// Bopl batched rollout forward (BOPL_BATCHED_FWD=1).
//
// One buffer step used to run 1 + banks separate policy chains (~15 kernels
// and 3-4 cuBLAS split-K launches each, ~80 records per step) plus four H2D
// memcpys and one D2H memcpy. Here the live rows of a buffer chunk -- laid out
// [learner | bank 0 | ... | bank n-1 | idle] by native_vector.py, described by
// pufferl->bank_layout -- go through ONE chain of seven kernels whose blocks
// pick their weights by row group:
//
//   br_encode   set encoder: obs row -> flat_input, pooled entity/slime
//               channels of joined; also copies the obs row and the reward /
//               terminal into the rollout tensors (the old cast kernels).
//               Stages only the populated slots of the row (header counts).
//   br_gemm     grouped small-M GEMM: mma.sync tf32 with the weights stored
//               in A-fragment order (BOPL_TF32=1) or fp32 FMAs (=0); 16-row
//               tiles, 8 warps split K in 64-chunks, fixed-order reduction,
//               optional ReLU epilogue: flat, fusion, MinGRU, decoder.
//   br_gate     MinGRU gate with the terminal reset fused in, state updated
//               in place (the old mask_state_rows + gate + state copy).
//   br_sample   warp-per-row sampler (same Philox stream per row as
//               sample_logits_warp) reading the env's byte action mask; writes
//               the actions to the env buffer AND the rollout, the float mask
//               to the rollout.
//
// Env buffers (BOPL_ZERO_COPY): 0 (default) one H2D memcpy of the live obs
// rows with rewards | terminals | mask packed into the idle tail behind them
// (vecenv.h bopl_upload) and one D2H memcpy of the actions, both captured
// into the buffer's CUDA graph (BOPL_GRAPH_COPIES=1); 1 / 2 read the pinned
// host buffers directly through their device aliases (slower: PCIe reads
// from SMs stall the shared partition).
//
// Numerics: the fp32 path is deterministic and within 3e-8 of cuBLAS fp32;
// the tf32 path rounds inputs to tf32 like cuBLAS did (BOPL_TF32 default).
// The chain has no atomics; runs are bit-identical.
#pragma once
#include <mma.h>

// Programmatic dependent launch for the per-step chain (graph-captured: the
// attribute becomes a programmatic graph edge): every chain kernel waits for
// its predecessor's completion (and memory flush) before its first read and
// releases its dependents (griddepcontrol.launch_dependents) before its
// epilogue, so the dependent's launch and scheduling overlap the tail. The
// release sits late, not at entry: released dependents occupy SM slots (a
// br_gemm block holds 32 KB of shared) spinning in the wait, and with the
// release at entry the overlapped training pass lost 36% (train_s 40 -> 55
// ms at w128 b4 p2) for a 13% faster rollout: a net loss. Both are no-ops
// for a kernel launched without the attribute (eager self-checks, v3).
// Because the wait is on the immediate predecessor only, every kernel in a
// PDL-launched chain waits unconditionally, so completion stays transitive.
// A kernel that exits without releasing counts as released.
__device__ __forceinline__ void br_pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void br_pdl_release() { asm volatile("griddepcontrol.launch_dependents;"); }
template<typename... KArgs, typename... Args>
static void br_launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args&&... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization; attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
    if (e != cudaSuccess) { fprintf(stderr, "br_launch_pdl: %s\n", cudaGetErrorString(e)); abort(); }
}
// An event the host polls: an external event node of the graph when the
// stream is capturing (the flag is an error outside capture).
static void br_record_external(cudaEvent_t e, cudaStream_t stream) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cs);
    cudaEventRecordWithFlags(e, stream, cs == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault);
}

#define BR_MAX_GROUPS 9   // learner + up to 8 frozen banks
#define BR_MAX_LAYERS 4

struct BrGroups {
    int n;
    int end[BR_MAX_GROUPS + 1];
};
struct BrLinear { const float* w[BR_MAX_GROUPS]; };

__device__ __forceinline__ int br_group_of(const BrGroups& groups, int row) {
    int group=0;
    while(group+1<groups.n && row>=groups.end[group+1]) group++;
    return group;
}

// Layout-v3 rollout encoder. The copy/flat path and the three small sets share
// one row-sized staging block. The entity set runs in its own 64-thread block:
// keeping its much longer 64x33 loop in the 256-thread block left six warps
// idle and limited each SM to only six useful entity warps.
static constexpr int BR_ENCODE_THREADS=192;

struct BrEncodeArgs {
    const float* obs;
    const float* rewards;
    const float* terminals;
    float* rollout_obs;
    float* rollout_rewards;
    float* rollout_terminals;
    float* flat_input;
    float* joined;
    int H, rows, sparse;
    BrGroups g;
    BrLinear w_player, w_slot, w_entity, w_slime, w_embedding;
};

static constexpr int BR_ROW_STRIDE=(BS_OBS+3)&~3;

__global__ void __launch_bounds__(BR_ENCODE_THREADS, 3) br_encode(BrEncodeArgs a) {
    __shared__ __align__(16) float row_data[BR_ROW_STRIDE];
    int row=blockIdx.x, thread=threadIdx.x;
    if(row>=a.rows) return;
    const float* source=a.obs+(long)row*BS_OBS;
    int entity_count=BS_N_ENT, slime_count=BS_N_SLIME;
    if(a.sparse) {
        entity_count=max(0,min(BS_N_ENT,__float2int_rn(source[4]*BS_N_ENT)));
        slime_count=max(0,min(BS_N_SLIME,__float2int_rn(source[6]*BS_N_SLIME)));
    }
    for(int i=thread;i<BR_ROW_STRIDE;i+=BR_ENCODE_THREADS) {
        bool read=i<BS_OBS && (!a.sparse || i<BS_ENTITY_OFFSET
            || (i>=BS_ENTITY_OFFSET && i<BS_ENTITY_OFFSET+entity_count*BS_W_ENT)
            || (i>=BS_SLIME_OFFSET && i<BS_SLIME_OFFSET+slime_count*BS_W_SLIME)
            || i>=BS_PREVIOUS_OFFSET);
        row_data[i]=read?source[i]:0.0f;
    }
    if(thread==0) {
        a.rollout_rewards[row]=a.rewards[row];
        a.rollout_terminals[row]=a.terminals[row];
    }
    __syncthreads();
    for(int i=thread;i<BS_OBS;i+=BR_ENCODE_THREADS)
        a.rollout_obs[(long)row*BS_OBS+i]=row_data[i];
    for(int i=thread;i<BS_FLAT;i+=BR_ENCODE_THREADS) {
        if(i<BS_PREFIX) a.flat_input[(long)row*BS_FLAT+i]=row_data[i];
        else if(i<BS_PREFIX+BOPL_SELF_FEATURES)
            a.flat_input[(long)row*BS_FLAT+i]=row_data[BOPL_OBS_SELF_OFFSET+i-BS_PREFIX];
        else a.flat_input[(long)row*BS_FLAT+i]=row_data[BS_PREVIOUS_OFFSET+i-BS_PREFIX-BOPL_SELF_FEATURES];
    }

    int pool=thread/BS_E, channel=thread%BS_E;
    int group=br_group_of(a.g,row);
    const float *block=nullptr, *weights=nullptr, *embedding=nullptr;
    int count=0, width=0, projected=0, id_column=-1;
    if(pool==0) {
        block=row_data+BS_PLAYER_OFFSET; weights=a.w_player.w[group];
        count=BS_N_PLAYER; width=BS_W_PLAYER; projected=BS_K_PLAYER;
    } else if(pool==1) {
        block=row_data+BS_SLOT_OFFSET; weights=a.w_slot.w[group]; embedding=a.w_embedding.w[group];
        count=BS_N_SLOT; width=BS_W_SLOT; projected=BS_K_SLOT; id_column=BOPL_SLOT_ABILITY_ID;
    } else {
        block=row_data+BS_SLIME_OFFSET; weights=a.w_slime.w[group];
        count=BS_N_SLIME; width=BS_W_SLIME; projected=BS_K_SLIME;
    }
    float best=0.0f;
    for(int slot=0;slot<count;slot++) {
        const float* x=block+slot*width;
        if(x[0]==0.0f) break;
        float sum=0.0f;
        if(id_column>=0) sum=embedding[bs_embedding_id(x[id_column])*BS_E+channel];
        for(int k=0;k<projected;k++) sum=fmaf(x[k],weights[channel*projected+k],sum);
        best=fmaxf(best,sum);
    }
    int joined_pool=pool<2 ? pool : 3;
    a.joined[(long)row*(a.H+BS_JOINED_POOLS)+a.H+joined_pool*BS_E+channel]=best;
}

__global__ void __launch_bounds__(BS_E, 12) br_encode_entity(BrEncodeArgs a) {
    int row=blockIdx.x, channel=threadIdx.x;
    if(row>=a.rows || channel>=BS_E) return;
    const float* source=a.obs+(long)row*BS_OBS;
    int count=BS_N_ENT;
    if(a.sparse) count=max(0,min(BS_N_ENT,__float2int_rn(source[4]*BS_N_ENT)));
    int group=br_group_of(a.g,row);
    const float* weights=a.w_entity.w[group];
    const float* embedding=a.w_embedding.w[group];
    const float* block=source+BS_ENTITY_OFFSET;
    float best=0.0f;
    for(int slot=0;slot<count;slot++) {
        const float* x=block+slot*BS_W_ENT;
        if(x[0]==0.0f) break;
        float sum=embedding[bs_embedding_id(x[BOPL_ENTITY_ABILITY_ID])*BS_E+channel];
        #pragma unroll
        for(int k=0;k<BS_K_ENT;k++) sum=fmaf(x[k],weights[channel*BS_K_ENT+k],sum);
        best=fmaxf(best,sum);
    }
    a.joined[(long)row*(a.H+BS_JOINED_POOLS)+a.H+2*BS_E+channel]=best;
}

// ---------------------------------------------------------------------------
// Grouped GEMM: out[r, n] = act(sum_k in[r, k] * Wt_g[k, n]) for r in group g,
// with the weights pre-transposed to (K, N) row-major (br_transpose, once per
// rollout) so a warp's lanes read consecutive columns of one k row coalesced.
// The problem is tiny (<= 256 rows, K <= 946) and latency-bound, so the block
// is built around a short dependency chain: a block owns 16 rows and its 8
// warps split K in 64-wide chunks; when K has fewer chunks than warps the
// spare warps take further 32-column subtiles (K=128: 2 warps per subtile, 4
// subtiles = 128 columns per block, so the small GEMMs fit one wave). A warp
// runs at most two chunks, so there is no software pipelining: the block is
// kept at <= 128 registers per thread so two blocks (and other buffers' chains)
// share an SM. The warps' partial tiles are reduced through shared memory in a
// fixed order (deterministic). TF32 (default, like cuBLAS' BOPL_TF32 mode): mma.sync
// m16n8k8 tensor-core steps on the transposed product (the weights are the A
// operand, the rows B: see br_frag_index), inputs rounded to tf32, fp32
// accumulate; the weights are stored in the A-fragment order
// (br_transpose_frag) so a lane's operands are one float4 and a warp's load is
// 512 contiguous bytes, and a lane reads its B operands straight from the rows. fp32
// (BOPL_TF32=0): CUDA-core FMAs over the plain transposed weights (K, ldb), a
// thread owns one column and all 16 rows. Row tiles never straddle a group
// (tiles[] = cumulative 16-row tiles per group). Slots past the chunk count
// (K = 304: 5 chunks for the 8 slots, K = 384: 6) have no warp, the block is
// that much narrower: their partials were zero tiles, and adding +0 to the
// slot sum (which starts at +0, so is never -0) does not change it. K must
// be even.
#define BR_TM 16
#define BR_TN 32
#define BR_TK 64
#define BR_KG 8
#define BR_THREADS (BR_TN * BR_KG)
struct BrGemmArgs {
    const float* in; int ld_in;
    float* out; int ld_out;
    int K, N, ldb, relu;  // ldb: fp32 = row stride of the (K, ldb) transposed weights; tf32 = m-tiles (ceil(N/16))
    BrGroups g;
    BrLinear wt;  // per-group weights: fp32 (K, ldb) transposed; tf32 fragment order (see br_frag_index)
    int tiles[BR_MAX_GROUPS + 1];
};

// TF32 fragment-order weight layout: element (k, n) of the (K, N) weights.
// The mma computes the output transposed, C^T = W^T X^T: the weights are the
// m16n8k8 A operand (16 output columns an m-tile) and the rows the B operand
// (8 rows an n-tile), so a group of 8 rows (a frozen bank's) fills one n-tile
// whole instead of half an m16 row tile. A fragment: lane = 4 groupID + tid4
// holds a0 = A[gid][tid4], a1 = A[gid + 8][tid4], a2 = A[gid][tid4 + 4], a3 =
// A[gid + 8][tid4 + 4] of a 16 x 8 (n, k) tile; one k-step (8 k) of one
// m-tile is the lane's float4 {a0, a1, a2, a3}, a warp's load 512 contiguous
// bytes. The k of a step go to the fragment's k slots permuted: slot tid4
// takes k = 2 tid4, slot tid4 + 4 takes k = 2 tid4 + 1, so a lane's B
// operands of a step (b0 = B[tid4][gid], b1 = B[tid4 + 4][gid]: row gid) are
// two adjacent floats of a row, one float2 load from global, with no shared
// memory round. The 8-product sum does not depend on the order of its
// products (measured: 0 of 384k sums differ under four permutations, wide
// exponent range, with and without an accumulator) nor on which operand a
// factor is (A B = (B^T A^T)^T: each output's 8 products are the same pairs
// in the same slots), so the bits are those of the unpermuted, unswapped step.
__host__ __device__ __forceinline__ long br_frag_index(int k, int n, int mtiles) {
    int kstep = k >> 3, kk = k & 7, mtile = n >> 4, gid = n & 7, hi = (n >> 3) & 1;
    int tid4 = kk >> 1, half = kk & 1;
    return (((long)kstep * mtiles + mtile) * 32 + gid * 4 + tid4) * 4 + half * 2 + hi;
}
__host__ __device__ __forceinline__ long br_frag_floats(int K, int N) {
    return (long)((K + 7) / 8) * 8 * ((N + 15) / 16) * 16;
}

__device__ __forceinline__ unsigned br_tf32(float x) {
    return __float_as_uint(nvcuda::wmma::__float_to_tf32(x));
}
__device__ __forceinline__ void br_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Warps per column subtile: the K slots that have a chunk.
__host__ __device__ __forceinline__ int br_gemm_slots(int K, int* sub) {
    const int nchunks = (K + BR_TK - 1) / BR_TK;
    int s = 1;
    while (s < nchunks && s < BR_KG) s <<= 1;
    *sub = s;
    return nchunks < s ? nchunks : s;
}

// Shared floats per warp: fp32 stages its A chunk (16 x 64) there, TF32 only
// its 16 x 32 C tile (the fp32 C tile aliases the A chunk). The TF32 tile's
// rows are 36 floats apart: a lane's C entries (rows 2 tid4 (+1), column gid
// of an m-tile) then fall in 32 distinct banks per store.
template <bool TF32> __host__ __device__ constexpr int br_gemm_ldc() { return TF32 ? BR_TN + 4 : BR_TN; }
template <bool TF32> __host__ __device__ constexpr int br_gemm_warp_smem() { return TF32 ? BR_TM * br_gemm_ldc<TF32>() : BR_TM * BR_TK; }

// The GEMM's block body: the row tile from row0 of group grp, column subtile
// set ctile (the BR_KG / sub subtiles one block covers), the warps' shared
// tiles at As (br_gemm_warp_smem floats each). pdl: griddepcontrol.wait
// before the first read of A (the predecessor's output) and
// launch_dependents under the epilogue.
template <bool TF32>
__device__ __forceinline__ void br_gemm_tile(const BrGemmArgs& a, int grp, int row0, int ctile, float* As, bool pdl) {
    constexpr int WS = br_gemm_warp_smem<TF32>();
    // PDL: the B operand (weights) does not depend on the predecessor, so
    // the first chunk's B loads are issued before griddepcontrol.wait and
    // their latency overlaps the predecessor's tail; A (the predecessor's
    // output) is read after it. A warp with no chunk waits after the loop.
    bool waited = !pdl;
    const int row_end = a.g.end[grp + 1];
    const int K = a.K, N = a.N, ldb = a.ldb;
    const int nchunks = (K + BR_TK - 1) / BR_TK;
    // sub = K slots per column subtile (power of two >= nchunks, <= BR_KG), used of them with a chunk (a warp each)
    int sub;
    const int used = br_gemm_slots(K, &sub);
    const int subtiles = BR_KG / sub;
    const int tid = threadIdx.x, warp = tid / BR_TN, c = tid % BR_TN, nthreads = subtiles * used * BR_TN;
    const int kg = warp % used;  // this warp's K slot within its subtile
    const int col0 = (ctile * subtiles + warp / used) * BR_TN;
    const float* __restrict__ Wt = a.wt.w[grp];
    float* Aw = As + warp * WS;
    const int lane = c;
    const int gid = lane >> 2, tid4 = lane & 3;
    if constexpr (TF32) {
        // A chunk (the weights): 8 k-steps (64 k) x 2 m-tiles (32 columns), one
        // float4 per lane each, a warp instruction reads 512 contiguous bytes.
        // k-steps past K are outside the buffer (br_frag_floats rounds K up to
        // 8) and so are m-tiles past mtiles (the block's column span exceeds
        // N, e.g. narrow decoders): zeros, their products are +0 and their
        // columns are dropped by the epilogue.
        const int mtiles = ldb, mtile0 = col0 >> 4, ksteps = (K + 7) >> 3;
        float4 wf[BR_TK / 8][2];
        auto load_w = [&](int k0) {
            #pragma unroll
            for (int ks = 0; ks < BR_TK / 8; ks++)
                #pragma unroll
                for (int t = 0; t < 2; t++) {
                    long kstep = (k0 >> 3) + ks;
                    wf[ks][t] = (kstep < ksteps && mtile0 + t < mtiles)
                        ? *(const float4*)(Wt + ((kstep * mtiles + mtile0 + t) * 32 + lane) * 4)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
                }
        };
        // acc[t][h]: m-tile t (columns col0 + 16 t ..) of n-tile h (rows row0 + 8 h ..)
        float acc[2][2][4];
        #pragma unroll
        for (int t = 0; t < 2; t++)
            #pragma unroll
            for (int h = 0; h < 2; h++) { acc[t][h][0] = acc[t][h][1] = acc[t][h][2] = acc[t][h][3] = 0.f; }
        // B operands of step ks (k slots permuted, see br_frag_index): row gid
        // of each n-tile at k = 8 ks + 2 tid4 + {0, 1}, a float2 (K is even: a
        // pair is past K whole); 8 loads per chunk and n-tile, every 32-byte
        // sector used whole. Scalar loads for rows that are not 8-byte aligned.
        // A tile whose second n-tile has no row (a group's last 8 rows or
        // fewer: a frozen bank) runs the first only, half the loads and mmas.
        const bool a2 = (a.ld_in & 1) == 0 && (((size_t)a.in) & 7) == 0;
        const bool r0 = row0 + gid < row_end, r1 = row0 + gid + 8 < row_end;
        const float* x0 = a.in + (long)(row0 + gid) * a.ld_in;
        const float* x1 = x0 + 8L * a.ld_in;
        auto chunks = [&](auto two_c) {
            constexpr bool TWO = decltype(two_c)::value;
            for (int ch = kg; ch < nchunks; ch += sub) {
                load_w(ch * BR_TK);
                if (!waited) { br_pdl_wait(); waited = true; }
                float2 xa[BR_TK / 8], xb[BR_TK / 8];
                #pragma unroll
                for (int ks = 0; ks < BR_TK / 8; ks++) {
                    int k = ch * BR_TK + ks * 8 + 2 * tid4;
                    xa[ks] = r0 && k < K ? (a2 ? *(const float2*)(x0 + k) : make_float2(x0[k], x0[k + 1])) : make_float2(0.f, 0.f);
                    if (TWO) xb[ks] = r1 && k < K ? (a2 ? *(const float2*)(x1 + k) : make_float2(x1[k], x1[k + 1])) : make_float2(0.f, 0.f);
                }
                #pragma unroll
                for (int ks = 0; ks < BR_TK / 8; ks++) {
                    const unsigned b0 = br_tf32(xa[ks].x), b1 = br_tf32(xa[ks].y);
                    unsigned c0 = 0, c1 = 0;
                    if (TWO) { c0 = br_tf32(xb[ks].x); c1 = br_tf32(xb[ks].y); }
                    #pragma unroll
                    for (int t = 0; t < 2; t++) {
                        const float4 w = wf[ks][t];
                        const unsigned af[4] = {br_tf32(w.x), br_tf32(w.y), br_tf32(w.z), br_tf32(w.w)};
                        br_mma(acc[t][0], af, b0, b1);
                        if (TWO) br_mma(acc[t][1], af, c0, c1);
                    }
                }
            }
        };
        if (row0 + 8 < row_end) chunks(std::true_type{});
        else chunks(std::false_type{});
        // C^T fragment: c0 (column gid, row 2 tid4), c1 (gid, 2 tid4 + 1), c2/c3 at column gid + 8
        constexpr int LDC = br_gemm_ldc<TF32>();
        float* Cw = Aw;  // [16][LDC]: rows of the tile x its 32 columns
        #pragma unroll
        for (int t = 0; t < 2; t++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                float* r = Cw + (h * 8 + 2 * tid4) * LDC + t * 16 + gid;
                r[0] = acc[t][h][0]; r[LDC] = acc[t][h][1]; r[8] = acc[t][h][2]; r[LDC + 8] = acc[t][h][3];
            }
    } else {
        // A chunk: 16 rows x 64 k = 256 float4 over the warp (element i: row
        // i%16, k-quad i/16), staged as [k][row] for float4 row reads. K is
        // even; the tail quad past K is masked per pair. float4 row loads when
        // the rows are 16-byte aligned (v3's flat block has a stride of 98),
        // scalar loads otherwise: the same values.
        float4 av[BR_TK / 8];
        const bool a4 = (a.ld_in & 3) == 0 && (((size_t)a.in) & 15) == 0;
        auto load_a = [&](int k0) {
            #pragma unroll
            for (int j = 0; j < BR_TK / 8; j++) {
                int i = c + j * 32, r = i & 15, k = k0 + ((i >> 4) << 2);
                const float* src = a.in + (long)(row0 + r) * a.ld_in + k;
                if (row0 + r < row_end && k + 3 < K && a4) av[j] = *(const float4*)src;
                else if (row0 + r < row_end && k < K)
                    av[j] = make_float4(src[0], k + 1 < K ? src[1] : 0.f, k + 2 < K ? src[2] : 0.f, k + 3 < K ? src[3] : 0.f);
                else av[j] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        };
        auto store_a = [&]() {
            #pragma unroll
            for (int j = 0; j < BR_TK / 8; j++) {
                int i = c + j * 32, r = i & 15, k = (i >> 4) << 2;
                Aw[(k + 0) * BR_TM + r] = av[j].x; Aw[(k + 1) * BR_TM + r] = av[j].y;
                Aw[(k + 2) * BR_TM + r] = av[j].z; Aw[(k + 3) * BR_TM + r] = av[j].w;
            }
        };
        const int col = col0 + c;
        const float* __restrict__ Wc = Wt + min(col, N - 1);
        float acc[BR_TM];
        #pragma unroll
        for (int r = 0; r < BR_TM; r++) acc[r] = 0.f;
        float b[BR_TK];
        auto load_b = [&](int k0) {
            #pragma unroll
            for (int j = 0; j < BR_TK; j++) { int k = k0 + j; b[j] = k < K ? Wc[(long)k * ldb] : 0.f; }
        };
        for (int ch = kg; ch < nchunks; ch += sub) {
            load_b(ch * BR_TK);
            if (!waited) { br_pdl_wait(); waited = true; }
            load_a(ch * BR_TK);
            __syncwarp();
            store_a();
            __syncwarp();
            #pragma unroll 8
            for (int kk = 0; kk < BR_TK; kk++) {
                const float w = b[kk];
                const float4* x = (const float4*)(Aw + kk * BR_TM);
                float4 x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
                acc[0] = fmaf(x0.x, w, acc[0]); acc[1] = fmaf(x0.y, w, acc[1]); acc[2] = fmaf(x0.z, w, acc[2]); acc[3] = fmaf(x0.w, w, acc[3]);
                acc[4] = fmaf(x1.x, w, acc[4]); acc[5] = fmaf(x1.y, w, acc[5]); acc[6] = fmaf(x1.z, w, acc[6]); acc[7] = fmaf(x1.w, w, acc[7]);
                acc[8] = fmaf(x2.x, w, acc[8]); acc[9] = fmaf(x2.y, w, acc[9]); acc[10] = fmaf(x2.z, w, acc[10]); acc[11] = fmaf(x2.w, w, acc[11]);
                acc[12] = fmaf(x3.x, w, acc[12]); acc[13] = fmaf(x3.y, w, acc[13]); acc[14] = fmaf(x3.z, w, acc[14]); acc[15] = fmaf(x3.w, w, acc[15]);
            }
        }
        __syncwarp();  // the warp is done with its A chunk: reuse as its C tile
        #pragma unroll
        for (int r = 0; r < BR_TM; r++) Aw[r * BR_TN + c] = acc[r];
    }
    if (!waited) { br_pdl_wait(); waited = true; }
    __syncthreads();
    if (pdl) br_pdl_release();   // the dependent launches under this epilogue
    // Per subtile, sum its warps' partial tiles in slot order and write the
    // valid entries, 4 adjacent columns a thread (each its own sum; the
    // scalar form took 0.2 us more of a critic GEMM).
    const bool o4 = (a.ld_out & 3) == 0 && (((size_t)a.out) & 15) == 0;
    for (int o = tid; o < subtiles * BR_TM * BR_TN / 4; o += nthreads) {
        int st = o / (BR_TM * BR_TN / 4), e = (o % (BR_TM * BR_TN / 4)) * 4, r = e / BR_TN, cc = e % BR_TN;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int w = st * used; w < (st + 1) * used; w++) {
            float4 p = *(const float4*)&As[w * WS + r * br_gemm_ldc<TF32>() + cc];
            v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
        }
        if (a.relu) { v.x = fmaxf(0.f, v.x); v.y = fmaxf(0.f, v.y); v.z = fmaxf(0.f, v.z); v.w = fmaxf(0.f, v.w); }
        int row = row0 + r, cl = (ctile * subtiles + st) * BR_TN + cc;
        if (row < row_end) {
            float* dst = a.out + (long)row * a.ld_out + cl;
            if (cl + 3 < N && o4) *(float4*)dst = v;
            else {
                const float vv[4] = {v.x, v.y, v.z, v.w};
                #pragma unroll
                for (int q = 0; q < 4; q++) if (cl + q < N) dst[q] = vv[q];
            }
        }
    }
}

template <bool TF32>
__global__ void __launch_bounds__(BR_THREADS, 2) br_gemm(BrGemmArgs a) {
    __shared__ __align__(16) float As[BR_KG * br_gemm_warp_smem<TF32>()];   // TF32 16 KB, fp32 32 KB
    const int rt = blockIdx.x;
    int grp = 0;
    while (grp + 1 < a.g.n && rt >= a.tiles[grp + 1]) grp++;
    br_gemm_tile<TF32>(a, grp, a.g.end[grp] + (rt - a.tiles[grp]) * BR_TM, blockIdx.y, As, true);
}

// Column-subtile sets of br_gemm's grid for a GEMM of depth K over N columns.
static int br_gemm_col_tiles(int K, int N) {
    int sub;
    br_gemm_slots(K, &sub);
    int cols = (BR_KG / sub) * BR_TN;
    return (N + cols - 1) / cols;
}

// TF32: dst_g = src_g (N, K) in A-fragment order (br_frag_index), zero padded to
// 8 k x 16 n; one thread per destination float4 (a lane's fragment: columns
// n, n + 8 at k0, k0 + 1), so the stores are whole float4s in order and a
// warp's loads are 16 columns x 8 consecutive k (32-byte runs).
// The thread-per-element form strided both sides (1.03 ms per update).
__global__ void br_transpose_frag(BrLinear src, BrLinear dst, int N, int K) {
    const float* s = src.w[blockIdx.z];
    float4* d = (float4*)dst.w[blockIdx.z];
    int mtiles = (N + 15) / 16, Ks = (K + 7) / 8;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;   /* destination float4 */
    if (i >= (long)Ks * mtiles * 32) return;
    int lane = (int)(i & 31), tile = (int)(i >> 5);
    int kstep = tile / mtiles, mtile = tile - kstep * mtiles;
    int n0 = mtile * 16 + (lane >> 2), k0 = kstep * 8 + 2 * (lane & 3);
    float v[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {   /* e = half * 2 + hi: k = k0 + half, n = n0 + 8 hi (br_frag_index) */
        int k = k0 + (e >> 1), n = n0 + (e & 1) * 8;
        v[e] = (k < K && n < N) ? s[(long)n * K + k] : 0.f;
    }
    d[i] = make_float4(v[0], v[1], v[2], v[3]);
}
// fp32: dst_g[k][n] = src_g[n][k] for every group (grid.z), 32x32 tiles through shared memory.
__global__ void br_transpose(BrLinear src, BrLinear dst, int N, int K, int ldt) {
    __shared__ float tile[32][33];
    const float* s = src.w[blockIdx.z];
    float* d = (float*)dst.w[blockIdx.z];
    int n0 = blockIdx.y * 32, k0 = blockIdx.x * 32;
    int tx = threadIdx.x, ty = threadIdx.y;  // 32 x 8
    for (int i = ty; i < 32; i += 8) {
        int n = n0 + i, k = k0 + tx;
        if (n < N && k < K) tile[i][tx] = s[(long)n * K + k];
    }
    __syncthreads();
    for (int i = ty; i < 32; i += 8) {
        int k = k0 + i, n = n0 + tx;
        if (k < K && n < N) d[(long)k * ldt + n] = tile[tx][i];
    }
}

// ---------------------------------------------------------------------------
// MinGRU gate with the episode reset fused in; the layer's state is updated in
// place (each thread reads then writes its own element).
struct BrGateArgs {
    const float* combined;   // (rows, 3H)
    const float* x;          // (rows, H) layer input
    const float* terminals;  // (rows,) env terminals: nonzero -> reset state first
    float* out;              // (rows, H)
    float* state[BR_MAX_GROUPS];  // group g: layer-l state of row r at state[g] + (l * state_rows[g] + r) * H
    int state_rows[BR_MAX_GROUPS];
    int layer, H, rows;
    BrGroups g;
};

__global__ void br_gate(BrGateArgs a) {
    br_pdl_wait(); br_pdl_release();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.rows * a.H) return;
    int row = idx / a.H, h = idx - row * a.H;
    int grp = br_group_of(a.g, row);
    float* st = a.state[grp] + ((long)a.layer * a.state_rows[grp] + (row - a.g.end[grp])) * a.H + h;
    long cb = (long)row * 3 * a.H;
    float hidden = a.combined[cb + h];
    float gate = a.combined[cb + a.H + h];
    float proj = a.combined[cb + 2 * a.H + h];
    float state = a.terminals[row] != 0.0f ? 0.0f : *st;
    float x = a.x[idx];
    float gate_sigmoid = sigmoid(gate);
    float hidden_tilde = (hidden >= 0.0f) ? hidden + 0.5f : fast_sigmoid(hidden);
    float mingru_out = lerp(state, hidden_tilde, gate_sigmoid);
    *st = mingru_out;
    float proj_sigmoid = sigmoid(proj);
    a.out[idx] = proj_sigmoid * mingru_out + (1.0f - proj_sigmoid) * x;
}

// ---------------------------------------------------------------------------
// Host side.
struct BrActs {
    float *flat, *joined, *enc, *comb, *gru[2], *dec;
};
struct BrWt {  // one group's transposed weights, (K, ld) row-major
    float *flat, *fusion, *gru[BR_MAX_LAYERS], *dec;
};
struct Br4Plan;      // policy v4 chain (bopl_rollout_v4.cu); nullptr = layout-v3 set encoder chain
struct BrPlan {
    int enabled;         // batched chain in use
    Br4Plan* v4;
    int zero_copy;       // 0 memcpys (obs + packed small inputs), 1 obs memcpy + rest zero-copy, 2 all zero-copy
    int actions_zero_copy;  // sampler writes actions into the mapped host buffer
    int graph_copies;    // H2D / D2H memcpys captured into the forward graph (BOPL_GRAPH_COPIES, default 1)
    int transport_check; // dump mode (train.py --dump-rows): count device-row floats that differ from the host row
    const float* packed_mapped;  // the device alias of vec->packed_host (bopl4_unpack_rows reads the slots' tails there)
    int tf32;            // tensor-core tf32 GEMMs (BOPL_TF32, default 1) or fp32 FMAs
    int sparse;          // stage only the populated obs slots (BOPL_OBS_SPARSE, default 1)
    int rows;            // live rows per buffer (bank_layout end)
    int H, layers, A_total, num_atns, mask_stride;
    int dec_cols;      // raw decoder logit columns (== A_total unless the aim head is reparameterised)
    int aim_fourier, aim_ar;
    int ld_dec;          // decoder columns padded to a multiple of 8
    BrGroups g;
    int tiles[BR_MAX_GROUPS + 1];
    BrActs* acts;        // [num_buffers]
    BrWt wt[BR_MAX_GROUPS];
};

static inline int br_pad8(int n) { return (n + 7) & ~7; }

static int br_tiles(BrGroups& g, int* tiles) {
    tiles[0] = 0;
    for (int k = 0; k < g.n; k++) tiles[k + 1] = tiles[k] + (g.end[k + 1] - g.end[k] + BR_TM - 1) / BR_TM;
    return tiles[g.n];
}

static void br_gemm_launch(const BrPlan& p, const float* in, int ld_in, float* out, int ld_out,
        int K, int N, int ldb, int relu, const BrLinear& wt, cudaStream_t stream) {
    BrGemmArgs a = {};
    a.in = in; a.ld_in = ld_in; a.out = out; a.ld_out = ld_out; a.K = K; a.N = N; a.ldb = ldb; a.relu = relu;
    if (p.tf32) a.ldb = (N + 15) / 16;  // m-tiles of the fragment layout
    a.g = p.g; a.wt = wt;
    memcpy(a.tiles, p.tiles, sizeof(a.tiles));
    int sub, used = br_gemm_slots(K, &sub);
    dim3 grid(p.tiles[p.g.n], br_gemm_col_tiles(K, N)), block((BR_KG / sub) * used * BR_TN);
    if (p.tf32) br_launch_pdl(br_gemm<true>, grid, block, 0, stream, a);
    else br_launch_pdl(br_gemm<false>, grid, block, 0, stream, a);
}

// dst[g] = the GEMM-side layout of src[g] (N, K) for groups 0..n-1: tf32 fragment
// order, or the (K, ldt) transpose for the fp32 path.
static void br_transpose_launch(const BrPlan& p, const BrLinear& src, const BrLinear& dst, int n, int N, int K, int ldt, cudaStream_t stream) {
    if (p.tf32) {
        long total = br_frag_floats(K, N) / 4;
        br_transpose_frag<<<dim3((unsigned)((total + 255) / 256), 1, n), 256, 0, stream>>>(src, dst, N, K);
    } else {
        dim3 grid((K + 31) / 32, (N + 31) / 32, n);
        br_transpose<<<grid, dim3(32, 8), 0, stream>>>(src, dst, N, K, ldt);
    }
}

constexpr int BR4_UNPACK_THREADS = 256;   // bopl4_unpack_rows (bopl_rollout_v4.cu)
// The words at the head of every packed slot the H2D copy carries (a 2D copy
// of the buffer's slots); bopl4_unpack_rows reads the rest of a slot's used
// words from the mapped host slot. A mode-4 slot uses ~1000 of its 2048 words
// (median; p90 ~1400). 2026-10-07 (fwd_bench, 200 spec steps; actions ready,
// us): the whole-slot memcpy 177 (1 stream) / 371 (4 streams at once, per
// stream); every used word read by the unpack 152 / 388 (the GPU's own PCIe
// reads are slower than the copy engine's under load); heads of 512, 640,
// 768, 1024, 1280 words 158, 155, 155, 157, 170 / 367, 364, 361, 360, 365.
constexpr int BR4_PACK_DMA_WORDS = 768;
static_assert(BR4_PACK_DMA_WORDS % 4 == 0 && BR4_PACK_DMA_WORDS >= BOPL4_PACK_HEADER, "whole float4s, the slot's header included");
// bopl4_unpack_rows' grid: a block per row, then the pack's 16-byte units a thread each.
static int bopl4_unpack_blocks(int rows, int tail_bytes) { return rows + (tail_bytes + 16 * BR4_UNPACK_THREADS - 1) / (16 * BR4_UNPACK_THREADS); }
#ifndef BR_KERNELS_ONLY  // the trainer-side part (needs pufferlib.cu); microbenchmarks include only the kernels above
// Policy v4 chain hooks (bopl_rollout_v4.cu, included at the end of this file).
static const char* br4_check(PuffeRL* pufferl);
static void br4_init(PuffeRL* pufferl, BrPlan* p);
static void br4_prepare(PuffeRL* pufferl, BrPlan& p, cudaStream_t stream);
static void br4_free(PuffeRL* pufferl, BrPlan* p);
static const float* br4_dec_rows(BrPlan& p, int buf);
// Packed observation transport (bopl_obs_pack.h): the unpack of one buffer's
// slots into its dense device rows, the dump-mode check of those rows against
// the host rows, and the env-side registration (binding.c: the env struct).
__global__ void bopl4_unpack_rows(const float* __restrict__ host_packed, float* __restrict__ packed, const float* __restrict__ host_rows,
                                  float* __restrict__ dense, int rows, int tail_bytes);
__global__ void bopl4_transport_check(const float* __restrict__ dense, const float* __restrict__ host_rows, int rows, int* mismatches);
extern "C" void bopl4_rays_init(const float* packed, long slots);
extern "C" void bopl4_rays_launch(const float* packed, float* dense, int rows, cudaStream_t stream);
extern "C" void bopl_pack_register(void* envs, int start, int count, const float* base, float* out, long row_begin, long row_end);
static void br_check_pre(PuffeRL* pufferl, int buf, cudaStream_t stream);
static void br_check_post(PuffeRL* pufferl, int buf, int t, const float* terminals, const float* dec_rows, cudaStream_t stream);
static void br4_chain(PuffeRL* pufferl, BrPlan& p, int buf, const float* obs, const float* rewards, const float* terminals,
                      float* rollout_obs, float* rollout_rewards, float* rollout_terminals, float* values, cudaStream_t stream);
static void br4_join(BrPlan& p, int buf, cudaStream_t stream);
static int br4_value_rows(const BrPlan& p);
#endif  // BR_KERNELS_ONLY
// ---------------------------------------------------------------------------
// Sampler: sample_logits_warp with the env's byte mask as input and the
// actions written to both the env buffer (float) and the rollout. (Kernels
// only: SAMPLE_WARP_ROWS, finite_or_clamp, bopl_game_logit come from
// pufferlib.cu, or the forward bench.)
struct BrSampleArgs {
    const float* dec; int ld_dec, A_total; // (rows, ld_dec >= dec_cols + 1): raw logits then the value
    int dec_cols, aim_fourier, aim_ar;       // bopl_aim.h: A_total game logits are materialised from dec_cols columns
    const int* act_sizes; int num_atns;
    int value_rows;                        // rows below this have their value written by the chain (the v4 critic)
    const unsigned char* env_mask; int mask_stride;  // NULL when the env has no mask
    float* rollout_mask;                   // (rows, mask_stride) float copy for training
    float* env_actions;                    // (rows, num_atns)
    float* rollout_actions;                // (rows, num_atns)
    float* logprobs; float* values;        // (rows,)
    curandStatePhilox4_32_10_t* rng;       // (rows,) this buffer's states
    const signed char* head_consume; int hc_stride;
    int rows;
};

__global__ void __launch_bounds__(SAMPLE_WARP_ROWS * 32) br_sample(BrSampleArgs a) {
    __shared__ float s_logits[SAMPLE_WARP_ROWS][SAMPLE_MAX_LOGITS];
    __shared__ int s_buttons[SAMPLE_WARP_ROWS];
    br_pdl_wait(); br_pdl_release();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * SAMPLE_WARP_ROWS, idx = row0 + warp;
    const int live_rows = min(SAMPLE_WARP_ROWS, a.rows - row0);
    const bool live = idx < a.rows;   // (the other warps only join the block's aim pass)
    const int A_total = a.A_total;
    const float* logits = a.dec + (long)idx * a.ld_dec;
    const unsigned char* mask = live && a.env_mask ? a.env_mask + (long)idx * a.mask_stride : nullptr;
    float* row = s_logits[warp];
    // The mask may live in host memory: issue every lane's loads before the first use.
    constexpr int MASK_ITERS = (SAMPLE_MAX_LOGITS + 31) / 32;
    unsigned char mv[MASK_ITERS];
    if (mask) {
        #pragma unroll
        for (int it = 0; it < MASK_ITERS; it++) { int i = lane + it * 32; mv[it] = i < A_total ? mask[i] : 1; }
    }
    if (mask) {
        #pragma unroll
        for (int it = 0; it < MASK_ITERS; it++) {
            int i = lane + it * 32;
            if (i < A_total) a.rollout_mask[(long)idx * a.mask_stride + i] = (float)mv[it];
        }
    }
    const bool aim = a.aim_fourier > 0 || a.aim_ar;
    // Fourier aim: the block's rows' aim logits into s_logits, one bin per
    // thread, so a bin's basis (cosf, sinf and the rotations) is computed once
    // for the block's rows rather than per row; each (row, bin) takes the
    // operations of bopl_aim_fourier_logit_dev in its order. The buttons of
    // each row's aim block from s_buttons (autoregressive). All warps.
    // 2026-10-07 (fwd_bench, Fourier 4): the kernel 14.7 -> 9.9 us.
    auto fourier_bins = [&]() {
        __syncthreads();
        const int K = a.aim_fourier;
        const float* blk[SAMPLE_WARP_ROWS];
        #pragma unroll
        for (int r = 0; r < SAMPLE_WARP_ROWS; r++)
            blk[r] = a.dec + (long)(row0 + min(r, live_rows - 1)) * a.ld_dec + bopl_aim_block_offset(K, a.aim_ar, a.aim_ar ? s_buttons[r] : 0);
        for (int bin = threadIdx.x; bin < BOPL_ACT_AIM; bin += blockDim.x) {
            float sum[SAMPLE_WARP_ROWS];
            if (bin == 0) {
                #pragma unroll
                for (int r = 0; r < SAMPLE_WARP_ROWS; r++) sum[r] = blk[r][0];
            } else {
                BoplAimBasis b = bopl_aim_basis(bin);
                #pragma unroll
                for (int r = 0; r < SAMPLE_WARP_ROWS; r++) sum[r] = 0.0f;
                for (int k = 1; k <= K; k++) {
                    #pragma unroll
                    for (int r = 0; r < SAMPLE_WARP_ROWS; r++) sum[r] += blk[r][2 * k - 1] * b.ck + blk[r][2 * k] * b.sk;
                    bopl_aim_basis_next(&b);
                }
                #pragma unroll
                for (int r = 0; r < SAMPLE_WARP_ROWS; r++) sum[r] = sum[r] * bopl_aim_fourier_scale(K);
            }
            #pragma unroll
            for (int r = 0; r < SAMPLE_WARP_ROWS; r++) if (r < live_rows) s_logits[r][bin] = sum[r];
        }
        __syncthreads();
    };
    // Stage game logits [g0, g1) from the raw row (bopl_aim.h; identity unless
    // aim; the Fourier aim bins as fourier_bins left them in the row).
    auto stage = [&](int g0, int g1, int buttons) {
        #pragma unroll
        for (int it = 0; it < MASK_ITERS; it++) {
            int i = lane + it * 32;
            if (i < g0 || i >= g1) continue;
            float l = a.aim_fourier > 0 && i < BOPL_ACT_AIM ? row[i]
                    : aim ? bopl_game_logit(logits, a.aim_fourier, a.aim_ar, buttons, i) : logits[i];
            l = finite_or_clamp(l, -1.0e4f, 1.0e4f);
            if (mask && mv[it] == 0) l = -1e4f;
            row[i] = l;
        }
        __syncwarp();
    };

    curandStatePhilox4_32_10_t state;
    if (live && lane == 0) state = a.rng[idx];
    float total_log_prob = 0.0f;
    int verb = 0;
    int sampled[MAX_ATN_HEADS];
    auto sample_head = [&](int h) {
        const int A = a.act_sizes[h];
        int logits_offset = 0;
        for (int i = 0; i < h; ++i) logits_offset += a.act_sizes[i];
        const int chunk = (A + 31) >> 5;
        const int a0 = lane * chunk;
        const int a1 = min(a0 + chunk, A);
        float local_max = -INFINITY;
        for (int i = a0; i < a1; ++i) local_max = fmaxf(local_max, row[logits_offset + i]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, off));
        float local_sum = 0.0f;
        for (int i = a0; i < a1; ++i) local_sum += expf(row[logits_offset + i] - local_max);
        float sum_exp = local_sum;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum_exp += __shfl_xor_sync(0xffffffffu, sum_exp, off);
        const float logsumexp = local_max + logf(sum_exp);

        float rand_val = 0.0f;
        if (lane == 0) rand_val = curand_uniform(&state);
        rand_val = __shfl_sync(0xffffffffu, rand_val, 0);

        float local_mass = 0.0f;
        for (int i = a0; i < a1; ++i) local_mass += expf(row[logits_offset + i] - logsumexp);
        float prefix = local_mass;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float v = __shfl_up_sync(0xffffffffu, prefix, off);
            if (lane >= off) prefix += v;
        }
        float cumsum = prefix - local_mass;
        int candidate = INT_MAX;
        for (int i = a0; i < a1; ++i) {
            cumsum += expf(row[logits_offset + i] - logsumexp);
            if (rand_val < cumsum) { candidate = i; break; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) candidate = min(candidate, __shfl_xor_sync(0xffffffffu, candidate, off));
        int sampled_action = candidate;
        if (sampled_action == INT_MAX) {
            sampled_action = A - 1;
            if (mask) {
                for (int i = A - 1; i >= 0; --i) {
                    if (mask[logits_offset + i] != 0) { sampled_action = i; break; }
                }
            }
        }
        const float log_prob = row[logits_offset + sampled_action] - logsumexp;
        if (h == 0) verb = sampled_action;
        const int used = (a.head_consume == nullptr || h == 0) ? 1 : (int)a.head_consume[verb * a.hc_stride + h];
        if (used) total_log_prob += log_prob;
        if (lane == 0) {
            a.rollout_actions[(long)idx * a.num_atns + h] = (float)sampled_action;
            a.env_actions[(long)idx * a.num_atns + h] = (float)sampled_action;
        }
        sampled[h] = sampled_action;
    };
    if (a.aim_ar) {
        // Autoregressive aim: buttons (head 1) first, then the aim block of
        // that combo (head 0), then the rest. Same order as sample_logits_warp.
        if (live) {
            stage(BOPL_ACT_AIM, A_total, 0);
            sample_head(1);
            if (lane == 0) s_buttons[warp] = sampled[1];
        }
        if (a.aim_fourier > 0) fourier_bins();
        if (!live) return;
        stage(0, BOPL_ACT_AIM, sampled[1]);
        sample_head(0);
        for (int h = 2; h < a.num_atns; ++h) sample_head(h);
    } else {
        if (a.aim_fourier > 0) fourier_bins();
        if (!live) return;
        stage(0, A_total, 0);
        for (int h = 0; h < a.num_atns; ++h) sample_head(h);
    }
    if (lane == 0) {
        a.logprobs[idx] = total_log_prob;
        if (idx >= a.value_rows) a.values[idx] = logits[a.dec_cols];
        a.rng[idx] = state;
    }
}

#ifndef BR_KERNELS_ONLY
// Decide whether the batched chain applies and allocate its per-buffer
// activations. Called once after the frozen banks exist and before cudagraph
// capture. Falls back to the per-bank chain (pufferl->br == nullptr) when the
// encoder is not the fused set encoder, a bank differs in shape, or
// BOPL_BATCHED_FWD=0.
static int br_env_int(const char* name, int fallback) {
    const char* v = getenv(name);
    return (v && v[0]) ? atoi(v) : fallback;
}

static void br_init(PuffeRL* pufferl) {
    pufferl->br = nullptr;
    if (!br_env_int("BOPL_BATCHED_FWD", 1)) return;
    StaticVec* vec = pufferl->vec;
    HypersT& hypers = pufferl->hypers;
    const char* why = nullptr;
    DecoderWeights* dw = (DecoderWeights*)pufferl->weights.decoder;
    MinGRUWeights* mw = (MinGRUWeights*)pufferl->weights.network;
    const bool is_v4 = bopl_is_v4_encoder(pufferl->policy.encoder);
    if (pufferl->policy.encoder.forward != bs_forward && !is_v4) why = "encoder is not the fused set encoder or policy v4";
    else if (dw->continuous) why = "continuous decoder";
    else if (pufferl->policy.network.forward != mingru_forward) why = "network is not MinGRU";
    else if (mw->num_layers > BR_MAX_LAYERS) why = "too many MinGRU layers";
    else if (1 + pufferl->num_frozen_banks > BR_MAX_GROUPS) why = "too many frozen banks";
    else if (pufferl->policy.num_atns > SAMPLE_MAX_LOGITS) why = "too many logits for the warp sampler";
    for (int b = 0; !why && b < pufferl->num_frozen_banks; b++) {
        WeightBank* fb = &pufferl->frozen_banks[b];
        if (fb->hidden_size != mw->hidden || fb->num_layers != mw->num_layers
                || (!is_v4 && fb->policy.encoder.forward != bs_forward) || fb->policy.network.forward != mingru_forward)
            why = "a frozen bank differs from the learner in shape";
    }
    if (!why && is_v4) why = br4_check(pufferl);
    if (why) {
        fprintf(stderr, "bopl batched rollout forward disabled: %s\n", why);
        return;
    }
    int apb = vec->total_agents / hypers.num_buffers;
    BrPlan* p = new BrPlan{};
    p->enabled = 1;
    p->H = mw->hidden;
    p->layers = mw->num_layers;
    p->g.n = 1 + pufferl->num_frozen_banks;
    p->g.end[0] = 0;
    for (int k = 0; k < p->g.n; k++)
        p->g.end[k + 1] = pufferl->bank_layout ? pufferl->bank_layout[k + 1] : apb;
    p->rows = p->g.end[p->g.n];
    br_tiles(p->g, p->tiles);
    p->A_total = pufferl->policy.num_atns;  // game logits (sum of the head sizes)
    p->dec_cols = dw->output_dim;           // raw decoder columns (bopl_aim.h)
    {
        const BoplAimConfig aim_cfg = bopl_aim_config();
        const bool aim = p->dec_cols != p->A_total;
        p->aim_fourier = aim ? aim_cfg.fourier : 0;
        p->aim_ar = aim ? aim_cfg.ar : 0;
        assert((!aim || p->dec_cols == aim_cfg.width) && "decoder width does not match BOPL_AIM_FOURIER/BOPL_AIM_AR");
    }
    p->num_atns = pufferl->env.actions.shape[1];
    p->mask_stride = pufferl->env.action_mask.data ? (int)pufferl->env.action_mask.shape[1] : 0;
    // Kernels reading pinned host memory through PCIe are latency-bound and
    // stall the (shared) SMs; DMA uploads of the live rows measured faster in
    // the trainer (eval_gpu 112 vs 127 vs 135 us for zero_copy 0 / 1 / 2 at
    // w128 b4), so the memcpy path is the default.
    p->zero_copy = vec->mapped ? br_env_int("BOPL_ZERO_COPY", 0) : 0;
    p->actions_zero_copy = vec->mapped ? br_env_int("BOPL_ZERO_COPY_ACTIONS", 0) != 0 : 0;
    // Every cudaMemcpyAsync costs ~10 us of host time on the buffer thread
    // (w128 b4: h2d 45 us for four copies, 27 us for two); as graph nodes the
    // copies cost nothing to issue.
    p->graph_copies = p->zero_copy == 0 ? br_env_int("BOPL_GRAPH_COPIES", 1) != 0 : 0;
    p->tf32 = br_env_int("BOPL_TF32", 1) != 0;
    p->sparse = br_env_int("BOPL_OBS_SPARSE", 1) != 0;
    p->ld_dec = br_pad8(p->dec_cols + 1);
    p->acts = (BrActs*)calloc(hypers.num_buffers, sizeof(BrActs));
    int H = p->H, JW = H + BS_JOINED_POOLS;
    auto zalloc = [](float** ptr, size_t n) {
        assert(cudaMalloc((void**)ptr, n * sizeof(float)) == cudaSuccess);
        cudaMemset(*ptr, 0, n * sizeof(float));
    };
    if (is_v4) br4_init(pufferl, p);
    for (int b = 0; !is_v4 && b < hypers.num_buffers; b++) {
        BrActs& a = p->acts[b];
        zalloc(&a.flat, (size_t)apb * BS_FLAT); zalloc(&a.joined, (size_t)apb * JW); zalloc(&a.enc, (size_t)apb * H);
        zalloc(&a.comb, (size_t)apb * 3 * H); zalloc(&a.gru[0], (size_t)apb * H); zalloc(&a.gru[1], (size_t)apb * H);
        zalloc(&a.dec, (size_t)apb * p->ld_dec);
    }
    for (int g = 0; !is_v4 && g < p->g.n; g++) {
        BrWt& w = p->wt[g];
        zalloc(&w.flat, (size_t)br_frag_floats(BS_FLAT, H)); zalloc(&w.fusion, (size_t)br_frag_floats(JW, H));
        for (int l = 0; l < p->layers; l++) zalloc(&w.gru[l], (size_t)br_frag_floats(H, 3 * H));
        zalloc(&w.dec, (size_t)br_frag_floats(H, p->ld_dec));
    }
    vec->zero_copy = p->zero_copy;
    vec->actions_zero_copy = p->actions_zero_copy;
    vec->live_per_buffer = p->rows;
    vec->graph_copies = p->graph_copies;
    if (br4_value_rows(*p) && (p->graph_copies || p->actions_zero_copy)) {
        // The env step waits for the actions alone (br_forward records the
        // event once they are in host memory), not for the critic's branch.
        vec->actions_ready = (cudaEvent_t*)calloc(pufferl->hypers.num_buffers, sizeof(cudaEvent_t));
        for (int b = 0; b < pufferl->hypers.num_buffers; b++)
            assert(cudaEventCreateWithFlags(&vec->actions_ready[b], cudaEventDisableTiming) == cudaSuccess);
    }
    if (p->zero_copy == 0) {
        int nb = pufferl->hypers.num_buffers;
        size_t row_bytes = (size_t)BS_OBS * sizeof(float);
        vec->pack_bytes = p->rows * (int)(2 * sizeof(float) + p->mask_stride);
        vec->pack_host = (char**)calloc(nb, sizeof(char*));
        vec->pack_dev = (char**)calloc(nb, sizeof(char*));
        // The idle tail rows of a buffer are never written or read (compact
        // layout), so the pack lives there when it fits: one memcpy per step.
        vec->pack_in_tail = (size_t)(apb - p->rows) * row_bytes >= (size_t)vec->pack_bytes && br_env_int("BOPL_PACK_TAIL", 1);
        const bool packed_obs = is_v4 && p->graph_copies && vec->mapped;   // the block below re-points pack_* into the packed slots
        for (int b = 0; !packed_obs && b < nb; b++) {
            if (vec->pack_in_tail) {
                size_t off = ((size_t)b * apb + p->rows) * row_bytes;
                vec->pack_host[b] = (char*)vec->observations + off;
                vec->pack_dev[b] = (char*)vec->gpu_observations + off;
            } else {
                cudaHostAlloc((void**)&vec->pack_host[b], vec->pack_bytes, cudaHostAllocPortable);
                cudaMalloc((void**)&vec->pack_dev[b], vec->pack_bytes);
            }
        }
        // Packed observation transport (bopl_obs_pack.h): policy v4 rows go up
        // as 8 KB slots instead of 19 KB dense rows; the unpack kernel rebuilds
        // the dense device row (mode-1 slots from the mapped host row) and
        // bopl4_rays_rows writes the terrain and rays of mode-2 rows.
        if (is_v4 && p->graph_copies && vec->mapped) {
            size_t slot_bytes = (size_t)BOPL4_PACK_P * sizeof(float), n = (size_t)vec->total_agents * slot_bytes;
            assert(cudaHostAlloc((void**)&vec->packed_host, n, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess);
            assert(cudaHostGetDevicePointer((void**)&p->packed_mapped, vec->packed_host, 0) == cudaSuccess);
            assert(cudaMalloc((void**)&vec->packed_dev, n) == cudaSuccess);
            assert(cudaMalloc((void**)&vec->transport_mismatch, sizeof(int)) == cudaSuccess);
            cudaMemset(vec->transport_mismatch, 0, sizeof(int));
            for (long r = 0; r < vec->total_agents; r++) ((unsigned*)(vec->packed_host + r * BOPL4_PACK_P))[0] = BOPL4_PACK_MODE_DENSE;
            cudaMemcpy(vec->packed_dev, vec->packed_host, n, cudaMemcpyHostToDevice);
            bopl4_rays_init(vec->packed_dev, vec->total_agents);   // rows with a scene (modes 2, 3) or a state (4): the GPU writes them (bopl_rays_v4.cu)
            // The reward / terminal / mask pack moves to the idle tail of the
            // buffer's packed slots when it fits there (one memcpy per step).
            vec->pack_in_tail = (size_t)(apb - p->rows) * slot_bytes >= (size_t)vec->pack_bytes && br_env_int("BOPL_PACK_TAIL", 1);
            for (int b = 0; b < nb; b++) {
                if (vec->pack_in_tail) {
                    size_t off = ((size_t)b * apb + p->rows) * slot_bytes;
                    vec->pack_host[b] = (char*)vec->packed_host + off;
                    vec->pack_dev[b] = (char*)vec->packed_dev + off;
                } else if (!vec->pack_host[b]) {
                    cudaHostAlloc((void**)&vec->pack_host[b], vec->pack_bytes, cudaHostAllocPortable);
                    cudaMalloc((void**)&vec->pack_dev[b], vec->pack_bytes);
                }
                bopl_pack_register(vec->envs, vec->buffer_env_starts[b], vec->buffer_env_counts[b], (const float*)vec->observations,
                                   vec->packed_host, (long)b * apb, (long)b * apb + p->rows);
            }
            fprintf(stderr, "bopl batched rollout forward: packed observation rows (%d floats per slot, pack %s)\n",
                    BOPL4_PACK_P, vec->pack_in_tail ? "in the packed tail" : "separate");
        }
        // Reset semantics of the device copies (static_vec_reset): rewards and
        // terminals zero, the mask as the envs left it. bopl_vec_reset refreshes
        // the pack again when it raises every terminal.
        bopl_pack_sync(vec, 3);
    }
    pufferl->br = p;
    fprintf(stderr, "bopl batched rollout forward: %d groups, %d of %d rows per buffer, H=%d, layers=%d, zero_copy=%d (actions %d), tf32=%d, sparse_obs=%d\n",
        p->g.n, p->rows, apb, p->H, p->layers, p->zero_copy, p->actions_zero_copy, p->tf32, p->sparse);
    if (p->graph_copies) fprintf(stderr, "bopl batched rollout forward: memcpys captured in the graph, pack %s\n", vec->pack_in_tail ? "in the idle tail" : "separate");
}

// Group g's policy weights and recurrent state for buffer `buf` (group 0 = the
// learner, reading the actor snapshot when the trainer runs asynchronously).
static void br_group_weights(PuffeRL* pufferl, int g, int buf, PolicyWeights** w, PrecisionTensor** s) {
    if (g == 0) {
        *w = pufferl->num_slots > 1 ? &pufferl->actor_weights : &pufferl->weights;
        if (s) *s = &pufferl->buffer_states[buf];
    } else {
        *w = &pufferl->frozen_banks[g - 1].weights;
        if (s) *s = &pufferl->frozen_banks[g - 1].buffer_states[buf];
    }
}

// Re-transpose every group's GEMM weights (rollout start: the learner snapshot
// was just taken; bank loads happen between rollouts). ~5 MB of traffic once
// per rollout.
static void br_prepare(PuffeRL* pufferl, cudaStream_t stream) {
    BrPlan& p = *pufferl->br;
    if (p.v4) { br4_prepare(pufferl, p, stream); return; }
    int H = p.H, JW = H + BS_JOINED_POOLS;
    BrLinear s_flat = {}, s_fus = {}, s_dec = {}, s_gru[BR_MAX_LAYERS] = {};
    BrLinear d_flat = {}, d_fus = {}, d_dec = {}, d_gru[BR_MAX_LAYERS] = {};
    for (int g = 0; g < p.g.n; g++) {
        PolicyWeights* w; br_group_weights(pufferl, g, 0, &w, nullptr);
        BoplSetWeights* sw = (BoplSetWeights*)w->encoder;
        MinGRUWeights* mw = (MinGRUWeights*)w->network;
        DecoderWeights* dw = (DecoderWeights*)w->decoder;
        s_flat.w[g] = sw->flat.data; d_flat.w[g] = p.wt[g].flat;
        s_fus.w[g] = sw->fusion.data; d_fus.w[g] = p.wt[g].fusion;
        s_dec.w[g] = dw->weight.data; d_dec.w[g] = p.wt[g].dec;
        for (int l = 0; l < p.layers; l++) { s_gru[l].w[g] = mw->weights[l].data; d_gru[l].w[g] = p.wt[g].gru[l]; }
    }
    br_transpose_launch(p, s_flat, d_flat, p.g.n, H, BS_FLAT, H, stream);
    br_transpose_launch(p, s_fus, d_fus, p.g.n, H, JW, H, stream);
    for (int l = 0; l < p.layers; l++) br_transpose_launch(p, s_gru[l], d_gru[l], p.g.n, 3 * H, H, 3 * H, stream);
    br_transpose_launch(p, s_dec, d_dec, p.g.n, p.dec_cols + 1, H, p.ld_dec, stream);
}

static void br_free(PuffeRL* pufferl) {
    BrPlan* p = pufferl->br;
    if (!p) return;
    const bool is_v4 = p->v4 != nullptr;
    br4_free(pufferl, p);
    for (int b = 0; !is_v4 && b < pufferl->hypers.num_buffers; b++) {
        BrActs& a = p->acts[b];
        for (float* ptr : {a.flat, a.joined, a.enc, a.comb, a.gru[0], a.gru[1], a.dec}) cudaFree(ptr);
    }
    for (int g = 0; !is_v4 && g < p->g.n; g++) {
        BrWt& w = p->wt[g];
        cudaFree(w.flat); cudaFree(w.fusion); cudaFree(w.dec);
        for (int l = 0; l < p->layers; l++) cudaFree(w.gru[l]);
    }
    StaticVec* vec = (StaticVec*)pufferl->vec;
    if (vec && vec->actions_ready) {
        for (int b = 0; b < pufferl->hypers.num_buffers; b++) cudaEventDestroy(vec->actions_ready[b]);
        free(vec->actions_ready); vec->actions_ready = nullptr;
    }
    if (vec && vec->pack_host) {
        if (!vec->pack_in_tail)
            for (int b = 0; b < pufferl->hypers.num_buffers; b++) { cudaFreeHost(vec->pack_host[b]); cudaFree(vec->pack_dev[b]); }
        free(vec->pack_host); free(vec->pack_dev); vec->pack_host = vec->pack_dev = nullptr;
    }
    if (vec && vec->packed_host) {   // the envs are closed before this (their pack pointers die with them)
        cudaFreeHost(vec->packed_host); cudaFree(vec->packed_dev); cudaFree(vec->transport_mismatch);
        vec->packed_host = vec->packed_dev = nullptr; vec->transport_mismatch = nullptr;
    }
    free(p->acts);
    delete p;
    pufferl->br = nullptr;
}

// One buffer step: the live rows of buffer `buf` through the batched chain on
// `stream` (captured into the step's cudagraph like the per-bank chain).
static void br_forward(PuffeRL* pufferl, int buf, int t, cudaStream_t stream) {
    BrPlan& p = *pufferl->br;
    HypersT& hypers = pufferl->hypers;
    StaticVec* vec = pufferl->vec;
    RolloutBuf rollouts = rollout_slot_view(pufferl->rollouts, pufferl->write_slot, hypers.horizon);
    int apb = vec->total_agents / hypers.num_buffers;
    int start = buf * apb, rows = p.rows, H = p.H, JW = H + BS_JOINED_POOLS;
    BrActs& A = p.acts[buf];

    // Env buffers: mapped host memory (zero_copy), the packed device staging
    // copy (zero_copy 0, bopl_upload) or the legacy device copies.
    const float* obs = (const float*)(p.zero_copy >= 2 ? vec->dev_observations : vec->gpu_observations) + (long)start * BS_OBS;
    const float* rewards = (p.zero_copy ? vec->dev_rewards : vec->gpu_rewards) + start;
    const float* terminals = (p.zero_copy ? vec->dev_terminals : vec->gpu_terminals) + start;
    const unsigned char* mask = p.mask_stride
        ? (p.zero_copy ? vec->dev_action_mask : vec->gpu_action_mask) + (long)start * p.mask_stride : nullptr;
    if (!p.zero_copy && vec->pack_dev) {
        const char* pk = vec->pack_dev[buf];
        rewards = (const float*)pk;
        terminals = (const float*)(pk + rows * sizeof(float));
        mask = p.mask_stride ? (const unsigned char*)(pk + 2 * rows * sizeof(float)) : nullptr;
    }
    float* env_actions = (p.actions_zero_copy ? vec->dev_actions : vec->gpu_actions) + (long)start * p.num_atns;
    if (p.graph_copies && vec->packed_host) {
        // Packed rows: one 2D copy of the head of every slot of the buffer,
        // then the dense device rows rebuilt for the chain (the unpack reads
        // the rest of each slot's used words, and the pack in their tail,
        // from the mapped host slots).
        size_t slot_bytes = (size_t)BOPL4_PACK_P * sizeof(float), off = (size_t)start * slot_bytes;
        cudaMemcpy2DAsync((char*)vec->packed_dev + off, slot_bytes, (const char*)vec->packed_host + off, slot_bytes,
                          (size_t)BR4_PACK_DMA_WORDS * sizeof(float), rows, cudaMemcpyHostToDevice, stream);
        if (!vec->pack_in_tail) cudaMemcpyAsync(vec->pack_dev[buf], vec->pack_host[buf], vec->pack_bytes, cudaMemcpyHostToDevice, stream);
        const int tail = vec->pack_in_tail ? vec->pack_bytes : 0;
        // The dense row is rebuilt straight into the rollout tensor's rows for
        // this step (the trainer reads them there) and the chain reads it from
        // there too: no device copy of the observation row (br4_sets used to
        // copy gpu_observations -> rollout_obs, 19 KB per row per step).
        const float* host_rows = (const float*)vec->dev_observations + (long)start * BOPL_OBS_STRIDE;
        float* dense = puf_slice(rollouts.observations, t, start, rows).data;
        bopl4_unpack_rows<<<bopl4_unpack_blocks(rows, tail), BR4_UNPACK_THREADS, 0, stream>>>(
            p.packed_mapped + (long)start * BOPL4_PACK_P, vec->packed_dev + (long)start * BOPL4_PACK_P, host_rows, dense, rows, tail);
        // The terrain and rays of the rows with a scene. (The rays on a side
        // stream beside br4_sets, joined before the flat GEMM, measured no
        // faster, 2026-10-06; again with br4_sets at 23.5 us, 2026-10-07
        // (fwd_bench): 1 stream -5.5 us, 4 streams at once +2 us; the rays
        // slow from 38 to 45 us beside the items and sets.)
        bopl4_rays_launch(vec->packed_dev + (long)start * BOPL4_PACK_P, dense, rows, stream);
        if (p.transport_check) bopl4_transport_check<<<rows, BR4_UNPACK_THREADS, 0, stream>>>(dense, host_rows, rows, vec->transport_mismatch);
        obs = dense;
    } else if (p.graph_copies) {
        size_t row_bytes = (size_t)BS_OBS * sizeof(float), off = (size_t)start * row_bytes;
        size_t bytes = (size_t)rows * row_bytes + (vec->pack_in_tail ? (size_t)vec->pack_bytes : 0);
        cudaMemcpyAsync((char*)vec->gpu_observations + off, (const char*)vec->observations + off, bytes, cudaMemcpyHostToDevice, stream);
        if (!vec->pack_in_tail) cudaMemcpyAsync(vec->pack_dev[buf], vec->pack_host[buf], vec->pack_bytes, cudaMemcpyHostToDevice, stream);
    }

    PrecisionTensor obs_dst = puf_slice(rollouts.observations, t, start, rows);
    PrecisionTensor rew_dst = puf_slice(rollouts.rewards, t, start, rows);
    PrecisionTensor term_dst = puf_slice(rollouts.terminals, t, start, rows);
    PrecisionTensor act_dst = puf_slice(rollouts.actions, t, start, rows);
    PrecisionTensor lp_dst = puf_slice(rollouts.logprobs, t, start, rows);
    PrecisionTensor val_dst = puf_slice(rollouts.values, t, start, rows);
    PrecisionTensor mask_dst = {};
    if (p.mask_stride) mask_dst = puf_slice(rollouts.action_mask, t, start, rows);
    const float* dec_rows = A.dec;
    br_check_pre(pufferl, buf, stream);
    if (p.v4) {
        br4_chain(pufferl, p, buf, obs, rewards, terminals, obs_dst.data, rew_dst.data, term_dst.data, val_dst.data, stream);
        dec_rows = br4_dec_rows(p, buf);
    } else {
    // Per-group weights (set-encoder MLPs as stored, GEMM weights transposed by
    // br_prepare) and recurrent state (group 0 = learner snapshot).
    BrLinear w_player = {}, w_slot = {}, w_entity = {}, w_slime = {}, w_embedding = {};
    BrLinear w_flat = {}, w_fusion = {}, w_dec = {}, w_gru[BR_MAX_LAYERS] = {};
    BrGateArgs gate = {};
    for (int g = 0; g < p.g.n; g++) {
        PolicyWeights* w; PrecisionTensor* s;
        br_group_weights(pufferl, g, buf, &w, &s);
        BoplSetWeights* sw = (BoplSetWeights*)w->encoder;
        w_player.w[g] = sw->player.data; w_slot.w[g] = sw->slot.data;
        w_entity.w[g] = sw->entity.data; w_slime.w[g] = sw->slime.data;
        w_embedding.w[g] = sw->embedding.data;
        w_flat.w[g] = p.wt[g].flat; w_fusion.w[g] = p.wt[g].fusion;
        for (int l = 0; l < p.layers; l++) w_gru[l].w[g] = p.wt[g].gru[l];
        w_dec.w[g] = p.wt[g].dec;
        gate.state[g] = s->data;
        gate.state_rows[g] = (int)s->shape[1];
    }

    BrEncodeArgs enc = {};
    enc.obs = obs; enc.rewards = rewards; enc.terminals = terminals;
    enc.rollout_obs = obs_dst.data; enc.rollout_rewards = rew_dst.data; enc.rollout_terminals = term_dst.data;
    enc.flat_input = A.flat; enc.joined = A.joined; enc.H = H; enc.rows = rows; enc.sparse = p.sparse; enc.g = p.g;
    enc.w_player = w_player; enc.w_slot = w_slot; enc.w_entity = w_entity;
    enc.w_slime = w_slime; enc.w_embedding = w_embedding;
    br_encode<<<rows, BR_ENCODE_THREADS, 0, stream>>>(enc);
    br_encode_entity<<<rows, BS_E, 0, stream>>>(enc);

    br_gemm_launch(p, A.flat, BS_FLAT, A.joined, JW, BS_FLAT, H, H, 1, w_flat, stream);
    br_gemm_launch(p, A.joined, JW, A.enc, H, JW, H, H, 1, w_fusion, stream);
    const float* x = A.enc;
    for (int l = 0; l < p.layers; l++) {
        br_gemm_launch(p, x, H, A.comb, 3 * H, H, 3 * H, 3 * H, 0, w_gru[l], stream);
        gate.combined = A.comb; gate.x = x; gate.terminals = terminals;
        gate.out = A.gru[l & 1]; gate.layer = l; gate.H = H; gate.rows = rows; gate.g = p.g;
        br_gate<<<grid_size(rows * H), BLOCK_SIZE, 0, stream>>>(gate);
        x = A.gru[l & 1];
    }
    br_gemm_launch(p, x, H, A.dec, p.ld_dec, H, p.dec_cols + 1, p.ld_dec, 0, w_dec, stream);
    }  // v3 chain
    br_check_post(pufferl, buf, t, terminals, dec_rows, stream);

    int hc_stride = 0;
    const signed char* hc_dev = get_head_consume_dev(&hc_stride);
    BrSampleArgs smp = {};
    smp.dec = dec_rows; smp.ld_dec = p.ld_dec; smp.A_total = p.A_total;
    smp.dec_cols = p.dec_cols; smp.aim_fourier = p.aim_fourier; smp.aim_ar = p.aim_ar;
    smp.act_sizes = pufferl->act_sizes_puf.data; smp.num_atns = p.num_atns;
    smp.value_rows = br4_value_rows(p);
    smp.env_mask = mask; smp.mask_stride = p.mask_stride; smp.rollout_mask = mask_dst.data;
    smp.env_actions = env_actions; smp.rollout_actions = act_dst.data;
    smp.logprobs = lp_dst.data; smp.values = val_dst.data;
    smp.rng = pufferl->rng_states[buf];
    smp.head_consume = hc_dev; smp.hc_stride = hc_stride; smp.rows = rows;
    br_launch_pdl(br_sample, dim3((rows + SAMPLE_WARP_ROWS - 1) / SAMPLE_WARP_ROWS), dim3(SAMPLE_WARP_ROWS * 32), 0, stream, smp);
    if (p.graph_copies && !p.actions_zero_copy)
        cudaMemcpyAsync(vec->actions + (long)start * p.num_atns, vec->gpu_actions + (long)start * p.num_atns,
                        (size_t)rows * p.num_atns * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (vec->actions_ready) br_record_external(vec->actions_ready[buf], stream);
    br4_join(p, buf, stream);
}
#include "bopl_rollout_v4.cu"

// The batched chain's decoder rows on given observations (check_dump.py via
// _C.bopl_chain_rows): the same br4_chain call br_forward makes, for buffer 0,
// on a scratch copy of the rows (the learner's first `count` rows; the other
// rows of the buffer zero) from a zero recurrent state, with the weights the
// trainer holds now (the actor snapshot is refreshed and re-transposed first,
// as rollouts_start does, so a checkpoint loaded with load_weights is what the
// chain runs). Returns count x (dec_cols + 1) floats: logits, then the value.
// Buffer 0's learner state is zeroed and left so: for a stopped trainer only.
// This is the rollout chain's own gate; bopl_policy_rows drives the eager
// per-layer path, which shares no kernel with it.
static void br_chain_rows(PuffeRL* pufferl, const float* host_obs, int count, int obs_width, float* host_dec) {
    if (!pufferl->br || !pufferl->br->v4) throw std::runtime_error("bopl_chain_rows: the v4 batched chain is not active");
    BrPlan& p = *pufferl->br;
    if (pufferl->rollout_in_flight) throw std::runtime_error("bopl_chain_rows: a rollout is in flight");
    if (count < 1 || count > p.g.end[1] - p.g.end[0]) throw std::runtime_error("bopl_chain_rows: count exceeds the learner's rows");
    if (obs_width > BOPL_OBS_STRIDE) throw std::runtime_error("bopl_chain_rows: observation width exceeds the row stride");
    cudaStream_t stream = 0;
    int rows = p.rows;
    if (pufferl->num_slots > 1)
        cudaMemcpyAsync(pufferl->actor_param_puf.data, pufferl->param_puf.data,
                        numel(pufferl->param_puf.shape) * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
    br_prepare(pufferl, stream);
    float *obs, *rewards, *terminals, *rew_copy, *term_copy, *values;
    size_t obs_bytes = (size_t)rows * BOPL_OBS_STRIDE * sizeof(float), row_bytes = (size_t)rows * sizeof(float);
    cudaMalloc((void**)&obs, obs_bytes); cudaMemsetAsync(obs, 0, obs_bytes, stream);
    cudaMalloc((void**)&rewards, row_bytes); cudaMemsetAsync(rewards, 0, row_bytes, stream);
    cudaMalloc((void**)&terminals, row_bytes); cudaMemsetAsync(terminals, 0, row_bytes, stream);
    cudaMalloc((void**)&rew_copy, row_bytes); cudaMalloc((void**)&term_copy, row_bytes); cudaMalloc((void**)&values, row_bytes);
    cudaMemcpy2DAsync(obs + (long)p.g.end[0] * BOPL_OBS_STRIDE, BOPL_OBS_STRIDE * sizeof(float), host_obs, (size_t)obs_width * sizeof(float),
                      (size_t)obs_width * sizeof(float), count, cudaMemcpyHostToDevice, stream);
    PrecisionTensor* state = &pufferl->buffer_states[0];
    cudaMemsetAsync(state->data, 0, (size_t)numel(state->shape) * sizeof(float), stream);
    br4_chain(pufferl, p, 0, obs, rewards, terminals, obs, rew_copy, term_copy, values, stream);
    br4_join(p, 0, stream);
    const float* dec = br4_dec_rows(p, 0) + (long)p.g.end[0] * p.ld_dec;
    cudaMemcpy2DAsync(host_dec, (size_t)(p.dec_cols + 1) * sizeof(float), dec, (size_t)p.ld_dec * sizeof(float),
                      (size_t)(p.dec_cols + 1) * sizeof(float), count, cudaMemcpyDeviceToHost, stream);
    cudaError_t e = cudaStreamSynchronize(stream);
    cudaFree(obs); cudaFree(rewards); cudaFree(terminals); cudaFree(rew_copy); cudaFree(term_copy); cudaFree(values);
    if (e != cudaSuccess) throw std::runtime_error(std::string("bopl_chain_rows: ") + cudaGetErrorString(e));
}
#endif  // BR_KERNELS_ONLY
