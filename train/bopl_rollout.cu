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
//               in B-fragment order (BOPL_TF32=1) or fp32 FMAs (=0); 16-row
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
// m16n8k8 tensor-core steps, inputs rounded to tf32, fp32 accumulate; the
// weights are stored in the B-fragment order (br_transpose_frag) so a lane's
// operands are one float4 and a warp's load is 512 contiguous bytes. fp32
// (BOPL_TF32=0): CUDA-core FMAs over the plain transposed weights (K, ldb), a
// thread owns one column and all 16 rows. Row tiles never straddle a group
// (tiles[] = cumulative 16-row tiles per group). K must be even.
#define BR_TM 16
#define BR_TN 32
#define BR_TK 64
#define BR_KG 8
#define BR_THREADS (BR_TN * BR_KG)
struct BrGemmArgs {
    const float* in; int ld_in;
    float* out; int ld_out;
    int K, N, ldb, relu;  // ldb: fp32 = row stride of the (K, ldb) transposed weights; tf32 = n-tiles (ceil(N/8))
    int units;            // column-subtile rounds per block: a block covers units * (8 / sub) subtiles
    BrGroups g;
    BrLinear wt;  // per-group weights: fp32 (K, ldb) transposed; tf32 fragment order (see br_frag_index)
    int tiles[BR_MAX_GROUPS + 1];
};

// TF32 fragment-order weight layout: element (k, n) of the (K, N) operand B.
// mma.m16n8k8 B fragment: lane = 4*groupID + tid4 holds b0 = B[tid4][groupID],
// b1 = B[tid4 + 4][groupID] of an 8x8 (k, n) tile; two k-steps (16 k) of one
// n-tile form the lane's float4 {b0(s0), b1(s0), b0(s1), b1(s1)}.
__host__ __device__ __forceinline__ long br_frag_index(int k, int n, int ntiles) {
    int kstep = k >> 3, kk = k & 7, ntile = n >> 3, gid = n & 7;
    int kpair = kstep >> 1, s = kstep & 1, tid4 = kk & 3, half = kk >> 2;
    return (((long)kpair * ntiles + ntile) * 32 + gid * 4 + tid4) * 4 + s * 2 + half;
}
__host__ __device__ __forceinline__ long br_frag_floats(int K, int N) {
    return (long)((K + 15) / 16) * 16 * ((N + 7) / 8) * 8;
}

__device__ __forceinline__ unsigned br_tf32(float x) {
    return __float_as_uint(nvcuda::wmma::__float_to_tf32(x));
}
__device__ __forceinline__ void br_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The GEMM's block body as a device function: row tile rt of the grouped
// problem a, column-subtile groups ctile0 .. ctile0 + units - 1 (a group is
// the BR_KG / sub subtiles one round of the block covers), A tiles in As. The
// row-tile / column-tile arithmetic is the caller's, so the same body serves
// br_gemm (one block per (rt, ctile)) and a kernel that runs several GEMMs
// of a chain in sequence for one row tile. pdl: griddepcontrol.wait before
// the first read of A (the predecessor's output) and launch_dependents under
// the last epilogue; a chain kernel orders its stages itself and passes
// false. The values are those of br_gemm as it was launched per GEMM: the
// same chunk -> warp assignment, mma order and partial-sum order for any
// (ctile0, units) split, so the sums are bit-identical to the one-round grid.
template <bool TF32>
__device__ __forceinline__ void br_gemm_tile(const BrGemmArgs& a, int rt, int ctile0, int units, float (*As)[BR_TM * BR_TK], bool pdl) {
    // PDL: the B operand (weights) does not depend on the predecessor, so
    // the first chunk's B loads are issued before griddepcontrol.wait and
    // their latency overlaps the predecessor's tail; A (the predecessor's
    // output) is read after it. A warp with no chunk waits after the loop.
    bool waited = !pdl;
    int grp = 0;
    while (grp + 1 < a.g.n && rt >= a.tiles[grp + 1]) grp++;
    const int row0 = a.g.end[grp] + (rt - a.tiles[grp]) * BR_TM;
    const int row_end = a.g.end[grp + 1];
    const int K = a.K, N = a.N, ldb = a.ldb;
    const int nchunks = (K + BR_TK - 1) / BR_TK;
    // sub = warps per column subtile (power of two >= nchunks, <= BR_KG).
    int sub = 1;
    while (sub < nchunks && sub < BR_KG) sub <<= 1;
    const int subtiles = BR_KG / sub;
    const int tid = threadIdx.x, warp = tid / BR_TN, c = tid % BR_TN;
    const int kg = warp % sub;  // this warp's K slot within its subtile
    const float* __restrict__ Wt = a.wt.w[grp];
    float* Aw = As[warp];
    // Rounds over the block's column subtiles (units of them per warp): the
    // same chunk -> warp assignment and partial-sum order as one round per
    // block, so the sums are bit-identical for any units; fewer, longer blocks
    // hold fewer SMs while the training pass runs beside the rollout.
    for (int u = 0; u < units; u++) {
    const int col0 = ((ctile0 + u) * subtiles + warp / sub) * BR_TN;

    // A chunk: 16 rows x 64 k = 256 float4 over the warp (element i: row i%16,
    // k-quad i/16), 8 instructions per chunk (the float2 form took 16). The
    // lanes stay along the rows: the fragment layout's row stride is 16
    // floats, so lanes spanning rows and k-halves keep the smem stores at
    // 4-way conflicts (lanes along k would make them 8-way). K is even; the
    // tail quad past K is masked per pair.
    // A chunk: 16 rows x 64 k = 256 float4 over the warp (element i: row i%16,
    // k-quad i/16), 8 instructions per chunk (the float2 form took 16). The
    // lanes stay along the rows: the fragment layout's row stride is 16
    // floats, so lanes spanning rows and k-halves keep the smem stores at
    // 4-way conflicts (lanes along k would make them 8-way). K is even; the
    // tail quad past K is masked per pair.
    float4 av[BR_TK / 8];
    // float4 row loads when the rows are 16-byte aligned (every v4 input; v3's
    // flat block has a stride of 98), scalar loads otherwise: the same values.
    // (An alignment abort here instead broke v3; check_dump caught it.)
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
    // TF32 layout: A fragment order [kstep][lane][4] (a0 = A[gid][tid4], a1 = A[gid+8][tid4],
    // a2 = A[gid][tid4+4], a3 = A[gid+8][tid4+4]) so a lane's k-step operand is one float4.
    auto store_a = [&]() {
        #pragma unroll
        for (int j = 0; j < BR_TK / 8; j++) {
            int i = c + j * 32, r = i & 15, k = (i >> 4) << 2;
            const float v[4] = {av[j].x, av[j].y, av[j].z, av[j].w};
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int kq = k + q;
                if (TF32) {
                    int kstep = kq >> 3, kk = kq & 7, lane = (r & 7) * 4 + (kk & 3), slot = (r >> 3) + 2 * (kk >> 2);
                    Aw[(kstep * 32 + lane) * 4 + slot] = v[q];
                } else {
                    Aw[kq * BR_TM + r] = v[q];
                }
            }
        }
    };
    if constexpr (TF32) {
        const int lane = c;
        const int gid = lane >> 2, tid4 = lane & 3;
        const int ntiles = ldb, ntile0 = col0 >> 3;
        float acc[4][4];
        #pragma unroll
        for (int t = 0; t < 4; t++) { acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f; }
        // B chunk: 4 k-pairs (64 k) x 4 n-tiles (32 columns), one float4 per lane each,
        // a warp instruction reads 512 contiguous bytes. k-pairs past K are zero padding
        // in the fragment buffer (br_frag_floats rounds K up to 16). n-tiles past ntiles
        // (the block's column span exceeds N, e.g. narrow decoders) are outside the
        // buffer: skip them, their columns are dropped by the epilogue anyway.
        float4 bf[4][4];
        auto load_b = [&](int k0) {
            #pragma unroll
            for (int kp = 0; kp < 4; kp++)
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    long kpair = (k0 >> 4) + kp;
                    bf[kp][t] = (ntile0 + t < ntiles)
                        ? *(const float4*)(Wt + ((kpair * ntiles + ntile0 + t) * 32 + lane) * 4)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
                }
        };
        for (int ch = kg; ch < nchunks; ch += sub) {
            load_b(ch * BR_TK);
            if (!waited) { br_pdl_wait(); waited = true; }
            load_a(ch * BR_TK);
            __syncwarp();
            store_a();
            __syncwarp();
            #pragma unroll
            for (int ks = 0; ks < BR_TK / 8; ks++) {
                float4 av4 = *(const float4*)(Aw + (ks * 32 + lane) * 4);
                unsigned a[4] = {br_tf32(av4.x), br_tf32(av4.y), br_tf32(av4.z), br_tf32(av4.w)};
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    float4 b = bf[ks >> 1][t];
                    unsigned b0 = br_tf32((ks & 1) ? b.z : b.x), b1 = br_tf32((ks & 1) ? b.w : b.y);
                    br_mma(acc[t], a, b0, b1);
                }
            }
        }
        __syncthreads();  // every warp is done with its A tile: reuse as C tiles
        float* Cw = As[warp];  // [16][32]: c0 (gid, 2 tid4), c1 (gid, 2 tid4 + 1), c2/c3 at row gid + 8
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            Cw[gid * BR_TN + t * 8 + tid4 * 2] = acc[t][0];
            Cw[gid * BR_TN + t * 8 + tid4 * 2 + 1] = acc[t][1];
            Cw[(gid + 8) * BR_TN + t * 8 + tid4 * 2] = acc[t][2];
            Cw[(gid + 8) * BR_TN + t * 8 + tid4 * 2 + 1] = acc[t][3];
        }
    } else {
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
        __syncthreads();
        float* Cw = As[warp];
        #pragma unroll
        for (int r = 0; r < BR_TM; r++) Cw[r * BR_TN + c] = acc[r];
    }
    if (!waited) { br_pdl_wait(); waited = true; }
    __syncthreads();
    if (pdl && u == units - 1) br_pdl_release();   // the dependent launches under this epilogue
    // Per subtile, sum its warps' partial tiles in warp order and write the valid entries.
    for (int o = tid; o < subtiles * BR_TM * BR_TN; o += BR_THREADS) {
        int st = o / (BR_TM * BR_TN), e = o % (BR_TM * BR_TN), r = e / BR_TN, cc = e % BR_TN;
        float v = 0.f;
        for (int w = st * sub; w < (st + 1) * sub; w++) v += As[w][r * BR_TN + cc];
        int row = row0 + r, cl = ((ctile0 + u) * subtiles + st) * BR_TN + cc;
        if (row < row_end && cl < N) {
            if (a.relu) v = fmaxf(0.f, v);
            a.out[(long)row * a.ld_out + cl] = v;
        }
    }
    __syncthreads();  // the C tiles are read before the next round's A tiles overwrite them
    }
}

template <bool TF32>
__global__ void __launch_bounds__(BR_THREADS, 2) br_gemm(BrGemmArgs a) {
    // Per-warp A tile (16 x 64): TF32 in A-fragment order [kstep][lane][4],
    // fp32 as [k][row] (float4 row reads). The 16 x 32 C tiles alias it after
    // the K loop.
    __shared__ __align__(16) float As[BR_KG][BR_TM * BR_TK];  // 32 KB
    br_gemm_tile<TF32>(a, blockIdx.x, blockIdx.y * a.units, a.units, As, true);
}

// Column-tile count of br_gemm's grid for a GEMM of depth K over N columns,
// and the rounds per block: 1, one subtile set per block (more rounds per
// block shrink the grid u-fold and were measured a wash: +1% wall, +5%
// rollout time; kept as a parameter of the kernel, fixed here).
static int br_gemm_units() { return 1; }
static int br_gemm_col_tiles(int K, int N, int* units) {
    int nchunks = (K + BR_TK - 1) / BR_TK, sub = 1;
    while (sub < nchunks && sub < BR_KG) sub <<= 1;
    int cols = (BR_KG / sub) * BR_TN;
    int tiles = (N + cols - 1) / cols;
    *units = std::min(br_gemm_units(), tiles);
    return (tiles + *units - 1) / *units;
}

// TF32: dst_g = src_g (N, K) in B-fragment order (br_frag_index), zero padded to
// 16 k x 8 n; one thread per destination element.
__global__ void br_transpose_frag(BrLinear src, BrLinear dst, int N, int K) {
    const float* s = src.w[blockIdx.z];
    float* d = (float*)dst.w[blockIdx.z];
    int ntiles = (N + 7) / 8, Kp = (K + 15) / 16 * 16, Np = ntiles * 8;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)Kp * Np) return;
    int k = (int)(i / Np), n = (int)(i % Np);
    d[br_frag_index(k, n, ntiles)] = (k < K && n < N) ? s[(long)n * K + k] : 0.f;
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
    if (p.tf32) a.ldb = (N + 7) / 8;  // n-tiles of the fragment layout
    a.g = p.g; a.wt = wt;
    memcpy(a.tiles, p.tiles, sizeof(a.tiles));
    dim3 grid(p.tiles[p.g.n], br_gemm_col_tiles(K, N, &a.units));
    if (p.tf32) br_launch_pdl(br_gemm<true>, grid, dim3(BR_THREADS), 0, stream, a);
    else br_launch_pdl(br_gemm<false>, grid, dim3(BR_THREADS), 0, stream, a);
}

// dst[g] = the GEMM-side layout of src[g] (N, K) for groups 0..n-1: tf32 fragment
// order, or the (K, ldt) transpose for the fp32 path.
static void br_transpose_launch(const BrPlan& p, const BrLinear& src, const BrLinear& dst, int n, int N, int K, int ldt, cudaStream_t stream) {
    if (p.tf32) {
        long total = br_frag_floats(K, N);
        br_transpose_frag<<<dim3((unsigned)((total + 255) / 256), 1, n), 256, 0, stream>>>(src, dst, N, K);
    } else {
        dim3 grid((K + 31) / 32, (N + 31) / 32, n);
        br_transpose<<<grid, dim3(32, 8), 0, stream>>>(src, dst, N, K, ldt);
    }
}

constexpr int BR4_UNPACK_THREADS = 256;   // bopl4_unpack_rows (bopl_rollout_v4.cu)
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
__global__ void bopl4_unpack_rows(const float* __restrict__ packed, const float* __restrict__ host_rows, float* __restrict__ dense, int rows);
__global__ void bopl4_transport_check(const float* __restrict__ dense, const float* __restrict__ host_rows, int rows, int* mismatches);
extern "C" void bopl4_rays_init(const float* packed, long slots);
extern "C" void bopl4_rays_launch(const float* packed, float* dense, int rows, cudaStream_t stream);
extern "C" void bopl_pack_register(void* envs, int start, int count, const float* base, float* out, long row_begin, long row_end);
static void br_check_pre(PuffeRL* pufferl, int buf, cudaStream_t stream);
static void br_check_post(PuffeRL* pufferl, int buf, int t, const float* terminals, const float* dec_rows, cudaStream_t stream);
static void br4_chain(PuffeRL* pufferl, BrPlan& p, int buf, const float* obs, const float* rewards, const float* terminals,
                      float* rollout_obs, float* rollout_rewards, float* rollout_terminals, cudaStream_t stream);
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
    br_pdl_wait(); br_pdl_release();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int idx = blockIdx.x * SAMPLE_WARP_ROWS + warp;
    if (idx >= a.rows) return;
    const int A_total = a.A_total;
    const float* logits = a.dec + (long)idx * a.ld_dec;
    const unsigned char* mask = a.env_mask ? a.env_mask + (long)idx * a.mask_stride : nullptr;
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
    // Stage game logits [g0, g1) from the raw row (bopl_aim.h; identity unless aim).
    auto stage = [&](int g0, int g1, int buttons) {
        #pragma unroll
        for (int it = 0; it < MASK_ITERS; it++) {
            int i = lane + it * 32;
            if (i < g0 || i >= g1) continue;
            float l = aim ? bopl_game_logit(logits, a.aim_fourier, a.aim_ar, buttons, i) : logits[i];
            l = finite_or_clamp(l, -1.0e4f, 1.0e4f);
            if (mask && mv[it] == 0) l = -1e4f;
            row[i] = l;
        }
        __syncwarp();
    };

    curandStatePhilox4_32_10_t state;
    if (lane == 0) state = a.rng[idx];
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
        stage(BOPL_ACT_AIM, A_total, 0);
        sample_head(1);
        stage(0, BOPL_ACT_AIM, sampled[1]);
        sample_head(0);
        for (int h = 2; h < a.num_atns; ++h) sample_head(h);
    } else {
        stage(0, A_total, 0);
        for (int h = 0; h < a.num_atns; ++h) sample_head(h);
    }
    if (lane == 0) {
        a.logprobs[idx] = total_log_prob;
        a.values[idx] = logits[a.dec_cols];
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
            assert(cudaHostAlloc((void**)&vec->packed_host, n, cudaHostAllocPortable) == cudaSuccess);
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
        // Packed rows: one memcpy of the buffer's slots (+ the pack in their
        // tail), then the dense device rows rebuilt for the chain.
        size_t slot_bytes = (size_t)BOPL4_PACK_P * sizeof(float), off = (size_t)start * slot_bytes;
        size_t bytes = (size_t)rows * slot_bytes + (vec->pack_in_tail ? (size_t)vec->pack_bytes : 0);
        cudaMemcpyAsync((char*)vec->packed_dev + off, (const char*)vec->packed_host + off, bytes, cudaMemcpyHostToDevice, stream);
        if (!vec->pack_in_tail) cudaMemcpyAsync(vec->pack_dev[buf], vec->pack_host[buf], vec->pack_bytes, cudaMemcpyHostToDevice, stream);
        // The dense row is rebuilt straight into the rollout tensor's rows for
        // this step (the trainer reads them there) and the chain reads it from
        // there too: no device copy of the observation row (br4_sets used to
        // copy gpu_observations -> rollout_obs, 19 KB per row per step).
        const float* host_rows = (const float*)vec->dev_observations + (long)start * BOPL_OBS_STRIDE;
        float* dense = puf_slice(rollouts.observations, t, start, rows).data;
        bopl4_unpack_rows<<<rows, BR4_UNPACK_THREADS, 0, stream>>>(vec->packed_dev + (long)start * BOPL4_PACK_P, host_rows, dense, rows);
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
        br4_chain(pufferl, p, buf, obs, rewards, terminals, obs_dst.data, rew_dst.data, term_dst.data, stream);
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
    smp.env_mask = mask; smp.mask_stride = p.mask_stride; smp.rollout_mask = mask_dst.data;
    smp.env_actions = env_actions; smp.rollout_actions = act_dst.data;
    smp.logprobs = lp_dst.data; smp.values = val_dst.data;
    smp.rng = pufferl->rng_states[buf];
    smp.head_consume = hc_dev; smp.hc_stride = hc_stride; smp.rows = rows;
    br_launch_pdl(br_sample, dim3((rows + SAMPLE_WARP_ROWS - 1) / SAMPLE_WARP_ROWS), dim3(SAMPLE_WARP_ROWS * 32), 0, stream, smp);
    if (p.graph_copies && !p.actions_zero_copy)
        cudaMemcpyAsync(vec->actions + (long)start * p.num_atns, vec->gpu_actions + (long)start * p.num_atns,
                        (size_t)rows * p.num_atns * sizeof(float), cudaMemcpyDeviceToHost, stream);
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
    float *obs, *rewards, *terminals, *rew_copy, *term_copy;
    size_t obs_bytes = (size_t)rows * BOPL_OBS_STRIDE * sizeof(float), row_bytes = (size_t)rows * sizeof(float);
    cudaMalloc((void**)&obs, obs_bytes); cudaMemsetAsync(obs, 0, obs_bytes, stream);
    cudaMalloc((void**)&rewards, row_bytes); cudaMemsetAsync(rewards, 0, row_bytes, stream);
    cudaMalloc((void**)&terminals, row_bytes); cudaMemsetAsync(terminals, 0, row_bytes, stream);
    cudaMalloc((void**)&rew_copy, row_bytes); cudaMalloc((void**)&term_copy, row_bytes);
    cudaMemcpy2DAsync(obs + (long)p.g.end[0] * BOPL_OBS_STRIDE, BOPL_OBS_STRIDE * sizeof(float), host_obs, (size_t)obs_width * sizeof(float),
                      (size_t)obs_width * sizeof(float), count, cudaMemcpyHostToDevice, stream);
    PrecisionTensor* state = &pufferl->buffer_states[0];
    cudaMemsetAsync(state->data, 0, (size_t)numel(state->shape) * sizeof(float), stream);
    br4_chain(pufferl, p, 0, obs, rewards, terminals, obs, rew_copy, term_copy, stream);
    const float* dec = br4_dec_rows(p, 0) + (long)p.g.end[0] * p.ld_dec;
    cudaMemcpy2DAsync(host_dec, (size_t)(p.dec_cols + 1) * sizeof(float), dec, (size_t)p.ld_dec * sizeof(float),
                      (size_t)(p.dec_cols + 1) * sizeof(float), count, cudaMemcpyDeviceToHost, stream);
    cudaError_t e = cudaStreamSynchronize(stream);
    cudaFree(obs); cudaFree(rewards); cudaFree(terminals); cudaFree(rew_copy); cudaFree(term_copy);
    if (e != cudaSuccess) throw std::runtime_error(std::string("bopl_chain_rows: ") + cudaGetErrorString(e));
}
#endif  // BR_KERNELS_ONLY
