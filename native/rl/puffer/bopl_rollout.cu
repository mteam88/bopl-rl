// Bopl batched rollout forward (THROUGHPUT.md 3.C, BOPL_BATCHED_FWD=1).
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

#define BR_MAX_GROUPS 9   // learner + up to 8 frozen banks
#define BR_MAX_LAYERS 4

// Rollout encoder tiling (the original bs_fused_forward layout: 12 slot
// groups x 8 channels per BS_FWD_TPR=96-thread row). The training encoder in
// bopl_set_encoder.cu moved to its own tiling and no longer defines these.
static constexpr int BS_FWD_SLOTS = 8;                 // slots per thread
static constexpr int BS_XT_STRIDE = 100;               // >= BS_N_ENT, multiple of 4
static constexpr int BS_WSL_STRIDE = BS_W_SLIME + 1;   // conflict-free per-channel reads
static_assert(BS_N_ENT / BS_FWD_SLOTS * BS_FWD_CH == BS_FWD_TPR, "rollout encoder tiling");

struct BrGroups {
    int n;
    int end[BR_MAX_GROUPS + 1];  // end[0] = 0; group g owns rows [end[g], end[g+1])
};
struct BrLinear { const float* w[BR_MAX_GROUPS]; };  // per-group (N, K) row-major weights

__device__ __forceinline__ int br_group_of(const BrGroups& g, int row) {
    int k = 0;
    while (k + 1 < g.n && row >= g.end[k + 1]) k++;
    return k;
}

// ---------------------------------------------------------------------------
// Set encoder, one row per block (BS_FWD_TPR threads). Same math as
// bs_fused_forward: the row is staged in shared memory once (it may live in
// host memory), then feeds the rollout copy, flat_input and the entity tile.
struct BrEncodeArgs {
    const float* obs;          // (rows, BS_OBS) env observations (chunk base)
    const float* rewards;      // (rows,) env rewards (chunk base)
    const float* terminals;    // (rows,) env terminals (chunk base)
    float* rollout_obs;        // (rows, BS_OBS) rollout slice for this step
    float* rollout_rewards;    // (rows,)
    float* rollout_terminals;  // (rows,)
    float* flat_input;         // (rows, BS_FLAT)
    float* joined;             // (rows, H + 2*BS_E)
    int H, rows;
    int sparse;                // 1: read only the populated slots (see br_encode)
    BrGroups g;
    BrLinear w_ent, w_slime;
};

static constexpr int BR_ROW_STRIDE = BS_OBS + 2;  // even, keeps float2 staging aligned
static_assert(BS_OBS % 2 == 0 && BS_FLAT % 2 == 0, "float2 staging");
static_assert(BS_PREFIX == 10 + 6 * BOPL_PLAYER_FEATURES && BOPL_PLAYER_FEATURES % 2 == 0, "sparse staging: player block");
static_assert(BS_ENTITY_OFFSET % 2 == 0 && BS_SLIME_OFFSET % 2 == 0 && BS_PREVIOUS_OFFSET % 2 == 0 && BS_W_SLIME % 2 == 0, "sparse staging");

// The observation row is written by bopl_encode (bopl_puffer.h): the header
// holds obs[3] = player_count / 6, obs[4] = min(entities, 96) / 96 and
// obs[6] = min(slimes, 12) / 12, the players / entities / slimes are packed
// from slot 0 and every slot past the counts is zero. Reading only the packed
// slots (plus the previous-decision tail) is therefore exact and cuts the
// PCIe traffic of a zero-copy row by 3-4x. BOPL_OBS_SPARSE=0 stages the whole
// row (same result, used to check the assumption).
struct BrRanges { int lo[4], hi[4]; };  // float2 units, concatenated

__device__ __forceinline__ int br_count(float h, int per, int cap) {
    int c = __float2int_rn(h * (float)per);
    return c < 0 ? 0 : (c > cap ? cap : c);
}

__global__ void __launch_bounds__(BS_FWD_TPR, 3) br_encode(BrEncodeArgs a) {
    __shared__ __align__(16) float Row[BR_ROW_STRIDE];
    __shared__ __align__(16) float Xt[BS_W_ENT * BS_XT_STRIDE];  // [k][slot]
    __shared__ __align__(16) float Wt[BS_W_ENT][BS_E];          // [k][channel]
    __shared__ float Wsl[BS_E * BS_WSL_STRIDE];
    __shared__ float red_v[2][BS_E];
    __shared__ int red_i[2][BS_E];
    __shared__ BrRanges rng;

    const int row = blockIdx.x;
    if (row >= a.rows) return;
    const int tr = threadIdx.x;
    const int warp_in_row = tr >> 5, lane = tr & 31;
    const int sg = tr >> 3, cg = tr & 7;
    const int JW = a.H + 2 * BS_E;
    const int grp = br_group_of(a.g, row);
    const float* w_ent = a.w_ent.w[grp];
    const float* w_slime = a.w_slime.w[grp];
    const float2* src = (const float2*)(a.obs + (long)row * BS_OBS);
    // Header first (one dependent round trip to host memory); the zero fill of
    // the staging row overlaps it.
    float hdr = 0.f;
    if (warp_in_row == 0 && lane < 8 && a.sparse) hdr = ((const float*)src)[lane];
    for (int i = tr; i < BR_ROW_STRIDE / 2; i += BS_FWD_TPR) ((float2*)Row)[i] = make_float2(0.f, 0.f);
    if (warp_in_row == 0) {
        float h3 = __shfl_sync(0xffffffffu, hdr, 3), h4 = __shfl_sync(0xffffffffu, hdr, 4), h6 = __shfl_sync(0xffffffffu, hdr, 6);
        if (lane == 0) {
            if (a.sparse) {
                int pc = br_count(h3, 6, 6), ec = br_count(h4, BS_N_ENT, BS_N_ENT), sc = br_count(h6, BS_N_SLIME, BS_N_SLIME);
                int ent = (ec * BS_W_ENT + 1) & ~1;  // an odd count reads one zero float past the last slot
                rng.lo[0] = 0;                        rng.hi[0] = (10 + pc * BOPL_PLAYER_FEATURES) / 2;
                rng.lo[1] = BS_ENTITY_OFFSET / 2;     rng.hi[1] = (BS_ENTITY_OFFSET + ent) / 2;
                rng.lo[2] = BS_SLIME_OFFSET / 2;      rng.hi[2] = (BS_SLIME_OFFSET + sc * BS_W_SLIME) / 2;
                rng.lo[3] = BS_PREVIOUS_OFFSET / 2;   rng.hi[3] = BS_OBS / 2;
            } else {
                rng.lo[0] = 0; rng.hi[0] = BS_OBS / 2;
                rng.lo[1] = rng.hi[1] = rng.lo[2] = rng.hi[2] = rng.lo[3] = rng.hi[3] = 0;
            }
        }
    }
    __syncthreads();
    // Stage the populated ranges: every load in flight (one batch covers a
    // whole row) while the weights are staged.
    {
        constexpr int BR_STAGE = (BS_OBS / 2 + BS_FWD_TPR - 1) / BS_FWD_TPR;
        const int n0 = rng.hi[0] - rng.lo[0], n1 = rng.hi[1] - rng.lo[1], n2 = rng.hi[2] - rng.lo[2], n3 = rng.hi[3] - rng.lo[3];
        const int total = n0 + n1 + n2 + n3;
        float2 v[BR_STAGE]; int at[BR_STAGE];
        #pragma unroll
        for (int it = 0; it < BR_STAGE; it++) {
            int i = it * BS_FWD_TPR + tr, j = i;
            at[it] = -1;
            if (i < total) {
                if (j < n0) at[it] = rng.lo[0] + j;
                else if ((j -= n0) < n1) at[it] = rng.lo[1] + j;
                else if ((j -= n1) < n2) at[it] = rng.lo[2] + j;
                else at[it] = rng.lo[3] + (j - n2);
                v[it] = src[at[it]];
            }
        }
        for (int i = tr; i < BS_E * BS_W_ENT; i += BS_FWD_TPR) Wt[i % BS_W_ENT][i / BS_W_ENT] = w_ent[i];
        for (int i = tr; i < BS_E * BS_W_SLIME; i += BS_FWD_TPR) Wsl[(i / BS_W_SLIME) * BS_WSL_STRIDE + i % BS_W_SLIME] = w_slime[i];
        float2* d = (float2*)Row;
        #pragma unroll
        for (int it = 0; it < BR_STAGE; it++) if (at[it] >= 0) d[at[it]] = v[it];
    }
    if (tr == 0) {
        a.rollout_rewards[row] = a.rewards[row];
        a.rollout_terminals[row] = a.terminals[row];
    }
    __syncthreads();
    // Rollout copy, flat input (prefix + previous), entity tile [k][slot].
    {
        const float2* s = (const float2*)Row;
        float2* d = (float2*)(a.rollout_obs + (long)row * BS_OBS);
        for (int i = tr; i < BS_OBS / 2; i += BS_FWD_TPR) d[i] = s[i];
        float2* flat = (float2*)(a.flat_input + (long)row * BS_FLAT);
        for (int i = tr; i < BS_PREFIX / 2; i += BS_FWD_TPR) flat[i] = s[i];
        for (int i = tr; i < BS_PREVIOUS / 2; i += BS_FWD_TPR) flat[BS_PREFIX / 2 + i] = s[BS_PREVIOUS_OFFSET / 2 + i];
        for (int e = tr; e < BS_ENT_FLOATS; e += BS_FWD_TPR) {
            int slot = e / BS_W_ENT, k = e - slot * BS_W_ENT;
            Xt[k * BS_XT_STRIDE + slot] = Row[BS_ENTITY_OFFSET + e];
        }
    }
    __syncthreads();

    // Entity projections for this thread's 8 slots x 8 channels.
    float acc[BS_FWD_SLOTS][BS_FWD_CH];
    #pragma unroll
    for (int s = 0; s < BS_FWD_SLOTS; s++)
        #pragma unroll
        for (int j = 0; j < BS_FWD_CH; j++) acc[s][j] = 0.0f;
    const float* xt = Xt + sg * BS_FWD_SLOTS;
    #pragma unroll
    for (int k = 0; k < BS_W_ENT; k++) {
        float4 w0 = *(const float4*)&Wt[k][cg * 4], w1 = *(const float4*)&Wt[k][32 + cg * 4];
        float4 x0 = *(const float4*)&xt[k * BS_XT_STRIDE], x1 = *(const float4*)&xt[k * BS_XT_STRIDE + 4];
        const float w[BS_FWD_CH] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const float x[BS_FWD_SLOTS] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        #pragma unroll
        for (int s = 0; s < BS_FWD_SLOTS; s++)
            #pragma unroll
            for (int j = 0; j < BS_FWD_CH; j++) acc[s][j] = fmaf(x[s], w[j], acc[s][j]);
    }
    float bv[BS_FWD_CH]; int bi[BS_FWD_CH];
    #pragma unroll
    for (int j = 0; j < BS_FWD_CH; j++) { bv[j] = 0.0f; bi[j] = -1; }
    #pragma unroll
    for (int s = 0; s < BS_FWD_SLOTS; s++) {
        bool valid = xt[s] != 0.0f;  // feature 0 of the slot
        #pragma unroll
        for (int j = 0; j < BS_FWD_CH; j++)
            if (valid && acc[s][j] > bv[j]) { bv[j] = acc[s][j]; bi[j] = sg * BS_FWD_SLOTS + s; }
    }
    #pragma unroll
    for (int off = 8; off <= 16; off <<= 1)
        #pragma unroll
        for (int j = 0; j < BS_FWD_CH; j++) {
            float ov = __shfl_xor_sync(0xffffffffu, bv[j], off);
            int oi = __shfl_xor_sync(0xffffffffu, bi[j], off);
            bs_better(bv[j], bi[j], ov, oi);
        }
    if (warp_in_row > 0 && lane < 8) {
        #pragma unroll
        for (int j = 0; j < BS_FWD_CH; j++) {
            int c = bs_fwd_channel(cg, j);
            red_v[warp_in_row - 1][c] = bv[j]; red_i[warp_in_row - 1][c] = bi[j];
        }
    }
    // Slime: thread = channel for the first 64 threads.
    float sv = 0.0f;
    if (tr < BS_E) {
        const float* wsl = Wsl + tr * BS_WSL_STRIDE;
        #pragma unroll 1
        for (int s = 0; s < BS_N_SLIME; s++) {
            const float* xs = Row + BS_SLIME_OFFSET + s * BS_W_SLIME;
            if (xs[0] == 0.0f) continue;
            float acc_s = 0.0f;
            #pragma unroll
            for (int k = 0; k < BS_W_SLIME; k++) acc_s = fmaf(xs[k], wsl[k], acc_s);
            if (acc_s > sv) sv = acc_s;
        }
    }
    __syncthreads();
    if (warp_in_row == 0 && lane < 8) {
        #pragma unroll
        for (int j = 0; j < BS_FWD_CH; j++) {
            int c = bs_fwd_channel(cg, j);
            bs_better(bv[j], bi[j], red_v[0][c], red_i[0][c]);
            bs_better(bv[j], bi[j], red_v[1][c], red_i[1][c]);
            a.joined[(long)row * JW + a.H + c] = bv[j];
        }
    }
    if (tr < BS_E) a.joined[(long)row * JW + a.H + BS_E + tr] = sv;
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

template <bool TF32>
__global__ void __launch_bounds__(BR_THREADS, 2) br_gemm(BrGemmArgs a) {
    // Per-warp A tile (16 x 64): TF32 in A-fragment order [kstep][lane][4],
    // fp32 as [k][row] (float4 row reads). The 16 x 32 C tiles alias it after
    // the K loop.
    __shared__ __align__(16) float As[BR_KG][BR_TM * BR_TK];  // 32 KB
    int rt = blockIdx.x, grp = 0;
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
    const int col0 = (blockIdx.y * subtiles + warp / sub) * BR_TN;
    const float* __restrict__ Wt = a.wt.w[grp];
    float* Aw = As[warp];

    // A chunk: 16 rows x 64 k = 512 float2 over the warp (element i: row i%16, k-pair i/16).
    float2 av[BR_TK / 4];
    auto load_a = [&](int k0) {
        #pragma unroll
        for (int j = 0; j < BR_TK / 4; j++) {
            int i = c + j * 32, r = i & 15, k = k0 + ((i >> 4) << 1);
            av[j] = (row0 + r < row_end && k < K) ? *(const float2*)(a.in + (long)(row0 + r) * a.ld_in + k) : make_float2(0.f, 0.f);
        }
    };
    // TF32 layout: A fragment order [kstep][lane][4] (a0 = A[gid][tid4], a1 = A[gid+8][tid4],
    // a2 = A[gid][tid4+4], a3 = A[gid+8][tid4+4]) so a lane's k-step operand is one float4.
    auto store_a = [&]() {
        #pragma unroll
        for (int j = 0; j < BR_TK / 4; j++) {
            int i = c + j * 32, r = i & 15, k = (i >> 4) << 1;
            if (TF32) {
                int kstep = k >> 3, kk = k & 7, lane = (r & 7) * 4 + (kk & 3), slot = (r >> 3) + 2 * (kk >> 2);
                Aw[(kstep * 32 + lane) * 4 + slot] = av[j].x;
                Aw[(kstep * 32 + lane + 1) * 4 + slot] = av[j].y;
            } else {
                Aw[k * BR_TM + r] = av[j].x; Aw[(k + 1) * BR_TM + r] = av[j].y;
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
        // in the fragment buffer (br_frag_floats rounds K up to 16).
        float4 bf[4][4];
        auto load_b = [&](int k0) {
            #pragma unroll
            for (int kp = 0; kp < 4; kp++)
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    long kpair = (k0 >> 4) + kp;
                    bf[kp][t] = *(const float4*)(Wt + ((kpair * ntiles + ntile0 + t) * 32 + lane) * 4);
                }
        };
        for (int ch = kg; ch < nchunks; ch += sub) {
            load_a(ch * BR_TK); load_b(ch * BR_TK);
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
            load_a(ch * BR_TK); load_b(ch * BR_TK);
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
    __syncthreads();
    // Per subtile, sum its warps' partial tiles in warp order and write the valid entries.
    for (int o = tid; o < subtiles * BR_TM * BR_TN; o += BR_THREADS) {
        int st = o / (BR_TM * BR_TN), e = o % (BR_TM * BR_TN), r = e / BR_TN, cc = e % BR_TN;
        float v = 0.f;
        for (int w = st * sub; w < (st + 1) * sub; w++) v += As[w][r * BR_TN + cc];
        int row = row0 + r, cl = (blockIdx.y * subtiles + st) * BR_TN + cc;
        if (row < row_end && cl < N) {
            if (a.relu) v = fmaxf(0.f, v);
            a.out[(long)row * a.ld_out + cl] = v;
        }
    }
}

// Column-tile count of br_gemm's grid for a GEMM of depth K over N columns.
static int br_gemm_col_tiles(int K, int N) {
    int nchunks = (K + BR_TK - 1) / BR_TK, sub = 1;
    while (sub < nchunks && sub < BR_KG) sub <<= 1;
    int cols = (BR_KG / sub) * BR_TN;
    return (N + cols - 1) / cols;
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
struct BrPlan {
    int enabled;         // batched chain in use
    int zero_copy;       // 0 memcpys (obs + packed small inputs), 1 obs memcpy + rest zero-copy, 2 all zero-copy
    int actions_zero_copy;  // sampler writes actions into the mapped host buffer
    int graph_copies;    // H2D / D2H memcpys captured into the forward graph (BOPL_GRAPH_COPIES, default 1)
    int tf32;            // tensor-core tf32 GEMMs (BOPL_TF32, default 1) or fp32 FMAs
    int sparse;          // stage only the populated obs slots (BOPL_OBS_SPARSE, default 1)
    int rows;            // live rows per buffer (bank_layout end)
    int H, layers, A_total, num_atns, mask_stride;
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
    dim3 grid(p.tiles[p.g.n], br_gemm_col_tiles(K, N));
    if (p.tf32) br_gemm<true><<<grid, BR_THREADS, 0, stream>>>(a);
    else br_gemm<false><<<grid, BR_THREADS, 0, stream>>>(a);
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

#ifndef BR_KERNELS_ONLY  // the trainer-side part (needs pufferlib.cu); microbenchmarks include only the kernels above
// ---------------------------------------------------------------------------
// Sampler: sample_logits_warp with the env's byte mask as input and the
// actions written to both the env buffer (float) and the rollout.
struct BrSampleArgs {
    const float* dec; int ld_dec, A_total; // (rows, ld_dec >= A_total + 1): logits then the value
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
    #pragma unroll
    for (int it = 0; it < MASK_ITERS; it++) {
        int i = lane + it * 32;
        if (i >= A_total) break;
        float l = finite_or_clamp(logits[i], -1.0e4f, 1.0e4f);
        if (mask) {
            float m = (float)mv[it];
            a.rollout_mask[(long)idx * a.mask_stride + i] = m;
            if (m == 0.0f) l = -1e4f;
        }
        row[i] = l;
    }
    __syncwarp();

    curandStatePhilox4_32_10_t state;
    if (lane == 0) state = a.rng[idx];
    float total_log_prob = 0.0f;
    int logits_offset = 0, verb = 0;
    for (int h = 0; h < a.num_atns; ++h) {
        const int A = a.act_sizes[h];
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
        logits_offset += A;
    }
    if (lane == 0) {
        a.logprobs[idx] = total_log_prob;
        a.values[idx] = logits[A_total];
        a.rng[idx] = state;
    }
}

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
    if (pufferl->policy.encoder.forward != bs_forward) why = "encoder is not the fused set encoder";
    else if (dw->continuous) why = "continuous decoder";
    else if (pufferl->policy.network.forward != mingru_forward) why = "network is not MinGRU";
    else if (mw->num_layers > BR_MAX_LAYERS) why = "too many MinGRU layers";
    else if (1 + pufferl->num_frozen_banks > BR_MAX_GROUPS) why = "too many frozen banks";
    else if (dw->output_dim > SAMPLE_MAX_LOGITS) why = "too many logits for the warp sampler";
    for (int b = 0; !why && b < pufferl->num_frozen_banks; b++) {
        WeightBank* fb = &pufferl->frozen_banks[b];
        if (fb->hidden_size != mw->hidden || fb->num_layers != mw->num_layers
                || fb->policy.encoder.forward != bs_forward || fb->policy.network.forward != mingru_forward)
            why = "a frozen bank differs from the learner in shape";
    }
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
    p->A_total = dw->output_dim;
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
    p->ld_dec = br_pad8(p->A_total + 1);
    p->acts = (BrActs*)calloc(hypers.num_buffers, sizeof(BrActs));
    int H = p->H, JW = H + 2 * BS_E;
    auto zalloc = [](float** ptr, size_t n) {
        assert(cudaMalloc((void**)ptr, n * sizeof(float)) == cudaSuccess);
        cudaMemset(*ptr, 0, n * sizeof(float));
    };
    for (int b = 0; b < hypers.num_buffers; b++) {
        BrActs& a = p->acts[b];
        zalloc(&a.flat, (size_t)apb * BS_FLAT); zalloc(&a.joined, (size_t)apb * JW); zalloc(&a.enc, (size_t)apb * H);
        zalloc(&a.comb, (size_t)apb * 3 * H); zalloc(&a.gru[0], (size_t)apb * H); zalloc(&a.gru[1], (size_t)apb * H);
        zalloc(&a.dec, (size_t)apb * p->ld_dec);
    }
    for (int g = 0; g < p->g.n; g++) {
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
        for (int b = 0; b < nb; b++) {
            if (vec->pack_in_tail) {
                size_t off = ((size_t)b * apb + p->rows) * row_bytes;
                vec->pack_host[b] = (char*)vec->observations + off;
                vec->pack_dev[b] = (char*)vec->gpu_observations + off;
            } else {
                cudaHostAlloc((void**)&vec->pack_host[b], vec->pack_bytes, cudaHostAllocPortable);
                cudaMalloc((void**)&vec->pack_dev[b], vec->pack_bytes);
            }
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
    int H = p.H, JW = H + 2 * BS_E;
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
    br_transpose_launch(p, s_dec, d_dec, p.g.n, p.A_total + 1, H, p.ld_dec, stream);
}

static void br_free(PuffeRL* pufferl) {
    BrPlan* p = pufferl->br;
    if (!p) return;
    for (int b = 0; b < pufferl->hypers.num_buffers; b++) {
        BrActs& a = p->acts[b];
        for (float* ptr : {a.flat, a.joined, a.enc, a.comb, a.gru[0], a.gru[1], a.dec}) cudaFree(ptr);
    }
    for (int g = 0; g < p->g.n; g++) {
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
    int start = buf * apb, rows = p.rows, H = p.H, JW = H + 2 * BS_E;
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
    if (p.graph_copies) {
        size_t row_bytes = (size_t)BS_OBS * sizeof(float), off = (size_t)start * row_bytes;
        size_t bytes = (size_t)rows * row_bytes + (vec->pack_in_tail ? (size_t)vec->pack_bytes : 0);
        cudaMemcpyAsync((char*)vec->gpu_observations + off, (const char*)vec->observations + off, bytes, cudaMemcpyHostToDevice, stream);
        if (!vec->pack_in_tail) cudaMemcpyAsync(vec->pack_dev[buf], vec->pack_host[buf], vec->pack_bytes, cudaMemcpyHostToDevice, stream);
    }

    // Per-group weights (set-encoder MLPs as stored, GEMM weights transposed by
    // br_prepare) and recurrent state (group 0 = learner snapshot).
    BrLinear w_ent = {}, w_slime = {}, w_flat = {}, w_fusion = {}, w_dec = {}, w_gru[BR_MAX_LAYERS] = {};
    BrGateArgs gate = {};
    for (int g = 0; g < p.g.n; g++) {
        PolicyWeights* w; PrecisionTensor* s;
        br_group_weights(pufferl, g, buf, &w, &s);
        BoplSetWeights* sw = (BoplSetWeights*)w->encoder;
        w_ent.w[g] = sw->entity.data; w_slime.w[g] = sw->slime.data;
        w_flat.w[g] = p.wt[g].flat; w_fusion.w[g] = p.wt[g].fusion;
        for (int l = 0; l < p.layers; l++) w_gru[l].w[g] = p.wt[g].gru[l];
        w_dec.w[g] = p.wt[g].dec;
        gate.state[g] = s->data;
        gate.state_rows[g] = (int)s->shape[1];
    }

    PrecisionTensor obs_dst = puf_slice(rollouts.observations, t, start, rows);
    PrecisionTensor rew_dst = puf_slice(rollouts.rewards, t, start, rows);
    PrecisionTensor term_dst = puf_slice(rollouts.terminals, t, start, rows);
    PrecisionTensor act_dst = puf_slice(rollouts.actions, t, start, rows);
    PrecisionTensor lp_dst = puf_slice(rollouts.logprobs, t, start, rows);
    PrecisionTensor val_dst = puf_slice(rollouts.values, t, start, rows);
    PrecisionTensor mask_dst = {};
    if (p.mask_stride) mask_dst = puf_slice(rollouts.action_mask, t, start, rows);

    BrEncodeArgs enc = {};
    enc.obs = obs; enc.rewards = rewards; enc.terminals = terminals;
    enc.rollout_obs = obs_dst.data; enc.rollout_rewards = rew_dst.data; enc.rollout_terminals = term_dst.data;
    enc.flat_input = A.flat; enc.joined = A.joined; enc.H = H; enc.rows = rows; enc.sparse = p.sparse; enc.g = p.g;
    enc.w_ent = w_ent; enc.w_slime = w_slime;
    br_encode<<<rows, BS_FWD_TPR, 0, stream>>>(enc);

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
    br_gemm_launch(p, x, H, A.dec, p.ld_dec, H, p.A_total + 1, p.ld_dec, 0, w_dec, stream);

    int hc_stride = 0;
    const signed char* hc_dev = get_head_consume_dev(&hc_stride);
    BrSampleArgs smp = {};
    smp.dec = A.dec; smp.ld_dec = p.ld_dec; smp.A_total = p.A_total;
    smp.act_sizes = pufferl->act_sizes_puf.data; smp.num_atns = p.num_atns;
    smp.env_mask = mask; smp.mask_stride = p.mask_stride; smp.rollout_mask = mask_dst.data;
    smp.env_actions = env_actions; smp.rollout_actions = act_dst.data;
    smp.logprobs = lp_dst.data; smp.values = val_dst.data;
    smp.rng = pufferl->rng_states[buf];
    smp.head_consume = hc_dev; smp.hc_stride = hc_stride; smp.rows = rows;
    br_sample<<<(rows + SAMPLE_WARP_ROWS - 1) / SAMPLE_WARP_ROWS, SAMPLE_WARP_ROWS * 32, 0, stream>>>(smp);
    if (p.graph_copies && !p.actions_zero_copy)
        cudaMemcpyAsync(vec->actions + (long)start * p.num_atns, vec->gpu_actions + (long)start * p.num_atns,
                        (size_t)rows * p.num_atns * sizeof(float), cudaMemcpyDeviceToHost, stream);
}
#endif  // BR_KERNELS_ONLY
