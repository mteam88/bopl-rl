// Bopl batched rollout forward for policy v4 (train/bopl_v4.cu is the
// per-policy trainer path and the parity reference).
//
// Included by bopl_rollout.cu. The v3 chain's machinery is reused unchanged:
// the row groups [learner | bank 0 | ...] (BrGroups), the grouped small-M
// GEMM (br_gemm, weights in TF32 fragment order or fp32 transposed, per group),
// the MinGRU gate with the reset fused in (br_gate), the warp sampler
// (br_sample) and the H2D / D2H copy plan. What differs is the encoder's set
// stage and the decoder (FFN, wide critic), which become row-wise kernels:
//
//   br4_sets     a block per (set, run of samples of one group): the valid
//                rows of all its samples together through relu(W1 x + E[id])
//                -> relu(W2 h1) in bf16-rounded fp32 (the trainer's WMMA path
//                rounds the same operands to bf16; W1, W2 packed to bf16 once
//                per rollout by br4_prepare), max || mean pooled per sample
//                into joined[:, FD + 2Ds ..]. The players' blocks also copy
//                the obs rows, rewards and terminals into the rollout tensors.
//   br_gemm      flat (obs[:, :FLAT] -> relu -> joined[:, :FD]), fusion,
//                MinGRU layers, FFN in/out, actor, critic in / b1 / b2.
//   br4_rows     the row-wise glue with a mode per call site: LayerNorm,
//                residual + LayerNorm of [y, privileged], the critic's last
//                residual + LayerNorm + dot with critic_out into the value
//                column of the decoder rows.
//
// ~18 launches per buffer step for every policy together (1 + banks chains of
// ~60 each before). Rows past a set's training budget are not dropped here
// (the trainer logs set_overflow when it drops any).
#pragma once

struct Br4SetWeights {   // per group g and set s: W1^T bf16 (features, I), W2 bf16 run-major (br4_w2_index) packed by br4_prepare, E fp32 (IDS, I) as stored
    const __nv_bfloat16* w1t[BR_MAX_GROUPS][4];
    const __nv_bfloat16* w2[BR_MAX_GROUPS][4];
    const float* emb[BR_MAX_GROUPS];
};
struct Br4SetsArgs {
    const float* obs;          // (rows, STRIDE) live rows of this buffer
    const float* rewards; const float* terminals;
    float* rollout_obs; float* rollout_rewards; float* rollout_terminals;
    float* joined; int ldj, col0;   // pooled channels at joined[row, col0 + 2 D s + {0, D}]
    int D, I, mean_pool, rows;
    int sm_h1, sm_meta;   // the dynamic shared layout (br4_set_smem), set by br4_sets_launch
    BrGroups g;
    Br4SetWeights w;
};
#define BR4_SET_THREADS 256   // = BR4_POOL_SPLIT: thread (d, part) of the pooling below
#define BR4_POOL_SPLIT 256    // the mean pool's summation order (br4_set_block)
#define BR4_MAX_SET_ROWS 48   // the largest set
#define BR4_BLOCK_SET_ROWS 96 // set rows (valid or not) one block holds: samples per block x the set's rows
static_assert(BR4_SET_THREADS == BR4_POOL_SPLIT, "a thread per (channel, pooling part)");
static_assert(BOPL4_PLAYERS <= BR4_MAX_SET_ROWS && BOPL4_SLOTS <= BR4_MAX_SET_ROWS && BOPL4_ENTITIES <= BR4_MAX_SET_ROWS && BOPL4_SLIMES <= BR4_MAX_SET_ROWS, "set rows");

__device__ __forceinline__ float br4_bf(float v) { return __bfloat162float(__float2bfloat16(v)); }

// Samples per block for set s: as many as fit BR4_BLOCK_SET_ROWS set rows,
// the players 4 (their pooling runs sample after sample: 10 a block made the
// players' blocks the longest; 4 measured 23.8 -> 21.8 us the launch).
// 96 rows (2 entity, 4 slot, 3 slime samples; 172 blocks for 128 rows) over
// 64 (1, 2, 2; 288 blocks, three waves at two blocks an SM): a block's cost
// is mostly its latency chain, not its rows; 23.7 -> 19.7 us the launch.
__host__ __device__ constexpr int br4_samples_of(int s) {
    return s == 0 ? 4 : BR4_BLOCK_SET_ROWS / (s == 1 ? BOPL4_SLOTS : s == 2 ? BOPL4_ENTITIES : BOPL4_SLIMES);
}
static_assert(br4_samples_of(0) <= 32 && br4_samples_of(1) <= 32 && br4_samples_of(2) <= 32 && br4_samples_of(3) <= 32, "samples per block");
constexpr int br4_cmax(int x, int y) { return x > y ? x : y; }
constexpr int br4_align16(int b) { return (b + 15) / 16 * 16; }
__host__ __device__ constexpr int br4_stage_width(int s) { return (v4::SPEC[s].width + 3) / 4 * 4; }   // a staged row's stride: float4 aligned
constexpr int br4_stage_floats() {   // the largest samples x staged obs block (floats)
    return br4_cmax(br4_cmax(br4_samples_of(0) * v4::SPEC[0].rows * br4_stage_width(0), br4_samples_of(1) * v4::SPEC[1].rows * br4_stage_width(1)),
                    br4_cmax(br4_samples_of(2) * v4::SPEC[2].rows * br4_stage_width(2), br4_samples_of(3) * v4::SPEC[3].rows * br4_stage_width(3)));
}
// Dynamic shared, per block (the largest set's sizes):
//   xs     the block's samples' obs blocks, fp32 bf16-rounded (the features rounded as the trainer
//          rounds them), rows br4_stage_width apart (then the pooling's parts: (samples, parts, D) max and sum)
//   h1     (valid rows of the block, I) bf16 (the values are bf16-rounded; h2 reads 8 a load)
//   meta   per staged row: valid flag, embedding id; per valid row: its xs offset and id;
//          per sample: valid rows and the first one's index
struct Br4SetSmem { int h1, meta, total; };
constexpr int BR4_SET_META = 4 * BR4_BLOCK_SET_ROWS + 2 * 32;
constexpr int br4_max_samples() { return br4_cmax(br4_cmax(br4_samples_of(0), br4_samples_of(1)), br4_cmax(br4_samples_of(2), br4_samples_of(3))); }
constexpr Br4SetSmem br4_set_smem(int I) {
    return Br4SetSmem{br4_cmax(br4_align16(br4_stage_floats() * 4), br4_max_samples() * BR4_POOL_SPLIT * 8),
                      br4_cmax(br4_align16(br4_stage_floats() * 4), br4_max_samples() * BR4_POOL_SPLIT * 8) + BR4_BLOCK_SET_ROWS * I * 2,
                      br4_cmax(br4_align16(br4_stage_floats() * 4), br4_max_samples() * BR4_POOL_SPLIT * 8) + BR4_BLOCK_SET_ROWS * I * 2 + BR4_SET_META * 4};
}
static size_t br4_sets_smem(int I) { return (size_t)br4_set_smem(I).total; }
// One group's set weights into the chain's layout: W1 transposed to (features, I)
// bf16 (thread i of the h1 loop reads column i: coalesced), W2 bf16 in runs of
// 8 columns, run-major ((I / 8, D, 8): thread d of the h2 loop reads its row's
// run q at q D + d, so a warp's load is 512 contiguous bytes, not 32 rows'
// 16 apiece). The values are the bf16 roundings br4_sets applied on the fly before.
__host__ __device__ inline long br4_w2_index(int d, int i, int D) { return ((long)(i >> 3) * D + d) * 8 + (i & 7); }
__global__ void br4_pack_set_weights(const float* w1, int features, const float* w2, int D, int I, __nv_bfloat16* w1t, __nv_bfloat16* w2b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, n1 = I * features, n2 = D * I;
    if (i < n1) { int f = i / I, ii = i - f * I; w1t[i] = __float2bfloat16(w1[(long)ii * features + f]); }
    else if (i < n1 + n2) { int d = (i - n1) / I, ii = i - n1 - d * I; w2b[br4_w2_index(d, ii, D)] = __float2bfloat16(w2[i - n1]); }
}
// Blocks of set s per group: its rows in runs of br4_samples_of(s) samples (one group's weights per block).
__host__ __device__ inline int br4_set_blocks(const BrGroups& g, int s) {
    int nb = 0, k = br4_samples_of(s);
    for (int q = 0; q < g.n; q++) nb += (g.end[q + 1] - g.end[q] + k - 1) / k;
    return nb;
}
__host__ __device__ constexpr int br4_set_order(int y) { return y == 0 ? 2 : y == 1 ? 1 : y == 2 ? 3 : 0; }   // entities (the most rows) first

// h1 of the valid rows m0, m0 + js, ... (NR of them, all < M) for channel i: see br4_set_block
template <int S, int I, int NR>
__device__ __forceinline__ void br4_h1_rows(const float* xs, const int* vrow, const int* vid, const float* E,
                                            const float (&w)[v4::SPEC[S].features], unsigned short* h1, int m0, int js, int i) {
    constexpr v4::Spec sp = v4::SPEC[S];
    constexpr int F = sp.features;
    const float* xr[NR]; float acc[NR], ev[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        const int m = m0 + r * js;
        xr[r] = xs + vrow[m]; acc[r] = 0.0f;
        ev[r] = sp.id_col >= 0 ? br4_bf(E[(long)vid[m] * I + i]) : 0.0f;
    }
    #pragma unroll
    for (int f = 0; f < F / 4 * 4; f += 4) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            const float4 x = *(const float4*)(xr[r] + f);
            acc[r] = fmaf(w[f], x.x, acc[r]); acc[r] = fmaf(w[f + 1], x.y, acc[r]);
            acc[r] = fmaf(w[f + 2], x.z, acc[r]); acc[r] = fmaf(w[f + 3], x.w, acc[r]);
        }
    }
    #pragma unroll
    for (int f = F / 4 * 4; f < F; f++) {
        #pragma unroll
        for (int r = 0; r < NR; r++) acc[r] = fmaf(w[f], xr[r][f], acc[r]);
    }
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        float v = acc[r];
        if (sp.id_col >= 0) v += ev[r];
        h1[(m0 + r * js) * I + i] = __bfloat16_as_ushort(__float2bfloat16(fmaxf(v, 0.0f)));   // br4_bf's value, stored as bf16
    }
}
// acc + w.lo x.lo, then + w.hi x.hi for two bf16 pairs: fma.rn.f32.bf16 is fmaf of the bf16 values (exact in
// fp32, one rounding: bit-identical to fmaf(float(w), float(x), acc) over every bf16 w, a sample of x and
// specials, subnormals and NaNs in acc), with no conversion instructions (FHFMA.BF16 reads either half).
__device__ __forceinline__ float br4_fma_bf2(unsigned w, unsigned x, float acc) {
    asm("{ .reg .b16 wl, wh, xl, xh; mov.b32 {wl, wh}, %1; mov.b32 {xl, xh}, %2;\n\t"
        "fma.rn.f32.bf16 %0, wl, xl, %0;\n\tfma.rn.f32.bf16 %0, wh, xh, %0; }" : "+f"(acc) : "r"(w), "r"(x));
    return acc;
}
// h2 of the valid rows m0, m0 + parts, ... (NR of them, all < M) for this thread's channel, pooled
// into its samples' running max and sum: see br4_set_block. W2's row and h1 stay bf16: a 16-byte load
// is 8 of a row's h1 (half the shared loads of fp32 h1), each pair a bf16 fma.
template <int K, int I, int NR>
__device__ __forceinline__ void br4_h2_rows(const unsigned short* h1, const uint4 (&wreg)[I / 8], int m0, int parts,
                                            const int (&kb)[K], const int (&kn)[K], float (&best)[K], float (&sum)[K]) {
    const uint4* hr[NR]; float acc[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) { hr[r] = (const uint4*)(h1 + (m0 + r * parts) * I); acc[r] = 0.0f; }
    #pragma unroll
    for (int q = 0; q < I / 8; q++) {
        const uint4 w = wreg[q];
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            const uint4 x = hr[r][q];
            acc[r] = br4_fma_bf2(w.x, x.x, acc[r]); acc[r] = br4_fma_bf2(w.y, x.y, acc[r]);
            acc[r] = br4_fma_bf2(w.z, x.z, acc[r]); acc[r] = br4_fma_bf2(w.w, x.w, acc[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        const int m = m0 + r * parts;
        const float v = br4_bf(fmaxf(acc[r], 0.0f));
        #pragma unroll
        for (int k = 0; k < K; k++) if (m >= kb[k] && m < kb[k] + kn[k]) { best[k] = fmaxf(best[k], v); sum[k] += v; }
    }
}

// One block per (set, run of samples of one group). Every value keeps its own
// fmaf chain in a fixed order (h1: f = 0 .. features - 1 from 0, then
// + bf(E[id]); h2: i = 0 .. I - 1 from 0), so it does not depend on how the
// work is spread over threads: a thread carries up to R rows' chains at once
// (as many as are left, so no row is computed twice or for nothing). The
// pooling has a fixed order too: per channel d, parts = BR4_POOL_SPLIT / D
// partial max and sum, part p over the sample's valid rows j = p, p + parts,
// ... in order from 0, combined part 0, 1, ...; the mean is the sum / n.
// The h2 thread (d, c) computes the rows m = c, c + parts, ... in order, and
// those are, per sample, exactly the rows of one part (p = c - the sample's
// first row, mod parts) in the part's order: it pools them as they come, h2
// is never stored, and the parts combine through shared. Global reads are
// one round: the weights before the PDL wait (they do not depend on the
// previous kernel), then the samples' obs blocks and E[id] rows. The weights are read once per block of samples.
template <int S, int I>
__device__ __forceinline__ void br4_set_block(const Br4SetsArgs& a, int grp, int row0, int cnt, unsigned char* smem) {
    constexpr v4::Spec sp = v4::SPEC[S];
    constexpr int T = BR4_SET_THREADS, R = 4, K = br4_samples_of(S), N = sp.rows * sp.width, F = sp.features;
    static_assert(T % I == 0 && I % 8 == 0, "set inner width");
    constexpr int W4 = br4_stage_width(S);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, D = a.D, parts = BR4_POOL_SPLIT / D;
    float* xs = (float*)smem;
    unsigned short* h1 = (unsigned short*)(smem + a.sm_h1);
    int* flag = (int*)(smem + a.sm_meta);             // (K, rows)
    int* ids = flag + BR4_BLOCK_SET_ROWS;          // (K, rows)
    int* vrow = ids + BR4_BLOCK_SET_ROWS;          // valid row m: xs offset
    int* vid = vrow + BR4_BLOCK_SET_ROWS;          // valid row m: embedding id
    int* nk = vid + BR4_BLOCK_SET_ROWS;            // sample k: valid rows
    int* bk = nk + 32;                             // sample k: its first valid row m
    // weights: this thread's W1 column (its h1 channel i) and W2 row into registers
    float w1c[F];
    {
        const __nv_bfloat16* src = a.w.w1t[grp][S] + tid % I;
        #pragma unroll
        for (int f = 0; f < F; f++) w1c[f] = __bfloat162float(__ldg(src + (long)f * I));
    }
    const int dw = tid % D;   // the h2 channel (and W2 row) of this thread
    uint4 wreg[I / 8];
    {
        const uint4* wrow = (const uint4*)a.w.w2[grp][S] + dw;
        #pragma unroll
        for (int k = 0; k < I / 8; k++) wreg[k] = __ldg(wrow + k * D);
    }
    br_pdl_wait();
    if (S == 0) {   // the rows' copies into the rollout tensors (the old cast kernels), by the players' blocks
        if (tid < cnt) { a.rollout_rewards[row0 + tid] = a.rewards[row0 + tid]; a.rollout_terminals[row0 + tid] = a.terminals[row0 + tid]; }
        // The packed transport unpacks straight into the rollout rows (obs ==
        // rollout_obs); the dense paths still copy. Rows are 8-byte aligned.
        if (a.obs != a.rollout_obs) {
            for (int k = 0; k < cnt; k++) {
                const float* o = a.obs + (long)(row0 + k) * BOPL_OBS_STRIDE;
                const float2* src = (const float2*)o; float2* dst = (float2*)(a.rollout_obs + (long)(row0 + k) * BOPL_OBS_STRIDE);
                for (int i = tid; i < BOPL_OBS_STRIDE / 2; i += T) dst[i] = src[i];
                if (BOPL_OBS_STRIDE & 1) if (tid == 0) a.rollout_obs[(long)(row0 + k) * BOPL_OBS_STRIDE + BOPL_OBS_STRIDE - 1] = o[BOPL_OBS_STRIDE - 1];
            }
        }
    }
    // the samples' obs blocks in one pass (every load in flight at once); a
    // row is valid when its first feature is nonzero
    // (all loads first: the compiler does not move a load past a shared store it cannot tell apart)
    constexpr int NE = (K * N + T - 1) / T;
    float v[NE];
    #pragma unroll
    for (int t = 0; t < NE; t++) {
        int e = t * T + tid, k = e / N, rem = e - k * N;
        v[t] = e < K * N && k < cnt ? a.obs[(long)(row0 + k) * BOPL_OBS_STRIDE + sp.offset + rem] : 0.0f;
    }
    #pragma unroll
    for (int t = 0; t < NE; t++) {
        int e = t * T + tid, k = e / N, rem = e - k * N;
        if (e < K * N && k < cnt) {
            int r = rem / sp.width, c = rem - r * sp.width;
            xs[k * sp.rows * W4 + r * W4 + c] = br4_bf(v[t]);
            if (c == 0) flag[k * sp.rows + r] = v[t] != 0.0f;
            if (sp.id_col >= 0 && c == sp.id_col) ids[k * sp.rows + r] = v4::embedding_id(v[t]);
        }
    }
    __syncthreads();
    // valid rows: per sample by ballot (warp w: samples w, w + 8, ...), then the block's list in sample order
    for (int k = warp; k < cnt; k += T / 32) {
        int n = 0;
        #pragma unroll
        for (int r0 = 0; r0 < sp.rows; r0 += 32) n += __popc(__ballot_sync(0xffffffffu, r0 + lane < sp.rows && flag[k * sp.rows + r0 + lane]));
        if (lane == 0) nk[k] = n;
    }
    __syncthreads();
    if (tid < 32) {   // exclusive scan of nk over the block's samples (K <= 32)
        int n = lane < cnt ? nk[lane] : 0, x = n;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
        if (lane < cnt) bk[lane] = x - n;
    }
    __syncthreads();
    for (int k = warp; k < cnt; k += T / 32) {
        int m = bk[k];
        #pragma unroll
        for (int r0 = 0; r0 < sp.rows; r0 += 32) {
            int r = r0 + lane;
            bool v = r < sp.rows && flag[k * sp.rows + r];
            unsigned bal = __ballot_sync(0xffffffffu, v);
            if (v) {
                int at = m + __popc(bal & ((1u << lane) - 1u));
                vrow[at] = (k * sp.rows + r) * W4;
                vid[at] = sp.id_col >= 0 ? ids[k * sp.rows + r] : 0;
            }
            m += __popc(bal);
        }
    }
    __syncthreads();
    const int M = bk[cnt - 1] + nk[cnt - 1];
    // h1[m][i] = bf(relu(sum_f bf(W1[i][f]) bf(x[m][f]) + bf(E[id_m][i]))) (bf16-stored, like the trainer's packed rows)
    {
        const float* E = a.w.emb[grp];
        constexpr int js = T / I;
        const int i = tid % I;
        for (int m0 = tid / I; m0 < M; m0 += R * js) {   // the warp's rows: m0 is the same over a warp (I >= 32)
            const int nr = (M - m0 + js - 1) / js;
            if (nr >= R) br4_h1_rows<S, I, R>(xs, vrow, vid, E, w1c, h1, m0, js, i);
            else if (nr == 3) br4_h1_rows<S, I, 3>(xs, vrow, vid, E, w1c, h1, m0, js, i);
            else if (nr == 2) br4_h1_rows<S, I, 2>(xs, vrow, vid, E, w1c, h1, m0, js, i);
            else br4_h1_rows<S, I, 1>(xs, vrow, vid, E, w1c, h1, m0, js, i);
        }
    }
    __syncthreads();
    // h2[m][d] = bf(relu(sum_i W2[d][i] h1[m][i])) (the trainer pools bf16-stored h2), pooled per sample
    // as it comes (thread (d, c): rows c, c + parts, ...); the W2 row converted from bf16 8 at a time
    const int c = tid / D;
    int kb[K], kn[K]; float best[K], sum[K];
    #pragma unroll
    for (int k = 0; k < K; k++) { kb[k] = k < cnt ? bk[k] : M; kn[k] = k < cnt ? nk[k] : 0; best[k] = 0.0f; sum[k] = 0.0f; }
    for (int m0 = c; m0 < M; m0 += R * parts) {   // c is the same over a warp (D >= 32)
        const int nr = (M - m0 + parts - 1) / parts;
        if (nr >= R) br4_h2_rows<K, I, R>(h1, wreg, m0, parts, kb, kn, best, sum);
        else if (nr == 3) br4_h2_rows<K, I, 3>(h1, wreg, m0, parts, kb, kn, best, sum);
        else if (nr == 2) br4_h2_rows<K, I, 2>(h1, wreg, m0, parts, kb, kn, best, sum);
        else br4_h2_rows<K, I, 1>(h1, wreg, m0, parts, kb, kn, best, sum);
    }
    br_pdl_release();   // the flat GEMM launches under the parts' combine
    // the parts (sample k, part p, channel d) over xs and w1 (h1 is done with them), then combined in part order
    float2* pp = (float2*)smem;
    #pragma unroll
    for (int k = 0; k < K; k++) if (k < cnt) pp[(k * parts + (c - kb[k] % parts + parts) % parts) * D + dw] = make_float2(best[k], sum[k]);
    __syncthreads();
    for (int e = tid; e < cnt * D; e += T) {
        const int k = e / D, d = e - k * D, n = nk[k];
        float2 t = pp[k * parts * D + d];
        float tb = t.x, ts = t.y;
        for (int p = 1; p < parts; p++) { float2 o = pp[(k * parts + p) * D + d]; tb = fmaxf(tb, o.x); ts += o.y; }
        float* out = a.joined + (long)(row0 + k) * a.ldj + a.col0 + 2 * D * S;
        out[d] = n ? tb : 0.0f;
        out[D + d] = n && a.mean_pool ? ts / n : 0.0f;
    }
}
// grid (blocks of the four sets): the entities' blocks first (the longest), then slots, slimes, players
template <int I>
__global__ void __launch_bounds__(BR4_SET_THREADS, I == 64 ? 2 : 1) br4_sets(Br4SetsArgs a) {
    extern __shared__ __align__(16) unsigned char smem[];
    int b = blockIdx.x;
    #pragma unroll
    for (int y = 0; y < 4; y++) {
        const int s = br4_set_order(y), k = br4_samples_of(s);
        for (int q = 0; q < a.g.n; q++) {
            int len = a.g.end[q + 1] - a.g.end[q], nb = (len + k - 1) / k;
            if (b < nb) {
                const int row0 = a.g.end[q] + b * k, cnt = min(k, a.g.end[q + 1] - row0);
                if (s == 2) br4_set_block<2, I>(a, q, row0, cnt, smem);
                else if (s == 1) br4_set_block<1, I>(a, q, row0, cnt, smem);
                else if (s == 3) br4_set_block<3, I>(a, q, row0, cnt, smem);
                else br4_set_block<0, I>(a, q, row0, cnt, smem);
                return;
            }
            b -= nb;
        }
    }
}
static void br4_sets_init(int D, int I) {
    int smem = (int)br4_sets_smem(I);
    auto k = I == 64 ? br4_sets<64> : br4_sets<128>;
    assert(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);
    assert(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared) == cudaSuccess);
}
static void br4_sets_launch(Br4SetsArgs a, cudaStream_t stream) {
    const Br4SetSmem L = br4_set_smem(a.I);
    a.sm_h1 = L.h1; a.sm_meta = L.meta;
    int blocks = 0;
    for (int s = 0; s < 4; s++) blocks += br4_set_blocks(a.g, s);
    size_t smem = br4_sets_smem(a.I);
    if (a.I == 64) br_launch_pdl(br4_sets<64>, dim3(blocks), dim3(BR4_SET_THREADS), smem, stream, a);
    else br_launch_pdl(br4_sets<128>, dim3(blocks), dim3(BR4_SET_THREADS), smem, stream, a);
}

// Row-wise glue. One block per row, 256 threads.
enum { BR4_LN = 0, BR4_RESIDUAL_LN_PRIV = 1, BR4_CRITIC_OUT = 2 };
struct Br4RowsArgs {
    int mode, rows, W;
    const float* x; int ldx;          // LN input (mode 0), FFN output or h (mode 1), z2 (mode 2)
    const float* res; int ldres;      // residual added to x first (modes 1, 2), or nullptr
    float* y; int ldy;                // mode 1: x + res written back here (the trunk output y), mode 2: unused
    float* out; int ldout;            // LN output; mode 2: the decoder row (value at column vcol)
    float* values;                    // mode 2: the value again, (rows,) (the rollout's values)
    const float* priv; int ldpriv, P; // mode 1: privileged block appended before the LN (P = 0: none)
    int vcol;
    BrGroups g;
    BrLinear cout;                    // mode 2: per-group critic_out (C,)
};
__device__ __forceinline__ float br4_block_sum(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) total += scratch[w];
    return total;
}
__global__ void __launch_bounds__(256) br4_rows(Br4RowsArgs a) {
    __shared__ float scratch[8];
    __shared__ float rowbuf[1024];   // W <= 1024 (hidden + privileged, critic width)
    br_pdl_wait(); br_pdl_release();
    const int row = blockIdx.x, W = a.W;
    const float* x = a.x + (long)row * a.ldx;
    const float* res = a.res ? a.res + (long)row * a.ldres : nullptr;
    int n = W;
    if (a.mode == BR4_RESIDUAL_LN_PRIV) n = W - a.P;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        float v = x[c] + (res ? res[c] : 0.0f);
        rowbuf[c] = v;
        if (a.mode == BR4_RESIDUAL_LN_PRIV) a.y[(long)row * a.ldy + c] = v;
    }
    if (a.mode == BR4_RESIDUAL_LN_PRIV)
        for (int c = threadIdx.x; c < a.P; c += blockDim.x) rowbuf[n + c] = a.priv[(long)row * a.ldpriv + c];
    __syncthreads();
    float s = 0.0f;
    for (int c = threadIdx.x; c < W; c += blockDim.x) s += rowbuf[c];
    float mu = br4_block_sum(s, scratch) / W;
    float q = 0.0f;
    for (int c = threadIdx.x; c < W; c += blockDim.x) { float d = rowbuf[c] - mu; q += d * d; }
    float r = rsqrtf(br4_block_sum(q, scratch) / W + v4::LN_EPS);
    if (a.mode == BR4_CRITIC_OUT) {
        const float* cout = a.cout.w[br_group_of(a.g, row)];
        float dot = 0.0f;
        for (int c = threadIdx.x; c < W; c += blockDim.x) dot = fmaf((rowbuf[c] - mu) * r, cout[c], dot);
        dot = br4_block_sum(dot, scratch);
        if (threadIdx.x == 0) { a.out[(long)row * a.ldout + a.vcol] = dot; a.values[row] = dot; }
    } else {
        for (int c = threadIdx.x; c < W; c += blockDim.x) a.out[(long)row * a.ldout + c] = (rowbuf[c] - mu) * r;
    }
}

struct Br4Wt {   // one group's GEMM-side weights (br_prepare) and the set MLPs' bf16 packs (br4_pack_set_weights)
    float *flat, *fusion, *gru[BR_MAX_LAYERS], *ffn_in, *ffn_out, *actor, *cin, *b1, *b2;
    __nv_bfloat16 *w1t[4], *w2b[4];
};
struct Br4Acts {
    float *joined, *enc, *comb, *gru[2], *ln, *f, *y, *z, *lnc, *b, *z2, *dec;
    cudaStream_t critic; cudaEvent_t fork, join;   // the critic's branch (br4_run, br4_join)
};
struct Br4Plan {
    int FD, D, I, F, C, P, mean_pool, JW, W, LNW;
    Br4Wt wt[BR_MAX_GROUPS];
    Br4Acts* acts;   // [num_buffers]
};

// The v4 chain from the live obs rows to the decoder rows A.dec (logits, value
// at column dec_cols), rollout copies included. The caller samples. Kernels
// only: the trainer (br4_chain) and the forward bench (fwd_bench.cu) call it.
// The critic runs on its own branch beside the actor and the sampler and
// writes the learner rows' values itself: the caller samples on `stream`
// (values from the decoder row for the rows from p.g.end[1] on) and joins
// the branch with br4_join before anything that reads the value column; the
// env step waits only for the actions (br_forward's actions_ready).
// 2026-10-07 (fwd_bench, 200 spec steps): the step's graph 195 -> 178 us on
// 1 stream, 407 -> 388 us per stream with 4 at once; spec trainer eval_gpu
// 0.348 -> 0.305 ms per buffer step (with the sampler's aim pass).
struct Br4Groups { const float* emb[BR_MAX_GROUPS]; const float* cout[BR_MAX_GROUPS]; float* state[BR_MAX_GROUPS]; int state_rows[BR_MAX_GROUPS]; };
static void br4_run(BrPlan& p, int buf, const Br4Groups& gr, const float* obs, const float* rewards, const float* terminals,
                    float* rollout_obs, float* rollout_rewards, float* rollout_terminals, float* values, cudaStream_t stream) {
    Br4Plan& q = *p.v4; Br4Acts& A = q.acts[buf];
    int rows = p.rows, H = p.H;
    Br4SetsArgs sa = {};
    sa.obs = obs; sa.rewards = rewards; sa.terminals = terminals;
    sa.rollout_obs = rollout_obs; sa.rollout_rewards = rollout_rewards; sa.rollout_terminals = rollout_terminals;
    sa.joined = A.joined; sa.ldj = q.JW; sa.col0 = q.FD; sa.D = q.D; sa.I = q.I; sa.mean_pool = q.mean_pool; sa.rows = rows; sa.g = p.g;
    BrLinear w_flat = {}, w_fus = {}, w_fi = {}, w_fo = {}, w_act = {}, w_cin = {}, w_b1 = {}, w_b2 = {}, w_cout = {}, w_gru[BR_MAX_LAYERS] = {};
    BrGateArgs gate = {};
    for (int g = 0; g < p.g.n; g++) {
        for (int k = 0; k < 4; k++) { sa.w.w1t[g][k] = q.wt[g].w1t[k]; sa.w.w2[g][k] = q.wt[g].w2b[k]; }
        sa.w.emb[g] = gr.emb[g];
        w_flat.w[g] = q.wt[g].flat; w_fus.w[g] = q.wt[g].fusion;
        for (int l = 0; l < p.layers; l++) w_gru[l].w[g] = q.wt[g].gru[l];
        w_fi.w[g] = q.wt[g].ffn_in; w_fo.w[g] = q.wt[g].ffn_out; w_act.w[g] = q.wt[g].actor;
        w_cin.w[g] = q.wt[g].cin; w_b1.w[g] = q.wt[g].b1; w_b2.w[g] = q.wt[g].b2; w_cout.w[g] = gr.cout[g];
        gate.state[g] = gr.state[g]; gate.state_rows[g] = gr.state_rows[g];
    }
    // encoder
    br4_sets_launch(sa, stream);
    br_gemm_launch(p, obs, BOPL_OBS_STRIDE, A.joined, q.JW, v4::FLAT, q.FD, q.FD, 1, w_flat, stream);
    br_gemm_launch(p, A.joined, q.JW, A.ln, q.LNW, q.JW, H, H, 0, w_fus, stream);
    Br4RowsArgs ra = {}; ra.rows = rows; ra.g = p.g;
    ra.mode = BR4_LN; ra.W = H; ra.x = A.ln; ra.ldx = q.LNW; ra.out = A.enc; ra.ldout = H;
    br_launch_pdl(br4_rows, dim3(rows), dim3(256), 0, stream, ra);
    // MinGRU
    const float* x = A.enc;
    for (int l = 0; l < p.layers; l++) {
        br_gemm_launch(p, x, H, A.comb, 3 * H, H, 3 * H, 3 * H, 0, w_gru[l], stream);
        gate.combined = A.comb; gate.x = x; gate.terminals = terminals;
        gate.out = A.gru[l & 1]; gate.layer = l; gate.H = H; gate.rows = rows; gate.g = p.g;
        br_launch_pdl(br_gate, dim3(grid_size(rows * H)), dim3(BLOCK_SIZE), 0, stream, gate);
        x = A.gru[l & 1];
    }
    // decoder: y = h + FFN(LN(h)); then LN([y, priv]) for the critic. Without
    // the FFN y is h itself: the actor reads h and the critic's branch forks
    // before the LN, which only the critic reads (the LN's y = h + 0.0f
    // differs from h at most in the sign of a zero, which the GEMM's sum from
    // +0 drops). 2026-10-08 (fwd_bench, 200 spec steps): actions ready
    // 145 -> 142 us on 1 stream.
    const float* priv = obs + BOPL4_OBS_PRIVILEGED_OFFSET;
    ra = Br4RowsArgs{}; ra.rows = rows; ra.g = p.g;
    const float* y = x;
    if (q.F) {
        ra.mode = BR4_LN; ra.W = H; ra.x = x; ra.ldx = H; ra.out = A.ln; ra.ldout = q.LNW;
        br_launch_pdl(br4_rows, dim3(rows), dim3(256), 0, stream, ra);
        br_gemm_launch(p, A.ln, q.LNW, A.f, q.F, H, q.F, q.F, 1, w_fi, stream);
        br_gemm_launch(p, A.f, q.F, A.y, H, q.F, H, H, 0, w_fo, stream);
        ra = Br4RowsArgs{}; ra.rows = rows; ra.g = p.g;
        ra.mode = BR4_RESIDUAL_LN_PRIV; ra.W = q.W; ra.x = A.y; ra.ldx = H; ra.res = x; ra.ldres = H;
        y = A.y;
    } else {
        ra.mode = BR4_RESIDUAL_LN_PRIV; ra.W = q.W; ra.x = x; ra.ldx = H;
    }
    ra.y = A.y; ra.ldy = H; ra.out = A.ln; ra.ldout = q.LNW; ra.priv = priv; ra.ldpriv = BOPL_OBS_STRIDE; ra.P = q.P;
    if (q.F) br_launch_pdl(br4_rows, dim3(rows), dim3(256), 0, stream, ra);   // A.y = y (trunk output), A.ln = LN([y, priv])
    if (q.C) {
        cudaEventRecord(A.fork, stream);
        cudaStreamWaitEvent(A.critic, A.fork, 0);
    }
    br_gemm_launch(p, y, H, A.dec, p.ld_dec, H, p.dec_cols + (q.C ? 0 : 1), p.ld_dec, 0, w_act, stream);
    if (q.C) {
        cudaStream_t cs = A.critic;
        if (!q.F) br_launch_pdl(br4_rows, dim3(p.g.end[1]), dim3(256), 0, cs, ra);   // A.ln = LN([h, priv]), the learner's rows
        // The critic runs over the learner's rows only (group 0, its own row
        // tiles): the frozen banks' values are never read (their advantages
        // are zeroed, the statistics take learner rows), and their value
        // column stays the zero A.dec was allocated with. The banks' rows are
        // 4 of 16 per tile, so this is 4 of the 7 tiles of the three widest
        // GEMMs (K, N = 512) and the two LNs.
        BrPlan pl = p; pl.g.n = 1;
        const int crows = p.g.end[1];
        br_gemm_launch(pl, A.ln, q.LNW, A.z, q.C, q.W, q.C, q.C, 0, w_cin, cs);
        ra = Br4RowsArgs{}; ra.rows = crows; ra.g = p.g;
        ra.mode = BR4_LN; ra.W = q.C; ra.x = A.z; ra.ldx = q.C; ra.out = A.lnc; ra.ldout = q.C;
        br_launch_pdl(br4_rows, dim3(crows), dim3(256), 0, cs, ra);
        br_gemm_launch(pl, A.lnc, q.C, A.b, q.C, q.C, q.C, q.C, 1, w_b1, cs);
        br_gemm_launch(pl, A.b, q.C, A.z2, q.C, q.C, q.C, q.C, 0, w_b2, cs);
        ra = Br4RowsArgs{}; ra.rows = crows; ra.g = p.g;
        ra.mode = BR4_CRITIC_OUT; ra.W = q.C; ra.x = A.z2; ra.ldx = q.C; ra.res = A.z; ra.ldres = q.C;
        ra.out = A.dec; ra.ldout = p.ld_dec; ra.vcol = p.dec_cols; ra.cout = w_cout; ra.values = values;
        br_launch_pdl(br4_rows, dim3(crows), dim3(256), 0, cs, ra);
        cudaEventRecord(A.join, cs);
    }
}
static void br4_join(BrPlan& p, int buf, cudaStream_t stream) {
    if (p.v4 && p.v4->C) cudaStreamWaitEvent(stream, p.v4->acts[buf].join, 0);
}
// The rows whose value the critic writes (the sampler writes the rest from the decoder row).
static int br4_value_rows(const BrPlan& p) { return p.v4 && p.v4->C ? p.g.end[1] : 0; }


#ifndef BR_KERNELS_ONLY
static void br_group_weights(PuffeRL* pufferl, int g, int buf, PolicyWeights** w, PrecisionTensor** s);

// Shape agreement of the learner and the banks (all v4, same widths); nullptr when fine.
static const char* br4_check(PuffeRL* pufferl) {
    auto* ew = (v4::EncWeights*)pufferl->weights.encoder;
    auto* dw = (v4::DecWeights*)pufferl->weights.decoder;
    if (ew->D > BR4_POOL_SPLIT || BR4_POOL_SPLIT % ew->D || (ew->I != 64 && ew->I != 128)) return "v4 set width must divide 256 (inner 64 or 128)";
    if (ew->hidden + v4::PRIV > 1024 || dw->cdim > 1024 || dw->ffn > 1024) return "v4 widths above 1024 are not supported by the batched chain";
    if (ew->flat_dim % 8 || dw->cdim % 8 || dw->ffn % 8) return "v4 flat / ffn / critic widths must be multiples of 8";
    for (int b = 0; b < pufferl->num_frozen_banks; b++) {
        WeightBank* fb = &pufferl->frozen_banks[b];
        if (!bopl_is_v4_encoder(fb->policy.encoder)) return "a frozen bank is not policy v4";
        auto* fe = (v4::EncWeights*)fb->weights.encoder; auto* fd = (v4::DecWeights*)fb->weights.decoder;
        if (fe->D != ew->D || fe->I != ew->I || fe->flat_dim != ew->flat_dim || fe->mean_pool != ew->mean_pool
                || fd->ffn != dw->ffn || fd->cdim != dw->cdim || fd->privileged != dw->privileged || fd->od != dw->od)
            return "a frozen bank differs from the learner in v4 shape";
    }
    return nullptr;
}

static void br4_init(PuffeRL* pufferl, BrPlan* p) {
    auto* ew = (v4::EncWeights*)pufferl->weights.encoder;
    auto* dw = (v4::DecWeights*)pufferl->weights.decoder;
    int apb = pufferl->vec->total_agents / pufferl->hypers.num_buffers, H = p->H;
    Br4Plan* q = new Br4Plan{};
    q->FD = ew->flat_dim; q->D = ew->D; q->I = ew->I; q->mean_pool = ew->mean_pool;
    q->F = dw->ffn; q->C = dw->cdim; q->P = dw->cdim && dw->privileged ? v4::PRIV : 0;
    q->JW = q->FD + 4 * 2 * q->D; q->W = H + q->P;
    q->LNW = std::max(H, std::max(q->W, q->C));
    auto zalloc = [](float** ptr, size_t n) {
        if (!n) { *ptr = nullptr; return; }
        assert(cudaMalloc((void**)ptr, n * sizeof(float)) == cudaSuccess);
        cudaMemset(*ptr, 0, n * sizeof(float));
    };
    q->acts = (Br4Acts*)calloc(pufferl->hypers.num_buffers, sizeof(Br4Acts));
    for (int b = 0; b < pufferl->hypers.num_buffers; b++) {
        Br4Acts& a = q->acts[b];
        zalloc(&a.joined, (size_t)apb * q->JW); zalloc(&a.enc, (size_t)apb * H); zalloc(&a.comb, (size_t)apb * 3 * H);
        zalloc(&a.gru[0], (size_t)apb * H); zalloc(&a.gru[1], (size_t)apb * H);
        zalloc(&a.ln, (size_t)apb * q->LNW); zalloc(&a.f, (size_t)apb * q->F); zalloc(&a.y, (size_t)apb * H);
        zalloc(&a.z, (size_t)apb * q->C); zalloc(&a.lnc, (size_t)apb * q->C); zalloc(&a.b, (size_t)apb * q->C); zalloc(&a.z2, (size_t)apb * q->C);
        zalloc(&a.dec, (size_t)apb * p->ld_dec);
        if (q->C) {   // same (highest) priority and partition as the buffer streams
            int prio_least = 0, prio_greatest = 0;
            cudaDeviceGetStreamPriorityRange(&prio_least, &prio_greatest);
            a.critic = bopl_make_stream(pufferl, true, prio_greatest);
            assert(cudaEventCreateWithFlags(&a.fork, cudaEventDisableTiming) == cudaSuccess);
            assert(cudaEventCreateWithFlags(&a.join, cudaEventDisableTiming) == cudaSuccess);
        }
    }
    for (int g = 0; g < p->g.n; g++) {
        Br4Wt& w = q->wt[g];
        zalloc(&w.flat, (size_t)br_frag_floats(v4::FLAT, q->FD)); zalloc(&w.fusion, (size_t)br_frag_floats(q->JW, H));
        for (int l = 0; l < p->layers; l++) zalloc(&w.gru[l], (size_t)br_frag_floats(H, 3 * H));
        zalloc(&w.ffn_in, q->F ? (size_t)br_frag_floats(H, q->F) : 0); zalloc(&w.ffn_out, q->F ? (size_t)br_frag_floats(q->F, H) : 0);
        zalloc(&w.actor, (size_t)br_frag_floats(H, p->ld_dec));
        zalloc(&w.cin, q->C ? (size_t)br_frag_floats(q->W, q->C) : 0);
        zalloc(&w.b1, q->C ? (size_t)br_frag_floats(q->C, q->C) : 0); zalloc(&w.b2, q->C ? (size_t)br_frag_floats(q->C, q->C) : 0);
        for (int k = 0; k < 4; k++) {
            assert(cudaMalloc((void**)&w.w1t[k], (size_t)v4::SPEC[k].features * q->I * sizeof(__nv_bfloat16)) == cudaSuccess);
            assert(cudaMalloc((void**)&w.w2b[k], (size_t)q->D * q->I * sizeof(__nv_bfloat16)) == cudaSuccess);
        }
    }
    size_t smem = br4_sets_smem(q->I);
    br4_sets_init(q->D, q->I);
    p->v4 = q;
    fprintf(stderr, "bopl batched rollout forward: policy v4 chain (set %d/%d, flat %d, ffn %d, critic %d%s), %.1f KB shared per set block, W2 rows in registers\n",
            q->D, q->I, q->FD, q->F, q->C, q->P ? " privileged" : "", smem / 1024.0);
}

// Every group's GEMM weights into the GEMM-side layout (rollout start).
static void br4_prepare(PuffeRL* pufferl, BrPlan& p, cudaStream_t stream) {
    Br4Plan& q = *p.v4; int H = p.H;
    BrLinear s_flat = {}, s_fus = {}, s_fi = {}, s_fo = {}, s_act = {}, s_cin = {}, s_b1 = {}, s_b2 = {}, s_gru[BR_MAX_LAYERS] = {};
    BrLinear d_flat = {}, d_fus = {}, d_fi = {}, d_fo = {}, d_act = {}, d_cin = {}, d_b1 = {}, d_b2 = {}, d_gru[BR_MAX_LAYERS] = {};
    for (int g = 0; g < p.g.n; g++) {
        PolicyWeights* w; br_group_weights(pufferl, g, 0, &w, nullptr);
        auto* ew = (v4::EncWeights*)w->encoder; auto* mw = (MinGRUWeights*)w->network; auto* dw = (v4::DecWeights*)w->decoder;
        s_flat.w[g] = ew->flat.data; d_flat.w[g] = q.wt[g].flat;
        s_fus.w[g] = ew->fusion.data; d_fus.w[g] = q.wt[g].fusion;
        for (int l = 0; l < p.layers; l++) { s_gru[l].w[g] = mw->weights[l].data; d_gru[l].w[g] = q.wt[g].gru[l]; }
        s_fi.w[g] = dw->ffn_in.data; d_fi.w[g] = q.wt[g].ffn_in;
        s_fo.w[g] = dw->ffn_out.data; d_fo.w[g] = q.wt[g].ffn_out;
        s_act.w[g] = dw->actor.data; d_act.w[g] = q.wt[g].actor;
        s_cin.w[g] = dw->critic_in.data; d_cin.w[g] = q.wt[g].cin;
        s_b1.w[g] = dw->critic_b1.data; d_b1.w[g] = q.wt[g].b1;
        s_b2.w[g] = dw->critic_b2.data; d_b2.w[g] = q.wt[g].b2;
        for (int k = 0; k < 4; k++) {
            int n = q.I * v4::SPEC[k].features + q.D * q.I;
            br4_pack_set_weights<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(ew->w1[k].data, v4::SPEC[k].features, ew->w2[k].data, q.D, q.I, q.wt[g].w1t[k], q.wt[g].w2b[k]);
        }
    }
    br_transpose_launch(p, s_flat, d_flat, p.g.n, q.FD, v4::FLAT, q.FD, stream);
    br_transpose_launch(p, s_fus, d_fus, p.g.n, H, q.JW, H, stream);
    for (int l = 0; l < p.layers; l++) br_transpose_launch(p, s_gru[l], d_gru[l], p.g.n, 3 * H, H, 3 * H, stream);
    if (q.F) { br_transpose_launch(p, s_fi, d_fi, p.g.n, q.F, H, q.F, stream); br_transpose_launch(p, s_fo, d_fo, p.g.n, H, q.F, H, stream); }
    br_transpose_launch(p, s_act, d_act, p.g.n, p.dec_cols + (q.C ? 0 : 1), H, p.ld_dec, stream);   /* cdim 0: the value is the actor's last row */
    if (q.C) {
        br_transpose_launch(p, s_cin, d_cin, p.g.n, q.C, q.W, q.C, stream);
        br_transpose_launch(p, s_b1, d_b1, p.g.n, q.C, q.C, q.C, stream);
        br_transpose_launch(p, s_b2, d_b2, p.g.n, q.C, q.C, q.C, stream);
    }
}

static const float* br4_dec_rows(BrPlan& p, int buf) { return p.v4->acts[buf].dec; }

static void br4_free(PuffeRL* pufferl, BrPlan* p) {
    Br4Plan* q = p->v4;
    if (!q) return;
    for (int b = 0; b < pufferl->hypers.num_buffers; b++) {
        Br4Acts& a = q->acts[b];
        for (float* ptr : {a.joined, a.enc, a.comb, a.gru[0], a.gru[1], a.ln, a.f, a.y, a.z, a.lnc, a.b, a.z2, a.dec}) cudaFree(ptr);
        if (a.critic) { cudaStreamDestroy(a.critic); cudaEventDestroy(a.fork); cudaEventDestroy(a.join); }
    }
    for (int g = 0; g < p->g.n; g++) {
        Br4Wt& w = q->wt[g];
        for (float* ptr : {w.flat, w.fusion, w.ffn_in, w.ffn_out, w.actor, w.cin, w.b1, w.b2}) cudaFree(ptr);
        for (int k = 0; k < 4; k++) { cudaFree(w.w1t[k]); cudaFree(w.w2b[k]); }
        for (int l = 0; l < p->layers; l++) cudaFree(w.gru[l]);
    }
    free(q->acts);
    delete q;
    p->v4 = nullptr;
}

// The trainer's side of one buffer step: per group, the tensors br4_run reads
// that are not the plan's own (the embedding and critic_out as stored, the
// recurrent state of this buffer).
static void br4_chain(PuffeRL* pufferl, BrPlan& p, int buf, const float* obs, const float* rewards, const float* terminals,
                      float* rollout_obs, float* rollout_rewards, float* rollout_terminals, float* values, cudaStream_t stream) {
    Br4Groups gr = {};
    for (int g = 0; g < p.g.n; g++) {
        PolicyWeights* w; PrecisionTensor* s;
        br_group_weights(pufferl, g, buf, &w, &s);
        gr.emb[g] = ((v4::EncWeights*)w->encoder)->embedding.data;
        gr.cout[g] = ((v4::DecWeights*)w->decoder)->critic_out.data;
        gr.state[g] = s->data; gr.state_rows[g] = (int)s->shape[1];
    }
    br4_run(p, buf, gr, obs, rewards, terminals, rollout_obs, rollout_rewards, rollout_terminals, values, stream);
}

// ---------------------------------------------------------------------------
// Self-check (BOPL_BR_CHECK=1, eager steps only): the batched chain's decoder
// rows against the per-policy chain (policy_forward: bopl_v4.cu / the set
// encoder + upstream MinGRU + decoder) on the same observations and the state
// the step started from. Prints the max abs difference per group for the
// logits and the value; the two paths differ by TF32 / bf16 rounding order
// only (~1e-3 abs on logits of order 1). Group 0 is the learner.
struct BrCheckState { float* saved[BR_MAX_GROUPS]; size_t bytes[BR_MAX_GROUPS]; int on; };
static BrCheckState* br_check_state(PuffeRL* pufferl) {
    static BrCheckState st = {};
    static int init = 0;
    if (!init) {
        init = 1;
        st.on = br_env_int("BOPL_BR_CHECK", 0);
        if (st.on) for (int g = 0; g < pufferl->br->g.n; g++) {
            PolicyWeights* w; PrecisionTensor* s; br_group_weights(pufferl, g, 0, &w, &s);
            st.bytes[g] = (size_t)numel(s->shape) * sizeof(float);
            assert(cudaMalloc((void**)&st.saved[g], st.bytes[g]) == cudaSuccess);
        }
    }
    return &st;
}
static bool br_check_eager(cudaStream_t stream) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cs);
    return cs == cudaStreamCaptureStatusNone;
}
static void br_check_pre(PuffeRL* pufferl, int buf, cudaStream_t stream) {
    BrCheckState* st = br_check_state(pufferl);
    if (!st->on || !br_check_eager(stream)) return;
    for (int g = 0; g < pufferl->br->g.n; g++) {
        PolicyWeights* w; PrecisionTensor* s; br_group_weights(pufferl, g, buf, &w, &s);
        cudaMemcpyAsync(st->saved[g], s->data, st->bytes[g], cudaMemcpyDeviceToDevice, stream);
    }
}
static void br_check_post(PuffeRL* pufferl, int buf, int t, const float* terminals, const float* dec_rows, cudaStream_t stream) {
    BrCheckState* st = br_check_state(pufferl);
    if (!st->on || !br_check_eager(stream)) return;
    BrPlan& p = *pufferl->br;
    HypersT& hypers = pufferl->hypers;
    RolloutBuf rollouts = rollout_slot_view(pufferl->rollouts, pufferl->write_slot, hypers.horizon);
    int apb = pufferl->vec->total_agents / hypers.num_buffers, start = buf * apb;
    br4_join(p, buf, stream);
    cudaStreamSynchronize(stream);
    std::vector<float> ours((size_t)p.rows * p.ld_dec);
    cudaMemcpy(ours.data(), dec_rows, ours.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (int g = 0; g < p.g.n; g++) {
        int off = p.g.end[g], n = p.g.end[g + 1] - off;
        if (!n) continue;
        Policy* pol; PolicyWeights* w; PolicyActivations* a; PrecisionTensor* s;
        if (g == 0) { pol = &pufferl->policy; a = &pufferl->buffer_activations[buf]; }
        else { pol = &pufferl->frozen_banks[g - 1].policy; a = &pufferl->frozen_banks[g - 1].buffer_activations[buf]; }
        br_group_weights(pufferl, g, buf, &w, &s);
        // the step's starting state, reset where the env's terminal is set, in a scratch tensor
        PrecisionTensor state = *s; state.data = st->saved[g];
        int layers = (int)s->shape[0], H = (int)s->shape[2];
        bopl_mask_state_rows<<<grid_size(n * H), BLOCK_SIZE, 0, stream>>>(state.data, terminals + off, n, layers, (int)s->shape[1], H);
        PrecisionTensor obs = puf_slice(rollouts.observations, t, start + off, n);
        PrecisionTensor dec = policy_forward(pol, *w, *a, obs, state, stream);
        cudaStreamSynchronize(stream);
        int od1 = (int)dec.shape[1];
        std::vector<float> theirs((size_t)n * od1);
        cudaMemcpy(theirs.data(), dec.data, theirs.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double dl = 0, dv = 0, scale = 0;
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < p.dec_cols; c++) {
                double o = ours[(size_t)(off + r) * p.ld_dec + c], e = theirs[(size_t)r * od1 + c];
                dl = std::max(dl, std::fabs(o - e)); scale = std::max(scale, std::fabs(e));
            }
            if (g == 0 || !p.v4)   /* v4: the banks' value column is not computed (the critic runs on learner rows only) */
                dv = std::max(dv, std::fabs((double)ours[(size_t)(off + r) * p.ld_dec + p.dec_cols] - theirs[(size_t)r * od1 + p.dec_cols]));
        }
        fprintf(stderr, "bopl batched forward check: buffer %d step %d group %d (%d rows): logits max|diff| %.3g (max|logit| %.3g), value max|diff| %.3g\n",
                buf, t, g, n, dl, scale, dv);
    }
}
#endif  // BR_KERNELS_ONLY

// ---- packed observation transport (bopl_obs_pack.h) ----------------------
// One block per row, after the H2D copy of the slots' first
// BR4_PACK_DMA_WORDS words: first the rest of the slot's used words from the
// mapped host slot (bopl4_slot_used), then per mode. The blocks past `rows` copy the reward /
// terminal / mask pack in the buffer's idle tail (tail_bytes, 0 when the
// pack is copied on its own). Mode 4 rows are done then. Mode 0: the flat and privileged blocks, then each set's
// prefix rows followed by zeros to the block's end; mode 2 the same without
// the terrain and rays, which bopl4_rays_rows writes next. Mode 1 (the slot did not
// fit, or the row has no writer record): the dense row straight from the
// mapped host buffer; mode 3 the same, its terrain and rays from bopl4_rays_rows. Either way the device row equals the host row bit for
// bit (bopl4_transport_check counts the floats that differ, dump mode only).
// A slot's used words, from its head in device memory: a mode-4 slot's head
// gives them, a mode-1 slot is its header, a mode-3 slot its header and scene
// (bopl4f_scene_words; the counts are in the DMA head), the rest the whole
// slot. 2026-10-08 (fwd_bench, 1 stream, nsys): mode 3 read its whole slot,
// 5 KB of PCIe reads a row for a scene the DMA head already held; ~half the
// dumps carry 2 to 10 such rows (45+ entities), and the unpack's mean drops
// 13.1 to 11.3 us, the sample's end 2.0 us.
__device__ __forceinline__ int bopl4_slot_used(const unsigned* hdr) {
    const unsigned mode = hdr[0];
    if (mode == BOPL4_PACK_MODE_STATE) return (int)hdr[BOPL4_PACK_STATE_WORDS];
    if (mode == BOPL4_PACK_MODE_DENSE) return BOPL4_PACK_HEADER;
    if (mode == BOPL4_PACK_MODE_DENSE_SCENE && hdr[5] + 4 <= (unsigned)BR4_PACK_DMA_WORDS) {
        const unsigned* sc = hdr + hdr[5];
        return (int)hdr[5] + BOPL4F_SCENE_HEAD + 5 * (int)sc[1] + 9 * (int)sc[2] + 6 * (int)sc[3];
    }
    return BOPL4_PACK_P;
}
__global__ void bopl4_unpack_rows(const float* __restrict__ host_packed, float* __restrict__ packed, const float* __restrict__ host_rows,
                                  float* __restrict__ dense, int rows, int tail_bytes) {
    br_pdl_wait(); br_pdl_release();   /* launched after a memcpy (no PDL edge in); lets br4_sets pre-launch */
    const int r = blockIdx.x;
    if (r >= rows) {   // (the 16-byte units round up into the idle tail's own slots)
        const float4* src = (const float4*)(host_packed + (long)rows * BOPL4_PACK_P);
        float4* dst = (float4*)(packed + (long)rows * BOPL4_PACK_P);
        const int i = (r - rows) * blockDim.x + threadIdx.x;
        if (i < (tail_bytes + 15) / 16) dst[i] = src[i];
        return;
    }
    float* slot = packed + (long)r * BOPL4_PACK_P;
    const unsigned* hdr = (const unsigned*)slot;
    const unsigned mode = hdr[0];
    {
        const float4* src = (const float4*)(host_packed + (long)r * BOPL4_PACK_P);
        for (int i = BR4_PACK_DMA_WORDS / 4 + threadIdx.x; i < (bopl4_slot_used(hdr) + 3) / 4; i += blockDim.x) ((float4*)slot)[i] = src[i];
    }
    float* out = dense + (long)r * BOPL_OBS_STRIDE;
    if (mode == BOPL4_PACK_MODE_STATE) return;   /* bopl4_rays_rows writes the whole row */
    if (mode == BOPL4_PACK_MODE_DENSE || mode == BOPL4_PACK_MODE_DENSE_SCENE) {
        // Every load in flight before the first store: one PCIe round trip, not one per 4 float4s.
        constexpr int N = BOPL_OBS_STRIDE / 4, PER = (N + BR4_UNPACK_THREADS - 1) / BR4_UNPACK_THREADS;
        const float4* src = (const float4*)(host_rows + (long)r * BOPL_OBS_STRIDE);
        float4 v[PER];
        #pragma unroll
        for (int k = 0; k < PER; k++) { const int i = k * BR4_UNPACK_THREADS + threadIdx.x; if (i < N) v[k] = src[i]; }
        #pragma unroll
        for (int k = 0; k < PER; k++) { const int i = k * BR4_UNPACK_THREADS + threadIdx.x; if (i < N) ((float4*)out)[i] = v[k]; }
        return;
    }
    __syncthreads();
    // Mode 2 leaves the terrain and rays (the flat block's tail) to bopl4_rays_rows.
    const int scene = mode == BOPL4_PACK_MODE_SCENE, flat = scene ? BOPL4_OBS_TERRAIN_OFFSET : BOPL4_FLAT;
    for (int i = threadIdx.x; i < flat; i += blockDim.x) out[i] = slot[BOPL4_PACK_HEADER + i];
    for (int i = threadIdx.x; i < BOPL4_PRIVILEGED; i += blockDim.x) out[BOPL4_OBS_PRIVILEGED_OFFSET + i] = slot[BOPL4_PACK_HEADER + flat + i];
    const Bopl4PackSet sets[BOPL4_PACK_SETS] = BOPL4_PACK_TABLE;
    int base = scene ? BOPL4_PACK_SCENE_ROWS_OFFSET : BOPL4_PACK_ROWS_OFFSET;
    #pragma unroll
    for (int s = 0; s < BOPL4_PACK_SETS; s++) {
        const int present = (int)hdr[1 + s] * sets[s].width, block = sets[s].slots * sets[s].width;
        float* dst = out + sets[s].offset;
        for (int i = threadIdx.x; i < block; i += blockDim.x) dst[i] = i < present ? slot[base + i] : 0.0f;
        base += present;
    }
}
__global__ void bopl4_transport_check(const float* __restrict__ dense, const float* __restrict__ host_rows, int rows, int* mismatches) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const unsigned* a = (const unsigned*)(dense + (long)r * BOPL_OBS_STRIDE);
    const unsigned* b = (const unsigned*)(host_rows + (long)r * BOPL_OBS_STRIDE);
    int bad = 0;
    for (int i = threadIdx.x; i < BOPL_OBS_STRIDE; i += blockDim.x) bad += a[i] != b[i];
    if (bad) atomicAdd(mismatches, bad);
}
