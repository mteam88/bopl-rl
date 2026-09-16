/* Policy v4 for the PufferLib CUDA trainer (POLICY_V4_PLAN.md sections 3, 5):
 * the observation encoder and the decoder, selected by BOPL_NATIVE_ENCODER=v4
 * (ocean.cu). research/policy_v4.py is the oracle; puffer/check_v4.py the
 * parity harness.
 *
 * Encoder (in_dim = BOPL_OBS_STRIDE rows of layout v4, out_dim = hidden):
 *   flat   = ReLU(W_flat obs[:BOPL4_FLAT])                     cuBLAS
 *   rows   = the valid rows of every set, compacted per set across the
 *            minibatch by a deterministic per-sample prefix sum into one
 *            packed bf16 matrix (rows, KPAD); the row count lives on the
 *            device so the launch shape is fixed (CUDA-graph capture) while
 *            the tensor-core work is proportional to the valid rows
 *   h1     = ReLU(W1 x + E[id])   h2 = ReLU(W2 h1)               bf16 WMMA
 *   [attention hook: bopl_v4_set_stage, below]
 *   pool   = [max_rows h2, mean_rows h2] per set (empty set -> 0)
 *   out    = LN(W_fusion [flat, pool x 4])                      cuBLAS
 * Decoder (hidden -> [logits..., value] row, stride od1_pad as upstream):
 *   y      = h + W_o ReLU(W_i LN(h))
 *   logits = W_actor y
 *   z      = W_cin LN([y (detached if critic_detach), privileged])
 *   value  = W_v LN(z + W_b2 ReLU(W_b1 LN(z)))
 * The privileged block comes from the observation rows the encoder gathered
 * (the decoder only receives h): the encoder packs it into a (B, 48) buffer
 * the decoder reads (g_v4_privileged), in the same policy_forward call.
 *
 * Row budget: each set's packed capacity is B x BOPL4_BUDGET_<set> rows
 * (BOPL4_ROW_BUDGET="p,s,e,l" overrides); rows past the capacity are dropped
 * in sample order and counted (bopl_v4_overflow), never silently.
 * Determinism: every reduction is a fixed-order chunked partial sum.
 */
#ifndef BOPL_V4_CU
#define BOPL_V4_CU
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "bopl_layout_v4.h"

namespace v4 {
using bf16 = __nv_bfloat16;
using namespace nvcuda;

constexpr int STRIDE = BOPL_OBS_STRIDE;
constexpr int FLAT = BOPL4_FLAT;
constexpr int PRIV = BOPL4_PRIVILEGED;
constexpr int PRIV_OFF = BOPL4_OBS_PRIVILEGED_OFFSET;
constexpr int IDS = BOPL_EMBEDDING_IDS;
constexpr int SETS = 4;
struct Spec { int offset, rows, width, id_col, features, kpad; };
__host__ __device__ constexpr int round16(int x) { return (x + 15) / 16 * 16; }
constexpr Spec SPEC[SETS] = {
    {BOPL4_OBS_PLAYER_OFFSET, BOPL4_PLAYERS, BOPL4_PLAYER_FEATURES, -1, BOPL4_PLAYER_FEATURES, round16(BOPL4_PLAYER_FEATURES)},
    {BOPL4_OBS_SLOT_OFFSET, BOPL4_SLOTS, BOPL4_SLOT_FEATURES, BOPL4_SLOT_ABILITY_ID, BOPL4_SLOT_FEATURES - 1, round16(BOPL4_SLOT_FEATURES - 1)},
    {BOPL4_OBS_ENTITY_OFFSET, BOPL4_ENTITIES, BOPL4_ENTITY_FEATURES, BOPL4_ENTITY_ABILITY_ID, BOPL4_ENTITY_FEATURES - 1, round16(BOPL4_ENTITY_FEATURES - 1)},
    {BOPL4_OBS_SLIME_OFFSET, BOPL4_SLIMES, BOPL4_SLIME_FEATURES, -1, BOPL4_SLIME_FEATURES, round16(BOPL4_SLIME_FEATURES)},
};
__constant__ Spec c_spec[SETS] = {
    {BOPL4_OBS_PLAYER_OFFSET, BOPL4_PLAYERS, BOPL4_PLAYER_FEATURES, -1, BOPL4_PLAYER_FEATURES, round16(BOPL4_PLAYER_FEATURES)},
    {BOPL4_OBS_SLOT_OFFSET, BOPL4_SLOTS, BOPL4_SLOT_FEATURES, BOPL4_SLOT_ABILITY_ID, BOPL4_SLOT_FEATURES - 1, round16(BOPL4_SLOT_FEATURES - 1)},
    {BOPL4_OBS_ENTITY_OFFSET, BOPL4_ENTITIES, BOPL4_ENTITY_FEATURES, BOPL4_ENTITY_ABILITY_ID, BOPL4_ENTITY_FEATURES - 1, round16(BOPL4_ENTITY_FEATURES - 1)},
    {BOPL4_OBS_SLIME_OFFSET, BOPL4_SLIMES, BOPL4_SLIME_FEATURES, -1, BOPL4_SLIME_FEATURES, round16(BOPL4_SLIME_FEATURES)},
};
constexpr int MAX_KPAD = 48;
static_assert(SPEC[0].kpad <= MAX_KPAD && SPEC[1].kpad <= MAX_KPAD && SPEC[2].kpad <= MAX_KPAD && SPEC[3].kpad <= MAX_KPAD, "kpad");
constexpr int DEFAULT_BUDGET[SETS] = {6, 12, 20, 8};   /* average valid rows per sample the packed buffers hold */
constexpr int TILE = 64, BK = 16;                      /* WMMA block tile: 64 x 64, K step 16, 4 warps */
constexpr int CHUNK = 512;                             /* rows per weight-gradient partial (one block column each) */
constexpr float LN_EPS = 1e-5f;

/* Stage timing for the harness (check_v4.cu): when enabled, an event per
 * mark() on the stream; never under graph capture. */
struct Prof { bool on = false; int n = 0; const char* name[64]; cudaEvent_t ev[64]; };
static Prof g_prof;
static void prof_mark(const char* name, cudaStream_t stream) {
    if (!g_prof.on || g_prof.n >= 64) return;
    if (!g_prof.ev[g_prof.n]) cudaEventCreate(&g_prof.ev[g_prof.n]);
    g_prof.name[g_prof.n] = name; cudaEventRecord(g_prof.ev[g_prof.n], stream); g_prof.n++;
}

struct Gather { const float* obs; const int* idx; int T, B_total; };
__device__ __forceinline__ const float* row_ptr(const float* input, const Gather& g, int row) {
    if (g.obs == nullptr) return input + (long)row * STRIDE;
    int segment = row / g.T, tick = row - segment * g.T;
    return g.obs + ((long)tick * g.B_total + g.idx[segment]) * STRIDE;
}
__device__ __forceinline__ int embedding_id(float value) {
    int id = (int)value;
    return id >= 0 && id < IDS && value == (float)id ? id : 0;
}
__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 tobf(float v) { return __float2bfloat16(v); }

/* ---- packing ---------------------------------------------------------- */
/* Valid rows (column 0 != 0) of each set per sample. */
__global__ void count_rows(const float* input, Gather g, int B, int* counts /* (SETS, B) */) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * SETS) return;
    int s = i / B, b = i - s * B;
    const float* row = row_ptr(input, g, b);
    const Spec sp = c_spec[s];
    int n = 0;
    for (int r = 0; r < sp.rows; r++) n += row[sp.offset + r * sp.width] != 0.0f;
    counts[i] = n;
}
/* Exclusive scan of one set's counts (one block per set, fixed order), the
 * clamped total and the overflow past the capacity. */
__global__ void scan_rows(const int* counts, int B, int4 caps, int* offsets, int* totals, int* overflow) {
    __shared__ int sums[1024];
    __shared__ int carry;
    int s = blockIdx.x;
    const int* c = counts + (long)s * B; int* o = offsets + (long)s * B;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < B; base += blockDim.x) {
        int i = base + threadIdx.x;
        int v = i < B ? c[i] : 0;
        sums[threadIdx.x] = v;
        __syncthreads();
        for (int stride = 1; stride < blockDim.x; stride <<= 1) {   /* Hillis-Steele inclusive scan */
            int add = threadIdx.x >= stride ? sums[threadIdx.x - stride] : 0;
            __syncthreads();
            sums[threadIdx.x] += add;
            __syncthreads();
        }
        if (i < B) o[i] = carry + sums[threadIdx.x] - v;
        __syncthreads();
        if (threadIdx.x == 0) carry += sums[blockDim.x - 1];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int cap = s == 0 ? caps.x : s == 1 ? caps.y : s == 2 ? caps.z : caps.w;
        totals[s] = carry < cap ? carry : cap;
        overflow[s] = carry > cap ? carry - cap : 0;
    }
}
/* Scatter set s's valid rows into the packed bf16 matrix (KPAD columns, zero
 * padded), with their embedding id and sample index. */
template<int S>
__global__ void scatter_rows(const float* input, Gather g, int B, const int* offsets, int cap,
                             bf16* packed, int* ids, int* sample) {
    const Spec sp = c_spec[S];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * sp.rows) return;
    int b = i / sp.rows, r = i - b * sp.rows;
    const float* row = row_ptr(input, g, b) + sp.offset;
    if (row[r * sp.width] == 0.0f) return;
    int rank = 0;
    for (int j = 0; j < r; j++) rank += row[j * sp.width] != 0.0f;
    int dst = offsets[(long)S * B + b] + rank;
    if (dst >= cap) return;
    const float* src = row + r * sp.width;
    bf16* out = packed + (long)dst * sp.kpad;
    for (int k = 0; k < sp.kpad; k++) out[k] = tobf(k < sp.features ? src[k] : 0.0f);
    ids[dst] = sp.id_col >= 0 ? embedding_id(src[sp.id_col]) : 0;
    sample[dst] = b;
}
/* The flat block and the privileged block of every row. */
__global__ void pack_flat(const float* input, Gather g, int B, float* flat, float* priv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * (FLAT + PRIV)) return;
    int b = i / (FLAT + PRIV), c = i - b * (FLAT + PRIV);
    const float* row = row_ptr(input, g, b);
    if (c < FLAT) flat[(long)b * FLAT + c] = row[c];
    else priv[(long)b * PRIV + c - FLAT] = row[PRIV_OFF + c - FLAT];
}
__global__ void to_bf16(const float* src, bf16* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = tobf(src[i]);
}
/* W (rows, cols) -> bf16 (rows, kpad) with zero padding. */
__global__ void to_bf16_padded(const float* src, bf16* dst, int rows, int cols, int kpad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * kpad) return;
    int r = i / kpad, c = i - r * kpad;
    dst[i] = tobf(c < cols ? src[r * cols + c] : 0.0f);
}

/* ---- the masked-M tensor-core GEMM -------------------------------------
 * C[m][n] = sum_k A(m,k) B(k,n) for m < *m_count (rows past the count are
 * skipped, and read as zero along the reduction). A(m,k) is A[m*lda+k] or,
 * A_COL, A[k*lda+m]; B(k,n) is B[k*ldb+n] or, B_COL, B[n*ldb+k]. The K range
 * is [k_begin, k_end) with k_end clamped by *k_count when given (the
 * weight-gradient reduction over packed rows). The epilogue receives
 * (m, n, value) for every element of the tile within M and N. */
struct EpiRelu {            /* h = ReLU(v + E[id[m]][n]) -> bf16 */
    bf16* out; int ldo; const bf16* E; const int* ids; int ldE;
    static constexpr bool store_zero = false;
    __device__ void chunk(int) {}
    __device__ void operator()(int m, int n, float v) const {
        if (E) v += bf(E[(long)ids[m] * ldE + n]);
        out[(long)m * ldo + n] = tobf(v > 0.0f ? v : 0.0f);
    }
};
struct EpiMask {            /* g = v where act > 0 else 0 -> bf16 */
    bf16* out; int ldo; const bf16* act;
    static constexpr bool store_zero = false;
    __device__ void chunk(int) {}
    __device__ void operator()(int m, int n, float v) const {
        out[(long)m * ldo + n] = tobf(bf(act[(long)m * ldo + n]) > 0.0f ? v : 0.0f);
    }
};
struct EpiStore {           /* fp32 partial of chunk blockIdx.z: out + z * per_chunk; an empty chunk stores zeros */
    float* out; int ldo; long per_chunk;
    static constexpr bool store_zero = true;
    __device__ void chunk(int z) { out += z * per_chunk; }
    __device__ void operator()(int m, int n, float v) const { out[(long)m * ldo + n] = v; }
};
template<bool A_COL, bool B_COL, class Epi>
__global__ void __launch_bounds__(128) wmma_gemm(const bf16* __restrict__ A, int lda, const bf16* __restrict__ B, int ldb,
                                                 int M, const int* m_count, int N, int k_begin, int k_end, const int* k_count,
                                                 int k_chunk, Epi epi) {
    __shared__ __align__(32) bf16 sA[TILE][BK + 8];
    __shared__ __align__(32) bf16 sB[BK][TILE + 8];
    __shared__ __align__(32) float sC[TILE][TILE + 4];
    int m0 = blockIdx.x * TILE, n0 = blockIdx.y * TILE;
    int m_valid = m_count ? min(M, *m_count) : M;
    if (m0 >= m_valid) return;
    if (k_chunk) { k_begin += blockIdx.z * k_chunk; k_end = min(k_end, k_begin + k_chunk); }
    if (k_count) k_end = min(k_end, *k_count);
    epi.chunk(blockIdx.z);
    if (k_begin >= k_end && !epi.store_zero) return;   /* an empty chunk of a partial stores zeros */
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);
    for (int k0 = k_begin; k0 < k_end; k0 += BK) {
        /* A tile (TILE x BK) and B tile (BK x TILE): 1024 elements each, 8 per thread. */
        for (int e = threadIdx.x; e < TILE * BK; e += 128) {
            /* consecutive threads read consecutive addresses of the source layout */
            int am = A_COL ? e % TILE : e / BK, ak = A_COL ? e / TILE : e % BK;
            int m = m0 + am, k = k0 + ak;
            bf16 v = tobf(0.0f);
            if (m < m_valid && k < k_end) v = A_COL ? A[(long)k * lda + m] : A[(long)m * lda + k];
            sA[am][ak] = v;
            int bk = B_COL ? e % BK : e / TILE, bn = B_COL ? e / BK : e % TILE;
            int kk = k0 + bk, n = n0 + bn;
            bf16 w = tobf(0.0f);
            if (kk < k_end && n < N) w = B_COL ? B[(long)n * ldb + kk] : B[(long)kk * ldb + n];
            sB[bk][bn] = w;
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> fb[2];
        for (int i = 0; i < 2; i++) wmma::load_matrix_sync(fa[i], &sA[wm + i * 16][0], BK + 8);
        for (int j = 0; j < 2; j++) wmma::load_matrix_sync(fb[j], &sB[0][wn + j * 16], TILE + 8);
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        __syncthreads();
    }
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
        wmma::store_matrix_sync(&sC[wm + i * 16][wn + j * 16], acc[i][j], TILE + 4, wmma::mem_row_major);
    __syncthreads();
    for (int e = threadIdx.x; e < TILE * TILE; e += 128) {
        int cm = e / TILE, cn = e - cm * TILE;
        int m = m0 + cm, n = n0 + cn;
        if (m < m_valid && n < N) epi(m, n, sC[cm][cn]);
    }
    (void)lane;
}
template<bool A_COL, bool B_COL, class Epi>
static void gemm(const bf16* A, int lda, const bf16* B, int ldb, int M, const int* m_count, int N,
                 int k_begin, int k_end, const int* k_count, Epi epi, cudaStream_t stream) {
    dim3 grid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    wmma_gemm<A_COL, B_COL, Epi><<<grid, 128, 0, stream>>>(A, lda, B, ldb, M, m_count, N, k_begin, k_end, k_count, 0, epi);
}
/* Weight gradient dW (N x K) = sum over packed rows m of dY[m][n] X[m][k],
 * as fixed-order chunk partials then one reduction. dY (rows, N) bf16, X
 * (rows, K) bf16; rows past *row_count read as zero. partials (chunks, N, K). */
__global__ void reduce_partials(const float* partials, int chunks, long per_chunk, int n, int k_full, int k_keep,
                                float* out, int accumulate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * k_keep) return;
    int r = i / k_keep, c = i - r * k_keep;
    float sum = 0.0f;
    for (int ch = 0; ch < chunks; ch++) sum += partials[(long)ch * per_chunk + (long)r * k_full + c];
    out[i] = accumulate ? out[i] + sum : sum;
}
static void weight_gradient(const bf16* dY, int N, const bf16* X, int K, int cap, const int* row_count,
                            float* partials, float* out, int k_keep, int accumulate, cudaStream_t stream) {
    int chunks = (cap + CHUNK - 1) / CHUNK;
    long per_chunk = (long)N * K;
    /* C[n][k] over m: A(n, m) = dY[m][n] (A_COL, lda N), B(m, k) = X[m][k] (ldb K); one z block per chunk of rows. */
    dim3 grid((N + TILE - 1) / TILE, (K + TILE - 1) / TILE, chunks);
    wmma_gemm<true, false, EpiStore><<<grid, 128, 0, stream>>>(dY, N, X, K, N, nullptr, K, 0, cap, row_count, CHUNK,
                                                              EpiStore{partials, K, per_chunk});
    reduce_partials<<<grid_size(N * k_keep), BLOCK_SIZE, 0, stream>>>(partials, chunks, per_chunk, N, K, k_keep, out, accumulate);
}

/* ---- pooling -------------------------------------------------------------- */
/* joined[b][col0 + s*2D + c] = max, [.. + D + c] = mean over the sample's rows. */
__global__ void pool_rows(const bf16* h2, int D, const int* offsets, const int* counts, int cap, int total_index,
                          const int* totals, float* joined, int ldj, int col0, int* winner, int mean_pool) {
    int b = blockIdx.x, c = threadIdx.x;
    int off = offsets[b], n = counts[b], total = totals[total_index];
    if (off + n > total) n = max(0, total - off);   /* rows past the capacity were dropped */
    float best = 0.0f, sum = 0.0f; int win = -1;
    for (int r = 0; r < n; r++) {
        float v = bf(h2[(long)(off + r) * D + c]);
        if (v > best) { best = v; win = r; }
        sum += v;
    }
    joined[(long)b * ldj + col0 + c] = best;
    joined[(long)b * ldj + col0 + D + c] = (mean_pool && n > 0) ? sum / n : 0.0f;
    winner[(long)b * D + c] = win;
    (void)cap;
}
/* dh2[r][c] = ReLU'(h2) * (winner == r ? gmax : 0) + gmean / n  for packed row r. */
__global__ void unpool_rows(const bf16* h2, bf16* dh2, int D, const int* offsets, const int* counts, const int* sample,
                            const int* totals, int total_index, const float* grad_joined, int ldj, int col0,
                            const int* winner, int mean_pool) {
    int r = blockIdx.x, c = threadIdx.x;
    if (r >= totals[total_index]) return;
    int b = sample[r], off = offsets[b], n = counts[b], total = totals[total_index];
    if (off + n > total) n = max(0, total - off);
    float g = 0.0f;
    if (winner[(long)b * D + c] == r - off) g += grad_joined[(long)b * ldj + col0 + c];
    if (mean_pool && n > 0) g += grad_joined[(long)b * ldj + col0 + D + c] / n;
    dh2[(long)r * D + c] = tobf(bf(h2[(long)r * D + c]) > 0.0f ? g : 0.0f);
}
/* Embedding gradient partials: one block per chunk of packed rows, fixed order. */
__global__ void embedding_partials(const bf16* dh1, int I, const int* ids, const int* totals, int total_index, int cap,
                                   float* partials /* (chunks, IDS, I) */) {
    extern __shared__ float acc[];   /* IDS x I */
    int ch = blockIdx.x, i = threadIdx.x;
    for (int e = i; e < IDS * I; e += blockDim.x) acc[e] = 0.0f;
    __syncthreads();
    int total = totals[total_index];
    int begin = ch * CHUNK, end = min(cap, min(total, begin + CHUNK));
    for (int r = begin; r < end; r++) acc[ids[r] * I + i] += bf(dh1[(long)r * I + i]);
    __syncthreads();
    for (int e = i; e < IDS * I; e += blockDim.x) partials[(long)ch * IDS * I + e] = acc[e];
}

/* ---- elementwise and LayerNorm ------------------------------------------ */
__global__ void relu_into(const float* src, float* dst, int rows, int cols, int ldd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i - r * cols;
    float v = src[i];
    dst[(long)r * ldd + c] = v > 0.0f ? v : 0.0f;
}
/* dst[r][c] = grad[r][c] * (act[r][c] > 0), reading grad with stride ldg. */
__global__ void relu_grad_from(const float* grad, int ldg, const float* act, int lda, float* dst, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i - r * cols;
    dst[i] = act[(long)r * lda + c] > 0.0f ? grad[(long)r * ldg + c] : 0.0f;
}
__global__ void add_into(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}
__global__ void copy_cols(const float* src, int lds, int col0, float* dst, int ldd, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i - r * cols;
    dst[(long)r * ldd + c] = src[(long)r * lds + col0 + c];
}
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) total += scratch[w];
    return total;
}
/* y = (x - mean) * rstd over W columns, one block (256 threads) per row. */
__global__ void ln_forward(const float* x, float* y, float* mean, float* rstd, int W) {
    __shared__ float scratch[8];
    const float* xr = x + (long)blockIdx.x * W; float* yr = y + (long)blockIdx.x * W;
    float s = 0.0f;
    for (int c = threadIdx.x; c < W; c += blockDim.x) s += xr[c];
    float mu = block_sum(s, scratch) / W;
    float q = 0.0f;
    for (int c = threadIdx.x; c < W; c += blockDim.x) { float d = xr[c] - mu; q += d * d; }
    float var = block_sum(q, scratch) / W;
    float r = rsqrtf(var + LN_EPS);
    for (int c = threadIdx.x; c < W; c += blockDim.x) yr[c] = (xr[c] - mu) * r;
    if (threadIdx.x == 0) { mean[blockIdx.x] = mu; rstd[blockIdx.x] = r; }
}
/* dx (+)= rstd * (dy - mean(dy) - xhat * mean(dy * xhat)). */
__global__ void ln_backward(const float* dy, const float* x, const float* mean, const float* rstd, float* dx, int W, int accumulate) {
    __shared__ float scratch[8];
    long row = blockIdx.x;
    const float* dyr = dy + row * W; const float* xr = x + row * W; float* dxr = dx + row * W;
    float mu = mean[row], r = rstd[row];
    float s1 = 0.0f, s2 = 0.0f;
    for (int c = threadIdx.x; c < W; c += blockDim.x) { float xh = (xr[c] - mu) * r; s1 += dyr[c]; s2 += dyr[c] * xh; }
    float m1 = block_sum(s1, scratch) / W;
    float m2 = block_sum(s2, scratch) / W;
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        float xh = (xr[c] - mu) * r;
        float v = r * (dyr[c] - m1 - xh * m2);
        dxr[c] = accumulate ? dxr[c] + v : v;
    }
}

/* ---- attention hook ------------------------------------------------------
 * Insertion point between the row MLP and the pool (POLICY_V4_PLAN.md,
 * section 3). A stage receives every set's packed rows and must transform
 * them in place:
 *   rows     bf16 (cap, D) row-major, the valid rows are [0, *total)
 *   sample   int (cap): the minibatch sample of each packed row; rows of one
 *            sample are contiguous, [offsets[b], offsets[b] + counts[b])
 *   offsets, counts   int (B); B samples; D channels; the CUDA stream
 * A stage that needs activations for its backward registers them through
 * reg (Allocator* acts, int cap) and receives grad rows (cap, D) bf16 in
 * backward, to be overwritten with the gradient wrt its input rows. nullptr
 * = identity (the shipped v4). */
struct SetStage {
    void (*reg)(Allocator* acts, int set, int cap, int D);
    void (*forward)(int set, bf16* rows, const int* sample, const int* offsets, const int* counts, const int* total, int B, int D, cudaStream_t stream);
    void (*backward)(int set, const bf16* rows, bf16* grad_rows, const int* sample, const int* offsets, const int* counts, const int* total, int B, int D, cudaStream_t stream);
};
static SetStage g_set_stage = {nullptr, nullptr, nullptr};
void bopl_v4_set_stage(SetStage stage) { g_set_stage = stage; }

/* ---- encoder module -------------------------------------------------------- */
struct EncWeights {
    PrecisionTensor w1[SETS], w2[SETS], embedding, flat, fusion;
    int hidden, D, I, flat_dim, mean_pool;
    int budget[SETS];
};
struct EncActs {
    Gather gather;
    const float* saved_input;
    int B, cap[SETS];
    PrecisionTensor flat_input, flat_pre, joined, fused, out, priv;
    FloatTensor ln_mean, ln_rstd;
    IntTensor counts, offsets, totals, overflow, ids[SETS], sample[SETS], winner[SETS];
    IntTensor w1b[SETS], w2b[SETS], embb;              /* bf16 weight copies, 2 per int */
    IntTensor xpack[SETS], h1[SETS], h2[SETS];         /* bf16 packed rows, 2 per int */
    /* training */
    PrecisionTensor grad_fused, grad_joined, grad_flat, partials;
    IntTensor dh2[SETS];
    PrecisionTensor wgrad_w1[SETS], wgrad_w2[SETS], wgrad_embedding, wgrad_flat, wgrad_fusion;
};
static const float* g_v4_privileged = nullptr;    /* (B, PRIV) of the last encoder forward, for the decoder */
static int g_v4_privileged_rows = 0;
static IntTensor* g_v4_overflow = nullptr;        /* (SETS) dropped rows of the last training forward */

static bf16* bfp(IntTensor& t) { return (bf16*)t.data; }
static int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::atoi(v) : fallback;
}

static PrecisionTensor enc_forward(void* weights, void* activations, PrecisionTensor input, cudaStream_t stream) {
    auto* w = (EncWeights*)weights; auto* a = (EncActs*)activations;
    int B = a->B, H = w->hidden, D = w->D, I = w->I, FD = w->flat_dim, LDJ = FD + SETS * 2 * D;
    a->saved_input = input.data;
    /* bf16 weights */
    for (int s = 0; s < SETS; s++) {
        to_bf16_padded<<<grid_size(I * SPEC[s].kpad), BLOCK_SIZE, 0, stream>>>(w->w1[s].data, bfp(a->w1b[s]), I, SPEC[s].features, SPEC[s].kpad);
        to_bf16<<<grid_size(D * I), BLOCK_SIZE, 0, stream>>>(w->w2[s].data, bfp(a->w2b[s]), D * I);
    }
    to_bf16<<<grid_size(IDS * I), BLOCK_SIZE, 0, stream>>>(w->embedding.data, bfp(a->embb), IDS * I);
    /* flat block and privileged block */
    pack_flat<<<grid_size(B * (FLAT + PRIV)), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->flat_input.data, a->priv.data);
    g_v4_privileged = a->priv.data; g_v4_privileged_rows = B;
    prof_mark("enc.pack_flat", stream);
    puf_mm(&a->flat_input, &w->flat, &a->flat_pre, stream);
    relu_into<<<grid_size(B * FD), BLOCK_SIZE, 0, stream>>>(a->flat_pre.data, a->joined.data, B, FD, LDJ);
    prof_mark("enc.flat_gemm", stream);
    /* sets */
    count_rows<<<grid_size(B * SETS), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->counts.data);
    scan_rows<<<SETS, 1024, 0, stream>>>(a->counts.data, B, make_int4(a->cap[0], a->cap[1], a->cap[2], a->cap[3]), a->offsets.data, a->totals.data, a->overflow.data);
    scatter_rows<0><<<grid_size(B * SPEC[0].rows), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[0], bfp(a->xpack[0]), a->ids[0].data, a->sample[0].data);
    scatter_rows<1><<<grid_size(B * SPEC[1].rows), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[1], bfp(a->xpack[1]), a->ids[1].data, a->sample[1].data);
    scatter_rows<2><<<grid_size(B * SPEC[2].rows), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[2], bfp(a->xpack[2]), a->ids[2].data, a->sample[2].data);
    scatter_rows<3><<<grid_size(B * SPEC[3].rows), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[3], bfp(a->xpack[3]), a->ids[3].data, a->sample[3].data);
    prof_mark("enc.count_scan_scatter", stream);
    for (int s = 0; s < SETS; s++) {
        const int* total = a->totals.data + s;
        int K = SPEC[s].kpad;
        /* h1 = ReLU(X W1^T + E[id]): A = X (cap, K), B(k, n) = W1[n][k] (B_COL, ldb K) */
        gemm<false, true>(bfp(a->xpack[s]), K, bfp(a->w1b[s]), K, a->cap[s], total, I, 0, K, nullptr,
                          EpiRelu{bfp(a->h1[s]), I, SPEC[s].id_col >= 0 ? bfp(a->embb) : nullptr, a->ids[s].data, I}, stream);
        /* h2 = ReLU(h1 W2^T) */
        gemm<false, true>(bfp(a->h1[s]), I, bfp(a->w2b[s]), I, a->cap[s], total, D, 0, I, nullptr,
                          EpiRelu{bfp(a->h2[s]), D, nullptr, nullptr, 0}, stream);
        if (g_set_stage.forward)
            g_set_stage.forward(s, bfp(a->h2[s]), a->sample[s].data, a->offsets.data + (long)s * B, a->counts.data + (long)s * B, total, B, D, stream);
        pool_rows<<<B, D, 0, stream>>>(bfp(a->h2[s]), D, a->offsets.data + (long)s * B, a->counts.data + (long)s * B, a->cap[s], s,
                                       a->totals.data, a->joined.data, LDJ, FD + s * 2 * D, a->winner[s].data, w->mean_pool);
        prof_mark("enc.set_mlp_pool", stream);
    }
    puf_mm(&a->joined, &w->fusion, &a->fused, stream);
    ln_forward<<<B, 256, 0, stream>>>(a->fused.data, a->out.data, a->ln_mean.data, a->ln_rstd.data, H);
    prof_mark("enc.fusion_ln", stream);
    return a->out;
}

static void enc_backward(void* weights, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    auto* w = (EncWeights*)weights; auto* a = (EncActs*)activations;
    int B = a->B, H = w->hidden, D = w->D, I = w->I, FD = w->flat_dim, LDJ = FD + SETS * 2 * D;
    ln_backward<<<B, 256, 0, stream>>>(grad.data, a->fused.data, a->ln_mean.data, a->ln_rstd.data, a->grad_fused.data, H, 0);
    puf_mm_tn(&a->grad_fused, &a->joined, &a->wgrad_fusion, stream);
    puf_mm_nn(&a->grad_fused, &w->fusion, &a->grad_joined, stream);
    relu_grad_from<<<grid_size(B * FD), BLOCK_SIZE, 0, stream>>>(a->grad_joined.data, LDJ, a->joined.data, LDJ, a->grad_flat.data, B, FD);
    puf_mm_tn(&a->grad_flat, &a->flat_input, &a->wgrad_flat, stream);
    prof_mark("bwd.enc.fusion_flat", stream);
    for (int s = SETS - 1; s >= 0; s--) {
        const int* total = a->totals.data + s;
        int K = SPEC[s].kpad, cap = a->cap[s];
        unpool_rows<<<cap, D, 0, stream>>>(bfp(a->h2[s]), bfp(a->dh2[s]), D, a->offsets.data + (long)s * B, a->counts.data + (long)s * B,
                                           a->sample[s].data, a->totals.data, s, a->grad_joined.data, LDJ, FD + s * 2 * D,
                                           a->winner[s].data, w->mean_pool);
        if (g_set_stage.backward)
            g_set_stage.backward(s, bfp(a->h2[s]), bfp(a->dh2[s]), a->sample[s].data, a->offsets.data + (long)s * B, a->counts.data + (long)s * B, total, B, D, stream);
        prof_mark("bwd.enc.unpool", stream);
        /* dW2 = dh2^T h1 */
        weight_gradient(bfp(a->dh2[s]), D, bfp(a->h1[s]), I, cap, total, a->partials.data, a->wgrad_w2[s].data, I, 0, stream);
        prof_mark("bwd.enc.dW2", stream);
        /* dh1 = (dh2 W2) * ReLU'(h1), written over h2 (no longer needed) */
        bf16* dh1 = bfp(a->h2[s]);
        gemm<false, false>(bfp(a->dh2[s]), D, bfp(a->w2b[s]), I, cap, total, I, 0, D, nullptr, EpiMask{dh1, I, bfp(a->h1[s])}, stream);
        prof_mark("bwd.enc.dh1", stream);
        /* dW1 = dh1^T X (keep the first `features` of the padded K) */
        weight_gradient(dh1, I, bfp(a->xpack[s]), K, cap, total, a->partials.data, a->wgrad_w1[s].data, SPEC[s].features, 0, stream);
        prof_mark("bwd.enc.dW1", stream);
        if (SPEC[s].id_col >= 0) {
            int chunks = (cap + CHUNK - 1) / CHUNK;
            embedding_partials<<<chunks, I, IDS * I * sizeof(float), stream>>>(dh1, I, a->ids[s].data, a->totals.data, s, cap, a->partials.data);
            reduce_partials<<<grid_size(IDS * I), BLOCK_SIZE, 0, stream>>>(a->partials.data, chunks, (long)IDS * I, IDS, I, I,
                                                                          a->wgrad_embedding.data, s != 2 /* entity set first (s=2), slot adds */);
            prof_mark("bwd.enc.embedding", stream);
        }
    }
}

static void enc_init(void* weights, ulong* seed, cudaStream_t stream) {
    auto* w = (EncWeights*)weights;
    for (int s = 0; s < SETS; s++) { puf_kaiming_init(&w->w1[s], std::sqrt(2.0f), (*seed)++, stream); puf_kaiming_init(&w->w2[s], std::sqrt(2.0f), (*seed)++, stream); }
    puf_kaiming_init(&w->embedding, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&w->flat, std::sqrt(2.0f), (*seed)++, stream);
    puf_kaiming_init(&w->fusion, 1.0f, (*seed)++, stream);
}
/* save_weights dumps the allocator's total_elems floats back to back while the
 * device offsets are 16-byte aligned: every tensor must be a multiple of 4
 * floats for the file and the device to agree (research/layout.py aligned). */
static void reg_aligned(Allocator* alloc, PrecisionTensor* t) {
    if (numel(t->shape) % 4) throw std::runtime_error("Bopl v4: a weight tensor of " + std::to_string(numel(t->shape)) + " floats is not 16-byte aligned");
    alloc_register(alloc, t);
}
static void enc_reg_params(void* weights, Allocator* alloc) {
    auto* w = (EncWeights*)weights; int H = w->hidden, D = w->D, I = w->I;
    for (int s = 0; s < SETS; s++) {
        w->w1[s] = {.shape = {I, SPEC[s].features}}; reg_aligned(alloc, &w->w1[s]);
        w->w2[s] = {.shape = {D, I}}; reg_aligned(alloc, &w->w2[s]);
    }
    w->embedding = {.shape = {IDS, I}}; reg_aligned(alloc, &w->embedding);
    w->flat = {.shape = {w->flat_dim, FLAT}}; reg_aligned(alloc, &w->flat);
    w->fusion = {.shape = {H, w->flat_dim + SETS * 2 * D}}; reg_aligned(alloc, &w->fusion);
}
static void enc_reg_forward(EncWeights* w, EncActs* a, Allocator* alloc, int B) {
    int H = w->hidden, D = w->D, I = w->I, FD = w->flat_dim;
    a->B = B; a->gather = Gather{};
    a->flat_input = {.shape = {B, FLAT}}; a->flat_pre = {.shape = {B, FD}};
    a->joined = {.shape = {B, FD + SETS * 2 * D}}; a->fused = {.shape = {B, H}}; a->out = {.shape = {B, H}};
    a->priv = {.shape = {B, PRIV}};
    for (auto* t : {&a->flat_input, &a->flat_pre, &a->joined, &a->fused, &a->out, &a->priv}) alloc_register(alloc, t);
    a->ln_mean = {.shape = {B}}; a->ln_rstd = {.shape = {B}};
    alloc_register(alloc, &a->ln_mean); alloc_register(alloc, &a->ln_rstd);
    a->counts = {.shape = {SETS, B}}; a->offsets = {.shape = {SETS, B}}; a->totals = {.shape = {SETS}}; a->overflow = {.shape = {SETS}};
    for (auto* t : {&a->counts, &a->offsets, &a->totals, &a->overflow}) alloc_register(alloc, t);
    for (int s = 0; s < SETS; s++) {
        int cap = (B * w->budget[s] + TILE - 1) / TILE * TILE;
        if (cap > B * SPEC[s].rows) cap = (B * SPEC[s].rows + TILE - 1) / TILE * TILE;
        a->cap[s] = cap;
        a->ids[s] = {.shape = {cap}}; a->sample[s] = {.shape = {cap}}; a->winner[s] = {.shape = {B, D}};
        a->w1b[s] = {.shape = {I, SPEC[s].kpad / 2}}; a->w2b[s] = {.shape = {D, I / 2}};
        a->xpack[s] = {.shape = {cap, SPEC[s].kpad / 2}}; a->h1[s] = {.shape = {cap, I / 2}}; a->h2[s] = {.shape = {cap, D / 2}};
        for (auto* t : {&a->ids[s], &a->sample[s], &a->winner[s], &a->w1b[s], &a->w2b[s], &a->xpack[s], &a->h1[s], &a->h2[s]}) alloc_register(alloc, t);
        if (g_set_stage.reg) g_set_stage.reg(alloc, s, cap, D);
    }
    a->embb = {.shape = {IDS, I / 2}}; alloc_register(alloc, &a->embb);
}
/* BOPL_V4_MEMLOG=1: print what each registration adds to the activation allocator. */
static void memlog(const char* what, int B, const Allocator* alloc, long before) {
    static int on = -1;
    if (on < 0) { const char* f = getenv("BOPL_V4_MEMLOG"); on = f && atoi(f) ? 1 : 0; }
    if (on) fprintf(stderr, "bopl v4 memory: %s B=%d %.1f MB (%.1f KB/row)\n", what, B, (alloc->total_elems - before) * 4.0 / 1048576, (alloc->total_elems - before) * 4.0 / 1024 / B);
}
static void enc_reg_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    long before = alloc->total_elems;
    enc_reg_forward((EncWeights*)weights, (EncActs*)activations, alloc, B);
    memlog("encoder rollout activations", B, alloc, before);
}
static void enc_reg_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    auto* w = (EncWeights*)weights; auto* a = (EncActs*)activations; int H = w->hidden, D = w->D, I = w->I, FD = w->flat_dim;
    long before = acts->total_elems;
    enc_reg_forward(w, a, acts, B);
    a->grad_fused = {.shape = {B, H}}; a->grad_joined = {.shape = {B, FD + SETS * 2 * D}}; a->grad_flat = {.shape = {B, FD}};
    for (auto* t : {&a->grad_fused, &a->grad_joined, &a->grad_flat}) alloc_register(acts, t);
    long partial = 0;
    for (int s = 0; s < SETS; s++) {
        a->dh2[s] = {.shape = {a->cap[s], D / 2}}; alloc_register(acts, &a->dh2[s]);
        long chunks = (a->cap[s] + CHUNK - 1) / CHUNK;
        long need = chunks * std::max((long)D * I, std::max((long)I * SPEC[s].kpad, (long)IDS * I));
        partial = std::max(partial, need);
    }
    a->partials = {.shape = {partial}}; alloc_register(acts, &a->partials);
    /* gradients in reg_params order */
    for (int s = 0; s < SETS; s++) {
        a->wgrad_w1[s] = {.shape = {I, SPEC[s].features}}; alloc_register(grads, &a->wgrad_w1[s]);
        a->wgrad_w2[s] = {.shape = {D, I}}; alloc_register(grads, &a->wgrad_w2[s]);
    }
    a->wgrad_embedding = {.shape = {IDS, I}}; alloc_register(grads, &a->wgrad_embedding);
    a->wgrad_flat = {.shape = {FD, FLAT}}; alloc_register(grads, &a->wgrad_flat);
    a->wgrad_fusion = {.shape = {H, FD + SETS * 2 * D}}; alloc_register(grads, &a->wgrad_fusion);
    g_v4_overflow = &a->overflow;
    memlog("encoder train activations", B, acts, before);
}
static void* enc_create(void* self) {
    auto* encoder = (Encoder*)self;
    if (encoder->in_dim != STRIDE)
        throw std::runtime_error("Bopl v4 expects " + std::to_string(STRIDE) + " observation floats per row, got " + std::to_string(encoder->in_dim));
    auto* w = (EncWeights*)calloc(1, sizeof(EncWeights));
    w->hidden = encoder->out_dim;
    w->D = env_int("BOPL4_SET_DIM", BOPL4_SET_DIM);
    w->I = env_int("BOPL4_SET_INNER", BOPL4_SET_INNER);
    w->flat_dim = env_int("BOPL4_FLAT_DIM", BOPL4_FLAT_DIM);
    w->mean_pool = env_int("BOPL4_MEAN_POOL", 1);
    if (w->D % 64 || w->I % 64 || w->D > 1024 || w->I > 1024) throw std::runtime_error("Bopl v4: set_dim and set_inner must be multiples of 64");
    for (int s = 0; s < SETS; s++) w->budget[s] = DEFAULT_BUDGET[s];
    if (const char* b = std::getenv("BOPL4_ROW_BUDGET")) {
        int p, sl, e, l;
        if (sscanf(b, "%d,%d,%d,%d", &p, &sl, &e, &l) == 4) { w->budget[0] = p; w->budget[1] = sl; w->budget[2] = e; w->budget[3] = l; }
    }
    return w;
}
static void enc_free(void* v) { free(v); }

static void enc_bind_gather(void* activations, const precision_t* rollout_obs, const int* idx, int T, int B_total) {
    ((EncActs*)activations)->gather = Gather{rollout_obs, idx, T, B_total};
}

/* ---- decoder module ------------------------------------------------------ */
struct DecWeights {
    DecoderWeights base;       /* the trainer casts decoder weights to DecoderWeights to read .continuous */
    PrecisionTensor ffn_in, ffn_out, actor, critic_in, critic_b1, critic_b2, critic_out;
    int hidden, od, ffn, cdim, privileged, detach;
};
struct DecActs {
    DecoderActivations base;   /* out, grad_out, saved_input, grad_input, ..., od1_pad: read by the trainer */
    int B, od1;
    PrecisionTensor actor_pad, wgrad_actor_pad;
    PrecisionTensor ln1, f, y, zin, lnin, z, ln2, bblk, z2, ln3, value;
    FloatTensor m1, r1, min_, rin, m2, r2, m3, r3;
    /* training */
    PrecisionTensor g_y, g_f, g_ln1, gv, g_ln3, g_z2, g_b, g_ln2, g_z, g_lnin, g_zin;
    PrecisionTensor wgrad_ffn_in, wgrad_ffn_out, wgrad_actor, wgrad_critic_in, wgrad_b1, wgrad_b2, wgrad_critic_out;
};
__global__ void pad_rows(float* dst, const float* src, int rows, int rows_pad, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_pad * cols) return;
    dst[i] = i < rows * cols ? src[i] : 0.0f;
}
__global__ void write_column(float* out, int ld, int col, const float* v, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) out[(long)i * ld + col] = v[i];
}
__global__ void read_column(const float* out, int ld, int col, float* v, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) v[i] = out[(long)i * ld + col];
}
/* zin = [y | priv] */
__global__ void concat_priv(const float* y, int H, const float* priv, int P, float* zin, int rows) {
    int W = H + P;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * W) return;
    int r = i / W, c = i - r * W;
    zin[i] = c < H ? y[(long)r * H + c] : (priv ? priv[(long)r * P + c - H] : 0.0f);
}
static PrecisionTensor dec_forward(void* weights, void* activations, PrecisionTensor input, cudaStream_t stream) {
    auto* w = (DecWeights*)weights; auto* a = (DecActs*)activations;
    int B = a->B, H = w->hidden, od1 = a->od1, pad = a->base.od1_pad > 0 ? a->base.od1_pad : od1;
    if (a->base.saved_input.data) puf_copy(&a->base.saved_input, &input, stream);
    prof_mark("dec.start", stream);
    PrecisionTensor h = input;
    PrecisionTensor vout = {.data = w->critic_out.data, .shape = {1, w->cdim}};
    /* y = h + FFN(LN(h)) */
    PrecisionTensor* y = &a->y;
    if (w->ffn) {
        ln_forward<<<B, 256, 0, stream>>>(h.data, a->ln1.data, a->m1.data, a->r1.data, H);
        puf_mm(&a->ln1, &w->ffn_in, &a->f, stream);
        relu_into<<<grid_size(B * w->ffn), BLOCK_SIZE, 0, stream>>>(a->f.data, a->f.data, B, w->ffn, w->ffn);
        puf_mm(&a->f, &w->ffn_out, &a->y, stream);
        add_into<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->y.data, h.data, B * H);
    } else {
        puf_copy(&a->y, &h, stream);
    }
    /* logits: actor rows then zero rows to the padded width */
    pad_rows<<<grid_size(pad * H), BLOCK_SIZE, 0, stream>>>(a->actor_pad.data, w->actor.data, w->od, pad, H);
    puf_mm(y, &a->actor_pad, &a->base.out, stream);
    /* value */
    if (w->cdim) {
        const float* priv = w->privileged ? g_v4_privileged : nullptr;
        if (w->privileged && g_v4_privileged_rows != B) throw std::runtime_error("Bopl v4 decoder: privileged rows do not match the batch");
        int P = w->privileged ? PRIV : 0, W = H + P;
        concat_priv<<<grid_size(B * W), BLOCK_SIZE, 0, stream>>>(y->data, H, priv, P, a->zin.data, B);
        ln_forward<<<B, 256, 0, stream>>>(a->zin.data, a->lnin.data, a->min_.data, a->rin.data, W);
        puf_mm(&a->lnin, &w->critic_in, &a->z, stream);
        ln_forward<<<B, 256, 0, stream>>>(a->z.data, a->ln2.data, a->m2.data, a->r2.data, w->cdim);
        puf_mm(&a->ln2, &w->critic_b1, &a->bblk, stream);
        relu_into<<<grid_size(B * w->cdim), BLOCK_SIZE, 0, stream>>>(a->bblk.data, a->bblk.data, B, w->cdim, w->cdim);
        puf_mm(&a->bblk, &w->critic_b2, &a->z2, stream);
        add_into<<<grid_size(B * w->cdim), BLOCK_SIZE, 0, stream>>>(a->z2.data, a->z.data, B * w->cdim);
        ln_forward<<<B, 256, 0, stream>>>(a->z2.data, a->ln3.data, a->m3.data, a->r3.data, w->cdim);
        puf_mm(&a->ln3, &vout, &a->value, stream);
        write_column<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(a->base.out.data, pad, w->od, a->value.data, B);
    }
    prof_mark("dec.done", stream);
    return a->base.out;
}
static PrecisionTensor dec_backward(void* weights, void* activations, FloatTensor grad_logits, FloatTensor grad_logstd,
                                    FloatTensor grad_value, cudaStream_t stream) {
    auto* w = (DecWeights*)weights; auto* a = (DecActs*)activations;
    int B = a->B, H = w->hidden, od = w->od, pad = a->base.od1_pad;
    if (!g_bopl_decoder_grad_fused)
        assemble_decoder_grad<<<grid_size(B * pad), BLOCK_SIZE, 0, stream>>>(a->base.grad_out.data, grad_logits.data, grad_value.data, B, od, pad);
    prof_mark("bwd.dec.start", stream);
    /* actor */
    puf_mm_tn(&a->base.grad_out, &a->y, &a->wgrad_actor_pad, stream);
    cudaMemcpyAsync(a->wgrad_actor.data, a->wgrad_actor_pad.data, (size_t)od * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    puf_mm_nn(&a->base.grad_out, &a->actor_pad, &a->g_y, stream);   /* rows >= od of actor_pad are zero */
    /* critic */
    if (w->cdim) {
        int C = w->cdim, P = w->privileged ? PRIV : 0, W = H + P;
        read_column<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(a->base.grad_out.data, pad, od, a->gv.data, B);
        puf_mm_tn(&a->gv, &a->ln3, &a->wgrad_critic_out, stream);
        PrecisionTensor vout = {.data = w->critic_out.data, .shape = {1, C}};
        puf_mm_nn(&a->gv, &vout, &a->g_ln3, stream);
        ln_backward<<<B, 256, 0, stream>>>(a->g_ln3.data, a->z2.data, a->m3.data, a->r3.data, a->g_z2.data, C, 0);
        puf_mm_tn(&a->g_z2, &a->bblk, &a->wgrad_b2, stream);
        puf_mm_nn(&a->g_z2, &w->critic_b2, &a->g_b, stream);
        relu_grad_from<<<grid_size(B * C), BLOCK_SIZE, 0, stream>>>(a->g_b.data, C, a->bblk.data, C, a->g_b.data, B, C);
        puf_mm_tn(&a->g_b, &a->ln2, &a->wgrad_b1, stream);
        puf_mm_nn(&a->g_b, &w->critic_b1, &a->g_ln2, stream);
        puf_copy(&a->g_z, &a->g_z2, stream);
        ln_backward<<<B, 256, 0, stream>>>(a->g_ln2.data, a->z.data, a->m2.data, a->r2.data, a->g_z.data, C, 1);
        puf_mm_tn(&a->g_z, &a->lnin, &a->wgrad_critic_in, stream);
        puf_mm_nn(&a->g_z, &w->critic_in, &a->g_lnin, stream);
        ln_backward<<<B, 256, 0, stream>>>(a->g_lnin.data, a->zin.data, a->min_.data, a->rin.data, a->g_zin.data, W, 0);
        if (!w->detach) {
            /* g_y += g_zin[:, :H] */
            copy_cols<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->g_zin.data, W, 0, a->g_ln1.data, H, B, H);
            add_into<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->g_y.data, a->g_ln1.data, B * H);
        }
    }
    /* FFN: grad_h = g_y + LN_bwd(W_i^T (ReLU' * W_o^T g_y)) */
    puf_copy(&a->base.grad_input, &a->g_y, stream);
    if (w->ffn) {
        puf_mm_tn(&a->g_y, &a->f, &a->wgrad_ffn_out, stream);
        puf_mm_nn(&a->g_y, &w->ffn_out, &a->g_f, stream);
        relu_grad_from<<<grid_size(B * w->ffn), BLOCK_SIZE, 0, stream>>>(a->g_f.data, w->ffn, a->f.data, w->ffn, a->g_f.data, B, w->ffn);
        puf_mm_tn(&a->g_f, &a->ln1, &a->wgrad_ffn_in, stream);
        puf_mm_nn(&a->g_f, &w->ffn_in, &a->g_ln1, stream);
        ln_backward<<<B, 256, 0, stream>>>(a->g_ln1.data, a->base.saved_input.data, a->m1.data, a->r1.data, a->base.grad_input.data, H, 1);
    }
    (void)grad_logstd;
    prof_mark("bwd.dec.done", stream);
    return a->base.grad_input;
}
static void dec_init(void* weights, ulong* seed, cudaStream_t stream) {
    auto* w = (DecWeights*)weights;
    if (w->ffn) { puf_kaiming_init(&w->ffn_in, std::sqrt(2.0f), (*seed)++, stream); puf_kaiming_init(&w->ffn_out, 0.1f, (*seed)++, stream); }
    puf_kaiming_init(&w->actor, 1.0f, (*seed)++, stream);
    if (w->cdim) {
        puf_kaiming_init(&w->critic_in, std::sqrt(2.0f), (*seed)++, stream);
        puf_kaiming_init(&w->critic_b1, std::sqrt(2.0f), (*seed)++, stream);
        puf_kaiming_init(&w->critic_b2, 0.1f, (*seed)++, stream);
        PrecisionTensor v = {.data = w->critic_out.data, .shape = {1, w->cdim}};
        puf_kaiming_init(&v, 1.0f, (*seed)++, stream);
    }
}
static void dec_reg_params(void* weights, Allocator* alloc) {
    auto* w = (DecWeights*)weights; int H = w->hidden;
    if (w->ffn) {
        w->ffn_in = {.shape = {w->ffn, H}}; reg_aligned(alloc, &w->ffn_in);
        w->ffn_out = {.shape = {H, w->ffn}}; reg_aligned(alloc, &w->ffn_out);
    }
    w->actor = {.shape = {w->od, H}}; reg_aligned(alloc, &w->actor);
    if (w->cdim) {
        w->critic_in = {.shape = {w->cdim, H + (w->privileged ? PRIV : 0)}}; reg_aligned(alloc, &w->critic_in);
        w->critic_b1 = {.shape = {w->cdim, w->cdim}}; reg_aligned(alloc, &w->critic_b1);
        w->critic_b2 = {.shape = {w->cdim, w->cdim}}; reg_aligned(alloc, &w->critic_b2);
        /* 1-D: Muon leaves vectors to momentum SGD; the CPU loader reads it as (1, cdim). */
        w->critic_out = {.shape = {w->cdim}}; reg_aligned(alloc, &w->critic_out);
    }
}
static void dec_reg_forward(DecWeights* w, DecActs* a, Allocator* alloc, int B, int pad) {
    int H = w->hidden, C = w->cdim, W = H + (w->privileged ? PRIV : 0);
    a->B = B; a->od1 = w->od + 1;
    a->base.out = {.shape = {B, pad}}; alloc_register(alloc, &a->base.out);
    a->actor_pad = {.shape = {pad, H}}; alloc_register(alloc, &a->actor_pad);
    a->y = {.shape = {B, H}}; alloc_register(alloc, &a->y);
    if (w->ffn) {
        a->ln1 = {.shape = {B, H}}; a->f = {.shape = {B, w->ffn}};
        alloc_register(alloc, &a->ln1); alloc_register(alloc, &a->f);
        a->m1 = {.shape = {B}}; a->r1 = {.shape = {B}}; alloc_register(alloc, &a->m1); alloc_register(alloc, &a->r1);
    }
    if (C) {
        a->zin = {.shape = {B, W}}; a->lnin = {.shape = {B, W}}; a->z = {.shape = {B, C}}; a->ln2 = {.shape = {B, C}};
        a->bblk = {.shape = {B, C}}; a->z2 = {.shape = {B, C}}; a->ln3 = {.shape = {B, C}}; a->value = {.shape = {B, 1}};
        for (auto* t : {&a->zin, &a->lnin, &a->z, &a->ln2, &a->bblk, &a->z2, &a->ln3, &a->value}) alloc_register(alloc, t);
        for (auto* t : {&a->min_, &a->rin, &a->m2, &a->r2, &a->m3, &a->r3}) { *t = {.shape = {B}}; alloc_register(alloc, t); }
    }
}
static void dec_reg_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    auto* w = (DecWeights*)weights; auto* a = (DecActs*)activations;
    a->base.od1_pad = 0;
    dec_reg_forward(w, a, alloc, B, w->od + 1);
}
static void dec_reg_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    auto* w = (DecWeights*)weights; auto* a = (DecActs*)activations; int H = w->hidden, C = w->cdim, W = H + (w->privileged ? PRIV : 0);
    int pad = decoder_padded_width(w->od + 1);
    a->base.od1_pad = pad;
    long before = acts->total_elems;
    dec_reg_forward(w, a, acts, B, pad);
    a->base.grad_out = {.shape = {B, pad}}; a->base.saved_input = {.shape = {B, H}}; a->base.grad_input = {.shape = {B, H}};
    a->wgrad_actor_pad = {.shape = {pad, H}}; a->g_y = {.shape = {B, H}}; a->g_ln1 = {.shape = {B, H}};
    for (auto* t : {&a->base.grad_out, &a->base.saved_input, &a->base.grad_input, &a->wgrad_actor_pad, &a->g_y, &a->g_ln1}) alloc_register(acts, t);
    if (w->ffn) { a->g_f = {.shape = {B, w->ffn}}; alloc_register(acts, &a->g_f); }
    if (C) {
        a->gv = {.shape = {B, 1}}; a->g_ln3 = {.shape = {B, C}}; a->g_z2 = {.shape = {B, C}}; a->g_b = {.shape = {B, C}};
        a->g_ln2 = {.shape = {B, C}}; a->g_z = {.shape = {B, C}}; a->g_lnin = {.shape = {B, W}}; a->g_zin = {.shape = {B, W}};
        for (auto* t : {&a->gv, &a->g_ln3, &a->g_z2, &a->g_b, &a->g_ln2, &a->g_z, &a->g_lnin, &a->g_zin}) alloc_register(acts, t);
    }
    /* gradients in reg_params order */
    if (w->ffn) {
        a->wgrad_ffn_in = {.shape = {w->ffn, H}}; alloc_register(grads, &a->wgrad_ffn_in);
        a->wgrad_ffn_out = {.shape = {H, w->ffn}}; alloc_register(grads, &a->wgrad_ffn_out);
    }
    a->wgrad_actor = {.shape = {w->od, H}}; alloc_register(grads, &a->wgrad_actor);
    if (C) {
        a->wgrad_critic_in = {.shape = {C, W}}; alloc_register(grads, &a->wgrad_critic_in);
        a->wgrad_b1 = {.shape = {C, C}}; alloc_register(grads, &a->wgrad_b1);
        a->wgrad_b2 = {.shape = {C, C}}; alloc_register(grads, &a->wgrad_b2);
        a->wgrad_critic_out = {.shape = {1, C}}; alloc_register(grads, &a->wgrad_critic_out);
    }
    memlog("decoder train activations", B, acts, before);
}
static void* dec_create(void* self) {
    auto* d = (Decoder*)self;
    auto* w = (DecWeights*)calloc(1, sizeof(DecWeights));
    w->base.hidden_dim = d->hidden_dim; w->base.output_dim = d->output_dim; w->base.continuous = false;
    w->hidden = d->hidden_dim; w->od = d->output_dim;
    w->ffn = env_int("BOPL4_FFN_DIM", BOPL4_FFN_DIM);
    w->cdim = env_int("BOPL4_CRITIC_DIM", BOPL4_CRITIC_DIM);
    w->privileged = env_int("BOPL4_CRITIC_PRIVILEGED", 1);
    w->detach = env_int("BOPL4_CRITIC_DETACH", 0);
    if (d->continuous) throw std::runtime_error("Bopl v4 decoder is discrete");
    return w;
}
static void dec_free(void* v) { free(v); }

} // namespace v4

static void bopl_create_v4_encoder(Encoder* encoder) {
    *encoder = Encoder{
        .forward = v4::enc_forward, .backward = v4::enc_backward, .init_weights = v4::enc_init, .reg_params = v4::enc_reg_params,
        .reg_train = v4::enc_reg_train, .reg_rollout = v4::enc_reg_rollout, .create_weights = v4::enc_create,
        .free_weights = v4::enc_free, .free_activations = v4::enc_free, .in_dim = encoder->in_dim, .out_dim = encoder->out_dim,
        .activation_size = sizeof(v4::EncActs),
    };
}
static void bopl_create_v4_decoder(Decoder* decoder) {
    *decoder = Decoder{
        .forward = v4::dec_forward, .backward = v4::dec_backward, .init_weights = v4::dec_init, .reg_params = v4::dec_reg_params,
        .reg_train = v4::dec_reg_train, .reg_rollout = v4::dec_reg_rollout, .create_weights = v4::dec_create,
        .free_weights = v4::dec_free, .free_activations = v4::dec_free,
        .hidden_dim = decoder->hidden_dim, .output_dim = decoder->output_dim, .continuous = decoder->continuous,
        .activation_size = (int)sizeof(v4::DecActs),
    };
}
static bool bopl_is_v4_encoder(const Encoder& e) { return e.forward == v4::enc_forward; }
static void bopl_v4_bind_gather(void* activations, const precision_t* rollout_obs, const int* idx, int T, int B_total) {
    v4::enc_bind_gather(activations, rollout_obs, idx, T, B_total);
}
/* Dropped rows of the last training forward (per set), for the trainer's log. */
int bopl_v4_overflow(int* out) {
    if (!v4::g_v4_overflow || !v4::g_v4_overflow->data) return -1;
    cudaMemcpy(out, v4::g_v4_overflow->data, v4::SETS * sizeof(int), cudaMemcpyDeviceToHost);
    return v4::SETS;
}
#endif
