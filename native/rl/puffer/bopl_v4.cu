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
 *            device so the launch shape is fixed (CUDA-graph capture); the
 *            GEMMs run over the fixed capacity with the tail kept at zero
 *   h1     = ReLU(W1 x + E[id])   h2 = ReLU(W2 h1)               cuBLAS bf16
 *   [attention hook: bopl_v4_set_stage, below]
 *   pool   = [max_rows h2, mean_rows h2] per set (empty set -> 0)
 *   out    = LN(W_fusion [flat, pool x 4])                      cuBLAS
 * Decoder (hidden -> [logits..., value] row, stride od1_pad as upstream):
 *   y      = h + W_o ReLU(W_i LN(h))
 *   logits = W_actor y
 *   z      = W_cin LN([y (detached if critic_detach), privileged])
 *   value  = W_v LN(z + W_b2 ReLU(W_b1 LN(z)))          W_cin, W_b1, W_b2: cuBLAS bf16
 * The privileged block comes from the observation rows the encoder gathered
 * (the decoder only receives h): the encoder packs it into a (B, 48) buffer
 * the decoder reads (g_v4_privileged), in the same policy_forward call.
 *
 * Row budget: each set's packed capacity is B x BOPL4_BUDGET_<set> rows
 * (BOPL4_ROW_BUDGET="p,s,e,l" overrides); rows past the capacity are dropped
 * in sample order and counted (bopl_v4_overflow), never silently.
 * Determinism: the embedding gradient is a fixed-order chunked partial sum;
 * the GEMMs are cuBLAS with a fixed shape and kernel choice.
 */
#ifndef BOPL_V4_CU
#define BOPL_V4_CU
#include <cuda_bf16.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include "bopl_layout_v4.h"

namespace v4 {
using bf16 = __nv_bfloat16;

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
constexpr int TILE = 64;                               /* packed capacities are rounded up to this many rows */
constexpr int CHUNK = 512;                             /* packed rows per embedding-gradient partial */
constexpr float LN_EPS = 1e-5f;

/* Stage timing for the harness (check_v4.cu): when enabled, an event per
 * mark() on the stream; never under graph capture. */
static int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::atoi(v) : fallback;
}

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
/* Scatter set S's valid rows into the packed bf16 matrix (KPAD columns, zero
 * padded), with their embedding id and sample index: one warp per sample, the
 * valid mask by ballot, rows copied by the warp in row order (the thread-per-
 * row form recomputed each row's rank with an O(rows) strided walk). */
template<int S>
__global__ void scatter_rows(const float* input, Gather g, int B, const int* offsets, int cap,
                             bf16* packed, int* ids, int* sample) {
    const Spec sp = c_spec[S];
    int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (b >= B) return;
    const float* row = row_ptr(input, g, b) + sp.offset;
    int dst = offsets[(long)S * B + b];
    for (int r0 = 0; r0 < sp.rows; r0 += 32) {
        int r = r0 + lane;
        unsigned valid = __ballot_sync(0xffffffffu, r < sp.rows && row[r * sp.width] != 0.0f);
        for (; valid && dst < cap; valid &= valid - 1, dst++) {
            const float* src = row + (r0 + __ffs(valid) - 1) * sp.width;
            bf16* out = packed + (long)dst * sp.kpad;
            for (int k = lane; k < sp.kpad; k += 32) out[k] = tobf(k < sp.features ? src[k] : 0.0f);
            if (lane == 0) { ids[dst] = sp.id_col >= 0 ? embedding_id(src[sp.id_col]) : 0; sample[dst] = b; }
        }
    }
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

/* Fixed-order sum of chunk partials (chunks, n, k_full), keeping the first k_keep columns. */
__global__ void reduce_partials(const float* partials, int chunks, long per_chunk, int n, int k_full, int k_keep,
                                float* out, int accumulate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * k_keep) return;
    int r = i / k_keep, c = i - r * k_keep;
    float sum = 0.0f;
    for (int ch = 0; ch < chunks; ch++) sum += partials[(long)ch * per_chunk + (long)r * k_full + c];
    out[i] = accumulate ? out[i] + sum : sum;
}

/* ---- the set MLPs as cublasLt bf16 GEMMs over the fixed capacity -----------
 * Row-major D (M, N) = epilogue(op(A) op(B) + C) with A, B bf16, fp32
 * accumulation, D bf16 (activations, ReLU fused in the epilogue) or fp32
 * (weight gradients). M is the fixed capacity so the shape is graph-
 * capturable; rows past the count are kept at zero (zero_tail on the packed
 * input, embed_rows / unpool_rows on the addend and the gradients) so they
 * contribute nothing. a_t / b_t: the operand is stored transposed (K, M) /
 * (N, K). C, when given, is a bf16 (M, N) addend (the gathered embedding
 * rows, beta = 1). Plans (layouts + the heuristic's first algorithm, split-K
 * restricted to the deterministic workspace reductions) are cached per shape
 * under a mutex: the buffer threads' per-bank chains and the trainer run
 * concurrently. The workspace is allocated once at enc_create. */
struct LtKey {
    int a_t, b_t, M, N, K, lda, ldb, ldd, d_bf16, has_c, epi;
    const void* aux;   /* RELU_AUX / DRELU bitmask, part of the plan (a descriptor attribute) */
    bool operator==(const LtKey& o) const {
        return a_t == o.a_t && b_t == o.b_t && M == o.M && N == o.N && K == o.K && lda == o.lda && ldb == o.ldb && ldd == o.ldd
            && d_bf16 == o.d_bf16 && has_c == o.has_c && epi == o.epi && aux == o.aux;
    }
};
struct LtPlan { LtKey key; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc, ld; cublasLtMatmulAlgo_t algo; };
static std::vector<LtPlan> g_lt_plans;
static std::mutex g_lt_mutex;
static void* g_lt_ws = nullptr;
constexpr size_t LT_WS_BYTES = 32u << 20;
static void lt_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("Bopl v4 cublasLt ") + what + ": " + cublasLtGetStatusString(st));
}
static void lt_workspace() {
    if (g_lt_ws) return;
    if (cudaMalloc(&g_lt_ws, LT_WS_BYTES) != cudaSuccess) throw std::runtime_error("Bopl v4: cublasLt workspace allocation failed");
}
/* Candidates for the shape from the heuristic; the first one is used unless
 * the operands are given (eager call, not under capture) in which case the
 * candidates are timed on the stream and the fastest kept: the heuristic's
 * first choice for the long-K weight gradients (512 x 512 output over 12288
 * rows) ran several times slower than its alternatives. BOPL4_LT_TUNE=0
 * keeps the heuristic's choice. */
constexpr int LT_CANDIDATES = 8;
static const LtPlan& lt_plan(cublasLtHandle_t lt, const LtKey& k, const void* A, const void* B, const void* C, void* D, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_lt_mutex);
    for (const auto& p : g_lt_plans) if (p.key == k) return p;
    LtPlan p; p.key = k;
    /* column-major: D^T (N, M) = op(B)^T op(A)^T, so cublas's A is our B and its B our A */
    cublasOperation_t ta = k.b_t ? CUBLAS_OP_T : CUBLAS_OP_N, tb = k.a_t ? CUBLAS_OP_T : CUBLAS_OP_N;
    int m = k.N, n = k.M, kk = k.K;
    lt_check(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta), "transa");
    lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb), "transb");
    cublasLtEpilogue_t epi = (cublasLtEpilogue_t)k.epi;
    lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi), "epilogue");
    if (k.aux) {   /* one bit per element, N bits (a multiple of 128) per packed row */
        int64_t ld = k.N;
        lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &k.aux, sizeof k.aux), "aux pointer");
        lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld, sizeof ld), "aux ld");
    }
    lt_check(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, ta == CUBLAS_OP_N ? m : kk, ta == CUBLAS_OP_N ? kk : m, k.ldb), "layout A");
    lt_check(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, tb == CUBLAS_OP_N ? kk : n, tb == CUBLAS_OP_N ? n : kk, k.lda), "layout B");
    cudaDataType dt = k.d_bf16 ? CUDA_R_16BF : CUDA_R_32F;
    lt_check(cublasLtMatrixLayoutCreate(&p.lc, dt, m, n, k.ldd), "layout C");
    lt_check(cublasLtMatrixLayoutCreate(&p.ld, dt, m, n, k.ldd), "layout D");
    cublasLtMatmulPreference_t pref;
    lt_check(cublasLtMatmulPreferenceCreate(&pref), "preference");
    size_t ws = LT_WS_BYTES;
    lt_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws), "workspace");
    uint32_t red = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE | CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE;   /* no in-place atomics: deterministic */
    lt_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof red), "reduction");
    cublasLtMatmulHeuristicResult_t results[LT_CANDIDATES]; int found = 0;
    lt_check(cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.la, p.lb, p.lc, p.ld, pref, LT_CANDIDATES, results, &found), "heuristic");
    cublasLtMatmulPreferenceDestroy(pref);
    if (found < 1) throw std::runtime_error("Bopl v4 cublasLt: no algorithm for M=" + std::to_string(k.M) + " N=" + std::to_string(k.N) + " K=" + std::to_string(k.K));
    int best = 0;
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &capturing);
    static int tune = env_int("BOPL4_LT_TUNE", 1);
    if (tune && found > 1 && A && capturing == cudaStreamCaptureStatusNone) {
        float alpha = 1.0f, beta = C ? 1.0f : 0.0f;
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        float best_ms = 1e30f;
        for (int i = 0; i < found; i++) {
            float ms_min = 1e30f;
            for (int rep = 0; rep < 3; rep++) {
                cudaEventRecord(t0, stream);
                if (cublasLtMatmul(lt, p.op, &alpha, B, p.la, A, p.lb, &beta, C ? C : D, p.lc, D, p.ld, &results[i].algo, g_lt_ws, LT_WS_BYTES, stream) != CUBLAS_STATUS_SUCCESS) { ms_min = 1e30f; break; }
                cudaEventRecord(t1, stream);
                cudaEventSynchronize(t1);
                float ms = 0; cudaEventElapsedTime(&ms, t0, t1); ms_min = std::min(ms_min, ms);
            }
            if (ms_min < best_ms) { best_ms = ms_min; best = i; }
        }
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        if (env_int("BOPL_V4_MEMLOG", 0))
            fprintf(stderr, "bopl v4 cublasLt: M=%d N=%d K=%d %s%s%s: candidate %d of %d, %.3f ms\n", k.M, k.N, k.K, k.a_t ? "A^T " : "", k.b_t ? "B^T " : "",
                    k.d_bf16 ? "bf16 out" : "fp32 out", best, found, best_ms);
    }
    p.algo = results[best].algo;
    g_lt_plans.push_back(p);
    return g_lt_plans.back();
}
/* epi: DEFAULT, RELU, RELU_AUX (ReLU, writing its bitmask to aux) or DRELU
 * (multiply by the bitmask read from aux). */
static void lt_gemm(bool a_t, bool b_t, int M, int N, int K, const bf16* A, int lda, const bf16* B, int ldb,
                    const bf16* C, void* D, int ldd, bool d_bf16, cublasLtEpilogue_t epi, cudaStream_t stream, void* aux = nullptr) {
    cublasLtHandle_t lt = (cublasLtHandle_t)cublas_get_handle(stream);
    LtKey key = {a_t, b_t, M, N, K, lda, ldb, ldd, d_bf16, C != nullptr, (int)epi, aux};
    const LtPlan& p = lt_plan(lt, key, A, B, C, D, stream);
    float alpha = 1.0f, beta = C ? 1.0f : 0.0f;
    lt_check(cublasLtMatmul(lt, p.op, &alpha, B, p.la, A, p.lb, &beta, C ? (const void*)C : D, p.lc, D, p.ld, &p.algo, g_lt_ws, LT_WS_BYTES, stream), "matmul");
}
/* Zero the packed rows [total, cap) so the GEMMs over the capacity see finite zeros. */
__global__ void zero_tail(bf16* rows, int K, const int* totals, int total_index, int cap) {
    long n = (long)(cap - totals[total_index]) * K;
    bf16* tail = rows + (long)totals[total_index] * K;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) tail[i] = tobf(0.0f);
}
/* rows[m] = E[id[m]] for m < total, 0 past it: the h1 GEMM's addend. */
__global__ void embed_rows(const bf16* E, int I, const int* ids, const int* totals, int total_index, int cap, bf16* rows) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)cap * I) return;
    int m = (int)(i / I), n = (int)(i - (long)m * I);
    rows[i] = m < totals[total_index] ? E[(long)ids[m] * I + n] : tobf(0.0f);
}
/* g[i] = g[i] where act[i] > 0 else 0, in place (rows past the count have act == 0):
 * the ReLU backward for sets whose capacity is below LT_AUX_MIN_ROWS. Above it
 * the h1 GEMM's RELU_AUX epilogue writes a bitmask and the dh1 GEMM's DRELU
 * epilogue applies it, saving this pass over cap x I (~0.7 ms at B=12288).
 * The threshold: cublasLt's small-M RELU_AUX kernels round the epilogue
 * differently from RELU (double rounding, ~1 bf16 ulp on h1, enough to flip
 * max-pool winners against the oracle); from M=1600 up they agree exactly
 * (measured K=32/48/128, M=64..73728). */
constexpr int LT_AUX_MIN_ROWS = 2048;
__global__ void mask_rows(bf16* g, const bf16* act, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n && !(bf(act[i]) > 0.0f)) g[i] = tobf(0.0f);
}
static int grid_long(long n) { return (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE); }

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
/* dh2[off + r][c] = ReLU'(h2) * (winner == r ? gmax : 0) + gmean / n for the
 * sample's rows: one block per sample like pool_rows (the per-packed-row form
 * was latency bound on its sample -> offsets/counts/winner load chain). Rows
 * past the count are zeroed separately (zero_tail). */
__global__ void unpool_rows(const bf16* h2, bf16* dh2, int D, const int* offsets, const int* counts, const int* totals,
                            int total_index, const float* grad_joined, int ldj, int col0, const int* winner, int mean_pool) {
    int b = blockIdx.x, c = threadIdx.x;
    int off = offsets[b], n = counts[b], total = totals[total_index];
    if (off + n > total) n = max(0, total - off);
    int win = winner[(long)b * D + c];
    float gmax = grad_joined[(long)b * ldj + col0 + c];
    float gmean = (mean_pool && n > 0) ? grad_joined[(long)b * ldj + col0 + D + c] / n : 0.0f;
    for (int r = 0; r < n; r++) {
        long i = (long)(off + r) * D + c;
        float g = gmean + (win == r ? gmax : 0.0f);
        dh2[i] = tobf(bf(h2[i]) > 0.0f ? g : 0.0f);
    }
}
/* Embedding gradient partials: one block per chunk of packed rows, fixed
 * order. EMB_LANES row lanes each own an (IDS, I) accumulator and walk every
 * EMB_LANES-th row with EMB_UNROLL loads in flight (the one-row-at-a-time
 * walk was latency bound: 512 dependent load-add steps per block); the lanes
 * are summed in fixed order at the end so the result stays deterministic. */
constexpr int EMB_LANES = 2, EMB_UNROLL = 4;
__global__ void embedding_partials(const bf16* dh1, int I, const int* ids, const int* totals, int total_index, int cap,
                                   float* partials /* (chunks, IDS, I) */) {
    extern __shared__ float acc[];   /* EMB_LANES x IDS x I */
    int ch = blockIdx.x, i = threadIdx.x % I, lane = threadIdx.x / I;
    float* mine = acc + lane * IDS * I;
    for (int e = threadIdx.x; e < EMB_LANES * IDS * I; e += blockDim.x) acc[e] = 0.0f;
    __syncthreads();
    int total = totals[total_index];
    int begin = ch * CHUNK, end = min(cap, min(total, begin + CHUNK));
    for (int r0 = begin + lane; r0 < end; r0 += EMB_LANES * EMB_UNROLL) {
        float v[EMB_UNROLL]; int id[EMB_UNROLL];
        #pragma unroll
        for (int u = 0; u < EMB_UNROLL; u++) {
            int r = r0 + u * EMB_LANES;
            v[u] = r < end ? bf(dh1[(long)r * I + i]) : 0.0f;
            id[u] = r < end ? ids[r] : 0;
        }
        #pragma unroll
        for (int u = 0; u < EMB_UNROLL; u++) mine[id[u] * I + i] += v[u];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < IDS * I; e += blockDim.x) {
        float sum = 0.0f;
        for (int l = 0; l < EMB_LANES; l++) sum += acc[l * IDS * I + e];
        partials[(long)ch * IDS * I + e] = sum;
    }
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
    IntTensor erows[SETS];                               /* bf16 gathered embedding rows (id sets), the h1 GEMM's addend */
    PrecisionTensor grad_fused, grad_joined, grad_flat, partials, acc;
    IntTensor dh2[SETS];
    IntTensor relu_mask[SETS];   /* h1's ReLU bitmask (cap, I/32) for the DRELU epilogue; unset in rollout, at small caps, or when I % 128 != 0 */
    PrecisionTensor wgrad_w1[SETS], wgrad_w2[SETS], wgrad_embedding, wgrad_flat, wgrad_fusion;
};
/* (B, PRIV) of the last encoder forward on this thread, for the decoder of the
 * same policy_forward. Thread-local: the buffer threads' rollout chains and the
 * trainer's forward run concurrently (a shared global handed one thread's
 * privileged rows to another's decoder). */
static thread_local const float* g_v4_privileged = nullptr;
static thread_local int g_v4_privileged_rows = 0;
static IntTensor* g_v4_overflow = nullptr;        /* (SETS) dropped rows of the last training forward */

static bf16* bfp(IntTensor& t) { return (bf16*)t.data; }

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
    scatter_rows<0><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[0], bfp(a->xpack[0]), a->ids[0].data, a->sample[0].data);
    scatter_rows<1><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[1], bfp(a->xpack[1]), a->ids[1].data, a->sample[1].data);
    scatter_rows<2><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[2], bfp(a->xpack[2]), a->ids[2].data, a->sample[2].data);
    scatter_rows<3><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[3], bfp(a->xpack[3]), a->ids[3].data, a->sample[3].data);
    prof_mark("enc.count_scan_scatter", stream);
    for (int s = 0; s < SETS; s++) {
        const int* total = a->totals.data + s;
        int K = SPEC[s].kpad, cap = a->cap[s];
        zero_tail<<<256, BLOCK_SIZE, 0, stream>>>(bfp(a->xpack[s]), K, a->totals.data, s, cap);
        /* h1 = ReLU(X W1^T + E[id]) over the capacity; W1 stored (I, K), E[id] gathered as the bf16 addend */
        const bf16* erows = nullptr;
        if (SPEC[s].id_col >= 0) {
            erows = bfp(a->erows[s]);
            embed_rows<<<grid_long((long)cap * I), BLOCK_SIZE, 0, stream>>>(bfp(a->embb), I, a->ids[s].data, a->totals.data, s, cap, bfp(a->erows[s]));
        }
        void* mask = a->relu_mask[s].data;   /* the backward's DRELU bitmask when registered (training, cap >= LT_AUX_MIN_ROWS) */
        lt_gemm(false, true, cap, I, K, bfp(a->xpack[s]), K, bfp(a->w1b[s]), K, erows, bfp(a->h1[s]), I, true,
                mask ? CUBLASLT_EPILOGUE_RELU_AUX : CUBLASLT_EPILOGUE_RELU, stream, mask);
        /* h2 = ReLU(h1 W2^T); W2 stored (D, I) */
        lt_gemm(false, true, cap, D, I, bfp(a->h1[s]), I, bfp(a->w2b[s]), I, nullptr, bfp(a->h2[s]), D, true, CUBLASLT_EPILOGUE_RELU, stream);
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
        unpool_rows<<<B, D, 0, stream>>>(bfp(a->h2[s]), bfp(a->dh2[s]), D, a->offsets.data + (long)s * B, a->counts.data + (long)s * B,
                                         a->totals.data, s, a->grad_joined.data, LDJ, FD + s * 2 * D, a->winner[s].data, w->mean_pool);
        zero_tail<<<256, BLOCK_SIZE, 0, stream>>>(bfp(a->dh2[s]), D, a->totals.data, s, cap);
        if (g_set_stage.backward)
            g_set_stage.backward(s, bfp(a->h2[s]), bfp(a->dh2[s]), a->sample[s].data, a->offsets.data + (long)s * B, a->counts.data + (long)s * B, total, B, D, stream);
        prof_mark("bwd.enc.unpool", stream);
        /* dW2 (D, I) = dh2^T h1 over the capacity (zero rows past the count) */
        lt_gemm(true, false, D, I, cap, bfp(a->dh2[s]), D, bfp(a->h1[s]), I, nullptr, a->wgrad_w2[s].data, I, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        prof_mark("bwd.enc.dW2", stream);
        /* dh1 = (dh2 W2) * ReLU'(h1), written over h2 (no longer needed) */
        bf16* dh1 = bfp(a->h2[s]);
        void* mask = a->relu_mask[s].data;
        lt_gemm(false, false, cap, I, D, bfp(a->dh2[s]), D, bfp(a->w2b[s]), I, nullptr, dh1, I, true,
                mask ? CUBLASLT_EPILOGUE_DRELU : CUBLASLT_EPILOGUE_DEFAULT, stream, mask);
        if (!mask) mask_rows<<<grid_long((long)cap * I), BLOCK_SIZE, 0, stream>>>(dh1, bfp(a->h1[s]), (long)cap * I);
        prof_mark("bwd.enc.dh1", stream);
        /* dW1 (I, K) = dh1^T X into the scratch, then keep the first `features` of the padded K */
        lt_gemm(true, false, I, K, cap, dh1, I, bfp(a->xpack[s]), K, nullptr, a->acc.data, K, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        reduce_partials<<<grid_size(I * SPEC[s].features), BLOCK_SIZE, 0, stream>>>(a->acc.data, 1, 0, I, K, SPEC[s].features, a->wgrad_w1[s].data, 0);
        prof_mark("bwd.enc.dW1", stream);
        if (SPEC[s].id_col >= 0) {
            int chunks = (cap + CHUNK - 1) / CHUNK;
            embedding_partials<<<chunks, EMB_LANES * I, EMB_LANES * IDS * I * sizeof(float), stream>>>(dh1, I, a->ids[s].data, a->totals.data, s, cap, a->partials.data);
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
    for (int s = 0; s < SETS; s++) if (SPEC[s].id_col >= 0) { a->erows[s] = {.shape = {a->cap[s], I / 2}}; alloc_register(alloc, &a->erows[s]); }
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
    long partial = 0, acc = 0;   /* embedding-gradient chunk partials; dW1's fp32 (I, kpad) */
    for (int s = 0; s < SETS; s++) {
        a->dh2[s] = {.shape = {a->cap[s], D / 2}}; alloc_register(acts, &a->dh2[s]);
        if (I % 128 == 0 && a->cap[s] >= LT_AUX_MIN_ROWS && env_int("BOPL4_DRELU", 1)) {
            a->relu_mask[s] = {.shape = {a->cap[s], I / 32}}; alloc_register(acts, &a->relu_mask[s]);
        }
        long chunks = (a->cap[s] + CHUNK - 1) / CHUNK;
        if (SPEC[s].id_col >= 0) partial = std::max(partial, chunks * (long)IDS * I);
        acc = std::max(acc, (long)I * SPEC[s].kpad);
    }
    a->acc = {.shape = {acc}}; alloc_register(acts, &a->acc);
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
    lt_workspace();
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
    int arows;                 /* actor rows: od, or od + 1 when cdim == 0 (v3-style shared linear value, the v4-critic-shared ablation) */
};
struct DecActs {
    DecoderActivations base;   /* out, grad_out, saved_input, grad_input, ..., od1_pad: read by the trainer */
    int B, od1;
    PrecisionTensor actor_pad, wgrad_actor_pad;
    PrecisionTensor ln1, f, y, zin, lnin, z, ln2, bblk, z2, ln3, value;
    FloatTensor m1, r1, min_, rin, m2, r2, m3, r3;
    /* bf16 operands of the critic GEMMs (cublasLt, fp32 accumulation): the
     * weights, the forward inputs (reused by the weight gradients) and the
     * output gradients; 2 per int */
    IntTensor cin_b, b1_b, b2_b, lnin_b, ln2_b, bblk_b, g_z_b, g_b_b, g_z2_b;
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
    prof_mark("dec.ffn", stream);
    /* logits: actor rows then zero rows to the padded width */
    pad_rows<<<grid_size(pad * H), BLOCK_SIZE, 0, stream>>>(a->actor_pad.data, w->actor.data, w->arows, pad, H);
    puf_mm(y, &a->actor_pad, &a->base.out, stream);
    prof_mark("dec.actor", stream);
    /* value */
    if (w->cdim) {
        const float* priv = w->privileged ? g_v4_privileged : nullptr;
        if (w->privileged && g_v4_privileged_rows != B) throw std::runtime_error("Bopl v4 decoder: privileged rows do not match the batch");
        int P = w->privileged ? PRIV : 0, W = H + P;
        int C = w->cdim;
        concat_priv<<<grid_size(B * W), BLOCK_SIZE, 0, stream>>>(y->data, H, priv, P, a->zin.data, B);
        ln_forward<<<B, 256, 0, stream>>>(a->zin.data, a->lnin.data, a->min_.data, a->rin.data, W);
        /* the critic GEMMs run bf16 on tensor cores (fp32 accumulation, fp32 outputs) */
        to_bf16<<<grid_size(C * W), BLOCK_SIZE, 0, stream>>>(w->critic_in.data, bfp(a->cin_b), C * W);
        to_bf16<<<grid_size(C * C), BLOCK_SIZE, 0, stream>>>(w->critic_b1.data, bfp(a->b1_b), C * C);
        to_bf16<<<grid_size(C * C), BLOCK_SIZE, 0, stream>>>(w->critic_b2.data, bfp(a->b2_b), C * C);
        to_bf16<<<grid_size(B * W), BLOCK_SIZE, 0, stream>>>(a->lnin.data, bfp(a->lnin_b), B * W);
        lt_gemm(false, true, B, C, W, bfp(a->lnin_b), W, bfp(a->cin_b), W, nullptr, a->z.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        prof_mark("dec.critic_in", stream);
        ln_forward<<<B, 256, 0, stream>>>(a->z.data, a->ln2.data, a->m2.data, a->r2.data, C);
        to_bf16<<<grid_size(B * C), BLOCK_SIZE, 0, stream>>>(a->ln2.data, bfp(a->ln2_b), B * C);
        lt_gemm(false, true, B, C, C, bfp(a->ln2_b), C, bfp(a->b1_b), C, nullptr, a->bblk.data, C, false, CUBLASLT_EPILOGUE_RELU, stream);
        to_bf16<<<grid_size(B * C), BLOCK_SIZE, 0, stream>>>(a->bblk.data, bfp(a->bblk_b), B * C);
        lt_gemm(false, true, B, C, C, bfp(a->bblk_b), C, bfp(a->b2_b), C, nullptr, a->z2.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        add_into<<<grid_size(B * C), BLOCK_SIZE, 0, stream>>>(a->z2.data, a->z.data, B * C);
        prof_mark("dec.critic_block", stream);
        ln_forward<<<B, 256, 0, stream>>>(a->z2.data, a->ln3.data, a->m3.data, a->r3.data, w->cdim);
        puf_mm(&a->ln3, &vout, &a->value, stream);
        write_column<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(a->base.out.data, pad, w->od, a->value.data, B);
    }
    prof_mark("dec.value", stream);
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
    cudaMemcpyAsync(a->wgrad_actor.data, a->wgrad_actor_pad.data, (size_t)w->arows * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    puf_mm_nn(&a->base.grad_out, &a->actor_pad, &a->g_y, stream);   /* rows >= arows of actor_pad are zero */
    prof_mark("bwd.dec.actor", stream);
    /* critic */
    if (w->cdim) {
        int C = w->cdim, P = w->privileged ? PRIV : 0, W = H + P;
        read_column<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(a->base.grad_out.data, pad, od, a->gv.data, B);
        puf_mm_tn(&a->gv, &a->ln3, &a->wgrad_critic_out, stream);
        PrecisionTensor vout = {.data = w->critic_out.data, .shape = {1, C}};
        puf_mm_nn(&a->gv, &vout, &a->g_ln3, stream);
        prof_mark("bwd.dec.value_gemms", stream);
        ln_backward<<<B, 256, 0, stream>>>(a->g_ln3.data, a->z2.data, a->m3.data, a->r3.data, a->g_z2.data, C, 0);
        prof_mark("bwd.dec.ln3_bwd", stream);
        /* bf16 operands as in the forward: dW = dY^T X, dX = dY W */
        to_bf16<<<grid_size(B * C), BLOCK_SIZE, 0, stream>>>(a->g_z2.data, bfp(a->g_z2_b), B * C);
        lt_gemm(true, false, C, C, B, bfp(a->g_z2_b), C, bfp(a->bblk_b), C, nullptr, a->wgrad_b2.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        lt_gemm(false, false, B, C, C, bfp(a->g_z2_b), C, bfp(a->b2_b), C, nullptr, a->g_b.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        relu_grad_from<<<grid_size(B * C), BLOCK_SIZE, 0, stream>>>(a->g_b.data, C, a->bblk.data, C, a->g_b.data, B, C);
        to_bf16<<<grid_size(B * C), BLOCK_SIZE, 0, stream>>>(a->g_b.data, bfp(a->g_b_b), B * C);
        lt_gemm(true, false, C, C, B, bfp(a->g_b_b), C, bfp(a->ln2_b), C, nullptr, a->wgrad_b1.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        lt_gemm(false, false, B, C, C, bfp(a->g_b_b), C, bfp(a->b1_b), C, nullptr, a->g_ln2.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        prof_mark("bwd.dec.critic_block_gemms", stream);
        puf_copy(&a->g_z, &a->g_z2, stream);
        ln_backward<<<B, 256, 0, stream>>>(a->g_ln2.data, a->z.data, a->m2.data, a->r2.data, a->g_z.data, C, 1);
        prof_mark("bwd.dec.ln2_bwd", stream);
        to_bf16<<<grid_size(B * C), BLOCK_SIZE, 0, stream>>>(a->g_z.data, bfp(a->g_z_b), B * C);
        lt_gemm(true, false, C, W, B, bfp(a->g_z_b), C, bfp(a->lnin_b), W, nullptr, a->wgrad_critic_in.data, W, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        lt_gemm(false, false, B, W, C, bfp(a->g_z_b), C, bfp(a->cin_b), W, nullptr, a->g_lnin.data, W, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        ln_backward<<<B, 256, 0, stream>>>(a->g_lnin.data, a->zin.data, a->min_.data, a->rin.data, a->g_zin.data, W, 0);
        if (!w->detach) {
            /* g_y += g_zin[:, :H] */
            copy_cols<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->g_zin.data, W, 0, a->g_ln1.data, H, B, H);
            add_into<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->g_y.data, a->g_ln1.data, B * H);
        }
        prof_mark("bwd.dec.critic_in", stream);
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
    prof_mark("bwd.dec.ffn", stream);
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
    w->arows = w->od + (w->cdim ? 0 : 1);
    w->actor = {.shape = {w->arows, H}}; reg_aligned(alloc, &w->actor);
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
        if (W % 8 || C % 8) throw std::runtime_error("Bopl v4: the critic widths must be multiples of 8 (bf16 GEMM operands)");
        a->cin_b = {.shape = {C, W / 2}}; a->b1_b = {.shape = {C, C / 2}}; a->b2_b = {.shape = {C, C / 2}};
        a->lnin_b = {.shape = {B, W / 2}}; a->ln2_b = {.shape = {B, C / 2}}; a->bblk_b = {.shape = {B, C / 2}};
        for (auto* t : {&a->cin_b, &a->b1_b, &a->b2_b, &a->lnin_b, &a->ln2_b, &a->bblk_b}) alloc_register(alloc, t);
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
        a->g_z_b = {.shape = {B, C / 2}}; a->g_b_b = {.shape = {B, C / 2}}; a->g_z2_b = {.shape = {B, C / 2}};
        for (auto* t : {&a->g_z_b, &a->g_b_b, &a->g_z2_b}) alloc_register(acts, t);
    }
    /* gradients in reg_params order */
    if (w->ffn) {
        a->wgrad_ffn_in = {.shape = {w->ffn, H}}; alloc_register(grads, &a->wgrad_ffn_in);
        a->wgrad_ffn_out = {.shape = {H, w->ffn}}; alloc_register(grads, &a->wgrad_ffn_out);
    }
    a->wgrad_actor = {.shape = {w->arows, H}}; alloc_register(grads, &a->wgrad_actor);
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
    lt_workspace();
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
