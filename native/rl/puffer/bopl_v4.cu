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
 *            GEMMs run over the capacity, or in training over the batch's
 *            row count rounded up to a bucket (enc_prepass: one train graph
 *            per bucket tuple), with the tail kept at zero
 *   h1     = ReLU(W1 x + E[id])   h2 = ReLU(W2 h1)               cuBLAS bf16
 *            (E[id] as the one-hot id block of the packed row against E^T
 *            appended to W1: one GEMM, no gathered addend)
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
 * Determinism: the GEMMs are cuBLAS with a fixed shape and kernel choice and
 * deterministic split-K reductions; the embedding gradient is dW1's one-hot
 * columns, so it inherits that.
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
/* A set's packed row is [features | one-hot(id) (IDS, id sets only)] padded
 * to kpad: the embedding add E[id] is the GEMM's product with the one-hot
 * block against [W1 | E^T] (pack_w1), so h1 = ReLU([x | 1_id] [W1 | E^T]^T) =
 * ReLU(W1 x + E[id]) in one GEMM without a gathered addend, and the embedding
 * gradient is dW1's one-hot columns (the same fp32-accumulated GEMM). */
struct Spec { int offset, rows, width, id_col, features, kpad; };
__host__ __device__ constexpr int round16(int x) { return (x + 15) / 16 * 16; }
__host__ __device__ constexpr int kpad_of(int features, int id_col) { return round16(features + (id_col >= 0 ? IDS : 0)); }
#define BOPL4_SPEC_TABLE { \
    {BOPL4_OBS_PLAYER_OFFSET, BOPL4_PLAYERS, BOPL4_PLAYER_FEATURES, -1, BOPL4_PLAYER_FEATURES, kpad_of(BOPL4_PLAYER_FEATURES, -1)}, \
    {BOPL4_OBS_SLOT_OFFSET, BOPL4_SLOTS, BOPL4_SLOT_FEATURES, BOPL4_SLOT_ABILITY_ID, BOPL4_SLOT_FEATURES - 1, kpad_of(BOPL4_SLOT_FEATURES - 1, BOPL4_SLOT_ABILITY_ID)}, \
    {BOPL4_OBS_ENTITY_OFFSET, BOPL4_ENTITIES, BOPL4_ENTITY_FEATURES, BOPL4_ENTITY_ABILITY_ID, BOPL4_ENTITY_FEATURES - 1, kpad_of(BOPL4_ENTITY_FEATURES - 1, BOPL4_ENTITY_ABILITY_ID)}, \
    {BOPL4_OBS_SLIME_OFFSET, BOPL4_SLIMES, BOPL4_SLIME_FEATURES, -1, BOPL4_SLIME_FEATURES, kpad_of(BOPL4_SLIME_FEATURES, -1)}, \
}
constexpr Spec SPEC[SETS] = BOPL4_SPEC_TABLE;
__constant__ Spec c_spec[SETS] = BOPL4_SPEC_TABLE;
constexpr int MAX_KPAD = 96;
static_assert(SPEC[0].kpad <= MAX_KPAD && SPEC[1].kpad <= MAX_KPAD && SPEC[2].kpad <= MAX_KPAD && SPEC[3].kpad <= MAX_KPAD, "kpad");
constexpr int DEFAULT_BUDGET[SETS] = {6, 12, 20, 8};   /* average valid rows per sample the packed buffers hold */
constexpr int TILE = 64;                               /* packed capacities are rounded up to this many rows */
constexpr float LN_EPS = 1e-5f;

/* The shape variables train.py exports (BOPL4_SET_DIM, ...) and the memory log. */
static int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::atoi(v) : fallback;
}

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
/* Valid rows (column 0 != 0) of each set per sample. (A warp-per-sample form
 * with ballots measured +0.2%, inside the noise: notes-v4.md.) */
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
/* Scatter set S's valid rows into the packed bf16 matrix ([features |
 * one-hot id] in KPAD columns, zero padded), with their embedding id and
 * sample index: one warp per sample, the valid mask by ballot, rows copied by
 * the warp in row order (the thread-per-row form recomputed each row's rank
 * with an O(rows) strided walk). */
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
            int id = sp.id_col >= 0 ? embedding_id(src[sp.id_col]) : 0;
            for (int k = lane; k < sp.kpad; k += 32)
                out[k] = tobf(k < sp.features ? src[k] : (sp.id_col >= 0 && k - sp.features == id) ? 1.0f : 0.0f);
            if (lane == 0) { ids[dst] = id; sample[dst] = b; }
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
/* [W1 (I, features) | E^T (I, IDS) if the set has ids] -> bf16 (I, kpad), zero padded. */
__global__ void pack_w1(const float* w1, int features, const float* E, bf16* dst, int I, int kpad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I * kpad) return;
    int r = i / kpad, c = i - r * kpad;
    float v = c < features ? w1[r * features + c] : (E && c - features < IDS) ? E[(long)(c - features) * I + r] : 0.0f;
    dst[i] = tobf(v);
}
/* wgrad_embedding (IDS, I) (+)= the one-hot columns of dW1 (I, K) transposed. */
__global__ void embed_grad_from_w1(const float* dw1, int K, int features, int I, float* out, int accumulate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= IDS * I) return;
    int id = i / I, c = i - id * I;
    float v = dw1[(long)c * K + features + id];
    out[i] = accumulate ? out[i] + v : v;
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
 * input, zero_tail / unpool_rows on the gradients) so they contribute
 * nothing. a_t / b_t: the operand is stored transposed (K, M) / (N, K). C,
 * when given, is a bf16 (M, N) addend (beta = 1; unused since the embedding
 * moved into the GEMM, kept for the attention hook). Plans (layouts + the heuristic's first algorithm, split-K
 * restricted to the deterministic workspace reductions) are cached per shape
 * under a mutex: the buffer threads' per-bank chains and the trainer run
 * concurrently. The workspace is allocated once at enc_create. */
struct LtKey {
    int a_t, b_t, M, N, K, lda, ldb, ldd, d_bf16, has_c, epi;
    int ab_f32;        /* A, B fp32 (the flat / FFN GEMMs: TF32 tensor cores per BOPL_TF32, as puf_mm) instead of bf16 */
    const void* aux;   /* RELU_AUX / DRELU bitmask, part of the plan (a descriptor attribute) */
    bool operator==(const LtKey& o) const {
        return a_t == o.a_t && b_t == o.b_t && M == o.M && N == o.N && K == o.K && lda == o.lda && ldb == o.ldb && ldd == o.ldd
            && d_bf16 == o.d_bf16 && has_c == o.has_c && epi == o.epi && ab_f32 == o.ab_f32 && aux == o.aux;
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
    cudaDataType ab = k.ab_f32 ? CUDA_R_32F : CUDA_R_16BF;
    lt_check(cublasLtMatmulDescCreate(&p.op, k.ab_f32 ? bopl_cublas_compute_type() : CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta), "transa");
    lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb), "transb");
    cublasLtEpilogue_t epi = (cublasLtEpilogue_t)k.epi;
    lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi), "epilogue");
    if (k.aux) {   /* one bit per element, N bits (a multiple of 128) per packed row */
        int64_t ld = k.N;
        lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &k.aux, sizeof k.aux), "aux pointer");
        lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld, sizeof ld), "aux ld");
    }
    lt_check(cublasLtMatrixLayoutCreate(&p.la, ab, ta == CUBLAS_OP_N ? m : kk, ta == CUBLAS_OP_N ? kk : m, k.ldb), "layout A");
    lt_check(cublasLtMatrixLayoutCreate(&p.lb, ab, tb == CUBLAS_OP_N ? kk : n, tb == CUBLAS_OP_N ? n : kk, k.lda), "layout B");
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
    if (found < 1) {
        cublasLtMatmulDescDestroy(p.op); cublasLtMatrixLayoutDestroy(p.la); cublasLtMatrixLayoutDestroy(p.lb); cublasLtMatrixLayoutDestroy(p.lc); cublasLtMatrixLayoutDestroy(p.ld);
        throw std::runtime_error("Bopl v4 cublasLt: no algorithm for M=" + std::to_string(k.M) + " N=" + std::to_string(k.N) + " K=" + std::to_string(k.K) + " epilogue " + std::to_string(k.epi));
    }
    int best = 0;
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &capturing);
    if (found > 1 && A && capturing == cudaStreamCaptureStatusNone) {   /* time the candidates once per shape (never under capture) */
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
 * (multiply by the bitmask read from aux). C: an addend of D's type (beta 1). */
static void lt_matmul(const LtKey& key, const void* A, const void* B, const void* C, void* D, cudaStream_t stream) {
    cublasLtHandle_t lt = (cublasLtHandle_t)cublas_get_handle(stream);
    const LtPlan& p = lt_plan(lt, key, A, B, C, D, stream);
    float alpha = 1.0f, beta = C ? 1.0f : 0.0f;
    lt_check(cublasLtMatmul(lt, p.op, &alpha, B, p.la, A, p.lb, &beta, C ? C : D, p.lc, D, p.ld, &p.algo, g_lt_ws, LT_WS_BYTES, stream), "matmul");
}
static void lt_gemm(bool a_t, bool b_t, int M, int N, int K, const bf16* A, int lda, const bf16* B, int ldb,
                    const void* C, void* D, int ldd, bool d_bf16, cublasLtEpilogue_t epi, cudaStream_t stream, void* aux = nullptr) {
    lt_matmul({a_t, b_t, M, N, K, lda, ldb, ldd, d_bf16, C != nullptr, (int)epi, 0, aux}, A, B, C, D, stream);
}
/* fp32 operands and output (the compute type follows BOPL_TF32 like puf_mm). */
static void lt_gemm32(bool a_t, bool b_t, int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                      const float* C, float* D, int ldd, cublasLtEpilogue_t epi, cudaStream_t stream, void* aux = nullptr) {
    lt_matmul({a_t, b_t, M, N, K, lda, ldb, ldd, 0, C != nullptr, (int)epi, 1, aux}, A, B, C, D, stream);
}
/* Zero the packed rows [total, cap) of one or two (rows2 nullable) row
 * matrices so the GEMMs over the capacity see finite zeros. */
__global__ void zero_tail(bf16* rows, int K, bf16* rows2, int K2, const int* totals, int total_index, int cap) {
    int total = totals[total_index];
    long n = (long)(cap - total) * K, n2 = rows2 ? (long)(cap - total) * K2 : 0;
    bf16* tail = rows + (long)total * K; bf16* tail2 = rows2 + (long)total * K2;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n + n2; i += (long)gridDim.x * blockDim.x)
        (i < n ? tail[i] : tail2[i - n]) = tobf(0.0f);
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
/* The same pools with 4 columns per thread (8-byte bf16 loads, float4 / int4
 * stores): POOL_SAMPLES samples per block of POOL_THREADS, sample = warp when
 * D = 128. Column c's max / sum / winner are the same sequential scan over the
 * rows as pool_rows / unpool_rows, so the results are bit-identical; the
 * thread count per pass is a quarter (the scalar forms issued 2-byte loads).
 * Used when D % 4 == 0 and D / 4 divides POOL_THREADS. */
constexpr int POOL_THREADS = 128;
__device__ __forceinline__ void bf4_load(const bf16* p, float* v) {
    uint2 u = *reinterpret_cast<const uint2*>(p);
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162*>(&u.x), hi = *reinterpret_cast<__nv_bfloat162*>(&u.y);
    v[0] = __low2float(lo); v[1] = __high2float(lo); v[2] = __low2float(hi); v[3] = __high2float(hi);
}
__device__ __forceinline__ void bf4_store(bf16* p, const float* v) {
    __nv_bfloat162 lo = __floats2bfloat162_rn(v[0], v[1]), hi = __floats2bfloat162_rn(v[2], v[3]);
    uint2 u; u.x = *reinterpret_cast<unsigned*>(&lo); u.y = *reinterpret_cast<unsigned*>(&hi);
    *reinterpret_cast<uint2*>(p) = u;
}
__global__ void __launch_bounds__(POOL_THREADS) pool_rows4(const bf16* h2, int D, const int* offsets, const int* counts, int total_index,
                                                          const int* totals, float* joined, int ldj, int col0, int* winner, int mean_pool, int B) {
    const int tps = D / 4;   /* threads per sample */
    const int b = blockIdx.x * (POOL_THREADS / tps) + threadIdx.x / tps, c = (threadIdx.x % tps) * 4;
    if (b >= B) return;
    int off = offsets[b], n = counts[b], total = totals[total_index];
    if (off + n > total) n = max(0, total - off);
    float best[4] = {0.0f, 0.0f, 0.0f, 0.0f}, sum[4] = {0.0f, 0.0f, 0.0f, 0.0f}; int win[4] = {-1, -1, -1, -1};
    for (int r = 0; r < n; r++) {
        float v[4]; bf4_load(h2 + (long)(off + r) * D + c, v);
        #pragma unroll
        for (int j = 0; j < 4; j++) { if (v[j] > best[j]) { best[j] = v[j]; win[j] = r; } sum[j] += v[j]; }
    }
    *reinterpret_cast<float4*>(joined + (long)b * ldj + col0 + c) = make_float4(best[0], best[1], best[2], best[3]);
    float4 m = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (mean_pool && n > 0) m = make_float4(sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n);
    *reinterpret_cast<float4*>(joined + (long)b * ldj + col0 + D + c) = m;
    *reinterpret_cast<int4*>(winner + (long)b * D + c) = make_int4(win[0], win[1], win[2], win[3]);
}
__global__ void __launch_bounds__(POOL_THREADS) unpool_rows4(const bf16* h2, bf16* dh2, int D, const int* offsets, const int* counts, const int* totals,
                                                            int total_index, const float* grad_joined, int ldj, int col0, const int* winner, int mean_pool, int B) {
    const int tps = D / 4;
    const int b = blockIdx.x * (POOL_THREADS / tps) + threadIdx.x / tps, c = (threadIdx.x % tps) * 4;
    if (b >= B) return;
    int off = offsets[b], n = counts[b], total = totals[total_index];
    if (off + n > total) n = max(0, total - off);
    int4 w4 = *reinterpret_cast<const int4*>(winner + (long)b * D + c);
    float4 gm4 = *reinterpret_cast<const float4*>(grad_joined + (long)b * ldj + col0 + c);
    float gmax[4] = {gm4.x, gm4.y, gm4.z, gm4.w}, gmean[4] = {0.0f, 0.0f, 0.0f, 0.0f}; int win[4] = {w4.x, w4.y, w4.z, w4.w};
    if (mean_pool && n > 0) {
        float4 t = *reinterpret_cast<const float4*>(grad_joined + (long)b * ldj + col0 + D + c);
        gmean[0] = t.x / n; gmean[1] = t.y / n; gmean[2] = t.z / n; gmean[3] = t.w / n;
    }
    for (int r = 0; r < n; r++) {
        long i = (long)(off + r) * D + c;
        float v[4], g[4]; bf4_load(h2 + i, v);
        #pragma unroll
        for (int j = 0; j < 4; j++) { float gg = gmean[j] + (win[j] == r ? gmax[j] : 0.0f); g[j] = v[j] > 0.0f ? gg : 0.0f; }
        bf4_store(dh2 + i, g);
    }
}
static bool pool4_ok(int D, int ldj, int col0) { return D % 4 == 0 && POOL_THREADS % (D / 4) == 0 && ldj % 4 == 0 && col0 % 4 == 0; }
static int pool4_grid(int D, int B) { int per = POOL_THREADS / (D / 4); return (B + per - 1) / per; }

/* ---- fused set forward: h1 = ReLU(X W1^T), h2 = ReLU(h1 W2^T), pool ---------
 * One kernel per set replaces the two Lt GEMMs and pool_rows: a block owns
 * SF_SAMPLES consecutive samples, i.e. one contiguous range of the packed
 * rows (offsets are prefix sums, so a sample's rows never straddle blocks),
 * and walks it in 64-row tiles: the h1 tile by bf16 mma.sync (m16n8k16, fp32
 * accumulate, ReLU, bf16 round: the same rounding points as the Lt RELU
 * epilogue), kept in shared for the h2 tile by the same path, both tiles
 * written to global from shared with 16-byte stores (h1 for the backward's
 * dW2 and ReLU' mask, h2 for unpool). The pool is the sequential scan of
 * pool_rows over the tile in shared: thread = column, rows in order, strict
 * max with the first winner, fp32 sum in row order, so max / winner / mean
 * are bit-identical to pool_rows on the same h2. What differs from the Lt
 * path is the GEMMs' fp32 summation order (the check_v4 class). Rows past a
 * sample's clamped count, [total, act), are not touched here: zero_tail
 * zeroes h1's tail with xpack's for the dW2 GEMM over the active rows. */
constexpr int SF_THREADS = 128, SF_WARPS = SF_THREADS / 32, SF_ROWS = 64, SF_SAMPLES = 16, SF_PAD = 8;
__device__ __forceinline__ void mma_bf16_16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned bf2_pack_relu(float x, float y) {
    __nv_bfloat162 v = __floats2bfloat162_rn(fmaxf(x, 0.0f), fmaxf(y, 0.0f));
    return *reinterpret_cast<unsigned*>(&v);
}
static size_t set_fwd_smem(int I, int D) { return (size_t)SF_ROWS * ((I + SF_PAD) + (D + SF_PAD)) * sizeof(bf16); }
/* The tile kernel's shape coverage (the shipped shapes: I 64 / 128, D 128);
 * other shapes take the Lt GEMMs + pool_rows. */
static bool set_fwd_ok(int I, int D, int ldj, int col0) {
    return I % 16 == 0 && I <= 128 && D % SF_THREADS == 0 && D <= 4 * SF_THREADS && ldj % 4 == 0 && col0 % 4 == 0;
}
/* bf16 transpose of a (rows, cols) fp32 matrix into (cols, rows): W2^T for set_bwd's B operand. */
__global__ void to_bf16_t(const float* src, bf16* dst, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i - r * cols;
    dst[(long)c * rows + r] = tobf(src[i]);
}
template<int NCOLS>   /* columns per thread in the pool = D / SF_THREADS */
__global__ void __launch_bounds__(SF_THREADS) set_fwd(const bf16* __restrict__ xpack, int K, const bf16* __restrict__ w1b, int I,
                                                       const bf16* __restrict__ w2b, int D, const int* __restrict__ offsets,
                                                       const int* __restrict__ counts, const int* __restrict__ totals, int s, int B,
                                                       bf16* __restrict__ h1, bf16* __restrict__ h2, float* __restrict__ joined, int ldj,
                                                       int col0, int* __restrict__ winner, int mean_pool) {
    extern __shared__ __align__(16) unsigned char sf_smem[];
    bf16* h1s = (bf16*)sf_smem;                       /* [SF_ROWS][I + SF_PAD] */
    bf16* h2s = h1s + SF_ROWS * (I + SF_PAD);         /* [SF_ROWS][D + SF_PAD] */
    const int L1 = I + SF_PAD, L2 = D + SF_PAD;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tid4 = lane & 3;
    const int b0 = blockIdx.x * SF_SAMPLES, b1 = min(b0 + SF_SAMPLES, B);
    const int total = totals[s];
    const int row0 = min(offsets[b0], total);
    const int row1 = min(b1 < B ? offsets[b1] : total, total);
    /* pool state: thread = column c (and c + SF_THREADS ... when D > SF_THREADS) of the current sample */
    float best[NCOLS], sum[NCOLS]; int win[NCOLS];
    int b = b0, off = offsets[b0], n = max(0, min(counts[b0], total - offsets[b0]));
    #pragma unroll
    for (int q = 0; q < NCOLS; q++) { best[q] = 0.0f; sum[q] = 0.0f; win[q] = -1; }
    auto flush = [&]() {
        #pragma unroll
        for (int q = 0; q < NCOLS; q++) {
            int c = tid + q * SF_THREADS;
            joined[(long)b * ldj + col0 + c] = best[q];
            joined[(long)b * ldj + col0 + D + c] = (mean_pool && n > 0) ? sum[q] / n : 0.0f;
            winner[(long)b * D + c] = win[q];
            best[q] = 0.0f; sum[q] = 0.0f; win[q] = -1;
        }
        b++;
        if (b < b1) { off = offsets[b]; n = max(0, min(counts[b], total - off)); }
    };
    float acc[16][4];   /* 16 n-tiles of 8 columns: h1 (I <= 128) or one 128-column chunk of h2 */
    for (int tile0 = row0; tile0 < row1; tile0 += SF_ROWS) {
        const int wrow = tile0 + warp * 16;   /* this warp's 16 rows */
        /* h1 tile: acc[nt] = X[wrow.., :] W1[nt*8.., :]^T */
        const int nt1 = I / 8;
        #pragma unroll
        for (int t = 0; t < 16; t++) { acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f; }
        const int ra = wrow + gid, rb = wrow + gid + 8;
        const bf16* xa = xpack + (long)ra * K; const bf16* xb = xpack + (long)rb * K;
        for (int k0 = 0; k0 < K; k0 += 16) {
            unsigned a[4];
            const int ka = k0 + tid4 * 2;
            a[0] = ra < row1 ? *(const unsigned*)(xa + ka) : 0u;
            a[1] = rb < row1 ? *(const unsigned*)(xb + ka) : 0u;
            a[2] = ra < row1 ? *(const unsigned*)(xa + ka + 8) : 0u;
            a[3] = rb < row1 ? *(const unsigned*)(xb + ka + 8) : 0u;
            #pragma unroll
            for (int t = 0; t < 16; t++) if (t < nt1) {
                const bf16* wr = w1b + (long)(t * 8 + gid) * K + ka;
                mma_bf16_16816(acc[t], a, *(const unsigned*)wr, *(const unsigned*)(wr + 8));
            }
        }
        #pragma unroll
        for (int t = 0; t < 16; t++) if (t < nt1) {
            *(unsigned*)(h1s + (warp * 16 + gid) * L1 + t * 8 + tid4 * 2) = bf2_pack_relu(acc[t][0], acc[t][1]);
            *(unsigned*)(h1s + (warp * 16 + gid + 8) * L1 + t * 8 + tid4 * 2) = bf2_pack_relu(acc[t][2], acc[t][3]);
        }
        __syncthreads();
        /* h2 tile in 128-column chunks: acc[nt] = h1s[wrow.., :] W2[c0 + nt*8.., :]^T */
        for (int c0 = 0; c0 < D; c0 += 128) {
            const int nt2 = min(D - c0, 128) / 8;
            #pragma unroll
            for (int t = 0; t < 16; t++) { acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f; }
            const bf16* ha = h1s + (warp * 16 + gid) * L1; const bf16* hb = ha + 8 * L1;
            for (int k0 = 0; k0 < I; k0 += 16) {
                unsigned a[4];
                const int ka = k0 + tid4 * 2;
                a[0] = *(const unsigned*)(ha + ka); a[1] = *(const unsigned*)(hb + ka);
                a[2] = *(const unsigned*)(ha + ka + 8); a[3] = *(const unsigned*)(hb + ka + 8);
                #pragma unroll
                for (int t = 0; t < 16; t++) if (t < nt2) {
                    const bf16* wr = w2b + (long)(c0 + t * 8 + gid) * I + ka;
                    mma_bf16_16816(acc[t], a, *(const unsigned*)wr, *(const unsigned*)(wr + 8));
                }
            }
            #pragma unroll
            for (int t = 0; t < 16; t++) if (t < nt2) {
                *(unsigned*)(h2s + (warp * 16 + gid) * L2 + c0 + t * 8 + tid4 * 2) = bf2_pack_relu(acc[t][0], acc[t][1]);
                *(unsigned*)(h2s + (warp * 16 + gid + 8) * L2 + c0 + t * 8 + tid4 * 2) = bf2_pack_relu(acc[t][2], acc[t][3]);
            }
        }
        __syncthreads();
        /* both tiles to global, 16-byte stores along the rows (rows < row1 only) */
        const int rows_here = min(SF_ROWS, row1 - tile0);
        for (int i = tid; i < rows_here * (I / 8); i += SF_THREADS) {
            int r = i / (I / 8), c8 = (i - r * (I / 8)) * 8;
            *(uint4*)(h1 + (long)(tile0 + r) * I + c8) = *(const uint4*)(h1s + r * L1 + c8);
        }
        for (int i = tid; i < rows_here * (D / 8); i += SF_THREADS) {
            int r = i / (D / 8), c8 = (i - r * (D / 8)) * 8;
            *(uint4*)(h2 + (long)(tile0 + r) * D + c8) = *(const uint4*)(h2s + r * L2 + c8);
        }
        /* the pool: rows in order, samples advanced as their rows end */
        for (int r = 0; r < rows_here; r++) {
            const int row = tile0 + r;
            while (row >= off + n) flush();   /* empty samples flush zeros */
            #pragma unroll
            for (int q = 0; q < NCOLS; q++) {
                float v = bf(h2s[r * L2 + tid + q * SF_THREADS]);
                if (v > best[q]) { best[q] = v; win[q] = row - off; }
                sum[q] += v;
            }
        }
        __syncthreads();   /* the tiles are read before the next tile overwrites them */
    }
    while (b < b1) flush();
}
/* The backward of the same tiles: dh2 = ReLU'(h2) * (winner term + mean
 * term) as unpool_rows (thread = column, rows in order, the sample's
 * winner / gmax / gmean loaded as its rows begin), written to global for the
 * dW2 GEMM and kept in shared as the A operand of dh1 = dh2 W2 (B = W2^T
 * (I, D) bf16, fp32 accumulate, bf16 round: the Lt dh1 GEMM's rounding
 * points), masked by ReLU'(h1) in a coalesced pass into its own rows (the
 * Lt path wrote dh1 over h2, whose (cap, D) rows a fused kernel's other
 * blocks are still reading: dh1 (cap, I) here). Replaces unpool_rows4 + the
 * dh1 Lt GEMM + mask_rows; zero_tail zeroes dh2's and dh1's tails
 * [total, act) for the weight GEMMs. */
static size_t set_bwd_smem(int I, int D) { return set_fwd_smem(I, D); }
template<int NCOLS>
__global__ void __launch_bounds__(SF_THREADS) set_bwd(const bf16* __restrict__ h1, const bf16* __restrict__ h2, const bf16* __restrict__ w2tb,
                                                       int I, int D, const int* __restrict__ offsets, const int* __restrict__ counts,
                                                       const int* __restrict__ totals, int s, int B, const float* __restrict__ grad_joined,
                                                       int ldj, int col0, const int* __restrict__ winner, int mean_pool,
                                                       bf16* __restrict__ dh2, bf16* __restrict__ dh1) {
    extern __shared__ __align__(16) unsigned char sf_smem[];
    bf16* dh1s = (bf16*)sf_smem;                      /* [SF_ROWS][I + SF_PAD] */
    bf16* dh2s = dh1s + SF_ROWS * (I + SF_PAD);       /* [SF_ROWS][D + SF_PAD] */
    const int L1 = I + SF_PAD, L2 = D + SF_PAD;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tid4 = lane & 3;
    const int b0 = blockIdx.x * SF_SAMPLES, b1 = min(b0 + SF_SAMPLES, B);
    const int total = totals[s];
    const int row0 = min(offsets[b0], total);
    const int row1 = min(b1 < B ? offsets[b1] : total, total);
    float gmax[NCOLS], gmean[NCOLS]; int win[NCOLS];
    int b = b0, off = offsets[b0], n = max(0, min(counts[b0], total - offsets[b0]));
    auto load_sample = [&]() {
        #pragma unroll
        for (int q = 0; q < NCOLS; q++) {
            int c = tid + q * SF_THREADS;
            win[q] = winner[(long)b * D + c];
            gmax[q] = grad_joined[(long)b * ldj + col0 + c];
            gmean[q] = (mean_pool && n > 0) ? grad_joined[(long)b * ldj + col0 + D + c] / n : 0.0f;
        }
    };
    auto next_sample = [&]() { b++; if (b < b1) { off = offsets[b]; n = max(0, min(counts[b], total - off)); load_sample(); } };
    if (row0 < row1) load_sample();
    float acc[16][4];
    for (int tile0 = row0; tile0 < row1; tile0 += SF_ROWS) {
        const int rows_here = min(SF_ROWS, row1 - tile0);
        /* the h2 tile into shared with 16-byte row loads (the per-row walk
         * below then reads shared: reading h2 from global per row cost ~30 us
         * of dependent latency per tile, 886 us per launch: notes-v4.md);
         * rows of the tile past row1 are zeroed, they are multiplied but not stored */
        for (int i = tid; i < SF_ROWS * (D / 8); i += SF_THREADS) {
            int r = i / (D / 8), c8 = (i - r * (D / 8)) * 8;
            *(uint4*)(dh2s + r * L2 + c8) = r < rows_here ? *(const uint4*)(h2 + (long)(tile0 + r) * D + c8) : make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
        /* dh2 rows in place: thread = column, rows in order */
        for (int r = 0; r < rows_here; r++) {
            const int row = tile0 + r;
            while (row >= off + n) next_sample();
            #pragma unroll
            for (int q = 0; q < NCOLS; q++) {
                int c = tid + q * SF_THREADS;
                float v = bf(dh2s[r * L2 + c]);
                float g = gmean[q] + (win[q] == row - off ? gmax[q] : 0.0f);
                dh2s[r * L2 + c] = tobf(v > 0.0f ? g : 0.0f);
            }
        }
        __syncthreads();
        /* dh2 to global for the dW2 GEMM, 16 bytes per thread along the rows */
        for (int i = tid; i < rows_here * (D / 8); i += SF_THREADS) {
            int r = i / (D / 8), c8 = (i - r * (D / 8)) * 8;
            *(uint4*)(dh2 + (long)(tile0 + r) * D + c8) = *(const uint4*)(dh2s + r * L2 + c8);
        }
        /* dh1 tile: acc[nt] = dh2s[wrow.., :] W2^T[nt*8.., :]^T, i.e. dh2 W2 */
        const int nt1 = I / 8;
        #pragma unroll
        for (int t = 0; t < 16; t++) { acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f; }
        const bf16* ga = dh2s + (warp * 16 + gid) * L2; const bf16* gb = ga + 8 * L2;
        for (int k0 = 0; k0 < D; k0 += 16) {
            unsigned a[4];
            const int ka = k0 + tid4 * 2;
            a[0] = *(const unsigned*)(ga + ka); a[1] = *(const unsigned*)(gb + ka);
            a[2] = *(const unsigned*)(ga + ka + 8); a[3] = *(const unsigned*)(gb + ka + 8);
            #pragma unroll
            for (int t = 0; t < 16; t++) if (t < nt1) {
                const bf16* wr = w2tb + (long)(t * 8 + gid) * D + ka;
                mma_bf16_16816(acc[t], a, *(const unsigned*)wr, *(const unsigned*)(wr + 8));
            }
        }
        #pragma unroll
        for (int t = 0; t < 16; t++) if (t < nt1) {
            __nv_bfloat162 lo = __floats2bfloat162_rn(acc[t][0], acc[t][1]), hi = __floats2bfloat162_rn(acc[t][2], acc[t][3]);
            *(__nv_bfloat162*)(dh1s + (warp * 16 + gid) * L1 + t * 8 + tid4 * 2) = lo;
            *(__nv_bfloat162*)(dh1s + (warp * 16 + gid + 8) * L1 + t * 8 + tid4 * 2) = hi;
        }
        __syncthreads();
        /* dh1 = dh1 * ReLU'(h1) to global, 16 bytes per thread along the rows */
        for (int i = tid; i < rows_here * (I / 8); i += SF_THREADS) {
            int r = i / (I / 8), c8 = (i - r * (I / 8)) * 8;
            uint4 hv = *(const uint4*)(h1 + (long)(tile0 + r) * I + c8), gv = *(const uint4*)(dh1s + r * L1 + c8);
            const bf16* hp = (const bf16*)&hv; bf16* gp = (bf16*)&gv;
            #pragma unroll
            for (int j = 0; j < 8; j++) if (!(bf(hp[j]) > 0.0f)) gp[j] = tobf(0.0f);
            *(uint4*)(dh1 + (long)(tile0 + r) * I + c8) = gv;
        }
        __syncthreads();
    }
}
/* The launchers follow both definitions: a template declared without the
 * definition's __restrict__ qualifiers mangles to another device symbol, and
 * the launch then fails with "invalid device function" without running
 * (found by check_dump; the error is checked here so it can never pass silently). */
template<typename... A> static void set_bwd_launch(int ncols, int grid, size_t smem, cudaStream_t stream, A... args) {
    switch (ncols) {
        case 1: set_bwd<1><<<grid, SF_THREADS, smem, stream>>>(args...); break;
        case 2: set_bwd<2><<<grid, SF_THREADS, smem, stream>>>(args...); break;
        case 3: set_bwd<3><<<grid, SF_THREADS, smem, stream>>>(args...); break;
        default: set_bwd<4><<<grid, SF_THREADS, smem, stream>>>(args...); break;
    }
}
template<typename... A> static void set_fwd_launch(int ncols, int grid, size_t smem, cudaStream_t stream, A... args) {
    switch (ncols) {
        case 1: set_fwd<1><<<grid, SF_THREADS, smem, stream>>>(args...); break;
        case 2: set_fwd<2><<<grid, SF_THREADS, smem, stream>>>(args...); break;
        case 3: set_fwd<3><<<grid, SF_THREADS, smem, stream>>>(args...); break;
        default: set_fwd<4><<<grid, SF_THREADS, smem, stream>>>(args...); break;
    }
}
static void set_launch_check(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("Bopl v4 ") + what + " launch: " + cudaGetErrorString(e));
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
/* Four columns per thread (cols, ldg, lda multiples of 4): float4 loads and
 * stores, the same elementwise values. */
__global__ void relu_grad_from4(const float* grad, int ldg, const float* act, int lda, float* dst, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c4 = cols >> 2;
    if (i >= rows * c4) return;
    int r = i / c4, c = (i - r * c4) << 2;
    float4 g = *(const float4*)(grad + (long)r * ldg + c), a = *(const float4*)(act + (long)r * lda + c);
    float4 o = make_float4(a.x > 0.0f ? g.x : 0.0f, a.y > 0.0f ? g.y : 0.0f, a.z > 0.0f ? g.z : 0.0f, a.w > 0.0f ? g.w : 0.0f);
    *(float4*)(dst + (long)r * cols + c) = o;
}
/* out[r][c] = bf16(grad[r][c] * (act[r][c] > 0)) for a bf16 activation: the
 * critic block's ReLU backward, written straight into the bf16 GEMM operand. */
__global__ void relu_grad_bf16(const float* grad, const bf16* act, bf16* out, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = tobf(bf(act[i]) > 0.0f ? grad[i] : 0.0f);
}
/* Four elements per thread (n a multiple of 4). */
__global__ void relu_grad_bf16x4(const float* grad, const bf16* act, bf16* out, long n4) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 g = *(const float4*)(grad + i * 4);
    uint2 av = *(const uint2*)(act + i * 4);
    const bf16* a = (const bf16*)&av;
    bf16 o[4] = {tobf(bf(a[0]) > 0.0f ? g.x : 0.0f), tobf(bf(a[1]) > 0.0f ? g.y : 0.0f), tobf(bf(a[2]) > 0.0f ? g.z : 0.0f), tobf(bf(a[3]) > 0.0f ? g.w : 0.0f)};
    *(uint2*)(out + i * 4) = *(const uint2*)o;
}
static bool ew4_ok(int cols, int ldg, int lda, const void* g, const void* a, const void* d) {
    return cols % 4 == 0 && ldg % 4 == 0 && lda % 4 == 0 && ((size_t)g & 15) == 0 && ((size_t)a & 15) == 0 && ((size_t)d & 15) == 0;
}
__global__ void add_into(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}
/* ---- LayerNorm: one warp per row, LN_ROWS rows per block, the lanes striding
 * the columns (W <= LN_MAXW), the row held in registers, the
 * two moments by warp shuffles (the one-block-per-row form spent its time in
 * __syncthreads and re-read the row three times). The forward reads the row as
 * the concatenation [x (H) | tail (P)] (the critic's zin = [y | priv]; P = 0
 * for a plain input), optionally saving it (xsave: the backward's input) and
 * writes fp32 and/or bf16 outputs (y16: the GEMM operand a separate to_bf16
 * pass produced before). The backward writes dx's first `cols` columns with
 * leading dimension ldx (the critic input LN's gradient to y is its first H
 * columns, accumulated into g_y directly) and optionally a bf16 copy. */
constexpr int LN_ROWS = 8, LN_MAXW = 512;
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
/* The lanes stride the row in units of VEC columns (VEC = 4: float4 loads and
 * stores when every width and leading dimension is a multiple of 4, which the
 * shipped shapes are; VEC = 1: the scalar form for any other width). */
template <int VEC> struct LnVec { static constexpr int PER_LANE = LN_MAXW / (32 * VEC); };
template <int VEC> __device__ __forceinline__ void ln_load(const float* p, float* v) {
    if (VEC == 4) { float4 t = *reinterpret_cast<const float4*>(p); v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w; }
    else v[0] = *p;
}
template <int VEC> __device__ __forceinline__ void ln_store(float* p, const float* v) {
    if (VEC == 4) *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    else *p = v[0];
}
template <int VEC> __device__ __forceinline__ void ln_store16(bf16* p, const float* v) {
    if (VEC == 4) {
        __nv_bfloat162 lo = __floats2bfloat162_rn(v[0], v[1]), hi = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u; u.x = *reinterpret_cast<unsigned*>(&lo); u.y = *reinterpret_cast<unsigned*>(&hi);
        *reinterpret_cast<uint2*>(p) = u;
    } else *p = tobf(v[0]);
}
template <int VEC>
__global__ void __launch_bounds__(LN_ROWS * 32) ln_forward(const float* x, int H, const float* tail, int P, int B, float* xsave,
                                                            float* y32, bf16* y16, float* mean, float* rstd) {
    constexpr int PL = LnVec<VEC>::PER_LANE;
    int W = H + P, lane = threadIdx.x & 31;
    long row = (long)blockIdx.x * LN_ROWS + (threadIdx.x >> 5);
    if (row >= B) return;
    float v[PL][VEC];
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < PL; k++) {
        int c = (lane + k * 32) * VEC;
        if (c < H) ln_load<VEC>(x + row * H + c, v[k]);
        else if (c < W) { if (tail) ln_load<VEC>(tail + row * P + c - H, v[k]); else { for (int j = 0; j < VEC; j++) v[k][j] = 0.0f; } }
        else { for (int j = 0; j < VEC; j++) v[k][j] = 0.0f; }
        #pragma unroll
        for (int j = 0; j < VEC; j++) s += v[k][j];
    }
    float mu = warp_sum(s) / W;
    float q = 0.0f;
    #pragma unroll
    for (int k = 0; k < PL; k++) {
        int c = (lane + k * 32) * VEC;
        if (c < W) {
            #pragma unroll
            for (int j = 0; j < VEC; j++) { float d = v[k][j] - mu; q += d * d; }
        }
    }
    float r = rsqrtf(warp_sum(q) / W + LN_EPS);
    #pragma unroll
    for (int k = 0; k < PL; k++) {
        int c = (lane + k * 32) * VEC;
        if (c < W) {
            float y[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; j++) y[j] = (v[k][j] - mu) * r;
            if (xsave) ln_store<VEC>(xsave + row * W + c, v[k]);
            if (y32) ln_store<VEC>(y32 + row * W + c, y);
            if (y16) ln_store16<VEC>(y16 + row * W + c, y);
        }
    }
    if (lane == 0) { mean[row] = mu; rstd[row] = r; }
}
/* dx[:, :cols] (+)= rstd * (dy - mean(dy) - xhat * mean(dy * xhat)) */
template <int VEC>
__global__ void __launch_bounds__(LN_ROWS * 32) ln_backward(const float* dy, const float* x, const float* mean, const float* rstd, int W, int B,
                                                             float* dx, int ldx, int cols, int accumulate, bf16* dx16) {
    constexpr int PL = LnVec<VEC>::PER_LANE;
    int lane = threadIdx.x & 31;
    long row = (long)blockIdx.x * LN_ROWS + (threadIdx.x >> 5);
    if (row >= B) return;
    float mu = mean[row], r = rstd[row];
    float g[PL][VEC], xh[PL][VEC];
    float s1 = 0.0f, s2 = 0.0f;
    #pragma unroll
    for (int k = 0; k < PL; k++) {
        int c = (lane + k * 32) * VEC;
        if (c < W) {
            ln_load<VEC>(dy + row * W + c, g[k]);
            ln_load<VEC>(x + row * W + c, xh[k]);
            #pragma unroll
            for (int j = 0; j < VEC; j++) { xh[k][j] = (xh[k][j] - mu) * r; s1 += g[k][j]; s2 += g[k][j] * xh[k][j]; }
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; j++) { g[k][j] = 0.0f; xh[k][j] = 0.0f; }
        }
    }
    float m1 = warp_sum(s1) / W, m2 = warp_sum(s2) / W;
    #pragma unroll
    for (int k = 0; k < PL; k++) {
        int c = (lane + k * 32) * VEC;
        if (c < cols) {
            float o[VEC];
            long i = row * ldx + c;
            if (accumulate) ln_load<VEC>(dx + i, o);
            #pragma unroll
            for (int j = 0; j < VEC; j++) { float v = r * (g[k][j] - m1 - xh[k][j] * m2); o[j] = accumulate ? o[j] + v : v; }
            ln_store<VEC>(dx + i, o);
            if (dx16) ln_store16<VEC>(dx16 + row * W + c, o);
        }
    }
}
static int ln_grid(int B, int W) {
    if (W > LN_MAXW) throw std::runtime_error("Bopl v4 LayerNorm: width " + std::to_string(W) + " is above " + std::to_string(LN_MAXW));
    return (B + LN_ROWS - 1) / LN_ROWS;
}
/* Launchers: the float4 form when the widths (and, for the backward, the
 * output's leading dimension and column count) are multiples of 4. */
static void ln_fwd(const float* x, int H, const float* tail, int P, int B, float* xsave, float* y32, bf16* y16, float* mean, float* rstd, cudaStream_t stream) {
    int grid = ln_grid(B, H + P);
    if (H % 4 == 0 && P % 4 == 0) ln_forward<4><<<grid, LN_ROWS * 32, 0, stream>>>(x, H, tail, P, B, xsave, y32, y16, mean, rstd);
    else ln_forward<1><<<grid, LN_ROWS * 32, 0, stream>>>(x, H, tail, P, B, xsave, y32, y16, mean, rstd);
}
static void ln_bwd(const float* dy, const float* x, const float* mean, const float* rstd, int W, int B, float* dx, int ldx, int cols, int accumulate, bf16* dx16, cudaStream_t stream) {
    int grid = ln_grid(B, W);
    if (W % 4 == 0 && ldx % 4 == 0 && cols % 4 == 0) ln_backward<4><<<grid, LN_ROWS * 32, 0, stream>>>(dy, x, mean, rstd, W, B, dx, ldx, cols, accumulate, dx16);
    else ln_backward<1><<<grid, LN_ROWS * 32, 0, stream>>>(dy, x, mean, rstd, W, B, dx, ldx, cols, accumulate, dx16);
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
/* ---- row cache: the observation-side packing once per update ---------------
 * Every training pass resamples its minibatch (prio_replay draws segments
 * with replacement), so one pass's packing is not the next's; but the rows
 * they draw from are the slot's learner rows, fixed for the update.
 * enc_precompute packs those once per update with the packing kernels
 * themselves (count_rows, scatter_rows into a fixed slab of rows[s] slots per
 * row, pack_flat), and the passes gather from the cache instead of walking
 * the 19 KB observation rows again: counts by row, packed set rows as 16-byte
 * copies, flat / privileged rows by row. Same source floats, same
 * conversions, same drop order (the pass's scan and cap): bit-identical to
 * packing from the observations, 4 x (count 0.42 + scatter 0.71 ms) -> 1 x
 * plus 4 gathers per update. Filled through the bound Gather (its obs is the
 * update's slot); valid until the trainer's next enc_precompute. */
struct RowCache {
    int valid, N, T, B_total, apb, primary, n_agents;
    int* agents;          /* (n_agents) learner agent ids: the cache's row order is agent position x T + tick */
    int* agent_pos;       /* (B_total) position of an agent in agents, -1 for a non-learner */
    int* counts;          /* (SETS, N) */
    int* offsets;         /* (SETS, N) the slab's row offsets r * rows[s], scatter_rows' destinations */
    bf16* slab[SETS];     /* (N * rows[s], kpad) */
    int* ids[SETS];       /* (N * rows[s]) */
    int* scratch[SETS];   /* (N * rows[s]) scatter_rows' sample output, unused (the pass writes its own) */
    float* flat;          /* (N, FLAT) */
    float* priv;          /* (N, PRIV) */
};
struct EncActs {
    Gather gather;
    RowCache cache;
    const float* saved_input;
    int B, cap[SETS];
    int act[SETS];          /* rows this pass's GEMMs / tails run over: cap, or the prepass's bucket (see enc_prepass) */
    int prepassed;          /* enc_prepass ran count_rows / scan_rows for the coming forward */
    int* host_totals;       /* pinned, mapped (SETS): the prepass's readback, written by publish_totals */
    int* host_totals_dev;   /* its device pointer */
    cudaEvent_t ready_event;   /* blocking-sync event the prepass sleeps on */
    PrecisionTensor flat_input, joined, fused, out, priv;
    FloatTensor ln_mean, ln_rstd;
    IntTensor counts, offsets, totals, overflow, ids[SETS], sample[SETS], winner[SETS];
    IntTensor w1b[SETS], w2b[SETS];                    /* bf16 weight copies ([W1 | E^T] per set, W2), 2 per int */
    IntTensor xpack[SETS], h1[SETS], h2[SETS];         /* bf16 packed rows, 2 per int */
    /* training */
    PrecisionTensor grad_fused, grad_joined, grad_flat, acc;
    IntTensor dh2[SETS];
    IntTensor dh1[SETS], w2tb[SETS];   /* set_bwd's dh1 (cap, I) and W2^T (I, D) bf16, 2 per int */
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
static IntTensor* g_v4_totals = nullptr;          /* (SETS) packed rows (clamped to the capacity) of the same forward */
static int g_v4_train_rows = 0;

static bf16* bfp(IntTensor& t) { return (bf16*)t.data; }

/* ---- row cache kernels (the passes' side) ---------------------------------- */
__device__ __forceinline__ long cache_row(const Gather& g, const int* agent_pos, int b) {
    int segment = b / g.T, tick = b - segment * g.T;
    return (long)agent_pos[g.idx[segment]] * g.T + tick;
}
__global__ void cache_counts(Gather g, const int* agent_pos, int B, int N, const int* counts_all, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * SETS) return;
    int s = i / B, b = i - s * B;
    counts[i] = counts_all[(long)s * N + cache_row(g, agent_pos, b)];
}
/* One warp per sample: its count rows from the slab into the packed matrix at the
 * pass's offset, kpad bf16 per row as int4, rows past the capacity dropped as
 * scatter_rows drops them. */
template<int S>
__global__ void cache_copy_rows(Gather g, const int* agent_pos, int B, const int* counts, const int* offsets, int cap,
                                const bf16* slab, const int* slab_ids, bf16* packed, int* ids, int* sample) {
    const Spec sp = c_spec[S];
    int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (b >= B) return;
    int n = counts[(long)S * B + b], dst = offsets[(long)S * B + b];
    if (dst + n > cap) n = cap - dst > 0 ? cap - dst : 0;
    long src = cache_row(g, agent_pos, b) * sp.rows;
    int v4s = sp.kpad / 8;   /* int4 per packed row */
    const int4* in = (const int4*)(slab + src * sp.kpad);
    int4* out = (int4*)(packed + (long)dst * sp.kpad);
    for (int t = lane; t < n * v4s; t += 32) out[t] = in[t];
    for (int j = lane; j < n; j += 32) { ids[dst + j] = slab_ids[src + j]; sample[dst + j] = b; }
}
__global__ void cache_flat(Gather g, const int* agent_pos, int B, const float4* flat_all, const float4* priv_all, float4* flat, float4* priv) {
    constexpr int F4 = FLAT / 4, P4 = PRIV / 4;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * (F4 + P4)) return;
    int b = i / (F4 + P4), c = i - b * (F4 + P4);
    long r = cache_row(g, agent_pos, b);
    if (c < F4) flat[(long)b * F4 + c] = flat_all[r * F4 + c];
    else priv[(long)b * P4 + c - F4] = priv_all[r * P4 + c - F4];
}
static void* cache_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes ? bytes : 16) != cudaSuccess) throw std::runtime_error("Bopl v4: row cache allocation failed (" + std::to_string(bytes >> 20) + " MB)");
    return p;
}
/* Pack the update's learner rows (agents with agent % apb < primary, all T
 * ticks) into the cache from the bound Gather's observations. Eager, never
 * under capture; the pass kernels bake the cache pointers into the graphs, so
 * the buffers are allocated once and the shape may not change afterwards. */
static void enc_precompute(void* activations, int apb, int primary, cudaStream_t stream) {
    auto* a = (EncActs*)activations; RowCache& c = a->cache; const Gather& g = a->gather;
    static_assert(FLAT % 4 == 0 && PRIV % 4 == 0, "cache_flat copies float4");
    if (!g.obs) throw std::runtime_error("Bopl v4: the row cache needs a bound observation gather");
    if (c.N == 0) {
        std::vector<int> agents, pos(g.B_total, -1);
        for (int agent = 0; agent < g.B_total; agent++)
            if (agent % apb < primary) { pos[agent] = (int)agents.size(); agents.push_back(agent); }
        c.n_agents = (int)agents.size(); c.T = g.T; c.B_total = g.B_total; c.apb = apb; c.primary = primary;
        c.N = c.n_agents * g.T;
        if (c.N == 0) throw std::runtime_error("Bopl v4: the row cache has no learner rows");
        c.agents = (int*)cache_alloc(agents.size() * sizeof(int));
        c.agent_pos = (int*)cache_alloc(pos.size() * sizeof(int));
        c.counts = (int*)cache_alloc((size_t)SETS * c.N * sizeof(int));
        c.offsets = (int*)cache_alloc((size_t)SETS * c.N * sizeof(int));
        std::vector<int> off((size_t)SETS * c.N);
        long bytes = 0;
        for (int s = 0; s < SETS; s++) {
            size_t slots = (size_t)c.N * SPEC[s].rows;
            for (long r = 0; r < c.N; r++) off[(size_t)s * c.N + r] = (int)(r * SPEC[s].rows);
            c.slab[s] = (bf16*)cache_alloc(slots * SPEC[s].kpad * sizeof(bf16));
            c.ids[s] = (int*)cache_alloc(slots * sizeof(int));
            c.scratch[s] = (int*)cache_alloc(slots * sizeof(int));
            bytes += slots * (SPEC[s].kpad * sizeof(bf16) + 2 * sizeof(int));
        }
        c.flat = (float*)cache_alloc((size_t)c.N * FLAT * sizeof(float));
        c.priv = (float*)cache_alloc((size_t)c.N * PRIV * sizeof(float));
        bytes += (size_t)c.N * (FLAT + PRIV) * sizeof(float);
        if (cudaMemcpy(c.agents, agents.data(), agents.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpy(c.agent_pos, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpy(c.offsets, off.data(), off.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Bopl v4: row cache upload failed");
        if (env_int("BOPL_V4_MEMLOG", 0)) fprintf(stderr, "bopl v4 memory: row cache N=%d %.1f MB\n", c.N, bytes / 1048576.0);
    } else if (c.T != g.T || c.B_total != g.B_total || c.apb != apb || c.primary != primary) {
        /* Another shape (the check harness's cases; the trainer's is fixed): drop
         * and reallocate. Graphs captured on the old buffers are stale. */
        for (void* p : {(void*)c.agents, (void*)c.agent_pos, (void*)c.counts, (void*)c.offsets, (void*)c.flat, (void*)c.priv}) cudaFree(p);
        for (int s = 0; s < SETS; s++) { cudaFree(c.slab[s]); cudaFree(c.ids[s]); cudaFree(c.scratch[s]); }
        c = RowCache{};
        enc_precompute(activations, apb, primary, stream);
        return;
    }
    Gather all{g.obs, c.agents, g.T, g.B_total};
    int N = c.N;
    count_rows<<<grid_size(N * SETS), BLOCK_SIZE, 0, stream>>>(nullptr, all, N, c.counts);
    scatter_rows<0><<<grid_size(N * 32), BLOCK_SIZE, 0, stream>>>(nullptr, all, N, c.offsets, N * SPEC[0].rows, c.slab[0], c.ids[0], c.scratch[0]);
    scatter_rows<1><<<grid_size(N * 32), BLOCK_SIZE, 0, stream>>>(nullptr, all, N, c.offsets, N * SPEC[1].rows, c.slab[1], c.ids[1], c.scratch[1]);
    scatter_rows<2><<<grid_size(N * 32), BLOCK_SIZE, 0, stream>>>(nullptr, all, N, c.offsets, N * SPEC[2].rows, c.slab[2], c.ids[2], c.scratch[2]);
    scatter_rows<3><<<grid_size(N * 32), BLOCK_SIZE, 0, stream>>>(nullptr, all, N, c.offsets, N * SPEC[3].rows, c.slab[3], c.ids[3], c.scratch[3]);
    pack_flat<<<grid_size(N * (FLAT + PRIV)), BLOCK_SIZE, 0, stream>>>(nullptr, all, N, c.flat, c.priv);
    c.valid = 1;
}

/* ---- active rows ------------------------------------------------------------
 * The set GEMMs' M is baked into the captured train graph, but a batch packs
 * far fewer rows than the capacity (the budget sizes the buffers for the
 * worst batch; the bench workload needs 2 / 6 / 11 / 2 rows per sample against
 * 6 / 12 / 20 / 8), so most of the set MLP work was on the zero tail. The
 * trainer calls enc_prepass before the graph launch / capture: it runs
 * count_rows + scan_rows on the stream, reads the totals back (one host
 * sync) and rounds each set's total up to a granule (cap / ACTIVE_BUCKETS,
 * a TILE multiple) -> act[s]; the pass's GEMMs, tails and masks run over
 * act[s] rows and the bucket tuple is the trainer's graph key (one captured
 * graph per tuple seen). Rows [total, act) are zero as before, rows
 * [act, cap) are never touched; the drop semantics (scatter's cap) are
 * unchanged. The GEMM shapes change with the bucket (tile order, split-K),
 * so fp32 sums reassociate: the check_v4 class, not bit-identical.
 * BOPL4_ACTIVE_ROWS=0 restores the fixed capacity. */
constexpr int ACTIVE_BUCKETS = 16;
static int active_granule(int cap) { int g = (cap / ACTIVE_BUCKETS + TILE - 1) / TILE * TILE; return g < TILE ? TILE : g; }
__global__ void publish_totals(const int* totals, int* host) { if (threadIdx.x < SETS) host[threadIdx.x] = totals[threadIdx.x]; }
static void count_and_scan(EncActs* a, PrecisionTensor input, cudaStream_t stream) {
    int B = a->B;
    if (a->cache.valid) cache_counts<<<grid_size(B * SETS), BLOCK_SIZE, 0, stream>>>(a->gather, a->cache.agent_pos, B, a->cache.N, a->cache.counts, a->counts.data);
    else count_rows<<<grid_size(B * SETS), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->counts.data);
    scan_rows<<<SETS, 1024, 0, stream>>>(a->counts.data, B, make_int4(a->cap[0], a->cap[1], a->cap[2], a->cap[3]), a->offsets.data, a->totals.data, a->overflow.data);
}
/* Returns the graph key (>= 1) of the coming forward. A forward without a
 * prepass (the harness) runs at act = cap with count / scan inside it. Not
 * for use under capture. */
static int enc_prepass(void* activations, PrecisionTensor input, cudaStream_t stream) {
    auto* a = (EncActs*)activations;
    if (!a->host_totals) {
        if (cudaHostAlloc((void**)&a->host_totals, SETS * sizeof(int), cudaHostAllocMapped) != cudaSuccess
            || cudaHostGetDevicePointer((void**)&a->host_totals_dev, a->host_totals, 0) != cudaSuccess)
            throw std::runtime_error("Bopl v4: pinned totals allocation failed");
    }
    count_and_scan(a, input, stream);
    /* The readback waits behind the previous pass's graph (~25 ms): a kernel
     * writes the totals into mapped host memory (a cudaMemcpyAsync queued that
     * long on the copy engine stalled the buffer streams' action copies behind
     * it: rollout 0.089 -> 0.125 s) and the thread sleeps on a blocking event,
     * as the trainer does between updates. */
    if (!a->ready_event && cudaEventCreateWithFlags(&a->ready_event, cudaEventDisableTiming | cudaEventBlockingSync) != cudaSuccess)
        throw std::runtime_error("Bopl v4: prepass event creation failed");
    publish_totals<<<1, 32, 0, stream>>>(a->totals.data, a->host_totals_dev);
    if (cudaEventRecord(a->ready_event, stream) != cudaSuccess || cudaEventSynchronize(a->ready_event) != cudaSuccess)
        throw std::runtime_error("Bopl v4: prepass readback failed");
    int key = 0;
    for (int s = 0; s < SETS; s++) {
        int g = active_granule(a->cap[s]), act = (a->host_totals[s] + g - 1) / g * g;
        if (act < g) act = g;
        if (act > a->cap[s]) act = a->cap[s];
        a->act[s] = act;
        key = key * (ACTIVE_BUCKETS + 1) + (act + g - 1) / g;
    }
    a->prepassed = 1;
    return key + 1;
}

static PrecisionTensor enc_forward(void* weights, void* activations, PrecisionTensor input, cudaStream_t stream) {
    auto* w = (EncWeights*)weights; auto* a = (EncActs*)activations;
    int B = a->B, H = w->hidden, D = w->D, I = w->I, FD = w->flat_dim, LDJ = FD + SETS * 2 * D;
    a->saved_input = input.data;
    /* bf16 weights */
    for (int s = 0; s < SETS; s++) {
        pack_w1<<<grid_size(I * SPEC[s].kpad), BLOCK_SIZE, 0, stream>>>(w->w1[s].data, SPEC[s].features, SPEC[s].id_col >= 0 ? w->embedding.data : nullptr, bfp(a->w1b[s]), I, SPEC[s].kpad);
        to_bf16<<<grid_size(D * I), BLOCK_SIZE, 0, stream>>>(w->w2[s].data, bfp(a->w2b[s]), D * I);
    }
    /* flat block and privileged block */
    const RowCache& c = a->cache;
    if (c.valid) cache_flat<<<grid_size(B * (FLAT + PRIV) / 4), BLOCK_SIZE, 0, stream>>>(a->gather, c.agent_pos, B, (const float4*)c.flat, (const float4*)c.priv, (float4*)a->flat_input.data, (float4*)a->priv.data);
    else pack_flat<<<grid_size(B * (FLAT + PRIV)), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->flat_input.data, a->priv.data);
    g_v4_privileged = a->priv.data; g_v4_privileged_rows = B;
    prof_mark("enc.pack_flat", stream);
    /* joined[:, :FD] = ReLU(flat_input W_flat^T): the ReLU in the epilogue, written at the joined stride */
    lt_gemm32(false, true, B, FD, FLAT, a->flat_input.data, FLAT, w->flat.data, FLAT, nullptr, a->joined.data, LDJ, CUBLASLT_EPILOGUE_RELU, stream);
    prof_mark("enc.flat_gemm", stream);
    /* sets */
    if (!a->prepassed) { count_and_scan(a, input, stream); for (int s = 0; s < SETS; s++) a->act[s] = a->cap[s]; }
    a->prepassed = 0;
    if (c.valid) {
        cache_copy_rows<0><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(a->gather, c.agent_pos, B, a->counts.data, a->offsets.data, a->cap[0], c.slab[0], c.ids[0], bfp(a->xpack[0]), a->ids[0].data, a->sample[0].data);
        cache_copy_rows<1><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(a->gather, c.agent_pos, B, a->counts.data, a->offsets.data, a->cap[1], c.slab[1], c.ids[1], bfp(a->xpack[1]), a->ids[1].data, a->sample[1].data);
        cache_copy_rows<2><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(a->gather, c.agent_pos, B, a->counts.data, a->offsets.data, a->cap[2], c.slab[2], c.ids[2], bfp(a->xpack[2]), a->ids[2].data, a->sample[2].data);
        cache_copy_rows<3><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(a->gather, c.agent_pos, B, a->counts.data, a->offsets.data, a->cap[3], c.slab[3], c.ids[3], bfp(a->xpack[3]), a->ids[3].data, a->sample[3].data);
    } else {
        scatter_rows<0><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[0], bfp(a->xpack[0]), a->ids[0].data, a->sample[0].data);
        scatter_rows<1><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[1], bfp(a->xpack[1]), a->ids[1].data, a->sample[1].data);
        scatter_rows<2><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[2], bfp(a->xpack[2]), a->ids[2].data, a->sample[2].data);
        scatter_rows<3><<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(input.data, a->gather, B, a->offsets.data, a->cap[3], bfp(a->xpack[3]), a->ids[3].data, a->sample[3].data);
    }
    prof_mark("enc.count_scan_scatter", stream);
    for (int s = 0; s < SETS; s++) {
        const int* total = a->totals.data + s;
        int K = SPEC[s].kpad, act = a->act[s];
        const int col0 = FD + s * 2 * D;
        if (set_fwd_ok(I, D, LDJ, col0)) {
            /* the fused tile kernel: h1, h2 and the pool; xpack's and h1's tails zeroed for the dW2 / dW1 GEMMs over the active rows */
            zero_tail<<<256, BLOCK_SIZE, 0, stream>>>(bfp(a->xpack[s]), K, bfp(a->h1[s]), I, a->totals.data, s, act);
            set_fwd_launch(D / SF_THREADS, (B + SF_SAMPLES - 1) / SF_SAMPLES, set_fwd_smem(I, D), stream,
                bfp(a->xpack[s]), K, bfp(a->w1b[s]), I, bfp(a->w2b[s]), D, a->offsets.data + (long)s * B, a->counts.data + (long)s * B,
                a->totals.data, s, B, bfp(a->h1[s]), bfp(a->h2[s]), a->joined.data, LDJ, col0, a->winner[s].data, w->mean_pool);
            set_launch_check("set_fwd");
            if (g_set_stage.forward)
                g_set_stage.forward(s, bfp(a->h2[s]), a->sample[s].data, a->offsets.data + (long)s * B, a->counts.data + (long)s * B, total, B, D, stream);
            prof_mark("enc.set_mlp_pool", stream);
            continue;
        }
        /* h1 = ReLU([X | 1_id] [W1 | E^T]^T) = ReLU(W1 x + E[id]) over the active rows */
        zero_tail<<<256, BLOCK_SIZE, 0, stream>>>(bfp(a->xpack[s]), K, nullptr, 0, a->totals.data, s, act);
        /* the backward's DRELU bitmask when registered (training) and the pass is wide enough for RELU_AUX to round like RELU */
        void* mask = act >= LT_AUX_MIN_ROWS ? a->relu_mask[s].data : nullptr;
        lt_gemm(false, true, act, I, K, bfp(a->xpack[s]), K, bfp(a->w1b[s]), K, nullptr, bfp(a->h1[s]), I, true,
                mask ? CUBLASLT_EPILOGUE_RELU_AUX : CUBLASLT_EPILOGUE_RELU, stream, mask);
        /* h2 = ReLU(h1 W2^T); W2 stored (D, I) */
        lt_gemm(false, true, act, D, I, bfp(a->h1[s]), I, bfp(a->w2b[s]), I, nullptr, bfp(a->h2[s]), D, true, CUBLASLT_EPILOGUE_RELU, stream);
        if (g_set_stage.forward)
            g_set_stage.forward(s, bfp(a->h2[s]), a->sample[s].data, a->offsets.data + (long)s * B, a->counts.data + (long)s * B, total, B, D, stream);
        if (pool4_ok(D, LDJ, FD + s * 2 * D))
            pool_rows4<<<pool4_grid(D, B), POOL_THREADS, 0, stream>>>(bfp(a->h2[s]), D, a->offsets.data + (long)s * B, a->counts.data + (long)s * B, s,
                                                                     a->totals.data, a->joined.data, LDJ, FD + s * 2 * D, a->winner[s].data, w->mean_pool, B);
        else
            pool_rows<<<B, D, 0, stream>>>(bfp(a->h2[s]), D, a->offsets.data + (long)s * B, a->counts.data + (long)s * B, a->cap[s], s,
                                           a->totals.data, a->joined.data, LDJ, FD + s * 2 * D, a->winner[s].data, w->mean_pool);
        prof_mark("enc.set_mlp_pool", stream);
    }
    puf_mm(&a->joined, &w->fusion, &a->fused, stream);
    ln_fwd(a->fused.data, H, nullptr, 0, B, nullptr, a->out.data, nullptr, a->ln_mean.data, a->ln_rstd.data, stream);
    prof_mark("enc.fusion_ln", stream);
    return a->out;
}

/* dW1 (I, K) = dh1^T X into the scratch, then keep the first `features` of
 * the padded K; the embedding gradient is dW1's one-hot columns (entity set
 * first (s=2), slot adds). */
static void set_wgrad_w1(EncActs* a, int s, const bf16* dh1, int K, int act, int I, cudaStream_t stream) {
    lt_gemm(true, false, I, K, act, dh1, I, bfp(a->xpack[s]), K, nullptr, a->acc.data, K, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
    reduce_partials<<<grid_size(I * SPEC[s].features), BLOCK_SIZE, 0, stream>>>(a->acc.data, 1, 0, I, K, SPEC[s].features, a->wgrad_w1[s].data, 0);
    prof_mark("bwd.enc.dW1", stream);
    if (SPEC[s].id_col >= 0) {
        embed_grad_from_w1<<<grid_size(IDS * I), BLOCK_SIZE, 0, stream>>>(a->acc.data, K, SPEC[s].features, I, a->wgrad_embedding.data, s != 2);
        prof_mark("bwd.enc.embedding", stream);
    }
}
static void enc_backward(void* weights, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    auto* w = (EncWeights*)weights; auto* a = (EncActs*)activations;
    int B = a->B, H = w->hidden, D = w->D, I = w->I, FD = w->flat_dim, LDJ = FD + SETS * 2 * D;
    ln_bwd(grad.data, a->fused.data, a->ln_mean.data, a->ln_rstd.data, H, B, a->grad_fused.data, H, H, 0, nullptr, stream);
    puf_mm_tn(&a->grad_fused, &a->joined, &a->wgrad_fusion, stream);
    puf_mm_nn(&a->grad_fused, &w->fusion, &a->grad_joined, stream);
    if (ew4_ok(FD, LDJ, LDJ, a->grad_joined.data, a->joined.data, a->grad_flat.data))
        relu_grad_from4<<<grid_size(B * FD / 4), BLOCK_SIZE, 0, stream>>>(a->grad_joined.data, LDJ, a->joined.data, LDJ, a->grad_flat.data, B, FD);
    else relu_grad_from<<<grid_size(B * FD), BLOCK_SIZE, 0, stream>>>(a->grad_joined.data, LDJ, a->joined.data, LDJ, a->grad_flat.data, B, FD);
    puf_mm_tn(&a->grad_flat, &a->flat_input, &a->wgrad_flat, stream);
    prof_mark("bwd.enc.fusion_flat", stream);
    for (int s = SETS - 1; s >= 0; s--) {
        const int* total = a->totals.data + s;
        int K = SPEC[s].kpad, act = a->act[s];
        if (set_fwd_ok(I, D, LDJ, FD + s * 2 * D)) {
            to_bf16_t<<<grid_size(D * I), BLOCK_SIZE, 0, stream>>>(w->w2[s].data, bfp(a->w2tb[s]), D, I);
            zero_tail<<<256, BLOCK_SIZE, 0, stream>>>(bfp(a->dh2[s]), D, bfp(a->dh1[s]), I, a->totals.data, s, act);
            set_bwd_launch(D / SF_THREADS, (B + SF_SAMPLES - 1) / SF_SAMPLES, set_bwd_smem(I, D), stream,
                bfp(a->h1[s]), bfp(a->h2[s]), bfp(a->w2tb[s]), I, D, a->offsets.data + (long)s * B, a->counts.data + (long)s * B,
                a->totals.data, s, B, a->grad_joined.data, LDJ, FD + s * 2 * D, a->winner[s].data, w->mean_pool, bfp(a->dh2[s]), bfp(a->dh1[s]));
            set_launch_check("set_bwd");
            if (g_set_stage.backward)
                g_set_stage.backward(s, bfp(a->h2[s]), bfp(a->dh2[s]), a->sample[s].data, a->offsets.data + (long)s * B, a->counts.data + (long)s * B, total, B, D, stream);
            prof_mark("bwd.enc.unpool_dh1", stream);
            bf16* dh1 = bfp(a->dh1[s]);
            /* dW2 (D, I) = dh2^T h1 over the active rows (zero rows past the count) */
            lt_gemm(true, false, D, I, act, bfp(a->dh2[s]), D, bfp(a->h1[s]), I, nullptr, a->wgrad_w2[s].data, I, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
            prof_mark("bwd.enc.dW2", stream);
            set_wgrad_w1(a, s, dh1, K, act, I, stream);
            continue;
        }
        if (pool4_ok(D, LDJ, FD + s * 2 * D))
            unpool_rows4<<<pool4_grid(D, B), POOL_THREADS, 0, stream>>>(bfp(a->h2[s]), bfp(a->dh2[s]), D, a->offsets.data + (long)s * B, a->counts.data + (long)s * B,
                                                                       a->totals.data, s, a->grad_joined.data, LDJ, FD + s * 2 * D, a->winner[s].data, w->mean_pool, B);
        else
            unpool_rows<<<B, D, 0, stream>>>(bfp(a->h2[s]), bfp(a->dh2[s]), D, a->offsets.data + (long)s * B, a->counts.data + (long)s * B,
                                             a->totals.data, s, a->grad_joined.data, LDJ, FD + s * 2 * D, a->winner[s].data, w->mean_pool);
        zero_tail<<<256, BLOCK_SIZE, 0, stream>>>(bfp(a->dh2[s]), D, nullptr, 0, a->totals.data, s, act);
        if (g_set_stage.backward)
            g_set_stage.backward(s, bfp(a->h2[s]), bfp(a->dh2[s]), a->sample[s].data, a->offsets.data + (long)s * B, a->counts.data + (long)s * B, total, B, D, stream);
        prof_mark("bwd.enc.unpool", stream);
        /* dW2 (D, I) = dh2^T h1 over the active rows (zero rows past the count) */
        lt_gemm(true, false, D, I, act, bfp(a->dh2[s]), D, bfp(a->h1[s]), I, nullptr, a->wgrad_w2[s].data, I, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        prof_mark("bwd.enc.dW2", stream);
        /* dh1 = (dh2 W2) * ReLU'(h1), written over h2 (no longer needed) */
        bf16* dh1 = bfp(a->h2[s]);
        /* as the forward decided: the fused tile kernel writes no bitmask (mask_rows below) */
        void* mask = (!set_fwd_ok(I, D, LDJ, FD + s * 2 * D) && act >= LT_AUX_MIN_ROWS) ? a->relu_mask[s].data : nullptr;
        lt_gemm(false, false, act, I, D, bfp(a->dh2[s]), D, bfp(a->w2b[s]), I, nullptr, dh1, I, true,
                mask ? CUBLASLT_EPILOGUE_DRELU : CUBLASLT_EPILOGUE_DEFAULT, stream, mask);
        if (!mask) mask_rows<<<grid_long((long)act * I), BLOCK_SIZE, 0, stream>>>(dh1, bfp(a->h1[s]), (long)act * I);
        prof_mark("bwd.enc.dh1", stream);
        set_wgrad_w1(a, s, dh1, K, act, I, stream);
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
    a->flat_input = {.shape = {B, FLAT}};
    a->joined = {.shape = {B, FD + SETS * 2 * D}}; a->fused = {.shape = {B, H}}; a->out = {.shape = {B, H}};
    a->priv = {.shape = {B, PRIV}};
    for (auto* t : {&a->flat_input, &a->joined, &a->fused, &a->out, &a->priv}) alloc_register(alloc, t);
    a->ln_mean = {.shape = {B}}; a->ln_rstd = {.shape = {B}};
    alloc_register(alloc, &a->ln_mean); alloc_register(alloc, &a->ln_rstd);
    a->counts = {.shape = {SETS, B}}; a->offsets = {.shape = {SETS, B}}; a->totals = {.shape = {SETS}}; a->overflow = {.shape = {SETS}};
    for (auto* t : {&a->counts, &a->offsets, &a->totals, &a->overflow}) alloc_register(alloc, t);
    for (int s = 0; s < SETS; s++) {
        int cap = (B * w->budget[s] + TILE - 1) / TILE * TILE;
        if (cap > B * SPEC[s].rows) cap = (B * SPEC[s].rows + TILE - 1) / TILE * TILE;
        a->cap[s] = a->act[s] = cap;
        a->ids[s] = {.shape = {cap}}; a->sample[s] = {.shape = {cap}}; a->winner[s] = {.shape = {B, D}};
        a->w1b[s] = {.shape = {I, SPEC[s].kpad / 2}}; a->w2b[s] = {.shape = {D, I / 2}};
        a->xpack[s] = {.shape = {cap, SPEC[s].kpad / 2}}; a->h1[s] = {.shape = {cap, I / 2}}; a->h2[s] = {.shape = {cap, D / 2}};
        for (auto* t : {&a->ids[s], &a->sample[s], &a->winner[s], &a->w1b[s], &a->w2b[s], &a->xpack[s], &a->h1[s], &a->h2[s]}) alloc_register(alloc, t);
        if (g_set_stage.reg) g_set_stage.reg(alloc, s, cap, D);
    }
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
    long acc = 0;   /* dW1's fp32 (I, kpad) */
    for (int s = 0; s < SETS; s++) {
        a->dh2[s] = {.shape = {a->cap[s], D / 2}}; alloc_register(acts, &a->dh2[s]);
        if (set_fwd_ok(I, D, FD + SETS * 2 * D, FD + s * 2 * D)) {
            a->dh1[s] = {.shape = {a->cap[s], I / 2}}; alloc_register(acts, &a->dh1[s]);
            a->w2tb[s] = {.shape = {I, D / 2}}; alloc_register(acts, &a->w2tb[s]);
        }
        if (I % 128 == 0 && a->cap[s] >= LT_AUX_MIN_ROWS) {
            a->relu_mask[s] = {.shape = {a->cap[s], I / 32}}; alloc_register(acts, &a->relu_mask[s]);
        }
        acc = std::max(acc, (long)I * SPEC[s].kpad);
    }
    a->acc = {.shape = {acc}}; alloc_register(acts, &a->acc);
    /* gradients in reg_params order */
    for (int s = 0; s < SETS; s++) {
        a->wgrad_w1[s] = {.shape = {I, SPEC[s].features}}; alloc_register(grads, &a->wgrad_w1[s]);
        a->wgrad_w2[s] = {.shape = {D, I}}; alloc_register(grads, &a->wgrad_w2[s]);
    }
    a->wgrad_embedding = {.shape = {IDS, I}}; alloc_register(grads, &a->wgrad_embedding);
    a->wgrad_flat = {.shape = {FD, FLAT}}; alloc_register(grads, &a->wgrad_flat);
    a->wgrad_fusion = {.shape = {H, FD + SETS * 2 * D}}; alloc_register(grads, &a->wgrad_fusion);
    g_v4_overflow = &a->overflow; g_v4_totals = &a->totals; g_v4_train_rows = B;
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
    auto* a = (EncActs*)activations;
    a->gather = Gather{rollout_obs, idx, T, B_total};
    a->cache.valid = 0;   /* filled for this binding by the next enc_precompute */
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
    PrecisionTensor ln1, f, y, zin, z, z2, ln3, value;   /* lnin, ln2 and bblk exist only as the bf16 GEMM operands below */
    FloatTensor m1, r1, min_, rin, m2, r2, m3, r3;
    /* bf16 operands of the critic GEMMs (cublasLt, fp32 accumulation): the
     * weights, the forward inputs (reused by the weight gradients) and the
     * output gradients; 2 per int */
    IntTensor cin_b, b1_b, b2_b, lnin_b, ln2_b, bblk_b, g_z_b, g_b_b, g_z2_b;
    /* The critic block's ReLU bitmask (B, C / 32), written by the forward
     * GEMM's RELU_AUX epilogue and applied by the backward's DRELU epilogue
     * (training, B >= LT_AUX_MIN_ROWS); without it the ReLU backward is a
     * kernel. Only the bf16 GEMMs: for fp32 operands cublasLt runs RELU_AUX
     * and DRELU as a separate epilogue kernel (measured: 0.38 / 0.45 ms per
     * call on the FFN, slower than the elementwise kernels they replace), so
     * the FFN keeps the plain fused RELU forward and a kernel backward. */
    IntTensor b1_mask;
    /* training */
    PrecisionTensor g_y, g_f, g_ln1, gv, g_ln3, g_b, g_ln2, g_z, g_lnin;
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
        ln_fwd(h.data, H, nullptr, 0, B, nullptr, a->ln1.data, nullptr, a->m1.data, a->r1.data, stream);
        int F = w->ffn;
        /* f = ReLU(ln1 W_in^T) with the ReLU in the epilogue; y = h + f W_out^T with h as the addend */
        lt_gemm32(false, true, B, F, H, a->ln1.data, H, w->ffn_in.data, H, nullptr, a->f.data, F, CUBLASLT_EPILOGUE_RELU, stream);
        lt_gemm32(false, true, B, H, F, a->f.data, F, w->ffn_out.data, F, h.data, a->y.data, H, CUBLASLT_EPILOGUE_DEFAULT, stream);
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
        /* zin = [y | priv], saved for the backward; LN(zin) straight to its bf16 GEMM operand.
         * The critic GEMMs run bf16 on tensor cores (fp32 accumulation, fp32 outputs). */
        ln_fwd(y->data, H, priv, P, B, a->zin.data, nullptr, bfp(a->lnin_b), a->min_.data, a->rin.data, stream);
        to_bf16<<<grid_size(C * W), BLOCK_SIZE, 0, stream>>>(w->critic_in.data, bfp(a->cin_b), C * W);
        to_bf16<<<grid_size(C * C), BLOCK_SIZE, 0, stream>>>(w->critic_b1.data, bfp(a->b1_b), C * C);
        to_bf16<<<grid_size(C * C), BLOCK_SIZE, 0, stream>>>(w->critic_b2.data, bfp(a->b2_b), C * C);
        lt_gemm(false, true, B, C, W, bfp(a->lnin_b), W, bfp(a->cin_b), W, nullptr, a->z.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        prof_mark("dec.critic_in", stream);
        ln_fwd(a->z.data, C, nullptr, 0, B, nullptr, nullptr, bfp(a->ln2_b), a->m2.data, a->r2.data, stream);
        /* bblk = ReLU(ln2 B1^T) written bf16 by the epilogue (the next GEMM's operand), its bitmask for the
         * backward's DRELU when registered; z2 = z + bblk B2^T with z as the addend */
        void* b1mask = a->b1_mask.data;
        lt_gemm(false, true, B, C, C, bfp(a->ln2_b), C, bfp(a->b1_b), C, nullptr, bfp(a->bblk_b), C, true,
                b1mask ? CUBLASLT_EPILOGUE_RELU_AUX : CUBLASLT_EPILOGUE_RELU, stream, b1mask);
        lt_gemm(false, true, B, C, C, bfp(a->bblk_b), C, bfp(a->b2_b), C, a->z.data, a->z2.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        prof_mark("dec.critic_block", stream);
        ln_fwd(a->z2.data, C, nullptr, 0, B, nullptr, a->ln3.data, nullptr, a->m3.data, a->r3.data, stream);
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
        /* g_z2 = LN3 backward, into g_z (the residual: g_z = g_z2 + LN2 backward below) and its bf16 copy for the GEMMs */
        ln_bwd(a->g_ln3.data, a->z2.data, a->m3.data, a->r3.data, C, B, a->g_z.data, C, C, 0, bfp(a->g_z2_b), stream);
        prof_mark("bwd.dec.ln3_bwd", stream);
        /* bf16 operands as in the forward: dW = dY^T X, dX = dY W */
        lt_gemm(true, false, C, C, B, bfp(a->g_z2_b), C, bfp(a->bblk_b), C, nullptr, a->wgrad_b2.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        if (a->b1_mask.data)   /* g_b_b = (g_z2 B2) * ReLU'(bblk) by the DRELU epilogue, bf16 */
            lt_gemm(false, false, B, C, C, bfp(a->g_z2_b), C, bfp(a->b2_b), C, nullptr, bfp(a->g_b_b), C, true, CUBLASLT_EPILOGUE_DRELU, stream, a->b1_mask.data);
        else {
            lt_gemm(false, false, B, C, C, bfp(a->g_z2_b), C, bfp(a->b2_b), C, nullptr, a->g_b.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
            if (ew4_ok(C, C, C, a->g_b.data, bfp(a->bblk_b), bfp(a->g_b_b)))
                relu_grad_bf16x4<<<grid_long((long)B * C / 4), BLOCK_SIZE, 0, stream>>>(a->g_b.data, bfp(a->bblk_b), bfp(a->g_b_b), (long)B * C / 4);
            else relu_grad_bf16<<<grid_long((long)B * C), BLOCK_SIZE, 0, stream>>>(a->g_b.data, bfp(a->bblk_b), bfp(a->g_b_b), (long)B * C);
        }
        lt_gemm(true, false, C, C, B, bfp(a->g_b_b), C, bfp(a->ln2_b), C, nullptr, a->wgrad_b1.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        lt_gemm(false, false, B, C, C, bfp(a->g_b_b), C, bfp(a->b1_b), C, nullptr, a->g_ln2.data, C, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        prof_mark("bwd.dec.critic_block_gemms", stream);
        ln_bwd(a->g_ln2.data, a->z.data, a->m2.data, a->r2.data, C, B, a->g_z.data, C, C, 1, bfp(a->g_z_b), stream);
        prof_mark("bwd.dec.ln2_bwd", stream);
        lt_gemm(true, false, C, W, B, bfp(a->g_z_b), C, bfp(a->lnin_b), W, nullptr, a->wgrad_critic_in.data, W, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        lt_gemm(false, false, B, W, C, bfp(a->g_z_b), C, bfp(a->cin_b), W, nullptr, a->g_lnin.data, W, false, CUBLASLT_EPILOGUE_DEFAULT, stream);
        /* g_y += LN_in backward [:, :H] (the privileged columns' gradient goes nowhere; detached: nothing flows to y) */
        if (!w->detach)
            ln_bwd(a->g_lnin.data, a->zin.data, a->min_.data, a->rin.data, W, B, a->g_y.data, H, H, 1, nullptr, stream);
        prof_mark("bwd.dec.critic_in", stream);
    }
    /* FFN: grad_h = g_y + LN_bwd(W_i^T (ReLU' * W_o^T g_y)) */
    puf_copy(&a->base.grad_input, &a->g_y, stream);
    if (w->ffn) {
        int F = w->ffn;
        puf_mm_tn(&a->g_y, &a->f, &a->wgrad_ffn_out, stream);
        puf_mm_nn(&a->g_y, &w->ffn_out, &a->g_f, stream);
        if (ew4_ok(F, F, F, a->g_f.data, a->f.data, a->g_f.data))
            relu_grad_from4<<<grid_size(B * F / 4), BLOCK_SIZE, 0, stream>>>(a->g_f.data, F, a->f.data, F, a->g_f.data, B, F);
        else relu_grad_from<<<grid_size(B * F), BLOCK_SIZE, 0, stream>>>(a->g_f.data, F, a->f.data, F, a->g_f.data, B, F);
        puf_mm_tn(&a->g_f, &a->ln1, &a->wgrad_ffn_in, stream);
        puf_mm_nn(&a->g_f, &w->ffn_in, &a->g_ln1, stream);
        ln_bwd(a->g_ln1.data, a->base.saved_input.data, a->m1.data, a->r1.data, H, B, a->base.grad_input.data, H, H, 1, nullptr, stream);
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
        a->zin = {.shape = {B, W}}; a->z = {.shape = {B, C}}; a->z2 = {.shape = {B, C}}; a->ln3 = {.shape = {B, C}}; a->value = {.shape = {B, 1}};
        for (auto* t : {&a->zin, &a->z, &a->z2, &a->ln3, &a->value}) alloc_register(alloc, t);
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
    /* the critic block's ReLU bitmask (RELU_AUX rounds like RELU from LT_AUX_MIN_ROWS up; the width must be a multiple of 128 bits) */
    if (C && B >= LT_AUX_MIN_ROWS && C % 128 == 0) { a->b1_mask = {.shape = {B, C / 32}}; alloc_register(acts, &a->b1_mask); }
    if (C) {
        a->gv = {.shape = {B, 1}}; a->g_ln3 = {.shape = {B, C}};
        a->g_ln2 = {.shape = {B, C}}; a->g_z = {.shape = {B, C}}; a->g_lnin = {.shape = {B, W}};
        for (auto* t : {&a->gv, &a->g_ln3, &a->g_ln2, &a->g_z, &a->g_lnin}) alloc_register(acts, t);
        if (!a->b1_mask.data) { a->g_b = {.shape = {B, C}}; alloc_register(acts, &a->g_b); }   /* the fp32 ReLU-backward input when no DRELU */
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
/* Before a training pass (never under capture): pack-count the minibatch and
 * return the pass's graph key (>= 1; 0 = fixed capacity, see enc_prepass).
 * The trainer keeps one captured train graph per key. */
static int bopl_v4_train_prepass(void* activations, PrecisionTensor obs, cudaStream_t stream) {
    return v4::enc_prepass(activations, obs, stream);
}
/* Once per update, before its passes (never under capture): pack the slot's
 * learner rows (agent % apb < primary) into the row cache the passes gather
 * from (v4::enc_precompute). Needs the gather bound to the slot. */
static void bopl_v4_train_precompute(void* activations, int apb, int primary, cudaStream_t stream) {
    v4::enc_precompute(activations, apb, primary, stream);
}
/* The last training forward's packing, for the trainer's log: out[0..3] the
 * dropped rows per set, out[4..7] the rows packed per set (clamped to the
 * capacity: with no drops, the batch's actual rows, so packed / batch rows is
 * the row budget the batch needed), out[8] the batch rows. */
int bopl_v4_overflow(int* out) {
    if (!v4::g_v4_overflow || !v4::g_v4_overflow->data) return -1;
    cudaMemcpy(out, v4::g_v4_overflow->data, v4::SETS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out + v4::SETS, v4::g_v4_totals->data, v4::SETS * sizeof(int), cudaMemcpyDeviceToHost);
    out[2 * v4::SETS] = v4::g_v4_train_rows;
    return 2 * v4::SETS + 1;
}
#endif
