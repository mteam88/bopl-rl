// Bopl batched rollout forward for policy v4 (POLICY_V4_PLAN.md section 3,
// bopl_v4.cu is the per-policy trainer path and the parity reference).
//
// Included by bopl_rollout.cu. The v3 chain's machinery is reused unchanged:
// the row groups [learner | bank 0 | ...] (BrGroups), the grouped small-M
// GEMM (br_gemm, weights in TF32 fragment order or fp32 transposed, per group),
// the MinGRU gate with the reset fused in (br_gate), the warp sampler
// (br_sample) and the H2D / D2H copy plan. What differs is the encoder's set
// stage and the decoder (FFN, wide critic), which become row-wise kernels:
//
//   br4_sets     grid (rows, 4 sets), block per (sample, set): the set's valid
//                rows through relu(W1 x + E[id]) -> relu(W2 h1) in bf16-rounded
//                fp32 (the trainer's WMMA path rounds the same operands to
//                bf16), max || mean pooled into joined[:, FD + 2Ds ..]. The set
//                0 block also copies the obs row, reward and terminal into the
//                rollout tensors.
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

struct Br4SetWeights {   // per group g and set s, as stored: W1 (I, features), W2 (D, I), E (IDS, I)
    const float* w1[BR_MAX_GROUPS][4];
    const float* w2[BR_MAX_GROUPS][4];
    const float* emb[BR_MAX_GROUPS];
};
struct Br4SetsArgs {
    const float* obs;          // (rows, STRIDE) live rows of this buffer
    const float* rewards; const float* terminals;
    float* rollout_obs; float* rollout_rewards; float* rollout_terminals;
    float* joined; int ldj, col0;   // pooled channels at joined[row, col0 + 2 D s + {0, D}]
    int D, I, mean_pool, rows;
    BrGroups g;
    Br4SetWeights w;
};
#define BR4_SET_THREADS 256
#define BR4_MAX_SET_ROWS 48   // the largest set
static_assert(BOPL4_PLAYERS <= BR4_MAX_SET_ROWS && BOPL4_SLOTS <= BR4_MAX_SET_ROWS && BOPL4_ENTITIES <= BR4_MAX_SET_ROWS && BOPL4_SLIMES <= BR4_MAX_SET_ROWS, "set rows");

__device__ __forceinline__ float br4_bf(float v) { return __bfloat162float(__float2bfloat16(v)); }

// Dynamic shared: valid[48] ints, h1 (n <= 48, I) bf16, W2 (D, I/2 + 1) bf162 (padded row: conflict-free column reads).
static size_t br4_sets_smem(int D, int I) {
    return BR4_MAX_SET_ROWS * sizeof(int) + (size_t)BR4_MAX_SET_ROWS * I * sizeof(__nv_bfloat16) + (size_t)D * (I / 2 + 1) * sizeof(__nv_bfloat162);
}
__global__ void __launch_bounds__(BR4_SET_THREADS) br4_sets(Br4SetsArgs a) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int row = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    const int I = a.I, D = a.D;
    int* valid = (int*)smem;
    __nv_bfloat16* h1 = (__nv_bfloat16*)(valid + BR4_MAX_SET_ROWS);
    __nv_bfloat162* W2s = (__nv_bfloat162*)(h1 + BR4_MAX_SET_ROWS * I);
    const int ldw = I / 2 + 1;
    const int grp = br_group_of(a.g, row);
    const v4::Spec sp = v4::c_spec[s];
    const float* obs = a.obs + (long)row * BOPL_OBS_STRIDE;
    if (s == 0) {  // the row's copy into the rollout tensors (the old cast kernels)
        if (tid == 0) { a.rollout_rewards[row] = a.rewards[row]; a.rollout_terminals[row] = a.terminals[row]; }
        // rows are 8-byte aligned (BOPL_OBS_STRIDE is even), not 16
        const float2* src = (const float2*)obs; float2* dst = (float2*)(a.rollout_obs + (long)row * BOPL_OBS_STRIDE);
        for (int i = tid; i < BOPL_OBS_STRIDE / 2; i += BR4_SET_THREADS) dst[i] = src[i];
        if (BOPL_OBS_STRIDE & 1) if (tid == 0) a.rollout_obs[(long)row * BOPL_OBS_STRIDE + BOPL_OBS_STRIDE - 1] = obs[BOPL_OBS_STRIDE - 1];
    }
    // valid rows of the set (first feature nonzero), in order
    __shared__ int n_valid;
    if (tid == 0) {
        int n = 0;
        for (int r = 0; r < sp.rows; r++) if (obs[sp.offset + r * sp.width] != 0.0f) valid[n++] = r;
        n_valid = n;
    }
    // W2 of this group into shared as bf16 pairs
    const float* W2 = a.w.w2[grp][s];
    for (int e = tid; e < D * (I / 2); e += BR4_SET_THREADS) {
        int d = e / (I / 2), i2 = e - d * (I / 2);
        W2s[d * ldw + i2] = __floats2bfloat162_rn(W2[(long)d * I + 2 * i2], W2[(long)d * I + 2 * i2 + 1]);
    }
    __syncthreads();
    const int n = n_valid;
    float* out_max = a.joined + (long)row * a.ldj + a.col0 + 2 * D * s;
    if (n == 0) {
        for (int d = tid; d < D; d += BR4_SET_THREADS) { out_max[d] = 0.0f; out_max[D + d] = 0.0f; }
        return;
    }
    // h1[j][i] = relu(sum_f bf(W1[i][f]) bf(x[j][f]) + E[id_j][i]) (bf16 stored, like the trainer's packed rows)
    const float* W1 = a.w.w1[grp][s];
    const float* E = a.w.emb[grp];
    for (int e = tid; e < n * I; e += BR4_SET_THREADS) {
        int j = e / I, i = e - j * I;
        const float* x = obs + sp.offset + valid[j] * sp.width;
        const float* w = W1 + (long)i * sp.features;
        float acc = 0.0f;
        for (int f = 0; f < sp.features; f++) acc = fmaf(br4_bf(w[f]), br4_bf(x[f]), acc);
        if (sp.id_col >= 0) acc += br4_bf(E[(long)v4::embedding_id(x[sp.id_col]) * I + i]);
        h1[j * I + i] = __float2bfloat16(fmaxf(acc, 0.0f));
    }
    __syncthreads();
    // h2[j][d] = relu(sum_i W2[d][i] h1[j][i]); thread (d, part) owns the rows j = part, part + parts, ...
    const int parts = BR4_SET_THREADS / D, d = tid % D, part = tid / D;
    float best = 0.0f, sum = 0.0f;
    const __nv_bfloat162* wrow = W2s + d * ldw;
    for (int j = part; j < n; j += parts) {
        const __nv_bfloat162* hrow = (const __nv_bfloat162*)(h1 + j * I);
        float acc = 0.0f;
        #pragma unroll 8
        for (int i2 = 0; i2 < I / 2; i2++) {
            float2 w = __bfloat1622float2(wrow[i2]), h = __bfloat1622float2(hrow[i2]);
            acc = fmaf(w.x, h.x, acc); acc = fmaf(w.y, h.y, acc);
        }
        float v = br4_bf(fmaxf(acc, 0.0f));   // the trainer pools bf16-stored h2
        best = fmaxf(best, v); sum += v;
    }
    // reduce the parts through shared (reuse the valid[] area is too small: use W2s' tail? h1 is still live; use a dedicated array)
    __shared__ float red_max[BR4_SET_THREADS], red_sum[BR4_SET_THREADS];
    red_max[tid] = best; red_sum[tid] = sum;
    __syncthreads();
    if (part == 0) {
        for (int q = 1; q < parts; q++) { best = fmaxf(best, red_max[q * D + d]); sum += red_sum[q * D + d]; }
        out_max[d] = best;
        out_max[D + d] = a.mean_pool ? sum / n : 0.0f;
    }
}

// Row-wise glue. One block per row, 256 threads.
enum { BR4_LN = 0, BR4_RESIDUAL_LN_PRIV = 1, BR4_CRITIC_OUT = 2 };
struct Br4RowsArgs {
    int mode, rows, W;
    const float* x; int ldx;          // LN input (mode 0), FFN output or h (mode 1), z2 (mode 2)
    const float* res; int ldres;      // residual added to x first (modes 1, 2), or nullptr
    float* y; int ldy;                // mode 1: x + res written back here (the trunk output y), mode 2: unused
    float* out; int ldout;            // LN output; mode 2: the decoder row (value at column vcol)
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
        if (threadIdx.x == 0) a.out[(long)row * a.ldout + a.vcol] = dot;
    } else {
        for (int c = threadIdx.x; c < W; c += blockDim.x) a.out[(long)row * a.ldout + c] = (rowbuf[c] - mu) * r;
    }
}

struct Br4Wt {   // one group's GEMM-side weights (br_prepare)
    float *flat, *fusion, *gru[BR_MAX_LAYERS], *ffn_in, *ffn_out, *actor, *cin, *b1, *b2;
};
struct Br4Acts {
    float *joined, *enc, *comb, *gru[2], *ln, *f, *y, *z, *lnc, *b, *z2, *dec;
};
struct Br4Plan {
    int FD, D, I, F, C, P, mean_pool, JW, W, LNW;
    Br4Wt wt[BR_MAX_GROUPS];
    Br4Acts* acts;   // [num_buffers]
};

#ifndef BR_KERNELS_ONLY
static void br_group_weights(PuffeRL* pufferl, int g, int buf, PolicyWeights** w, PrecisionTensor** s);

// Shape agreement of the learner and the banks (all v4, same widths); nullptr when fine.
static const char* br4_check(PuffeRL* pufferl) {
    auto* ew = (v4::EncWeights*)pufferl->weights.encoder;
    auto* dw = (v4::DecWeights*)pufferl->weights.decoder;
    if (ew->D > BR4_SET_THREADS || BR4_SET_THREADS % ew->D || ew->I % 2) return "v4 set width must divide 256 (inner even)";
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
    }
    for (int g = 0; g < p->g.n; g++) {
        Br4Wt& w = q->wt[g];
        zalloc(&w.flat, (size_t)br_frag_floats(v4::FLAT, q->FD)); zalloc(&w.fusion, (size_t)br_frag_floats(q->JW, H));
        for (int l = 0; l < p->layers; l++) zalloc(&w.gru[l], (size_t)br_frag_floats(H, 3 * H));
        zalloc(&w.ffn_in, q->F ? (size_t)br_frag_floats(H, q->F) : 0); zalloc(&w.ffn_out, q->F ? (size_t)br_frag_floats(q->F, H) : 0);
        zalloc(&w.actor, (size_t)br_frag_floats(H, p->ld_dec));
        zalloc(&w.cin, q->C ? (size_t)br_frag_floats(q->W, q->C) : 0);
        zalloc(&w.b1, q->C ? (size_t)br_frag_floats(q->C, q->C) : 0); zalloc(&w.b2, q->C ? (size_t)br_frag_floats(q->C, q->C) : 0);
    }
    size_t smem = br4_sets_smem(q->D, q->I);
    assert(cudaFuncSetAttribute(br4_sets, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess);
    p->v4 = q;
    fprintf(stderr, "bopl batched rollout forward: policy v4 chain (set %d/%d, flat %d, ffn %d, critic %d%s), %.1f KB shared per set block\n",
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
    }
    br_transpose_launch(p, s_flat, d_flat, p.g.n, q.FD, v4::FLAT, q.FD, stream);
    br_transpose_launch(p, s_fus, d_fus, p.g.n, H, q.JW, H, stream);
    for (int l = 0; l < p.layers; l++) br_transpose_launch(p, s_gru[l], d_gru[l], p.g.n, 3 * H, H, 3 * H, stream);
    if (q.F) { br_transpose_launch(p, s_fi, d_fi, p.g.n, q.F, H, q.F, stream); br_transpose_launch(p, s_fo, d_fo, p.g.n, H, q.F, H, stream); }
    br_transpose_launch(p, s_act, d_act, p.g.n, p.dec_cols, H, p.ld_dec, stream);
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
    }
    for (int g = 0; g < p->g.n; g++) {
        Br4Wt& w = q->wt[g];
        for (float* ptr : {w.flat, w.fusion, w.ffn_in, w.ffn_out, w.actor, w.cin, w.b1, w.b2}) cudaFree(ptr);
        for (int l = 0; l < p->layers; l++) cudaFree(w.gru[l]);
    }
    free(q->acts);
    delete q;
    p->v4 = nullptr;
}

// The v4 chain from the live obs rows to the decoder rows A.dec (logits, value
// at column dec_cols), rollout copies included. The caller samples.
static void br4_chain(PuffeRL* pufferl, BrPlan& p, int buf, const float* obs, const float* rewards, const float* terminals,
                      float* rollout_obs, float* rollout_rewards, float* rollout_terminals, cudaStream_t stream) {
    Br4Plan& q = *p.v4; Br4Acts& A = q.acts[buf];
    int rows = p.rows, H = p.H;
    Br4SetsArgs sa = {};
    sa.obs = obs; sa.rewards = rewards; sa.terminals = terminals;
    sa.rollout_obs = rollout_obs; sa.rollout_rewards = rollout_rewards; sa.rollout_terminals = rollout_terminals;
    sa.joined = A.joined; sa.ldj = q.JW; sa.col0 = q.FD; sa.D = q.D; sa.I = q.I; sa.mean_pool = q.mean_pool; sa.rows = rows; sa.g = p.g;
    BrLinear w_flat = {}, w_fus = {}, w_fi = {}, w_fo = {}, w_act = {}, w_cin = {}, w_b1 = {}, w_b2 = {}, w_cout = {}, w_gru[BR_MAX_LAYERS] = {};
    BrGateArgs gate = {};
    for (int g = 0; g < p.g.n; g++) {
        PolicyWeights* w; PrecisionTensor* s;
        br_group_weights(pufferl, g, buf, &w, &s);
        auto* ew = (v4::EncWeights*)w->encoder; auto* dw = (v4::DecWeights*)w->decoder;
        for (int k = 0; k < 4; k++) { sa.w.w1[g][k] = ew->w1[k].data; sa.w.w2[g][k] = ew->w2[k].data; }
        sa.w.emb[g] = ew->embedding.data;
        w_flat.w[g] = q.wt[g].flat; w_fus.w[g] = q.wt[g].fusion;
        for (int l = 0; l < p.layers; l++) w_gru[l].w[g] = q.wt[g].gru[l];
        w_fi.w[g] = q.wt[g].ffn_in; w_fo.w[g] = q.wt[g].ffn_out; w_act.w[g] = q.wt[g].actor;
        w_cin.w[g] = q.wt[g].cin; w_b1.w[g] = q.wt[g].b1; w_b2.w[g] = q.wt[g].b2; w_cout.w[g] = dw->critic_out.data;
        gate.state[g] = s->data; gate.state_rows[g] = (int)s->shape[1];
    }
    // encoder
    br4_sets<<<dim3(rows, 4), BR4_SET_THREADS, br4_sets_smem(q.D, q.I), stream>>>(sa);
    br_gemm_launch(p, obs, BOPL_OBS_STRIDE, A.joined, q.JW, v4::FLAT, q.FD, q.FD, 1, w_flat, stream);
    br_gemm_launch(p, A.joined, q.JW, A.ln, q.LNW, q.JW, H, H, 0, w_fus, stream);
    Br4RowsArgs ra = {}; ra.rows = rows; ra.g = p.g;
    ra.mode = BR4_LN; ra.W = H; ra.x = A.ln; ra.ldx = q.LNW; ra.out = A.enc; ra.ldout = H;
    br4_rows<<<rows, 256, 0, stream>>>(ra);
    // MinGRU
    const float* x = A.enc;
    for (int l = 0; l < p.layers; l++) {
        br_gemm_launch(p, x, H, A.comb, 3 * H, H, 3 * H, 3 * H, 0, w_gru[l], stream);
        gate.combined = A.comb; gate.x = x; gate.terminals = terminals;
        gate.out = A.gru[l & 1]; gate.layer = l; gate.H = H; gate.rows = rows; gate.g = p.g;
        br_gate<<<grid_size(rows * H), BLOCK_SIZE, 0, stream>>>(gate);
        x = A.gru[l & 1];
    }
    // decoder: y = h + FFN(LN(h)); then LN([y, priv]) for the critic
    const float* priv = obs + BOPL4_OBS_PRIVILEGED_OFFSET;
    ra = Br4RowsArgs{}; ra.rows = rows; ra.g = p.g;
    if (q.F) {
        ra.mode = BR4_LN; ra.W = H; ra.x = x; ra.ldx = H; ra.out = A.ln; ra.ldout = q.LNW;
        br4_rows<<<rows, 256, 0, stream>>>(ra);
        br_gemm_launch(p, A.ln, q.LNW, A.f, q.F, H, q.F, q.F, 1, w_fi, stream);
        br_gemm_launch(p, A.f, q.F, A.y, H, q.F, H, H, 0, w_fo, stream);
        ra = Br4RowsArgs{}; ra.rows = rows; ra.g = p.g;
        ra.mode = BR4_RESIDUAL_LN_PRIV; ra.W = q.W; ra.x = A.y; ra.ldx = H; ra.res = x; ra.ldres = H;
    } else {
        ra.mode = BR4_RESIDUAL_LN_PRIV; ra.W = q.W; ra.x = x; ra.ldx = H;
    }
    ra.y = A.y; ra.ldy = H; ra.out = A.ln; ra.ldout = q.LNW; ra.priv = priv; ra.ldpriv = BOPL_OBS_STRIDE; ra.P = q.P;
    br4_rows<<<rows, 256, 0, stream>>>(ra);   // A.y = y (trunk output), A.ln = LN([y, priv])
    br_gemm_launch(p, A.y, H, A.dec, p.ld_dec, H, p.dec_cols, p.ld_dec, 0, w_act, stream);
    if (q.C) {
        br_gemm_launch(p, A.ln, q.LNW, A.z, q.C, q.W, q.C, q.C, 0, w_cin, stream);
        ra = Br4RowsArgs{}; ra.rows = rows; ra.g = p.g;
        ra.mode = BR4_LN; ra.W = q.C; ra.x = A.z; ra.ldx = q.C; ra.out = A.lnc; ra.ldout = q.C;
        br4_rows<<<rows, 256, 0, stream>>>(ra);
        br_gemm_launch(p, A.lnc, q.C, A.b, q.C, q.C, q.C, q.C, 1, w_b1, stream);
        br_gemm_launch(p, A.b, q.C, A.z2, q.C, q.C, q.C, q.C, 0, w_b2, stream);
        ra = Br4RowsArgs{}; ra.rows = rows; ra.g = p.g;
        ra.mode = BR4_CRITIC_OUT; ra.W = q.C; ra.x = A.z2; ra.ldx = q.C; ra.res = A.z; ra.ldres = q.C;
        ra.out = A.dec; ra.ldout = p.ld_dec; ra.vcol = p.dec_cols; ra.cout = w_cout;
        br4_rows<<<rows, 256, 0, stream>>>(ra);
    } else {
        cudaMemset2DAsync(A.dec + p.dec_cols, (size_t)p.ld_dec * sizeof(float), 0, sizeof(float), rows, stream);
    }
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
            dv = std::max(dv, std::fabs((double)ours[(size_t)(off + r) * p.ld_dec + p.dec_cols] - theirs[(size_t)r * od1 + p.dec_cols]));
        }
        fprintf(stderr, "bopl batched forward check: buffer %d step %d group %d (%d rows): logits max|diff| %.3g (max|logit| %.3g), value max|diff| %.3g\n",
                buf, t, g, n, dl, scale, dv);
    }
}
#endif  // BR_KERNELS_ONLY
