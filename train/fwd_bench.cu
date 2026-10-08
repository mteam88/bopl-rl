// The policy v4 rollout forward of one buffer step, standalone: the trainer's
// launch sequence (br_forward's packed-row path: the H2D copy of the slots,
// bopl4_unpack_rows, bopl4_rays_launch, br4_run, br_sample, the D2H copy of
// the actions) on buffer steps dumped from a running trainer (real weights,
// rows, recurrent state, RNG). Two jobs:
//
//   exactness  every dumped step run eagerly from its dumped inputs; the
//              outputs (dense rows, joined, decoder rows, actions, logprobs,
//              values, mask copy, recurrent state, RNG) compared bit for bit
//              with what the trainer produced (--check), and saved (--save
//              DIR) or compared with another build's saved outputs
//              (--compare DIR: per array the floats that differ, max abs and
//              rel difference);
//   latency    the step captured into a CUDA graph per stream as the trainer
//              does, replayed on 1 stream and on 4 streams at once (CUDA
//              events, GPU-side; median / p10 / p90 over --reps).
//
//   fwd_bench <dump dir> [--check] [--save DIR] [--compare DIR] [--reps N] [--steps N] [--streams 1|4 (default both)]
//
// The dump layout (s###/ per step, w<n>/ per weight snapshot) is written
// by train/fwd_bench_dump.cuh, spliced into the trainer by hand. Build:
// make fwd-bench (after train/build.py --device cuda, whose overlay it includes).
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include "models.cu"
#include "bopl_set_encoder.cu"
#include "bopl_v4.cu"
#include "bopl_aim.h"
#include "bopl_obs_pack.h"

// ---- pufferlib.cu's sampler helpers (verbatim; --check proves them against the trainer's outputs) ----
#define SAMPLE_WARP_ROWS 8
#define SAMPLE_MAX_LOGITS 512
__device__ __forceinline__ float finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) {
        return 0.0f;
    }
    if (isinf(x)) {
        return x > 0.0f ? hi : lo;
    }
    return fminf(hi, fmaxf(lo, x));
}
template<typename T>
__device__ __forceinline__ float bopl_aim_fourier_logit_dev(const T* block, int harmonics, int bin) {
    if (bin <= 0) return to_float(block[0]);
    BoplAimBasis b = bopl_aim_basis(bin);
    float sum = 0.0f;
    for (int k = 1; k <= harmonics; k++) {
        sum += to_float(block[2 * k - 1]) * b.ck + to_float(block[2 * k]) * b.sk;
        bopl_aim_basis_next(&b);
    }
    return sum * bopl_aim_fourier_scale(harmonics);
}
template<typename T>
__device__ __forceinline__ float bopl_game_logit(const T* row, int fourier, int ar, int buttons, int g) {
    if (g < BOPL_ACT_AIM) {
        const T* block = row + bopl_aim_block_offset(fourier, ar, buttons);
        return fourier > 0 ? bopl_aim_fourier_logit_dev(block, fourier, g) : to_float(block[g]);
    }
    return to_float(row[bopl_decoder_buttons_offset(fourier, ar) + g - BOPL_ACT_AIM]);
}

#define BR_KERNELS_ONLY
#include "bopl_rollout.cu"
#include "bopl_rollout_v4.cu"
extern "C" void bopl4_rays_init(const float* packed, long slots);
extern "C" void bopl4_rays_launch(const float* packed, float* dense, int rows, cudaStream_t stream);

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}
static std::vector<char> slurp(const std::string& path, bool required = true) {
    std::vector<char> out;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { if (required) { fprintf(stderr, "missing %s\n", path.c_str()); exit(1); } return out; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize(n);
    if (n && fread(out.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path.c_str()); exit(1); }
    fclose(f);
    return out;
}
static void spill(const std::string& path, const void* p, size_t n) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f || fwrite(p, 1, n, f) != n) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
    fclose(f);
}
template <typename T> static T* dev_alloc(size_t n) {
    T* p = nullptr; ck(cudaMalloc((void**)&p, n * sizeof(T) + 16), "cudaMalloc"); ck(cudaMemset(p, 0, n * sizeof(T) + 16), "memset"); return p;
}
static void* dev_upload(const std::vector<char>& h) {
    void* p = nullptr; ck(cudaMalloc(&p, h.size() + 16), "cudaMalloc"); ck(cudaMemcpy(p, h.data(), h.size(), cudaMemcpyHostToDevice), "upload"); return p;
}

struct Meta { std::map<std::string, std::string> kv; int i(const std::string& k) const { auto it = kv.find(k); if (it == kv.end()) { fprintf(stderr, "meta: no %s\n", k.c_str()); exit(1); } return atoi(it->second.c_str()); } };
static Meta read_meta(const std::string& path) {
    Meta m; std::vector<char> b = slurp(path); b.push_back(0);
    char* save = nullptr;
    for (char* line = strtok_r(b.data(), "\n", &save); line; line = strtok_r(nullptr, "\n", &save)) {
        char* sp = strchr(line, ' '); if (!sp) continue; *sp = 0; m.kv[line] = sp + 1;
    }
    return m;
}

// One weight snapshot: every group's chain-side weights (the plan's Br4Wt) and the tensors read as stored.
struct Weights { Br4Wt wt[BR_MAX_GROUPS]; const float* emb[BR_MAX_GROUPS]; const float* cout[BR_MAX_GROUPS]; };
// One dumped step's host data.
struct Step {
    std::string dir, wdir;
    int t;
    std::vector<char> packed, host_rows, rng_pre, state_pre[BR_MAX_GROUPS];
};
// One stream's (buffer's) device state.
struct Lane {
    cudaStream_t s;
    float *dense, *rew, *term, *actions, *env_actions, *logprobs, *values, *mask;
    curandStatePhilox4_32_10_t* rng;
    float* state[BR_MAX_GROUPS];
    cudaGraphExec_t graph = nullptr;
    cudaEvent_t e0, e1, ready;   // ready: the actions in host memory (br_forward's actions_ready; the critic may still run)
};

struct Bench {
    Meta m;
    int rows, apb, nb = 4, H, G, num_atns, mask_stride, pack_bytes, state_numel[BR_MAX_GROUPS], state_rows[BR_MAX_GROUPS];
    BrPlan p = {};
    Br4Plan q = {};
    std::map<std::string, Weights> weights;
    const Weights* cur = nullptr;
    std::vector<Step> steps;
    std::vector<Lane> lanes;
    float *packed_dev, *packed_host, *packed_mapped, *host_rows, *host_rows_dev, *h_actions;
    int* act_sizes;

    void load(const std::string& root, int max_steps) {
        std::vector<std::string> names;
        DIR* d = opendir(root.c_str()); if (!d) { fprintf(stderr, "cannot open %s\n", root.c_str()); exit(1); }
        while (dirent* e = readdir(d)) if (e->d_name[0] == 's') names.push_back(e->d_name);
        closedir(d);
        std::sort(names.begin(), names.end());
        if (max_steps > 0 && (int)names.size() > max_steps) names.resize(max_steps);
        if (names.empty()) { fprintf(stderr, "no steps in %s\n", root.c_str()); exit(1); }
        m = read_meta(root + "/" + names[0] + "/meta");
        rows = m.i("rows"); apb = m.i("apb"); H = m.i("H"); G = m.i("groups"); num_atns = m.i("num_atns"); mask_stride = m.i("mask_stride");
        pack_bytes = m.i("pack_bytes");
        if (!m.i("pack_in_tail")) { fprintf(stderr, "the bench handles the pack in the slots' tail only\n"); exit(1); }
        for (int g = 0; g < G; g++) { state_numel[g] = m.i("state_numel" + std::to_string(g)); state_rows[g] = m.i("state_rows" + std::to_string(g)); }
        // the plan, as br_init / br4_init build it
        p.enabled = 1; p.v4 = &q; p.tf32 = 1; p.rows = rows; p.H = H; p.layers = m.i("layers");
        p.A_total = m.i("A_total"); p.num_atns = num_atns; p.mask_stride = mask_stride; p.dec_cols = m.i("dec_cols");
        p.aim_fourier = m.i("aim_fourier"); p.aim_ar = m.i("aim_ar"); p.ld_dec = m.i("ld_dec");
        p.g.n = G; for (int g = 0; g <= G; g++) p.g.end[g] = m.i("end" + std::to_string(g));
        br_tiles(p.g, p.tiles);
        q.FD = m.i("FD"); q.D = m.i("D"); q.I = m.i("I"); q.F = m.i("F"); q.C = m.i("C"); q.P = m.i("P"); q.mean_pool = m.i("mean_pool");
        q.JW = m.i("JW"); q.W = m.i("W"); q.LNW = m.i("LNW");
        if (q.JW != q.FD + 8 * q.D || q.W != H + q.P || q.LNW != std::max(H, std::max(q.W, q.C))) { fprintf(stderr, "plan widths disagree with the dump\n"); exit(1); }
        for (auto& n : names) {
            Step s; s.dir = root + "/" + n;
            Meta sm = read_meta(s.dir + "/meta");
            s.wdir = sm.kv["weights"]; s.t = sm.i("t");
            s.packed = slurp(s.dir + "/packed"); s.host_rows = slurp(s.dir + "/host_rows"); s.rng_pre = slurp(s.dir + "/rng_pre");
            for (int g = 0; g < G; g++) s.state_pre[g] = slurp(s.dir + "/state_pre" + std::to_string(g));
            if (!weights.count(s.wdir)) weights[s.wdir] = load_weights(s.wdir);
            steps.push_back(std::move(s));
        }
        std::vector<char> as = slurp(root + "/" + names[0] + "/act_sizes");
        act_sizes = (int*)dev_upload(as);
        fprintf(stderr, "fwd_bench: %zu steps, %zu weight snapshots, %d rows (%d groups, %d tiles), H %d, set %d/%d, critic %d\n",
                steps.size(), weights.size(), rows, G, p.tiles[G], H, q.D, q.I, q.C);
    }
    Weights load_weights(const std::string& wdir) {
        Weights w = {};
        for (int g = 0; g < G; g++) {
            std::string pre = wdir + "/g" + std::to_string(g) + "_";
            auto up = [&](const char* name) { return (float*)dev_upload(slurp(pre + name)); };
            Br4Wt& t = w.wt[g];
            t.flat = up("flat"); t.fusion = up("fusion");
            for (int l = 0; l < p.layers; l++) t.gru[l] = up(("gru" + std::to_string(l)).c_str());
            if (q.F) { t.ffn_in = up("ffn_in"); t.ffn_out = up("ffn_out"); }
            t.actor = up("actor");
            if (q.C) { t.cin = up("cin"); t.b1 = up("b1"); t.b2 = up("b2"); w.cout[g] = up("cout"); }
            for (int s = 0; s < 4; s++) {
                t.w1t[s] = (__nv_bfloat16*)up(("w1t" + std::to_string(s)).c_str());
                t.w2b[s] = (__nv_bfloat16*)up(("w2b" + std::to_string(s)).c_str());
            }
            w.emb[g] = up("emb");
        }
        return w;
    }
    void setup() {
        size_t slot = BOPL4_PACK_P, slots = (size_t)nb * apb;
        ck(cudaMalloc((void**)&packed_dev, slots * slot * sizeof(float)), "packed_dev");
        ck(cudaHostAlloc((void**)&packed_host, slots * slot * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "packed_host");
        ck(cudaHostGetDevicePointer((void**)&packed_mapped, packed_host, 0), "packed_host alias");
        for (size_t r = 0; r < slots; r++) ((unsigned*)(packed_host + r * slot))[0] = BOPL4_PACK_MODE_DENSE;
        ck(cudaMemcpy(packed_dev, packed_host, slots * slot * sizeof(float), cudaMemcpyHostToDevice), "packed init");
        ck(cudaHostAlloc((void**)&host_rows, slots * BOPL_OBS_STRIDE * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable), "host_rows");
        ck(cudaHostGetDevicePointer((void**)&host_rows_dev, host_rows, 0), "host_rows alias");
        ck(cudaHostAlloc((void**)&h_actions, slots * num_atns * sizeof(float), cudaHostAllocPortable), "h_actions");
        bopl4_rays_init(packed_dev, (long)slots);
        q.acts = (Br4Acts*)calloc(nb, sizeof(Br4Acts));
        lanes.resize(nb);
        int lo, hi; cudaDeviceGetStreamPriorityRange(&lo, &hi);
        for (int b = 0; b < nb; b++) {
            Br4Acts& a = q.acts[b];
            auto z = [&](size_t n) { return n ? dev_alloc<float>(n) : nullptr; };
            a.joined = z((size_t)apb * q.JW); a.enc = z((size_t)apb * H); a.comb = z((size_t)apb * 3 * H);
            a.gru[0] = z((size_t)apb * H); a.gru[1] = z((size_t)apb * H); a.ln = z((size_t)apb * q.LNW); a.f = z((size_t)apb * q.F);
            a.y = z((size_t)apb * H); a.z = z((size_t)apb * q.C); a.lnc = z((size_t)apb * q.C); a.b = z((size_t)apb * q.C); a.z2 = z((size_t)apb * q.C);
            a.dec = z((size_t)apb * p.ld_dec);
            Lane& L = lanes[b];
            ck(cudaStreamCreateWithPriority(&L.s, cudaStreamNonBlocking, hi), "stream");
            ck(cudaStreamCreateWithPriority(&a.critic, cudaStreamNonBlocking, hi), "critic stream");
            cudaEventCreateWithFlags(&a.fork, cudaEventDisableTiming); cudaEventCreateWithFlags(&a.join, cudaEventDisableTiming);
            L.dense = z((size_t)rows * BOPL_OBS_STRIDE); L.rew = z(rows); L.term = z(rows);
            L.actions = z((size_t)rows * num_atns); L.env_actions = z((size_t)rows * num_atns); L.logprobs = z(rows); L.values = z(rows);
            L.mask = z((size_t)rows * mask_stride);
            L.rng = dev_alloc<curandStatePhilox4_32_10_t>(rows);
            for (int g = 0; g < G; g++) L.state[g] = z(state_numel[g]);
            cudaEventCreate(&L.e0); cudaEventCreate(&L.e1); cudaEventCreate(&L.ready);
        }
        br4_sets_init(q.D, q.I);
    }
    void use_weights(const std::string& wdir) { cur = &weights.at(wdir); for (int g = 0; g < G; g++) q.wt[g] = cur->wt[g]; }
    // A step's inputs into buffer b's host slots / device state (as the env workers and the previous step left them).
    void stage(const Step& s, int b) {
        size_t slot_bytes = (size_t)BOPL4_PACK_P * sizeof(float);
        memcpy((char*)packed_host + (size_t)b * apb * slot_bytes, s.packed.data(), s.packed.size());
        memcpy((char*)host_rows + (size_t)b * apb * BOPL_OBS_STRIDE * sizeof(float), s.host_rows.data(), s.host_rows.size());
        Lane& L = lanes[b];
        ck(cudaMemcpy(L.rng, s.rng_pre.data(), s.rng_pre.size(), cudaMemcpyHostToDevice), "rng");
        for (int g = 0; g < G; g++) ck(cudaMemcpy(L.state[g], s.state_pre[g].data(), s.state_pre[g].size(), cudaMemcpyHostToDevice), "state");
    }
    // br_forward's packed-row path for buffer b.
    void forward(int b) {
        Lane& L = lanes[b];
        const int start = b * apb;
        size_t slot_bytes = (size_t)BOPL4_PACK_P * sizeof(float), off = (size_t)start * slot_bytes;
        ck(cudaMemcpy2DAsync((char*)packed_dev + off, slot_bytes, (const char*)packed_host + off, slot_bytes,
                             (size_t)BR4_PACK_DMA_WORDS * sizeof(float), rows, cudaMemcpyHostToDevice, L.s), "h2d");
        const float* hr = host_rows_dev + (size_t)start * BOPL_OBS_STRIDE;
        bopl4_unpack_rows<<<bopl4_unpack_blocks(rows, pack_bytes), BR4_UNPACK_THREADS, 0, L.s>>>(
            packed_mapped + (size_t)start * BOPL4_PACK_P, packed_dev + (size_t)start * BOPL4_PACK_P, hr, L.dense, rows, pack_bytes);
        bopl4_rays_launch(packed_dev + (size_t)start * BOPL4_PACK_P, L.dense, rows, L.s);
        const char* pk = (const char*)packed_dev + (size_t)(start + rows) * slot_bytes;
        const float* rewards = (const float*)pk;
        const float* terminals = (const float*)(pk + rows * sizeof(float));
        const unsigned char* mask = mask_stride ? (const unsigned char*)(pk + 2 * rows * sizeof(float)) : nullptr;
        Br4Groups gr = {};
        for (int g = 0; g < G; g++) { gr.emb[g] = cur->emb[g]; gr.cout[g] = cur->cout[g]; gr.state[g] = L.state[g]; gr.state_rows[g] = state_rows[g]; }
        br4_run(p, b, gr, L.dense, rewards, terminals, L.dense, L.rew, L.term, L.values, L.s);
        BrSampleArgs smp = {};
        smp.dec = q.acts[b].dec; smp.ld_dec = p.ld_dec; smp.A_total = p.A_total;
        smp.dec_cols = p.dec_cols; smp.aim_fourier = p.aim_fourier; smp.aim_ar = p.aim_ar;
        smp.act_sizes = act_sizes; smp.num_atns = num_atns; smp.value_rows = br4_value_rows(p);
        smp.env_mask = mask; smp.mask_stride = mask_stride; smp.rollout_mask = L.mask;
        smp.env_actions = L.env_actions; smp.rollout_actions = L.actions;
        smp.logprobs = L.logprobs; smp.values = L.values; smp.rng = L.rng;
        smp.head_consume = nullptr; smp.hc_stride = 0; smp.rows = rows;
        br_launch_pdl(br_sample, dim3((rows + SAMPLE_WARP_ROWS - 1) / SAMPLE_WARP_ROWS), dim3(SAMPLE_WARP_ROWS * 32), 0, L.s, smp);
        ck(cudaMemcpyAsync(h_actions + (size_t)start * num_atns, L.env_actions, (size_t)rows * num_atns * sizeof(float), cudaMemcpyDeviceToHost, L.s), "d2h");
        br_record_external(L.ready, L.s);
        br4_join(p, b, L.s);
    }
    // The outputs of buffer b's last forward, by name (the dump's out_* / *_post files).
    std::vector<std::pair<std::string, std::vector<char>>> outputs(int b) {
        Lane& L = lanes[b];
        ck(cudaStreamSynchronize(L.s), "sync");
        std::vector<std::pair<std::string, std::vector<char>>> o;
        auto get = [&](const char* name, const void* dp, size_t n) {
            std::vector<char> h(n); ck(cudaMemcpy(h.data(), dp, n, cudaMemcpyDeviceToHost), name); o.emplace_back(name, std::move(h));
        };
        get("out_obs", L.dense, (size_t)rows * BOPL_OBS_STRIDE * 4);
        get("out_joined", q.acts[b].joined, (size_t)rows * q.JW * 4);
        get("out_dec", q.acts[b].dec, (size_t)rows * p.ld_dec * 4);
        get("out_rewards", L.rew, rows * 4); get("out_terminals", L.term, rows * 4);
        get("out_actions", L.actions, (size_t)rows * num_atns * 4);
        get("out_logprobs", L.logprobs, rows * 4); get("out_values", L.values, rows * 4);
        if (mask_stride) get("out_mask", L.mask, (size_t)rows * mask_stride * 4);
        std::vector<char> ea((size_t)rows * num_atns * 4); memcpy(ea.data(), h_actions + (size_t)b * apb * num_atns, ea.size());
        o.emplace_back("out_env_actions", std::move(ea));
        get("rng_post", L.rng, (size_t)rows * sizeof(curandStatePhilox4_32_10_t));
        for (int g = 0; g < G; g++) get(("state_post" + std::to_string(g)).c_str(), L.state[g], (size_t)state_numel[g] * 4);
        return o;
    }
};

// Bits that differ between two output arrays; for float arrays also the max abs / rel difference.
struct Diff { long words = 0, bad = 0; double max_abs = 0, max_rel = 0; };
static void diff_into(Diff& d, const std::string& name, const std::vector<char>& a, const std::vector<char>& b) {
    if (a.size() != b.size()) { fprintf(stderr, "%s: sizes differ (%zu vs %zu)\n", name.c_str(), a.size(), b.size()); d.bad++; return; }
    const uint32_t* x = (const uint32_t*)a.data(); const uint32_t* y = (const uint32_t*)b.data();
    const bool floats = name.rfind("rng", 0) != 0;
    for (size_t i = 0; i < a.size() / 4; i++) {
        d.words++;
        if (x[i] == y[i]) continue;
        d.bad++;
        if (floats) {
            float u, v; memcpy(&u, &x[i], 4); memcpy(&v, &y[i], 4);
            double ad = std::fabs((double)u - v), rd = ad / std::max(1e-30, std::max(std::fabs((double)u), std::fabs((double)v)));
            if (std::isfinite(ad)) { d.max_abs = std::max(d.max_abs, ad); d.max_rel = std::max(d.max_rel, rd); }
        }
    }
}

static double pct(std::vector<float> v, double q) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[std::min(v.size() - 1, (size_t)(q * v.size()))]; }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "fwd_bench <dump dir> [--check] [--save DIR] [--compare DIR] [--reps N] [--steps N] [--streams 1|4]\n"); return 2; }
    std::string root = argv[1], save, compare;
    bool check = false; int reps = 2000, max_steps = 0, streams = 0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--check") check = true;
        else if (a == "--save" && i + 1 < argc) save = argv[++i];
        else if (a == "--compare" && i + 1 < argc) compare = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--steps" && i + 1 < argc) max_steps = atoi(argv[++i]);
        else if (a == "--streams" && i + 1 < argc) streams = atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    Bench B;
    B.load(root, max_steps);
    B.setup();
    // ---- exactness: every step eagerly on buffer 0, from its dumped inputs ----
    if (check || !save.empty() || !compare.empty()) {
        std::map<std::string, Diff> vs_trainer, vs_other;
        for (size_t k = 0; k < B.steps.size(); k++) {
            const Step& s = B.steps[k];
            B.use_weights(s.wdir);
            B.stage(s, 0);
            B.forward(0);
            auto out = B.outputs(0);
            std::string name = s.dir.substr(s.dir.rfind('/') + 1);
            if (!save.empty()) { mkdir(save.c_str(), 0755); mkdir((save + "/" + name).c_str(), 0755); }
            for (auto& [n, data] : out) {
                if (check) diff_into(vs_trainer[n], n, data, slurp(s.dir + "/" + n));
                if (!save.empty()) spill(save + "/" + name + "/" + n, data.data(), data.size());
                if (!compare.empty()) diff_into(vs_other[n], n, slurp(compare + "/" + name + "/" + n), data);
            }
        }
        long rows = (long)B.steps.size() * B.rows;
        auto report = [&](const char* what, std::map<std::string, Diff>& m) {
            long bad = 0;
            for (auto& [n, d] : m) {
                bad += d.bad;
                printf("  %-16s %10ld words %8ld differ  max|abs| %.3g  max rel %.3g\n", n.c_str(), d.words, d.bad, d.max_abs, d.max_rel);
            }
            printf("%s: %ld rows, %ld words differ in total -> %s\n", what, rows, bad, bad ? "MISMATCH" : "bit-identical");
            return bad;
        };
        long bad = 0;
        if (check) bad += report("vs the trainer's outputs", vs_trainer);
        if (!compare.empty()) bad += report(("vs " + compare).c_str(), vs_other);
        if (bad) return 1;
        if (reps == 0) return 0;
    }
    if (reps == 0) return 0;
    // ---- latency: one graph per buffer, captured as the trainer captures ----
    B.use_weights(B.steps[0].wdir);
    for (int b = 0; b < B.nb; b++) {
        B.stage(B.steps[b % B.steps.size()], b);
        Lane& L = B.lanes[b];
        cudaGraph_t g;
        ck(cudaStreamBeginCapture(L.s, cudaStreamCaptureModeThreadLocal), "capture");
        B.forward(b);
        ck(cudaStreamEndCapture(L.s, &g), "end capture");
        ck(cudaGraphInstantiate(&L.graph, g, 0), "instantiate");
        cudaGraphDestroy(g);
    }
    // Inputs rotate over the dumped steps (outside the timed region); weights stay at the first snapshot.
    auto rotate = [&](int r) {
        for (int b = 0; b < B.nb; b++) {
            const Step& s = B.steps[(r * B.nb + b) % B.steps.size()];
            size_t slot_bytes = (size_t)BOPL4_PACK_P * sizeof(float);
            memcpy((char*)B.packed_host + (size_t)b * B.apb * slot_bytes, s.packed.data(), s.packed.size());
            memcpy((char*)B.host_rows + (size_t)b * B.apb * BOPL_OBS_STRIDE * sizeof(float), s.host_rows.data(), s.host_rows.size());
        }
    };
    for (int w = 0; w < 50; w++) for (int b = 0; b < B.nb; b++) { ck(cudaGraphLaunch(B.lanes[b].graph, B.lanes[b].s), "warm"); ck(cudaStreamSynchronize(B.lanes[b].s), "warm sync"); }
    std::vector<float> one, four, span, one_ready, four_ready;
    for (int r = 0; r < reps && streams != 4; r++) {
        rotate(r);
        Lane& L = B.lanes[0];
        cudaEventRecord(L.e0, L.s); cudaGraphLaunch(L.graph, L.s); cudaEventRecord(L.e1, L.s);
        ck(cudaStreamSynchronize(L.s), "sync");
        float ms; cudaEventElapsedTime(&ms, L.e0, L.e1); one.push_back(ms * 1000);
        cudaEventElapsedTime(&ms, L.e0, L.ready); one_ready.push_back(ms * 1000);
    }
    for (int r = 0; r < reps && streams != 1; r++) {
        rotate(r);
        for (int b = 0; b < B.nb; b++) { Lane& L = B.lanes[b]; cudaEventRecord(L.e0, L.s); cudaGraphLaunch(L.graph, L.s); cudaEventRecord(L.e1, L.s); }
        for (int b = 0; b < B.nb; b++) ck(cudaStreamSynchronize(B.lanes[b].s), "sync");
        float first = 0, last = 0;
        for (int b = 0; b < B.nb; b++) {
            float ms, a; cudaEventElapsedTime(&ms, B.lanes[b].e0, B.lanes[b].e1); four.push_back(ms * 1000);
            cudaEventElapsedTime(&ms, B.lanes[b].e0, B.lanes[b].ready); four_ready.push_back(ms * 1000);
            cudaEventElapsedTime(&a, B.lanes[0].e0, B.lanes[b].e0); first = std::min(first, a);
            cudaEventElapsedTime(&a, B.lanes[0].e0, B.lanes[b].e1); last = std::max(last, a);
        }
        span.push_back((last - first) * 1000);
    }
    printf("latency (us, GPU events around the step's graph, %d reps): 1 stream median %.1f p10 %.1f p90 %.1f | 4 streams at once: per stream median %.1f p10 %.1f p90 %.1f, all four done in median %.1f"
           " | actions ready: 1 stream median %.1f, 4 streams per stream median %.1f\n",
           reps, pct(one, 0.5), pct(one, 0.1), pct(one, 0.9), pct(four, 0.5), pct(four, 0.1), pct(four, 0.9), pct(span, 0.5),
           pct(one_ready, 0.5), pct(four_ready, 0.5));
    return 0;
}
