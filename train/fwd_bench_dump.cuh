// The trainer hook that writes fwd_bench's dumps. Not built by anything: to
// take a dump, include it in the generated overlay's pufferlib.cu right after
// `#include "bopl_rollout.cu"`, and wrap the replay in net_callback_wrapper's
// rollout_captured branch:
//
//     std::string brd_path; bool brd = br_dump_want(pufferl, buf);
//     if (brd) br_dump_pre(pufferl, buf, t, current_stream, brd_path);
//     ...cudaGraphLaunch(...)...
//     if (brd) br_dump_post(pufferl, buf, t, current_stream, brd_path);
//
// then build _C without regenerating the overlay and run the trainer with
// BOPL_FWD_DUMP=<dir> (buffer 0, every BOPL_FWD_DUMP_EVERY-th replayed step
// after BOPL_FWD_DUMP_SKIP, up to BOPL_FWD_DUMP_N steps). Weight snapshots
// are keyed by their bytes, not the epoch: the learner publishes
// asynchronously, so the epoch does not say which weights a step used.
#include <sys/stat.h>
#include <atomic>
#include <string>
static std::string brd_dir; static int brd_every = 7, brd_skip = 300, brd_n = 120, brd_init = 0;
static std::atomic<int> brd_calls{0}, brd_done{0};
static std::string brd_wdir;
static void brd_write(const std::string& path, const void* p, size_t bytes, bool dev) {
    std::vector<char> h(bytes);
    if (dev) { if (cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { fprintf(stderr, "brd copy %s failed\n", path.c_str()); abort(); } }
    else memcpy(h.data(), p, bytes);
    FILE* f = fopen(path.c_str(), "wb"); fwrite(h.data(), 1, bytes, f); fclose(f);
}
static bool br_dump_want(PuffeRL* pufferl, int buf) {
    if (!brd_init) {
        brd_init = 1;
        const char* d = getenv("BOPL_FWD_DUMP"); if (d && *d) brd_dir = d;
        brd_every = br_env_int("BOPL_FWD_DUMP_EVERY", 7); brd_skip = br_env_int("BOPL_FWD_DUMP_SKIP", 300); brd_n = br_env_int("BOPL_FWD_DUMP_N", 120);
    }
    if (brd_dir.empty() || buf != 0 || !pufferl->br || !pufferl->br->v4) return false;
    int c = brd_calls++;
    return c >= brd_skip && (c - brd_skip) % brd_every == 0 && brd_done.load() < brd_n;
}
static void br_dump_pre(PuffeRL* pufferl, int buf, int t, cudaStream_t stream, std::string& dir) {
    BrPlan& p = *pufferl->br; Br4Plan& q = *p.v4;
    StaticVec* vec = pufferl->vec;
    int apb = vec->total_agents / pufferl->hypers.num_buffers, start = buf * apb, rows = p.rows, H = p.H;
    cudaStreamSynchronize(stream);
    int k = brd_done.load();
    char name[64]; snprintf(name, sizeof(name), "/s%03d", k); dir = brd_dir + name;
    mkdir(brd_dir.c_str(), 0755); mkdir(dir.c_str(), 0755);
    // weights: a new snapshot directory whenever their bytes change (the learner's at each rollout start)
    struct Arr { std::string name; const void* p; size_t bytes; };
    std::vector<Arr> arrs;
    for (int g = 0; g < p.g.n; g++) {
        Br4Wt& w = q.wt[g];
        std::string pre = "/g" + std::to_string(g) + "_";
        arrs.push_back({pre + "flat", w.flat, (size_t)br_frag_floats(v4::FLAT, q.FD) * 4});
        arrs.push_back({pre + "fusion", w.fusion, (size_t)br_frag_floats(q.JW, H) * 4});
        for (int l = 0; l < p.layers; l++) arrs.push_back({pre + "gru" + std::to_string(l), w.gru[l], (size_t)br_frag_floats(H, 3 * H) * 4});
        if (q.F) { arrs.push_back({pre + "ffn_in", w.ffn_in, (size_t)br_frag_floats(H, q.F) * 4}); arrs.push_back({pre + "ffn_out", w.ffn_out, (size_t)br_frag_floats(q.F, H) * 4}); }
        arrs.push_back({pre + "actor", w.actor, (size_t)br_frag_floats(H, p.ld_dec) * 4});
        if (q.C) {
            arrs.push_back({pre + "cin", w.cin, (size_t)br_frag_floats(q.W, q.C) * 4});
            arrs.push_back({pre + "b1", w.b1, (size_t)br_frag_floats(q.C, q.C) * 4});
            arrs.push_back({pre + "b2", w.b2, (size_t)br_frag_floats(q.C, q.C) * 4});
        }
        for (int s = 0; s < 4; s++) {
            arrs.push_back({pre + "w1t" + std::to_string(s), w.w1t[s], (size_t)v4::SPEC[s].features * q.I * 2});
            arrs.push_back({pre + "w2b" + std::to_string(s), w.w2b[s], (size_t)q.D * q.I * 2});
        }
        PolicyWeights* pw; br_group_weights(pufferl, g, buf, &pw, nullptr);
        auto* ew = (v4::EncWeights*)pw->encoder; auto* dw = (v4::DecWeights*)pw->decoder;
        arrs.push_back({pre + "emb", ew->embedding.data, (size_t)numel(ew->embedding.shape) * 4});
        if (q.C) arrs.push_back({pre + "cout", dw->critic_out.data, (size_t)numel(dw->critic_out.shape) * 4});
    }
    std::vector<std::vector<char>> host(arrs.size());
    unsigned long long hash = 1469598103934665603ull;
    for (size_t i = 0; i < arrs.size(); i++) {
        host[i].resize(arrs[i].bytes);
        cudaMemcpy(host[i].data(), arrs[i].p, arrs[i].bytes, cudaMemcpyDeviceToHost);
        for (char c : host[i]) hash = (hash ^ (unsigned char)c) * 1099511628211ull;
    }
    static unsigned long long last_hash = 0; static int snapshots = 0;
    if (hash != last_hash) {
        last_hash = hash;
        brd_wdir = brd_dir + "/w" + std::to_string(snapshots++);
        mkdir(brd_wdir.c_str(), 0755);
        for (size_t i = 0; i < arrs.size(); i++) brd_write(brd_wdir + arrs[i].name, host[i].data(), host[i].size(), false);
    }
    std::string wdir = brd_wdir;
    FILE* m = fopen((dir + "/meta").c_str(), "w");
    fprintf(m, "weights %s\nrows %d\napb %d\nstart %d\nt %d\nH %d\nlayers %d\ndec_cols %d\nld_dec %d\nA_total %d\nnum_atns %d\nmask_stride %d\n"
               "aim_fourier %d\naim_ar %d\npack_in_tail %d\npack_bytes %d\nD %d\nI %d\nFD %d\nF %d\nC %d\nP %d\nmean_pool %d\nJW %d\nW %d\nLNW %d\ngroups %d\n",
            wdir.c_str(), rows, apb, start, t, H, p.layers, p.dec_cols, p.ld_dec, p.A_total, p.num_atns, p.mask_stride, p.aim_fourier, p.aim_ar,
            (int)vec->pack_in_tail, vec->pack_bytes, q.D, q.I, q.FD, q.F, q.C, q.P, q.mean_pool, q.JW, q.W, q.LNW, p.g.n);
    for (int g = 0; g <= p.g.n; g++) fprintf(m, "end%d %d\n", g, p.g.end[g]);
    for (int g = 0; g < p.g.n; g++) { PrecisionTensor* s; PolicyWeights* pw; br_group_weights(pufferl, g, buf, &pw, &s); fprintf(m, "state_rows%d %d\nstate_numel%d %ld\n", g, (int)s->shape[1], g, (long)numel(s->shape)); }
    int hc_stride = 0; const signed char* hc = get_head_consume_dev(&hc_stride);
    int nv = 0, na = 0; const signed char* hch = (hc && env_head_consume_map) ? env_head_consume_map(&nv, &na) : nullptr;
    fprintf(m, "hc_rows %d\nhc_stride %d\n", hch ? nv : 0, hc ? hc_stride : 0);
    fprintf(m, "frag 4\n");   // the GEMM weights in br_frag_pack_lane's order (fwd_bench converts the older orders 1 (no key) to 3)
    fprintf(m, "setw 2\n");   // the sets' W2 in br4_w2_index's order (fwd_bench converts the older (D, I) order, 1 (no key))
    fprintf(m, "slot_words %d\n", (int)BOPL4_PACK_P);   // the packed slots' stride (fwd_bench restrides; no key: 2048)
    fclose(m);
    if (hch) brd_write(dir + "/head_consume", hch, (size_t)nv * na, false);
    brd_write(dir + "/act_sizes", pufferl->act_sizes_puf.data, (size_t)p.num_atns * 4, true);
    size_t slot_bytes = (size_t)BOPL4_PACK_P * 4;
    brd_write(dir + "/packed", (const char*)vec->packed_host + (size_t)start * slot_bytes, (size_t)rows * slot_bytes + (vec->pack_in_tail ? vec->pack_bytes : 0), false);
    if (!vec->pack_in_tail) brd_write(dir + "/pack", vec->pack_host[buf], vec->pack_bytes, false);
    brd_write(dir + "/host_rows", (const float*)vec->observations + (size_t)start * BOPL_OBS_STRIDE, (size_t)rows * BOPL_OBS_STRIDE * 4, false);
    brd_write(dir + "/rng_pre", pufferl->rng_states[buf], (size_t)rows * sizeof(curandStatePhilox4_32_10_t), true);
    for (int g = 0; g < p.g.n; g++) {
        PrecisionTensor* s; PolicyWeights* pw; br_group_weights(pufferl, g, buf, &pw, &s);
        brd_write(dir + "/state_pre" + std::to_string(g), s->data, (size_t)numel(s->shape) * 4, true);
    }
}
static void br_dump_post(PuffeRL* pufferl, int buf, int t, cudaStream_t stream, const std::string& dir) {
    BrPlan& p = *pufferl->br; Br4Plan& q = *p.v4;
    StaticVec* vec = pufferl->vec;
    HypersT& hypers = pufferl->hypers;
    int apb = vec->total_agents / hypers.num_buffers, start = buf * apb, rows = p.rows;
    cudaStreamSynchronize(stream);
    RolloutBuf rollouts = rollout_slot_view(pufferl->rollouts, pufferl->write_slot, hypers.horizon);
    brd_write(dir + "/out_obs", puf_slice(rollouts.observations, t, start, rows).data, (size_t)rows * BOPL_OBS_STRIDE * 4, true);
    brd_write(dir + "/out_rewards", puf_slice(rollouts.rewards, t, start, rows).data, (size_t)rows * 4, true);
    brd_write(dir + "/out_terminals", puf_slice(rollouts.terminals, t, start, rows).data, (size_t)rows * 4, true);
    brd_write(dir + "/out_actions", puf_slice(rollouts.actions, t, start, rows).data, (size_t)rows * p.num_atns * 4, true);
    brd_write(dir + "/out_logprobs", puf_slice(rollouts.logprobs, t, start, rows).data, (size_t)rows * 4, true);
    brd_write(dir + "/out_values", puf_slice(rollouts.values, t, start, rows).data, (size_t)rows * 4, true);
    if (p.mask_stride) brd_write(dir + "/out_mask", puf_slice(rollouts.action_mask, t, start, rows).data, (size_t)rows * p.mask_stride * 4, true);
    brd_write(dir + "/out_env_actions", vec->actions + (size_t)start * p.num_atns, (size_t)rows * p.num_atns * 4, false);
    brd_write(dir + "/out_dec", q.acts[buf].dec, (size_t)rows * p.ld_dec * 4, true);
    brd_write(dir + "/out_joined", q.acts[buf].joined, (size_t)rows * q.JW * 4, true);
    brd_write(dir + "/rng_post", pufferl->rng_states[buf], (size_t)rows * sizeof(curandStatePhilox4_32_10_t), true);
    for (int g = 0; g < p.g.n; g++) {
        PrecisionTensor* s; PolicyWeights* pw; br_group_weights(pufferl, g, buf, &pw, &s);
        brd_write(dir + "/state_post" + std::to_string(g), s->data, (size_t)numel(s->shape) * 4, true);
    }
    int k = ++brd_done;
    fprintf(stderr, "bopl fwd dump: %s (t %d), %d of %d\n", dir.c_str(), t, k, brd_n);
}
