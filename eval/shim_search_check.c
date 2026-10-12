/* Offline proof that the in-game search (bopl_agent_search, train/bopl_agent.c,
 * docs/INGAME_SEARCH.md) plays exactly as the research library's search
 * (train/bopl_search.h on a stepped BoplPuffer, what libbopl_research.so's
 * research_search_act runs).
 *
 *   shim_search_check <q32 libbopl_sim.so> <policy-*.bin> [options]
 *     --players P      2 or 3 (FFA: one team per player)         (2)
 *     --budget B       k,n,depth[,h][,mix=M][,vs=V][,latched=0]  (8,1,1)
 *     --episodes N     episodes, maps and seeds below cycled     (4)
 *     --maps a,b,...   map ids                                   (0,3,36)
 *     --limit L        episode-tick limit for both sides          (the sidecar's)
 *     --short L        every other episode uses this limit, so rollouts and
 *                      the episode reach the cap                  (0 = off)
 *     --seed S         base seed                                  (1)
 *
 * The research side R is a P-player BoplPuffer on the q32 Rust sim with every
 * row layout v4, stepped tick by tick like research_step_begin/tick/finish;
 * the agent (the last player, where RlAgent appends it) is decided by
 * bopl_search_act on R. R's world stands in for the running game: each tick
 * its state S_e goes, in the RlObservation.cs wire format with RlAgent's
 * quirks (no telemetry), to a shadow (bopl_shadow_check_bytes) and to the
 * shim (bopl_agent_tick_bytes, search on that shadow); then R and the shadow
 * are stepped with every player's packet, the agent's being the shim's
 * output. The other players are "humans": the same checkpoint sampled with
 * the shortest hold (a per-tick input has no latch), so the shim's packet
 * mapping recovers their decisions exactly.
 *
 * Checked at every decision: the shim's choice equals R's search choice, the
 * agent's packet equals R's latched action, every row (all P, bitwise) and
 * every mask equals R's, a shim with the search attached whose shadow never
 * tracks (so it falls back at once, with every row v4) decides exactly as the
 * plain raw shim (own row, logits, choice), and the search shim's own row
 * matches the raw shim's outside its own-action and critic blocks. R's
 * truncated last record is not checked against the shadow (the game never
 * truncates). Also reported: per-decision wall time of the shim tick (search
 * included) and of R's search.
 * Exit status: 0 identical, 1 a mismatch, 2 usage. */
#include "../train/bopl_agent.c"
#include <time.h>

static double now_seconds(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec+1e-9*t.tv_nsec; }

/* RlObservation.cs record of `state` with RlAgent's quirks: telemetry is never
 * counted by the game (zeros). */
static size_t wire(const BoplRlState *state, unsigned char **buffer, size_t *capacity) {
    uint32_t P=state->header.player_count, E=state->header.entity_count, S=state->header.slime_count;
    size_t need=sizeof(BoplRlHeader)+P*sizeof(BoplRlPlayer)+E*sizeof(BoplRlEntity)+S*sizeof(BoplRlSlime)+P*sizeof(BoplRlTelemetry);
    if (need>*capacity) { *buffer=realloc(*buffer, need); *capacity=need; if (!*buffer) abort(); }
    unsigned char *p=*buffer;
    memcpy(p, &state->header, sizeof(BoplRlHeader)); p+=sizeof(BoplRlHeader);
    memcpy(p, state->players, P*sizeof(BoplRlPlayer)); p+=P*sizeof(BoplRlPlayer);
    if (E) memcpy(p, state->entities, E*sizeof(BoplRlEntity));
    p+=E*sizeof(BoplRlEntity);
    if (S) memcpy(p, state->slimes, S*sizeof(BoplRlSlime));
    p+=S*sizeof(BoplRlSlime);
    memset(p, 0, P*sizeof(BoplRlTelemetry));
    return need;
}

/* The BOPLREP2 setup RlAgent hands bopl_shadow_begin (ReplayArchive.Encode's
 * header; frames are not read). */
static size_t setup_archive(const BoplRlConfig *c, const char *source, unsigned char *out) {
    memset(out, 0, 512);
    memcpy(out, "BOPLREP2", 8);
    memcpy(out+8, source, strlen(source)<64?strlen(source):64);
    size_t at=136;
#define PUT(value, width) do { uint32_t v_=(uint32_t)(value); for (int b_=0; b_<(width); b_++) out[at++]=(unsigned char)(v_>>(8*b_)); } while (0)
    PUT(c->seed, 4); PUT(c->map, 1); PUT(c->mode, 1); PUT(c->stocks, 4); PUT(c->players, 1);
    for (uint32_t p=0; p<c->players; p++) {
        PUT(p+1, 1); PUT(c->player[p].color, 1); PUT(c->player[p].team, 1); PUT(c->player[p].keyboard, 1); PUT(c->slots, 1);
        for (uint32_t k=0; k<c->slots; k++) PUT(c->player[p].abilities[k], 2);
    }
#undef PUT
    return at<150?150:at;
}

static int usage(void) {
    fprintf(stderr, "usage: shim_search_check <q32 libbopl_sim.so> <policy-*.bin> [--players P] [--budget B] [--episodes N] [--maps a,b] [--limit L] [--short L] [--seed S]\n");
    return 2;
}

int main(int argc, char **argv) {
    if (argc<3) return usage();
    const char *lib=argv[1], *checkpoint=argv[2], *budget="8,1,1", *maps_text="0,3,36";
    int P=2, episodes=4, limit=0, short_limit=0; uint32_t base_seed=1;
    for (int i=3; i<argc; i++) {
        if (i+1>=argc) return usage();
        if (!strcmp(argv[i], "--players")) P=atoi(argv[++i]);
        else if (!strcmp(argv[i], "--budget")) budget=argv[++i];
        else if (!strcmp(argv[i], "--episodes")) episodes=atoi(argv[++i]);
        else if (!strcmp(argv[i], "--maps")) maps_text=argv[++i];
        else if (!strcmp(argv[i], "--limit")) limit=atoi(argv[++i]);
        else if (!strcmp(argv[i], "--short")) short_limit=atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")) base_seed=(uint32_t)strtoul(argv[++i], NULL, 10);
        else return usage();
    }
    if (P<2 || P>BOPL_MAX_PLAYERS) return usage();
    int maps[64], map_count=0;
    for (char *copy=strdup(maps_text), *t=strtok(copy, ","); t && map_count<64; t=strtok(NULL, ",")) maps[map_count++]=atoi(t);
    if (!map_count) return usage();
    setenv("BOPL_RL_GAME", lib, 1); setenv("BOPL_RL_BOOTSTRAP", "/tmp/bopl-shim-search-check.rep", 1);
    setenv("BOPL_RL_LOG_DIR", "/tmp", 1); setenv("BOPL_AGENT_SHADOW_LOG", "/tmp/bopl-shim-search-check-shadow.log", 1);
    int me=P-1;
    char error[1024];
    BoplAgent *agent=bopl_agent_open(checkpoint, P, me, error, sizeof(error));
    BoplAgent *raw=bopl_agent_open(checkpoint, P, me, error, sizeof(error));
    BoplAgent *fallback=bopl_agent_open(checkpoint, P, me, error, sizeof(error));
    if (!agent || !raw || !fallback) { fprintf(stderr, "open: %s\n", error); return 2; }
    if (!limit) limit=agent->limit;
    BoplShadow *shadow=bopl_shadow_open(lib, error, sizeof(error));
    if (!shadow) { fprintf(stderr, "shadow: %s\n", error); return 2; }
    if (bopl_agent_search(agent, shadow, budget, error, sizeof(error))) { fprintf(stderr, "search: %s\n", error); return 2; }
    BoplShadow *dead=bopl_shadow_open(lib, error, sizeof(error));   /* never begun: the fallback shim's */
    if (!dead || bopl_agent_search(fallback, dead, budget, error, sizeof(error))) { fprintf(stderr, "fallback: %s\n", error); return 2; }
    int k, n, depth, halving, latched; float mix, scale;
    bopl_agent_parse_budget(budget, &k, &n, &depth, &halving, &latched, &mix, &scale);

    /* R: the research environment, every row v4 with the checkpoint's bits. */
    BoplPuffer *R=calloc(1, sizeof(BoplPuffer));
    R->players=P; R->num_agents=P; R->repeat=agent->repeat; R->observation_delay=agent->delay; R->no_autoreset=1; R->rng=base_seed;
    R->reward_kill=0.1f; R->reward_timeout=0;
    bopl_rl_default_config(&R->config, (uint32_t)P);
    R->config.stocks=0; R->config.warmup_ticks=(uint32_t)agent->warmup;
    for (int p=0; p<P; p++) {
        for (int s=0; s<3; s++) R->config.player[p].abilities[s]=(uint32_t)agent->ability[s];
        R->layout[p]=BOPL4_LAYOUT_VERSION; bopl_set_ablate_of(R, p, agent->ablate);
    }
    R->observations=calloc((size_t)P*BOPL_OBS_STRIDE, sizeof(float)); R->actions=calloc((size_t)P*BOPL_NUM_ATNS, sizeof(float));
    R->rewards=calloc((size_t)P, sizeof(float)); R->terminals=calloc((size_t)P, sizeof(float)); R->action_mask=calloc((size_t)P*BOPL_LOGITS, 1);
    bopl_search_route(R);
    R->config.map=(uint32_t)maps[0]; R->config.max_episode_ticks=(uint32_t)limit;
    init(R); c_reset(R);
    BoplSearch *search=bopl_search_open(R, agent->checkpoint, agent->checkpoint_count, agent->hidden, agent->layers,
                                        agent->aim_fourier, agent->aim_ar, agent->shape, k*n, 1);
    if (!search || bopl_search_configure(search, k, n, depth, halving, latched, 0.995f, 0.2883693f, -1.0f, mix, scale)) { fprintf(stderr, "research search\n"); return 2; }
    /* The humans: the same checkpoint on their own rows, a batch-P evaluator. */
    void *human=bopl_policy_eval_create_v4(agent->checkpoint, agent->checkpoint_count, P, agent->hidden, agent->layers,
                                           agent->aim_fourier, agent->aim_ar, agent->shape);
    int width=bopl_policy_eval_width(human);
    size_t hstate=(size_t)agent->layers*P*agent->hidden;
    float *hs=calloc(hstate, sizeof(float)), *hn=calloc(hstate, sizeof(float));
    float *hrows=calloc((size_t)P*(width+1), sizeof(float)), *hlogits=calloc((size_t)P*BOPL_DECODER, sizeof(float));
    unsigned char *bytes=NULL; size_t capacity=0;
    long critic_diffs=0, decisions=0, searched_total=0, mismatches=0, row_diffs=0, mask_diffs=0, raw_diffs=0, truncated=0, capped_rollouts=0, shadow_bad=0;
    double shim_seconds=0, shim_max=0, research_seconds=0, research_max=0;
    float *shim_times=calloc(1<<20, sizeof(float)); long shim_samples=0;
    for (int episode=0; episode<episodes; episode++) {
        int map=maps[episode%map_count];
        int episode_limit=short_limit && episode%2?short_limit:limit;
        uint32_t seed=base_seed+(uint32_t)episode*7919u;
        /* R's episode (research_reset), then the shim's limit and map. */
        R->rng=seed; R->config.map=(uint32_t)map; R->config.max_episode_ticks=(uint32_t)episode_limit; c_reset(R);
        BoplAgent *both[3]={agent, raw, fallback};
        for (int w=0; w<3; w++) {
            both[w]->limit=episode_limit; both[w]->env.config.max_episode_ticks=(uint32_t)episode_limit;
            bopl_agent_set_map(both[w], (uint32_t)map);
            bopl_agent_reset(both[w], seed);
        }
        bopl_search_set_limit(agent->search, (uint32_t)episode_limit);
        bopl_search_reset(search, agent->search->rng);  /* the shim's round stream */
        memset(hs, 0, hstate*sizeof(float));
        uint64_t human_rng=(uint64_t)seed*0xD1B54A32D192ED03ull+17;
        unsigned char archive[512];
        size_t archive_length=setup_archive(&R->config, bopl_rl_source_hash(R->world), archive);
        if (bopl_shadow_begin(shadow, archive, archive_length, (uint32_t)agent->warmup)!=BOPL_SHADOW_TRACKING) {
            fprintf(stderr, "shadow begin: %s\n", bopl_shadow_report(shadow)); return 1;
        }
        BoplStepPhase phase; int open=0, ticks=0, sampled=0;
        long episode_decisions=0, episode_searched=0; int ended=0, failed=0;
        for (uint32_t e=0; !ended && !failed; e++) {
            const BoplRlState *state=bopl_rl_state(R->world);
            int terminal=(state->header.flags&BOPL_RL_TERMINAL)!=0;
            size_t length=wire(state, &bytes, &capacity);
            /* The game never truncates a round (RlAgent only logs the cap), so
             * R's truncated last record has no in-game counterpart: the shadow
             * is not checked against it, only the shims see it. */
            int status=(state->header.flags&BOPL_RL_TRUNCATED)?BOPL_SHADOW_TRACKING:bopl_shadow_check_bytes(shadow, bytes, length);
            if (status!=BOPL_SHADOW_TRACKING) { fprintf(stderr, "episode %d tick %u: shadow %s: %s\n", episode, e, bopl_shadow_status_name(status), bopl_shadow_report(shadow)); shadow_bad++; failed=1; break; }
            int decision=!terminal && !open;
            int out_r[3]={0, 0, 0}, searched=0;
            if (decision) {
                /* R decides: the agent by the research search, the humans by the policy (shortest hold). */
                double t0=now_seconds();
                if (bopl_search_act(search, me, out_r, &searched)) { fprintf(stderr, "research search: %s\n", search->error); return 1; }
                double dt=now_seconds()-t0; research_seconds+=dt; if (dt>research_max) research_max=dt;
                bopl_policy_eval_step_raw(human, R->observations, hs, hrows, hn);
                bopl_policy_eval_game_logits(human, hrows, NULL, hlogits);
                float *sw=hs; hs=hn; hn=sw;
                for (int p=0; p<P; p++) {
                    float *act=R->actions+(size_t)p*BOPL_NUM_ATNS;
                    if (p==me) { for (int h=0; h<3; h++) act[h]=(float)out_r[h]; continue; }
                    const float *row=hlogits+(size_t)p*BOPL_DECODER; const unsigned char *mask=R->action_mask+(size_t)p*BOPL_LOGITS;
                    act[0]=(float)bopl_sample_head(row, mask, BOPL_ACT_AIM, &human_rng);
                    act[1]=(float)bopl_sample_head(row+BOPL_ACT_AIM, mask+BOPL_ACT_AIM, BOPL_ACT_BUTTONS, &human_rng);
                    act[2]=0;
                }
                bopl_step_begin(R, &phase); open=1; ticks=0; sampled=0;
            }
            /* The game: RlAgent hands S_e to the shim (search on the shadow inside). */
            BoplRlAction applied_shim, applied_raw;
            double t0=now_seconds();
            int result=bopl_agent_tick_bytes(agent, bytes, length, &applied_shim);
            double dt=now_seconds()-t0;
            int result_raw=bopl_agent_tick_bytes(raw, bytes, length, &applied_raw);
            BoplRlAction applied_fallback;
            if (bopl_agent_tick_bytes(fallback, bytes, length, &applied_fallback)<0) result_raw=-1;
            if (result<0 || result_raw<0) { fprintf(stderr, "episode %d tick %u: shim tick %d / raw %d\n", episode, e, result, result_raw); return 1; }
            if (terminal) { ended=1; if (state->header.flags&BOPL_RL_TRUNCATED) truncated++; break; }
            if (!bopl_agent_search_active(agent)) { fprintf(stderr, "episode %d tick %u: the shim fell back: %s\n", episode, e, bopl_agent_search_note(agent)); failed=1; break; }
            if (decision) {
                decisions++; episode_decisions++; searched_total+=searched; episode_searched+=searched;
                shim_seconds+=dt; if (dt>shim_max) shim_max=dt;
                if (shim_samples<(1<<20)) shim_times[shim_samples++]=(float)dt;
                int out_s[3]; bopl_agent_decision(agent, out_s);
                if (memcmp(out_s, out_r, sizeof(out_s))) {
                    mismatches++;
                    fprintf(stderr, "episode %d map %d tick %u: shim chose %d/%d/%d, research %d/%d/%d\n", episode, map, e,
                            out_s[0], out_s[1], out_s[2], out_r[0], out_r[1], out_r[2]);
                    failed=1;
                }
                for (int p=0; p<P; p++) {
                    if (memcmp(agent->env.observations+(size_t)p*BOPL_OBS_STRIDE, R->observations+(size_t)p*BOPL_OBS_STRIDE, BOPL4_OBS_SIZE*sizeof(float))) {
                        if (!row_diffs) {
                            const float *x=agent->env.observations+(size_t)p*BOPL_OBS_STRIDE, *y=R->observations+(size_t)p*BOPL_OBS_STRIDE;
                            for (int f=0; f<BOPL4_OBS_SIZE; f++) if (memcmp(&x[f], &y[f], sizeof(float))) { fprintf(stderr, "episode %d tick %u: row %d feature %d shim %.9g research %.9g\n", episode, e, p, f, x[f], y[f]); break; }
                        }
                        row_diffs++; failed=1;
                    }
                    if (memcmp(agent->env.action_mask+(size_t)p*BOPL_LOGITS, R->action_mask+(size_t)p*BOPL_LOGITS, BOPL_LOGITS)) {
                        if (!mask_diffs) fprintf(stderr, "episode %d tick %u: mask %d differs\n", episode, e, p);
                        mask_diffs++; failed=1;
                    }
                }
                /* The raw path. A shim with the search attached whose shadow never
                 * tracks (it falls back at its first decision, every row v4,
                 * nothing mapped) decides exactly as the plain raw shim: same
                 * own row, logits, value and choice. */
                int out_f[3], out_w[3]; bopl_agent_decision(fallback, out_f); bopl_agent_decision(raw, out_w);
                if (memcmp(raw->env.observations+(size_t)me*BOPL_OBS_STRIDE, fallback->env.observations+(size_t)me*BOPL_OBS_STRIDE, BOPL4_OBS_SIZE*sizeof(float))
                    || memcmp(raw->decoder, fallback->decoder, BOPL_DECODER*sizeof(float)) || memcmp(out_f, out_w, sizeof(out_f))) {
                    if (!raw_diffs) fprintf(stderr, "episode %d tick %u: the fallen-back shim differs from the raw shim\n", episode, e);
                    raw_diffs++; failed=1;
                }
                /* The search shim's own row matches the raw shim's in everything
                 * the state alone decides: its own-action tail (its own, searched
                 * choices) and the critic block (the humans' real inputs) differ. */
                const float *xr=raw->env.observations+(size_t)me*BOPL_OBS_STRIDE, *xs=agent->env.observations+(size_t)me*BOPL_OBS_STRIDE;
                if (memcmp(xr, xs, BOPL4_OBS_PREVIOUS_OFFSET*sizeof(float))
                    || memcmp(xr+BOPL4_OBS_TERRAIN_OFFSET, xs+BOPL4_OBS_TERRAIN_OFFSET, (BOPL4_OBS_PRIVILEGED_OFFSET-BOPL4_OBS_TERRAIN_OFFSET)*sizeof(float))) {
                    if (!raw_diffs) fprintf(stderr, "episode %d tick %u: the search shim's state features differ from the raw shim's\n", episode, e);
                    raw_diffs++; failed=1;
                }
                if (memcmp(xr+BOPL4_OBS_PRIVILEGED_OFFSET, xs+BOPL4_OBS_PRIVILEGED_OFFSET, (BOPL4_OBS_SIZE-BOPL4_OBS_PRIVILEGED_OFFSET)*sizeof(float))) critic_diffs++;
            }
            if (applied_shim.angle!=phase.actions[me].angle || applied_shim.buttons!=phase.actions[me].buttons) {
                fprintf(stderr, "episode %d tick %u: shim packet %u/%u, research latched %u/%u\n", episode, e,
                        applied_shim.angle, applied_shim.buttons, phase.actions[me].angle, phase.actions[me].buttons);
                mismatches++; failed=1; break;
            }
            /* One tick of the game and of the shadow with every player's packet. */
            if (bopl_rl_step(R->world, phase.actions, (size_t)P, 1)) { fprintf(stderr, "step: %s\n", bopl_rl_error(R->world)); return 1; }
            if (bopl_shadow_step(shadow, phase.actions, (size_t)P)!=BOPL_SHADOW_TRACKING) { fprintf(stderr, "shadow step: %s\n", bopl_shadow_report(shadow)); shadow_bad++; failed=1; break; }
            ticks++;
            const BoplRlState *after=bopl_rl_state(R->world);
            int done=(after->header.flags&BOPL_RL_TERMINAL)!=0;
            if (R->observation_delay && !sampled && (ticks==R->repeat-1 || done)) { bopl_step_sample(R, after); sampled=1; }
            if (ticks>=R->repeat || done) { bopl_step_finish(R, &phase); open=0; }
        }
        const BoplRlState *last=bopl_rl_state(R->world);
        if (ended) bopl_shadow_end(shadow);
        printf("episode %d map %d limit %d: %u ticks%s, %ld decisions (%ld searched), kills %u/%u%s%s\n", episode, map, episode_limit,
               last->header.episode_ticks, (last->header.flags&BOPL_RL_TRUNCATED)?" (truncated)":"", episode_decisions, episode_searched,
               last->players[0].kills, last->players[me].kills, failed?" FAILED":"", ended?"":" (stopped)");
        fflush(stdout);
        if (failed) break;
    }
    (void)capped_rollouts;
    qsort(shim_times, (size_t)shim_samples, sizeof(float), bopl_float_order);
    double p50=shim_samples?shim_times[(shim_samples-1)/2]:0, p95=shim_samples?shim_times[(long)((shim_samples-1)*0.95)]:0;
    printf("players %d budget %s: %ld decisions (%ld searched), %ld choice mismatches, %ld row diffs, %ld mask diffs, %ld raw-path diffs, %ld shadow failures, %ld truncated episodes\n",
           P, budget, decisions, searched_total, mismatches, row_diffs, mask_diffs, raw_diffs, shadow_bad, truncated);
    printf("shim tick at a decision (search + raw forward + encode): mean %.2f ms, p50 %.2f, p95 %.2f, max %.2f; research search mean %.2f ms, max %.2f\n",
           1e3*shim_seconds/(decisions?decisions:1), 1e3*p50, 1e3*p95, 1e3*shim_max, 1e3*research_seconds/(decisions?decisions:1), 1e3*research_max);
    printf("raw path: the fallen-back shim (every row v4) played as the plain raw shim; the search shim's critic block differed from the raw shim's at %ld of %ld decisions (the humans' mapped inputs); fallback note: %s\n",
           critic_diffs, decisions, bopl_agent_search_note(fallback));
    int ok=decisions>0 && !mismatches && !row_diffs && !mask_diffs && !raw_diffs && !shadow_bad;
    printf("%s\n", ok?"IDENTICAL":"MISMATCH");
    bopl_search_close(search); bopl_policy_eval_close(human);
    bopl_agent_close(agent); bopl_agent_close(raw); bopl_agent_close(fallback); bopl_shadow_close(shadow); bopl_shadow_close(dead);
    return ok?0:1;
}
