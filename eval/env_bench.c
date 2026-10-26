/* The trainer's per-env step cost on the state distribution a trained policy
 * visits: a corpus of BOPLREP2 games (native/artifacts/rl/tp-spec/
 * record_corpus.py: generalist-111 against itself, Matthew's 15 kits, the 111
 * training maps, 3-minute games, delay 9) replayed through c_step's phases on
 * one core, each phase timed on its own:
 *   reset    bopl_rl_reset (the 240 warmup ticks included), per episode
 *   sim      bopl_rl_step_sampled(world, actions, 2, 3, 1): one 4-tick decision
 *   sample   the phase-3 state copy + bopl_count_overflow
 *   lite     bopl_encode of the delayed state, writer bits 10 (aim/LOS casts)
 *   full     the same state, writer bits 0 (terrain block + 32-ray fan + aim)
 *   tail     previous action + v4 history + action mask, both players
 *   priv     bopl_encode_privileged from the undelayed state
 *   pack     bopl_pack_rows of the lite rows
 *   train    the trainer's full-writer finish (bopl_step_finish's writer
 *            calls in its order): the full encode deferred (bopl4_defer)
 *            with the overflow telemetry counted, which ships each row
 *            whose state fits as that state (pack mode 4) for the GPU
 *            writer (bopl_rays_v4.cu) and writes the rest on the CPU
 *            with their ray scene (mode 2), then the tail, the privileged
 *            block and bopl_pack_rows
 * lite and full each write into their own env (their own written-row records)
 * and swap order every decision. The replay's per-tick inputs are held for a
 * decision (the trainer's repeat 4); groups whose ticks disagree are counted.
 * One JSON line per game (map, kits, decisions, us per phase per decision),
 * then a summary over the corpus and per-map / per-kit aggregates go to the
 * python side. load1 travels with every line. Every game's hash folds each
 * decision's sampled and final states (replay_ab.c's strict state hash) and
 * both writers' rows (hash), the train phase's packed slots and masks
 * (train_hash) and its overflow telemetry (telemetry_hash, the counters):
 * equal hashes between two builds = the exact gate. A change to what a slot
 * ships for the same row moves train_hash alone; eval/encode_check.c then
 * proves the GPU's rows from it equal to the CPU writer's.
 * Each phase's user-space instructions per decision (<phase>_inst, from
 * perf_event_open; -1 when unavailable) go in the summary: deterministic,
 * the evidence for a writer change when wall time is noisy.
 * BOPL_ENV_BENCH_STYLE=<style.bin> (train/style_compile.py -o) adds the
 * style phase: bopl_step_finish's style call (bopl_style_events +
 * bopl_style_reward) for both players on every decision's final state,
 * history slots zeroed per game; style_pct is its share of env_train_us
 * (reset + sim + sample + train: the trainer's c_step), style_inst_pct of
 * the same phases' instructions (docs/STYLES.md: < 2%
 * with three styles). The payouts fold into style_hash, outside the timer.
 *
 *   env_bench <libbopl_sim.so> <corpus_dir> [max_games=0 (all)] [warmup=240] [limit=10800]
 *   BOPL_ENV_BENCH_ABLATE=<bits> (default 0): the writers' bits (lite: 10 | bits)
 *
 * Pin and nice it (taskset -c 11 nice -n 19). Build: make env-bench. */
#define _GNU_SOURCE
#include "bopl_puffer.h"
#include <dirent.h>
#include <inttypes.h>
#include <time.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>

static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec+t.tv_nsec/1e9;
}
/* This thread's user-space instruction count (0 without perf access). */
static int inst_fd=-2;
static uint64_t instructions(void) {
    if (inst_fd==-2) {
        struct perf_event_attr attr; memset(&attr, 0, sizeof(attr));
        attr.size=sizeof(attr); attr.type=PERF_TYPE_HARDWARE; attr.config=PERF_COUNT_HW_INSTRUCTIONS;
        attr.exclude_kernel=1; attr.exclude_hv=1;
        inst_fd=(int)syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
    }
    uint64_t v=0;
    if (inst_fd<0 || read(inst_fd, &v, sizeof(v))!=sizeof(v)) return 0;
    return v;
}
static double load1(void) { double l[1]; return getloadavg(l, 1)==1?l[0]:-1.0; }

typedef struct { BoplRlConfig config; uint32_t tick_count; BoplRlAction *actions; } Replay;
typedef struct { const unsigned char *data; size_t size, offset; int failed; } Cursor;
static uint32_t take_word(Cursor *c, int bytes) {
    if (c->offset+(size_t)bytes>c->size) { c->failed=1; return 0; }
    uint32_t v=0; for (int i=0; i<bytes; i++) v|=(uint32_t)c->data[c->offset+(size_t)i]<<(8*i);
    c->offset+=(size_t)bytes; return v;
}
static uint32_t take_byte(Cursor *c) { return take_word(c, 1); }
/* eval/rust_ab/replay_ab.c's decoder. */
static int replay_decode(const char *path, uint32_t warmup, uint32_t limit, Replay *out) {
    FILE *file=fopen(path, "rb");
    if (!file) { fprintf(stderr, "%s: cannot open\n", path); return 1; }
    fseek(file, 0, SEEK_END); long size=ftell(file); fseek(file, 0, SEEK_SET);
    if (size<140) { fclose(file); fprintf(stderr, "%s: too short\n", path); return 1; }
    unsigned char *raw=malloc((size_t)size);
    if (!raw || fread(raw, 1, (size_t)size, file)!=(size_t)size) { fclose(file); free(raw); return 1; }
    fclose(file);
    if (memcmp(raw, "BOPLREP2", 8)) { free(raw); fprintf(stderr, "%s: not BOPLREP2\n", path); return 1; }
    Cursor c={raw, (size_t)size, 136, 0};
    memset(out, 0, sizeof(*out));
    bopl_rl_default_config(&out->config, 2);
    out->config.seed=take_word(&c, 4); out->config.map=take_byte(&c); out->config.mode=take_byte(&c);
    out->config.stocks=take_word(&c, 4);
    uint32_t players=take_byte(&c);
    if (players!=2) { free(raw); fprintf(stderr, "%s: %u players, the bench is 1v1\n", path, players); return 1; }
    out->config.players=players; out->config.warmup_ticks=warmup; out->config.max_episode_ticks=limit;
    for (uint32_t p=0; p<players; p++) {
        take_byte(&c);
        out->config.player[p].color=take_byte(&c); out->config.player[p].team=take_byte(&c); out->config.player[p].keyboard=take_byte(&c);
        uint32_t slots=take_byte(&c);
        if (slots<1 || slots>3) { free(raw); return 1; }
        memset(out->config.player[p].abilities, 0, sizeof(out->config.player[p].abilities));
        for (uint32_t s=0; s<slots; s++) out->config.player[p].abilities[s]=take_word(&c, 2);
        out->config.slots=slots;
    }
    out->tick_count=take_word(&c, 4);
    if (c.failed || out->tick_count>1000000u) { free(raw); return 1; }
    out->actions=calloc((size_t)out->tick_count*players, sizeof(BoplRlAction));
    for (uint32_t t=0; t<out->tick_count; t++) for (uint32_t p=0; p<players; p++) {
        take_word(&c, 4);
        BoplRlAction *a=&out->actions[(size_t)t*players+p];
        a->angle=(uint8_t)take_byte(&c); a->reserved=0; a->buttons=(uint16_t)take_word(&c, 2); take_byte(&c);
    }
    free(raw);
    if (c.failed) { free(out->actions); return 1; }
    return 0;
}
/* replay_ab.c's state hash (strict): every decision's sampled and final
 * state and both writers' rows fold into the game's hash, the exact gate
 * between two sim builds or writer builds. */
static uint64_t mix(uint64_t h, uint64_t v) { h^=v; h*=0x9E3779B97F4A7C15ull; return h^(h>>29); }
static uint64_t mix_words(uint64_t h, const void *data, size_t bytes) {
    const unsigned char *p=data;
    for (; bytes>=8; p+=8, bytes-=8) { uint64_t v; memcpy(&v, p, 8); h=mix(h, v); }
    if (bytes) { uint64_t v=0; memcpy(&v, p, bytes); h=mix(h, v); }
    return h;
}
static uint64_t state_hash(uint64_t h, const BoplRlState *s) {
    const BoplRlHeader *hd=&s->header;
    h=mix(h, hd->player_count); h=mix(h, hd->entity_count); h=mix(h, hd->slime_count);
    h=mix(h, hd->flags); h=mix(h, hd->episode_ticks); h=mix(h, hd->sudden_death);
    h=mix(h, hd->warmup_ticks); h=mix(h, (uint64_t)hd->simulation_time);
    h=mix(h, (uint64_t)hd->player_time_scale); h=mix(h, (uint64_t)hd->simulation_time_since_level_loaded);
    for (uint32_t i=0; i<hd->player_count; i++) { h=mix_words(h, &s->players[i], sizeof(s->players[i])); h=mix_words(h, &s->telemetry[i], sizeof(s->telemetry[i])); }
    h=mix_words(h, s->slimes, (size_t)hd->slime_count*sizeof(*s->slimes));
    return mix_words(h, s->entities, (size_t)hd->entity_count*sizeof(*s->entities));
}
static int cmp_name(const void *a, const void *b) { return strcmp(*(char *const*)a, *(char *const*)b); }

enum { PLAYERS=2, PHASES=10 };
enum { P_RESET, P_SIM, P_SAMPLE, P_LITE, P_FULL, P_TAIL, P_PRIV, P_PACK, P_TRAIN, P_STYLE };
static const char *PHASE_NAMES[PHASES]={"reset", "sim", "sample", "lite", "full", "tail", "priv", "pack", "train", "style"};

static void env_init(BoplPuffer *env, BoplRl *world, float *scratch, float *pack_out, unsigned char (*masks)[BOPL_LOGITS], int ablate) {
    memset(env, 0, sizeof(*env));
    env->players=PLAYERS; env->num_agents=PLAYERS; env->scratch=scratch; env->world=world;
    env->pack_base=scratch; env->pack_out=pack_out; env->pack_row_begin=0; env->pack_row_end=PLAYERS;
    for (int i=0; i<PLAYERS; i++) { env->layout[i]=BOPL4_LAYOUT_VERSION; bopl_set_ablate_of(env, i, ablate); env->mask_ptr[i]=masks[i]; }
}

#ifndef ENV_BENCH_NO_MAIN   /* eval/encode_check.c reuses the decoder */
int main(int argc, char **argv) {
    if (argc<3 || argc>6) { fprintf(stderr, "env_bench <libbopl_sim.so> <corpus_dir> [max_games] [warmup] [limit]\n"); return 2; }
    int max_games=argc>3?atoi(argv[3]):0;
    uint32_t warmup=argc>4?(uint32_t)atoi(argv[4]):240, limit=argc>5?(uint32_t)atoi(argv[5]):10800;
    DIR *dir=opendir(argv[2]);
    if (!dir) { fprintf(stderr, "%s: cannot open\n", argv[2]); return 1; }
    char **names=NULL; int n=0, cap=0; struct dirent *e;
    while ((e=readdir(dir))) {
        size_t len=strlen(e->d_name);
        if (len<5 || strcmp(e->d_name+len-4, ".rep")) continue;
        if (n==cap) { cap=cap?cap*2:256; names=realloc(names, sizeof(*names)*(size_t)cap); }
        names[n++]=strdup(e->d_name);
    }
    closedir(dir);
    qsort(names, (size_t)n, sizeof(*names), cmp_name);
    if (max_games>0 && n>max_games) n=max_games;
    if (!n) { fprintf(stderr, "no .rep files in %s\n", argv[2]); return 1; }

    char error[512], log[64]; snprintf(log, sizeof(log), "/tmp/env-bench-%d.log", (int)getpid());
    BoplRl *world=bopl_rl_open(argv[1], "/dev/null", log, 30000, error, sizeof(error));
    if (!world) { fprintf(stderr, "open failed: %s\n", error); return 1; }

    static float scratch_lite[PLAYERS*BOPL_OBS_STRIDE], scratch_full[PLAYERS*BOPL_OBS_STRIDE];
    static float scratch_train[PLAYERS*BOPL_OBS_STRIDE];
    static float pack_out[PLAYERS*BOPL4_PACK_P], pack_train[PLAYERS*BOPL4_PACK_P];
    static unsigned char masks_lite[PLAYERS][BOPL_LOGITS], masks_full[PLAYERS][BOPL_LOGITS], masks_train[PLAYERS][BOPL_LOGITS];
    BoplPuffer *lite=calloc(1, sizeof(BoplPuffer)), *full=calloc(1, sizeof(BoplPuffer)), *train=calloc(1, sizeof(BoplPuffer));
    /* BOPL_ENV_BENCH_ABLATE=<bits>: the full and train writers at these
     * writer bits (default 0), the lite one at them with the terrain and rays
     * off and the aim casts on (10 | bits), so a writer revision's bit-off
     * gate covers rows written with the other revisions on (16, 48). */
    const char *ablate_env=getenv("BOPL_ENV_BENCH_ABLATE");
    int bits=ablate_env&&ablate_env[0]?atoi(ablate_env):0;
    env_init(lite, world, scratch_lite, pack_out, masks_lite, 10|bits);
    env_init(full, world, scratch_full, pack_out, masks_full, bits);
    env_init(train, world, scratch_train, pack_train, masks_train, bits);
    BoplStateCopy delayed[2]={0}, sample={0};

    /* BOPL_ENV_BENCH_EVICT_KB: between decisions, outside the timers, sweep
     * a buffer this large through the cache, as the trainer's env threads do
     * when they step the pool's other worlds between two of this one's. */
    /* BOPL_ENV_BENCH_SIM_ONLY skips the writers, so `perf stat` counts the
     * sim's instructions undiluted (the hash then covers the states alone). */
    int sim_only=getenv("BOPL_ENV_BENCH_SIM_ONLY")!=NULL;
    /* BOPL_ENV_BENCH_TRAIN_ONLY skips the lite and full writers and the lite
     * env's tail, privileged block and pack, so a profile of the run shows
     * the trainer's writer path (sample + train) next to the sim alone (the
     * hash then covers the states and the train phase). */
    int train_only=getenv("BOPL_ENV_BENCH_TRAIN_ONLY")!=NULL;
    const char *evict_env=getenv("BOPL_ENV_BENCH_EVICT_KB");
    size_t evict_bytes=evict_env?(size_t)atol(evict_env)*1024:0;
    unsigned char *evict=evict_bytes?calloc(1, evict_bytes):NULL;
    const char *style_path=getenv("BOPL_ENV_BENCH_STYLE");
    static BoplStyleProgram style_program;
    BoplPuffer *styled=NULL;
    if (style_path && *style_path) {
        FILE *f=fopen(style_path, "rb");
        if (!f) { fprintf(stderr, "%s: cannot open\n", style_path); return 1; }
        fseek(f, 0, SEEK_END); long len=ftell(f); fseek(f, 0, SEEK_SET);
        void *bytes=malloc((size_t)len);
        if (fread(bytes, 1, (size_t)len, f)!=(size_t)len) { fprintf(stderr, "%s: short read\n", style_path); return 1; }
        fclose(f);
        if (bopl_style_load(&style_program, bytes, (size_t)len, error, sizeof(error))) { fprintf(stderr, "%s: %s\n", style_path, error); return 1; }
        free(bytes);
        styled=train;   /* the trainer's env: its style pointer, slots and payout counters */
        styled->style=&style_program;
        styled->style_slots=calloc((size_t)PLAYERS*style_program.hist_floats+1, sizeof(float));
    }
    uint64_t style_hash=0x243F6A8885A308D3ull, style_entities=0;
    /* What one timed block costs with nothing in it (two clock reads and two
     * counter reads): every phase carries it once per decision; the style
     * phase is small enough for it to matter, so it is reported net too. */
    double timer_s=0.0; uint64_t timer_inst=0;
    for (int k=0; k<4096; k++) {
        double a=seconds(); uint64_t ia=instructions();
        timer_s+=seconds()-a; timer_inst+=instructions()-ia;
    }
    timer_s/=4096.0; timer_inst/=4096;
    double total[PHASES]={0}; uint64_t total_inst[PHASES]={0}, total_fallback=0, total_decisions=0, total_ticks=0, total_split=0, corpus_hash=0, corpus_train=0, corpus_telemetry=0;
    double wall=seconds();
    for (int g=0; g<n; g++) {
        char path[4096]; snprintf(path, sizeof(path), "%s/%s", argv[2], names[g]);
        Replay replay;
        if (replay_decode(path, warmup, limit, &replay)) return 1;
        if (replay.tick_count<=warmup) { fprintf(stderr, "%s: shorter than the warmup\n", path); return 1; }
        double t[PHASES]={0}, t0=seconds(); uint64_t ic[PHASES]={0}, i0=instructions();
        if (bopl_rl_reset(world, &replay.config)!=BOPL_RL_OK) { fprintf(stderr, "%s: reset failed: %s\n", path, bopl_rl_error(world)); return 1; }
        t[P_RESET]=seconds()-t0; ic[P_RESET]=instructions()-i0;
        lite->config=full->config=train->config=replay.config;
        bopl_written_forget(lite); bopl_written_forget(full); bopl_written_forget(train);
        if (styled) memset(styled->style_slots, 0, (size_t)PLAYERS*style_program.hist_floats*sizeof(float));
        memset(lite->action_history, 0, sizeof(lite->action_history)); memset(full->action_history, 0, sizeof(full->action_history));
        memset(train->action_history, 0, sizeof(train->action_history));
        const BoplRlState *state=bopl_rl_state(world);
        for (int i=0; i<2; i++) bopl_state_copy(&delayed[i], state);
        int cursor=0; uint64_t decisions=0, split=0, hash=state_hash(0x243F6A8885A308D3ull, state), train_hash=0x243F6A8885A308D3ull, telemetry_hash=train_hash;
        uint32_t tick=1;   /* the replay's first post-warmup tick */
        while (!(state->header.flags&BOPL_RL_TERMINAL) && warmup+tick-1<replay.tick_count) {
            BoplRlAction actions[PLAYERS];
            for (size_t b=0; b<evict_bytes; b+=64) evict[b]++;
            const BoplRlAction *first=&replay.actions[(size_t)(warmup+tick-1)*PLAYERS];
            memcpy(actions, first, sizeof(actions));
            for (uint32_t k=1; k<4 && warmup+tick-1+k<replay.tick_count; k++)
                if (memcmp(first, &replay.actions[(size_t)(warmup+tick-1+k)*PLAYERS], sizeof(actions))) { split++; break; }
            uint32_t before=state->header.episode_ticks;
            t0=seconds(); i0=instructions();
            if (bopl_rl_step_sampled(world, actions, PLAYERS, 3, 1)) { fprintf(stderr, "%s: step failed: %s\n", path, bopl_rl_error(world)); return 1; }
            double t1=seconds(); uint64_t i1=instructions(); t[P_SIM]+=t1-t0; ic[P_SIM]+=i1-i0;
            bopl_state_copy(&sample, bopl_rl_sample(world)); bopl_count_overflow(lite, bopl_rl_sample(world));
            double t2=seconds(); t[P_SAMPLE]+=t2-t1; ic[P_SAMPLE]+=instructions()-i1;
            state=bopl_rl_state(world);
            hash=state_hash(state_hash(hash, bopl_rl_sample(world)), state);
            tick+=state->header.episode_ticks-before;
            decisions++;
            if (styled) {
                float paid[PLAYERS];
                double a=seconds(); uint64_t ia=instructions();
                const BoplRlEvent *events=NULL; uint32_t event_count=0; int support=0;
                events=bopl_style_events(world, &event_count, &support);
                for (int i=0; i<PLAYERS; i++) paid[i]=bopl_style_reward(styled, state, events, event_count, support, i);
                t[P_STYLE]+=seconds()-a; ic[P_STYLE]+=instructions()-ia;
                style_hash=mix_words(style_hash, paid, sizeof(paid)); style_entities+=state->header.entity_count;
            }
            if (state->header.flags&BOPL_RL_TERMINAL) break;
            if (sim_only) continue;
            const BoplRlState *old=bopl_state_copy_get(&delayed[cursor]);
            for (int k=0; k<2 && !train_only; k++) {
                BoplPuffer *env=((decisions+k)&1)?lite:full; int phase=env==lite?P_LITE:P_FULL;
                double a=seconds(); uint64_t ia=instructions(); bopl_encode(env, old, 0); t[phase]+=seconds()-a; ic[phase]+=instructions()-ia;
            }
            if (!train_only) hash=mix_words(mix_words(hash, scratch_lite, sizeof(scratch_lite)), scratch_full, sizeof(scratch_full));
            {
                double a=seconds(); uint64_t ia=instructions();
                for (int i=0; i<PLAYERS; i++) train->applied[i]=actions[i];
                bopl4_defer=1; bopl4_pending_count=0; bopl4_state_count=0; bopl_encode_copy(train, &delayed[cursor], 1);
                for (int i=0; i<PLAYERS; i++) {
                    float *previous=bopl_obs_row(train, i)+bopl_previous_offset(train, i);
                    bopl_previous_action(previous, actions[i], train->hold_left[i], train->hold_run[i]);
                    bopl4_previous_history(previous, train->action_history[i], actions[i]);
                    bopl_write_mask(train, i);
                }
                bopl_encode_privileged(train, state, actions);
                bopl4_defer=0;
                bopl_pack_rows(train);
                t[P_TRAIN]+=seconds()-a; ic[P_TRAIN]+=instructions()-ia;
                train_hash=mix_words(mix_words(train_hash, pack_train, sizeof(pack_train)), masks_train, sizeof(masks_train));
                telemetry_hash=mix_words(mix(telemetry_hash, (uint64_t)train->log.observation_overflow), &train->counters, sizeof(train->counters));
                for (int i=0; i<PLAYERS; i++) total_fallback+=((const uint32_t*)(pack_train+(size_t)i*BOPL4_PACK_P))[0]!=BOPL4_PACK_MODE_STATE;
            }
            BoplStateCopy published=delayed[cursor]; delayed[cursor]=sample; sample=published; cursor^=1;
            if (train_only) continue;
            double t3=seconds(); uint64_t i3=instructions();
            for (int i=0; i<PLAYERS; i++) {
                lite->applied[i]=actions[i];
                float *previous=bopl_obs_row(lite, i)+bopl_previous_offset(lite, i);
                bopl_previous_action(previous, actions[i], lite->hold_left[i], lite->hold_run[i]);
                bopl4_previous_history(previous, lite->action_history[i], actions[i]);
                bopl_write_mask(lite, i);
            }
            double t4=seconds(); uint64_t i4=instructions(); t[P_TAIL]+=t4-t3; ic[P_TAIL]+=i4-i3;
            bopl_encode_privileged(lite, state, actions);
            double t5=seconds(); uint64_t i5=instructions(); t[P_PRIV]+=t5-t4; ic[P_PRIV]+=i5-i4;
            bopl_pack_rows(lite);
            t[P_PACK]+=seconds()-t5; ic[P_PACK]+=instructions()-i5;
        }
        uint32_t ticks=state->header.episode_ticks;
        printf("{\"game\": \"%s\", \"map\": %u, \"kits\": [[%u,%u,%u],[%u,%u,%u]], \"decisions\": %"PRIu64", \"ticks\": %u, \"split\": %"PRIu64", \"hash\": \"%016"PRIx64"\", \"train_hash\": \"%016"PRIx64"\", \"telemetry_hash\": \"%016"PRIx64"\"",
               names[g], replay.config.map, replay.config.player[0].abilities[0], replay.config.player[0].abilities[1], replay.config.player[0].abilities[2],
               replay.config.player[1].abilities[0], replay.config.player[1].abilities[1], replay.config.player[1].abilities[2], decisions, ticks, split, hash, train_hash, telemetry_hash);
        for (int p=0; p<PHASES; p++) printf(", \"%s_us\": %.3f", PHASE_NAMES[p], p==P_RESET?t[p]*1e6:t[p]*1e6/(double)(decisions?decisions:1));
        printf(", \"load1\": %.2f}\n", load1());
        fflush(stdout);
        for (int p=0; p<PHASES; p++) { total[p]+=t[p]; total_inst[p]+=ic[p]; }
        total_decisions+=decisions; total_ticks+=ticks; total_split+=split; corpus_hash=mix(corpus_hash, hash);
        corpus_train=mix(corpus_train, train_hash); corpus_telemetry=mix(corpus_telemetry, telemetry_hash);
        free(replay.actions);
    }
    wall=seconds()-wall;
    printf("{\"summary\": true, \"games\": %d, \"decisions\": %"PRIu64", \"ticks\": %"PRIu64", \"split\": %"PRIu64", \"wall_s\": %.2f, \"hash\": \"%016"PRIx64"\", \"train_hash\": \"%016"PRIx64"\", \"telemetry_hash\": \"%016"PRIx64"\"",
           n, total_decisions, total_ticks, total_split, wall, corpus_hash, corpus_train, corpus_telemetry);
    for (int p=0; p<PHASES; p++) printf(", \"%s_us\": %.3f", PHASE_NAMES[p], total[p]*1e6/(double)total_decisions);
    for (int p=0; p<PHASES; p++) printf(", \"%s_inst\": %.1f", PHASE_NAMES[p], inst_fd<0?-1.0:(double)total_inst[p]/(double)total_decisions);
    printf(", \"train_fallback\": %.5f", (double)total_fallback/(double)(PLAYERS*total_decisions));
    double env_full=total[P_RESET]+total[P_SIM]+total[P_SAMPLE]+total[P_FULL]+total[P_TAIL]+total[P_PRIV]+total[P_PACK];
    /* The trainer's own c_step: reset, sim, sample and the train-writer finish. */
    double env_train=total[P_RESET]+total[P_SIM]+total[P_SAMPLE]+total[P_TRAIN];
    uint64_t env_train_inst=total_inst[P_RESET]+total_inst[P_SIM]+total_inst[P_SAMPLE]+total_inst[P_TRAIN];
    if (styled) {
        double net_s=total[P_STYLE]-timer_s*(double)total_decisions, net_inst=(double)total_inst[P_STYLE]-(double)timer_inst*(double)total_decisions;
        printf(", \"style_terms\": %u, \"style_hash\": \"%016"PRIx64"\", \"style_entities\": %.1f, \"style_pct\": %.3f, \"style_inst_pct\": %.3f"
               ", \"timer_us\": %.3f, \"timer_inst\": %"PRIu64", \"style_net_us\": %.3f, \"style_net_inst\": %.1f, \"style_net_pct\": %.3f, \"style_net_inst_pct\": %.3f, \"style_pay\": [",
               style_program.n_terms, style_hash, (double)style_entities/(double)total_decisions, 100.0*total[P_STYLE]/env_train,
               inst_fd<0?-1.0:100.0*(double)total_inst[P_STYLE]/(double)env_train_inst,
               timer_s*1e6, timer_inst, net_s*1e6/(double)total_decisions, net_inst/(double)total_decisions,
               100.0*net_s/env_train, inst_fd<0?-1.0:100.0*net_inst/(double)env_train_inst);
        for (uint32_t k=0; k<style_program.n_terms; k++) printf("%s%.6g", k?", ":"", styled->style_pay[k]);
        printf("]");
    }
    printf(", \"env_train_us\": %.3f", env_train*1e6/(double)total_decisions);
    printf(", \"env_lite_us\": %.3f, \"env_full_us\": %.3f, \"load1\": %.2f}\n",
           (total[P_RESET]+total[P_SIM]+total[P_SAMPLE]+total[P_LITE]+total[P_TAIL]+total[P_PRIV]+total[P_PACK])*1e6/(double)total_decisions,
           env_full*1e6/(double)total_decisions, load1());
    bopl_rl_close(world);
    return 0;
}
#endif
