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
 * lite and full each write into their own env (their own written-row records)
 * and swap order every decision. The replay's per-tick inputs are held for a
 * decision (the trainer's repeat 4); groups whose ticks disagree are counted.
 * One JSON line per game (map, kits, decisions, us per phase per decision),
 * then a summary over the corpus and per-map / per-kit aggregates go to the
 * python side. load1 travels with every line. Every game's hash folds each
 * decision's sampled and final states (replay_ab.c's strict state hash) and
 * both writers' rows: equal hashes between two builds = the exact gate.
 *
 *   env_bench <libbopl_sim.so> <corpus_dir> [max_games=0 (all)] [warmup=240] [limit=10800]
 *
 * Pin and nice it (taskset -c 11 nice -n 19). Build: make env-bench. */
#define _GNU_SOURCE
#include "bopl_puffer.h"
#include <dirent.h>
#include <inttypes.h>
#include <time.h>

static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec+t.tv_nsec/1e9;
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

enum { PLAYERS=2, PHASES=8 };
enum { P_RESET, P_SIM, P_SAMPLE, P_LITE, P_FULL, P_TAIL, P_PRIV, P_PACK };
static const char *PHASE_NAMES[PHASES]={"reset", "sim", "sample", "lite", "full", "tail", "priv", "pack"};

static void env_init(BoplPuffer *env, BoplRl *world, float *scratch, float *pack_out, unsigned char (*masks)[BOPL_LOGITS], int ablate) {
    memset(env, 0, sizeof(*env));
    env->players=PLAYERS; env->num_agents=PLAYERS; env->scratch=scratch; env->world=world;
    env->pack_base=scratch; env->pack_out=pack_out; env->pack_row_begin=0; env->pack_row_end=PLAYERS;
    for (int i=0; i<PLAYERS; i++) { env->layout[i]=BOPL4_LAYOUT_VERSION; bopl_set_ablate_of(env, i, ablate); env->mask_ptr[i]=masks[i]; }
}

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
    static float pack_out[PLAYERS*BOPL4_PACK_P];
    static unsigned char masks_lite[PLAYERS][BOPL_LOGITS], masks_full[PLAYERS][BOPL_LOGITS];
    BoplPuffer *lite=calloc(1, sizeof(BoplPuffer)), *full=calloc(1, sizeof(BoplPuffer));
    env_init(lite, world, scratch_lite, pack_out, masks_lite, 10);
    env_init(full, world, scratch_full, pack_out, masks_full, 0);
    BoplStateCopy delayed[2]={0}, sample={0};

    double total[PHASES]={0}; uint64_t total_decisions=0, total_ticks=0, total_split=0, corpus_hash=0;
    double wall=seconds();
    for (int g=0; g<n; g++) {
        char path[4096]; snprintf(path, sizeof(path), "%s/%s", argv[2], names[g]);
        Replay replay;
        if (replay_decode(path, warmup, limit, &replay)) return 1;
        if (replay.tick_count<=warmup) { fprintf(stderr, "%s: shorter than the warmup\n", path); return 1; }
        double t[PHASES]={0}, t0=seconds();
        if (bopl_rl_reset(world, &replay.config)!=BOPL_RL_OK) { fprintf(stderr, "%s: reset failed: %s\n", path, bopl_rl_error(world)); return 1; }
        t[P_RESET]=seconds()-t0;
        lite->config=full->config=replay.config;
        bopl_written_forget(lite); bopl_written_forget(full);
        memset(lite->action_history, 0, sizeof(lite->action_history)); memset(full->action_history, 0, sizeof(full->action_history));
        const BoplRlState *state=bopl_rl_state(world);
        for (int i=0; i<2; i++) bopl_state_copy(&delayed[i], state);
        int cursor=0; uint64_t decisions=0, split=0, hash=state_hash(0x243F6A8885A308D3ull, state);
        uint32_t tick=1;   /* the replay's first post-warmup tick */
        while (!(state->header.flags&BOPL_RL_TERMINAL) && warmup+tick-1<replay.tick_count) {
            BoplRlAction actions[PLAYERS];
            const BoplRlAction *first=&replay.actions[(size_t)(warmup+tick-1)*PLAYERS];
            memcpy(actions, first, sizeof(actions));
            for (uint32_t k=1; k<4 && warmup+tick-1+k<replay.tick_count; k++)
                if (memcmp(first, &replay.actions[(size_t)(warmup+tick-1+k)*PLAYERS], sizeof(actions))) { split++; break; }
            uint32_t before=state->header.episode_ticks;
            t0=seconds();
            if (bopl_rl_step_sampled(world, actions, PLAYERS, 3, 1)) { fprintf(stderr, "%s: step failed: %s\n", path, bopl_rl_error(world)); return 1; }
            double t1=seconds(); t[P_SIM]+=t1-t0;
            bopl_state_copy(&sample, bopl_rl_sample(world)); bopl_count_overflow(lite, bopl_rl_sample(world));
            double t2=seconds(); t[P_SAMPLE]+=t2-t1;
            state=bopl_rl_state(world);
            hash=state_hash(state_hash(hash, bopl_rl_sample(world)), state);
            tick+=state->header.episode_ticks-before;
            decisions++;
            if (state->header.flags&BOPL_RL_TERMINAL) break;
            const BoplRlState *old=bopl_state_copy_get(&delayed[cursor]);
            for (int k=0; k<2; k++) {
                BoplPuffer *env=((decisions+k)&1)?lite:full; int phase=env==lite?P_LITE:P_FULL;
                double a=seconds(); bopl_encode(env, old, 0); t[phase]+=seconds()-a;
            }
            hash=mix_words(mix_words(hash, scratch_lite, sizeof(scratch_lite)), scratch_full, sizeof(scratch_full));
            BoplStateCopy published=delayed[cursor]; delayed[cursor]=sample; sample=published; cursor^=1;
            double t3=seconds();
            for (int i=0; i<PLAYERS; i++) {
                lite->applied[i]=actions[i];
                float *previous=bopl_obs_row(lite, i)+bopl_previous_offset(lite, i);
                bopl_previous_action(previous, actions[i], lite->hold_left[i], lite->hold_run[i]);
                bopl4_previous_history(previous, lite->action_history[i], actions[i]);
                bopl_write_mask(lite, i);
            }
            double t4=seconds(); t[P_TAIL]+=t4-t3;
            bopl_encode_privileged(lite, state, actions);
            double t5=seconds(); t[P_PRIV]+=t5-t4;
            bopl_pack_rows(lite);
            t[P_PACK]+=seconds()-t5;
        }
        uint32_t ticks=state->header.episode_ticks;
        printf("{\"game\": \"%s\", \"map\": %u, \"kits\": [[%u,%u,%u],[%u,%u,%u]], \"decisions\": %"PRIu64", \"ticks\": %u, \"split\": %"PRIu64", \"hash\": \"%016"PRIx64"\"",
               names[g], replay.config.map, replay.config.player[0].abilities[0], replay.config.player[0].abilities[1], replay.config.player[0].abilities[2],
               replay.config.player[1].abilities[0], replay.config.player[1].abilities[1], replay.config.player[1].abilities[2], decisions, ticks, split, hash);
        for (int p=0; p<PHASES; p++) printf(", \"%s_us\": %.3f", PHASE_NAMES[p], p==P_RESET?t[p]*1e6:t[p]*1e6/(double)(decisions?decisions:1));
        printf(", \"load1\": %.2f}\n", load1());
        fflush(stdout);
        for (int p=0; p<PHASES; p++) total[p]+=t[p];
        total_decisions+=decisions; total_ticks+=ticks; total_split+=split; corpus_hash=mix(corpus_hash, hash);
        free(replay.actions);
    }
    wall=seconds()-wall;
    printf("{\"summary\": true, \"games\": %d, \"decisions\": %"PRIu64", \"ticks\": %"PRIu64", \"split\": %"PRIu64", \"wall_s\": %.2f, \"hash\": \"%016"PRIx64"\"", n, total_decisions, total_ticks, total_split, wall, corpus_hash);
    for (int p=0; p<PHASES; p++) printf(", \"%s_us\": %.3f", PHASE_NAMES[p], total[p]*1e6/(double)total_decisions);
    printf(", \"env_lite_us\": %.3f, \"env_full_us\": %.3f, \"load1\": %.2f}\n",
           (total[P_RESET]+total[P_SIM]+total[P_SAMPLE]+total[P_LITE]+total[P_TAIL]+total[P_PRIV]+total[P_PACK])*1e6/(double)total_decisions,
           (total[P_RESET]+total[P_SIM]+total[P_SAMPLE]+total[P_FULL]+total[P_TAIL]+total[P_PRIV]+total[P_PACK])*1e6/(double)total_decisions, load1());
    bopl_rl_close(world);
    return 0;
}
