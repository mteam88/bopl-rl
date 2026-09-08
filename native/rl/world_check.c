/* Many-worlds parity: every world of one N-world game process must produce,
 * tick for tick, the same fixed-point state, sampled observations and replay
 * bytes as a single-world process given the same reset config and actions,
 * while the other worlds run different matches (different maps, seeds and
 * loadouts that exercise the scene-scoped Find sites) at the same time and
 * reset at unrelated moments. Worlds are driven from concurrent threads.
 *
 * world_check <absolute game> <absolute bootstrap.rep> <absolute log dir> [worlds=4] [rounds=3] */
#include "bopl_rl.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_WORLDS=64, MAX_STEPS=400 };
typedef struct { unsigned char *bytes; size_t length; } Blob;
static Blob snapshot(const BoplRlState *state) {
    size_t entities=state->header.entity_count*sizeof(BoplRlEntity);
    size_t slimes=state->header.slime_count*sizeof(BoplRlSlime);
    Blob result={NULL, sizeof(state->header)+sizeof(state->players)+entities+slimes+sizeof(state->telemetry)};
    result.bytes=malloc(result.length);
    if (!result.bytes) abort();
    unsigned char *at=result.bytes;
    memcpy(at, &state->header, sizeof(state->header)); at+=sizeof(state->header);
    memcpy(at, state->players, sizeof(state->players)); at+=sizeof(state->players);
    if (entities) memcpy(at, state->entities, entities);
    at+=entities;
    if (slimes) memcpy(at, state->slimes, slimes);
    at+=slimes;
    memcpy(at, state->telemetry, sizeof(state->telemetry));
    return result;
}
static Blob replay_blob(BoplRl *w) {
    const void *bytes; size_t length;
    if (bopl_rl_replay(w, &bytes, &length)) { fprintf(stderr, "replay: %s\n", bopl_rl_error(w)); abort(); }
    Blob result={malloc(length?length:1), length};
    if (!result.bytes) abort();
    memcpy(result.bytes, bytes, length);
    return result;
}
static void actions_for(BoplRlAction *actions, uint32_t step, uint32_t salt) {
    for (uint32_t i=0; i<6; i++) {
        actions[i].angle=(uint8_t)(1+(step*17+i*39+salt*7)%255);
        actions[i].buttons=(uint16_t)((step+3*i+salt)%16 | ((step%5==0)?2:0) | ((step%7==0)?4:0) | ((step%11==0)?8:0));
        actions[i].reserved=0;
    }
}
/* Loadouts that hit the scene-scoped Find sites (Mjolnir 37, Portal gun 38,
 * Poke ball 33, Tack 35, Freeze ray 36, Gold mushroom 34, Spike 11, Beam 28,
 * Throw/Macho 21, Dash 2). */
static const uint32_t LOADOUTS[][3]={{37,38,33},{35,36,34},{11,28,21},{2,37,36},{38,33,35},{34,11,2}};
static void config_for(BoplRlConfig *config, uint32_t maps, uint32_t match) {
    bopl_rl_default_config(config, 2);
    config->seed=90001+match*7919;
    config->map=match%maps;
    config->max_episode_ticks=600+(match%3)*120;
    config->warmup_ticks=240;
    for (uint32_t i=0; i<2; i++) for (uint32_t j=0; j<3; j++)
        config->player[i].abilities[j]=LOADOUTS[(match+i*2)%6][j];
}
/* One match: reset, then sampled steps until terminal or MAX_STEPS. Records
 * a snapshot of sample+state per step, and the replay bytes at the end. */
typedef struct { Blob states[2*MAX_STEPS+1]; uint32_t steps; Blob replay; int failed; char error[512]; } Trace;
static void free_trace(Trace *t) {
    for (uint32_t i=0; i<2*t->steps+1; i++) free(t->states[i].bytes);
    free(t->replay.bytes); memset(t, 0, sizeof(*t));
}
static void run_match(BoplRl *w, const BoplRlConfig *config, uint32_t salt, Trace *t) {
    memset(t, 0, sizeof(*t));
    if (bopl_rl_reset(w, config)) { t->failed=1; snprintf(t->error, sizeof(t->error), "reset: %s", bopl_rl_error(w)); return; }
    t->states[0]=snapshot(bopl_rl_state(w));
    while (!(bopl_rl_state(w)->header.flags&BOPL_RL_TERMINAL) && t->steps<MAX_STEPS) {
        BoplRlAction actions[6];
        actions_for(actions, t->steps, salt);
        if (bopl_rl_step_sampled(w, actions, 2, 3, 1)) { t->failed=1; snprintf(t->error, sizeof(t->error), "step: %s", bopl_rl_error(w)); return; }
        t->states[2*t->steps+1]=snapshot(bopl_rl_sample(w));
        t->states[2*t->steps+2]=snapshot(bopl_rl_state(w));
        t->steps++;
    }
    t->replay=replay_blob(w);
}
static int equal_trace(const Trace *a, const Trace *b, const char **why) {
    if (a->failed || b->failed) { *why="failure"; return 0; }
    if (a->steps!=b->steps) { *why="step count"; return 0; }
    for (uint32_t i=0; i<2*a->steps+1; i++)
        if (a->states[i].length!=b->states[i].length || memcmp(a->states[i].bytes, b->states[i].bytes, a->states[i].length)) { *why="state"; return 0; }
    if (a->replay.length!=b->replay.length || memcmp(a->replay.bytes, b->replay.bytes, a->replay.length)) { *why="replay"; return 0; }
    return 1;
}
typedef struct { BoplRl *world; uint32_t index, worlds, rounds, maps; Trace *traces; } Job;
/* Each world thread plays `rounds` matches back to back; world k's match r is
 * global match k+r*worlds, so worlds are on different maps/loadouts and are
 * desynchronised because their episode lengths differ. */
static void *drive(void *arg) {
    Job *job=arg;
    for (uint32_t r=0; r<job->rounds; r++) {
        uint32_t match=job->index+r*job->worlds;
        BoplRlConfig config; config_for(&config, job->maps, match);
        run_match(job->world, &config, match, &job->traces[r]);
        if (job->traces[r].failed) return NULL;
    }
    return NULL;
}
int main(int argc, char **argv) {
    if (argc<4 || argc>6) { fprintf(stderr, "world_check <absolute game> <absolute bootstrap.rep> <absolute log dir> [worlds=4] [rounds=3]\n"); return 2; }
    uint32_t worlds=argc>4?(uint32_t)strtoul(argv[4], NULL, 10):4, rounds=argc>5?(uint32_t)strtoul(argv[5], NULL, 10):3;
    if (worlds<2 || worlds>MAX_WORLDS || rounds<1 || rounds>16) return 2;
    char error[512], path[4096];
    /* Reference: a single-world process (legacy path when the build has one). */
    snprintf(path, sizeof(path), "%s/world_check-single.log", argv[3]);
    BoplRl *single=bopl_rl_open(argv[1], argv[2], path, 30000, error, sizeof(error));
    if (!single) { fprintf(stderr, "single: %s\n", error); return 1; }
    snprintf(path, sizeof(path), "%s/world_check-%u-worlds.log", argv[3], worlds);
    BoplRl *many[MAX_WORLDS];
    if (bopl_rl_open_worlds(argv[1], argv[2], path, 30000, worlds, many, error, sizeof(error))) { fprintf(stderr, "many: %s\n", error); return 1; }
    if (bopl_rl_world_count(many[0])!=worlds) { fprintf(stderr, "world count\n"); return 1; }
    uint32_t maps=bopl_rl_map_count(single);
    if (maps>5) maps=5;
    Trace *traces=calloc((size_t)worlds*rounds, sizeof(Trace));
    Job *jobs=calloc(worlds, sizeof(Job));
    pthread_t *threads=calloc(worlds, sizeof(pthread_t));
    if (!traces || !jobs || !threads) abort();
    for (uint32_t k=0; k<worlds; k++) {
        jobs[k]=(Job){many[k], k, worlds, rounds, maps, traces+(size_t)k*rounds};
        if (pthread_create(&threads[k], NULL, drive, &jobs[k])) abort();
    }
    for (uint32_t k=0; k<worlds; k++) pthread_join(threads[k], NULL);
    int status=0; uint32_t compared=0, snapshots=0, total_ticks=0;
    for (uint32_t k=0; k<worlds && !status; k++) for (uint32_t r=0; r<rounds && !status; r++) {
        Trace *got=&traces[(size_t)k*rounds+r];
        if (got->failed) { fprintf(stderr, "world %u round %u: %s\n", k, r, got->error); status=1; break; }
        uint32_t match=k+r*worlds;
        BoplRlConfig config; config_for(&config, maps, match);
        Trace want; run_match(single, &config, match, &want);
        if (want.failed) { fprintf(stderr, "single round %u: %s\n", match, want.error); status=1; free_trace(&want); break; }
        const char *why="";
        if (!equal_trace(&want, got, &why)) {
            fprintf(stderr, "world %u round %u (map %u, seed %u) diverged: %s (single %u steps, world %u steps)\n", k, r, config.map, config.seed, why, want.steps, got->steps);
            status=1;
        } else {
            compared++; snapshots+=2*got->steps+1;
            total_ticks+=((const BoplRlHeader*)got->states[2*got->steps].bytes)->episode_ticks;
            printf("{\"world\":%u,\"round\":%u,\"map\":%u,\"steps\":%u,\"episode_ticks\":%u,\"replay_bytes\":%zu,\"exact\":true}\n",
                   k, r, config.map, got->steps, ((const BoplRlHeader*)got->states[2*got->steps].bytes)->episode_ticks, got->replay.length);
            fflush(stdout);
        }
        free_trace(&want);
    }
    /* Closing order: an inner handle first, then the rest, then the reference. */
    for (uint32_t k=0; k<worlds; k++) free_trace(&traces[(size_t)k*rounds+0]);
    for (uint32_t k=0; k<worlds; k++) for (uint32_t r=1; r<rounds; r++) free_trace(&traces[(size_t)k*rounds+r]);
    bopl_rl_close(many[worlds/2]);
    for (uint32_t k=0; k<worlds; k++) if (k!=worlds/2) bopl_rl_close(many[k]);
    bopl_rl_close(single);
    free(traces); free(jobs); free(threads);
    if (!status) printf("{\"status\":\"MANY_WORLDS_PARITY_OK\",\"worlds\":%u,\"rounds\":%u,\"matches\":%u,\"exact_snapshots\":%u,\"episode_ticks\":%u}\n", worlds, rounds, compared, snapshots, total_ticks);
    return status;
}
