/* The state after warmup must not depend on the episode seed. If it does not,
 * one post-warmup snapshot per map can serve every episode, and a restore only
 * has to rebuild Updater.random (InitSeed(seed), then one RandomFix(0,1) per
 * ground, exactly what GameSessionHandler.Init draws). Compares the reset
 * state and the state 120 scripted ticks later across seeds, per map. A
 * loadout holding the Random ability (catalog id 1) draws at SpawnPlayers and
 * is expected to differ; it is reported, not failed. */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned char *bytes; size_t length; } Snapshot;
static Snapshot snapshot(const BoplRlState *state) {
    size_t entities=state->header.entity_count*sizeof(BoplRlEntity);
    size_t slimes=state->header.slime_count*sizeof(BoplRlSlime);
    Snapshot result={NULL, sizeof(state->header)+sizeof(state->players)+entities+slimes};
    result.bytes=malloc(result.length);
    if (!result.bytes) abort();
    unsigned char *at=result.bytes;
    memcpy(at, &state->header, sizeof(state->header)); at+=sizeof(state->header);
    memcpy(at, state->players, sizeof(state->players)); at+=sizeof(state->players);
    if (entities) memcpy(at, state->entities, entities);
    at+=entities;
    if (slimes) memcpy(at, state->slimes, slimes);
    return result;
}
static int same(Snapshot a, Snapshot b) { return a.length==b.length && !memcmp(a.bytes, b.bytes, a.length); }
static int action(BoplRl *world, unsigned int step, unsigned int repeat) {
    BoplRlAction actions[6]={{0}};
    for (uint32_t i=0; i<6; i++) {
        actions[i].angle=(uint8_t)(1+(step*17+i*39)%255);
        actions[i].buttons=(uint16_t)((step+3*i)%16);
    }
    return bopl_rl_step(world, actions, 6, repeat);
}
/* Reset with the configuration, snapshot, play 120 ticks, snapshot again. */
static int run(BoplRl *world, const BoplRlConfig *config, Snapshot *fresh, Snapshot *later) {
    int status=bopl_rl_reset(world, config);
    if (status) return status;
    *fresh=snapshot(bopl_rl_state(world));
    for (unsigned int step=0; step<30 && !(bopl_rl_state(world)->header.flags&BOPL_RL_TERMINAL); step++) {
        status=action(world, step, 4);
        if (status) return status;
    }
    *later=snapshot(bopl_rl_state(world));
    return 0;
}
int main(int argc, char **argv) {
    if (argc<4 || argc>6) { fprintf(stderr, "warmup_check <absolute game> <absolute bootstrap.rep> <absolute game.log> [maps=5] [seeds=4]\n"); return 2; }
    unsigned int maps=argc>4?(unsigned int)atoi(argv[4]):5, seeds=argc>5?(unsigned int)atoi(argv[5]):4;
    if (seeds<2 || seeds>16) { fprintf(stderr, "seeds must be within 2..16\n"); return 2; }
    char error[512];
    BoplRl *world=bopl_rl_open(argv[1], argv[2], argv[3], 30000, error, sizeof(error));
    if (!world) { fprintf(stderr, "%s\n", error); return 1; }
    if (maps>bopl_rl_map_count(world)) maps=bopl_rl_map_count(world);
    int status=0, dependent=0;
    for (uint32_t map=0; map<maps; map++) {
        Snapshot fresh[16], later[16];
        for (unsigned int k=0; k<seeds; k++) {
            BoplRlConfig config; bopl_rl_default_config(&config, 2);
            config.map=map; config.seed=17+map*7919+k*1000003;
            status=run(world, &config, &fresh[k], &later[k]);
            if (status) goto failed;
        }
        for (unsigned int k=1; k<seeds; k++) {
            int first=same(fresh[0], fresh[k]), after=same(later[0], later[k]);
            dependent|=!first||!after;
            printf("{\"map\":%u,\"seed\":%u,\"warmup_ticks\":%u,\"post_warmup_identical\":%s,\"after_120_ticks_identical\":%s}\n",
                   map, 17+map*7919+k*1000003, bopl_rl_state(world)->header.warmup_ticks, first?"true":"false", after?"true":"false");
        }
        for (unsigned int k=0; k<seeds; k++) { free(fresh[k].bytes); free(later[k].bytes); }
        fflush(stdout);
    }
    {   /* Control: the Random ability draws from the seeded stream at spawn. */
        Snapshot fresh[2], later[2];
        for (unsigned int k=0; k<2; k++) {
            BoplRlConfig config; bopl_rl_default_config(&config, 2);
            config.seed=901+k*1000003;
            for (int i=0; i<6; i++) for (int j=0; j<3; j++) config.player[i].abilities[j]=1;
            status=run(world, &config, &fresh[k], &later[k]);
            if (status) goto failed;
        }
        printf("{\"control\":\"random_ability\",\"post_warmup_identical\":%s,\"after_120_ticks_identical\":%s}\n",
               same(fresh[0], fresh[1])?"true":"false", same(later[0], later[1])?"true":"false");
        for (unsigned int k=0; k<2; k++) { free(fresh[k].bytes); free(later[k].bytes); }
    }
    printf("{\"status\":\"%s\",\"maps\":%u,\"seeds\":%u}\n", dependent?"WARMUP_SEED_DEPENDENT":"WARMUP_SEED_INDEPENDENT_OK", maps, seeds);
failed:
    if (status) fprintf(stderr, "Warmup check failed (%d): %s\n", status, bopl_rl_error(world));
    bopl_rl_close(world);
    return (status||dependent)?1:0;
}
