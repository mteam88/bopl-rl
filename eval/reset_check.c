/* A reset after dirty gameplay must reproduce a fresh seeded trajectory. */
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
static int action(BoplRl *world, unsigned int step, unsigned int repeat) {
    BoplRlAction actions[6]={{0}};
    for (uint32_t i=0; i<6; i++) {
        actions[i].angle=(uint8_t)(1+(step*17+i*39)%255);
        actions[i].buttons=(uint16_t)((step+3*i)%16);
    }
    return bopl_rl_step(world, actions, 6, repeat);
}
int main(int argc, char **argv) {
    if (argc!=4) { fprintf(stderr, "reset_check <absolute game> <absolute bootstrap.rep> <absolute game.log>\n"); return 2; }
    char error[512];
    BoplRl *world=bopl_rl_open(argv[1], argv[2], argv[3], 30000, error, sizeof(error));
    if (!world) { fprintf(stderr, "%s\n", error); return 1; }
    BoplRlConfig config; bopl_rl_default_config(&config, 6);
    config.seed=112481; config.max_episode_ticks=600;
    Snapshot expected[151]={{0}};
    uint32_t count=0, comparisons=0;
    int status=bopl_rl_reset(world, &config);
    if (status) goto failed;
    for (;;) {
        expected[count++]=snapshot(bopl_rl_state(world));
        if (bopl_rl_state(world)->header.flags&BOPL_RL_TERMINAL) break;
        if (count>=151) { status=1; goto failed; }
        status=action(world, count-1, 4);
        if (status) goto failed;
    }
    for (uint32_t ability=1; ability<bopl_rl_ability_count(world); ability++) {
        BoplRlConfig dirty=config; dirty.seed=75426+ability; dirty.map=ability%bopl_rl_map_count(world);
        for (int i=0; i<6; i++) for (int j=0; j<3; j++) dirty.player[i].abilities[j]=ability;
        status=bopl_rl_reset(world, &dirty);
        if (status) goto failed;
        for (unsigned int step=0; step<32 && !(bopl_rl_state(world)->header.flags&BOPL_RL_TERMINAL); step++) {
            status=action(world, step, 16);
            if (status) goto failed;
        }
        status=bopl_rl_reset(world, &config);
        if (status) goto failed;
        for (uint32_t index=0; index<count; index++) {
            Snapshot got=snapshot(bopl_rl_state(world));
            int equal=got.length==expected[index].length && !memcmp(got.bytes, expected[index].bytes, got.length);
            free(got.bytes);
            if (!equal) {
                fprintf(stderr, "Reset after ability %u diverged at action %u\n", ability, index);
                status=1; goto failed;
            }
            comparisons++;
            if (index+1<count) { status=action(world, index, 4); if (status) goto failed; }
        }
        printf("{\"dirty_ability\":%u,\"exact_snapshots\":%u}\n", ability, count); fflush(stdout);
    }
    printf("{\"status\":\"RESET_ISOLATION_OK\",\"exact_snapshots\":%u}\n", comparisons);
failed:
    if (status) fprintf(stderr, "Reset isolation failed (%d): %s\n", status, bopl_rl_error(world));
    for (uint32_t i=0; i<count; i++) free(expected[i].bytes);
    bopl_rl_close(world);
    return status?1:0;
}
