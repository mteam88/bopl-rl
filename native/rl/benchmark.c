#define _POSIX_C_SOURCE 200809L
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec+time.tv_nsec/1e9;
}
static uint32_t random_u32(uint32_t *state) {
    uint32_t x=*state; x^=x<<13; x^=x>>17; x^=x<<5; return *state=x;
}
static int check(BoplRl *world, int status, const char *operation) {
    if (status==BOPL_RL_OK) return 1;
    fprintf(stderr, "%s failed (%d): %s\n", operation, status, bopl_rl_error(world));
    return 0;
}
int main(int argc, char **argv) {
    if (argc<4 || argc>7) {
        fprintf(stderr, "benchmark <absolute game> <absolute bootstrap.rep> <absolute game.log> [ticks=3000] [players=6] [repeat=1]\n");
        return 2;
    }
    uint64_t wanted=argc>4?strtoull(argv[4], NULL, 10):3000;
    uint32_t players=argc>5?(uint32_t)strtoul(argv[5], NULL, 10):6;
    uint32_t repeat=argc>6?(uint32_t)strtoul(argv[6], NULL, 10):1;
    if (!wanted || players<1 || players>6 || repeat<1 || repeat>16) return 2;
    char error[512]; double start=seconds();
    BoplRl *world=bopl_rl_open(argv[1], argv[2], argv[3], 30000, error, sizeof(error));
    if (!world) { fprintf(stderr, "open failed: %s\n", error); return 1; }
    double startup=seconds()-start;
    BoplRlConfig config; bopl_rl_default_config(&config, players);
    config.max_episode_ticks=1200;
    double reset_start=seconds();
    if (!check(world, bopl_rl_reset(world, &config), "initial reset")) goto failed;
    double reset_seconds=seconds()-reset_start;
    const BoplRlState *state=bopl_rl_state(world);
    if (!state || state->header.player_count!=players || state->header.tick!=240) {
        fprintf(stderr, "Invalid reset state\n"); goto failed;
    }
    BoplRlHeader header=state->header;
    BoplRlPlayer saved_players[6]; memcpy(saved_players, state->players, sizeof(saved_players));
    size_t entity_bytes=state->header.entity_count*sizeof(BoplRlEntity);
    BoplRlEntity *saved_entities=malloc(entity_bytes?entity_bytes:1);
    if (!saved_entities) goto failed;
    memcpy(saved_entities, state->entities, entity_bytes);
    size_t slime_bytes=state->header.slime_count*sizeof(BoplRlSlime);
    BoplRlSlime *saved_slimes=malloc(slime_bytes?slime_bytes:1);
    if (!saved_slimes) { free(saved_entities); goto failed; }
    memcpy(saved_slimes, state->slimes, slime_bytes);
    reset_start=seconds();
    if (!check(world, bopl_rl_reset(world, &config), "repeated reset")) { free(saved_slimes); free(saved_entities); goto failed; }
    reset_seconds+=seconds()-reset_start;
    state=bopl_rl_state(world);
    int repeat_match=!memcmp(&header, &state->header, sizeof(header))
        && !memcmp(saved_players, state->players, sizeof(saved_players))
        && !memcmp(saved_entities, state->entities, entity_bytes)
        && !memcmp(saved_slimes, state->slimes, slime_bytes);
    free(saved_slimes); free(saved_entities);
    if (!repeat_match) { fprintf(stderr, "Repeated reset state mismatch\n"); goto failed; }
    BoplRlAction actions[6]={{0}};
    if (bopl_rl_step(world, actions, 0, repeat)!=BOPL_RL_ARGUMENT || bopl_rl_state(world)->header.tick!=240) {
        fprintf(stderr, "Invalid action count advanced state\n"); goto failed;
    }
    uint64_t ticks=0, calls=0, resets=2, alive_ticks=0, all_dead_ticks=0;
    uint32_t rng=932155, max_entities=state->header.entity_count;
    double step_seconds=0;
    while (ticks<wanted) {
        state=bopl_rl_state(world);
        if (state->header.flags & BOPL_RL_TERMINAL) {
            config.seed++;
            reset_start=seconds();
            if (!check(world, bopl_rl_reset(world, &config), "episode reset")) goto failed;
            reset_seconds+=seconds()-reset_start; resets++;
            state=bopl_rl_state(world);
            if (state->header.flags & BOPL_RL_TERMINAL) { fprintf(stderr, "Reset returned an ended episode\n"); goto failed; }
        }
        if (calls%15==0) for (uint32_t i=0; i<players; i++) {
            uint32_t r=random_u32(&rng);
            actions[i].angle=(uint8_t)(r>>16); actions[i].buttons=(uint16_t)(r&15);
        }
        uint32_t before=state->header.episode_ticks;
        double step_start=seconds();
        if (!check(world, bopl_rl_step(world, actions, players, repeat), "step")) {
            fprintf(stderr, "Failure after %"PRIu64" measured ticks, episode tick %u\n", ticks, before);
            goto failed;
        }
        step_seconds+=seconds()-step_start; calls++;
        state=bopl_rl_state(world);
        uint32_t advanced=state->header.episode_ticks-before;
        if (!advanced || advanced>repeat) { fprintf(stderr, "Invalid tick advance\n"); goto failed; }
        ticks+=advanced;
        uint32_t alive=0;
        for (uint32_t i=0; i<players; i++) alive+=(state->players[i].flags&1)!=0;
        alive_ticks+=(uint64_t)alive*advanced; all_dead_ticks+=alive?0:advanced;
        if (state->header.entity_count>max_entities) max_entities=state->header.entity_count;
    }
    printf("{\"status\":\"C_RL_SMOKE_OK\",\"players\":%u,\"repeat\":%u,\"ticks\":%"PRIu64",\"calls\":%"PRIu64
           ",\"resets\":%"PRIu64",\"alive_player_ticks\":%"PRIu64",\"all_dead_ticks\":%"PRIu64
           ",\"max_entities\":%u,\"repeated_reset_exact\":true,\"startup_seconds\":%.6f,\"step_seconds\":%.6f,\"reset_seconds\":%.6f"
           ",\"simulation_ticks_per_step_second\":%.2f,\"simulation_ticks_per_second_including_resets\":%.2f"
           ",\"source_hash\":\"%s\",\"catalog_hash\":\"%s\"}\n",
           players, repeat, ticks, calls, resets, alive_ticks, all_dead_ticks, max_entities, startup, step_seconds, reset_seconds,
           ticks/step_seconds, ticks/(step_seconds+reset_seconds), bopl_rl_source_hash(world), bopl_rl_catalog_hash(world));
    bopl_rl_close(world); return 0;
failed:
    bopl_rl_close(world); return 1;
}
