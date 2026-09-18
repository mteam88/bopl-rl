/* Per-ability throughput driver, derived from native/rl/benchmark.c.
 * usage: ability_bench <game.so> <ticks> <players> <ability_id|-1 for default>
 * All three slots of every player are set to <ability_id>; actions cast often.
 * Prints: "<ability> <ticks_per_step_second> <resets> <max_entities>" */
#define _POSIX_C_SOURCE 200809L
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
static uint32_t random_u32(uint32_t *s) {
    uint32_t x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return *s = x;
}
int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s <game.so> <ticks> <players> <ability|-1>\n", argv[0]); return 2; }
    uint64_t wanted = strtoull(argv[2], NULL, 10);
    uint32_t players = (uint32_t)strtoul(argv[3], NULL, 10);
    long ability = strtol(argv[4], NULL, 10);
    char error[512];
    BoplRl *world = bopl_rl_open(
        argv[1], "/dev/null",
        "/home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject/ability-port/native/target/perf/ability-bench.log",
        30000, error, sizeof(error));
    if (!world) { fprintf(stderr, "open failed: %s\n", error); return 1; }
    BoplRlConfig config; bopl_rl_default_config(&config, players);
    config.max_episode_ticks = 1200;
    if (ability >= 0)
        for (uint32_t i = 0; i < 6; i++)
            for (int s = 0; s < 3; s++) config.player[i].abilities[s] = (uint32_t)ability;
    if (bopl_rl_reset(world, &config) != BOPL_RL_OK) {
        fprintf(stderr, "reset failed for ability %ld: %s\n", ability, bopl_rl_error(world));
        bopl_rl_close(world); return 1;
    }
    BoplRlAction actions[6] = {{0, 0, 0}};
    uint64_t ticks = 0, calls = 0, resets = 1;
    uint32_t rng = 932155, max_entities = bopl_rl_state(world)->header.entity_count;
    double step_seconds = 0;
    while (ticks < wanted) {
        const BoplRlState *state = bopl_rl_state(world);
        if (state->header.flags & BOPL_RL_TERMINAL) {
            config.seed++;
            if (bopl_rl_reset(world, &config) != BOPL_RL_OK) {
                fprintf(stderr, "episode reset failed: %s\n", bopl_rl_error(world));
                bopl_rl_close(world); return 1;
            }
            resets++; state = bopl_rl_state(world);
        }
        /* New actions every 5 calls, ability buttons pressed most of the time. */
        if (calls % 5 == 0) for (uint32_t i = 0; i < players; i++) {
            uint32_t r = random_u32(&rng);
            actions[i].angle = (uint8_t)(r >> 16);
            /* jump/move from the low bits, ability bits 2/4/8 set often */
            actions[i].buttons = (uint16_t)((r & 0x3c1) | (1u << (1 + (r >> 8) % 3)));
        }
        double start = seconds();
        if (bopl_rl_step(world, actions, players, 1) != BOPL_RL_OK) {
            fprintf(stderr, "step failed: %s\n", bopl_rl_error(world));
            bopl_rl_close(world); return 1;
        }
        step_seconds += seconds() - start; calls++; ticks++;
        state = bopl_rl_state(world);
        if (state->header.entity_count > max_entities) max_entities = state->header.entity_count;
    }
    printf("%ld %.1f %" PRIu64 " %u\n", ability, ticks / step_seconds, resets, max_entities);
    bopl_rl_close(world);
    return 0;
}
