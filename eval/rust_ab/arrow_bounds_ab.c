/* Arrow boundary A/B: fire out of the left side of map 4 and report the
 * last active-projectile tick. Arrow.prefab destroys below water or beyond
 * either horizontal blast bound, but deliberately survives above the top.
 */
#include "bopl_rl.h"
#include <stdio.h>

static unsigned arrows(const BoplRlState *state, double *x) {
    unsigned count = 0;
    for (uint32_t i = 0; i < state->header.entity_count; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->layer == 9 && entity->owner == 1 && (entity->flags & 1)) {
            count++;
            *x = entity->x / 4294967296.0;
        }
    }
    return count;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1],
            "/proc/self/cwd/native/fixtures/movement.rep",
            "/proc/self/cwd/native/target/arrow_bounds_ab.log",
            120000, error, sizeof error);
        if (!world[backend]) { fprintf(stderr, "%s\n", error); return 1; }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7; config.map = 4; config.stocks = 100;
    config.warmup_ticks = 240; config.max_episode_ticks = 4000;
    for (int player = 0; player < 2; player++) {
        config.player[player].abilities[0] = 2;
        config.player[player].abilities[1] = 4;
        config.player[player].abilities[2] = 14;
    }
    for (int backend = 0; backend < 2; backend++)
        if (bopl_rl_reset(world[backend], &config) != BOPL_RL_OK) return 1;

    BoplRlAction action[2] = {{0}};
    unsigned last_tick[2] = {0}, max_arrows[2] = {0};
    double last_x[2] = {0};
    for (unsigned tick = 1; tick <= 900; tick++) {
        action[0].angle = 112; /* up-left: clears terrain before side exit */
        action[0].buttons = tick >= 260 && tick <= 320 ? 4 : 0;
        for (int backend = 0; backend < 2; backend++) {
            if (bopl_rl_step(world[backend], action, 2, 1) != BOPL_RL_OK) return 1;
            double x = 0;
            unsigned count = arrows(bopl_rl_state(world[backend]), &x);
            if (count) { last_tick[backend] = tick; last_x[backend] = x; }
            if (count > max_arrows[backend]) max_arrows[backend] = count;
        }
    }
    printf("net_last=%u net_x=%.6f rust_last=%u rust_x=%.6f max=%u/%u\n",
        last_tick[0], last_x[0], last_tick[1], last_x[1], max_arrows[0], max_arrows[1]);
    bopl_rl_close(world[0]); bopl_rl_close(world[1]);
    return 0;
}
