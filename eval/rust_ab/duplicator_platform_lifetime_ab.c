/* Permanent duplicated platforms: a second copy from the same gun expires the first. */
#include "bopl_rl.h"
#include <stdio.h>

static unsigned platform_count(const BoplRlState *state) {
    unsigned count = 0;
    for (unsigned i = 0; i < state->header.entity_count; i++)
        if (state->entities[i].kind == 2) count++;
    return count;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/duplicator_platform_lifetime.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }

    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 123;
    config.map = 0;
    config.stocks = 5;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1400;
    config.player[0].abilities[0] = 29;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    for (int backend = 0; backend < 2; backend++) {
        int status = bopl_rl_reset(world[backend], &config);
        if (status != BOPL_RL_OK) {
            fprintf(stderr, "reset backend %d failed: %s\n", backend,
                    bopl_rl_error(world[backend]));
            return 1;
        }
    }

    const unsigned initial = platform_count(bopl_rl_state(world[0]));
    unsigned casts = 0, held = 0, last_cast = 0;
    unsigned first_extra_tick = 0, second_extra_tick = 0, first_removed_tick = 0;
    for (unsigned tick = 1; tick <= 1200; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (casts < 2 && held == 0 && tick > 180 && tick > last_cast + 120 &&
            !(before->slimes[0].ability_flags[0] & 8) &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
            casts++;
            held = 60;
            last_cast = tick;
        }
        if (held > 0) {
            action[0].buttons = 2;
            action[0].angle = 192;
            held--;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "tick %u backend %d failed: %s\n", tick,
                        backend, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        unsigned reference_count = platform_count(bopl_rl_state(world[0]));
        unsigned rust_count = platform_count(bopl_rl_state(world[1]));
        if (reference_count != rust_count) {
            fprintf(stderr, "tick %u platform count differs: %u/%u\n", tick,
                    reference_count, rust_count);
            return 1;
        }
        if (!first_extra_tick && reference_count == initial + 1)
            first_extra_tick = tick;
        if (!second_extra_tick && reference_count == initial + 2)
            second_extra_tick = tick;
        if (second_extra_tick && !first_removed_tick && reference_count == initial + 1)
            first_removed_tick = tick;
    }
    if (casts != 2 || !first_extra_tick || !second_extra_tick || !first_removed_tick) {
        fprintf(stderr,
                "incomplete casts=%u first=%u second=%u removed=%u initial=%u final=%u\n",
                casts, first_extra_tick, second_extra_tick, first_removed_tick, initial,
                platform_count(bopl_rl_state(world[0])));
        return 1;
    }
    printf("DUPLICATOR_PLATFORM_LIFETIME_GOLDEN_OK first=%u second=%u removed=%u\n",
           first_extra_tick, second_extra_tick, first_removed_tick);
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    return 0;
}
