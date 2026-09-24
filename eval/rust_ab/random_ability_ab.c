/* Random loadout resolution golden: compare the concrete ids and authored
 * cooldowns produced by GameSessionHandler.SpawnPlayers with the Rust port.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/random_ability_ab \
 *   eval/rust_ab/random_ability_ab.c sim/abi/bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static int reset(BoplRl *world, BoplRlConfig *config, const char *label) {
    int status = bopl_rl_reset(world, config);
    if (status != BOPL_RL_OK) {
        fprintf(stderr, "%s reset failed (%d): %s\n", label, status,
                bopl_rl_error(world));
        return 1;
    }
    return 0;
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
                                      "/tmp/random_ability_ab.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }

    static const uint32_t seeds[] = {1, 7, 37, 785922, 0x12345678u, 0xfffffffbu};
    unsigned comparisons = 0;
    for (unsigned all_random = 0; all_random < 2; all_random++) {
        for (unsigned n = 0; n < sizeof seeds / sizeof seeds[0]; n++) {
            BoplRlConfig config;
            bopl_rl_default_config(&config, 2);
            config.seed = seeds[n];
            config.map = n % 5;
            config.stocks = 5;
            config.warmup_ticks = 240;
            config.max_episode_ticks = 60;
            if (all_random) {
                for (unsigned player = 0; player < 2; player++)
                    for (unsigned slot = 0; slot < 3; slot++)
                        config.player[player].abilities[slot] = 1;
            } else {
                config.player[0].abilities[0] = 1;
                config.player[0].abilities[1] = 2;
                config.player[0].abilities[2] = 3;
                config.player[1].abilities[0] = 4;
                config.player[1].abilities[1] = 5;
                config.player[1].abilities[2] = 6;
            }
            if (reset(world[0], &config, "reference") ||
                reset(world[1], &config, "rust"))
                return 1;
            const BoplRlState *reference = bopl_rl_state(world[0]);
            const BoplRlState *rust = bopl_rl_state(world[1]);
            if (reference->header.slime_count != rust->header.slime_count) {
                fprintf(stderr, "seed %" PRIu32 ": slime count %u != %u\n", seeds[n],
                        reference->header.slime_count, rust->header.slime_count);
                return 1;
            }
            for (unsigned player = 0; player < reference->header.slime_count; player++) {
                for (unsigned slot = 0; slot < 3; slot++) {
                    uint32_t expected_id = reference->slimes[player].abilities[slot];
                    uint32_t actual_id = rust->slimes[player].abilities[slot];
                    int64_t expected_cd = reference->slimes[player].cooldowns[slot];
                    int64_t actual_cd = rust->slimes[player].cooldowns[slot];
                    comparisons++;
                    if (expected_id != actual_id || expected_cd != actual_cd) {
                        fprintf(stderr,
                                "seed %" PRIu32 " all_random=%u player=%u slot=%u: "
                                "id %u != %u or cooldown %" PRId64 " != %" PRId64 "\n",
                                seeds[n], all_random, player, slot, expected_id,
                                actual_id, expected_cd, actual_cd);
                        return 1;
                    }
                }
            }
        }
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    printf("RANDOM_ABILITY_GOLDEN_OK comparisons=%u\n", comparisons);
    return 0;
}
