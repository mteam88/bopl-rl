/* When each backend first PUBLISHES header.sudden_death.
 *
 * GameSessionHandler.UpdateSim (sim/Gameplay/GameSessionHandler.cs:502) raises
 * the flag from inside the `updatables` pass, testing
 * Updater.SimTimeSinceLevelLoaded -- which TickSimulationInner only advances at
 * the very end of the tick (sim/Gameplay/Updater.cs:353). The observation is
 * written after the whole tick, so the value the reference publishes on the
 * pass that reports `header.tick == N` was decided from `(N - 1) * dt`.
 * `rollout::session_tick` runs after `self.tick` has already been bumped and so
 * tests `N * dt`: one pass ahead.
 *
 * This probe reports, per map, the first tick each backend publishes the flag
 * and the clock each one had there, so the offset is a number rather than an
 * argument. Exit status 1 when they disagree.
 *
 * cc -O2 -I. -o ../target/release/sudden_death_publication_ab \
 *   research/rust_ab/sudden_death_publication_ab.c bopl_rl.c -ldl -lm
 *
 * Usage: sudden_death_publication_ab <BoplSimNative.so> <libbopl_sim.so> */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>

#define TICKS 9000

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <reference.so> <candidate.so>\n", argv[0]); return 2; }
    char error[512] = {0};
    BoplRl *world[2];
    for (int b = 0; b < 2; b++) {
        world[b] = bopl_rl_open(argv[b + 1],
            "/tmp/gate_bootstrap.rep",
            "/tmp/sudden_death_publication_ab.log",
            600000, error, sizeof error);
        if (!world[b]) { fprintf(stderr, "%s\n", error); return 1; }
    }
    int failures = 0;
    for (unsigned map = 0; map < 5; map++) {
        BoplRlConfig config;
        bopl_rl_default_config(&config, 2);
        config.seed = 7; config.map = map; config.stocks = 100;
        config.warmup_ticks = 0; config.max_episode_ticks = TICKS + 100;
        for (int p = 0; p < 2; p++) {
            config.player[p].abilities[0] = 14;
            config.player[p].abilities[1] = 2;
            config.player[p].abilities[2] = 4;
        }
        for (int b = 0; b < 2; b++)
            if (bopl_rl_reset(world[b], &config) != BOPL_RL_OK) {
                fprintf(stderr, "map %u reset failed: %s\n", map, bopl_rl_error(world[b]));
                return 1;
            }
        unsigned flip[2] = {0, 0};
        double clock[2] = {0, 0};
        BoplRlAction actions[2] = {{0}};
        for (unsigned tick = 0; tick < TICKS; tick++) {
            for (int b = 0; b < 2; b++) {
                if (bopl_rl_step(world[b], actions, 2, 1) != BOPL_RL_OK) {
                    fprintf(stderr, "map %u step failed: %s\n", map, bopl_rl_error(world[b]));
                    return 1;
                }
                const BoplRlState *state = bopl_rl_state(world[b]);
                if (!flip[b] && state->header.sudden_death) {
                    flip[b] = state->header.tick;
                    clock[b] = f(state->header.simulation_time_since_level_loaded);
                }
            }
            if (flip[0] && flip[1]) break;
        }
        printf("map %u reference first sudden_death tick %u (clock %.6f)  "
               "rust %u (clock %.6f)%s\n", map, flip[0], clock[0], flip[1], clock[1],
               flip[0] == flip[1] ? "" : "   <<< MISMATCH");
        if (flip[0] != flip[1]) failures++;
    }
    bopl_rl_close(world[0]);
    bopl_rl_close(world[1]);
    printf("%s maps_mismatched=%d\n",
           failures ? "SUDDEN_DEATH_PUBLICATION_MISMATCH" : "SUDDEN_DEATH_PUBLICATION_OK", failures);
    return failures ? 1 : 0;
}
