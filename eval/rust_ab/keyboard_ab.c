/* Keyboard-input ABI A/B. A keyboard-configured player receives movement from
 * W/A/S/D while its joystick byte remains an independent ability aim.
 *
 * cc -O2 -I sim/abi -o /tmp/keyboard_ab eval/rust_ab/keyboard_ab.c sim/abi/bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <stdio.h>

static double f(long long raw) { return (double)raw / 4294967296.0; }

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *world[2];
    for (int b = 0; b < 2; b++) {
        world[b] = bopl_rl_open(argv[b + 1], "/tmp/gate_bootstrap.rep",
                                "/tmp/keyboard_ab.log", 120000, error, sizeof error);
        if (!world[b]) {
            fprintf(stderr, "open: %s\n", error);
            return 1;
        }
    }

    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7;
    config.map = 0;
    config.stocks = 0;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 7200;
    for (int p = 0; p < 2; p++) {
        config.player[p].keyboard = 1;
        config.player[p].abilities[0] = 2;
        config.player[p].abilities[1] = 4;
        config.player[p].abilities[2] = 14;
    }
    for (int b = 0; b < 2; b++) {
        if (bopl_rl_reset(world[b], &config) != BOPL_RL_OK) {
            fprintf(stderr, "reset: %s\n", bopl_rl_error(world[b]));
            return 1;
        }
    }

    BoplRlAction action[2] = {{0}};
    action[0].buttons = 512; /* D */
    for (int tick = 1; tick <= 40; tick++) {
        for (int b = 0; b < 2; b++) {
            if (bopl_rl_step(world[b], action, 2, 1) != BOPL_RL_OK) {
                fprintf(stderr, "step: %s\n", bopl_rl_error(world[b]));
                return 1;
            }
        }
        if (tick == 1 || tick == 10 || tick == 20 || tick == 40) {
            const BoplRlState *net = bopl_rl_state(world[0]);
            const BoplRlState *rust = bopl_rl_state(world[1]);
            printf("tick %2d net=(%.9f,%.9f) rust=(%.9f,%.9f) state=%u/%u\n",
                   tick, f(net->players[0].x), f(net->players[0].y),
                   f(rust->players[0].x), f(rust->players[0].y),
                   net->slimes[0].state, rust->slimes[0].state);
        }
    }
    return 0;
}
