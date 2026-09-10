/* Grounded Bow fire/landing exit order and following slime movement. */
#include "bopl_rl.h"
#include <stdio.h>

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n", argv[0]);
        return 2;
    }
    char error[512] = {0}; BoplRl *world[2];
    for (int b = 0; b < 2; b++) {
        world[b] = bopl_rl_open(argv[b + 1], "/tmp/gate_bootstrap.rep",
                                "/tmp/bow_ground_exit_ab.log", 120000,
                                error, sizeof error);
        if (!world[b]) { fprintf(stderr, "%s\n", error); return 1; }
    }
    BoplRlConfig c; bopl_rl_default_config(&c, 2);
    c.seed = 7; c.map = 0; c.stocks = 0; c.warmup_ticks = 400;
    for (int p = 0; p < 2; p++) {
        c.player[p].abilities[0] = 2; c.player[p].abilities[1] = 4;
        c.player[p].abilities[2] = 14;
    }
    for (int b = 0; b < 2; b++) if (bopl_rl_reset(world[b], &c)) return 1;
    puts("tick,net_x,rust_x,net_bow,rust_bow");
    BoplRlAction a[2] = {{0}};
    for (unsigned tick = 1; tick <= 100; tick++) {
        a[1].angle = tick < 64 ? 129 : 128;
        a[1].buttons = (tick == 48 ? 1 : 0) | (tick >= 64 && tick < 70 ? 4 : 0);
        for (int b = 0; b < 2; b++) if (bopl_rl_step(world[b], a, 2, 1)) return 1;
        if (tick >= 62) {
            const BoplRlState *n = bopl_rl_state(world[0]), *r = bopl_rl_state(world[1]);
            printf("%u,%.9f,%.9f,%u,%u\n", tick,
                   f(n->players[1].x), f(r->players[1].x),
                   !!(n->slimes[1].ability_flags[1] & 8),
                   !!(r->slimes[1].ability_flags[1] & 8));
        }
    }
    for (int b = 0; b < 2; b++) bopl_rl_close(world[b]);
    return 0;
}
