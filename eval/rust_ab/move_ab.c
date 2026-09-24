/* Fixed-action movement A/B: angle sweep, print positions per decision.
 * cc -O2 -o /tmp/move_ab /tmp/move_ab.c bopl_rl.c -ldl -lm */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>

static double f(long long raw) { return (double)raw / 4294967296.0; }

int main(int argc, char **argv) {
    char error[512];
    BoplRl *w[2];
    for (int b = 0; b < 2; b++) {
        w[b] = bopl_rl_open(argv[1 + b], "/tmp/gate_bootstrap.rep", "/tmp/move_ab.log",
                            120000, error, sizeof error);
        if (!w[b]) { fprintf(stderr, "open: %s\n", error); return 1; }
    }
    int angle = argc > 3 ? atoi(argv[3]) : 64;
    int buttons = argc > 4 ? atoi(argv[4]) : 0;
    BoplRlConfig c;
    bopl_rl_default_config(&c, 2);
    c.seed = 7; c.map = 0; c.stocks = 0; c.warmup_ticks = 240; c.max_episode_ticks = 7200;
    for (int p = 0; p < 2; p++) {
        c.player[p].abilities[0] = 2; c.player[p].abilities[1] = 4; c.player[p].abilities[2] = 14;
    }
    for (int b = 0; b < 2; b++)
        if (bopl_rl_reset(w[b], &c) != BOPL_RL_OK) { fprintf(stderr, "reset\n"); return 1; }
    BoplRlAction a[2];
    for (int d = 1; d <= 10; d++) {
        for (int p = 0; p < 2; p++) { a[p].angle = (unsigned char)angle; a[p].reserved = 0; a[p].buttons = (unsigned short)buttons; }
        for (int b = 0; b < 2; b++) bopl_rl_step(w[b], a, 2, 4);
        const BoplRlState *s0 = bopl_rl_state(w[0]), *s1 = bopl_rl_state(w[1]);
        printf("d%02d p0 net=(%8.4f,%8.4f) rust=(%8.4f,%8.4f)  p1 net=(%8.4f,%8.4f) rust=(%8.4f,%8.4f)\n",
               d, f(s0->players[0].x), f(s0->players[0].y), f(s1->players[0].x), f(s1->players[0].y),
               f(s0->players[1].x), f(s0->players[1].y), f(s1->players[1].x), f(s1->players[1].y));
    }
    return 0;
}
