/* SlimeController state A/B: idle, run right, turn, stop; print flags/timers/state.
 * cc -O2 -I sim/abi -o /tmp/state_ab /tmp/state_ab.c bopl_rl.c -ldl -lm */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>

static double f(long long raw) { return (double)raw / 4294967296.0; }

int main(int argc, char **argv) {
    char error[512];
    BoplRl *w[2];
    for (int b = 0; b < 2; b++) {
        w[b] = bopl_rl_open(argv[1 + b], "/tmp/gate_bootstrap.rep", "/tmp/bow_ab.log", 120000, error, sizeof error);
        if (!w[b]) { fprintf(stderr, "open: %s\n", error); return 1; }
    }
    
    BoplRlConfig c;
    bopl_rl_default_config(&c, 2);
    c.seed = 7; c.map = 0; c.stocks = 0; c.warmup_ticks = 240; c.max_episode_ticks = 7200;
    for (int p = 0; p < 2; p++) {
        c.player[p].abilities[0] = 2; c.player[p].abilities[1] = 4; c.player[p].abilities[2] = 14;
    }
    for (int b = 0; b < 2; b++)
        if (bopl_rl_reset(w[b], &c) != BOPL_RL_OK) { fprintf(stderr, "reset\n"); return 1; }
    BoplRlAction a[2];
    /* wait 20 decisions (cooldowns ready), then hold arrow, then release */
    for (int d = 1; d <= 40; d++) {
        /* 1-10 idle, 11-18 run right, 19-26 run left (turn), 27-40 neutral (stop->idle) */
        unsigned char angle = d <= 10 ? 0 : d <= 18 ? 64 : d <= 26 ? 128 : 0;
        for (int p = 0; p < 2; p++) { a[p].angle = angle; a[p].reserved = 0; a[p].buttons = 0; }
        for (int b = 0; b < 2; b++) bopl_rl_step(w[b], a, 2, 4);
        const BoplRlState *s0 = bopl_rl_state(w[0]), *s1 = bopl_rl_state(w[1]);
        const BoplRlSlime *m0 = &s0->slimes[0], *m1 = &s1->slimes[0];
        printf("d%02d ang%u | net f=%3u cd=%5.2f tm=%6.2f st=%u | rust f=%3u cd=%5.2f tm=%6.2f st=%u\n",
               d, (unsigned)a[0].angle,
               m0->ability_flags[1], f(m0->cooldowns[1]), f(m0->timers[1]), m0->state,
               m1->ability_flags[1], f(m1->cooldowns[1]), f(m1->timers[1]), m1->state);
    }
    return 0;
}
