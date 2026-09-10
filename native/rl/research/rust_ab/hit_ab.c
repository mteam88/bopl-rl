/* p0 charges and fires toward p1; report hit outcome, death tick and cause.
 * cc -O2 -I. -o /tmp/hit_ab /tmp/hit_ab.c bopl_rl.c -ldl -lm */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>

static double f(long long raw) { return (double)raw / 4294967296.0; }

int main(int argc, char **argv) {
    char error[512];
    static const int angles[] = {208, 216, 224, 232, 240, 248};
    for (int ai = 0; ai < 6; ai++) {
        for (int hold = 4; hold <= 16; hold += 6) {
            BoplRl *w = bopl_rl_open(argv[1], "/tmp/gate_bootstrap.rep", "/tmp/hit.log", 120000, error, sizeof error);
            if (!w) { fprintf(stderr, "open: %s\n", error); return 1; }
            BoplRlConfig c;
            bopl_rl_default_config(&c, 2);
            c.seed = 7; c.map = atoi(argv[2]); c.stocks = 0; c.warmup_ticks = 240; c.max_episode_ticks = 7200;
            for (int p = 0; p < 2; p++) {
                c.player[p].abilities[0] = 2; c.player[p].abilities[1] = 4; c.player[p].abilities[2] = 14;
            }
            bopl_rl_reset(w, &c);
            BoplRlAction a[2] = {{0}};
            int death_d = -1; unsigned cause = 0, victim = 0; int terminal = 0;
            double p1x = 0, p1y = 0;
            for (int d = 1; d <= 120 && !terminal; d++) {
                unsigned short buttons = (d > 20 && d <= 20 + hold) ? 4 : 0;
                a[0].angle = (unsigned char)angles[ai]; a[0].buttons = buttons;
                a[1].angle = 0; a[1].buttons = 0;
                bopl_rl_step(w, a, 2, 4);
                const BoplRlState *s = bopl_rl_state(w);
                terminal = (s->header.flags & BOPL_RL_TERMINAL) != 0;
                for (int p = 0; p < 2; p++)
                    if (s->players[p].deaths > 0 && death_d < 0) {
                        death_d = d; cause = s->players[p].cause_of_death; victim = s->players[p].id;
                    }
                p1x = f(s->players[1].x); p1y = f(s->players[1].y);
            }
            printf("ang %3d hold %2d: death_d=%3d victim=%u cause=%u p1=(%6.2f,%6.2f)\n",
                   angles[ai], hold, death_d, victim, cause, p1x, p1y);
            bopl_rl_close(w);
        }
    }
    return 0;
}
