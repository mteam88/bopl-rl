/* Fire a full-charge arrow at each angle byte; print arrow velocity.
 * cc -O2 -I. -o /tmp/aim_map /tmp/aim_map.c bopl_rl.c -ldl -lm */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>

static double f(long long raw) { return (double)raw / 4294967296.0; }

int main(int argc, char **argv) {
    char error[512];
    for (int ang = 0; ang < 256; ang += 32) {
        BoplRl *w = bopl_rl_open(argv[1], "/tmp/gate_bootstrap.rep", "/tmp/aim.log", 120000, error, sizeof error);
        if (!w) { fprintf(stderr, "open: %s\n", error); return 1; }
        BoplRlConfig c;
        bopl_rl_default_config(&c, 2);
        c.seed = 7; c.map = 0; c.stocks = 0; c.warmup_ticks = 240; c.max_episode_ticks = 7200;
        for (int p = 0; p < 2; p++) {
            c.player[p].abilities[0] = 2; c.player[p].abilities[1] = 4; c.player[p].abilities[2] = 14;
        }
        bopl_rl_reset(w, &c);
        BoplRlAction a[2] = {{0}};
        double vx = 0, vy = 0; int seen = 0;
        for (int d = 1; d <= 42 && !seen; d++) {
            unsigned short buttons = (d > 20 && d <= 36) ? 4 : 0;
            a[0].angle = (unsigned char)ang; a[0].buttons = buttons;
            a[1].angle = 0; a[1].buttons = 0;
            bopl_rl_step(w, a, 2, 4);
            const BoplRlState *s = bopl_rl_state(w);
            for (unsigned i = 0; i < s->header.entity_count; i++) {
                const BoplRlEntity *e = &s->entities[i];
                if (e->layer == 9 && (e->flags & 1) && e->owner == 1) {
                    vx = f(e->vx); vy = f(e->vy); seen = 1;
                }
            }
        }
        printf("angle %3d -> arrow v=(%7.2f, %7.2f)\n", ang, vx, vy);
        bopl_rl_close(w);
    }
    return 0;
}
