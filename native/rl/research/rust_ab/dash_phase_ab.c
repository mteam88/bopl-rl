/* Dash phase A/B: current-input activation, anticipation and displacement. */
#include "bopl_rl.h"
#include <stdio.h>

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

static void sample(const BoplRlState *s, unsigned *sf, unsigned *af,
                   unsigned *df, double *px, double *py) {
    *sf = *af = *df = 0;
    *px = f(s->players[0].x); *py = f(s->players[0].y);
    for (uint32_t i = 0; i < s->header.slime_count; i++)
        if (s->slimes[i].owner == 1) {
            *sf = s->slimes[i].flags;
            *af = s->slimes[i].ability_flags[0];
        }
    for (uint32_t i = 0; i < s->header.entity_count; i++) {
        const BoplRlEntity *e = &s->entities[i];
        if (e->kind == 1 && e->hierarchy >= 100000015 &&
            e->hierarchy < 100000020) *df = e->flags;
    }
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0}; BoplRl *w[2];
    for (int b = 0; b < 2; b++) {
        w[b] = bopl_rl_open(argv[b + 1],
            "/proc/self/cwd/native/fixtures/movement.rep",
            "/proc/self/cwd/native/target/dash_phase_ab.log", 120000,
            error, sizeof error);
        if (!w[b]) { fprintf(stderr, "%s\n", error); return 1; }
    }
    BoplRlConfig c; bopl_rl_default_config(&c, 2);
    c.seed = 7; c.map = 0; c.stocks = 0; c.warmup_ticks = 240;
    for (int p = 0; p < 2; p++) {
        c.player[p].abilities[0] = 2; c.player[p].abilities[1] = 4;
        c.player[p].abilities[2] = 14;
    }
    for (int b = 0; b < 2; b++) if (bopl_rl_reset(w[b], &c)) return 1;
    puts("tick,net_slime_flags,net_ability_flags,net_dash_flags,net_x,net_y,"
         "rust_slime_flags,rust_ability_flags,rust_dash_flags,rust_x,rust_y");
    BoplRlAction a[2] = {{0}}; a[0].angle = 1;
    for (unsigned tick = 1; tick <= 310; tick++) {
        a[0].buttons = tick == 276 ? 2 : 0;
        for (int b = 0; b < 2; b++) if (bopl_rl_step(w[b], a, 2, 1)) return 1;
        if (tick < 270) continue;
        unsigned ns,nf,nd,rs,rf,rd; double nx,ny,rx,ry;
        sample(bopl_rl_state(w[0]),&ns,&nf,&nd,&nx,&ny);
        sample(bopl_rl_state(w[1]),&rs,&rf,&rd,&rx,&ry);
        printf("%u,%u,%u,%u,%.12f,%.12f,%u,%u,%u,%.12f,%.12f\n",
               tick,ns,nf,nd,nx,ny,rs,rf,rd,rx,ry);
    }
    bopl_rl_close(w[0]); bopl_rl_close(w[1]);
    return 0;
}
