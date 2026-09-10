/* Bow phase A/B: activation, charge flags, release and arrow creation. */
#include "bopl_rl.h"
#include <math.h>
#include <stdio.h>

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

typedef struct {
    unsigned slime_flags, ability_flags, bow_flags, arrows;
    double px, py, bvx, bvy, ax, ay, avx, avy, arot;
} Sample;

static Sample sample(const BoplRlState *s) {
    Sample o = {.px=f(s->players[0].x),.py=f(s->players[0].y),
                .ax=NAN,.ay=NAN,.avx=NAN,.avy=NAN};
    for (uint32_t i=0;i<s->header.slime_count;i++) if (s->slimes[i].owner==1) {
        o.slime_flags=s->slimes[i].flags; o.ability_flags=s->slimes[i].ability_flags[1];
    }
    for (uint32_t i=0;i<s->header.entity_count;i++) {
        const BoplRlEntity *e=&s->entities[i];
        if (e->kind==1 && e->hierarchy>=100000022 && e->hierarchy<100000027) {
            o.bow_flags=e->flags; o.bvx=f(e->vx); o.bvy=f(e->vy);
        }
        if (e->layer==9 && e->owner==1 && (e->flags&1)) {
            o.arrows++; o.ax=f(e->x); o.ay=f(e->y); o.avx=f(e->vx); o.avy=f(e->vy);
            o.arot=f(e->rotation);
        }
    }
    return o;
}

int main(int argc,char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n", argv[0]);
        return 2;
    }
    char error[512]={0}; BoplRl *w[2];
    for (int b=0; b<2; b++) {
        w[b]=bopl_rl_open(argv[b+1],"/proc/self/cwd/native/fixtures/movement.rep",
          "/proc/self/cwd/native/target/bow_phase_ab.log",120000,error,sizeof error);
        if (!w[b]) { fprintf(stderr,"%s\n",error); return 1; }
    }
    BoplRlConfig c;bopl_rl_default_config(&c,2);c.seed=7;c.map=0;c.stocks=0;c.warmup_ticks=240;
    for (int p=0; p<2; p++) {
        c.player[p].abilities[0]=2;
        c.player[p].abilities[1]=4;
        c.player[p].abilities[2]=14;
    }
    for (int b=0; b<2; b++) if (bopl_rl_reset(w[b],&c)) return 1;
    puts("tick,net_slime_flags,net_ability_flags,net_bow_flags,net_arrows,net_px,net_py,net_bvx,net_bvy,net_ax,net_ay,net_avx,net_avy,"
         "net_arot,rust_slime_flags,rust_ability_flags,rust_bow_flags,rust_arrows,rust_px,rust_py,rust_bvx,rust_bvy,rust_ax,rust_ay,rust_avx,rust_avy,rust_arot");
    BoplRlAction a[2]={{0}};a[0].angle=192;
    for (unsigned tick=1; tick<=345; tick++) {
        a[0].buttons=tick==260?1:(tick>=276&&tick<=332?4:0);
        for (int b=0; b<2; b++) if (bopl_rl_step(w[b],a,2,1)) return 1;
        if (tick<270) continue;
        Sample n=sample(bopl_rl_state(w[0])),r=sample(bopl_rl_state(w[1]));
        printf("%u,%u,%u,%u,%u,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%u,%u,%u,%u,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f\n",
          tick,n.slime_flags,n.ability_flags,n.bow_flags,n.arrows,n.px,n.py,n.bvx,n.bvy,n.ax,n.ay,n.avx,n.avy,n.arot,
          r.slime_flags,r.ability_flags,r.bow_flags,r.arrows,r.px,r.py,r.bvx,r.bvy,r.ax,r.ay,r.avx,r.avy,r.arot);
    }
    bopl_rl_close(w[0]);bopl_rl_close(w[1]);return 0;
}
