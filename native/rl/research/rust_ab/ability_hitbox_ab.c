/* Incoming-arrow A/B while the target changes from its slime hurtbox to Bow. */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct { int death_tick; unsigned victim, arrows; } Result;

static Result run(BoplRl *world, int jump_tick, int bow_tick) {
    BoplRlConfig c;
    bopl_rl_default_config(&c, 2);
    c.seed=7; c.map=0; c.stocks=0; c.warmup_ticks=240;
    for (int p=0; p<2; p++) {
        c.player[p].abilities[0]=2;
        c.player[p].abilities[1]=4;
        c.player[p].abilities[2]=14;
    }
    if (bopl_rl_reset(world,&c)) return (Result){.death_tick=-2};
    Result out={.death_tick=-1};
    BoplRlAction a[2]={{0}};
    for (int tick=1; tick<=210 && out.death_tick<0; tick++) {
        /* Player 1 jumps, then fires right from an elevated Bow. */
        a[0].angle=1;
        a[0].buttons=(tick==jump_tick?1:0)|((tick>=80 && tick<=139)?4:0);
        /* Player 2 changes to the larger Bow hurtbox while staying grounded. */
        a[1].angle=tick>=bow_tick?64:0;
        a[1].buttons=tick>=bow_tick?4:0;
        if (bopl_rl_step(world,a,2,1)) break;
        const BoplRlState *s=bopl_rl_state(world);
        for (unsigned p=0;p<2;p++) if (s->players[p].deaths) {
            out.death_tick=tick; out.victim=s->players[p].id;
        }
        out.arrows=0;
        for (unsigned i=0;i<s->header.entity_count;i++)
            if (s->entities[i].layer==9 && (s->entities[i].flags&1)) out.arrows++;
    }
    return out;
}

int main(int argc,char **argv) {
    if (argc!=3) {
        fprintf(stderr,"usage: %s <BoplSimNative.so> <libbopl_sim.so>\n",argv[0]);
        return 2;
    }
    char error[512]={0}; BoplRl *world[2];
    for (int b=0;b<2;b++) {
        world[b]=bopl_rl_open(argv[b+1],"/tmp/gate_bootstrap.rep",
                              "/tmp/ability_hitbox_ab.log",120000,error,sizeof error);
        if (!world[b]) { fprintf(stderr,"%s\n",error); return 1; }
    }
    Result n=run(world[0],40,80),r=run(world[1],40,80);
    printf("jump=40 bow=80 net=%d/%u rust=%d/%u arrows=%u/%u%s\n",
           n.death_tick,n.victim,r.death_tick,r.victim,n.arrows,r.arrows,
           (n.death_tick!=r.death_tick || n.victim!=r.victim)?" mismatch":"");
    for (int b=0;b<2;b++) bopl_rl_close(world[b]);
    return 0;
}
