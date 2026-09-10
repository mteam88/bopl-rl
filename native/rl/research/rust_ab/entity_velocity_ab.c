/* Observation A/B: trigger PhysicsBody velocity versus PlayerBody velocity. */
#include "bopl_rl.h"
#include <stdio.h>

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

static const BoplRlEntity *body(const BoplRlState *s) {
    for (uint32_t i=0;i<s->header.entity_count;i++)
        if (s->entities[i].kind==1 && s->entities[i].layer==8 && s->entities[i].owner==1)
            return &s->entities[i];
    return NULL;
}

int main(int argc,char **argv) {
    if (argc!=3) {
        fprintf(stderr,"usage: %s <BoplSimNative.so> <libbopl_sim.so>\n",argv[0]);
        return 2;
    }
    char error[512]={0}; BoplRl *world[2];
    for (int b=0;b<2;b++) {
        world[b]=bopl_rl_open(argv[b+1],"/tmp/gate_bootstrap.rep",
                              "/tmp/entity_velocity_ab.log",120000,error,sizeof error);
        if (!world[b]) { fprintf(stderr,"%s\n",error); return 1; }
    }
    BoplRlConfig c; bopl_rl_default_config(&c,2);
    c.seed=7;c.map=0;c.stocks=0;c.warmup_ticks=240;
    for (int p=0;p<2;p++) {
        c.player[p].abilities[0]=2;c.player[p].abilities[1]=4;c.player[p].abilities[2]=14;
    }
    for (int b=0;b<2;b++) if (bopl_rl_reset(world[b],&c)) return 1;
    puts("tick,net_player_x,net_player_y,net_body_x,net_body_y,net_body_vx,net_body_vy,"
         "net_slime_x,net_slime_y,net_slime_vx,net_slime_vy,"
         "rust_player_x,rust_player_y,rust_body_x,rust_body_y,rust_body_vx,rust_body_vy,"
         "rust_slime_x,rust_slime_y,rust_slime_vx,rust_slime_vy");
    BoplRlAction a[2]={{0}}; a[0].angle=1;
    for (unsigned tick=1;tick<=12;tick++) {
        a[0].buttons=tick==1?1:0;
        for (int b=0;b<2;b++) if (bopl_rl_step(world[b],a,2,1)) return 1;
        const BoplRlState *n=bopl_rl_state(world[0]),*r=bopl_rl_state(world[1]);
        const BoplRlEntity *nb=body(n),*rb=body(r);
        printf("%u,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
               "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",tick,
               f(n->players[0].x),f(n->players[0].y),f(nb->x),f(nb->y),f(nb->vx),f(nb->vy),
               f(n->slimes[0].x),f(n->slimes[0].y),f(n->slimes[0].vx),f(n->slimes[0].vy),
               f(r->players[0].x),f(r->players[0].y),f(rb->x),f(rb->y),f(rb->vx),f(rb->vy),
               f(r->slimes[0].x),f(r->slimes[0].y),f(r->slimes[0].vx),f(r->slimes[0].vy));
    }
    for (int b=0;b<2;b++) bopl_rl_close(world[b]);
    return 0;
}
