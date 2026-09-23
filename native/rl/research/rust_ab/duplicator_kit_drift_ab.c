/* Duplicator/Blink/Black-hole kit divergence probe (loadout 29/6/18).
 *
 * usage: <binary> <BoplSimNative.so> <libbopl_sim.so> <map> <seed> <ticks>
 *                 <ability0> <ability1> <ability2> <button-mask>
 *
 * Drives both backends with identical scripted pseudo-random movement and
 * periodic ability presses, and reports the first tick at which a DISCRETE
 * state (entity identity/layer/flags, slime flags/state, entity or slime
 * count) differs, along with the largest positional drift seen up to that
 * point.  Positions are compared for magnitude only, not equality: a reset
 * leaves a ~22 raw-Q32 (5e-9 unit) difference in the players' resting pose on
 * a rotated platform, so bit-exact comparison is not meaningful here.
 *
 * Control result (2026-09-22, HEAD 64b5d0e1 + freshly published oracle):
 *   maps 0/1/2, loadout 2/4/14, abilities used : no divergence in 900 ticks,
 *                                                drift <= 2e-6 units.
 *   maps 0/1/2, loadout 29/6/18, no abilities  : no divergence in 900 ticks.
 *   maps 0/1/2, loadout 29/6/18, abilities used: DIVERGES at ticks 113-193.
 * Isolated per ability (slot 0 only, mask 2), the divergence reproduces for
 * Blink (181/113/193), Black hole (274/215/307, after ~12 units of drift) and
 * Duplicator (map 1 at 184, map 2 at 535; map 0 clean).
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
static int64_t d(int64_t a,int64_t b){return a>b?a-b:b-a;}
/* argv: ref rust map seed ticks a0 a1 a2 buttonmask */
int main(int argc,char**argv){
    if(argc!=10){fprintf(stderr,"usage: %s <ref.so> <rust.so> <map> <seed> <ticks> <a0> <a1> <a2> <mask>\n",argv[0]);return 2;}
    unsigned map=atoi(argv[3]),seed=atoi(argv[4]),nt=atoi(argv[5]);
    unsigned a0=atoi(argv[6]),a1=atoi(argv[7]),a2=atoi(argv[8]),bm=atoi(argv[9]);
    char err[512]={0};BoplRl*w[2];
    for(int b=0;b<2;b++){w[b]=bopl_rl_open(argv[b+1],"/tmp/gate_bootstrap.rep","/tmp/drift3.log",120000,err,sizeof err);
        if(!w[b]){fprintf(stderr,"open: %s\n",err);return 1;}}
    BoplRlConfig c;bopl_rl_default_config(&c,2);
    c.seed=seed;c.map=map;c.stocks=5;c.warmup_ticks=240;c.max_episode_ticks=100000;
    for(int p=0;p<2;p++){c.player[p].abilities[0]=a0;c.player[p].abilities[1]=a1;c.player[p].abilities[2]=a2;}
    for(int b=0;b<2;b++)if(bopl_rl_reset(w[b],&c)!=BOPL_RL_OK){fprintf(stderr,"reset: %s\n",bopl_rl_error(w[b]));return 1;}
    int64_t maxpos=0;
    for(unsigned t=0;t<=nt;t++){
        const BoplRlState*A=bopl_rl_state(w[0]),*B=bopl_rl_state(w[1]);
        if(A->header.entity_count!=B->header.entity_count||A->header.slime_count!=B->header.slime_count){
            printf("  DIVERGE tick %u HEADER ents %u/%u slimes %u/%u (drift %.4g u)\n",t,
                A->header.entity_count,B->header.entity_count,A->header.slime_count,B->header.slime_count,(double)maxpos/4294967296.0);return 3;}
        for(unsigned i=0;i<A->header.entity_count;i++){
            const BoplRlEntity*a=&A->entities[i],*b=&B->entities[i];
            if(a->id!=b->id||a->kind!=b->kind||a->layer!=b->layer||a->flags!=b->flags){
                printf("  DIVERGE tick %u ENTITY %u id %u/%u kind %u/%u layer %u/%u flags %u/%u (drift %.4g u)\n",
                    t,i,a->id,b->id,a->kind,b->kind,a->layer,b->layer,a->flags,b->flags,(double)maxpos/4294967296.0);return 3;}
            if(d(a->x,b->x)>maxpos)maxpos=d(a->x,b->x);
            if(d(a->y,b->y)>maxpos)maxpos=d(a->y,b->y);}
        for(unsigned i=0;i<A->header.slime_count;i++){
            const BoplRlSlime*a=&A->slimes[i],*b=&B->slimes[i];
            if(a->flags!=b->flags||a->state!=b->state){
                printf("  DIVERGE tick %u SLIME %u flags %u/%u state %u/%u (drift %.4g u)\n",t,i,a->flags,b->flags,a->state,b->state,(double)maxpos/4294967296.0);return 3;}
            if(d(a->x,b->x)>maxpos)maxpos=d(a->x,b->x);
            if(d(a->y,b->y)>maxpos)maxpos=d(a->y,b->y);}
        BoplRlAction act[2]={{0}};
        unsigned h=t*2654435761u;
        act[0].buttons=(h>>3)&1?1:0; act[0].angle=(unsigned char)(h>>8);
        act[1].buttons=(h>>11)&1?1:0; act[1].angle=(unsigned char)(h>>17);
        if(((h>>5)&15)==0)act[0].buttons|=bm;
        if(((h>>13)&15)==0)act[1].buttons|=bm;
        for(int b=0;b<2;b++)if(bopl_rl_step(w[b],act,2,1)!=BOPL_RL_OK){fprintf(stderr,"step: %s\n",bopl_rl_error(w[b]));return 1;}
    }
    printf("  OK %u ticks drift=%.4g u\n",nt,(double)maxpos/4294967296.0);return 0;}
