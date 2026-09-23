/* Replay-driven differential for a saved RL game.
 *
 * usage: <binary> <BoplSimNative.so> <libbopl_sim.so> <inputs>
 *
 * <inputs> is produced from a BOPLREP2 .rep by the decoder in
 * native/rl/research/replay_data.py: a header of
 *   u32 ticks, u32 players, u32 seed, u32 level, i32 stocks,
 *   u32 abilities[players][3]
 * followed by per tick, per player: u8 angle, u16 buttons.
 * The first 240 ticks are the harness warmup and are skipped, since
 * bopl_rl_reset runs its own.
 *
 * Both backends are stepped with the identical recorded input stream and the
 * run reports the first tick at which a player's alive/deaths counters differ.
 *
 * Result on study-duplicator/evaluations/9253158912/map0-s1-ba.rep
 * (2026-09-22): the Rust port lets player 2 ride a descending platform to
 * y = -887 and beyond while still alive, thousands of ticks after crossing
 * the y = -11.3 water plane that DestroyIfOutsideSceneBounds kills at.  The
 * NativeAOT oracle never produces such a state and kills player 1 at
 * y = -11.34 on tick 7078, where the port still has it alive at y = -13.12.
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static double U(int64_t v){return (double)v/4294967296.0;}
int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: %s ref rust inputs\n",argv[0]);return 2;}
    FILE*f=fopen(argv[3],"rb"); if(!f){perror("inputs");return 1;}
    uint32_t ticks,count,seed,level; int32_t stocks;
    fread(&ticks,4,1,f);fread(&count,4,1,f);fread(&seed,4,1,f);fread(&level,4,1,f);fread(&stocks,4,1,f);
    uint32_t ab[6][3]={{0}};
    for(uint32_t i=0;i<count;i++)fread(ab[i],4,3,f);
    unsigned char*ang=malloc(ticks*count); uint16_t*btn=malloc(ticks*count*2);
    for(uint32_t t=0;t<ticks;t++)for(uint32_t c=0;c<count;c++){fread(&ang[t*count+c],1,1,f);fread(&btn[t*count+c],2,1,f);}
    fclose(f);
    printf("replay ticks=%u players=%u seed=%u level=%u stocks=%d abilities=[%u,%u,%u]\n",ticks,count,seed,level,stocks,ab[0][0],ab[0][1],ab[0][2]);
    char err[512]={0};BoplRl*w[2];
    for(int b=0;b<2;b++){w[b]=bopl_rl_open(argv[b+1],"/tmp/gate_bootstrap.rep","/tmp/replay_ab.log",600000,err,sizeof err);
        if(!w[b]){fprintf(stderr,"open %d: %s\n",b,err);return 1;}}
    BoplRlConfig c;bopl_rl_default_config(&c,count);
    c.seed=seed;c.map=level;c.stocks=stocks;c.warmup_ticks=240;c.max_episode_ticks=100000;
    for(uint32_t p=0;p<count;p++)for(int k=0;k<3;k++)c.player[p].abilities[k]=ab[p][k];
    for(int b=0;b<2;b++)if(bopl_rl_reset(w[b],&c)!=BOPL_RL_OK){fprintf(stderr,"reset %d: %s\n",b,bopl_rl_error(w[b]));return 1;}
    unsigned first_alive_diff=0;
    for(uint32_t t=240;t<ticks;t++){
        BoplRlAction act[6]={{0}};
        for(uint32_t p=0;p<count;p++){act[p].angle=ang[t*count+p];act[p].buttons=btn[t*count+p];}
        for(int b=0;b<2;b++)if(bopl_rl_step(w[b],act,count,1)!=BOPL_RL_OK){fprintf(stderr,"step %u b%d: %s\n",t,b,bopl_rl_error(w[b]));return 1;}
        const BoplRlState*A=bopl_rl_state(w[0]),*B=bopl_rl_state(w[1]);
        if((t-240)%400==0||t==ticks-1)
            printf("t=%5u REF p1=(%7.2f,%9.2f)f%u p2=(%7.2f,%9.2f)f%u | RST p1=(%7.2f,%9.2f)f%u p2=(%7.2f,%9.2f)f%u ents %u/%u\n",
                t-240,U(A->players[0].x),U(A->players[0].y),A->players[0].flags,U(A->players[1].x),U(A->players[1].y),A->players[1].flags,
                U(B->players[0].x),U(B->players[0].y),B->players[0].flags,U(B->players[1].x),U(B->players[1].y),B->players[1].flags,
                A->header.entity_count,B->header.entity_count);
        if(!first_alive_diff)for(uint32_t p=0;p<count;p++)
            if(A->players[p].alive_bodies!=B->players[p].alive_bodies||A->players[p].deaths!=B->players[p].deaths){
                first_alive_diff=t;
                printf(">>> t=%u OUTCOME DIVERGES p%u: REF alive=%u deaths=%u cod=%u y=%.2f | RST alive=%u deaths=%u cod=%u y=%.2f\n",
                    t-240,p,A->players[p].alive_bodies,A->players[p].deaths,A->players[p].cause_of_death,U(A->players[p].y),
                    B->players[p].alive_bodies,B->players[p].deaths,B->players[p].cause_of_death,U(B->players[p].y));}
    }
    const BoplRlState*A=bopl_rl_state(w[0]),*B=bopl_rl_state(w[1]);
    for(uint32_t p=0;p<count;p++)
        printf("FINAL p%u REF alive=%u deaths=%u kills=%u cod=%u y=%.2f | RST alive=%u deaths=%u kills=%u cod=%u y=%.2f\n",p,
            A->players[p].alive_bodies,A->players[p].deaths,A->players[p].kills,A->players[p].cause_of_death,U(A->players[p].y),
            B->players[p].alive_bodies,B->players[p].deaths,B->players[p].kills,B->players[p].cause_of_death,U(B->players[p].y));
    return 0;}
