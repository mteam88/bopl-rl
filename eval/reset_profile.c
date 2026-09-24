// Reset cost versus warmup ticks, plus sampled-step tick cost, on one game process.
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
int main(int argc,char**argv){
    (void)argc; char error[512]; BoplRl*w=bopl_rl_open(argv[1],argv[2],argv[3],30000,error,sizeof(error));
    if(!w){fprintf(stderr,"%s\n",error);return 1;}
    BoplRlConfig c; bopl_rl_default_config(&c,2); c.stocks=3; c.max_episode_ticks=3600;
    for(int i=0;i<2;i++){c.player[i].abilities[0]=2;c.player[i].abilities[1]=4;c.player[i].abilities[2]=14;}
    BoplRlAction a[2]={{0}};
    unsigned warm[]={240,240,120,0,240,0,120,240,0,120};
    for(int k=0;k<10;k++){
        c.seed=100+k; c.warmup_ticks=warm[k];
        double t=now(); if(bopl_rl_reset(w,&c)){fprintf(stderr,"%s\n",bopl_rl_error(w));return 1;} double r=now()-t;
        t=now(); for(int s=0;s<25;s++) if(bopl_rl_step_sampled(w,a,2,3,1)){fprintf(stderr,"%s\n",bopl_rl_error(w));return 1;} double st=(now()-t)/100.0;
        printf("warmup=%3u reset=%.1f ms  step-tick=%.3f ms  ticks_after_reset=%u\n",warm[k],r,st,bopl_rl_state(w)->header.tick);
    }
    bopl_rl_close(w); return 0;
}
