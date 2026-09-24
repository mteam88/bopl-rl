// Reset cost versus warmup ticks, plus sampled-step tick cost, on one game process.
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
int main(int argc,char**argv){
    (void)argc; char error[512]; BoplRl*w=bopl_rl_open(argv[1],argv[2],argv[3],30000,error,sizeof(error));
    if(!w){fprintf(stderr,"%s\n",error);return 1;}
    BoplRlConfig c; bopl_rl_default_config(&c,2); c.stocks=3; c.max_episode_ticks=7200; c.seed=5;
    for(int i=0;i<2;i++){c.player[i].abilities[0]=2;c.player[i].abilities[1]=4;c.player[i].abilities[2]=14;}
    BoplRlAction a[2]={{0}};
    unsigned repeats[]={16,1,4,16,1,4,8,2};
    for(int k=0;k<8;k++){
        if(bopl_rl_reset(w,&c)) return 1;
        unsigned r=repeats[k], ticks=960; double t=now();
        for(unsigned s=0;s<ticks/r;s++) if(bopl_rl_step(w,a,2,r)){fprintf(stderr,"%s\n",bopl_rl_error(w));return 1;}
        double per_tick=(now()-t)/ticks;
        printf("repeat=%2u  commands/frame=%2u  per-tick=%.4f ms  per-command=%.4f ms  per-frame=%.3f ms\n",r,16/r,per_tick,per_tick*r,per_tick*16);
    }
    bopl_rl_close(w); return 0;
}
