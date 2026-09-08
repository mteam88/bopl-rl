/* Generate a lossless episode through the public C API for replay verification. */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc<6 || argc>9) { fprintf(stderr, "record <game> <bootstrap.rep> <game.log> <output.rep> <players> [ability_group=0] [ticks=180] [map=0]\n"); return 2; }
    char error[512];
    BoplRl *world=bopl_rl_open(argv[1], argv[2], argv[3], 60000, error, sizeof(error));
    if (!world) { fprintf(stderr, "%s\n", error); return 1; }
    BoplRlConfig config; bopl_rl_default_config(&config, (uint32_t)atoi(argv[5]));
    uint32_t group=argc>6?(uint32_t)atoi(argv[6]):0;
    config.seed=785921; config.max_episode_ticks=argc>7?(uint32_t)atoi(argv[7]):180;
    config.map=argc>8?(uint32_t)atoi(argv[8]):0;
    if (group>4) { bopl_rl_close(world); return 2; }
    if (group) {
        config.seed+=group; config.stocks=5;
        for (uint32_t i=0; i<config.players; i++) for (uint32_t slot=0; slot<3; slot++) {
            uint32_t ability=1+18*(group-1)+3*i+slot;
            config.player[i].abilities[slot]=group==4?(slot==0?2:slot==1?4:14):(ability<39?ability:2);
        }
    }
    int status=bopl_rl_reset(world, &config);
    uint32_t step=0;
    while (!status && !(bopl_rl_state(world)->header.flags&BOPL_RL_TERMINAL)) {
        BoplRlAction actions[6]={{0}};
        for (uint32_t i=0; i<config.players; i++) {
            actions[i].angle=(uint8_t)(1+(step*17+i*39)%255);
            if (group) {
                uint32_t phase=(4*step)%120;
                actions[i].buttons=(uint16_t)(phase<12?2:phase<24?4:phase<36?8:phase<40?15:0);
                if (step%17==0) actions[i].buttons|=1;
            } else actions[i].buttons=(uint16_t)((step+3*i)%16);
        }
        status=bopl_rl_step(world, actions, config.players, 4); step++;
    }
    if (!status) {
        const void *bytes=NULL; size_t length=0;
        uint32_t ticks=bopl_rl_state(world)->header.tick;
        status=bopl_rl_replay(world, &bytes, &length);
        if (!status) {
            FILE *file=fopen(argv[4], "wb");
            if (!file) status=1;
            else { if (fwrite(bytes, 1, length, file)!=length) status=1; if (fclose(file)) status=1; }
            if (!status) printf("{\"ticks\":%u,\"players\":%u,\"replay_bytes\":%zu}\n", ticks, config.players, length);
        }
    }
    if (status) fprintf(stderr, "Recording failed (%d): %s\n", status, bopl_rl_error(world));
    bopl_rl_close(world);
    return status?1:0;
}
