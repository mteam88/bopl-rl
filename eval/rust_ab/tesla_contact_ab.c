/* Tesla Coil noncoincident-line contact differential. */
#include "bopl_rl.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n",
                argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(
            argv[backend + 1], "/tmp/gate_bootstrap.rep",
            "/tmp/tesla_contact_golden.log", 120000, error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }

    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 123;
    config.map = 0;
    config.stocks = 5;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 2000;
    config.player[0].abilities[0] = 23;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    for (int backend = 0; backend < 2; backend++) {
        if (bopl_rl_reset(world[backend], &config) != BOPL_RL_OK) {
            fprintf(stderr, "reset failed: %s\n", bopl_rl_error(world[backend]));
            return 1;
        }
    }

    unsigned casts = 0;
    unsigned death_tick = 0;
    for (unsigned tick = 1; tick <= 700; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (tick >= 230 && tick < 260)
            action[0].angle = 1;
        unsigned cast_after = casts == 0 ? 180 : 300;
        if (casts < 2 && tick > cast_after &&
            !(before->slimes[0].ability_flags[0] & 8) &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
            casts++;
            action[0].buttons = 2;
        }
        for (int backend = 0; backend < 2; backend++) {
            if (bopl_rl_step(world[backend], action, 2, 1) != BOPL_RL_OK) {
                fprintf(stderr, "tick %u step failed: %s\n", tick,
                        bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        const BoplRlState *rust = bopl_rl_state(world[1]);
        for (unsigned player = 0; player < 2; player++) {
            if (reference->players[player].deaths != rust->players[player].deaths ||
                memcmp(&reference->telemetry[player], &rust->telemetry[player],
                       sizeof reference->telemetry[player]) != 0) {
                fprintf(stderr, "tick %u player %u death/telemetry differs\n",
                        tick, player);
                return 1;
            }
        }
        if (death_tick == 0 && reference->players[0].deaths != 0)
            death_tick = tick;
    }

    const BoplRlState *final = bopl_rl_state(world[0]);
    if (casts != 2 || death_tick != 381 || final->players[0].deaths != 1 ||
        final->players[1].deaths != 0 ||
        final->telemetry[0].deaths_other != 1 ||
        final->telemetry[0].deaths_self != 0 ||
        final->telemetry[0].deaths_opponent != 0) {
        fprintf(stderr,
                "unexpected casts=%u death_tick=%u deaths=%u/%u "
                "other=%u self=%u opponent=%u\n",
                casts, death_tick, final->players[0].deaths,
                final->players[1].deaths,
                final->telemetry[0].deaths_other,
                final->telemetry[0].deaths_self,
                final->telemetry[0].deaths_opponent);
        return 1;
    }
    for (int backend = 0; backend < 2; backend++)
        bopl_rl_close(world[backend]);
    printf("TESLA_CONTACT_GOLDEN_OK ticks=700 death_tick=%u\n", death_tick);
    return 0;
}
