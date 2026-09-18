/* Flying Grappling Hook attachment to a grounded third player. The contact
 * occurs after that player's update and must not apply the grounded-rope
 * correction until the following tick.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/grapple_player_attach_ab \
 *   research/rust_ab/grapple_player_attach_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static unsigned kind_count(const BoplRlState *state, unsigned kind) {
    unsigned count = 0;
    for (unsigned i = 0; i < state->header.entity_count; i++)
        count += state->entities[i].kind == kind;
    return count;
}

static int compare(const BoplRlState *reference, const BoplRlState *rust,
                   unsigned tick) {
    if (reference->header.tick != rust->header.tick ||
        reference->header.episode_ticks != rust->header.episode_ticks ||
        reference->header.flags != rust->header.flags ||
        reference->header.player_count != rust->header.player_count ||
        reference->header.entity_count != rust->header.entity_count ||
        reference->header.slime_count != rust->header.slime_count ||
        reference->header.simulation_time != rust->header.simulation_time ||
        reference->header.player_time_scale != rust->header.player_time_scale) {
        fprintf(stderr, "tick %u: header differs (entities %u/%u)\n",
                tick, reference->header.entity_count,
                rust->header.entity_count);
        return 1;
    }
    for (unsigned i = 0; i < reference->header.player_count; i++) {
        if (memcmp(&reference->players[i], &rust->players[i],
                   sizeof reference->players[i]) != 0 ||
            memcmp(&reference->telemetry[i], &rust->telemetry[i],
                   sizeof reference->telemetry[i]) != 0) {
            fprintf(stderr, "tick %u: player %u differs pos (%" PRId64
                    ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") flags %u/%u ability-time %" PRId64 "/%" PRId64
                    " activations %u/%u\n", tick, i,
                    reference->players[i].x, reference->players[i].y,
                    rust->players[i].x, rust->players[i].y,
                    reference->players[i].flags, rust->players[i].flags,
                    reference->players[i].ability_time,
                    rust->players[i].ability_time,
                    reference->telemetry[i].ability_activations[1],
                    rust->telemetry[i].ability_activations[1]);
            return 1;
        }
    }
    for (unsigned i = 0; i < reference->header.slime_count; i++) {
        if (memcmp(&reference->slimes[i], &rust->slimes[i],
                   sizeof reference->slimes[i]) != 0) {
            const BoplRlSlime *a = &reference->slimes[i], *b = &rust->slimes[i];
            fprintf(stderr,
                    "tick %u: slime %u differs: position (%" PRId64
                    ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") velocity "
                    "(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                    tick, i, a->x, a->y, b->x, b->y,
                    a->vx, a->vy, b->vx, b->vy);
            fprintf(stderr,
                    "  flags %u/%u state %u/%u rotation %" PRId64 "/%" PRId64
                    " scale %" PRId64 "/%" PRId64 " cooldown %" PRId64
                    "/%" PRId64 " timer %" PRId64 "/%" PRId64 "\n",
                    a->flags, b->flags, a->state, b->state,
                    a->rotation, b->rotation, a->scale, b->scale,
                    a->cooldowns[0], b->cooldowns[0], a->timers[0], b->timers[0]);
            const unsigned char *aa = (const unsigned char *)a;
            const unsigned char *bb = (const unsigned char *)b;
            for (unsigned z = 0; z < sizeof *a; z++)
                if (aa[z] != bb[z])
                    fprintf(stderr, "  byte %u %u/%u\n", z, aa[z], bb[z]);
            return 1;
        }
    }
    unsigned ri = 0;
    for (unsigned i = 0; i < reference->header.entity_count; i++) {
        if (reference->entities[i].kind != 3) continue;
        while (ri < rust->header.entity_count && rust->entities[ri].kind != 3) ri++;
        if (ri == rust->header.entity_count ||
            memcmp(&reference->entities[i], &rust->entities[ri],
                   sizeof reference->entities[i]) != 0) {
            const BoplRlEntity *a = &reference->entities[i];
            const BoplRlEntity *b = ri < rust->header.entity_count
                ? &rust->entities[ri] : a;
            fprintf(stderr, "tick %u: rope segment %u hierarchy %u/%u differs pos (%" PRId64
                    ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") right (%" PRId64
                    ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n", tick, i,
                    a->hierarchy, b->hierarchy, a->x, a->y, b->x, b->y, a->right_x, a->right_y,
                    b->right_x, b->right_y);
            return 1;
        }
        ri++;
    }
    if (kind_count(reference, 3) != kind_count(rust, 3)) {
        fprintf(stderr, "tick %u: rope count differs\n", tick);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1],
                                      "/tmp/gate_bootstrap.rep",
                                      "/tmp/grapple_player_attach_ab.log", 180000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }

    BoplRlConfig config;
    bopl_rl_default_config(&config, 3);
    config.seed = 7;
    config.map = 0;
    config.stocks = 10;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 3000;
    config.player[1].team = 0;
    config.player[2].team = 1;
    for (unsigned player = 0; player < 3; player++) {
        config.player[player].keyboard = 1;
        config.player[player].abilities[0] = 27;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    for (int backend = 0; backend < 2; backend++) {
        int status = bopl_rl_reset(world[backend], &config);
        if (status != BOPL_RL_OK) {
            fprintf(stderr, "reset backend %d failed: %s\n", backend,
                    bopl_rl_error(world[backend]));
            return 1;
        }
    }

    unsigned attach_tick = 0;
    for (unsigned tick = 1; tick <= 198; tick++) {
        BoplRlAction actions[3] = {{0}};
        if (tick >= 181 && tick < 220) {
            actions[0].buttons = 2;
            actions[0].angle = 1;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], actions, 3, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "tick %u backend %d failed: %s\n", tick,
                        backend, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        const BoplRlState *rust = bopl_rl_state(world[1]);
        if (!attach_tick && kind_count(reference, 3) == 27 &&
            reference->slimes[2].vx == 0 && reference->slimes[2].vy == 0)
            attach_tick = tick;
        if (compare(reference, rust, tick)) {
            fprintf(stderr, "attach=%u rope=%u\n", attach_tick,
                    kind_count(reference, 3));
            return 1;
        }
    }
    unsigned final_ropes = kind_count(bopl_rl_state(world[0]), 3);
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (attach_tick != 197 || final_ropes != 27) {
        fprintf(stderr, "incomplete coverage: attach=%u ropes=%u\n",
                attach_tick, final_ropes);
        return 1;
    }
    printf("GRAPPLE_PLAYER_ATTACH_GOLDEN_OK ticks=198 attach=%u ropes=%u\n",
           attach_tick, final_ropes);
    return 0;
}
