/* Web Swing differential golden: misses on four maps and the complete map-3
 * latch, swing, release boost, detached-strand observation, and cleanup.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/web_swing_ability_ab \
 *   eval/rust_ab/web_swing_ability_ab.c sim/abi/bopl_rl.c -ldl -lm
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
                   unsigned map, unsigned tick) {
    if (reference->header.tick != rust->header.tick ||
        reference->header.episode_ticks != rust->header.episode_ticks ||
        reference->header.flags != rust->header.flags ||
        reference->header.player_count != rust->header.player_count ||
        reference->header.entity_count != rust->header.entity_count ||
        reference->header.slime_count != rust->header.slime_count ||
        reference->header.simulation_time != rust->header.simulation_time ||
        reference->header.player_time_scale != rust->header.player_time_scale) {
        fprintf(stderr, "map %u tick %u: header differs (entities %u/%u)\n",
                map, tick, reference->header.entity_count,
                rust->header.entity_count);
        return 1;
    }
    for (unsigned i = 0; i < reference->header.player_count; i++) {
        if (memcmp(&reference->players[i], &rust->players[i],
                   sizeof reference->players[i]) != 0 ||
            memcmp(&reference->telemetry[i], &rust->telemetry[i],
                   sizeof reference->telemetry[i]) != 0) {
            fprintf(stderr, "map %u tick %u: player %u differs\n", map, tick, i);
            return 1;
        }
    }
    for (unsigned i = 0; i < reference->header.slime_count; i++) {
        if (memcmp(&reference->slimes[i], &rust->slimes[i],
                   sizeof reference->slimes[i]) != 0) {
            const BoplRlSlime *a = &reference->slimes[i], *b = &rust->slimes[i];
            fprintf(stderr,
                    "map %u tick %u: slime %u differs: position (%" PRId64
                    ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") velocity "
                    "(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                    map, tick, i, a->x, a->y, b->x, b->y,
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
            fprintf(stderr, "map %u tick %u: rope segment %u differs\n",
                    map, tick, i);
            return 1;
        }
        ri++;
    }
    if (kind_count(reference, 3) != kind_count(rust, 3)) {
        fprintf(stderr, "map %u tick %u: rope count differs\n", map, tick);
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
                                      "/tmp/web_swing_ability_ab.log", 180000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }

    unsigned stages = 0, hits = 0, misses = 0, released = 0, cleaned = 0;
    for (unsigned map = 0; map < 5; map++) {
        BoplRlConfig config;
        bopl_rl_default_config(&config, 2);
        config.seed = 7;
        config.map = map;
        config.stocks = 10;
        config.warmup_ticks = 240;
        config.max_episode_ticks = 3000;
        for (unsigned player = 0; player < 2; player++) {
            config.player[player].keyboard = 1;
            config.player[player].abilities[0] = 32;
            config.player[player].abilities[1] = 2;
            config.player[player].abilities[2] = 4;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_reset(world[backend], &config);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "map %u reset backend %d failed: %s\n", map,
                        backend, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        for (unsigned tick = 1; tick <= (map == 3 ? 430 : 220); tick++) {
            BoplRlAction actions[2] = {{0}};
            if (tick >= 181 && tick < 360) {
                actions[0].buttons = 2;
                actions[0].angle = 1;
            }
            for (int backend = 0; backend < 2; backend++) {
                int status = bopl_rl_step(world[backend], actions, 2, 1);
                if (status != BOPL_RL_OK) {
                    fprintf(stderr, "map %u tick %u backend %d failed: %s\n",
                            map, tick, backend, bopl_rl_error(world[backend]));
                    return 1;
                }
            }
            const BoplRlState *reference = bopl_rl_state(world[0]);
            const BoplRlState *rust = bopl_rl_state(world[1]);
            if (compare(reference, rust, map, tick)) return 1;
            stages++;
            if (tick == 181) {
                if (kind_count(reference, 3) == 11) hits++;
                else if (kind_count(reference, 3) == 0) misses++;
            }
            if (map == 3 && tick == 360 &&
                reference->slimes[0].cooldowns[0] == INT64_C(5368709120) &&
                kind_count(reference, 3) == 11)
                released = 1;
            if (map == 3 && tick == 413 && kind_count(reference, 3) == 0)
                cleaned = 1;
        }
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (hits != 1 || misses != 4 || !released || !cleaned) {
        fprintf(stderr,
                "incomplete coverage: hits=%u misses=%u release=%u cleanup=%u\n",
                hits, misses, released, cleaned);
        return 1;
    }
    printf("WEB_SWING_GOLDEN_OK stages=%u hits=%u misses=%u\n",
           stages, hits, misses);
    return 0;
}
