/* Invisibility golden: cast, chain a second slot on the exact expiry frame,
 * observe restoration, then recast after the six-second restored cooldown.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/invisibility_ab \
 *   research/rust_ab/invisibility_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>

#define ONE INT64_C(4294967296)
#define DT INT64_C(71582788)

static int reset(BoplRl *world, BoplRlConfig *config, const char *label) {
    int status = bopl_rl_reset(world, config);
    if (status != BOPL_RL_OK) {
        fprintf(stderr, "%s reset failed (%d): %s\n", label, status,
                bopl_rl_error(world));
        return 1;
    }
    return 0;
}

static int compare(const BoplRlState *a, const BoplRlState *b, unsigned tick) {
    const BoplRlPlayer *pa = &a->players[0], *pb = &b->players[0];
    const BoplRlSlime *sa = &a->slimes[0], *sb = &b->slimes[0];
    if ((pa->flags & 4) != (pb->flags & 4) ||
        pa->invisible_time != pb->invisible_time || pa->ability_time != pb->ability_time) {
        fprintf(stderr,
                "tick %u player: flags %u/%u invisible_time %" PRId64 "/%" PRId64
                " ability_time %" PRId64 "/%" PRId64 "\n",
                tick, pa->flags, pb->flags, pa->invisible_time, pb->invisible_time,
                pa->ability_time, pb->ability_time);
        return 1;
    }
    for (unsigned slot = 0; slot < 3; slot++) {
        if (sa->abilities[slot] != sb->abilities[slot] ||
            sa->cooldowns[slot] != sb->cooldowns[slot] ||
            sa->timers[slot] != sb->timers[slot] ||
            a->telemetry[0].ability_activations[slot] !=
                b->telemetry[0].ability_activations[slot]) {
            fprintf(stderr,
                    "tick %u slot %u: id %u/%u cooldown %" PRId64 "/%" PRId64
                    " timer %" PRId64 "/%" PRId64 " activations %u/%u\n",
                    tick, slot, sa->abilities[slot], sb->abilities[slot],
                    sa->cooldowns[slot], sb->cooldowns[slot], sa->timers[slot],
                    sb->timers[slot], a->telemetry[0].ability_activations[slot],
                    b->telemetry[0].ability_activations[slot]);
            return 1;
        }
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
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/invisibility_ab.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 785921;
    config.map = 0;
    config.stocks = 5;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1200;
    for (unsigned slot = 0; slot < 3; slot++) config.player[0].abilities[slot] = 19;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    if (reset(world[0], &config, "reference") || reset(world[1], &config, "rust")) return 1;

    int first_cast = 0, chained = 0, saw_visible = 0, recast = 0;
    unsigned completed_ticks = 0;
    for (unsigned tick = 1; tick <= 1000; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (!first_cast && before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
            action[0].buttons = 2;
            first_cast = 1;
        } else if (first_cast && !chained && (before->players[0].flags & 4) &&
                   before->players[0].invisible_time + DT > 5 * ONE) {
            action[0].buttons = 4;
            chained = 1;
        } else if (chained && !(before->players[0].flags & 4)) {
            saw_visible = 1;
            if (!recast && before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
                action[0].buttons = 2;
                recast = 1;
            }
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "tick %u backend %d failed (%d): %s\n", tick,
                        backend, status, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), tick)) return 1;
        completed_ticks = tick;
        if (recast && bopl_rl_state(world[0])->players[0].flags & 4) break;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (!first_cast || !chained || !saw_visible || !recast) {
        fprintf(stderr, "incomplete scenario: first=%d chain=%d visible=%d recast=%d\n",
                first_cast, chained, saw_visible, recast);
        return 1;
    }
    printf("INVISIBILITY_GOLDEN_OK ticks=%u\n", completed_ticks);
    return 0;
}
