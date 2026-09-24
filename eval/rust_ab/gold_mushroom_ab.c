/* Gold Mushroom golden: cast its instant effect, exercise an airborne jump,
 * observe four-second expiry/five-second cooldown restoration, then recast.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/gold_mushroom_ab \
 *   research/rust_ab/gold_mushroom_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>

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
    const BoplRlSlime *sa = &a->slimes[0], *sb = &b->slimes[0];
    if (sa->y != sb->y || sa->vx != sb->vx || sa->vy != sb->vy ||
        sa->cooldowns[0] != sb->cooldowns[0] || sa->timers[0] != sb->timers[0] ||
        sa->ability_flags[0] != sb->ability_flags[0] ||
        a->telemetry[0].ability_activations[0] !=
            b->telemetry[0].ability_activations[0]) {
        fprintf(stderr,
                "tick %u: y %" PRId64 "/%" PRId64 " velocity (%" PRId64
                ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") cooldown %" PRId64
                "/%" PRId64 " timer %" PRId64 "/%" PRId64
                " flags %u/%u activations %u/%u\n",
                tick, sa->y, sb->y, sa->vx, sa->vy, sb->vx, sb->vy,
                sa->cooldowns[0], sb->cooldowns[0], sa->timers[0], sb->timers[0],
                sa->ability_flags[0], sb->ability_flags[0],
                a->telemetry[0].ability_activations[0],
                b->telemetry[0].ability_activations[0]);
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
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/gold_mushroom_ab.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 785955;
    config.map = 0;
    config.stocks = 5;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1000;
    config.player[0].abilities[0] = 34;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    if (reset(world[0], &config, "reference") || reset(world[1], &config, "rust")) return 1;

    int cast = 0, jumped = 0, restored = 0, recast = 0;
    unsigned completed_ticks = 0;
    for (unsigned tick = 1; tick <= 900; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (!cast && before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
            action[0].buttons = 2;
            cast = 1;
        } else if (cast && !jumped && before->slimes[0].cooldowns[0] > INT64_C(1000) * INT64_C(4294967296)) {
            action[0].buttons = 1;
            jumped = 1;
        } else if (jumped && before->slimes[0].cooldowns[0] == INT64_C(5) * INT64_C(4294967296)) {
            restored = 1;
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
        if (recast && bopl_rl_state(world[0])->slimes[0].cooldowns[0] >
                          INT64_C(1000) * INT64_C(4294967296))
            break;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (!cast || !jumped || !restored || !recast) {
        fprintf(stderr, "incomplete scenario: cast=%d jump=%d restored=%d recast=%d\n",
                cast, jumped, restored, recast);
        return 1;
    }
    printf("GOLD_MUSHROOM_GOLDEN_OK ticks=%u\n", completed_ticks);
    return 0;
}
