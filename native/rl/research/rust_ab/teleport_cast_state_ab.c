/* Teleport cast-tick slime state differential.
 *
 * SlimeController.OldUpdate enters an ability and returns before its own state
 * switch:
 *
 *   if (player.AbilityButtonIsDown(k) && isAbilityReady(k) &&
 *       isAbilityCastable(k) && !suppressInput) { EnterAbility(k, flag); return; }
 *
 * so the tick a cast starts never runs the run/stop/turn machine. A form
 * ability keeps the machine frozen afterwards too, but InstantAbility
 * (Teleport, and every other instant) enters and leaves inside EnterAbility
 * without setting `isInAbility`, so the freeze is exactly one tick and is
 * visible only on that tick.
 *
 * The port used to read the cast tick's stick direction as a locomotion input,
 * which turned a standing Teleport cast with a horizontal aim into
 * run -> stop -> idle over the following seven ticks. Aim and movement share
 * one stick, so a Teleport policy trips this on essentially every cast.
 * teleport_ability_ab.c casts with a neutral aim and cannot see it.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/teleport_cast_state_ab \
 *   research/rust_ab/teleport_cast_state_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#define STATE_IDLE 0u
#define MARKER_LAYER 27u

static int64_t distance(int64_t a, int64_t b) {
    return a > b ? a - b : b - a;
}

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy && a->x == b->x && a->y == b->y &&
           a->vx == b->vx && a->vy == b->vy &&
           distance(a->right_x, b->right_x) <= 1 &&
           distance(a->right_y, b->right_y) <= 1 &&
           distance(a->up_x, b->up_x) <= 1 &&
           distance(a->up_y, b->up_y) <= 1 &&
           distance(a->radius, b->radius) <= 1 &&
           a->rotation == b->rotation &&
           a->angular_velocity == b->angular_velocity &&
           a->inverse_mass == b->inverse_mass;
}

static int compare(const BoplRlState *a, const BoplRlState *b, unsigned tick,
                   unsigned aim) {
    if (memcmp(&a->header, &b->header, sizeof a->header) != 0) {
        fprintf(stderr, "aim %u tick %u header differs: entities %u/%u\n", aim,
                tick, a->header.entity_count, b->header.entity_count);
        return 1;
    }
    for (unsigned i = 0; i < a->header.player_count; i++)
        if (memcmp(&a->players[i], &b->players[i], sizeof a->players[i]) != 0 ||
            memcmp(&a->telemetry[i], &b->telemetry[i],
                   sizeof a->telemetry[i]) != 0) {
            fprintf(stderr,
                    "aim %u tick %u player %u differs: position (%" PRId64
                    ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                    aim, tick, i, a->players[i].x, a->players[i].y,
                    b->players[i].x, b->players[i].y);
            return 1;
        }
    for (unsigned i = 0; i < a->header.slime_count; i++)
        if (memcmp(&a->slimes[i], &b->slimes[i], sizeof a->slimes[i]) != 0) {
            fprintf(stderr,
                    "aim %u tick %u slime %u differs: state %u/%u position "
                    "(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") velocity (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ")\n",
                    aim, tick, i, a->slimes[i].state, b->slimes[i].state,
                    a->slimes[i].x, a->slimes[i].y, b->slimes[i].x,
                    b->slimes[i].y, a->slimes[i].vx, a->slimes[i].vy,
                    b->slimes[i].vx, b->slimes[i].vy);
            return 1;
        }
    for (unsigned i = 0; i < a->header.entity_count; i++)
        if (!entity_equal(&a->entities[i], &b->entities[i])) {
            fprintf(stderr,
                    "aim %u tick %u entity %u differs: id %u/%u layer %u/%u "
                    "flags %u/%u position (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ")\n",
                    aim, tick, i, a->entities[i].id, b->entities[i].id,
                    a->entities[i].layer, b->entities[i].layer,
                    a->entities[i].flags, b->entities[i].flags,
                    a->entities[i].x, a->entities[i].y, b->entities[i].x,
                    b->entities[i].y);
            return 1;
        }
    return 0;
}

static int run_case(BoplRl *world[2], unsigned aim, unsigned *casts,
                    unsigned *marker_ticks) {
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 123;
    config.map = 0;
    config.stocks = 5;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 2000;
    config.player[0].abilities[0] = 25;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    for (int backend = 0; backend < 2; backend++) {
        int status = bopl_rl_reset(world[backend], &config);
        if (status != BOPL_RL_OK) {
            fprintf(stderr, "aim %u reset backend %d failed (%d): %s\n", aim,
                    backend, status, bopl_rl_error(world[backend]));
            return 1;
        }
    }
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0, aim))
        return 1;

    unsigned first = 0, second = 0;
    *casts = 0;
    *marker_ticks = 0;
    for (unsigned tick = 1; tick <= 250; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        unsigned state_before = before->slimes[0].state;
        BoplRlAction action[2] = {{0}};
        int casting = 0;
        if (!first && before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
            first = tick;
            casting = 1;
        } else if (first && !second && tick == first + 40) {
            second = tick;
            casting = 1;
        }
        if (casting) {
            action[0].buttons = 2;
            action[0].angle = (unsigned char)aim; /* aim doubles as the stick */
            (*casts)++;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "aim %u tick %u backend %d failed (%d): %s\n",
                        aim, tick, backend, status, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), tick, aim)) return 1;
        /* Only the marker-placing cast leaves the body alone; the second cast
         * swaps the player to the marker and UnGrounds it later in the frame,
         * which legitimately moves the state to fall. */
        if (casting && tick == first && reference->slimes[0].state != state_before) {
            fprintf(stderr,
                    "aim %u tick %u: the reference changed slime state %u -> %u "
                    "across a cast tick; OldUpdate should have returned first\n",
                    aim, tick, state_before, reference->slimes[0].state);
            return 1;
        }
        for (unsigned i = 0; i < reference->header.entity_count; i++)
            if (reference->entities[i].layer == MARKER_LAYER &&
                (reference->entities[i].flags & 1))
                (*marker_ticks)++;
    }
    if (*casts != 2 || !*marker_ticks) {
        fprintf(stderr, "aim %u incomplete: casts=%u marker_ticks=%u\n", aim,
                *casts, *marker_ticks);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n",
                argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] =
            bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                         "/tmp/teleport_cast_state_ab.log", 120000, error,
                         sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    /* 1 is +x and 128 is -x: the two aims whose ground alignment clears
     * GROUNDALIGNMENTDEADZONE and would drive the run/stop machine. */
    static const unsigned aims[] = {1, 128, 64, 192};
    unsigned total_markers = 0;
    for (unsigned i = 0; i < sizeof aims / sizeof *aims; i++) {
        unsigned casts = 0, marker_ticks = 0;
        if (run_case(world, aims[i], &casts, &marker_ticks)) return 1;
        total_markers += marker_ticks;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    printf("TELEPORT_CAST_STATE_GOLDEN_OK aims=4 marker_ticks=%u\n",
           total_markers);
    return 0;
}
