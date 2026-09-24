/* Tesla Coil plant-exit state differential.
 *
 * PlantObjectOnPlatform.ExitAbility decides between SlimeController.EnterStop
 * and EnterIdle from the ground-aligned movement input held on the tick the
 * cast ends:
 *
 *   Vec2 inputVector = Vec2.NormalizedSafe(player.InputVector());
 *   Fix value = playerPhysics.InputGroundAlignment(inputVector);
 *   if (Fix.Abs(value) >= Constants.GROUNDALIGNMENTDEADZONE &&
 *       (float)Fix.Sign2(value) * transform.localScale.x >= 0f)
 *       exitInfo.enterTurn = true;
 *
 * OnEnterAbility set localScale.x to -1 while facing right, so enterTurn (and
 * therefore EnterStop) is reached when the held direction *opposes* the facing,
 * and the state installed by the recover callback survives the exit tick
 * because PlantObjectOnPlatform runs after SlimeController's own pass.
 *
 * tesla_ability_ab.c casts with no movement input, so the deadzone rejects
 * every exit and neither branch is exercised. This probe holds a ground
 * direction through the release in both orientations and requires the
 * reference to show both EnterStop and EnterIdle.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/tesla_exit_state_ab \
 *   eval/rust_ab/tesla_exit_state_ab.c sim/abi/bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#define STATE_IDLE 0u
#define STATE_RUN 1u
#define STATE_STOP 7u

static int64_t distance(int64_t a, int64_t b) {
    return a > b ? a - b : b - a;
}

/* Collider basis vectors carry the same single-raw-unit slack the other
 * goldens allow; everything discrete stays exact. */
static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy && distance(a->x, b->x) <= 16 &&
           distance(a->y, b->y) <= 16 && a->vx == b->vx && a->vy == b->vy &&
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
                   unsigned angle) {
    if (memcmp(&a->header, &b->header, sizeof a->header) != 0) {
        fprintf(stderr,
                "angle %u tick %u header differs: entities %u/%u slimes %u/%u\n",
                angle, tick, a->header.entity_count, b->header.entity_count,
                a->header.slime_count, b->header.slime_count);
        return 1;
    }
    for (unsigned i = 0; i < a->header.player_count; i++) {
        if (memcmp(&a->players[i], &b->players[i], sizeof a->players[i]) != 0 ||
            memcmp(&a->telemetry[i], &b->telemetry[i],
                   sizeof a->telemetry[i]) != 0) {
            fprintf(stderr,
                    "angle %u tick %u player %u differs: position "
                    "(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") flags %u/%u\n",
                    angle, tick, i, a->players[i].x, a->players[i].y,
                    b->players[i].x, b->players[i].y, a->players[i].flags,
                    b->players[i].flags);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        if (memcmp(&a->slimes[i], &b->slimes[i], sizeof a->slimes[i]) != 0) {
            fprintf(stderr,
                    "angle %u tick %u slime %u differs: state %u/%u position "
                    "(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") velocity (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") ability flags %u/%u\n",
                    angle, tick, i, a->slimes[i].state, b->slimes[i].state,
                    a->slimes[i].x, a->slimes[i].y, b->slimes[i].x,
                    b->slimes[i].y, a->slimes[i].vx, a->slimes[i].vy,
                    b->slimes[i].vx, b->slimes[i].vy,
                    a->slimes[i].ability_flags[0], b->slimes[i].ability_flags[0]);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.entity_count; i++) {
        if (!entity_equal(&a->entities[i], &b->entities[i])) {
            fprintf(stderr,
                    "angle %u tick %u entity %u differs: id %u/%u layer %u/%u "
                    "flags %u/%u position (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ")\n",
                    angle, tick, i, a->entities[i].id, b->entities[i].id,
                    a->entities[i].layer, b->entities[i].layer,
                    a->entities[i].flags, b->entities[i].flags,
                    a->entities[i].x, a->entities[i].y, b->entities[i].x,
                    b->entities[i].y);
            return 1;
        }
    }
    return 0;
}

/* Runs one orientation. Reports the reference's state on the exit tick. */
static int run_case(BoplRl *world[2], unsigned angle, unsigned *exit_state,
                    unsigned *exit_tick) {
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
        int status = bopl_rl_reset(world[backend], &config);
        if (status != BOPL_RL_OK) {
            fprintf(stderr, "angle %u reset backend %d failed (%d): %s\n", angle,
                    backend, status, bopl_rl_error(world[backend]));
            return 1;
        }
    }
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0, angle))
        return 1;

    unsigned cast_tick = 0;
    int in_form = 0;
    *exit_state = 0xffffffffu;
    *exit_tick = 0;
    for (unsigned tick = 1; tick <= 400; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (!cast_tick &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0])
            cast_tick = tick;
        if (cast_tick) {
            /* The direction is held continuously, including the release tick
             * the plant-exit decision reads. */
            action[0].angle = (unsigned char)angle;
            if (tick < cast_tick + 30) action[0].buttons = 2;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "angle %u tick %u backend %d failed (%d): %s\n",
                        angle, tick, backend, status,
                        bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), tick, angle)) return 1;
        unsigned flags = reference->slimes[0].ability_flags[0];
        if (flags & 8) in_form = 1;
        if (in_form && !(flags & 8) && !*exit_tick) {
            *exit_tick = tick;
            *exit_state = reference->slimes[0].state;
        }
    }
    if (!cast_tick || !in_form || !*exit_tick) {
        fprintf(stderr, "angle %u incomplete: cast=%u form=%d exit=%u\n", angle,
                cast_tick, in_form, *exit_tick);
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
                         "/tmp/tesla_exit_state_ab.log", 120000, error,
                         sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    /* angle 1 is +x, angle 128 is -x; player 0 spawns facing right. */
    unsigned forward_state = 0, forward_tick = 0;
    unsigned reverse_state = 0, reverse_tick = 0;
    if (run_case(world, 1, &forward_state, &forward_tick)) return 1;
    if (run_case(world, 128, &reverse_state, &reverse_tick)) return 1;
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);

    if (forward_state != STATE_IDLE) {
        fprintf(stderr,
                "reference exit with the facing direction held was state %u, "
                "expected idle (%u)\n",
                forward_state, STATE_IDLE);
        return 1;
    }
    if (reverse_state != STATE_STOP) {
        fprintf(stderr,
                "reference exit against the facing direction was state %u, "
                "expected stop (%u)\n",
                reverse_state, STATE_STOP);
        return 1;
    }
    printf("TESLA_EXIT_STATE_GOLDEN_OK forward=idle@%u reverse=stop@%u\n",
           forward_tick, reverse_tick);
    return 0;
}
