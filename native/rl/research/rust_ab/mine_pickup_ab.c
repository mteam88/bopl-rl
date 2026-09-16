/* Mine pickup golden: pursue the first authored spawn while casting Mine,
 * retain the pickup across the form, then validate the released item,
 * post-exit collection, third-slot replacement, deferred registration and the
 * source's same-pass held-button cast of the acquired Revival.
 *
 * BOPL_RL_ABILITY_SPAWNS=1 is required for both backends.
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int64_t distance(int64_t a, int64_t b) {
    return a > b ? a - b : b - a;
}

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    int64_t vector_tolerance = a->layer == 13 ? INT64_C(100000) : 1;
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy &&
           distance(a->x, b->x) <= vector_tolerance &&
           distance(a->y, b->y) <= vector_tolerance &&
           distance(a->vx, b->vx) <= vector_tolerance &&
           distance(a->vy, b->vy) <= vector_tolerance &&
           distance(a->right_x, b->right_x) <= 1 &&
           distance(a->right_y, b->right_y) <= 1 &&
           distance(a->up_x, b->up_x) <= 1 &&
           distance(a->up_y, b->up_y) <= 1 &&
           distance(a->radius, b->radius) <= 1 &&
           a->rotation == b->rotation &&
           a->angular_velocity == b->angular_velocity &&
           a->inverse_mass == b->inverse_mass;
}

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
    if (memcmp(&a->header, &b->header, sizeof a->header) != 0) {
        fprintf(stderr,
                "tick %u header differs: ticks %u/%u entities %u/%u slimes %u/%u\n",
                tick, a->header.tick, b->header.tick, a->header.entity_count,
                b->header.entity_count, a->header.slime_count,
                b->header.slime_count);
        return 1;
    }
    for (unsigned i = 0; i < a->header.player_count; i++) {
        if (memcmp(&a->players[i], &b->players[i], sizeof a->players[i]) != 0 ||
            memcmp(&a->telemetry[i], &b->telemetry[i],
                   sizeof a->telemetry[i]) != 0) {
            fprintf(stderr,
                    "tick %u player %u differs: flags %u/%u alive %u/%u "
                    "position (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") scale %" PRId64 "/%" PRId64
                    " activations %u/%u\n",
                    tick, i, a->players[i].flags, b->players[i].flags,
                    a->players[i].alive_bodies, b->players[i].alive_bodies,
                    a->players[i].x, a->players[i].y, b->players[i].x,
                    b->players[i].y, a->players[i].scale,
                    b->players[i].scale,
                    a->telemetry[i].ability_activations[0],
                    b->telemetry[i].ability_activations[0]);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        if (memcmp(&a->slimes[i], &b->slimes[i], sizeof a->slimes[i]) != 0) {
            fprintf(stderr,
                    "tick %u slime %u differs: position (%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") ability flags %u/%u\n",
                    tick, i, a->slimes[i].x, a->slimes[i].y, b->slimes[i].x,
                    b->slimes[i].y, a->slimes[i].ability_flags[0],
                    b->slimes[i].ability_flags[0]);
            fprintf(stderr,
                    "  velocity (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") state %u/%u\n",
                    a->slimes[i].vx, a->slimes[i].vy, b->slimes[i].vx,
                    b->slimes[i].vy, a->slimes[i].state, b->slimes[i].state);
            for (unsigned e = 0; e < a->header.entity_count; e++) {
                if (a->entities[e].kind == 2)
                    fprintf(stderr,
                            "  platform %u p (%" PRId64 ",%" PRId64
                            ")/(%" PRId64 ",%" PRId64 ") v (%" PRId64
                            ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                            e, a->entities[e].x, a->entities[e].y,
                            b->entities[e].x, b->entities[e].y,
                            a->entities[e].vx, a->entities[e].vy,
                            b->entities[e].vx, b->entities[e].vy);
            }
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.entity_count; i++) {
        if (!entity_equal(&a->entities[i], &b->entities[i])) {
            fprintf(stderr,
                    "tick %u entity %u differs: id %u/%u kind %u/%u flags %u/%u "
                    "position (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") velocity (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ")\n",
                    tick, i, a->entities[i].id, b->entities[i].id,
                    a->entities[i].kind, b->entities[i].kind,
                    a->entities[i].flags, b->entities[i].flags,
                    a->entities[i].x, a->entities[i].y, b->entities[i].x,
                    b->entities[i].y, a->entities[i].vx, a->entities[i].vy,
                    b->entities[i].vx, b->entities[i].vy);
            fprintf(stderr,
                    "  hierarchy %u/%u layer %u/%u owner %u/%u right "
                    "(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") up (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") radius %" PRId64 "/%" PRId64 " rotation %" PRId64
                    "/%" PRId64 " angular %" PRId64 "/%" PRId64
                    " inverse_mass %" PRId64 "/%" PRId64 "\n",
                    a->entities[i].hierarchy, b->entities[i].hierarchy,
                    a->entities[i].layer, b->entities[i].layer,
                    a->entities[i].owner, b->entities[i].owner,
                    a->entities[i].right_x, a->entities[i].right_y,
                    b->entities[i].right_x, b->entities[i].right_y,
                    a->entities[i].up_x, a->entities[i].up_y,
                    b->entities[i].up_x, b->entities[i].up_y,
                    a->entities[i].radius, b->entities[i].radius,
                    a->entities[i].rotation, b->entities[i].rotation,
                    a->entities[i].angular_velocity,
                    b->entities[i].angular_velocity,
                    a->entities[i].inverse_mass, b->entities[i].inverse_mass);
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
        world[backend] = bopl_rl_open(argv[backend + 1],
            "/proc/self/cwd/native/fixtures/movement.rep",
            "/proc/self/cwd/native/target/mine_pickup_ab.log",
            120000, error, sizeof error);
        if (!world[backend]) { fprintf(stderr, "open failed: %s\n", error); return 1; }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7; config.map = 0; config.stocks = 100;
    config.warmup_ticks = 240; config.max_episode_ticks = 5000;
    for (int player = 0; player < 2; player++) {
        config.player[player].keyboard = 1;
        config.player[player].abilities[0] = 14;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 24;
    }
    if (reset(world[0], &config, "reference") || reset(world[1], &config, "rust")) return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    BoplRlAction action[2] = {{0}};
    int jump_cooldown = 0, saw_pickup = 0, collected = 0, registered = 0;
    int saw_mine = 0, mine_exited = 0;
    int saw_released_mine = 0, cast = 0, saw_acquired_active = 0;
    int saw_marker = 0, saw_acquired_exit = 0;
    unsigned collect_tick = 0, register_tick = 0, cast_tick = 0, finish_tick = 0;
    for (unsigned episode_tick = 0; episode_tick < 2700; episode_tick++) {
        action[0].buttons = 0;
        const BoplRlState *before = bopl_rl_state(world[0]);
        const BoplRlEntity *pickup = NULL;
        for (unsigned i = 0; i < before->header.entity_count; i++)
            if (before->entities[i].layer == 13 && (before->entities[i].flags & 1)) {
                pickup = &before->entities[i]; break;
            }
        if (episode_tick >= 2160 && pickup) {
            int64_t dx = pickup->x - before->players[0].x;
            if (dx > INT64_C(2000000000)) action[0].buttons |= 512;
            else if (dx < -INT64_C(2000000000)) action[0].buttons |= 128;
            if (jump_cooldown > 0) jump_cooldown--;
            if (distance(pickup->x, before->players[0].x) < INT64_C(50000000000) &&
                before->players[0].y + INT64_C(5000000000) < pickup->y &&
                jump_cooldown == 0) {
                action[0].buttons |= 1; jump_cooldown = 45;
            }
        }
        if (episode_tick >= 2200 && episode_tick < 2320) {
            action[0].buttons |= 8;
            action[0].angle = 64;
        }
        if (registered && !cast) {
            action[0].buttons = 8;
            cast = 1;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "episode tick %u backend %d failed: %s\n",
                        episode_tick + 1, backend, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *state[2] = {bopl_rl_state(world[0]), bopl_rl_state(world[1])};
        if (compare(state[0], state[1], episode_tick + 1)) return 1;
        const BoplRlEntity *live = NULL, *destroyed = NULL, *revival = NULL;
        for (unsigned i = 0; i < state[0]->header.entity_count; i++) {
            const BoplRlEntity *entity = &state[0]->entities[i];
            if (entity->layer == 13 && (entity->flags & 1)) live = entity;
            if (entity->id == 16 && entity->flags == 4) destroyed = entity;
            if (entity->id == 18 && entity->layer == 8) revival = entity;
            if (entity->layer == 10 && (entity->flags & 1))
                saw_released_mine = 1;
            if (entity->layer == 27 && entity->owner == 1 &&
                (entity->flags & 1)) saw_marker = 1;
        }
        if (live) saw_pickup = 1;
        const BoplRlSlime *player = &state[0]->slimes[0];
        int mine_active = (player->ability_flags[2] & 8) != 0;
        if (mine_active && !collected) {
            saw_mine = 1;
            if (!live) {
                fprintf(stderr, "pickup collected through active Mine form at tick %u\n",
                        state[0]->header.tick);
                return 1;
            }
        } else if (saw_mine && !collected) {
            mine_exited = 1;
        }
        if (!collected && saw_pickup && !live) {
            const BoplRlSlime *slime = &state[0]->slimes[0];
            if (!destroyed || slime->abilities[2] != 15 ||
                slime->timers[2] != INT64_C(429496729600000) ||
                slime->ability_flags[2] != 7) {
                fprintf(stderr, "collection payload mismatch at tick %u\n", state[0]->header.tick);
                return 1;
            }
            collected = 1; collect_tick = state[0]->header.tick;
        } else if (collected && revival && !registered) {
            /* The replacement registers one pass later while the original
             * slot-three hold is still down, so SlimeController immediately
             * enters the acquired Revival in that same pass. */
            if (revival->flags != 11 || revival->hierarchy != 100000075 ||
                revival->owner != 1 ||
                !(state[0]->slimes[0].ability_flags[2] & 8)) {
                fprintf(stderr, "deferred Revival activation mismatch at tick %u\n",
                        state[0]->header.tick);
                return 1;
            }
            registered = 1; register_tick = state[0]->header.tick;
            cast = 1; cast_tick = state[0]->header.tick;
            saw_acquired_active = 1;
        }
        if (cast && !cast_tick && (state[0]->slimes[0].ability_flags[2] & 8)) {
            cast_tick = state[0]->header.tick;
            saw_acquired_active = 1;
        }
        if (saw_acquired_active && !(state[0]->slimes[0].ability_flags[2] & 8))
            saw_acquired_exit = 1;
        if (saw_marker && saw_acquired_exit) {
            finish_tick = state[0]->header.tick;
            break;
        }
    }
    if (!saw_mine || !mine_exited || !saw_released_mine || !collected ||
        !registered || !cast ||
        !saw_acquired_active || !saw_marker || !saw_acquired_exit ||
        !finish_tick || register_tick != collect_tick + 1) {
        fprintf(stderr, "Mine pickup incomplete mine=%d exit=%d item=%d "
                "collect=%u register=%u cast=%d active=%d marker=%d "
                "acquired_exit=%d\n", saw_mine, mine_exited,
                saw_released_mine, collect_tick, register_tick, cast,
                saw_acquired_active,
                saw_marker, saw_acquired_exit);
        return 1;
    }
    printf("MINE_PICKUP_GOLDEN_OK ticks=%u collect=%u register=%u cast=%u "
           "ability=15\n", finish_tick, collect_tick, register_tick, cast_tick);
    bopl_rl_close(world[0]); bopl_rl_close(world[1]);
    return 0;
}
