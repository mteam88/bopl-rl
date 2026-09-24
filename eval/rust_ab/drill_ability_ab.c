/* Drill differential golden: entry/loop/exit animation timing, dynamically
 * spawned combo hitboxes, air steering, burrowing, bounds death, and stock reconstruction.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/drill_ability_ab \
 *   eval/rust_ab/drill_ability_ab.c sim/abi/bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/* Q32 differs by at most two raw units in the scaled entry offset and by
 * one raw unit in unrelated inactive prefab axes. */
#define SPATIAL_TOLERANCE 8

static int64_t distance(int64_t a, int64_t b) {
    return a > b ? a - b : b - a;
}

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy && distance(a->x, b->x) <= SPATIAL_TOLERANCE && distance(a->y, b->y) <= SPATIAL_TOLERANCE &&
           a->vx == b->vx && a->vy == b->vy &&
           distance(a->right_x, b->right_x) <= SPATIAL_TOLERANCE &&
           distance(a->right_y, b->right_y) <= SPATIAL_TOLERANCE &&
           distance(a->up_x, b->up_x) <= SPATIAL_TOLERANCE &&
           distance(a->up_y, b->up_y) <= SPATIAL_TOLERANCE &&
           distance(a->radius, b->radius) <= SPATIAL_TOLERANCE &&
           a->rotation == b->rotation &&
           a->angular_velocity == b->angular_velocity &&
           a->inverse_mass == b->inverse_mass;
}

static int player_equal(const BoplRlPlayer *a, const BoplRlPlayer *b) {
    BoplRlPlayer ac = *a, bc = *b;
    if (distance(ac.x, bc.x) > SPATIAL_TOLERANCE || distance(ac.y, bc.y) > SPATIAL_TOLERANCE)
        return 0;
    ac.x = bc.x; ac.y = bc.y;
    return memcmp(&ac, &bc, sizeof ac) == 0;
}

static int slime_equal(const BoplRlSlime *a, const BoplRlSlime *b) {
    BoplRlSlime ac = *a, bc = *b;
    if (distance(ac.x, bc.x) > SPATIAL_TOLERANCE || distance(ac.y, bc.y) > SPATIAL_TOLERANCE ||
        distance(ac.vx, bc.vx) > SPATIAL_TOLERANCE || distance(ac.vy, bc.vy) > SPATIAL_TOLERANCE)
        return 0;
    ac.x = bc.x; ac.y = bc.y; ac.vx = bc.vx; ac.vy = bc.vy;
    return memcmp(&ac, &bc, sizeof ac) == 0;
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
        if (!player_equal(&a->players[i], &b->players[i]) ||
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
            for (unsigned z = 0; z < a->header.slime_count; z++) fprintf(stderr, "  slime %u owner %u/%u pos (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") vel (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") flags %u/%u\n", z, a->slimes[z].owner, b->slimes[z].owner, a->slimes[z].x,a->slimes[z].y,b->slimes[z].x,b->slimes[z].y,a->slimes[z].vx,a->slimes[z].vy,b->slimes[z].vx,b->slimes[z].vy,a->slimes[z].ability_flags[0],b->slimes[z].ability_flags[0]);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        if (!slime_equal(&a->slimes[i], &b->slimes[i])) {
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
            const unsigned char *aa=(const unsigned char *)&a->slimes[i];
            const unsigned char *bb=(const unsigned char *)&b->slimes[i];
            for (unsigned z=0; z<sizeof a->slimes[i]; z++) if (aa[z]!=bb[z]) fprintf(stderr, "  byte %u %u/%u\n",z,aa[z],bb[z]);
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


static int step_pair(BoplRl *world[2], BoplRlAction action[2], unsigned tick) {
    for (int backend = 0; backend < 2; backend++) {
        int status = bopl_rl_step(world[backend], action, 2, 1);
        if (status != BOPL_RL_OK) {
            fprintf(stderr, "tick %u backend %d failed (%d): %s\n", tick,
                    backend, status, bopl_rl_error(world[backend]));
            return 1;
        }
    }
    return compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), tick);
}

static int run_scenario(BoplRl *world[2], BoplRlConfig *config, int dive) {
    if (reset(world[0], config, "reference") || reset(world[1], config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;
    unsigned spawned = 0, lethal = 0, exit_tick = 0, death = 0, clone = 0;
    int saw_form = 0;
    for (unsigned tick = 1; tick <= 420; tick++) {
        BoplRlAction action[2] = {{0}};
        if (tick >= 181 && tick < (dive ? 340u : 199u)) {
            action[0].buttons = 2;
            action[0].angle = dive ? 192 : 64;
        } else if (!dive && tick >= 199 && tick < 240) {
            action[0].angle = 64;
        }
        if (step_pair(world, action, tick)) return 1;
        const BoplRlState *state = bopl_rl_state(world[0]);
        int form = state->slimes[0].ability_flags[0] & 8;
        if (form) saw_form = 1;
        if (saw_form && !form && !exit_tick) exit_tick = tick;
        unsigned hitboxes = 0, active_hitboxes = 0;
        for (unsigned i = 0; i < state->header.entity_count; i++) {
            if (state->entities[i].layer == 18 && state->entities[i].owner == 1) {
                hitboxes++;
                if (state->entities[i].flags & 1) active_hitboxes++;
            }
        }
        if (!spawned && hitboxes == 3) spawned = tick;
        if (!lethal && active_hitboxes == 3) lethal = tick;
        if (!death && state->players[0].deaths) death = tick;
        if (death && !clone && state->header.slime_count == 2) clone = tick;
    }
    if (!saw_form || spawned != 181 || lethal != 199 ||
        (dive ? (!death || !clone) : !exit_tick)) {
        fprintf(stderr, "%s incomplete form=%d spawn=%u lethal=%u exit=%u death=%u clone=%u\n",
                dive ? "dive" : "release", saw_form, spawned, lethal,
                exit_tick, death, clone);
        return 1;
    }
    printf("%s spawn=%u lethal=%u exit=%u death=%u clone=%u ",
           dive ? "dive" : "release", spawned, lethal, exit_tick, death, clone);
    return 0;
}

static int run_contact(BoplRl *world[2], BoplRlConfig *config) {
    if (reset(world[0], config, "reference") || reset(world[1], config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;
    unsigned kill_tick = 0, exit_tick = 0, clone_tick = 0;
    int saw_form = 0;
    for (unsigned tick = 1; tick <= 260; tick++) {
        BoplRlAction action[3] = {{0}};
        if (tick >= 181 && tick < 240) {
            action[0].buttons = 2;
            action[0].angle = 128;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 3, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "contact tick %u backend %d failed (%d): %s\n",
                        tick, backend, status, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *state = bopl_rl_state(world[0]);
        if (compare(state, bopl_rl_state(world[1]), tick)) return 1;
        int form = state->slimes[0].ability_flags[0] & 8;
        if (form) saw_form = 1;
        if (saw_form && !form && !exit_tick) exit_tick = tick;
        if (!kill_tick && state->players[1].deaths) kill_tick = tick;
        if (kill_tick && !clone_tick && state->header.slime_count == 3) clone_tick = tick;
    }
    if (!saw_form || kill_tick != 199 || exit_tick != 200 || clone_tick != 200) {
        fprintf(stderr, "contact incomplete form=%d kill=%u exit=%u clone=%u\n",
                saw_form, kill_tick, exit_tick, clone_tick);
        return 1;
    }
    printf("contact kill=%u exit=%u clone=%u ", kill_tick, exit_tick, clone_tick);
    return 0;
}

static int run_time_stop(BoplRl *world[2], BoplRlConfig *config) {
    if (reset(world[0], config, "reference") || reset(world[1], config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;
    unsigned cast_tick = 0, frozen_ticks = 0;
    int saw_drill = 0, saw_stopped_drill = 0, saw_restored = 0;
    for (unsigned tick = 1; tick <= 1600; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (!cast_tick && tick > 180 &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0])
            cast_tick = tick;
        if (cast_tick && tick < cast_tick + 600) action[0].buttons = 2;
        if (tick >= 720 && tick < 1100) {
            action[1].buttons = 2;
            action[1].angle = 64;
        }
        if (step_pair(world, action, tick)) return 1;
        const BoplRlState *state = bopl_rl_state(world[0]);
        int drilling = state->slimes[1].ability_flags[0] & 8;
        if (drilling) saw_drill = 1;
        if (state->header.player_time_scale == 0) {
            frozen_ticks++;
            if (drilling) saw_stopped_drill = 1;
        } else if (saw_stopped_drill) {
            saw_restored = 1;
        }
    }
    if (!cast_tick || !saw_drill || !saw_stopped_drill || !saw_restored ||
        frozen_ticks != 690) {
        fprintf(stderr,
                "time-stop incomplete cast=%u drill=%d stopped=%d restored=%d frozen=%u\n",
                cast_tick, saw_drill, saw_stopped_drill, saw_restored,
                frozen_ticks);
        return 1;
    }
    printf("time-stop frozen=%u ", frozen_ticks);
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
                                      "/tmp/drill_ability_golden.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7;
    config.map = 0;
    config.stocks = 100;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1000;
    for (unsigned player = 0; player < 2; player++) {
        config.player[player].keyboard = 1;
        config.player[player].abilities[0] = 26;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    if (run_scenario(world, &config, 1) || run_scenario(world, &config, 0))
        return 1;
    bopl_rl_default_config(&config, 3);
    config.seed = 7;
    config.map = 0;
    config.stocks = 100;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1000;
    for (unsigned player = 0; player < 3; player++) {
        config.player[player].team = player == 2 ? 1 : 0;
        config.player[player].keyboard = 1;
        config.player[player].abilities[0] = 26;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    if (run_contact(world, &config)) return 1;
    bopl_rl_default_config(&config, 2);
    config.seed = 123;
    config.map = 0;
    config.stocks = 5;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 2000;
    config.player[0].abilities[0] = 12;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 26;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    if (run_time_stop(world, &config)) return 1;
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    puts("DRILL_ABILITY_GOLDEN_OK ticks=2700");
    return 0;
}
