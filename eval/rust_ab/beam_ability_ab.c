/* Beam differential golden: ground and air entry, firing and collision latency,
 * cancellation, maximum duration, retained air-orb recast, and beam crossings.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/beam_ability_ab \
 *   research/rust_ab/beam_ability_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int64_t distance(int64_t a, int64_t b) {
    return a > b ? a - b : b - a;
}

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy && distance(a->x, b->x) <= 1 && distance(a->y, b->y) <= 1 &&
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

static int step_pair(BoplRl *world[2], BoplRlAction action[2],
                     unsigned tick, const char *scenario) {
    for (int backend = 0; backend < 2; backend++) {
        int status = bopl_rl_step(world[backend], action, 2, 1);
        if (status != BOPL_RL_OK) {
            fprintf(stderr, "%s step %u backend %d failed: %s\n", scenario,
                    tick, backend, bopl_rl_error(world[backend]));
            return 1;
        }
    }
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), tick)) {
        fprintf(stderr, "%s scenario\n", scenario);
        return 1;
    }
    return 0;
}

static unsigned count_kind(const BoplRlState *state, unsigned kind) {
    unsigned count = 0;
    for (unsigned i = 0; i < state->header.entity_count; i++)
        count += state->entities[i].kind == kind;
    return count;
}

static int configure(BoplRl *world[2], BoplRlConfig *config,
                     const char *scenario) {
    if (reset(world[0], config, scenario) || reset(world[1], config, scenario))
        return 1;
    return compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0);
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1],
                                      "/tmp/gate_bootstrap.rep",
                                      "/tmp/beam_ability_ab.log", 180000,
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
    config.stocks = 10;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 3000;
    for (unsigned player = 0; player < 2; player++) {
        config.player[player].keyboard = 1;
        config.player[player].abilities[0] = 28;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }

    /* Horizontal ground cast: the orb charges, BeamObject appears one pass
     * before its callback kills player 2, release destroys the orb, and the
     * form returns to the ordinary slime. */
    if (configure(world, &config, "ground lifecycle")) return 1;
    unsigned saw_orb = 0, saw_trace = 0, saw_kill = 0, saw_cleanup = 0;
    for (unsigned tick = 1; tick <= 700; tick++) {
        BoplRlAction action[2] = {{0}};
        if (tick >= 181 && tick < 500) {
            action[0].buttons = 2;
            action[0].angle = 1;
        }
        if (step_pair(world, action, tick, "ground lifecycle")) return 1;
        const BoplRlState *state = bopl_rl_state(world[0]);
        for (unsigned i = 0; i < state->header.entity_count; i++)
            saw_orb |= state->entities[i].kind == 0 &&
                       state->entities[i].layer == 18;
        saw_trace |= count_kind(state, 4) != 0;
        saw_kill |= state->telemetry[1].deaths_opponent != 0;
        saw_cleanup |= saw_trace && tick > 520 && count_kind(state, 4) == 0;
    }
    if (!saw_orb || !saw_trace || !saw_kill || !saw_cleanup) {
        fprintf(stderr, "ground lifecycle missed an expected phase\n");
        return 1;
    }

    /* Releasing during ground charge must never create a BeamObject and must
     * hand the moving platform-local pose back exactly on the exit tick. */
    if (configure(world, &config, "charge cancellation")) return 1;
    for (unsigned tick = 1; tick <= 360; tick++) {
        BoplRlAction action[2] = {{0}};
        if (tick == 170) action[0].buttons = 1;
        if (tick >= 181 && tick < 300) {
            action[0].buttons |= 2;
            action[0].angle = 65;
        }
        if (step_pair(world, action, tick, "charge cancellation")) return 1;
        if (count_kind(bopl_rl_state(world[0]), 4) != 0) {
            fprintf(stderr, "charge cancellation fired a beam\n");
            return 1;
        }
    }

    /* Jump immediately before cooldown readiness so the cast enters the air
     * form and exercises its three frame-driven orb poses and short exit. */
    if (configure(world, &config, "air lifecycle")) return 1;
    saw_trace = 0;
    for (unsigned tick = 1; tick <= 500; tick++) {
        BoplRlAction action[2] = {{0}};
        if (tick >= 181 && tick < 420) {
            action[0].buttons = 2;
            action[0].angle = 65;
        }
        if (tick == 240) action[0].buttons |= 1;
        if (step_pair(world, action, tick, "air lifecycle")) return 1;
        saw_trace |= count_kind(bopl_rl_state(world[0]), 4) != 0;
    }
    if (!saw_trace) {
        fprintf(stderr, "air lifecycle never fired\n");
        return 1;
    }

    /* Air exit is shorter than whiteExit, so the equipped Beam prefab retains
     * an inactive orb. A second air cast reuses its id and hierarchy. */
    if (configure(world, &config, "retained orb recast")) return 1;
    for (unsigned tick = 1; tick <= 1000; tick++) {
        BoplRlAction action[2] = {{0}};
        if ((tick >= 181 && tick < 420) || (tick >= 730 && tick < 900)) {
            action[0].buttons = 2;
            action[0].angle = 65;
        }
        if (tick == 240 || tick == 750) action[0].buttons |= 1;
        if (step_pair(world, action, tick, "retained orb recast")) return 1;
    }
    if (bopl_rl_state(world[0])->telemetry[0].ability_activations[0] != 2) {
        fprintf(stderr, "retained orb scenario cast %u times\n", bopl_rl_state(world[0])->telemetry[0].ability_activations[0]);
        return 1;
    }

    /* Holding through maxTime exercises exitScaleAnim and the automatic stop
     * path, including the final FirePos update before whiteExit. */
    if (configure(world, &config, "maximum duration")) return 1;
    saw_trace = saw_cleanup = 0;
    for (unsigned tick = 1; tick <= 750; tick++) {
        BoplRlAction action[2] = {{0}};
        if (tick >= 181) {
            action[0].buttons = 2;
            action[0].angle = 1;
        }
        if (step_pair(world, action, tick, "maximum duration")) return 1;
        unsigned traces = count_kind(bopl_rl_state(world[0]), 4);
        saw_trace |= traces != 0;
        saw_cleanup |= saw_trace && traces == 0;
    }
    if (!saw_trace || !saw_cleanup) {
        fprintf(stderr, "maximum-duration beam did not stop itself\n");
        return 1;
    }

    /* Two upward ground beams meet above the stage. The persistent crossing
     * graph emits two parent traces plus the shared child trace. */
    if (configure(world, &config, "crossing beams")) return 1;
    unsigned saw_crossing = 0;
    for (unsigned tick = 1; tick <= 500; tick++) {
        BoplRlAction action[2] = {{0}};
        if (tick >= 181 && tick < 430) {
            action[0].buttons = 2;
            action[0].angle = 33;
            action[1].buttons = 2;
            action[1].angle = 97;
        }
        if (step_pair(world, action, tick, "crossing beams")) return 1;
        saw_crossing |= count_kind(bopl_rl_state(world[0]), 4) >= 3;
    }
    if (!saw_crossing) {
        fprintf(stderr, "crossing scenario produced no child trace\n");
        return 1;
    }

    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    puts("BEAM_ABILITY_AB_OK");
    return 0;
}
