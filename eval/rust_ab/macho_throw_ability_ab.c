/* Throw differential golden: platform shrinking, held and quick boulders,
 * launch/collision physics, form exit, death, and stock reconstruction.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/macho_throw_ability_ab \
 *   eval/rust_ab/macho_throw_ability_ab.c sim/abi/bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/* Repeated collider reciprocal/multiply round trips while the ground shrinks
 * differ by at most four raw Q32 units. Positions, velocities, lifecycle,
 * contact response, and the Throw form collider are otherwise exact. */
#define SPATIAL_TOLERANCE 4

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

static const BoplRlEntity *find_boulder(const BoplRlState *state) {
    for (unsigned i = 0; i < state->header.entity_count; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->kind == 2 && entity->layer == 11 &&
            entity->hierarchy >= 100000000u)
            return entity;
    }
    return NULL;
}

static const BoplRlEntity *find_entity(const BoplRlState *state, unsigned id) {
    for (unsigned i = 0; i < state->header.entity_count; i++)
        if (state->entities[i].id == id) return &state->entities[i];
    return NULL;
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

static int run_scenario(BoplRl *world[2], BoplRlConfig *config, int quick) {
    if (reset(world[0], config, "reference") || reset(world[1], config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    const BoplRlEntity *initial_ground = find_entity(bopl_rl_state(world[0]), 11);
    if (!initial_ground) return 1;
    int64_t initial_radius = initial_ground->radius;
    unsigned spawn_tick = 0, launch_tick = 0, shrink_tick = 0;
    unsigned death_tick = 0, clone_tick = 0;
    int saw_form = 0, saw_exit = 0;
    for (unsigned tick = 1; tick <= 500; tick++) {
        BoplRlAction action[2] = {{0}};
        if ((quick && tick == 181) || (!quick && tick >= 181 && tick < 270)) {
            action[0].buttons = 2;
            action[0].angle = 64;
        }
        if (step_pair(world, action, tick)) return 1;
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (reference->slimes[0].ability_flags[0] & 8) saw_form = 1;
        if (saw_form && !(reference->slimes[0].ability_flags[0] & 8)) saw_exit = 1;
        const BoplRlEntity *boulder = find_boulder(reference);
        if (boulder && !spawn_tick) spawn_tick = tick;
        if (boulder && !launch_tick && (boulder->vx || boulder->vy))
            launch_tick = tick;
        const BoplRlEntity *ground = find_entity(reference, 11);
        if (ground && !shrink_tick && ground->radius != initial_radius)
            shrink_tick = tick;
        if (!death_tick && reference->header.slime_count == 1) death_tick = tick;
        if (death_tick && !clone_tick && reference->header.slime_count == 2)
            clone_tick = tick;
    }
    if (!saw_form || !saw_exit || !spawn_tick || !launch_tick || !shrink_tick ||
        (!quick && (!death_tick || !clone_tick))) {
        fprintf(stderr,
                "%s incomplete form=%d exit=%d spawn=%u launch=%u shrink=%u "
                "death=%u clone=%u\n",
                quick ? "quick" : "held", saw_form, saw_exit, spawn_tick,
                launch_tick, shrink_tick, death_tick, clone_tick);
        return 1;
    }
    if (quick && spawn_tick != launch_tick) {
        fprintf(stderr, "quick boulder was held: spawn=%u launch=%u\n",
                spawn_tick, launch_tick);
        return 1;
    }
    printf("%s spawn=%u launch=%u shrink=%u death=%u clone=%u ",
           quick ? "quick" : "held", spawn_tick, launch_tick, shrink_tick,
           death_tick, clone_tick);
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
                                      "/tmp/macho_throw_ability_golden.log", 120000,
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
        config.player[player].abilities[0] = 21;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    if (run_scenario(world, &config, 1) || run_scenario(world, &config, 0))
        return 1;
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    printf("MACHO_THROW_ABILITY_GOLDEN_OK ticks=2x500\n");
    return 0;
}
