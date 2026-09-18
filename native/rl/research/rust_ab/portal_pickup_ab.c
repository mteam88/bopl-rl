/* AbilitySpawner pickup traversal through an older linked Portal Gun pair.
 * The golden covers updater hierarchy ordering, capture, mapped momentum,
 * exit and continued hover motion. */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int64_t distance(int64_t a, int64_t b) {
    return a > b ? a - b : b - a;
}

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy &&
           (a->layer == 13 ? distance(a->x, b->x) <= 16384 : a->x == b->x) &&
           (a->layer == 13 ? distance(a->y, b->y) <= 16384 : a->y == b->y) &&
           (a->layer == 13 ? distance(a->vx, b->vx) <= 16384 : a->vx == b->vx) &&
           (a->layer == 13 ? distance(a->vy, b->vy) <= 16384 : a->vy == b->vy) &&
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

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    if (setenv("BOPL_RL_ABILITY_SPAWNS", "1", 1) != 0) return 1;
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1],
            "/proc/self/cwd/native/fixtures/movement.rep",
            "/proc/self/cwd/native/target/portal_pickup_ab.log", 120000,
            error, sizeof error);
        if (!world[backend]) { fprintf(stderr, "open: %s\n", error); return 1; }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7; config.map = 0; config.stocks = 100;
    config.warmup_ticks = 240; config.max_episode_ticks = 4000;
    for (unsigned player = 0; player < 2; player++) {
        config.player[player].keyboard = 1;
        config.player[player].abilities[0] = 38;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    if (reset(world[0], &config, "reference") ||
        reset(world[1], &config, "rust")) return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    unsigned spawn_tick = 0, enabled_tick = 0, transit_tick = 0;
    uint32_t pickup_id = 0;
    int64_t previous_x = 0, transit_vx = 0, transit_vy = 0;
    int64_t max_pickup_delta = 0;
    for (unsigned tick = 1; tick <= 2400; tick++) {
        BoplRlAction action[2] = {{0}};
        if (tick == 1950 || tick == 2040) {
            action[0].buttons = 2;
            action[0].angle = tick == 1950 ? 44 : 192;
        }
        for (int backend = 0; backend < 2; backend++)
            if (bopl_rl_step(world[backend], action, 2, 1) != BOPL_RL_OK) return 1;
        const BoplRlState *reference = bopl_rl_state(world[0]);
        const BoplRlState *rust = bopl_rl_state(world[1]);
        if (compare(reference, rust, tick)) return 1;
        for (unsigned i = 0; i < reference->header.entity_count; i++) {
            const BoplRlEntity *entity = &reference->entities[i];
            if (entity->layer != 13 || !(entity->flags & 1)) continue;
            const BoplRlEntity *other = &rust->entities[i];
            int64_t differences[4] = {
                distance(entity->x, other->x), distance(entity->y, other->y),
                distance(entity->vx, other->vx), distance(entity->vy, other->vy)};
            for (unsigned field = 0; field < 4; field++)
                if (differences[field] > max_pickup_delta)
                    max_pickup_delta = differences[field];
            if (!pickup_id) {
                pickup_id = entity->id;
                spawn_tick = tick;
                previous_x = entity->x;
            }
            if (!enabled_tick && entity->flags & 2) enabled_tick = tick;
            if (!transit_tick && previous_x > -INT64_C(42949672960) &&
                entity->x < -INT64_C(42949672960)) {
                transit_tick = tick;
                transit_vx = entity->vx;
                transit_vy = entity->vy;
            }
            previous_x = entity->x;
        }
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (spawn_tick != 2161 || enabled_tick != 2168 || transit_tick != 2233 ||
        pickup_id != 16 || transit_vx >= 0 || transit_vy <= 0 ||
        max_pickup_delta > 16384) {
        fprintf(stderr, "incomplete spawn=%u enabled=%u transit=%u id=%u "
                "velocity=(%" PRId64 ",%" PRId64 ") max_delta=%" PRId64 "\n",
                spawn_tick, enabled_tick, transit_tick, pickup_id,
                transit_vx, transit_vy, max_pickup_delta);
        return 1;
    }
    printf("PORTAL_PICKUP_GOLDEN_OK ticks=2400 spawn=%u enabled=%u "
           "transit=%u velocity=(%" PRId64 ",%" PRId64 ") max_q32_delta=%" PRId64 "\n",
           spawn_tick, enabled_tick, transit_tick, transit_vx, transit_vy,
           max_pickup_delta);
    return 0;
}
