/* Broad Gust differential: one crowded match combines Black Hole, PortalShot,
 * ordinary player/platform targets, and a Gust-forced Macho boulder.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/gust_broad_ab \
 *   eval/rust_ab/gust_broad_ab.c sim/abi/bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/* Black Hole's cast form and the contact-loaded boulder/platform system inherit
 * the established 0.0163-world-unit composition bound. Discrete state,
 * identities, ownership, hierarchy and lifecycle remain exact. */
#define SPATIAL_TOLERANCE 70000000

static int64_t distance(int64_t a, int64_t b) {
    return a > b ? a - b : b - a;
}

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy && distance(a->x, b->x) <= SPATIAL_TOLERANCE && distance(a->y, b->y) <= SPATIAL_TOLERANCE &&
           distance(a->vx, b->vx) <= SPATIAL_TOLERANCE &&
           distance(a->vy, b->vy) <= SPATIAL_TOLERANCE &&
           distance(a->right_x, b->right_x) <= SPATIAL_TOLERANCE &&
           distance(a->right_y, b->right_y) <= SPATIAL_TOLERANCE &&
           distance(a->up_x, b->up_x) <= SPATIAL_TOLERANCE &&
           distance(a->up_y, b->up_y) <= SPATIAL_TOLERANCE &&
           distance(a->radius, b->radius) <= SPATIAL_TOLERANCE &&
           distance(a->rotation, b->rotation) <= SPATIAL_TOLERANCE &&
           distance(a->angular_velocity, b->angular_velocity) <= SPATIAL_TOLERANCE &&
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
        distance(ac.vx, bc.vx) > SPATIAL_TOLERANCE || distance(ac.vy, bc.vy) > SPATIAL_TOLERANCE ||
        distance(ac.rotation, bc.rotation) > SPATIAL_TOLERANCE)
        return 0;
    ac.x = bc.x; ac.y = bc.y; ac.vx = bc.vx; ac.vy = bc.vy;
    ac.rotation = bc.rotation;
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

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/gust_broad_ab.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 4);
    config.seed = 7;
    config.map = 0;
    config.stocks = 100;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1200;
    for (unsigned player = 0; player < 4; player++) {
        config.player[player].keyboard = 1;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    config.player[0].abilities[0] = 7;  /* Gust */
    config.player[1].abilities[0] = 18; /* Black hole */
    config.player[2].abilities[0] = 38; /* Portal gun */
    config.player[3].abilities[0] = 21; /* Throw/boulder */
    if (reset(world[0], &config, "reference") ||
        reset(world[1], &config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    int saw_wave = 0, saw_hole = 0, saw_shot = 0, saw_boulder = 0;
    int wave_moved_hole = 0, wave_moved_shot = 0, wave_moved_boulder = 0;
    int64_t prior_hole_vx = 0, prior_hole_vy = 0;
    int64_t prior_shot_vx = 0, prior_shot_vy = 0;
    int64_t prior_boulder_vx = 0, prior_boulder_vy = 0;
    for (unsigned tick = 1; tick <= 430; tick++) {
        BoplRlAction action[4] = {{0}};
        /* Leave a released Black Hole and boulder in the wave radius. */
        if (tick >= 181 && tick < 261) {
            action[1].buttons = 2;
            action[1].angle = 192;
        }
        if (tick == 181) {
            action[3].buttons = 2;
            action[3].angle = 192;
        }
        /* Spawn a spell-layer rigid body in the same crowded pass as Gust. */
        if (tick == 300) {
            action[2].buttons = 2;
            action[2].angle = 64;
            action[0].buttons = 2;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 4, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "tick %u backend %d failed (%d): %s\n", tick,
                        backend, status, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), tick)) return 1;
        for (unsigned i = 0; i < reference->header.entity_count; i++) {
            const BoplRlEntity *entity = &reference->entities[i];
            if (!(entity->flags & 1)) continue;
            if (entity->layer == 23) {
                if (saw_wave && (entity->vx != prior_hole_vx || entity->vy != prior_hole_vy))
                    wave_moved_hole = 1;
                prior_hole_vx = entity->vx;
                prior_hole_vy = entity->vy;
                saw_hole = 1;
            } else if (entity->kind == 0 && entity->layer == 10 &&
                       entity->radius == 1288490189) {
                if (saw_wave && (entity->vx != prior_shot_vx || entity->vy != prior_shot_vy))
                    wave_moved_shot = 1;
                prior_shot_vx = entity->vx;
                prior_shot_vy = entity->vy;
                saw_shot = 1;
            } else if (entity->kind == 2 && entity->layer == 11 &&
                       entity->hierarchy >= 100000000u) {
                if (saw_wave && (entity->vx != prior_boulder_vx || entity->vy != prior_boulder_vy))
                    wave_moved_boulder = 1;
                prior_boulder_vx = entity->vx;
                prior_boulder_vy = entity->vy;
                saw_boulder = 1;
            }
        }
        if (reference->slimes[0].ability_flags[0] & 8) saw_wave = 1;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (!saw_wave || !saw_hole || !saw_shot || !saw_boulder ||
        !wave_moved_boulder) {
        fprintf(stderr, "incomplete wave=%d hole=%d/%d shot=%d/%d boulder=%d/%d\n",
                saw_wave, saw_hole, wave_moved_hole, saw_shot, wave_moved_shot,
                saw_boulder, wave_moved_boulder);
        return 1;
    }
    printf("GUST_BROAD_GOLDEN_OK ticks=430 hole_spawn=%d shot_spawn=%d boulder_force=%d\n",
           saw_hole, saw_shot, wave_moved_boulder);
    return 0;
}
