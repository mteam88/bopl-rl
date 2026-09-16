/* Broad Shrink differential: an independently cast persistent Black Hole is
 * hit by Shrink through the complete RigidBodyAffector branch.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/shrink_broad_ab \
 *   research/rust_ab/shrink_broad_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <math.h>
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

static unsigned aim_byte(int64_t sx, int64_t sy, int64_t tx, int64_t ty) {
    double angle = atan2((double)(ty - sy), (double)(tx - sx));
    if (angle < 0.0) angle += 6.28318530717958647692;
    return (unsigned)floor(angle / 6.28318530717958647692 * 255.0) + 1;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/shrink_broad_ab.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) { fprintf(stderr, "open failed: %s\n", error); return 1; }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 123;
    config.map = 0;
    config.stocks = 50;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1200;
    config.player[0].abilities[0] = 18; /* Black hole */
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 17;  /* Shrink */
    config.player[1].abilities[1] = 2;
    config.player[1].abilities[2] = 14;
    if (reset(world[0], &config, "reference") || reset(world[1], &config, "rust")) return 1;
    unsigned shrink_cast = 0, hole_cast = 0, hole_seen = 0, shrink_exit = 0;
    unsigned shrink_held = 0, hole_held = 0;
    int64_t first_radius = 0, max_radius = 0;
    for (unsigned tick = 1; tick <= 520; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (!hole_cast && tick > 180 && before->slimes[0].timers[0] > before->slimes[0].cooldowns[0])
            hole_cast = tick;
        if (hole_cast && hole_held++ < 80) {
            action[0].buttons = 2;
            action[0].angle = 64;
        }
        if (!shrink_cast && hole_cast && tick >= hole_cast + 21 &&
            before->slimes[1].timers[0] > before->slimes[1].cooldowns[0])
            shrink_cast = tick;
        if (shrink_cast) {
            action[1].angle = 128;
            for (unsigned e = 0; e < before->header.entity_count; e++) {
                if (before->entities[e].layer == 23 && (before->entities[e].flags & 1)) {
                    action[1].angle = aim_byte(before->players[1].x, before->players[1].y,
                                               before->entities[e].x, before->entities[e].y);
                    break;
                }
            }
            if (shrink_held++ < 60) action[1].buttons = 2;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "tick %u backend %d: %s\n", tick, backend,
                        bopl_rl_error(world[backend])); return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), tick)) return 1;
        if (shrink_cast && !shrink_exit && tick > shrink_cast &&
            !(reference->slimes[1].ability_flags[0] & 8)) shrink_exit = tick;
        for (unsigned e = 0; e < reference->header.entity_count; e++) {
            if (reference->entities[e].layer != 23 || !(reference->entities[e].flags & 1)) continue;
            if (!hole_seen) { hole_seen = tick; first_radius = reference->entities[e].radius; }
            if (reference->entities[e].radius > max_radius) max_radius = reference->entities[e].radius;
        }
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (!shrink_cast || !shrink_exit || !hole_seen || max_radius <= first_radius) {
        fprintf(stderr, "incomplete shrink=%u exit=%u hole=%u radius=%" PRId64 "/%" PRId64 "\n",
                shrink_cast, shrink_exit, hole_seen, first_radius, max_radius); return 1;
    }
    printf("SHRINK_BROAD_GOLDEN_OK ticks=520 cast=%u exit=%u hole=%u radius=%" PRId64 "..%" PRId64 "\n",
           shrink_cast, shrink_exit, hole_seen, first_radius, max_radius);
    return 0;
}
