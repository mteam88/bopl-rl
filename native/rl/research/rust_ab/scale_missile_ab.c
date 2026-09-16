/* Duplicator Missile probe. */
#include "bopl_rl.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int64_t distance(int64_t a, int64_t b) {
    return a > b ? a - b : b - a;
}

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    int64_t form_tolerance = 3;
    int64_t rotation_tolerance =
        1;
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy && a->x == b->x &&
           a->y == b->y && a->vx == b->vx &&
           a->vy == b->vy &&
           distance(a->right_x, b->right_x) <= form_tolerance &&
           distance(a->right_y, b->right_y) <= form_tolerance &&
           distance(a->up_x, b->up_x) <= form_tolerance &&
           distance(a->up_y, b->up_y) <= form_tolerance &&
           distance(a->radius, b->radius) <= 1 &&
           distance(a->rotation, b->rotation) <= rotation_tolerance &&
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
        BoplRlPlayer ap = a->players[i], bp = b->players[i];
        int player_position_diff = distance(ap.x, bp.x) > 0 || distance(ap.y, bp.y) > 0;
        ap.x = ap.y = bp.x = bp.y = 0;
        if (player_position_diff || memcmp(&ap, &bp, sizeof ap) != 0 ||
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
            for (unsigned e = 0; e < a->header.entity_count; e++) if (!entity_equal(&a->entities[e], &b->entities[e])) fprintf(stderr, "  entity %u flags %u/%u p (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") rot %" PRId64 "/%" PRId64 "\n", e, a->entities[e].flags,b->entities[e].flags,a->entities[e].x,a->entities[e].y,b->entities[e].x,b->entities[e].y,a->entities[e].rotation,b->entities[e].rotation);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        BoplRlSlime as = a->slimes[i], bs = b->slimes[i];
        int slime_vector_diff = distance(as.x, bs.x) > 0 || distance(as.y, bs.y) > 0 ||
            distance(as.vx, bs.vx) > 0 || distance(as.vy, bs.vy) > 0;
        as.x = as.y = as.vx = as.vy = bs.x = bs.y = bs.vx = bs.vy = 0;
        if (slime_vector_diff || memcmp(&as, &bs, sizeof as) != 0) {
            fprintf(stderr,
                    "tick %u slime %u differs: position (%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") ability flags %u/%u\n",
                    tick, i, a->slimes[i].x, a->slimes[i].y, b->slimes[i].x,
                    b->slimes[i].y, a->slimes[i].ability_flags[0],
                    b->slimes[i].ability_flags[0]);
            fprintf(stderr,
                    "  velocity (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") state %u/%u ability1 %u/%u\n",
                    a->slimes[i].vx, a->slimes[i].vy, b->slimes[i].vx,
                    b->slimes[i].vy, a->slimes[i].state, b->slimes[i].state,
                    a->slimes[i].ability_flags[1], b->slimes[i].ability_flags[1]);
            const unsigned char *aa=(const unsigned char *)&a->slimes[i];
            const unsigned char *bb=(const unsigned char *)&b->slimes[i];
            for (unsigned z=0; z<sizeof a->slimes[i]; z++) if (aa[z]!=bb[z]) fprintf(stderr, "  byte %u %u/%u\n",z,aa[z],bb[z]);
            for (unsigned e = 0; e < a->header.entity_count; e++) {
                if (!entity_equal(&a->entities[e], &b->entities[e]))
                    fprintf(stderr, "  entity %u layer %u/%u flags %u/%u p (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n", e,a->entities[e].layer,b->entities[e].layer,a->entities[e].flags,b->entities[e].flags,a->entities[e].x,a->entities[e].y,b->entities[e].x,b->entities[e].y);
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
            for (unsigned z = 0; z < a->header.entity_count; z++) {
                if (a->entities[z].kind == 2)
                    fprintf(stderr, "  platform %u id %u p (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") rot %" PRId64 "/%" PRId64 "\n",
                            z, a->entities[z].id, a->entities[z].x, a->entities[z].y,
                            b->entities[z].x, b->entities[z].y,
                            a->entities[z].rotation, b->entities[z].rotation);
            }
            return 1;
        }
    }
    return 0;
}

static unsigned aim_byte(int64_t sx, int64_t sy, int64_t tx, int64_t ty) {
    const double pi = 3.14159265358979323846;
    double angle = atan2((double)(ty - sy), (double)(tx - sx));
    if (angle < 0.0)
        angle += 2.0 * pi;
    return (unsigned)floor(angle / (2.0 * pi) * 255.0) + 1;
}

static unsigned active_items(const BoplRlState *state) {
    unsigned count = 0;
    for (unsigned i = 0; i < state->header.entity_count; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->layer == 10 && (entity->flags & 1))
            count++;
    }
    return count;
}

static unsigned active_grenade_explosions(const BoplRlState *state) {
    unsigned count = 0;
    for (unsigned i = 0; i < state->header.entity_count; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->layer == 18 && (entity->flags & 1))
            count++;
    }
    return count;
}

static int run_case(BoplRl **world, BoplRlConfig *config, unsigned ability,
                    const char *label, int64_t expected_right_x,
                    int64_t expected_right_y, int64_t expected_up_x,
                    int64_t expected_up_y) {
    config->player[0].abilities[0] = 10;      /* Missile */
    config->player[0].abilities[1] = ability; /* Growth or Shrink */
    config->player[0].abilities[2] = 14;
    if (reset(world[0], config, label) || reset(world[1], config, label)) return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), ability * 1000))
        return 1;

    unsigned missile_cast = 0, missile_registered = 0, missile_exit = 0;
    unsigned ray_cast = 0, ray_release = 0, first_scaled = 0;
    unsigned missile_end = 0, explosion_registered = 0;
    unsigned explosion_disabled = 0, explosion_end = 0;
    unsigned previous_missiles = 0, previous_explosions = 0;
    int64_t right_x_260 = 0, right_y_260 = 0, up_x_260 = 0, up_y_260 = 0;
    for (unsigned tick = 1; tick <= 450; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (!missile_cast &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0] &&
            before->slimes[0].timers[1] > before->slimes[0].cooldowns[1])
            missile_cast = tick;
        if (missile_cast && tick < missile_cast + 45) {
            action[0].buttons = 2;
            action[0].angle = 128;
        }
        if (missile_cast && !missile_exit && tick > missile_cast + 10 &&
            !(before->slimes[0].ability_flags[0] & 8))
            missile_exit = tick;
        if (missile_exit && !ray_cast &&
            before->slimes[0].timers[1] > before->slimes[0].cooldowns[1])
            ray_cast = tick;
        if (ray_cast && tick <= ray_cast + 45) {
            int found = 0;
            int64_t target_x = 0, target_y = 0;
            for (unsigned i = 0; i < before->header.entity_count; i++) {
                const BoplRlEntity *entity = &before->entities[i];
                if (entity->kind == 1 && entity->layer == 10 &&
                    (entity->flags & 1)) {
                    target_x = entity->x;
                    target_y = entity->y;
                    found = 1;
                    break;
                }
            }
            action[0].angle = found ? aim_byte(before->players[0].x,
                                               before->players[0].y,
                                               target_x, target_y)
                                    : 1;
            if (tick < ray_cast + 45)
                action[0].buttons = 4;
            else
                ray_release = tick;
        }
        for (int backend = 0; backend < 2; backend++) {
            if (bopl_rl_step(world[backend], action, 2, 1) != BOPL_RL_OK) {
                fprintf(stderr, "%s step %u/%d failed: %s\n", label, tick,
                        backend, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), ability * 1000 + tick))
            return 1;

        unsigned missiles = 0, explosions = 0;
        int disabled = 0;
        for (unsigned i = 0; i < reference->header.entity_count; i++) {
            const BoplRlEntity *entity = &reference->entities[i];
            if (entity->kind == 1 && entity->layer == 10 &&
                (entity->flags & 1)) {
                missiles++;
                if (!missile_registered) missile_registered = tick;
                if (ray_release && tick >= ray_release && !first_scaled &&
                    entity->right_x != INT64_C(1286531713))
                    first_scaled = tick;
                if (tick == 260) {
                    right_x_260 = entity->right_x;
                    right_y_260 = entity->right_y;
                    up_x_260 = entity->up_x;
                    up_y_260 = entity->up_y;
                    if (entity->inverse_mass != INT64_C(858993459)) {
                        fprintf(stderr, "%s changed zero-inertia Missile mass\n", label);
                        return 1;
                    }
                }
            }
            if (entity->kind == 1 && entity->layer == 18) {
                explosions += (entity->flags & 1) != 0;
                disabled |= entity->flags == 9;
            }
        }
        if (previous_missiles && !missiles && !missile_end) missile_end = tick;
        if (!previous_explosions && explosions && !explosion_registered)
            explosion_registered = tick;
        if (disabled && !explosion_disabled) explosion_disabled = tick;
        if (previous_explosions && !explosions && !explosion_end)
            explosion_end = tick;
        previous_missiles = missiles;
        previous_explosions = explosions;
    }
    const BoplRlState *final = bopl_rl_state(world[0]);
    for (unsigned i = 0; i < final->header.player_count; i++) {
        if (final->players[i].deaths || final->telemetry[i].deaths_other) {
            fprintf(stderr, "%s unexpected death/credit for player %u\n", label, i);
            return 1;
        }
    }
    if (missile_cast != 154 || missile_registered != 173 || missile_exit != 206 ||
        ray_cast != 206 || ray_release != 251 || first_scaled != 252 ||
        missile_end != 261 || explosion_registered != 262 ||
        explosion_disabled != 263 || explosion_end != 344 ||
        right_x_260 != expected_right_x || right_y_260 != expected_right_y ||
        up_x_260 != expected_up_x || up_y_260 != expected_up_y) {
        fprintf(stderr,
                "%s stages=%u/%u/%u/%u/%u/%u/%u/%u/%u/%u "
                "basis=%" PRId64 ",%" PRId64 "/%" PRId64 ",%" PRId64 "\n",
                label, missile_cast, missile_registered, missile_exit, ray_cast,
                ray_release, first_scaled, missile_end, explosion_registered,
                explosion_disabled, explosion_end, right_x_260, right_y_260,
                up_x_260, up_y_260);
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
                                      "/tmp/scale_missile_ab.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 123;
    config.map = 0;
    config.stocks = 5;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1000;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    if (run_case(world, &config, 8, "growth-missile",
                 INT64_C(1362790008), INT64_C(2521506519),
                 INT64_C(-3402032728), INT64_C(1838684998)) ||
        run_case(world, &config, 17, "shrink-missile",
                 INT64_C(1210273412), INT64_C(2239312206),
                 INT64_C(-3021294355), INT64_C(1632908631)))
        return 1;
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    printf("SCALE_MISSILE_GOLDEN_OK cases=2 ticks=900 "
           "stages=154/173/206/251/252/261/262/263/344\n");
    return 0;
}
