/* Portal Gun differential golden: same-pass projectile physics, timeout and
 * wall landing, pair linking/replacement, player transit and rigid-body transit.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/portal_gun_ability_ab \
 *   eval/rust_ab/portal_gun_ability_ab.c sim/abi/bopl_rl.c -ldl -lm
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
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/portal_gun_ability_ab.log", 120000,
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
        config.player[player].abilities[0] = 38;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    if (reset(world[0], &config, "reference") ||
        reset(world[1], &config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    unsigned first_shot = 0, timeout_tick = 0, wall_hit = 0;
    unsigned player_transit = 0, replacement_shot = 0, replacement_timeout = 0;
    int64_t previous_x = bopl_rl_state(world[0])->slimes[0].x;
    for (unsigned tick = 1; tick <= 900; tick++) {
        BoplRlAction action[2] = {{0}};
        if (tick == 181 || tick == 270 || tick == 500) {
            action[0].buttons = 2;
            action[0].angle = tick == 270 ? 192 : 1;
        }
        if (tick >= 300 && tick < 380) action[1].buttons = 512;
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "tick %u backend %d failed (%d): %s\n", tick,
                        backend, status, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), tick)) return 1;
        unsigned live_shots = 0;
        for (unsigned i = 0; i < reference->header.entity_count; i++) {
            const BoplRlEntity *entity = &reference->entities[i];
            if (entity->kind != 0 || entity->layer != 10 || entity->radius != 1288490189)
                continue;
            live_shots++;
            if (!first_shot) first_shot = tick;
            if (entity->flags == 4) wall_hit = tick;
            if (entity->id >= 16 && !replacement_shot) replacement_shot = tick;
        }
        if (first_shot && !timeout_tick && tick > first_shot && tick < 270 && !live_shots)
            timeout_tick = tick;
        if (replacement_shot && !replacement_timeout && tick > replacement_shot && !live_shots)
            replacement_timeout = tick;
        if (!player_transit && previous_x < 0 && reference->slimes[0].x > 0)
            player_transit = tick;
        previous_x = reference->slimes[0].x;
    }
    if (first_shot != 181 || timeout_tick != 254 || wall_hit != 270 ||
        player_transit != 275 || replacement_shot != 500 || replacement_timeout != 573) {
        fprintf(stderr,
                "incomplete first=%u timeout=%u wall=%u transit=%u replacement=%u replacement_timeout=%u\n",
                first_shot, timeout_tick, wall_hit, player_transit,
                replacement_shot, replacement_timeout);
        return 1;
    }

    /* A grenade starts 0.0185 units apart because the existing ThrowItem2
     * port composes its animated hand offset differently. Portal capture
     * must still occur on the same pass and reset both positions to the same
     * exit mouth; mapped velocity may retain at most 0.0024 units/s of that
     * pre-existing input difference. */
    for (unsigned player = 0; player < 2; player++)
        config.player[player].abilities[1] = 3;
    if (reset(world[0], &config, "reference objects") ||
        reset(world[1], &config, "rust objects"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;
    unsigned body_transit[2] = {0, 0};
    BoplRlEntity transit_entity[2] = {{0}};
    for (unsigned tick = 1; tick <= 390; tick++) {
        BoplRlAction action[2] = {{0}};
        if (tick == 181 || tick == 270) {
            action[0].buttons = 2;
            action[0].angle = tick == 270 ? 192 : 1;
        }
        if (tick >= 300 && tick < 380) action[1].buttons = 512;
        if (tick >= 360 && tick < 367) {
            action[0].buttons |= 4;
            action[0].angle = 1;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "object tick %u backend %d failed (%d): %s\n",
                        tick, backend, status, bopl_rl_error(world[backend]));
                return 1;
            }
            const BoplRlState *state = bopl_rl_state(world[backend]);
            for (unsigned i = 0; i < state->header.entity_count; i++) {
                const BoplRlEntity *entity = &state->entities[i];
                if (entity->kind == 0 && entity->layer == 10 &&
                    entity->radius == 3994319616 && entity->x < -10LL * 4294967296LL &&
                    !body_transit[backend]) {
                    body_transit[backend] = tick;
                    transit_entity[backend] = *entity;
                }
            }
        }
        if (tick < 360 && compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), tick))
            return 1;
    }
    if (body_transit[0] != 383 || body_transit[1] != 383 ||
        distance(transit_entity[0].x, transit_entity[1].x) > 10000000 ||
        distance(transit_entity[0].y, transit_entity[1].y) > 10000000 ||
        distance(transit_entity[0].vx, transit_entity[1].vx) > 10000000 ||
        distance(transit_entity[0].vy, transit_entity[1].vy) > 10000000) {
        fprintf(stderr, "rigid transit differs: ticks %u/%u pos (%" PRId64 ",%" PRId64
                ")/(%" PRId64 ",%" PRId64 ") vel (%" PRId64 ",%" PRId64
                ")/(%" PRId64 ",%" PRId64 ")\n",
                body_transit[0], body_transit[1], transit_entity[0].x,
                transit_entity[0].y, transit_entity[1].x, transit_entity[1].y,
                transit_entity[0].vx, transit_entity[0].vy, transit_entity[1].vx,
                transit_entity[1].vy);
        return 1;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    printf("PORTAL_GUN_ABILITY_GOLDEN_OK first=%u timeout=%u wall=%u player_transit=%u replacement=%u replacement_timeout=%u body_transit=%u ticks=1290\n",
           first_shot, timeout_tick, wall_hit, player_transit,
           replacement_shot, replacement_timeout, body_transit[0]);
    return 0;
}
