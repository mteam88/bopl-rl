/* Grenade non-owner-contact golden: the caster throws upward through an
 * airborne opponent, retreats outside the blast, and receives kill credit.
 * This covers immediate non-owner detonation before the one-second owner
 * gate, deferred explosion registration, and the victim's stock respawn.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/grenade_non_owner_ab \
 *   research/rust_ab/grenade_non_owner_ab.c bopl_rl.c -ldl -lm
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
           a->hierarchy == b->hierarchy && distance(a->x, b->x) <= 1 &&
           distance(a->y, b->y) <= 1 &&
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
                                      "/tmp/grenade_non_owner_ab.log", 120000,
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
    config.player[0].abilities[0] = 3;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    if (reset(world[0], &config, "reference") ||
        reset(world[1], &config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    unsigned cast_tick = 0;
    unsigned item_tick = 0, explosion_tick = 0;
    int saw_item = 0, saw_form_exit = 0, saw_destroyed_item = 0;
    int saw_explosion = 0, saw_disabled_explosion = 0, saw_explosion_end = 0;
    int saw_death = 0, saw_choice = 0, saw_respawn = 0;
    for (unsigned tick = 1; tick <= 700; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        // Cross under the grenade, jump into its upward arc, then release
        // movement before the stock chooser appears.
        if (tick < 85) action[1].angle = 129;
        if (tick == 40) action[1].buttons = 1;
        if (!cast_tick &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
            cast_tick = tick;
        }
        if (cast_tick && tick < cast_tick + 7) {
            action[0].buttons = 2;
            action[0].angle = 16;
        } else if (cast_tick && tick == cast_tick + 7) {
            // ThrowItem2 samples aim on the release pass.
            action[0].angle = 16;
        } else if (cast_tick && tick < 150) {
            // Retreat so only the non-owner is inside the explosion box.
            action[0].angle = 129;
        }
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
        for (unsigned i = 0; i < reference->header.entity_count; i++) {
            const BoplRlEntity *entity = &reference->entities[i];
            if (entity->kind == 0 && entity->layer == 10) {
                if (entity->flags & 1) {
                    saw_item = 1;
                    if (!item_tick) item_tick = tick;
                }
                if (entity->flags == 4) saw_destroyed_item = 1;
            }
            if (entity->kind == 1 && entity->layer == 18) {
                saw_explosion = 1;
                if (!explosion_tick) explosion_tick = tick;
                if (entity->flags == 9) saw_disabled_explosion = 1;
            }
            if (entity->kind == 0 && entity->layer == 27) saw_choice = 1;
        }
        if (cast_tick && tick > cast_tick + 25 &&
            reference->players[0].deaths == 0 &&
            reference->slimes[0].ability_flags[0] == 1) {
            saw_form_exit = 1;
        }
        if (reference->players[1].deaths == 1 &&
            reference->players[0].deaths == 0 &&
            reference->players[0].kills == 1) saw_death = 1;
        if (saw_explosion) {
            int live_explosion = 0;
            for (unsigned i = 0; i < reference->header.entity_count; i++)
                live_explosion |= reference->entities[i].layer == 18;
            if (!live_explosion) saw_explosion_end = 1;
        }
        if (saw_choice && tick > 550) {
            for (unsigned i = 0; i < reference->header.slime_count; i++)
                if (reference->slimes[i].owner == 2 &&
                    (reference->slimes[i].flags & 2)) saw_respawn = 1;
        }
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (!cast_tick || !saw_item || !saw_form_exit || !saw_destroyed_item ||
        !saw_explosion || !saw_disabled_explosion || !saw_explosion_end ||
        !saw_death || !saw_choice || !saw_respawn || !item_tick ||
        !explosion_tick || explosion_tick - item_tick >= 60) {
        fprintf(stderr,
                "incomplete cast=%u item=%d exit=%d destroyed=%d explosion=%d "
                "disabled=%d ended=%d death=%d choice=%d respawn=%d "
                "item_tick=%u explosion_tick=%u\n",
                cast_tick, saw_item, saw_form_exit, saw_destroyed_item,
                saw_explosion, saw_disabled_explosion, saw_explosion_end,
                saw_death, saw_choice, saw_respawn, item_tick, explosion_tick);
        return 1;
    }
    printf("GRENADE_NON_OWNER_GOLDEN_OK ticks=700 hit=%u\n", explosion_tick);
    return 0;
}
