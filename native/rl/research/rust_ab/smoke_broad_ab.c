/* Broad Smoke differential: charged airborne release, a non-igniting Bow
 * Arrow contact after the owner gate, later ordinary-contact detonation, all
 * four randomized cloud registrations, and persistent cloud animation. */
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
           distance(a->y, b->y) <= 1 && distance(a->vx, b->vx) <= 1 &&
           distance(a->vy, b->vy) <= 1 &&
           distance(a->right_x, b->right_x) <= 1 &&
           distance(a->right_y, b->right_y) <= 1 &&
           distance(a->up_x, b->up_x) <= 1 &&
           distance(a->up_y, b->up_y) <= 1 &&
           distance(a->radius, b->radius) <= 1 &&
           distance(a->rotation, b->rotation) <= 1 &&
           distance(a->angular_velocity, b->angular_velocity) <= 1 &&
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
        fprintf(stderr, "tick %u header differs: entities %u/%u slimes %u/%u\n",
                tick, a->header.entity_count, b->header.entity_count,
                a->header.slime_count, b->header.slime_count);
        return 1;
    }
    for (unsigned i = 0; i < a->header.player_count; i++) {
        if (memcmp(&a->players[i], &b->players[i], sizeof a->players[i]) != 0 ||
            memcmp(&a->telemetry[i], &b->telemetry[i],
                   sizeof a->telemetry[i]) != 0) {
            fprintf(stderr, "tick %u player %u differs\n", tick, i);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        if (memcmp(&a->slimes[i], &b->slimes[i], sizeof a->slimes[i]) != 0) {
            fprintf(stderr,
                    "tick %u slime %u differs: p (%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") v (%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") flags %u/%u\n",
                    tick, i, a->slimes[i].x, a->slimes[i].y,
                    b->slimes[i].x, b->slimes[i].y, a->slimes[i].vx,
                    a->slimes[i].vy, b->slimes[i].vx, b->slimes[i].vy,
                    a->slimes[i].ability_flags[0], b->slimes[i].ability_flags[0]);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.entity_count; i++) {
        if (!entity_equal(&a->entities[i], &b->entities[i])) {
            fprintf(stderr,
                    "tick %u entity %u differs: id %u/%u kind %u/%u layer %u/%u "
                    "flags %u/%u p (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") v (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ")\n",
                    tick, i, a->entities[i].id, b->entities[i].id,
                    a->entities[i].kind, b->entities[i].kind,
                    a->entities[i].layer, b->entities[i].layer,
                    a->entities[i].flags, b->entities[i].flags,
                    a->entities[i].x, a->entities[i].y, b->entities[i].x,
                    b->entities[i].y, a->entities[i].vx, a->entities[i].vy,
                    b->entities[i].vx, b->entities[i].vy);
            fprintf(stderr,
                    "  radius %" PRId64 "/%" PRId64 " rotation %" PRId64
                    "/%" PRId64 " angular %" PRId64 "/%" PRId64
                    " inverse_mass %" PRId64 "/%" PRId64 " owner %u/%u "
                    "hierarchy %u/%u\n",
                    a->entities[i].radius, b->entities[i].radius,
                    a->entities[i].rotation, b->entities[i].rotation,
                    a->entities[i].angular_velocity,
                    b->entities[i].angular_velocity,
                    a->entities[i].inverse_mass, b->entities[i].inverse_mass,
                    a->entities[i].owner, b->entities[i].owner,
                    a->entities[i].hierarchy, b->entities[i].hierarchy);
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
                                      "/tmp/smoke_broad_ab.log", 120000,
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
    config.player[0].abilities[0] = 13;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    if (reset(world[0], &config, "reference") ||
        reset(world[1], &config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    unsigned cast_tick = 0, smoke_registered = 0, arrow_registered = 0;
    unsigned contact_tick = 0, clouds_registered = 0;
    int previous_arrow = 0;
    unsigned survived_after_contact = 0;
    for (unsigned tick = 1; tick <= 500; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (!cast_tick && tick > 150 &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0])
            cast_tick = tick;
        if (cast_tick && tick < cast_tick + 35) {
            action[0].buttons = 2;
            action[0].angle = 32;
        }
        if (cast_tick && tick < cast_tick + 40) {
            action[1].buttons = 4;
            action[1].angle = 80;
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
        int live_arrow = 0, destroyed_arrow = 0, live_smoke = 0;
        unsigned clouds = 0;
        for (unsigned i = 0; i < reference->header.entity_count; i++) {
            const BoplRlEntity *entity = &reference->entities[i];
            if (entity->kind == 0 && entity->layer == 9) {
                live_arrow |= (entity->flags & 1) != 0;
                destroyed_arrow |= entity->flags == 4;
            }
            if (entity->kind == 0 && entity->layer == 10 &&
                (entity->flags & 1))
                live_smoke = 1;
            if (entity->kind == 0 && entity->layer == 22 && (entity->flags & 1))
                clouds++;
        }
        if (!smoke_registered && live_smoke) smoke_registered = tick;
        if (!arrow_registered && live_arrow) arrow_registered = tick;
        if (previous_arrow && destroyed_arrow && !contact_tick) {
            contact_tick = tick;
            if (!live_smoke) {
                fprintf(stderr, "Smoke did not survive non-igniting Arrow on tick %u\n", tick);
                return 1;
            }
        }
        if (contact_tick && tick > contact_tick && tick <= contact_tick + 12 &&
            live_smoke)
            survived_after_contact++;
        if (!clouds_registered && clouds == 4) clouds_registered = tick;
        previous_arrow = live_arrow;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (cast_tick != 151 || smoke_registered != 186 || arrow_registered != 192 ||
        contact_tick != 218 || clouds_registered != 266 || survived_after_contact != 12) {
        fprintf(stderr,
                "unexpected stages cast=%u smoke=%u arrow=%u contact=%u clouds=%u "
                "survival=%u\n", cast_tick, smoke_registered, arrow_registered,
                contact_tick, clouds_registered, survived_after_contact);
        return 1;
    }
    printf("SMOKE_BROAD_GOLDEN_OK ticks=500 stages=%u/%u/%u/%u/%u survival=%u\n",
           cast_tick, smoke_registered, arrow_registered, contact_tick,
           clouds_registered, survived_after_contact);
    return 0;
}
