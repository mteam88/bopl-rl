/* Tesla Coil differential golden. */
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
           a->hierarchy == b->hierarchy &&
           distance(a->x, b->x) <= 16 &&
           distance(a->y, b->y) <= 16 &&
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
            for (unsigned z = 0; z < a->header.slime_count; z++) fprintf(stderr, "  slime %u owner %u/%u pos (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") vel (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") flags %u/%u\n", z, a->slimes[z].owner, b->slimes[z].owner, a->slimes[z].x,a->slimes[z].y,b->slimes[z].x,b->slimes[z].y,a->slimes[z].vx,a->slimes[z].vy,b->slimes[z].vx,b->slimes[z].vy,a->slimes[z].ability_flags[0],b->slimes[z].ability_flags[0]);
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
                                      "/tmp/tesla_ability_golden.log", 120000,
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
    config.max_episode_ticks = 2000;
    config.player[0].abilities[0] = 23;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    if (reset(world[0], &config, "reference") ||
        reset(world[1], &config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    unsigned cast_count = 0;
    int saw_form = 0, saw_exit = 0, saw_two_nodes = 0;
    int saw_loading_line = 0, saw_one_lethal = 0, saw_two_lethal = 0;
    int saw_active_line = 0, saw_flicker = 0, saw_reenable = 0;
    int saw_platform_composite = 0, nodes_after_exit = 0;
    int saw_destroyed_line = 0, saw_ring_pair = 0;
    for (unsigned tick = 1; tick <= 900; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (cast_count < 3 && tick > 180 &&
            !(before->slimes[0].ability_flags[0] & 8) &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
            cast_count++;
            action[0].buttons = 2;
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
        unsigned ability_flags = 0;
        for (unsigned i = 0; i < reference->header.slime_count; i++) {
            if (reference->slimes[i].owner == 1) {
                ability_flags = reference->slimes[i].ability_flags[0];
                break;
            }
        }
        if (ability_flags & 8) saw_form = 1;
        if (saw_form && !(ability_flags & 8)) saw_exit = 1;

        unsigned nodes = 0, lethal_nodes = 0;
        for (unsigned i = 0; i < reference->header.entity_count; i++) {
            const BoplRlEntity *entity = &reference->entities[i];
            if (entity->kind == 0 && entity->owner == 1 &&
                (entity->layer == 21 || entity->layer == 16) &&
                (entity->flags & 16)) {
                nodes++;
                if (entity->layer == 16) lethal_nodes++;
            }
            if (entity->kind == 1 && entity->layer == 18 && entity->owner == 1) {
                if (entity->flags == 10) saw_loading_line = 1;
                if (entity->flags == 11) {
                    if (saw_flicker) saw_reenable = 1;
                    saw_active_line = 1;
                }
                if (saw_active_line && entity->flags == 9) saw_flicker = 1;
            }
            if (entity->kind == 1 && entity->layer == 18 &&
                entity->owner == 0 && entity->flags == 12)
                saw_destroyed_line = 1;
            if (entity->kind == 2 && (entity->flags & 16))
                saw_platform_composite = 1;
        }
        if (nodes == 2) saw_two_nodes = 1;
        if (cast_count == 3 && nodes == 2) saw_ring_pair = 1;
        if (lethal_nodes == 1) saw_one_lethal = 1;
        if (lethal_nodes == 2) saw_two_lethal = 1;
        if (saw_exit && nodes != 0) nodes_after_exit = 1;
    }
    const BoplRlState *final = bopl_rl_state(world[0]);
    if (cast_count != 3 || !saw_form || !saw_exit || !saw_two_nodes ||
        !saw_loading_line || !saw_one_lethal || !saw_two_lethal ||
        !saw_active_line || !saw_flicker || !saw_reenable ||
        !saw_platform_composite || !nodes_after_exit || !saw_destroyed_line ||
        !saw_ring_pair ||
        final->players[0].deaths != 0) {
        fprintf(stderr,
                "incomplete casts=%u form=%d exit=%d nodes=%d loading=%d "
                "one_lethal=%d two_lethal=%d active=%d flicker=%d "
                "reenable=%d composite=%d persistent=%d destroyed=%d "
                "ring_pair=%d deaths=%u\n",
                cast_count, saw_form, saw_exit, saw_two_nodes, saw_loading_line,
                saw_one_lethal, saw_two_lethal, saw_active_line, saw_flicker,
                saw_reenable, saw_platform_composite, nodes_after_exit,
                saw_destroyed_line, saw_ring_pair, final->players[0].deaths);
        return 1;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    printf("TESLA_ABILITY_GOLDEN_OK ticks=900 casts=%u\n", cast_count);
    return 0;
}
