/* Freeze ray probe. */
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
           a->hierarchy == b->hierarchy && distance(a->x, b->x) <= 64 &&
           distance(a->y, b->y) <= 64 && distance(a->vx, b->vx) <= 64 &&
           distance(a->vy, b->vy) <= 64 &&
           distance(a->right_x, b->right_x) <= 1 &&
           distance(a->right_y, b->right_y) <= 1 &&
           distance(a->up_x, b->up_x) <= 1 &&
           distance(a->up_y, b->up_y) <= 1 &&
           distance(a->radius, b->radius) <= 64 &&
           distance(a->rotation, b->rotation) <= 64 &&
           distance(a->angular_velocity, b->angular_velocity) <= 64 &&
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
        int player_position_diff = distance(ap.x, bp.x) > 64 || distance(ap.y, bp.y) > 64;
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
        int slime_vector_diff = distance(as.x, bs.x) > 64 || distance(as.y, bs.y) > 64 ||
            distance(as.vx, bs.vx) > 64 || distance(as.vy, bs.vy) > 64;
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

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/freeze_ability_golden.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) { fprintf(stderr, "open failed: %s\n", error); return 1; }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 123;
    config.map = 0;
    config.stocks = 5;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1200;
    config.player[0].abilities[0] = 36;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    if (reset(world[0], &config, "reference") || reset(world[1], &config, "rust")) return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;
    unsigned cast_tick = 0, exit_tick = 0, held = 0;
    for (unsigned tick = 1; tick <= 500; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (!cast_tick && tick > 180 && !(before->slimes[0].ability_flags[0] & 8) &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
            cast_tick = tick;
        }
        if (cast_tick && held++ < 60) {
            action[0].buttons = 2;
            action[0].angle = 1;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) { fprintf(stderr, "tick %u backend %d: %s\n", tick, backend, bopl_rl_error(world[backend])); return 1; }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), tick)) return 1;
        if (cast_tick && !exit_tick && tick > cast_tick && !(reference->slimes[0].ability_flags[0] & 8))
            exit_tick = tick;
    }
    if (!cast_tick || !exit_tick) {
        fprintf(stderr, "incomplete cast=%u exit=%u\n", cast_tick, exit_tick);
        return 1;
    }
    unsigned ground_cast = cast_tick, ground_exit = exit_tick;

    /* Airborne entry, diagonal aim, recoil flight, and animation exit. */
    if (reset(world[0], &config, "reference-air") || reset(world[1], &config, "rust-air")) return 1;
    cast_tick = exit_tick = held = 0;
    for (unsigned tick = 1; tick <= 500; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (tick == 250) action[0].buttons = 1;
        if (!cast_tick && tick > 260 && !(before->slimes[0].ability_flags[0] & 8) &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) cast_tick = tick;
        if (cast_tick && held++ < 35) {
            action[0].buttons = 2;
            action[0].angle = 32;
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) { fprintf(stderr, "air tick %u backend %d: %s\n", tick, backend, bopl_rl_error(world[backend])); return 1; }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), tick + 1000)) return 1;
        if (cast_tick && !exit_tick && tick > cast_tick && !(reference->slimes[0].ability_flags[0] & 8)) exit_tick = tick;
    }
    if (!cast_tick || !exit_tick) { fprintf(stderr, "incomplete air cast=%u exit=%u\n", cast_tick, exit_tick); return 1; }
    unsigned air_cast = cast_tick, air_exit = exit_tick;

    printf("FREEZE_ABILITY_GOLDEN_OK scenarios=2 ticks=1000 ground=%u/%u air=%u/%u\n",
           ground_cast, ground_exit, air_cast, air_exit);
    for (int i=0;i<2;i++) bopl_rl_close(world[i]);
    return 0;
}
