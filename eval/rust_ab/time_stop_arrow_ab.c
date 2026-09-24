/* Time Stop golden: grounded channel/cast/exit ordering, team protection,
 * global time-scale lifetime, frozen opponent motion/cooldowns, and protected
 * caster motion/cooldown semantics. */
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
                "tick %u header differs: ticks %u/%u entities %u/%u slimes %u/%u scale %" PRId64 "/%" PRId64 " flags %u/%u sim %" PRId64 "/%" PRId64 " episode %u/%u\n",
                tick, a->header.tick, b->header.tick, a->header.entity_count,
                b->header.entity_count, a->header.slime_count,
                b->header.slime_count, a->header.player_time_scale, b->header.player_time_scale,
                a->header.flags, b->header.flags, a->header.simulation_time, b->header.simulation_time,
                a->header.episode_ticks, b->header.episode_ticks);
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
            for (unsigned z = 0; z < a->header.slime_count; z++) fprintf(stderr, "  slime %u pos (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") player (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n", z,a->slimes[z].x,a->slimes[z].y,b->slimes[z].x,b->slimes[z].y,a->players[z].x,a->players[z].y,b->players[z].x,b->players[z].y);
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
                                      "/tmp/time_stop_arrow_ab.log", 120000,
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
    config.player[0].abilities[0] = 12;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    if (reset(world[0], &config, "reference") ||
        reset(world[1], &config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    unsigned cast_tick = 0, arrow_tick = 0, stop_tick = 0, restore_tick = 0;
    unsigned frozen_ticks = 0;
    int64_t frozen_x = 0, frozen_y = 0;
    int saw_arrow = 0, saw_restored_motion = 0, saw_post_stop_death = 0;
    for (unsigned tick = 1; tick <= 1450; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (!cast_tick && tick > 180 &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0])
            cast_tick = tick;
        if (cast_tick && tick < cast_tick + 600) action[0].buttons = 2;
        action[1].angle = 64;
        if (tick >= 680 && tick <= 700) action[1].buttons = 4;

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

        const BoplRlEntity *arrow = NULL;
        for (unsigned i = 0; i < reference->header.entity_count; i++) {
            if (reference->entities[i].layer == 9 &&
                (reference->entities[i].flags & 1)) {
                arrow = &reference->entities[i];
                break;
            }
        }
        if (arrow && !saw_arrow) {
            saw_arrow = 1;
            arrow_tick = tick;
        }
        if (reference->header.player_time_scale == 0) {
            if (!arrow) {
                fprintf(stderr, "tick %u arrow absent during time stop\n", tick);
                return 1;
            }
            if (!stop_tick) {
                stop_tick = tick;
                frozen_x = arrow->x;
                frozen_y = arrow->y;
            } else if (arrow->x != frozen_x || arrow->y != frozen_y) {
                fprintf(stderr,
                        "tick %u arrow advanced while stopped: (%" PRId64
                        ",%" PRId64 ") -> (%" PRId64 ",%" PRId64 ")\n",
                        tick, frozen_x, frozen_y, arrow->x, arrow->y);
                return 1;
            }
            frozen_ticks++;
        } else if (stop_tick) {
            if (!restore_tick) restore_tick = tick;
            if (arrow && (arrow->x != frozen_x || arrow->y != frozen_y))
                saw_restored_motion = 1;
            if (reference->players[0].deaths || reference->players[1].deaths)
                saw_post_stop_death = 1;
            if (saw_restored_motion || saw_post_stop_death) break;
        }
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (!cast_tick || !saw_arrow || !arrow_tick || !stop_tick || !restore_tick ||
        frozen_ticks < 600 || (!saw_restored_motion && !saw_post_stop_death)) {
        fprintf(stderr,
                "incomplete cast=%u arrow=%u stop=%u restore=%u frozen=%u "
                "motion=%d death=%d\n",
                cast_tick, arrow_tick, stop_tick, restore_tick, frozen_ticks,
                saw_restored_motion, saw_post_stop_death);
        return 1;
    }
    printf("TIME_STOP_ARROW_GOLDEN_OK cast=%u arrow=%u stop=%u restore=%u "
           "frozen=%u motion=%d death=%d\n",
           cast_tick, arrow_tick, stop_tick, restore_tick, frozen_ticks,
           saw_restored_motion, saw_post_stop_death);
    return 0;
}
