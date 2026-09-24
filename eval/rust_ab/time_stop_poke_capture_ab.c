/* Time Stop + Poke Ball capture golden: capture immediately before the global
 * stop, pinned invisible ragdoll and item state, then restored expiry/ejection.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/poke_ball_capture_ab \
 *   eval/rust_ab/poke_ball_capture_ab.c sim/abi/bopl_rl.c -ldl -lm
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
           a->hierarchy == b->hierarchy && distance(a->x,b->x) <= 750000000 && distance(a->y,b->y) <= 750000000 &&
           distance(a->vx,b->vx) <= 750000000 && distance(a->vy,b->vy) <= 750000000 &&
           distance(a->right_x, b->right_x) <= 750000000 &&
           distance(a->right_y, b->right_y) <= 750000000 &&
           distance(a->up_x, b->up_x) <= 750000000 &&
           distance(a->up_y, b->up_y) <= 750000000 &&
           distance(a->radius, b->radius) <= 1 &&
           distance(a->rotation,b->rotation) <= 750000000 &&
           distance(a->angular_velocity,b->angular_velocity) <= 750000000 &&
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
        int spatial_player = distance(ap.x,bp.x) <= 750000000 && distance(ap.y,bp.y) <= 750000000;
        ap.x=bp.x; ap.y=bp.y;
        if (!spatial_player || memcmp(&ap, &bp, sizeof ap) != 0 ||
            memcmp(&a->telemetry[i], &b->telemetry[i], sizeof a->telemetry[i]) != 0) {
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
        BoplRlSlime as = a->slimes[i], bs = b->slimes[i];
        int spatial_slime = distance(as.x,bs.x) <= 750000000 && distance(as.y,bs.y) <= 750000000 &&
                            distance(as.vx,bs.vx) <= 750000000 && distance(as.vy,bs.vy) <= 750000000 && distance(as.rotation,bs.rotation) <= 750000000;
        as.x=bs.x; as.y=bs.y; as.vx=bs.vx; as.vy=bs.vy; as.rotation=bs.rotation;
        if (!spatial_slime || memcmp(&as, &bs, sizeof as) != 0) {
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
                                      "/tmp/time_stop_poke_capture_ab.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 3);
    config.seed = 123;
    config.map = 0;
    config.stocks = 5;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 2000;
    config.player[0].abilities[0] = 33;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    config.player[2].abilities[0] = 12;
    config.player[2].abilities[1] = 2;
    config.player[2].abilities[2] = 14;
    if (reset(world[0], &config, "reference") ||
        reset(world[1], &config, "rust"))
        return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    unsigned cast_tick = 0, stop_cast_tick = 0;
    int saw_air_form = 0, saw_item = 0, saw_raycast_exit = 0;
    int saw_owner_binding = 0, saw_capture = 0, saw_release = 0;
    int saw_stopped = 0, saw_restored = 0, saw_protection = 0;
    unsigned capture_tick = 0, release_tick = 0;
    unsigned frozen_ticks = 0;
    uint32_t frozen_ball_id = 0;
    int64_t frozen_ball_x = 0, frozen_ball_y = 0;
    int64_t frozen_ball_vx = 0, frozen_ball_vy = 0, frozen_ball_angular = 0;
    int64_t frozen_victim_x = 0, frozen_victim_y = 0;
    int64_t frozen_victim_timers[3] = {0};
    for (unsigned tick = 1; tick <= 1600; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[3] = {{0}};
        /* Jump well before the 3.5-second cooldown becomes ready so the
         * ability copies a live airborne body. */
        if (tick >= 580 && tick < 665) { action[0].angle = 1; action[1].angle = 129; }
        if (tick == 645) action[1].buttons = 1;
        if (tick == 670) action[0].buttons = 1;
        if (tick >= 680 && tick < 790) action[1].angle = 129;
        if (!stop_cast_tick && tick > 180) {
            for (unsigned i = 0; i < before->header.slime_count; i++)
                if (before->slimes[i].owner == 3 &&
                    before->slimes[i].timers[0] > before->slimes[i].cooldowns[0]) {
                    stop_cast_tick = tick;
                    break;
                }
        }
        if (stop_cast_tick && tick < stop_cast_tick + 600)
            action[2].buttons = 2;
        if (!cast_tick && tick >= 680 &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
            cast_tick = tick;
        }
        if (cast_tick && tick < cast_tick + 7) { action[0].buttons = 2; action[0].angle = 1; }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 3, 1);
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
        if (ability_flags & 8) saw_air_form = 1;
        if (saw_air_form && !(ability_flags & 8)) saw_raycast_exit = 1;
        const BoplRlEntity *ball = NULL;
        for (unsigned i = 0; i < reference->header.entity_count; i++) {
            const BoplRlEntity *entity = &reference->entities[i];
            if (entity->kind != 0 || entity->layer != 10 ||
                !(entity->flags & 1)) continue;
            ball = entity;
            saw_item = 1;
            if (entity->owner == 1) saw_owner_binding = 1;
        }
        if ((reference->players[1].flags & 4) && !saw_capture) { saw_capture = 1; capture_tick = tick; }
        if (saw_capture && !(reference->players[1].flags & 4) && !saw_release) { saw_release = 1; release_tick = tick; }
        const BoplRlSlime *victim = NULL;
        for (unsigned i = 0; i < reference->header.slime_count; i++)
            if (reference->slimes[i].owner == 2) victim = &reference->slimes[i];
        if (reference->header.player_time_scale == 0) {
            if (!saw_stopped) {
                if (!saw_capture || !ball || !victim ||
                    !(reference->players[1].flags & 4)) {
                    fprintf(stderr, "tick %u stop began outside Poke Ball capture\n", tick);
                    return 1;
                }
                saw_stopped = 1;
                frozen_ball_id = ball->id;
                frozen_ball_x = ball->x;
                frozen_ball_y = ball->y;
                frozen_ball_vx = ball->vx;
                frozen_ball_vy = ball->vy;
                frozen_ball_angular = ball->angular_velocity;
                frozen_victim_x = reference->players[1].x;
                frozen_victim_y = reference->players[1].y;
                memcpy(frozen_victim_timers, victim->timers,
                       sizeof frozen_victim_timers);
            } else {
                /* PokeBall.UpdateSim pins the captured slime before its
                 * Player.Position publication catches up on the next pass. */
                if (frozen_ticks == 1) {
                    frozen_victim_x = reference->players[1].x;
                    frozen_victim_y = reference->players[1].y;
                }
                if (!ball || !victim || ball->id != frozen_ball_id ||
                       ball->x != frozen_ball_x || ball->y != frozen_ball_y ||
                       ball->vx != frozen_ball_vx || ball->vy != frozen_ball_vy ||
                       ball->angular_velocity != frozen_ball_angular ||
                       reference->players[1].x != frozen_victim_x ||
                       reference->players[1].y != frozen_victim_y ||
                       !(reference->players[1].flags & 4) ||
                       memcmp(victim->timers, frozen_victim_timers,
                              sizeof frozen_victim_timers) != 0) {
                fprintf(stderr, "tick %u captured Poke Ball state advanced during stop\n",
                        tick);
                fprintf(stderr,
                        " ball id %u/%u p (%" PRId64 ",%" PRId64 ")/(%" PRId64
                        ",%" PRId64 ") v (%" PRId64 ",%" PRId64 ")/(%" PRId64
                        ",%" PRId64 ") a %" PRId64 "/%" PRId64
                        " victim (%" PRId64 ",%" PRId64 ")/(%" PRId64
                        ",%" PRId64 ") flags=%u\n",
                        ball ? ball->id : 0, frozen_ball_id,
                        ball ? ball->x : 0, ball ? ball->y : 0,
                        frozen_ball_x, frozen_ball_y,
                        ball ? ball->vx : 0, ball ? ball->vy : 0,
                        frozen_ball_vx, frozen_ball_vy,
                        ball ? ball->angular_velocity : 0, frozen_ball_angular,
                        reference->players[1].x, reference->players[1].y,
                        frozen_victim_x, frozen_victim_y,
                        reference->players[1].flags);
                return 1;
                }
            }
            frozen_ticks++;
            if ((reference->players[2].flags & 8) &&
                !(reference->players[0].flags & 8) &&
                !(reference->players[1].flags & 8))
                saw_protection = 1;
        } else if (saw_stopped) {
            saw_restored = 1;
        }
    }
    const BoplRlState *final = bopl_rl_state(world[0]);
    if (!cast_tick || !stop_cast_tick || !saw_air_form || !saw_item || !saw_raycast_exit ||
        !saw_owner_binding || !saw_capture || !saw_release ||
        !saw_stopped || !saw_restored || !saw_protection || frozen_ticks != 690 ||
        release_tick - capture_tick != 806 ||
        final->players[0].deaths != 0) {
        fprintf(stderr,
                "incomplete cast=%u air=%d item=%d exit=%d owner=%d "
                "capture=%d release=%d stopped=%d restored=%d protect=%d frozen=%u duration=%u deaths=%u\n",
                cast_tick, saw_air_form, saw_item, saw_raycast_exit,
                saw_owner_binding, saw_capture, saw_release, saw_stopped,
                saw_restored, saw_protection, frozen_ticks, release_tick - capture_tick,
                final->players[0].deaths);
        return 1;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    printf("TIME_STOP_POKE_CAPTURE_GOLDEN_OK ticks=1600 capture=%u release=%u frozen=%u\n",
           capture_tick, release_tick, frozen_ticks);
    return 0;
}
