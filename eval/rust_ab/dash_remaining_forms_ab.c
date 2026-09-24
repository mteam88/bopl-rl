/* Dash ray through eighteen remaining nonlethal ability forms. Their active
 * PlayerCollision owns the callback, so the ordinary-slime push must not run.
 * Meteor inherits its measured 8,192-raw-Q32 grounded form pose bound. */
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
           a->hierarchy == b->hierarchy && distance(a->x, b->x) <= 8192 &&
           distance(a->y, b->y) <= 8192 && distance(a->vx, b->vx) <= 8192 &&
           distance(a->vy, b->vy) <= 8192 &&
           distance(a->right_x, b->right_x) <= 1 &&
           distance(a->right_y, b->right_y) <= 1 &&
           distance(a->up_x, b->up_x) <= 1 &&
           distance(a->up_y, b->up_y) <= 1 &&
           distance(a->radius, b->radius) <= 8192 &&
           distance(a->rotation, b->rotation) <= 8192 &&
           distance(a->angular_velocity, b->angular_velocity) <= 8192 &&
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
        int player_position_diff = distance(ap.x, bp.x) > 8192 || distance(ap.y, bp.y) > 8192;
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
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        BoplRlSlime as = a->slimes[i], bs = b->slimes[i];
        int slime_vector_diff = distance(as.x, bs.x) > 8192 || distance(as.y, bs.y) > 8192 ||
            distance(as.vx, bs.vx) > 8192 || distance(as.vy, bs.vy) > 8192;
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
                                      "/tmp/dash_remaining_forms_ab.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    static const uint32_t abilities[] = {5, 7, 11, 12, 14, 15, 18, 19, 20, 21, 22, 23, 25, 26, 28, 31, 32, 38};
    static const char *names[] = {
        "Engine", "Gust", "Spike", "TimeStop", "Platform", "Revival", "BlackHole",
        "Invisibility", "Meteor", "Throw", "Push", "Tesla", "Teleport",
        "Drill", "Beam", "Rewind", "WebSwing", "PortalGun"
    };
    for (unsigned scenario = 0; scenario < sizeof abilities / sizeof abilities[0]; scenario++) {
        BoplRlConfig config;
        bopl_rl_default_config(&config, 5);
        config.seed = 123;
        config.map = 0;
        config.stocks = 5;
        config.warmup_ticks = 240;
        config.max_episode_ticks = 1000;
        for (unsigned player = 0; player < 5; player++) {
            config.player[player].keyboard = 1;
            config.player[player].abilities[0] = 2;
            config.player[player].abilities[1] = 4;
            config.player[player].abilities[2] = 14;
        }
        config.player[4].abilities[0] = abilities[scenario];
        config.player[4].abilities[1] = 2;
        if (abilities[scenario] == 14) config.player[4].abilities[2] = 4;
        if (reset(world[0], &config, "reference") ||
            reset(world[1], &config, "rust"))
            return 1;
        if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

        unsigned dash_cast = 0;
        int saw_form = 0, form_at_dash = 0;
        int64_t separation_at_cast = 0;
        int64_t target_velocity_after_cast = 0, rust_target_velocity_after_cast = 0;
        for (unsigned tick = 1; tick <= 180; tick++) {
            const BoplRlState *before = bopl_rl_state(world[0]);
            BoplRlAction action[5] = {{0}};
            int64_t separation = distance(before->players[0].x, before->players[4].x);
            if (tick >= 149) {
                action[4].buttons = 2;
                /* Neutral aim keeps Push's supporting platform still, isolating
                 * the form's PlayerCollision callback from platform chaos. */
                action[4].angle = abilities[scenario] == 22 ? 0 : 1;
            }
            if (!dash_cast && tick > 148 && separation > INT64_C(3000000000)) {
                action[0].buttons = 128;
            } else if (!dash_cast && tick > 148 &&
                       before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) {
                dash_cast = tick;
                separation_at_cast = separation;
                action[0].buttons = 2;
                action[0].angle = 128;
            }
            for (int backend = 0; backend < 2; backend++) {
                int status = bopl_rl_step(world[backend], action, 5, 1);
                if (status != BOPL_RL_OK) {
                    fprintf(stderr, "tick %u backend %d: %s\n", tick, backend,
                            bopl_rl_error(world[backend]));
                    return 1;
                }
            }
            const BoplRlState *reference = bopl_rl_state(world[0]);
            if (compare(reference, bopl_rl_state(world[1]), tick)) return 1;
            int active_form = 0;
            for (unsigned entity = 0; entity < reference->header.entity_count; entity++)
                if (reference->entities[entity].layer == 8 &&
                    reference->entities[entity].owner == 5 &&
                    (reference->entities[entity].flags & 1)) {
                    active_form = 1;
                    break;
                }
            if (active_form) saw_form = 1;
            if (tick == dash_cast) {
                form_at_dash = active_form;
                target_velocity_after_cast = reference->slimes[4].vx;
                rust_target_velocity_after_cast = bopl_rl_state(world[1])->slimes[4].vx;
            }
        }
        if (!dash_cast || !saw_form || !form_at_dash ||
            separation_at_cast >= INT64_C(3000000000) ||
            target_velocity_after_cast != rust_target_velocity_after_cast) {
            fprintf(stderr, "%s unexpected dash=%u form=%d/%d separation=%" PRId64
                            " target-vx=%" PRId64 "/%" PRId64 "\n",
                    names[scenario], dash_cast, saw_form, form_at_dash,
                    separation_at_cast, target_velocity_after_cast,
                    rust_target_velocity_after_cast);
            return 1;
        }
        printf("DASH_REMAINING_FORM_OK ability=%u name=%s ticks=180 dash=%u separation=%" PRId64
               " target-vx=%" PRId64 "\n",
               abilities[scenario], names[scenario], dash_cast,
               separation_at_cast, target_velocity_after_cast);
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    return 0;
}
