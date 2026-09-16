/* Time Stop freezes an older linked Portal pair, then an ordinary player
 * traverses after the pair would have expired on the unfrozen clock. */
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
           a->hierarchy == b->hierarchy && distance(a->x, b->x) <= 1 && distance(a->y, b->y) <= 1 &&
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

static const BoplRlSlime *slime_for(const BoplRlState *state, unsigned owner) {
    for (unsigned i = 0; i < state->header.slime_count; i++)
        if (state->slimes[i].owner == owner) return &state->slimes[i];
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1],
            "/proc/self/cwd/native/fixtures/movement.rep",
            "/proc/self/cwd/native/target/time_stop_portal_ab.log",
            120000, error, sizeof error);
        if (!world[backend]) return 1;
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7; config.map = 0; config.stocks = 10;
    config.warmup_ticks = 240; config.max_episode_ticks = 3000;
    for (unsigned player = 0; player < 2; player++) {
        config.player[player].keyboard = 1;
        config.player[player].abilities[0] = 2;
        config.player[player].abilities[1] = 4;
        config.player[player].abilities[2] = 14;
    }
    config.player[0].abilities[0] = 38;
    config.player[0].abilities[1] = 12;
    if (reset(world[0], &config, "reference") ||
        reset(world[1], &config, "rust")) return 1;

    unsigned stopped = 0, restored = 0, transit = 0;
    const BoplRlSlime *initial_opponent = slime_for(bopl_rl_state(world[0]), 2);
    if (!initial_opponent) return 1;
    int64_t previous_x = initial_opponent->x;
    for (unsigned tick = 1; tick <= 2200; tick++) {
        BoplRlAction action[2] = {{0}};
        if (tick == 181 || tick == 270) {
            action[0].buttons = 2;
            action[0].angle = tick == 270 ? 192 : 1;
        }
        if (tick >= 300 && tick < 900) {
            action[0].buttons |= 4;
            action[0].angle = 1;
        }
        /* The unfrozen 25-second portal clock would expire at about tick
         * 1770. Start the ordinary player afterward to prove the 690 stopped
         * passes did not age either linked mouth. */
        if (tick >= 1800 && tick < 2200) action[1].buttons = 512;
        for (int backend = 0; backend < 2; backend++) {
            if (bopl_rl_step(world[backend], action, 2, 1) != BOPL_RL_OK) {
                fprintf(stderr, "step %u backend %d: %s\n", tick, backend,
                        bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), tick)) return 1;
        if (reference->header.player_time_scale == 0) {
            if (!stopped) stopped = tick;
        } else if (stopped && !restored) {
            restored = tick;
        }
        const BoplRlSlime *opponent = slime_for(reference, 2);
        if (!opponent) return 1;
        if (restored && !transit &&
            distance(opponent->x, previous_x) > INT64_C(42949672960))
            transit = tick;
        previous_x = opponent->x;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    if (stopped != 857 || restored != 1547 || transit != 1811) {
        fprintf(stderr, "incomplete stop=%u restore=%u transit=%u\n",
                stopped, restored, transit);
        return 1;
    }
    printf("TIME_STOP_PORTAL_GOLDEN_OK ticks=2200 stop=%u restore=%u "
           "transit=%u frozen=%u\n", stopped, restored, transit,
           restored - stopped);
    return 0;
}
