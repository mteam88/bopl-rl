/* Broad Freeze differential: active-form cancellation followed by a second
 * Freeze while the same victim is already pinned. */
/* Freeze ray cancels and freezes an active Bow form. */
#include "bopl_rl.h"
#include <inttypes.h>
#include <math.h>
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
        int position_diff = distance(ap.x, bp.x) > 64 || distance(ap.y, bp.y) > 64;
        ap.x = ap.y = bp.x = bp.y = 0;
        if (position_diff || memcmp(&ap, &bp, sizeof ap) != 0 ||
            memcmp(&a->telemetry[i], &b->telemetry[i],
                   sizeof a->telemetry[i]) != 0) {
            fprintf(stderr,
                    "tick %u player %u differs: flags %u/%u alive %u/%u "
                    "position (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") activations %u/%u\n",
                    tick, i, a->players[i].flags, b->players[i].flags,
                    a->players[i].alive_bodies, b->players[i].alive_bodies,
                    a->players[i].x, a->players[i].y, b->players[i].x,
                    b->players[i].y, a->telemetry[i].ability_activations[0],
                    b->telemetry[i].ability_activations[0]);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        BoplRlSlime as = a->slimes[i], bs = b->slimes[i];
        int vector_diff = distance(as.x, bs.x) > 64 || distance(as.y, bs.y) > 64 ||
            distance(as.vx, bs.vx) > 64 || distance(as.vy, bs.vy) > 64;
        as.x = as.y = as.vx = as.vy = bs.x = bs.y = bs.vx = bs.vy = 0;
        if (vector_diff || memcmp(&as, &bs, sizeof as) != 0) {
            fprintf(stderr,
                    "tick %u slime %u differs: p (%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") v (%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") state %u/%u flags %u/%u\n",
                    tick, i, a->slimes[i].x, a->slimes[i].y,
                    b->slimes[i].x, b->slimes[i].y, a->slimes[i].vx,
                    a->slimes[i].vy, b->slimes[i].vx, b->slimes[i].vy,
                    a->slimes[i].state, b->slimes[i].state,
                    a->slimes[i].ability_flags[0], b->slimes[i].ability_flags[0]);
            fprintf(stderr, "  rotation=%" PRId64 "/%" PRId64 " scale=%" PRId64
                    "/%" PRId64 "\n", a->slimes[i].rotation,
                    b->slimes[i].rotation, a->slimes[i].scale, b->slimes[i].scale);
            const unsigned char *aa = (const unsigned char *)&a->slimes[i];
            const unsigned char *bb = (const unsigned char *)&b->slimes[i];
            for (unsigned z = 0; z < sizeof a->slimes[i]; z++)
                if (aa[z] != bb[z]) fprintf(stderr, "  byte %u %u/%u\n", z, aa[z], bb[z]);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.entity_count; i++) {
        if (!entity_equal(&a->entities[i], &b->entities[i])) {
            fprintf(stderr,
                    "tick %u entity %u differs: id %u/%u kind %u/%u layer %u/%u "
                    "flags %u/%u p (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") v (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") hierarchy %u/%u\n",
                    tick, i, a->entities[i].id, b->entities[i].id,
                    a->entities[i].kind, b->entities[i].kind,
                    a->entities[i].layer, b->entities[i].layer,
                    a->entities[i].flags, b->entities[i].flags,
                    a->entities[i].x, a->entities[i].y,
                    b->entities[i].x, b->entities[i].y,
                    a->entities[i].vx, a->entities[i].vy,
                    b->entities[i].vx, b->entities[i].vy,
                    a->entities[i].hierarchy, b->entities[i].hierarchy);
            for (unsigned z = 0; z < a->header.entity_count; z++) {
                fprintf(stderr, "  e%u id=%u/%u layer=%u/%u flags=%u/%u p=(%" PRId64
                        ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n", z,
                        a->entities[z].id, b->entities[z].id, a->entities[z].layer,
                        b->entities[z].layer, a->entities[z].flags,
                        b->entities[z].flags, a->entities[z].x, a->entities[z].y,
                        b->entities[z].x, b->entities[z].y);
            }
            return 1;
        }
    }
    return 0;
}

static unsigned aim_byte(int64_t sx, int64_t sy, int64_t tx, int64_t ty) {
    double angle = atan2((double)(ty - sy), (double)(tx - sx));
    if (angle < 0.0) angle += 6.28318530717958647692;
    return (unsigned)floor(angle / 6.28318530717958647692 * 255.0) + 1;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/freeze_broad_ab.log", 120000,
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
    config.player[0].abilities[1] = 36;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    if (reset(world[0], &config, "reference") || reset(world[1], &config, "rust")) return 1;

    unsigned cast[2] = {0}, held[2] = {0}, frozen = 0, thawed = 0;
    for (unsigned tick = 1; tick <= 620; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (tick >= 225 && tick <= 500) {
            action[1].buttons = 4;
            action[1].angle = 128;
        }
        if (!cast[0] && tick > 180 && before->slimes[0].timers[0] > before->slimes[0].cooldowns[0])
            cast[0] = tick;
        if (!cast[1] && tick > 340 && before->slimes[0].timers[1] > before->slimes[0].cooldowns[1])
            cast[1] = tick;
        action[0].angle = aim_byte(before->players[0].x, before->players[0].y,
                                   before->players[1].x, before->players[1].y);
        if (cast[0] && held[0]++ < 60) action[0].buttons = 2;
        else if (cast[1] && held[1]++ < 60) action[0].buttons = 4;
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "tick %u backend %d: %s\n", tick, backend,
                        bopl_rl_error(world[backend])); return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), tick)) return 1;
        if (!frozen && reference->slimes[1].state == 10) frozen = tick;
        if (frozen && !thawed && tick > frozen && reference->slimes[1].state != 10) thawed = tick;
    }
    const BoplRlState *final = bopl_rl_state(world[0]);
    unsigned final_state = final->slimes[1].state;
    for (int i = 0; i < 2; i++) bopl_rl_close(world[i]);
    if (!cast[0] || !cast[1] || !frozen || !thawed || thawed - frozen < 150) {
        fprintf(stderr, "incomplete casts=%u/%u frozen=%u thawed=%u final_state=%u\n",
                cast[0], cast[1], frozen, thawed, final_state);
        return 1;
    }
    printf("FREEZE_BROAD_GOLDEN_OK ticks=620 casts=%u/%u frozen=%u thawed=%u span=%u\n",
           cast[0], cast[1], frozen, thawed, thawed - frozen);
    return 0;
}
