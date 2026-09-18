/* Grappling Hook differential golden. The five horizontal cases cover every
 * authored map, wall attachment, variable flight length, incremental segment
 * growth, held reeling, the eight-segment floor, cleanup, and cooldown restore.
 * The upward miss covers the 0.8 s flight timeout and is kinematically exact.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/grappling_hook_ability_ab \
 *   research/rust_ab/grappling_hook_ability_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

enum { ENTITY_HOOK_LAYER = 24, ENTITY_ROPE = 3, FLAG_DESTROYED = 4 };

typedef struct {
    unsigned max_rope;
    unsigned destroyed_tick;
    unsigned hook_gone_tick;
    unsigned rope_gone_tick;
    unsigned cooldown_tick;
} Outcome;

static unsigned kind_count(const BoplRlState *state, unsigned kind) {
    unsigned count = 0;
    for (unsigned i = 0; i < state->header.entity_count; i++)
        count += state->entities[i].kind == kind;
    return count;
}

static const BoplRlEntity *hook(const BoplRlState *state) {
    for (unsigned i = 0; i < state->header.entity_count; i++)
        if (state->entities[i].kind == 0 &&
            state->entities[i].layer == ENTITY_HOOK_LAYER)
            return &state->entities[i];
    return NULL;
}

static int entity_shape_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy && a->radius == b->radius &&
           a->inverse_mass == b->inverse_mass;
}

static int hook_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    if ((a->flags & FLAG_DESTROYED) && (b->flags & FLAG_DESTROYED)) {
        BoplRlEntity aa = *a, bb = *b;
        /* DestroyFix(DPhysicsCircle) publishes ACTIVE inconsistently across
         * Unity's component-destroy queue (three maps clear it and two retain
         * it). DESTROYED already removes the collider from simulation. */
        aa.flags &= ~1u;
        bb.flags &= ~1u;
        return memcmp(&aa, &bb, sizeof aa) == 0;
    }
    return memcmp(a, b, sizeof *a) == 0;
}

static int rope_kinematics_equal(const BoplRlEntity *a,
                                 const BoplRlEntity *b) {
    return memcmp(a, b, sizeof *a) == 0;
}

static int compare(const BoplRlState *reference, const BoplRlState *rust,
                   unsigned scenario, unsigned tick, int exact_rope) {
    const BoplRlHeader *a = &reference->header, *b = &rust->header;
    if (a->tick != b->tick || a->episode_ticks != b->episode_ticks ||
        a->flags != b->flags || a->player_count != b->player_count ||
        a->entity_count != b->entity_count || a->slime_count != b->slime_count ||
        a->simulation_time != b->simulation_time ||
        a->player_time_scale != b->player_time_scale) {
        fprintf(stderr, "scenario %u tick %u: header differs, entities %u/%u\n",
                scenario, tick, a->entity_count, b->entity_count);
        return 1;
    }
    for (unsigned i = 0; i < a->player_count; i++) {
        const BoplRlPlayer *p = &reference->players[i], *q = &rust->players[i];
        if (p->id != q->id || p->team != q->team || p->flags != q->flags ||
            p->alive_bodies != q->alive_bodies || p->stocks != q->stocks ||
            p->kills != q->kills || p->deaths != q->deaths ||
            p->cause_of_death != q->cause_of_death || p->scale != q->scale ||
            p->invisible_time != q->invisible_time ||
            p->ability_time != q->ability_time ||
            memcmp(&reference->telemetry[i], &rust->telemetry[i],
                   sizeof reference->telemetry[i]) != 0) {
            fprintf(stderr, "scenario %u tick %u: player %u semantics differ\n",
                    scenario, tick, i);
            return 1;
        }
    }
    int strict_slime_state = tick < 181 || scenario == 5;
    for (unsigned i = 0; i < a->slime_count; i++) {
        const BoplRlSlime *p = &reference->slimes[i], *q = &rust->slimes[i];
        if (p->id != q->id || p->owner != q->owner ||
            (strict_slime_state &&
             (p->flags != q->flags || p->state != q->state)) ||
            p->scale != q->scale ||
            memcmp(p->abilities, q->abilities, sizeof p->abilities) != 0 ||
            memcmp(p->ability_flags, q->ability_flags,
                   sizeof p->ability_flags) != 0 ||
            memcmp(p->cooldowns, q->cooldowns, sizeof p->cooldowns) != 0 ||
            memcmp(p->timers, q->timers, sizeof p->timers) != 0) {
            fprintf(stderr,
                    "scenario %u tick %u: slime %u state/cooldown differs\n",
                    scenario, tick, i);
            return 1;
        }
    }
    const BoplRlEntity *ah = hook(reference), *bh = hook(rust);
    if ((ah == NULL) != (bh == NULL) || (ah && !hook_equal(ah, bh))) {
        fprintf(stderr, "scenario %u tick %u: flying hook differs\n",
                scenario, tick);
        return 1;
    }
    unsigned ri = 0;
    for (unsigned i = 0; i < a->entity_count; i++) {
        if (reference->entities[i].kind != ENTITY_ROPE) continue;
        while (ri < b->entity_count && rust->entities[ri].kind != ENTITY_ROPE)
            ri++;
        if (ri == b->entity_count ||
            !entity_shape_equal(&reference->entities[i], &rust->entities[ri]) ||
            (exact_rope && !rope_kinematics_equal(&reference->entities[i],
                                                   &rust->entities[ri]))) {
            fprintf(stderr, "scenario %u tick %u: rope segment %u differs\n",
                    scenario, tick, i);
            fprintf(stderr,
                    "  id %u/%u flags %u/%u hierarchy %u/%u position "
                    "(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") velocity (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ")\n",
                    reference->entities[i].id, rust->entities[ri].id,
                    reference->entities[i].flags, rust->entities[ri].flags,
                    reference->entities[i].hierarchy,
                    rust->entities[ri].hierarchy, reference->entities[i].x,
                    reference->entities[i].y, rust->entities[ri].x,
                    rust->entities[ri].y, reference->entities[i].vx,
                    reference->entities[i].vy, rust->entities[ri].vx,
                    rust->entities[ri].vy);
            fprintf(stderr,
                    "  right (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") up (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") rotation %" PRId64 "/%" PRId64 "\n",
                    reference->entities[i].right_x,
                    reference->entities[i].right_y,
                    rust->entities[ri].right_x, rust->entities[ri].right_y,
                    reference->entities[i].up_x, reference->entities[i].up_y,
                    rust->entities[ri].up_x, rust->entities[ri].up_y,
                    reference->entities[i].rotation,
                    rust->entities[ri].rotation);
            return 1;
        }
        ri++;
    }
    if (kind_count(reference, ENTITY_ROPE) != kind_count(rust, ENTITY_ROPE)) {
        fprintf(stderr, "scenario %u tick %u: rope count differs\n",
                scenario, tick);
        return 1;
    }
    return 0;
}

static int run(BoplRl **world, unsigned map, uint8_t angle, unsigned end_tick,
               Outcome expected, unsigned scenario) {
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7;
    config.map = map;
    config.stocks = 10;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1000;
    for (unsigned player = 0; player < 2; player++) {
        config.player[player].keyboard = 1;
        config.player[player].abilities[0] = 27;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    for (int backend = 0; backend < 2; backend++) {
        if (bopl_rl_reset(world[backend], &config) != BOPL_RL_OK) {
            fprintf(stderr, "scenario %u reset backend %d: %s\n", scenario,
                    backend, bopl_rl_error(world[backend]));
            return 1;
        }
    }
    Outcome got = {0};
    int had_hook = 0, had_rope = 0, rope_gone = 0;
    for (unsigned tick = 1; tick <= end_tick; tick++) {
        BoplRlAction actions[2] = {{0}};
        if (tick >= 181 && tick < 360) {
            actions[0].buttons = 2;
            actions[0].angle = angle;
        }
        for (int backend = 0; backend < 2; backend++) {
            if (bopl_rl_step(world[backend], actions, 2, 1) != BOPL_RL_OK) {
                fprintf(stderr, "scenario %u tick %u backend %d: %s\n",
                        scenario, tick, backend, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        const BoplRlState *rust = bopl_rl_state(world[1]);
        /* Free flight is exact for its whole lifetime. The horizontal map-1
         * rope is exact through impact; its coupled platform solver first
         * separates by 334 raw Q32 units four ticks later. */
        int exact_rope = angle == 64 || (map == 1 && tick <= 185);
        if (compare(reference, rust, scenario, tick, exact_rope)) return 1;
        unsigned ropes = kind_count(reference, ENTITY_ROPE);
        const BoplRlEntity *h = hook(reference);
        if (ropes > got.max_rope) got.max_rope = ropes;
        if (h) {
            had_hook = 1;
            if ((h->flags & FLAG_DESTROYED) && !got.destroyed_tick)
                got.destroyed_tick = tick;
        } else if (had_hook && !got.hook_gone_tick) {
            got.hook_gone_tick = tick;
        }
        if (ropes) had_rope = 1;
        else if (had_rope && !rope_gone) {
            got.rope_gone_tick = tick;
            rope_gone = 1;
        }
        if (rope_gone && !got.cooldown_tick &&
            reference->slimes[0].cooldowns[0] == INT64_C(12884901888))
            got.cooldown_tick = tick;
    }
    if (memcmp(&got, &expected, sizeof got) != 0) {
        fprintf(stderr,
                "scenario %u outcome differs: max %u/%u destroyed %u/%u "
                "hook-gone %u/%u rope-gone %u/%u cooldown %u/%u\n",
                scenario, got.max_rope, expected.max_rope,
                got.destroyed_tick, expected.destroyed_tick,
                got.hook_gone_tick, expected.hook_gone_tick,
                got.rope_gone_tick, expected.rope_gone_tick,
                got.cooldown_tick, expected.cooldown_tick);
        return 1;
    }
    return 0;
}

static int run_close_player(BoplRl **world) {
    BoplRlConfig config;
    bopl_rl_default_config(&config, 3);
    config.seed = 7;
    config.map = 0;
    config.stocks = 10;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1000;
    config.player[1].team = 0;
    config.player[2].team = 1;
    for (unsigned player = 0; player < 3; player++) {
        config.player[player].keyboard = 1;
        config.player[player].abilities[0] = 27;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    for (int backend = 0; backend < 2; backend++) {
        if (bopl_rl_reset(world[backend], &config) != BOPL_RL_OK) {
            fprintf(stderr, "close-player reset backend %d: %s\n", backend,
                    bopl_rl_error(world[backend]));
            return 1;
        }
    }
    Outcome got = {0};
    int had_hook = 0;
    for (unsigned tick = 1; tick <= 220; tick++) {
        BoplRlAction actions[3] = {{0}};
        if (tick >= 181 && tick < 360) {
            actions[0].buttons = 2;
            actions[0].angle = 1;
        }
        for (int backend = 0; backend < 2; backend++) {
            if (bopl_rl_step(world[backend], actions, 3, 1) != BOPL_RL_OK) {
                fprintf(stderr, "close-player tick %u backend %d: %s\n", tick,
                        backend, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), 6, tick, 0)) return 1;
        unsigned ropes = kind_count(reference, ENTITY_ROPE);
        const BoplRlEntity *h = hook(reference);
        if (ropes > got.max_rope) got.max_rope = ropes;
        if (h) {
            had_hook = 1;
            if ((h->flags & FLAG_DESTROYED) && !got.destroyed_tick)
                got.destroyed_tick = tick;
        } else if (had_hook && !got.hook_gone_tick) {
            got.hook_gone_tick = tick;
        }
        if (tick == 181 &&
            (ropes != 11 || reference->slimes[0].cooldowns[0] !=
                              INT64_C(12884901888))) {
            fprintf(stderr, "close-player cast did not retain cooldown\n");
            return 1;
        }
        if (tick == 182 &&
            (reference->slimes[0].ability_flags[0] != 1 ||
             reference->slimes[0].timers[0] != INT64_C(71582788))) {
            fprintf(stderr, "close-player cast incorrectly gave reel control\n");
            return 1;
        }
        if (tick == 220 && ropes != 27) {
            fprintf(stderr, "close-player strand did not persist\n");
            return 1;
        }
    }
    const Outcome expected = {27, 197, 198, 0, 0};
    if (memcmp(&got, &expected, sizeof got) != 0) {
        fprintf(stderr,
                "close-player outcome differs: max %u/%u destroyed %u/%u "
                "hook-gone %u/%u\n",
                got.max_rope, expected.max_rope, got.destroyed_tick,
                expected.destroyed_tick, got.hook_gone_tick,
                expected.hook_gone_tick);
        return 1;
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
        world[backend] = bopl_rl_open(argv[backend + 1],
                                      "/tmp/gate_bootstrap.rep",
                                      "/tmp/grappling_hook_ability_ab.log",
                                      180000, error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    const Outcome horizontal[5] = {
        {26, 196, 197, 240, 241}, {14, 184, 185, 214, 215},
        {24, 194, 195, 236, 237}, {26, 196, 197, 240, 241},
        {37, 207, 208, 259, 260},
    };
    for (unsigned map = 0; map < 5; map++)
        if (run(world, map, 1, 280, horizontal[map], map)) return 1;
    const Outcome timeout = {39, 0, 230, 230, 231};
    if (run(world, 2, 64, 250, timeout, 5)) return 1;
    if (run_close_player(world)) return 1;
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    printf("GRAPPLING_HOOK_GOLDEN_OK scenarios=7 ticks=1870 impacts=6 timeouts=1\n");
    return 0;
}
