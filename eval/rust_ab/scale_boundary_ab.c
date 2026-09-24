/* Collider-rescale vs entry-pass boundary check.
 *
 * `rollout::player_tick` tests the scene boundary on an ability *entry* pass
 * by running `move_only` on a COPY of the body and testing that position
 * (the reference's DestroyIfOutsideSceneBounds runs after PlayerPhysics.Move,
 * so the pre-Move position is the wrong one to test). The copy is only a
 * faithful stand-in if it carries the same collider the real Move will use.
 *
 * `move_only` does not call `sync_collider`, so the copy runs its landing
 * rays against whatever hurtbox basis the previous pass left behind. Every
 * held ability slot that rescales the collider on entry is therefore a
 * candidate for clone and real Move disagreeing: Dash swaps to its own
 * extents (`world.rs` sync_collider, Dash branch), and Growth/Shrink drive
 * the slime's extents along a ScaleChanger curve for the whole animation.
 *
 * The divergence needs three things at once, which is why it needs a probe
 * rather than an argument: the body must be GROUNDED (airborne, clone and
 * real agree -- there is no surface for Move to advance along), it must be
 * within reach of the kill plane so the boundary test is live, and the
 * collider must change on that same pass.
 *
 * Construction: sudden death sinks the authored platforms (AnimateVelocity
 * drives homePosition down past SceneBounds.WaterHeight = -11.3), carrying a
 * standing player below the plane while still grounded. Player 0 presses its
 * Dash slot at a swept offset around the crossing, so some case in the sweep
 * lands the press on the exact pass the body is grounded below the plane.
 * A second family holds the button for several passes instead of one, to
 * catch an entry deferred by the ability's own readiness gate.
 *
 * Both backends receive identical input from the same reset and every public
 * field is compared exactly -- header, players, slimes and entities -- so a
 * single raw Q32 unit fails. Growth/Shrink of the caster itself can only come
 * from an opponent's ray and is not constructed here; see README.
 *
 * cc -O2 -I sim/abi -o ../target/release/scale_boundary_ab \
 *   eval/rust_ab/scale_boundary_ab.c sim/abi/bopl_rl.c -ldl -lm
 *
 * Usage: scale_boundary_ab <BoplSimNative.so> <libbopl_sim.so> */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Map 0's SceneBounds.WaterHeight. It is per-map: map 2's water trigger sits
 * at (2.04, -18.95) with half-extents (134, 7), so its plane is -11.95. Each
 * subject therefore carries its own. */
#define PLAYERS 2
/* Sudden death begins at 120 s (7200 ticks); the platforms need a while
 * after that to carry a standing player under the plane. */
#define TICKS 9000

static double f(int64_t v) { return (double)v / 4294967296.0; }

struct sample {
    unsigned tick, entity_count, slime_count, stocks, deaths, flags;
    int64_t x, y;
};

/* Compare every public field of both states for this tick. Returns 0 when
 * they are identical, and describes the first difference otherwise. */
static int diff_state(const BoplRlState *a, const BoplRlState *b,
                      unsigned tick, const char *case_name) {
    const BoplRlHeader ha = a->header, hb = b->header;
    if (memcmp(&ha, &hb, sizeof ha) != 0) {
        printf("%s tick %u header differs: entities %u/%u slimes %u/%u "
               "flags %u/%u sudden %u/%u\n", case_name, tick,
               ha.entity_count, hb.entity_count,
               ha.slime_count, hb.slime_count, ha.flags, hb.flags,
               a->header.sudden_death, b->header.sudden_death);
        return 1;
    }
    for (unsigned i = 0; i < PLAYERS; i++) {
        const BoplRlPlayer *p = &a->players[i], *q = &b->players[i];
        if (memcmp(p, q, sizeof *p) != 0) {
            printf("%s tick %u player %u differs: stocks %u/%u deaths %u/%u "
                   "cause %u/%u flags %u/%u x %.12f/%.12f y %.12f/%.12f\n",
                   case_name, tick, i, p->stocks, q->stocks, p->deaths,
                   q->deaths, p->cause_of_death, q->cause_of_death,
                   p->flags, q->flags, f(p->x), f(q->x), f(p->y), f(q->y));
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        const BoplRlSlime *p = &a->slimes[i], *q = &b->slimes[i];
        if (memcmp(p, q, sizeof *p) != 0) {
            printf("%s tick %u slime %u differs: id %u/%u flags %u/%u "
                   "state %u/%u x %.12f/%.12f y %.12f/%.12f scale %.12f/%.12f\n",
                   case_name, tick, i, p->id, q->id, p->flags, q->flags,
                   p->state, q->state, f(p->x), f(q->x), f(p->y), f(q->y),
                   f(p->scale), f(q->scale));
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.entity_count; i++) {
        const BoplRlEntity *p = &a->entities[i], *q = &b->entities[i];
        if (memcmp(p, q, sizeof *p) != 0) {
            printf("%s tick %u entity %u differs: id %u/%u kind %u/%u "
                   "flags %u/%u x %.12f/%.12f y %.12f/%.12f "
                   "right_x %" PRId64 "/%" PRId64 " up_y %" PRId64 "/%" PRId64 "\n",
                   case_name, tick, i, p->id, q->id, p->kind, q->kind,
                   p->flags, q->flags, f(p->x), f(q->x), f(p->y), f(q->y),
                   p->right_x, q->right_x, p->up_y, q->up_y);
            return 1;
        }
    }
    return 0;
}

/* One case: press the Dash slot for `hold` passes starting `offset` ticks
 * after player 0's body first goes under the kill plane while grounded.
 * Returns 1 on a divergence. */
static int run_case(BoplRl *net, BoplRl *rust, BoplRlConfig *config,
                    int offset, unsigned hold, unsigned map, double plane,
                    struct sample *seen) {
    char case_name[96];
    snprintf(case_name, sizeof case_name, "map %u offset %+d hold %u",
             map, offset, hold);
    config->map = map;
    if (bopl_rl_reset(net, config) != BOPL_RL_OK ||
        bopl_rl_reset(rust, config) != BOPL_RL_OK) {
        printf("%s: reset failed\n", case_name);
        return 1;
    }
    unsigned cross = 0, pressed_until = 0;
    memset(seen, 0, sizeof *seen);
    for (unsigned tick = 1; tick <= TICKS; tick++) {
        const BoplRlState *sn = bopl_rl_state(net);
        /* Drive both backends from the reference's view so the script is
         * identical even once they disagree. */
        if (!cross && sn->header.slime_count > 0) {
            for (unsigned i = 0; i < sn->header.slime_count; i++) {
                if (sn->slimes[i].owner != 1) continue;
                if (f(sn->slimes[i].y) < plane) {
                    cross = tick;
                    long start = (long)tick + offset;
                    pressed_until = (unsigned)(start < 1 ? 1 : start) + hold;
                    seen->tick = tick;
                    seen->y = sn->slimes[i].y;
                }
            }
        }
        BoplRlAction action[PLAYERS];
        memset(action, 0, sizeof action);
        if (cross) {
            long start = (long)cross + offset;
            if ((long)tick >= start && tick < pressed_until) {
                action[0].buttons = 2; /* slot 0 = Dash */
            }
        }
        if (bopl_rl_step(net, action, PLAYERS, 1) != BOPL_RL_OK ||
            bopl_rl_step(rust, action, PLAYERS, 1) != BOPL_RL_OK) {
            break; /* episode ended in both; nothing further to compare */
        }
        if (diff_state(bopl_rl_state(net), bopl_rl_state(rust), tick, case_name))
            return 1;
        if (cross && tick > pressed_until + 120) break;
    }
    if (!cross) {
        printf("%s: player never went under the plane while alive\n", case_name);
        return 2;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <reference.so> <rust.so>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *net = bopl_rl_open(argv[1], "/proc/self/cwd/native/fixtures/movement.rep",
                               "/proc/self/cwd/native/target/scale_boundary_ab.log",
                               600000, error, sizeof error);
    BoplRl *rust = bopl_rl_open(argv[2], "/proc/self/cwd/native/fixtures/movement.rep",
                                "/proc/self/cwd/native/target/scale_boundary_ab.log",
                                600000, error, sizeof error);
    if (!net || !rust) { fprintf(stderr, "open: %s\n", error); return 1; }

    BoplRlConfig config;
    bopl_rl_default_config(&config, PLAYERS);
    config.seed = 7;
    config.stocks = 10;
    config.warmup_ticks = 240;
    config.max_episode_ticks = TICKS + 600;
    for (unsigned p = 0; p < PLAYERS; p++) {
        config.player[p].abilities[0] = 2;  /* Dash: rescales the collider */
        config.player[p].abilities[1] = 8;  /* Growth ray */
        config.player[p].abilities[2] = 17; /* Shrink ray */
    }

    const int offsets[] = {-3, -2, -1, 0, 1, 2};
    const unsigned holds[] = {1, 6};
    /* One entry per map that actually carries a living, grounded body under
     * its kill plane. Map 2 was here and was removed: sudden death drops its
     * platforms fast enough that the player drowns on the same pass it first
     * dips below -11.95, so no published tick ever shows it alive under the
     * plane and all twelve of its cases were `unreached`. Twelve vacuous
     * cases reading as twenty-four passing ones is exactly the laundering
     * this file exists to prevent; a map only belongs here once it reaches. */
    const struct { unsigned map; double plane; } subjects[] = {
        {0, -11.3},
    };
    int failures = 0, cases = 0, unreached = 0;
    struct sample seen;
    for (unsigned m = 0; m < sizeof subjects / sizeof *subjects; m++) {
        int map_cases = 0, map_unreached = 0, map_failures = 0;
        for (unsigned h = 0; h < sizeof holds / sizeof *holds; h++) {
            for (unsigned o = 0; o < sizeof offsets / sizeof *offsets; o++) {
                int rc = run_case(net, rust, &config, offsets[o], holds[h],
                                  subjects[m].map, subjects[m].plane, &seen);
                map_cases++;
                if (rc == 1) map_failures++;
                else if (rc == 2) map_unreached++;
            }
        }
        printf("SCALE_BOUNDARY_MAP map=%u plane=%.2f cases=%d reached=%d "
               "unreached=%d failures=%d\n", subjects[m].map,
               subjects[m].plane, map_cases, map_cases - map_unreached,
               map_unreached, map_failures);
        /* An unreached case tested nothing. A map where none of them reach
         * is a broken construction, not a pass, and has to say so in its own
         * exit status - otherwise the next reader counts it as coverage. */
        if (map_unreached == map_cases) {
            printf("SCALE_BOUNDARY_VACUOUS map=%u -- no case put a living "
                   "body under the plane; this map tests nothing and must be "
                   "reconstructed or removed\n", subjects[m].map);
            map_failures += map_cases;
        }
        cases += map_cases;
        unreached += map_unreached;
        failures += map_failures;
    }
    bopl_rl_close(net);
    bopl_rl_close(rust);
    if (failures) {
        printf("SCALE_BOUNDARY_DIFF cases=%d reached=%d failures=%d "
               "unreached=%d\n", cases, cases - unreached, failures, unreached);
        return 1;
    }
    if (unreached) {
        /* Partial coverage still hid a divergence once; name the shortfall
         * rather than folding it into a pass count. */
        printf("SCALE_BOUNDARY_SHORT cases=%d reached=%d unreached=%d -- the "
               "reached cases agree, but the unreached ones tested nothing\n",
               cases, cases - unreached, unreached);
        return 1;
    }
    printf("SCALE_BOUNDARY_OK cases=%d reached=%d -- clone and real Move "
           "agree across the sweep\n", cases, cases);
    return 0;
}
