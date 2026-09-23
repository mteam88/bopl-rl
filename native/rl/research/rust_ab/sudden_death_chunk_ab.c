/* Sudden-death endgame differential.
 *
 * Sudden death (GameSessionHandler.cs:502, 120 s = level tick 7200) drives the
 * authored level bodies down through AnimateVelocity's homePosition until the
 * last player drowns. `map-sweep-3000` stops 4,200 ticks before that starts,
 * so until this probe every contact in the collapsing endgame -- and the round
 * resolution itself -- was outside every gate.
 *
 * This probe drives a plain two-player game with NO input at all -- no button
 * is ever pressed -- and compares every public field of both backends each
 * tick, from reset through the tick that ends the round. Any difference is a
 * real divergence: under --features q32 both backends run identical Q32.32
 * arithmetic, so the pass criterion is exact equality and there is no
 * tolerance anywhere in this file.
 *
 * What it caught: map 2's AntiLockPlatform orbiter (entity 18, hierarchy 162).
 * `WinGame` calls `ActivateSuddenDeath(activate: false)` before anything else
 * (GameSessionHandler.cs:655), so on the pass that resolves the round the
 * global `SuddenDeathInProgress` is already down and AntiLockPlatform.cs:35
 * runs once more, kicking the orbiter toward its next waypoint. The port held
 * one flag for both that session state and AnimateVelocity's per-component
 * `inSuddenDeath` latch, so it never applied that force.
 *
 * Set SD_CHUNK_DUMP=<first>[:<last>] to print the entity rows for that tick
 * range instead of stopping at the first difference, and SD_CHUNK_NEAR=<id>
 * to restrict the dump to the neighbourhood of one entity (SD_CHUNK_SLACK
 * widens that neighbourhood beyond the two bodies' radii, default 2.0).
 *
 * cc -O2 -I. -o ../target/release/sudden_death_chunk_ab \
 *   research/rust_ab/sudden_death_chunk_ab.c bopl_rl.c -ldl -lm
 *
 * Usage: sudden_death_chunk_ab <BoplSimNative.so> <libbopl_sim.so> [map] [ticks]
 * With no [map] it sweeps maps 0-4. */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PLAYERS 2
#define DEFAULT_TICKS 9000

static double f(int64_t v) { return (double)v / 4294967296.0; }

static unsigned dump_from, dump_to, near_id;
static double slack = 2.0;
static int dumping;

static void describe(const char *label, const BoplRlEntity *e) {
    /* res0/res1 are compared by the memcmp below, so they have to be
     * printed: a row whose only difference is a retained `reserved[0]`
     * otherwise prints two identical lines and reads as a probe bug. */
    printf("  %-4s id=%u kind=%u layer=%u owner=%u flags=%u hier=%u "
           "res=(%u,%u) clk=%.9f "
           "p=(%.9f,%.9f) v=(%.9f,%.9f) rot=%.9f av=%.9f "
           "right=(%.9f,%.9f) up=(%.9f,%.9f) r=%.9f invm=%.9f\n",
           label, e->id, e->kind, e->layer, e->owner, e->flags, e->hierarchy,
           e->reserved[0], e->reserved[1], f(e->clock_rate),
           f(e->x), f(e->y), f(e->vx), f(e->vy), f(e->rotation),
           f(e->angular_velocity), f(e->right_x), f(e->right_y),
           f(e->up_x), f(e->up_y), f(e->radius), f(e->inverse_mass));
}

/* 0 clean, 1 divergence, 2 harness failure. */
static int run_map(BoplRl *world[2], unsigned map, unsigned ticks) {
    const char *labels[2] = {"net", "rust"};
    BoplRlConfig config;
    bopl_rl_default_config(&config, PLAYERS);
    config.seed = 7;
    config.map = map;
    config.stocks = 10;
    config.warmup_ticks = 240;
    config.max_episode_ticks = ticks + 600;
    for (unsigned p = 0; p < PLAYERS; p++) {
        config.player[p].abilities[0] = 2;
        config.player[p].abilities[1] = 8;
        config.player[p].abilities[2] = 17;
    }
    for (int b = 0; b < 2; b++) {
        if (bopl_rl_reset(world[b], &config) != BOPL_RL_OK) {
            fprintf(stderr, "reset %s map %u: %s\n", labels[b], map,
                    bopl_rl_error(world[b]));
            return 2;
        }
    }

    BoplRlAction action[PLAYERS];
    memset(action, 0, sizeof action); /* no input, ever */

    for (unsigned tick = 1; tick <= ticks; tick++) {
        for (int b = 0; b < 2; b++) {
            if (bopl_rl_step(world[b], action, PLAYERS, 1) != BOPL_RL_OK) {
                fprintf(stderr, "step %s map %u tick %u: %s\n", labels[b], map,
                        tick, bopl_rl_error(world[b]));
                return 2;
            }
        }
        const BoplRlState *a = bopl_rl_state(world[0]);
        const BoplRlState *b = bopl_rl_state(world[1]);

        if (dumping && tick >= dump_from && tick <= dump_to) {
            printf("tick %u (level %u) entities %u/%u flags %u/%u "
                   "stocks %u,%u/%u,%u alive %u,%u/%u,%u\n", tick,
                   a->header.tick, a->header.entity_count,
                   b->header.entity_count, a->header.flags, b->header.flags,
                   a->players[0].stocks, a->players[1].stocks,
                   b->players[0].stocks, b->players[1].stocks,
                   a->players[0].alive_bodies, a->players[1].alive_bodies,
                   b->players[0].alive_bodies, b->players[1].alive_bodies);
            const BoplRlEntity *anchor = NULL;
            if (near_id) {
                for (unsigned i = 0; i < a->header.entity_count; i++)
                    if (a->entities[i].id == near_id) anchor = &a->entities[i];
            }
            for (unsigned i = 0; i < a->header.entity_count; i++) {
                const BoplRlEntity *e = &a->entities[i];
                if (anchor) {
                    /* Bodies here are rounded rects: a 0.01 x 0.01 chunk with
                     * radius 6.03 reaches far beyond its extents, so the
                     * neighbourhood must allow for both radii. */
                    double dx = f(e->x) - f(anchor->x), dy = f(e->y) - f(anchor->y);
                    double reach = f(anchor->radius) + f(e->radius) + slack;
                    if (dx * dx + dy * dy > reach * reach) continue;
                }
                describe("net", e);
                if (i < b->header.entity_count) describe("rust", &b->entities[i]);
            }
            if ((a->header.flags & BOPL_RL_TERMINAL) ||
                (b->header.flags & BOPL_RL_TERMINAL)) {
                printf("episode ended at tick %u (level %u)\n", tick,
                       a->header.tick);
                return 0;
            }
            continue;
        }

        if (a->header.entity_count != b->header.entity_count ||
            a->header.slime_count != b->header.slime_count ||
            a->header.player_count != b->header.player_count) {
            printf("SD_CHUNK_DIFF map=%u tick %u: counts e %u/%u s %u/%u "
                   "p %u/%u\n", map, tick, a->header.entity_count,
                   b->header.entity_count, a->header.slime_count,
                   b->header.slime_count, a->header.player_count,
                   b->header.player_count);
            return 1;
        }
        for (unsigned i = 0; i < a->header.entity_count; i++) {
            BoplRlEntity x = a->entities[i], y = b->entities[i];
            if (memcmp(&x, &y, sizeof x) != 0) {
                printf("SD_CHUNK_DIFF map=%u tick %u (level %u) entity "
                       "index %u\n", map, tick, a->header.tick, i);
                describe("net", &a->entities[i]);
                describe("rust", &b->entities[i]);
                return 1;
            }
        }
        for (unsigned i = 0; i < a->header.slime_count; i++) {
            if (memcmp(&a->slimes[i], &b->slimes[i], sizeof a->slimes[i])) {
                printf("SD_CHUNK_DIFF map=%u tick %u (level %u) slime "
                       "index %u: net p=(%.9f,%.9f) v=(%.9f,%.9f) flags=%u "
                       "state=%u | rust p=(%.9f,%.9f) v=(%.9f,%.9f) flags=%u "
                       "state=%u\n", map, tick, a->header.tick, i,
                       f(a->slimes[i].x), f(a->slimes[i].y),
                       f(a->slimes[i].vx), f(a->slimes[i].vy),
                       a->slimes[i].flags, a->slimes[i].state,
                       f(b->slimes[i].x), f(b->slimes[i].y),
                       f(b->slimes[i].vx), f(b->slimes[i].vy),
                       b->slimes[i].flags, b->slimes[i].state);
                return 1;
            }
        }
        for (unsigned i = 0; i < a->header.player_count; i++) {
            if (memcmp(&a->players[i], &b->players[i], sizeof a->players[i])) {
                printf("SD_CHUNK_DIFF map=%u tick %u (level %u) player %u: "
                       "net p=(%.9f,%.9f) flags=%u stocks=%u alive=%u "
                       "cause=%u | rust p=(%.9f,%.9f) flags=%u stocks=%u "
                       "alive=%u cause=%u\n", map, tick, a->header.tick, i,
                       f(a->players[i].x), f(a->players[i].y),
                       a->players[i].flags, a->players[i].stocks,
                       a->players[i].alive_bodies, a->players[i].cause_of_death,
                       f(b->players[i].x), f(b->players[i].y),
                       b->players[i].flags, b->players[i].stocks,
                       b->players[i].alive_bodies, b->players[i].cause_of_death);
                return 1;
            }
        }
        /* The round ends when the last opponent drowns, well before `ticks`.
         * Stepping past it is a transport error, not a result, so stop here
         * and say which tick ended it - a backend that ends on a different
         * tick is itself a divergence. */
        if ((a->header.flags & BOPL_RL_TERMINAL) ||
            (b->header.flags & BOPL_RL_TERMINAL)) {
            if ((a->header.flags & BOPL_RL_TERMINAL) !=
                (b->header.flags & BOPL_RL_TERMINAL)) {
                printf("SD_CHUNK_DIFF map=%u tick %u: terminal %u/%u\n", map,
                       tick, a->header.flags, b->header.flags);
                return 1;
            }
            printf("SD_CHUNK_MAP map=%u ended=%u (level %u)\n", map, tick,
                   a->header.tick);
            return 0;
        }
    }
    /* Never reaching sudden death means the probe compared the early game
     * only, which `map-sweep-3000` already covers. Say so rather than
     * reporting it as endgame coverage. */
    printf("SD_CHUNK_UNFINISHED map=%u ticks=%u -- the round had not resolved "
           "when the probe stopped\n", map, ticks);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s <reference.so> <rust.so> [map] [ticks]\n",
                argv[0]);
        return 2;
    }
    const unsigned ticks = argc > 4 ? (unsigned)strtoul(argv[4], NULL, 10)
                                    : DEFAULT_TICKS;
    unsigned maps[] = {0, 1, 2, 3, 4};
    unsigned map_count = sizeof maps / sizeof *maps;
    if (argc > 3) {
        maps[0] = (unsigned)strtoul(argv[3], NULL, 10);
        map_count = 1;
    }

    const char *dump = getenv("SD_CHUNK_DUMP");
    if (dump) {
        dumping = 1;
        dump_from = (unsigned)strtoul(dump, NULL, 10);
        const char *colon = strchr(dump, ':');
        dump_to = colon ? (unsigned)strtoul(colon + 1, NULL, 10) : dump_from;
    }
    const char *near_text = getenv("SD_CHUNK_NEAR");
    near_id = near_text ? (unsigned)strtoul(near_text, NULL, 10) : 0;
    const char *slack_text = getenv("SD_CHUNK_SLACK");
    if (slack_text) slack = atof(slack_text);

    char error[512] = {0};
    const char *labels[2] = {"net", "rust"};
    BoplRl *world[2];
    for (int b = 0; b < 2; b++) {
        world[b] = bopl_rl_open(argv[1 + b],
                                "/proc/self/cwd/native/fixtures/movement.rep",
                                "/proc/self/cwd/native/target/sudden_death_chunk_ab.log",
                                600000, error, sizeof error);
        if (!world[b]) {
            fprintf(stderr, "open %s: %s\n", labels[b], error);
            return 1;
        }
    }

    int failures = 0;
    for (unsigned m = 0; m < map_count; m++) {
        int rc = run_map(world, maps[m], ticks);
        if (rc == 2) { for (int b = 0; b < 2; b++) bopl_rl_close(world[b]); return 2; }
        if (rc) failures++;
        if (dumping) break;
    }
    for (int b = 0; b < 2; b++) bopl_rl_close(world[b]);
    if (dumping) return 0;
    if (failures) {
        printf("SD_CHUNK_DIFF maps=%u failures=%d\n", map_count, failures);
        return 1;
    }
    printf("SD_CHUNK_OK maps=%u -- every field exact from reset through the "
           "tick that ends the round\n", map_count);
    return 0;
}
