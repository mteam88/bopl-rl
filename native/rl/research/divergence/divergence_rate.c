/* How long does the port agree with the oracle, and about what?
 *
 *   divergence_rate <oracle.so> <q32.so> <f64.so>
 *
 * Every existing differential in research/rust_ab checks something somebody
 * thought to check. This one checks nothing in particular: it plays whole
 * games from randomly drawn seeds, maps and loadouts, drives the .NET oracle
 * and BOTH Rust builds in lockstep off one action stream, and records the
 * first tick at which each build stops agreeing with the oracle. The product
 * is a survival curve of ticks-to-first-divergence, not a bug list.
 *
 * Three simulations per game, one action stream:
 *
 *   oracle   sim/Native NativeAOT BoplSimNative.so, the decompiled C# compiled
 *            outside Unity.
 *   q32      cargo build --features q32: the port's bit-exact i64 Q32.32
 *            arithmetic. oracle-vs-q32 isolates genuine LOGIC divergence.
 *   f64      cargo build --release: the IEEE double `Fix` the library policies
 *            actually train in (native/src/fixed.rs:111). oracle-vs-f64 is the
 *            divergence that actually reaches training.
 *
 * The gap between the two curves separates arithmetic drift from logic bugs.
 * Both arms are measured on the SAME games with the SAME action stream, so the
 * two curves are paired and their difference is a within-game quantity.
 *
 * Two field lenses, per replay_ab.c and bopl_puffer.h:316:
 *   all      every field the ABI publishes: "the simulators disagree".
 *   visible  only rows a policy can read (ACTIVE && !DESTROYED), compared as
 *            the compacted sequences the observation writer actually encodes:
 *            "the policy can tell they disagree".
 *
 * Action sources (DR_CELL):
 *   random          uniform aim + button bits re-drawn every 4 ticks, the
 *                   decision cadence the trainer uses (writer_bench.c:135).
 *                   Broad and unrealistic: it reaches state a played game
 *                   never visits.
 *   bot:<kind>:<lv> the scripted opponents (puffer/bopl_bots.h), driven
 *                   through the environment's exact latch and forced-release
 *                   contract. Realistic play, which concentrates in a very
 *                   different part of the state space.
 *
 * The bots read the q32 Rust state and fork the q32 handle (only the Rust sim
 * exports bopl_rl_fork, so the search kind cannot be driven off the oracle),
 * and the actions they produce are sent identically to all three backends. A
 * divergence therefore never changes the action stream: every game's three
 * runs see byte-identical input from tick 1 to the end.
 *
 * Storage: nothing per-tick is ever written. One JSON line per game on stdout,
 * ~600 bytes, carrying the first divergence of each (arm, lens) and the
 * coverage evidence for that game.
 *
 * Environment:
 *   DR_CELL      action source, above (default random)
 *   DR_GAMES     games to play (default 100)
 *   DR_FIRST     index of the first game, so shards do not overlap (default 0)
 *   DR_SEED      base seed for the configuration draw (default 0x5eed)
 *   DR_TICKS     max_episode_ticks, and the comparison horizon (default 3600)
 *   DR_PLAYERS   players per game (default 2)
 *   DR_STOCKS    config.stocks (default 0, the library default)
 *   DR_TOL       numeric tolerance in raw Q32.32 units (default 1 = 2.33e-10)
 *   DR_POOL      train | all | fixed | <comma-separated ids>
 *                train = 1..38 less Time Stop (12) and Revival (15), the
 *                        exclusions evaluation_protocol.DEFAULT_POOL_EXCLUSIONS
 *                        applies because ABILITY_PORT_GAPS.md documents them as
 *                        unfinished; this is the pool training actually uses.
 *                all   = 1..38, the documented gaps included.
 *                fixed = Dash/Bow/Platform {2,4,14}, which is the only loadout
 *                        the scripted bots are written for (bopl_bots.h:143).
 *   DR_MAPS      all | <comma-separated ids>. Games are assigned maps by
 *                stratification (game index modulo the map count), not by
 *                drawing, so map coverage is exactly uniform.
 *   DR_VISIBLE_ALIGNED  1 = index-align the visible lens like replay_ab.c
 *                       instead of compacting it.
 *   DR_PROGRESS  print a progress line to stderr every N games (default 0)
 */
#include "bopl_rl.h"
#include "divergence_compare.h"
#include "bopl_bots.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { DR_MAX_PLAYERS = 6, DR_ARMS = 2, DR_LENSES = 2, DR_ABILITIES = 39 };

/* ---- configuration draw ------------------------------------------------ */

static uint32_t dr_rand(uint64_t *s) {   /* splitmix64, low 32 bits */
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)((z ^ (z >> 31)) >> 32);
}

/* bopl_puffer.h:270-281, so a bot's seed is drawn the same way the trainer
 * draws it and the play distribution is the trainer's. */
static uint32_t dr_bot_seed(uint32_t episode_seed, int player) {
    uint32_t x = (episode_seed ^ 0x9E3779B9u) * 2654435761u + (uint32_t)(player + 1) * 0x85EBCA6Bu;
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    return x;
}

/* ---- per-game record --------------------------------------------------- */

typedef struct {
    DrCombo combo[DR_ARMS][DR_LENSES];
    unsigned ticks;            /* episode ticks actually compared (excluding tick 0) */
    unsigned compared;         /* comparisons performed, tick 0 included */
    int end_reason;            /* 0 limit, 1 terminal, 2 backends ended apart */
    unsigned end_tick[3];      /* per backend terminal tick, 0 = never */
    uint32_t map, seed;
    uint32_t loadout[DR_MAX_PLAYERS][3];
    uint32_t activations[DR_MAX_PLAYERS][3];   /* oracle telemetry: slot presses that took */
    /* The loadout as the game actually resolved it. Ability 1 (Random) is
     * replaced at spawn, so the configured id attributes a divergence to the
     * wrong ability; this is read off the oracle's slimes. */
    uint32_t resolved[DR_MAX_PLAYERS][3];
    uint64_t ability_entities;                 /* bit i = an entity sourced by ability i existed */
    unsigned max_entities;
} DrGame;

/* ---- the environment's button latch (bopl_puffer.h:777) ---------------- */

typedef struct { int hold_left; uint16_t held; int hold_run[4]; } DrLatch;

static uint16_t dr_latch(DrLatch *l, uint16_t buttons, int hold) {
    if (l->hold_left > 0) l->hold_left--;
    else { l->held = buttons; l->hold_left = BOPL_HOLD_LENGTHS[hold] - 1; }
    uint16_t applied = l->held;
    for (int bit = 0; bit < 4; bit++)
        l->hold_run[bit] = (applied & (1 << bit)) ? l->hold_run[bit] + 1 : 0;
    return applied;
}

/* ---- action sources ---------------------------------------------------- */

typedef struct {
    int bot_kind, bot_level;       /* 0 = random actions */
    BoplBot bots[DR_MAX_PLAYERS];
    DrLatch latch[DR_MAX_PLAYERS];
    uint64_t rng;
} DrDriver;

/* One decision for every player, already latched. `pilot` is the state the
 * decision is taken from and the handle the search kind forks; it is the q32
 * Rust build for every arm, so the action stream does not depend on which
 * candidate is being scored. */
static void dr_decide(DrDriver *d, BoplRl *pilot, const BoplRlState *state, int players,
                      BoplRlAction *actions) {
    for (int i = 0; i < players; i++) {
        uint16_t buttons; uint8_t aim; int hold;
        if (d->bot_kind) {
            uint8_t out[3];
            d->bots[i].world = pilot;                     /* bopl_puffer.h:1163 */
            bopl_bot_decide(&d->bots[i], state, i, d->latch[i].hold_left, d->latch[i].held,
                            d->latch[i].hold_run, out);
            aim = out[0]; buttons = out[1]; hold = out[2];
        } else {
            /* writer_bench.c:135's shape: an aim byte, the jump and three
             * ability bits, and some WASD. Hold index 0, so every decision is
             * fresh and the latch never pins: random play should not be given
             * the bots' charge discipline. */
            uint32_t r = dr_rand(&d->rng);
            aim = (uint8_t)(1 + (r >> 16) % 255);
            buttons = (uint16_t)((r & 15) | ((r >> 4) & (64 | 128 | 256 | 512)));
            hold = 0;
        }
        actions[i].angle = aim;
        actions[i].reserved = 0;
        actions[i].buttons = dr_latch(&d->latch[i], buttons, hold);
    }
    if (d->bot_kind)
        for (int i = 0; i < players; i++)
            memcpy(d->bots[i].applied, actions, sizeof(BoplRlAction) * DR_MAX_PLAYERS);
}

/* ---- one game ---------------------------------------------------------- */

/* DR_DUMP=<from>[:<to>[:<entity>]] prints both backends' entity rows over a
 * tick window, for pulling one divergence apart by hand. */
static unsigned dr_dump_from = 0, dr_dump_to = 0;
static int dr_dump_entity = -1;

static const char *DR_ARM_NAME[DR_ARMS] = {"q32", "f64"};
static const char *DR_LENS_NAME[DR_LENSES] = {"all", "visible"};

static int dr_complete(const DrGame *g) {
    for (int a = 0; a < DR_ARMS; a++)
        for (int l = 0; l < DR_LENSES; l++)
            if (!g->combo[a][l].any.found || !g->combo[a][l].disc.found ||
                !g->combo[a][l].any_ex.found || !g->combo[a][l].disc_ex.found) return 0;
    return 1;
}

/* world[0] oracle, world[1] q32, world[2] f64. */
static int dr_play(BoplRl *world[3], const BoplRlConfig *config, DrDriver *driver,
                   int64_t tolerance, unsigned horizon, DrGame *g) {
    int players = (int)config->players;
    for (int i = 0; i < 3; i++)
        if (bopl_rl_reset(world[i], config) != BOPL_RL_OK) {
            fprintf(stderr, "reset backend %d map %u seed %u: %s\n", i, config->map, config->seed,
                    bopl_rl_error(world[i]));
            return 1;
        }
    if (driver->bot_kind) {
        for (int i = 0; i < players; i++)
            bopl_bot_reset(&driver->bots[i], driver->bot_kind, driver->bot_level,
                           dr_bot_seed(config->seed, i));
        memset(driver->latch, 0, sizeof driver->latch);
    } else {
        memset(driver->latch, 0, sizeof driver->latch);
    }

    const BoplRlState *state[3];
    for (int i = 0; i < 3; i++) state[i] = bopl_rl_state(world[i]);

    for (int a = 0; a < DR_ARMS; a++)
        for (int l = 0; l < DR_LENSES; l++)
            dr_compare(state[0], state[a + 1], 0, tolerance, l, &g->combo[a][l]);
    g->compared = 1;
    /* slime.owner is a player ID (bopl_puffer.h:333 passes players[i].id),
     * not an index into players[]. */
    for (uint32_t i = 0; i < state[0]->header.slime_count; i++) {
        const BoplRlSlime *slime = &state[0]->slimes[i];
        for (uint32_t q = 0; q < state[0]->header.player_count && q < DR_MAX_PLAYERS; q++)
            if (state[0]->players[q].id == slime->owner)
                for (int s = 0; s < 3; s++) g->resolved[q][s] = slime->abilities[s];
    }

    BoplRlAction actions[DR_MAX_PLAYERS];
    memset(actions, 0, sizeof actions);
    int ended[3] = {0, 0, 0};
    unsigned tick = 0;
    for (tick = 1; tick <= horizon; tick++) {
        if ((tick - 1) % 4 == 0)
            dr_decide(driver, world[1], state[1], players, actions);
        int live = 0;
        for (int i = 0; i < 3; i++) {
            if (ended[i]) continue;
            int status = bopl_rl_step(world[i], actions, (size_t)players, 1);
            if (status == BOPL_RL_ENDED) { ended[i] = 1; if (!g->end_tick[i]) g->end_tick[i] = tick; continue; }
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "step backend %d map %u seed %u tick %u: %s\n", i, config->map,
                        config->seed, tick, bopl_rl_error(world[i]));
                return 1;
            }
            state[i] = bopl_rl_state(world[i]);
            if (state[i]->header.flags & BOPL_RL_TERMINAL) { ended[i] = 1; g->end_tick[i] = tick; }
            else live++;
        }
        g->ticks = tick;
        if (dr_dump_from && tick >= dr_dump_from && tick <= dr_dump_to) {
            for (int i = 0; i < 3; i++) {
                const BoplRlState *v = state[i];
                for (uint32_t e = 0; e < v->header.entity_count; e++) {
                    if (dr_dump_entity >= 0 && (uint32_t)dr_dump_entity != e) continue;
                    const BoplRlEntity *r = &v->entities[e];
                    fprintf(stderr, "DUMP t%-5u b%d e%-3u id%-4u kind%u lay%-2u own%u fl%-6u ab%-3u "
                            "x%14.9f y%14.9f rx%14.9f ry%14.9f ux%14.9f uy%14.9f rot%12.7f rad%10.6f\n",
                            tick, i, e, r->id, r->kind, r->layer, r->owner, r->flags, r->reserved[0],
                            dr_units(r->x), dr_units(r->y), dr_units(r->right_x), dr_units(r->right_y),
                            dr_units(r->up_x), dr_units(r->up_y), dr_units(r->rotation), dr_units(r->radius));
                }
            }
        }
        /* Coverage evidence, taken from the oracle: which equipped abilities
         * actually fired, and which abilities put an entity on the field. An
         * ability that is merely equipped proves nothing ran. */
        const BoplRlState *o = state[0];
        for (int p = 0; p < players && p < (int)o->header.player_count; p++)
            for (int s = 0; s < 3; s++)
                if (o->telemetry[p].ability_activations[s] > g->activations[p][s])
                    g->activations[p][s] = o->telemetry[p].ability_activations[s];
        if (o->header.entity_count > g->max_entities) g->max_entities = o->header.entity_count;
        for (uint32_t e = 0; e < o->header.entity_count; e++) {
            uint32_t source = o->entities[e].reserved[0];
            if (source < DR_ABILITIES) g->ability_entities |= 1ull << source;
        }
        if (!dr_complete(g)) {
            g->compared++;
            for (int a = 0; a < DR_ARMS; a++)
                for (int l = 0; l < DR_LENSES; l++) {
                    DrCombo *c = &g->combo[a][l];
                    if (c->any.found && c->disc.found && c->any_ex.found && c->disc_ex.found) continue;
                    dr_compare(state[0], state[a + 1], tick, tolerance, l, c);
                }
        }
        if (!live) break;
    }
    g->end_reason = (g->end_tick[0] || g->end_tick[1] || g->end_tick[2]) ? 1 : 0;
    if (g->end_tick[0] != g->end_tick[1] || g->end_tick[0] != g->end_tick[2]) g->end_reason = 2;
    return 0;
}

/* ---- reporting --------------------------------------------------------- */

static void dr_print_hit(const char *label, const DrHit *h) {
    if (!h->found) { printf(",\"%s\":null", label); return; }
    char field[64];
    dr_generalize(h->field, field, sizeof field);
    printf(",\"%s\":{\"tick\":%u,\"field\":\"%s\",\"discrete\":%d,\"ability\":%d,\"kind\":%d",
           label, h->tick, field, h->discrete, h->source_ability, h->entity_kind);
    if (h->discrete) printf(",\"left\":%" PRId64 ",\"right\":%" PRId64, h->left, h->right);
    else printf(",\"gap\":%.6g", dr_units(h->left > h->right ? h->left - h->right : h->right - h->left));
    printf(",\"detail\":\"%s\"}", h->context);
}

static const char *dr_env(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return value && *value ? value : fallback;
}
static long dr_envl(const char *name, long fallback) {
    const char *value = getenv(name);
    return value && *value ? strtol(value, NULL, 10) : fallback;
}
static double dr_seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <oracle.so> <q32.so> <f64.so>\n", argv[0]);
        return 2;
    }
    const char *cell = dr_env("DR_CELL", "random");
    unsigned games = (unsigned)dr_envl("DR_GAMES", 100);
    unsigned first = (unsigned)dr_envl("DR_FIRST", 0);
    uint32_t base_seed = (uint32_t)dr_envl("DR_SEED", 0x5eed);
    unsigned horizon = (unsigned)dr_envl("DR_TICKS", 3600);
    int players = (int)dr_envl("DR_PLAYERS", 2);
    uint32_t stocks = (uint32_t)dr_envl("DR_STOCKS", 0);
    int64_t tolerance = dr_envl("DR_TOL", 1);
    unsigned progress = (unsigned)dr_envl("DR_PROGRESS", 0);
    dr_visible_aligned = (int)dr_envl("DR_VISIBLE_ALIGNED", 0);
    /* Default: the inactive-prefab hierarchy label. A row with no ACTIVE bit
     * is never encoded into an observation, and the two ports number a few
     * such rows' hierarchy slots one apart (Roll and Shrink Ray prefabs at
     * reset); it is the same class as the ENABLED-bit exclusion replay_ab.c
     * already carries. Both curves are reported, so nothing is hidden. */
    dr_ignore_list = dr_env("DR_IGNORE", "entity[].hierarchy");
    if (getenv("DR_DUMP")) {
        dr_dump_to = 0;
        if (sscanf(getenv("DR_DUMP"), "%u:%u:%d", &dr_dump_from, &dr_dump_to, &dr_dump_entity) < 2)
            dr_dump_to = dr_dump_from;
    }
    if (players < 1 || players > DR_MAX_PLAYERS) { fprintf(stderr, "DR_PLAYERS 1..6\n"); return 2; }

    /* Ability pool. */
    int pool[DR_ABILITIES], pool_n = 0, pool_fixed = 0;
    const char *pool_spec = dr_env("DR_POOL", "train");
    if (!strcmp(pool_spec, "train")) {
        for (int a = 1; a < DR_ABILITIES; a++) if (a != 12 && a != 15) pool[pool_n++] = a;
    } else if (!strcmp(pool_spec, "all")) {
        for (int a = 1; a < DR_ABILITIES; a++) pool[pool_n++] = a;
    } else if (!strcmp(pool_spec, "fixed")) {
        pool_fixed = 1; pool[pool_n++] = 2; pool[pool_n++] = 4; pool[pool_n++] = 14;
    } else {
        char buffer[512]; snprintf(buffer, sizeof buffer, "%s", pool_spec);
        for (char *at = strtok(buffer, ","); at && pool_n < DR_ABILITIES; at = strtok(NULL, ","))
            pool[pool_n++] = atoi(at);
    }
    if (!pool_n) { fprintf(stderr, "empty DR_POOL\n"); return 2; }

    /* Action source. */
    DrDriver driver;
    memset(&driver, 0, sizeof driver);
    char cell_label[64];
    snprintf(cell_label, sizeof cell_label, "%s", cell);
    if (!strncmp(cell, "bot:", 4)) {
        if (sscanf(cell + 4, "%d:%d", &driver.bot_kind, &driver.bot_level) != 2) {
            fprintf(stderr, "DR_CELL bot:<kind>:<level>\n"); return 2;
        }
        if (driver.bot_kind < 1 || driver.bot_kind > BOPL_BOT_KINDS ||
            driver.bot_level < 0 || driver.bot_level >= BOPL_BOT_LEVELS) {
            fprintf(stderr, "bot kind 1..%d level 0..2\n", BOPL_BOT_KINDS); return 2;
        }
    } else if (strcmp(cell, "random")) { fprintf(stderr, "DR_CELL random|bot:k:l\n"); return 2; }

    char error[512] = {0};
    BoplRl *world[3];
    for (int i = 0; i < 3; i++) {
        char log[64]; snprintf(log, sizeof log, "/tmp/divergence-rate-%d.log", (int)getpid());
        world[i] = bopl_rl_open(argv[i + 1], "/dev/null", log, 300000, error, sizeof error);
        if (!world[i]) { fprintf(stderr, "open %s: %s\n", argv[i + 1], error); return 2; }
    }
    uint32_t map_count = bopl_rl_map_count(world[0]);
    for (int i = 1; i < 3; i++)
        if (bopl_rl_map_count(world[i]) != map_count || bopl_rl_ability_count(world[i]) != DR_ABILITIES) {
            fprintf(stderr, "backend %d catalog disagrees\n", i); return 2;
        }
    /* The search kind forks; prove the pilot can before playing thousands of
     * games into an abort() (bopl_bots_v2.h:237). */
    if (driver.bot_kind == BOPL_BOT_SEARCH) {
        /* A handle with no world yet cannot be forked, so reset one first:
         * probing a fresh handle reports "not forkable" for every backend. */
        BoplRlConfig probe_config;
        bopl_rl_default_config(&probe_config, (uint32_t)players);
        probe_config.max_episode_ticks = 600;
        if (bopl_rl_reset(world[1], &probe_config) != BOPL_RL_OK) {
            fprintf(stderr, "fork probe reset: %s\n", bopl_rl_error(world[1])); return 2;
        }
        BoplRl *probe = bopl_rl_fork(world[1], NULL);
        if (!probe) { fprintf(stderr, "the search bot needs a forkable pilot; %s is not\n", argv[2]); return 2; }
        bopl_rl_close(probe);
    }

    /* Map selection: stratified, so coverage is exact rather than sampled. */
    unsigned maps[512], map_n = 0;
    const char *map_spec = dr_env("DR_MAPS", "all");
    if (!strcmp(map_spec, "all")) { for (uint32_t m = 0; m < map_count && map_n < 512; m++) maps[map_n++] = m; }
    else {
        char buffer[2048]; snprintf(buffer, sizeof buffer, "%s", map_spec);
        for (char *at = strtok(buffer, ","); at && map_n < 512; at = strtok(NULL, ",")) maps[map_n++] = (unsigned)atoi(at);
    }
    if (!map_n) { fprintf(stderr, "empty DR_MAPS\n"); return 2; }

    printf("{\"record\":\"run\",\"cell\":\"%s\",\"oracle\":\"%s\",\"q32\":\"%s\",\"f64\":\"%s\","
           "\"oracle_hash\":\"%s\",\"rust_hash\":\"%s\",\"games\":%u,\"first\":%u,\"seed\":%u,"
           "\"ticks\":%u,\"players\":%d,\"stocks\":%u,\"tol\":%" PRId64 ",\"pool\":\"%s\",\"pool_n\":%d,"
           "\"maps\":%u,\"map_count\":%u,\"visible_aligned\":%d,\"ignore\":\"%s\"}\n",
           cell_label, argv[1], argv[2], argv[3], bopl_rl_source_hash(world[0]),
           bopl_rl_source_hash(world[1]), games, first, base_seed, horizon, players, stocks,
           tolerance, pool_spec, pool_n, map_n, map_count, dr_visible_aligned, dr_ignore_list);
    fflush(stdout);

    unsigned failed = 0;
    double started = dr_seconds();
    unsigned long long ticks_total = 0, compared_total = 0;
    for (unsigned n = 0; n < games; n++) {
        unsigned index = first + n;
        uint64_t draw = ((uint64_t)base_seed << 32) ^ (uint64_t)index * 0x9E3779B97F4A7C15ull;
        dr_rand(&draw);
        BoplRlConfig config;
        bopl_rl_default_config(&config, (uint32_t)players);
        config.map = maps[index % map_n];
        config.seed = dr_rand(&draw);
        config.stocks = stocks;
        config.slots = 3;
        config.warmup_ticks = 240;
        config.max_episode_ticks = horizon;
        for (int p = 0; p < players; p++)
            for (int s = 0; s < 3; s++)
                config.player[p].abilities[s] = pool_fixed ? (uint32_t)pool[s]
                                                           : (uint32_t)pool[dr_rand(&draw) % (uint32_t)pool_n];
        driver.rng = draw ^ 0xD1B54A32D192ED03ull;

        DrGame game;
        memset(&game, 0, sizeof game);
        game.map = config.map;
        game.seed = config.seed;
        for (int p = 0; p < players; p++)
            for (int s = 0; s < 3; s++) game.loadout[p][s] = config.player[p].abilities[s];
        if (dr_play(world, &config, &driver, tolerance, horizon, &game)) { failed++; continue; }
        ticks_total += game.ticks;
        compared_total += game.compared;

        printf("{\"record\":\"game\",\"cell\":\"%s\",\"i\":%u,\"map\":%u,\"seed\":%u,\"ticks\":%u,"
               "\"compared\":%u,\"end\":%d,\"end_ticks\":[%u,%u,%u],\"max_entities\":%u,"
               "\"ability_entities\":%" PRIu64 ",\"loadout\":[",
               cell_label, index, game.map, game.seed, game.ticks, game.compared, game.end_reason,
               game.end_tick[0], game.end_tick[1], game.end_tick[2], game.max_entities,
               game.ability_entities);
        for (int p = 0; p < players; p++)
            printf("%s[%u,%u,%u]", p ? "," : "", game.loadout[p][0], game.loadout[p][1], game.loadout[p][2]);
        printf("],\"resolved\":[");
        for (int p = 0; p < players; p++)
            printf("%s[%u,%u,%u]", p ? "," : "", game.resolved[p][0], game.resolved[p][1], game.resolved[p][2]);
        printf("],\"activations\":[");
        for (int p = 0; p < players; p++)
            printf("%s[%u,%u,%u]", p ? "," : "", game.activations[p][0], game.activations[p][1],
                   game.activations[p][2]);
        printf("]");
        for (int a = 0; a < DR_ARMS; a++)
            for (int l = 0; l < DR_LENSES; l++) {
                char label[32];
                snprintf(label, sizeof label, "%s_%s", DR_ARM_NAME[a], DR_LENS_NAME[l]);
                char any[40], disc[40];
                snprintf(any, sizeof any, "%s_any", label);
                snprintf(disc, sizeof disc, "%s_disc", label);
                dr_print_hit(any, &game.combo[a][l].any);
                dr_print_hit(disc, &game.combo[a][l].disc);
                snprintf(any, sizeof any, "%s_anyx", label);
                snprintf(disc, sizeof disc, "%s_discx", label);
                dr_print_hit(any, &game.combo[a][l].any_ex);
                dr_print_hit(disc, &game.combo[a][l].disc_ex);
                printf(",\"%s_maxgap\":%.6g", label, dr_units(game.combo[a][l].max_gap));
            }
        printf("}\n");
        if (progress && (n + 1) % progress == 0) {
            double elapsed = dr_seconds() - started;
            fprintf(stderr, "%s: %u/%u games, %.1f games/s, %llu ticks\n", cell_label, n + 1, games,
                    (n + 1) / elapsed, ticks_total);
        }
        if ((n & 63) == 0) fflush(stdout);
    }
    if (driver.bot_kind)
        for (int i = 0; i < DR_MAX_PLAYERS; i++) bopl_rl_close(driver.bots[i].fork);
    for (int i = 0; i < 3; i++) bopl_rl_close(world[i]);
    double elapsed = dr_seconds() - started;
    printf("{\"record\":\"done\",\"cell\":\"%s\",\"games\":%u,\"failed\":%u,\"ticks\":%llu,"
           "\"compared\":%llu,\"seconds\":%.1f}\n",
           cell_label, games - failed, failed, ticks_total, compared_total, elapsed);
    return failed ? 1 : 0;
}
