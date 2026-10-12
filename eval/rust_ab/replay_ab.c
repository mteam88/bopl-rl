/* Real-game replay differential between two bopl_rl.h backends.
 *
 *   replay_ab <reference.so> <candidate.so> <path> [path...]
 *
 * Each path is a BOPLREP2 archive (`<game>.rep`, the lossless per-tick input
 * recording `evaluate.py` saves beside every `<game>.json`) or a directory
 * scanned recursively for `*.rep`. For every archive the tool
 *
 *   1. decodes the recorded setup (seed, map, mode, stocks, per-player colour,
 *      team, keyboard and ability loadout) and the per-tick action stream,
 *   2. resets both backends with exactly that configuration, and
 *   3. steps both one tick at a time with the identical recorded actions,
 *      comparing the complete observable state after every tick.
 *
 * This is the piece `fidelity_gate.c` cannot supply: the inputs are the ones a
 * trained policy actually pressed, so ability combinations and timings that a
 * pseudo-random stream never reaches are exercised.
 *
 * Divergences are reported in two classes, because they mean different things:
 *
 *   DISCRETE - an integer, flag, count, identity, cause or entity-population
 *              field differs. The two simulations have taken different
 *              decisions; everything after is unrelated.
 *   NUMERIC  - a fixed-point quantity differs by more than the tolerance.
 *              Small early drift is the seed a discrete divergence grows from.
 *
 * The first of each is reported per game with the field path, both values and
 * the surrounding context (entity kind/layer/owner, slime owner/abilities).
 *
 * Environment:
 *   REPLAY_AB_WARMUP     warmup ticks used for the recording (default 240)
 *   REPLAY_AB_LIMIT      max_episode_ticks used for the recording (default 7200)
 *   REPLAY_AB_TOL        numeric tolerance in raw Q32.32 units (default 8192,
 *                        1.9e-6 world units); 0 demands bit-exactness
 *   REPLAY_AB_STRICT     1 = drop the inactive-prefab ENABLED-bit exclusion
 *   REPLAY_AB_VISIBLE    1 = compare only what the v4 observation writer can
 *                        read: entity rows that are ACTIVE and not DESTROYED
 *                        (bopl_puffer.h:316), without `id` or `hierarchy`,
 *                        and the visible entity count instead of the raw one.
 *                        This is the ranking that says whether a divergence
 *                        could have changed a trained policy's behaviour.
 *   REPLAY_AB_ALL        1 = print every differing field, not only the first
 *   REPLAY_AB_DUMP       tick: print both backends' entity/slime tables there
 *   REPLAY_AB_IGNORE     comma-separated generalized field names to skip,
 *                        e.g. "entity[].flags,slime[].timers[]" 
 *   REPLAY_AB_MAX_TICKS  stop each game after this many episode ticks (0 = all)
 *   REPLAY_AB_GAMES      stop after this many archives (0 = all)
 *   REPLAY_AB_QUIET      1 = only print diverging games and the summary
 *   REPLAY_AB_FULL       1 = keep stepping after the first divergence, to
 *                        measure how the two runs separate
 *   REPLAY_AB_TRACE      path: write per-tick candidate player positions as
 *                        TSV (tick, player, x, y in world units) for the
 *                        single game being replayed; used by the recorded-JSON
 *                        self check
 *   REPLAY_AB_LOG        path: the libraries' own log (default
 *                        /tmp/replay_ab.log); give concurrent runs their own
 *   REPLAY_AB_CACHE      directory: keep a per-tick hash of every compared
 *                        reference field, keyed by the reference library's
 *                        identity and the game. A game whose hashes are cached
 *                        steps only the candidate; any mismatch reruns it live
 *                        for the report. Bit-exact runs only (TOL=0, no
 *                        VISIBLE/IGNORE/ALL/FULL/DUMP/TRACE).
 *
 * Exit status: 0 when every game agrees, 1 when any diverges, 2 on usage or
 * I/O failure.
 *
 *   cc -O3 -std=c11 -I. -o replay_ab eval/rust_ab/replay_ab.c sim/abi/bopl_rl.c -ldl -lm
 */
#define _DEFAULT_SOURCE /* realpath, st_mtim */
#include "bopl_rl.h"
#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_PLAYERS 6

/* ---- BOPLREP2 decoding (mirrors native/src/rollout/replay.rs) ---- */

typedef struct {
    BoplRlConfig config;
    uint32_t tick_count;
    /* tick_count rows of `players` actions, warmup included. */
    BoplRlAction *actions;
} Replay;

typedef struct {
    const unsigned char *raw;
    size_t length, at;
    int failed;
} Cursor;

static uint8_t take_byte(Cursor *c) {
    if (c->at >= c->length) { c->failed = 1; return 0; }
    return c->raw[c->at++];
}

static uint32_t take_word(Cursor *c, size_t width) {
    uint32_t value = 0;
    for (size_t i = 0; i < width; i++) value |= (uint32_t)take_byte(c) << (8 * i);
    return value;
}

static void replay_free(Replay *replay) { free(replay->actions); replay->actions = NULL; }

static int replay_decode(const char *path, uint32_t warmup, uint32_t limit, Replay *out) {
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "%s: cannot open\n", path); return 1; }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (size < 140) { fclose(file); fprintf(stderr, "%s: too short\n", path); return 1; }
    unsigned char *raw = malloc((size_t)size);
    if (!raw || fread(raw, 1, (size_t)size, file) != (size_t)size) {
        fclose(file); free(raw); fprintf(stderr, "%s: read failed\n", path); return 1;
    }
    fclose(file);
    if (memcmp(raw, "BOPLREP2", 8) != 0) { free(raw); fprintf(stderr, "%s: not a BOPLREP2 archive\n", path); return 1; }

    Cursor cursor = {raw, (size_t)size, 136, 0};
    memset(out, 0, sizeof *out);
    bopl_rl_default_config(&out->config, 2);
    out->config.seed = take_word(&cursor, 4);
    out->config.map = take_byte(&cursor);
    out->config.mode = take_byte(&cursor);
    out->config.stocks = take_word(&cursor, 4);
    uint32_t players = take_byte(&cursor);
    if (players < 1 || players > MAX_PLAYERS) { free(raw); fprintf(stderr, "%s: invalid player count\n", path); return 1; }
    out->config.players = players;
    out->config.warmup_ticks = warmup;
    out->config.max_episode_ticks = limit;
    for (uint32_t p = 0; p < players; p++) {
        take_byte(&cursor); /* id */
        out->config.player[p].color = take_byte(&cursor);
        out->config.player[p].team = take_byte(&cursor);
        out->config.player[p].keyboard = take_byte(&cursor);
        uint32_t slots = take_byte(&cursor);
        if (slots < 1 || slots > 3) { free(raw); fprintf(stderr, "%s: unsupported slot count\n", path); return 1; }
        memset(out->config.player[p].abilities, 0, sizeof out->config.player[p].abilities);
        for (uint32_t s = 0; s < slots; s++) out->config.player[p].abilities[s] = take_word(&cursor, 2);
        out->config.slots = slots;
    }
    out->tick_count = take_word(&cursor, 4);
    if (cursor.failed || out->tick_count > 1000000u) { free(raw); fprintf(stderr, "%s: corrupt archive\n", path); return 1; }
    out->actions = calloc((size_t)out->tick_count * players, sizeof(BoplRlAction));
    if (!out->actions) { free(raw); fprintf(stderr, "%s: out of memory\n", path); return 1; }
    for (uint32_t t = 0; t < out->tick_count; t++) {
        for (uint32_t p = 0; p < players; p++) {
            take_word(&cursor, 4); /* recorded tick index */
            BoplRlAction *action = &out->actions[(size_t)t * players + p];
            action->angle = take_byte(&cursor);
            action->reserved = 0;
            action->buttons = (uint16_t)take_word(&cursor, 2);
            take_byte(&cursor); /* per-input delay, presentation only */
        }
    }
    free(raw);
    if (cursor.failed) { replay_free(out); fprintf(stderr, "%s: truncated archive\n", path); return 1; }
    return 0;
}

/* ---- Divergence reporting (comparator shared with the in-game shadow) ---- */

#include "bopl_rl_compare.h"

typedef BoplDivergence Divergence;
typedef BoplCompare Compare;

static int report_all = 0;              /* print every differing field, not just the first */
static int strict = 0;                  /* 1 = no built-in exclusions at all */
static int visible_only = 0;            /* 1 = compare only what a policy can read */
static const char *ignore_list = NULL;  /* comma separated generalized field names */

static int64_t magnitude(int64_t a, int64_t b) { return bopl_cmp_magnitude(a, b); }
static double units(int64_t raw) { return bopl_cmp_units(raw); }
static void generalize(const char *field, char *out, size_t capacity) { bopl_cmp_generalize(field, out, capacity); }

/* Returns 1 when anything differs at this tick. */
static int compare_state(const BoplRlState *a, const BoplRlState *b, unsigned tick,
                         int64_t tolerance, Compare *out) {
    BoplCompareOptions options = {.tolerance = tolerance, .strict = strict, .visible_only = visible_only,
                                  .report_all = report_all, .ignore = ignore_list};
    return bopl_compare_state(a, b, tick, &options, out);
}

/* Side-by-side entity/slime tables at one tick, for identifying which row a
 * divergence belongs to (REPLAY_AB_DUMP=<tick>). */
static void dump_tables(const BoplRlState *a, const BoplRlState *b, unsigned tick) {
    printf("--- tick %u reference %u entities / candidate %u entities ---\n", tick,
           a->header.entity_count, b->header.entity_count);
    uint32_t count = a->header.entity_count > b->header.entity_count ? a->header.entity_count
                                                                     : b->header.entity_count;
    printf("%4s | %6s %4s %5s %5s %5s %6s %10s %10s | %6s %4s %5s %5s %5s %6s %10s %10s\n",
           "idx", "id", "kind", "layer", "owner", "flags", "abil", "x", "y",
           "id", "kind", "layer", "owner", "flags", "abil", "x", "y");
    for (uint32_t i = 0; i < count; i++) {
        const BoplRlEntity *ea = i < a->header.entity_count ? &a->entities[i] : NULL;
        const BoplRlEntity *eb = i < b->header.entity_count ? &b->entities[i] : NULL;
        printf("%4u |", i);
        if (ea) printf(" %6u %4u %5u %5u %5u %6u %10.4f %10.4f |", ea->id, ea->kind, ea->layer,
                       ea->owner, ea->flags, ea->reserved[0], units(ea->x), units(ea->y));
        else printf(" %58s |", "-");
        if (eb) printf(" %6u %4u %5u %5u %5u %6u %10.4f %10.4f", eb->id, eb->kind, eb->layer,
                       eb->owner, eb->flags, eb->reserved[0], units(eb->x), units(eb->y));
        else printf(" %58s", "-");
        printf("%s\n", (ea && eb && (ea->id != eb->id || ea->reserved[0] != eb->reserved[0] ||
                                      ea->flags != eb->flags)) ? "  <<" : "");
    }
    /* The box/rounded-rect extents and the rotation, for rows whose shape or
     * orientation is what differs rather than the centre. */
    printf("%4s | %10s %10s %10s %10s %10s %10s | %10s %10s %10s %10s %10s %10s\n",
           "idx", "right_x", "right_y", "up_x", "up_y", "radius", "rot",
           "right_x", "right_y", "up_x", "up_y", "radius", "rot");
    for (uint32_t i = 0; i < count; i++) {
        const BoplRlEntity *ea = i < a->header.entity_count ? &a->entities[i] : NULL;
        const BoplRlEntity *eb = i < b->header.entity_count ? &b->entities[i] : NULL;
        printf("%4u |", i);
        if (ea) printf(" %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f |", units(ea->right_x),
                       units(ea->right_y), units(ea->up_x), units(ea->up_y), units(ea->radius),
                       units(ea->rotation));
        else printf(" %70s |", "-");
        if (eb) printf(" %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f", units(eb->right_x),
                       units(eb->right_y), units(eb->up_x), units(eb->up_y), units(eb->radius),
                       units(eb->rotation));
        else printf(" %70s", "-");
        printf("\n");
    }
    /* Raw Q32.32 motion of rows whose motion differs, for hand-checking the
     * arithmetic of a last-bit divergence. */
    for (uint32_t i = 0; i < count && i < a->header.entity_count && i < b->header.entity_count; i++) {
        const BoplRlEntity *ea = &a->entities[i], *eb = &b->entities[i];
        if (ea->x == eb->x && ea->y == eb->y && ea->vx == eb->vx && ea->vy == eb->vy) continue;
        printf("raw %2u | x %" PRId64 " y %" PRId64 " vx %" PRId64 " vy %" PRId64
               " | x %" PRId64 " y %" PRId64 " vx %" PRId64 " vy %" PRId64 "\n", i,
               ea->x, ea->y, ea->vx, ea->vy, eb->x, eb->y, eb->vx, eb->vy);
    }
    for (uint32_t i = 0; i < a->header.player_count && i < b->header.player_count; i++) {
        const BoplRlPlayer *pa = &a->players[i], *pb = &b->players[i];
        printf("player %u | flags %u alive %u stocks %u deaths %u cause %u x %.8f y %.8f | flags %u alive %u stocks %u deaths %u cause %u x %.8f y %.8f\n",
               i, pa->flags, pa->alive_bodies, pa->stocks, pa->deaths, pa->cause_of_death, units(pa->x), units(pa->y),
               pb->flags, pb->alive_bodies, pb->stocks, pb->deaths, pb->cause_of_death, units(pb->x), units(pb->y));
    }
    for (uint32_t i = 0; i < a->header.slime_count && i < b->header.slime_count; i++) {
        const BoplRlSlime *sa = &a->slimes[i], *sb = &b->slimes[i];
        printf("slime %u | id %u owner %u flags %u state %u af %u/%u/%u | id %u owner %u flags %u state %u af %u/%u/%u\n",
               i, sa->id, sa->owner, sa->flags, sa->state, sa->ability_flags[0], sa->ability_flags[1],
               sa->ability_flags[2], sb->id, sb->owner, sb->flags, sb->state, sb->ability_flags[0],
               sb->ability_flags[1], sb->ability_flags[2]);
        printf("slime %u | x %.8f y %.8f vx %.8f vy %.8f | x %.8f y %.8f vx %.8f vy %.8f\n", i,
               units(sa->x), units(sa->y), units(sa->vx), units(sa->vy),
               units(sb->x), units(sb->y), units(sb->vx), units(sb->vy));
    }
    fflush(stdout);
}

static long dump_tick = -1;

/* ---- Reference hash cache (REPLAY_AB_CACHE) ---- */

/* One hash per tick over exactly the fields compare_state compares, with the
 * same inactive-row ENABLED exclusion, so equal hashes mean a bit-exact match. */
static uint64_t mix(uint64_t h, uint64_t v) { h ^= v; h *= 0x9E3779B97F4A7C15ull; return h ^ (h >> 29); }

static uint64_t mix_words(uint64_t h, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (; bytes >= 8; p += 8, bytes -= 8) { uint64_t v; memcpy(&v, p, 8); h = mix(h, v); }
    if (bytes) { uint64_t v = 0; memcpy(&v, p, bytes); h = mix(h, v); }
    return h;
}

static uint64_t state_hash(const BoplRlState *s) {
    const BoplRlHeader *hd = &s->header;
    uint64_t h = 0x243F6A8885A308D3ull;
    h = mix(h, hd->player_count); h = mix(h, hd->entity_count); h = mix(h, hd->slime_count);
    h = mix(h, hd->flags); h = mix(h, hd->episode_ticks); h = mix(h, hd->sudden_death);
    h = mix(h, hd->warmup_ticks); h = mix(h, (uint64_t)hd->simulation_time);
    h = mix(h, (uint64_t)hd->player_time_scale); h = mix(h, (uint64_t)hd->simulation_time_since_level_loaded);
    for (uint32_t i = 0; i < hd->player_count; i++) {
        h = mix_words(h, &s->players[i], sizeof s->players[i]);
        h = mix_words(h, &s->telemetry[i], sizeof s->telemetry[i]);
    }
    h = mix_words(h, s->slimes, (size_t)hd->slime_count * sizeof *s->slimes);
    for (uint32_t i = 0; i < hd->entity_count; i++) {
        BoplRlEntity e = s->entities[i];
        if (!strict && !(e.flags & BOPL_ENTITY_ACTIVE)) e.flags &= ~(uint32_t)BOPL_ENTITY_ENABLED;
        h = mix_words(h, &e, sizeof e);
    }
    return h;
}

/* The reference's hash at every tick it produced a state (0 = after reset),
 * and the episode tick it ended on (0 = ran out of input). */
typedef struct { uint64_t *hash; unsigned count, capacity, end; int complete; } Hashes;

static void hashes_push(Hashes *h, uint64_t value) {
    if (h->count == h->capacity) {
        h->capacity = h->capacity ? h->capacity * 2 : 16384;
        h->hash = realloc(h->hash, h->capacity * sizeof *h->hash);
    }
    h->hash[h->count++] = value;
}

static const char *cache_root = NULL;
static char cache_dir[4096];

static uint64_t mix_text(uint64_t h, const char *text) { return mix_words(h, text, strlen(text)); }

static int cache_setup(const char *reference) {
    struct stat st;
    char real[4096];
    if (!realpath(reference, real) || stat(real, &st)) return 1;
    uint64_t key = mix_text(0x13198A2E03707344ull, real);
    key = mix(key, (uint64_t)st.st_size); key = mix(key, (uint64_t)st.st_ino);
    key = mix(key, (uint64_t)st.st_mtim.tv_sec); key = mix(key, (uint64_t)st.st_mtim.tv_nsec);
    snprintf(cache_dir, sizeof cache_dir, "%s/%016" PRIx64, cache_root, key);
    mkdir(cache_root, 0755);
    mkdir(cache_dir, 0755);
    return 0;
}

static void cache_file(const char *path, uint32_t warmup, uint32_t limit, unsigned max_ticks,
                       char *out, size_t capacity) {
    char real[4096];
    uint64_t key = mix_text(0xA4093822299F31D0ull, realpath(path, real) ? real : path);
    key = mix(key, warmup); key = mix(key, limit); key = mix(key, max_ticks); key = mix(key, (uint64_t)strict);
    snprintf(out, capacity, "%s/%016" PRIx64 ".h64", cache_dir, key);
}

static int cache_load(const char *file, Hashes *h) {
    FILE *f = fopen(file, "rb");
    if (!f) return 1;
    uint32_t head[2];
    int ok = fread(head, sizeof head, 1, f) == 1 && head[0] > 0;
    if (ok) {
        h->hash = malloc(head[0] * sizeof *h->hash);
        h->count = h->capacity = head[0];
        h->end = head[1];
        ok = fread(h->hash, sizeof *h->hash, head[0], f) == head[0];
    }
    fclose(f);
    return !ok;
}

static void cache_save(const char *file, const Hashes *h) {
    char temporary[4400];
    snprintf(temporary, sizeof temporary, "%s.%d", file, (int)getpid());
    FILE *f = fopen(temporary, "wb");
    if (!f) return;
    uint32_t head[2] = {h->count, h->end};
    int ok = fwrite(head, sizeof head, 1, f) == 1 && fwrite(h->hash, sizeof *h->hash, h->count, f) == h->count;
    ok = !fclose(f) && ok;
    if (ok) rename(temporary, file); else unlink(temporary);
}

/* ---- Per-game replay ---- */

typedef struct {
    Compare compare;
    unsigned episode_ticks;      /* ticks actually stepped */
    unsigned reference_end, candidate_end; /* episode tick each backend ended on, 0 = ran out of input */
    int failed;
    char loadout[64];
    uint32_t map, seed, players;
} GameResult;

static void loadout_text(const BoplRlConfig *config, char *out, size_t capacity) {
    snprintf(out, capacity, "%u,%u,%u", config->player[0].abilities[0],
             config->player[0].abilities[1], config->player[0].abilities[2]);
}

static int replay_game(BoplRl *reference, BoplRl *candidate, const char *path,
                       uint32_t warmup, uint32_t limit, int64_t tolerance,
                       unsigned max_ticks, int full, FILE *trace, Hashes *record, GameResult *result) {
    Replay replay;
    memset(result, 0, sizeof *result);
    if (replay_decode(path, warmup, limit, &replay)) { result->failed = 1; return 1; }
    result->map = replay.config.map;
    result->seed = replay.config.seed;
    result->players = replay.config.players;
    loadout_text(&replay.config, result->loadout, sizeof result->loadout);
    if (replay.tick_count <= warmup) {
        fprintf(stderr, "%s: recording shorter than the warmup (%u ticks)\n", path, replay.tick_count);
        replay_free(&replay); result->failed = 1; return 1;
    }
    BoplRl *world[2] = {reference, candidate};
    for (int backend = 0; backend < 2; backend++) {
        if (bopl_rl_reset(world[backend], &replay.config) != BOPL_RL_OK) {
            fprintf(stderr, "%s: backend %d reset failed: %s\n", path, backend, bopl_rl_error(world[backend]));
            replay_free(&replay); result->failed = 1; return 1;
        }
    }
    const BoplRlState *state[2] = {bopl_rl_state(reference), bopl_rl_state(candidate)};
    if (trace) {
        for (uint32_t p = 0; p < state[1]->header.player_count; p++)
            fprintf(trace, "0\t%u\t%.6f\t%.6f\n", p, units(state[1]->players[p].x), units(state[1]->players[p].y));
    }
    if (dump_tick == 0) dump_tables(state[0], state[1], 0);
    compare_state(state[0], state[1], 0, tolerance, &result->compare);
    result->compare.compared_ticks = 1;
    if (record) hashes_push(record, state_hash(state[0]));

    unsigned total = replay.tick_count - warmup;
    if (max_ticks && total > max_ticks) total = max_ticks;
    int ended[2] = {0, 0};
    unsigned tick;
    for (tick = 1; tick <= total; tick++) {
        const BoplRlAction *actions = &replay.actions[(size_t)(warmup + tick - 1) * replay.config.players];
        for (int backend = 0; backend < 2; backend++) {
            if (ended[backend]) continue;
            int status = bopl_rl_step(world[backend], actions, replay.config.players, 1);
            if (status == BOPL_RL_ENDED) {
                ended[backend] = 1;
                if (backend == 0 && record) record->end = tick;
                continue;
            }
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "%s: backend %d step %u failed: %s\n", path, backend, tick,
                        bopl_rl_error(world[backend]));
                replay_free(&replay); result->failed = 1; return 1;
            }
            const BoplRlState *now = bopl_rl_state(world[backend]);
            if (backend == 0 && record) hashes_push(record, state_hash(now));
            if (now->header.flags & BOPL_RL_TERMINAL) {
                ended[backend] = 1;
                if (backend == 0) result->reference_end = tick; else result->candidate_end = tick;
                if (backend == 0 && record) record->end = tick;
            }
        }
        state[0] = bopl_rl_state(reference);
        state[1] = bopl_rl_state(candidate);
        if (trace) {
            for (uint32_t p = 0; p < state[1]->header.player_count; p++)
                fprintf(trace, "%u\t%u\t%.6f\t%.6f\n", tick, p, units(state[1]->players[p].x),
                        units(state[1]->players[p].y));
        }
        result->episode_ticks = tick;
        result->compare.compared_ticks++;
        if (dump_tick >= 0 && (unsigned)dump_tick == tick) dump_tables(state[0], state[1], tick);
        int diverged = compare_state(state[0], state[1], tick, tolerance, &result->compare);
        if (diverged && !full) break;
        if (ended[0] && ended[1]) break;
    }
    /* A game that diverged still records the rest of the reference, so the
     * next candidate can check every tick against the cache. */
    for (tick++; record && !ended[0] && tick <= total; tick++) {
        const BoplRlAction *actions = &replay.actions[(size_t)(warmup + tick - 1) * replay.config.players];
        int status = bopl_rl_step(reference, actions, replay.config.players, 1);
        if (status == BOPL_RL_ENDED) { ended[0] = 1; record->end = tick; break; }
        if (status != BOPL_RL_OK) { replay_free(&replay); return 0; }
        const BoplRlState *now = bopl_rl_state(reference);
        hashes_push(record, state_hash(now));
        if (now->header.flags & BOPL_RL_TERMINAL) { ended[0] = 1; record->end = tick; }
    }
    if (record) record->complete = 1;
    replay_free(&replay);
    return 0;
}

/* The candidate alone against a cached reference. Returns 0 when every tick
 * matches, so the game is clean; anything else is rerun live for the report. */
static int cached_game(BoplRl *candidate, const char *path, uint32_t warmup, uint32_t limit,
                       unsigned max_ticks, const Hashes *reference, GameResult *result) {
    Replay replay;
    memset(result, 0, sizeof *result);
    if (replay_decode(path, warmup, limit, &replay)) return 1;
    result->map = replay.config.map;
    result->seed = replay.config.seed;
    result->players = replay.config.players;
    loadout_text(&replay.config, result->loadout, sizeof result->loadout);
    int bad = replay.tick_count <= warmup || bopl_rl_reset(candidate, &replay.config) != BOPL_RL_OK
              || state_hash(bopl_rl_state(candidate)) != reference->hash[0];
    result->compare.compared_ticks = 1;
    unsigned total = bad ? 0 : replay.tick_count - warmup;
    if (max_ticks && total > max_ticks) total = max_ticks;
    int ended = 0;
    for (unsigned tick = 1; tick <= total; tick++) {
        if (!ended) {
            const BoplRlAction *actions = &replay.actions[(size_t)(warmup + tick - 1) * replay.config.players];
            int status = bopl_rl_step(candidate, actions, replay.config.players, 1);
            if (status == BOPL_RL_ENDED) ended = 1;
            else if (status != BOPL_RL_OK) { bad = 1; break; }
            else if (bopl_rl_state(candidate)->header.flags & BOPL_RL_TERMINAL) ended = 1;
        }
        result->episode_ticks = tick;
        result->compare.compared_ticks++;
        unsigned at = tick < reference->count ? tick : reference->count - 1;
        if (state_hash(bopl_rl_state(candidate)) != reference->hash[at]) { bad = 1; break; }
        if (ended && reference->end && tick >= reference->end) break;
    }
    replay_free(&replay);
    return bad;
}

/* ---- Directory walking ---- */

typedef struct { char **paths; size_t count, capacity; } Paths;

static void paths_push(Paths *set, const char *path) {
    if (set->count == set->capacity) {
        set->capacity = set->capacity ? set->capacity * 2 : 64;
        set->paths = realloc(set->paths, set->capacity * sizeof *set->paths);
    }
    size_t length = strlen(path) + 1;
    char *copy = malloc(length);
    memcpy(copy, path, length);
    set->paths[set->count++] = copy;
}

static int compare_paths(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static void collect(const char *path, Paths *set) {
    struct stat info;
    if (stat(path, &info)) { fprintf(stderr, "%s: cannot stat\n", path); return; }
    if (S_ISREG(info.st_mode)) {
        size_t length = strlen(path);
        if (length > 4 && !strcmp(path + length - 4, ".rep")) paths_push(set, path);
        return;
    }
    if (!S_ISDIR(info.st_mode)) return;
    DIR *dir = opendir(path);
    if (!dir) { fprintf(stderr, "%s: cannot open directory\n", path); return; }
    size_t start = set->count;
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char child[4096];
        snprintf(child, sizeof child, "%s/%s", path, entry->d_name);
        collect(child, set);
    }
    closedir(dir);
    if (set->count > start) qsort(set->paths + start, set->count - start, sizeof *set->paths, compare_paths);
}

/* ---- Summary bucketing ---- */

typedef struct {
    char field[96];
    unsigned games;
    unsigned earliest;
    char example[512];
} Bucket;

static Bucket *bucket_for(Bucket *buckets, size_t *count, size_t capacity, const char *key) {
    for (size_t i = 0; i < *count; i++) if (!strcmp(buckets[i].field, key)) return &buckets[i];
    if (*count == capacity) return NULL;
    Bucket *bucket = &buckets[(*count)++];
    memset(bucket, 0, sizeof *bucket);
    snprintf(bucket->field, sizeof bucket->field, "%s", key);
    bucket->earliest = (unsigned)-1;
    return bucket;
}

static int bucket_order(const void *a, const void *b) {
    const Bucket *x = a, *y = b;
    if (x->earliest != y->earliest) return x->earliest < y->earliest ? -1 : 1;
    if (x->games != y->games) return x->games > y->games ? -1 : 1;
    return strcmp(x->field, y->field);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <reference.so> <candidate.so> <game.rep|directory>...\n", argv[0]);
        return 2;
    }
    uint32_t warmup = (uint32_t)(getenv("REPLAY_AB_WARMUP") ? atoi(getenv("REPLAY_AB_WARMUP")) : 240);
    uint32_t limit = (uint32_t)(getenv("REPLAY_AB_LIMIT") ? atoi(getenv("REPLAY_AB_LIMIT")) : 7200);
    /* 8192 raw = 1.9e-6 world units: above the largest static authored-geometry
     * gap measured at reset (1.06e-6 on a map trigger half-extent, the
     * separately rounded rounded-rectangle geometry)
     * and far below any real physics drift. */
    int64_t tolerance = getenv("REPLAY_AB_TOL") ? strtoll(getenv("REPLAY_AB_TOL"), NULL, 10) : 8192;
    unsigned max_ticks = (unsigned)(getenv("REPLAY_AB_MAX_TICKS") ? atoi(getenv("REPLAY_AB_MAX_TICKS")) : 0);
    unsigned max_games = (unsigned)(getenv("REPLAY_AB_GAMES") ? atoi(getenv("REPLAY_AB_GAMES")) : 0);
    int quiet = getenv("REPLAY_AB_QUIET") && atoi(getenv("REPLAY_AB_QUIET"));
    int full = getenv("REPLAY_AB_FULL") && atoi(getenv("REPLAY_AB_FULL"));
    const char *trace_path = getenv("REPLAY_AB_TRACE");
    const char *log_path = getenv("REPLAY_AB_LOG") ? getenv("REPLAY_AB_LOG") : "/tmp/replay_ab.log";
    report_all = getenv("REPLAY_AB_ALL") && atoi(getenv("REPLAY_AB_ALL"));
    ignore_list = getenv("REPLAY_AB_IGNORE");
    strict = getenv("REPLAY_AB_STRICT") && atoi(getenv("REPLAY_AB_STRICT"));
    visible_only = getenv("REPLAY_AB_VISIBLE") && atoi(getenv("REPLAY_AB_VISIBLE"));
    dump_tick = getenv("REPLAY_AB_DUMP") ? atol(getenv("REPLAY_AB_DUMP")) : -1;

    Paths set = {0};
    for (int i = 3; i < argc; i++) collect(argv[i], &set);
    if (!set.count) { fprintf(stderr, "no .rep archives found\n"); return 2; }
    if (max_games && set.count > max_games) set.count = max_games;

    cache_root = getenv("REPLAY_AB_CACHE");
    if (cache_root && (!*cache_root || tolerance || visible_only || ignore_list || report_all || full
                       || trace_path || dump_tick >= 0 || cache_setup(argv[1])))
        cache_root = NULL;

    /* The reference opens on first use: a fully cached run never needs it. */
    char error[512] = {0};
    BoplRl *world[2] = {NULL, NULL};
    for (int backend = cache_root ? 1 : 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      log_path, 180000, error, sizeof error);
        if (!world[backend]) { fprintf(stderr, "open %s failed: %s\n", argv[backend + 1], error); return 2; }
    }

    FILE *trace = trace_path ? fopen(trace_path, "w") : NULL;
    if (trace_path && !trace) { fprintf(stderr, "%s: cannot write trace\n", trace_path); return 2; }

    enum { BUCKETS = 256 };
    Bucket discrete[BUCKETS], numeric[BUCKETS];
    size_t discrete_count = 0, numeric_count = 0;
    unsigned games = 0, clean = 0, failed = 0, discrete_games = 0, numeric_games = 0;
    unsigned long long ticks_total = 0;
    int64_t worst_gap = 0;
    char worst_game[512] = "";

    for (size_t i = 0; i < set.count; i++) {
        GameResult result;
        char file[4300];
        Hashes cached = {0}, record = {0};
        int hit = 0;
        if (cache_root) {
            cache_file(set.paths[i], warmup, limit, max_ticks, file, sizeof file);
            hit = !cache_load(file, &cached) && !cached_game(world[1], set.paths[i], warmup, limit, max_ticks,
                                                            &cached, &result);
            free(cached.hash);
            if (!world[0] && !hit) {
                world[0] = bopl_rl_open(argv[1], "/tmp/gate_bootstrap.rep", log_path, 180000,
                                        error, sizeof error);
                if (!world[0]) { fprintf(stderr, "open %s failed: %s\n", argv[1], error); return 2; }
            }
        }
        if (!hit) {
            int bad = replay_game(world[0], world[1], set.paths[i], warmup, limit, tolerance, max_ticks,
                                  full, trace, cache_root ? &record : NULL, &result);
            if (cache_root && record.complete) cache_save(file, &record);
            free(record.hash);
            if (bad) { failed++; continue; }
        }
        games++;
        ticks_total += result.compare.compared_ticks;
        if (result.compare.max_gap > worst_gap) {
            worst_gap = result.compare.max_gap;
            snprintf(worst_game, sizeof worst_game, "%s", set.paths[i]);
        }
        int bad = result.compare.discrete.found || result.compare.numeric.found;
        if (!bad) {
            clean++;
            if (!quiet)
                printf("OK    %-70s map %u seed %u loadout %s ticks %u maxgap %.3g\n",
                       set.paths[i], result.map, result.seed, result.loadout,
                       result.episode_ticks, units(result.compare.max_gap));
            continue;
        }
        if (result.compare.discrete.found) {
            discrete_games++;
            char key[96];
            generalize(result.compare.discrete.field, key, sizeof key);
            Bucket *bucket = bucket_for(discrete, &discrete_count, BUCKETS, key);
            if (bucket) {
                bucket->games++;
                if (result.compare.discrete.tick < bucket->earliest) {
                    bucket->earliest = result.compare.discrete.tick;
                    snprintf(bucket->example, sizeof bucket->example, "%s tick %u %s %" PRId64 " vs %" PRId64 " [%s]",
                             set.paths[i], result.compare.discrete.tick, result.compare.discrete.field,
                             result.compare.discrete.left, result.compare.discrete.right,
                             result.compare.discrete.context);
                }
            }
        }
        if (result.compare.numeric.found) {
            numeric_games++;
            char key[96];
            generalize(result.compare.numeric.field, key, sizeof key);
            Bucket *bucket = bucket_for(numeric, &numeric_count, BUCKETS, key);
            if (bucket) {
                bucket->games++;
                if (result.compare.numeric.tick < bucket->earliest) {
                    bucket->earliest = result.compare.numeric.tick;
                    snprintf(bucket->example, sizeof bucket->example, "%s tick %u %s %.9g vs %.9g [%s]",
                             set.paths[i], result.compare.numeric.tick, result.compare.numeric.field,
                             units(result.compare.numeric.left), units(result.compare.numeric.right),
                             result.compare.numeric.context);
                }
            }
        }
        printf("DIFF  %-70s map %u seed %u loadout %s ticks %u\n", set.paths[i], result.map,
               result.seed, result.loadout, result.episode_ticks);
        if (result.compare.numeric.found)
            printf("      numeric  tick %5u %-34s %.9g vs %.9g   (%s; raw gap %" PRId64 ")\n", result.compare.numeric.tick,
                   result.compare.numeric.field, units(result.compare.numeric.left),
                   units(result.compare.numeric.right), result.compare.numeric.context,
                   magnitude(result.compare.numeric.left, result.compare.numeric.right));
        if (result.compare.discrete.found)
            printf("      discrete tick %5u %-34s %" PRId64 " vs %" PRId64 "   (%s)\n",
                   result.compare.discrete.tick, result.compare.discrete.field,
                   result.compare.discrete.left, result.compare.discrete.right,
                   result.compare.discrete.context);
        fflush(stdout);
    }
    if (trace) fclose(trace);
    for (int backend = 0; backend < 2; backend++) if (world[backend]) bopl_rl_close(world[backend]);

    printf("\n=== replay_ab summary ===\n");
    printf("reference %s\ncandidate %s\n", argv[1], argv[2]);
    printf("games %u (clean %u, numeric %u, discrete %u, unreadable %u), compared ticks %llu, tolerance %" PRId64 " raw\n",
           games, clean, numeric_games, discrete_games, failed, ticks_total, tolerance);
    printf("largest positional gap %.6g world units in %s\n", units(worst_gap), worst_game[0] ? worst_game : "-");
    qsort(discrete, discrete_count, sizeof *discrete, bucket_order);
    qsort(numeric, numeric_count, sizeof *numeric, bucket_order);
    if (numeric_count) {
        printf("\nfirst numeric divergence by field (earliest tick, games):\n");
        for (size_t i = 0; i < numeric_count; i++)
            printf("  %-40s tick %6u  %4u games  %s\n", numeric[i].field, numeric[i].earliest,
                   numeric[i].games, numeric[i].example);
    }
    if (discrete_count) {
        printf("\nfirst discrete divergence by field (earliest tick, games):\n");
        for (size_t i = 0; i < discrete_count; i++)
            printf("  %-40s tick %6u  %4u games  %s\n", discrete[i].field, discrete[i].earliest,
                   discrete[i].games, discrete[i].example);
    }
    for (size_t i = 0; i < set.count; i++) free(set.paths[i]);
    free(set.paths);
    return (discrete_games || numeric_games || failed) ? 1 : 0;
}
