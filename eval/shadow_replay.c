/* Offline proof of the in-game shadow path (train/bopl_shadow.h, docs/SHADOW.md).
 *
 *   shadow_replay <game.so> <shadow.so> <path> [path...]
 *
 * For every BOPLREP2 archive (or directory of them) this plays the round the
 * way RlAgent.cs does with BOPL_AGENT_SHADOW=1, with <game.so> (the .NET
 * reference build of the game's own C#, BoplSimNative.so) standing in for the
 * running game:
 *
 *   1. bopl_shadow_begin with the archive's bytes (setup + source hash), the
 *      call RlAgent makes from ReplaySession.Recording after level tick 1;
 *   2. the "game" is reset with the recorded setup and runs the zero-input
 *      warmup, as RlAgent forces;
 *   3. each tick the game's state is serialized to the RlObservation.cs wire
 *      format, with the wrapper fields RlAgent writes (STARTED only after the
 *      warmup, telemetry never counted, simulation_time = level ticks x
 *      FixedTimeStep), and handed to bopl_shadow_check_bytes; then both are
 *      stepped with the recorded packets of every player (PlayerList order),
 *      the shadow through bopl_shadow_step;
 *   4. on the game's terminal state the round is closed with bopl_shadow_end.
 *
 * Before the corpus it runs two self tests on the first archive: a wrong
 * source hash must be refused, packets perturbed from tick 150 on, fed to the
 * shadow only, must be reported as a desync, and prefab hierarchy numbers
 * shifted as in a later live round must not be.
 *
 * Environment:
 *   SHADOW_REPLAY_GAMES   stop after this many archives (0 = all)
 *   SHADOW_REPLAY_SHARD   "i/n": only archives with index % n == i
 *   SHADOW_REPLAY_QUIET   1 = print only non-OK games and the summary
 *   SHADOW_REPLAY_LOG     the libraries' log (default /tmp/shadow_replay.log)
 *   BOPL_AGENT_SHADOW_*   the shadow's own options (bopl_shadow.h)
 * Exit status: 0 all in sync, 1 any desync/failure, 2 usage or I/O. */
#define _DEFAULT_SOURCE
#include "bopl_rl.h"
#include "bopl_shadow.h"
#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define WARMUP 240u
#define LIMIT 2160000u

typedef struct {
    unsigned char *raw; size_t length;
    BoplRlConfig config;
    uint32_t ticks;
    BoplRlAction *actions;   /* ticks x players, warmup included */
} Archive;

static uint32_t take(const unsigned char *raw, size_t length, size_t *at, size_t width) {
    uint32_t value = 0;
    for (size_t i = 0; i < width; i++, (*at)++)
        if (*at < length) value |= (uint32_t)raw[*at] << (8 * i);
    return value;
}

static int load(const char *path, Archive *a) {
    memset(a, 0, sizeof *a);
    FILE *file = fopen(path, "rb");
    if (!file) return -1;
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (size < 150) { fclose(file); return -1; }
    a->raw = malloc((size_t)size);
    a->length = (size_t)size;
    if (!a->raw || fread(a->raw, 1, a->length, file) != a->length) { fclose(file); return -1; }
    fclose(file);
    if (memcmp(a->raw, "BOPLREP2", 8)) return -1;
    size_t at = 136;
    bopl_rl_default_config(&a->config, 2);
    a->config.seed = take(a->raw, a->length, &at, 4);
    a->config.map = take(a->raw, a->length, &at, 1);
    a->config.mode = take(a->raw, a->length, &at, 1);
    a->config.stocks = take(a->raw, a->length, &at, 4);
    uint32_t players = take(a->raw, a->length, &at, 1);
    if (players < 1 || players > 6) return -1;
    a->config.players = players;
    a->config.warmup_ticks = WARMUP;
    a->config.max_episode_ticks = LIMIT;
    for (uint32_t p = 0; p < players; p++) {
        (void)take(a->raw, a->length, &at, 1);
        a->config.player[p].color = take(a->raw, a->length, &at, 1);
        a->config.player[p].team = take(a->raw, a->length, &at, 1);
        a->config.player[p].keyboard = take(a->raw, a->length, &at, 1);
        uint32_t slots = take(a->raw, a->length, &at, 1);
        if (slots < 1 || slots > 3) return -1;
        memset(a->config.player[p].abilities, 0, sizeof a->config.player[p].abilities);
        for (uint32_t s = 0; s < slots; s++) a->config.player[p].abilities[s] = take(a->raw, a->length, &at, 2);
        a->config.slots = slots;
    }
    a->ticks = take(a->raw, a->length, &at, 4);
    if (a->ticks > 1000000u) return -1;
    a->actions = calloc((size_t)a->ticks * players + 1, sizeof(BoplRlAction));
    if (!a->actions) return -1;
    for (uint32_t t = 0; t < a->ticks; t++)
        for (uint32_t p = 0; p < players; p++) {
            (void)take(a->raw, a->length, &at, 4);
            BoplRlAction *action = &a->actions[(size_t)t * players + p];
            action->angle = (uint8_t)take(a->raw, a->length, &at, 1);
            action->buttons = (uint16_t)take(a->raw, a->length, &at, 2);
            (void)take(a->raw, a->length, &at, 1);
        }
    return at > a->length ? -1 : 0;
}

static void unload(Archive *a) { free(a->raw); free(a->actions); memset(a, 0, sizeof *a); }

/* RlObservation.Write's record, with RlAgent's wrapper fields. */
typedef struct { unsigned char *bytes; size_t length, capacity; } Wire;
static void put(Wire *w, const void *data, size_t size) {
    if (w->length + size > w->capacity) {
        w->capacity = (w->length + size) * 2;
        w->bytes = realloc(w->bytes, w->capacity);
        if (!w->bytes) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    memcpy(w->bytes + w->length, data, size);
    w->length += size;
}
/* Self test: the live game's prefab HierarchyNumbers carry earlier rounds'
 * prefabs (Updater.PreLevelLoad keeps prefabsAdded); this shift mimics it. */
static uint32_t hierarchy_shift;

static void serialize(Wire *w, const BoplRlState *s, uint32_t level_tick, int64_t step, int terminal) {
    w->length = 0;
    BoplRlHeader h = s->header;
    h.tick = level_tick;
    h.flags = (terminal ? (uint32_t)BOPL_RL_TERMINAL : 0u) | (h.episode_ticks > 0 ? (uint32_t)BOPL_RL_STARTED : 0u);
    h.simulation_time = (int64_t)level_tick * step;
    put(w, &h, sizeof h);
    put(w, s->players, (size_t)h.player_count * sizeof(BoplRlPlayer));
    for (uint32_t i = 0; i < h.entity_count; i++) {
        BoplRlEntity e = s->entities[i];
        if (hierarchy_shift && e.hierarchy >= 100000000u) e.hierarchy += hierarchy_shift;
        put(w, &e, sizeof e);
    }
    put(w, s->slimes, (size_t)h.slime_count * sizeof(BoplRlSlime));
    BoplRlTelemetry zero[6];
    memset(zero, 0, sizeof zero);
    put(w, zero, (size_t)h.player_count * sizeof(BoplRlTelemetry));
}

typedef struct {
    unsigned games, ok, desync, failed, game_failed, unreadable, players[7], sudden, deaths_games, max_deaths;
    unsigned long long ticks;
    unsigned char maps[256], abilities[256];
} Summary;

/* One round. perturb_tick > 0: from that episode tick on, the shadow alone gets
 * every player's jump bit flipped and stick turned a half circle (self test; a
 * single flipped bit can land on an airborne player and change nothing). Returns the shadow's final status. */
static int round_once(BoplRl *game, BoplShadow *shadow, const Archive *a, Wire *wire, unsigned perturb_tick,
                      Summary *summary, unsigned *ticks_out, int *deaths_out, int *sudden_out) {
    *ticks_out = 0; *deaths_out = 0; *sudden_out = 0;
    int status = bopl_shadow_begin(shadow, a->raw, a->length, WARMUP);
    if (status != BOPL_SHADOW_TRACKING) return status;
    if (bopl_rl_reset(game, &a->config) != BOPL_RL_OK) {
        fprintf(stderr, "game reset failed: %s\n", bopl_rl_error(game));
        bopl_shadow_end(shadow);
        return -1;
    }
    const BoplRlState *state = bopl_rl_state(game);
    int64_t step = 0;
    uint32_t players = a->config.players;
    BoplRlAction packets[6];
    unsigned total = a->ticks > WARMUP ? a->ticks - WARMUP : 0;
    serialize(wire, state, WARMUP, 0, 0);
    int64_t base = state->header.simulation_time; /* the game's clock at S_0, to learn FixedTimeStep */
    int terminal = 0;
    for (unsigned e = 0;; e++) {
        if (e == 1) step = state->header.simulation_time - base;
        if (e > 0) serialize(wire, state, WARMUP + e, step, terminal);
        /* S_0's clock: RlAgent's is warmup x FixedTimeStep. Unknown until the
         * first step, so S_0 is serialized with the game's own value. */
        if (e == 0) { BoplRlHeader *h = (BoplRlHeader *)wire->bytes; h->simulation_time = base; }
        status = bopl_shadow_check_bytes(shadow, wire->bytes, wire->length);
        if (status != BOPL_SHADOW_TRACKING) break;
        for (uint32_t p = 0; p < players; p++) {
            if (state->players[p].deaths > (uint32_t)*deaths_out) *deaths_out = (int)state->players[p].deaths;
        }
        if (state->header.sudden_death) *sudden_out = 1;
        if (terminal || e >= total) { status = bopl_shadow_end(shadow); break; }
        memcpy(packets, &a->actions[(size_t)(WARMUP + e) * players], players * sizeof(BoplRlAction));
        int result = bopl_rl_step(game, packets, players, 1);
        if (result != BOPL_RL_OK && result != BOPL_RL_ENDED) {
            fprintf(stderr, "game step failed: %s\n", bopl_rl_error(game));
            bopl_shadow_end(shadow);
            return -3; /* the stand-in game (.NET reference) broke, not the shadow */
        }
        state = bopl_rl_state(game);
        terminal = (state->header.flags & BOPL_RL_TERMINAL) != 0;
        if (perturb_tick && e + 1 >= perturb_tick)
            for (uint32_t p = 0; p < players; p++) {
                packets[p].buttons ^= 1;
                packets[p].angle = packets[p].angle ? (uint8_t)(1 + (packets[p].angle - 1 + 127) % 255) : 64;
            }
        status = bopl_shadow_step(shadow, packets, players);
        if (status != BOPL_SHADOW_TRACKING) break;
        (*ticks_out)++;
    }
    (void)summary;
    return status;
}

static int collect(const char *path, char ***paths, size_t *count, size_t *capacity) {
    struct stat info;
    if (stat(path, &info)) return -1;
    if (S_ISDIR(info.st_mode)) {
        struct dirent **entries;
        int n = scandir(path, &entries, NULL, alphasort);
        if (n < 0) return -1;
        for (int i = 0; i < n; i++) {
            const char *name = entries[i]->d_name;
            if (name[0] != '.') {
                char child[4096];
                snprintf(child, sizeof child, "%s/%s", path, name);
                struct stat sub;
                size_t len = strlen(name);
                if (!stat(child, &sub) && (S_ISDIR(sub.st_mode) || (len > 4 && !strcmp(name + len - 4, ".rep"))))
                    collect(child, paths, count, capacity);
            }
            free(entries[i]);
        }
        free(entries);
        return 0;
    }
    if (*count == *capacity) {
        *capacity = *capacity ? *capacity * 2 : 256;
        *paths = realloc(*paths, *capacity * sizeof **paths);
    }
    (*paths)[(*count)++] = strdup(path);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <game.so> <shadow.so> <path> [path...]\n", argv[0]); return 2; }
    const char *log = getenv("SHADOW_REPLAY_LOG") ? getenv("SHADOW_REPLAY_LOG") : "/tmp/shadow_replay.log";
    unsigned limit = getenv("SHADOW_REPLAY_GAMES") ? (unsigned)atoi(getenv("SHADOW_REPLAY_GAMES")) : 0;
    int quiet = getenv("SHADOW_REPLAY_QUIET") && atoi(getenv("SHADOW_REPLAY_QUIET"));
    unsigned shard = 0, shards = 1;
    if (getenv("SHADOW_REPLAY_SHARD")) sscanf(getenv("SHADOW_REPLAY_SHARD"), "%u/%u", &shard, &shards);
    if (!shards) shards = 1;
    char error[1024];
    BoplRl *game = bopl_rl_open(argv[1], "/tmp/gate_bootstrap.rep", log, 180000, error, sizeof error);
    if (!game) { fprintf(stderr, "game %s: %s\n", argv[1], error); return 2; }
    BoplShadow *shadow = bopl_shadow_open(argv[2], error, sizeof error);
    if (!shadow) { fprintf(stderr, "%s\n", error); return 2; }
    printf("game   %s (source %s)\nshadow %s\n", argv[1], bopl_rl_source_hash(game), bopl_shadow_library(shadow));

    char **paths = NULL; size_t count = 0, capacity = 0;
    for (int i = 3; i < argc; i++)
        if (collect(argv[i], &paths, &count, &capacity)) { fprintf(stderr, "%s: cannot read\n", argv[i]); return 2; }
    Wire wire = {0};
    Summary summary;
    memset(&summary, 0, sizeof summary);
    int bad = 0;

    /* Self tests on the first readable archive. */
    for (size_t i = 0; i < count; i++) {
        Archive a;
        if (load(paths[i], &a) || a.ticks <= WARMUP + 200) { unload(&a); continue; }
        unsigned ticks; int deaths, sudden;
        unsigned char saved = a.raw[8];
        a.raw[8] = a.raw[8] == '0' ? '1' : '0';
        int refused = bopl_shadow_begin(shadow, a.raw, a.length, WARMUP);
        printf("selftest source-mismatch %s: %s\n  %s\n", refused == BOPL_SHADOW_FAILED ? "PASS" : "FAIL",
               paths[i], bopl_shadow_report(shadow));
        if (refused != BOPL_SHADOW_FAILED) bad = 1;
        a.raw[8] = saved;
        int caught = round_once(game, shadow, &a, &wire, 150, &summary, &ticks, &deaths, &sudden);
        const char *report = bopl_shadow_report(shadow);
        const char *newline = strchr(report, '\n');
        printf("selftest perturbed-packet@150 %s: %.*s\n", caught == BOPL_SHADOW_DESYNC ? "PASS" : "FAIL",
               newline ? (int)(newline - report) : (int)strlen(report), report);
        if (caught != BOPL_SHADOW_DESYNC) bad = 1;
        int plain = round_once(game, shadow, &a, &wire, 0, &summary, &ticks, &deaths, &sudden);
        uint32_t plain_compared = bopl_shadow_compared(shadow);
        hierarchy_shift = 184;
        int carried = round_once(game, shadow, &a, &wire, 0, &summary, &ticks, &deaths, &sudden);
        hierarchy_shift = 0;
        report = bopl_shadow_report(shadow);
        newline = strchr(report, '\n');
        /* Same outcome as unshifted (an archive may have a real sim desync). */
        int carried_ok = carried == plain && bopl_shadow_compared(shadow) == plain_compared;
        printf("selftest carried-prefab-hierarchy+184 %s: %.*s\n", carried_ok ? "PASS" : "FAIL",
               newline ? (int)(newline - report) : (int)strlen(report), report);
        if (!carried_ok) bad = 1;
        unload(&a);
        break;
    }

    for (size_t i = 0; i < count; i++) {
        if (i % shards != shard) continue;
        if (limit && summary.games >= limit) break;
        Archive a;
        summary.games++;
        if (load(paths[i], &a)) { summary.unreadable++; unload(&a); if (!quiet) printf("SKIP  %s unreadable/not BOPLREP2\n", paths[i]); continue; }
        if (a.ticks <= WARMUP) { summary.unreadable++; unload(&a); if (!quiet) printf("SKIP  %s shorter than the warmup\n", paths[i]); continue; }
        unsigned warm_nonzero = 0;
        for (size_t k = 0; k < (size_t)WARMUP * a.config.players; k++)
            warm_nonzero += a.actions[k].angle || a.actions[k].buttons;
        unsigned ticks; int deaths, sudden;
        int status = round_once(game, shadow, &a, &wire, 0, &summary, &ticks, &deaths, &sudden);
        summary.ticks += ticks;
        summary.players[a.config.players]++;
        summary.maps[a.config.map & 255] = 1;
        for (uint32_t p = 0; p < a.config.players; p++)
            for (uint32_t s = 0; s < a.config.slots; s++) summary.abilities[a.config.player[p].abilities[s] & 255] = 1;
        if (sudden) summary.sudden++;
        if (deaths > 0) summary.deaths_games++;
        if ((unsigned)deaths > summary.max_deaths) summary.max_deaths = (unsigned)deaths;
        const char *tag = status == BOPL_SHADOW_ENDED ? "OK   " : status == BOPL_SHADOW_DESYNC ? "DESYNC"
                        : status == -3 ? "GAMEFAIL" : "FAIL ";
        if (status == BOPL_SHADOW_ENDED) summary.ok++;
        else if (status == BOPL_SHADOW_DESYNC) { summary.desync++; bad = 1; }
        else if (status == -3) { summary.game_failed++; bad = 1; }
        else { summary.failed++; bad = 1; }
        if (!quiet || status != BOPL_SHADOW_ENDED)
            printf("%s %s players %u map %u seed %u ticks %u deaths %d sudden %d%s\n  %s\n", tag, paths[i],
                   a.config.players, a.config.map, a.config.seed, ticks, deaths, sudden,
                   warm_nonzero ? " (recorded warmup inputs zeroed, as RlAgent forces)" : "", bopl_shadow_report(shadow));
        unload(&a);
    }
    unsigned maps = 0, abilities = 0;
    for (int i = 0; i < 256; i++) { maps += summary.maps[i]; abilities += summary.abilities[i]; }
    printf("\n=== shadow_replay summary ===\n");
    printf("games %u: in sync %u, desync %u, failed %u, stand-in game (.NET) broke %u, unreadable %u; %llu ticks "
           "shadowed\n", summary.games, summary.ok, summary.desync, summary.failed, summary.game_failed,
           summary.unreadable, summary.ticks);
    printf("players: 1p %u, 2p %u, 3p %u, 4p %u, 5p %u, 6p %u; maps %u; ability ids %u; games with deaths %u "
           "(max deaths %u); games reaching sudden death %u\n",
           summary.players[1], summary.players[2], summary.players[3], summary.players[4], summary.players[5],
           summary.players[6], maps, abilities, summary.deaths_games, summary.max_deaths, summary.sudden);
    printf("ability ids:");
    for (int i = 0; i < 256; i++) if (summary.abilities[i]) printf(" %d", i);
    printf("\n");
    bopl_shadow_close(shadow);
    bopl_rl_close(game);
    for (size_t i = 0; i < count; i++) free(paths[i]);
    free(paths); free(wire.bytes);
    return bad;
}
