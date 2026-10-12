/* In-game shadow sim; see bopl_shadow.h and docs/SHADOW.md. */
#define _GNU_SOURCE /* dladdr, open_memstream */
#include "bopl_shadow.h"
#include "bopl_rl_compare.h"
#include <dlfcn.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SHADOW_ROUND_LIMIT 2160000u /* the game, not the shadow, decides when a round ends */
#define PREFAB_HIERARCHY 100000000u /* Updater.EndRegisterPrefab's base */
#define SHADOW_ALL_FIELDS 4096      /* bytes of "every differing field" kept in the report */

static const char RUST_PLACEHOLDER[] = "c0debab5c0debab5c0debab5c0debab5c0debab5c0debab5c0debab5c0debab5";

struct BoplShadow {
    BoplRl *world;
    char library[PATH_MAX];
    BoplCompareOptions options;
    int status, reset_pending, sim_ended;
    uint32_t players, warmup, steps, compared;
    /* The game's Player.Kills/Deaths run for the whole match; the shadow's
     * start at zero every round. Compared relative to the round's first check. */
    int baseline_set;
    uint32_t base_kills[6], base_deaths[6];
    /* Stock Updater.PreLevelLoad keeps prefabsAdded across levels, so every
     * prefab-born HierarchyNumber (>= PREFAB_HIERARCHY) of round N is offset by
     * the prefabs of earlier rounds; the sims start at zero. Hierarchy numbers
     * are only compared for equality and sorted, so a constant offset is
     * behaviour-neutral: it is learned at the first check and subtracted. */
    int64_t hierarchy_offset;
    BoplCompare compare;
    char report[1024 + SHADOW_ALL_FIELDS];
    /* Decoded game state (the wire format's entity/slime arrays). */
    BoplRlState game;
    BoplRlEntity *entities; uint32_t ecap;
    BoplRlSlime *slimes; uint32_t scap;
};

static void say(char *error, size_t capacity, const char *text) {
    if (error && capacity) snprintf(error, capacity, "%s", text);
}

static int env_int(const char *name, int fallback) {
    const char *value = getenv(name);
    return value && *value ? atoi(value) : fallback;
}

/* libbopl_sim_q32.{so,dylib} in the directory this shim was loaded from. */
static int beside_shim(char *out, size_t capacity) {
    Dl_info info;
    if (!dladdr((void *)&bopl_shadow_open, &info) || !info.dli_fname) return 0;
    const char *slash = strrchr(info.dli_fname, '/');
    if (!slash) return 0;
    int directory = (int)(slash - info.dli_fname);
    static const char *names[] = {"libbopl_sim_q32.so", "libbopl_sim_q32.dylib"};
    for (size_t i = 0; i < sizeof names / sizeof *names; i++) {
        snprintf(out, capacity, "%.*s/%s", directory, info.dli_fname, names[i]);
        if (!access(out, R_OK)) return 1;
    }
    return 0;
}

BoplShadow *bopl_shadow_open(const char *library, char *error, size_t capacity) {
    BoplShadow *s = calloc(1, sizeof *s);
    if (!s) { say(error, capacity, "shadow: out of memory"); return NULL; }
    const char *chosen = library && *library ? library : getenv("BOPL_AGENT_SHADOW_LIB");
    if (chosen && *chosen) snprintf(s->library, sizeof s->library, "%s", chosen);
    else if (!beside_shim(s->library, sizeof s->library)) {
        say(error, capacity, "shadow: no sim library (set BOPL_AGENT_SHADOW_LIB to the q32 Rust libbopl_sim, "
                             "or put libbopl_sim_q32.so beside libbopl_agent)");
        free(s); return NULL;
    }
    if (s->library[0] != '/') {
        char absolute[PATH_MAX];
        if (!realpath(s->library, absolute)) { say(error, capacity, "shadow: sim library not found"); free(s); return NULL; }
        snprintf(s->library, sizeof s->library, "%s", absolute);
    }
    if (!bopl_rl_in_process(s->library)) {
        say(error, capacity, "shadow: the sim library must be an in-process .so/.dylib"); free(s); return NULL;
    }
    const char *log = getenv("BOPL_AGENT_SHADOW_LOG");
    char message[512] = "";
    /* The in-process library ignores the bootstrap replay; the path need not exist. */
    s->world = bopl_rl_open(s->library, "/tmp/bopl-shadow-bootstrap.rep",
                            log && log[0] == '/' ? log : "/tmp/bopl-shadow.log", 60000, message, sizeof message);
    if (!s->world) {
        char text[PATH_MAX + 600];
        snprintf(text, sizeof text, "shadow: %s: %s", s->library, message);
        say(error, capacity, text); free(s); return NULL;
    }
    s->options.tolerance = getenv("BOPL_AGENT_SHADOW_TOL") ? strtoll(getenv("BOPL_AGENT_SHADOW_TOL"), NULL, 10) : 0;
    s->options.visible_only = env_int("BOPL_AGENT_SHADOW_VISIBLE", 0);
    s->options.strict = env_int("BOPL_AGENT_SHADOW_STRICT", 0);
    s->options.ignore = getenv("BOPL_AGENT_SHADOW_IGNORE");
    s->options.flags_mask = (uint32_t)env_int("BOPL_AGENT_SHADOW_FLAGS", BOPL_RL_TERMINAL);
    s->options.no_telemetry = !env_int("BOPL_AGENT_SHADOW_TELEMETRY", 0);
    say(error, capacity, "");
    return s;
}

static int fail(BoplShadow *s, const char *format, const char *detail) {
    s->status = BOPL_SHADOW_FAILED;
    snprintf(s->report, sizeof s->report, format, detail ? detail : "");
    return s->status;
}

static int finish_reset(BoplShadow *s) {
    if (!s->reset_pending) return 0;
    s->reset_pending = 0;
    if (bopl_rl_reset_finish(s->world) != BOPL_RL_OK) {
        fail(s, "shadow reset failed: %s", bopl_rl_error(s->world));
        return -1;
    }
    return 0;
}

/* Little-endian field; reading past the end leaves *at > length. */
static uint32_t take(const unsigned char *raw, size_t length, size_t *at, size_t width) {
    uint32_t value = 0;
    for (size_t i = 0; i < width; i++, (*at)++)
        if (*at < length) value |= (uint32_t)raw[*at] << (8 * i);
    return value;
}

/* BOPLREP2 setup (eval/rust_ab/replay_ab.c replay_decode, without frames). */
static int decode_setup(const unsigned char *raw, size_t length, uint32_t warmup, BoplRlConfig *config,
                        char game_source[65], char *why, size_t capacity) {
    if (length < 150 || memcmp(raw, "BOPLREP2", 8)) { snprintf(why, capacity, "not a BOPLREP2 archive"); return -1; }
    memcpy(game_source, raw + 8, 64); game_source[64] = 0;
    size_t at = 136;
#define TAKE(width) take(raw, length, &at, (width))
    bopl_rl_default_config(config, 2);
    config->seed = TAKE(4);
    config->map = TAKE(1);
    config->mode = TAKE(1);
    config->stocks = TAKE(4);
    uint32_t players = TAKE(1);
    if (players < 1 || players > 6) { snprintf(why, capacity, "invalid player count %u", players); return -1; }
    config->players = players;
    config->warmup_ticks = warmup;
    config->max_episode_ticks = SHADOW_ROUND_LIMIT;
    for (uint32_t p = 0; p < players; p++) {
        (void)TAKE(1); /* id */
        config->player[p].color = TAKE(1);
        config->player[p].team = TAKE(1);
        config->player[p].keyboard = TAKE(1);
        uint32_t slots = TAKE(1);
        if (slots < 1 || slots > 3) { snprintf(why, capacity, "unsupported slot count %u", slots); return -1; }
        memset(config->player[p].abilities, 0, sizeof config->player[p].abilities);
        for (uint32_t k = 0; k < slots; k++) config->player[p].abilities[k] = TAKE(2);
        config->slots = slots;
    }
#undef TAKE
    if (at > length) { snprintf(why, capacity, "truncated archive setup"); return -1; }
    return 0;
}

int bopl_shadow_begin(BoplShadow *s, const void *archive, size_t length, uint32_t warmup) {
    if (!s) return BOPL_SHADOW_FAILED;
    if (s->reset_pending) { s->reset_pending = 0; (void)bopl_rl_reset_finish(s->world); }
    s->status = BOPL_SHADOW_TRACKING;
    s->steps = s->compared = 0; s->sim_ended = 0; s->baseline_set = 0;
    memset(&s->compare, 0, sizeof s->compare);
    s->report[0] = 0;
    if (!archive) return fail(s, "shadow: no round archive%s", NULL);
    BoplRlConfig config;
    char game_source[65], why[256];
    if (decode_setup(archive, length, warmup, &config, game_source, why, sizeof why)) return fail(s, "shadow: round setup: %s", why);
    const char *sim_source = bopl_rl_source_hash(s->world);
    if (!sim_source || strcmp(sim_source, game_source)) {
        const char *accepted = getenv("BOPL_AGENT_SHADOW_SOURCE");
        if (!accepted || !*accepted) accepted = BOPL_SHADOW_RUST_SOURCE;
        int placeholder = sim_source && !strcmp(sim_source, RUST_PLACEHOLDER);
        if (!placeholder || (strcmp(accepted, "any") && strcmp(accepted, game_source))) {
            snprintf(s->report, sizeof s->report,
                     "shadow refused: game source %s, shadow %s reports %s%s%s", game_source, s->library,
                     sim_source ? sim_source : "(none)",
                     placeholder ? " (the Rust port's placeholder; it is checked against " : "",
                     placeholder ? accepted : "");
            if (placeholder) strncat(s->report, ")", sizeof s->report - strlen(s->report) - 1);
            s->status = BOPL_SHADOW_FAILED;
            return s->status;
        }
    }
    if (bopl_rl_reset_begin(s->world, &config) != BOPL_RL_OK)
        return fail(s, "shadow reset rejected the round setup: %s", bopl_rl_error(s->world));
    s->reset_pending = 1;
    s->players = config.players;
    s->warmup = warmup;
    snprintf(s->report, sizeof s->report, "shadow tracking: map %u seed %u players %u stocks %u mode %u, %s",
             config.map, config.seed, config.players, config.stocks, config.mode, s->library);
    return s->status;
}

static void describe(char *out, size_t capacity, const char *kind, const BoplDivergence *d) {
    if (!d->found) { out[0] = 0; return; }
    if (d->numeric)
        snprintf(out, capacity, "%s %s game=%.9g shadow=%.9g (raw %" PRId64 " vs %" PRId64 ") [%s]", kind, d->field,
                 bopl_cmp_units(d->left), bopl_cmp_units(d->right), d->left, d->right, d->context);
    else
        snprintf(out, capacity, "%s %s game=%" PRId64 " shadow=%" PRId64 " [%s]", kind, d->field, d->left, d->right,
                 d->context);
}

int bopl_shadow_check(BoplShadow *s, const BoplRlState *game) {
    if (!s || !game) return BOPL_SHADOW_FAILED;
    if (s->status != BOPL_SHADOW_TRACKING) return s->status;
    if (finish_reset(s)) return s->status;
    const BoplRlState *shadow = bopl_rl_state(s->world);
    if (!shadow) return fail(s, "shadow has no state: %s", bopl_rl_error(s->world));
    unsigned tick = game->header.episode_ticks;
    uint32_t players = game->header.player_count < 6 ? game->header.player_count : 6;
    if (!s->baseline_set) {
        for (uint32_t i = 0; i < players; i++) { s->base_kills[i] = game->players[i].kills; s->base_deaths[i] = game->players[i].deaths; }
        s->hierarchy_offset = 0;
        uint32_t common = game->header.entity_count < shadow->header.entity_count ? game->header.entity_count
                                                                                   : shadow->header.entity_count;
        for (uint32_t i = 0; i < common; i++)
            if (game->entities[i].hierarchy >= PREFAB_HIERARCHY && shadow->entities[i].hierarchy >= PREFAB_HIERARCHY) {
                s->hierarchy_offset = (int64_t)game->entities[i].hierarchy - shadow->entities[i].hierarchy;
                break;
            }
        s->baseline_set = 1;
    }
    BoplRlState relative = *game;
    if (s->hierarchy_offset && game->header.entity_count) {
        uint32_t count = game->header.entity_count;
        if (game->entities != s->entities) { /* caller's buffer: work on a copy */
            if (count > s->ecap) {
                free(s->entities); s->entities = calloc(count, sizeof(BoplRlEntity));
                s->ecap = s->entities ? count : 0;
                if (!s->entities) return fail(s, "shadow: out of memory%s", "");
            }
            memcpy(s->entities, game->entities, (size_t)count * sizeof(BoplRlEntity));
            relative.entities = s->entities;
        }
        /* relative.entities is s->entities now (check_bytes decodes into it). */
        for (uint32_t i = 0; i < count; i++)
            if (s->entities[i].hierarchy >= PREFAB_HIERARCHY)
                s->entities[i].hierarchy = (uint32_t)((int64_t)s->entities[i].hierarchy - s->hierarchy_offset);
    }
    for (uint32_t i = 0; i < players; i++) {
        relative.players[i].kills -= s->base_kills[i];
        relative.players[i].deaths -= s->base_deaths[i];
    }
    game = &relative;
    if (!bopl_compare_state(game, shadow, tick, &s->options, &s->compare)) { s->compared++; return s->status; }
    s->status = BOPL_SHADOW_DESYNC;
    char discrete[512], numeric[512];
    describe(discrete, sizeof discrete, "discrete", &s->compare.discrete);
    describe(numeric, sizeof numeric, "numeric", &s->compare.numeric);
    int used = snprintf(s->report, sizeof s->report,
                        "shadow DESYNC at episode tick %u (game SimulationTicks %u, shadow episode tick %u, %u ticks matched): %s%s%s",
                        tick, game->header.tick, shadow->header.episode_ticks, s->compared, discrete,
                        discrete[0] && numeric[0] ? "; " : "", numeric);
    /* Every differing field of this tick, for the log. */
    char *all = NULL; size_t all_length = 0;
    FILE *stream = open_memstream(&all, &all_length);
    if (stream) {
        BoplCompareOptions every = s->options;
        BoplCompare scratch;
        memset(&scratch, 0, sizeof scratch);
        every.report_all = 1; every.all_out = stream;
        bopl_compare_state(game, shadow, tick, &every, &scratch);
        fclose(stream);
        if (all && used > 0 && (size_t)used < sizeof s->report)
            snprintf(s->report + used, sizeof s->report - (size_t)used, "\n  every differing field:\n%s", all);
        free(all);
    }
    return s->status;
}

int bopl_shadow_check_bytes(BoplShadow *s, const void *bytes, size_t length) {
    if (!s || !bytes) return -2;
    if (s->status != BOPL_SHADOW_TRACKING) return s->status;
    if (length < sizeof(BoplRlHeader)) return -2;
    const unsigned char *p = bytes;
    BoplRlState *g = &s->game;
    memcpy(&g->header, p, sizeof(BoplRlHeader)); p += sizeof(BoplRlHeader);
    uint32_t players = g->header.player_count, entities = g->header.entity_count, slimes = g->header.slime_count;
    if (players > 6) return -2;
    size_t need = sizeof(BoplRlHeader) + (size_t)players * sizeof(BoplRlPlayer) + (size_t)entities * sizeof(BoplRlEntity)
                  + (size_t)slimes * sizeof(BoplRlSlime) + (size_t)players * sizeof(BoplRlTelemetry);
    if (length != need) return -2;
    memset(g->players, 0, sizeof g->players);
    memset(g->telemetry, 0, sizeof g->telemetry);
    memcpy(g->players, p, (size_t)players * sizeof(BoplRlPlayer)); p += (size_t)players * sizeof(BoplRlPlayer);
    if (entities > s->ecap) {
        free(s->entities); s->entities = calloc(entities, sizeof(BoplRlEntity));
        s->ecap = s->entities ? entities : 0;
        if (!s->entities) return fail(s, "shadow: out of memory%s", NULL);
    }
    if (slimes > s->scap) {
        free(s->slimes); s->slimes = calloc(slimes, sizeof(BoplRlSlime));
        s->scap = s->slimes ? slimes : 0;
        if (!s->slimes) return fail(s, "shadow: out of memory%s", NULL);
    }
    if (entities) memcpy(s->entities, p, (size_t)entities * sizeof(BoplRlEntity));
    p += (size_t)entities * sizeof(BoplRlEntity);
    if (slimes) memcpy(s->slimes, p, (size_t)slimes * sizeof(BoplRlSlime));
    p += (size_t)slimes * sizeof(BoplRlSlime);
    memcpy(g->telemetry, p, (size_t)players * sizeof(BoplRlTelemetry));
    g->entities = s->entities; g->slimes = s->slimes;
    return bopl_shadow_check(s, g);
}

int bopl_shadow_step(BoplShadow *s, const BoplRlAction *actions, size_t count) {
    if (!s || !actions) return BOPL_SHADOW_FAILED;
    if (s->status != BOPL_SHADOW_TRACKING) return s->status;
    if (finish_reset(s)) return s->status;
    if (count != s->players) {
        char detail[64];
        snprintf(detail, sizeof detail, "%zu packets for %u players", count, s->players);
        return fail(s, "shadow step: %s", detail);
    }
    if (s->sim_ended) { s->steps++; return s->status; }
    int result = bopl_rl_step(s->world, actions, count, 1);
    if (result == BOPL_RL_ENDED) { s->sim_ended = 1; s->steps++; return s->status; }
    if (result != BOPL_RL_OK) return fail(s, "shadow step failed: %s", bopl_rl_error(s->world));
    s->steps++;
    return s->status;
}

int bopl_shadow_end(BoplShadow *s) {
    if (!s) return BOPL_SHADOW_FAILED;
    if (s->reset_pending) { s->reset_pending = 0; (void)bopl_rl_reset_finish(s->world); }
    if (s->status == BOPL_SHADOW_TRACKING) {
        s->status = BOPL_SHADOW_ENDED;
        snprintf(s->report, sizeof s->report, "shadow in sync for the whole round: %u ticks compared, %u stepped",
                 s->compared, s->steps);
    }
    return s->status;
}

int bopl_shadow_status(const BoplShadow *s) { return s ? s->status : BOPL_SHADOW_FAILED; }
const char *bopl_shadow_status_name(int status) {
    switch (status) {
    case BOPL_SHADOW_OFF: return "off";
    case BOPL_SHADOW_TRACKING: return "tracking";
    case BOPL_SHADOW_DESYNC: return "desync";
    case BOPL_SHADOW_ENDED: return "ended";
    case BOPL_SHADOW_FAILED: return "failed";
    default: return "?";
    }
}
const char *bopl_shadow_report(const BoplShadow *s) { return s ? s->report : ""; }
uint32_t bopl_shadow_compared(const BoplShadow *s) { return s ? s->compared : 0; }
uint32_t bopl_shadow_episode_tick(const BoplShadow *s) { return s ? s->steps : 0; }
BoplRl *bopl_shadow_world(BoplShadow *s) {
    if (!s || s->status != BOPL_SHADOW_TRACKING || finish_reset(s)) return NULL;
    return s->world;
}
const BoplRlState *bopl_shadow_state(BoplShadow *s) {
    BoplRl *world = bopl_shadow_world(s);
    return world ? bopl_rl_state(world) : NULL;
}
const char *bopl_shadow_library(const BoplShadow *s) { return s ? s->library : ""; }
void bopl_shadow_close(BoplShadow *s) {
    if (!s) return;
    bopl_rl_close(s->world);
    free(s->entities); free(s->slimes);
    free(s);
}
