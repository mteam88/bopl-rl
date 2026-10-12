#ifndef BOPL_RL_COMPARE_H
#define BOPL_RL_COMPARE_H
/* Field-by-field comparison of two BoplRlState records, header-only.
 *
 * This is eval/rust_ab/replay_ab.c's compare_state, moved here so the in-game
 * shadow (train/bopl_shadow.c) checks the running game against its shadow sim
 * with exactly the comparator the replay differential uses. replay_ab.c
 * includes this header; its output is unchanged.
 *
 * Divergences come in two classes:
 *
 *   DISCRETE - an integer, flag, count, identity, cause or entity-population
 *              field differs. The two simulations took different decisions.
 *   NUMERIC  - a fixed-point quantity differs by more than the tolerance.
 *
 * Options (BoplCompareOptions):
 *   tolerance     numeric tolerance in raw Q32.32 units (0 = bit-exact)
 *   strict        1 = drop the inactive-prefab ENABLED-bit exclusion
 *   visible_only  1 = compare only what the v4 observation writer can read:
 *                 entity rows that are ACTIVE and not DESTROYED
 *                 (bopl_puffer.h:316), without `id` or `hierarchy`, and the
 *                 visible entity count instead of the raw one
 *   report_all    1 = print every differing field (to all_out, default
 *                 stdout), not only the first
 *   ignore        comma-separated generalized field names to skip,
 *                 e.g. "entity[].flags,slime[].timers[]"
 *   flags_mask    header.flags bits compared (0 = all). The game-side wrapper
 *                 (RlAgent.cs) and the sims set BOPL_RL_STARTED at different
 *                 ticks, so the shadow compares BOPL_RL_TERMINAL only.
 *   no_telemetry  1 = skip the telemetry[] counters (research bookkeeping that
 *                 RlAgent does not count; not game state). */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    int found;
    unsigned tick;          /* episode tick (0 = state right after reset) */
    char field[96];         /* e.g. "entity[7].x" */
    char context[256];      /* identifying detail for the responsible system */
    int64_t left, right;    /* reference, candidate */
    int numeric;            /* 1 when the field is fixed point */
} BoplDivergence;

typedef struct {
    BoplDivergence discrete, numeric;
    int64_t max_gap;          /* largest raw gap seen */
    unsigned max_gap_tick;
    unsigned compared_ticks;
} BoplCompare;

typedef struct {
    int64_t tolerance;
    int strict, visible_only, report_all, no_telemetry;
    const char *ignore;
    uint32_t flags_mask;
    FILE *all_out;            /* report_all destination (NULL = stdout) */
} BoplCompareOptions;

static inline int64_t bopl_cmp_magnitude(int64_t a, int64_t b) { return a > b ? a - b : b - a; }
static inline double bopl_cmp_units(int64_t raw) { return (double)raw / 4294967296.0; }

/* Collapse "entity[7].x" to "entity[].x" so the same system aggregates. */
static inline void bopl_cmp_generalize(const char *field, char *out, size_t capacity) {
    size_t o = 0;
    for (size_t i = 0; field[i] && o + 1 < capacity; i++) {
        out[o++] = field[i];
        if (field[i] == '[') { while (field[i + 1] && field[i + 1] != ']') i++; }
    }
    out[o] = 0;
}

static inline int bopl_cmp_ignored(const char *ignore, const char *field) {
    if (!ignore || !*ignore) return 0;
    char key[96];
    bopl_cmp_generalize(field, key, sizeof key);
    size_t length = strlen(key);
    for (const char *at = ignore; at && *at;) {
        const char *end = strchr(at, ',');
        size_t span = end ? (size_t)(end - at) : strlen(at);
        if (span == length && !strncmp(at, key, span)) return 1;
        at = end ? end + 1 : NULL;
    }
    return 0;
}

/* The observation writer's entity filter (bopl_puffer.h:316-317): a row that
 * is not ACTIVE, or is DESTROYED, is never encoded, so nothing it carries can
 * reach a policy. `entity[].id` and `entity[].hierarchy` are not read by the
 * v4 writer at all. */
static inline int bopl_cmp_visible(const BoplRlEntity *e) {
    return (e->flags & BOPL_ENTITY_ACTIVE) && !(e->flags & BOPL_ENTITY_DESTROYED);
}

static inline void bopl_cmp_record(BoplDivergence *d, unsigned tick, const char *field, const char *context,
                                   int64_t left, int64_t right, int numeric) {
    if (d->found) return;
    d->found = 1;
    d->tick = tick;
    d->numeric = numeric;
    d->left = left;
    d->right = right;
    snprintf(d->field, sizeof d->field, "%s", field);
    snprintf(d->context, sizeof d->context, "%s", context ? context : "");
}

#define BOPL_CMP_DISCRETE(field, context, l, r)                               \
    do {                                                                      \
        if ((int64_t)(l) != (int64_t)(r) && !bopl_cmp_ignored(o->ignore, field)) { \
            if (o->report_all)                                                \
                fprintf(o->all_out ? o->all_out : stdout, "      all      tick %5u %-34s %" PRId64 " vs %" PRId64 "   (%s)\n", \
                       tick, field, (int64_t)(l), (int64_t)(r), context);     \
            bopl_cmp_record(&out->discrete, tick, field, context, (int64_t)(l), (int64_t)(r), 0); \
            diverged = 1;                                                     \
        }                                                                     \
    } while (0)

#define BOPL_CMP_NUMERIC(field, context, l, r)                                \
    do {                                                                      \
        int64_t gap_ = bopl_cmp_magnitude((int64_t)(l), (int64_t)(r));        \
        if (gap_ > out->max_gap) { out->max_gap = gap_; out->max_gap_tick = tick; } \
        if (gap_ > o->tolerance && !bopl_cmp_ignored(o->ignore, field)) {     \
            if (o->report_all)                                                \
                fprintf(o->all_out ? o->all_out : stdout, "      all      tick %5u %-34s %.9g vs %.9g   (%s; raw gap %" PRId64 ")\n",\
                       tick, field, bopl_cmp_units((int64_t)(l)), bopl_cmp_units((int64_t)(r)), context, gap_); \
            bopl_cmp_record(&out->numeric, tick, field, context, (int64_t)(l), (int64_t)(r), 1); \
            diverged = 1;                                                     \
        }                                                                     \
    } while (0)

/* Returns 1 when anything differs at this tick. a is the reference (the game,
 * for the shadow), b the candidate. */
static inline int bopl_compare_state(const BoplRlState *a, const BoplRlState *b, unsigned tick,
                                     const BoplCompareOptions *o, BoplCompare *out) {
    int diverged = 0;
    char field[96], context[256];
    const BoplRlHeader *ha = &a->header, *hb = &b->header;
    uint32_t mask = o->flags_mask ? o->flags_mask : 0xFFFFFFFFu;

    BOPL_CMP_DISCRETE("header.player_count", "", ha->player_count, hb->player_count);
    if (o->visible_only) {
        uint32_t va = 0, vb = 0;
        for (uint32_t i = 0; i < ha->entity_count; i++) va += (uint32_t)bopl_cmp_visible(&a->entities[i]);
        for (uint32_t i = 0; i < hb->entity_count; i++) vb += (uint32_t)bopl_cmp_visible(&b->entities[i]);
        BOPL_CMP_DISCRETE("header.visible_entities", "entity population (lifecycle)", va, vb);
    } else {
        BOPL_CMP_DISCRETE("header.entity_count", "entity population (lifecycle)", ha->entity_count, hb->entity_count);
    }
    BOPL_CMP_DISCRETE("header.slime_count", "slime population (clones/stocks)", ha->slime_count, hb->slime_count);
    BOPL_CMP_DISCRETE("header.flags", "terminal/truncated/started/failed bits", ha->flags & mask, hb->flags & mask);
    BOPL_CMP_DISCRETE("header.episode_ticks", "", ha->episode_ticks, hb->episode_ticks);
    BOPL_CMP_DISCRETE("header.sudden_death", "session/sudden-death", ha->sudden_death, hb->sudden_death);
    BOPL_CMP_DISCRETE("header.warmup_ticks", "", ha->warmup_ticks, hb->warmup_ticks);
    BOPL_CMP_NUMERIC("header.simulation_time", "clock", ha->simulation_time, hb->simulation_time);
    BOPL_CMP_NUMERIC("header.player_time_scale", "time scale (Time Stop)", ha->player_time_scale, hb->player_time_scale);
    BOPL_CMP_NUMERIC("header.simulation_time_since_level_loaded", "clock", ha->simulation_time_since_level_loaded,
                     hb->simulation_time_since_level_loaded);
    if (diverged && !o->report_all) return 1;
    /* Rows are walked up to the shorter side (players are a fixed array of 6;
     * entities/slimes are pointers that must not be read past their count). */
    uint32_t player_rows = ha->player_count < 6 ? ha->player_count : 6;
    uint32_t slime_rows = ha->slime_count < hb->slime_count ? ha->slime_count : hb->slime_count;
    uint32_t entity_rows = ha->entity_count < hb->entity_count ? ha->entity_count : hb->entity_count;

    for (uint32_t i = 0; i < player_rows; i++) {
        const BoplRlPlayer *pa = &a->players[i], *pb = &b->players[i];
        snprintf(context, sizeof context, "player %u id %u team %u", i, pa->id, pa->team);
        snprintf(field, sizeof field, "player[%u].id", i);          BOPL_CMP_DISCRETE(field, context, pa->id, pb->id);
        snprintf(field, sizeof field, "player[%u].team", i);        BOPL_CMP_DISCRETE(field, context, pa->team, pb->team);
        snprintf(field, sizeof field, "player[%u].flags", i);       BOPL_CMP_DISCRETE(field, context, pa->flags, pb->flags);
        snprintf(field, sizeof field, "player[%u].alive_bodies", i);BOPL_CMP_DISCRETE(field, context, pa->alive_bodies, pb->alive_bodies);
        snprintf(field, sizeof field, "player[%u].stocks", i);      BOPL_CMP_DISCRETE(field, context, pa->stocks, pb->stocks);
        snprintf(field, sizeof field, "player[%u].kills", i);       BOPL_CMP_DISCRETE(field, context, pa->kills, pb->kills);
        snprintf(field, sizeof field, "player[%u].deaths", i);      BOPL_CMP_DISCRETE(field, context, pa->deaths, pb->deaths);
        snprintf(field, sizeof field, "player[%u].cause_of_death", i);
        BOPL_CMP_DISCRETE(field, context, pa->cause_of_death, pb->cause_of_death);
        snprintf(field, sizeof field, "player[%u].x", i);           BOPL_CMP_NUMERIC(field, context, pa->x, pb->x);
        snprintf(field, sizeof field, "player[%u].y", i);           BOPL_CMP_NUMERIC(field, context, pa->y, pb->y);
        snprintf(field, sizeof field, "player[%u].scale", i);       BOPL_CMP_NUMERIC(field, context, pa->scale, pb->scale);
        snprintf(field, sizeof field, "player[%u].invisible_time", i);
        BOPL_CMP_NUMERIC(field, context, pa->invisible_time, pb->invisible_time);
        snprintf(field, sizeof field, "player[%u].ability_time", i);
        BOPL_CMP_NUMERIC(field, context, pa->ability_time, pb->ability_time);
        if (o->no_telemetry) continue;
        const BoplRlTelemetry *ta = &a->telemetry[i], *tb = &b->telemetry[i];
        for (int s = 0; s < 3; s++) {
            snprintf(field, sizeof field, "telemetry[%u].ability_activations[%d]", i, s);
            BOPL_CMP_DISCRETE(field, context, ta->ability_activations[s], tb->ability_activations[s]);
        }
        snprintf(field, sizeof field, "telemetry[%u].arrows_fired", i);   BOPL_CMP_DISCRETE(field, context, ta->arrows_fired, tb->arrows_fired);
        snprintf(field, sizeof field, "telemetry[%u].arrow_hits", i);     BOPL_CMP_DISCRETE(field, context, ta->arrow_hits, tb->arrow_hits);
        snprintf(field, sizeof field, "telemetry[%u].deaths_environment", i);
        BOPL_CMP_DISCRETE(field, context, ta->deaths_environment, tb->deaths_environment);
        snprintf(field, sizeof field, "telemetry[%u].deaths_opponent", i);
        BOPL_CMP_DISCRETE(field, context, ta->deaths_opponent, tb->deaths_opponent);
        snprintf(field, sizeof field, "telemetry[%u].deaths_self", i);
        BOPL_CMP_DISCRETE(field, context, ta->deaths_self, tb->deaths_self);
        snprintf(field, sizeof field, "telemetry[%u].deaths_other", i);
        BOPL_CMP_DISCRETE(field, context, ta->deaths_other, tb->deaths_other);
    }

    for (uint32_t i = 0; i < slime_rows; i++) {
        const BoplRlSlime *sa = &a->slimes[i], *sb = &b->slimes[i];
        snprintf(context, sizeof context, "slime %u owner %u abilities %u/%u/%u flags %u vs %u state %u vs %u",
                 i, sa->owner, sa->abilities[0], sa->abilities[1], sa->abilities[2],
                 sa->flags, sb->flags, sa->state, sb->state);
        snprintf(field, sizeof field, "slime[%u].id", i);      BOPL_CMP_DISCRETE(field, context, sa->id, sb->id);
        snprintf(field, sizeof field, "slime[%u].owner", i);   BOPL_CMP_DISCRETE(field, context, sa->owner, sb->owner);
        snprintf(field, sizeof field, "slime[%u].flags", i);   BOPL_CMP_DISCRETE(field, context, sa->flags, sb->flags);
        snprintf(field, sizeof field, "slime[%u].state", i);   BOPL_CMP_DISCRETE(field, context, sa->state, sb->state);
        for (int s = 0; s < 3; s++) {
            snprintf(field, sizeof field, "slime[%u].abilities[%d]", i, s);
            BOPL_CMP_DISCRETE(field, context, sa->abilities[s], sb->abilities[s]);
            snprintf(field, sizeof field, "slime[%u].ability_flags[%d]", i, s);
            BOPL_CMP_DISCRETE(field, context, sa->ability_flags[s], sb->ability_flags[s]);
            snprintf(field, sizeof field, "slime[%u].cooldowns[%d]", i, s);
            BOPL_CMP_NUMERIC(field, context, sa->cooldowns[s], sb->cooldowns[s]);
            snprintf(field, sizeof field, "slime[%u].timers[%d]", i, s);
            BOPL_CMP_NUMERIC(field, context, sa->timers[s], sb->timers[s]);
        }
        snprintf(field, sizeof field, "slime[%u].x", i);        BOPL_CMP_NUMERIC(field, context, sa->x, sb->x);
        snprintf(field, sizeof field, "slime[%u].y", i);        BOPL_CMP_NUMERIC(field, context, sa->y, sb->y);
        snprintf(field, sizeof field, "slime[%u].vx", i);       BOPL_CMP_NUMERIC(field, context, sa->vx, sb->vx);
        snprintf(field, sizeof field, "slime[%u].vy", i);       BOPL_CMP_NUMERIC(field, context, sa->vy, sb->vy);
        snprintf(field, sizeof field, "slime[%u].rotation", i); BOPL_CMP_NUMERIC(field, context, sa->rotation, sb->rotation);
        snprintf(field, sizeof field, "slime[%u].scale", i);    BOPL_CMP_NUMERIC(field, context, sa->scale, sb->scale);
    }

    for (uint32_t i = 0; i < entity_rows; i++) {
        const BoplRlEntity *ea = &a->entities[i], *eb = &b->entities[i];
        if (o->visible_only && !bopl_cmp_visible(ea) && !bopl_cmp_visible(eb)) continue;
        snprintf(context, sizeof context,
                 "entity %u kind %u/%u layer %u/%u owner %u/%u source-ability %u/%u flags %u/%u",
                 i, ea->kind, eb->kind, ea->layer, eb->layer, ea->owner, eb->owner,
                 ea->reserved[0], eb->reserved[0], ea->flags, eb->flags);
        if (!o->visible_only) {
            snprintf(field, sizeof field, "entity[%u].id", i);    BOPL_CMP_DISCRETE(field, context, ea->id, eb->id);
        }
        snprintf(field, sizeof field, "entity[%u].kind", i);      BOPL_CMP_DISCRETE(field, context, ea->kind, eb->kind);
        snprintf(field, sizeof field, "entity[%u].layer", i);     BOPL_CMP_DISCRETE(field, context, ea->layer, eb->layer);
        snprintf(field, sizeof field, "entity[%u].owner", i);     BOPL_CMP_DISCRETE(field, context, ea->owner, eb->owner);
        /* An inactive prefab (no BOPL_ENTITY_ACTIVE) reports its component's
         * inert BOPL_ENTITY_ENABLED bit differently in the two ports; the
         * observation writer drops every inactive row (bopl_obs_v4.h:141), so
         * that bit cannot reach a policy. strict compares it. */
        uint32_t flags_a = ea->flags, flags_b = eb->flags;
        if (!o->strict && !(flags_a & BOPL_ENTITY_ACTIVE) && !(flags_b & BOPL_ENTITY_ACTIVE)) {
            flags_a &= ~(uint32_t)BOPL_ENTITY_ENABLED;
            flags_b &= ~(uint32_t)BOPL_ENTITY_ENABLED;
        }
        snprintf(field, sizeof field, "entity[%u].flags", i);     BOPL_CMP_DISCRETE(field, context, flags_a, flags_b);
        if (!o->visible_only) {
            snprintf(field, sizeof field, "entity[%u].hierarchy", i);
            BOPL_CMP_DISCRETE(field, context, ea->hierarchy, eb->hierarchy);
        }
        snprintf(field, sizeof field, "entity[%u].source_ability", i);
        BOPL_CMP_DISCRETE(field, context, ea->reserved[0], eb->reserved[0]);
        snprintf(field, sizeof field, "entity[%u].timer_ticks", i);
        BOPL_CMP_DISCRETE(field, context, ea->reserved[1], eb->reserved[1]);
        snprintf(field, sizeof field, "entity[%u].x", i);         BOPL_CMP_NUMERIC(field, context, ea->x, eb->x);
        snprintf(field, sizeof field, "entity[%u].y", i);         BOPL_CMP_NUMERIC(field, context, ea->y, eb->y);
        snprintf(field, sizeof field, "entity[%u].vx", i);        BOPL_CMP_NUMERIC(field, context, ea->vx, eb->vx);
        snprintf(field, sizeof field, "entity[%u].vy", i);        BOPL_CMP_NUMERIC(field, context, ea->vy, eb->vy);
        snprintf(field, sizeof field, "entity[%u].right_x", i);   BOPL_CMP_NUMERIC(field, context, ea->right_x, eb->right_x);
        snprintf(field, sizeof field, "entity[%u].right_y", i);   BOPL_CMP_NUMERIC(field, context, ea->right_y, eb->right_y);
        snprintf(field, sizeof field, "entity[%u].up_x", i);      BOPL_CMP_NUMERIC(field, context, ea->up_x, eb->up_x);
        snprintf(field, sizeof field, "entity[%u].up_y", i);      BOPL_CMP_NUMERIC(field, context, ea->up_y, eb->up_y);
        snprintf(field, sizeof field, "entity[%u].radius", i);    BOPL_CMP_NUMERIC(field, context, ea->radius, eb->radius);
        snprintf(field, sizeof field, "entity[%u].rotation", i);  BOPL_CMP_NUMERIC(field, context, ea->rotation, eb->rotation);
        snprintf(field, sizeof field, "entity[%u].angular_velocity", i);
        BOPL_CMP_NUMERIC(field, context, ea->angular_velocity, eb->angular_velocity);
        snprintf(field, sizeof field, "entity[%u].inverse_mass", i);
        BOPL_CMP_NUMERIC(field, context, ea->inverse_mass, eb->inverse_mass);
        snprintf(field, sizeof field, "entity[%u].clock_rate", i);
        BOPL_CMP_NUMERIC(field, context, ea->clock_rate, eb->clock_rate);
    }
    return diverged;
}
#undef BOPL_CMP_DISCRETE
#undef BOPL_CMP_NUMERIC
#endif
