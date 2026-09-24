#ifndef DIVERGENCE_COMPARE_H
#define DIVERGENCE_COMPARE_H
/* Field-by-field state comparison between two bopl_rl.h backends, recording
 * the FIRST tick at which they disagree rather than stopping at it.
 *
 * The field set, the tolerance semantics, the inactive-prefab ENABLED-bit
 * exclusion and the policy-visible lens are the same ones
 * eval/rust_ab/replay_ab.c uses, and the two must stay in step: this
 * header is the same comparison rewritten so that a tick with no divergence
 * costs no string formatting at all (replay_ab.c snprintf's every field name
 * of every entity on every tick, which is affordable for a few hundred
 * replays and not for a hundred thousand games). Every field name, every
 * exclusion and the DISCRETE/NUMERIC split below are copied from it verbatim;
 * divergence_selfcheck proves the two agree on real captures.
 *
 * DISCRETE - an integer, flag, count, identity, cause or population field
 *            differs: the two simulations took different decisions.
 * NUMERIC  - a fixed-point quantity differs by more than the tolerance.
 */
#include "bopl_rl.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* One recorded divergence. */
typedef struct {
    int found;
    unsigned tick;
    int discrete;             /* 1 discrete, 0 numeric */
    char field[64];           /* "entity[7].x" */
    char context[192];
    int64_t left, right;      /* reference, candidate */
    int source_ability;       /* ability id responsible, -1 when not attributable */
    int entity_kind;          /* entity kind for entity[] fields, -1 otherwise */
} DrHit;

/* Per (arm, lens) accumulator. `any` is the first divergence of either class,
 * `disc` the first discrete one; a combo is complete when both are found. */
typedef struct {
    DrHit any, disc;
    /* The same two, restricted to fields not named in dr_ignore_list. A
     * divergence that only a field on that list reports is structurally
     * different from one the physics reports, and hiding it inside the
     * headline curve would be the same mistake as excluding it silently:
     * both are recorded, and the report carries both curves. */
    DrHit any_ex, disc_ex;
    int64_t max_gap;
    unsigned max_gap_tick;
} DrCombo;

/* Comma-separated generalized field names ("entity[].hierarchy"), as
 * REPLAY_AB_IGNORE. */
static const char *dr_ignore_list = NULL;

static int dr_ignored(const char *field);

static int dr_obs_visible(const BoplRlEntity *e) {
    return (e->flags & BOPL_ENTITY_ACTIVE) && !(e->flags & BOPL_ENTITY_DESTROYED);
}
static double dr_units(int64_t raw) { return (double)raw / 4294967296.0; }
static int64_t dr_magnitude(int64_t a, int64_t b) { return a > b ? a - b : b - a; }

/* Collapse "entity[7].x" to "entity[].x" so the same system aggregates. */
static void dr_generalize(const char *field, char *out, size_t capacity) {
    size_t o = 0;
    for (size_t i = 0; field[i] && o + 1 < capacity; i++) {
        out[o++] = field[i];
        if (field[i] == '[') { while (field[i + 1] && field[i + 1] != ']') i++; }
    }
    out[o] = 0;
}

typedef enum { DR_CTX_HEADER, DR_CTX_PLAYER, DR_CTX_SLIME, DR_CTX_ENTITY } DrCtxKind;

/* Everything the comparison needs that is not one of the two states. */
typedef struct {
    const BoplRlState *a, *b;
    int64_t tolerance;
    unsigned tick;
    int visible_only;
    DrCombo *out;
    int diverged;             /* set when this tick differs at all */
    unsigned ctx_ib;          /* b-side entity row, which the compacted visible
                               * lens lets drift from the a-side row */
} DrPass;

/* 1 = compare the visible lens index-aligned, like replay_ab.c, instead of
 * comparing the compacted visible sequences. */
static int dr_visible_aligned = 0;

/* Build the identifying context lazily - only a field that actually differs
 * pays for it. Also resolves the ability and entity kind the row belongs to,
 * which is what the ability/subsystem attribution is keyed on. */
static void dr_context(DrPass *p, DrCtxKind kind, unsigned index, char *out, size_t capacity,
                       int *source_ability, int *entity_kind) {
    const BoplRlState *a = p->a, *b = p->b;
    *source_ability = -1;
    *entity_kind = -1;
    switch (kind) {
    case DR_CTX_HEADER:
        snprintf(out, capacity, "header");
        break;
    case DR_CTX_PLAYER: {
        const BoplRlPlayer *pa = &a->players[index];
        snprintf(out, capacity, "player %u id %u team %u", index, pa->id, pa->team);
        break;
    }
    case DR_CTX_SLIME: {
        const BoplRlSlime *sa = &a->slimes[index], *sb = &b->slimes[index];
        snprintf(out, capacity, "slime %u owner %u abilities %u/%u/%u flags %u vs %u state %u vs %u",
                 index, sa->owner, sa->abilities[0], sa->abilities[1], sa->abilities[2],
                 sa->flags, sb->flags, sa->state, sb->state);
        break;
    }
    case DR_CTX_ENTITY: {
        const BoplRlEntity *ea = &a->entities[index], *eb = &b->entities[p->ctx_ib];
        snprintf(out, capacity,
                 "entity %u/%u kind %u/%u layer %u/%u owner %u/%u source-ability %u/%u flags %u/%u",
                 index, p->ctx_ib, ea->kind, eb->kind, ea->layer, eb->layer, ea->owner, eb->owner,
                 ea->reserved[0], eb->reserved[0], ea->flags, eb->flags);
        *source_ability = (int)ea->reserved[0];
        *entity_kind = (int)ea->kind;
        break;
    }
    }
}

/* Record one differing field. Called only when the values actually differ. */
static void dr_hit(DrPass *p, int discrete, const char *fmt, unsigned index,
                   DrCtxKind kind, unsigned ctx_index, int64_t left, int64_t right,
                   int slot_ability) {
    p->diverged = 1;
    DrCombo *c = p->out;
    int want = (!c->any.found || (discrete && !c->disc.found));
    int want_ex = (!c->any_ex.found || (discrete && !c->disc_ex.found));
    if (!want && !want_ex) return;
    char field[64], context[192];
    int source_ability, entity_kind;
    snprintf(field, sizeof field, fmt, index);
    dr_context(p, kind, ctx_index, context, sizeof context, &source_ability, &entity_kind);
    if (slot_ability >= 0) source_ability = slot_ability;
    if (want_ex && dr_ignored(field)) want_ex = 0;
    if (want && !c->any.found) {
        c->any.found = 1; c->any.tick = p->tick; c->any.discrete = discrete;
        c->any.left = left; c->any.right = right;
        c->any.source_ability = source_ability; c->any.entity_kind = entity_kind;
        memcpy(c->any.field, field, sizeof c->any.field);
        memcpy(c->any.context, context, sizeof c->any.context);
    }
    if (want && discrete && !c->disc.found) {
        c->disc.found = 1; c->disc.tick = p->tick; c->disc.discrete = 1;
        c->disc.left = left; c->disc.right = right;
        c->disc.source_ability = source_ability; c->disc.entity_kind = entity_kind;
        memcpy(c->disc.field, field, sizeof c->disc.field);
        memcpy(c->disc.context, context, sizeof c->disc.context);
    }
    if (want_ex && !c->any_ex.found) {
        c->any_ex.found = 1; c->any_ex.tick = p->tick; c->any_ex.discrete = discrete;
        c->any_ex.left = left; c->any_ex.right = right;
        c->any_ex.source_ability = source_ability; c->any_ex.entity_kind = entity_kind;
        memcpy(c->any_ex.field, field, sizeof c->any_ex.field);
        memcpy(c->any_ex.context, context, sizeof c->any_ex.context);
    }
    if (want_ex && discrete && !c->disc_ex.found) {
        c->disc_ex.found = 1; c->disc_ex.tick = p->tick; c->disc_ex.discrete = 1;
        c->disc_ex.left = left; c->disc_ex.right = right;
        c->disc_ex.source_ability = source_ability; c->disc_ex.entity_kind = entity_kind;
        memcpy(c->disc_ex.field, field, sizeof c->disc_ex.field);
        memcpy(c->disc_ex.context, context, sizeof c->disc_ex.context);
    }
}

/* Defined after dr_generalize, which it needs. */
static int dr_ignored(const char *field) {
    if (!dr_ignore_list || !*dr_ignore_list) return 0;
    char key[64];
    dr_generalize(field, key, sizeof key);
    size_t length = strlen(key);
    for (const char *at = dr_ignore_list; at && *at;) {
        const char *end = strchr(at, ',');
        size_t span = end ? (size_t)(end - at) : strlen(at);
        if (span == length && !strncmp(at, key, span)) return 1;
        at = end ? end + 1 : NULL;
    }
    return 0;
}

#define DR_D(fmt, idx, kind, ctxidx, l, r) \
    do { if ((int64_t)(l) != (int64_t)(r)) dr_hit(p, 1, fmt, (unsigned)(idx), kind, (unsigned)(ctxidx), (int64_t)(l), (int64_t)(r), -1); } while (0)
#define DR_DA(fmt, idx, kind, ctxidx, l, r, ab) \
    do { if ((int64_t)(l) != (int64_t)(r)) dr_hit(p, 1, fmt, (unsigned)(idx), kind, (unsigned)(ctxidx), (int64_t)(l), (int64_t)(r), (ab)); } while (0)
#define DR_N(fmt, idx, kind, ctxidx, l, r) \
    do { int64_t g_ = dr_magnitude((int64_t)(l), (int64_t)(r)); \
         if (g_ > p->out->max_gap) { p->out->max_gap = g_; p->out->max_gap_tick = p->tick; } \
         if (g_ > p->tolerance) dr_hit(p, 0, fmt, (unsigned)(idx), kind, (unsigned)(ctxidx), (int64_t)(l), (int64_t)(r), -1); } while (0)
#define DR_NA(fmt, idx, kind, ctxidx, l, r, ab) \
    do { int64_t g_ = dr_magnitude((int64_t)(l), (int64_t)(r)); \
         if (g_ > p->out->max_gap) { p->out->max_gap = g_; p->out->max_gap_tick = p->tick; } \
         if (g_ > p->tolerance) dr_hit(p, 0, fmt, (unsigned)(idx), kind, (unsigned)(ctxidx), (int64_t)(l), (int64_t)(r), (ab)); } while (0)

/* One entity row pair. The two indices are equal in the all-fields lens and
 * may differ in the compacted visible lens. */
static void dr_entity_fields(DrPass *p, uint32_t ia, uint32_t ib, int visible_only) {
    const BoplRlEntity *ea = &p->a->entities[ia], *eb = &p->b->entities[ib];
    p->ctx_ib = ib;
    if (!visible_only) DR_D("entity[%u].id", ia, DR_CTX_ENTITY, ia, ea->id, eb->id);
    DR_D("entity[%u].kind", ia, DR_CTX_ENTITY, ia, ea->kind, eb->kind);
    DR_D("entity[%u].layer", ia, DR_CTX_ENTITY, ia, ea->layer, eb->layer);
    DR_D("entity[%u].owner", ia, DR_CTX_ENTITY, ia, ea->owner, eb->owner);
    /* An inactive prefab reports its component's inert ENABLED bit differently
     * in the two ports and the observation writer drops every inactive row, so
     * that bit cannot reach a policy (replay_ab.c). */
    uint32_t fa = ea->flags, fb = eb->flags;
    if (!(fa & BOPL_ENTITY_ACTIVE) && !(fb & BOPL_ENTITY_ACTIVE)) {
        fa &= ~(uint32_t)BOPL_ENTITY_ENABLED;
        fb &= ~(uint32_t)BOPL_ENTITY_ENABLED;
    }
    DR_D("entity[%u].flags", ia, DR_CTX_ENTITY, ia, fa, fb);
    if (!visible_only) DR_D("entity[%u].hierarchy", ia, DR_CTX_ENTITY, ia, ea->hierarchy, eb->hierarchy);
    DR_D("entity[%u].source_ability", ia, DR_CTX_ENTITY, ia, ea->reserved[0], eb->reserved[0]);
    DR_D("entity[%u].timer_ticks", ia, DR_CTX_ENTITY, ia, ea->reserved[1], eb->reserved[1]);
    DR_N("entity[%u].x", ia, DR_CTX_ENTITY, ia, ea->x, eb->x);
    DR_N("entity[%u].y", ia, DR_CTX_ENTITY, ia, ea->y, eb->y);
    DR_N("entity[%u].vx", ia, DR_CTX_ENTITY, ia, ea->vx, eb->vx);
    DR_N("entity[%u].vy", ia, DR_CTX_ENTITY, ia, ea->vy, eb->vy);
    DR_N("entity[%u].right_x", ia, DR_CTX_ENTITY, ia, ea->right_x, eb->right_x);
    DR_N("entity[%u].right_y", ia, DR_CTX_ENTITY, ia, ea->right_y, eb->right_y);
    DR_N("entity[%u].up_x", ia, DR_CTX_ENTITY, ia, ea->up_x, eb->up_x);
    DR_N("entity[%u].up_y", ia, DR_CTX_ENTITY, ia, ea->up_y, eb->up_y);
    DR_N("entity[%u].radius", ia, DR_CTX_ENTITY, ia, ea->radius, eb->radius);
    DR_N("entity[%u].rotation", ia, DR_CTX_ENTITY, ia, ea->rotation, eb->rotation);
    DR_N("entity[%u].angular_velocity", ia, DR_CTX_ENTITY, ia, ea->angular_velocity, eb->angular_velocity);
    DR_N("entity[%u].inverse_mass", ia, DR_CTX_ENTITY, ia, ea->inverse_mass, eb->inverse_mass);
    DR_N("entity[%u].clock_rate", ia, DR_CTX_ENTITY, ia, ea->clock_rate, eb->clock_rate);
}

/* Returns 1 when anything differs at this tick. Unlike replay_ab.c's
 * compare_state this never returns early: the caller wants the first tick for
 * every lens, and a lens whose first divergence is already recorded is skipped
 * by the caller instead. */
static int dr_compare(const BoplRlState *a, const BoplRlState *b, unsigned tick,
                      int64_t tolerance, int visible_only, DrCombo *out) {
    DrPass pass = {a, b, tolerance, tick, visible_only, out, 0, 0};
    DrPass *p = &pass;
    const BoplRlHeader *ha = &a->header, *hb = &b->header;

    DR_D("header.player_count", 0, DR_CTX_HEADER, 0, ha->player_count, hb->player_count);
    if (visible_only) {
        uint32_t va = 0, vb = 0;
        for (uint32_t i = 0; i < ha->entity_count; i++) va += (uint32_t)dr_obs_visible(&a->entities[i]);
        for (uint32_t i = 0; i < hb->entity_count; i++) vb += (uint32_t)dr_obs_visible(&b->entities[i]);
        DR_D("header.visible_entities", 0, DR_CTX_HEADER, 0, va, vb);
    } else {
        DR_D("header.entity_count", 0, DR_CTX_HEADER, 0, ha->entity_count, hb->entity_count);
    }
    DR_D("header.slime_count", 0, DR_CTX_HEADER, 0, ha->slime_count, hb->slime_count);
    DR_D("header.flags", 0, DR_CTX_HEADER, 0, ha->flags, hb->flags);
    DR_D("header.episode_ticks", 0, DR_CTX_HEADER, 0, ha->episode_ticks, hb->episode_ticks);
    DR_D("header.sudden_death", 0, DR_CTX_HEADER, 0, ha->sudden_death, hb->sudden_death);
    DR_D("header.warmup_ticks", 0, DR_CTX_HEADER, 0, ha->warmup_ticks, hb->warmup_ticks);
    DR_N("header.simulation_time", 0, DR_CTX_HEADER, 0, ha->simulation_time, hb->simulation_time);
    DR_N("header.player_time_scale", 0, DR_CTX_HEADER, 0, ha->player_time_scale, hb->player_time_scale);
    DR_N("header.simulation_time_since_level_loaded", 0, DR_CTX_HEADER, 0,
         ha->simulation_time_since_level_loaded, hb->simulation_time_since_level_loaded);

    uint32_t players = ha->player_count < hb->player_count ? ha->player_count : hb->player_count;
    for (uint32_t i = 0; i < players; i++) {
        const BoplRlPlayer *pa = &a->players[i], *pb = &b->players[i];
        DR_D("player[%u].id", i, DR_CTX_PLAYER, i, pa->id, pb->id);
        DR_D("player[%u].team", i, DR_CTX_PLAYER, i, pa->team, pb->team);
        DR_D("player[%u].flags", i, DR_CTX_PLAYER, i, pa->flags, pb->flags);
        DR_D("player[%u].alive_bodies", i, DR_CTX_PLAYER, i, pa->alive_bodies, pb->alive_bodies);
        DR_D("player[%u].stocks", i, DR_CTX_PLAYER, i, pa->stocks, pb->stocks);
        DR_D("player[%u].kills", i, DR_CTX_PLAYER, i, pa->kills, pb->kills);
        DR_D("player[%u].deaths", i, DR_CTX_PLAYER, i, pa->deaths, pb->deaths);
        DR_D("player[%u].cause_of_death", i, DR_CTX_PLAYER, i, pa->cause_of_death, pb->cause_of_death);
        DR_N("player[%u].x", i, DR_CTX_PLAYER, i, pa->x, pb->x);
        DR_N("player[%u].y", i, DR_CTX_PLAYER, i, pa->y, pb->y);
        DR_N("player[%u].scale", i, DR_CTX_PLAYER, i, pa->scale, pb->scale);
        DR_N("player[%u].invisible_time", i, DR_CTX_PLAYER, i, pa->invisible_time, pb->invisible_time);
        DR_N("player[%u].ability_time", i, DR_CTX_PLAYER, i, pa->ability_time, pb->ability_time);
        const BoplRlTelemetry *ta = &a->telemetry[i], *tb = &b->telemetry[i];
        for (int s = 0; s < 3; s++)
            DR_D("telemetry[%u].ability_activations[]", i, DR_CTX_PLAYER, i,
                 ta->ability_activations[s], tb->ability_activations[s]);
        DR_D("telemetry[%u].arrows_fired", i, DR_CTX_PLAYER, i, ta->arrows_fired, tb->arrows_fired);
        DR_D("telemetry[%u].arrow_hits", i, DR_CTX_PLAYER, i, ta->arrow_hits, tb->arrow_hits);
        DR_D("telemetry[%u].deaths_environment", i, DR_CTX_PLAYER, i, ta->deaths_environment, tb->deaths_environment);
        DR_D("telemetry[%u].deaths_opponent", i, DR_CTX_PLAYER, i, ta->deaths_opponent, tb->deaths_opponent);
        DR_D("telemetry[%u].deaths_self", i, DR_CTX_PLAYER, i, ta->deaths_self, tb->deaths_self);
        DR_D("telemetry[%u].deaths_other", i, DR_CTX_PLAYER, i, ta->deaths_other, tb->deaths_other);
    }

    uint32_t slimes = ha->slime_count < hb->slime_count ? ha->slime_count : hb->slime_count;
    for (uint32_t i = 0; i < slimes; i++) {
        const BoplRlSlime *sa = &a->slimes[i], *sb = &b->slimes[i];
        DR_D("slime[%u].id", i, DR_CTX_SLIME, i, sa->id, sb->id);
        DR_D("slime[%u].owner", i, DR_CTX_SLIME, i, sa->owner, sb->owner);
        DR_D("slime[%u].flags", i, DR_CTX_SLIME, i, sa->flags, sb->flags);
        DR_D("slime[%u].state", i, DR_CTX_SLIME, i, sa->state, sb->state);
        for (int s = 0; s < 3; s++) {
            int ab = (int)sa->abilities[s];
            DR_DA("slime[%u].abilities[]", i, DR_CTX_SLIME, i, sa->abilities[s], sb->abilities[s], ab);
            DR_DA("slime[%u].ability_flags[]", i, DR_CTX_SLIME, i, sa->ability_flags[s], sb->ability_flags[s], ab);
            DR_NA("slime[%u].cooldowns[]", i, DR_CTX_SLIME, i, sa->cooldowns[s], sb->cooldowns[s], ab);
            DR_NA("slime[%u].timers[]", i, DR_CTX_SLIME, i, sa->timers[s], sb->timers[s], ab);
        }
        DR_N("slime[%u].x", i, DR_CTX_SLIME, i, sa->x, sb->x);
        DR_N("slime[%u].y", i, DR_CTX_SLIME, i, sa->y, sb->y);
        DR_N("slime[%u].vx", i, DR_CTX_SLIME, i, sa->vx, sb->vx);
        DR_N("slime[%u].vy", i, DR_CTX_SLIME, i, sa->vy, sb->vy);
        DR_N("slime[%u].rotation", i, DR_CTX_SLIME, i, sa->rotation, sb->rotation);
        DR_N("slime[%u].scale", i, DR_CTX_SLIME, i, sa->scale, sb->scale);
    }

    /* Entity rows. The all-fields lens walks the two tables index by index,
     * which is what replay_ab.c does and what the map sweeps established is
     * meaningful: both backends build the table in the same order.
     *
     * The policy-visible lens instead walks the two COMPACTED sequences of
     * visible rows, because that is what the observation writer encodes
     * (bopl_obs_v4.h iterates the table and emits the rows that pass
     * bopl_puffer.h:316 in order). If one backend carries an extra hidden row
     * mid-table, index-aligned comparison would report every later row as
     * divergent while the policy sees identical input; compacting is the
     * question actually being asked. DR_VISIBLE_ALIGNED=1 restores the
     * index-aligned behaviour so the two can be compared. */
    uint32_t entities;
    if (visible_only && !dr_visible_aligned) {
        uint32_t ia = 0, ib = 0;
        for (;;) {
            while (ia < ha->entity_count && !dr_obs_visible(&a->entities[ia])) ia++;
            while (ib < hb->entity_count && !dr_obs_visible(&b->entities[ib])) ib++;
            if (ia >= ha->entity_count || ib >= hb->entity_count) break;
            dr_entity_fields(p, ia, ib, visible_only);
            ia++; ib++;
        }
        return pass.diverged;
    }
    entities = ha->entity_count < hb->entity_count ? ha->entity_count : hb->entity_count;
    for (uint32_t i = 0; i < entities; i++) {
        if (visible_only && !dr_obs_visible(&a->entities[i]) && !dr_obs_visible(&b->entities[i])) continue;
        dr_entity_fields(p, i, i, visible_only);
    }
    return pass.diverged;
}
#endif
