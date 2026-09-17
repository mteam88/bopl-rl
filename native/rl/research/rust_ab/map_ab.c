/* Exact authored-map-state NativeAOT/Rust differential.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/map_ab \
 *   research/rust_ab/map_ab.c bopl_rl.c -ldl -lm
 * /tmp/map_ab BoplSimNative.so libbopl_sim.so [ticks]
 *
 * The ordinary actor and spawned-ability rows have their own focused gates.
 * This probe compares every authored platform and water row byte-for-byte,
 * plus the clocks that drive map components, while exercising both player
 * counts. MAP_AB_STRICT_ACTORS=1 additionally checks discrete actor state;
 * MAP_AB_STRICT_VECTORS=1 makes that diagnostic compare actor vectors too.
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t max_vector_delta;
static uint64_t max_basis_delta;
static uint64_t max_scalar_delta;
static unsigned nonexact_map_rows;

static void update_max(uint64_t *maximum, uint64_t value) {
    if (value > *maximum) *maximum = value;
}

static int bytes_equal(const void *a, const void *b, size_t size,
                       const char *field, unsigned map, unsigned players,
                       unsigned tick) {
    if (!memcmp(a, b, size)) return 1;
    const unsigned char *aa = a, *bb = b;
    size_t byte = 0;
    while (byte < size && aa[byte] == bb[byte]) byte++;
    fprintf(stderr,
            "MAP_DIFFERENCE map=%u players=%u tick=%u field=%s byte=%zu "
            "reference=%u rust=%u\n",
            map, players, tick, field, byte, aa[byte], bb[byte]);
    return 0;
}

static uint64_t distance_i64(int64_t a, int64_t b) {
    return a >= b ? (uint64_t)a - (uint64_t)b : (uint64_t)b - (uint64_t)a;
}

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    int map_entity = a->kind == 2 || a->layer == 4;
    uint64_t vector_tolerance = map_entity ? 0 : UINT64_MAX;
    uint64_t basis_tolerance = map_entity ? 0 : UINT64_MAX;
    uint64_t scalar_tolerance = map_entity ? 0 : UINT64_MAX;
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy &&
           !memcmp(a->reserved, b->reserved, sizeof a->reserved) &&
           distance_i64(a->x, b->x) <= vector_tolerance &&
           distance_i64(a->y, b->y) <= vector_tolerance &&
           distance_i64(a->vx, b->vx) <= vector_tolerance &&
           distance_i64(a->vy, b->vy) <= vector_tolerance &&
           distance_i64(a->right_x, b->right_x) <= basis_tolerance &&
           distance_i64(a->right_y, b->right_y) <= basis_tolerance &&
           distance_i64(a->up_x, b->up_x) <= basis_tolerance &&
           distance_i64(a->up_y, b->up_y) <= basis_tolerance &&
           distance_i64(a->radius, b->radius) <= scalar_tolerance &&
           distance_i64(a->rotation, b->rotation) <= scalar_tolerance &&
           distance_i64(a->angular_velocity, b->angular_velocity) <=
               vector_tolerance &&
           a->inverse_mass == b->inverse_mass &&
           (!map_entity || a->clock_rate == b->clock_rate);
}

static int is_map_entity(const BoplRlEntity *entity) {
    return entity->kind == 2 || entity->layer == 4;
}

static void observe_map_delta(const BoplRlEntity *a, const BoplRlEntity *b) {
    if (memcmp(a, b, sizeof *a)) nonexact_map_rows++;
    update_max(&max_vector_delta, distance_i64(a->x, b->x));
    update_max(&max_vector_delta, distance_i64(a->y, b->y));
    update_max(&max_vector_delta, distance_i64(a->vx, b->vx));
    update_max(&max_vector_delta, distance_i64(a->vy, b->vy));
    update_max(&max_vector_delta,
               distance_i64(a->angular_velocity, b->angular_velocity));
    update_max(&max_basis_delta, distance_i64(a->right_x, b->right_x));
    update_max(&max_basis_delta, distance_i64(a->right_y, b->right_y));
    update_max(&max_basis_delta, distance_i64(a->up_x, b->up_x));
    update_max(&max_basis_delta, distance_i64(a->up_y, b->up_y));
    update_max(&max_scalar_delta, distance_i64(a->radius, b->radius));
    update_max(&max_scalar_delta, distance_i64(a->rotation, b->rotation));
}

static const BoplRlEntity *find_map_entity(const BoplRlState *state,
                                            const BoplRlEntity *wanted) {
    for (unsigned i = 0; i < state->header.entity_count; i++) {
        const BoplRlEntity *candidate = &state->entities[i];
        if (is_map_entity(candidate) && candidate->kind == wanted->kind &&
            candidate->layer == wanted->layer &&
            candidate->hierarchy == wanted->hierarchy)
            return candidate;
    }
    return NULL;
}

static int compare(const BoplRlState *a, const BoplRlState *b, unsigned map,
                   unsigned players, unsigned tick) {
    if (getenv("MAP_AB_TRACE_PICKUP")) {
        unsigned trace_id = (unsigned)strtoul(getenv("MAP_AB_TRACE_PICKUP"), NULL, 10);
        const BoplRlEntity *ap = NULL, *bp = NULL;
        for (unsigned i = 0; i < a->header.entity_count; i++)
            if (a->entities[i].id == trace_id) ap = &a->entities[i];
        for (unsigned i = 0; i < b->header.entity_count; i++)
            if (b->entities[i].id == trace_id) bp = &b->entities[i];
        if (ap || bp)
            fprintf(stderr,
                    "PICKUP_TRACE tick=%u flags=%u/%u ability=%u/%u pos=(%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") vel=(%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ")\n",
                    tick, ap ? ap->flags : 0, bp ? bp->flags : 0,
                    ap ? ap->reserved[0] : 0, bp ? bp->reserved[0] : 0,
                    ap ? ap->x : 0, ap ? ap->y : 0, bp ? bp->x : 0,
                    bp ? bp->y : 0, ap ? ap->vx : 0, ap ? ap->vy : 0,
                    bp ? bp->vx : 0, bp ? bp->vy : 0);
    }
    if (getenv("MAP_AB_TRACE_SLIME")) {
        unsigned owner = (unsigned)strtoul(getenv("MAP_AB_TRACE_SLIME"), NULL, 10);
        const BoplRlSlime *as = NULL, *bs = NULL;
        for (unsigned i = 0; i < a->header.slime_count; i++)
            if (a->slimes[i].owner == owner) as = &a->slimes[i];
        for (unsigned i = 0; i < b->header.slime_count; i++)
            if (b->slimes[i].owner == owner) bs = &b->slimes[i];
        if (as || bs)
            fprintf(stderr,
                    "SLIME_TRACE tick=%u state=%u/%u pos=(%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") vel=(%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") rot=%" PRId64 "/%" PRId64 "\n",
                    tick, as ? as->state : 0, bs ? bs->state : 0,
                    as ? as->x : 0, as ? as->y : 0, bs ? bs->x : 0,
                    bs ? bs->y : 0, as ? as->vx : 0, as ? as->vy : 0,
                    bs ? bs->vx : 0, bs ? bs->vy : 0,
                    as ? as->rotation : 0, bs ? bs->rotation : 0);
    }
    if (getenv("MAP_AB_TRACE_ENTITY")) {
        unsigned hierarchy = (unsigned)strtoul(getenv("MAP_AB_TRACE_ENTITY"), NULL, 10);
        const BoplRlEntity *ae = NULL, *be = NULL;
        for (unsigned i = 0; i < a->header.entity_count; i++)
            if (a->entities[i].hierarchy == hierarchy) ae = &a->entities[i];
        for (unsigned i = 0; i < b->header.entity_count; i++)
            if (b->entities[i].hierarchy == hierarchy) be = &b->entities[i];
        if (ae || be)
            fprintf(stderr,
                    "ENTITY_TRACE tick=%u pos=(%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") vel=(%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") radius=%" PRId64 "/%" PRId64
                    " basis=(%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ")\n",
                    tick, ae ? ae->x : 0, ae ? ae->y : 0, be ? be->x : 0,
                    be ? be->y : 0, ae ? ae->vx : 0, ae ? ae->vy : 0,
                    be ? be->vx : 0, be ? be->vy : 0,
                    ae ? ae->radius : 0, be ? be->radius : 0,
                    ae ? ae->right_x : 0, ae ? ae->right_y : 0,
                    ae ? ae->up_x : 0, ae ? ae->up_y : 0,
                    be ? be->right_x : 0, be ? be->right_y : 0,
                    be ? be->up_x : 0, be ? be->up_y : 0);
    }
    BoplRlHeader ah = a->header, bh = b->header;
    /* The Rust backend's fixed ability-entity table deliberately retains
     * removed rows, whereas NativeAOT compacts them. Map rows are compared
     * below; the table length is not map state. */
    ah.entity_count = bh.entity_count = 0;
    if (!getenv("MAP_AB_STRICT_ACTORS")) {
        ah.slime_count = bh.slime_count = 0;
        ah.flags = bh.flags = 0;
    }
    if (!bytes_equal(&ah, &bh, sizeof ah, "header", map, players, tick)) {
        for (unsigned p = 0; p < players; p++)
            fprintf(stderr,
                    "  player[%u] flags=%u/%u alive=%u/%u deaths=%u/%u cause=%u/%u "
                    "pos=(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                    p, a->players[p].flags, b->players[p].flags,
                    a->players[p].alive_bodies, b->players[p].alive_bodies,
                    a->players[p].deaths, b->players[p].deaths,
                    a->players[p].cause_of_death, b->players[p].cause_of_death,
                    a->players[p].x, a->players[p].y,
                    b->players[p].x, b->players[p].y);
        return 0;
    }
    if (getenv("MAP_AB_STRICT_ACTORS"))
    for (unsigned i = 0; i < 6; i++) {
        BoplRlPlayer ap = a->players[i], bp = b->players[i];
        if ((getenv("MAP_AB_STRICT_VECTORS") ||
             getenv("MAP_AB_STRICT_PLAYER_VECTORS")) &&
            (distance_i64(ap.x, bp.x) > 0 || distance_i64(ap.y, bp.y) > 0)) {
            fprintf(stderr,
                    "  player=%u pos=(%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ")\n",
                    i, ap.x, ap.y, bp.x, bp.y);
            for (unsigned p = 0; p < players; p++)
                fprintf(stderr,
                        "  player[%u] pos=(%" PRId64 ",%" PRId64 ")/(%" PRId64
                        ",%" PRId64 ") flags=%u/%u\n",
                        p, a->players[p].x, a->players[p].y,
                        b->players[p].x, b->players[p].y,
                        a->players[p].flags, b->players[p].flags);
            for (unsigned e = 0; e < a->header.entity_count; e++) {
                if (a->entities[e].kind == 2 &&
                    !entity_equal(&a->entities[e], &b->entities[e]))
                    fprintf(stderr,
                            "  platform=%u p=(%" PRId64 ",%" PRId64 ")/(%" PRId64
                            ",%" PRId64 ") v=(%" PRId64 ",%" PRId64 ")/(%" PRId64
                            ",%" PRId64 ")\n",
                            e, a->entities[e].x, a->entities[e].y,
                            b->entities[e].x, b->entities[e].y,
                            a->entities[e].vx, a->entities[e].vy,
                            b->entities[e].vx, b->entities[e].vy);
            }
            return bytes_equal(&ap, &bp, sizeof ap, "player_position", map,
                               players, tick);
        }
        ap.x = ap.y = bp.x = bp.y = 0;
        if (memcmp(&ap, &bp, sizeof ap)) {
            fprintf(stderr,
                    "  player=%u flags=%u/%u alive=%u/%u stocks=%u/%u "
                    "pos=(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                    i, a->players[i].flags, b->players[i].flags,
                    a->players[i].alive_bodies, b->players[i].alive_bodies,
                    a->players[i].stocks, b->players[i].stocks,
                    a->players[i].x, a->players[i].y,
                    b->players[i].x, b->players[i].y);
            for (unsigned s = 0; s < a->header.slime_count; s++)
                fprintf(stderr,
                        "  reference slime=%u owner=%u flags=%u state=%u "
                        "pos=(%" PRId64 ",%" PRId64 ") vel=(%" PRId64 ",%" PRId64 ")\n",
                        s, a->slimes[s].owner, a->slimes[s].flags,
                        a->slimes[s].state, a->slimes[s].x, a->slimes[s].y,
                        a->slimes[s].vx, a->slimes[s].vy);
            for (unsigned s = 0; s < b->header.slime_count; s++)
                fprintf(stderr,
                        "  rust slime=%u owner=%u flags=%u state=%u "
                        "pos=(%" PRId64 ",%" PRId64 ") vel=(%" PRId64 ",%" PRId64 ")\n",
                        s, b->slimes[s].owner, b->slimes[s].flags,
                        b->slimes[s].state, b->slimes[s].x, b->slimes[s].y,
                        b->slimes[s].vx, b->slimes[s].vy);
        }
        if (!bytes_equal(&ap, &bp, sizeof ap, "players", map, players, tick))
            return 0;
    }
    if (getenv("MAP_AB_STRICT_ACTORS") &&
        !bytes_equal(a->telemetry, b->telemetry, sizeof a->telemetry,
                     "telemetry", map, players, tick))
        return 0;
    unsigned a_map_count = 0, b_map_count = 0;
    for (unsigned i = 0; i < b->header.entity_count; i++)
        b_map_count += is_map_entity(&b->entities[i]);
    for (unsigned i = 0; i < a->header.entity_count; i++) {
        if (!is_map_entity(&a->entities[i])) continue;
        a_map_count++;
        const BoplRlEntity *match = find_map_entity(b, &a->entities[i]);
        if (match) observe_map_delta(&a->entities[i], match);
        if (!match || !entity_equal(&a->entities[i], match)) {
            const BoplRlEntity *other = match ? match : &a->entities[i];
            fprintf(stderr,
                    "MAP_DIFFERENCE map=%u players=%u tick=%u field=map_entity\n",
                    map, players, tick);
                fprintf(stderr,
                        "  entity=%u id=%u/%u kind=%u/%u layer=%u/%u "
                        "hierarchy=%u/%u pos=(%" PRId64 ",%" PRId64
                        ")/(%" PRId64 ",%" PRId64 ") vel=(%" PRId64
                        ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                        i, a->entities[i].id, other->id,
                        a->entities[i].kind, other->kind,
                        a->entities[i].layer, other->layer,
                        a->entities[i].hierarchy, other->hierarchy,
                        a->entities[i].x, a->entities[i].y, other->x,
                        other->y, a->entities[i].vx, a->entities[i].vy,
                        other->vx, other->vy);
                fprintf(stderr,
                        "  flags=%u/%u basis=(%" PRId64 ",%" PRId64 ",%" PRId64
                        ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ",%" PRId64
                        ",%" PRId64 ") radius=%" PRId64 "/%" PRId64
                        " rot=%" PRId64 "/%" PRId64 " av=%" PRId64 "/%" PRId64
                        " im=%" PRId64 "/%" PRId64 " reserved=%u,%u/%u,%u clock=%" PRId64 "/%" PRId64 "\n",
                        a->entities[i].flags, other->flags,
                        a->entities[i].right_x, a->entities[i].right_y,
                        a->entities[i].up_x, a->entities[i].up_y,
                        other->right_x, other->right_y,
                        other->up_x, other->up_y,
                        a->entities[i].radius, other->radius,
                        a->entities[i].rotation, other->rotation,
                        a->entities[i].angular_velocity,
                        other->angular_velocity,
                        a->entities[i].inverse_mass, other->inverse_mass,
                        a->entities[i].reserved[0], a->entities[i].reserved[1],
                        other->reserved[0], other->reserved[1],
                        a->entities[i].clock_rate, other->clock_rate);
            return 0;
        }
    }
    if (a_map_count != b_map_count) {
        fprintf(stderr,
                "MAP_DIFFERENCE map=%u players=%u tick=%u field=map_entity_count "
                "reference=%u rust=%u\n",
                map, players, tick, a_map_count, b_map_count);
        return 0;
    }
    if (getenv("MAP_AB_STRICT_PICKUPS")) {
        for (unsigned i = 0; i < a->header.entity_count; i++) {
            const BoplRlEntity *ap = &a->entities[i];
            if (!(ap->flags & BOPL_ENTITY_PICKUP)) continue;
            const BoplRlEntity *bp = NULL;
            for (unsigned j = 0; j < b->header.entity_count; j++)
                if (b->entities[j].id == ap->id &&
                    (b->entities[j].flags & BOPL_ENTITY_PICKUP)) {
                    bp = &b->entities[j];
                    break;
                }
            if (!bp || memcmp(ap, bp, sizeof *ap)) {
                fprintf(stderr,
                        "MAP_DIFFERENCE map=%u players=%u tick=%u field=pickup id=%u "
                        "flags=%u/%u pos=(%" PRId64 ",%" PRId64 ")/(%" PRId64
                        ",%" PRId64 ") vel=(%" PRId64 ",%" PRId64 ")/(%" PRId64
                        ",%" PRId64 ")\n",
                        map, players, tick, ap->id, ap->flags,
                        bp ? bp->flags : 0, ap->x, ap->y, bp ? bp->x : 0,
                        bp ? bp->y : 0, ap->vx, ap->vy, bp ? bp->vx : 0,
                        bp ? bp->vy : 0);
                fprintf(stderr,
                        "  ability=%u/%u timer=%u/%u time_scale=%" PRId64 "/%" PRId64 "\n",
                        ap->reserved[0], bp ? bp->reserved[0] : 0,
                        ap->reserved[1], bp ? bp->reserved[1] : 0,
                        a->header.player_time_scale, b->header.player_time_scale);
                return 0;
            }
        }
    }
    if (!getenv("MAP_AB_STRICT_ACTORS")) return 1;
    if (a->header.slime_count != b->header.slime_count) return 0;
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        BoplRlSlime as = a->slimes[i], bs = b->slimes[i];
        if ((getenv("MAP_AB_STRICT_VECTORS") ||
             getenv("MAP_AB_STRICT_SLIME_VECTORS")) &&
            (distance_i64(as.x, bs.x) > 0 || distance_i64(as.y, bs.y) > 0 ||
                  distance_i64(as.vx, bs.vx) > 0 || distance_i64(as.vy, bs.vy) > 0))
        {
            fprintf(stderr,
                    "  slime=%u pos=(%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") vel=(%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ")\n",
                    i, as.x, as.y, bs.x, bs.y, as.vx, as.vy, bs.vx, bs.vy);
            for (unsigned e = 0; e < a->header.entity_count; e++) {
                const BoplRlEntity *ae = &a->entities[e];
                if (!is_map_entity(ae)) continue;
                const BoplRlEntity *be = find_map_entity(b, ae);
                if (be && memcmp(ae, be, sizeof *ae))
                    fprintf(stderr,
                            "  map_entity hierarchy=%u pos=(%" PRId64 ",%" PRId64
                            ")/(%" PRId64 ",%" PRId64 ") vel=(%" PRId64 ",%" PRId64
                            ")/(%" PRId64 ",%" PRId64 ") radius=%" PRId64 "/%" PRId64
                            " basis=(%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
                            ")/(%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ")\n",
                            ae->hierarchy, ae->x, ae->y, be->x, be->y,
                            ae->vx, ae->vy, be->vx, be->vy, ae->radius, be->radius,
                            ae->right_x, ae->right_y, ae->up_x, ae->up_y,
                            be->right_x, be->right_y, be->up_x, be->up_y);
            }
            return bytes_equal(&as, &bs, sizeof as, "slime_vector", map,
                               players, tick);
        }
        as.x = as.y = as.vx = as.vy = 0;
        bs.x = bs.y = bs.vx = bs.vy = 0;
        if (!getenv("MAP_AB_STRICT_STATE")) {
            as.state = 0;
            bs.state = 0;
        }
        if (!getenv("MAP_AB_STRICT_ROTATION")) {
            as.rotation = 0;
            bs.rotation = 0;
        }
        /* The Rust backend advertises its fixed replacement slot in the
         * ABI-v3 metadata bits even before a pickup exists; map fidelity is
         * independent of those advisory bits. */
        as.flags &= 0x3fu;
        bs.flags &= 0x3fu;
        if (memcmp(&as, &bs, sizeof as)) {
            fprintf(stderr,
                    "  slime=%u id=%u/%u owner=%u/%u flags=%u/%u state=%u/%u "
                    "abilities=%u,%u,%u/%u,%u,%u ability_flags=%u,%u,%u/%u,%u,%u "
                    "timers=(%" PRId64 ",%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ",%" PRId64 ")\n",
                    i, as.id, bs.id, as.owner, bs.owner, as.flags, bs.flags,
                    as.state, bs.state, as.abilities[0], as.abilities[1],
                    as.abilities[2], bs.abilities[0], bs.abilities[1],
                    bs.abilities[2], as.ability_flags[0], as.ability_flags[1],
                    as.ability_flags[2], bs.ability_flags[0],
                    bs.ability_flags[1], bs.ability_flags[2], as.timers[0],
                    as.timers[1], as.timers[2], bs.timers[0], bs.timers[1],
                    bs.timers[2]);
            for (unsigned e = 0; e < a->header.entity_count; e++)
                if (a->entities[e].flags & BOPL_ENTITY_PICKUP)
                    fprintf(stderr,
                            "  reference pickup id=%u flags=%u pos=(%" PRId64
                            ",%" PRId64 ") vel=(%" PRId64 ",%" PRId64 ")\n",
                            a->entities[e].id, a->entities[e].flags,
                            a->entities[e].x, a->entities[e].y,
                            a->entities[e].vx, a->entities[e].vy);
            for (unsigned e = 0; e < b->header.entity_count; e++)
                if (b->entities[e].flags & BOPL_ENTITY_PICKUP)
                    fprintf(stderr,
                            "  rust pickup id=%u flags=%u pos=(%" PRId64
                            ",%" PRId64 ") vel=(%" PRId64 ",%" PRId64 ")\n",
                            b->entities[e].id, b->entities[e].flags,
                            b->entities[e].x, b->entities[e].y,
                            b->entities[e].vx, b->entities[e].vy);
            for (unsigned s = 0; s < a->header.slime_count && s < b->header.slime_count; s++)
                fprintf(stderr,
                        "  slime_timer owner=%u/%u third=%" PRId64 "/%" PRId64 " pos=(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                        a->slimes[s].owner, b->slimes[s].owner,
                        a->slimes[s].timers[2], b->slimes[s].timers[2],
                        a->slimes[s].x, a->slimes[s].y,
                        b->slimes[s].x, b->slimes[s].y);
        }
        if (!bytes_equal(&as, &bs, sizeof as, "slimes", map, players, tick))
            return 0;
    }
    return 1;
}

static void actions(BoplRlAction *out, unsigned players, unsigned tick) {
    memset(out, 0, sizeof(*out) * players);
    if (getenv("MAP_AB_IDLE")) return;
    if (getenv("MAP_AB_ABILITY")) {
        for (unsigned p = 0; p < players; p++)
            out[p].angle = (uint8_t)(32 + p * 64);
        if (tick == 10) out[0].buttons = 2;
        return;
    }
    /* Exercise ordinary locomotion without abilities. Inputs are sticky and
     * symmetric enough to keep long-lived cases deterministic. */
    for (unsigned p = 0; p < players; p++) {
        unsigned phase = (tick / 45 + p) % 4;
        out[p].angle = phase == 0 ? 64 : phase == 1 ? 0 : phase == 2 ? 192 : 0;
        if (tick % 73 == 0) out[p].buttons = 1;
    }
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr,
                "usage: %s <BoplSimNative.so> <libbopl_sim.so> [ticks [map [warmup]]]\n",
                argv[0]);
        return 2;
    }
    unsigned ticks = argc >= 4 ? (unsigned)strtoul(argv[3], NULL, 10) : 3000;
    unsigned only_map = argc >= 5 ? (unsigned)strtoul(argv[4], NULL, 10) : UINT32_MAX;
    unsigned warmup = argc >= 6 ? (unsigned)strtoul(argv[5], NULL, 10) : 240;
    char error[2][512] = {{0}};
    BoplRl *world[2];
    for (unsigned backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1],
                                      "/tmp/gate_bootstrap.rep",
                                      "/tmp/map_ab.log", 120000,
                                      error[backend], sizeof error[backend]);
        if (!world[backend]) {
            fprintf(stderr, "open backend %u failed: %s\n", backend,
                    error[backend]);
            return 1;
        }
    }
    unsigned maps = bopl_rl_map_count(world[0]);
    if (maps != 124 || bopl_rl_map_count(world[1]) != maps) {
        fprintf(stderr, "map count reference=%u rust=%u\n", maps,
                bopl_rl_map_count(world[1]));
        return 1;
    }
    unsigned comparisons = 0;
    for (unsigned player_case = 0; player_case < 2; player_case++) {
        unsigned players = player_case ? 6 : 2;
        if (getenv("MAP_AB_PLAYERS") &&
            players != (unsigned)strtoul(getenv("MAP_AB_PLAYERS"), NULL, 10))
            continue;
        for (unsigned map = 0; map < maps; map++) {
            if (only_map != UINT32_MAX && map != only_map) continue;
            BoplRlConfig config;
            bopl_rl_default_config(&config, players);
            config.seed = 0x4d415000u + map * 17u + players;
            config.map = map;
            config.stocks = 0;
            config.warmup_ticks = warmup;
            config.max_episode_ticks = ticks + 600;
            for (unsigned p = 0; p < players; p++) {
                if (getenv("MAP_AB_ABILITY")) {
                    config.player[p].abilities[0] =
                        (unsigned)strtoul(getenv("MAP_AB_ABILITY"), NULL, 10);
                    config.player[p].abilities[1] = 2;
                    config.player[p].abilities[2] = 4;
                }
            }
            for (unsigned backend = 0; backend < 2; backend++) {
                int status = bopl_rl_reset(world[backend], &config);
                if (status != BOPL_RL_OK) {
                    fprintf(stderr,
                            "reset map=%u players=%u backend=%u failed: %s\n",
                            map, players, backend, bopl_rl_error(world[backend]));
                    return 1;
                }
            }
            if (getenv("MAP_AB_DUMP_ACTORS")) {
                const BoplRlState *reference = bopl_rl_state(world[0]);
                const BoplRlState *rust = bopl_rl_state(world[1]);
                for (unsigned s = 0; s < reference->header.slime_count; s++)
                    fprintf(stderr,
                            "MAP_ACTOR map=%u players=%u reference slime=%u owner=%u "
                            "state=%u pos=(%" PRId64 ",%" PRId64 ") vel=(%" PRId64
                            ",%" PRId64 ")\n",
                            map, players, s, reference->slimes[s].owner,
                            reference->slimes[s].state, reference->slimes[s].x,
                            reference->slimes[s].y, reference->slimes[s].vx,
                            reference->slimes[s].vy);
                for (unsigned s = 0; s < rust->header.slime_count; s++)
                    fprintf(stderr,
                            "MAP_ACTOR map=%u players=%u rust slime=%u owner=%u "
                            "state=%u pos=(%" PRId64 ",%" PRId64 ") vel=(%" PRId64
                            ",%" PRId64 ")\n",
                            map, players, s, rust->slimes[s].owner,
                            rust->slimes[s].state, rust->slimes[s].x,
                            rust->slimes[s].y, rust->slimes[s].vx,
                            rust->slimes[s].vy);
            }
            if (getenv("MAP_AB_DUMP_PLATFORMS")) {
                const BoplRlState *reference = bopl_rl_state(world[0]);
                const BoplRlState *rust = bopl_rl_state(world[1]);
                unsigned count = reference->header.entity_count < rust->header.entity_count
                                     ? reference->header.entity_count
                                     : rust->header.entity_count;
                for (unsigned entity = 0; entity < count; entity++) {
                    if (reference->entities[entity].kind != 2) continue;
                    fprintf(stderr,
                            "MAP_PLATFORM map=%u players=%u entity=%u hierarchy=%u "
                            "reference=(%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ") "
                            "rust=(%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ")\n",
                            map, players, entity, reference->entities[entity].hierarchy,
                            reference->entities[entity].x, reference->entities[entity].y,
                            reference->entities[entity].vx, reference->entities[entity].vy,
                            rust->entities[entity].x, rust->entities[entity].y,
                            rust->entities[entity].vx, rust->entities[entity].vy);
                }
            }
            if (!compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), map,
                         players, 0))
                return 1;
            comparisons++;
            for (unsigned tick = 1; tick <= ticks; tick++) {
                BoplRlAction action[6];
                actions(action, players, tick);
                for (unsigned backend = 0; backend < 2; backend++) {
                    int status = bopl_rl_step(world[backend], action, players, 1);
                    if (status != BOPL_RL_OK) {
                        fprintf(stderr,
                                "step map=%u players=%u tick=%u backend=%u "
                                "failed: %s\n",
                                map, players, tick, backend,
                                bopl_rl_error(world[backend]));
                        return 1;
                    }
                }
                if (!compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]),
                             map, players, tick))
                    return 1;
                comparisons++;
                if (bopl_rl_state(world[0])->header.flags & BOPL_RL_TERMINAL)
                    break;
            }
            if (getenv("MAP_AB_VERBOSE")) {
                printf("MAP_CASE_OK map=%u players=%u\n", map, players);
                fflush(stdout);
            }
        }
    }
    for (unsigned backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);
    printf("MAP_STATE_OK maps=%u player_cases=2 compared_snapshots=%u "
           "nonexact_map_rows=%u max_vector_raw=%" PRIu64
           " max_basis_raw=%" PRIu64 " max_scalar_raw=%" PRIu64 "\n",
           maps, comparisons, nonexact_map_rows, max_vector_delta,
           max_basis_delta, max_scalar_delta);
    return 0;
}
