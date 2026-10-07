/* The policy v4 row's features from a world state, one source for the CPU
 * writer (bopl_obs_v4.h bopl_encode_v4_row) and the trainer's CUDA writer
 * (bopl_rays_v4.cu bopl4_encode_row) (2026-10-07). They agree bit for bit:
 * the arithmetic is IEEE double as written, with no FMA contraction (the
 * pragma below; nvcc -fmad=false for the CUDA unit), and CUDA's double
 * division and sqrt are correctly rounded like the CPU's. What the device
 * cannot reproduce stays with the host: sinf/cosf of the slime rotations
 * come in as arguments (the host computes them per slime), the aim
 * table's directions come as floats, and bopl4_sector (atan2 near the
 * octant edges) is host only: the host ships each entity's sector.
 *
 * Everything here reads a BoplRlState and writes floats; nothing knows the
 * environment (BoplPuffer). Included by bopl_puffer.h and by the CUDA unit. */
#ifndef BOPL_ROWS_V4_H
#define BOPL_ROWS_V4_H
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "bopl_rl.h"
#include "bopl_layout.h"
#include "bopl_layout_v4.h"
#include "bopl_rays_v4.h"

#ifdef __CUDACC__
#define BOPL4D_FN static __host__ __device__ inline
#else
#define BOPL4D_FN static inline
#endif
#if defined(__clang__) && !defined(__CUDACC__)
#pragma clang fp contract(off)
#elif defined(__GNUC__) && !defined(__CUDACC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

/* Writer ablations (bopl_obs_v4.h). */
enum { BOPL4_ABLATE_RELATIONAL=1, BOPL4_ABLATE_RAYS=2, BOPL4_ABLATE_RAYS16=4, BOPL4_ABLATE_AIM=8 };
#define BOPL4_SLIME_RADIUS 0.76   /* bopl_bots.h BOT_SLIME_RADIUS */

/* fmax(-1, fmin(1, value)) without the libm calls (gcc will not inline them
 * for their NaN rules): the same result for every input, NaN included (1).
 * Written as selects so it compiles to minsd/maxsd, whose operand order gives
 * exactly these results (value == +-1 takes the other arm, the same bits),
 * where an if compiled to a data-dependent branch. */
BOPL4D_FN float bopl_clip(double value) {
    value=value<1.0?value:1.0;
    value=value>-1.0?value:-1.0;
    return (float)value;
}
BOPL4D_FN double bopl_fix(int64_t value) { return value/4294967296.0; }
/* x/c for a constant c, rounded as the division is. The device's IEEE
 * double division is a dozen dependent FP64 operations, and FP64 is the GPU
 * writer's bound (1/64 rate on the RTX 5070), so there it is Markstein's
 * correction from the rounded reciprocal rc = RN(1/c): q = RN(x rc) is within
 * an ulp of x/c, r = x - q c is exact (an FMA), and RN(q + r rc) is RN(x/c)
 * (Markstein 1990; Muller et al., Handbook of Floating-Point Arithmetic,
 * Theorem 5.10), away from overflow and underflow, which no feature nears.
 * When r is 0, q is the exact quotient and keeps its sign (RN(-0 + 0) would
 * be +0).
 * The host divides. */
BOPL4D_FN double bopl4_div_const(double x, double c, double rc) {
#ifdef __CUDA_ARCH__
    double q=x*rc, r=fma(-q, c, x);
    return r==0.0?q:fma(r, rc, q);   /* r == 0: q is x/c exactly, its sign kept (x = -0 gives -0) */
#else
    (void)rc;
    return x/c;
#endif
}
#define BOPL4_DIVC(x, c) bopl4_div_const((double)(x), (c), 1.0/(c))
/* The ballistic arc's sample times k/15 (bopl4_relation) for the device:
 * the compiler folds each quotient correctly rounded, as the host's division
 * rounds it. */
#ifdef __CUDACC__
__constant__ static const double bopl4_arc_t[31]={
    0.0, 1/15.0, 2/15.0, 3/15.0, 4/15.0, 5/15.0, 6/15.0, 7/15.0, 8/15.0, 9/15.0, 10/15.0, 11/15.0, 12/15.0, 13/15.0, 14/15.0,
    15/15.0, 16/15.0, 17/15.0, 18/15.0, 19/15.0, 20/15.0, 21/15.0, 22/15.0, 23/15.0, 24/15.0, 25/15.0, 26/15.0, 27/15.0, 28/15.0, 29/15.0, 30/15.0};
#endif
BOPL4D_FN int bopl_ally(const BoplRlState *state, uint32_t owner, const BoplRlPlayer *self) {
    for (uint32_t i=0; i<state->header.player_count; i++)
        if (state->players[i].id==owner) return state->players[i].team==self->team;
    return 0;
}
/* Stable bounded nearest-neighbor selection. Overflow is an explicit feature
 * and metric; the C game state itself retains all entities. */
BOPL4D_FN void bopl_nearest(uint32_t *indices, double *distances, int *count, int capacity, uint32_t index, double distance) {
    int position=*count;
    if (position==capacity && distance>=distances[position-1]) return;
    if (position<capacity) (*count)++; else position--;
    while (position>0 && distance<distances[position-1]) {
        distances[position]=distances[position-1]; indices[position]=indices[position-1]; position--;
    }
    distances[position]=distance; indices[position]=index;
}
BOPL4D_FN const BoplRlPlayer *bopl_player_by_id(const BoplRlState *state, uint32_t id) {
    for (uint32_t i=0; i<state->header.player_count; i++)
        if (state->players[i].id==id) return &state->players[i];
    return NULL;
}
/* Unity's Invisibility fade is EaseInOut(0,1,1,0): a smoothstep from opaque
 * to transparent. During the final invisible second it fades back in using
 * 5-timeSpentInvisible, exactly like SetAlphaBasedOnPlayerInvisibiltyTime. */
BOPL4D_FN double bopl_invisibility_alpha(const BoplRlPlayer *player) {
    double remaining=5.0-bopl_fix(player->invisible_time);
    double t=remaining<=1.0?remaining:bopl_fix(player->ability_time);
    if (t<0.0) t=0.0; else if (t>1.0) t=1.0;
    return 1.0-3.0*t*t+2.0*t*t*t;
}
BOPL4D_FN int bopl_spatial_hidden(const BoplRlState *state, uint32_t owner, uint32_t self) {
    if (!owner || owner==self) return 0;
    const BoplRlPlayer *player=bopl_player_by_id(state, owner);
    return player && (player->flags&4) && bopl_invisibility_alpha(player)<0.2;
}
BOPL4D_FN int bopl_entity_visible(const BoplRlState *state, const BoplRlEntity *entity, uint32_t self) {
    if (!(entity->flags&BOPL_ENTITY_ACTIVE) || (entity->flags&BOPL_ENTITY_DESTROYED)) return 0;
    /* Only the invisible player's Player-layer body/hurtbox hierarchy hides.
     * Independent owned objects remain visible, including disabled smoke. */
    if (entity->layer==8 && bopl_spatial_hidden(state, entity->owner, self)
        && !(entity->flags&(BOPL_ENTITY_PICKUP|BOPL_ENTITY_PROJECTILE|BOPL_ENTITY_SPARK_NODE
                            |BOPL_ENTITY_FLAMMABLE_SMOKE|BOPL_ENTITY_EXPLOSIVE))) return 0;
    return 1;
}
BOPL4D_FN int bopl_slime_inventory(const BoplRlSlime *slime) {
    int count=0;
    for (int slot=0; slot<3; slot++)
        if ((slime->ability_flags[slot]&1) && slime->abilities[slot]>0
            && slime->abilities[slot]<=BOPL_ABILITY_COUNT) count++;
    return count;
}
BOPL4D_FN const BoplRlSlime *bopl_controlling_slime(const BoplRlState *state,
                                                     uint32_t owner, double x, double y) {
    const BoplRlSlime *best=NULL; double best_distance=0.0; int preferred=0;
    for (uint32_t i=0; i<state->header.slime_count; i++) {
        const BoplRlSlime *slime=&state->slimes[i];
        if (slime->owner!=owner || !bopl_slime_inventory(slime)) continue;
        int candidate_preferred=(slime->flags&BOPL_SLIME_SUPPRESS_INPUT)==0;
        double dx=bopl_fix(slime->x)-x, dy=bopl_fix(slime->y)-y, distance=dx*dx+dy*dy;
        if (!best || candidate_preferred>preferred
            || (candidate_preferred==preferred && distance<best_distance)) {
            best=slime; best_distance=distance; preferred=candidate_preferred;
        }
    }
    return best;
}
BOPL4D_FN int bopl_rope_group_end(const BoplRlState *state, uint32_t begin) {
    const BoplRlEntity *first=&state->entities[begin];
    uint32_t end=begin+1, hierarchy=first->hierarchy;
    while (end<state->header.entity_count) {
        const BoplRlEntity *next=&state->entities[end];
        if (next->kind!=3 || next->owner!=first->owner || next->hierarchy<=hierarchy) break;
        hierarchy=next->hierarchy; end++;
    }
    return (int)end;
}
BOPL4D_FN int bopl_rope_span_kept(int ordinal, int count) {
    if (count<=3) return 1;
    return ordinal==0 || ordinal==(count-1)/2 || ordinal==count-1;
}
BOPL4D_FN double bopl_surface_distance(const BoplRlEntity *entity, double x, double y) {
    double px=bopl_fix(entity->x), py=bopl_fix(entity->y), radius=bopl_fix(entity->radius);
    if (entity->kind==3 || entity->kind==4) {
        double sx=bopl_fix(entity->right_x), sy=bopl_fix(entity->right_y);
        double length=sx*sx+sy*sy, t=length>0?((x-px)*sx+(y-py)*sy)/length:0.0;
        if (t<0.0) t=0.0; else if (t>1.0) t=1.0;
        double dx=x-(px+t*sx), dy=y-(py+t*sy);
        return sqrt(dx*dx+dy*dy)-radius;
    }
    double dx=px-x, dy=py-y;
    return sqrt(dx*dx+dy*dy)-radius;
}
BOPL4D_FN void bopl_player_features(float *out, const BoplRlState *state,
                                    const BoplRlPlayer *player, const BoplRlPlayer *self,
                                    double self_x, double self_y) {
    memset(out, 0, BOPL_PLAYER_FEATURES*sizeof(float));
    int hidden=bopl_spatial_hidden(state, player->id, self->id), bodies=0;
    for (uint32_t i=0; i<state->header.slime_count; i++)
        if (state->slimes[i].owner==player->id) bodies++;
    out[0]=1; out[1]=player->id==self->id; out[2]=player->team==self->team;
    out[3]=(player->flags&1)!=0; out[4]=(player->flags&2)!=0;
    out[5]=bopl_clip(BOPL4_DIVC(player->alive_bodies, 6.0)); out[6]=bopl_clip(BOPL4_DIVC(player->stocks, 10.0));
    out[7]=bopl_clip(BOPL4_DIVC(player->kills, 10.0)); out[8]=bopl_clip(BOPL4_DIVC(player->deaths, 10.0));
    if (!hidden) {
        out[9]=bopl_clip((bopl_fix(player->x)-self_x)/64.0);
        out[10]=bopl_clip((bopl_fix(player->y)-self_y)/64.0);
        out[11]=bopl_clip(BOPL4_DIVC(bopl_fix(player->scale), 3.0));
        out[16]=1;
    }
    out[12]=(player->flags&4)!=0; out[13]=(player->flags&8)!=0;
    out[14]=(player->flags&16)!=0; out[15]=(player->flags&(32|64))!=0;
    out[17]=bopl_clip(BOPL4_DIVC(bopl_fix(player->invisible_time), 5.0));
    out[18]=bopl_clip(BOPL4_DIVC(bopl_fix(player->ability_time), 5.0));
    out[19]=bopl_clip(bodies/(double)BOPL_OBS_SLIMES);
}

/* Relational block (BOPL4_RELATION) of a body at relative position r moving
 * at relative velocity v (both versus the observer), with `radius` taken off
 * the distance. `ballistic` adds the gravity-arc closest approach (projectiles). */
BOPL4D_FN void bopl4_relation(float *out, double dx, double dy, double dvx, double dvy, double radius, int ballistic, int ablation) {
    double r=sqrt(dx*dx+dy*dy), rr=r>1e-9?1.0/r:0.0;
    double rv=dx*dvx+dy*dvy, vv=dvx*dvx+dvy*dvy;
    double tca=vv>1e-9?-rv/vv:0.0;
    if (tca<0.0) tca=0.0; else if (tca>BOPL4_HORIZON) tca=BOPL4_HORIZON;
    double mx=dx+dvx*tca, my=dy+dvy*tca, miss=sqrt(mx*mx+my*my)-radius;
    out[0]=bopl_clip((r-radius)/BOPL4_POSITION_SCALE);
    out[1]=bopl_clip(dy*rr); out[2]=bopl_clip(dx*rr);
    out[3]=bopl_clip(BOPL4_DIVC(dvx, BOPL4_VELOCITY_SCALE)); out[4]=bopl_clip(BOPL4_DIVC(dvy, BOPL4_VELOCITY_SCALE));
    out[5]=bopl_clip(BOPL4_DIVC(-rv*rr, BOPL4_VELOCITY_SCALE));                 /* closing speed (>0 approaching) */
    out[6]=bopl_clip(BOPL4_DIVC((dx*dvy-dy*dvx)*rr, BOPL4_VELOCITY_SCALE));     /* tangential speed */
    out[7]=bopl_clip(tca/BOPL4_HORIZON);
    out[8]=bopl_clip(miss/16.0);
    out[9]=bopl_clip(BOPL4_DIVC(sqrt(vv), BOPL4_VELOCITY_SCALE));
    out[10]=0; out[11]=0;
    if (ballistic) {
        /* Arc under gravity (observer assumed on the ground), sampled every
         * 1/15 s to the horizon; the nearest sample's distance and time. */
        double best=r-radius, best_t=0.0, best_squared=dx*dx+dy*dy;
        for (int k=1; k<=30; k++) {
#ifdef __CUDA_ARCH__
            double t=bopl4_arc_t[k];
#else
            double t=k/15.0;
#endif
            double px=dx+dvx*t, py=dy+dvy*t-0.5*BOPL4_GRAVITY*t*t, squared=px*px+py*py;
            /* sqrt and -radius are monotone, so a sample can only beat `best`
             * when its square is smaller: skip the sqrt otherwise. */
            if (squared>=best_squared) continue;
            double d=sqrt(squared)-radius;
            if (d<best) { best=d; best_t=t; best_squared=squared; }
        }
        out[10]=bopl_clip(best/16.0); out[11]=bopl_clip(best_t/BOPL4_HORIZON);
    }
    if (ablation&BOPL4_ABLATE_RELATIONAL) memset(out, 0, BOPL4_RELATION*sizeof(float));
}
/* Controlling-slime block (BOPL4_BODY); NULL or hidden bodies are all zero.
 * sin_r, cos_r: sinf and cosf of the body's rotation (as a float). */
BOPL4D_FN void bopl4_body(float *out, const BoplRlSlime *slime, float sin_r, float cos_r) {
    memset(out, 0, BOPL4_BODY*sizeof(float));
    if (!slime) return;
    double vx=bopl_fix(slime->vx), vy=bopl_fix(slime->vy);
    out[0]=1; out[1]=bopl_clip(BOPL4_DIVC(vx, BOPL4_VELOCITY_SCALE)); out[2]=bopl_clip(BOPL4_DIVC(vy, BOPL4_VELOCITY_SCALE));
    out[3]=bopl_clip(BOPL4_DIVC(sqrt(vx*vx+vy*vy), BOPL4_VELOCITY_SCALE));
    out[4]=sin_r; out[5]=cos_r;
    if (slime->state<11) out[6+slime->state]=1;
    out[17]=(slime->flags&BOPL_SLIME_IN_ABILITY)!=0; out[18]=(slime->flags&BOPL_SLIME_SUPPRESS_INPUT)!=0;
    out[19]=(slime->flags&BOPL_SLIME_SPAWNED)!=0; out[20]=(slime->flags&BOPL_SLIME_ABILITYLESS)!=0;
    out[21]=bopl_clip(BOPL4_DIVC(bopl_fix(slime->scale), 3.0));
    out[22]=slime->vx==0 && slime->vy==0;
    out[23]=((slime->flags>>BOPL_SLIME_PICKUP_SLOT_SHIFT)&BOPL_SLIME_PICKUP_SLOT_MASK)/2.0f;
}
/* The primary enemy of `self`: the enemy player whose controlling slime is
 * nearest among the visible ones, else the first enemy (position unknown). */
typedef struct {
    const BoplRlPlayer *player; const BoplRlSlime *slime;
    int known; double x, y, vx, vy;
} Bopl4Enemy;
BOPL4D_FN Bopl4Enemy bopl4_primary_enemy(const BoplRlState *state, const BoplRlPlayer *self, double self_x, double self_y) {
    Bopl4Enemy best; memset(&best, 0, sizeof(best));
    double best_distance=0.0;
    for (uint32_t i=0; i<state->header.player_count; i++) {
        const BoplRlPlayer *player=&state->players[i];
        if (player->id==self->id || player->team==self->team) continue;
        int hidden=bopl_spatial_hidden(state, player->id, self->id);
        double x=bopl_fix(player->x), y=bopl_fix(player->y);
        const BoplRlSlime *slime=bopl_controlling_slime(state, player->id, x, y);
        double dx=x-self_x, dy=y-self_y, distance=dx*dx+dy*dy;
        int candidate_known=!hidden;
        if (!best.player || candidate_known>best.known || (candidate_known==best.known && distance<best_distance)) {
            best.player=player; best.slime=hidden?NULL:slime; best.known=candidate_known;
            best.x=x; best.y=y; best.vx=slime?bopl_fix(slime->vx):0.0; best.vy=slime?bopl_fix(slime->vy):0.0;
            best_distance=distance;
        }
    }
    return best;
}
/* The eighth of the circle (atan2(dy, dx)+pi)*8/(2 pi) falls in, clamped to
 * 0-7. Away from the octant edges (the axes and diagonals) it is read off the
 * signs and which of |dx|, |dy| is larger: there the float atan2 is ~1e-16
 * from the exact angle and the edges are 1e-6 away, so the two agree. Near an
 * edge it takes the atan2 path as written. */
#ifndef __CUDACC__
static inline int bopl4_sector(double dx, double dy) {
    double ax=fabs(dx), ay=fabs(dy), big=fmax(ax, ay);
    if (fmin(ax, ay)<=1e-6*big || fabs(ax-ay)<=1e-6*big) {
        double angle=atan2(dy, dx)+3.14159265358979323846;
        int sector=(int)(angle*(8.0/6.28318530717958647692));
        if (sector<0) sector=0; else if (sector>7) sector=7;
        return sector;
    }
    int steep=ay>ax;
    if (dy<0) return dx<0 ? (steep?1:0) : (steep?2:3);
    return dx>0 ? (steep?5:4) : (steep?6:7);
}
#endif

/* The global block (BOPL4_GLOBAL) from the entity and slime passes. Lane
 * `lane` of `lanes` writes its share (the host: 0 of 1, the whole block;
 * the device a warp, the per-ability and per-sector columns side by side). */
typedef struct {
    int count, active_count, n_nonstatic, n_pickup, visible_slimes, time_stop_known;
    uint32_t time_stop_remaining;
    int sectors[8], source_counts[BOPL_ABILITY_COUNT];
} Bopl4Counts;
BOPL4D_FN void bopl4_global_block(float *obs, const BoplRlState *state, const Bopl4Counts *c, uint32_t max_episode_ticks,
                                  int space, const BoplRlSlime *controller, int lane, int lanes) {
    for (int id=lane; id<BOPL_ABILITY_COUNT; id+=lanes) obs[14+id]=bopl_clip(c->source_counts[id]/(double)BOPL_COUNT_SCALE);
    for (int sector=lane; sector<8; sector+=lanes) obs[55+sector]=bopl_clip(c->sectors[sector]/(double)BOPL_COUNT_SCALE);
    if (lane) return;
    obs[0]=bopl_clip(BOPL4_DIVC(bopl_fix(state->header.simulation_time), 120.0));
    obs[1]=bopl_clip(bopl_fix(state->header.player_time_scale));
    obs[2]=bopl_clip((double)state->header.episode_ticks/max_episode_ticks);
    obs[3]=state->header.player_count/6.0f;
    obs[4]=bopl_clip(BOPL4_DIVC((double)c->active_count, BOPL4_ENTITIES));
    obs[5]=bopl_clip(BOPL4_DIVC((double)(c->active_count-c->count), BOPL4_ENTITIES));
    obs[6]=bopl_clip((double)c->visible_slimes/BOPL4_SLIMES);
    obs[7]=bopl_clip((double)(c->visible_slimes>BOPL4_SLIMES?c->visible_slimes-BOPL4_SLIMES:0)/BOPL4_SLIMES);
    /* v3's absolute self position was a map lookup key and is dropped (the
     * terrain block replaces it). obs[8] reuses the first dead column for the
     * one mechanical map property: LevelType.space, which halves the gravity
     * modifier on players and ropes, removes water and moves the lower blast
     * boundary. Every non-space map leaves it 0, exactly as the dead column
     * was, so nothing measured on grass or snow changes. Projectile arcs are
     * NOT affected by space (gravityAccel carries no level-type factor), so
     * BOPL4_GRAVITY stays one constant. */
    obs[8]=(float)space; obs[9]=0;
    double level_time=bopl_fix(state->header.simulation_time_since_level_loaded);
    obs[10]=bopl_clip(BOPL4_DIVC(fmax(0.0, 120.0-level_time), 120.0));
    obs[11]=state->header.sudden_death!=0;
    obs[12]=state->header.sudden_death?bopl_clip(BOPL4_DIVC(fmax(0.0, level_time-120.0), 30.0)):0.0f;
    obs[13]=bopl_clip((max_episode_ticks-(double)state->header.episode_ticks)/max_episode_ticks);
    obs[53]=bopl_clip(BOPL4_DIVC(c->n_nonstatic, (double)BOPL4_ENTITIES));
    obs[54]=bopl_clip(c->n_pickup/(double)BOPL_COUNT_SCALE);
    obs[63]=c->time_stop_known?bopl_clip(BOPL4_DIVC(c->time_stop_remaining, 300.0)):0.0f;
    obs[64]=c->time_stop_known;
    obs[65]=controller?((controller->flags>>BOPL_SLIME_PICKUP_SLOT_SHIFT)&BOPL_SLIME_PICKUP_SLOT_MASK)/2.0f:0.0f;
}
/* The slime pass: the nearest visible slimes (nearby_slimes, by squared
 * distance) and the nearest ability slots of the visible and the two
 * controlling bodies (slot_indices: slime*3+slot; the observer's first, then
 * the enemy's), with the visible slime and slot counts. */
BOPL4D_FN void bopl4_slime_pass(const BoplRlState *state, const BoplRlPlayer *self, double self_x, double self_y,
                                const BoplRlSlime *controller, const BoplRlSlime *enemy_controller,
                                uint32_t *nearby_slimes, int *slime_count, uint32_t *slot_indices, int *slot_count,
                                int *visible_slimes, int *visible_slots) {
    double slime_distances[BOPL4_SLIMES], slot_distances[BOPL4_SLOTS];
    *slime_count=*slot_count=*visible_slimes=*visible_slots=0;
    for (uint32_t i=0; i<state->header.slime_count; i++) {
        const BoplRlSlime *slime=&state->slimes[i];
        int hidden=bopl_spatial_hidden(state, slime->owner, self->id);
        int forced=slime==controller || slime==enemy_controller;
        if (hidden && !forced) continue;
        double dx=bopl_fix(slime->x)-self_x, dy=bopl_fix(slime->y)-self_y;
        double distance=dx*dx+dy*dy;
        if (!hidden) {
            bopl_nearest(nearby_slimes, slime_distances, slime_count, BOPL4_SLIMES, i, distance);
            (*visible_slimes)++;
        }
        for (int slot=0; slot<3; slot++) {
            uint32_t id=slime->abilities[slot], flags=slime->ability_flags[slot];
            if (!(flags&1) || id==0 || id>BOPL_ABILITY_COUNT) continue;
            double priority=distance+slot*1.0e-9;
            if (slime==controller) priority-=1.0e12; else if (slime==enemy_controller) priority-=1.0e11;
            bopl_nearest(slot_indices, slot_distances, slot_count, BOPL4_SLOTS, i*3u+(uint32_t)slot, priority);
            (*visible_slots)++;
        }
    }
}
/* Whether entity k is kept for the row (rope thinning, visibility), from
 * the rope group around it: k continues k-1's group when both are rope
 * pieces of one owner with rising hierarchy (bopl_rope_group_end). */
BOPL4D_FN int bopl4_rope_continues(const BoplRlState *state, uint32_t k) {
    if (k==0) return 0;
    const BoplRlEntity *e=&state->entities[k], *p=&state->entities[k-1];
    return e->kind==3 && p->kind==3 && e->owner==p->owner && e->hierarchy>p->hierarchy;
}
BOPL4D_FN int bopl4_entity_kept(const BoplRlState *state, uint32_t k, uint32_t self_id) {
    const BoplRlEntity *entity=&state->entities[k];
    if (entity->kind==3) {
        uint32_t begin=k, end=k+1;
        while (bopl4_rope_continues(state, begin)) begin--;
        while (end<state->header.entity_count && bopl4_rope_continues(state, end)) end++;
        if (!bopl_rope_span_kept((int)(k-begin), (int)(end-begin))) return 0;
    }
    return bopl_entity_visible(state, entity, self_id);
}

/* The opponent block (BOPL4_OPPONENT): the primary enemy's player row and
 * body, its relation to the observer. */
BOPL4D_FN void bopl4_opponent_block(float *opp, const BoplRlState *state, const Bopl4Enemy *enemy, const BoplRlPlayer *self,
                                    double self_x, double self_y, double self_vx, double self_vy,
                                    float sin_r, float cos_r, int ablation) {
    memset(opp, 0, BOPL4_OPPONENT*sizeof(float));
    if (!enemy->player) return;
    bopl_player_features(opp, state, enemy->player, self, self_x, self_y);
    bopl4_body(opp+BOPL_PLAYER_FEATURES, enemy->slime, sin_r, cos_r);
    if (enemy->known) bopl4_relation(opp+BOPL4_SELF, enemy->x-self_x, enemy->y-self_y, enemy->vx-self_vx, enemy->vy-self_vy, 0.0, 0, ablation);
}
/* Player row i (the player at `index`): its v3 row and its relation. */
BOPL4D_FN void bopl4_player_row(float *row, const BoplRlState *state, const BoplRlPlayer *player, const BoplRlPlayer *self,
                                double self_x, double self_y, double self_vx, double self_vy, int ablation) {
    bopl_player_features(row, state, player, self, self_x, self_y);
    memset(row+BOPL_PLAYER_FEATURES, 0, BOPL4_RELATION*sizeof(float));
    if (!bopl_spatial_hidden(state, player->id, self->id)) {
        double x=bopl_fix(player->x), y=bopl_fix(player->y);
        const BoplRlSlime *body=bopl_controlling_slime(state, player->id, x, y);
        bopl4_relation(row+BOPL_PLAYER_FEATURES, x-self_x, y-self_y,
                       (body?bopl_fix(body->vx):0.0)-self_vx, (body?bopl_fix(body->vy):0.0)-self_vy, 0.0, 0, ablation);
    }
}
/* An ability slot row (BOPL4_SLOT_FEATURES): slot `slot` of `slime`. */
BOPL4D_FN void bopl4_slot_row(float *out, const BoplRlState *state, const BoplRlSlime *slime, uint32_t slot,
                              const BoplRlPlayer *self, double self_x, double self_y, const Bopl4Enemy *enemy) {
    uint32_t flags=slime->ability_flags[slot], id=slime->abilities[slot];
    double cooldown=bopl_fix(slime->cooldowns[slot]), timer=bopl_fix(slime->timers[slot]);
    int hidden=bopl_spatial_hidden(state, slime->owner, self->id);
    double dx=bopl_fix(slime->x)-self_x, dy=bopl_fix(slime->y)-self_y;
    float row[BOPL4_SLOT_FEATURES]={0};
    row[0]=1; row[1]=slime->owner==self->id; row[2]=bopl_ally(state, slime->owner, self);
    row[3]=slot/2.0f; row[4]=bopl_clip(cooldown>0?timer/cooldown:0.0);
    row[5]=bopl_clip(BOPL4_DIVC(cooldown, 60.0)); row[6]=(flags&2)!=0; row[7]=(flags&4)!=0;
    row[8]=(flags&BOPL_ABILITY_ACTIVE)!=0;
    row[10]=id==4;
    row[9]=row[10]?((flags>>BOPL_ABILITY_CHARGE_SHIFT)&BOPL_ABILITY_CHARGE_MASK)/4.0f:0.0f;
    if (!hidden) { row[11]=bopl_clip(dx/64.0); row[12]=bopl_clip(dy/64.0); row[16]=bopl_clip(sqrt(dx*dx+dy*dy)/64.0); }
    row[13]=bopl_clip(BOPL4_DIVC(fmax(0.0, cooldown-timer), 10.0));
    row[14]=!hidden;
    row[15]=enemy->player && slime->owner==enemy->player->id;
    row[BOPL4_SLOT_ABILITY_ID]=(float)id;
    memcpy(out, row, sizeof(row));
}
/* An entity row (BOPL4_ENTITY_FEATURES). */
BOPL4D_FN void bopl4_entity_row(float *out, const BoplRlState *state, const BoplRlEntity *entity, const BoplRlPlayer *self,
                                double self_x, double self_y, double self_vx, double self_vy, const Bopl4Enemy *enemy, int ablation) {
    float row[BOPL4_ENTITY_FEATURES]={0};
    double ex=bopl_fix(entity->x), ey=bopl_fix(entity->y), evx=bopl_fix(entity->vx), evy=bopl_fix(entity->vy);
    double radius=bopl_fix(entity->radius);
    row[0]=1; if (entity->kind<5) row[1+entity->kind]=1;
    row[6]=entity->owner==self->id; row[7]=bopl_ally(state, entity->owner, self);
    row[8]=entity->owner!=0 && !row[6] && !row[7];
    row[9]=bopl_clip((ex-self_x)/64.0); row[10]=bopl_clip((ey-self_y)/64.0);
    row[11]=bopl_clip(BOPL4_DIVC(evx, BOPL4_VELOCITY_SCALE)); row[12]=bopl_clip(BOPL4_DIVC(evy, BOPL4_VELOCITY_SCALE));
    row[13]=bopl_clip(bopl_fix(entity->right_x)/64.0); row[14]=bopl_clip(bopl_fix(entity->right_y)/64.0);
    row[15]=bopl_clip(bopl_fix(entity->up_x)/64.0); row[16]=bopl_clip(bopl_fix(entity->up_y)/64.0);
    row[17]=bopl_clip(radius/16.0);
    row[18]=(entity->flags&BOPL_ENTITY_ENABLED)!=0;
    row[19]=(entity->flags&BOPL_ENTITY_TRIGGER)!=0;
    row[20]=(entity->flags&BOPL_ENTITY_COMPOSITE)!=0;
    row[21]=entity->layer/32.0f;
    row[22]=(entity->flags&BOPL_ENTITY_TIMER_KNOWN)?bopl_clip(BOPL4_DIVC(entity->reserved[1], (double)BOPL_TIMER_TICKS)):0.0f;
    row[23]=(entity->flags&BOPL_ENTITY_TIMER_KNOWN)!=0;
    row[24]=bopl_clip(bopl_fix(entity->clock_rate));
    row[25]=(entity->flags&BOPL_ENTITY_PICKUP)!=0;
    row[26]=(entity->flags&BOPL_ENTITY_PROJECTILE)!=0;
    row[27]=(entity->flags&BOPL_ENTITY_SPARK_NODE)!=0;
    row[28]=(entity->flags&BOPL_ENTITY_FLAMMABLE_SMOKE)!=0;
    row[29]=(entity->flags&BOPL_ENTITY_EXPLOSIVE)!=0;
    row[30]=(entity->flags&BOPL_ENTITY_HITBOX)!=0;
    row[31]=(entity->flags&BOPL_ENTITY_CONTACT_DAMAGE)!=0;
    row[32]=(entity->flags&BOPL_ENTITY_CONTACT_DAMAGE_KNOWN)!=0;
    int projectile=(entity->flags&BOPL_ENTITY_PROJECTILE)!=0 && entity->inverse_mass>0;
    /* Relation to the observer: for segments (ropes, platform edges) the
     * surface distance is the segment's, the direction is to the centre. */
    bopl4_relation(row+33, ex-self_x, ey-self_y, evx-self_vx, evy-self_vy, radius, projectile, ablation);
    row[33]=bopl_clip(bopl_surface_distance(entity, self_x, self_y)/64.0);
    if (enemy->known) {
        /* The same body seen from the primary enemy: is it about to hit them. */
        float versus[BOPL4_RELATION];
        bopl4_relation(versus, ex-enemy->x, ey-enemy->y, evx-enemy->vx, evy-enemy->vy, radius, projectile, ablation);
        row[45]=versus[0]; row[46]=projectile?versus[10]:versus[8];
    }
    row[BOPL4_ENTITY_ABILITY_ID]=entity->reserved[0]<=BOPL_ABILITY_COUNT?(float)entity->reserved[0]:0.0f;
    memcpy(out, row, sizeof(row));
}
/* A slime row (BOPL4_SLIME_FEATURES); `controlling`: the slime is the
 * observer's or the primary enemy's controlling body. */
BOPL4D_FN void bopl4_slime_row(float *out, const BoplRlState *state, const BoplRlSlime *slime, const BoplRlPlayer *self,
                               double self_x, double self_y, double self_vx, double self_vy, const Bopl4Enemy *enemy,
                               int controlling, float sin_r, float cos_r, int ablation) {
    const BoplRlPlayer *owner=bopl_player_by_id(state, slime->owner);
    float row[BOPL4_SLIME_FEATURES]={0};
    double sx=bopl_fix(slime->x), sy=bopl_fix(slime->y), svx=bopl_fix(slime->vx), svy=bopl_fix(slime->vy);
    row[0]=1; row[1]=slime->owner==self->id; row[2]=bopl_ally(state, slime->owner, self);
    row[3]=(slime->flags&BOPL_SLIME_ACTIVE)!=0; row[4]=(slime->flags&BOPL_SLIME_SPAWNED)!=0;
    row[5]=(slime->flags&BOPL_SLIME_IN_ABILITY)!=0; row[6]=(slime->flags&BOPL_SLIME_SUPPRESS_INPUT)!=0;
    row[7]=bopl_clip((sx-self_x)/64.0); row[8]=bopl_clip((sy-self_y)/64.0);
    row[9]=bopl_clip(BOPL4_DIVC(svx, BOPL4_VELOCITY_SCALE)); row[10]=bopl_clip(BOPL4_DIVC(svy, BOPL4_VELOCITY_SCALE));
    row[11]=bopl_clip(BOPL4_DIVC(bopl_fix(slime->scale), 3.0));
    if (slime->state<11) row[12+slime->state]=1;
    row[23]=sin_r; row[24]=cos_r;
    row[25]=owner?bopl_clip(BOPL4_DIVC(owner->stocks, 10.0)):0.0f;
    row[26]=enemy->player && slime->owner==enemy->player->id;
    row[27]=controlling;   /* a controlling body */
    row[28]=(slime->flags&BOPL_SLIME_ABILITYLESS)!=0; row[29]=1;
    bopl4_relation(row+30, sx-self_x, sy-self_y, svx-self_vx, svy-self_vy, 0.0, 0, ablation);
    memcpy(out, row, sizeof(row));
}

/* ---- The rays' scene (bopl_rays_v4.h Bopl4Scene), piece by piece ---- */
/* The scene's head: flags from the writer bits, the body, the aim applied
 * this decision (aimed: its direction from the aim table) and the enemy. */
BOPL4D_FN void bopl4_scene_head(Bopl4Scene *s, int ablation, double x, double y, double svx, double svy,
                                int aimed, float aim_dx, float aim_dy, const Bopl4Enemy *enemy) {
    s->flags=(ablation&BOPL4_ABLATE_RAYS?0:BOPL4F_FAN)|(ablation&BOPL4_ABLATE_RAYS16?BOPL4F_HALF:0)|(ablation&BOPL4_ABLATE_AIM?BOPL4F_AIM:0);
    s->x=(float)x; s->y=(float)y; s->svx=(float)svx; s->svy=(float)svy;
    s->aim_dx=s->aim_dy=0.0f;
    if (aimed) { s->flags|=BOPL4F_AIMED; s->aim_dx=aim_dx; s->aim_dy=aim_dy; }
    s->enemy_x=s->enemy_y=0.0f;
    if (enemy->known) { s->flags|=BOPL4F_ENEMY; s->enemy_x=(float)enemy->x; s->enemy_y=(float)enemy->y; }
    s->axis_count=s->rot_count=s->circle_count=0; s->axis_lethal=s->rot_lethal=s->rot_skew=0;
}
/* What entity e contributes to the scene's terrain: nothing (0), water (1,
 * *top its top), a wall candidate (2 aligned, 3 rotated) with its fields in
 * w[]: aligned cx, cy, radius, ex, ey; rotated cx, cy, radius, rx, ry, ux,
 * uy, ex, ey; *lethal (layer 16) and *skew (rotated, not orthogonal). The
 * scene keeps the first BOPL4F_WALLS candidates in entity order. */
enum { BOPL4_SCENE_NONE=0, BOPL4_SCENE_WATER=1, BOPL4_SCENE_AXIS=2, BOPL4_SCENE_ROT=3 };
BOPL4D_FN int bopl4_scene_entity(const BoplRlEntity *e, float *w, double *top, int *lethal, int *skew) {
    if (!(e->flags&1) || (e->flags&4)) return BOPL4_SCENE_NONE;
    if (e->layer==BOPL4_WATER_LAYER) {
        *top=bopl_fix(e->y)+fabs(bopl_fix(e->up_y))+fabs(bopl_fix(e->right_y));
        return BOPL4_SCENE_WATER;
    }
    if (e->kind!=2 || !(e->layer==11 || e->layer==16 || e->layer==21)) return BOPL4_SCENE_NONE;
    double rx=bopl_fix(e->right_x), ry=bopl_fix(e->right_y), ux=bopl_fix(e->up_x), uy=bopl_fix(e->up_y);
    double ex=sqrt(rx*rx+ry*ry), ey=sqrt(ux*ux+uy*uy);
    if (ex>1e-6) { rx/=ex; ry/=ex; } else { rx=1; ry=0; }
    if (ey>1e-6) { ux/=ey; uy/=ey; } else { ux=-ry; uy=rx; }
    *lethal=e->layer==16;
    int straight=(rx==1||rx==-1) && ry==0 && ux==0 && (uy==1||uy==-1);
    int turned=rx==0 && (ry==1||ry==-1) && (ux==1||ux==-1) && uy==0;
    w[0]=(float)bopl_fix(e->x); w[1]=(float)bopl_fix(e->y); w[2]=(float)bopl_fix(e->radius);
    if (straight||turned) {
        w[3]=(float)(straight?ex:ey); w[4]=(float)(straight?ey:ex);
        *skew=0;
        return BOPL4_SCENE_AXIS;
    }
    w[3]=(float)rx; w[4]=(float)ry; w[5]=(float)ux; w[6]=(float)uy; w[7]=(float)ex; w[8]=(float)ey;
    *skew=fabs(rx*ux+ry*uy)>1e-9;
    return BOPL4_SCENE_ROT;
}
/* Store wall candidate w (bopl4_scene_entity) as aligned wall k or rotated wall k. */
BOPL4D_FN void bopl4_scene_axis(Bopl4Scene *s, int k, const float *w, int lethal) {
    s->axis_cx[k]=w[0]; s->axis_cy[k]=w[1]; s->axis_radius[k]=w[2]; s->axis_ex[k]=w[3]; s->axis_ey[k]=w[4];
    if (lethal) s->axis_lethal|=1ull<<k;
}
BOPL4D_FN void bopl4_scene_rot(Bopl4Scene *s, int k, const float *w, int lethal, int skew) {
    s->rot_cx[k]=w[0]; s->rot_cy[k]=w[1]; s->rot_radius[k]=w[2];
    s->rot_rx[k]=w[3]; s->rot_ry[k]=w[4]; s->rot_ux[k]=w[5]; s->rot_uy[k]=w[6]; s->rot_ex[k]=w[7]; s->rot_ey[k]=w[8];
    if (lethal) s->rot_lethal|=1ull<<k;
    if (skew) s->rot_skew|=1ull<<k;
}
/* A circle of the scene: an observed entity with mass (not a rope or a
 * platform edge; class 2 own, 0 otherwise), or a visible enemy slime (class
 * 1). Returns 0 when it is none. */
BOPL4D_FN int bopl4_entity_circle(const BoplRlEntity *e, const BoplRlPlayer *self, float *c, int32_t *cls) {
    if (e->inverse_mass<=0 || e->kind==3 || e->kind==4) return 0;
    c[0]=(float)bopl_fix(e->x); c[1]=(float)bopl_fix(e->y); c[2]=(float)fmax(bopl_fix(e->radius), 0.25);
    c[3]=(float)bopl_fix(e->vx); c[4]=(float)bopl_fix(e->vy);
    *cls=e->owner==self->id?2:0;
    return 1;
}
BOPL4D_FN int bopl4_slime_circle(const BoplRlState *state, const BoplRlSlime *slime, const BoplRlPlayer *self, float *c, int32_t *cls) {
    if (!(slime->flags&BOPL_SLIME_ACTIVE)) return 0;
    const BoplRlPlayer *owner=bopl_player_by_id(state, slime->owner);
    if (!owner || owner->id==self->id || owner->team==self->team) return 0;
    double r=BOPL4_SLIME_RADIUS*bopl_fix(slime->scale);
    c[0]=(float)bopl_fix(slime->x); c[1]=(float)bopl_fix(slime->y); c[2]=(float)(r>0?r:BOPL4_SLIME_RADIUS);
    c[3]=(float)bopl_fix(slime->vx); c[4]=(float)bopl_fix(slime->vy);
    *cls=1;
    return 1;
}
BOPL4D_FN void bopl4_scene_circle(Bopl4Scene *s, int k, const float *c, int32_t cls) {
    s->cx[k]=c[0]; s->cy[k]=c[1]; s->cr[k]=c[2]; s->cvx[k]=c[3]; s->cvy[k]=c[4]; s->cclass[k]=cls;
}

/* privileged[43, 44]: the critic's true enemy offset `d` (the undelayed
 * enemy position minus the observer's) minus the actor's delayed one, the
 * opponent block's column 9 or 10. */
BOPL4D_FN float bopl4_stale_enemy(double d, float delayed) { return bopl_clip(d/64.0-delayed); }

/* ---- The packed state a row ships instead of its features (pack mode 4) ----
 * bopl_obs_pack.h: the slot holds this head, the row's own-action block
 * (BOPL4_PREVIOUS) and critic block (BOPL4_PRIVILEGED) as the host wrote
 * them, then the delayed state the row is encoded from: its header, the six
 * players, the entity rows, the slime rows, per slime sinf and cosf of
 * its rotation, and per entity its sector. The GPU writes everything else
 * (bopl_rays_v4.cu). */
typedef struct {
    uint32_t mode, player, ablation, flags;   /* flags: BOPL4_STATE_* */
    uint32_t max_episode_ticks, words, entity_count, slime_count;   /* words: the slot's used length */
    float aim_dx, aim_dy, pad[2];
    double enemy_dx, enemy_dy;   /* privileged[43, 44] before the delayed view is taken off (bopl4_privileged) */
} Bopl4StateHead;
enum { BOPL4_STATE_SPACE=1, BOPL4_STATE_AIMED=2, BOPL4_STATE_ENEMY=4 };
enum {
    BOPL4_STATE_HEAD_WORDS=(int)(sizeof(Bopl4StateHead)/4),                                   /* 16 */
    BOPL4_STATE_FIXED_OFFSET=BOPL4_STATE_HEAD_WORDS,                                          /* previous, privileged */
    BOPL4_STATE_OFFSET=BOPL4_STATE_FIXED_OFFSET+BOPL4_PREVIOUS+BOPL4_PRIVILEGED,              /* 90: the state */
    BOPL4_STATE_BASE_WORDS=(int)((sizeof(BoplRlHeader)+6*sizeof(BoplRlPlayer))/4),           /* 124 */
    BOPL4_STATE_ENTITY_WORDS=(int)(sizeof(BoplRlEntity)/4), BOPL4_STATE_SLIME_WORDS=(int)(sizeof(BoplRlSlime)/4)+2
};
/* The slot words a state of e entities and s slimes takes: then a byte per
 * entity, its sector from the observer (bopl4_sector; 0 for no mass). */
BOPL4D_FN int bopl4_state_words(uint32_t e, uint32_t s) {
    return BOPL4_STATE_OFFSET+BOPL4_STATE_BASE_WORDS+(int)e*BOPL4_STATE_ENTITY_WORDS+(int)s*BOPL4_STATE_SLIME_WORDS+(int)(e+3)/4;
}

#if defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__)
#pragma GCC pop_options
#endif
#endif
