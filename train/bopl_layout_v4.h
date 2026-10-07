/* Observation layout v4, next to layout v3
 * (bopl_layout.h). Both are written from the same wire state (bopl_rl.h,
 * version 3); which one a player's row holds is selected per player
 * (BoplPuffer.layout), so a v3 checkpoint and a v4 checkpoint can share an
 * evaluation game. Row memory is BOPL_OBS_STRIDE floats, the larger layout.
 *
 * Flat block (read by one Linear): global | self | opponent | previous |
 * terrain | rays. Rays and the terrain block are how the network sees the
 * map: egocentric and map-invariant (no absolute position anywhere in the
 * row: the v3 prefix's self_x/self_y columns are zero in v4, a lookup key
 * that let a three-map policy memorise instead of reading geometry). The
 * static-collider entity rows stay as the secondary, exact description.
 * Sets (per-row MLP + max||mean pool): players, slots, entities, slimes; the
 * ability id is the last column of slot and entity rows, an exact integer used
 * only to gather the embedding. Privileged block: critic only.
 *
 * Scales: positions /64 relative to the observer, velocities /100 (walk 19,
 * jump 58, arrow 75 u/s), times /2 s unless stated. Gravity is 40 u/s^2
 * (measured on the sim: projectile vy changes by -40/s). */
#ifndef BOPL_LAYOUT_V4_H
#define BOPL_LAYOUT_V4_H
#include "bopl_layout.h"

#define BOPL4_ARCHITECTURE "bopl-v4-mingru"

enum {
    BOPL4_LAYOUT_VERSION=4,
    BOPL4_GLOBAL=66,                       /* the v3 prefix, same semantics */
    BOPL4_BODY=24,                         /* controlling-slime block shared by self and opponent */
    BOPL4_SELF=BOPL_PLAYER_FEATURES+BOPL4_BODY,           /* 44 */
    BOPL4_RELATION=12,                     /* relational features of one body versus the observer */
    BOPL4_OPPONENT=BOPL4_SELF+BOPL4_RELATION,             /* 56 */
    /* Own-action tail: the v3 12 (the decision as applied, latch, hold runs)
     * then the two decisions before it (none flag, aim cos/sin, 4 buttons
     * each): the three decisions applied inside the 9-tick observation delay,
     * so the network can dead-reckon its own state through the delay. */
    BOPL4_HISTORY=2, BOPL4_HISTORY_FEATURES=7,
    BOPL4_PREVIOUS=BOPL_PREVIOUS_FEATURES+BOPL4_HISTORY*BOPL4_HISTORY_FEATURES,   /* 26 */
    /* Egocentric terrain (map-invariant; bopl_obs_v4.h bopl4_terrain): drop to
     * the surface below, ceiling above, edge distance left/right and whether a
     * wall blocks instead, height above the water, distance to lethal terrain,
     * ground within one dash, SDF at the body and its gradient. */
    BOPL4_TERRAIN=12,
    /* Rays from the body (the primary geometry channel:
     * 32+4 rays alone beat the whole v3 observation on falls and geometry):
     * BOPL4_RAYS_AROUND uniform rays, angle 2 pi k / BOPL4_RAYS_AROUND from +x,
     * then BOPL4_RAYS_DOWN in a 120-degree fan about straight down (-135,
     * -105, -75, -45 degrees), length BOPL4_RAY_LENGTH. Per ray: distance
     * over the length (1 = nothing), a one-hot of the hit (wall, lethal wall,
     * water / blast plane, hostile projectile or object, opponent body, own
     * object), the hit's closing speed along the ray (bodies and objects). */
    BOPL4_RAYS_AROUND=32, BOPL4_RAYS_DOWN=4, BOPL4_RAYS=BOPL4_RAYS_AROUND+BOPL4_RAYS_DOWN, BOPL4_RAY_FEATURES=8,
    BOPL4_FLAT=BOPL4_GLOBAL+BOPL4_SELF+BOPL4_OPPONENT+BOPL4_PREVIOUS+BOPL4_TERRAIN+BOPL4_RAYS*BOPL4_RAY_FEATURES, /* 492 */
    BOPL4_PLAYERS=6, BOPL4_PLAYER_FEATURES=BOPL_PLAYER_FEATURES+BOPL4_RELATION,    /* 32 */
    BOPL4_SLOTS=24, BOPL4_SLOT_FEATURES=18,
    BOPL4_ENTITIES=48, BOPL4_ENTITY_FEATURES=48,
    BOPL4_SLIMES=32, BOPL4_SLIME_FEATURES=42,
    BOPL4_PRIVILEGED=48,
    BOPL4_OBS_GLOBAL_OFFSET=0,
    BOPL4_OBS_SELF_OFFSET=BOPL4_OBS_GLOBAL_OFFSET+BOPL4_GLOBAL,
    BOPL4_OBS_OPPONENT_OFFSET=BOPL4_OBS_SELF_OFFSET+BOPL4_SELF,
    BOPL4_OBS_PREVIOUS_OFFSET=BOPL4_OBS_OPPONENT_OFFSET+BOPL4_OPPONENT,
    BOPL4_OBS_TERRAIN_OFFSET=BOPL4_OBS_PREVIOUS_OFFSET+BOPL4_PREVIOUS,
    BOPL4_OBS_RAY_OFFSET=BOPL4_OBS_TERRAIN_OFFSET+BOPL4_TERRAIN,
    BOPL4_OBS_PLAYER_OFFSET=BOPL4_FLAT,
    BOPL4_OBS_SLOT_OFFSET=BOPL4_OBS_PLAYER_OFFSET+BOPL4_PLAYERS*BOPL4_PLAYER_FEATURES,
    BOPL4_OBS_ENTITY_OFFSET=BOPL4_OBS_SLOT_OFFSET+BOPL4_SLOTS*BOPL4_SLOT_FEATURES,
    BOPL4_OBS_SLIME_OFFSET=BOPL4_OBS_ENTITY_OFFSET+BOPL4_ENTITIES*BOPL4_ENTITY_FEATURES,
    BOPL4_OBS_PRIVILEGED_OFFSET=BOPL4_OBS_SLIME_OFFSET+BOPL4_SLIMES*BOPL4_SLIME_FEATURES,
    BOPL4_OBS_SIZE=BOPL4_OBS_PRIVILEGED_OFFSET+BOPL4_PRIVILEGED,                /* 4812 */
    BOPL4_SLOT_ABILITY_ID=BOPL4_SLOT_FEATURES-1,
    BOPL4_ENTITY_ABILITY_ID=BOPL4_ENTITY_FEATURES-1,
    /* Layout v4.1 (BOPL4_ABLATE_AIM): the down rays at -135 and -45 degrees
     * duplicate around rays 20 and 28 and carry the aim ray and the line of
     * sight instead (bopl_rays_v4.h). */
    BOPL4_AIM_RAY=BOPL4_RAYS_AROUND+0, BOPL4_LOS_RAY=BOPL4_RAYS_AROUND+3,
    /* Row memory of every observation buffer shared by the layouts. */
    BOPL_OBS_STRIDE=BOPL4_OBS_SIZE>BOPL_OBS_SIZE?BOPL4_OBS_SIZE:BOPL_OBS_SIZE
};
#ifdef __cplusplus
static_assert(BOPL4_OBS_SIZE==4812, "layout v4 size");
#else
_Static_assert(BOPL4_OBS_SIZE==4812, "layout v4 size");
#endif

/* Network shape defaults; the trainer flags and
 * the checkpoint sidecar override them, the evaluator reads the sidecar. */
enum { BOPL4_SET_DIM=128, BOPL4_SET_INNER=128, BOPL4_FLAT_DIM=256, BOPL4_FFN_DIM=512, BOPL4_CRITIC_DIM=512 };

#define BOPL4_VELOCITY_SCALE 100.0
#define BOPL4_POSITION_SCALE 64.0
#define BOPL4_GRAVITY 40.0
#define BOPL4_HORIZON 2.0
#define BOPL4_RAY_LENGTH 40.0
#define BOPL4_DASH_LENGTH 13.0
#define BOPL4_WATER_LAYER 4

#endif
