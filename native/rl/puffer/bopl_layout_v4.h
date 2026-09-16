/* Observation layout v4 (POLICY_V4_PLAN.md section 2), next to layout v3
 * (bopl_layout.h). Both are written from the same wire state (bopl_rl.h,
 * version 3); which one a player's row holds is selected per player
 * (BoplPuffer.layout), so a v3 checkpoint and a v4 checkpoint can share an
 * evaluation game. Row memory is BOPL_OBS_STRIDE floats, the larger layout.
 *
 * Flat block (read by one Linear): global | self | opponent | previous | rays.
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
    BOPL4_PREVIOUS=BOPL_PREVIOUS_FEATURES, /* 12, the v3 own-action tail */
    BOPL4_RAYS=0, BOPL4_RAY_FEATURES=4,    /* reserved: distance, hit class, hit speed along the ray, hit known */
    BOPL4_FLAT=BOPL4_GLOBAL+BOPL4_SELF+BOPL4_OPPONENT+BOPL4_PREVIOUS+BOPL4_RAYS*BOPL4_RAY_FEATURES, /* 178 */
    BOPL4_PLAYERS=6, BOPL4_PLAYER_FEATURES=BOPL_PLAYER_FEATURES+BOPL4_RELATION,    /* 32 */
    BOPL4_SLOTS=24, BOPL4_SLOT_FEATURES=18,
    BOPL4_ENTITIES=48, BOPL4_ENTITY_FEATURES=48,
    BOPL4_SLIMES=32, BOPL4_SLIME_FEATURES=42,
    BOPL4_PRIVILEGED=48,
    BOPL4_OBS_GLOBAL_OFFSET=0,
    BOPL4_OBS_SELF_OFFSET=BOPL4_OBS_GLOBAL_OFFSET+BOPL4_GLOBAL,
    BOPL4_OBS_OPPONENT_OFFSET=BOPL4_OBS_SELF_OFFSET+BOPL4_SELF,
    BOPL4_OBS_PREVIOUS_OFFSET=BOPL4_OBS_OPPONENT_OFFSET+BOPL4_OPPONENT,
    BOPL4_OBS_RAY_OFFSET=BOPL4_OBS_PREVIOUS_OFFSET+BOPL4_PREVIOUS,
    BOPL4_OBS_PLAYER_OFFSET=BOPL4_FLAT,
    BOPL4_OBS_SLOT_OFFSET=BOPL4_OBS_PLAYER_OFFSET+BOPL4_PLAYERS*BOPL4_PLAYER_FEATURES,
    BOPL4_OBS_ENTITY_OFFSET=BOPL4_OBS_SLOT_OFFSET+BOPL4_SLOTS*BOPL4_SLOT_FEATURES,
    BOPL4_OBS_SLIME_OFFSET=BOPL4_OBS_ENTITY_OFFSET+BOPL4_ENTITIES*BOPL4_ENTITY_FEATURES,
    BOPL4_OBS_PRIVILEGED_OFFSET=BOPL4_OBS_SLIME_OFFSET+BOPL4_SLIMES*BOPL4_SLIME_FEATURES,
    BOPL4_OBS_SIZE=BOPL4_OBS_PRIVILEGED_OFFSET+BOPL4_PRIVILEGED,                /* 4498 */
    BOPL4_SLOT_ABILITY_ID=BOPL4_SLOT_FEATURES-1,
    BOPL4_ENTITY_ABILITY_ID=BOPL4_ENTITY_FEATURES-1,
    /* Row memory of every observation buffer shared by the layouts. */
    BOPL_OBS_STRIDE=BOPL4_OBS_SIZE>BOPL_OBS_SIZE?BOPL4_OBS_SIZE:BOPL_OBS_SIZE
};
#ifdef __cplusplus
static_assert(BOPL4_OBS_SIZE==4498, "layout v4 size");
#else
_Static_assert(BOPL4_OBS_SIZE==4498, "layout v4 size");
#endif

/* Network shape defaults (POLICY_V4_PLAN.md section 3); the trainer flags and
 * the checkpoint sidecar override them, the evaluator reads the sidecar. */
enum { BOPL4_SET_DIM=128, BOPL4_SET_INNER=128, BOPL4_FLAT_DIM=256, BOPL4_FFN_DIM=512, BOPL4_CRITIC_DIM=512 };

#define BOPL4_VELOCITY_SCALE 100.0
#define BOPL4_POSITION_SCALE 64.0
#define BOPL4_GRAVITY 40.0
#define BOPL4_HORIZON 2.0

#endif
