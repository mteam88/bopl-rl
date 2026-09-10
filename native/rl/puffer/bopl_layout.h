/* Policy interface layout shared by the environment (bopl_puffer.h), the
 * CUDA set encoder (bopl_set_encoder.cu), the CPU evaluation encoder
 * (bopl_set_encoder.h / policy_eval.c) and, by value, the Python side
 * (research/layout.py, checked against research_layout()). Change it in one
 * place; the checkpoint architecture string below must change with it.
 *
 * Layout v2 (2026-09-09, "hold head"):
 *   - per-slot ability block grew 44 -> 46: [44] ability active this decision,
 *     [45] bow charge level / 4 (0 for other abilities);
 *   - the own-action tail grew 7 -> 12: [7] latched decisions remaining / 16,
 *     [8..11] consecutive decisions each button has been held / 16;
 *   - a third action head chooses how many decisions the button state is
 *     latched for: {1, 2, 4, 8, 16}. Aim is never latched. */
#ifndef BOPL_LAYOUT_H
#define BOPL_LAYOUT_H

#define BOPL_ARCHITECTURE "bopl-set-mingru-v2"

enum {
    BOPL_OBS_ENTITIES=96, BOPL_OBS_SLIMES=12, BOPL_ABILITY_COUNT=39,
    BOPL_ABILITY_FEATURES=46,                       /* 39 ids + overflow + timer, cooldown, ready, over, active, charge */
    BOPL_PLAYER_FEATURES=16+3*BOPL_ABILITY_FEATURES, /* 154 */
    BOPL_ENTITY_FEATURES=21, BOPL_SLIME_FEATURES=28,
    BOPL_PREVIOUS_FEATURES=12,                      /* own last decision + latch state, never delayed */
    BOPL_OBS_PREFIX=10+6*BOPL_PLAYER_FEATURES,      /* 934 */
    BOPL_OBS_ENTITY_OFFSET=BOPL_OBS_PREFIX,
    BOPL_OBS_SLIME_OFFSET=BOPL_OBS_ENTITY_OFFSET+BOPL_OBS_ENTITIES*BOPL_ENTITY_FEATURES,
    BOPL_OBS_PREVIOUS_OFFSET=BOPL_OBS_SLIME_OFFSET+BOPL_OBS_SLIMES*BOPL_SLIME_FEATURES,
    BOPL_OBS_SIZE=BOPL_OBS_PREVIOUS_OFFSET+BOPL_PREVIOUS_FEATURES /* 3298 */
};
/* Action heads: aim angle (0 neutral, 1..255), button bits (jump, dash,
 * arrow, platform), hold length index. */
enum { BOPL_ACT_AIM=256, BOPL_ACT_BUTTONS=16, BOPL_ACT_HOLD=5, BOPL_NUM_ATNS=3,
       BOPL_HOLD_MAX=16, BOPL_LOGITS=BOPL_ACT_AIM+BOPL_ACT_BUTTONS+BOPL_ACT_HOLD,
       BOPL_DECODER=BOPL_LOGITS+1 /* + value */ };
static const int BOPL_HOLD_LENGTHS[BOPL_ACT_HOLD]={1, 2, 4, 8, 16};
/* Button bit positions in the button head's index. The arrow may not be held
 * across a free decision (bopl_write_mask): pressing it is charging, and the
 * charge fires when the latch expires. */
enum { BOPL_BUTTON_JUMP=0, BOPL_BUTTON_DASH=1, BOPL_BUTTON_ARROW=2, BOPL_BUTTON_PLATFORM=3 };
/* Slime ability_flags bits written by RlObservation.cs: 1 present, 2 ready,
 * 4 over cooldown, 8 active (in use this tick), 0x70 bow charge level 0..4
 * (BowTransform.loadingFrame while active), 0x80 bow already fired. */
enum { BOPL_ABILITY_ACTIVE=8, BOPL_ABILITY_CHARGE_SHIFT=4, BOPL_ABILITY_CHARGE_MASK=7, BOPL_ABILITY_FIRED=0x80 };

#endif
