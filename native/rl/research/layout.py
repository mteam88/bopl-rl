"""Policy interface layout, mirrored from native/rl/puffer/bopl_layout.h (v2).

Every Python consumer of observation offsets or action head sizes imports these
instead of literal numbers. environment.py asserts them against the compiled
research library (research_layout) on import, so a drift fails loudly.
"""
ARCHITECTURE = "bopl-set-mingru-v2"

OBS_ENTITIES = 96
OBS_SLIMES = 12
ABILITY_COUNT = 39
ABILITY_FEATURES = 46            # 39 ids + overflow + timer, cooldown, ready, over, active, charge
PLAYER_FEATURES = 16 + 3 * ABILITY_FEATURES   # 154
ENTITY_FEATURES = 21
SLIME_FEATURES = 28
PREVIOUS_FEATURES = 12           # angle==0, cos, sin, 4 buttons, hold left/16, 4 hold runs/16
OBS_PREFIX = 10 + 6 * PLAYER_FEATURES         # 934
OBS_ENTITY_OFFSET = OBS_PREFIX
OBS_SLIME_OFFSET = OBS_ENTITY_OFFSET + OBS_ENTITIES * ENTITY_FEATURES
OBS_PREVIOUS_OFFSET = OBS_SLIME_OFFSET + OBS_SLIMES * SLIME_FEATURES
OBS_SIZE = OBS_PREVIOUS_OFFSET + PREVIOUS_FEATURES   # 3298
PLAYER0 = 10                     # first player block

ACT_AIM = 256
ACT_BUTTONS = 16
ACT_HOLD = 5
ACTION_SIZES = [ACT_AIM, ACT_BUTTONS, ACT_HOLD]
NUM_ATNS = len(ACTION_SIZES)
HOLD_LENGTHS = [1, 2, 4, 8, 16]
HOLD_MAX = 16
LOGITS = sum(ACTION_SIZES)       # 277
DECODER = LOGITS + 1             # + value = 278

# Set encoder parameter count (bopl_set_encoder.h): flat, entity, slime, fusion.
SET_DIM = 64
SET_FLAT = OBS_PREFIX + PREVIOUS_FEATURES     # 946


def set_encoder_weights(hidden):
    return hidden * SET_FLAT + SET_DIM * (ENTITY_FEATURES + SLIME_FEATURES) + hidden * (hidden + 2 * SET_DIM)


def policy_weights(hidden, num_layers, architecture):
    encoder = set_encoder_weights(hidden) if architecture == ARCHITECTURE else hidden * OBS_SIZE
    return encoder + DECODER * hidden + num_layers * 3 * hidden ** 2


LAYOUT_FIELDS = (OBS_SIZE, PLAYER_FEATURES, ABILITY_FEATURES, ENTITY_FEATURES, SLIME_FEATURES, PREVIOUS_FEATURES,
                 OBS_ENTITIES, OBS_SLIMES, ACT_AIM, ACT_BUTTONS, ACT_HOLD, HOLD_MAX)  # research_layout() order
