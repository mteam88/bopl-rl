"""Observation layout v3, mirrored from native/rl/puffer/bopl_layout.h."""
ARCHITECTURE = "bopl-set-mingru-v3"

ABILITY_COUNT = 39
EMBEDDING_IDS = ABILITY_COUNT + 1
OBS_PREFIX = 66
SELF_FEATURES = 20
OBS_PLAYERS = 6
PLAYER_FEATURES = 20
OBS_SLOTS = 24
SLOT_FEATURES = 14
ABILITY_FEATURES = SLOT_FEATURES
OBS_ENTITIES = 64
ENTITY_FEATURES = 34
OBS_SLIMES = 32
SLIME_FEATURES = 30
PREVIOUS_FEATURES = 12

OBS_SELF_OFFSET = OBS_PREFIX
OBS_PLAYER_OFFSET = OBS_SELF_OFFSET + SELF_FEATURES
OBS_SLOT_OFFSET = OBS_PLAYER_OFFSET + OBS_PLAYERS * PLAYER_FEATURES
OBS_ENTITY_OFFSET = OBS_SLOT_OFFSET + OBS_SLOTS * SLOT_FEATURES
OBS_SLIME_OFFSET = OBS_ENTITY_OFFSET + OBS_ENTITIES * ENTITY_FEATURES
OBS_PREVIOUS_OFFSET = OBS_SLIME_OFFSET + OBS_SLIMES * SLIME_FEATURES
OBS_SIZE = OBS_PREVIOUS_OFFSET + PREVIOUS_FEATURES
PLAYER0 = OBS_PLAYER_OFFSET
SLOT_ABILITY_ID = SLOT_FEATURES - 1
ENTITY_ABILITY_ID = ENTITY_FEATURES - 1
COUNT_SCALE = 64
TIMER_TICKS = 600

ACT_AIM = 256
ACT_BUTTONS = 16
ACT_HOLD = 5
ACTION_SIZES = [ACT_AIM, ACT_BUTTONS, ACT_HOLD]
NUM_ATNS = len(ACTION_SIZES)
HOLD_LENGTHS = [1, 2, 4, 8, 16]
HOLD_MAX = 16
LOGITS = sum(ACTION_SIZES)
DECODER = LOGITS + 1
AIM_MAX_HARMONICS = 64


def aim_block_width(harmonics):
    return 1 + 2 * harmonics if harmonics > 0 else ACT_AIM


def decoder_buttons_offset(harmonics, autoregressive):
    return (ACT_BUTTONS if autoregressive else 1) * aim_block_width(harmonics)


def decoder_logits(harmonics=0, autoregressive=0):
    return decoder_buttons_offset(harmonics, autoregressive) + ACT_BUTTONS + ACT_HOLD


def aim_block_offset(harmonics, autoregressive, buttons):
    return (buttons if autoregressive else 0) * aim_block_width(harmonics)


SET_DIM = 64
SET_FLAT = OBS_PREFIX + SELF_FEATURES + PREVIOUS_FEATURES
SET_PLAYER_PROJECTION = PLAYER_FEATURES
SET_SLOT_PROJECTION = SLOT_FEATURES - 1
SET_ENTITY_PROJECTION = ENTITY_FEATURES - 1
SET_SLIME_PROJECTION = SLIME_FEATURES


def set_encoder_weights(hidden):
    projections = SET_DIM * (SET_PLAYER_PROJECTION + SET_SLOT_PROJECTION
                             + SET_ENTITY_PROJECTION + SET_SLIME_PROJECTION)
    embedding = EMBEDDING_IDS * SET_DIM
    fusion = hidden * (hidden + 4 * SET_DIM)
    return hidden * SET_FLAT + projections + embedding + fusion


def policy_weights(hidden, num_layers, architecture, aim_fourier=0, aim_ar=0):
    encoder = set_encoder_weights(hidden) if architecture == ARCHITECTURE else hidden * OBS_SIZE
    return encoder + (decoder_logits(aim_fourier, aim_ar) + 1) * hidden + num_layers * 3 * hidden ** 2


LAYOUT_FIELDS = (OBS_SIZE, OBS_PREFIX, SELF_FEATURES,
                 OBS_PLAYERS, PLAYER_FEATURES, OBS_SLOTS, SLOT_FEATURES,
                 OBS_ENTITIES, ENTITY_FEATURES, OBS_SLIMES, SLIME_FEATURES,
                 PREVIOUS_FEATURES, OBS_SELF_OFFSET, OBS_PLAYER_OFFSET, OBS_SLOT_OFFSET,
                 OBS_ENTITY_OFFSET, OBS_SLIME_OFFSET, OBS_PREVIOUS_OFFSET,
                 SLOT_ABILITY_ID, ENTITY_ABILITY_ID, EMBEDDING_IDS,
                 ACT_AIM, ACT_BUTTONS, ACT_HOLD, HOLD_MAX)
