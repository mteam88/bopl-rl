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


# ---- Observation layout v4 (native/rl/puffer/bopl_layout_v4.h, POLICY_V4_PLAN.md) ----
V4_ARCHITECTURE = "bopl-v4-mingru"
V4_LAYOUT_VERSION = 4
V4_GLOBAL = 66
V4_BODY = 24
V4_SELF = PLAYER_FEATURES + V4_BODY            # 44
V4_RELATION = 12
V4_OPPONENT = V4_SELF + V4_RELATION            # 56
V4_HISTORY, V4_HISTORY_FEATURES = 2, 7   # the two decisions before the last, behind the v3 tail
V4_PREVIOUS = PREVIOUS_FEATURES + V4_HISTORY * V4_HISTORY_FEATURES   # 26
V4_TERRAIN = 12
V4_RAYS_AROUND, V4_RAYS_DOWN = 32, 4
V4_RAYS, V4_RAY_FEATURES = V4_RAYS_AROUND + V4_RAYS_DOWN, 8
V4_FLAT = V4_GLOBAL + V4_SELF + V4_OPPONENT + V4_PREVIOUS + V4_TERRAIN + V4_RAYS * V4_RAY_FEATURES   # 492
V4_PLAYERS, V4_PLAYER_FEATURES = 6, PLAYER_FEATURES + V4_RELATION   # 32
V4_SLOTS, V4_SLOT_FEATURES = 24, 18
V4_ENTITIES, V4_ENTITY_FEATURES = 48, 48
V4_SLIMES, V4_SLIME_FEATURES = 32, 42
V4_PRIVILEGED = 48
V4_SELF_OFFSET = V4_GLOBAL
V4_OPPONENT_OFFSET = V4_SELF_OFFSET + V4_SELF
V4_PREVIOUS_OFFSET = V4_OPPONENT_OFFSET + V4_OPPONENT
V4_TERRAIN_OFFSET = V4_PREVIOUS_OFFSET + V4_PREVIOUS
V4_RAY_OFFSET = V4_TERRAIN_OFFSET + V4_TERRAIN
V4_PLAYER_OFFSET = V4_FLAT
V4_SLOT_OFFSET = V4_PLAYER_OFFSET + V4_PLAYERS * V4_PLAYER_FEATURES
V4_ENTITY_OFFSET = V4_SLOT_OFFSET + V4_SLOTS * V4_SLOT_FEATURES
V4_SLIME_OFFSET = V4_ENTITY_OFFSET + V4_ENTITIES * V4_ENTITY_FEATURES
V4_PRIVILEGED_OFFSET = V4_SLIME_OFFSET + V4_SLIMES * V4_SLIME_FEATURES
V4_OBS_SIZE = V4_PRIVILEGED_OFFSET + V4_PRIVILEGED   # 4812
V4_SLOT_ABILITY_ID = V4_SLOT_FEATURES - 1
V4_ENTITY_ABILITY_ID = V4_ENTITY_FEATURES - 1
OBS_STRIDE = max(OBS_SIZE, V4_OBS_SIZE)
assert V4_OBS_SIZE == 4812
V4_LAYOUT_FIELDS = (V4_OBS_SIZE, V4_GLOBAL, V4_SELF, V4_OPPONENT, V4_PREVIOUS, V4_TERRAIN, V4_RAYS, V4_RAY_FEATURES,
                    V4_FLAT, V4_PLAYERS, V4_PLAYER_FEATURES, V4_SLOTS, V4_SLOT_FEATURES,
                    V4_ENTITIES, V4_ENTITY_FEATURES, V4_SLIMES, V4_SLIME_FEATURES, V4_PRIVILEGED,
                    V4_SELF_OFFSET, V4_OPPONENT_OFFSET, V4_PREVIOUS_OFFSET, V4_TERRAIN_OFFSET, V4_RAY_OFFSET,
                    V4_PLAYER_OFFSET, V4_SLOT_OFFSET, V4_ENTITY_OFFSET, V4_SLIME_OFFSET,
                    V4_PRIVILEGED_OFFSET, V4_SLOT_ABILITY_ID, V4_ENTITY_ABILITY_ID, OBS_STRIDE)
# Sets in the order the encoder pools them: (name, row offset, rows, width, id column or None).
V4_SETS = (("player", V4_PLAYER_OFFSET, V4_PLAYERS, V4_PLAYER_FEATURES, None),
           ("slot", V4_SLOT_OFFSET, V4_SLOTS, V4_SLOT_FEATURES, V4_SLOT_ABILITY_ID),
           ("entity", V4_ENTITY_OFFSET, V4_ENTITIES, V4_ENTITY_FEATURES, V4_ENTITY_ABILITY_ID),
           ("slime", V4_SLIME_OFFSET, V4_SLIMES, V4_SLIME_FEATURES, None))
V4_DEFAULTS = dict(set_dim=128, set_inner=128, flat_dim=256, ffn_dim=512, critic_dim=512,
                   critic_privileged=1, critic_detach=0, mean_pool=1,
                   ablate=0)   # writer ablation bitmask (bopl_obs_v4.h): 1 relational, 2 terrain + rays, 4 rays16


def obs_size_of(layout_version):
    return V4_OBS_SIZE if layout_version == V4_LAYOUT_VERSION else OBS_SIZE


def layout_of(architecture):
    """Observation layout version a checkpoint architecture reads."""
    return V4_LAYOUT_VERSION if architecture == V4_ARCHITECTURE else 3


def aligned(count):
    """v4 tensors are stored back to back in the trainer's flat weight dump
    (bindings.cu save_weights: total_elems floats, no padding); the device
    allocator aligns every tensor to 16 bytes, so bopl_v4.cu requires every
    tensor's element count to be a multiple of 4 and the two layouts agree."""
    if count % 4:
        raise ValueError(f"v4 tensor of {count} floats is not 16-byte aligned")
    return count


def policy_v4_tensors(hidden, num_layers, aim_fourier=4, aim_ar=0, set_dim=128, set_inner=128,
                      flat_dim=256, ffn_dim=512, critic_dim=512, critic_privileged=1, **_):
    """(name, shape) of every v4 tensor in flat-buffer order (bopl_v4.h / bopl_v4_encoder.cu).

    Row MLPs: W1 (inner, features_without_id) then W2 (set_dim, inner) per set; the
    embedding (ids, inner) is added before the first ReLU. Pooled width per set is
    2*set_dim (max || mean). ffn_dim 0 drops the FFN block; critic_dim 0 the critic.
    """
    tensors = []
    for name, _, _, width, id_column in V4_SETS:
        features = width - (1 if id_column is not None else 0)
        tensors += [(f"{name}_w1", (set_inner, features)), (f"{name}_w2", (set_dim, set_inner))]
    tensors += [("embedding", (EMBEDDING_IDS, set_inner)),
                ("flat", (flat_dim, V4_FLAT)),
                ("fusion", (hidden, flat_dim + 4 * 2 * set_dim))]
    for layer in range(num_layers):
        tensors.append((f"mingru{layer}", (3 * hidden, hidden)))
    if ffn_dim:
        tensors += [("ffn_in", (ffn_dim, hidden)), ("ffn_out", (hidden, ffn_dim))]
    # critic_dim 0: v3-style shared value, one extra actor row (the v4-critic-shared ablation)
    tensors.append(("actor", (decoder_logits(aim_fourier, aim_ar) + (0 if critic_dim else 1), hidden)))
    if critic_dim:
        tensors += [("critic_in", (critic_dim, hidden + (V4_PRIVILEGED if critic_privileged else 0))),
                    ("critic_b1", (critic_dim, critic_dim)), ("critic_b2", (critic_dim, critic_dim)),
                    ("critic_out", (1, critic_dim))]
    return tensors


def policy_v4_weights(hidden, num_layers, **shape):
    """Flat float count of a v4 checkpoint (every tensor 8-aligned, like the trainer's allocator)."""
    total = 0
    for _, (rows, cols) in policy_v4_tensors(hidden, num_layers, **shape):
        total += aligned(rows * cols)
    return total


LAYOUT_FIELDS = (OBS_SIZE, OBS_PREFIX, SELF_FEATURES,
                 OBS_PLAYERS, PLAYER_FEATURES, OBS_SLOTS, SLOT_FEATURES,
                 OBS_ENTITIES, ENTITY_FEATURES, OBS_SLIMES, SLIME_FEATURES,
                 PREVIOUS_FEATURES, OBS_SELF_OFFSET, OBS_PLAYER_OFFSET, OBS_SLOT_OFFSET,
                 OBS_ENTITY_OFFSET, OBS_SLIME_OFFSET, OBS_PREVIOUS_OFFSET,
                 SLOT_ABILITY_ID, ENTITY_ABILITY_ID, EMBEDDING_IDS,
                 ACT_AIM, ACT_BUTTONS, ACT_HOLD, HOLD_MAX)
