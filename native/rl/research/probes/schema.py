"""Probe dataset schema: compact observation storage and column names.

A v3 observation is 3,690 floats, most of them zero rows of the set blocks
(64 entity, 32 slime, 24 slot, 6 player rows). In 1v1 Dash/Bow/Platform games
the populated rows are far fewer, so the dataset stores each block truncated
to the caps below (rows are nearest-first, so truncation drops the farthest)
and `expand` rebuilds the exact 3,690-float row; the set encoder skips zero
rows, so the two are equivalent for every probe and for the encoder itself.
"""
import numpy as np
import layout

PLAYER_CAP, SLOT_CAP, ENTITY_CAP, SLIME_CAP = 2, 8, 32, 4
BLOCKS = (  # (name, full offset, rows kept, rows in v3, width)
    ("prefix", 0, 1, 1, layout.OBS_PREFIX),
    ("self", layout.OBS_SELF_OFFSET, 1, 1, layout.SELF_FEATURES),
    ("players", layout.OBS_PLAYER_OFFSET, PLAYER_CAP, layout.OBS_PLAYERS, layout.PLAYER_FEATURES),
    ("slots", layout.OBS_SLOT_OFFSET, SLOT_CAP, layout.OBS_SLOTS, layout.SLOT_FEATURES),
    ("entities", layout.OBS_ENTITY_OFFSET, ENTITY_CAP, layout.OBS_ENTITIES, layout.ENTITY_FEATURES),
    ("slimes", layout.OBS_SLIME_OFFSET, SLIME_CAP, layout.OBS_SLIMES, layout.SLIME_FEATURES),
    ("previous", layout.OBS_PREVIOUS_OFFSET, 1, 1, layout.PREVIOUS_FEATURES),
)
COMPACT_OFFSETS = {}
_offset = 0
for _name, _full, _kept, _rows, _width in BLOCKS:
    COMPACT_OFFSETS[_name] = _offset
    _offset += _kept * _width
COMPACT_SIZE = _offset  # 1458


def compact(obs):
    """(n, 3690) -> (n, COMPACT_SIZE), plus the number of populated rows dropped per sample."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1, layout.OBS_SIZE)
    out = np.zeros((obs.shape[0], COMPACT_SIZE), dtype=np.float32)
    dropped = np.zeros(obs.shape[0], dtype=np.int32)
    for name, full, kept, rows, width in BLOCKS:
        block = obs[:, full:full + rows * width].reshape(-1, rows, width)
        out[:, COMPACT_OFFSETS[name]:COMPACT_OFFSETS[name] + kept * width] = block[:, :kept].reshape(-1, kept * width)
        if kept < rows:
            dropped += (block[:, kept:, 0] != 0).sum(1)
    return out, dropped


def expand(compact_obs):
    """(n, COMPACT_SIZE) -> (n, 3690) float32."""
    compact_obs = np.asarray(compact_obs, dtype=np.float32).reshape(-1, COMPACT_SIZE)
    out = np.zeros((compact_obs.shape[0], layout.OBS_SIZE), dtype=np.float32)
    for name, full, kept, rows, width in BLOCKS:
        out[:, full:full + kept * width] = compact_obs[:, COMPACT_OFFSETS[name]:COMPACT_OFFSETS[name] + kept * width]
    return out


def block(compact_obs, name):
    """One set block of compact rows as (n, rows_kept, width)."""
    _, _, kept, _, width = next(b for b in BLOCKS if b[0] == name)
    start = COMPACT_OFFSETS[name]
    return np.asarray(compact_obs)[:, start:start + kept * width].reshape(-1, kept, width)


# Per-row metadata columns (int32) and label columns (float32), by name.
META = ("game", "decision", "player", "controller", "opponent_controller", "map", "seed", "tick",
        "aim", "buttons", "hold", "latched", "killer_row", "killer_row_now", "entity_rows", "entity_rows_now",
        "dropped", "dropped_now", "terminal", "split")
# Controllers (who decided this row's action): see generate.CONTROLLERS.
LABELS = (
    # from the idle fork (both players hold no input for 60 ticks)
    "idle_death_tick", "idle_cause", "hit_15", "hit_30", "hit_60", "hit_inflight_30", "hit_inflight_60", "killer_known", "self_hit_60", "fall_30", "fall_60", "squish_60",
    "any_death_30", "any_death_60", "own_arrow_in_flight", "my_shot_hits_60",
    "idle_x10", "idle_y10", "idle_x30", "idle_y30", "idle_opp_x10", "idle_opp_y10", "idle_opp_x30", "idle_opp_y30",
    # geometry of the live state (probe_geometry)
    "grounded", "drop", "support", "edge_distance", "lands_dry", "lands_dry_any_steer", "terrain_within_dash",
    "terrain_min_ray", "vx", "vy", "foe_vx", "foe_vy", "foe_grounded", "foe_dash_ready", "foe_bow_ready",
    "foe_platform_ready", "foe_dash_cd", "foe_bow_cd", "foe_platform_cd",
    # live kinematics
    "x", "y", "opp_x", "opp_y", "opp_alive",
    # the realised future along the rollout (NaN past the end of the episode)
    "real_x12", "real_y12", "real_opp_x12", "real_opp_y12", "real_opp_x32", "real_opp_y32",
    "real_death_60", "real_death_tick", "real_cause",
    "win", "loss", "timeout", "ticks_to_end", "reward", "return_disc", "value_online",
)
RAY_SETS = {  # name: (n, extra_down, hits_per_ray)
    "rays16": (16, 4, 1),
    "rays32": (32, 4, 1),
    "rays32x3": (32, 4, 3),
}
RAY_FIELDS = 8
GAMMA = 0.995
IDLE_HORIZON = 60
