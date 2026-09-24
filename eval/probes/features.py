"""Candidate feature families computed from the raw observation (feature study A).

Relational per-row features are pure functions of what is already in the
observation row plus the observer's own velocity (which v3 carries only in the
own slime row, not in the self block): distance, bearing, relative velocity in
the self frame, radial (closing) and tangential speed, linear time to closest
approach and the miss distance at that time. Nothing here reads the sim.
Scales follow the observation: positions /64 u, velocities /256 u/s.
"""
import numpy as np
import layout
import schema

REL_PER_ROW = 9
REL_NAMES = ("dist", "inv_dist", "bearing_cos", "bearing_sin", "rel_vx", "rel_vy", "closing", "tangential", "miss")
TCA_MAX_TICKS = 60.0


def self_velocity(compact_obs):
    """Own slime velocity (obs scale /256) from the self-owned active slime row, else 0."""
    slimes = schema.block(compact_obs, "slimes")
    own = (slimes[:, :, 0] != 0) & (slimes[:, :, 1] != 0) & (slimes[:, :, 3] != 0)
    idx = np.argmax(own, axis=1)
    has = own.any(1)
    v = slimes[np.arange(len(slimes)), idx][:, 9:11]
    return np.where(has[:, None], v, 0.0).astype(np.float32)


def relational(dx, dy, rvx, rvy, valid):
    """dx, dy in /64 units; rvx, rvy in /256 units (entity minus self). Returns (n, R, REL_PER_ROW)."""
    px, py = dx * 64.0, dy * 64.0            # world units
    vx, vy = rvx * 256.0, rvy * 256.0        # u/s
    dist = np.sqrt(px * px + py * py) + 1e-6
    cos, sin = px / dist, py / dist
    closing = -(px * vx + py * vy) / dist     # u/s toward me
    tangential = (px * vy - py * vx) / dist
    speed2 = vx * vx + vy * vy + 1e-6
    tca = np.clip(-(px * vx + py * vy) / speed2, 0.0, TCA_MAX_TICKS / 60.0)  # seconds
    miss = np.sqrt((px + vx * tca) ** 2 + (py + vy * tca) ** 2)
    out = np.stack([dist / 64.0, np.clip(1.0 / dist, 0, 4.0) / 4.0, cos, sin, vx / 256.0, vy / 256.0,
                    closing / 256.0, tangential / 256.0, np.clip(miss, 0, 32.0) / 32.0], -1).astype(np.float32)
    return np.where(valid[..., None], out, 0.0)


def entity_relational(compact_obs):
    """(n, ENTITY_CAP * REL_PER_ROW) for the entity rows."""
    ent = schema.block(compact_obs, "entities")
    sv = self_velocity(compact_obs)
    valid = ent[:, :, 0] != 0
    return relational(ent[:, :, 9], ent[:, :, 10], ent[:, :, 11] - sv[:, None, 0], ent[:, :, 12] - sv[:, None, 1], valid).reshape(len(ent), -1)


def opponent_relational(compact_obs):
    """(n, REL_PER_ROW): the nearest non-self visible player row against the nearest non-self active slime's velocity."""
    players = schema.block(compact_obs, "players")
    slimes = schema.block(compact_obs, "slimes")
    other = (players[:, :, 0] != 0) & (players[:, :, 1] == 0) & (players[:, :, 16] != 0)
    pi = np.argmax(other, axis=1)
    p = players[np.arange(len(players)), pi]
    foe_slime = (slimes[:, :, 0] != 0) & (slimes[:, :, 1] == 0) & (slimes[:, :, 3] != 0)
    si = np.argmax(foe_slime, axis=1)
    s = slimes[np.arange(len(slimes)), si]
    sv = self_velocity(compact_obs)
    fv = np.where(foe_slime.any(1)[:, None], s[:, 9:11], 0.0)
    valid = other.any(1)
    return relational(p[:, 9], p[:, 10], fv[:, 0] - sv[:, 0], fv[:, 1] - sv[:, 1], valid).astype(np.float32)


def relational_features(compact_obs):
    """Feature family A: entity rows (32 x 9) + opponent (9) + own velocity (2) = 299 floats."""
    compact_obs = np.asarray(compact_obs, dtype=np.float32)
    return np.concatenate([entity_relational(compact_obs), opponent_relational(compact_obs), self_velocity(compact_obs)], 1)


REL_SIZE = schema.ENTITY_CAP * REL_PER_ROW + REL_PER_ROW + 2


def hostile_projectile_rows(compact_obs):
    """Boolean (n, ENTITY_CAP): entity rows that are enemy-owned projectiles."""
    ent = schema.block(compact_obs, "entities")
    return (ent[:, :, 0] != 0) & (ent[:, :, 26] != 0) & (ent[:, :, 8] != 0)


def per_row_features(compact_obs, row_index):
    """The raw 34 entity-row features and the 9 relational ones of one row per sample: (n, 43)."""
    compact_obs = np.asarray(compact_obs, dtype=np.float32)
    ent = schema.block(compact_obs, "entities")
    n = len(ent)
    rows = ent[np.arange(n), row_index]
    rel = entity_relational(compact_obs).reshape(n, schema.ENTITY_CAP, REL_PER_ROW)[np.arange(n), row_index]
    return rows, rel
