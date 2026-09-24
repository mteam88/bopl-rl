"""The research Environment plus the probe binding's privileged readers.

Import this before `environment`: it points BOPL_RESEARCH_LIBRARY at the probe
library, which contains the whole research evaluator (eval_binding.c) plus
probe_* exports, so `Environment` and the labels share one C environment.
"""
import ctypes as C
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIBRARY = ROOT / "native/artifacts/rl/probes/libbopl_probe.so"
if not LIBRARY.exists():
    raise RuntimeError(f"{LIBRARY} missing; run eval/probes/build.py")
os.environ["BOPL_RESEARCH_LIBRARY"] = str(LIBRARY)
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
from environment import Environment, LIB, State, OBS_SIZE  # noqa: E402
import layout  # noqa: E402

LIB.probe_observed_state.argtypes = [C.c_void_p]; LIB.probe_observed_state.restype = C.POINTER(State)
LIB.probe_pack.argtypes = [C.c_void_p, C.POINTER(State), C.POINTER(C.c_float)]
LIB.probe_entity_rows.argtypes = [C.POINTER(State), C.c_int, C.POINTER(C.c_uint32), C.c_int]; LIB.probe_entity_rows.restype = C.c_int
LIB.probe_idle_future.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_float)]; LIB.probe_idle_future.restype = C.c_int
LIB.probe_geometry.argtypes = [C.POINTER(State), C.c_int, C.POINTER(C.c_float)]; LIB.probe_geometry.restype = C.c_int
LIB.probe_rays.argtypes = [C.POINTER(State), C.c_int, C.c_int, C.c_int, C.c_int, C.c_float, C.POINTER(C.c_float)]; LIB.probe_rays.restype = C.c_int
LIB.probe_rays_bench.argtypes = [C.POINTER(State), C.c_int, C.c_int, C.c_int, C.c_int, C.c_float, C.c_int]; LIB.probe_rays_bench.restype = C.c_double
LIB.probe_pack_bench.argtypes = [C.c_void_p, C.c_int]; LIB.probe_pack_bench.restype = C.c_double
LIB.probe_idle_fields.restype = C.c_int; LIB.probe_geometry_fields.restype = C.c_int; LIB.probe_ray_fields.restype = C.c_int
IDLE_FIELDS = LIB.probe_idle_fields()
GEOMETRY_FIELDS = LIB.probe_geometry_fields()
RAY_FIELDS = LIB.probe_ray_fields()
RAY_TYPES = ("wall", "kill", "hostile_projectile", "opponent", "own", "none")
IDLE_NAMES = ("death_tick", "cause", "killer_id", "killer_x", "killer_y", "killer_vx", "killer_vy", "killer_known",
              "x10", "y10", "x30", "y30", "own_arrow_in_flight")
GEOMETRY_NAMES = ("view_ok", "grounded", "drop", "support", "edge_distance", "lands_dry", "lands_dry_any_steer",
                  "terrain_within_dash", "terrain_min_ray", "vx", "vy", "foe_vx", "foe_vy", "foe_grounded",
                  "foe_dash_ready", "foe_bow_ready", "foe_platform_ready", "foe_dash_cd", "foe_bow_cd", "foe_platform_cd")
CAUSES = ("alive", "environment", "squished", "opponent", "self", "other")
FIX = 1 / 4294967296.0


def fptr(array):
    return array.ctypes.data_as(C.POINTER(C.c_float))


class ProbeEnvironment(Environment):
    """Environment with privileged readers of the live and the observed state."""

    def observed_state(self):
        """The state the published observation rows were encoded from (nine ticks old)."""
        return LIB.probe_observed_state(self.handle).contents

    def observed_state_pointer(self):
        return LIB.probe_observed_state(self.handle)

    def live_state_pointer(self):
        return LIB.research_state(self.handle)

    def pack(self, state_pointer):
        """Undelayed encode of a state for both players: (2, OBS_SIZE), own-action tail zero."""
        out = np.zeros((2, OBS_SIZE), dtype=np.float32)
        LIB.probe_pack(self.handle, state_pointer, fptr(out))
        return out

    def entity_rows(self, state_pointer, player):
        ids = np.zeros(layout.OBS_ENTITIES, dtype=np.uint32)
        count = LIB.probe_entity_rows(state_pointer, player, ids.ctypes.data_as(C.POINTER(C.c_uint32)), layout.OBS_ENTITIES)
        return ids, count

    def idle_future(self, horizon=60):
        """Fork the live world, hold no input for both players for `horizon` ticks: (2, IDLE_FIELDS)."""
        out = np.zeros((2, IDLE_FIELDS), dtype=np.float32)
        players = LIB.probe_idle_future(self.handle, horizon, fptr(out))
        if players < 0:
            raise RuntimeError(f"probe_idle_future failed ({players}); the game must be the forkable Rust sim")
        return out

    def geometry(self, state_pointer, player):
        out = np.zeros(GEOMETRY_FIELDS, dtype=np.float32)
        LIB.probe_geometry(state_pointer, player, fptr(out))
        return out

    def rays(self, state_pointer, player, n=16, extra_down=4, hits=1, range_=40.0):
        out = np.zeros(((n + extra_down) * hits, RAY_FIELDS), dtype=np.float32)
        count = LIB.probe_rays(state_pointer, player, n, extra_down, hits, range_, fptr(out))
        if count < 0:
            raise ValueError("probe_rays rejected its arguments")
        return out

    def rays_bench(self, state_pointer, player, n, extra_down, hits, range_=40.0, iterations=2000):
        return LIB.probe_rays_bench(state_pointer, player, n, extra_down, hits, range_, iterations)

    def pack_bench(self, iterations=2000):
        return LIB.probe_pack_bench(self.handle, iterations)


def player_xy(state, player):
    p = state.players[player]
    return p.x * FIX, p.y * FIX


def slime_of(state, owner_id):
    for i in range(state.header.slime_count):
        if state.slimes[i].owner == owner_id:
            return state.slimes[i]
    return None
