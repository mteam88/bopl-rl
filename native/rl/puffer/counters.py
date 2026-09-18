"""Cumulative learner counters of a native vector, in the shape research/native_metrics.py consumes.

The C environment keeps process-lifetime counters over the decisions it actually
played (bopl_vec_counters in binding.c): warmup never reaches the environment
step, unfinished episodes are included, frozen-opponent rows never count as
learner work. Read them between rollouts only.
"""
import ctypes

FIELDS = 16
MAPS = 64   # BOPL_COUNTER_MAPS
SIZE = 5 + 4 + FIELDS + MAPS

def bind(lib):
    lib.bopl_vec_counters.restype = ctypes.c_int
    lib.bopl_vec_counters.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.bopl_counter_size.restype = ctypes.c_int
    if lib.bopl_counter_size() != SIZE: raise RuntimeError('Native counter layout changed')
    return lib

def snapshot(lib, vec):
    out = (ctypes.c_double * SIZE)()
    if lib.bopl_vec_counters(ctypes.c_void_p(vec.gpu_obs_ptr), out, SIZE) != SIZE: raise RuntimeError('Unknown native vector')
    v = list(out)
    maps = v[9+FIELDS:]
    return dict(learner_steps=int(v[0]), game_steps=int(v[1]), matches=int(v[2]), opponent_game_decisions=int(v[3]),
                kills=int(v[4]), buttons=[int(x) for x in v[5:9]], telemetry=v[9:9+FIELDS],
                map_matches={str(m): int(n) for m, n in enumerate(maps) if n})
