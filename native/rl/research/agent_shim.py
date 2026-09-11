"""ctypes view of the in-game agent shim (native/rl/puffer/bopl_agent.h,
native/artifacts/rl/libbopl_agent.so; build with `make -C native/rl agent`).

The same library the Unity build loads (RlAgent.cs): one tick() per simulation
tick with the game state in the RlObservation.cs wire format, an InputPacket
back. Used by check_agent_fidelity.py to prove that what the game applies is
what the research environment would have decided."""
import ctypes as C
import os
from pathlib import Path
import numpy as np
import layout

ROOT = Path(__file__).resolve().parents[3]
LIBRARY = Path(os.environ.get('BOPL_AGENT_LIBRARY') or ROOT/'native/artifacts/rl/libbopl_agent.so')

class Action(C.Structure):
    _fields_ = [('angle', C.c_uint8), ('reserved', C.c_uint8), ('buttons', C.c_uint16)]

_lib = None
def library():
    global _lib
    if _lib is not None: return _lib
    lib = C.CDLL(str(LIBRARY))
    lib.bopl_agent_abi_version.restype = C.c_uint32
    lib.bopl_agent_open.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_char_p, C.c_size_t]; lib.bopl_agent_open.restype = C.c_void_p
    lib.bopl_agent_reset.argtypes = [C.c_void_p, C.c_uint32]
    lib.bopl_agent_tick_bytes.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(Action)]; lib.bopl_agent_tick_bytes.restype = C.c_int
    lib.bopl_agent_close.argtypes = [C.c_void_p]
    for name in ('ability',):
        getattr(lib, 'bopl_agent_'+name).argtypes = [C.c_void_p, C.c_int]; getattr(lib, 'bopl_agent_'+name).restype = C.c_int
    for name in ('warmup_ticks', 'repeat', 'observation_delay', 'max_episode_ticks', 'is_policy', 'decisions'):
        getattr(lib, 'bopl_agent_'+name).argtypes = [C.c_void_p]; getattr(lib, 'bopl_agent_'+name).restype = C.c_int
    lib.bopl_agent_id.argtypes = [C.c_void_p]; lib.bopl_agent_id.restype = C.c_char_p
    lib.bopl_agent_observation.argtypes = [C.c_void_p]; lib.bopl_agent_observation.restype = C.POINTER(C.c_float)
    lib.bopl_agent_mask.argtypes = [C.c_void_p]; lib.bopl_agent_mask.restype = C.POINTER(C.c_ubyte)
    lib.bopl_agent_logits.argtypes = [C.c_void_p]; lib.bopl_agent_logits.restype = C.POINTER(C.c_float)
    lib.bopl_agent_decision.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
    lib.bopl_agent_obs_size.restype = C.c_int; lib.bopl_agent_logits_size.restype = C.c_int
    if lib.bopl_agent_abi_version() != 1: raise RuntimeError('agent shim ABI')
    if lib.bopl_agent_obs_size() != layout.OBS_SIZE or lib.bopl_agent_logits_size() != layout.LOGITS:
        raise RuntimeError('agent shim layout differs from research/layout.py; rebuild it')
    _lib = lib
    return lib

def state_bytes(state):
    """Serialise a research State (environment.py ctypes struct) as the wire format."""
    h = state.header
    n, e, s = h.player_count, h.entity_count, h.slime_count
    parts = [C.string_at(C.addressof(h), C.sizeof(h)),
             C.string_at(C.addressof(state.players), n*C.sizeof(state.players[0])),
             C.string_at(state.entities, e*C.sizeof(state.entities[0])) if e else b'',
             C.string_at(state.slimes, s*C.sizeof(state.slimes[0])) if s else b'',
             C.string_at(C.addressof(state.telemetry), n*C.sizeof(state.telemetry[0]))]
    return b''.join(parts)

class Agent:
    def __init__(self, spec, players=2, player=1):
        lib = library()
        error = C.create_string_buffer(1024)
        self.handle = lib.bopl_agent_open(os.fsencode(str(spec)), players, player, error, len(error))
        if not self.handle: raise ValueError(error.value.decode())
        self.lib = lib; self.players = players; self.player = player
        self.observation = np.ctypeslib.as_array(lib.bopl_agent_observation(self.handle), shape=(layout.OBS_SIZE,))
        self.mask = np.ctypeslib.as_array(lib.bopl_agent_mask(self.handle), shape=(layout.LOGITS,))
        logits = lib.bopl_agent_logits(self.handle)
        self.logits = np.ctypeslib.as_array(logits, shape=(layout.DECODER,)) if logits else None
    @property
    def id(self): return self.lib.bopl_agent_id(self.handle).decode()
    @property
    def is_policy(self): return bool(self.lib.bopl_agent_is_policy(self.handle))
    @property
    def abilities(self): return [self.lib.bopl_agent_ability(self.handle, s) for s in range(3)]
    @property
    def warmup_ticks(self): return self.lib.bopl_agent_warmup_ticks(self.handle)
    @property
    def max_episode_ticks(self): return self.lib.bopl_agent_max_episode_ticks(self.handle)
    @property
    def decisions(self): return self.lib.bopl_agent_decisions(self.handle)
    @property
    def decision(self):
        out = (C.c_int*3)(); self.lib.bopl_agent_decision(self.handle, out); return tuple(out)
    def reset(self, seed): self.lib.bopl_agent_reset(self.handle, seed)
    def tick(self, data):
        """data: wire bytes (or a research State). Returns (status, (angle, buttons))."""
        if not isinstance(data, (bytes, bytearray)): data = state_bytes(data)
        out = Action()
        status = self.lib.bopl_agent_tick_bytes(self.handle, data, len(data), C.byref(out))
        if status < 0: raise RuntimeError(f'agent tick failed ({status})')
        return status, (out.angle, out.buttons)
    def close(self):
        if self.handle: self.lib.bopl_agent_close(self.handle); self.handle = None
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
