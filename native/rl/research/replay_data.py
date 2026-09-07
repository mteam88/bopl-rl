"""Strict human replay decoding and tick-level access to the real game.

No Unity state or policy features are approximated in Python. C owns encoding;
callers must copy observations before stepping again.
"""
import ctypes as C
import hashlib
import os
from pathlib import Path
import struct
import sys
import numpy as np
from environment import LIB, ROOT, OBS_SIZE, State
sys.path.insert(0, str(ROOT / "native/tools"))
from reference import resolve_game, player_sources

INPUT_DTYPE = np.dtype([('seq','<u4'),('angle','u1'),('buttons','<u2'),('delay','u1')])

def hold_lengths(buttons, bit, origin=240):
    """Completed presses beginning at/after origin; exclude censored holds."""
    on = (buttons & (1 << bit)) != 0
    starts = np.flatnonzero(~on[:-1] & on[1:]) + 1
    ends = np.flatnonzero(on[:-1] & ~on[1:]) + 1
    result = []
    for start in starts[starts >= origin]:
        release = ends[ends > start]
        if len(release): result.append(int(release[0]-start))
    return result

def decode(path):
    path = Path(path)
    raw = path.read_bytes()
    if not raw.startswith(b'BOPLREP2'):
        raise ValueError(f'{path}: only lossless BOPLREP2 is supported')
    if hashlib.sha256(raw[:-32]).digest() != raw[-32:]:
        raise ValueError(f'{path}: checksum mismatch')
    seed, level, mode, stocks, count = struct.unpack_from('<IBBiB', raw, 136)
    if not 1 <= count <= 6:
        raise ValueError('Invalid player count')
    offset, players = 147, []
    for index in range(count):
        pid, color, team, keyboard, slots = struct.unpack_from('5B', raw, offset)
        offset += 5
        if pid != index + 1 or not 1 <= slots <= 3 or keyboard > 1:
            raise ValueError('Unsupported player layout')
        abilities = list(struct.unpack_from('<' + 'H'*slots, raw, offset))
        offset += slots*2
        players.append(dict(id=pid, color=color, team=team, keyboard=bool(keyboard), abilities=abilities))
    ticks, = struct.unpack_from('<i', raw, offset)
    offset += 4
    if ticks < 1 or offset + ticks*count*8 != len(raw)-32:
        raise ValueError('Invalid replay length')
    inputs = np.frombuffer(raw, INPUT_DTYPE, count=ticks*count, offset=offset).reshape(ticks, count)
    if np.any(inputs['buttons'] & ~np.uint16(1023)):
        raise ValueError('Unknown input buttons')
    meta = dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                source=raw[8:72].decode(), catalog=raw[72:136].decode(), seed=seed,
                level=level, mode=mode, stocks=stocks, players=players, ticks=ticks)
    return meta, inputs

class PlayerConfig(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('color','team','keyboard')] + [('abilities', C.c_uint32*3)]
class Config(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('seed','map','players','slots','stocks','mode','warmup_ticks','max_episode_ticks')] + [('player', PlayerConfig*6)]
class Action(C.Structure):
    _fields_ = [('angle',C.c_uint8),('reserved',C.c_uint8),('buttons',C.c_uint16)]

LIB.bopl_rl_open.argtypes = [C.c_char_p,C.c_char_p,C.c_char_p,C.c_uint32,C.c_char_p,C.c_size_t]
LIB.bopl_rl_open.restype = C.c_void_p
LIB.bopl_rl_reset.argtypes = [C.c_void_p,C.POINTER(Config)]
LIB.bopl_rl_step.argtypes = [C.c_void_p,C.POINTER(Action),C.c_size_t,C.c_uint32]
LIB.bopl_rl_state.argtypes = [C.c_void_p]; LIB.bopl_rl_state.restype = C.POINTER(State)
LIB.bopl_rl_close.argtypes = [C.c_void_p]
LIB.bopl_rl_replay.argtypes = [C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)]
for name in ('error','source_hash','catalog_hash'):
    function = getattr(LIB, 'bopl_rl_'+name)
    function.argtypes = [C.c_void_p]; function.restype = C.c_char_p
LIB.research_pack.argtypes = [C.POINTER(State),C.POINTER(Config),C.POINTER(C.c_float)]
LIB.research_previous.argtypes = [C.POINTER(C.c_float),C.POINTER(Action),C.c_int]

def action_array(actions):
    return (Action*len(actions))(*(Action(int(a),0,int(b)) for a,b in actions))

def previous(observations, actions):
    LIB.research_previous(observations.ctypes.data_as(C.POINTER(C.c_float)), action_array(actions), len(actions))

class ReplayWorld:
    def __init__(self, meta, log, limit=None, game=None):
        self.game = resolve_game(game)
        self.meta = meta
        self.config = config = Config()
        config.seed, config.map, config.players = meta['seed'], meta['level'], len(meta['players'])
        config.slots = len(meta['players'][0]['abilities'])
        config.stocks, config.mode = meta['stocks'], meta['mode']
        config.max_episode_ticks = limit or meta['ticks']
        for i, player in enumerate(meta['players']):
            if len(player['abilities']) != config.slots:
                raise ValueError('Unequal ability slots cannot be represented by the RL reset API')
            if any(a < 1 or a > 38 for a in player['abilities']):
                raise ValueError('Unresolved or unsupported ability')
            config.player[i] = PlayerConfig(player['color'], player['team'], player['keyboard'],
                                           (C.c_uint32*3)(*player['abilities']))
        log = Path(log).resolve(); log.parent.mkdir(parents=True, exist_ok=True)
        error = C.create_string_buffer(512)
        self.handle = LIB.bopl_rl_open(os.fsencode(self.game),
            os.fsencode(ROOT/'native/fixtures/movement.rep'), os.fsencode(log), 60000, error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode())
        try:
            for key in ('source','catalog'):
                if getattr(LIB,'bopl_rl_'+key+'_hash')(self.handle).decode() != meta[key]:
                    raise ValueError(f'Replay {key} identity differs from game')
            self.check(LIB.bopl_rl_reset(self.handle, C.byref(config)))
        except BaseException:
            self.close(); raise
    def check(self, status):
        if status:
            raise RuntimeError(LIB.bopl_rl_error(self.handle).decode())
    @property
    def state(self): return LIB.bopl_rl_state(self.handle).contents
    @property
    def ended(self): return bool(self.state.header.flags & 1)
    def pack(self, origin=240, episode_limit=3600):
        # The RL clock starts after its neutral 240-tick warmup. Human playback
        # starts at tick zero, so translate only this policy feature's clock.
        pointer = LIB.bopl_rl_state(self.handle)
        state = State.from_buffer_copy(pointer.contents)
        state.header.episode_ticks = max(0, state.header.episode_ticks-origin)
        config = Config.from_buffer_copy(self.config)
        config.max_episode_ticks = episode_limit
        obs = np.empty((config.players,OBS_SIZE), np.float32)
        LIB.research_pack(C.byref(state), C.byref(config), obs.ctypes.data_as(C.POINTER(C.c_float)))
        if not np.isfinite(obs).all(): raise ValueError('Nonfinite observation')
        return obs
    def step(self, actions, repeat=1):
        self.check(LIB.bopl_rl_step(self.handle, action_array(actions), self.config.players, repeat))
    def replay(self, path):
        data, length = C.c_void_p(), C.c_size_t()
        self.check(LIB.bopl_rl_replay(self.handle,C.byref(data),C.byref(length)))
        Path(path).write_bytes(C.string_at(data,length.value))
    def close(self):
        if self.handle: LIB.bopl_rl_close(self.handle); self.handle=None
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
