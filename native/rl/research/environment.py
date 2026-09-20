"""Small read-only evaluator interface around the same C environment used by PufferLib."""
import ctypes as C
import os
import math
from pathlib import Path
import sys
import numpy as np
from telemetry import EVENT_FIELDS,FIELDS
import layout
from layout import ACTION_SIZES,NUM_ATNS
from evaluation_protocol import DEFAULT_LOADOUT,resolve_loadouts

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'native/tools'))
from reference import resolve_game

class Player(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('id','team','flags','alive_bodies','stocks','kills','deaths','cause_of_death')] + [(name,C.c_int64) for name in ('x','y','scale','invisible_time','ability_time')]
class Entity(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('id','kind','layer','owner','flags','hierarchy','reserved0','reserved1')] + [(name,C.c_int64) for name in ('x','y','vx','vy','right_x','right_y','up_x','up_y','radius','rotation','angular_velocity','inverse_mass','clock_rate')]
class Slime(C.Structure):
    _fields_ = [(name,C.c_uint32) for name in ('id','owner','flags','state')] + [(name,C.c_int64) for name in ('x','y','vx','vy','rotation','scale')] + [('abilities',C.c_uint32*3),('ability_flags',C.c_uint32*3),('cooldowns',C.c_int64*3),('timers',C.c_int64*3)]
class Header(C.Structure):
    _fields_ = ([(name,C.c_uint32) for name in ('version','tick','episode_ticks','flags','player_count','entity_count','warmup_ticks','slime_count')]
                + [(name,C.c_int64) for name in ('simulation_time','player_time_scale','simulation_time_since_level_loaded')]
                + [(name,C.c_uint32) for name in ('sudden_death','reserved')])
class Telemetry(C.Structure):
    _fields_ = [(name,C.c_uint32) for name in EVENT_FIELDS]
class State(C.Structure):
    _fields_ = [('header',Header),('players',Player*6),('entities',C.POINTER(Entity)),('slimes',C.POINTER(Slime)),('telemetry',Telemetry*6)]

LIB = C.CDLL(os.environ.get('BOPL_RESEARCH_LIBRARY') or str(ROOT/'native/artifacts/rl/libbopl_research.so'))
LIB.research_obs_size.restype = C.c_int
LIB.research_obs_stride.restype = C.c_int
LIB.research_obs_size_of.argtypes = [C.c_int]; LIB.research_obs_size_of.restype = C.c_int
LIB.research_layout_v4.argtypes = [C.POINTER(C.c_int),C.c_int]; LIB.research_layout_v4.restype = C.c_int
LIB.research_set_layout.argtypes = [C.c_void_p,C.c_int,C.c_int]; LIB.research_set_layout.restype = C.c_int
LIB.research_get_layout.argtypes = [C.c_void_p,C.c_int]; LIB.research_get_layout.restype = C.c_int
LIB.research_set_ablation.argtypes = [C.c_void_p,C.c_int,C.c_int]; LIB.research_set_ablation.restype = C.c_int
LIB.research_get_ablation.argtypes = [C.c_void_p,C.c_int]; LIB.research_get_ablation.restype = C.c_int
LIB.research_telemetry_size.restype = C.c_int
LIB.research_num_atns.restype = C.c_int
LIB.research_layout.argtypes = [C.POINTER(C.c_int),C.c_int]; LIB.research_layout.restype = C.c_int
LIB.research_check.argtypes = [C.c_void_p]; LIB.research_check.restype = C.c_int
LIB.research_overflow.argtypes = [C.c_void_p]; LIB.research_overflow.restype = C.c_ulong
if LIB.research_telemetry_size()!=len(FIELDS): raise RuntimeError('Rebuild the research telemetry binding')
# The compiled layout (bopl_layout.h) must match the Python mirror (layout.py) field for field.
_fields = (C.c_int*len(layout.LAYOUT_FIELDS))()
if (LIB.research_layout(_fields,len(_fields))!=len(layout.LAYOUT_FIELDS) or tuple(_fields)!=layout.LAYOUT_FIELDS
        or LIB.research_num_atns()!=NUM_ATNS or LIB.research_obs_size()!=layout.OBS_SIZE):
    raise RuntimeError(f'Research library layout {tuple(_fields)} differs from research/layout.py {layout.LAYOUT_FIELDS}; rebuild it')
_fields4 = (C.c_int*len(layout.V4_LAYOUT_FIELDS))()
if (LIB.research_layout_v4(_fields4,len(_fields4))!=len(layout.V4_LAYOUT_FIELDS) or tuple(_fields4)!=layout.V4_LAYOUT_FIELDS
        or LIB.research_obs_stride()!=layout.OBS_STRIDE):
    raise RuntimeError(f'Research library layout v4 {tuple(_fields4)} differs from research/layout.py {layout.V4_LAYOUT_FIELDS}; rebuild it')
LIB.research_open.argtypes = [C.c_uint32,C.c_uint32,C.c_uint32,C.c_int]
LIB.research_open.restype = C.c_void_p
LIB.research_reset.argtypes = [C.c_void_p,C.c_uint32,C.c_uint32]
LIB.research_configure_loadouts.argtypes = [C.c_void_p,C.POINTER(C.c_uint32),C.c_uint64,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int]
LIB.research_configure_loadouts.restype = C.c_int
LIB.research_loadouts.argtypes = [C.c_void_p,C.POINTER(C.c_uint8),C.c_int]; LIB.research_loadouts.restype = C.c_int
LIB.research_ability_telemetry.argtypes = [C.c_void_p,C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_double),C.c_int]
LIB.research_ability_telemetry.restype = C.c_int
LIB.research_step.argtypes = [C.c_void_p,C.POINTER(C.c_float)]
LIB.research_step_begin.argtypes = [C.c_void_p,C.POINTER(C.c_float)]; LIB.research_step_begin.restype = C.c_int
LIB.research_step_tick.argtypes = [C.c_void_p]; LIB.research_step_tick.restype = C.c_int
LIB.research_step_finish.argtypes = [C.c_void_p]; LIB.research_step_finish.restype = C.c_int
LIB.research_obs.argtypes = [C.c_void_p]; LIB.research_obs.restype = C.POINTER(C.c_float)
LIB.research_telemetry.argtypes = [C.c_void_p]; LIB.research_telemetry.restype = C.POINTER(C.c_float)
LIB.research_state.argtypes = [C.c_void_p]; LIB.research_state.restype = C.POINTER(State)
LIB.research_replay.argtypes = [C.c_void_p,C.c_char_p]
LIB.research_close.argtypes = [C.c_void_p]
LIB.research_bot.argtypes = [C.c_void_p,C.c_int,C.c_int,C.c_int]; LIB.research_bot.restype = C.c_int
LIB.research_bot_forks.argtypes = [C.c_int]; LIB.research_bot_forks.restype = C.c_int
LIB.research_forks.argtypes = [C.c_void_p]; LIB.research_forks.restype = C.c_int
LIB.research_mask.argtypes = [C.c_void_p]; LIB.research_mask.restype = C.POINTER(C.c_ubyte)
LIB.research_mask_size.restype = C.c_int
if LIB.research_mask_size()!=layout.LOGITS: raise RuntimeError('Research library action mask width differs from layout.LOGITS')
OBS_SIZE = LIB.research_obs_size()      # the layout-v3 row width (the default layout)
OBS_STRIDE = LIB.research_obs_stride()  # row memory: fits layout v4 too

def selected_game():
    """The player the C environment will actually launch (explicit > BOPL_RL_GAME > IL2CPP default)."""
    return resolve_game()

def game_identity(game=None):
    """Identify the selected player by its runtime payload, so a run is attributable to a build.

    Mono players carry the game code in Managed/Assembly-CSharp.dll. IL2CPP players compile it
    into GameAssembly.so plus il2cpp_data/Metadata/global-metadata.dat and ship no managed
    assembly, so hashing the Mono path alone would silently describe the wrong build."""
    import hashlib
    game=Path(game) if game else selected_game()
    data=game.parent/(game.name.split('.')[0]+'_Data')
    payload={}
    managed=data/'Managed/Assembly-CSharp.dll'
    if managed.exists(): payload['assembly']=managed
    native=game.parent/'GameAssembly.so'
    if native.exists():
        payload['game_assembly']=native
        metadata=data/'il2cpp_data/Metadata/global-metadata.dat'
        if metadata.exists(): payload['metadata']=metadata
    # The NativeAOT .NET sim (sim/RL.md) compiles the game code into the executable
    # itself and ships no payload files: the player hash alone identifies the build.
    backend='il2cpp' if 'game_assembly' in payload else 'mono' if payload else 'dotnet-aot'
    digests={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in payload.items()}
    return {'game':str(game),'backend':backend,'player_sha256':hashlib.sha256(game.read_bytes()).hexdigest(),
            **{name+'_sha256':digest for name,digest in digests.items()},
            'payload':{name:str(path) for name,path in payload.items()}}

def configure(logs, game=None):
    logs=Path(logs).resolve(); logs.mkdir(parents=True,exist_ok=True)
    os.environ['BOPL_RL_GAME']=str(resolve_game(game))
    os.environ['BOPL_RL_BOOTSTRAP']=str(ROOT/'native/fixtures/movement.rep')
    os.environ['BOPL_RL_LOG_DIR']=str(logs)

class Environment:
    def __init__(self, seed=73, map_id=0, limit=3600, delay=9, reward=None, loadouts=None):
        setter = None
        if reward is not None:
            kill, timeout = float(reward['kill']), float(reward['timeout'])
            if (not math.isfinite(kill) or not math.isfinite(timeout) or timeout > 0
                    or reward.get('win') != 1 or reward.get('loss') != -1
                    or reward.get('death') != -kill):
                raise ValueError('Unsupported evaluation reward configuration')
            setter = getattr(LIB, 'research_rewards', None)
            if setter is None:
                raise RuntimeError('Rebuild the evaluator to configure native checkpoint rewards')
            setter.argtypes = [C.c_void_p, C.c_float, C.c_float]
            setter.restype = C.c_int
        self.handle=LIB.research_open(seed,map_id,limit,delay)
        if not self.handle: raise RuntimeError('Could not open Bopl evaluator')
        try: self.configure_loadouts(loadouts)
        except BaseException:
            self.close(); raise
        # research_open already reset the default configuration; replay the
        # same seed/map once so the requested sampler owns episode zero.
        LIB.research_reset(self.handle,seed,map_id)
        if setter is not None and setter(self.handle, kill, timeout):
            self.close()
            raise ValueError('Evaluation reward values are outside native float range')
        # Observation rows of both players in one strided buffer. `obs` is the
        # (2, width) view over the widest layout in use (v3's 3690 columns by
        # default, so v3 callers see exactly the rows they always did); a v4
        # player's row is 4498 wide (set_layout). Slice obs[:, :policy.obs_size]
        # to feed a checkpoint, or take obs_row(player).
        self._rows=np.ctypeslib.as_array(LIB.research_obs(self.handle),shape=(2*OBS_STRIDE,)).reshape(2,OBS_STRIDE)
        self._layouts=[3,3]
        self._view()
        self.telemetry=np.ctypeslib.as_array(LIB.research_telemetry(self.handle),shape=(2*len(FIELDS),)).reshape(2,len(FIELDS))
        # Action mask of the next decision (1 = allowed), the one the trainer samples with:
        # while a player's buttons are latched only the held button state and hold length 1 remain.
        self.mask=np.ctypeslib.as_array(LIB.research_mask(self.handle),shape=(2*layout.LOGITS,)).reshape(2,layout.LOGITS)
        try: self.check()
        except BaseException:
            self.close();raise
    def configure_loadouts(self,loadouts):
        """Install a loadout selection (evaluation_protocol.resolve_loadouts); None = the
        fixed production loadout. The next reset draws its episode from it, so a rater
        that wants one named lineup per game reconfigures between resets: the C call only
        copies the rows (eval_binding.research_configure_loadouts, at most 256 x 6 bytes)
        and the research environment has no spare world pre-drawing the next config."""
        config=loadouts or resolve_loadouts(DEFAULT_LOADOUT)
        fixed=(C.c_uint32*3)(*config['fixed'])
        pool=sum(1<<ability for ability in config['pool'])
        flat=[ability for row in config['lineups'] for ability in row]
        sets=(C.c_uint8*len(flat))(*flat) if flat else None
        if LIB.research_configure_loadouts(self.handle,fixed,pool,sets,len(config['lineups']),
                                           int(config['no_dup']),round(config['mirror_share']*1_000_000)):
            raise ValueError('Invalid native loadout selection')
        self.loadout_config=config
    def _view(self):
        width=max(layout.obs_size_of(v) for v in self._layouts)
        self.obs=self._rows[:,:width]
    def set_layout(self,player,version,ablate=None):
        """Encode player's row in observation layout `version` (3 or 4, layout.py) and, for
        v4, with writer ablation bits `ablate` (the checkpoint's v4_shape.ablate,
        native_policy.writer_ablation; None keeps the player's current bits, which start
        at the process default BOPL4_ABLATE). Call
        after reset, like bot(); persists across resets. Selected per checkpoint via
        layout.layout_of(architecture) so v3 and v4 policies can share a game."""
        if ablate is not None and LIB.research_set_ablation(self.handle,player,int(ablate)): raise ValueError('Bad writer ablation selection')
        if LIB.research_set_layout(self.handle,player,int(version)): raise ValueError('Bad observation layout selection')
        self._layouts[player]=int(version); self._view()
    def ablation_of(self,player):
        """Player's v4 writer bits; -1 = the process default (BOPL4_ABLATE)."""
        return LIB.research_get_ablation(self.handle,player)
    def layout_of(self,player): return LIB.research_get_layout(self.handle,player)
    @property
    def rows(self):
        """Both players' observation rows at the padded stride (players x OBS_STRIDE), each in
        its own layout; slice row p to layout.obs_size_of(self.layout_of(p))."""
        return self._rows
    def obs_row(self,player):
        """Player's observation row in its own layout's width."""
        return self._rows[player,:layout.obs_size_of(self._layouts[player])]
    def check(self):
        if LIB.research_check(self.handle): raise RuntimeError('Game error during evaluation')
    @property
    def observation_overflow(self):
        """Cumulative sampled states truncated by the policy layout; non-fatal."""
        return int(LIB.research_overflow(self.handle))
    @property
    def loadouts(self):
        values=(C.c_uint8*6)()
        count=LIB.research_loadouts(self.handle,values,len(values))
        if count!=6: raise RuntimeError('Unexpected native loadout width')
        return [list(values[:3]),list(values[3:6])]
    def ability_telemetry(self):
        activations=(C.c_double*39)(); rows=(C.c_double*39)(); overflow=(C.c_double*39)()
        if LIB.research_ability_telemetry(self.handle,activations,rows,overflow,39)!=39:
            raise RuntimeError('Unexpected native ability telemetry width')
        return dict(activations=[int(value) for value in activations],
                    entity_rows=[int(value) for value in rows],
                    entity_overflow=[int(value) for value in overflow])
    @property
    def state(self): return LIB.research_state(self.handle).contents
    @property
    def ended(self): return bool(self.state.header.flags&1)
    def reset(self,seed,map_id=0):
        LIB.research_reset(self.handle,seed,map_id);self.check()
    def bot(self,player,kind,level=2):
        """Pin a scripted bot (bopl_bots.h) on a player; kind 0 detaches. Call after reset."""
        if LIB.research_bot(self.handle,player,kind,level): raise ValueError('Bad scripted bot selection')
    def forks(self):
        """Whether this environment's game backend can fork the world (bopl_rl_fork: the
        in-process Rust sim). Kinds with bot_forks(kind) abort on a backend that cannot."""
        return bool(LIB.research_forks(self.handle))
    @staticmethod
    def bot_forks(kind):
        """Whether a scripted kind decides by rollouts on a forked game (the search bot)."""
        return bool(LIB.research_bot_forks(kind))
    def step(self,actions):
        """actions: 2 players x NUM_ATNS floats (aim angle, button bits, hold length index)."""
        actions=np.ascontiguousarray(actions,dtype=np.float32).reshape(2,NUM_ATNS)
        LIB.research_step(self.handle,actions.ctypes.data_as(C.POINTER(C.c_float)))
        self.check()
    def step_ticks(self,actions):
        """step() split into its game ticks: a generator that yields after every
        tick (read self.state / frame() there) and completes the decision when
        exhausted. Same decision, same observations, masks, telemetry and replay
        as step(); play.py uses it to show all 60 ticks/s. Must be run to the end."""
        actions=np.ascontiguousarray(actions,dtype=np.float32).reshape(2,NUM_ATNS)
        if LIB.research_step_begin(self.handle,actions.ctypes.data_as(C.POINTER(C.c_float))): raise RuntimeError('A decision is already in progress')
        try:
            while True:
                left=LIB.research_step_tick(self.handle)
                if left<0: raise RuntimeError('No decision in progress')
                self.check()
                yield left
                if left==0: break
        finally:
            if LIB.research_step_finish(self.handle): raise RuntimeError('Decision finished before its ticks were run')
            self.check()
    def replay(self,path):
        if LIB.research_replay(self.handle,os.fsencode(path)): raise RuntimeError('Replay export failed')
    def frame(self):
        s=self.state
        fix=lambda x: round(x/4294967296.,5)
        return dict(t=s.header.episode_ticks, flags=s.header.flags,
            players=[dict(id=p.id,flags=p.flags,stocks=p.stocks,kills=p.kills,deaths=p.deaths,cause_of_death=p.cause_of_death,x=fix(p.x),y=fix(p.y),scale=fix(p.scale)) for p in s.players[:s.header.player_count]],
            telemetry=[{k:getattr(s.telemetry[i],k) for k in EVENT_FIELDS} for i in range(s.header.player_count)],
            entities=[dict(id=e.id,kind=e.kind,layer=e.layer,owner=e.owner,flags=e.flags,x=fix(e.x),y=fix(e.y),vx=fix(e.vx),vy=fix(e.vy),rx=fix(e.right_x),ry=fix(e.right_y),ux=fix(e.up_x),uy=fix(e.up_y),radius=fix(e.radius),rotation=fix(e.rotation)) for e in s.entities[:s.header.entity_count] if e.flags&1 and not e.flags&4],
            slimes=[dict(owner=x.owner,flags=x.flags,state=x.state,x=fix(x.x),y=fix(x.y),vx=fix(x.vx),vy=fix(x.vy),scale=fix(x.scale),abilities=list(x.abilities),cooldowns=[fix(v) for v in x.cooldowns],timers=[fix(v) for v in x.timers]) for x in s.slimes[:s.header.slime_count]])
    def close(self):
        if self.handle: LIB.research_close(self.handle); self.handle=None
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
