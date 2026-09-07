"""Small read-only evaluator interface around the same C environment used by PufferLib."""
import ctypes as C
import os
import math
from pathlib import Path
import sys
import numpy as np
from telemetry import EVENT_FIELDS,FIELDS

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'native/tools'))
from reference import resolve_game

class Player(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('id','team','flags','alive_bodies','stocks','kills','deaths','cause_of_death')] + [(name,C.c_int64) for name in ('x','y','scale','invisible_time','ability_time')]
class Entity(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('id','kind','layer','owner','flags','hierarchy','reserved0','reserved1')] + [(name,C.c_int64) for name in ('x','y','vx','vy','right_x','right_y','up_x','up_y','radius','rotation','angular_velocity','inverse_mass')]
class Slime(C.Structure):
    _fields_ = [(name,C.c_uint32) for name in ('id','owner','flags','state')] + [(name,C.c_int64) for name in ('x','y','vx','vy','rotation','scale')] + [('abilities',C.c_uint32*3),('ability_flags',C.c_uint32*3),('cooldowns',C.c_int64*3),('timers',C.c_int64*3)]
class Header(C.Structure):
    _fields_ = [(name,C.c_uint32) for name in ('version','tick','episode_ticks','flags','player_count','entity_count','warmup_ticks','slime_count')] + [('simulation_time',C.c_int64),('player_time_scale',C.c_int64)]
class Telemetry(C.Structure):
    _fields_ = [(name,C.c_uint32) for name in EVENT_FIELDS]
class State(C.Structure):
    _fields_ = [('header',Header),('players',Player*6),('entities',C.POINTER(Entity)),('slimes',C.POINTER(Slime)),('telemetry',Telemetry*6)]

LIB = C.CDLL(str(ROOT/'native/artifacts/rl/libbopl_research.so'))
LIB.research_obs_size.restype = C.c_int
LIB.research_telemetry_size.restype = C.c_int
LIB.research_check.argtypes = [C.c_void_p]; LIB.research_check.restype = C.c_int
if LIB.research_telemetry_size()!=len(FIELDS): raise RuntimeError('Rebuild the research telemetry binding')
LIB.research_open.argtypes = [C.c_uint32,C.c_uint32,C.c_uint32,C.c_int]
LIB.research_open.restype = C.c_void_p
LIB.research_reset.argtypes = [C.c_void_p,C.c_uint32,C.c_uint32]
LIB.research_step.argtypes = [C.c_void_p,C.POINTER(C.c_float)]
LIB.research_obs.argtypes = [C.c_void_p]; LIB.research_obs.restype = C.POINTER(C.c_float)
LIB.research_telemetry.argtypes = [C.c_void_p]; LIB.research_telemetry.restype = C.POINTER(C.c_float)
LIB.research_state.argtypes = [C.c_void_p]; LIB.research_state.restype = C.POINTER(State)
LIB.research_replay.argtypes = [C.c_void_p,C.c_char_p]
LIB.research_step_mode.argtypes = [C.c_void_p]; LIB.research_step_mode.restype = C.c_int
STEP_MODES={2:'sampled',1:'split',0:'fallback'}
LIB.research_close.argtypes = [C.c_void_p]
OBS_SIZE = LIB.research_obs_size()

def configure(logs, game=None):
    logs=Path(logs).resolve(); logs.mkdir(parents=True,exist_ok=True)
    os.environ['BOPL_RL_GAME']=str(resolve_game(game))
    os.environ['BOPL_RL_BOOTSTRAP']=str(ROOT/'native/fixtures/movement.rep')
    os.environ['BOPL_RL_LOG_DIR']=str(logs)

class Environment:
    def __init__(self, seed=73, map_id=0, limit=3600, delay=9, reward=None):
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
        if setter is not None and setter(self.handle, kill, timeout):
            self.close()
            raise ValueError('Evaluation reward values are outside native float range')
        self.obs=np.ctypeslib.as_array(LIB.research_obs(self.handle),shape=(2*OBS_SIZE,)).reshape(2,OBS_SIZE)
        self.telemetry=np.ctypeslib.as_array(LIB.research_telemetry(self.handle),shape=(2*len(FIELDS),)).reshape(2,len(FIELDS))
        try: self.check()
        except BaseException:
            self.close();raise
    def check(self):
        if LIB.research_check(self.handle): raise RuntimeError('Game error or observation overflow during evaluation')
    @property
    def state(self): return LIB.research_state(self.handle).contents
    @property
    def ended(self): return bool(self.state.header.flags&1)
    def reset(self,seed,map_id=0):
        LIB.research_reset(self.handle,seed,map_id);self.check()
    def step(self,actions):
        actions=np.ascontiguousarray(actions,dtype=np.float32).reshape(2,2)
        LIB.research_step(self.handle,actions.ctypes.data_as(C.POINTER(C.c_float)))
        self.check()
    @property
    def step_mode(self):
        """Report sampled, forced split, or old-game fallback after stepping."""
        return STEP_MODES[LIB.research_step_mode(self.handle)]
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
