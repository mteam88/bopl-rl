"""Gymnasium adapter for the verified movement/dash slice.

One policy controls player 1; the optional opponent controls player 2 using the
same action encoding. Without an opponent player 2 stays still. This is a
locomotion/survival scaffold, not a complete combat training task.
"""
import gymnasium as gym
import numpy as np
from bopl_native import World

ONE=2**32

def features(player):
    result=[]
    def vector(v):result.extend([v['x']/ONE,v['y']/ONE])
    for name in ('position','velocity','external_velocity','up'):vector(player[name])
    for name in ('rotation','local','grounded_speed','radius','cooldown'):result.append(player[name]/ONE)
    result.extend([float(player[k]) for k in ('alive','grounded','previous_jump')])
    result.append((player['ground'] or 0)/150)
    d=player['dash'];result.extend([float(d['active']),float(d['in_dash']),d['updates']/4,d['time']/ONE])
    vector(d['input']);result.extend([(d['hit_ground'] or 0)/150,d['hit_local']/ONE,d['local']/ONE,d['saved_local']/ONE])
    for key in ('center','right','up','inverse_extents'):vector(player['collider'][key])
    return result

class BoplEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,opponent=None,max_steps=1200):
        if max_steps<=0:raise ValueError('max_steps must be positive')
        self.world=World();self.opponent=opponent;self.max_steps=max_steps;self.elapsed=0;self._ended=False
        self.action_space=gym.spaces.MultiDiscrete([256,4])
        shape=(2*len(features(self.world.state()['players'][0])),)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=shape,dtype=np.float32)
    def _observation(self,state):return np.asarray([v for p in state['players'] for v in features(p)],dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if options:raise ValueError('reset options are unsupported; the native reset is pinned')
        self.world.reset();self.elapsed=0;self._ended=False
        return self._observation(self.world.state()),{'reference_reset_seed':12345}
    def step(self,action):
        if self._ended:raise RuntimeError('reset required after episode end')
        if not self.action_space.contains(action):raise ValueError('invalid action')
        opponent=[0,0] if self.opponent is None else self.opponent(self.world.state(),self.np_random)
        self.world.step([[int(action[0]),int(action[1])],[int(v) for v in opponent]])
        self.elapsed+=1;state=self.world.state();terminated=self.world.terminal
        truncated=self.elapsed>=self.max_steps and not terminated
        self._ended=terminated or truncated
        # Explicit starter objective: stay alive. Replace this for a combat task
        # when damaging abilities and match scoring have parity coverage.
        reward=1/60 if state['players'][0]['alive'] else -1.0
        return self._observation(state),reward,terminated,truncated,{'simulation_tick':state['tick']}
    def close(self):self.world.close()
