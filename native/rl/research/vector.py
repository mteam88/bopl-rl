"""Pinned PufferLib C vector runner with tensors shared with the CUDA backend.

`groups=1` (default) is the original synchronous path: `step` copies actions,
synchronises the device and runs the upstream gpu_step. `groups>1` splits the
workers into contiguous groups that step on their own C thread and CUDA stream
(binding.c), so the trainer can run inference for one group while another
group's games are ticking. Every group keeps the exact per-environment
semantics of c_step: nine-tick observation delay, reward/terminal alignment,
autoreset and telemetry are unchanged because each environment is still
stepped by the same code, once per decision.
"""
import sys
import ctypes
import numpy as np
import torch
from telemetry import FIELDS
from pufferlib import _C,pufferl
from pufferlib.torch_pufferl import _CudaPtr, _cpu_tensor

class Vector:
    def __init__(self,workers,seed,limit=3600,delay=9,groups=1,*,training_maps=(0,)):
        if not training_maps or len(set(training_maps))!=len(training_maps) or any(m<0 or m>30 for m in training_maps):
            raise ValueError('Distinct training map IDs between 0 and 30 required')
        saved=sys.argv; sys.argv=[saved[0]]
        try: args=pufferl.load_config('bopl')
        finally: sys.argv=saved
        args['vec'].update(total_agents=2*workers,num_threads=workers,num_buffers=1)
        args['env'].update(players=2,seed=seed,stocks=3,map=training_maps[0],ability1=2,ability2=4,ability3=14,
                           training_map_mask=sum(1<<m for m in training_maps) if len(training_maps)>1 else 0,
                           action_repeat=4,observation_delay=delay,max_episode_ticks=limit,
                           random_maps=0,random_loadouts=0)
        # Retained for the complete native vector lifetime. Workers own disjoint
        # slices, and Python reads only after synchronous gpu_step returns (or
        # after wait_group for that group's slice).
        self.telemetry=np.zeros((2*workers,len(FIELDS)),dtype=np.float32)
        self.lib=ctypes.CDLL(_C.__file__)
        self.lib.bopl_vec_step_mode.restype=ctypes.c_int; self.lib.bopl_vec_step_mode.argtypes=[ctypes.c_void_p]
        if self.lib.bopl_puffer_telemetry_size()!=len(FIELDS):
            raise RuntimeError('Rebuild the PufferLib telemetry binding')
        pointer=self.telemetry.ctypes.data
        if int(float(pointer))!=pointer: raise RuntimeError('Telemetry pointer cannot cross numeric config safely')
        args['env']['telemetry_ptr']=pointer
        args['env']['telemetry_agents']=2*workers
        self.config=args
        if not _C.gpu: raise RuntimeError('Build the CUDA PufferLib backend first')
        if groups<1 or workers%groups: raise ValueError('Pipeline groups must divide the worker count')
        self.vec=_C.create_vec(args,True)
        self.size=self.vec.total_agents; self.obs_size=self.vec.obs_size
        self.obs=torch.as_tensor(_CudaPtr(self.vec.gpu_obs_ptr,(self.size,self.obs_size),torch.float32))
        self.rewards=torch.as_tensor(_CudaPtr(self.vec.gpu_rewards_ptr,(self.size,),torch.float32))
        self.dones=torch.as_tensor(_CudaPtr(self.vec.gpu_terminals_ptr,(self.size,),torch.float32))
        # These are useful for episode boundaries without a separate GPU sync.
        self.cpu_dones=_cpu_tensor(self.vec.terminals_ptr,(self.size,),torch.float32)
        self.vec.reset()
        self.workers=workers; self.group_count=groups; self.handle=None; self.groups=[]; self.actions=[]
        if groups>1:
            lib=self.lib
            lib.bopl_groups_open.restype=ctypes.c_void_p
            lib.bopl_groups_open.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
            lib.bopl_groups_close.argtypes=[ctypes.c_void_p]
            for name in ('bopl_group_agent_start','bopl_group_agent_count','bopl_group_pending'):
                getattr(lib,name).restype=ctypes.c_int; getattr(lib,name).argtypes=[ctypes.c_void_p,ctypes.c_int]
            lib.bopl_group_actions_ptr.restype=ctypes.c_void_p; lib.bopl_group_actions_ptr.argtypes=[ctypes.c_void_p,ctypes.c_int]
            lib.bopl_group_launch.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
            lib.bopl_group_wait.restype=ctypes.c_int; lib.bopl_group_wait.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
            for name in ('bopl_group_env_ms','bopl_group_total_ms'):
                getattr(lib,name).restype=ctypes.c_double; getattr(lib,name).argtypes=[ctypes.c_void_p,ctypes.c_int]
            lib.bopl_group_launches.restype=ctypes.c_ulong; lib.bopl_group_launches.argtypes=[ctypes.c_void_p,ctypes.c_int]
            self.handle=lib.bopl_groups_open(self.vec.gpu_obs_ptr,groups,workers)
            if not self.handle: raise RuntimeError('Could not create pipeline groups')
            for g in range(groups):
                start=lib.bopl_group_agent_start(self.handle,g); count=lib.bopl_group_agent_count(self.handle,g)
                self.groups.append(slice(start,start+count))
                # The group owns this device buffer; a copy into it on the torch
                # stream is the only hand-off, so the caller's action tensor may
                # be freed or overwritten (e.g. a CUDA graph output) immediately.
                self.actions.append(torch.as_tensor(_CudaPtr(lib.bopl_group_actions_ptr(self.handle,g),(count,2),torch.float32)))
            if self.groups[-1].stop!=self.size: raise RuntimeError('Pipeline groups do not cover the vector')
        self.workers_per_group=workers//groups
    def step(self,actions):
        """Original synchronous full-batch step (groups==1 only)."""
        if self.handle: raise RuntimeError('Use step_group/wait_group with pipeline groups')
        actions=actions.to(dtype=torch.float32).contiguous()
        torch.cuda.synchronize()
        self.vec.gpu_step(actions.data_ptr())
    def step_group(self,g,actions):
        """Start stepping group g with `actions` (rows for that group only).
        Returns immediately; the group's slices of obs/rewards/dones/cpu_dones/
        telemetry are undefined until wait_group(g)."""
        stream=torch.cuda.current_stream()
        self.actions[g].copy_(actions)
        self.lib.bopl_group_launch(self.handle,g,ctypes.c_void_p(stream.cuda_stream))
    def wait_group(self,g):
        """Block until group g's step is complete on the host and make the
        current torch stream wait for its device uploads."""
        stream=torch.cuda.current_stream()
        return bool(self.lib.bopl_group_wait(self.handle,g,ctypes.c_void_p(stream.cuda_stream)))
    def wait_all(self):
        for g in range(len(self.groups)): self.wait_group(g)
    def pending(self,g): return bool(self.lib.bopl_group_pending(self.handle,g))
    def group_profile(self):
        if not self.handle: return {}
        return {g:dict(env_ms=self.lib.bopl_group_env_ms(self.handle,g),total_ms=self.lib.bopl_group_total_ms(self.handle,g),
                       launches=int(self.lib.bopl_group_launches(self.handle,g))) for g in range(len(self.groups))}
    STEP_MODES={2:'sampled',1:'split',0:'fallback'}
    def step_mode(self):
        """'sampled': every game uses the single-roundtrip sampled command;
        'split': the two-roundtrip oracle was forced with BOPL_SPLIT_STEP;
        'fallback': some game rejected the sampled command (old game build)."""
        return self.STEP_MODES[self.lib.bopl_vec_step_mode(self.vec.gpu_obs_ptr)]
    def log(self):
        if self.handle: self.wait_all()
        return dict(self.vec.log(),step_mode=self.step_mode())
    def close(self):
        """Stop group threads, then wait for every queued torch consumer of the
        vector's device buffers (inference reading obs, copies out of rewards)
        before the native close frees those allocations."""
        if self.handle:
            if any(self.pending(g) for g in range(len(self.groups))): self.wait_all()
            self.lib.bopl_groups_close(self.handle); self.handle=None
        if self.vec is not None: torch.cuda.synchronize()
        self.obs=self.rewards=self.dones=self.cpu_dones=None; self.actions=[]
        if self.vec is not None: self.vec.close(); self.vec=None
        self.telemetry=None
