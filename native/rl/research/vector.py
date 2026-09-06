"""Pinned PufferLib C vector runner with tensors shared with the CUDA backend."""
import sys
import ctypes
import numpy as np
import torch
from telemetry import FIELDS
from pufferlib import _C,pufferl
from pufferlib.torch_pufferl import _CudaPtr, _cpu_tensor

class Vector:
    def __init__(self,workers,seed,limit=3600,delay=9):
        saved=sys.argv; sys.argv=[saved[0]]
        try: args=pufferl.load_config('bopl')
        finally: sys.argv=saved
        args['vec'].update(total_agents=2*workers,num_threads=workers,num_buffers=1)
        args['env'].update(players=2,seed=seed,stocks=3,map=0,ability1=2,ability2=4,ability3=14,
                           action_repeat=4,observation_delay=delay,max_episode_ticks=limit,
                           random_maps=0,random_loadouts=0)
        # Retained for the complete native vector lifetime. Workers own disjoint
        # slices, and Python reads only after synchronous gpu_step returns.
        self.telemetry=np.zeros((2*workers,len(FIELDS)),dtype=np.float32)
        if ctypes.CDLL(_C.__file__).bopl_puffer_telemetry_size()!=len(FIELDS):
            raise RuntimeError('Rebuild the PufferLib telemetry binding')
        pointer=self.telemetry.ctypes.data
        if int(float(pointer))!=pointer: raise RuntimeError('Telemetry pointer cannot cross numeric config safely')
        args['env']['telemetry_ptr']=pointer
        args['env']['telemetry_agents']=2*workers
        self.config=args
        if not _C.gpu: raise RuntimeError('Build the CUDA PufferLib backend first')
        self.vec=_C.create_vec(args,True)
        self.size=self.vec.total_agents; self.obs_size=self.vec.obs_size
        self.obs=torch.as_tensor(_CudaPtr(self.vec.gpu_obs_ptr,(self.size,self.obs_size),torch.float32))
        self.rewards=torch.as_tensor(_CudaPtr(self.vec.gpu_rewards_ptr,(self.size,),torch.float32))
        self.dones=torch.as_tensor(_CudaPtr(self.vec.gpu_terminals_ptr,(self.size,),torch.float32))
        # These are useful for episode boundaries without a separate GPU sync.
        self.cpu_dones=_cpu_tensor(self.vec.terminals_ptr,(self.size,),torch.float32)
        self.vec.reset()
    def step(self,actions):
        actions=actions.to(dtype=torch.float32).contiguous()
        torch.cuda.synchronize()
        self.vec.gpu_step(actions.data_ptr())
    def log(self): return dict(self.vec.log())
    def close(self):
        self.obs=self.rewards=self.dones=self.cpu_dones=None
        self.vec.close()
        self.telemetry=None
