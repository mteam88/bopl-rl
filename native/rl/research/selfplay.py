"""Historical-opponent bookkeeping for the recurrent PPO rollout.

train.py assigns every odd worker a frozen historical checkpoint. Which
checkpoint and which side change only when a match ends, but the original
loop rebuilt Python index lists and pushed them to the GPU several times per
version on every step; each list-indexed copy was a pageable transfer that
also synchronised the GPU. This module caches one index tensor per version
and refreshes it only when an assignment changes.

The order of work is the original one: versions are visited in `set(versions)`
iteration order and each version's actions are sampled with its own
distribution() call, so random-number consumption is unchanged and results
are bit-identical to the original loop under the same seed.
"""
import torch
from policy import distribution

class Opponents:
    def __init__(self,workers,device):
        self.workers=workers; self.device=device; self.key=None
        self.groups=[]; self.active=torch.ones(2*workers,dtype=torch.bool,device=device)
    def refresh(self,versions,historical,side):
        """Rebuild cached indices if any worker's opponent assignment changed."""
        key=(tuple(versions),tuple(side))
        if key==self.key: return
        self.key=key; groups=[]
        for version in set(versions):
            ids=[2*w+side[w] for w in range(self.workers) if historical[w] and versions[w]==version]
            if ids: groups.append((version,torch.tensor(ids,dtype=torch.long,device=self.device)))
        self.groups=groups
        self.active=torch.ones(2*self.workers,dtype=torch.bool,device=self.device)
        for _,ids in groups: self.active[ids]=False
    def act(self,pool,obs,frozen_state,chosen):
        """Replace historical agents' rows of `chosen` in place, advance their
        frozen recurrent state in place, and return the learner-active mask."""
        for version,ids in self.groups:
            opponent_logits,_,next_state=pool[version].step(obs.index_select(0,ids),frozen_state.index_select(1,ids))
            opponent_actions,_,_=distribution(opponent_logits)
            chosen.index_copy_(0,ids,opponent_actions); frozen_state.index_copy_(1,ids,next_state)
        return self.active
