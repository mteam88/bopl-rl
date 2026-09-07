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

One instance covers one pipeline group of `workers` consecutive workers
starting at global worker `first`; indices are local to that group's agent
slice. With CUDA-graph steppers (fixed batch, forward only) each version's rows are
padded to the group's historical agent count by repeating real rows; sampling
happens eagerly on the real rows so random-number use matches the eager path.
"""
import torch
from policy import distribution

class Opponents:
    def __init__(self,workers,device,first=0):
        self.workers=workers; self.device=device; self.first=first; self.key=None
        self.groups=[]; self.active=torch.ones(2*workers,dtype=torch.bool,device=device)
        self.capacity=sum(1 for w in range(first,first+workers) if w%2==1)
    def refresh(self,versions,historical,side):
        """Rebuild cached indices if any of this group's assignments changed.
        Arguments are the global per-worker lists."""
        span=range(self.first,self.first+self.workers)
        key=(tuple(versions[w] for w in span),tuple(side[w] for w in span),tuple(historical[w] for w in span))
        if key==self.key: return
        self.key=key; groups=[]
        for version in set(versions[w] for w in span):
            ids=[2*(w-self.first)+side[w] for w in span if historical[w] and versions[w]==version]
            if ids:
                padded=(ids*self.capacity)[:self.capacity]
                groups.append((version,torch.tensor(ids,dtype=torch.long,device=self.device),
                               torch.tensor(padded,dtype=torch.long,device=self.device)))
        self.groups=groups
        self.active=torch.ones(2*self.workers,dtype=torch.bool,device=self.device)
        for _,ids,_ in groups: self.active[ids]=False
    def act(self,pool,obs,frozen_state,chosen,steppers=None):
        """Replace historical agents' rows of `chosen` in place, advance their
        frozen recurrent state in place, and return the learner-active mask.
        `steppers` maps version -> Stepper of batch `capacity` (graph mode)."""
        for version,ids,padded in self.groups:
            if steppers is None:
                opponent_logits,_,next_state=pool[version].step(obs.index_select(0,ids),frozen_state.index_select(1,ids))
                opponent_actions,_,_=distribution(opponent_logits)
            else:
                k=len(ids)
                opponent_logits,_,next_state=steppers[version](obs.index_select(0,padded),frozen_state.index_select(1,padded))
                opponent_actions,_,_=distribution(opponent_logits[:k]); next_state=next_state[:,:k]
            chosen.index_copy_(0,ids,opponent_actions); frozen_state.index_copy_(1,ids,next_state)
        return self.active
