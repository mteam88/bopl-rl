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
slice. With CUDA-graph steppers (fixed batch) each version's rows are padded to the
group's historical agent count by repeating real rows. A stepper captured with
sample='actions' draws the real rows' actions inside the graph, bit-identically
to eager sampling of the unpadded rows (see inference.py); a forward-only
stepper is sampled eagerly on its real rows.
"""
import os
import torch
from policy import distribution,sample

# BOPL_OPPONENT_SAMPLING selects how historical opponents' actions are drawn
# (all three give bit-identical draws from the same generator state):
#   graph        sampled inside each historical stepper's CUDA graph (default)
#   slim         eager policy.sample on the real rows (no Distribution objects)
#   distribution eager Categorical-based sampling (the original loop)
OPPONENT_SAMPLING=os.environ.get('BOPL_OPPONENT_SAMPLING','graph')
if OPPONENT_SAMPLING not in ('graph','slim','distribution'): raise ValueError(f'BOPL_OPPONENT_SAMPLING={OPPONENT_SAMPLING}')
SAMPLE=distribution if OPPONENT_SAMPLING=='distribution' else lambda logits:(sample(logits),None,None)
# BOPL_OPPONENT_GATHER=direct: index_select straight into each historical
# stepper's static graph inputs (default 'copy': gather to a temporary, then copy).
GATHER_DIRECT=os.environ.get('BOPL_OPPONENT_GATHER','copy')=='direct'
# BOPL_OPPONENT_STREAMS=1: replay each historical stepper on its own CUDA stream
# (forked from and joined back to the trainer stream) so the versions' small
# graphs run concurrently on the GPU. CPU issue order, and with it every
# generator offset a replay consumes, is unchanged, so the draws are identical.
OPPONENT_STREAMS=os.environ.get('BOPL_OPPONENT_STREAMS')=='1'

class Opponents:
    def __init__(self,workers,device,first=0):
        self.workers=workers; self.device=device; self.first=first; self.key=None
        self.groups=[]; self.lists=[]; self.active=torch.ones(2*workers,dtype=torch.bool,device=device)
        self.capacity=sum(1 for w in range(first,first+workers) if w%2==1)
        self.streams=[]
    def refresh(self,versions,historical,side):
        """Rebuild cached indices if any of this group's assignments changed.
        Arguments are the global per-worker lists."""
        span=range(self.first,self.first+self.workers)
        key=(tuple(versions[w] for w in span),tuple(side[w] for w in span),tuple(historical[w] for w in span))
        if key==self.key: return
        self.key=key; groups=[]; lists=[]
        for version in set(versions[w] for w in span):
            ids=[2*(w-self.first)+side[w] for w in span if historical[w] and versions[w]==version]
            if ids:
                padded=(ids*self.capacity)[:self.capacity]
                lists.append((version,ids,padded))
                groups.append((version,torch.tensor(ids,dtype=torch.long,device=self.device),
                               torch.tensor(padded,dtype=torch.long,device=self.device)))
        self.groups=groups; self.lists=lists   # lists: the same assignments as host lists (turn.py)
        self.active=torch.ones(2*self.workers,dtype=torch.bool,device=self.device)
        for _,ids,_ in groups: self.active[ids]=False
    def act(self,pool,obs,frozen_state,chosen,steppers=None):
        """Replace historical agents' rows of `chosen` in place, advance their
        frozen recurrent state in place, and return the learner-active mask.
        `steppers` maps version -> Stepper of batch `capacity` (graph mode)."""
        if OPPONENT_STREAMS and steppers is not None and self.groups:
            main=torch.cuda.current_stream()
            while len(self.streams)<len(self.groups): self.streams.append(torch.cuda.Stream())
            pending=[]
            for (version,ids,padded),stream in zip(self.groups,self.streams):
                k=len(ids); stepper=steppers[version]
                stream.wait_stream(main)
                with torch.cuda.stream(stream):
                    out=stepper(obs.index_select(0,padded),frozen_state.index_select(1,padded))
                    if stepper.sample=='actions':
                        opponent_actions,_,next_state=out
                        opponent_actions=opponent_actions[:k]; next_state=next_state[:,:k]
                    else:
                        opponent_logits,_,next_state=out
                        opponent_actions,_,_=SAMPLE(opponent_logits[:k]); next_state=next_state[:,:k]
                pending.append((ids,opponent_actions,next_state,stream))
            for ids,opponent_actions,next_state,stream in pending:
                main.wait_stream(stream)
                chosen.index_copy_(0,ids,opponent_actions); frozen_state.index_copy_(1,ids,next_state)
            return self.active
        for version,ids,padded in self.groups:
            if steppers is None:
                opponent_logits,_,next_state=pool[version].step(obs.index_select(0,ids),frozen_state.index_select(1,ids))
                opponent_actions,_,_=SAMPLE(opponent_logits)
            else:
                k=len(ids); stepper=steppers[version]
                if stepper.graph is not None and GATHER_DIRECT:
                    # Gather straight into the graph's static inputs (one launch
                    # each instead of index_select + copy_). Same values.
                    torch.index_select(obs,0,padded,out=stepper.obs)
                    torch.index_select(frozen_state,1,padded,out=stepper.state)
                    stepper.graph.replay(); out=stepper.out
                else:
                    out=stepper(obs.index_select(0,padded),frozen_state.index_select(1,padded))
                if stepper.sample=='actions':
                    opponent_actions,_,next_state=out
                    opponent_actions=opponent_actions[:k]; next_state=next_state[:,:k]
                else:
                    opponent_logits,_,next_state=out
                    opponent_actions,_,_=SAMPLE(opponent_logits[:k]); next_state=next_state[:,:k]
            chosen.index_copy_(0,ids,opponent_actions); frozen_state.index_copy_(1,ids,next_state)
        return self.active
