"""One CUDA graph per pipeline group and historical-version order for a whole
inference turn (BOPL_TURN_GRAPH=1).

The original turn issues, per group, the learner's graph replay (two input
copies, one launch), then for each historical version present in the group two
gathers, a graph replay, and two index copies, all from Python. Measured with
BOPL_TRACE_TURN=1 that is ~0.35 ms of main-thread time per group turn, during
which the other group's games have finished and are waiting. This module
captures the identical sequence of kernels once per (group, tuple of versions
in the order Opponents.refresh visits them) and replays it as a single graph.

Exactness. The graph reads the vector's observation slice and the trainer's
recurrent-state slices at their fixed addresses, so no input copies are
needed; the learner's `policy.step` + `distribution` and each version's
`policy.step` + `sample` are issued in the original order, so the default CUDA
generator is consumed identically (replay advances it by the captured Philox
offsets, as inference.Stepper does). Historical rows are padded to the group's
historical capacity exactly as selfplay.Opponents does, and scattered back with
a static gather+where instead of index_copy_ on a variable-length list: the
same values land in the same rows. Index buffers are static and refilled when
assignments change; only a change in the *order of versions* needs a new
capture, so a handful of graphs per pool composition covers a run. Captures use
capture_error_mode='thread_local' because the other group's C thread issues
its own CUDA calls while this one is captured.
"""
import time
import torch
from policy import distribution
from selfplay import SAMPLE
from inference import ROLLOUT_ENTROPY

class TurnGraph:
    def __init__(self,policy,pool,opponents,obs,state,frozen_state,device,slots=4):
        """obs: the vector's observation rows of this group (fixed device address);
        state/frozen_state: the trainer's recurrent-state slices for this group."""
        self.policy=policy; self.pool=pool; self.opponents=opponents
        self.obs=obs; self.state=state; self.frozen=frozen_state; self.device=device
        n=obs.shape[0]; cap=max(1,opponents.capacity); self.slots=slots
        self.padded=torch.zeros(slots,cap,dtype=torch.long,device=device)
        self.source=torch.zeros(slots,n,dtype=torch.long,device=device)
        self.mask=torch.zeros(slots,n,dtype=torch.bool,device=device)
        self.graphs={}; self.key=None; self.current=None
        self.captures=0; self.capture_seconds=0.
    def refresh(self):
        """Call after opponents.refresh: refill the index buffers if the group's
        assignments changed and select (capturing if new) the graph for the
        current order of versions."""
        opp=self.opponents
        if opp.key==self.key: return
        self.key=opp.key
        if len(opp.lists)>self.slots: raise RuntimeError('More historical versions than turn-graph slots')
        n=self.obs.shape[0]; cap=self.padded.shape[1]
        padded=torch.zeros(self.slots,cap,dtype=torch.long); source=torch.zeros(self.slots,n,dtype=torch.long)
        mask=torch.zeros(self.slots,n,dtype=torch.bool)
        for i,(version,ids,pad) in enumerate(opp.lists):
            padded[i,:len(pad)]=torch.tensor(pad)
            for j,row in enumerate(ids): source[i,row]=j; mask[i,row]=True
        self.padded.copy_(padded); self.source.copy_(source); self.mask.copy_(mask)
        versions=tuple(version for version,_,_ in opp.lists)
        if versions not in self.graphs: self.graphs[versions]=self.capture(versions)
        self.current=self.graphs[versions]
    def _run(self,versions):
        logits,value,next_state=self.policy.step(self.obs,self.state)
        chosen,logp,_=distribution(logits,entropy=ROLLOUT_ENTROPY)
        self.state.copy_(next_state)
        for i,version in enumerate(versions):
            padded=self.padded[i]
            opponent_logits,_,next_frozen=self.pool[version].step(self.obs.index_select(0,padded),self.frozen.index_select(1,padded))
            opponent_actions,_,_=SAMPLE(opponent_logits)
            source=self.source[i]; mask=self.mask[i]
            chosen=torch.where(mask.unsqueeze(-1),opponent_actions.index_select(0,source),chosen)
            self.frozen.copy_(torch.where(mask.view(1,-1,1),next_frozen.index_select(1,source),self.frozen))
        return chosen,logp,value
    def capture(self,versions):
        started=time.perf_counter()
        rng=torch.cuda.get_rng_state(); state=self.state.clone(); frozen=self.frozen.clone()
        main=torch.cuda.current_stream(); stream=torch.cuda.Stream(); stream.wait_stream(main)
        with torch.cuda.stream(stream), torch.no_grad():
            self._run(versions)   # warm-up: lazy library initialisation happens outside the capture
            graph=torch.cuda.CUDAGraph()
            graph.capture_begin(capture_error_mode='thread_local')
            try: out=self._run(versions)
            finally: graph.capture_end()
        main.wait_stream(stream); torch.cuda.synchronize()
        # Warm-up and capture consumed random numbers and advanced the state
        # slices; put everything back so the replayed turn starts where the
        # eager turn would have.
        torch.cuda.set_rng_state(rng); self.state.copy_(state); self.frozen.copy_(frozen)
        self.captures+=1; self.capture_seconds+=time.perf_counter()-started
        # The graph's kernels read these snapshots' parameters in place; keep them
        # alive until prune() drops the graph.
        return graph,out,[self.pool[v] for v in versions]
    def replay(self):
        graph,out,_=self.current
        graph.replay()
        return out
    def prune(self):
        """Drop graphs that reference a snapshot no longer in the pool."""
        for versions in [k for k in self.graphs if any(v not in self.pool for v in k)]:
            if self.current is self.graphs[versions]: self.current=None; self.key=None
            del self.graphs[versions]
    def stats(self):
        return dict(graphs=len(self.graphs),captures=self.captures,capture_seconds=self.capture_seconds)

class CollectGraph:
    """BOPL_COLLECT_GRAPH=1: the device part of train.py's collect() for one
    group as a single replayed graph: from the group's fresh terminal flags
    (fixed vector address) store the next reset flags and zero the learner and
    frozen recurrent states of agents whose episode ended. Same six kernels on
    the same buffers as the eager code, one launch; the rewards/terminal copies
    into the per-update rollout buffers stay eager (their destination moves)."""
    def __init__(self,dones,reset_before,state,frozen_state):
        self.dones=dones; self.reset_before=reset_before; self.state=state; self.frozen=frozen_state
        stream=torch.cuda.Stream(); main=torch.cuda.current_stream(); stream.wait_stream(main)
        saved=(dones.clone(),reset_before.clone(),state.clone(),frozen_state.clone())
        with torch.cuda.stream(stream), torch.no_grad():
            self._run()
            self.graph=torch.cuda.CUDAGraph()
            self.graph.capture_begin(capture_error_mode='thread_local')
            try: self._run()
            finally: self.graph.capture_end()
        main.wait_stream(stream); torch.cuda.synchronize()
        for buffer,value in zip((dones,reset_before,state,frozen_state),saved): buffer.copy_(value)
    def _run(self):
        ended_now=self.dones.bool().clone()
        self.reset_before.copy_(ended_now)
        keep_state=~ended_now.view(1,-1,1)
        self.state*=keep_state; self.frozen*=keep_state
    def replay(self): self.graph.replay()
