"""Fixed-shape policy stepping for rollouts, eager or as a replayed CUDA graph.

A Stepper runs `policy.step` followed by action sampling for one batch shape.
In eager mode it issues exactly the calls train.py always issued, so results
are bit-identical to the original loop. In graph mode the same kernels are
captured once and replayed; inputs are copied into static buffers and the
returned tensors alias static outputs that the next replay overwrites, so
callers copy what they keep before stepping the same batch again.

Random numbers: warm-up and capture run under a saved RNG state that is
restored afterwards, and replay advances the default CUDA generator by the
same Philox offset the eager kernels use, so a graphed step draws the same
samples as the eager step from the same generator state (checked by
check_pipeline.py). Historical policies (sample='actions') sample inside the
graph on their padded batch: torch's one-sample multinomial draws element i
from Philox subsequence i (first component) and advances the generator by a
fixed 4 counters whatever the batch size, so the real rows' draws and the
generator offset match eager sampling of the unpadded rows exactly (checked by
check_padded_sampling.py). sample=False captures the forward only, for the
eager fallback. Recurrent state flows through the static input/output buffers
and stays owned by the trainer.
"""
import os
import torch
from policy import distribution,sample

# BOPL_ROLLOUT_ENTROPY=0: the learner stepper does not compute the entropy term
# (its output is discarded during collection; actions/log-probabilities unchanged).
ROLLOUT_ENTROPY=os.environ.get('BOPL_ROLLOUT_ENTROPY','1')!='0'

class Stepper:
    def __init__(self,policy,batch,device,graph=False,warmup=3,sample=True):
        """sample=True: learner step (actions, log-probabilities, value, state).
        sample='actions': actions only, drawn with policy.sample (historical
        opponents; returns actions, value, state). sample=False: the network
        forward only (logits, value, state); the caller samples."""
        if sample not in (True,False,'actions'): raise ValueError(sample)
        self.policy=policy; self.batch=batch; self.graph=None; self.sample=sample
        if not graph: return
        self.obs=torch.zeros(batch,policy.obs_size,device=device)
        self.state=torch.zeros(1,batch,policy.hidden,device=device)
        rng=torch.cuda.get_rng_state()
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(warmup): self._run()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad(): self.out=self._run()
        torch.cuda.synchronize()
        torch.cuda.set_rng_state(rng)
    def _run(self,obs=None,state=None):
        logits,value,state=self.policy.step(self.obs if obs is None else obs,self.state if state is None else state)
        if not self.sample: return logits,value,state
        if self.sample=='actions': return sample(logits),value,state
        actions,logp,_=distribution(logits,entropy=ROLLOUT_ENTROPY)
        return actions,logp,value,state
    def __call__(self,obs,state):
        if self.graph is None: return self._run(obs,state)
        self.obs.copy_(obs); self.state.copy_(state)
        self.graph.replay()
        return self.out
    def close(self):
        # Dropping the graph and its static tensors returns the graph's private memory pool.
        self.graph=None; self.out=None; self.obs=None; self.state=None
