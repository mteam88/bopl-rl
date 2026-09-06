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
check_pipeline.py). Historical policies are captured forward-only (sample=False)
and sampled eagerly on their real rows, because their padded batch would
otherwise consume more draws than the eager path. Recurrent state flows through the static input/output
buffers and stays owned by the trainer.
"""
import torch
from policy import distribution

class Stepper:
    def __init__(self,policy,batch,device,graph=False,warmup=3,sample=True):
        """sample=False captures only the network forward (logits, value, state);
        the caller samples outside the graph. Historical policies use this so
        their padded fixed-shape batch draws no extra random numbers."""
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
        actions,logp,_=distribution(logits)
        return actions,logp,value,state
    def __call__(self,obs,state):
        if self.graph is None: return self._run(obs,state)
        self.obs.copy_(obs); self.state.copy_(state)
        self.graph.replay()
        return self.out
    def close(self):
        # Dropping the graph and its static tensors returns the graph's private memory pool.
        self.graph=None; self.out=None; self.obs=None; self.state=None
