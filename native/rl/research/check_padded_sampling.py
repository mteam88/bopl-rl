"""Historical opponents sampled inside a padded CUDA graph must draw exactly
what eager sampling of the unpadded rows draws.

torch's one-sample multinomial (Categorical.sample / policy.sample) draws
element i of the flattened probability tensor from Philox subsequence i and
advances the default CUDA generator by a fixed 4 counters per call whatever
the batch size, so rows are independent of how many rows follow them. This
check pins that down empirically for the shapes train.py uses: for every real
row count k <= capacity, a Stepper(sample='actions') captured at `capacity`
rows and replayed on selfplay's padded batch must give the real rows' actions,
the real rows' recurrent state and the same generator state afterwards as the
incumbent design (forward-only padded graph, real rows sampled eagerly with
policy.sample, which is checked against Categorical), over a sequence of steps
from one generator state. It also checks Opponents.act with graph steppers
against Opponents.act with forward-only steppers (the previous design).
"""
import argparse
import sys
import torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from policy import Policy,distribution,sample
from inference import Stepper
from selfplay import Opponents

def padded_ids(ids,capacity): return (ids*capacity)[:capacity]

def check_stepper(policy,capacity,k,steps,device,seed):
    """Reference: the incumbent design, a forward-only padded graph whose real
    rows are sampled eagerly (policy.sample, itself checked against Categorical).
    Candidate: the same padded graph with sampling captured inside it."""
    torch.manual_seed(seed)
    obs=[torch.randn(k,policy.obs_size,device=device) for _ in range(steps)]
    state0=torch.randn(1,k,policy.hidden,device=device)*.5
    ids=list(range(k)); pad=torch.tensor(padded_ids(ids,capacity),device=device)
    start=torch.cuda.get_rng_state()
    forward=Stepper(policy,capacity,device,graph=True,sample=False)
    reference=[]; state=state0.clone()
    for t in range(steps):
        logits,_,next_state=forward(obs[t].index_select(0,pad),state.index_select(1,pad))
        before=torch.cuda.get_rng_state()
        a=sample(logits[:k])
        after=torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(before)
        d,_,_=distribution(logits[:k])
        if not torch.equal(a,d): raise AssertionError(f'policy.sample differs from Categorical: capacity={capacity} k={k} t={t}')
        if not torch.equal(torch.cuda.get_rng_state(),after): raise AssertionError('generator offset differs between policy.sample and Categorical')
        state=next_state[:,:k].clone(); reference.append((a.clone(),state.clone()))
    reference_end=torch.cuda.get_rng_state()
    forward.close()
    torch.cuda.set_rng_state(start)
    stepper=Stepper(policy,capacity,device,graph=True,sample='actions')
    if not torch.equal(torch.cuda.get_rng_state(),start): raise AssertionError('graph capture changed the generator state')
    state=state0.clone()
    for t in range(steps):
        a,_,next_state=stepper(obs[t].index_select(0,pad),state.index_select(1,pad))
        a=a[:k].clone(); state=next_state[:,:k].clone()
        ra,rs=reference[t]
        if not torch.equal(a,ra): raise AssertionError(f'in-graph actions differ: capacity={capacity} k={k} t={t}\n{a.T}\n{ra.T}')
        if not torch.equal(state,rs): raise AssertionError(f'in-graph state differs: capacity={capacity} k={k} t={t}')
    if not torch.equal(torch.cuda.get_rng_state(),reference_end): raise AssertionError(f'generator offset differs after {steps} in-graph replays: capacity={capacity} k={k}')
    stepper.close()

def check_opponents(policy,workers,steps,device,seed):
    """Two Opponents instances over the same assignments: graph-sampled steppers
    versus forward-only steppers sampled eagerly (the previous design)."""
    torch.manual_seed(seed)
    first=workers  # second pipeline group, as in train.py
    versions=[0]*(2*workers); historical=[w%2==1 for w in range(2*workers)]; side=[w%2 for w in range(2*workers)]
    rng=torch.Generator().manual_seed(seed)
    pool={v:policy for v in (0,1,2)}
    for w in range(first,first+workers):
        versions[w]=int(torch.randint(3,(1,),generator=rng)); side[w]=int(torch.randint(2,(1,),generator=rng))
    n=2*workers
    obs=[torch.randn(n,policy.obs_size,device=device) for _ in range(steps)]
    chosen0=torch.randint(256,(n,2),device=device)
    frozen0=torch.randn(1,n,policy.hidden,device=device)*.5
    results=[]
    for mode in ('actions',False):
        opp=Opponents(workers,device,first=first)
        steppers={v:Stepper(policy,opp.capacity,device,graph=True,sample=mode) for v in pool}
        torch.manual_seed(seed+1)
        chosen_log=[]; frozen=frozen0.clone()
        for t in range(steps):
            chosen=chosen0.clone()
            opp.refresh(versions,historical,side)
            active=opp.act(pool,obs[t],frozen,chosen,steppers)
            chosen_log.append((chosen.clone(),frozen.clone(),active.clone()))
        results.append((chosen_log,torch.cuda.get_rng_state()))
        for s in steppers.values(): s.close()
    (a,ra),(b,rb)=results
    for t,((ca,fa,ma),(cb,fb,mb)) in enumerate(zip(a,b)):
        if not (torch.equal(ca,cb) and torch.equal(fa,fb) and torch.equal(ma,mb)): raise AssertionError(f'Opponents.act differs between graph-sampled and forward-only steppers at t={t}')
    if not torch.equal(ra,rb): raise AssertionError('generator state differs after Opponents.act sequences')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obs-size',type=int,default=3257)
    parser.add_argument('--steps',type=int,default=6)
    parser.add_argument('--capacities',default='4,8,16,64')
    parser.add_argument('--seed',type=int,default=7)
    opt=parser.parse_args()
    device='cuda'
    torch.manual_seed(opt.seed)
    policy=Policy(opt.obs_size).to(device).requires_grad_(False)
    policy.memory.flatten_parameters()
    cases=0
    for capacity in [int(x) for x in opt.capacities.split(',')]:
        ks=range(1,capacity+1) if capacity<=8 else sorted({1,2,3,capacity//2,capacity-1,capacity})
        for k in ks:
            for rep in range(2):
                check_stepper(policy,capacity,k,opt.steps,device,opt.seed+100*capacity+10*k+rep); cases+=1
    for workers in (4,8):
        check_opponents(policy,workers,opt.steps,device,opt.seed+workers); cases+=1
    print(f'padded in-graph sampling is bit-identical to eager sampling of the real rows: {cases} cases, {opt.steps} steps each')

if __name__=='__main__': main()
