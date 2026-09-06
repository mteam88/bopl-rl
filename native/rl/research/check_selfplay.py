"""Bit-exact equivalence of selfplay.Opponents against train.py's original opponent loop."""
import argparse,copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import torch,numpy as np
from policy import Policy,distribution
from selfplay import Opponents

def original(pool,workers,versions,historical,side,obs,frozen_state,chosen):
    n=2*workers; active=torch.ones(n,dtype=torch.bool,device=obs.device)
    for version in set(versions):
        ids=[2*w+side[w] for w in range(workers) if historical[w] and versions[w]==version]
        if not ids: continue
        opponent_logits,_,next_state=pool[version].step(obs[ids],frozen_state[:,ids])
        opponent_actions,_,_=distribution(opponent_logits)
        chosen[ids]=opponent_actions; frozen_state[:,ids]=next_state
        active[ids]=False
    return active

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda'); parser.add_argument('--trials',type=int,default=40)
    parser.add_argument('--workers',type=int,default=16)
    opt=parser.parse_args(); device=opt.device; workers=opt.workers; n=2*workers; obs_size=3257
    torch.manual_seed(1); policy=Policy(obs_size).to(device)
    def frozen():
        f=copy.deepcopy(policy).requires_grad_(False)
        with torch.no_grad():
            for p in f.parameters(): p.add_(torch.randn_like(p)*.05)
        f.memory.flatten_parameters(); return f
    pool={v:frozen() for v in range(5)}
    random=np.random.default_rng(3); historical=[w%2==1 for w in range(workers)]
    opp=Opponents(workers,device); failures=0
    versions=[0]*workers; side=[0]*workers
    for trial in range(opt.trials):
        # Change a random subset of assignments, as ended matches do in train.py.
        for w in np.flatnonzero(random.random(workers)<.3): versions[w]=int(random.choice(5)); side[w]=int(random.integers(2))
        obs=torch.randn(n,obs_size,device=device); fs0=torch.randn(1,n,128,device=device)
        with torch.no_grad(): logits,_,_=policy.step(obs,policy.initial(n,device))
        torch.manual_seed(100+trial)
        with torch.no_grad():
            chosen_a,_,_=distribution(logits); fs_a=fs0.clone(); active_a=original(pool,workers,versions,historical,side,obs,fs_a,chosen_a)
        torch.manual_seed(100+trial)
        with torch.no_grad():
            chosen_b,_,_=distribution(logits); fs_b=fs0.clone(); opp.refresh(versions,historical,side); active_b=opp.act(pool,obs,fs_b,chosen_b)
        same=torch.equal(chosen_a,chosen_b) and torch.equal(fs_a,fs_b) and torch.equal(active_a,active_b)
        if not same:
            failures+=1; print('MISMATCH trial',trial,'actions',torch.equal(chosen_a,chosen_b),'state',torch.equal(fs_a,fs_b),'mask',torch.equal(active_a,active_b))
    print(f'{opt.device}: {opt.trials-failures}/{opt.trials} trials bit-identical (actions, frozen state, learner mask)')
    sys.exit(1 if failures else 0)
if __name__=='__main__': main()
