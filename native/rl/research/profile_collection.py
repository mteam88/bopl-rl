"""Profile the recurrent PPO collection path without changing train.py.

Replicates the rollout loop of research/train.py (same Vector, Policy,
distribution and historical-opponent bookkeeping) and reports where the
wall time of one vectorised step goes. Modes:

  env    random GPU actions -> vec.step only (game + IPC + packing + copies)
  infer  policy inference + sampling + opponent path only, on a fixed batch
  loop   the full replicated rollout loop; --sync adds CUDA syncs at section
         boundaries so the breakdown attributes GPU time to its section

Also samples system-wide CPU busy time and per-process CPU time for the
Python process and its game children so CPU-bound versus latency-bound
behaviour can be distinguished. Use a new --output directory per run.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[3]

def cpu_snapshot(pid):
    with open('/proc/stat') as f: fields=f.readline().split()
    total=sum(int(x) for x in fields[1:]); idle=int(fields[4])+int(fields[5])
    procs={}
    for child in [pid,*children(pid)]:
        try:
            with open(f'/proc/{child}/stat') as f: parts=f.read().rsplit(')',1)[1].split()
            procs[child]=int(parts[11])+int(parts[12])
        except FileNotFoundError: pass
    return dict(total=total,idle=idle,procs=procs)

def children(pid):
    out=[]
    for task in Path(f'/proc/{pid}/task').iterdir():
        try: out+=[int(x) for x in (task/'children').read_text().split()]
        except FileNotFoundError: pass
    return out

def cpu_report(before,after,seconds,workers):
    hz=os.sysconf('SC_CLK_TCK'); ncpu=os.cpu_count()
    busy=((after['total']-after['idle'])-(before['total']-before['idle']))/hz
    own=(after['procs'].get(os.getpid(),0)-before['procs'].get(os.getpid(),0))/hz
    games=[(after['procs'][p]-before['procs'].get(p,0))/hz for p in after['procs'] if p!=os.getpid()]
    return dict(system_busy_cores=busy/seconds,cpu_count=ncpu,python_cores=own/seconds,
                game_total_cores=sum(games)/seconds,game_processes=len(games),
                game_cores_each_mean=sum(games)/max(1,len(games))/seconds,
                game_cores_each_max=max(games,default=0)/seconds,workers=workers)

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('env','infer','loop'),default='loop')
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--warmup',type=int,default=64)
    parser.add_argument('--seed',type=int,default=73)
    parser.add_argument('--sync',action='store_true',help='synchronise CUDA at section boundaries (attribution mode)')
    parser.add_argument('--pool',type=int,default=4,help='historical versions to simulate in the opponent pool')
    parser.add_argument('--torch-threads',type=int,default=None)
    parser.add_argument('--label',default='')
    parser.add_argument('--baseline',action='store_true',help='use the original per-step Python opponent loop instead of selfplay.Opponents')
    opt=parser.parse_args()
    out=opt.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    os.environ['OMP_NUM_THREADS']=str(opt.workers)
    os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
    sys.path.insert(0,str(Path(__file__).parent))
    from environment import configure
    configure(out/'workers')
    import numpy as np
    import torch
    from policy import Policy,distribution
    from vector import Vector
    from selfplay import Opponents
    torch.set_num_threads(opt.torch_threads if opt.torch_threads else opt.workers)
    torch.manual_seed(opt.seed); np.random.seed(opt.seed)
    torch.set_float32_matmul_precision('high'); torch.backends.cudnn.benchmark=True
    device='cuda'
    t0=time.monotonic()
    vec=Vector(opt.workers,opt.seed,3600)
    startup=time.monotonic()-t0
    n=vec.size; policy=Policy(vec.obs_size).to(device); h=policy.hidden
    import copy
    def frozen_copy():
        frozen=copy.deepcopy(policy).requires_grad_(False); frozen.memory.flatten_parameters(); return frozen
    pool={v:frozen_copy() for v in range(opt.pool)}
    recent=list(pool)
    random=np.random.default_rng(opt.seed)
    versions=[int(random.choice(recent)) for _ in range(opt.workers)]
    historical=[w%2==1 for w in range(opt.workers)]; side=[w%2 for w in range(opt.workers)]
    state=policy.initial(n,device); frozen_state=policy.initial(n,device)
    opponents=Opponents(opt.workers,device)
    reset_before=torch.ones(n,dtype=torch.bool,device=device)
    sections={}
    def mark(name,start):
        if opt.sync: torch.cuda.synchronize()
        now=time.perf_counter(); sections.setdefault(name,[]).append(now-start); return now
    steps_done=0; matches=0
    def random_actions():
        return torch.stack([torch.randint(0,256,(n,),device=device),torch.randint(0,16,(n,),device=device)],dim=-1)
    fixed_obs=vec.obs.clone()
    def one_step(t,record):
        nonlocal state,frozen_state,reset_before,matches
        start=time.perf_counter()
        if opt.mode=='env':
            chosen=random_actions()
            start=mark('sample',start)
            vec.step(chosen); start=mark('env_step',start)
            reset_before=vec.dones.bool().clone()
            ended=vec.cpu_dones.numpy().reshape(-1,2).all(axis=1); matches+=int(ended.sum())
            mark('bookkeeping',start); return
        obs_in=fixed_obs if opt.mode=='infer' else vec.obs
        with torch.no_grad():
            logits,value,state=policy.step(obs_in,state); start=mark('policy_step',start)
            chosen,logp,_=distribution(logits); start=mark('distribution',start)
            if opt.baseline:
                active=torch.ones(n,dtype=torch.bool,device=device)
                for version in set(versions):
                    ids=[2*w+side[w] for w in range(opt.workers) if historical[w] and versions[w]==version]
                    if not ids: continue
                    opponent_logits,_,next_state=pool[version].step(obs_in[ids],frozen_state[:,ids])
                    opponent_actions,_,_=distribution(opponent_logits)
                    chosen[ids]=opponent_actions; frozen_state[:,ids]=next_state
                    active[ids]=False
            else:
                opponents.refresh(versions,historical,side)
                active=opponents.act(pool,obs_in,frozen_state,chosen)
            start=mark('opponents',start)
            if opt.mode=='infer':
                mark('total',start); return
            vec.step(chosen); start=mark('env_step',start)
            reset_before=vec.dones.bool().clone()
            state*=~reset_before.view(1,-1,1); frozen_state*=~reset_before.view(1,-1,1)
            start=mark('post_step_gpu',start)
        ended=vec.cpu_dones.numpy().reshape(-1,2).all(axis=1)
        for w in np.flatnonzero(ended):
            matches+=1; versions[w]=int(random.choice(recent)); side[w]=int(random.integers(2))
        mark('bookkeeping',start)
    for t in range(opt.warmup): one_step(t,False)
    torch.cuda.synchronize(); sections.clear(); matches=0
    before=cpu_snapshot(os.getpid()); wall0=time.perf_counter()
    for t in range(opt.steps): one_step(t,True)
    torch.cuda.synchronize(); wall=time.perf_counter()-wall0
    after=cpu_snapshot(os.getpid())
    report=dict(label=opt.label,mode=opt.mode,baseline=opt.baseline,workers=opt.workers,agents=n,steps=opt.steps,sync=opt.sync,pool=opt.pool,
                torch_threads=torch.get_num_threads(),startup_seconds=startup,wall_seconds=wall,
                ms_per_step=1000*wall/opt.steps,decisions_per_second=n*opt.steps/wall,matches=matches,
                sections={k:dict(mean_ms=1000*statistics.fmean(v),median_ms=1000*statistics.median(v),
                                 p95_ms=1000*sorted(v)[int(.95*len(v))-1],share=sum(v)/wall) for k,v in sections.items()},
                cpu=cpu_report(before,after,wall,opt.workers),env_log=vec.log())
    (out/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=1))
    vec.close()

if __name__=='__main__': main()
