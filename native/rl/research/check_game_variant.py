"""Exactness check for a game/runtime variant selected through environment
variables (for example BOPL_RL_GAME=<other build> or a Unity/Mono knob).
Steps the same random action plan through the vector under the incumbent
environment and under the variant, over enough decisions to cover many
autoresets, and requires identical observations, rewards, terminals,
telemetry and episode logs. Reports the env-step time of both runs."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))

def own_games():
    """Game processes spawned by this process (other services may run their own games)."""
    me=str(os.getpid())
    return sum(1 for l in os.popen('ps -eo ppid,args').read().splitlines()[1:] if l.split()[0]==me and 'BoplBattle.x86_64' in l)

def run(workers,decisions,limit,groups,maps,env):
    import torch
    from vector import Vector
    saved={k:os.environ.get(k) for k in env}
    for k,v in env.items():
        if v is None: os.environ.pop(k,None)
        else: os.environ[k]=v
    try:
        rng=np.random.default_rng(1)
        plan=[np.stack([rng.integers(256,size=2*workers),rng.integers(16,size=2*workers)],axis=-1).astype(np.float32) for _ in range(decisions)]
        started=time.monotonic()
        vec=Vector(workers,73,limit,groups=groups,training_maps=maps)
        setup=time.monotonic()-started
        assert vec.step_mode()=='sampled'
        obs=[];rew=[];don=[];tel=[];step_seconds=0.0
        for t in range(decisions):
            actions=torch.as_tensor(plan[t],device='cuda')
            t0=time.monotonic()
            if groups==1: vec.step(actions)
            else:
                for g in range(groups): vec.step_group(g,actions[vec.groups[g]])
                for g in range(groups): vec.wait_group(g)
            torch.cuda.synchronize(); step_seconds+=time.monotonic()-t0
            obs.append(vec.obs.clone());rew.append(vec.rewards.clone());don.append(vec.dones.clone());tel.append(vec.telemetry.copy())
        log=vec.log()
        games=own_games()
        vec.close()
        # bopl_rl_close reaps each child, but count with a short grace period anyway
        # and record how long the processes lingered.
        closed=time.monotonic()
        while True:
            games_after=own_games()
            if games_after==0 or time.monotonic()-closed>15: break
            time.sleep(0.5)
        linger=time.monotonic()-closed
    finally:
        for k,v in saved.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k]=v
    return dict(obs=torch.stack(obs),rewards=torch.stack(rew),dones=torch.stack(don),telemetry=np.stack(tel)),log,dict(setup_seconds=setup,step_seconds=step_seconds,games=games,games_after=games_after,linger_seconds=linger,env=env)

def parse(items):
    env={}
    for item in items or []:
        k,_,v=item.partition('='); env[k]=v if v!='' else None
    return env

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--decisions',type=int,default=600); p.add_argument('--limit',type=int,default=300)
    p.add_argument('--groups',type=int,default=1); p.add_argument('--maps',default='0,1,2')
    p.add_argument('--base',action='append',help='KEY=VALUE applied to the incumbent run (KEY= unsets)')
    p.add_argument('--variant',action='append',required=True,help='KEY=VALUE applied to the variant run (KEY= unsets)')
    opt=p.parse_args(); out=opt.output; out.mkdir(parents=True,exist_ok=True)
    from environment import configure
    configure(out/'workers')
    maps=[int(m) for m in opt.maps.split(',')]
    import torch
    base=parse(opt.base); variant=parse(opt.variant)
    report=dict(workers=opt.workers,decisions=opt.decisions,limit=opt.limit,groups=opt.groups,maps=maps,base=base,variant=variant,cases={})
    ref,ref_log,ref_info=run(opt.workers,opt.decisions,opt.limit,opt.groups,maps,base)
    report['cases']['base']=dict(log=ref_log,**ref_info)
    got,log,info=run(opt.workers,opt.decisions,opt.limit,opt.groups,maps,variant)
    report['cases']['variant']=dict(log=log,**info)
    (out/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    for k in ('obs','rewards','dones'):
        if not torch.equal(got[k],ref[k]):
            bad=(got[k]!=ref[k]).nonzero()[:5].tolist(); raise AssertionError(f'{k} differs at {bad}')
    if not np.array_equal(got['telemetry'],ref['telemetry']): raise AssertionError('telemetry differs')
    for k in ref_log:
        if k in ('sps','environment_ticks'): continue
        if k=='step_mode': assert log[k]=='sampled' and ref_log[k]=='sampled'
        elif abs(float(log.get(k,0))-float(ref_log[k]))>1e-6: raise AssertionError(f'log {k} differs: {log.get(k)} vs {ref_log[k]}')
    assert info['games']==ref_info['games'], f'game process count differs: {ref_info["games"]} vs {info["games"]}'
    assert info['games_after']==0 and ref_info['games_after']==0, f'game processes were not closed: {ref_info["games_after"]} / {info["games_after"]}'
    matches=float(ref_log.get('n',0))/2
    assert matches>=2*opt.workers, f'only {matches} matches; the check must cover autoresets'
    report['matches']=matches; report['status']='exact'
    (out/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(dict(status='exact',matches=matches,base=ref_info,variant=info)))
if __name__=='__main__': main()
