"""Exactness check for BOPL_SPARE_GAME (a second, pre-reset game process per
environment). Steps the same action plan through the vector with the spare
off and on, over enough decisions to cover many autoresets, and requires
identical observations, rewards, terminals, telemetry and episode logs.
Also reports the env-step time of both runs (the spare's purpose)."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))

def run(workers,decisions,limit,groups,spare,maps):
    from environment import configure
    import torch
    from vector import Vector
    if spare: os.environ['BOPL_SPARE_GAME']='1'
    else: os.environ.pop('BOPL_SPARE_GAME',None)
    rng=np.random.default_rng(1)
    plan=[np.stack([rng.integers(256,size=2*workers),rng.integers(16,size=2*workers)],axis=-1).astype(np.float32) for _ in range(decisions)]
    started=time.monotonic()
    vec=Vector(workers,73,limit,groups=groups,training_maps=maps)
    setup=time.monotonic()-started
    assert vec.step_mode()=='sampled'
    obs=[];rew=[];don=[];tel=[];resets=0;step_seconds=0.0
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
    games=int(os.popen('ps -eo args | grep -c "[B]oplBattle.x86_64"').read())
    vec.close()
    games_after=int(os.popen('ps -eo args | grep -c "[B]oplBattle.x86_64"').read())
    return dict(obs=torch.stack(obs),rewards=torch.stack(rew),dones=torch.stack(don),telemetry=np.stack(tel)),log,dict(setup_seconds=setup,step_seconds=step_seconds,games=games,games_after=games_after)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--decisions',type=int,default=600); p.add_argument('--limit',type=int,default=300)
    p.add_argument('--groups',type=int,default=1); p.add_argument('--maps',default='0,1,2')
    opt=p.parse_args(); out=opt.output; out.mkdir(parents=True,exist_ok=True)
    from environment import configure
    configure(out/'workers')
    maps=[int(m) for m in opt.maps.split(',')]
    import torch
    report=dict(workers=opt.workers,decisions=opt.decisions,limit=opt.limit,groups=opt.groups,maps=maps,cases={})
    ref,ref_log,ref_info=run(opt.workers,opt.decisions,opt.limit,opt.groups,False,maps)
    report['cases']['base']=dict(log=ref_log,**ref_info)
    got,log,info=run(opt.workers,opt.decisions,opt.limit,opt.groups,True,maps)
    report['cases']['spare']=dict(log=log,**info)
    assert info['games']==2*opt.workers, f'expected {2*opt.workers} game processes with spares, saw {info["games"]}'
    assert info['games_after']==0, 'spare processes were not closed'
    assert ref_info['games']==opt.workers
    for k in ('obs','rewards','dones'):
        if not torch.equal(got[k],ref[k]):
            bad=(got[k]!=ref[k]).nonzero()[:5].tolist(); raise AssertionError(f'{k} differs at {bad}')
    if not np.array_equal(got['telemetry'],ref['telemetry']): raise AssertionError('telemetry differs')
    for k in ref_log:
        if k in ('sps','environment_ticks'): continue
        if k=='step_mode': assert log[k]=='sampled' and ref_log[k]=='sampled'
        elif abs(float(log.get(k,0))-float(ref_log[k]))>1e-6: raise AssertionError(f'log {k} differs: {log.get(k)} vs {ref_log[k]}')
    matches=float(ref_log.get('n',0))/2
    assert matches>=2*opt.workers, f'only {matches} matches; the check must cover autoresets'
    report['matches']=matches; report['status']='exact'
    (out/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(dict(status='exact',matches=matches,base=ref_info,spare=info)))
if __name__=='__main__': main()
