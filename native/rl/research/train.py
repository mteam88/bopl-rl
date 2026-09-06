"""Recurrent PPO research runner, using Bopl's pinned PufferLib C vector runner.

Current-vs-current self-play in half the games; current-vs-historical in the
other half. Historical actions are excluded from optimization. Episode resets,
rollout boundaries, rewards, and recurrent memory retain their explicit owners.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]

def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); temporary.replace(path)

def advantages(rewards,values,dones,last_value,gamma=.995,lam=.95):
    import torch
    result=torch.zeros_like(rewards); carry=torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        next_value=last_value if t==len(rewards)-1 else values[t+1]
        alive=1-dones[t]
        delta=rewards[t]+gamma*next_value*alive-values[t]
        carry=delta+gamma*lam*alive*carry
        result[t]=carry
    return result

def main():
    program_started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1_000_000)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=73)
    parser.add_argument('--horizon',type=int,default=128)
    parser.add_argument('--sequence',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--checkpoint-every',type=int,default=100_000)
    parser.add_argument('--eval-at',type=lambda value:[int(x) for x in value.split(',')],
                        help='Evaluate these decision milestones; does not change historical checkpoint cadence')
    parser.add_argument('--evaluation-games',type=int,default=8)
    parser.add_argument('--evaluation-heldout-games',type=int,default=4)
    parser.add_argument('--evaluation-workers',type=int,default=2)
    parser.add_argument('--evaluation-baseline',type=Path,
                        help='Frozen opponent shared by the experiment series; defaults to this run initial policy')
    parser.add_argument('--episode-ticks',type=int,default=3600)
    parser.add_argument('--no-eval',action='store_true')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--load',type=Path)
    opt=parser.parse_args()
    if min(opt.steps,opt.workers,opt.horizon,opt.sequence,opt.epochs,opt.checkpoint_every,opt.episode_ticks)<1 or opt.horizon%opt.sequence:
        parser.error('Positive counts and horizon divisible by sequence are required')
    if min(opt.evaluation_games,opt.evaluation_heldout_games,opt.evaluation_workers)<1:
        parser.error('Evaluation counts must be positive')
    if opt.eval_at is not None and (not opt.eval_at or any(x<=0 for x in opt.eval_at)):
        parser.error('Evaluation milestones must be positive')
    out=opt.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    os.environ['OMP_NUM_THREADS']=str(opt.workers)
    os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
    from environment import configure
    configure(out/'workers')
    import numpy as np
    import torch
    from policy import Policy,distribution
    from vector import Vector
    from telemetry import FIELDS,summarize
    torch.set_num_threads(opt.workers)
    torch.manual_seed(opt.seed); np.random.seed(opt.seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=True
    device='cuda'
    vec=Vector(opt.workers,opt.seed,opt.episode_ticks)
    policy=Policy(vec.obs_size).to(device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4,eps=1e-5)
    step=0; updates=0
    if opt.load:
        previous=torch.load(opt.load,map_location=device,weights_only=False)
        policy.load_state_dict(previous['policy']); optimizer.load_state_dict(previous['optimizer'])
        step=previous['step']; updates=previous['updates']
    config={**vars(opt),'output':str(out),'load':str(opt.load) if opt.load else None,
            'evaluation_baseline':str(opt.evaluation_baseline.resolve()) if opt.evaluation_baseline else None,
            'evaluation_baseline_sha256':hashlib.sha256(opt.evaluation_baseline.read_bytes()).hexdigest() if opt.evaluation_baseline else None,
            'schema':1,'obs_size':vec.obs_size,'hidden':policy.hidden,'device':device,
            'telemetry_schema':1,'telemetry_fields':FIELDS,'action_rates_scope':'learner decisions in this rollout',
            'startup_seconds':time.monotonic()-program_started,'torch_version':torch.__version__,
            'gpu':torch.cuda.get_device_name(),
            'algorithm':'recurrent PPO','gamma':.995,'gae_lambda':.95,'learning_rate':.0003,
            'action_hz':15,'observation_delay_ticks':9,'own_previous_actions':'immediate','loadout':['dash','arrow','platform'],
            'training_maps':[0],'eval_maps':[0,1,2],'stocks':3,'pool_size':4,
            'historical_game_fraction':.5,'reward':{'win':1,'loss':-1,'kill':.1,'death':-.1,'timeout':0},
            'timeout_semantics':'finite-horizon match; no winner bonus; terminal value zero',
            'puffer_revision':'42f70d6932c30ac977736f861006809c50168ba9',
            'assembly_sha256':hashlib.sha256((ROOT/'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll').read_bytes()).hexdigest()}
    sources={}
    for directory in (ROOT/'native/rl/puffer',Path(__file__).parent):
        for file in sorted(directory.rglob('*')):
            if file.is_file() and file.suffix in ('.py','.c','.h','.ini','.patch','.html','.css','.js','.lock'):
                relative=str(file.relative_to(ROOT)); sources[relative]=hashlib.sha256(file.read_bytes()).hexdigest()
    config['source_sha256']=sources
    from pufferlib import _C
    config['puffer_library_sha256']=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest()
    config['bootstrap_sha256']=hashlib.sha256((ROOT/'native/fixtures/movement.rep').read_bytes()).hexdigest()
    config['research_library_sha256']=hashlib.sha256((ROOT/'native/artifacts/rl/libbopl_research.so').read_bytes()).hexdigest()
    config['git_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    atomic_json(out/'config.json',config)
    atomic_json(out/'status.json',dict(status='starting',pid=os.getpid(),step=step,target_steps=opt.steps))
    (out/'checkpoints').mkdir(); (out/'evaluations').mkdir()
    stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    started=time.monotonic(); started_step=step; next_checkpoint=step+opt.checkpoint_every
    evaluation=None; pending_evaluation=None; evaluation_log=None
    def frozen_copy():
        frozen=copy.deepcopy(policy).requires_grad_(False)
        frozen.memory.flatten_parameters()
        return frozen
    pool={step:frozen_copy()}
    recent=[step]; versions=[step]*opt.workers
    historical=[w%2==1 for w in range(opt.workers)]
    side=[w%2 for w in range(opt.workers)]
    random=np.random.default_rng(opt.seed)
    n=vec.size; h=policy.hidden; tmax=opt.horizon; length=opt.sequence
    state=policy.initial(n,device); frozen_state=policy.initial(n,device)
    reset_before=torch.ones(n,dtype=torch.bool,device=device)
    completed_matches=0
    evaluation_milestones=sorted(set(opt.eval_at or []))
    metrics=(out/'metrics.jsonl').open('a',buffering=1)

    def checkpoint(label):
        path=out/'checkpoints'/f'{label}.pt'; tmp=path.with_suffix('.tmp')
        torch.save(dict(schema=1,policy=policy.state_dict(),optimizer=optimizer.state_dict(),
                        obs_size=vec.obs_size,hidden=h,step=step,updates=updates,config=config,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),tmp)
        tmp.replace(path)
        return path

    def launch_evaluation(path):
        nonlocal evaluation,evaluation_log
        evaluation_log=(out/'evaluation-worker.log').open('a')
        evaluation=subprocess.Popen([sys.executable,str(Path(__file__).with_name('evaluate.py')),
            '--run',str(out),'--checkpoint',str(path),'--initial',str(opt.evaluation_baseline or initial_path),
            '--games-per-opponent',str(opt.evaluation_games),'--heldout-games',str(opt.evaluation_heldout_games),
            '--workers',str(opt.evaluation_workers)],
            stdout=evaluation_log,stderr=subprocess.STDOUT,start_new_session=True)

    def service_evaluation(path=None):
        nonlocal evaluation,pending_evaluation,evaluation_log
        if opt.no_eval: return
        if path: pending_evaluation=path
        if evaluation and evaluation.poll() is not None:
            if evaluation.returncode: raise RuntimeError(f'Evaluation failed ({evaluation.returncode}); see evaluation-worker.log')
            evaluation=None; evaluation_log.close(); evaluation_log=None
        if evaluation is None and pending_evaluation:
            path=pending_evaluation; pending_evaluation=None; launch_evaluation(path)

    video_process=None; video_log=None
    if not opt.no_video and not opt.no_eval:
        video_log=(out/'video-worker.log').open('w')
        video_process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('watch_videos.py')),'--run',str(out)],
                                      stdout=video_log,stderr=subprocess.STDOUT)
    initial_path=checkpoint(f'policy-{step:09d}')
    service_evaluation(initial_path)
    status='running'; failure=None; result=None
    try:
        while step<opt.steps and not stopping:
            rollout_started=time.monotonic()
            obs=torch.empty(tmax,n,vec.obs_size,device=device)
            actions=torch.empty(tmax,n,2,dtype=torch.long,device=device)
            old_logp=torch.empty(tmax,n,device=device); values=torch.empty(tmax,n,device=device)
            rewards=torch.empty(tmax,n,device=device); dones=torch.empty(tmax,n,device=device)
            masks=torch.empty(tmax,n,dtype=torch.bool,device=device)
            resets=torch.empty(tmax,n,dtype=torch.bool,device=device)
            events=np.empty((tmax,n,len(FIELDS)),dtype=np.float32)
            memories=[]
            for t in range(tmax):
                if t%length==0: memories.append(state.detach().clone())
                obs[t].copy_(vec.obs); resets[t]=reset_before
                with torch.no_grad():
                    logits,value,state=policy.step(vec.obs,state)
                    chosen,logp,_=distribution(logits)
                    active=torch.ones(n,dtype=torch.bool,device=device)
                    for version in set(versions):
                        ids=[2*w+side[w] for w in range(opt.workers) if historical[w] and versions[w]==version]
                        if not ids: continue
                        opponent_logits,_,next_state=pool[version].step(vec.obs[ids],frozen_state[:,ids])
                        opponent_actions,_,_=distribution(opponent_logits)
                        chosen[ids]=opponent_actions; frozen_state[:,ids]=next_state
                        active[ids]=False
                    actions[t]=chosen; old_logp[t]=logp; values[t]=value; masks[t]=active
                    vec.step(chosen)
                    events[t]=vec.telemetry
                    rewards[t].copy_(vec.rewards); dones[t].copy_(vec.dones)
                    reset_before=vec.dones.bool().clone()
                    state*=~reset_before.view(1,-1,1); frozen_state*=~reset_before.view(1,-1,1)
                ended=vec.cpu_dones.numpy().reshape(-1,2).all(axis=1)
                for w in np.flatnonzero(ended):
                    completed_matches+=1
                    versions[w]=int(random.choice(recent)); side[w]=int(random.integers(2))
                # No snapshot can disappear while a match is still using it.
                keep=set(recent)|set(versions)
                for version in list(pool):
                    if version not in keep: del pool[version]
            with torch.no_grad():
                _,last_value,_=policy.step(vec.obs,state)
                adv=advantages(rewards,values,dones,last_value)
                returns=adv+values
            if not torch.isfinite(obs).all() or not torch.isfinite(adv).all():
                raise RuntimeError('Nonfinite observation or return')
            rollout_seconds=time.monotonic()-rollout_started
            optimize_started=time.monotonic()
            # Each segment retains its rollout-start memory and all internal
            # reset masks. Historical opponents never become learner samples.
            segments=tmax//length
            def pack(x):
                return x.reshape(segments,length,n,*x.shape[2:]).transpose(1,2).reshape(segments*n,length,*x.shape[2:]).transpose(0,1)
            bo,ba,bl,bv,br,bad,bm,bd=map(pack,(obs,actions,old_logp,values,returns,adv,masks,resets))
            initial=torch.cat(memories,dim=1)
            indices=torch.where(bm.any(dim=0))[0]
            losses=[]
            for epoch in range(opt.epochs):
                permutation=indices[torch.randperm(len(indices),device=device)]
                for ids in permutation.split(max(4,n)):
                    logits,new_value=policy.sequence(bo[:,ids],initial[:,ids],bd[:,ids])
                    _,logp,entropy=distribution(logits,ba[:,ids])
                    mask=bm[:,ids]
                    advantage=bad[:,ids][mask]
                    advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
                    logratio=(logp-bl[:,ids])[mask]; ratio=logratio.exp()
                    pg=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(.8,1.2)).mean()
                    value_loss=.5*(new_value[mask]-br[:,ids][mask]).square().mean()
                    entropy_loss=entropy[mask].mean()
                    loss=pg+.5*value_loss-.01*entropy_loss
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)
                    if not torch.isfinite(loss) or not torch.isfinite(grad): raise RuntimeError('Nonfinite loss or gradient')
                    optimizer.step()
                    losses.append([pg.item(),value_loss.item(),entropy_loss.item(),((ratio-1)-logratio).mean().item(),((ratio-1).abs()>.2).float().mean().item(),grad.item()])
            torch.cuda.synchronize()
            train_seconds=time.monotonic()-optimize_started
            step+=tmax*n; updates+=1
            learner_mask=masks.cpu().numpy()
            buttons=actions[:,:,1].cpu().numpy()[learner_mask]
            behavior=summarize(events[learner_mask])
            loss=np.mean(losses,axis=0)
            env=vec.log()
            if env.get('game_errors',0) or env.get('observation_overflow',0):
                raise RuntimeError(f'Environment errors or dropped observations: {env}')
            mean=np.mean(rewards.cpu().numpy())
            target=returns[masks]; prediction=values[masks]
            variance=target.var(unbiased=False)
            explained=(1-(target-prediction).var(unbiased=False)/variance).item() if variance>1e-12 else 0.0
            elapsed=time.monotonic()-started
            record=dict(step=step,update=updates,elapsed=elapsed,sps=tmax*n/(rollout_seconds+train_seconds),
                mean_reward=float(mean),matches=completed_matches,env=env,
                rollout_seconds=rollout_seconds,train_seconds=train_seconds,
                policy_loss=float(loss[0]),value_loss=float(loss[1]),entropy=float(loss[2]),
                kl=float(loss[3]),clip_fraction=float(loss[4]),gradient_norm=float(loss[5]),explained_variance=explained,
                action_rates={name:float(np.mean((buttons&(1<<i))!=0)) for i,name in enumerate(['jump','dash','arrow','platform'])},
                behavior=behavior,
                pool_steps=recent,learner_fraction=float(masks.float().mean().item()),
                cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            metrics.write(json.dumps(record,allow_nan=False)+'\n')
            atomic_json(out/'status.json',dict(status='running',pid=os.getpid(),**record,target_steps=opt.steps))
            if updates%5==0: print(json.dumps(record),flush=True)
            if step>=next_checkpoint:
                path=checkpoint(f'policy-{step:09d}')
                pool[step]=frozen_copy()
                recent=(recent+[step])[-4:]
                next_checkpoint=step+opt.checkpoint_every
                if opt.eval_at is None: service_evaluation(path)
            if evaluation_milestones and step>=evaluation_milestones[0]:
                service_evaluation(checkpoint(f'policy-{step:09d}'))
                evaluation_milestones=[x for x in evaluation_milestones if x>step]
            service_evaluation()
        status='stopped' if stopping else 'complete'
        final=checkpoint(f'policy-{step:09d}')
        service_evaluation(final)
        # Let the last evaluation finish so the run includes its final evidence.
        while evaluation or pending_evaluation:
            service_evaluation(); time.sleep(1)
        result=dict(status=status,step=step,updates=updates,
                    elapsed=time.monotonic()-started,checkpoint=str(final),matches=completed_matches)
    except BaseException as error:
        status='failed'; failure=str(error); traceback.print_exc()
        raise
    finally:
        if evaluation and evaluation.poll() is None:
            evaluation.terminate()
            try: evaluation.wait(timeout=15)
            except subprocess.TimeoutExpired: evaluation.kill(); evaluation.wait()
        metrics.close(); vec.close()
        atomic_json(out/'status.json',dict(status=status,pid=os.getpid(),step=step,updates=updates,
                    target_steps=opt.steps,elapsed=time.monotonic()-started,failure=failure,matches=completed_matches))
        if video_process:
            try: video_code=video_process.wait(timeout=150)
            except subprocess.TimeoutExpired:
                video_process.terminate(); video_process.wait(timeout=10)
                video_code=-1
            video_log.close()
            if video_code:
                atomic_json(out/'video-status.json',dict(status='failed',exit_code=video_code))
                if status=='complete':
                    atomic_json(out/'status.json',dict(status='failed',step=step,failure='Video worker failed; see video-worker.log'))
                    raise RuntimeError('Video worker failed; see video-worker.log')
        if result is not None:
            result['elapsed_including_artifacts']=time.monotonic()-started
            result['wall_seconds']=time.monotonic()-program_started
            atomic_json(out/'result.json',result)

if __name__=='__main__': main()
