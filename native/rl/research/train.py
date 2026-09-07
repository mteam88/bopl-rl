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
import shutil
import subprocess
import sys
import time
import traceback
from evaluation_protocol import map_ids,opponents as frozen_opponents

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
    parser.add_argument('--minibatch-segments',type=int,
                        help='Sequence segments per PPO minibatch; defaults to the number of agents (2*workers), the original behaviour')
    parser.add_argument('--checkpoint-every',type=int,default=100_000)
    parser.add_argument('--eval-at',type=lambda value:[int(x) for x in value.split(',')],
                        help='Evaluate these decision milestones; does not change historical checkpoint cadence')
    parser.add_argument('--evaluation-games',type=int,default=8)
    parser.add_argument('--evaluation-heldout-games',type=int,default=4)
    parser.add_argument('--training-maps',type=map_ids,default=[0])
    parser.add_argument('--evaluation-maps',type=map_ids,default=[0])
    parser.add_argument('--heldout-maps',type=map_ids,default=[1,2])
    parser.add_argument('--evaluation-opponents',type=Path)
    parser.add_argument('--evaluation-frozen-games',type=int,default=16)
    parser.add_argument('--evaluation-workers',type=int,default=2)
    parser.add_argument('--evaluation-baseline',type=Path,
                        help='Frozen opponent shared by the experiment series; defaults to this run initial policy')
    parser.add_argument('--episode-ticks',type=int,default=3600)
    parser.add_argument('--pipeline-groups',type=int,default=None,
                        help='Contiguous worker groups stepped on their own C thread/CUDA stream so one group ticks while another infers; 1 is the original synchronous loop. Default 2 for even worker counts (measured +24%% active throughput at 16 workers, bit-exact with --pipeline-serial), else 1')
    parser.add_argument('--pipeline-serial',action='store_true',
                        help='Test oracle: wait for each group right after launching it (same (t,group) inference/RNG order, no overlap)')
    parser.add_argument('--cuda-graphs',type=int,default=1,choices=(0,1),
                        help='Replay captured CUDA graphs for rollout inference of the learner and each historical policy (bit-exact with eager inference; default on)')
    parser.add_argument('--no-eval',action='store_true')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--load',type=Path)
    parser.add_argument('--continuation-context',type=Path,help='Restore a declared recent opponent pool; requires --load, fresh games')
    parser.add_argument('--initialize',type=Path,help='Weights only, fresh optimizer and decision count')
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--entropy-coefficient',type=float,default=0.01)
    parser.add_argument('--entropy-halflife',type=int,default=None,
                        help='Learner decisions per halving of the entropy coefficient (exponential decay); constant when omitted')
    parser.add_argument('--entropy-floor',type=float,default=0.0,help='Lower bound for the decayed entropy coefficient')
    parser.add_argument('--learner-steps',type=int,help='Stop by actions included in PPO, rather than all player actions')
    parser.add_argument('--scripted-opponents',type=Path)
    parser.add_argument('--scripted-fraction',type=float,default=2/3,
                        help='Fraction of historical-opponent assignments replaced by scripts')
    opt=parser.parse_args()
    if opt.continuation_context and not opt.load: parser.error('Continuation context requires --load')
    if opt.load and opt.initialize: parser.error('Choose resume or weights-only initialization')
    if opt.lr<=0 or not 0<=opt.scripted_fraction<=1: parser.error('Invalid learning rate or scripted fraction')
    if opt.learner_steps is not None and opt.learner_steps<1: parser.error('Positive learner budget required')
    if min(opt.steps,opt.workers,opt.horizon,opt.sequence,opt.epochs,opt.checkpoint_every,opt.episode_ticks)<1 or opt.horizon%opt.sequence:
        parser.error('Positive counts and horizon divisible by sequence are required')
    if opt.minibatch_segments is not None and opt.minibatch_segments<1: parser.error('Minibatch segments must be positive')
    if opt.pipeline_groups is None: opt.pipeline_groups=2 if opt.workers%2==0 else 1
    if opt.pipeline_groups<1 or opt.workers%opt.pipeline_groups: parser.error('Pipeline groups must divide the worker count')
    if opt.evaluation_workers<1 or any(n<2 or n%2 for n in
            (opt.evaluation_games,opt.evaluation_heldout_games,opt.evaluation_frozen_games)):
        parser.error('Evaluation needs positive workers and complete side-swapped game pairs')
    if opt.eval_at is not None and (not opt.eval_at or any(x<=0 for x in opt.eval_at)):
        parser.error('Evaluation milestones must be positive')
    if set(opt.heldout_maps)&(set(opt.training_maps)|set(opt.evaluation_maps)):
        parser.error('Held-out maps must be excluded from training and primary evaluation maps')
    frozen=frozen_opponents(opt.evaluation_opponents)
    out=opt.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    # BOPL_CPU_DMA_LATENCY_US=<us>: hold a kernel PM QoS request (/dev/cpu_dma_latency)
    # for the life of the run so idle CPUs stay out of deep C-states with wake-up
    # latency above the cap. Runtime only; the games and trainer compute nothing differently.
    if os.environ.get('BOPL_CPU_DMA_LATENCY_US'):
        import struct
        global dma_latency_request
        dma_latency_request=open('/dev/cpu_dma_latency','wb',buffering=0)
        dma_latency_request.write(struct.pack('i',int(os.environ['BOPL_CPU_DMA_LATENCY_US'])))
    os.environ['OMP_NUM_THREADS']=str(opt.workers)
    os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
    from environment import configure,game_identity
    TRACE=os.environ.get('BOPL_TRACE_TURN')=='1'
    configure(out/'workers')
    import numpy as np
    import torch
    from policy import Policy
    from ppo import update as ppo_update
    from vector import Vector
    from telemetry import FIELDS,summarize
    from selfplay import Opponents,OPPONENT_SAMPLING
    from inference import Stepper
    # BOPL_TURN_GRAPH=1: replay one captured CUDA graph per group turn (learner step,
    # sampling, every historical version's step and scatter; see turn.py). Needs --cuda-graphs 1.
    TURN_GRAPH=os.environ.get('BOPL_TURN_GRAPH')=='1' and bool(opt.cuda_graphs)
    if TURN_GRAPH: from turn import TurnGraph
    # BOPL_COLLECT_GRAPH=1: collect()'s reset-flag store and recurrent-state masking
    # for a group as one replayed graph (turn.py CollectGraph).
    COLLECT_GRAPH=os.environ.get('BOPL_COLLECT_GRAPH')=='1' and bool(opt.cuda_graphs)
    if COLLECT_GRAPH: from turn import CollectGraph
    torch.set_num_threads(opt.workers)
    torch.manual_seed(opt.seed); np.random.seed(opt.seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=True
    device='cuda'
    vec=Vector(opt.workers,opt.seed,opt.episode_ticks,groups=opt.pipeline_groups,training_maps=opt.training_maps)
    policy=Policy(vec.obs_size).to(device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=opt.lr,eps=1e-5)
    step=0; updates=0; learner_steps=0
    if opt.initialize:
        initial=torch.load(opt.initialize,map_location=device,weights_only=False)
        if initial.get('action_mode','absolute')!='absolute' or initial.get('input_mode','full')!='full' or initial.get('task')=='bow':
            raise ValueError('Initialization must explicitly use the full-game action contract')
        policy.load_state_dict(initial['policy'])
    if opt.load:
        previous=torch.load(opt.load,map_location=device,weights_only=False)
        policy.load_state_dict(previous['policy']); optimizer.load_state_dict(previous['optimizer'])
        step=previous['step']; updates=previous['updates']
        learner_steps=previous.get('learner_steps',0)
        if opt.learner_steps and 'learner_steps' not in previous:
            raise ValueError('Cannot resume a learner budget without its saved count')
    continuation=None
    if opt.continuation_context:
        from continuation import read_context
        continuation=read_context(opt.continuation_context,opt.load,step)
        if continuation['checkpoint_every']!=opt.checkpoint_every:
            raise ValueError('Continuation must preserve historical snapshot cadence')
    config={**vars(opt),'output':str(out),'load':str(opt.load) if opt.load else None,
            'continuation_context':str(opt.continuation_context.resolve()) if opt.continuation_context else None,
            'continuation':continuation,
            'initialize':str(opt.initialize.resolve()) if opt.initialize else None,
            'initialize_sha256':hashlib.sha256(opt.initialize.read_bytes()).hexdigest() if opt.initialize else None,
            'scripted_opponents':str(opt.scripted_opponents.resolve()) if opt.scripted_opponents else None,
            'scripted_manifest_sha256':hashlib.sha256(opt.scripted_opponents.read_bytes()).hexdigest() if opt.scripted_opponents else None,
            'evaluation_opponents':str(opt.evaluation_opponents.resolve()) if opt.evaluation_opponents else None,
            'frozen_opponents':frozen,
            'evaluation_baseline':str(opt.evaluation_baseline.resolve()) if opt.evaluation_baseline else None,
            'evaluation_baseline_sha256':hashlib.sha256(opt.evaluation_baseline.read_bytes()).hexdigest() if opt.evaluation_baseline else None,
            'schema':1,'obs_size':vec.obs_size,'hidden':policy.hidden,'device':device,
            'telemetry_schema':1,'telemetry_fields':FIELDS,'action_rates_scope':'learner decisions in this rollout',
            'startup_seconds':time.monotonic()-program_started,'torch_version':torch.__version__,
            'gpu':torch.cuda.get_device_name(),
            'pipeline_groups':opt.pipeline_groups,'cuda_graphs':opt.cuda_graphs,'split_step_env':os.environ.get('BOPL_SPLIT_STEP','0'),
            'decisions_per_update':opt.horizon*vec.size,'minibatch_segments':max(4,opt.minibatch_segments if opt.minibatch_segments is not None else vec.size),
            'algorithm':'recurrent PPO','gamma':.995,'gae_lambda':.95,'learning_rate':opt.lr,
            'action_hz':15,'observation_delay_ticks':9,'own_previous_actions':'immediate','loadout':['dash','arrow','platform'],
            'training_maps':opt.training_maps,'eval_maps':opt.evaluation_maps+opt.heldout_maps,'stocks':3,'pool_size':4,
            'fixed_opponent_game_fraction':.5,
            'historical_game_fraction':.5*(1-opt.scripted_fraction if opt.scripted_opponents else 1),
            'scripted_game_fraction':.5*opt.scripted_fraction if opt.scripted_opponents else 0,
            'reward':{'win':1,'loss':-1,'kill':.1,'death':-.1,'timeout':0},
            'timeout_semantics':'finite-horizon match; no winner bonus; terminal value zero',
            'puffer_revision':'42f70d6932c30ac977736f861006809c50168ba9',
            # The selected player (BOPL_RL_GAME) and its runtime payload; assembly_sha256 is only
            # present for Mono players, IL2CPP players are identified by GameAssembly.so + metadata.
            'game_identity':game_identity()}
    config['assembly_sha256']=config['game_identity'].get('assembly_sha256')
    sources={}
    for directory in (ROOT/'native/rl/puffer',Path(__file__).parent):
        for file in sorted(directory.rglob('*')):
            if file.is_file() and file.suffix in ('.py','.c','.h','.ini','.patch','.html','.css','.js','.lock'):
                relative=str(file.relative_to(ROOT)); sources[relative]=hashlib.sha256(file.read_bytes()).hexdigest()
    config['source_sha256']=sources
    for relative in sources:
        target=out/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    from pufferlib import _C
    config['puffer_library_sha256']=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest()
    config['bootstrap_sha256']=hashlib.sha256((ROOT/'native/fixtures/movement.rep').read_bytes()).hexdigest()
    config['research_library_sha256']=hashlib.sha256((ROOT/'native/artifacts/rl/libbopl_research.so').read_bytes()).hexdigest()
    config['git_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    atomic_json(out/'config.json',config)
    atomic_json(out/'status.json',dict(status='starting',pid=os.getpid(),step=step,target_steps=opt.steps,
                                     target_learner_steps=opt.learner_steps,learner_steps=learner_steps))
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
    if continuation:
        pool={}
        for record in continuation['historical']:
            opponent=frozen_copy()
            saved=torch.load(record['path'],map_location=device,weights_only=False)
            if saved['step']!=record['step']:raise ValueError('Historical step mismatch')
            opponent.load_state_dict(saved['policy']);pool[record['step']]=opponent
        recent=sorted(pool);versions=[recent[-1]]*opt.workers
        next_checkpoint=continuation['next_checkpoint']
    historical=[w%2==1 for w in range(opt.workers)]
    side=[w%2 for w in range(opt.workers)]
    random=np.random.default_rng(opt.seed)
    from scripted_opponents import load_specs,ScriptedOpponent
    script_specs=load_specs(opt.scripted_opponents) if opt.scripted_opponents else []
    script_rng=np.random.default_rng(opt.seed+1000003)
    scripts=[None]*opt.workers
    script_assignments={spec.name:0 for spec in script_specs}
    def assign_script(w):
        scripts[w]=None
        historical[w]=w%2==1
        if historical[w] and script_specs and script_rng.random()<opt.scripted_fraction:
            spec=script_specs[int(script_rng.integers(len(script_specs)))]
            scripts[w]=ScriptedOpponent(spec,int(script_rng.integers(2**31)))
            script_assignments[spec.name]+=1
            historical[w]=False
    for w in range(opt.workers): assign_script(w)
    n=vec.size; h=policy.hidden; tmax=opt.horizon; length=opt.sequence
    minibatch=max(4,opt.minibatch_segments if opt.minibatch_segments is not None else n)
    state=policy.initial(n,device); frozen_state=policy.initial(n,device)
    # Pipeline groups: contiguous worker ranges. Group g owns agents slices[g]
    # and workers [g*wpg,(g+1)*wpg). With one group this is the whole batch.
    groups=opt.pipeline_groups; wpg=opt.workers//groups
    slices=vec.groups if groups>1 else [slice(0,n)]
    opponents=[Opponents(wpg,device,first=g*wpg) for g in range(groups)]
    graphs=bool(opt.cuda_graphs)
    learner_stepper=Stepper(policy,n//groups,device,graph=graphs)
    # Historical-policy steppers are created outside the rollout (no group is
    # in flight during graph capture) and pruned once their snapshot leaves the pool.
    steppers={}
    def add_steppers(version):
        if not graphs: return
        for g in range(groups):
            capacity=opponents[g].capacity
            if capacity: steppers[version,g]=Stepper(pool[version],capacity,device,graph=True,sample='actions' if OPPONENT_SAMPLING=='graph' else False)
    def prune_steppers():
        for key in [k for k in steppers if k[0] not in pool]: steppers.pop(key).close()
    def group_steppers(g):
        if not graphs: return None
        return {version:stepper for (version,gg),stepper in steppers.items() if gg==g}
    for version in pool:add_steppers(version)
    turns=[TurnGraph(policy,pool,opponents[g],vec.obs[sl],state[:,sl],frozen_state[:,sl],device) for g,sl in enumerate(slices)] if TURN_GRAPH else []
    if continuation:
        # Game and private NumPy RNG state were not in the old checkpoint.
        # They restart from --seed; Torch sampling continues from saved state.
        torch.set_rng_state(previous['torch_rng'].cpu())
        torch.cuda.set_rng_state(previous['cuda_rng'].cpu())
    reset_before=torch.ones(n,dtype=torch.bool,device=device)
    collects=[CollectGraph(vec.dones[sl],reset_before[sl],state[:,sl],frozen_state[:,sl]) for sl in slices] if COLLECT_GRAPH else []
    # BOPL_TRACE_TURN=1: per-group-turn main-thread phase times (ms per turn, per update) in metrics.jsonl.
    trace_sum=[0.0]*7+[0]
    completed_matches=0
    evaluation_milestones=sorted(set(opt.eval_at or []))
    metrics=(out/'metrics.jsonl').open('a',buffering=1)

    def checkpoint(label):
        path=out/'checkpoints'/f'{label}.pt'; tmp=path.with_suffix('.tmp')
        torch.save(dict(schema=1,policy=policy.state_dict(),optimizer=optimizer.state_dict(),
                        obs_size=vec.obs_size,hidden=h,step=step,learner_steps=learner_steps,updates=updates,config=config,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),tmp)
        tmp.replace(path)
        return path

    def launch_evaluation(path):
        nonlocal evaluation,evaluation_log
        evaluation_log=(out/'evaluation-worker.log').open('a')
        evaluation=subprocess.Popen([sys.executable,str(Path(__file__).with_name('evaluate.py')),
            '--run',str(out),'--checkpoint',str(path),'--initial',str(opt.evaluation_baseline or initial_path),
            '--games-per-opponent',str(opt.evaluation_games),'--heldout-games',str(opt.evaluation_heldout_games),
            '--evaluation-maps',','.join(map(str,opt.evaluation_maps)),'--heldout-maps',','.join(map(str,opt.heldout_maps)),
            '--frozen-games',str(opt.evaluation_frozen_games),
            *(['--opponents',str(opt.evaluation_opponents.resolve())] if opt.evaluation_opponents else []),
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
        while (learner_steps<opt.learner_steps if opt.learner_steps else step<opt.steps) and not stopping:
            rollout_started=time.monotonic()
            obs=torch.empty(tmax,n,vec.obs_size,device=device)
            actions=torch.empty(tmax,n,2,dtype=torch.long,device=device)
            old_logp=torch.empty(tmax,n,device=device); values=torch.empty(tmax,n,device=device)
            rewards=torch.empty(tmax,n,device=device); dones=torch.empty(tmax,n,device=device)
            masks=torch.empty(tmax,n,dtype=torch.bool,device=device)
            resets=torch.empty(tmax,n,dtype=torch.bool,device=device)
            events=np.empty((tmax,n,len(FIELDS)),dtype=np.float32)
            memories=torch.empty(tmax//length,1,n,h,device=device)
            game_decisions={'current':0,'historical':0,**{spec.name:0 for spec in script_specs}}
            def collect(t,g):
                """Bank group g's results of decision t (its step is complete) and
                apply its episode boundaries before the group's next inference."""
                sl=slices[g]
                events[t,sl]=vec.telemetry[sl]
                rewards[t,sl].copy_(vec.rewards[sl]); dones[t,sl].copy_(vec.dones[sl])
                if collects: collects[g].replay()
                else:
                    ended_now=vec.dones[sl].bool().clone()
                    reset_before[sl]=ended_now
                    keep_state=~ended_now.view(1,-1,1)
                    state[:,sl]*=keep_state; frozen_state[:,sl]*=keep_state
                ended=vec.cpu_dones[sl].numpy().reshape(-1,2).all(axis=1)
                for w in np.flatnonzero(ended):
                    w+=g*wpg; matches[0]+=1
                    versions[w]=int(random.choice(recent)); side[w]=int(random.integers(2))
                    assign_script(w)
                # No snapshot can disappear while a match is still using it.
                keep=set(recent)|set(versions)
                for version in list(pool):
                    if version not in keep: del pool[version]
            matches=[completed_matches]
            dump=updates==0 and os.environ.get('BOPL_DUMP_ROLLOUT')
            states_seen=torch.empty(tmax,1,n,h,device=device) if dump else None
            for t in range(tmax):
                for g in range(groups):
                    sl=slices[g]
                    if TRACE: trace_t0=time.perf_counter()
                    if groups>1: vec.wait_group(g)
                    if TRACE: trace_t1=time.perf_counter()
                    if t: collect(t-1,g)
                    if TRACE: trace_t2=time.perf_counter()
                    if t%length==0: memories[t//length][:,sl].copy_(state[:,sl])
                    if dump: states_seen[t][:,sl].copy_(state[:,sl])
                    group_obs=vec.obs[sl]
                    obs[t,sl].copy_(group_obs); resets[t,sl]=reset_before[sl]
                    with torch.no_grad():
                        if TURN_GRAPH:
                            if TRACE: trace_t3=time.perf_counter()
                            opponents[g].refresh(versions,historical,side); turns[g].refresh()
                            if TRACE: trace_t4=time.perf_counter()
                            chosen,logp,value=turns[g].replay(); active=opponents[g].active
                        else:
                            chosen,logp,value,next_state=learner_stepper(group_obs,state[:,sl])
                            state[:,sl].copy_(next_state)
                            if TRACE: trace_t3=time.perf_counter()
                            opponents[g].refresh(versions,historical,side)
                            if TRACE: trace_t4=time.perf_counter()
                            active=opponents[g].act(pool,group_obs,frozen_state[:,sl],chosen,group_steppers(g))
                        if TRACE: trace_t5=time.perf_counter()
                        script_workers=[w for w in range(g*wpg,(g+1)*wpg) if scripts[w] is not None]
                        for w in range(g*wpg,(g+1)*wpg):
                            kind=scripts[w].spec.name if scripts[w] is not None else 'historical' if historical[w] else 'current'
                            game_decisions[kind]+=1
                        if script_workers:
                            # One device-to-host transfer per group. Scripts see
                            # precisely the same delayed observations as policy.
                            delayed=group_obs.cpu().numpy()
                            active=active.clone()
                            for w in script_workers:
                                row=2*(w-g*wpg)+side[w]
                                chosen[row]=torch.as_tensor(scripts[w].act(delayed[row]),device=device)
                                active[row]=False
                        actions[t,sl]=chosen; old_logp[t,sl]=logp; values[t,sl]=value; masks[t,sl]=active
                        if TRACE: trace_t6=time.perf_counter()
                        if groups>1:
                            vec.step_group(g,chosen)
                            if opt.pipeline_serial: vec.wait_group(g)
                        else: vec.step(chosen)
                        if TRACE:
                            trace_t7=time.perf_counter()
                            for i,(a,b) in enumerate(((trace_t0,trace_t1),(trace_t1,trace_t2),(trace_t2,trace_t3),(trace_t3,trace_t4),(trace_t4,trace_t5),(trace_t5,trace_t6),(trace_t6,trace_t7))):
                                trace_sum[i]+=b-a
                            trace_sum[7]+=1
            if groups>1: vec.wait_all()
            for g in range(groups): collect(tmax-1,g)
            completed_matches=matches[0]
            prune_steppers()
            for turn in turns: turn.prune()
            if updates==0 and os.environ.get('BOPL_DUMP_ROLLOUT'):
                # Test hook (check_pipeline.py): the first update's complete rollout buffers.
                torch.save(dict(obs=obs,actions=actions,old_logp=old_logp,values=values,rewards=rewards,dones=dones,masks=masks,
                                resets=resets,memories=memories,states=states_seen,events=torch.from_numpy(events),
                                final_state=state,final_reset=reset_before),os.environ['BOPL_DUMP_ROLLOUT'])
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
            entropy_coefficient=opt.entropy_coefficient
            if opt.entropy_halflife: entropy_coefficient=max(opt.entropy_floor,entropy_coefficient*0.5**(learner_steps/opt.entropy_halflife))
            losses=ppo_update(policy,optimizer,obs,actions,old_logp,values,returns,adv,
                              masks,resets,memories,length,opt.epochs,minibatch,entropy_coefficient=entropy_coefficient)
            torch.cuda.synchronize()
            train_seconds=time.monotonic()-optimize_started
            step+=tmax*n; updates+=1
            learner_steps+=int(masks.sum().item())
            learner_mask=masks.cpu().numpy()
            buttons=actions[:,:,1].cpu().numpy()[learner_mask]
            behavior=summarize(events[learner_mask])
            loss=np.mean(losses,axis=0)
            env=vec.log()
            if env.get('game_errors',0) or env.get('observation_overflow',0):
                raise RuntimeError(f'Environment errors or dropped observations: {env}')
            if env['step_mode']=='fallback': raise RuntimeError('Game build rejected the single-roundtrip sampled step; rebuild the player')
            if 'step_mode' not in config: config['step_mode']=env['step_mode']; atomic_json(out/'config.json',config)
            mean=np.mean(rewards.cpu().numpy())
            target=returns[masks]; prediction=values[masks]
            variance=target.var(unbiased=False)
            explained=(1-(target-prediction).var(unbiased=False)/variance).item() if variance>1e-12 else 0.0
            elapsed=time.monotonic()-started
            record=dict(step=step,learner_steps=learner_steps,script_assignments=script_assignments.copy(),update=updates,elapsed=elapsed,sps=tmax*n/(rollout_seconds+train_seconds),
                mean_reward=float(mean),matches=completed_matches,env=env,
                map_matches={key.removeprefix('map_').removesuffix('_fraction'):round(value*env['n']/2)
                             for key,value in env.items() if key.startswith('map_') and key.endswith('_fraction')},
                rollout_seconds=rollout_seconds,train_seconds=train_seconds,
                policy_loss=float(loss[0]),value_loss=float(loss[1]),entropy=float(loss[2]),entropy_coefficient=entropy_coefficient,
                kl=float(loss[3]),clip_fraction=float(loss[4]),gradient_norm=float(loss[5]),explained_variance=explained,
                action_rates={name:float(np.mean((buttons&(1<<i))!=0)) for i,name in enumerate(['jump','dash','arrow','platform'])},
                behavior=behavior,
                pool_steps=recent,learner_fraction=float(masks.float().mean().item()),opponent_game_decisions=game_decisions,
                cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,cuda_reserved_mb=torch.cuda.memory_reserved()/1048576)
            if groups>1: record['pipeline']=vec.group_profile()
            if turns: record['pipeline']={**record.get('pipeline',{}),'turn_graphs':[turn.stats() for turn in turns]}   # runtime bookkeeping, stripped by check_pipeline
            if TRACE and trace_sum[7]:
                record['turn_ms']=dict(zip(('wait','collect','learner','refresh','act','stores','launch'),[1000*x/trace_sum[7] for x in trace_sum[:7]]))
                record['turn_ms']['turns']=trace_sum[7]; trace_sum[:]=[0.0]*7+[0]
            # Snapshot work (checkpoint file, frozen copy, historical graph
            # capture) happens outside rollout/train time; time it explicitly
            # so loop sps cannot hide it. Evaluation launches are asynchronous.
            snapshot_started=time.monotonic()
            if step>=next_checkpoint:
                path=checkpoint(f'policy-{step:09d}')
                pool[step]=frozen_copy(); add_steppers(step)
                recent=(recent+[step])[-4:]
                next_checkpoint=step+opt.checkpoint_every
                if opt.eval_at is None: service_evaluation(path)
            if evaluation_milestones and step>=evaluation_milestones[0]:
                service_evaluation(checkpoint(f'policy-{step:09d}'))
                evaluation_milestones=[x for x in evaluation_milestones if x>step]
            service_evaluation()
            record['snapshot_seconds']=time.monotonic()-snapshot_started
            record['pool_versions']=len(pool); record['steppers']=len(steppers)
            metrics.write(json.dumps(record,allow_nan=False)+'\n')
            atomic_json(out/'status.json',dict(status='running',pid=os.getpid(),**record,target_steps=opt.steps,
                                              target_learner_steps=opt.learner_steps))
            if updates%5==0: print(json.dumps(record),flush=True)
        status='stopped' if stopping else 'complete'
        final=checkpoint(f'policy-{step:09d}')
        service_evaluation(final)
        # Let the last evaluation finish so the run includes its final evidence.
        while evaluation or pending_evaluation:
            service_evaluation(); time.sleep(1)
        result=dict(status=status,step=step,learner_steps=learner_steps,updates=updates,
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
                    target_steps=opt.steps,target_learner_steps=opt.learner_steps,learner_steps=learner_steps,
                    elapsed=time.monotonic()-started,failure=failure,matches=completed_matches))
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
