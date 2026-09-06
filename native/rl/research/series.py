"""Declare and execute a sequential, fixed-protocol training series."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from train import ROOT,atomic_json
from evaluation_protocol import map_ids,opponents

HERE=Path(__file__).resolve().parent

def identity():
    from pufferlib import _C
    paths=[ROOT/'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll',
           ROOT/'native/artifacts/rl/libbopl_research.so',Path(_C.__file__)]
    for directory in (HERE,ROOT/'native/rl/puffer'):
        paths.extend(p for p in directory.rglob('*') if p.suffix in ('.py','.c','.h','.ini','.patch'))
    paths.extend(p for p in (ROOT/'native/rl').iterdir() if p.suffix in ('.c','.h'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def prepare(args):
    out=args.output.resolve()
    if args.steps<1 or args.workers<1 or not args.seeds or len(set(args.seeds))!=len(args.seeds):
        raise ValueError('Positive steps/workers and distinct training seeds required')
    if args.pipeline_groups<1 or args.workers%args.pipeline_groups:
        raise ValueError('Pipeline groups must be positive and divide the worker count')
    if any(x<2 or x%2 for x in (args.diagnostic_games,args.diagnostic_heldout_games,args.final_games,args.final_heldout_games,
                               args.diagnostic_frozen_games,args.final_frozen_games)):
        raise ValueError('Evaluation game counts must be positive seed pairs')
    if set(args.training_maps)&set(args.heldout_maps):
        raise ValueError('Training and held-out maps must be disjoint')
    frozen=opponents(args.opponents)
    baseline=args.baseline.resolve()
    baseline_hash=hashlib.sha256(baseline.read_bytes()).hexdigest()
    source_identity=identity()
    out.mkdir(parents=True,exist_ok=False)
    if frozen: atomic_json(out/'opponents.json',dict(schema=1,opponents=frozen))
    plan=dict(schema=1,kind='bopl-training-series',name=out.name,
              created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              source_identity=source_identity,steps=args.steps,workers=args.workers,resource_sampling=True,
              pipeline_groups=args.pipeline_groups,cuda_graphs=args.cuda_graphs,split_step=args.split_step,
              training_maps=args.training_maps,evaluation_maps=args.training_maps,heldout_maps=args.heldout_maps,
              opponents=str(out/'opponents.json') if frozen else None,frozen_opponents=frozen,
              runs=[dict(name=f'{out.name}-s{seed}',seed=seed) for seed in args.seeds],
              baseline=str(baseline),baseline_sha256=baseline_hash,
              evaluation_milestones=[x for x in (1_000_000,3_000_000,args.steps) if x<=args.steps],
              diagnostic_games=args.diagnostic_games,diagnostic_heldout_games=args.diagnostic_heldout_games,evaluation_workers=4,
              diagnostic_frozen_games=args.diagnostic_frozen_games,final_frozen_games=args.final_frozen_games,
              final_games=args.final_games,final_heldout_games=args.final_heldout_games,final_seed_base=50001,
              final_heldout_seed_base=60001,final_selfplay_seed=70001,
              checkpoint_every=100_000,
              protocol='Fresh initialization per training seed; fixed PPO/reward/observation/training-opponent settings; explicit maps sampled at reset; frozen evaluation-only opponents; final seeds excluded from diagnostic evaluation.')
    atomic_json(out/'plan.json',plan)
    atomic_json(out/'status.json',dict(status='prepared',completed=[],active_run=None))
    print(out/'plan.json',flush=True)

def run(plan_path):
    plan_path=plan_path.resolve(); out=plan_path.parent; root=out.parent
    plan=json.loads(plan_path.read_text())
    if plan.get('kind')!='bopl-training-series': raise ValueError('Not a training series plan')
    baseline=Path(plan['baseline'])
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=plan['baseline_sha256']:
        raise ValueError('Frozen baseline identity changed')
    if identity()!=plan['source_identity']:
        raise ValueError('Training implementation changed since series preparation')
    def check_opponents():
        if hashlib.sha256(baseline.read_bytes()).hexdigest()!=plan['baseline_sha256']:
            raise ValueError('Frozen baseline identity changed')
        if opponents(plan.get('opponents'))!=plan.get('frozen_opponents',[]):
            raise ValueError('Frozen opponent manifest changed since preparation')
    check_opponents()
    evaluation_args=['--evaluation-maps',','.join(map(str,plan.get('evaluation_maps',[0]))),
                     '--heldout-maps',','.join(map(str,plan.get('heldout_maps',[1,2])))]
    completed=[]; process=None; monitor=None; monitor_log=None; stopping=False; started=time.monotonic()
    active=None; phase=None
    def status(state,**extra):
        atomic_json(out/'status.json',dict(status=state,completed=completed,active_run=active,
                    phase=phase,pid=os.getpid(),elapsed=time.monotonic()-started,**extra))
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if process is not None and process.poll() is None: process.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def execute(command,log):
        nonlocal process
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                     env=dict(os.environ,BOPL_SPLIT_STEP=str(plan.get('split_step',0)),BOPL_GROUP_JITTER_US='0',BOPL_DUMP_ROLLOUT=''))
            while process.poll() is None:
                if monitor is not None and monitor.poll() is not None:
                    raise RuntimeError('Resource sampler failed; see resource-monitor.log')
                status('stopping' if stopping else 'running',child_pid=process.pid)
                time.sleep(5)
            code=process.returncode; process=None
        if stopping: raise InterruptedError('Series stopped; unfinished runs are not silently resumed')
        if code: raise RuntimeError(f'Process exited {code}; see {log}')
    try:
        status('running')
        monitor_log=(out/'resource-monitor.log').open('w')
        monitor=subprocess.Popen([sys.executable,str(HERE/'resource_monitor.py'),'--series',str(out)],
                                 stdout=monitor_log,stderr=subprocess.STDOUT,start_new_session=True)
        for item in plan['runs']:
            if stopping: raise InterruptedError('Series stopped')
            active=item['name']; directory=root/active
            if identity()!=plan['source_identity']:
                raise ValueError('Training implementation changed during the series')
            check_opponents()
            phase='training'; status('running')
            execute([sys.executable,str(HERE/'train.py'),'--output',str(directory),
                     '--steps',str(plan['steps']),'--workers',str(plan['workers']),'--seed',str(item['seed']),
                     '--pipeline-groups',str(plan.get('pipeline_groups',1)),
                     '--cuda-graphs',str(plan.get('cuda_graphs',0)),
                     '--training-maps',','.join(map(str,plan.get('training_maps',[0]))),*evaluation_args,
                     '--evaluation-frozen-games',str(plan.get('diagnostic_frozen_games',16)),
                     *(['--evaluation-opponents',plan['opponents']] if plan.get('opponents') else []),
                     '--checkpoint-every',str(plan['checkpoint_every']),
                     '--eval-at',','.join(map(str,plan['evaluation_milestones'])),
                     '--evaluation-games',str(plan['diagnostic_games']),
                     '--evaluation-heldout-games',str(plan['diagnostic_heldout_games']),
                     '--evaluation-workers',str(plan['evaluation_workers']),
                     '--evaluation-baseline',str(baseline)],out/(active+'-train.log'))
            result=json.loads((directory/'result.json').read_text())
            if result['status']!='complete' or result['step']<plan['steps']:
                raise RuntimeError(f'{active} did not complete its training budget')
            phase='final evaluation'; status('running')
            check_opponents()
            execute([sys.executable,str(HERE/'evaluate.py'),'--run',str(directory),
                     '--checkpoint',result['checkpoint'],'--initial',str(baseline),'--suite','final',
                     *evaluation_args,'--frozen-games',str(plan.get('final_frozen_games',128)),
                     *(['--opponents',plan['opponents']] if plan.get('opponents') else []),
                     '--games-per-opponent',str(plan['final_games']),'--heldout-games',str(plan['final_heldout_games']),
                     '--seed-base',str(plan['final_seed_base']),'--heldout-seed-base',str(plan['final_heldout_seed_base']),
                     '--selfplay-seed',str(plan['final_selfplay_seed']),'--workers',str(plan['evaluation_workers'])],
                    out/(active+'-final-evaluation.log'))
            final=json.loads((directory/'final-evaluations'/f"{result['step']:09d}"/'result.json').read_text())
            completed.append(dict(name=active,seed=item['seed'],step=result['step'],matches=result['matches'],
                                  training=result,groups=final['groups']))
            status('running')
        active=None; phase=None; status('complete')
        atomic_json(out/'result.json',dict(status='complete',runs=completed,elapsed=time.monotonic()-started))
    except BaseException as error:
        status('stopped' if stopping else 'failed',failure=str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=180)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        if monitor is not None:
            monitor.terminate()
            try: monitor.wait(timeout=10)
            except subprocess.TimeoutExpired: monitor.kill();monitor.wait()
        if monitor_log is not None: monitor_log.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare'); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True); p.add_argument('--steps',type=int,default=10_000_000)
    p.add_argument('--workers',type=int,default=16); p.add_argument('--seeds',type=lambda s:[int(x) for x in s.split(',')],default=[73,173,273])
    p.add_argument('--training-maps',type=map_ids,default=[0]);p.add_argument('--heldout-maps',type=map_ids,default=[1,2])
    p.add_argument('--opponents',type=Path)
    p.add_argument('--pipeline-groups',type=int,default=2)
    p.add_argument('--cuda-graphs',type=int,choices=(0,1),default=1)
    p.add_argument('--split-step',type=int,choices=(0,1),default=0,
                   help='Use the original two-request game step instead of the combined request')
    p.add_argument('--diagnostic-games',type=int,default=64);p.add_argument('--diagnostic-heldout-games',type=int,default=16)
    p.add_argument('--diagnostic-frozen-games',type=int,default=16);p.add_argument('--final-frozen-games',type=int,default=128)
    p.add_argument('--final-games',type=int,default=256);p.add_argument('--final-heldout-games',type=int,default=64)
    p=commands.add_parser('run'); p.add_argument('plan',type=Path)
    args=parser.parse_args()
    prepare(args) if args.command=='prepare' else run(args.plan)

if __name__=='__main__': main()
