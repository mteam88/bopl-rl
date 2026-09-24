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
from common import ROOT,atomic_json
from evaluation_protocol import checkpoint_identity,map_ids,opponents

HERE=Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'native/tools'))
from reference import resolve_game,player_sources

def identity(game):
    from pufferlib import _C
    paths=[ROOT/'native/artifacts/rl/libbopl_research.so',
           ROOT/'native/artifacts/rl/libbopl_policy_eval.so',Path(_C.__file__)]
    for directory in (HERE,ROOT/'train'):
        paths.extend(p for p in directory.rglob('*') if p.suffix in ('.py','.c','.h','.cu','.cuh','.ini','.patch'))
    paths.extend(p for p in (ROOT/'sim/abi').iterdir() if p.suffix in ('.c','.h'))
    sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    sources.update({key:hashlib.sha256(p.read_bytes()).hexdigest() for key,p in player_sources(game).items()})
    return sources

def prepare(args):
    out=args.output.resolve()
    if args.steps<1 or args.workers<1 or not args.seeds or len(set(args.seeds))!=len(args.seeds):
        raise ValueError('Positive steps/workers and distinct training seeds required')
    if args.cudagraphs != -1 and args.cudagraphs < 1:
        raise ValueError('cudagraphs must be -1 or at least 1')
    if min(args.episode_ticks,args.checkpoint_every,args.evaluation_every) < 1:
        raise ValueError('Episode cap and checkpoint/evaluation intervals must be positive')
    if args.workers < 2 or args.workers % 2:
        raise ValueError('Native training requires an even number of games')
    if any(x<2 or x%2 for x in (args.diagnostic_games,args.diagnostic_heldout_games,args.final_games,args.final_heldout_games,
                               args.diagnostic_frozen_games,args.final_frozen_games)):
        raise ValueError('Evaluation game counts must be positive seed pairs')
    if set(args.training_maps)&set(args.heldout_maps):
        raise ValueError('Training and held-out maps must be disjoint')
    frozen=opponents(args.opponents)
    baseline=args.baseline.resolve()
    baseline_id=checkpoint_identity(baseline)
    game=resolve_game(args.game)
    source_identity=identity(game)
    out.mkdir(parents=True,exist_ok=False)
    if frozen: atomic_json(out/'opponents.json',dict(schema=2,opponents=frozen))
    plan=dict(schema=2,kind='bopl-native-training-series',name=out.name,
              created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              source_identity=source_identity,steps=args.steps,workers=args.workers,resource_sampling=True,
              encoder=args.encoder,banks=args.banks,cudagraphs=args.cudagraphs,game=str(game),
              step_unit='learner_decisions',episode_ticks=args.episode_ticks,
              training_maps=args.training_maps,evaluation_maps=args.training_maps,heldout_maps=args.heldout_maps,
              opponents=str(out/'opponents.json') if frozen else None,frozen_opponents=frozen,
              runs=[dict(name=f'{out.name}-s{seed}',seed=seed) for seed in args.seeds],
              baseline=str(baseline),baseline_identity=baseline_id,
              evaluation_every=args.evaluation_every,
              diagnostic_games=args.diagnostic_games,diagnostic_heldout_games=args.diagnostic_heldout_games,evaluation_workers=4,
              diagnostic_frozen_games=args.diagnostic_frozen_games,final_frozen_games=args.final_frozen_games,
              final_games=args.final_games,final_heldout_games=args.final_heldout_games,final_seed_base=50001,
              final_heldout_seed_base=60001,final_selfplay_seed=70001,
              checkpoint_every=args.checkpoint_every,
              protocol='Fresh initialization per training seed; fixed PPO/reward/observation/training-opponent settings; explicit maps sampled at reset; frozen evaluation-only opponents; final seeds excluded from diagnostic evaluation.')
    atomic_json(out/'plan.json',plan)
    atomic_json(out/'status.json',dict(status='prepared',completed=[],active_run=None))
    print(out/'plan.json',flush=True)

def run(plan_path):
    plan_path=plan_path.resolve(); out=plan_path.parent; root=out.parent
    plan=json.loads(plan_path.read_text())
    if plan.get('schema')!=2 or plan.get('kind')!='bopl-native-training-series':
        raise ValueError('Expected a native training series plan; prepare a new plan for this trainer')
    game=Path(plan['game'])
    baseline=Path(plan['baseline'])
    if checkpoint_identity(baseline)!=plan['baseline_identity']:
        raise ValueError('Frozen baseline identity changed')
    if identity(game)!=plan['source_identity']:
        raise ValueError('Training implementation changed since series preparation')
    def check_opponents():
        if checkpoint_identity(baseline)!=plan['baseline_identity']:
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
                                     env=dict(os.environ,BOPL_RL_GAME=str(game)))
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
            if identity(game)!=plan['source_identity']:
                raise ValueError('Training implementation changed during the series')
            check_opponents()
            phase='training'; status('running')
            execute([sys.executable,str(ROOT/'train/train.py'),'--output',str(directory),
                     '--steps',str(plan['steps']),'--workers',str(plan['workers']),'--seed',str(item['seed']),
                     '--encoder',plan['encoder'],'--banks',str(plan['banks']),
                     '--cudagraphs',str(plan['cudagraphs']),'--game',str(game),
                     '--episode-ticks',str(plan['episode_ticks']),
                     '--training-maps',','.join(map(str,plan.get('training_maps',[0]))),*evaluation_args,
                     '--frozen-games',str(plan.get('diagnostic_frozen_games',16)),
                     *(['--evaluation-opponents',plan['opponents']] if plan.get('opponents') else []),
                     '--checkpoint-every',str(plan['checkpoint_every']),
                     '--eval-every',str(plan['evaluation_every']),
                     '--evaluation-games',str(plan['diagnostic_games']),
                     '--heldout-games',str(plan['diagnostic_heldout_games']),
                     '--evaluation-workers',str(plan['evaluation_workers']),
                     '--evaluation-baseline',str(baseline)],out/(active+'-train.log'))
            result=json.loads((directory/'result.json').read_text())
            if result['status']!='complete' or result['learner_steps']<plan['steps']:
                raise RuntimeError(f'{active} did not complete its training budget')
            phase='final evaluation'; status('running')
            check_opponents()
            execute([sys.executable,str(HERE/'evaluate.py'),'--run',str(directory),
                     '--checkpoint',result['checkpoint'],'--initial',str(baseline),'--suite','final','--game',str(game),
                     *evaluation_args,'--frozen-games',str(plan.get('final_frozen_games',128)),
                     *(['--opponents',plan['opponents']] if plan.get('opponents') else []),
                     '--games-per-opponent',str(plan['final_games']),'--heldout-games',str(plan['final_heldout_games']),
                     '--seed-base',str(plan['final_seed_base']),'--heldout-seed-base',str(plan['final_heldout_seed_base']),
                     '--selfplay-seed',str(plan['final_selfplay_seed']),'--workers',str(plan['evaluation_workers'])],
                    out/(active+'-final-evaluation.log'))
            final=json.loads((directory/'final-evaluations'/f"{result['step']:09d}"/'result.json').read_text())
            completed.append(dict(name=active,seed=item['seed'],step=result['step'],learner_steps=result['learner_steps'],matches=result['matches'],
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
    p.add_argument('--baseline',type=Path,required=True); p.add_argument('--steps',type=int,default=10_000_000,help='Learner decisions per training seed')
    p.add_argument('--workers',type=int,default=16); p.add_argument('--seeds',type=lambda s:[int(x) for x in s.split(',')],default=[73,173,273])
    p.add_argument('--training-maps',type=map_ids,default=[0]);p.add_argument('--heldout-maps',type=map_ids,default=[1,2])
    p.add_argument('--opponents',type=Path)
    p.add_argument('--encoder',choices=('linear','set'),default='linear')
    p.add_argument('--banks',type=int,choices=range(2,9),default=4)
    p.add_argument('--game',type=Path)
    p.add_argument('--cudagraphs',type=int,default=1)
    p.add_argument('--episode-ticks',type=int,default=7200)
    p.add_argument('--checkpoint-every',type=int,default=1_000_000)
    p.add_argument('--evaluation-every',type=int,default=1_000_000)
    p.add_argument('--diagnostic-games',type=int,default=64);p.add_argument('--diagnostic-heldout-games',type=int,default=16)
    p.add_argument('--diagnostic-frozen-games',type=int,default=16);p.add_argument('--final-frozen-games',type=int,default=128)
    p.add_argument('--final-games',type=int,default=256);p.add_argument('--final-heldout-games',type=int,default=64)
    p=commands.add_parser('run'); p.add_argument('plan',type=Path)
    args=parser.parse_args()
    prepare(args) if args.command=='prepare' else run(args.plan)

if __name__=='__main__': main()
