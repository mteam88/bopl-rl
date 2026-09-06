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
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    if args.steps<1 or args.workers<1 or not args.seeds or len(set(args.seeds))!=len(args.seeds):
        raise ValueError('Positive steps/workers and distinct training seeds required')
    if any(x<2 or x%2 for x in (args.diagnostic_games,args.diagnostic_heldout_games,args.final_games,args.final_heldout_games)):
        raise ValueError('Evaluation game counts must be positive seed pairs')
    baseline=args.baseline.resolve()
    plan=dict(schema=1,kind='bopl-training-series',name=out.name,
              created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              source_identity=identity(),steps=args.steps,workers=args.workers,
              runs=[dict(name=f'{out.name}-s{seed}',seed=seed) for seed in args.seeds],
              baseline=str(baseline),baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
              evaluation_milestones=[x for x in (1_000_000,3_000_000,args.steps) if x<=args.steps],
              diagnostic_games=args.diagnostic_games,diagnostic_heldout_games=args.diagnostic_heldout_games,evaluation_workers=4,
              final_games=args.final_games,final_heldout_games=args.final_heldout_games,final_seed_base=50001,
              final_heldout_seed_base=60001,final_selfplay_seed=70001,
              checkpoint_every=100_000,
              protocol='Fresh initialization per training seed; unchanged PPO/reward/observation/opponent settings; common frozen baseline; final seeds excluded from diagnostic evaluation.')
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
    completed=[]; process=None; stopping=False; started=time.monotonic()
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
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                status('stopping' if stopping else 'running',child_pid=process.pid)
                time.sleep(5)
            code=process.returncode; process=None
        if stopping: raise InterruptedError('Series stopped; unfinished runs are not silently resumed')
        if code: raise RuntimeError(f'Process exited {code}; see {log}')
    try:
        for item in plan['runs']:
            if stopping: raise InterruptedError('Series stopped')
            active=item['name']; directory=root/active
            if identity()!=plan['source_identity']:
                raise ValueError('Training implementation changed during the series')
            phase='training'; status('running')
            execute([sys.executable,str(HERE/'train.py'),'--output',str(directory),
                     '--steps',str(plan['steps']),'--workers',str(plan['workers']),'--seed',str(item['seed']),
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
            execute([sys.executable,str(HERE/'evaluate.py'),'--run',str(directory),
                     '--checkpoint',result['checkpoint'],'--initial',str(baseline),'--suite','final',
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

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare'); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True); p.add_argument('--steps',type=int,default=10_000_000)
    p.add_argument('--workers',type=int,default=16); p.add_argument('--seeds',type=lambda s:[int(x) for x in s.split(',')],default=[73,173,273])
    p.add_argument('--diagnostic-games',type=int,default=64);p.add_argument('--diagnostic-heldout-games',type=int,default=16)
    p.add_argument('--final-games',type=int,default=256);p.add_argument('--final-heldout-games',type=int,default=64)
    p=commands.add_parser('run'); p.add_argument('plan',type=Path)
    args=parser.parse_args()
    prepare(args) if args.command=='prepare' else run(args.plan)

if __name__=='__main__': main()
