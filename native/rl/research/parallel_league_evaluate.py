"""Run independent frozen suites in separate Python processes.

An optional, validated series parent is paused while its current evaluation
child finishes. The parent is resumed in finally and skips completed suites.
No training process is paused; all nine runs must already be complete.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_league import schedule, verify
from train import ROOT, atomic_json


def evaluator_child(parent):
    proc=Path('/proc')/str(parent)
    try:children=(proc/'task'/str(parent)/'children').read_text().split()
    except FileNotFoundError:return None
    for child in children:
        try:args=(Path('/proc')/child/'cmdline').read_bytes().decode().strip('\0').split('\0')
        except FileNotFoundError:continue
        if any(x.endswith('/league_evaluate.py') for x in args):
            return int(child),Path(args[args.index('--output')+1]).resolve()
    return None


def pause_while_evaluating(parent):
    """Pause only after confirming the parent is waiting on a live evaluator."""
    deadline=time.monotonic()+60
    proc=Path('/proc')/str(parent)
    stopped=False
    try:
        while time.monotonic()<deadline:
            if not proc.exists():return False,None
            if evaluator_child(parent):
                os.kill(parent,signal.SIGSTOP);stopped=True
                for _ in range(100):
                    status=(proc/'status').read_text()
                    if any(line.startswith('State:') and 'T' in line for line in status.splitlines()):break
                    time.sleep(.01)
                child=evaluator_child(parent)
                if child:return True,child
                os.kill(parent,signal.SIGCONT);stopped=False
            time.sleep(.1)
        raise TimeoutError('Series runner did not enter a stable evaluation wait')
    except BaseException:
        if stopped:
            try:os.kill(parent,signal.SIGCONT)
            except ProcessLookupError:pass
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--runner-pid',type=int)
    args=parser.parse_args();out=args.output.resolve()
    if min(args.jobs,args.workers)<1:parser.error('Positive concurrency required')
    protocol=json.loads((out/'protocol.json').read_text());verify(out,protocol)
    jobs=[]
    for run in protocol['runs']:
        directory=out/'runs'/run['name']
        result=json.loads((directory/'result.json').read_text())
        if result['status']!='complete' or result['learner_steps']<protocol['learner_decisions']:
            raise ValueError('All training must finish before parallel evaluation')
        checkpoint=directory/'checkpoints'/Path(result['checkpoint']).name
        for suite in ('development','final'):
            jobs.append((run['name'],suite,checkpoint))
    if args.runner_pid:
        proc=Path('/proc')/str(args.runner_pid)
        if (proc/'cwd').resolve()!=ROOT or not any(x.endswith(b'/run_league.py') for x in (proc/'cmdline').read_bytes().split(b'\0')):
            raise ValueError('PID is not this workspace\'s series runner')
    started=time.monotonic();paused=False;completed=[];active_child=None

    def work(job):
        name,suite,checkpoint=job
        destination=out/'evaluations'/name/suite
        # A suite already owned by the original runner keeps that ownership.
        owned=active_child is not None and destination==active_child[1]
        if owned or destination.exists() and not (destination/'result.json').exists():
            deadline=time.monotonic()+1800
            while not (destination/'result.json').exists():
                if time.monotonic()>deadline:raise TimeoutError(f'Existing evaluation did not finish: {name}/{suite}')
                if owned:
                    proc=Path('/proc')/str(active_child[0])/'status'
                    if not proc.exists() or any(line.startswith('State:') and 'Z' in line for line in proc.read_text().splitlines()):
                        raise RuntimeError('Original evaluator exited without a complete result')
                time.sleep(1)
        if not (destination/'result.json').exists():
            verify(out,protocol)
            plan=out/(name+'-'+suite+'-schedule.json')
            atomic_json(plan,schedule(protocol,suite,checkpoint))
            with (out/(name+'-'+suite+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name('league_evaluate.py')),
                                '--schedule',str(plan),'--output',str(destination),'--workers',str(args.workers)],
                               stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        result=json.loads((destination/'result.json').read_text())
        spec=protocol[suite]
        if len(result['games'])!=len(spec['maps'])*len(spec['bots'])*spec['pairs_per_cell']*2:
            raise ValueError('Incomplete suite')
        return dict(run=name,suite=suite,games=len(result['games']))

    try:
        if args.runner_pid:
            paused,active_child=pause_while_evaluating(args.runner_pid)
        atomic_json(out/'parallel-status.json',dict(status='running',jobs=args.jobs,workers_per_job=args.workers,
                    elapsed=time.monotonic()-started,completed=[]))
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures={executor.submit(work,job):job for job in jobs}
            for future in as_completed(futures):
                record=future.result();completed.append(record)
                atomic_json(out/'parallel-status.json',dict(status='running',jobs=args.jobs,workers_per_job=args.workers,
                            elapsed=time.monotonic()-started,completed=completed))
                print(json.dumps(record),flush=True)
        atomic_json(out/'parallel-status.json',dict(status='complete',jobs=args.jobs,workers_per_job=args.workers,
                    elapsed=time.monotonic()-started,completed=completed))
    except BaseException as error:
        atomic_json(out/'parallel-status.json',dict(status='failed',error=str(error),completed=completed))
        raise
    finally:
        if paused:
            try:os.kill(args.runner_pid,signal.SIGCONT)
            except ProcessLookupError:pass


if __name__=='__main__':main()
