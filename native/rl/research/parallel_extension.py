"""Take over a waiting sequential parent and overlap independent fixed runs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from run_extension import verify, training_command
from train import ROOT, atomic_json


def read(path):
    return json.loads(path.read_text())


def process_command(pid):
    return (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runner-pid',type=int,required=True)
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if args.jobs not in (1,2):parser.error('At most two training runs on this host')
    out=args.output.resolve();protocol=read(out/'protocol.json');verify(out,protocol)
    parent=Path('/proc')/str(args.runner_pid)
    command=process_command(args.runner_pid)
    assert any(x.endswith(b'/run_extension.py') for x in command),command
    assert str(out).encode() in command or str(out.relative_to(ROOT)).encode() in command
    assert (parent/'cwd').resolve()==ROOT
    state=read(out/'training-status.json');assert state['status']=='running'
    existing=out/'runs'/state['run'];status=read(existing/'status.json');child=status['pid']
    children=[int(x) for x in (parent/'task'/str(args.runner_pid)/'children').read_text().split()]
    assert child in children and str(existing).encode() in process_command(child)
    assert any(x.endswith(b'/train.py') for x in process_command(child))
    before=read(existing/'status.json');paused=False;owned={};started=time.monotonic()
    try:
        os.kill(args.runner_pid,signal.SIGSTOP);paused=True
        deadline=time.monotonic()+5
        while True:
            line=next(x for x in (parent/'status').read_text().splitlines() if x.startswith('State:'))
            if '\tT' in line:break
            if time.monotonic()>deadline:raise RuntimeError('Could not confirm stopped queue parent')
            time.sleep(.02)
        # The child keeps training; only the sequential queue is stopped.
        assert str(existing).encode() in process_command(child)
        while True:
            completed=[];active=[];pending=[]
            for run in protocol['runs']:
                directory=out/'runs'/run['name'];result=directory/'result.json'
                if result.exists():
                    r=read(result);assert r['status']=='complete' and r['learner_steps']>=protocol['total_learner_target']
                    completed.append(run['name'])
                elif directory.exists():
                    if run['name']!=state['run'] and run['name'] not in owned:raise RuntimeError('Unexpected incomplete run: '+run['name'])
                    s=read(directory/'status.json') if (directory/'status.json').exists() else {'status':'starting'}
                    if s['status'] in ('failed','interrupted'):raise RuntimeError('Failed run: '+run['name'])
                    if run['name']==state['run'] and not (Path('/proc')/str(child)).exists():raise RuntimeError('Original child disappeared without a result')
                    active.append(run['name'])
                elif run['name'] in owned:
                    if owned[run['name']][0].poll() is not None:raise RuntimeError('Child exited before creating its run: '+run['name'])
                    active.append(run['name'])
                else:pending.append(run)
            for name,(process,log) in list(owned.items()):
                code=process.poll()
                if code is not None:
                    log.close()
                    if code:raise RuntimeError(f'{name} exited {code}')
                    if name not in completed:raise RuntimeError(f'{name} exited without a result')
            limit=args.jobs
            control=out/'training-concurrency.json'
            if control.exists():limit=int(read(control)['jobs'])
            if limit not in (1,2):raise ValueError('Concurrency control must be 1 or 2')
            for run in pending[:max(0,limit-len(active))]:
                verify(out,protocol);log=(out/(run['name']+'.train.log')).open('w')
                process=subprocess.Popen(training_command(out,protocol,run),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                owned[run['name']]=(process,log);active.append(run['name'])
                print('STARTED',run['name'],process.pid,flush=True)
            record=dict(status='running' if len(completed)<len(protocol['runs']) else 'complete',run=active[0] if active else None,
                        active=active,completed=completed,parallel_jobs=limit,elapsed=time.monotonic()-started,
                        resumed_from_elapsed=state['elapsed'],sequential_parent_pid=args.runner_pid)
            atomic_json(out/'parallel-training-status.json',record)
            atomic_json(out/'training-status.json',record)
            if len(completed)==len(protocol['runs']):break
            time.sleep(5)
    except BaseException as error:
        atomic_json(out/'parallel-training-status.json',dict(status='failed',error=str(error),elapsed=time.monotonic()-started))
        atomic_json(out/'training-status.json',dict(status='failed',error=str(error)))
        raise
    finally:
        # Preserve in-progress child work before allowing the old queue to proceed.
        for process,log in owned.values():process.wait();log.close()
        if paused and parent.exists():os.kill(args.runner_pid,signal.SIGCONT)
    print('PARALLEL_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
