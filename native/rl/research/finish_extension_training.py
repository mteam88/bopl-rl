"""Finish untouched trials after a recorded trial failure; never retry it."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from run_extension import verify,training_command
from train import ROOT,atomic_json


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();protocol=read(out/'protocol.json');verify(out,protocol)
    incidents=read(out/'training-incidents.json')['incidents'];failed={r['run'] for r in incidents};owned={};adopted={};started=time.monotonic()
    for run in protocol['runs']:
        directory=out/'runs'/run['name']
        if directory.exists() and not (directory/'result.json').exists() and run['name'] not in failed:
            status=read(directory/'status.json');pid=status['pid'];p=Path('/proc')/str(pid)
            assert p.exists() and str(directory).encode() in (p/'cmdline').read_bytes()
            adopted[run['name']]=pid
    while True:
        completed=[];active=[];pending=[]
        for run in protocol['runs']:
            name=run['name'];directory=out/'runs'/name
            if name in failed:continue
            if (directory/'result.json').exists():
                result=read(directory/'result.json');assert result['status']=='complete' and result['learner_steps']>=protocol['total_learner_target'];completed.append(name)
            elif name in owned or name in adopted:
                alive=owned[name][0].poll() is None if name in owned else (Path('/proc')/str(adopted[name])).exists()
                if alive:active.append(name)
                else:
                    status=read(directory/'status.json') if (directory/'status.json').exists() else {}
                    record=dict(run=name,kind='process_failure',last_completed_learner_steps=status.get('learner_steps'),note='Inspect the preserved training log; no automatic retry.')
                    incidents.append(record);failed.add(name);atomic_json(out/'training-incidents.json',dict(incidents=incidents))
            elif directory.exists():raise RuntimeError('Unknown incomplete run: '+name)
            else:pending.append(run)
        for run in pending[:max(0,2-len(active))]:
            verify(out,protocol);log=(out/(run['name']+'.train.log')).open('w')
            process=subprocess.Popen(training_command(out,protocol,run),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            owned[run['name']]=(process,log);active.append(run['name']);print('STARTED',run['name'],process.pid,flush=True)
        done=len(completed)+len(failed)==len(protocol['runs'])
        atomic_json(out/'training-status.json',dict(status='complete_with_failures' if done and failed else 'complete' if done else 'running',completed=completed,failed=sorted(failed),active=active,run=active[0] if active else None,parallel_jobs=2,elapsed=time.monotonic()-started))
        if done:break
        time.sleep(5)
    for process,log in owned.values():process.wait();log.close()
    print('ALL_TRIAL_ATTEMPTS_FINISHED',len(completed),'complete',sorted(failed),'interrupted',flush=True)


if __name__=='__main__':main()
