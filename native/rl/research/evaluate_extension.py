"""Evaluate frozen milestone jobs as atomic checkpoints become available."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from train import ROOT, atomic_json
from continuation import digest
from run_extension import verify
from run_league import schedule


def checkpoint(out,protocol,run,milestone):
    if milestone==1000000:return ROOT/run['source_checkpoint']
    learner_per_update=protocol['workers']*2*protocol['horizon']*3//4
    step=math.ceil(milestone/learner_per_update)*protocol['workers']*2*protocol['horizon']
    return out/'runs'/run['name']/'checkpoints'/f'policy-{step:09d}.pt'


def jobs(protocol):
    result=[]
    for milestone in protocol['crossplay']['milestones']:
        for a in protocol['crossplay']['seeds']:
            for b in protocol['crossplay']['seeds']:
                result.append(dict(name=f'crossplay-{milestone}-A{a}-B{b}',kind='crossplay',milestone=milestone,a=a,b=b))
    for run in protocol['runs']:
        for milestone in (2000000,3000000,4000000):
            for suite in (('development','transfer') if milestone==4000000 else ('development',)):
                result.append(dict(name=f"{run['name']}-{milestone}-{suite}",kind='script',run=run['name'],milestone=milestone,suite=suite))
    return result


def sources(out,protocol,job):
    runs={r['name']:r for r in protocol['runs']}
    names=[job['run']] if job['kind']=='script' else [f"A-s{job['a']}",f"B-s{job['b']}"]
    return [checkpoint(out,protocol,runs[name],job['milestone']) for name in names]


def plan(out,protocol,job):
    paths=sources(out,protocol,job)
    if job['kind']=='script':return schedule(protocol,job['suite'],paths[0])
    a,b=[dict(kind='policy',path=str(p.resolve()),sha256=digest(p)) for p in paths]
    spec=protocol['crossplay'];tasks=[]
    ai=spec['seeds'].index(job['a']);bi=spec['seeds'].index(job['b'])
    for m in spec['maps']:
        for pair in range(spec['pairs_per_cell']):
            for side in (0,1):
                name=f"A{job['a']}-B{job['b']}-m{m}-p{pair}-s{side}"
                tasks.append(dict(name=name,group=f"A{job['a']}-B{job['b']}-m{m}",map=m,side=side,
                                  seed=spec['seed_base']+ai*100000+bi*10000+m*100+pair,
                                  learner=a,opponent=b,save=pair==0))
    return dict(schema=1,limit=protocol['episode_ticks'],tasks=tasks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if min(args.jobs,args.workers)<1:parser.error('Positive workers required')
    out=args.output.resolve();protocol=json.loads((out/'protocol.json').read_text());verify(out,protocol)
    queue=jobs(protocol);completed=[];active={};started=time.monotonic()
    (out/'schedules').mkdir(exist_ok=True);(out/'evaluations').mkdir(exist_ok=True)

    def work(job):
        destination=out/'evaluations'/job['name'];expected=plan(out,protocol,job)
        if not (destination/'result.json').exists():
            if destination.exists():raise ValueError(f'Incomplete evaluation needs inspection: {destination}')
            verify(out,protocol);schedule_path=out/'schedules'/(job['name']+'.json')
            atomic_json(schedule_path,expected)
            with (out/(job['name']+'.eval.log')).open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name('league_evaluate.py')),
                                '--schedule',str(schedule_path),'--output',str(destination),'--workers',str(args.workers)],
                               stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        result=json.loads((destination/'result.json').read_text())
        assert len(result['games'])==len(expected['tasks'])
        planned={t['name']:t for t in expected['tasks']}
        for game in result['games']:
            task=planned[game['name']]
            for key in ('learner','opponent','seed','side','map'):assert game[key]==task[key]
        return dict(**job,games=len(result['games']))

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            while queue or active:
                for job in queue.copy():
                    if len(active)>=args.jobs:break
                    if all(p.exists() for p in sources(out,protocol,job)):
                        active[executor.submit(work,job)]=job;queue.remove(job)
                if active:
                    done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                    for future in done:
                        record=future.result();completed.append(record);active.pop(future)
                        print(json.dumps(record),flush=True)
                else:time.sleep(5)
                training=out/'training-status.json'
                if training.exists() and json.loads(training.read_text())['status']=='failed':
                    raise RuntimeError('Training failed; inspect training-status.json')
                if training.exists() and json.loads(training.read_text())['status']=='complete' and queue and not active:
                    raise RuntimeError('Training complete but a scheduled checkpoint is missing')
                atomic_json(out/'evaluation-status.json',dict(status='running',elapsed=time.monotonic()-started,
                            completed=completed,active=list(active.values()),pending=len(queue)))
        atomic_json(out/'evaluation-status.json',dict(status='complete',elapsed=time.monotonic()-started,completed=completed,pending=0,active=[]))
    except BaseException as error:
        atomic_json(out/'evaluation-status.json',dict(status='failed',error=str(error),elapsed=time.monotonic()-started,completed=completed));raise


if __name__=='__main__':main()
