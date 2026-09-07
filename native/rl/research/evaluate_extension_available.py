"""Complete all available fixed suites and a declared common-checkpoint fallback."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
from pathlib import Path
import subprocess
import sys
import time
from continuation import digest
from evaluate_extension import checkpoint,jobs,plan,sources
from run_extension import verify
from train import ROOT,atomic_json


def read(path):return json.loads(path.read_text())


def count(protocol,job):
    spec=protocol['crossplay'] if job['kind']=='crossplay' else protocol[job['suite']]
    return len(spec['maps'])*spec['pairs_per_cell']*2*(1 if job['kind']=='crossplay' else len(spec['bots']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.jobs<1 or args.workers<1:parser.error('Positive jobs and workers required')
    out=args.output.resolve();protocol=read(out/'protocol.json');verify(out,protocol)
    assert digest(out/'protocol.json')==read(out/'capacity-fallback-protocol.json')['primary_protocol_sha256']
    queue=jobs(protocol);completed=[];unavailable=[];active={};fallback_added=False;started=time.monotonic()
    def work(job):
        directory=out/'evaluations'/job['name'];expected=plan(out,protocol,job)
        if not (directory/'result.json').exists():
            if directory.exists():return dict(**job,games=count(protocol,job),reason='Preserved incomplete evaluation; no automatic retry')
            verify(out,protocol);schedule=out/'schedules'/(job['name']+'.json');atomic_json(schedule,expected)
            with (out/(job['name']+'.eval.log')).open('w') as log:
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('league_evaluate.py')),'--schedule',str(schedule),'--output',str(directory),'--workers',str(args.workers)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:return dict(**job,games=count(protocol,job),reason=f'Evaluation process exited {result.returncode}; raw partial data preserved')
        report=read(directory/'result.json');planned={t['name']:t for t in expected['tasks']}
        assert len(report['games'])==len(planned)
        for game in report['games']:
            for key in ('seed','side','map','learner','opponent'):assert game[key]==planned[game['name']][key]
        return dict(**job,games=len(report['games']))
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        while queue or active or not fallback_added:
            training=read(out/'training-status.json');failed=set(training.get('failed',[]));terminal=training['status'] in ('complete','complete_with_failures')
            if terminal and not fallback_added:
                common=max(m for m in (1000000,2000000,3000000) if all(checkpoint(out,protocol,r,m).exists() for r in protocol['runs']))
                fallback=dict(milestone=common,selection_protocol_sha256=digest(out/'capacity-fallback-protocol.json'),source_sha256={r['name']:digest(checkpoint(out,protocol,r,common)) for r in protocol['runs']})
                atomic_json(out/'fallback-checkpoint.json',fallback)
                if common>1000000:
                    for a in protocol['crossplay']['seeds']:
                        for b in protocol['crossplay']['seeds']:queue.append(dict(name=f'crossplay-{common}-A{a}-B{b}',kind='crossplay',milestone=common,a=a,b=b,supplemental=True))
                    for r in protocol['runs']:queue.append(dict(name=f"{r['name']}-{common}-transfer",kind='script',milestone=common,run=r['name'],suite='transfer',supplemental=True))
                fallback_added=True
            for job in queue.copy():
                paths=sources(out,protocol,job);missing=[p for p in paths if not p.exists()]
                if missing and (terminal or any(p.parent.parent.name in failed for p in missing)):
                    unavailable.append(dict(**job,games=count(protocol,job),reason='Required checkpoint unavailable after interrupted training'));queue.remove(job);continue
                if not missing and len(active)<args.jobs:active[executor.submit(work,job)]=job;queue.remove(job)
            if active:
                done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for future in done:
                    result=future.result();active.pop(future)
                    (unavailable if 'reason' in result else completed).append(result);print(json.dumps(result),flush=True)
            else:time.sleep(5)
            finished=not queue and not active and fallback_added
            state='complete_with_omissions' if finished and unavailable else 'complete' if finished else 'running'
            atomic_json(out/'evaluation-status.json',dict(status=state,elapsed=time.monotonic()-started,completed=completed,unavailable=unavailable,active=list(active.values()),pending=len(queue),fallback_added=fallback_added,expected_suites=len(completed)+len(unavailable)+len(active)+len(queue),expected_games=sum(j['games'] for j in completed+unavailable)+sum(count(protocol,j) for j in list(active.values())+queue)))
    print('ALL_AVAILABLE_EVALUATIONS_COMPLETE',len(completed),'complete',len(unavailable),'unavailable',flush=True)


if __name__=='__main__':main()
