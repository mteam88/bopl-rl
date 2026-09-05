"""Verify PlayerCollision selection with real components and recording effect sinks."""
import argparse
import hashlib
from pathlib import Path
import gzip
import json
import random
import subprocess
from collections import Counter
from reference import ROOT,first_difference,provenance,read_trace
from verify_beam import encoded,reference
from player_collision_cases import queries

FIXTURE=ROOT/'native/fixtures/player-collision.jsonl.gz'
SCHEMA='bopl-player-collision-v1'


def check(records,native_path=None):
    process=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=encoded(r['query'] for r in records),text=True,capture_output=True,check=True)
    if native_path:native_path.write_text(process.stdout)
    outputs=[json.loads(l) for l in process.stdout.splitlines()]
    if len(outputs)!=len(records):raise ValueError('Incomplete collision-policy output')
    counts=Counter();effects=Counter();causes=Counter();errors=Counter();failures=[]
    for index,(record,actual) in enumerate(zip(records,outputs)):
        expected=record['expected']
        if isinstance(expected,list):
            counts['complete_scenarios']+=1
            frames=expected
        else:
            counts['source_errors']+=1;errors[expected['error']]+=1
            frames=expected['completed']
            counts['failed_actions']+=1
        counts['completed_actions']+=len(frames)
        for frame in [*frames,*([expected['failed']] if isinstance(expected,dict) else [])]:
            for effect in frame['effects']:
                effects[effect['kind']]+=1
                if effect['kind']=='kill':causes[str(effect['cause'])]+=1
        if actual!=expected:
            counts['mismatches']+=1
            if len(failures)<10:
                difference=first_difference(expected,actual)
                compact=lambda value:dict(type=type(value).__name__,length=len(value)) if isinstance(value,(list,dict)) else value
                failures.append(dict(case=index,difference=[difference[0],compact(difference[1]),compact(difference[2])]))
    return dict(match=not counts['mismatches'],cases=len(records),**counts,effects=dict(effects),error_categories=dict(errors),kill_causes=dict(causes),first_failures=failures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,default=600);parser.add_argument('--seed',type=int,default=278844)
    parser.add_argument('--golden',action='store_true');parser.add_argument('--reuse-unity',action='store_true')
    parser.add_argument('--traces',type=Path,nargs='+');parser.add_argument('--failure',type=Path)
    parser.add_argument('--gameplay',action='store_true')
    args=parser.parse_args()
    fixture=ROOT/'native/fixtures/player-collision-gameplay.jsonl.gz' if args.gameplay or args.traces or args.failure else FIXTURE
    if args.traces or args.failure:
        records=[];sources=[]
        def append(stages,source):
            if not stages:raise ValueError('No collision decisions captured')
            for stage in stages:
                if 'capture_error' in stage:raise ValueError(stage['capture_error'])
                records.append(dict(stage,source=len(sources)))
            sources.append(source)
        for path in args.traces or []:
            source=provenance(path);stages=[]
            for row in read_trace(path):
                if row['kind']=='state':stages.extend(dict(stage,tick=row['value']['tick']) for stage in row['value'].get('playerCollisionStages',[]))
            append(stages,source)
        if args.failure:
            source=json.loads(args.failure.with_suffix('.provenance.json').read_text())
            if source['failure_sha256']!=hashlib.sha256(args.failure.read_bytes()).hexdigest():raise ValueError('Failure snapshot provenance mismatch')
            fault=json.loads(args.failure.read_text())
            append(fault['playerCollisionStages'],source)
        directory=ROOT/'native/artifacts/player-collision-gameplay';directory.mkdir(parents=True,exist_ok=True)
        report=dict(check(records,directory/'native.jsonl'),provenance=sources)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            header=dict(schema=SCHEMA,cases=len(records),provenance=sources,scope='OnCollide decisions; downstream handler changes projected out')
            fixture.write_bytes(gzip.compress(encoded([header,*records]).encode(),mtime=0))
    elif args.golden:
        with gzip.open(fixture,'rt') as stream:header=json.loads(next(stream));records=[json.loads(l) for l in stream]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid collision policy fixture')
        report=check(records)
    else:
        if args.cases<1:parser.error('--cases must be positive')
        directory=ROOT/'native/artifacts/player-collision'
        records,hashes=reference(directory,list(queries(args.cases,args.seed)),args.reuse_unity)
        report=dict(check(records,directory/"native.jsonl"),provenance=hashes)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            selected=set(range(min(24,len(records))))|set(random.Random(args.seed).sample(range(len(records)),min(150,len(records))))
            selected.update(i for i,r in enumerate(records) if not isinstance(r['expected'],list))
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,scope='collision decisions; downstream kill/attach handlers replaced by recording sinks')
            fixture.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])


if __name__=='__main__':raise SystemExit(main())
