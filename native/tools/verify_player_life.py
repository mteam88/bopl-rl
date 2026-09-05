"""Compare Player.Kill accounting and event order with the pinned Unity method."""
import argparse
import gzip
import json
import random
import subprocess
from pathlib import Path
from collections import Counter
from reference import ROOT, first_difference, provenance, read_trace
from verify_beam import encoded, reference

FIXTURE=ROOT/'native/fixtures/player-life.jsonl.gz'
SCHEMA='bopl-player-life-v1'


def queries(count, seed):
    rng=random.Random(seed)
    for case in range(count):
        ids=rng.sample(range(1,7),case%6+1)
        players=[]
        for i in ids:
            players.append(dict(id=i,team=rng.randrange(3),bodies_alive=rng.choice([1,1,1,2,3,6,0,-1,2147483647,-2147483648]),
                won_round=rng.randrange(9)==0,kills=rng.choice([0,1,20,2147483647,-2147483648]),deaths=rng.choice([0,1,5,2147483647,-2147483648]),
                still_alive_round=bool(rng.randrange(2)),killed_at=rng.choice([0,17,9223372036854775807]),cause=rng.randrange(6),rumble=rng.randrange(4)))
        state=dict(players=players,registered_kills=[rng.choice([0,1,2147483647]) for _ in range(6)],events=[])
        actions=[]
        for j in range(32):
            victim=j%len(players) if j<12 else rng.randrange(len(players))
            actions.append(dict(victim=victim,killer=ids[victim] if j%4==0 else rng.choice([*range(1,7),1000,0,-1,2147483647]),
                timestamp=rng.choice([j,1000+j,9223372036854775807,-9223372036854775808]),cause=rng.randrange(6)))
        yield dict(operation='dynamics_player_life',state=state,actions=actions)


def check(records):
    process=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=encoded(r['query'] for r in records),text=True,capture_output=True,check=True)
    output=[json.loads(l) for l in process.stdout.splitlines()]
    if len(output)!=len(records):raise ValueError('Incomplete death-accounting output')
    counts=Counter();failures=[]
    for index,(record,actual) in enumerate(zip(records,output)):
        expected=record['expected'];before=record['query']['state']
        counts['actions']+=len(expected)
        for action,after in zip(record['query']['actions'],expected,strict=True):
            player=before['players'][action['victim']]
            if player['won_round']:counts['winner_noops']+=1
            else:
                counts['final_body_deaths' if player['bodies_alive']==1 else 'other_body_deaths']+=1
                counts['self_kills']+=action['killer']==player['id']
                counts['missing_killers']+=not any(p['id']==action['killer'] for p in before['players'])
            counts['death_events']+=len(after['events'])-len(before['events'])
            before=after
        if actual!=expected:
            counts['mismatches']+=1
            if len(failures)<10:failures.append(dict(case=index,difference=first_difference(expected,actual)))
    return dict(match=not counts['mismatches'],cases=len(records),**counts,first_failures=failures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,default=1200)
    parser.add_argument('--seed',type=int,default=761338)
    parser.add_argument('--golden',action='store_true')
    parser.add_argument('--gameplay',action='store_true')
    parser.add_argument('--traces',type=Path,nargs='+')
    parser.add_argument('--reuse-unity',action='store_true')
    args=parser.parse_args()
    fixture=ROOT/'native/fixtures/player-life-gameplay.jsonl.gz' if args.gameplay or args.traces else FIXTURE
    if args.traces:
        records=[];sources=[]
        for path in args.traces:
            source=provenance(path);count=0
            for row in read_trace(path):
                if row['kind']=='state':
                    stages=row['value'].get('playerLifeStages',[])
                    records.extend(dict(stage,source=len(sources),tick=row['value']['tick']) for stage in stages)
                    count+=len(stages)
            if not count:raise ValueError(f'No Player.Kill calls captured in {path}')
            sources.append(dict(source,accounting_calls=count))
        report=dict(check(records),provenance=sources)
        if report['match']:
            header=dict(schema=SCHEMA,cases=len(records),provenance=sources)
            fixture.write_bytes(gzip.compress(encoded([header,*records]).encode(),mtime=0))
        directory=ROOT/'native/artifacts/player-life-gameplay';directory.mkdir(parents=True,exist_ok=True)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    elif args.golden:
        with gzip.open(fixture,'rt') as stream:
            header=json.loads(next(stream));records=[json.loads(line) for line in stream]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid player-life fixture')
        report=check(records)
    else:
        if args.cases<1:parser.error('--cases must be positive')
        directory=ROOT/'native/artifacts/player-life'
        records,hashes=reference(directory,list(queries(args.cases,args.seed)),args.reuse_unity)
        report=dict(check(records),provenance=hashes)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            indices=sorted(set(range(min(24,len(records))))|set(random.Random(args.seed).sample(range(len(records)),min(176,len(records)))))
            selected=[records[i] for i in indices]
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes)
            fixture.write_bytes(gzip.compress(encoded([header,*selected]).encode(),mtime=0))
    print(json.dumps(report,indent=2))
    return int(not report['match'])


if __name__=='__main__':raise SystemExit(main())
