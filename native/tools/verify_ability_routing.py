"""Ordinary ability cooldown/selection gates in the scheduled Slime simulation.

Casts are deliberately blocked by CanUseAbilities or released buttons. Direct
source predicate checks include true ready/castable results. This corpus does
not substitute a fake successful cast for an unimplemented ability handler.
"""
import argparse
import gzip
import json
import random
from collections import Counter
from reference import ROOT
from verify_beam import reference,encoded
from verify_actors import check,ONE,DT
from verify_ragdoll import queries as actor_queries

SCHEMA='bopl-ability-routing-v1'

def queries(count,seed,steps):
    rng=random.Random(seed+729)
    for case,q in enumerate(actor_queries(count,seed,steps)):
        q['routing_checks']=True
        for i,a in enumerate(q['runtime']['actors']):
            n=(case+i)%7
            pool=[dict(present=True,destroyed=False,cooldown=rng.choice([0,DT,DT-1,DT+1,ONE,5*ONE,-DT]),stopped_during_time_stop=bool(rng.randrange(2)),ground_only=bool(rng.randrange(2)),castable=bool(rng.randrange(2)),ignore_next_cooldown=bool(rng.randrange(2))) for _ in range(n)]
            slots=list(range(n))
            if n>1 and case%5==0:slots[-1]=slots[0]
            if n and case%11==0:pool[-1]['destroyed']=True
            if n and case%13==0:pool[-1]['present']=False
            if n and case%17==0:slots[-1]=None
            timers=[p['cooldown']+rng.choice([-DT,-1,0,1,DT]) for p in pool]
            if case%23==0 and n:timers.pop()
            indicators=[dict(seconds_since_reset=rng.choice([0.,11.,10000.,-1.])) for _ in range((case+i)%5)]
            if indicators and case%19==0:indicators[-1]=None
            state=dict(slots=slots,pool=pool,timers=timers,indicators=indicators,in_ability=case%7==0,can_use=False,buttons=[False]*3)
            a['slime']['abilities']=state
        for tick,action in enumerate(q['actions']):
            values=[]
            for actor in q['runtime']['actors']:
                # Predicate checks exercise positive casts independently; main
                # simulation preserves the real no-cast branch and its ordering.
                can_use=(case+tick)%3==0
                values.append(dict(can_use=can_use,buttons=[not can_use and bool(rng.randrange(2)) for _ in range(3)]))
            action['routing_inputs']=values
        yield q

def coverage(records):
    predicates=Counter();states=Counter();changed=0
    for r in records:
        expected=r['expected'];frames=expected if isinstance(expected,list) else expected['completed']
        previous=[a['slime']['abilities']['timers'] for a in r['query']['runtime']['actors']]
        for frame in frames:
            for checks in frame['routing_checks']:
                for item in checks:
                    for name,value in item.items():predicates[name+':'+(str(value).lower() if isinstance(value,bool) else value['error'])]+=1
            for i,a in enumerate(frame['actors']):
                state=a['slime']['abilities'];states['in_ability' if state['in_ability'] else 'ordinary']+=1
                states['ragdoll']+=a['slime']['state']['mode']==10
                changed+=sum(x!=y for x,y in zip(previous[i],state['timers']));previous[i]=state['timers']
    return dict(predicates=dict(predicates),routing_states=dict(states),timer_changes=changed)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=120);p.add_argument('--steps',type=int,default=120);p.add_argument('--seed',type=int,default=472193);p.add_argument('--golden',action='store_true');p.add_argument('--reuse-unity',action='store_true');a=p.parse_args()
    fixture=ROOT/'native/fixtures/ability-routing.jsonl.gz'
    if a.golden:
        with gzip.open(fixture,'rt') as f:header=json.loads(next(f));records=[json.loads(line) for line in f]
        assert header['schema']==SCHEMA and header['cases']==len(records)
        report=dict(check(records),**coverage(records))
    else:
        directory=ROOT/'native/artifacts/ability-routing';records,hashes=reference(directory,list(queries(a.cases,a.seed,a.steps)),a.reuse_unity)
        report=dict(check(records),**coverage(records),provenance=hashes);(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            selected=set(range(min(12,len(records))))|set(random.Random(a.seed).sample(range(len(records)),min(24,len(records))))
            selected.update(i for i,r in enumerate(records) if isinstance(r['expected'],dict))
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,scope='Ordinary ability cooldowns, no-cast selection path and direct ready/castable predicates; headless indicators, no rope abilities or ability entry')
            fixture.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])
if __name__=='__main__':raise SystemExit(main())
