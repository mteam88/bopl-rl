"""Ragdoll impulses/recovery inside the composed SlimeController simulation."""
import argparse
import copy
import gzip
import json
import random
from collections import Counter
from reference import ROOT
from verify_beam import encoded,reference
from verify_actors import check,vec,ONE,DT
from verify_slime_actors import queries as slime_queries,coverage
from verify_animator import asset_queries

SCHEMA='bopl-ragdoll-v1'
def ragdoll_animations():
    data,_=asset_queries();out={}
    for q in data:
        name=q['asset'].get('animation')
        if q['asset']['path'].endswith('/MCAnimData.asset') and name in ['ragdollFast','ragdollSlow'] and q['actions'][0]['op']=='begin':out[name]=q['actions'][0]['frames']
    assert len(out)==2;return out

def queries(count,seed,steps):
    animations=ragdoll_animations();rng=random.Random(seed+91)
    for case,q in enumerate(slime_queries(count,seed,steps)):
        actors=q['runtime']['actors'];n=len(actors);q['scripts']={}
        for i,a in enumerate(actors):
            a['slime']['animations'].update(copy.deepcopy(animations))
            duration=[0,DT-1,DT,DT+1,ONE//4,ONE,-DT][(case+i)%7]
            a['slime']['ragdoll']=dict(duration=duration,time=duration+[-1,0,1,-DT][(case+i)%4],speed=vec(3,-4),magnet_holder=None if case%3==0 else i%2)
            if case%3==0:a['slime']['state']['mode']=10
            if case%17==0:
                ground=a['controller']['physics']['motion']['ground']
                if ground is not None:q['runtime']['bindings'][ground]['platform']['destroyed']=True
            if case%19==0:a['controller']['physics']['motion'].update(grounded=True,ground=None)
        def command(actor):
            return dict(op='ragdoll',actor=actor,request=dict(duration=rng.choice([0,DT-1,DT,DT+1,ONE//4,ONE,-DT]),
                speed=rng.choice([vec(),vec(30,20),vec(-30,-20),vec(.0001,-.0002),vec(0,50),vec(100,-100)]),fast=bool(rng.randrange(2)),reset_animation=bool(rng.randrange(2)),magnet=rng.choice([None,0,1,2])))
        # Apply before a tick, and between different components of different
        # actors. Effects are not synchronized to a player-by-player loop.
        for tick,action in enumerate(q['actions']):
            if tick%13==0:action['commands']=[command(rng.randrange(n)) for _ in range(1+case%3)]
            if tick%17==0:action['dt']=[0,DT,DT*2,-DT][(case+tick//17)%4]
        for key in ['init:0','player_inputs:0','update:2','late:0','physics:*']:
            q['scripts'][key]=[([command(rng.randrange(n))] if j in [0,2,7] else []) for j in range(8)]
        yield q

def fault_queries(seed):
    base=list(queries(2,seed,120))[1]
    for name in ['ragdollFast','ragdollSlow','fallLeftLoop','renderer']:
        q=copy.deepcopy(base);q['fault']=name;q['scripts']={}
        for action in q['actions']:action['commands']=[]
        a=q['runtime']['actors'][0];a['slime']['state'].update(mode=2,has_spawned=True,suppress=False)
        a['slime']['ragdoll'].update(duration=0,time=0,magnet_holder=0)
        if name=='renderer':a['slime']['animator']['render']=None
        else:del a['slime']['animations'][name]
        request=dict(duration=0,speed=vec(9,-12),fast=name=='ragdollFast',reset_animation=name in ['ragdollFast','ragdollSlow'],magnet=2)
        command=dict(op='ragdoll',actor=0,request=request)
        q['scripts']['update:1']=[[command]]
        for action in q['actions']:action.update(scale=ONE,dt=DT,enabled=[True]*len(q['runtime']['components']))
        # Ensure the recovery happens inside the first Slime UpdateSim, before
        # any unrelated platform attachment or animation can end ragdoll.
        q['runtime']['scheduler']['pending']['items']=list(range(len(q['runtime']['components'])))
        for i,c in enumerate(q['runtime']['scheduler']['components']):c['hierarchy']=i
        yield q

def ragdoll_coverage(records):
    effects=Counter();counts=Counter()
    for r in records:
        e=r['expected'];frames=e if isinstance(e,list) else e['completed'];previous=[a['slime']['ragdoll']['magnet_holder'] for a in r['query']['runtime']['actors']]
        for frame in frames:
            for command in frame.get('effects',[]):
                request=command['request'];effects['applied']+=1;effects['reset_animation' if request['reset_animation'] else 'retained_animation']+=1;effects['fast' if request['fast'] else 'slow']+=1
                effects['zero_speed' if request['speed']==vec() else 'nonzero_speed']+=1
                effects['magnet_argument']+=request['magnet'] is not None
            for i,a in enumerate(frame['actors']):
                s=a['slime'];rag=s['ragdoll'];counts['ragdoll_frames']+=s['state']['mode']==10
                counts['ragdoll_with_callback']+=s['state']['mode']==10 and s['animator']['callback'] is not None
                counts['grounded_ragdoll_frames']+=s['state']['mode']==10 and a['controller']['physics']['motion']['grounded']
                counts['magnet_cleared']+=previous[i] is not None and rag['magnet_holder'] is None
                previous[i]=rag['magnet_holder']
    return dict(ragdoll=dict(counts),effects=dict(effects))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=120);p.add_argument('--steps',type=int,default=120);p.add_argument('--seed',type=int,default=149357);p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--faults',action='store_true');a=p.parse_args()
    name='ragdoll-failures' if a.faults else 'ragdoll';fixture=ROOT/f'native/fixtures/{name}.jsonl.gz'
    if a.golden:
        with gzip.open(fixture,'rt') as stream:header=json.loads(next(stream));records=[json.loads(line) for line in stream]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid ragdoll fixture')
        report=dict(check(records),**coverage(records),**ragdoll_coverage(records))
    else:
        directory=ROOT/f'native/artifacts/{name}';records,hashes=reference(directory,list(fault_queries(a.seed) if a.faults else queries(a.cases,a.seed,a.steps)),a.reuse_unity)
        report=dict(check(records),**coverage(records),**ragdoll_coverage(records),provenance=hashes);(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if a.faults:
            for r in records:
                e=r['expected'];assert isinstance(e,dict) and e['action']==0
                assert e['error']==('null_reference' if r['query']['fault']=='renderer' else 'animation_missing')
                assert e['failed']['report']['phases'][-1]==dict(kind='update',id=1)
                assert len(e['failed']['effects'])==1
        if report['match']:
            selected=set(range(min(12,len(records))))|set(random.Random(a.seed).sample(range(len(records)),min(24,len(records))))
            selected.update(i for i,r in enumerate(records) if isinstance(r['expected'],dict))
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,scope='Actual SlimeController ragdoll apply/recovery and invalid-ground detachment in composed scheduling; empty abilities, no ropes or lifecycle')
            fixture.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])
if __name__=='__main__':raise SystemExit(main())
