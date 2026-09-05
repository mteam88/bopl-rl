"""Differential Updater queue/lifecycle checks against the actual Unity tick."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
import random
import subprocess
from reference import ROOT,first_difference
from verify_beam import encoded,reference
from verify_player_ground import ONE

FIXTURE=ROOT/'native/fixtures/scheduler.jsonl.gz'
SCHEMA='bopl-scheduler-v1'


def queries(count,seed):
    rng=random.Random(seed)
    for case in range(count):
        nodes=[];components=[];up=[];plain=[]
        for i in range(rng.randint(8,25)):
            parent=rng.randrange(i) if i and rng.randrange(4) else None
            nodes.append(dict(parent=parent,children=[],active=bool(rng.randrange(3)),present=True))
            if parent is not None:nodes[parent]['children'].append(i)
            for _ in range(rng.randint(1,3)):
                up.append(len(components));components.append(dict(owner=i,kind='updatable',present=True,destroyed=False,hierarchy=rng.randint(-(1<<31),(1<<31)-1) if case%11==0 else rng.randint(-3,3),enabled=bool(rng.randrange(3)),ignore_activity=bool(rng.randrange(2))))
            for kind in ['fix','plain']:
                plain.append(len(components));components.append(dict(owner=i,kind=kind,present=True,destroyed=False,hierarchy=0,enabled=False,ignore_activity=False))
        for n in nodes:rng.shuffle(n['children'])
        if case%17==0:components[up[1]]['present']=False
        if case%29==0:
            nodes[-1]['present']=False
            for c in components:
                if c['owner']==len(nodes)-1:c['present']=False
        if case%13==0:components[up[2]]['destroyed']=True
        pending=rng.sample(up,len(up));pending+=rng.choices(up,k=3)
        if case%7==0:pending.insert(2,None)
        capacity=len(pending)+rng.choice([0,1,8,32]);storage=pending+[None]*(capacity-len(pending))
        inputs=[dict(jump=bool(rng.randrange(2)),pressed=bool(rng.randrange(2)),previous=bool(rng.randrange(2))) for _ in range(3)]
        players=[dict(id=i+1,protected=bool((case+i)%2),invisible=bool((case+i)%3),time_invisible=rng.choice([0,5*ONE-1,5*ONE,5*ONE+1]),time_ability=rng.randint(-ONE,8*ONE),input=rng.randrange(3)) for i in range(case%6+1)]
        if case%31==0:players[-1]['input']=None
        state=dict(nodes=nodes,components=components,players=players,inputs=inputs,live=[None,*rng.choices(up,k=5)],pending=dict(items=storage,length=len(pending)),prefab=[],destroy_buffer=[],destroyed_objects=[],destroyed_monos=[],
            added=(1<<31)-1 if case%19==0 else rng.randrange(200),prefabs_added=(1<<31)-1 if case%23==0 else rng.randrange(50),registering_prefab=False,
            loaded=case%3==0,stopped=bool(case%2),time_scale=ONE,time=rng.randrange(-ONE,ONE),level_time=rng.randrange(-ONE,ONE),ticks=(1<<31)-1 if case%5==0 else rng.randrange(1000),hitstop=rng.randrange(-1,8))
        reg=lambda id:dict(op='register',id=id)
        tick=lambda dt=ONE//60:dict(op='tick',dt=dt)
        leaf=len(nodes)-1
        scripts={f'init:{up[0]}':[[reg(up[1]),dict(op='begin'),reg(up[2]),dict(op='end')],[reg(None)]],
            f'update:{up[1]}':[[reg(up[3]),dict(op='destroy_mono',id=up[4]),dict(op='live_add',id=up[5])]],
            f'late:{up[2]}':[[dict(op='destroy_object',id=leaf)]],
            f'destroy_updatable:{up[4]}':[[reg(up[6])]],f'destroy_object:{leaf}':[[dict(op='destroy_mono',id=plain[0])]],
            'physics:*':[[reg(up[7])]],'duplicates:*':[[dict(op='mark',id=up[8],value=True)]],
            'magnet:*':[[dict(op='post_level')]],'player_update:0':[[dict(op='scale',value=0)]]}
        if case%5==0:scripts[f'init:{up[0]}']=[[dict(op='pre_level')]]
        if case%13==0:scripts[f'destroy_updatable:{up[2]}']=[[dict(op='live_clear')]]
        if case%17==0:scripts[f'update:{up[0]}']=[[dict(op='live_remove',index=0)]]
        actions=[dict(op='sort')] if case%2==0 else []
        actions.extend([reg(up[0]),dict(op='begin'),reg(up[1]),reg(up[1]),reg(up[2]),dict(op='end'),tick(0),
            dict(op='scale',value=ONE),tick(),dict(op='input',id=0,value=True),dict(op='hitstop',value=123),tick(-ONE),
            dict(op='scale',value=0),dict(op='post_level'),tick(),dict(op='input',id=0,value=False),tick(),
            dict(op='scale',value=ONE*2),dict(op='enable',id=up[1],value=True),reg(up[-1]),tick(),
            dict(op='destroy_object',id=leaf if case%3==0 else 0),tick(),reg(up[-1]),tick(),dict(op='destroy_mono',id=plain[0]),dict(op='destroy_mono',id=plain[0]),
            dict(op='pre_level'),dict(op='post_level'),tick(),dict(op='reinit'),tick(),dict(op='begin'),reg(up[0]),reg(None if case%3==0 else up[1]),dict(op='end')])
        if case%3:actions.append(tick())
        yield dict(operation='dynamics_scheduler',state=state,scripts=scripts,actions=actions,scene=dict(objects=[],composites=[],gravity=0,layers=[(1<<32)-1]*32))


def activation_queries(seed):
    base=list(queries(2,seed))[1]
    for deleted in [False,True]:
        q=deepcopy(base);q['targeted']='activation_deleted' if deleted else 'activation_live';q['scripts']={}
        for n in q['state']['nodes']:n['active']=True
        tick=lambda:dict(op='tick',dt=ONE//60)
        q['state']['loaded']=False
        q['actions']=[tick(),dict(op='active',id=0,value=False),tick(),dict(op='active',id=0,value=True),tick(),dict(op='destroy_mono',id=None),tick(),dict(op='destroy_object',id=0),tick()]
        if deleted:q['actions'].append(dict(op='active',id=0,value=False))
        yield q


def stress_queries(count,seed):
    base=next(queries(1,seed));rng=random.Random(seed)
    for case in range(count):
        q=deepcopy(base);s=q['state'];size=[171,256,512,1024,4096][case%5]
        keys=[0 for _ in range(size)] if case%6==0 else list(range(size)) if case%6==1 else list(reversed(range(size))) if case%6==2 else [min(i,size-i) for i in range(size)] if case%6==3 else [rng.randint(-8,8) for _ in range(size)] if case%6==4 else [rng.choice([-(1<<31),-1,0,1,(1<<31)-1]) for _ in range(size)]
        s['nodes']=[dict(parent=None,children=[],active=True,present=True)]
        s['components']=[dict(owner=0,kind='updatable',present=True,destroyed=False,hierarchy=h,enabled=True,ignore_activity=False) for h in keys]
        s['pending']=dict(items=list(range(size))+[None]*17,length=size);s['live']=[];s['prefab']=[]
        s['players']=[];s['inputs']=[];s['loaded']=False;s['stopped']=False
        s['added']=0;s['prefabs_added']=0;s['ticks']=0;s['hitstop']=0
        q['scripts']={};q['actions']=[dict(op='sort'),dict(op='tick',dt=0),dict(op='destroy_object',id=0),dict(op='tick',dt=ONE//60),dict(op='tick',dt=0)]
        yield q


def check(records):
    p=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=encoded(r['query'] for r in records),text=True,capture_output=True,check=True,timeout=300)
    outputs=[json.loads(line) for line in p.stdout.splitlines()]
    if len(outputs)!=len(records):raise ValueError('Incomplete scheduler output')
    counts,events,errors,actions=(Counter() for _ in range(4));failures=[]
    for i,(r,out) in enumerate(zip(records,outputs)):
        expected=r['expected']
        if isinstance(expected,list):frames=expected;counts['complete_scenarios']+=1
        else:frames=expected['completed'];errors[expected['error']]+=1
        counts['completed_actions']+=len(frames);actions.update(a['op'] for a in r['query']['actions'][:len(frames)])
        for f in frames:events.update(e['kind'] for e in f['events'])
        if out!=expected:
            counts['mismatches']+=1
            if len(failures)<10:
                diff=first_difference(expected,out);compact=lambda v:dict(type=type(v).__name__,length=len(v)) if isinstance(v,(dict,list)) else v
                failures.append(dict(case=i,difference=[diff[0],compact(diff[1]),compact(diff[2])]))
    return dict(match=not counts['mismatches'],cases=len(records),**counts,actions=dict(actions),events=dict(events),errors=dict(errors),first_failures=failures)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int);p.add_argument('--seed',type=int,default=91463)
    p.add_argument('--stress',action='store_true')
    p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');args=p.parse_args()
    args.cases=args.cases if args.cases is not None else 24 if args.stress else 1000
    fixture=FIXTURE.with_name('scheduler-stress.jsonl.gz') if args.stress else FIXTURE
    if args.golden:
        with gzip.open(fixture,'rt') as f:header=json.loads(next(f));records=[json.loads(l) for l in f]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid scheduler fixture')
        report=check(records)
    else:
        if args.cases<1:p.error('--cases must be positive')
        directory=ROOT/('native/artifacts/scheduler-stress' if args.stress else 'native/artifacts/scheduler')
        inputs=list((stress_queries if args.stress else queries)(args.cases,args.seed))
        if not args.stress:inputs.extend(activation_queries(args.seed))
        records,hashes=reference(directory,inputs,args.reuse_unity)
        report=dict(check(records),provenance=hashes);(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            selected=set(range(min(24,len(records))))|set(random.Random(args.seed).sample(range(len(records)),min(128,len(records))))
            selected.update(i for i,r in enumerate(records) if isinstance(r['expected'],dict) and r['expected']['action']<len(r['query']['actions'])-1)
            selected.update(i for i,r in enumerate(records) if 'targeted' in r['query'])
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,scope='Actual Updater with callback components, empty physics scene, frame-dependent jump input and invisibility clocks; no gameplay-specific Unity OnDestroy handlers or full world')
            fixture.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])


if __name__=='__main__':raise SystemExit(main())
