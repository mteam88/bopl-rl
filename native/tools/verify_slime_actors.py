"""Compare actual SlimeController locomotion in composed scheduled episodes."""
import argparse
import copy
import gzip
import json
import random
from collections import Counter
from reference import ROOT
from verify_beam import encoded,reference
from verify_actors import queries as actor_queries,check,vec,ONE
from verify_animator import asset_queries

FIXTURE=ROOT/'native/fixtures/slime-actors.jsonl.gz'
SCHEMA='bopl-slime-actors-v1'
def queries(count,seed,steps):
    assets,provenance=asset_queries();animations={};fall_map=None;template=None
    needed={'idle','run','stop','runTurnAround','land','fallLeftLoop'}
    for q in assets:
        if not q['asset']['path'].endswith('/MCAnimData.asset'):continue
        name=q['asset'].get('animation')
        if name in needed and q['actions'][0]['op']=='begin':animations[name]=q['actions'][0]['frames'];template=q['state']
        if q['asset'].get('velocity_animation')=='fallLeft':fall_map=q['state']['velocity_maps'][0]
    assert set(animations)==needed and fall_map is not None
    rng=random.Random(seed+87)
    for case,q in enumerate(actor_queries(count,seed,steps)):
        q['operation']='dynamics_slime_actors';q['animation_assets']=provenance
        r=q['runtime'];n=len(r['actors'])
        for i,a in enumerate(r['actors']):
            animator=copy.deepcopy(template);animator.update(current=copy.deepcopy(animations['idle']),render=dict(sprite=animations['idle'][0]['sprite'],flip_x=False,flip_y=False),callback=None)
            a['slime']=dict(state=dict(mode=[0,1,2,3,4,5,6,7,8,9][(case+i)%10],facing_right=bool((case+i)%2),was_upside_down=case%4==0,input=vec(),previous_input=0,
                suppress=case%7==0,has_spawned=case%13!=0,time_moonwalking=0,turn_input=0),animator=animator,speed=dict(velocity=vec(),rotation=0),animations=copy.deepcopy(animations),fall_map=copy.deepcopy(fall_map))
        r['components']=[dict(actor=i,role=role) for i in range(n) for role in ['physics','motor','body','animation']]
        priorities=list(range(n*4))
        if case%3==1:rng.shuffle(priorities)
        if case%3==2:priorities=[0]*len(priorities)
        r['scheduler']['components']=[dict(owner=c['actor'],kind='updatable',present=True,destroyed=False,hierarchy=priorities[i],enabled=True,ignore_activity=True) for i,c in enumerate(r['components'])]
        order=list(range(n*4));rng.shuffle(order);r['scheduler']['pending']=dict(items=order,length=len(order));r['scheduler']['added']=len(order)
        for tick,action in enumerate(q['actions']):
            for inp in action['inputs']:inp['direction']=rng.choice([vec(),vec(1,0),vec(-1,0),vec(1,1),vec(-1,-1),vec(0,-1)])
            action['enabled']=[rng.randrange(13)!=0 for _ in r['components']]
        yield q

def fault_queries(seed):
    base=list(queries(2,seed,120))[1]
    for name in ['land','fallLeftLoop','idle']:
        q=copy.deepcopy(base);q['fault']=name;r=q['runtime'];n=len(r['actors'])
        for a in r['actors']:
            a['slime']['state'].update(mode=0,facing_right=True,suppress=False,has_spawned=True,was_upside_down=False)
            a['controller']['skip_update']=False
        r['scheduler']['pending']['items']=list(range(n*4))
        for i,c in enumerate(r['scheduler']['components']):c['hierarchy']=i
        for action in q['actions']:
            action.update(dt=71582788,scale=ONE,enabled=[True]*(n*4),inputs=[dict(direction=vec(),jump=False) for _ in range(n)])
        a=r['actors'][0]
        if name=='land':
            a['slime']['animations']['land']=[]
            a['controller']['physics']['motion'].update(grounded=False,velocity=vec(0,-30))
        else:
            del a['slime']['animations'][name]
            if name=='fallLeftLoop':q['actions'][0]['inputs'][0]['jump']=True
            else:
                for action in q['actions'][:15]:action['inputs'][0]['direction']=vec(1,0)
        yield q

def coverage(records):
    states=Counter();kinds=Counter();callbacks=Counter();transitions=Counter()
    for r in records:
        e=r['expected'];frames=e if isinstance(e,list) else e['completed']
        previous=[a['slime']['state']['mode'] for a in r['query']['runtime']['actors']]
        for frame in frames:
            for i,a in enumerate(frame['actors']):
                s=a['slime'];mode=s['state']['mode'];states[mode]+=1;kinds[s['animator']['kind']]+=1
                if s['animator']['callback'] is not None:callbacks[s['animator']['callback']]+=1
                if mode!=previous[i]:transitions[f'{previous[i]}->{mode}']+=1
                previous[i]=mode
    return dict(slime_states=dict(states),animation_kinds=dict(kinds),pending_callbacks=dict(callbacks),state_transitions=dict(transitions))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=120);p.add_argument('--steps',type=int,default=120);p.add_argument('--seed',type=int,default=571937);p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--faults',action='store_true');a=p.parse_args()
    fixture=ROOT/'native/fixtures/slime-actors-failures.jsonl.gz' if a.faults else FIXTURE
    if a.golden:
        with gzip.open(fixture,'rt') as f:header=json.loads(next(f));records=[json.loads(line) for line in f]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid slime fixture')
        report=dict(check(records),**coverage(records))
    else:
        directory=ROOT/('native/artifacts/slime-actors-failures' if a.faults else 'native/artifacts/slime-actors');records,hashes=reference(directory,list(fault_queries(a.seed) if a.faults else queries(a.cases,a.seed,a.steps)),a.reuse_unity)
        report=dict(check(records),**coverage(records),provenance=hashes);(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if a.faults:
            expected_roles={'land':'physics','fallLeftLoop':'motor','idle':'animation'}
            for r in records:
                e=r['expected'];assert isinstance(e,dict) and e['error'] in ['index_out_of_range','animation_missing']
                phase=e['failed']['report']['phases'][-1];assert phase['kind']=='update'
                assert r['query']['runtime']['components'][phase['id']]['role']==expected_roles[r['query']['fault']]
        if report['match']:
            selected=set(range(min(12,len(records))))|set(random.Random(a.seed).sample(range(len(records)),min(24,len(records))))
            selected.update(i for i,r in enumerate(records) if isinstance(r['expected'],dict))
            seen=set()
            for i,r in enumerate(records):
                e=r['expected'];frames=e if isinstance(e,list) else e['completed']
                previous=[a['slime']['state']['mode'] for a in r['query']['runtime']['actors']]
                transitions=set()
                for frame in frames:
                    for j,actor in enumerate(frame['actors']):
                        mode=actor['slime']['state']['mode']
                        if mode!=previous[j]:transitions.add((previous[j],mode))
                        previous[j]=mode
                if transitions-seen:selected.add(i);seen.update(transitions)
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,scope='Original SlimeController UpdateSim with empty abilities, player/body/animation scheduling and static terrain; excludes ragdoll, swallowing, position-setter comparison and full lifecycle')
            fixture.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])
if __name__=='__main__':raise SystemExit(main())
