"""Independent scheduled Invisibility casts, expiration and cooldown restoration."""
import argparse
import copy
import gzip
import json
import random
from collections import Counter
from reference import ROOT
from verify_beam import reference,encoded
from verify_actors import check,ONE,DT,vec
from verify_slime_actors import queries as actor_queries

SCHEMA='bopl-invisibility-v1'
def info():
    return dict(player=0,slot=0,position=vec(),external_velocity=vec(),velocity=vec(),material=None,grounded=False,grounded_speed=0,rotation=0,forward=vec(),up=vec(),facing_right=False,ground=None,local=0,radius=0,sprite=None,rope=None,top=False,use_rope_on_player=False,rope_target=None,slime=None)

def asset_config():
    assets=json.loads((ROOT/'native/reference-export/assets.json').read_text())
    # The exporter stores a list of source objects; preserve the exact prefab
    # cooldown values and callback/audio identity, rather than guessing defaults.
    entries=assets if isinstance(assets,list) else assets.values()
    asset=next(a for a in entries if a['path']=='Assets/GameObject/Invisibility.prefab')
    c={c['type']:c['properties'] for c in asset['data']['components']}
    a=c['InstantAbility'];effect=c['Invisibility']
    assert a['funcOnEnter.m_PersistentCalls.m_Calls.Array.data[0].m_MethodName']=='CastInvisibility'
    assert a['EffectOnEnter'] is None and not a['isHookshot']
    return dict(cooldown=a['cooldown.m_rawValue'],restore=effect['cooldown.m_rawValue'],ground_only=a['OnlyCastableOnGround'],stop=a['cooldownStoppedDuringTimeStop'],gate=a['isInvisibilty'],audio=a['AudioOnCast'])

def queries(count,seed,steps):
    rng=random.Random(seed+390);asset=asset_config()
    for case,q in enumerate(actor_queries(count,seed,steps)):
        q['player_state']=True;q['routing_checks']=True;q['invisibility_asset']=asset
        for i,actor in enumerate(q['runtime']['actors']):
            slime=actor['slime'];slime['state'].update(mode=0,has_spawned=True,suppress=False)
            slime['entry']=dict(info=info(),rope_target=None,was_invisible=False)
            n=1+(case+i)%3;pool=[]
            for j in range(n):
                handler=dict(kind='invisibility',restore_cooldown=asset['restore'],gate_invisible=asset['gate'],component_present=True,linked=True,transform_present=True,position=vec(),active=False,info=info(),slime=None,audio=asset['audio'])
                pool.append(dict(present=True,destroyed=False,cooldown=asset['cooldown'],stopped_during_time_stop=asset['stop'],ground_only=asset['ground_only'],castable=True,ignore_next_cooldown=False,handler=handler))
            slots=list(range(n))
            if n>1 and case%5==0:slots[-1]=slots[0]
            slime['abilities']=dict(slots=slots,pool=pool,timers=[asset['cooldown']+rng.choice([-1,0,1,DT]) for _ in pool],indicators=[dict(seconds_since_reset=10000.) for _ in pool],in_ability=False,can_use=True,buttons=[False]*3)
        for i,p in enumerate(q['runtime']['scheduler']['players']):
            p.update(invisible=False,time_invisible=0,time_ability=0,flags=dict(spark=bool(i%2),macho=bool((i+case)%2),platform=True))
            if case>=6 and case%4==0:p.update(invisible=True,time_invisible=5*ONE-DT+[-1,0,1][i%3],time_ability=2*ONE)
        for tick,action in enumerate(q['actions']):
            action['scale']=ONE if case<6 or case%3!=2 else [ONE,ONE//4,0,ONE*2,-ONE][(tick//60)%5]
            action['dt']=DT
            if case<6:action['enabled']=[True]*len(q['runtime']['components'])
            action['routing_inputs']=[dict(can_use=case<6 or tick%61!=0,buttons=[True,True,True] if case<6 else [rng.randrange(8)!=0 for _ in range(3)]) for _ in q['runtime']['actors']]
        yield q

def fault_queries(seed):
    base=list(queries(2,seed,1))[1]
    for name in ['renderer','transform','link','missing_component','expiry_null_slot','expiry_destroyed','expiry_alias_tie']:
        q=copy.deepcopy(base);q['fault']=name;r=q['runtime'];actor=r['actors'][0];slime=actor['slime'];routing=slime['abilities']
        r['scheduler']['pending']['items']=list(range(len(r['components'])))
        for i,c in enumerate(r['scheduler']['components']):c['hierarchy']=i
        action=q['actions'][0];action['enabled']=[i==1 for i in range(len(r['components']))]
        action.update(scale=ONE,dt=DT)
        routing['timers']=[10*ONE]*len(routing['pool']);routing['in_ability']=False
        for v in action['routing_inputs']:v.update(can_use=True,buttons=[True]*3)
        actor['controller']['physics']['motion']['grounded']=False
        handler=routing['pool'][0]['handler']
        if name=='renderer':slime['animator']['render']=None
        elif name=='transform':handler['transform_present']=False
        elif name=='link':handler['linked']=False
        elif name=='missing_component':handler['component_present']=False
        else:
            slime['entry']['was_invisible']=True
            for v in action['routing_inputs']:v['buttons']=[False]*3
            for a in routing['pool']:a['cooldown']=10000*ONE
            if name=='expiry_null_slot':slime['state']['mode']=8;routing['slots']=[None]
            elif name=='expiry_destroyed':routing['pool'][0]['destroyed']=True;routing['timers']=[0]*len(routing['pool'])
            else:routing['slots']=[0,1,0];routing['timers']=[ONE]*3;routing['indicators']=[]
        yield q

def coverage(records):
    c=Counter();players=Counter()
    for r in records:
        e=r['expected'];frames=e if isinstance(e,list) else e['completed'];previous=[p['invisible'] for p in r['query']['runtime']['scheduler']['players']]
        for frame in frames:
            for i,p in enumerate(frame['players']):
                players['invisible' if p['invisible'] else 'visible']+=1
                c['became_invisible']+=not previous[i] and p['invisible'];c['expired']+=previous[i] and not p['invisible'];previous[i]=p['invisible']
            for event in frame['report']['events']:
                if event['event']==dict(kind='sound',value='invisibility'):c['casts']+=1
            for actor in frame['actors']:
                for ability in actor['slime']['abilities']['pool']:
                    c['sentinel_cooldown_snapshots']+=ability['cooldown']==10000*ONE
                    c['restored_cooldown_snapshots']+=ability['cooldown']==ability['handler']['restore_cooldown']
    return dict(invisibility=dict(c),player_states=dict(players))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=24);p.add_argument('--steps',type=int,default=900);p.add_argument('--seed',type=int,default=731249);p.add_argument('--faults',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--reuse-unity',action='store_true');a=p.parse_args()
    name='invisibility-failures' if a.faults else 'invisibility'
    fixture=ROOT/f'native/fixtures/{name}.jsonl.gz'
    if a.golden:
        with gzip.open(fixture,'rt') as f:header=json.loads(next(f));records=[json.loads(line) for line in f]
        assert header['schema']==SCHEMA and header['cases']==len(records)
        report=dict(check(records),**coverage(records))
    else:
        directory=ROOT/f'native/artifacts/{name}';records,hashes=reference(directory,list(fault_queries(a.seed) if a.faults else queries(a.cases,a.seed,a.steps)),a.reuse_unity)
        report=dict(check(records),**coverage(records),provenance=hashes);(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if a.faults:
            failures={'renderer','transform','link','expiry_null_slot'}
            for record in records:
                if record['query']['fault'] in failures:
                    e=record['expected'];assert e['error']=='null_reference' and e['action']==0
                    assert e['failed']['report']['phases'][-1]==dict(kind='update',id=1)
                else:assert len(record['expected'])==1
        if report['match']:
            selected=set(range(min(6,len(records))))|set(random.Random(a.seed).sample(range(len(records)),min(6,len(records))))
            selected.update(i for i,r in enumerate(records) if isinstance(r['expected'],dict))
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,scope='Independent supplied-state actor episodes with real Invisibility casts/expiration; headless indicators, no death/lifecycle/rope integration')
            fixture.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])
if __name__=='__main__':raise SystemExit(main())
