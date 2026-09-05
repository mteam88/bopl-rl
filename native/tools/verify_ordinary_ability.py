"""Real Ability entry/update/return between independently scheduled actor entities."""
import argparse
import copy
import json
import random
from actor_stream import ROOT, capture, verify, promote, golden
from verify_actor_entity import queries as actor_queries
from verify_slime_exit import exit_info
from verify_invisibility import info
from verify_actors import ONE, DT, vec


def queries(count, seed, steps):
    rng=random.Random(seed+494)
    for case,q in enumerate(actor_queries(count,seed,steps)):
        q['entry_events']=True
        r=q['runtime'];actors=r['actors'];n=len(actors);objects=r['scene']['objects'];next_id=max(o['id'] for o in objects)+10
        for i in range(n):
            source=actors[i];ordinary=copy.deepcopy(source);ordinary.pop('slime');ordinary['input']=vec()
            ordinary['object']=next_id+i;ordinary['collision_hierarchy']=2000+i
            ordinary['entity'].update(node=n+i,active=False,body_simulated=True)
            ordinary['body']['holder']=source['body']['holder']
            ordinary['ordinary']=dict(owner=i,info=info(),slime=None,callback=None,previous_scale=ONE,body_linked=True,collider_linked=True)
            ordinary['ordinary']['info']['player']=source['body']['holder']
            motion=ordinary['controller']['physics']['motion'];motion.update(grounded=False,ground=None,position=vec(4*i-8,8),velocity=vec(),external_velocity=vec(),rotation=0,right=vec(1,0),up=vec(0,1))
            ordinary['controller']['apply_gravity']=False
            obj=copy.deepcopy(next(o for o in objects if o['id']==source['object']));obj.update(id=ordinary['object'],active=False);obj['transform'].update(position=motion['position'],rotation=0,right=vec(1,0));objects.append(obj)
            actors.append(ordinary);r['scheduler']['nodes'].append(dict(parent=None,children=[],active=False,present=True))
            for role in ('ability','physics','body'):
                index=len(r['components']);r['components'].append(dict(actor=n+i,role=role));r['scheduler']['components'].append(dict(owner=n+i,kind='updatable',present=True,destroyed=False,hierarchy=index,enabled=True,ignore_activity=False))
            source['slime']['entry']['invisible_on_entry']=False
            source['slime']['abilities']=dict(pool=[dict(present=True,destroyed=False,cooldown=ONE//5,stopped_during_time_stop=False,ground_only=False,castable=True,ignore_next_cooldown=False,handler=dict(kind='entity',actor=n+i))],slots=[0],timers=[ONE],indicators=[dict(seconds_since_reset=3.,lifecycle=dict(active=False,follow=True))],in_ability=False,can_use=True,buttons=[False]*3)
        total=len(r['components']);r['scheduler'].update(live=[],added=total);r['scheduler']['pending']=dict(items=list(range(total)),length=total)
        order=list(range(total));rng.shuffle(order)
        for component,hierarchy in zip(r['scheduler']['components'],order):component['hierarchy']=hierarchy
        for player in r['scheduler']['players']:player['position']=dict(value=vec(),current=None,previous=None)
        for tick,action in enumerate(q['actions']):
            action['enabled']=[True]*total if case<6 else [rng.randrange(15)!=0 for _ in range(total)]
            action['routing_inputs']=[dict(can_use=True,buttons=[True,False,False]) for _ in range(n)]
            if tick>0 and tick%29==0:
                for i in range(n):
                    action['commands'].append(dict(op='ordinary_exit',actor=n+i,info=exit_info(actors[i],rng,len(r['bindings']))))
        yield q


def faults(seed):
    base=next(queries(1,seed,1))
    for name in ('missing_collider','destroyed_target','ready','exit_without_callback','zero_previous_scale'):
        q=copy.deepcopy(base);q['fault']=name;r=q['runtime'];q['scripts']={};a=q['actions'][0];a['commands']=[];a['enabled']=[True]*len(a['enabled'])
        for i,c in enumerate(r['scheduler']['components']):c['hierarchy']=i
        ability=r['actors'][1]['ordinary']
        if name=='missing_collider':ability['collider_linked']=False
        elif name=='destroyed_target':r['actors'][0]['slime']['abilities']['pool'][0]['destroyed']=True
        elif name=='exit_without_callback':a['commands']=[dict(op='ordinary_exit',actor=1,info=exit_info(r['actors'][0],random.Random(seed),len(r['bindings'])))]
        elif name=='zero_previous_scale':ability['previous_scale']=0;r['scheduler']['players'][0]['scale']=2*ONE
        yield q


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=48);p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=894152);p.add_argument('--faults',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--name',default='ordinary-ability');a=p.parse_args();fixture=ROOT/f'native/fixtures/{a.name}.jsonl.gz'
    if a.golden:report=golden(fixture)
    else:
        directory=ROOT/f'native/artifacts/{a.name}';inputs,hashes=capture(directory,list(faults(a.seed) if a.faults else queries(a.cases,a.seed,a.steps)),a.reuse_unity)
        report=dict(verify(directory,inputs),provenance=hashes);(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:promote(directory,fixture,inputs,set(range(min(12,len(inputs))))|set(report['error_cases']),hashes,'Real ordinary Ability bodies with input-driven entry, scheduled updates and actual exit callbacks; plain Ability has no concrete effect component')
    print(json.dumps(report,indent=2));return int(not report['match'])

if __name__=='__main__':raise SystemExit(main())
