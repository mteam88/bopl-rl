"""Real PlayerCollision.killPlayer, death accounting and scheduled actor removal."""
import argparse
import copy
import json
from actor_stream import ROOT,capture,verify,promote,golden
from verify_actor_entity import queries as entity_queries
from verify_actors import ONE,DT,vec


def queries(seed):
    for case,base in enumerate(entity_queries(6,seed,24,False)):
        for mode in ['death','clone','won','reviving','will_revive','invulnerable','override','inactive','missing_camera','missing_effect','double']:
            q=copy.deepcopy(base);q['death_case']=mode;q['scripts']={};q['routing_checks']=False
            r=q['runtime'];r['scheduler']['loaded']=False;n=len(r['actors'])
            book=dict(players=[dict(id=i+1,team=i%2,bodies_alive=1,won_round=False,kills=i,deaths=i+3,still_alive_round=True,killed_at=-1,cause=5,rumble=0) for i in range(n)],registered_kills=[0]*6)
            r['combat']=dict(book=book,revival=[dict(status=0,stocks=0,respawn_count=0) for _ in range(n)],events=[],trauma=0.0,max_trauma=1.0,camera_present=mode!='missing_camera',stocks_enabled=False,drop_enabled=False)
            for i,a in enumerate(r['actors']):
                a['entity'].update(present=True,destroyed=False)
                a['collision']=dict(invulnerable=1 if mode in ['invulnerable','override'] else 0,was_killed=True,death_trauma=0.5)
                a['slime']['managed_destroyed']=False
                routing=a['slime']['abilities'];routing.update(slots=[],pool=[],indicators=[],buttons=[False]*3)
                a['slime']['state'].update(has_spawned=True,suppress=False)
                cid=len(r['components']);r['components'].append(dict(actor=i,role='collision'))
                r['scheduler']['components'].append(dict(owner=i,kind='updatable',present=True,destroyed=False,hierarchy=100+i,enabled=True,ignore_activity=False))
                r['scheduler']['live'].append(cid)
            for action in q['actions']:
                action.update(commands=[],enabled=[True]*len(r['components']),scale=ONE,dt=DT)
                for inp in action['inputs']: inp.update(direction=vec(),jump=False)
            victim=case
            if mode=='inactive':r['actors'][victim]['entity']['active']=False
            if mode=='clone':book['players'][victim]['bodies_alive']=2
            if mode=='won':book['players'][victim]['won_round']=True
            if mode in ['reviving','will_revive']:r['combat']['revival'][victim]['status']=2 if mode=='reviving' else 1
            request=dict(killer=1 if victim else n,spawn_effect=mode in ['missing_effect','reviving'],ignore_invulnerability=mode=='override',cause=case%5)
            q['actions'][5]['commands']=[dict(op='kill_actor',actor=victim,request=request)]*(2 if mode=='double' else 1)
            yield q


def handoffs(seed):
    from verify_dash_ability import queries as dash_queries
    from verify_actor_ropes import queries as rope_queries
    deaths=[q for q in queries(seed) if q['death_case']=='death']
    for case,(base,death,ropes) in enumerate(zip(dash_queries(6,seed,24),deaths,rope_queries(6,seed,24))):
        for mode in ['dash','invulnerable_owner','inactive_owner']:
            q=copy.deepcopy(base);q['death_case']=mode;q['scripts']={};q['routing_checks']=False
            r=q['runtime'];n=len(r['scheduler']['players']);r['scheduler']['loaded']=False;r['combat']=copy.deepcopy(death['runtime']['combat'])
            r['actor_ropes']=copy.deepcopy(ropes['runtime']['actor_ropes']);r['scene']['rope_system']=copy.deepcopy(ropes['runtime']['scene']['rope_system'])
            r['actor_ropes']['max_stretch']=1000000*ONE
            for i,a in enumerate(r['actors']):
                a['entity'].update(present=True,destroyed=False)
                a['collision']=dict(invulnerable=1 if i<n and mode=='invulnerable_owner' else 0,was_killed=False,death_trauma=0.5)
                if 'slime' in a:
                    a['slime']['managed_destroyed']=False
                    a['slime']['state'].update(has_spawned=True,suppress=False)
                    for key in ['rope','top']:a['body'][key]=ropes['runtime']['actors'][i]['body'][key]
                else:a['ordinary']['dash']['pause']=10*ONE
                cid=len(r['components']);r['components'].append(dict(actor=i,role='collision'))
                r['scheduler']['components'].append(dict(owner=a['entity']['node'],kind='updatable',present=True,destroyed=False,hierarchy=100+i,enabled=True,ignore_activity=False))
                r['scheduler']['live'].append(cid)
            for tick,action in enumerate(q['actions']):
                action.update(commands=[],enabled=[True]*len(r['components']),scale=ONE,dt=DT)
                for inp in action['routing_inputs']:inp.update(can_use=True,buttons=[tick==0,False,False])
            target=case if mode=='inactive_owner' else n+case
            q['actions'][5]['commands']=[dict(op='kill_actor',actor=target,request=dict(killer=1 if case else n,spawn_effect=False,ignore_invulnerability=False,cause=case%5))]
            yield q


def contacts(seed):
    bases=[q for q in queries(seed) if q['death_case']=='death']
    for base in bases:
        for mode in ['lethal','invulnerable','missing_effect','duplicates']:
            q=copy.deepcopy(base);q['death_case']=mode;r=q['runtime'];r['combat']['collisions']=True;r['combat']['hitboxes']=[]
            r['scheduler']['loaded']=False
            for action in q['actions']:action['commands']=[]
            for i,a in enumerate(r['actors']):
                r['combat']['revival'][i]['status']=0 if mode=='missing_effect' else 2
                a['collision']['invulnerable']=1 if mode=='invulnerable' else 0
                a['collision']['death_trauma']=0.8
                actor_obj=next(o for o in r['scene']['objects'] if o['id']==a['object'])
                for repeat in range(2 if mode=='duplicates' else 1):
                    obj=copy.deepcopy(actor_obj);oid=10000+10*i+repeat
                    obj.update(id=oid,layer=16,active=True,callback=True,callback_destroyed=False,destroyed=False)
                    obj['collider']=dict(kind='circle',shape=dict(center=copy.deepcopy(a['controller']['physics']['motion']['position']),radius=2*ONE))
                    obj['body'].update(inverse_mass=-ONE,inverse_inertia=0,velocity=vec(),force=vec(),angular_velocity=0)
                    obj['transform'].update(position=copy.deepcopy(a['controller']['physics']['motion']['position']),offset=vec())
                    r['scene']['objects'].append(obj);r['combat']['hitboxes'].append(dict(object=oid,owner=1 if i else len(r['actors'])))
            r['scene']['layers'][8]|=1<<16;r['scene']['layers'][16]|=1<<8
            yield q


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=724548);p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--handoffs',action='store_true');p.add_argument('--contacts',action='store_true');args=p.parse_args()
    name='actor-death-contacts' if args.contacts else 'actor-death-handoffs' if args.handoffs else 'actor-deaths';fixture=ROOT/f'native/fixtures/{name}.jsonl.gz'
    if args.golden:result=golden(fixture)
    else:
        directory=ROOT/f'native/artifacts/{name}';inputs,hashes=capture(directory,list(contacts(args.seed) if args.contacts else handoffs(args.seed) if args.handoffs else queries(args.seed)),args.reuse_unity)
        result=dict(verify(directory,inputs),provenance=hashes);(directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['match']:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(result,indent=2));return int(not result['match'])

if __name__=='__main__':raise SystemExit(main())
