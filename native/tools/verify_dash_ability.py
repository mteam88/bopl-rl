"""Input-driven original Dash entry, scheduled windup/raycast, and automatic return."""
import argparse
import copy
import json
import random
from actor_stream import ROOT, capture, verify, promote, golden
from verify_ordinary_ability import queries as ordinary_queries
from verify_invisibility import info
from verify_actors import ONE, vec


def queries(count, seed, steps):
    rng=random.Random(seed+591)
    for case,q in enumerate(ordinary_queries(count,seed,steps)):
        r=q['runtime'];actors=r['actors'];n=len(r['scheduler']['players']);q['scripts']={}
        for i,inp in enumerate(r['scheduler']['inputs']):inp['angle']=0
        for player in r['scheduler']['players']:player['invisible']=False
        for i,o in enumerate(r['scene']['objects']):o['hierarchy']=i
        for i,actor in enumerate(actors):
            next(o for o in r['scene']['objects'] if o['id']==actor['object'])['hierarchy']=actor['collision_hierarchy']
            actor['entity']['body_simulated']=True
            if i<n:
                actor['entity']['active']=True
                actor['slime']['abilities']['timers']=[ONE]
                actor['slime']['entry']['invisible_on_entry']=False
                continue
            actor['ordinary']['dash']=dict(pause=rng.choice([42949672,ONE//10,ONE//3]),length=rng.choice([4,13,30])*ONE,leave_speed=100*ONE,fixed_updates=rng.choice([0,2,5]),mask=1<<11,anticipation=1000,input=vec(),info=info(),player=None,time=0,updates=0,in_dash=False,hits=[],hit_ground=None,hit_local=0,sprite_enabled=True,smear_enabled=False,sprite=None)
            index=len(r['components']);r['components'].append(dict(actor=i,role='dash'));r['scheduler']['components'].append(dict(owner=i,kind='updatable',present=True,destroyed=False,hierarchy=index,enabled=True,ignore_activity=False))
        total=len(r['components']);order=list(range(total));rng.shuffle(order)
        for c,h in zip(r['scheduler']['components'],order):c['hierarchy']=h
        r['scheduler'].update(live=[],added=total);r['scheduler']['pending']=dict(items=list(range(total)),length=total)
        for tick,action in enumerate(q['actions']):
            action['enabled']=[True]*total
            action['commands']=[]
            if case>=6 and tick%41==0:
                action['commands']=[dict(op='player_scale',player=i,value=rng.choice([ONE//2,ONE,2*ONE])) for i in range(n)]
            for inp in action['inputs']:inp['angle']=rng.randrange(256)
        yield q



def faults(seed):
    base=next(queries(1,seed,12))
    for name in ('zero_aim','exact_pause','water_exit','horizontal_below_water','nonwall_missing_collision','wall_missing_component','tied_twenty_hits','equal_twenty_hits','zero_time_scale','negative_frame_count'):
        q=copy.deepcopy(base);q['fault']=name;r=q['runtime'];a,b=r['actors'];ordinary=b['ordinary'];dash=ordinary['dash']
        a['entity']['active']=False;a['slime']['abilities']['in_ability']=True;b['entity']['active']=True
        ordinary.update(slime=0,callback=0);dash.update(player=1,info=copy.deepcopy(ordinary['info']),pause=ONE//10,time=0,fixed_updates=2,length=13*ONE,input=vec(1,0))
        for action in q['actions']:
            action['scale']=ONE;action['routing_inputs'][0]['buttons']=[False]*3;action['inputs'][0]['angle']=1
        for o in r['scene']['objects']:o['enabled']=False
        b['controller']['physics']['motion'].update(position=vec(0,8),velocity=vec(),external_velocity=vec(),grounded=False,ground=None)
        if name=='zero_aim':
            dash['pause']=0;dash['input']=vec()
            for action in q['actions']:action['inputs'][0]['angle']=0
        elif name=='exact_pause':dash['pause']=q['actions'][0]['dt'];dash['input']=vec()
        elif name in ('water_exit','horizontal_below_water'):
            dash.update(in_dash=True,updates=3,input=vec(0,-1) if name=='water_exit' else vec(1,0))
            dash['time']=dash['pause'];b['controller']['physics']['motion']['position']=vec(0,-10 if name=='water_exit' else -60)
        elif name=='zero_time_scale':
            for action in q['actions']:action['scale']=0
            r['scheduler']['players'][0]['protected']=False
        elif name=='negative_frame_count':dash.update(fixed_updates=-1,pause=0)
        else:
            dash['pause']=0;dash['input']=vec(1,0);dash['mask']=1<<(11 if name=='wall_missing_component' else 9)
            template=next(o for o in r['scene']['objects'] if o['id']==b['object'])
            for i in range(20 if 'twenty' in name else 1):
                o=copy.deepcopy(template);o.update(id=9000+i,active=True,enabled=True,layer=11 if name=='wall_missing_component' else 9,hierarchy=9 if name=='equal_twenty_hits' else 20-i)
                o['transform'].update(position=vec(3,8),rotation=0,right=vec(1,0));o['collider']['shape'].update(center=vec(3,8),right=vec(1,0),up=vec(0,1),inverse_extents=vec(1,1));r['scene']['objects'].append(o)
        yield q



def coverage(directory, queries):
    # There are no supplied activity/exit commands in the main corpus. Entry
    # events plus activity conservation count automatic returns even when the
    # scheduler exits and reenters an ability within the same frame.
    from collections import Counter
    counts=Counter();previous={}
    with (directory/'unity.jsonl').open() as stream:
        for row in map(json.loads,stream):
            if 'state' not in row:continue
            q=queries[row['case']];actors=row['state']['actors'];n=len(q['runtime']['scheduler']['players']);prev=previous.get(row['case'],q['runtime']['actors'])
            entries=Counter(e['actor'] for e in row['state']['report']['events'] if e['event']['kind']=='ability_entry')
            for i in range(n):
                actor=actors[n+i];d=actor['ordinary']['dash'];old=prev[n+i];active=actor['entity']['active']
                returned=entries[i]+old['entity']['active']-active
                assert returned>=0
                counts['automatic_returns_in_completed_frames']+=returned
                counts['returns_and_reentries_in_same_frame']+=returned>0 and entries[i]>0
                if active:counts['active_windup_snapshots' if d['time']<d['pause'] else 'active_dash_snapshots']+=1
                if d['in_dash'] and (not old['ordinary']['dash']['in_dash'] or entries[i]>0):
                    counts['wall_raycast_performs' if d['hit_ground'] is not None else 'air_raycast_performs']+=1
                    counts['multi_hit_raycast_performs']+=len(d['hits'])>1
                if returned and actor['controller']['physics']['motion']['position']['y']==(-50*ONE if q['runtime']['space'] else -48533131264):counts['returns_at_water_height']+=returned
            previous[row['case']]=actors
    return dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=48);p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=953174);p.add_argument('--faults',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--directory');p.add_argument('--name',default='dash-ability');a=p.parse_args();fixture=ROOT/f'native/fixtures/{a.name}.jsonl.gz'
    if a.golden:report=golden(fixture)
    else:
        directory=ROOT/(a.directory or f'native/artifacts/{a.name}');inputs,hashes=capture(directory,list(faults(a.seed) if a.faults else queries(a.cases,a.seed,a.steps)),a.reuse_unity)
        report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:report['dash_coverage']=coverage(directory,inputs)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:promote(directory,fixture,inputs,set(range(min(12,len(inputs))))|set(report['error_cases']),hashes,'Original Dash component with quantized aim, walls-only ray mask, scheduled automatic return; no PlayerCollision or rope transfer')
    print(json.dumps(report,indent=2));return int(not report['match'])

if __name__=='__main__':raise SystemExit(main())
