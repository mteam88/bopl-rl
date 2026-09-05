"""Compare composed scheduled player/body/physics episodes with original Unity methods."""
import argparse
from collections import Counter
import copy
import gzip
import json
import random
import struct
import subprocess
from reference import ROOT, first_difference
from verify_beam import encoded, reference

ONE=1<<32
DT=71582788
FIXTURE=ROOT/'native/fixtures/actors.jsonl.gz'
SCHEMA='bopl-actors-v1'
def vec(x=0,y=0):return dict(x=int(x*ONE),y=int(y*ONE))
def f32(x):return int(struct.unpack('<f',struct.pack('<f',x))[0]*ONE)
def rigid(inverse=-ONE):
    return dict(inverse_mass=inverse,velocity=vec(),force=vec(),angular_velocity=0,orientation=0,inverse_inertia=0,
        bounciness=0,static_friction=0,dynamic_friction=0,gravity_scale=ONE,air_drag=ONE,rotational_drag=ONE)
def collider(id,kind,shape,position,right,layer):
    return dict(id=id,collider=dict(kind=kind,shape=shape),body=rigid(),transform=dict(position=position,rotation=0,offset=vec(),right=right),
        layer=layer,active=True,callback=True,callback_destroyed=False,destroyed=False,composite=None,relative_pos=vec(),relative_orientation=0)
def queries(count,seed,steps):
    rng=random.Random(seed)
    reset=json.loads((ROOT/'native/data/level1-movement-6.json').read_text())
    for case in range(count):
        n=case%6+1;objects=[];bindings=[];grounds=[];actors=[]
        for p in reset['platforms']:
            objects.append(collider(p['id'],'rounded_rect',copy.deepcopy(p['collider']),copy.deepcopy(p['surface']['center']),copy.deepcopy(p['surface']['right']),11))
            bindings.append(dict(object=p['id'],platform=dict(surface=copy.deepcopy(p['surface']),destroyed=False,has_shape=True),transform_destroyed=False,active_self=True))
            grounds.append(dict(ice=bool(case%3==0),drop_frames=2 if case%11==0 else 0,dont_drop=case%7,drop_force=10*ONE))
        for i,p in enumerate(reset['players'][:n]):
            c=p['constants'];m={k:copy.deepcopy(p[k]) for k in ['position','rotation','up','velocity','external_velocity','radius','grounded','ground','local','grounded_speed']}
            m['right']=dict(x=m['up']['y'],y=-m['up']['x']);m['ground']=next(j for j,b in enumerate(bindings) if b['object']==m['ground'])
            if case%4==0:m['grounded']=False;m['position']['y']+=(i+2)*ONE;m['velocity']=vec(rng.randint(-10,10),rng.randint(-15,15))
            if case%5==0:m['position']=vec(i*.2 if case%10 else 0,2);m['external_velocity']=vec(rng.randint(-10,10),rng.randint(-10,10))
            config=dict(physics=dict(gravity=c['gravity_accel'],max_fall_speed=c['gravity_max_fall_speed'],gravity_modifier=c['gravity_modifier'],air_accel=c['air_accel'],max_speed=c['max_speed'],inverse_mass=35*ONE,
                **{k:c[k] for k in ['jump_strength','jump_extra_x','jump_momentum','jump_normal_scale','jump_teleport']},rope_strength=0),accel=c['accel'],friction=c['friction'],ice_friction=c['ice_friction'],alignment_deadzone=f32(.2))
            controller=dict(physics=dict(motion=m,scale=ONE,previous_scale=ONE,jumped=False,sync_rotation=True,cannot_unground=False),skip_update=case%7==0,apply_gravity=True,blocked=True)
            body=dict(position=copy.deepcopy(m['position']),velocity=copy.deepcopy(m['velocity']),external_velocity=copy.deepcopy(m['external_velocity']),previous_position=vec(),previous_velocity=vec(),previous_imposed=vec(),previous_external=vec(),
                extra_swing_force=vec(),rope=None,top=False,ignore_gravity=False,rope_mass=ONE//2,holder=i+1,physics=dict(grounded=m['grounded'],jumped=False))
            obj=collider(2000+i,'box',copy.deepcopy(p['collider']),copy.deepcopy(m['position']),copy.deepcopy(m['right']),8);obj['transform']['rotation']=m['rotation'];objects.append(obj)
            actors.append(dict(object=2000+i,controller=controller,body=body,config=config,input=vec(),push_strength=21474836 if case%4 else None,collision_hierarchy=rng.choice([i,-i,2147483647,-2147483648])))
        components=[dict(actor=i,role=role) for i in range(n) for role in ['physics','motor','body']]
        priorities=list(range(3*n))
        if case%3==1:rng.shuffle(priorities)
        if case%3==2:priorities=[0]*(3*n)
        order=list(range(3*n));rng.shuffle(order)
        schedule=dict(nodes=[dict(parent=None,children=[],active=True,present=True) for _ in range(n)],components=[dict(owner=c['actor'],kind='updatable',present=True,destroyed=False,hierarchy=priorities[i],enabled=True,ignore_activity=True) for i,c in enumerate(components)],
            players=[dict(id=i+1,protected=bool((case+i)%3==0),invisible=False,time_invisible=0,time_ability=0,input=i) for i in range(n)],inputs=[dict(jump=False,pressed=False,previous=False) for _ in range(n)],
            live=[],pending=dict(items=order,length=len(order)),prefab=[],destroy_buffer=[],destroyed_objects=[],destroyed_monos=[],added=3*n,prefabs_added=0,registering_prefab=False,
            loaded=case%5==0,stopped=False,time_scale=ONE,time=0,level_time=0,ticks=0,hitstop=0)
        runtime=dict(scheduler=schedule,components=components,actors=actors,scene=dict(objects=objects,composites=[],layers=[(1<<32)-1]*32,gravity=-ONE),bindings=bindings,grounds=grounds,space=case%4==1,player_layer=8)
        actions=[]
        for tick in range(steps):
            inputs=[dict(direction=rng.choice([vec(),vec(1,0),vec(-1,0),vec(.01,-1),vec(.6,.8)]),jump=rng.randrange(7)==0) for _ in range(n)]
            actions.append(dict(dt=0 if tick%31==30 else DT,scale=[ONE,ONE//4,0,2*ONE,-ONE,f32(.01)-1,f32(.01)][(case+tick//9)%7],inputs=inputs,enabled=[rng.randrange(11)!=0 for _ in components]))
        # Force coincident players through a loaded-level early return, then
        # a physics tick. This exercises signed/wrapping hierarchy tie breaking.
        if case%10==0:
            for action in actions[:2]:
                action.update(dt=0,inputs=[dict(direction=vec(),jump=False) for _ in range(n)],enabled=[True]*len(components))
        # Failure can follow disabled/skipped controller updates; retain the
        # state reached by other components before the failing call.

        if case%23==22:actors[-1]['controller']['physics']['previous_scale']=0
        yield dict(operation='dynamics_actors',runtime=runtime,actions=actions)

def check(records):
    p=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=encoded(r['query'] for r in records),text=True,capture_output=True,check=True,timeout=300)
    outputs=[json.loads(line) for line in p.stdout.splitlines()]
    if len(outputs)!=len(records):raise ValueError('Incomplete actor output')
    counts=Counter();errors=Counter();phases=Counter();events=Counter();players=Counter();failures=[]
    for i,(r,out) in enumerate(zip(records,outputs)):
        expected=r['expected'];players[len(r['query']['runtime']['actors'])]+=1
        if isinstance(expected,list):frames=expected;counts['complete_episodes']+=1
        else:frames=expected['completed'];errors[expected['error']]+=1
        counts['ticks']+=len(frames)
        for frame in frames:
            counts['shoves']+=len(frame['report']['pushes']);phases.update(e['kind'] for e in frame['report']['phases']);events.update(e['event']['kind'] for e in frame['report']['events'])
            if frame['report']['physics'] is not None:counts['collision_callbacks']+=len(frame['report']['physics']['callbacks'])
        if out!=expected:
            counts['mismatches']+=1
            if len(failures)<10:
                diff=first_difference(expected,out)
                compact=lambda x:dict(type=type(x).__name__,length=len(x)) if isinstance(x,(list,dict)) else x
                failures.append(dict(case=i,difference=[diff[0],compact(diff[1]),compact(diff[2])]))
    return dict(match=not counts['mismatches'],cases=len(records),**counts,players=dict(players),phases=dict(phases),events=dict(events),errors=dict(errors),first_failures=failures)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=120);p.add_argument('--steps',type=int,default=120);p.add_argument('--seed',type=int,default=96271);p.add_argument('--golden',action='store_true');p.add_argument('--reuse-unity',action='store_true');a=p.parse_args()
    if a.golden:
        with gzip.open(FIXTURE,'rt') as f:header=json.loads(next(f));records=[json.loads(line) for line in f]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid actor fixture')
        report=check(records)
    else:
        directory=ROOT/'native/artifacts/actors';records,hashes=reference(directory,list(queries(a.cases,a.seed,a.steps)),a.reuse_unity);report=dict(check(records),provenance=hashes)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            selected=set(range(min(12,len(records))))|set(random.Random(a.seed).sample(range(len(records)),min(24,len(records))))
            selected.update(i for i,r in enumerate(records) if isinstance(r['expected'],dict))
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,scope='Scheduled original PlayerPhysics and PlayerBody with scripted motor and shared collider transforms; scripted motor plus original equal-scale SlimeController shoves; excludes full SlimeController, bite, abilities and ropes')
            FIXTURE.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])
if __name__=='__main__':raise SystemExit(main())
