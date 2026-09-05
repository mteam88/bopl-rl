"""Verify the actual player movement controller and callback order in Unity."""
import argparse
from collections import Counter
import gzip
import json
import random
import subprocess
import struct
from reference import ROOT,first_difference
from verify_beam import encoded,reference
from verify_surface_motion import queries as surface_queries
from verify_player_physics import queries as player_queries
from verify_player_ground import vec
from verify_dynamics import ONE

FIXTURE=ROOT/'native/fixtures/player-controller.jsonl.gz'
SCHEMA='bopl-player-controller-v1'


def queries(count,seed):
    rng=random.Random(seed)
    for case,(terrain,player) in enumerate(zip(surface_queries(count,seed),player_queries(count,seed+1))):
        bindings=terrain['bindings']
        if case%3==0:
            # Roll supplies five exclusion slots. Keep at most five platforms
            # here so this corpus has a finite recursion boundary even in overlaps.
            kept={10000,*[b['object'] for b in bindings if b['object']!=10000][:4]}
            bindings[:]=[b for b in bindings if b['object'] in kept]
            terrain['scene']['objects']=[o for o in terrain['scene']['objects'] if o['collider']['kind']!='rounded_rect' or o['id'] in kept]
            terrain['scene']['composites']=[]
            for o in terrain['scene']['objects']:o['composite']=None
        root=next(i for i,b in enumerate(bindings) if b['object']==10000)
        surface=bindings[root]['platform']['surface']
        radius=ONE//2 if case%43 else 0
        physics=player['state'];motion=physics['motion']
        height=surface['extents']['y']+surface['radius']+radius
        motion.update(position={k:surface['center'][k]+surface['up'][k]*height//ONE for k in ['x','y']},
            ground=None if case%13==0 else root,grounded=case%4!=0,radius=radius,local=ONE//8,
            velocity=vec(rng.randint(-12,12),rng.randint(-12,12)),external_velocity=vec(rng.randint(-20,20),rng.randint(-20,20)))
        physics.update(scale=ONE if case%17 else 2*ONE,previous_scale=0 if case%47==0 else ONE)
        config=player['config'];config.update(gravity=int(1.6*ONE),max_fall_speed=0 if case%53==0 else 27*ONE,max_speed=19*ONE,air_accel=10*ONE)
        state=dict(physics=physics,skip_update=case%5==0,apply_gravity=case%7!=0,blocked=case%11!=0)
        grounds=[dict(ice=(i+case)%3==0,drop_frames=2 if case%5==0 else 0,dont_drop=(case+i)%7,drop_force=(10 if case%2 else -10)*ONE) for i in range(len(bindings))]
        players=[dict(id=i+1,protected=bool((case+i)%3==0)) for i in range(case%6+1)]
        holder=None if case%29==0 else 99 if case%31==0 else case%len(players)+1
        actions=[]
        for i in range(3):
            values=dict(current=rng.randint(-40,40)*ONE,target=rng.randint(-40,40)*ONE,accel=rng.randint(-10,100)*ONE,dt=[0,ONE//60,ONE][i])
            actions.extend([dict(op='increment',**values),dict(op='accelerate',**values)])
        actions.append(dict(op='alignment',input=vec(.6,.8)))
        if case%5==0:actions.append(dict(op='drop'))
        if case%7==0:actions.append(dict(op='hop',radius=radius,speed=30*ONE,dt=ONE))
        for i,dt in enumerate([ONE//60,0,ONE//10,ONE,-ONE]):
            direction=[vec(1,0),vec(),vec(-1,0),vec(.01,-1),vec(.2,1)][(case+i)%5]
            actions.extend([dict(op='update',dt=dt),dict(op='move',input=direction,dt=dt,surface=bool(i%3)),
                dict(op='align'),dict(op='macho',input=direction,dt=dt,multiplier=[0,ONE//2,ONE,2*ONE,-ONE][i]),
                dict(op='current',dt=dt),dict(op='drop')])
        actions.append(dict(op='hop',radius=radius,speed=50*ONE,dt=ONE//60))
        yield dict(operation='dynamics_player_controller',scene=terrain['scene'],bindings=bindings,grounds=grounds,
            state=state,config=dict(physics=config,accel=180*ONE,friction=ONE//2,ice_friction=int(struct.unpack('<f',struct.pack('<f',.87))[0]*ONE),
                alignment_deadzone=int(struct.unpack('<f',struct.pack('<f',.2))[0]*ONE)),players=players,holder=holder,
            time_scale=[0,ONE//4,ONE,2*ONE,-ONE][case%5],space=bool(case%3==0),ability=bool(case%2),
            hop_buffer=[-1]*(5 if case%3==0 else 64),detach_offset=vec(.01,-.02),detach_velocity=vec(3,-2),attach_offset=vec(.03,.01),exit_offset=vec(-.04,.02),actions=actions)


def check(records):
    p=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=encoded(r['query'] for r in records),text=True,capture_output=True,check=True,timeout=300)
    outputs=[json.loads(line) for line in p.stdout.splitlines()]
    if len(outputs)!=len(records):raise ValueError('Incomplete controller output')
    counts,actions,events,errors,buffers=(Counter() for _ in range(5));failures=[]
    for i,(r,out) in enumerate(zip(records,outputs)):
        expected=r['expected']
        buffers[len(r['query']['hop_buffer'])]+=1
        if isinstance(expected,list):frames=expected;counts['complete_scenarios']+=1
        else:frames=expected['completed'];errors[expected['error']]+=1
        counts['completed_actions']+=len(frames)
        actions.update(a['op'] for a in r['query']['actions'][:len(frames)])
        events.update(e['kind'] for f in frames for e in f['events'])
        previous=r['query']['hop_buffer']
        for action,frame in zip(r['query']['actions'],frames):
            if action['op']=='hop':
                visits=sum(x!=-1 for x in frame['hop_buffer'])-sum(x!=-1 for x in previous)
                counts['hop_platform_transitions']+=max(0,visits-1)
                counts['max_hop_platform_transitions']=max(counts['max_hop_platform_transitions'],max(0,visits-1))
            previous=frame['hop_buffer']
        if out!=expected:
            counts['mismatches']+=1
            if len(failures)<10:
                diff=first_difference(expected,out)
                compact=lambda v:dict(type=type(v).__name__,length=len(v)) if isinstance(v,(dict,list)) else v
                failures.append(dict(case=i,difference=[diff[0],compact(diff[1]),compact(diff[2])]))
    return dict(match=not counts['mismatches'],cases=len(records),**counts,actions=dict(actions),events=dict(events),errors=dict(errors),hop_buffer_sizes=dict(buffers),first_failures=failures)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=int,default=1000);p.add_argument('--seed',type=int,default=61439)
    p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');args=p.parse_args()
    if args.golden:
        with gzip.open(FIXTURE,'rt') as stream:header=json.loads(next(stream));records=[json.loads(line) for line in stream]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid controller fixture')
        report=check(records)
    else:
        if args.cases<1:p.error('--cases must be positive')
        directory=ROOT/'native/artifacts/player-controller'
        records,hashes=reference(directory,list(queries(args.cases,args.seed)),args.reuse_unity)
        report=dict(check(records),provenance=hashes);(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            selected=set(range(min(32,len(records))))|set(random.Random(args.seed).sample(range(len(records)),min(128,len(records))))
            selected.update(i for i,r in enumerate(records) if isinstance(r['expected'],dict))
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,
                scope='Actual PlayerPhysics UpdateSim, Move, MachoMove, drop/current movement and recursive hopping; mutable callback sinks and supplied world state; not full ability episodes')
            FIXTURE.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])


if __name__=='__main__':raise SystemExit(main())
