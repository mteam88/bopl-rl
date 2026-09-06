"""Compare original BoplBody methods on real circle/box/rounded-rectangle components."""
import argparse
import gzip
import json
import random
import subprocess
from collections import Counter
from reference import ROOT, first_difference
from verify_beam import encoded, reference
from verify_dynamics import ONE, body, box, sample, vec, vector

FIXTURE=ROOT/'native/fixtures/bopl-body.jsonl.gz'
SCHEMA='bopl-body-v1'


def queries(count,seed):
    rng=random.Random(seed)
    modes=['ordinary','resizable','zero_mass','zero_inertia','hold_zero','hold_negative','hold_overflow','scale_zero','get_zero_mass','set_zero_mass','get_zero_inertia','set_zero_inertia']
    for case in range(count):
        shape=['circle','box','rounded_rect'][case%3]
        initialized=bool((case//3)%2)
        composite=['none','active','destroyed'][(case//6)%3]
        mode=modes[(case//18)%len(modes)]
        center=vec(rng,-5,5)
        geometry=dict(center=center,radius=sample(rng,.1,2)) if shape=='circle' else box(rng,center,False)
        if shape=='rounded_rect':geometry=dict(inner=geometry,radius=sample(rng,.1,2))
        obj=dict(id=1,collider=dict(kind=shape,shape=geometry),body=body(rng),transform=dict(position=center,rotation=sample(rng,-3,3),offset=vec(rng,-1,1)),layer=0,active=bool(case%5),callback=False,callback_destroyed=False,destroyed=False,composite=None,relative_pos=vector(0,0),relative_orientation=0)
        comps=[]
        if composite!='none':
            comp=dict(body=body(rng),position=vec(rng,-5,5),destroyed=composite=='destroyed',circles=[],boxes=[],rounded_rects=[])
            comp[dict(circle='circles',box='boxes',rounded_rect='rounded_rects')[shape]]=[1]
            comps=[comp];obj['composite']=0
        config=dict(mass=sample(rng,.1,8),inertia=sample(rng,.1,10000),bounciness=sample(rng,-1,2),friction=sample(rng,-1,2),dynamic_friction=sample(rng,-1,2),gravity=sample(rng,-3,3),air_drag=sample(rng,0,1.1),rotational_drag=sample(rng,0,1.1),velocity=vec(rng,-10,10),angular_velocity=sample(rng,-10,10))
        if mode in ['zero_mass','get_zero_mass']:config['mass']=0
        if mode in ['zero_inertia','get_zero_inertia']:config['inertia']=0
        state=dict(config=config,resizable=mode=='resizable',prev_scale=ONE,frames_until_drop=-1,magnet=None,sorting=None if case%4==0 else dict(layer='Default',order=17),original_sorting=dict(layer='Default',order=4))
        actions=[dict(op='init'),dict(op='new_body',orientation=sample(rng,-5,5))]
        if mode in ['get_zero_mass','get_zero_inertia']:actions.append(dict(op='inverse_mass' if mode=='get_zero_mass' else 'inverse_inertia'))
        if mode in ['set_zero_mass','set_zero_inertia']:actions.append(dict(op='set_mass' if mode=='set_zero_mass' else 'set_inertia',value=0))
        actions.extend([dict(op='set_velocity',value=vec(rng,-10,10)),dict(op='set_angular',value=sample(rng,-10,10)),dict(op='new_body',orientation=sample(rng,-5,5)),dict(op='body'),dict(op='velocity'),dict(op='angular')])
        actions.extend([dict(op='force',force=vec(rng,-10,10),impulse=bool(i)) for i in range(2)])
        actions.extend([dict(op='force_at',force=vec(rng,-10,10),position=vec(rng,-10,10) if i else center,impulse=bool(i)) for i in range(2)])
        actions.extend([dict(op='transform_scale',value=2*ONE),dict(op='update',dt=0),dict(op='update',dt=71582788)])
        actions.extend([dict(op='transform_scale',value=-ONE//2),dict(op='update',dt=0)])
        frames=dict(hold_zero=0,hold_negative=-7,hold_overflow=-(1<<63)).get(mode,3)
        actions.extend([dict(op='hold',frames=frames,magnet=0,change_sorting=True),dict(op='held'),dict(op='update',dt=0),dict(op='held'),dict(op='update',dt=71582788),dict(op='held'),dict(op='update',dt=0),dict(op='held'),dict(op='drop'),dict(op='held')])
        actions.extend([dict(op='hold',frames=3,magnet=1,change_sorting=False),dict(op='hold',frames=2,magnet=None,change_sorting=True),dict(op='update',dt=0),dict(op='update',dt=0)])
        if mode=='scale_zero':actions.extend([dict(op='transform_scale',value=0),dict(op='update',dt=0)])
        if mode not in ['zero_mass','zero_inertia','get_zero_mass','get_zero_inertia','set_zero_mass','set_zero_inertia']:
            actions.extend([dict(op='inverse_mass'),dict(op='inverse_inertia')])
        actions.extend([dict(op='set_mass',value=sample(rng,.1,4)),dict(op='set_inertia',value=sample(rng,.1,4)),dict(op='new_body',orientation=0),dict(op='inverse_mass'),dict(op='inverse_inertia')])
        yield dict(operation='dynamics_bopl_body',body_case=mode,scene=dict(objects=[obj],composites=comps,layers=[0]*32,gravity=-10*ONE),state=state,object=1,initialized=initialized,transform_scale=ONE,actions=actions)


def check(records,output=None):
    run=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=encoded(r['query'] for r in records),text=True,capture_output=True,check=True)
    if output:output.write_text(run.stdout)
    actual=[json.loads(line) for line in run.stdout.splitlines()]
    if len(actual)!=len(records):raise ValueError('Incomplete BoplBody output')
    counts=Counter();errors=Counter();actions=Counter();modes=Counter();shapes=Counter();failures=[]
    for i,(record,result) in enumerate(zip(records,actual)):
        q=record['query'];expected=record['expected'];frames=expected if isinstance(expected,list) else expected['completed']
        counts['actions']+=len(frames);modes[q['body_case']]+=1;shapes[q['scene']['objects'][0]['collider']['kind']]+=1
        if isinstance(expected,dict):errors[expected['error']]+=1
        else:counts['completed_cases']+=1
        mode=q['body_case'];obj=q['scene']['objects'][0]
        live_composite=obj['composite'] is not None and not q['scene']['composites'][obj['composite']]['destroyed']
        must_error=(not q['initialized'] and mode in ['get_zero_mass','get_zero_inertia','set_zero_mass','set_zero_inertia']) or (mode=='scale_zero' and q['initialized'] and (not live_composite or obj['collider']['kind']=='rounded_rect'))
        if must_error != isinstance(expected,dict) or isinstance(expected,dict) and expected['error']!='division_by_zero':
            raise ValueError(f'Unexpected completion/error boundary for BoplBody case {i}')
        before=dict(state=q['state'],scene=q['scene'])
        for action,frame in zip(q['actions'],frames):
            actions[action['op']]+=1
            counts['scale_mass_changes']+=action['op']=='update' and (frame['scene']!=before['scene'])
            counts['magnet_cleared']+=before['state']['magnet'] is not None and frame['state']['magnet'] is None
            counts['negative_hold_retained']+=frame['state']['frames_until_drop']<0 and frame['state']['magnet'] is not None
            before=frame
        if result!=expected:
            counts['mismatches']+=1
            if len(failures)<10:failures.append(dict(case=i,difference=first_difference(expected,result)))
    return dict(match=not counts['mismatches'],cases=len(records),counts=dict(counts),errors=dict(errors),actions=dict(actions),modes=dict(modes),shapes=dict(shapes),failures=failures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,default=432);parser.add_argument('--seed',type=int,default=782083)
    parser.add_argument('--directory',default='bopl-body');parser.add_argument('--golden',action='store_true');parser.add_argument('--reuse-unity',action='store_true')
    args=parser.parse_args()
    if args.golden:
        with gzip.open(FIXTURE,'rt') as stream:header=json.loads(next(stream));records=[json.loads(line) for line in stream]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid BoplBody fixture')
        report=check(records)
    else:
        directory=ROOT/'native/artifacts'/args.directory
        records,hashes=reference(directory,list(queries(args.cases,args.seed)),args.reuse_unity)
        report=dict(check(records,directory/'native.jsonl'),provenance=hashes)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            header=dict(schema=SCHEMA,cases=len(records),provenance=hashes,scope='Actual BoplBody methods on real colliders with seeded physics; collider initialization and whole-scheduler integration are separate')
            FIXTURE.write_bytes(gzip.compress(encoded([header,*records]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])

if __name__=='__main__':raise SystemExit(main())
