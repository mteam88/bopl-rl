"""Compare multi-step rigid/composite scenes with actual Unity DetPhysics.Simulate."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from reference import ROOT, first_difference
from verify_dynamics import ONE, body, box, sample, vec, vector


def queries(count, seed):
    rng=random.Random(seed)
    for case in range(count):
        layers=[0]*32
        for a in range(32):
            for b in range(a,32):
                if rng.random()<.8:layers[a]|=1<<b;layers[b]|=1<<a
        objects=[]
        for i in range(case%19+1):
            kind=['circle','box','rounded_rect'][(case+i)%3]
            center=vec(rng,-7,7)
            shape={'center':center,'radius':sample(rng,.1,2)} if kind=='circle' else box(rng,center,case%7==0)
            if kind=='rounded_rect':shape={'inner':shape,'radius':sample(rng,.05,1.5)}
            ba=body(rng)
            # Include static and trigger bodies; avoid two static bodies in contact
            # in most cases so long trajectory coverage survives division-by-zero.
            if i==0 and case%4==0:ba['inverse_mass']=ba['inverse_inertia']=0
            if i%5==3:ba['inverse_mass']=-ONE
            ba['velocity']=vec(rng,-5,5);ba['angular_velocity']=sample(rng,-1,1)
            objects.append(dict(id=i+1,collider={'kind':kind,'shape':shape},body=ba,
                transform={'position':copy.deepcopy(center),'rotation':sample(rng,-3,3),'offset':vec(rng,-.2,.2)},
                layer=rng.randrange(32),active=rng.random()>.12,callback=True,callback_destroyed=rng.random()<.06,
                destroyed=case%17==0 and i%3==0,composite=None,relative_pos=vector(0,0),relative_orientation=0))
        composites=[]
        if case%2==0 and len(objects)>3:
            # Each group has one platform root and circle/box attachments.
            members=objects[:3]
            comp=dict(body=body(rng),position=vec(rng,-5,5),destroyed=case%14==0,circles=[],boxes=[],rounded_rects=[])
            comp['body']['velocity']=vec(rng,-3,3);comp['body']['angular_velocity']=sample(rng,-1,1)
            for o in members:
                o['composite']=0;o['destroyed']=False
                o['body']['inverse_mass']=sample(rng,.1,2)
                o['relative_pos']=vec(rng,-2,2);o['relative_orientation']=sample(rng,-1,1)
                comp[{'circle':'circles','box':'boxes','rounded_rect':'rounded_rects'}[o['collider']['kind']]].append(o['id'])
            composites.append(comp)
        # Registration order is intentionally different from stable IDs.
        rng.shuffle(objects)
        commands=[[] for _ in range(12)]
        if case%3==0 and len(objects)>3:
            composites=[]
            for o in objects:o['composite']=None
            root=next(o for o in objects if o['collider']['kind']=='rounded_rect')
            circle=next(o for o in objects if o['collider']['kind']=='circle')
            box_child=next(o for o in objects if o['collider']['kind']=='box')
            for o in (root,circle,box_child):
                o['destroyed']=False;o['body']['inverse_mass']=sample(rng,.1,2);o['body']['inverse_inertia']=sample(rng,.1,2)
            commands[0]=[{'op':'attach','id':o['id'],'root':root['id']} for o in (circle,box_child)]
            commands[1]=[{'op':'force','id':o['id'],'force':vec(rng,-100,100)} for o in (root,circle,box_child)]
            commands[3]=[{'op':'detach','id':circle['id']}]
            commands[4]=[{'op':'attach','id':circle['id'],'root':root['id']}]
            commands[6]=[{'op':'destroy','id':box_child['id']}]
            commands[8]=[{'op':'destroy','id':root['id']}]
        yield {'operation':'dynamics_scene','scene':{'objects':objects,'composites':composites,'layers':layers,'gravity':sample(rng,-12,0)},
               'steps':[0 if j%7==0 else rng.choice([71582788,71582788,35791394]) for j in range(12)],'commands':commands}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,default=1000);parser.add_argument('--seed',type=int,default=943184)
    parser.add_argument('--reuse-unity',action='store_true');parser.add_argument('--directory',type=Path,default=ROOT/'native/artifacts/physics-scene')
    args=parser.parse_args();d=args.directory.resolve();d.mkdir(parents=True,exist_ok=True)
    source=d/'input.jsonl';reference=d/'unity.jsonl';rust=d/'rust.jsonl'
    inputs=list(queries(args.cases,args.seed));raw=''.join(json.dumps(q,separators=(',',':'))+'\n' for q in inputs)
    if args.reuse_unity:
        if source.read_text()!=raw:raise ValueError('Cannot reuse different input corpus')
    else:
        source.write_text(raw);env=dict(os.environ,BOPL_KERNEL_INPUT=str(source),BOPL_KERNEL_OUTPUT=str(reference))
        for key in ('BOPL_TRACE','BOPL_TRACE_TICKS','BOPL_TRACE_BATCH'):env.pop(key,None)
        with (d/'stdout.log').open('w') as out:
            subprocess.run([str(ROOT/'Builds/Linux/BoplBattle.x86_64'),str(ROOT/'native/fixtures/movement.rep'),'-batchmode','-nographics','-logFile',str(d/'unity.log')],env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=600)
        if 'REFERENCE_KERNEL_COMPLETE' not in (d/'unity.log').read_text():raise ValueError('Incomplete Unity reference')
        (d/'reference-sha256.json').write_text(json.dumps({
            'assembly':hashlib.sha256((ROOT/'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll').read_bytes()).hexdigest(),
            'input':hashlib.sha256(source.read_bytes()).hexdigest(),'output':hashlib.sha256(reference.read_bytes()).hexdigest()}))
    hashes=json.loads((d/'reference-sha256.json').read_text())
    for key,p in [('input',source),('output',reference)]:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[key]:raise ValueError('Reference provenance mismatch')
    with source.open('rb') as inp,rust.open('wb') as out:subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],stdin=inp,stdout=out,check=True)
    failures=[];steps=callbacks=pairs=errors=composite_scenes=0;lines=0
    with reference.open() as a,rust.open() as b:
        for i,(al,bl) in enumerate(zip(a,b,strict=True)):
            lines+=1;left=json.loads(al);right=json.loads(bl)
            if left!=right and len(failures)<10:failures.append({'case':i,'difference':first_difference(left,right)})
            if isinstance(left,dict):errors+=1;continue
            steps+=len(left);composite_scenes+=any(step['scene']['composites'] for step in left)
            for step in left:
                callbacks+=len(step['report']['callbacks']);pairs+=sum(map(len,step['report']['pairs']))
    if lines!=len(inputs):raise ValueError('Incomplete output corpus')
    report={'match':not failures,'scenes':lines,'successful_steps':steps,'composite_scenes':composite_scenes,'callbacks':callbacks,'potential_pairs':pairs,'expected_errors':errors,'first_failures':failures,'provenance':hashes}
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())
