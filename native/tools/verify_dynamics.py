"""Compare contact manifolds and body dynamics against the actual Unity functions."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
from reference import ROOT

ONE = 1 << 32
OPS = ['circle_circle','circle_box','circle_rounded','box_box','box_rounded','rounded_rounded','integrate','resolve','correction']


def vector(x,y): return {'x':x,'y':y}
def sample(rng, low, high): return rng.randrange(int(low*ONE), int(high*ONE))
def vec(rng, low, high): return vector(sample(rng,low,high),sample(rng,low,high))

def body(rng):
    return dict(inverse_mass=sample(rng,0.05,5), velocity=vec(rng,-40,40), force=vec(rng,-100,100), angular_velocity=sample(rng,-15,15),
        orientation=sample(rng,-30,30), inverse_inertia=sample(rng,0.02,5), bounciness=sample(rng,0,2),
        static_friction=sample(rng,0,1), dynamic_friction=sample(rng,0,1), gravity_scale=sample(rng,-2,3),
        air_drag=sample(rng,0.9,1), rotational_drag=sample(rng,0.9,1))

def box(rng, center, axis_aligned):
    angle = 0 if axis_aligned else rng.uniform(-math.pi,math.pi)
    ex,ey = rng.uniform(.1,5),rng.uniform(.1,5)
    return {'center':center,'right':vector(int(math.cos(angle)*ex*ONE),int(math.sin(angle)*ex*ONE)),
        'up':vector(int(-math.sin(angle)*ey*ONE),int(math.cos(angle)*ey*ONE)), 'inverse_extents':vector(int(ONE/ex),int(ONE/ey))}

def queries(count,seed):
    rng=random.Random(seed)
    for i in range(count):
        ca=vec(rng,-20,20); cb={k:v+sample(rng,-5,5) for k,v in ca.items()}
        a=box(rng,ca,i%4==0); b=box(rng,cb,i%4==0)
        if i%11==0: b=copy.deepcopy(a);cb=b['center']
        if i%13==0:
            b=copy.deepcopy(a);b['center']['x']+=2*a['right']['x']+rng.choice([-1,0,1]);cb=b['center']
        ac={'center':ca,'radius':sample(rng,.1,4)};bc={'center':cb,'radius':sample(rng,.1,4)}
        ar={'inner':a,'radius':sample(rng,0,2)};br={'inner':b,'radius':sample(rng,0,2)}
        for op,x,y in [('circle_circle',ac,bc),('circle_box',ac,b),('circle_rounded',ac,br),('box_box',a,b),('box_rounded',a,br),('rounded_rounded',ar,br)]:
            yield {'operation':'dynamics_'+op,'a':x,'b':y}
        ba=body(rng);bb=body(rng)
        if i%3==0:ba['inverse_mass']=ba['inverse_inertia']=0
        if i%17==0:bb['inverse_mass']=bb['inverse_inertia']=0
        yield {'operation':'dynamics_integrate','body':ba,'dt':rng.choice([0,71582788,ONE//120,ONE//30]),'gravity':sample(rng,-20,0)}
        angle=rng.uniform(-math.pi,math.pi)
        contact={'penetration':sample(rng,0,5),'normal':vector(int(math.cos(angle)*ONE),int(math.sin(angle)*ONE)),'point':vec(rng,-20,20)}
        if i%9==0:contact['point']=ca
        yield {'operation':'dynamics_resolve','a':ba,'b':bb,'contact':contact,'pos_a':ca,'pos_b':cb}
        yield {'operation':'dynamics_correction','contact':contact,'inverse_a':ba['inverse_mass'],'inverse_b':bb['inverse_mass'],'full':i%2==0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,default=3000)
    parser.add_argument('--seed',type=int,default=67484)
    parser.add_argument('--reuse-unity',action='store_true')
    args=parser.parse_args()
    directory=ROOT/'native/artifacts/dynamics';directory.mkdir(parents=True,exist_ok=True)
    inputs=list(queries(args.cases,args.seed));raw=''.join(json.dumps(q,separators=(',',':'))+'\n' for q in inputs)
    source=directory/'input.jsonl'
    if args.reuse_unity:
        if source.read_text()!=raw:raise ValueError('Cannot reuse reference from different queries')
    else:source.write_text(raw)
    binary=ROOT/'Builds/Linux/BoplBattle.x86_64';assembly=binary.parent/'BoplBattle_Data/Managed/Assembly-CSharp.dll'
    reference=directory/'unity.jsonl';unity_log=directory/'unity.log'
    if not args.reuse_unity:
        env=dict(os.environ,BOPL_KERNEL_INPUT=str(source),BOPL_KERNEL_OUTPUT=str(reference))
        for key in ('BOPL_TRACE','BOPL_TRACE_TICKS','BOPL_TRACE_BATCH'):env.pop(key,None)
        with (directory/'stdout.log').open('w') as out:
            subprocess.run([str(binary),str(ROOT/'native/fixtures/movement.rep'),'-batchmode','-nographics','-logFile',str(unity_log)],env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=300)
        if 'REFERENCE_KERNEL_COMPLETE' not in unity_log.read_text():raise ValueError('Unity probe did not complete')
        (directory/'reference-sha256.json').write_text(json.dumps({'assembly':hashlib.sha256(assembly.read_bytes()).hexdigest(),'input':hashlib.sha256(source.read_bytes()).hexdigest(),'output':hashlib.sha256(reference.read_bytes()).hexdigest()}))
    hashes=json.loads((directory/'reference-sha256.json').read_text())
    if hashes['input']!=hashlib.sha256(source.read_bytes()).hexdigest() or hashes['output']!=hashlib.sha256(reference.read_bytes()).hexdigest():raise ValueError('Reference provenance differs')
    with source.open('rb') as inp,(directory/'rust.jsonl').open('wb') as out:
        subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],stdin=inp,stdout=out,check=True)
    left=[json.loads(l) for l in reference.read_text().splitlines()];right=[json.loads(l) for l in (directory/'rust.jsonl').read_text().splitlines()]
    if len(left)!=len(inputs) or len(right)!=len(inputs):raise ValueError('Incomplete query corpus')
    counts={op:{'cases':0,'hits':0,'expected_errors':0,'mismatches':0} for op in OPS};failures=[]
    for query,a,b in zip(inputs,left,right):
        c=counts[query['operation'].removeprefix('dynamics_')];c['cases']+=1
        if a is not None: c['hits']+=1
        if isinstance(a,dict) and 'error' in a:c['expected_errors']+=1
        if a!=b:
            c['mismatches']+=1
            if len(failures)<12:failures.append({'query':query,'unity':a,'rust':b})
    report={'match':not failures,'cases':len(inputs),'operations':counts,'first_failures':failures,'provenance':hashes}
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return int(bool(failures))

if __name__=='__main__':raise SystemExit(main())
