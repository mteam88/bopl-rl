"""Differential rope integration, platform contacts, growth and reeling checks."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from reference import ROOT, first_difference
from verify_dynamics import ONE, box, sample, vec, vector

FIXTURE=ROOT/'native/fixtures/rope.jsonl.gz'


def queries(count,seed):
    rng=random.Random(seed)
    for case in range(count):
        n=2+case%39
        segments=[vec(rng,-5,5) for _ in range(40)]
        previous=[{k:v+sample(rng,-.05,.05) for k,v in p.items()} for p in segments]
        if case%11==0:segments[1]=copy.deepcopy(segments[0]);previous[1]=copy.deepcopy(previous[0])
        rectangles=[{'inner':box(rng,vec(rng,-3,3),case%4==0),'radius':sample(rng,.1,2)} for _ in range(case%5)]
        rope={'segment':segments,'previous':previous,'count':n,'enabled':case%17!=0,'force_start':vec(rng,-20,20),'force_end':vec(rng,-20,20),
            'mass_start':rng.choice([0,ONE//2,ONE,sample(rng,0,1)]),'mass_end':rng.choice([0,ONE//2,ONE,sample(rng,0,1)]),
            'separation':0 if case%79==0 else sample(rng,.05,3),'time_since_hook_arrived':rng.choice([0,1,ONE]),'detached':False,
            'colliding_rects':list(range(len(rectangles)))+[0]*(32-len(rectangles)),'collision_count':len(rectangles)}
        actions=[{'op':'bounds'},{'op':'step'},{'op':'mass','value':sample(rng,-.5,1.5),'top':case%2==0,'scale':sample(rng,0,5)},
            {'op':'detach','top':case%3==0},{'op':'spawn','point':vec(rng,-5,5)},
            {'op':'reel','minimum':rng.choice([2,n])},{'op':'reverse_reel','minimum':rng.choice([2,n])},{'op':'step'}]
        yield {'operation':'dynamics_rope','rope':rope,'rectangles':rectangles,'actions':actions,
            'config':{'dt':rng.choice([0,71582788,35791394]),'gravity':sample(rng,-3,15),'space':case%3==0,'iterations':case%9,'stretchiness':sample(rng,0,5)}}


def check(records):
    p=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=''.join(json.dumps(r['query'],separators=(',',':'))+'\n' for r in records),text=True,capture_output=True,check=True)
    actual=[json.loads(l) for l in p.stdout.splitlines()]
    if len(actual)!=len(records):raise ValueError('Incomplete Rust rope corpus')
    failures=[];errors=actions=mismatches=0
    for i,(r,value) in enumerate(zip(records,actual)):
        expected=r['expected']
        if isinstance(expected,list):actions+=len(expected)
        else:errors+=1
        if expected!=value:
            mismatches+=1
            if len(failures)<10:failures.append({'case':i,'difference':first_difference(expected,value)})
    return {'match':mismatches==0,'cases':len(records),'actions':actions,'expected_errors':errors,'mismatches':mismatches,'first_failures':failures}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=2000);p.add_argument('--seed',type=int,default=42821);p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');a=p.parse_args()
    if a.golden:
        with gzip.open(FIXTURE,'rt') as f:header=json.loads(next(f));records=[json.loads(l) for l in f]
        if header['schema']!='bopl-rope-golden-v1' or len(records)!=header['cases']:raise ValueError('Invalid rope golden')
        report=check(records);print(json.dumps(report,indent=2));return int(not report['match'])
    d=ROOT/'native/artifacts/rope';d.mkdir(exist_ok=True,parents=True)
    source=d/'input.jsonl';output=d/'unity.jsonl';inputs=list(queries(a.cases,a.seed));raw=''.join(json.dumps(q,separators=(',',':'))+'\n' for q in inputs)
    if a.reuse_unity:
        if source.read_text()!=raw:raise ValueError('Cannot reuse different rope inputs')
    else:
        source.write_text(raw);env=dict(os.environ,BOPL_KERNEL_INPUT=str(source),BOPL_KERNEL_OUTPUT=str(output));env.pop('BOPL_TRACE',None)
        with (d/'stdout.log').open('w') as out:subprocess.run([str(ROOT/'Builds/Linux/BoplBattle.x86_64'),str(ROOT/'native/fixtures/movement.rep'),'-batchmode','-nographics','-logFile',str(d/'unity.log')],env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=600)
        if 'REFERENCE_KERNEL_COMPLETE' not in (d/'unity.log').read_text():raise ValueError('Incomplete Unity rope oracle')
        (d/'reference-sha256.json').write_text(json.dumps({'assembly':hashlib.sha256((ROOT/'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll').read_bytes()).hexdigest(),'input':hashlib.sha256(source.read_bytes()).hexdigest(),'output':hashlib.sha256(output.read_bytes()).hexdigest()}))
    hashes=json.loads((d/'reference-sha256.json').read_text())
    for key,path in [('input',source),('output',output)]:
        if hashes[key]!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Rope provenance mismatch')
    expected=[json.loads(l) for l in output.read_text().splitlines()]
    if len(expected)!=len(inputs):raise ValueError('Incomplete reference rope corpus')
    records=[{'query':q,'expected':e} for q,e in zip(inputs,expected)];report={**check(records),'provenance':hashes}
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if report['match']:
        selected=sorted(set(range(min(100,len(records))))|set(random.Random(a.seed).sample(range(len(records)),min(100,len(records)))))
        header={'schema':'bopl-rope-golden-v1','cases':len(selected),'provenance':hashes}
        raw=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in [header,*[records[i] for i in selected]])
        FIXTURE.write_bytes(gzip.compress(raw.encode(),mtime=0))
    return int(not report['match'])
if __name__=='__main__':raise SystemExit(main())
