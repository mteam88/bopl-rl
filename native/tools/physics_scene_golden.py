"""Standalone, multi-step regressions captured from the actual Unity physics world."""
import argparse
import gzip
import hashlib
import json
import random
import subprocess
from reference import ROOT, first_difference

FIXTURE=ROOT/'native/fixtures/physics-scene.jsonl.gz'


def export():
    directory=ROOT/'native/artifacts/physics-scene'
    report=json.loads((directory/'report.json').read_text())
    if not report['match']:raise ValueError('Cannot promote a failing scene corpus')
    source=directory/'input.jsonl';output=directory/'unity.jsonl'
    for key,path in [('input',source),('output',output)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=report['provenance'][key]:raise ValueError('Reference provenance mismatch')
    selected=sorted(set(range(20))|set(random.Random(943184).sample(range(report['scenes']),40)))
    records=[];lines=0
    with source.open() as a,output.open() as b:
        for i,(al,bl) in enumerate(zip(a,b,strict=True)):
            lines+=1
            if i in selected:records.append({'query':json.loads(al),'expected':json.loads(bl)})
    if lines!=report['scenes']:raise ValueError('Incomplete scene corpus')
    header={'schema':'bopl-physics-scene-golden-v1','scenes':len(records),'steps':sum(len(r['expected']) for r in records if isinstance(r['expected'],list)), 'reference_provenance':report['provenance']}
    raw=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in [header,*records])
    FIXTURE.write_bytes(gzip.compress(raw.encode(),mtime=0))
    print(json.dumps({'exported':str(FIXTURE),'scenes':header['scenes'],'steps':header['steps']}))


def verify():
    with gzip.open(FIXTURE,'rt') as stream:
        header=json.loads(next(stream));records=[json.loads(l) for l in stream]
    if header['schema']!='bopl-physics-scene-golden-v1' or len(records)!=header['scenes'] or not records:raise ValueError('Invalid scene golden')
    result=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=''.join(json.dumps(r['query'])+'\n' for r in records),text=True,capture_output=True,check=True)
    actual=[json.loads(l) for l in result.stdout.splitlines()]
    if len(actual)!=len(records):raise ValueError('Incomplete Rust output')
    for i,(r,value) in enumerate(zip(records,actual)):
        if r['expected']!=value:raise ValueError(json.dumps({'scene':i,'difference':first_difference(r['expected'],value)}))
    print(json.dumps({'physics_scene_match':True,'scenes':len(records),'steps':header['steps'],'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest()}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['export','verify']);args=parser.parse_args()
    export() if args.command=='export' else verify()
