"""Standalone regression checks for the Unity-verified rigid-body primitives."""
import argparse
import gzip
import hashlib
import json
import random
import subprocess
from reference import ROOT

FIXTURE=ROOT/'native/fixtures/dynamics.jsonl.gz'


def export():
    directory=ROOT/'native/artifacts/dynamics'
    report=json.loads((directory/'report.json').read_text())
    if not report['match']:raise ValueError('Cannot promote a failing dynamics corpus')
    source=directory/'input.jsonl';output=directory/'unity.jsonl'
    for name,path in [('input',source),('output',output)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=report['provenance'][name]:raise ValueError('Reference provenance differs')
    queries=[json.loads(l) for l in source.read_text().splitlines()];expected=[json.loads(l) for l in output.read_text().splitlines()]
    if len(queries)!=report['cases'] or len(expected)!=len(queries):raise ValueError('Incomplete reference')
    # Include the first boundary cases, then a seeded sample across the full corpus.
    selected=sorted(set(range(270))|set(random.Random(67484).sample(range(len(queries)),1500)))
    header={'schema':'bopl-dynamics-golden-v1','cases':len(selected),'reference_provenance':report['provenance']}
    records=[{'query':queries[i],'expected':expected[i]} for i in selected]
    raw=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in [header,*records])
    FIXTURE.write_bytes(gzip.compress(raw.encode(),mtime=0))
    print(json.dumps({'exported':str(FIXTURE),'cases':len(records)}))


def verify():
    with gzip.open(FIXTURE,'rt') as stream:
        header=json.loads(next(stream));records=[json.loads(l) for l in stream]
    if header['schema']!='bopl-dynamics-golden-v1' or len(records)!=header['cases'] or not records:raise ValueError('Invalid dynamics golden')
    result=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=''.join(json.dumps(r['query'])+'\n' for r in records),text=True,capture_output=True,check=True)
    outputs=[json.loads(l) for l in result.stdout.splitlines()]
    if len(outputs)!=len(records):raise ValueError('Incomplete Rust output')
    for index,(record,actual) in enumerate(zip(records,outputs)):
        if actual!=record['expected']:raise ValueError(json.dumps({'index':index,**record,'actual':actual}))
    print(json.dumps({'dynamics_match':True,'cases':len(records),'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['export','verify']);args=parser.parse_args()
    export() if args.command=='export' else verify()
