"""Keep exact physics-stage regressions from real gameplay captures.

Unity supplies the state before each stage. These are not full native rollouts.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from reference import ROOT, read_trace, provenance, first_difference

FIXTURE=ROOT/'native/fixtures/physics-gameplay.jsonl.gz'


def export(paths):
    records=[];sources=[]
    for path in paths:
        report=json.loads(path.with_suffix('.physics-report.json').read_text())
        source=provenance(path)
        if not report['match'] or report['provenance']!=source:raise ValueError('Cannot promote unverified gameplay stages')
        stages=[r['value']['physicsStage'] for r in read_trace(path) if r['kind']=='state' and r['value'].get('physicsStage') is not None]
        stages=[s for s in stages if not s['ropes'] and not s['beams']]
        # Preserve spawn-to-movement boundaries plus an even sample of later play.
        indices=sorted({i for i,s in enumerate(stages) if s['tick'] in range(237,246)}|set(range(0,len(stages),max(1,len(stages)//40))))
        for i in indices:
            s=stages[i]
            records.append({'source':len(sources),'tick':s['tick'],'query':{'operation':'dynamics_scene','scene':s['before'],'steps':[s['dt']]},
                'expected':{'scene':s['after'],'callbacks':s['callbacks']}})
        sources.append(source)
    header={'schema':'bopl-gameplay-physics-stage-v1','stages':len(records),'sources':sources,
        'scope':'Unity supplies the state before each independent physics stage; gameplay callback effects are excluded'}
    raw=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in [header,*records]);FIXTURE.write_bytes(gzip.compress(raw.encode(),mtime=0))
    print(json.dumps({'exported':str(FIXTURE),'stages':len(records)}))


def verify():
    with gzip.open(FIXTURE,'rt') as f:header=json.loads(next(f));records=[json.loads(l) for l in f]
    if header['schema']!='bopl-gameplay-physics-stage-v1' or len(records)!=header['stages'] or not records:raise ValueError('Invalid gameplay physics fixture')
    p=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=''.join(json.dumps(r['query'])+'\n' for r in records),text=True,capture_output=True,check=True)
    actual=[json.loads(l) for l in p.stdout.splitlines()]
    if len(actual)!=len(records):raise ValueError('Incomplete Rust output')
    for i,(r,out) in enumerate(zip(records,actual)):
        value={'scene':out[0]['scene'],'callbacks':out[0]['report']['callbacks']} if isinstance(out,list) else out
        if value!=r['expected']:raise ValueError(json.dumps({'stage':i,'tick':r['tick'],'difference':first_difference(r['expected'],value)}))
    print(json.dumps({'gameplay_physics_stage_match':True,'stages':len(records),'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest()}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['export','verify']);p.add_argument('traces',nargs='*',type=Path);a=p.parse_args()
    if a.command=='export':
        if not a.traces:p.error('export requires verified capture paths')
        export(a.traces)
    else:verify()
