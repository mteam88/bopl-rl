"""Check the independent Rust physics stage on states from actual Unity gameplay.

Each stage starts from Unity's pre-physics state. This is stage verification,
not an independent full-match rollout; gameplay callbacks execute only in Unity.
"""
import argparse
import json
from pathlib import Path
import subprocess
from reference import ROOT, read_trace, provenance, first_difference


def verify(path):
    source=provenance(path)
    cases=[];skipped={'no_physics_stage':0,'ropes':0,'beams':0};ticks=0
    for row in read_trace(path):
        if row['kind']!='state':continue
        ticks+=1;stage=row['value'].get('physicsStage')
        if stage is None:skipped['no_physics_stage']+=1;continue
        if stage['ropes'] or stage['beams']:
            skipped['ropes']+=bool(stage['ropes']);skipped['beams']+=bool(stage['beams']);continue
        if 'after' not in stage or 'callbacks' not in stage:raise ValueError('Incomplete physics stage')
        cases.append(stage)
    if not cases:raise ValueError('No supported physics stages captured')
    queries=[{'operation':'dynamics_scene','scene':s['before'],'steps':[s['dt']]} for s in cases]
    result=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=''.join(json.dumps(q,separators=(',',':'))+'\n' for q in queries),text=True,capture_output=True,check=True)
    outputs=[json.loads(l) for l in result.stdout.splitlines()]
    if len(outputs)!=len(cases):raise ValueError('Incomplete Rust output')
    failures=[];mismatches=0;callbacks=0
    for s,actual in zip(cases,outputs):
        expected={'scene':s['after'],'callbacks':s['callbacks']}
        value={'scene':actual[0]['scene'],'callbacks':actual[0]['report']['callbacks']} if isinstance(actual,list) else actual
        callbacks+=len(s['callbacks'])
        if value!=expected:
            mismatches+=1
            if len(failures)<10:failures.append({'tick':s['tick'],'difference':first_difference(expected,value)})
    report={'trace':str(path),'scope':'independent physics stages; Unity state supplied before each stage; no gameplay callback effects',
            'match':mismatches==0,'replay_ticks':ticks,'compared_stages':len(cases),'matched_stages':len(cases)-mismatches,
            'callbacks':callbacks,'skipped':skipped,'first_failures':failures,'provenance':source}
    path.with_suffix('.physics-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='provenance'},indent=2));return int(bool(mismatches))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=Path);a=p.parse_args();raise SystemExit(verify(a.trace))
