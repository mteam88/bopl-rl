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
    skipped={'no_physics_stage':0,'ropes':0,'beams':0};ticks=cases=rope_stages=max_ropes=0
    inputs=path.with_suffix('.physics-input.jsonl');expected_path=path.with_suffix('.physics-expected.jsonl');native=path.with_suffix('.physics-native.jsonl')
    with inputs.open('w') as inp,expected_path.open('w') as expected:
        for row in read_trace(path):
            if row['kind']!='state':continue
            ticks+=1;stage=row['value'].get('physicsStage')
            if stage is None:skipped['no_physics_stage']+=1;continue
            missing_ropes=bool(stage['ropes']) and 'rope_system' not in stage['before']
            if missing_ropes or stage['beams']:
                skipped['ropes']+=missing_ropes;skipped['beams']+=bool(stage['beams']);continue
            if 'after' not in stage or 'callbacks' not in stage:raise ValueError('Incomplete physics stage')
            cases+=1;rope_stages+=bool(stage['ropes']);max_ropes=max(max_ropes,stage['ropes'])
            query={'operation':'dynamics_scene','scene':stage['before'],'steps':[stage['dt']]}
            inp.write(json.dumps(query,separators=(',',':'))+'\n')
            expected.write(json.dumps({'tick':stage['tick'],'scene':stage['after'],'callbacks':stage['callbacks']},separators=(',',':'))+'\n')
    if not cases:raise ValueError('No supported physics stages captured')
    with inputs.open('rb') as inp,native.open('wb') as out:
        subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],stdin=inp,stdout=out,check=True)
    failures=[];mismatches=callbacks=outputs=0
    with expected_path.open() as left,native.open() as right:
        for al,bl in zip(left,right,strict=True):
            outputs+=1;s=json.loads(al);actual=json.loads(bl)
            expected={'scene':s['scene'],'callbacks':s['callbacks']}
            value={'scene':actual[0]['scene'],'callbacks':actual[0]['report']['callbacks']} if isinstance(actual,list) else actual
            callbacks+=len(s['callbacks'])
            if value!=expected:
                mismatches+=1
                if len(failures)<10:failures.append({'tick':s['tick'],'difference':first_difference(expected,value)})
    if outputs!=cases:raise ValueError('Incomplete Rust output')
    report={'trace':str(path),'scope':'independent physics stages; Unity state supplied before each stage; no gameplay callback effects',
            'match':mismatches==0,'replay_ticks':ticks,'compared_stages':cases,'matched_stages':cases-mismatches,'rope_stages':rope_stages,'max_active_ropes':max_ropes,
            'callbacks':callbacks,'skipped':skipped,'first_failures':failures,'provenance':source}
    path.with_suffix('.physics-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='provenance'},indent=2));return int(bool(mismatches))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=Path);a=p.parse_args();raise SystemExit(verify(a.trace))
