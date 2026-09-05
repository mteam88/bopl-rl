"""Verify a completed physics stage immediately before a Unity gameplay exception.

The failed tick is not a successful rollout. Only its pre-callback physics stage
is compared, with Unity supplying the input state.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from reference import ROOT, first_difference


def verify(path, fixture=None):
    source=json.loads(path.with_suffix('.provenance.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source['failure_sha256']:
        raise ValueError('Failure snapshot hash differs from provenance')
    fault=json.loads(path.read_text())
    if fault['schema']!='bopl-replay-failure-v1' or not fault['physics_stage_complete']:
        raise ValueError('No completed physics stage before the source exception')
    s=fault['physicsStage']
    if 'beams_before' not in s:raise ValueError('Missing persistent beam state')
    query={'operation':'dynamics_beam_capture','scene':s['before'],'beams':s['beams_before'],'dt':s['dt']}
    expected={'scene':s['after'],'beams':s['beams_after'],'callbacks':s['callbacks']}
    process=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=json.dumps(query)+'\n',text=True,capture_output=True,check=True)
    actual=json.loads(process.stdout)
    difference=first_difference(expected,actual)
    report={'scope':__doc__.strip(),'match':difference is None,'input_index':fault['input_index'],
        'source_exception':fault['exception'],'source_message':fault['message'],
        'queued_beams':s['beams'],'beam_owners':sorted({b['owner_id'] for b in s['beams_before']['pending']}),
        'callbacks':len(s['callbacks']),'difference':difference,'provenance':source}
    path.with_suffix('.physics-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='provenance'},indent=2))
    if difference:return 1
    if fixture:
        header={'schema':'bopl-gameplay-physics-stage-v1','stages':1,'sources':[source],
            'scope':'Final completed physics stage before a Unity exception; supplied pre-state; excludes gameplay callback effects'}
        record={'source':0,'tick':s['tick'],'query':query,'expected':expected,
            'subsequent_source_exception':fault['exception'],'input_index':fault['input_index']}
        raw=''.join(json.dumps(v,separators=(',',':'))+'\n' for v in (header,record))
        fixture.write_bytes(gzip.compress(raw.encode(),mtime=0))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('failure',type=Path)
    parser.add_argument('--fixture',type=Path)
    args=parser.parse_args()
    raise SystemExit(verify(args.failure,args.fixture))
