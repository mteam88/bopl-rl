"""Append one results.tsv row derived from a bench_throughput summary.json.

Every number in the row (medians, per-run ranges, wall seconds) and the
timestamp (summary.json modification time, i.e. when the benchmark finished)
comes from the saved artifact, so the record cannot drift from the data.

    python native/rl/autoresearch/record.py <summary.json> <experiment> <workload> <status> "<description>" [--table base-w16] [--commit HEAD]
"""
import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
COLUMNS=['timestamp_utc','commit','experiment','workload','learner_dps_wall','status','description','learner_dps_active','wall_seconds']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('summary'); p.add_argument('experiment'); p.add_argument('workload')
    p.add_argument('status',choices=['baseline','keep','discard','invalid']); p.add_argument('description')
    p.add_argument('--table',default=None,help='table key (default: the only one)')
    p.add_argument('--commit',default=None,help='commit measured (default: git HEAD)')
    p.add_argument('--results',default=str(HERE/'results.tsv'))
    a=p.parse_args()
    summary=json.load(open(a.summary))
    key=a.table or (list(summary['table'])[0] if len(summary['table'])==1 else None)
    if key is None: raise SystemExit('Several tables; pass --table: '+', '.join(summary['table']))
    t=summary['table'][key]; runs=[r for r in summary['runs'] if r['name'] in t['runs']]
    if len(runs)!=len(t['runs']): raise SystemExit('Runs missing from summary')
    wall=[r['learner_dps_wall'] for r in runs]; active=[r['learner_dps_active'] for r in runs]
    startup=[r.get('startup_seconds',0.0) for r in runs]; seconds=[r['wall_seconds'] for r in runs]
    commit=a.commit or subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=HERE,text=True).strip()
    stamp=datetime.datetime.fromtimestamp(os.path.getmtime(a.summary),datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    detail=(f'{len(runs)} runs: wall {min(wall):.2f}-{max(wall):.2f}, active {min(active):.2f}-{max(active):.2f} learner dps; '
            f'startup {min(startup):.1f}-{max(startup):.1f} s of {min(seconds):.1f}-{max(seconds):.1f} s wall')
    row=[stamp,commit,a.experiment,a.workload,f"{t['median_learner_dps_wall']:.0f}",a.status,
         a.description.replace('\t',' ').replace('\n',' ')+'; '+detail,
         f"{t['median_learner_dps_active']:.0f}",f"{t['median_wall_seconds']:.1f}"]
    path=Path(a.results)
    if not path.exists(): path.write_text('\t'.join(COLUMNS)+'\n')
    with open(path,'a') as f: f.write('\t'.join(row)+'\n')
    print('\t'.join(row))
main()
