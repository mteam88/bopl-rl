"""Append one results.tsv row derived from a bench_throughput summary.json.

Every number in the row (medians, per-run ranges, wall seconds) and the
timestamp (summary.json modification time, i.e. when the benchmark finished)
comes from the saved artifact, so the record cannot drift from the data.

    python native/rl/autoresearch/record.py <summary.json> <experiment> <workload> <status> "<description>" [--table base-w16] [--commit HEAD]

Campaign 3 (THROUGHPUT_V4.md, notes-v4.md) scores with bench_shape.sh, whose
rows live in <OUT>/results.jsonl. `--shape <label>` takes the last row of
that label from the results.jsonl passed as <summary.json>: a plain run row
or an A/B summary row (`"ab": true`). Column mapping for those rows:
learner_dps_wall = learner_sps (the score; the candidate arm's median for an
A/B), learner_dps_active = physical sps, wall_seconds = wall_s per update.

    python native/rl/autoresearch/record.py native/artifacts/rl/bench-shape/results.jsonl v4-base shape-v4-w128-b4-p4-u300 baseline "..." --shape v4-base
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
    p.add_argument('--shape',default=None,metavar='LABEL',help='<summary.json> is a bench_shape.sh results.jsonl; take the last row with this label')
    a=p.parse_args()
    if a.shape:
        return shape_row(a)
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

def shape_row(a):
    rows=[json.loads(l) for l in open(a.summary) if l.strip()]
    hits=[r for r in rows if r.get('label')==a.shape and not r.get('failed')]
    if not hits: raise SystemExit(f'no row labelled {a.shape} in {a.summary}')
    r=hits[-1]
    commit=a.commit or subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=HERE,text=True).strip()
    stamp=datetime.datetime.fromtimestamp(os.path.getmtime(a.summary),datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    if r.get('ab'):
        inc,cand,ratio=r['inc'],r['cand'],r.get('cand_over_inc',{})
        fmt=lambda arm,k: f"{arm[k][0]:.4g} [{arm[k][1]:.4g}, {arm[k][2]:.4g}]"
        detail=(f"A/B {r['ab_env']} {r['ab_args']}".strip()+f", {r['repeats']['cand']}x{r['repeats']['inc']} repeats: learner_sps cand {fmt(cand,'learner_sps')} vs inc {fmt(inc,'learner_sps')}"
                f" (x{ratio.get('learner_sps','?')}), sps x{ratio.get('sps','?')}, rollout_s cand {fmt(cand,'rollout_s')} vs {fmt(inc,'rollout_s')}, train_s cand {fmt(cand,'train_s')} vs {fmt(inc,'train_s')},"
                f" wall_s x{ratio.get('wall_s','?')}, every cand sps above every inc: {r.get('cand_sps_all_above_inc')}; load1 {r.get('load1')}")
        score,active,wall=cand['learner_sps'][0],cand['sps'][0],cand['wall_s'][0]
    else:
        ms=r.get('ms_per_step',{})
        detail=(f"{r['encoder']} w{r['workers']} b{r['buffers']} p{r['passes']}, {r['updates']} updates: learner_sps {r['learner_sps']}, sps {r['sps']}, rollout_s {r['rollout_s']}, train_s {r['train_s']}, wall_s {r['wall_s']},"
                f" per buffer-step ms rollout {ms.get('rollout')} eval_gpu {ms.get('eval_gpu')} eval_env {ms.get('eval_env')}"
                +(f", env_pool {r['env_pool']}" if r.get('env_pool') else '')+(f", gpu_step {r['gpu_step']}" if r.get('gpu_step') else '')
                +(f", c_step {r['c_step']}" if r.get('c_step') else '')+(f", set_overflow_max {r['set_overflow_max']}" if r.get('set_overflow_max') is not None else '')
                +f", cuda_mb {r.get('cuda_mb')}, load1 {r.get('load1')}")
        score,active,wall=r['learner_sps'],r['sps'],r['wall_s']
    row=[stamp,commit,a.experiment,a.workload,f'{score:.0f}',a.status,
         a.description.replace('\t',' ').replace('\n',' ')+'; '+detail.replace('\t',' ').replace('\n',' '),f'{active:.0f}',f'{wall:.4f}']
    path=Path(a.results)
    if not path.exists(): path.write_text('\t'.join(COLUMNS)+'\n')
    with open(path,'a') as f: f.write('\t'.join(row)+'\n')
    print('\t'.join(row))
main()
