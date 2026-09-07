"""Preserve the operator-interrupted attempt and adopt its identical-settings repeat."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from continuation import digest
from train import atomic_json


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();retry=out/'retry-A-s173'
    while True:
        if (retry/'result.json').exists() and read(out/'training-status.json')['status']=='complete_with_failures' and read(out/'evaluation-status.json')['status'].startswith('complete'):break
        state=read(retry/'status.json')
        if state['status']=='failed':raise RuntimeError('Replacement failed; inspect before proceeding')
        time.sleep(10)
    original=out/'runs/A-s173';archive=out/'incidents/orchestration-A-s173';archive.mkdir(parents=True,exist_ok=False)
    comparisons={}
    for file in ('policy-002668544.pt','policy-004001792.pt'):
        a=torch.load(original/'checkpoints'/file,map_location='cpu',weights_only=False)
        b=torch.load(retry/'checkpoints'/file,map_location='cpu',weights_only=False)
        checks={}
        for key in ('policy','optimizer','torch_rng','cuda_rng','step','learner_steps','updates'):
            try:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0);checks[key]=True
            except AssertionError:checks[key]=False
        comparisons[file]=dict(checks=checks,original_sha256=digest(original/'checkpoints'/file),replacement_sha256=digest(retry/'checkpoints'/file))
    atomic_json(archive/'checkpoint-comparison.json',comparisons)
    # Keep every report with the checkpoint identity that produced it.
    for sub in ('evaluations','schedules'):
        (archive/sub).mkdir()
        for file in (out/sub).iterdir():
            name=file.name
            dependent=(name.startswith('A-s173-') or (name.startswith('crossplay-') and '-A173-' in name and not name.startswith('crossplay-1000000-')))
            if dependent:file.rename(archive/sub/name)
    for name in ('training-status.json','evaluation-status.json','fallback-checkpoint.json'):
        atomic_json(archive/name,read(out/name))
    original.rename(archive/'run');retry.rename(original)
    training=read(out/'training-status.json');training['failed']=[r for r in training['failed'] if r!='A-s173'];training['completed'].append('A-s173');training['retry']='incidents/orchestration-A-s173/checkpoint-comparison.json'
    atomic_json(out/'training-status.json',training)
    incidents=read(out/'training-incidents.json');incidents['incidents'][1].update(retry_status='complete',original_attempt='incidents/orchestration-A-s173/run',comparison='incidents/orchestration-A-s173/checkpoint-comparison.json')
    atomic_json(out/'training-incidents.json',incidents)
    print('REPLACEMENT_ADOPTED',json.dumps(comparisons),flush=True)
    # A fresh queue reuses only reports whose original checkpoint identities match.
    subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_extension_available.py')),'--output',str(out),'--jobs','4'],check=True,start_new_session=True)
    atomic_json(out/'retry-evaluation-complete.json',dict(status='complete'))


if __name__=='__main__':main()
