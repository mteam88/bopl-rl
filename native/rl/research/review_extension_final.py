"""Reproduce the same selected ranged-opponent cases at the final checkpoint."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
from pathlib import Path
import torch
from continuation import digest
from evaluate_extension import checkpoint
from trace_league_failures import trace
from train import ROOT, atomic_json


def work(item):
    case,directory=item
    torch.set_num_threads(1)
    return trace(case,directory)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    milestone=json.loads((out/'fallback-checkpoint.json').read_text())['milestone'];review=f'failure-review-{milestone//1000000}m'
    plan=json.loads((out/(review+'-protocol.json')).read_text())
    assert digest(out/'failure-review-protocol.json')==plan['source_selection_sha256']
    protocol=json.loads((out/'protocol.json').read_text());runs={r['name']:r for r in protocol['runs']}
    directory=out/review;directory.mkdir(exist_ok=True);rows=[];cases=[]
    for selected in plan['cases']:
        original=out/'evaluations'/(selected['run']+f'-{milestone}-transfer')/(selected['case']+'.json')
        ckpt=checkpoint(out,protocol,runs[selected['run']],milestone)
        case=dict(**selected,original=str(original.relative_to(ROOT)),sha256=digest(original),checkpoint=str(ckpt.relative_to(ROOT)),selection=f'Same case selected for a B loss at 1M; evaluated at {milestone//1000000}M regardless of outcome.')
        cases.append((case,directory))
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as executor:
        for row in executor.map(work,cases):
            rows.append(row)
            atomic_json(directory/'result.json',dict(schema=1,protocol_sha256=digest(out/(review+'-protocol.json')),cases=rows))
            print(row['run'],row['case'],row['outcome'],flush=True)
    print('MATCHED_COMMON_REVIEW_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
