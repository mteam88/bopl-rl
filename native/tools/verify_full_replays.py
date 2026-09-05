"""Verify all-player Unity playback consistency; this does not certify the Rust port."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from reference import ROOT, provenance, read_trace, first_difference


def run_case(directory, case):
    name = case['name']
    evidence = directory / 'traces'
    evidence.mkdir(exist_ok=True)
    traces = []
    for batch in (1, 16):
        path = evidence / f'{name}-batch{batch}.jsonl.gz'
        if path.exists():
            provenance(path)
        else:
            command = [sys.executable, str(ROOT/'native/tools/reference.py'), 'capture', str(directory/(name+'.rep')), str(path), '--ticks', str(case['ticks']), '--batch', str(batch), '--timeout', '600']
            with (evidence/f'{name}-batch{batch}.capture.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        traces.append(path)
    result = subprocess.run([sys.executable, str(ROOT/'native/tools/reference.py'), 'compare', *map(str,traces), '--gameplay'], capture_output=True, text=True)
    report = {'name': name, 'players': case['players'], 'comparison': json.loads(result.stdout) if result.stdout else result.stderr, 'match': result.returncode == 0}
    # Verify the trace actually contains every requested input slot and runtime player.
    runtime_ids = set()
    nonneutral = set()
    for row in read_trace(traces[0]):
        if row['kind'] == 'input':
            if len(row['value']) != case['players']: raise ValueError('Input player count mismatch')
            for index, packet in enumerate(row['value']):
                if packet['InputPacket.joystickAngle'] or any(packet['InputPacket.'+b] for b in ('jump', 'ab1', 'ab2', 'ab3')): nonneutral.add(index+1)
        elif row['kind'] == 'end':
            report['ability_entries'] = row.get('abilityEntries', {})
        elif row['kind'] == 'state':
            for player in row['value']['players']:
                runtime_ids.add(player['Player.<Id>k__BackingField'])
    if runtime_ids != set(range(1,case['players']+1)) or nonneutral != runtime_ids: raise ValueError(f'Player execution missing: {runtime_ids} / {nonneutral}')
    report['runtime_player_ids'] = sorted(runtime_ids)
    report['nonneutral_input_slots'] = sorted(nonneutral)
    (evidence/f'{name}-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--filter', default='')
    parser.add_argument('--workers', type=int, default=2)
    args=parser.parse_args()
    cases=[c for c in json.loads((args.directory/'cases.json').read_text()) if args.filter in c['name']]
    if not cases: raise ValueError('No cases selected')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,args.directory,c) for c in cases]
        results=[]
        for case,future in zip(cases,futures):
            try: results.append(future.result())
            except Exception as error: results.append({'name':case['name'],'match':False,'error':str(error)})
    report={'scope':'Unity versus Unity only', 'cases':results,'passed':sum(r['match'] for r in results),'total':len(results)}
    (args.directory/'verification-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return int(report['passed'] != report['total'])

if __name__ == '__main__': raise SystemExit(main())
