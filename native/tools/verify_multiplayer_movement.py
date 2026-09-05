"""Verify independent movement rollouts from a single Unity reset, for 1..6 players."""
import argparse
import json
from pathlib import Path
import subprocess
from reference import ROOT,read_trace,first_difference,provenance
from project_state import project
from golden import comparable


def evaluate(trace):
    provenance(trace)
    initial=None;records=[];input_value=None;finished=False;players=None
    for row in read_trace(trace):
        if row['kind']=='header':
            setup=row['setup'];players=len(setup['players'])
            if any(p['abilities']!=[2] or p['keyboard'] for p in setup['players']):raise ValueError('Movement slice needs one dash slot and stick controls')
            if setup['stocks']!=0 or setup['level']!=0:raise ValueError('Movement slice needs Level1 with stocks off')
        elif row['kind']=='input':input_value=row['value']
        elif row['kind']=='state':
            tick=row['value']['tick']
            if tick==240:initial=project(row['value'])
            elif tick>240 and not finished:
                expected=project(row['value']);actions=[]
                for p in input_value:
                    if any(p['InputPacket.'+b] for b in ('ab2','ab3','start','select','w','a','s','d')):raise ValueError('Unsupported input')
                    actions.append([p['InputPacket.joystickAngle'],int(p['InputPacket.jump'])+2*int(p['InputPacket.ab1'])])
                records.append({'actions':actions,'expected':expected})
                finished=len(expected['players'])!=players or any(not p['alive'] for p in expected['players'])
    if initial is None or not records:raise ValueError('Missing reset or trajectory')
    output=ROOT/'native/artifacts/multiplayer-projections';output.mkdir(exist_ok=True)
    reset=output/f'{players}-reset.json';reset.write_text(json.dumps(initial,indent=2)+'\n')
    sequence=output/f'{players}-episode.json';sequence.write_text(json.dumps(records)+'\n')
    result=subprocess.run([str(ROOT/'native/target/release/sim-probe'),str(reset)],input=''.join(json.dumps(r['actions'])+'\n' for r in records),text=True,capture_output=True,check=True)
    actual=[json.loads(line) for line in result.stdout.splitlines()]
    if len(actual)!=len(records):raise ValueError(f'Incomplete Rust rollout {len(actual)}/{len(records)}')
    for record,state in zip(records,actual):
        diff=first_difference(comparable(record['expected']),comparable(state))
        if diff:return {'players':players,'match':False,'matched_ticks':state['tick']-241,'tick':state['tick'],'difference':diff}
    return {'players':players,'match':True,'ticks':len(records),'terminal':finished}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('traces',nargs='+',type=Path);args=parser.parse_args()
    result=[evaluate(t) for t in args.traces];print(json.dumps(result,indent=2))
    return int(any(not r['match'] for r in result))

if __name__=='__main__':raise SystemExit(main())
