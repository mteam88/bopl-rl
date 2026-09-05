"""Compare independent Rust movement rollouts with a Unity trace, field by field."""
import argparse
import json
from pathlib import Path
import subprocess
from project_state import project,states
from reference import first_difference
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace');p.add_argument('--fixture',default=str(ROOT/'native/fixtures/movement.json'));p.add_argument('--stop',type=int,default=480);args=p.parse_args()
    frames=json.loads(Path(args.fixture).read_text())['frames'][240:args.stop]
    result=subprocess.run([str(ROOT/'native/target/release/sim-probe')],input=''.join(json.dumps(f)+'\n' for f in frames),text=True,capture_output=True,check=True)
    rust=[json.loads(l) for l in result.stdout.splitlines()]
    unity=[project(s) for s in states(args.trace) if 240<s['tick']<=args.stop]
    terminal_tick=next((s['tick'] for s in unity if len(s['players'])<2 or any(not p['alive'] for p in s['players'])),None)
    expected_count=args.stop-240 if terminal_tick is None else terminal_tick-240
    if len(rust)!=expected_count or len(unity)!=len(frames):raise ValueError(f'incomplete rollout: Rust {len(rust)}, expected {expected_count}, Unity {len(unity)}')
    unity=unity[:expected_count]
    for i,(a,b) in enumerate(zip(unity,rust)):
        for player in b['players']:player['previous_jump']=False # Input history belongs to the Rust driver; not in this projection.
        if a['tick']==terminal_tick:
            alive_ids={p['id'] for p in a['players'] if p['alive']}
            if alive_ids!={p['id'] for p in b['players'] if p['alive']}:raise ValueError('terminal alive mask differs')
            a['players']=[p for p in a['players'] if p['alive']]
            b['players']=[p for p in b['players'] if p['alive']]
        for x,y in zip(a['players'],b['players']):
            x['dash'].pop('saved_local');y['dash'].pop('saved_local')
        diff=first_difference(a,b)
        if diff:
            print(json.dumps(dict(match=False,tick=i+241,difference=diff),indent=2));return 1
    print(json.dumps(dict(match=True,ticks=len(rust),players=len(rust[0]['players']),terminal_tick=terminal_tick)));return 0
if __name__=='__main__':raise SystemExit(main())
