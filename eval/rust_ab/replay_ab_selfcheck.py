#!/usr/bin/env python3
"""Prove replay_ab reconstructs a saved game faithfully in the backend that produced it.

`replay_ab` re-steps a `<game>.rep` recording with the recorded per-tick inputs.
That reconstruction is only trustworthy if, in the *same* backend that produced
the recording, it reproduces the state the evaluation actually saved in
`<game>.json`. This script checks exactly that: it runs `replay_ab` with the
Rust library on both sides (so the differential is a no-op) while dumping the
candidate's per-tick player positions, then compares every decision boundary
with the recorded frames.

    native/artifacts/rl/venv/bin/python \\
        native/rl/research/rust_ab/replay_ab_selfcheck.py \\
        --binary native/artifacts/rl/replay_ab \\
        --library native/target/release/libbopl_sim.so \\
        native/artifacts/rl/study-grenade/evaluations/*/map0-s0-ab.rep

Exit status 0 when every game round-trips, 1 otherwise. A failure means the
harness is wrong (input reconstruction, warmup, config), not the simulation.
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('games', nargs='+', type=Path, help='<game>.rep archives (siblings of <game>.json[.gz])')
parser.add_argument('--binary', type=Path, required=True, help='built replay_ab')
parser.add_argument('--library', type=Path, required=True, help='backend that produced the recordings')
parser.add_argument('--tolerance', type=float, default=1e-4, help='world units; the JSON stores 5 decimals')
parser.add_argument('--warmup', type=int, default=240)
parser.add_argument('--limit', type=int, default=7200)
args = parser.parse_args()


def load_json(rep: Path):
    plain = rep.with_suffix('.json')
    if plain.exists():
        return json.loads(plain.read_text())
    packed = Path(str(plain) + '.gz')
    if packed.exists():
        with gzip.open(packed, 'rt') as handle:
            return json.load(handle)
    return None


failures = 0
checked = 0
for rep in args.games:
    saved = load_json(rep)
    if saved is None:
        print(f'skip   {rep}: no recorded JSON beside it')
        continue
    with tempfile.NamedTemporaryFile('r', suffix='.tsv', delete=False) as handle:
        trace_path = handle.name
    environment = dict(os.environ, REPLAY_AB_TRACE=trace_path, REPLAY_AB_TOL='0',
                       REPLAY_AB_QUIET='1', REPLAY_AB_WARMUP=str(args.warmup),
                       REPLAY_AB_LIMIT=str(args.limit))
    run = subprocess.run([str(args.binary.resolve()), str(args.library.resolve()),
                          str(args.library.resolve()), str(rep.resolve())],
                         env=environment, capture_output=True, text=True)
    if run.returncode not in (0, 1):
        print(f'FAIL   {rep}: replay_ab exited {run.returncode}\n{run.stdout}{run.stderr}')
        failures += 1
        os.unlink(trace_path)
        continue
    trace = {}
    with open(trace_path) as handle:
        for line in handle:
            tick, player, x, y = line.split()
            trace[(int(tick), int(player))] = (float(x), float(y))
    os.unlink(trace_path)

    frames = saved['frames']
    worst = 0.0
    worst_where = ''
    for frame in frames:
        tick = frame['t']
        for index, player in enumerate(frame['players']):
            key = (tick, index)
            if key not in trace:
                print(f'FAIL   {rep}: replay produced no tick {tick} (recording has {len(frames)} frames, '
                      f'last replayed tick {max(t for t, _ in trace)})')
                failures += 1
                worst = float('inf')
                break
            x, y = trace[key]
            gap = max(abs(x - player['x']), abs(y - player['y']))
            if gap > worst:
                worst, worst_where = gap, f'tick {tick} player {index}: replay ({x},{y}) recorded ({player["x"]},{player["y"]})'
        if worst == float('inf'):
            break
    if worst == float('inf'):
        continue
    checked += 1
    recorded_ticks = saved['result']['ticks']
    replayed_ticks = max(t for t, _ in trace)
    status = 'OK  '
    if worst > args.tolerance or replayed_ticks != recorded_ticks:
        status = 'FAIL'
        failures += 1
    print(f'{status}   {rep.name}: {len(frames)} frames, recorded {recorded_ticks} ticks, '
          f'replayed {replayed_ticks}, worst player gap {worst:.3g}'
          + (f'  {worst_where}' if status == 'FAIL' else ''))

print(f'\n{checked} games round-tripped, {failures} failures')
sys.exit(1 if failures else 0)
