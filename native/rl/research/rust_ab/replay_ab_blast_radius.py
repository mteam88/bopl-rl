#!/usr/bin/env python3
"""Cumulative blast radius: how much of a recorded training corpus a library change moves.

Runs `replay_ab` twice over each kit's `.rep` corpus:

  1. baseline vs baseline  - establishes the true episode length of every game
     in the build that recorded it, and proves the differential is a no-op there.
  2. baseline vs candidate - the measurement.

Both passes use REPLAY_AB_VISIBLE=1 (only state the v4 observation writer can
read) and REPLAY_AB_TOL=1 (one raw unit; 0 trips on the q32 build's static
tick-0 ulp, and anything larger hides real bugs).

Per kit it reports how many games' policy-readable observation streams differ
and the distribution of the first diverging tick.

    replay_ab_blast_radius.py --binary <replay_ab> --baseline <base.so> \
        --candidate <main.so> --out rows.tsv <kit-evaluations-dir>...
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

OK = re.compile(r'^OK\s+(\S+)\s+map (\d+) seed (\d+) loadout (\S+) ticks (\d+)')
DIFF = re.compile(r'^DIFF\s+(\S+)\s+map (\d+) seed (\d+) loadout (\S+) ticks (\d+)')
KIND = re.compile(r'^\s+(numeric|discrete)\s+tick\s+(\d+)\s+(\S+)')


def run(binary, left, right, directory, extra=None):
    """Return {game path: dict} for one replay_ab pass."""
    environment = dict(os.environ, REPLAY_AB_VISIBLE='1', REPLAY_AB_TOL=str(TOLERANCE))
    environment.update(extra or {})
    proc = subprocess.run([str(binary), str(left), str(right), str(directory)],
                          env=environment, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.exit(f'replay_ab exited {proc.returncode} on {directory}\n{proc.stdout}{proc.stderr}')
    games, current = {}, None
    for line in proc.stdout.splitlines():
        if line.startswith('=== replay_ab summary'):
            break
        hit = OK.match(line) or DIFF.match(line)
        if hit:
            current = dict(path=hit.group(1), map=int(hit.group(2)), seed=int(hit.group(3)),
                           loadout=hit.group(4), ticks=int(hit.group(5)),
                           diverged=line.startswith('DIFF'), numeric=None, discrete=None,
                           field=None)
            games[current['path']] = current
            continue
        kind = KIND.match(line)
        if kind and current is not None:
            current[kind.group(1)] = int(kind.group(2))
            if current['field'] is None or int(kind.group(2)) < current.get('_at', 1 << 30):
                current['field'], current['_at'] = kind.group(3), int(kind.group(2))
    return games


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('kits', nargs='+', type=Path, help='<kit>/000/evaluations directories')
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--baseline', type=Path, required=True, help='library the policies trained in')
parser.add_argument('--candidate', type=Path, required=True, help='library under test')
parser.add_argument('--out', type=Path, help='per-game TSV')
parser.add_argument('--tolerance', type=int, default=1,
                    help='REPLAY_AB_TOL in raw units (1 raw = 2.33e-10 world units). '
                         'Never 0: the q32 build leaves one ulp on statically authored extents.')
args = parser.parse_args()
TOLERANCE = args.tolerance

rows = []
print(f'{"kit":<34} {"games":>6} {"differ":>7} {"pct":>6}  {"<100":>5} {"<1000":>6} '
      f'{">=1000":>6}  {"p25":>6} {"med":>6} {"p75":>6}')
for kit in args.kits:
    name = kit.parts[-3] if len(kit.parts) >= 3 else str(kit)
    lengths = run(args.binary, args.baseline, args.baseline, kit)
    clean = sum(1 for g in lengths.values() if g['diverged'])
    if clean:
        sys.exit(f'{name}: baseline is not self-consistent ({clean} games differ) - wrong baseline')
    measured = run(args.binary, args.baseline, args.candidate, kit)
    if set(measured) != set(lengths):
        sys.exit(f'{name}: the two passes read different game sets')
    first = []
    for path, game in sorted(measured.items()):
        ticks = [t for t in (game['numeric'], game['discrete']) if t is not None]
        at = min(ticks) if ticks else None
        rows.append((name, path, lengths[path]['map'], lengths[path]['seed'], lengths[path]['loadout'],
                     lengths[path]['ticks'], 1 if game['diverged'] else 0,
                     '' if at is None else at, game['field'] or ''))
        if game['diverged']:
            if at is None:
                sys.exit(f'{name}: {path} reported DIFF with no divergence tick')
            first.append(at)
    first.sort()
    total = len(measured)
    differ = len(first)
    early = sum(1 for t in first if t < 100)
    mid = sum(1 for t in first if 100 <= t < 1000)
    late = differ - early - mid
    p25, med, p75 = (quantile(first, q) for q in (0.25, 0.5, 0.75))
    fmt = lambda v: '-' if v is None else f'{v:.0f}'
    print(f'{name:<34} {total:>6} {differ:>7} {100*differ/total:>5.1f}%  {early:>5} {mid:>6} {late:>6}  '
          f'{fmt(p25):>6} {fmt(med):>6} {fmt(p75):>6}')

if args.out:
    with open(args.out, 'w') as handle:
        handle.write('kit\tgame\tmap\tseed\tloadout\tepisode_ticks\tdiverged\tfirst_tick\tfield\n')
        for row in rows:
            handle.write('\t'.join(str(value) for value in row) + '\n')
    print(f'\n{len(rows)} games written to {args.out}')
