"""Paired score difference between two evaluations played on the same seeds.

    venv/bin/python eval/paired.py X Y [--json]

X and Y are probe directories (eval/probe.py, read from their games.jsonl) or the
--games-out files of league/all_combos.py run with the same --seed-base and --pairs.
Games pair on (map, seed, side); each scores 1 for an A win, 0.5 for a draw, 0 otherwise.
The two side-swapped games of a seed share kits and map, so the interval is a 95% normal
one over per-seed mean differences (seed clusters). Also printed: the pairing ratio
r = var(paired diff) / var(unpaired diff) at the same seeds; an unpaired comparison needs
1/r times the games for the same interval.
"""
import argparse
import json
import math
from pathlib import Path


def scores(path):
    path = Path(path)
    rows = (json.loads(line) for line in (path / 'games.jsonl' if path.is_dir() else path).open())
    return {(r['map'], r['seed'], r['side']): {'a': 1.0, 'draw': 0.5}.get(r['outcome'], 0.0) for r in rows}


def variance(values):
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def paired(x, y):
    x, y = scores(x), scores(y)
    keys = sorted(x.keys() & y.keys())
    if not keys:
        raise SystemExit('no common games: were both run with the same --seed-base and --pairs?')
    clusters = {}
    for k in keys:
        clusters.setdefault(k[:2], []).append(k)
    mx = [sum(x[k] for k in ks) / len(ks) for ks in clusters.values()]
    my = [sum(y[k] for k in ks) / len(ks) for ks in clusters.values()]
    diffs = [a - b for a, b in zip(mx, my)]
    n, m = len(keys), len(diffs)
    mean = sum(x[k] - y[k] for k in keys) / n
    se = math.sqrt(variance(diffs) / m) if m > 1 else float('inf')
    unpaired = variance(mx) + variance(my) if m > 1 else 0.0
    return dict(games=n, seeds=m, only_x=len(x) - n, only_y=len(y) - n,
                x=sum(x[k] for k in keys) / n, y=sum(y[k] for k in keys) / n,
                diff=mean, se=se, ci=[mean - 1.96 * se, mean + 1.96 * se],
                pairing_ratio=variance(diffs) / unpaired if unpaired > 0 else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('x', type=Path)
    p.add_argument('y', type=Path)
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    r = paired(a.x, a.y)
    if a.json:
        print(json.dumps(r))
    else:
        ratio = 'n/a' if r['pairing_ratio'] is None else f"{r['pairing_ratio']:.2f}"
        print(f"{a.x.name} {r['x']:.4f} vs {a.y.name} {r['y']:.4f} over {r['games']} paired games ({r['seeds']} seeds): "
              f"diff {r['diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}], pairing ratio {ratio}")
        if r['only_x'] or r['only_y']:
            print(f"unpaired games dropped: {r['only_x']} only in {a.x.name}, {r['only_y']} only in {a.y.name}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
