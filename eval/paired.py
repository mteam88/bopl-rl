"""Paired score difference between two probes (eval/probe.py) played on the same seeds.

    venv/bin/python eval/paired.py DIR_X DIR_Y [--json]

Games pair on (map, seed, side); each scores 1 for an A win, 0.5 for a draw, 0 otherwise.
Prints X's score, Y's score, and X - Y with a 95% normal interval over the paired differences.
"""
import argparse
import json
import math
from pathlib import Path


def scores(directory):
    rows = (json.loads(line) for line in (Path(directory) / 'games.jsonl').open())
    return {(r['map'], r['seed'], r['side']): {'a': 1.0, 'draw': 0.5}.get(r['outcome'], 0.0) for r in rows}


def paired(x, y):
    x, y = scores(x), scores(y)
    keys = sorted(x.keys() & y.keys())
    if not keys:
        raise SystemExit('no common games: were both probes run with the same --seed-base and --pairs?')
    diffs = [x[k] - y[k] for k in keys]
    n = len(diffs)
    mean = sum(diffs) / n
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1) / n) if n > 1 else float('inf')
    return dict(games=n, x=sum(x[k] for k in keys) / n, y=sum(y[k] for k in keys) / n,
                diff=mean, ci=[mean - 1.96 * se, mean + 1.96 * se])


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
        print(f"{a.x.name} {r['x']:.4f} vs {a.y.name} {r['y']:.4f} over {r['games']} paired games: "
              f"diff {r['diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
