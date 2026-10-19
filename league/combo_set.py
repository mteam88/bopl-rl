"""all_combos.py on a fixed set of loadouts: two checkpoints, every ordered pair of the set's kits.

    venv/bin/python league/combo_set.py A.bin B.bin|random --set native/data/lineups-matthew.txt --repeat 2 --out results.jsonl

Seed pair i deals kits (x, y) from the set, cycling through all ordered pairs (mirrors included)
--repeat times, and plays them side-swapped like all_combos.py, so a strong kit favours neither
side. Maps, ticks and reaction delay are the rating condition's.

Per kit it reports two readings: `with`, A's score in the games where A holds the kit (skill and
kit strength together), and `pairs`, A's mean over every seed pair that deals the kit to either
side (kit-neutral: both players held it once, so it reads skill on the matchups the kit is in).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'eval'), str(HERE.parent / 'native/tools')]
import all_combos  # noqa: E402
from all_combos import elo_ladder, league_games, map_catalogue, registry  # noqa: E402
from evaluation_protocol import ability_names  # noqa: E402


def read_set(path):
    kits = []
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            kits.append(tuple(int(v) for v in line.split(',')))
    return kits


def tasks(kits, repeat, seed_base, maps):
    order = league_games.map_order(maps, seed_base)
    combos = [(x, y) for x in kits for y in kits] * repeat
    out = []
    for pair, (x, y) in enumerate(combos):
        for side in (0, 1):
            out.append(dict(a='A', b='B', map=order[pair % len(order)], seed=seed_base + pair, side=side,
                            kits=list(x + y) if side == 0 else list(y + x), a_kit=list(x),
                            pair_kits=(x, y)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a', type=Path)
    p.add_argument('b', help='checkpoint path, or "random" for the uniform random anchor')
    p.add_argument('--set', type=Path, required=True, help='one comma-separated three-ability kit per line, # comments')
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed-base', type=int, default=151001)
    p.add_argument('--label')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)

    built = {'A': elo_ladder.Model('A', a.a.resolve(), 'candidate'), 
             'B': elo_ladder.Random() if a.b == 'random' else elo_ladder.Model('B', Path(a.b).resolve(), 'candidate')}
    league_games.configure(a.out.parent / 'all-combos-workers')
    kits = read_set(a.set)
    todo = tasks(kits, a.repeat, a.seed_base, map_catalogue.parse_map_ids(registry.MAPS))
    meta = {(t['seed'], t['side']): t for t in todo}
    played = league_games.play(built, [{k: v for k, v in t.items() if k != 'pair_kits'} for t in todo],
                               a.workers, a.seed_base)

    points = lambda g: 1.0 if g['outcome'] == 'a' else 0.5 if g['outcome'] == 'draw' else 0.0
    per_pair, pair_kits, held = {}, {}, {k: [] for k in kits}
    for g in played:
        t = meta[(g['seed'], g['side'])]
        per_pair.setdefault(g['seed'], []).append(points(g))
        pair_kits[g['seed']] = t['pair_kits']
        held[tuple(t['a_kit'])].append(points(g))
    pairs = {s: float(np.mean(v)) for s, v in per_pair.items()}
    rng = np.random.default_rng(0)
    values = np.array(list(pairs.values()))
    boot = [rng.choice(values, values.size).mean() for _ in range(2000)]
    names = ability_names()
    name = lambda k: '/'.join(names.get(i, str(i)) for i in k)
    rows = []
    for k in kits:
        involved = [pairs[s] for s, (x, y) in pair_kits.items() if k in (x, y)]
        rows.append(dict(kit=list(k), name=name(k), with_=[round(float(np.mean(held[k])), 3), len(held[k])],
                         pairs=[round(float(np.mean(involved)), 3), len(involved)]))
    row = dict(label=a.label or f'{a.a} vs {a.b}', a=str(a.a), b=str(a.b), set=str(a.set), seed_base=a.seed_base,
               games=len(played), score=float(values.mean()),
               ci=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
               draws=sum(g['outcome'] == 'draw' for g in played), kits=rows)
    print(f"{row['label']}: A scores {row['score']:.3f} [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}] "
          f"over {row['games']} games ({row['draws']} draws)")
    for r in sorted(rows, key=lambda r: r['pairs'][0]):
        print(f"  {r['name']:<36} pairs {r['pairs'][0]:.3f} ({r['pairs'][1]})   with {r['with_'][0]:.3f} ({r['with_'][1]})")
    with open(a.out, 'a') as f:
        f.write(json.dumps(row) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
