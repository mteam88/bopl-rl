"""All-combo benchmark: two checkpoints on random kits drawn from every ability.

    venv/bin/python league/all_combos.py A.bin B.bin --pairs 500 --workers 6 --out results.jsonl

The league's rating condition plays each member on the kits it trained on (16 for
the generalist-111 lineage), so it cannot say how good a policy is on the other
~8,000 kits. Here seed pair i deals two random kits (three slots each, drawn with
replacement from POOL, the any-3 pool: every ability but 12 and Revival) and plays
them side-swapped, so A and B each play both kits and a strong kit favours neither.
Maps, ticks and reaction delay are the rating condition's (league/games.py).
Deterministic in --seed-base: two runs with the same seed play the same kits and maps,
so successive checkpoints are measured on the same games.

Prints A's score (wins + draws/2) with a seed-pair bootstrap interval, then A's score
by ability (games where A's kit holds it), weakest first.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'eval'), str(HERE.parent / 'native/tools')]
import registry  # noqa: E402
import games as league_games  # noqa: E402
import elo_ladder  # noqa: E402
import map_catalogue  # noqa: E402
from evaluation_protocol import ability_names  # noqa: E402

POOL = [a for a in range(1, 39) if a not in (12, 15)]


def tasks(pairs, seed_base, maps):
    rng = np.random.default_rng(seed_base)
    order = league_games.map_order(maps, seed_base)
    out = []
    for pair in range(pairs):
        x, y = [int(v) for v in rng.choice(POOL, 3)], [int(v) for v in rng.choice(POOL, 3)]
        for side in (0, 1):
            out.append(dict(a='A', b='B', map=order[pair % len(order)], seed=seed_base + pair, side=side,
                            kits=x + y if side == 0 else y + x, a_kit=x if side == 0 else y))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a', type=Path)
    p.add_argument('b', type=Path)
    p.add_argument('--pairs', type=int, default=500)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed-base', type=int, default=131001)
    p.add_argument('--label', help='name for the output row (default: the two paths)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)

    built = {'A': elo_ladder.Model('A', a.a.resolve(), 'candidate'),
             'B': elo_ladder.Model('B', a.b.resolve(), 'candidate')}
    league_games.configure(a.out.parent / 'all-combos-workers')
    todo = tasks(a.pairs, a.seed_base, map_catalogue.parse_map_ids(registry.MAPS))
    kit_of = {(t['seed'], t['side']): t['a_kit'] for t in todo}
    played = league_games.play(built, todo, a.workers, a.seed_base)

    points = lambda g: 1.0 if g['outcome'] == 'a' else 0.5 if g['outcome'] == 'draw' else 0.0
    per_pair = {}
    by_ability = {}
    for g in played:
        s = points(g)
        per_pair.setdefault(g['seed'], []).append(s)
        for ability in set(kit_of[(g['seed'], g['side'])]):
            by_ability.setdefault(ability, []).append(s)
    pairs = np.array([np.mean(v) for v in per_pair.values()])
    rng = np.random.default_rng(0)
    boot = [rng.choice(pairs, pairs.size).mean() for _ in range(2000)]
    names = ability_names()
    row = dict(label=a.label or f'{a.a} vs {a.b}', a=str(a.a), b=str(a.b), seed_base=a.seed_base,
               games=len(played), score=float(pairs.mean()),
               ci=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
               draws=sum(g['outcome'] == 'draw' for g in played),
               by_ability={names.get(k, str(k)) if isinstance(names, dict) else str(k):
                           [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(by_ability.items())})
    print(f"{row['label']}: A scores {row['score']:.3f} [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}] "
          f"over {row['games']} games ({row['draws']} draws)")
    for name, (score, n) in sorted(row['by_ability'].items(), key=lambda kv: kv[1][0]):
        print(f'  {name:<16} {score:.3f}  ({n} games)')
    with open(a.out, 'a') as f:
        f.write(json.dumps(row) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
