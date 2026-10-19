"""All-combo benchmark: two checkpoints on random kits drawn from every ability.

    venv/bin/python league/all_combos.py A.bin {B.bin|random} --pairs 500 --workers 6 --out results.jsonl

The league's rating condition plays each member on the kits it trained on (16 for
the generalist-111 lineage), so it cannot say how good a policy is on the other
~8,000 kits. Here seed pair i deals two random kits (three slots each, drawn with
replacement from POOL, the any-3 pool: every ability but 12 and Revival) and plays
them side-swapped: A keeps the first kit in both seats, B the second. B may also be a
league member by name; one that declares kits (registry.kits) plays them in turn, and a
scripted bot skips space maps (registry.plays_map). Maps default to the rating
condition's (league/games.py); --maps narrows them, ticks and reaction delay are fixed.
Deterministic in --seed-base: two runs with the same seed play the same kits and maps,
so successive checkpoints are measured on the same games.

Prints A's score (wins + draws/2) with a seed-pair bootstrap interval, then A's score
by ability (games where A's kit holds it), weakest first. --games-out F also writes one
row per game, so two runs on the same --seed-base and --pairs difference pair by pair
(eval/paired.py F_X F_Y).
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


def tasks(pairs, seed_base, maps, b_kits=None):
    rng = np.random.default_rng(seed_base)
    order = league_games.map_order(maps, seed_base)
    out = []
    for pair in range(pairs):
        x, y = [int(v) for v in rng.choice(POOL, 3)], [int(v) for v in rng.choice(POOL, 3)]
        if b_kits:
            y = [int(v) for v in b_kits[pair % len(b_kits)]]
        # kits are [seat 0 | seat 1] and A sits in seat `side`, so A plays x in both games.
        for side in (0, 1):
            out.append(dict(a='A', b='B', map=order[pair % len(order)], seed=seed_base + pair, side=side,
                            kits=x + y if side == 0 else y + x, a_kit=x))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a', help='checkpoint path, member:<league name>, or ensemble:<spec>+<spec>+... '
                   '(elo_ladder.participant; an ensemble pools its members\' logits)')
    p.add_argument('b', help='as A, "random" for the uniform random anchor, or a league member name '
                   '(plays its own kits and maps)')
    p.add_argument('--pairs', type=int, default=500)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--threads', action='store_true', help='play in worker threads instead of forked processes '
                   '(same games; threads share the GIL)')
    p.add_argument('--seed-base', type=int, default=131001)
    p.add_argument('--label', help='name for the output row (default: the two paths)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--maps', default=registry.MAPS, help='map IDs, ranges or named sets (map_catalogue.parse_map_ids)')
    p.add_argument('--games-out', type=Path, help='write one JSON row per game here (overwritten)')
    p.add_argument('--search', help='play A through the decision-time search (eval/search_policy.py) with this '
                   'budget: k,n,depth[,h][,mix=M][,vs=V] (search_policy.parse_budget)')
    p.add_argument('--temperature-a', help='sample A\'s heads at this softmax temperature: one value or aim,buttons,hold '
                   '(raw models only; < 1 sharpens)')
    a = p.parse_args(argv)

    if a.search:
        import search_policy
        built = {'A': search_policy.Search('A', Path(a.a).resolve(), 'candidate', **search_policy.parse_budget(a.search))}
    else:
        built = {'A': elo_ladder.participant('A', a.a)}
    maps, b_kits = map_catalogue.parse_map_ids(a.maps), None
    member = registry.by_name(registry.load()).get(a.b)
    if member:
        built['B'] = league_games.participant(member)
        built['B'].name = 'B'
        b_kits = registry.kits(member)
        maps = [m for m in maps if registry.plays_map(member, m)]
    else:
        built['B'] = elo_ladder.participant('B', a.b)
    if a.temperature_a:
        t = [float(v) for v in a.temperature_a.split(',')]
        built['A'].temperature = tuple(t * 3 if len(t) == 1 else t)
    league_games.configure(a.out.parent / 'all-combos-workers')
    todo = tasks(a.pairs, a.seed_base, maps, b_kits)
    kit_of = {(t['seed'], t['side']): t['a_kit'] for t in todo}
    # Forked workers unless searching: the search keeps per-thread handles and timing totals.
    played = league_games.play(built, todo, a.workers, a.seed_base, processes=not a.search and not a.threads)
    if a.games_out:
        with open(a.games_out, 'w') as f:
            for g in sorted(played, key=lambda g: (g['seed'], g['side'])):
                f.write(json.dumps(dict(a=str(a.a), b=str(a.b), seed_base=a.seed_base, pair=g['seed'] - a.seed_base,
                                        seed=g['seed'], map=g['map'], side=g['side'], kits=g['kits'], a_kit=g['a_kit'],
                                        outcome=g['outcome'], draw_kind=g.get('draw_kind'), ticks=g['ticks'],
                                        deaths=g['deaths'])) + '\n')
    search = None
    if a.search:
        built['A'].finish()
        search = built['A'].summary()
        print(f"search {a.search}: {search['ms_per_decision']:.1f} ms per decision, "
              f"{search['rollouts_per_searched']:.1f} rollouts per searched decision, {search['games']} games")
    ensemble = {k: v.summary() for k, v in built.items() if isinstance(v, elo_ladder.Ensemble)} or None
    for k, s in (ensemble or {}).items():
        print(f"ensemble {k} of {len(s['members'])}: {s['ms_per_decision']:.2f} ms per decision over {s['decisions']} decisions")

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
    row = dict(label=a.label or f'{a.a} vs {a.b}', a=str(a.a), b=str(a.b), seed_base=a.seed_base, maps=a.maps, search=search,
               temperature_a=a.temperature_a, ensemble=ensemble,
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
