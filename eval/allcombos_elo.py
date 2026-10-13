"""Elo and versatility on the all-combo condition, from games already played.

    venv/bin/python eval/allcombos_elo.py --member allkit1=PATH --member a20=PATH ... [--play 200] [--json OUT]

The league's ratings (league rate / candidate) are fitted on the rating condition, where every
member plays its own kits; an all-kit net has no kits of its own there and the round robin is
on an older sim build. This rates the named checkpoints where the goal is set, random kits from
every ability (league/all_combos.py), on the repo's one Elo scale: elo_ladder.bradley_terry_fit,
draw-aware, `random` anchored at 0.

Games are harvested from every all_combos row (results.jsonl, trend.jsonl) under the sweeps whose
A and B are both named members and that ran without search or a temperature, and from every
eval/probe.py run of a member against `random` (also random all-ability kits). --play N first
plays N seed pairs for every member pair with fewer than N pairs on record.

Versatility is the per-ability Elo (league/candidate.py's per-kit rating, here per ability):
the 1-D MLE of a member's rating from only the games where its kit holds that ability, with
its opponents held at their fitted ratings. A versatile net has no weak ability; the readout
is the weakest ability, the 10th percentile and the spread.
"""
import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'eval')]
import elo_ladder  # noqa: E402

SWEEPS = ROOT / 'native/artifacts/rl/sweeps'


def norm(path):
    return str((ROOT / path).resolve()) if path else None


def harvest(members):
    """Games between members: {(a, b): [wins_a, wins_b, draws]} and per-ability rows."""
    by_path = {norm(p): n for n, p in members.items()}
    records = defaultdict(lambda: [0.0, 0.0, 0.0])
    abilities = []  # (member, opponent, ability, score, games)
    seen = set()
    for f in sorted(SWEEPS.glob('**/*.jsonl')):
        if f.name not in ('results.jsonl', 'trend.jsonl'):
            continue
        for line in f.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not {'a', 'b', 'score', 'games'} <= r.keys() or r.get('search') or r.get('temperature_a'):
                continue
            a, b = by_path.get(norm(r['a'])), by_path.get(norm(r['b']))
            if not a or not b or a == b:
                continue
            key = (a, b, r.get('seed_base'), r['games'])
            if key in seen:  # the same games recorded twice (a rerun or a copied row)
                continue
            seen.add(key)
            g, d = float(r['games']), float(r.get('draws', 0))
            wa = r['score'] * g - d / 2
            rec = records[(a, b)]
            rec[0] += wa; rec[1] += g - wa - d; rec[2] += d
            for ability, (score, n) in (r.get('by_ability') or {}).items():
                abilities.append((a, b, ability, score, n))
    for status in sorted(SWEEPS.glob('**/config.json')):
        try:
            probe = json.load(status.open()).get('probe')
        except (json.JSONDecodeError, OSError):
            continue
        if not probe or probe.get('b') != 'random' or probe.get('search') or probe.get('temperature'):
            continue
        a = by_path.get(norm(probe['a']))
        games_file = status.parent / 'games.jsonl'
        if not a or not games_file.exists():
            continue
        key = (a, 'random', probe.get('seed_base'), probe.get('pairs'))
        if key in seen:
            continue
        seen.add(key)
        rec = records[(a, 'random')]
        for line in games_file.open():
            o = json.loads(line)['outcome']
            rec[0 if o == 'a' else 1 if o == 'b' else 2] += 1
    return dict(records), abilities


def pairs_on_record(records, a, b):
    return sum(sum(records.get(k, (0, 0, 0))) for k in ((a, b), (b, a))) / 2


def play_missing(members, records, pairs, workers):
    names = [n for n in members if n != 'random']
    out = SWEEPS / 'allcombos-elo' / 'results.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            have = pairs_on_record(records, a, b)
            if have >= pairs:
                continue
            print(f'playing {a} vs {b}: {have:.0f} pairs on record, playing {pairs}', flush=True)
            subprocess.run([sys.executable, str(ROOT / 'league/all_combos.py'), members[a], members[b],
                            '--pairs', str(pairs), '--workers', str(workers), '--seed-base', '151001',
                            '--label', f'elo-{a}-vs-{b}', '--out', str(out)], check=True, cwd=ROOT)


def ability_elo(member, rows, ratings):
    """1-D MLE per ability with the opponents held at their ratings (Newton on the logistic)."""
    by_ability = defaultdict(list)
    for a, b, ability, score, n in rows:
        if a == member and b in ratings:
            by_ability[ability].append((ratings[b], score, n))
    out = {}
    for ability, games in by_ability.items():
        n = sum(g[2] for g in games)
        if n < 20:
            continue
        r = ratings[member]
        for _ in range(50):
            p = [1 / (1 + math.exp(-(r - rb) / elo_ladder.ELO_SCALE)) for rb, _, _ in games]
            grad = sum(k * (s - pi) for (_, s, k), pi in zip(games, p))
            info = sum(k * pi * (1 - pi) for (_, _, k), pi in zip(games, p)) + 1e-9
            step = elo_ladder.ELO_SCALE * grad / info
            r += max(-200.0, min(200.0, step))
            if abs(step) < 1e-3:
                break
        out[ability] = (r, n)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--member', action='append', default=[], help='NAME=checkpoint path (repeat); random is implicit')
    p.add_argument('--play', type=int, default=0, help='first play this many seed pairs for every member pair short of it')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--json', type=Path)
    a = p.parse_args(argv)
    members = dict(m.split('=', 1) for m in a.member)
    records, rows = harvest(members)
    if a.play:
        play_missing(members, records, a.play, a.workers)
        records, rows = harvest(members)
    names = ['random'] + list(members)
    fit = elo_ladder.bradley_terry_fit(names, {k: tuple(v) for k, v in records.items()}, anchor='random')
    ratings, ci = fit['ratings'], fit['intervals']
    table = []
    for n in sorted(names, key=lambda n: -ratings[n]):
        games = sum(sum(v) for k, v in records.items() if n in k)
        per = ability_elo(n, rows, ratings) if n != 'random' else {}
        vals = sorted(r for r, _ in per.values())
        weakest = sorted(per.items(), key=lambda kv: kv[1][0])[:3]
        table.append(dict(name=n, elo=ratings[n], ci=ci[n], games=games, abilities=len(vals),
                          ability_min=vals[0] if vals else None,
                          ability_p10=float(np.percentile(vals, 10)) if vals else None,
                          ability_spread=float(np.std(vals)) if vals else None,
                          weakest=[(k, round(r), n_) for k, (r, n_) in weakest]))
    print(f"{'member':<22}{'Elo':>6}  {'95%':<14}{'games':>7}  {'abil min':>8}{'p10':>6}{'sd':>5}  weakest")
    for t in table:
        lo, hi = t['ci']
        v = (f"{t['ability_min']:>8.0f}{t['ability_p10']:>6.0f}{t['ability_spread']:>5.0f}"
             if t['abilities'] else ' ' * 19)
        print(f"{t['name']:<22}{t['elo']:>6.0f}  [{lo:>4.0f}, {hi:>4.0f}]  {t['games']:>7.0f}  {v}  "
              + ', '.join(f'{k} {r}' for k, r, _ in t['weakest']))
    missing = [(x, y) for i, x in enumerate(names) for y in names[i + 1:] if pairs_on_record(records, x, y) == 0]
    if missing:
        print(f'{len(missing)} member pairs have no games (ratings are linked through the others):',
              ', '.join(f'{x}-{y}' for x, y in missing))
    if a.json:
        a.json.write_text(json.dumps(dict(table=table, records={f'{k[0]}|{k[1]}': v for k, v in records.items()}), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
