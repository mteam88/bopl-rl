#!/usr/bin/env python3
"""Write a .roster for a human rating session (Tools/play-vs-agent-mac.sh <roster>).

The shim (train/bopl_agent.h) draws one roster member per round,
so ~100 rounds against this file spread over the pool and every round is a
Bradley-Terry observation of the human against a rated opponent. Members are
the pool's scripted bots (minus search-*, which the shim cannot run) and its
checkpoints, plus --extra ones. Weights follow the information a game
carries: a Gaussian in Elo distance from the human's current estimate
(reference_pool.rate_human on --human-dir; uniform when there are no games),
floored at --min-weight so every member keeps showing up.

    Tools/rating_roster.py                      # pool v6, Matthew's games, Mac paths
    Tools/rating_roster.py --out x.roster --mac-root ''   # paths for this machine

Then: rsync the roster (and the anchors it names) to the Mac and run
Tools/play-vs-agent-mac.sh <roster>. play_vs_agent_results.py attributes
rounds from the shim's draw log; games.jsonl rows carry the member as agent.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'train/sweep'), str(ROOT / 'eval')]
import reference_pool  # noqa: E402

DEFAULT_MAC_ROOT = '/Users/mte/Documents/Coding/bopl-re/ripped/ExportedProject'


def expected(human, member):
    return 1 / (1 + 10 ** ((member - human) / 400))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', type=Path, default=ROOT / 'native/artifacts/rl/sweeps/anchors/pool-v6/pool.json')
    parser.add_argument('--human-dir', type=Path, default=reference_pool.DEFAULT_HUMAN)
    parser.add_argument('--extra', action='append', default=[], help='NAME=PATH checkpoint outside the pool (weight 1, or NAME=PATH=RATING)')
    parser.add_argument('--band', type=float, default=150, help='Elo sigma of the weight Gaussian')
    parser.add_argument('--min-weight', type=float, default=0.25)
    parser.add_argument('--human', type=float, help='override the human Elo estimate')
    parser.add_argument('--mac-root', default=DEFAULT_MAC_ROOT, help="checkout root on the machine that plays ('' = this one)")
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()

    pool = reference_pool.load_pool(args.pool)
    pool_dir = args.pool.resolve().parent
    members = []
    for m in pool['members']:
        if m['kind'] == 'scripted' and not m['name'].startswith('search-'):
            members.append(dict(name=m['name'], spec=m['name'], rating=m['rating']))
        elif m['kind'] == 'checkpoint':
            path = (pool_dir / m['path']).resolve()
            if not path.is_file() or not Path(str(path) + '.json').is_file():
                print(f'{m["name"]}: {path} (or its .json) missing; skipped', file=sys.stderr)
                continue
            members.append(dict(name=m['name'], spec=str(path), rating=m['rating'], sha256=m.get('sha256')))
    for item in args.extra:
        parts = item.split('=')
        name, path = parts[0], Path(parts[1]).resolve()
        rating = float(parts[2]) if len(parts) > 2 else None
        if not path.is_file():
            parser.error(f'--extra {name}: {path} missing')
        members.append(dict(name=name, spec=str(path), rating=rating))

    human = args.human
    human_note = f'given {human:.0f}' if human is not None else 'no games yet: uniform weights'
    if human is None:
        try:
            result = reference_pool.rate_human(pool, args.human_dir)
            human = result['rating']
            human_note = (f"{human:.0f} [{result['interval'][0]:.0f}, {result['interval'][1]:.0f}] from {result['games']} "
                          f"games vs pool members in {args.human_dir}")
        except ValueError as error:
            human_note = f'{error}: uniform weights'

    for m in members:
        if human is None or m['rating'] is None:
            m['weight'] = 1.0
        else:
            m['weight'] = max(args.min_weight, math.exp(-0.5 * ((m['rating'] - human) / args.band) ** 2))
        m['p_human'] = expected(human, m['rating']) if human is not None and m['rating'] is not None else None

    out = args.out or ROOT / 'native/artifacts/rl/human/rosters' / f"{pool_dir.name}-{time.strftime('%Y%m%d')}.roster"
    out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(m['weight'] for m in members)
    lines = [f'# rating roster from {args.pool} (pool {pool["pool_sha256"][:8]}), written {time.strftime("%Y-%m-%d %H:%M")}',
             f'# human estimate: {human_note}',
             f'# weight = max({args.min_weight}, exp(-((rating - human) / {args.band:.0f})^2 / 2)); {len(members)} members',
             '# spec  weight        # name  pool-rating  share  P(human wins)']
    for m in sorted(members, key=lambda m: -(m['rating'] or 0)):
        spec = m['spec']
        if args.mac_root and spec.startswith(str(ROOT)):
            spec = args.mac_root + spec[len(str(ROOT)):]
        rating = f"{m['rating']:.0f}" if m['rating'] is not None else '?'
        p = f"{m['p_human']:.2f}" if m['p_human'] is not None else '?'
        lines.append(f"{spec} {m['weight']:.3f}    # {m['name']}  {rating}  {m['weight'] / total:.2f}  {p}")
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n-> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
