#!/usr/bin/env python3
"""Write a .roster for a human rating session (Tools/play-vs-agent-mac.sh <roster>).

The shim (train/bopl_agent.h) draws one roster member per round, so ~100 rounds
against this file spread over the league and every round is a Bradley-Terry
observation of the human against a rated member. Members are the league's rated
scripted bots (minus search-*, which the shim cannot run) and checkpoints that can
play --kit, plus --extra ones. Weights follow the information a game carries: a
Gaussian in Elo distance from the human's current estimate (their games in
--human-dir against league members, with the members' league ratings held fixed;
uniform when there are none), floored at --min-weight so every member keeps
showing up.

    Tools/rating_roster.py                      # Matthew's games, Mac paths
    Tools/rating_roster.py --out x.roster --mac-root ''   # paths for this machine

Then: rsync the roster (and the checkpoints it names) to the Mac and run
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
sys.path[:0] = [str(ROOT / 'league'), str(ROOT / 'eval')]
import candidate  # noqa: E402
import elo_ladder  # noqa: E402
import registry  # noqa: E402

DEFAULT_MAC_ROOT = '/Users/mte/Documents/Coding/bopl-re/ripped/ExportedProject'
DEFAULT_HUMAN = ROOT / 'native/artifacts/rl/human/mac'


def human_games(directory):
    """games.jsonl plus .session-*.jsonl round rows, deduped on (session, round),
    minus the rounds and whole sessions excluded.json names."""
    directory = Path(directory)
    rows, seen = [], set()
    if (directory / 'games.jsonl').is_file():
        for line in (directory / 'games.jsonl').read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen.add((row.get('session'), row.get('round')))
                rows.append(row)
    for path in sorted(directory.glob('.session-*.jsonl')):
        session = path.name[len('.session-'):-len('.jsonl')]
        for line in path.read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            if row and row.get('kind') in (None, 'round') and (session, row.get('round')) not in seen:
                seen.add((session, row.get('round')))
                rows.append(dict(row, session=session))
    excluded = json.loads((directory / 'excluded.json').read_text()) if (directory / 'excluded.json').is_file() else []
    sessions = {item['session'] for item in excluded if 'rounds' not in item}
    rounds = {(item['session'], int(r)) for item in excluded for r in item.get('rounds') or ()}
    return [row for row in rows if row.get('session') not in sessions and (row.get('session'), row.get('round')) not in rounds]


def human_rating(directory, ratings):
    """The human's rating against league members, from games.jsonl rows (a = human,
    b = 'policy:<sha256>' or 'bot:<name>', the member's league id)."""
    played = []
    for row in human_games(directory):
        b = row.get('b') or ''
        member = b.split(':', 1)[1] if ':' in b else None
        if member in ratings:
            played.append(dict(b=member, map=int(row.get('map', 0)), seed=int(row.get('seed') or 0),
                               outcome=row.get('outcome', 'draw')))
    if not played:
        raise ValueError(f'no games against rated league members in {directory}')
    return dict(rating=elo_ladder.bradley_terry_fixed(candidate.records(played), ratings),
                interval=candidate.interval(played, ratings), games=len(played))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kit', default='2,4,14', help='the kit the human plays: members declared on other kits are left out')
    parser.add_argument('--human-dir', type=Path, default=DEFAULT_HUMAN)
    parser.add_argument('--extra', action='append', default=[], help='NAME=PATH checkpoint outside the league (weight 1, or NAME=PATH=RATING)')
    parser.add_argument('--band', type=float, default=150, help='Elo sigma of the weight Gaussian')
    parser.add_argument('--min-weight', type=float, default=0.25)
    parser.add_argument('--human', type=float, help='override the human Elo estimate')
    parser.add_argument('--mac-root', default=DEFAULT_MAC_ROOT, help="checkout root on the machine that plays ('' = this one)")
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()

    kit = [int(x) for x in args.kit.split(',')]
    key, ratings = candidate.reference()
    members = []
    for row in registry.by_id(registry.load(verify=False)).values():
        kits = registry.kits(row)
        if row['id'] not in ratings or (kits is not None and kit not in [list(k) for k in kits]):
            continue
        if row['kind'] == 'scripted' and not row['name'].startswith('search-'):
            members.append(dict(name=row['name'], spec=row['name'], rating=ratings[row['id']]))
        elif row.get('path'):
            members.append(dict(name=row['name'], spec=str(ROOT / row['path']), rating=ratings[row['id']]))
    for item in args.extra:
        parts = item.split('=')
        name, path = parts[0], Path(parts[1]).resolve()
        if not path.is_file():
            parser.error(f'--extra {name}: {path} missing')
        members.append(dict(name=name, spec=str(path), rating=float(parts[2]) if len(parts) > 2 else None))

    human = args.human
    human_note = f'given {human:.0f}' if human is not None else 'no games yet: uniform weights'
    if human is None:
        try:
            result = human_rating(args.human_dir, ratings)
            human = result['rating']
            human_note = (f"{human:.0f} [{result['interval'][0]:.0f}, {result['interval'][1]:.0f}] from {result['games']} "
                          f"games vs league members in {args.human_dir} (Unity games on any map and kit)")
        except ValueError as error:
            human_note = f'{error}: uniform weights'

    for m in members:
        if human is None or m['rating'] is None:
            m['weight'] = 1.0
        else:
            m['weight'] = max(args.min_weight, math.exp(-0.5 * ((m['rating'] - human) / args.band) ** 2))
        m['p_human'] = candidate.expected(human, m['rating']) if human is not None and m['rating'] is not None else None

    out = args.out or ROOT / 'native/artifacts/rl/human/rosters' / f"league-{time.strftime('%Y%m%d')}.roster"
    out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(m['weight'] for m in members)
    lines = [f'# rating roster from the league ({key}), written {time.strftime("%Y-%m-%d %H:%M")}',
             f'# human estimate: {human_note}',
             f'# weight = max({args.min_weight}, exp(-((rating - human) / {args.band:.0f})^2 / 2)); {len(members)} members',
             '# spec  weight        # name  league-rating  share  P(human wins)']
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
