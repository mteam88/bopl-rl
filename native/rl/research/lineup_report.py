"""What a multi-ability run actually played, read off its in-run evaluations.

evaluate.py already records everything this needs per checkpoint: the resolved
loadout selection, per-ability activation and entity-row counters, the lineup of
every game and the per-group timeout counts. Nothing is replayed here; the point
is to see, without spending a game, whether a run is using the abilities it was
given (an ability with no activations is an ability the policy never learned to
press), whether the entity rows of the observation are overflowing on the busy
ones, and — for a lineup set — which lineups it is actually winning on.

The scripted bots (bopl_bots.h) reach for the Dash / Bow / Platform slots by
index, so on any other lineup they press whatever sits in those slots. Their
win rates off 2,4,14 are a property of the opponent, not only of the policy.

    python native/rl/research/lineup_report.py --run <run dir> [--last-k 3]
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluation_protocol import ability_names, scripted_opponent


def evaluations(run_dir, suite='evaluations'):
    """[(step, report)] in step order for a run's in-run evaluation results."""
    rows = []
    for path in sorted((Path(run_dir) / suite).glob('*/result.json')):
        try:
            report = json.loads(path.read_text())
        except ValueError:      # a result.json still being written
            continue
        rows.append((int(report.get('step') or path.parent.name), report))
    return sorted(rows)


def lineup_key(loadouts):
    """'2,4,14' for a mirror episode, '2,4,14|3,5,6' when the players differ."""
    left, right = [list(map(int, half)) for half in loadouts]
    text = ','.join(str(a) for a in left)
    return text if left == right else text + '|' + ','.join(str(a) for a in right)


def lineup_rows(report):
    """Win/timeout counts per (lineup, opponent family) over an evaluation's games.

    Only games against the scripted bots and `random` are counted: selfplay and
    the frozen/initial opponents drift with the run and say nothing about a lineup.
    """
    rows = {}
    for game in report.get('games') or ():
        if not game.get('loadouts'):
            continue
        opponent = game.get('opponent')
        family = 'scripted' if scripted_opponent(opponent) else opponent
        if family not in ('scripted', 'random'):
            continue
        row = rows.setdefault((lineup_key(game['loadouts']), family), dict(games=0, wins=0, timeouts=0))
        row['games'] += 1
        row['wins'] += bool(game.get('win'))
        row['timeouts'] += bool(game.get('timeout'))
    return rows


def ability_rows(report):
    """[(id, name, activations, entity rows, dropped rows)] for every ability that appeared."""
    activations = {int(k): v for k, v in (report.get('ability_activations') or {}).items()}
    overflow = {int(k): v for k, v in (report.get('entity_overflow') or {}).items()}
    names = ability_names()
    ids = sorted(set(activations) | {k for k, v in overflow.items() if v.get('rows')})
    return [(i, names.get(i, str(i)), activations.get(i, 0),
             (overflow.get(i) or {}).get('rows', 0), (overflow.get(i) or {}).get('dropped', 0))
            for i in ids]


def timeout_rate(report):
    """Timeouts over games across every evaluation group (the run's stalling rate)."""
    games = sum(group['games'] for group in report.get('groups') or ())
    timeouts = sum(group['timeouts'] for group in report.get('groups') or ())
    return timeouts / games if games else 0.0


def format_report(step, report):
    selection = report.get('loadout_selection') or {}
    lines = [f"step {step:>12}  mode {selection.get('mode', '?')}"
             f"  timeouts {timeout_rate(report):.1%}"
             + (f"  set {Path(selection['set_file']).name}" if selection.get('set_file') else '')]
    lines.append(f"  {'ability':<24} {'activations':>12} {'entity rows':>12} {'dropped':>9}")
    for ability, name, activations, rows, dropped in ability_rows(report):
        lines.append(f"  {ability:>2} {name:<21} {activations:>12} {rows:>12} "
                     + (f"{dropped:>9} ({dropped / rows:.1%})" if dropped else f"{dropped:>9}"))
    rows = lineup_rows(report)
    if len(set(key for key, _ in rows)) > 1:
        names = ability_names()
        lines.append(f"  {'lineup':<14} {'vs scripted':>16} {'vs random':>16}   abilities")
        cell = lambda row: f"{row['wins'] / row['games']:.0%} of {row['games']}" if row else '-'
        for key in sorted({key for key, _ in rows}):
            label = ' | '.join('/'.join(names.get(int(a), a) for a in half.split(','))
                               for half in key.split('|'))
            lines.append(f"  {key:<14} {cell(rows.get((key, 'scripted'))):>16} "
                         f"{cell(rows.get((key, 'random'))):>16}   {label}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--suite', choices=('evaluations', 'final-evaluations'), default='evaluations')
    parser.add_argument('--last-k', type=int, default=0, help='Only the last k checkpoints (0 = all)')
    parser.add_argument('--json', action='store_true', help='Machine-readable rows instead of the table')
    args = parser.parse_args()
    rows = evaluations(args.run, args.suite)
    if args.last_k > 0:
        rows = rows[-args.last_k:]
    if not rows:
        parser.error(f'No {args.suite}/*/result.json under {args.run}')
    if args.json:
        print(json.dumps([dict(step=step, loadout_selection=report.get('loadout_selection'),
                               timeout_rate=timeout_rate(report),
                               abilities=[dict(id=i, name=n, activations=a, entity_rows=r, dropped=d)
                                          for i, n, a, r, d in ability_rows(report)],
                               lineups=[dict(lineup=key, opponent=family, **row)
                                        for (key, family), row in sorted(lineup_rows(report).items())])
                          for step, report in rows], indent=1))
        return
    for step, report in rows:
        print(format_report(step, report))
        print()


if __name__ == '__main__':
    main()
