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

from evaluation_protocol import ability_names, scripted_opponent, DEFAULT_LOADOUT

# A run older than the multi-ability work has no loadout block anywhere: it
# trained and was evaluated on the production triple, so that is what it means.
DEFAULT_SELECTION = dict(mode='fixed', fixed=list(DEFAULT_LOADOUT), pool=[], lineups=[],
                         set_file=None, no_dup=False, mirror_share=0.0)
# Opponent families a lineup can be judged on. The scripted bots press ability
# SLOTS (bopl_bots.h), so their column is not comparable across lineups; frozen
# policies and selfplay drift with the run and are informative, not absolute.
FAMILIES = ('random', 'scripted', 'frozen', 'selfplay', 'initial')


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


def selection(record):
    """The loadout block of a run config or an evaluation result; fixed 2,4,14 when absent."""
    value = (record or {}).get('loadout_selection') or (record or {}).get('loadouts') or {}
    return dict(DEFAULT_SELECTION, **{key: value[key] for key in DEFAULT_SELECTION if key in value})


def named(ids):
    """[{id, name}] — the UI shows names and keeps the id for the tooltip."""
    names = ability_names()
    return [dict(id=int(a), name=names.get(int(a), str(a))) for a in ids]


def selection_abilities(sel):
    """Every ability the selection can hand out, sorted."""
    found = set(sel['pool'])
    for row in sel['lineups']:
        found.update(row)
    return sorted(found or set(sel['fixed']))


def selection_badge(sel):
    """'2·4·14', 'set lineups-8 ×8 (mirror 100%)', 'pool 36 (mirror 50%)'."""
    mirror = f" (mirror {sel['mirror_share']:.0%})" if sel['mirror_share'] else ''
    if sel['mode'] == 'set':
        name = Path(sel['set_file']).stem if sel['set_file'] else 'set'
        return f"set {name} ×{len(sel['lineups'])}{mirror}"
    if sel['mode'] == 'pool':
        return f"pool {len(sel['pool'])}{mirror}" + (' no-dup' if sel['no_dup'] else '')
    return '·'.join(str(a) for a in sel['fixed'])


def lineup_entry(key):
    """{key, label, abilities, mirror} for a lineup key, ability names resolved."""
    halves = key.split('|')
    return dict(key=key, label=lineup_label(key), mirror=len(halves) == 1,
                abilities=named(halves[0].split(',')),
                opponent=None if len(halves) == 1 else named(halves[1].split(',')))


def lineup_label(key):
    """'Dash/Bow/Platform' for a mirror key, both halves when the players differ."""
    names = ability_names()
    return ' | '.join('/'.join(names.get(int(a), a) for a in half.split(','))
                      for half in key.split('|'))


def selection_brief(sel):
    """Everything the dashboard prints about a selection, ids resolved to names."""
    return dict(mode=sel['mode'], badge=selection_badge(sel), fixed=named(sel['fixed']),
                pool=named(sel['pool']), mirror_share=sel['mirror_share'], no_dup=sel['no_dup'],
                set_file=sel['set_file'], set_name=Path(sel['set_file']).name if sel['set_file'] else None,
                lineups=[lineup_entry(lineup_key([row[:3], row[3:6]])) for row in sel['lineups']],
                abilities=named(selection_abilities(sel)))


def opponent_family(name):
    """Which column a game counts in: every scripted bot is one slot-blind family."""
    if scripted_opponent(name):
        return 'scripted'
    return name if name in ('random', 'selfplay', 'initial') else 'frozen'


def digest(report):
    """Per-evaluation counters, small enough to keep one per checkpoint of a run.

    Ability rows keep the abilities that were *available* and never fired (a
    lineup member with no activations is an ability the policy never learned to
    press), which is why presence is counted off the games rather than read off
    the activation table.
    """
    sel = selection(report)
    fixed = [list(sel['fixed']), list(sel['fixed'])]
    activations = {int(k): v for k, v in (report.get('ability_activations') or {}).items()}
    overflow = {int(k): v for k, v in (report.get('entity_overflow') or {}).items()}
    present, won, lineups = {}, {}, {}
    for game in report.get('games') or ():
        loadouts = game.get('loadouts') or fixed
        cell = lineups.setdefault(lineup_key(loadouts), {}).setdefault(
            opponent_family(game.get('opponent') or ''), dict(games=0, wins=0, timeouts=0))
        cell['games'] += 1
        cell['wins'] += bool(game.get('win'))
        cell['timeouts'] += bool(game.get('timeout'))
        # A free draw makes almost every lineup unique, so the ability is the
        # only axis with enough games behind it: keep its record too.
        for ability in {int(a) for a in loadouts[0]}:
            present[ability] = present.get(ability, 0) + 1
            won[ability] = won.get(ability, 0) + bool(game.get('win'))
    names = ability_names()
    # Evaluations older than the ability counters have no activation table at
    # all; reporting those abilities as "0 activations" would invent a finding.
    ids = (sorted(set(activations) | set(present) | {k for k, v in overflow.items() if v.get('rows')})
           if activations or overflow else [])
    return dict(
        step=int(report.get('step') or 0), games=len(report.get('games') or ()),
        selection=sel, timeout_rate=timeout_rate(report), telemetry=bool(ids),
        abilities=[dict(id=i, name=names.get(i, str(i)), activations=activations.get(i, 0),
                        games=present.get(i, 0), wins=won.get(i, 0),
                        rows=(overflow.get(i) or {}).get('rows', 0),
                        dropped=(overflow.get(i) or {}).get('dropped', 0)) for i in ids],
        lineups=lineups)


def merge_digests(rows):
    """Sum a window of digests into one per-ability and per-lineup table.

    Activations are reported per game the ability was in the policy's loadout —
    a raw count is meaningless across lineups that were drawn different numbers
    of times.
    """
    abilities, lineups = {}, {}
    for row in rows:
        for ability in row['abilities']:
            seen = abilities.setdefault(ability['id'], dict(id=ability['id'], name=ability['name'],
                                                           activations=0, games=0, wins=0, rows=0, dropped=0))
            for key in ('activations', 'games', 'wins', 'rows', 'dropped'):
                seen[key] += ability.get(key, 0)
        for key, families in row['lineups'].items():
            bucket = lineups.setdefault(key, {})
            for family, cell in families.items():
                seen = bucket.setdefault(family, dict(games=0, wins=0, timeouts=0))
                for field in ('games', 'wins', 'timeouts'):
                    seen[field] += cell[field]
    for ability in abilities.values():
        ability['per_game'] = ability['activations'] / ability['games'] if ability['games'] else None
        ability['win_rate'] = ability['wins'] / ability['games'] if ability['games'] else None
        ability['drop_rate'] = ability['dropped'] / ability['rows'] if ability['rows'] else 0.0
    return dict(evaluations=len(rows), games=sum(row['games'] for row in rows),
                telemetry=any(row.get('telemetry') for row in rows),
                abilities=sorted(abilities.values(), key=lambda row: row['id']),
                lineups=lineups)


def lineup_table(lineups):
    """[{key, label, abilities, families, games, ...}] strongest scripted record first."""
    rows = []
    for key, families in lineups.items():
        families = {family: dict(cell, win_rate=cell['wins'] / cell['games'] if cell['games'] else None,
                                 timeout_rate=cell['timeouts'] / cell['games'] if cell['games'] else None)
                    for family, cell in families.items()}
        games = sum(cell['games'] for cell in families.values())
        timeouts = sum(cell['timeouts'] for cell in families.values())
        rows.append(dict(lineup_entry(key), families=families, games=games,
                         timeout_rate=timeouts / games if games else None))
    rows.sort(key=lambda row: (-(row['families'].get('scripted') or {}).get('win_rate', -1), row['key']))
    return rows


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
        lines.append(f"  {'lineup':<14} {'vs scripted':>16} {'vs random':>16}   abilities")
        cell = lambda row: f"{row['wins'] / row['games']:.0%} of {row['games']}" if row else '-'
        for key in sorted({key for key, _ in rows}):
            lines.append(f"  {key:<14} {cell(rows.get((key, 'scripted'))):>16} "
                         f"{cell(rows.get((key, 'random'))):>16}   {lineup_label(key)}")
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
