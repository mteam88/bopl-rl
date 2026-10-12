"""How did A lose? Classify every saved non-win of a probe (eval/probe.py) from its frames.

    venv/bin/python eval/loss_causes.py DIR [DIR ...] [--out causes.jsonl]

Per game: who died first and when, how (water, blast zone, squished, eaten, beamed,
opponent, own, unattributed (the sim credits no one), or nothing recorded), where, the distance to the opponent at death,
and which of A's abilities it pressed in the last two seconds. Death geometry is the
one loss_taxonomy.py uses (SceneBounds.cs).
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluation_protocol import ability_names  # noqa: E402

WATER_Y = -11.3
BLAST_X = 131.25
BLAST_Y = 72.5
CODE = {1: 'squished', 3: 'eaten', 4: 'beamed'}
SLOT_BITS = (2, 4, 8)  # buttons bit per ability slot; bit 1 is jump
LOOKBACK = 30  # decisions, two seconds at 15 Hz


def death(frames, player, telemetry):
    """(frame index, cause) of player's first death, or (None, None) if it never died."""
    index = next((i for i, f in enumerate(frames) if not f['players'][player]['flags'] & 1), None)
    if index is None:
        return None, None
    p, before = frames[index]['players'][player], frames[index - 1]['players'][player]
    if p['cause_of_death'] in CODE:
        return index, CODE[p['cause_of_death']]
    if telemetry['deaths_opponent']:
        return index, 'opponent'
    if telemetry['deaths_self']:
        return index, 'own'
    if p['y'] < WATER_Y + 1.0 or (before['y'] < WATER_Y + 3 and p['y'] < 0):
        return index, 'water'
    if abs(p['x']) > BLAST_X - 5:
        return index, 'blast-x'
    if p['y'] > BLAST_Y - 5:
        return index, 'blast-y'
    if telemetry['deaths_other']:
        return index, 'unattributed'  # the sim credits no one: in the games seen, an ability projectile (Missile, Throw) next to the body
    return index, 'environment-other' if telemetry['deaths_environment'] else 'unrecorded'


def classify(path, names):
    visual = json.loads(path.read_text())
    frames, result = visual['frames'], visual['result']
    a = result['side']
    final = frames[-1]
    kits = result['lineup_ids'][:3], result['lineup_ids'][3:]  # side 0's kit, then side 1's
    row = dict(name=path.stem, probe=path.parents[2].name, map=result['map'], seed=result['seed'], side=a,
               outcome=result['outcome'], seconds=round(final['t'] / 60, 2),
               a_kit=[names[i] for i in kits[a]], b_kit=[names[i] for i in kits[1 - a]])
    deaths = {p: death(frames, p, final['telemetry'][p]) for p in (0, 1)}
    row['a_death'], row['b_death'] = deaths[a][1], deaths[1 - a][1]
    index = deaths[a][0]
    if index is None:
        return row
    me, them = frames[index]['players'][a], frames[index - 1]['players'][1 - a]
    row.update(death_seconds=round(frames[index]['t'] / 60, 2), death_xy=[round(me['x'], 1), round(me['y'], 1)],
               distance=round(math.hypot(me['x'] - them['x'], me['y'] - them['y']), 1))
    pressed = Counter()
    for f in frames[max(0, index - LOOKBACK):index]:
        buttons = f['actions'][a][1]
        for slot, bit in enumerate(SLOT_BITS):
            if buttons & bit:
                pressed[row['a_kit'][slot]] += 1
        if buttons & 1:
            pressed['jump'] += 1
    row['pressed_last_2s'] = dict(pressed)
    # Highest point in the last two seconds: a long drop into water reads as a launch, a short one as a walk-off.
    row['peak_y_last_2s'] = round(max(f['players'][a]['y'] for f in frames[max(0, index - LOOKBACK):index]), 1)
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dirs', nargs='+', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    names = ability_names()
    rows = [classify(path, names) for d in a.dirs for path in sorted((d / 'evaluations').glob('*/*.json'))
            if path.name != 'result.json']
    for d in a.dirs:
        mine = [r for r in rows if r['probe'] == d.name]
        print(f'{d.name}: {len(mine)} saved non-wins; A died by', dict(Counter(r['a_death'] for r in mine).most_common()))
    if a.out:
        a.out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
