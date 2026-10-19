"""Humans vs bots on one set of offense and behaviour statistics (HUMAN-GAP-2026-10.md).

    python human_gap.py HUMANS.jsonl NAME=OFFENSE_DIAG.jsonl[,...] ... [--json out.json]

HUMANS.jsonl is eval/human_gap.py's output (faithful 1v1 replays, one row per seat); the other
inputs are offense_diag.py rows (A = the bot under test). Groups from the human file:
  human-vs-bot     human seats of Mac human-vs-bot games
  human-vs-human   human seats of the older human-human recordings (gamepad vs gamepad)
  bot<-human:NAME  the bot seats of the human-vs-bot games, by bot era
Every group goes through offense_metrics.summarize (the same definitions), bootstrapped by session
for human rows (seed = session hash, map dropped) and by seed pair for sim rows. Added here:
  win             A's score
  le120           share of rounds whose decisive death came before sudden death (level tick 7200)
  loser_how       how the round's loser died (credited / self / env-near / env-unforced), share
  loser_cause     the loser's death cause (projectile, ring-out, squished, ...), share
  a_death_how     how A itself died, share of A's rounds lost
  press_mix       share of A's button-down decisions per ability, Dash/Bow/Platform kits
  dbp_per_min     A's activations per ability, arrows fired and lethal arrows per minute (RlTelemetry, slots
                  mapped to the seat's kit order), Dash/Bow/Platform kits; human file only
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offense_metrics as om  # noqa: E402

NAMES = {2: 'Dash', 4: 'Bow', 14: 'Platform', 3: 'Grenade', 26: 'Blink'}
DBP = sorted((2, 4, 14))


def a_kit(g):
    if 'a_kit' in g:
        return g['a_kit']
    k = g['kits']
    return k[g['side']] if isinstance(k[0], list) else k[:3]


def extras(rows):
    dec = [g for g in rows if g['outcome'] in ('a', 'b')]
    how = Counter(om.how(om.loser_death(g)) for g in dec)
    cause = Counter((om.loser_death(g) or {}).get('cause', 'none') for g in dec)
    lost = [g for g in rows if g['outcome'] == 'b']
    a_how = Counter(om.how(om.loser_death(g)) for g in lost)
    mix, act, minutes = Counter(), Counter(), 0.
    slots = ('dash_activations', 'bow_activations', 'platform_activations')   # RlTelemetry per-slot counters
    for g in rows:
        tel = g['players'][g['side']].get('tel', {})
        if sorted(a_kit(g)) == DBP and slots[0] in tel:
            minutes += g['ticks'] / 3600
            for slot, ability in enumerate(a_kit(g)):
                act[NAMES[ability]] += tel[slots[slot]]
            act['arrows_fired'] += tel['arrows_fired']; act['arrow_kills'] += tel['arrow_hits']
    for g in rows:
        kit = a_kit(g)
        if sorted(kit) != DBP:
            continue
        b = g['players'][g['side']]['buttons']
        for slot, ability in enumerate(kit):
            mix[NAMES.get(ability, ability)] += b[slot + 1] * g['decisions']
    share = lambda c: {k: round(v / max(sum(c.values()), 1e-9), 3) for k, v in c.most_common()}
    return dict(win=round(sum(om.pts(g) for g in rows) / max(len(rows), 1), 3),
                le120=round(sum(om.end_tick(g) < om.SUDDEN for g in dec) / max(len(dec), 1), 3),
                loser_how=share(how), loser_cause=share(cause), a_death_how=share(a_how), press_mix=share(mix),
                dbp_per_min={k: round(v / minutes, 2) for k, v in act.items()} if minutes else None,
                n_dbp=sum(sorted(a_kit(g)) == DBP for g in rows))


def groups(human_file):
    rows = [json.loads(line) for line in open(human_file)]
    rows = [dict(g, map=0) for g in rows if g['faithful']]
    out = {'human-vs-bot': [g for g in rows if g['who'] == 'human' and g['source'] == 'mac'],
           'human-vs-g111': [g for g in rows if g['who'] == 'human' and g['opponent'].startswith('g111-20b')],
           'human-vs-human': [g for g in rows if g['who'] == 'human' and g['opponent'] == 'human']}
    for g in rows:
        if g['source'] == 'mac' and g['who'] not in ('human', 'keyboard', 'agent'):
            era = 'g111-20b (2026-10-10)' if g['who'].startswith('g111-20b') else 'v4-full (2026-09-17)' \
                if '2313256960' in g['who'] else 'Sept nets (09-11..14)' if g['who'].endswith('.bin') else 'scripted'
            out.setdefault(f'bot<-human:{era}', []).append(g)
    return out


def main(argv):
    js = None
    if '--json' in argv:
        i = argv.index('--json'); js = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    sets = groups(argv[0])
    for spec in argv[1:]:
        name, files = spec.split('=', 1)
        sets[name] = [json.loads(line) for f in files.split(',') for line in open(f)]
    res = {}
    for name, rows in sets.items():
        s = om.summarize(rows)
        s.update(extras(rows))
        res[name] = s
        print(f'\n== {name} (n {s["n"]}, Dash/Bow/Platform {s["n_dbp"]})')
        for k, v in s.items():
            if isinstance(v, tuple):
                print(f'  {k:12s} {v[0]:8.3f} [{v[1]:.3f},{v[2]:.3f}] n {v[3]}')
        for k in ('win', 'le120', 'loser_how', 'loser_cause', 'a_death_how', 'press_mix', 'dbp_per_min'):
            print(f'  {k:12s} {s[k]}')
    if js:
        json.dump(res, open(js, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
