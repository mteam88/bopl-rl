#!/usr/bin/env python3
"""Survival curves, attribution and coverage from divergence_rate's JSONL.

    research/divergence/divergence_summary.py <dir-or-file>... [--json out.json]

Ticks-to-first-divergence is a survival time, and most games end (somebody
wins) long before the comparison horizon, so a game that was still exact when
its round finished is RIGHT-CENSORED, not a game that survived to the horizon.
Everything below is therefore Kaplan-Meier with Greenwood confidence bands;
reporting a raw "fraction of games that diverged" instead would understate the
rate by the share of games that simply ended early, which is most of them.

Four curves per cell, two arms by two lenses:
    q32 vs oracle   genuine logic divergence (the port's bit-exact arithmetic)
    f64 vs oracle   what training actually runs on
    all / visible   every ABI field, versus only what a policy can read
and within each, `any` (either class) and `disc` (a different DECISION, i.e.
an integer/flag/identity field), which are different questions.
"""
import json, math, sys, collections
from pathlib import Path

ABILITY_NAMES = {}
PARTIAL = {7: 'Gust', 12: 'Time stop', 15: 'Revival', 29: 'Duplicator'}
TRAIN_EXCLUDED = (12, 15)
HORIZONS = (100, 300, 1000, 3600)


def load(paths):
    runs, games = [], []
    for path in paths:
        path = Path(path)
        files = sorted(path.glob('*.jsonl')) if path.is_dir() else [path]
        for file in files:
            for line in file.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('record') == 'game':
                    games.append(row)
                elif row.get('record') == 'run':
                    runs.append(row)
    return runs, games


def kaplan_meier(observations):
    """observations: (time, event) with event=1 divergence, 0 censored.

    Returns (times, survival, variance) stepping down at each event time.
    """
    at_risk = len(observations)
    if not at_risk:
        return [], [], []
    by_time = collections.defaultdict(lambda: [0, 0])   # time -> [events, censored]
    for time, event in observations:
        by_time[time][0 if event else 1] += 1
    times, survival, variance = [], [], []
    s, v = 1.0, 0.0
    for time in sorted(by_time):
        events, censored = by_time[time]
        if events and at_risk:
            s *= 1.0 - events / at_risk
            v += events / (at_risk * (at_risk - events)) if at_risk > events else 0.0
            times.append(time); survival.append(s); variance.append(v)
        at_risk -= events + censored
    return times, survival, variance


def survival_at(km, t):
    times, surv, var = km
    s, v = 1.0, 0.0
    for time, value, variance in zip(times, surv, var):
        if time > t:
            break
        s, v = value, variance
    half = 1.96 * s * math.sqrt(v) if v > 0 else 0.0
    return s, max(0.0, s - half), min(1.0, s + half)


def quantile(km, q):
    """Smallest t with S(t) <= 1-q; None when the curve never gets there."""
    times, surv, _ = km
    for time, value in zip(times, surv):
        if value <= 1.0 - q:
            return time
    return None


def observations(games, key):
    """(time, event) for one of the eight (arm, lens, class) combinations."""
    out = []
    for game in games:
        hit = game.get(key)
        horizon = game['ticks']
        if hit is not None and hit['tick'] <= horizon:
            out.append((hit['tick'], 1))
        else:
            out.append((horizon, 0))
    return out


def wilson(k, n):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


KEYS = [(f'{arm}_{lens}_{cls}', arm, lens, cls)
        for arm in ('q32', 'f64') for lens in ('all', 'visible')
        for cls in ('anyx', 'discx', 'any', 'disc')]


def curve_table(games, label, keys=('q32_all_anyx', 'q32_all_discx', 'q32_visible_anyx',
                                    'q32_visible_discx', 'f64_all_anyx', 'f64_all_discx',
                                    'f64_visible_anyx', 'f64_visible_discx')):
    print(f'\n--- {label}  (n={len(games)} games, '
          f'{sum(g["ticks"] for g in games):,} ticks) ---')
    header = f'{"curve":28s} {"median":>8s} {"p25":>7s} {"p75":>7s}'
    for h in HORIZONS:
        header += f' {"S(" + str(h) + ")":>16s}'
    print(header)
    rows = {}
    for key in keys:
        km = kaplan_meier(observations(games, key))
        line = f'{key:28s} {str(quantile(km, 0.5)):>8s} {str(quantile(km, 0.25)):>7s} {str(quantile(km, 0.75)):>7s}'
        entry = {'median': quantile(km, 0.5), 'p25': quantile(km, 0.25), 'p75': quantile(km, 0.75)}
        for h in HORIZONS:
            s, lo, hi = survival_at(km, h)
            line += f'  {s:5.1%} [{lo:4.0%},{hi:4.0%}]'
            entry[f'S{h}'] = [s, lo, hi]
        print(line)
        rows[key] = entry
    return rows


SUBSYSTEM = [
    ('header.', 'session/clock'),
    ('player[].', 'player summary'),
    ('telemetry[].', 'event counters'),
    ('slime[].', 'slime body/abilities'),
    ('entity[].', 'entity'),
]


def subsystem(field):
    for prefix, name in SUBSYSTEM:
        if field.startswith(prefix):
            return name
    return 'other'


def attribution(games, key, label, top=14):
    hits = [g[key] for g in games if g.get(key)]
    if not hits:
        print(f'\n--- {label}: no divergences ---')
        return
    print(f'\n--- {label}: what diverges first ({len(hits)} divergent games) ---')
    by_field = collections.Counter(h['field'] for h in hits)
    earliest = {}
    for h in hits:
        earliest[h['field']] = min(earliest.get(h['field'], 1 << 30), h['tick'])
    print(f'{"field":38s} {"games":>7s} {"share":>7s} {"earliest":>9s} {"median tick":>12s}')
    ticks_by_field = collections.defaultdict(list)
    for h in hits:
        ticks_by_field[h['field']].append(h['tick'])
    for field, count in by_field.most_common(top):
        series = sorted(ticks_by_field[field])
        print(f'{field:38s} {count:7d} {count / len(hits):6.1%} {earliest[field]:9d} '
              f'{series[len(series) // 2]:12d}')
    print(f'\n{"subsystem":24s} {"games":>7s} {"share":>7s}')
    by_sub = collections.Counter(subsystem(h['field']) for h in hits)
    for name, count in by_sub.most_common():
        print(f'{name:24s} {count:7d} {count / len(hits):6.1%}')
    kinds = collections.Counter(h['kind'] for h in hits if h.get('kind', -1) >= 0)
    if kinds:
        print('entity kinds blamed: ' + ', '.join(f'{k}:{v}' for k, v in kinds.most_common(8)))


def by_map(games, key, label, top=12):
    """Divergence-by-300-ticks rate per map, with the exposure each map got."""
    stat = collections.defaultdict(lambda: [0, 0])
    for game in games:
        hit = game.get(key)
        # Only games observed to 300 ticks, or that diverged before then, can
        # answer "did it diverge by 300"; the rest are censored and dropped.
        if hit and hit['tick'] <= 300:
            stat[game['map']][0] += 1; stat[game['map']][1] += 1
        elif game['ticks'] >= 300:
            stat[game['map']][1] += 1
    ranked = sorted(((k / n, n, m) for m, (k, n) in stat.items() if n >= 8), reverse=True)
    if not ranked:
        return
    print(f'\n--- {label}: worst maps (P[diverge by tick 300], maps with >=8 usable games) ---')
    print(f'{"map":>5s} {"rate":>7s} {"95% CI":>16s} {"n":>6s}')
    for rate, n, m in ranked[:top]:
        p, lo, hi = wilson(round(rate * n), n)
        print(f'{m:5d} {p:6.1%} [{lo:5.0%},{hi:5.0%}] {n:6d}')
    total_k = sum(k for k, _ in stat.values()); total_n = sum(n for _, n in stat.values())
    p, lo, hi = wilson(total_k, total_n)
    print(f'{"ALL":>5s} {p:6.1%} [{lo:5.0%},{hi:5.0%}] {total_n:6d}   '
          f'({len(stat)} maps with any usable game)')


def by_ability(games, key, label, top=16):
    """Exposure-based attribution: does having ability X in the RESOLVED
    loadout raise the chance of diverging by tick 300?

    This does not rely on the divergence naming an ability, which most
    non-entity fields cannot, so it catches an ability whose effect shows up
    in the player body rather than in a spawned entity.
    """
    stat = collections.defaultdict(lambda: [0, 0])
    base_k = base_n = 0
    for game in games:
        hit = game.get(key)
        if hit and hit['tick'] <= 300:
            diverged, usable = 1, 1
        elif game['ticks'] >= 300:
            diverged, usable = 0, 1
        else:
            continue
        base_k += diverged; base_n += usable
        present = {a for row in game.get('resolved', []) for a in row if a}
        for ability in present:
            stat[ability][0] += diverged; stat[ability][1] += usable
    if not base_n:
        return
    base = base_k / base_n
    print(f'\n--- {label}: abilities by P[diverge by tick 300] (baseline {base:.1%}) ---')
    print(f'{"id":>4s} {"ability":18s} {"rate":>7s} {"95% CI":>16s} {"n":>6s}  {"status":8s}')
    ranked = sorted(((k / n, n, a) for a, (k, n) in stat.items() if n >= 25), reverse=True)
    for rate, n, ability in ranked[:top]:
        p, lo, hi = wilson(round(rate * n), n)
        name = ABILITY_NAMES.get(ability, f'#{ability}')
        status = 'partial' if ability in PARTIAL else ''
        print(f'{ability:4d} {name:18s} {p:6.1%} [{lo:5.0%},{hi:5.0%}] {n:6d}  {status:8s}')


def coverage(games, runs):
    print('\n=== coverage: what was actually exercised ===')
    maps = collections.Counter(g['map'] for g in games)
    map_count = max((r.get('map_count', 124) for r in runs), default=124)
    reached = sorted(maps)
    missing = [m for m in range(map_count) if m not in maps]
    print(f'maps reached {len(reached)}/{map_count}; unreached: {missing if missing else "none"}')
    print(f'  games per map: min {min(maps.values())} median '
          f'{sorted(maps.values())[len(maps) // 2]} max {max(maps.values())}')

    equipped = collections.Counter()
    activated = collections.Counter()
    activation_total = collections.Counter()
    entity_seen = collections.Counter()
    for game in games:
        for p, row in enumerate(game.get('resolved', [])):
            for s, ability in enumerate(row):
                if not ability:
                    continue
                equipped[ability] += 1
                n = game['activations'][p][s] if p < len(game['activations']) else 0
                if n:
                    activated[ability] += 1
                    activation_total[ability] += n
        mask = game.get('ability_entities', 0)
        for ability in range(39):
            if mask >> ability & 1:
                entity_seen[ability] += 1
    print(f'\n{"id":>4s} {"ability":18s} {"equipped":>9s} {"games fired":>12s} '
          f'{"activations":>12s} {"games w/ entity":>16s}  {"status":8s}')
    unreached_equip, unreached_fire = [], []
    for ability in range(1, 39):
        e = equipped.get(ability, 0)
        a = activated.get(ability, 0)
        if not e:
            unreached_equip.append(ability)
        elif not a and not entity_seen.get(ability, 0):
            unreached_fire.append(ability)
        print(f'{ability:4d} {ABILITY_NAMES.get(ability, "?"):18s} {e:9d} {a:12d} '
              f'{activation_total.get(ability, 0):12d} {entity_seen.get(ability, 0):16d}  '
              f'{"partial" if ability in PARTIAL else "":8s}')
    print(f'\nabilities never equipped: {unreached_equip or "none"}')
    print(f'abilities equipped but never observed firing or spawning: {unreached_fire or "none"}')
    cells = collections.Counter(g['cell'] for g in games)
    print('\ncells: ' + ', '.join(f'{k}={v}' for k, v in sorted(cells.items())))
    ends = collections.Counter(g['end'] for g in games)
    print(f'episode ends: natural/limit {ends.get(0,0)}, terminal {ends.get(1,0)}, '
          f'backends ended at different ticks {ends.get(2,0)}')


def end_tick_agreement(games):
    """Do the three backends agree on when the round is over? That is the
    coarsest possible divergence and the one a training run feels first."""
    same_q32 = same_f64 = both = n = 0
    for game in games:
        o, q, f = game['end_ticks']
        if not o:
            continue
        n += 1
        same_q32 += o == q
        same_f64 += o == f
        both += o == q == f
    if not n:
        return
    print('\n=== round length agreement (games where the oracle ended) ===')
    for label, k in (('q32 ends on the oracle tick', same_q32), ('f64 ends on the oracle tick', same_f64),
                     ('all three agree', both)):
        p, lo, hi = wilson(k, n)
        print(f'{label:34s} {p:6.2%} [{lo:5.1%},{hi:5.1%}]  ({k}/{n})')


def hierarchy_cost(games):
    """How much of the all-fields curve is the inactive-prefab hierarchy label
    the default ignore list removes?"""
    blamed = sum(1 for g in games
                 if g.get('q32_all_any') and g['q32_all_any']['field'] == 'entity[].hierarchy')
    earlier = sum(1 for g in games
                  if g.get('q32_all_any') and (not g.get('q32_all_anyx')
                       or g['q32_all_any']['tick'] < g['q32_all_anyx']['tick']))
    p, lo, hi = wilson(earlier, len(games))
    print('\n=== cost of the entity[].hierarchy exclusion (q32, all fields) ===')
    print(f'games whose FIRST divergence is entity[].hierarchy: {blamed}')
    print(f'games where excluding it moves the first divergence later: '
          f'{earlier}/{len(games)} = {p:.1%} [{lo:.1%},{hi:.1%}]')


def main(argv):
    paths = [a for a in argv[1:] if not a.startswith('--')]
    if not paths:
        print(__doc__); return 2
    try:
        catalogue = json.loads((Path(__file__).resolve().parents[3] / 'data/rollout.json').read_text())
        ABILITY_NAMES.update({int(r['id']): r['name'] for r in catalogue['abilities']})
    except Exception as error:                      # noqa: BLE001 - reporting only
        print(f'(ability names unavailable: {error})')
    runs, games = load(paths)
    if not games:
        print('no games found'); return 2
    print(f'=== divergence rate: {len(games):,} games, '
          f'{sum(g["ticks"] for g in games):,} compared ticks ===')
    for run in runs[:1]:
        print(f'oracle  {run["oracle"]}\n  hash {run["oracle_hash"]}')
        print(f'rust    {run["q32"]}\n  hash {run["rust_hash"]}')
        print(f'tolerance {run["tol"]} raw units ({run["tol"] * 2.3283064365386963e-10:.3g} world units), '
              f'ignore list "{run.get("ignore")}"')

    by_cell = collections.defaultdict(list)
    for game in games:
        by_cell[game['cell']].append(game)
    random_cells = [g for g in games if g['cell'] == 'random']
    bot_cells = [g for g in games if g['cell'] != 'random']

    print('\n\n############ SURVIVAL CURVES ############')
    curve_table(games, 'ALL CELLS POOLED')
    if random_cells:
        curve_table(random_cells, 'RANDOM ACTIONS')
    if bot_cells:
        curve_table(bot_cells, 'SCRIPTED PLAY (all bot cells)')

    print('\n\n############ ATTRIBUTION ############')
    for key, label in (('q32_all_anyx', 'q32 vs oracle, all fields'),
                       ('q32_visible_anyx', 'q32 vs oracle, policy-visible'),
                       ('f64_all_discx', 'f64 vs oracle, all fields, DISCRETE only'),
                       ('f64_visible_discx', 'f64 vs oracle, policy-visible, DISCRETE only')):
        attribution(games, key, label)
    by_map(games, 'q32_all_anyx', 'q32 vs oracle, all fields')
    by_ability(games, 'q32_all_anyx', 'q32 vs oracle, all fields')
    by_ability(games, 'q32_visible_anyx', 'q32 vs oracle, policy-visible')

    print('\n\n############ CROSS-CHECKS ############')
    end_tick_agreement(games)
    hierarchy_cost(games)

    print('\n\n############ COVERAGE ############')
    coverage(games, runs)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
