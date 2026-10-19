"""Failure-mode atlas: where and how a checkpoint loses, from all_combos --games-out rows.

    python eval/failure_atlas.py report GAMES.jsonl [...] [--confirm GAMES.jsonl ...] --out ATLAS.md [--json F]
    venv/bin/python eval/failure_atlas.py replay GAMES.jsonl --pick seed:side [...] --out DIR    # rust backend
    python3 eval/failure_atlas.py render DIR [--frames 12]                                       # needs matplotlib

`report` reads rows of league/all_combos.py --games-out (one file per opponent; the
opponent label is the file stem minus the prefix every input shares). It scores every
game from A's side (win 1, draw 0.5) and compares it with A's mean score against the
same opponent, so a pocket's residual is opponent-adjusted. Pockets are pre-game facts:
the map, the side, and each ability held by A's or the opponent's kit (and A-ability x
opponent-ability pairs with --pairs-min games). Each pocket reports its mean residual,
its excess loss (points below expectation, summed), and a 95% interval clustered on seed
pairs (the two side-swapped games share kits and map). p-values are normal, and q-values
are Benjamini-Hochberg over every pocket tested; Holm is printed beside them. Rows from
--confirm (independent seeds, e.g. sibling checkpoints of the same run) re-test the
discovered pockets out of sample.

Outcome facts (game length phase, death cause) are descriptive, not pockets: they are
compared as A's death rate by (cause, phase) minus the opponent's in the same games, a
paired difference with the same clustering. Causes, from the row's telemetry counts and
the sim's cause_of_death code (eval/readout.py deaths): squished (1), eaten (3), beamed
(4), opponent (credited), self (own projectile), fall (environment: water, the space
floor or the blast zone; the sim reports code 2 or 0, and replays of both are drownings),
uncredited (deaths_other).

`replay` re-plays chosen rows exactly (elo_ladder.play_games is deterministic in
seed, map, side and the participant names A/B all_combos uses) and keeps the native
replay (.rep, render with eval/render_video.py) and the physics visual (.json.gz, the
dashboard arena's format). It checks ticks and outcome against the row. `render` draws a
contact sheet per visual: evenly spaced frames plus the last two seconds before the
first death, and a trajectory strip (x, y, distance between bodies over time).
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUDDEN, WARMUP = 7200, 240  # header ticks; games-out `ticks` and frame `t` count from the end of warmup
PHASES = ((900, 'opening <15s'), (3600, 'early 15-60s'), (SUDDEN - WARMUP, 'late 60-116s'), (10 ** 9, 'sudden death'))
CAUSES = ('fall', 'opponent', 'self', 'squished', 'beamed', 'eaten', 'uncredited')
CODE = {1: 'squished', 3: 'eaten', 4: 'beamed'}
WATER_Y, SPACE_FLOOR = -11.3, -26.0  # board/web/arena.js BOUNDS (SceneBounds.cs)


def ability_names():
    rows = json.loads((ROOT / 'native/data/rollout.json').read_text())['abilities']
    return {int(row['id']): row['name'] for row in rows}


def map_names():
    data = json.loads((ROOT / 'native/data/rollout.json').read_text())['maps']
    heldout = set(json.loads((ROOT / 'native/data/map-sets.json').read_text()).get('heldout-13', []))
    out = {}
    for m in data:
        space = (m.get('session') or {}).get('level_type') == 2
        out[m['level']] = f"{m['level']} {m['scene']}" + (' space' if space else '') + (' heldout' if m['level'] in heldout else '')
    return out


def phase(ticks):
    return next(name for limit, name in PHASES if ticks < limit)


def cause(death):
    """One death record of a games-out row -> cause name, or None if that seat never died."""
    if death.get('code') is None and not any(death.get(k) for k in ('environment', 'opponent', 'self', 'other')):
        return None
    if death.get('code') in CODE:
        return CODE[death['code']]
    if death.get('self'):
        return 'self'
    if death.get('opponent'):
        return 'opponent'
    if death.get('environment'):
        return 'fall'
    return 'uncredited'


def load(paths):
    stems = [Path(p).name.removesuffix('.jsonl').removesuffix('.games') for p in paths]
    parts = [s.split('-') for s in stems]
    shared = 0
    while len(stems) > 1 and all(len(p) > shared + 1 and p[shared] == parts[0][shared] for p in parts):
        shared += 1
    rows = []
    for path, p in zip(paths, parts):
        label = '-'.join(p[shared:])
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            side = r['side']
            r['opp'] = label
            r['file'] = str(path)
            r['b_kit'] = r['kits'][3:] if side == 0 else r['kits'][:3]
            r['score'] = {'a': 1.0, 'draw': 0.5}.get(r['outcome'], 0.0)
            r['phase'] = phase(r['ticks'])
            r['a_cause'], r['b_cause'] = cause(r['deaths'][0]), cause(r['deaths'][1])
            r['cluster'] = (label, r['seed'])
            rows.append(r)
    expected = {o: np.mean([r['score'] for r in rows if r['opp'] == o]) for o in {r['opp'] for r in rows}}
    for r in rows:
        r['residual'] = r['score'] - expected[r['opp']]
    return rows


def cluster_mean(values, clusters):
    """Mean and its seed-pair-clustered standard error."""
    values = np.asarray(values, float)
    n = len(values)
    if n < 2:
        return float(values.mean()) if n else 0.0, float('inf')
    m = values.mean()
    sums = defaultdict(float)
    for v, c in zip(values, clusters):
        sums[c] += v - m
    k = len(sums)
    if k < 2:
        return float(m), float('inf')
    var = k / (k - 1) * sum(s * s for s in sums.values())
    return float(m), math.sqrt(var) / n


def p_normal(z):
    return math.erfc(abs(z) / math.sqrt(2))


def bh(pvalues):
    order = np.argsort(pvalues)
    m = len(pvalues)
    q = np.empty(m)
    running = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        running = min(running, pvalues[i] * m / rank)
        q[i] = running
    return q


def holm(pvalues):
    order = np.argsort(pvalues)
    m = len(pvalues)
    out = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, pvalues[i] * (m - rank)))
        out[i] = running
    return out


def pockets(rows, names, maps, pairs_min):
    """{key: (family, label, predicate)} over pre-game facts."""
    out = {}
    for m in sorted({r['map'] for r in rows}):
        out[f'map:{m}'] = ('map', maps.get(m, str(m)), lambda r, m=m: r['map'] == m)
    for s in (0, 1):
        out[f'side:{s}'] = ('side', f'side {s}', lambda r, s=s: r['side'] == s)
    abilities = sorted({a for r in rows for a in r['a_kit'] + r['b_kit']})
    for a in abilities:
        out[f'a:{a}'] = ('own ability', f'A holds {names.get(a, a)}', lambda r, a=a: a in r['a_kit'])
        out[f'b:{a}'] = ('opp ability', f'opp holds {names.get(a, a)}', lambda r, a=a: a in r['b_kit'])
    out['a:dup'] = ('own ability', 'A kit has a duplicate', lambda r: len(set(r['a_kit'])) < 3)
    if pairs_min:
        counts = Counter((x, y) for r in rows for x in set(r['a_kit']) for y in set(r['b_kit']))
        for (x, y), n in counts.items():
            if n >= pairs_min:
                out[f'ab:{x}:{y}'] = ('own x opp', f'A {names.get(x, x)} vs opp {names.get(y, y)}',
                                      lambda r, x=x, y=y: x in r['a_kit'] and y in r['b_kit'])
    return out


def test_pockets(rows, defs, min_games):
    tests = []
    for key, (family, label, pred) in defs.items():
        sub = [r for r in rows if pred(r)]
        if len(sub) < min_games:
            continue
        m, se = cluster_mean([r['residual'] for r in sub], [r['cluster'] for r in sub])
        z = m / se if se > 0 and math.isfinite(se) else 0.0
        tests.append(dict(key=key, family=family, label=label, n=len(sub), score=float(np.mean([r['score'] for r in sub])),
                          residual=m, ci=[m - 1.96 * se, m + 1.96 * se], excess_loss=-m * len(sub), z=z, p=p_normal(z),
                          loss_rate=sum(r['outcome'] == 'b' for r in sub) / len(sub)))
    if tests:
        ps = np.array([t['p'] for t in tests])
        for t, q, h in zip(tests, bh(ps), holm(ps)):
            t['q'], t['holm'] = float(q), float(h)
    return tests


def confirm(tests, rows, defs, top):
    """Re-test the top discovered weak pockets on independent rows: one-sided, Holm over the ones re-tested."""
    weak = [t for t in tests if t['residual'] < 0 and t['q'] < 0.05][:top]
    out = []
    for t in weak:
        sub = [r for r in rows if defs[t['key']][2](r)]
        if len(sub) < 10:
            continue
        m, se = cluster_mean([r['residual'] for r in sub], [r['cluster'] for r in sub])
        z = m / se if se > 0 and math.isfinite(se) else 0.0
        out.append(dict(key=t['key'], label=t['label'], n=len(sub), residual=m, ci=[m - 1.96 * se, m + 1.96 * se],
                        p_one=0.5 * math.erfc(-z / math.sqrt(2))))
    if out:
        for c, h in zip(out, holm(np.array([c['p_one'] for c in out]))):
            c['holm'] = float(h)
    return out


def paired_rates(rows):
    """Per (cause, phase): how often A lost that way vs how often the opponent did, in the same games.

    A is usually the stronger seat, so the difference is mostly negative; the ratio A/opp says
    where A's edge is thinnest (near or above 1). Ratio intervals by the delta method on the
    linearised d_i = a_i - R b_i, clustered on seed pairs."""
    out = []
    clusters = [r['cluster'] for r in rows]
    for c in CAUSES:
        for limit, ph in PHASES + ((None, 'all'),):
            def hit(r, seat):
                return float(r[seat] == c and (ph == 'all' or r['phase'] == ph) and r['outcome'] != ('a' if seat == 'a_cause' else 'b'))
            a = np.array([hit(r, 'a_cause') for r in rows])
            b = np.array([hit(r, 'b_cause') for r in rows])
            if a.sum() < 5 or b.sum() < 5:
                continue
            d, se = cluster_mean(a - b, clusters)
            ratio = a.mean() / b.mean()
            _, se_lin = cluster_mean(a - ratio * b, clusters)
            se_r = se_lin / b.mean()
            out.append(dict(cause=c, phase=ph, a=float(a.mean()), b=float(b.mean()), diff=d,
                            ci=[d - 1.96 * se, d + 1.96 * se], ratio=float(ratio),
                            ratio_ci=[ratio - 1.96 * se_r, ratio + 1.96 * se_r],
                            p_ratio=p_normal((ratio - 1) / se_r) if se_r > 0 else 1.0))
    for o, q in zip(out, bh(np.array([o['p_ratio'] for o in out]))):
        o['q_ratio'] = float(q)
    return out


def asymmetry(rows, names, min_games):
    """Per ability: residual when A holds it plus residual when the opponent holds it.

    A kit-power effect is symmetric (A gains with it what A loses against it), so the sum
    is ~0. A negative sum means A gains less from holding it than the opponents gain from
    holding it against A: A plays it worse, or defends against it worse, than they do."""
    out = []
    clusters = [r['cluster'] for r in rows]
    resid = np.array([r['residual'] for r in rows])
    for a in sorted({x for r in rows for x in r['a_kit'] + r['b_kit']}):
        ia = np.array([a in r['a_kit'] for r in rows]); ib = np.array([a in r['b_kit'] for r in rows])
        if ia.sum() < min_games or ib.sum() < min_games:
            continue
        ma, mb = resid[ia].mean(), resid[ib].mean()
        contrib = np.where(ia, (resid - ma) / ia.sum(), 0.0) + np.where(ib, (resid - mb) / ib.sum(), 0.0)
        sums = defaultdict(float)
        for v, c in zip(contrib, clusters):
            sums[c] += v
        k = len(sums)
        se = math.sqrt(k / (k - 1) * sum(s * s for s in sums.values()))
        asym = ma + mb
        out.append(dict(ability=names.get(a, str(a)), id=a, own=float(ma), opp=float(mb), n_own=int(ia.sum()), n_opp=int(ib.sum()),
                        asym=float(asym), ci=[asym - 1.96 * se, asym + 1.96 * se], p=p_normal(asym / se)))
    for o, q, h in zip(out, bh(np.array([o['p'] for o in out])), holm(np.array([o['p'] for o in out]))):
        o['q'], o['holm'] = float(q), float(h)
    return sorted(out, key=lambda o: o['asym'])


def seed_pairs(rows):
    """Side-swapped pairs: observed vs independent both-lost rate, and the within-pair score correlation."""
    by = defaultdict(dict)
    for r in rows:
        by[r['cluster']][r['side']] = r
    out = {}
    for opp in sorted({r['opp'] for r in rows}):
        pairs = [v for (o, _), v in by.items() if o == opp and len(v) == 2]
        if not pairs:
            continue
        l0 = np.mean([p[0]['outcome'] == 'b' for p in pairs])
        l1 = np.mean([p[1]['outcome'] == 'b' for p in pairs])
        both = np.mean([p[0]['outcome'] == 'b' and p[1]['outcome'] == 'b' for p in pairs])
        x = np.array([p[0]['score'] for p in pairs]); y = np.array([p[1]['score'] for p in pairs])
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
        out[opp] = dict(pairs=len(pairs), loss_side0=float(l0), loss_side1=float(l1), both_lost=float(both),
                        both_lost_if_independent=float(l0 * l1), corr=corr)
    pairs = [v for v in by.values() if len(v) == 2]
    lost_both = [v for v in pairs if v[0]['outcome'] == 'b' and v[1]['outcome'] == 'b']
    return out, pairs, lost_both


def kit_name(kit, names):
    return '/'.join(names.get(a, str(a)) for a in kit)


def fmt_ci(ci, digits=3):
    return f'[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]'


def report(args):
    names, maps = ability_names(), map_names()
    rows = load(args.games)
    defs = pockets(rows, names, maps, args.pairs_min)
    tests = test_pockets(rows, defs, args.min_games)
    tests.sort(key=lambda t: (t['q'] >= args.fdr, -t['excess_loss']))
    out = []
    w = out.append
    total = len(rows)
    score, se = cluster_mean([r['score'] for r in rows], [r['cluster'] for r in rows])
    w(f'# Failure atlas: {Path(rows[0]["a"]).name}\n')
    w(f'{total} games from {len(args.games)} files, {len({r["cluster"] for r in rows})} seed pairs. '
      f'A scores {score:.3f} [{score - 1.96 * se:.3f}, {score + 1.96 * se:.3f}] overall. '
      'Intervals are 95%, clustered on seed pairs.\n')
    w('## Opponents\n')
    w('| opponent | games | score | 95% CI | A losses | draws (timeout / both-dead / neither) | median ticks |')
    w('|---|---|---|---|---|---|---|')
    for opp in sorted({r['opp'] for r in rows}):
        sub = [r for r in rows if r['opp'] == opp]
        m, s = cluster_mean([r['score'] for r in sub], [r['cluster'] for r in sub])
        dk = Counter(r.get('draw_kind') for r in sub if r['outcome'] == 'draw')
        w(f"| {opp} | {len(sub)} | {m:.3f} | [{m - 1.96 * s:.3f}, {m + 1.96 * s:.3f}] | {sum(r['outcome'] == 'b' for r in sub)} | "
          f"{dk['timeout']} / {dk['both-dead']} / {dk['neither-won']} | {int(np.median([r['ticks'] for r in sub]))} |")
    w('')
    w('## Losses by game phase and cause\n')
    w('Share of all games that A lost, by when the game ended and how A died.\n')
    losses = [r for r in rows if r['outcome'] == 'b']
    w('| phase | games | A loss rate | ' + ' | '.join(CAUSES) + ' |')
    w('|---|---|---|' + '---|' * len(CAUSES))
    for _, ph in PHASES:
        sub = [r for r in rows if r['phase'] == ph]
        lost = [r for r in sub if r['outcome'] == 'b']
        cc = Counter(r['a_cause'] for r in lost)
        w(f"| {ph} | {len(sub)} | {len(lost) / max(1, len(sub)):.3f} | " + ' | '.join(str(cc[c]) for c in CAUSES) + ' |')
    cc = Counter(r['a_cause'] for r in losses)
    w(f"| all | {total} | {len(losses) / total:.3f} | " + ' | '.join(str(cc[c]) for c in CAUSES) + ' |')
    w('')
    w('## A vs opponent: decisive deaths by cause and phase\n')
    w("Per game: did the seat die this way and not win? A's rate and the opponent's in the same games. "
      "A is the stronger seat, so A/opp < 1 is normal; the cells nearest 1 are where A's edge is thinnest. "
      'q is BH over the table for ratio = 1.\n')
    w('| cause | phase | A | opp | A - opp | 95% CI | A/opp | 95% CI | q |')
    w('|---|---|---|---|---|---|---|---|---|')
    rates = paired_rates(rows)
    for o in sorted(rates, key=lambda o: -o['ratio']):
        w(f"| {o['cause']} | {o['phase']} | {o['a']:.4f} | {o['b']:.4f} | {o['diff']:+.4f} | {fmt_ci(o['ci'], 4)} | "
          f"{o['ratio']:.2f} | [{o['ratio_ci'][0]:.2f}, {o['ratio_ci'][1]:.2f}] | {o['q_ratio']:.2g} |")
    w('')
    w('## Abilities A plays worse than its opponents (asymmetry)\n')
    w('own = residual when A holds the ability, opp = residual when the opponent holds it; asym = own + opp. '
      'Kit power alone gives asym ~0 (Gold mushroom: big own gain, equal loss against it). '
      'asym < 0: A gets less out of the ability than the opponents get out of it against A.\n')
    w('| ability | own (n) | opp (n) | asym | 95% CI | p | q (BH) | Holm |')
    w('|---|---|---|---|---|---|---|---|')
    asym = asymmetry(rows, names, args.min_games)
    for o in asym:
        if o['q'] < 0.2:
            w(f"| {o['ability']} | {o['own']:+.3f} ({o['n_own']}) | {o['opp']:+.3f} ({o['n_opp']}) | {o['asym']:+.3f} | "
              f"{fmt_ci(o['ci'])} | {o['p']:.2g} | {o['q']:.2g} | {o['holm']:.2g} |")
    w('')
    w('## Paired seeds (same map and kits, sides swapped)\n')
    w('| opponent | pairs | A lost side 0 | side 1 | lost both | if independent | score corr |')
    w('|---|---|---|---|---|---|---|')
    stats, pairs, lost_both = seed_pairs(rows)
    for opp, s in stats.items():
        w(f"| {opp} | {s['pairs']} | {s['loss_side0']:.3f} | {s['loss_side1']:.3f} | {s['both_lost']:.3f} | "
          f"{s['both_lost_if_independent']:.3f} | {s['corr']:+.3f} |")
    w(f'\n{len(lost_both)} of {len(pairs)} seed pairs were lost on both sides.\n')
    w('## Loss pockets (pre-game facts, opponent-adjusted)\n')
    w(f'{len(tests)} pockets with >= {args.min_games} games tested. residual = pocket score minus A\'s score vs the '
      'same opponent; excess = points below expectation over the pocket. '
      f'Rows with BH q < {args.fdr}, most costly first, then the strongest pockets for contrast.\n')
    w('| pocket | family | games | score | residual | 95% CI | excess loss | p | q (BH) | Holm |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    sig = [t for t in tests if t['q'] < args.fdr]
    for t in [t for t in sig if t['residual'] < 0] + sorted([t for t in sig if t['residual'] > 0], key=lambda t: t['excess_loss'])[:8]:
        w(f"| {t['label']} | {t['family']} | {t['n']} | {t['score']:.3f} | {t['residual']:+.3f} | {fmt_ci(t['ci'])} | "
          f"{t['excess_loss']:+.1f} | {t['p']:.2g} | {t['q']:.2g} | {t['holm']:.2g} |")
    near = [t for t in tests if args.fdr <= t['q'] < 0.2 and t['residual'] < 0]
    if near:
        w(f"\nNear misses (q < 0.2): " + '; '.join(f"{t['label']} {t['residual']:+.3f} (n {t['n']}, q {t['q']:.2f})"
                                                for t in sorted(near, key=lambda t: -t['excess_loss'])[:12]) + '.')
    w('')
    confirmed = []
    if args.confirm:
        crow = load(args.confirm)
        confirmed = confirm(tests, crow, defs, args.confirm_top)
        w('## Out-of-sample confirmation\n')
        w(f'{len(crow)} games from {len(args.confirm)} files on independent seeds; the weak pockets above re-tested, '
          'one-sided, Holm over the rows.\n')
        w('| pocket | games | residual | 95% CI | p (one-sided) | Holm |')
        w('|---|---|---|---|---|---|')
        for c in confirmed:
            w(f"| {c['label']} | {c['n']} | {c['residual']:+.3f} | {fmt_ci(c['ci'])} | {c['p_one']:.2g} | {c['holm']:.2g} |")
        w('')
        casym = {o['id']: o for o in asymmetry(crow, names, args.min_games)}
        w('Asymmetry, same independent rows:\n')
        w('| ability | discovery asym | confirm asym | 95% CI | p (one-sided) |')
        w('|---|---|---|---|---|')
        for o in asym:
            c = casym.get(o['id'])
            if o['q'] < 0.2 and c:
                z = (c['asym'] / ((c['ci'][1] - c['ci'][0]) / 3.92)) * (1 if o['asym'] > 0 else -1)
                w(f"| {o['ability']} | {o['asym']:+.3f} | {c['asym']:+.3f} | {fmt_ci(c['ci'])} | {0.5 * math.erfc(z / math.sqrt(2)):.2g} |")
        w('')
    w('## What decides sudden-death games (descriptive)\n')
    sd = [dict(r) for r in rows if r['phase'] == 'sudden death']
    base = {o: np.mean([r['score'] for r in sd if r['opp'] == o]) for o in {r['opp'] for r in sd}}
    for r in sd:
        r['residual'] = r['score'] - base[r['opp']]
    sd_defs = {k: v for k, v in defs.items() if v[0] in ('own ability', 'opp ability')}
    sd_tests = sorted(test_pockets(sd, sd_defs, args.min_games), key=lambda t: t['residual'])
    w(f'{len(sd)} games reached sudden death (A {np.mean([r["score"] for r in sd]):.3f}); '
      f'{sum(r["outcome"] == "b" and r["a_cause"] == "fall" for r in sd)} of A\'s '
      f'{sum(r["outcome"] == "b" for r in sd)} losses there are falls. Phase is an outcome, so these pockets describe '
      'which kits survive the endgame, not causes. Rows with BH q < 0.05:\n')
    w('| pocket | games | score | residual | 95% CI | q (BH) |')
    w('|---|---|---|---|---|---|')
    for t in sd_tests:
        if t['q'] < 0.05:
            w(f"| {t['label']} | {t['n']} | {t['score']:.3f} | {t['residual']:+.3f} | {fmt_ci(t['ci'])} | {t['q']:.2g} |")
    w('')
    w('## Worst setups (lost on both sides)\n')
    w('| opponent | seed | map | A kit | opp kit | ticks (side 0, 1) | A death (side 0, 1) |')
    w('|---|---|---|---|---|---|---|')
    for v in sorted(lost_both, key=lambda v: v[0]['ticks'] + v[1]['ticks'])[:args.list]:
        r0, r1 = v[0], v[1]
        w(f"| {r0['opp']} | {r0['seed']} | {maps.get(r0['map'], r0['map'])} | {kit_name(r0['a_kit'], names)} | "
          f"{kit_name(r0['b_kit'], names)} | {r0['ticks']}, {r1['ticks']} | {r0['a_cause']}, {r1['a_cause']} |")
    w('')
    text = '\n'.join(out) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    if args.json:
        args.json.write_text(json.dumps(dict(pockets=tests, confirmed=confirmed, asymmetry=asym,
                                             paired_rates=rates, seed_pairs=stats), indent=1))
    return 0


# ---------------------------------------------------------------- replay ----
def replay(args):
    sys.path[:0] = [str(HERE), str(ROOT / 'league'), str(ROOT / 'native/tools')]
    import registry
    import games as league_games
    import elo_ladder
    from evaluation_protocol import DEFAULT_LOADOUT
    rows = [json.loads(l) for l in args.games.read_text().splitlines() if l.strip()]
    picks = [tuple(int(v) for v in p.split(':')) for p in args.pick]
    chosen = [r for r in rows if (r['seed'], r['side']) in picks]
    if len(chosen) != len(picks):
        raise SystemExit(f'found {len(chosen)} of {len(picks)} picks in {args.games}')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    league_games.configure(out / 'workers')
    from environment import Environment
    built = {}
    results = []
    for r in chosen:
        for name, path in (('A', r['a']), ('B', r['b'])):
            if built.get(name, (None,))[0] != path:
                built[name] = (path, elo_ladder.Model(name, Path(path), 'candidate'))
        task = dict(a='A', b='B', map=r['map'], seed=r['seed'], side=r['side'], kits=r['kits'])
        loadouts = dict(mode='set', fixed=list(DEFAULT_LOADOUT), pool=[], lineups=[list(r['kits'])],
                        set_file=None, no_dup=False, mirror_share=0.0)
        with Environment(seed=r['seed_base'], limit=registry.EPISODE_TICKS, delay=registry.DELAY, loadouts=loadouts) as env:
            env.configure_loadouts(loadouts)
            env.reset(task['seed'], task['map'])
            rng = np.random.default_rng((task['seed'], task['map'], task['side'],
                                         int(hashlib.sha256(b'AB').hexdigest()[:8], 16)))
            env.bot(0, 0); env.bot(1, 0)
            for name, side in (('A', task['side']), ('B', 1 - task['side'])):
                built[name][1].attach(env, side)
            actors = {task['side']: built['A'][1].start(task, task['side'], rng),
                      1 - task['side']: built['B'][1].start(task, 1 - task['side'], rng)}
            frames = [env.frame()]
            while not env.ended:
                observation = env.rows.copy(); mask = env.mask.copy()
                actions = np.stack([actors[0](observation, mask), actors[1](observation, mask)])
                env.step(actions)
                frames[-1]['actions'] = actions.tolist()
                frames.append(env.frame())
            final = frames[-1]
            timeout = bool(final['flags'] & 2)
            a_won = bool(final['players'][task['side']]['flags'] & 2) and not timeout
            b_won = bool(final['players'][1 - task['side']]['flags'] & 2) and not timeout
            outcome = 'draw' if timeout or a_won == b_won else ('a' if a_won else 'b')
            name = f"{args.tag + '-' if args.tag else ''}seed{r['seed']}-side{r['side']}-map{r['map']}-{outcome}"
            env.replay(out / (name + '.rep'))
        exact = final['t'] == r['ticks'] and outcome == r['outcome']
        result = dict(row=r, name=name, ticks=final['t'], outcome=outcome, exact=exact, lineup_ids=r['kits'])
        with gzip.open(out / (name + '.json.gz'), 'wt', compresslevel=4) as f:
            json.dump(dict(schema=1, step=0, task=dict(task, name=name), result=dict(result, side=r['side'], map=r['map'], seed=r['seed']),
                           frames=frames, rendering='physics state replay', tick_rate=60, decision_rate=15,
                           actions_timing='observation-before-action'), f, separators=(',', ':'))
        results.append(result)
        print(json.dumps(dict(name=name, ticks=final['t'], row_ticks=r['ticks'], outcome=outcome, row_outcome=r['outcome'],
                              exact=exact)), flush=True)
    with open(out / 'replays.jsonl', 'a') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    return 0 if all(r['exact'] for r in results) else 1


# ---------------------------------------------------------------- render ----
def first_death(frames, seat):
    return next((i for i, f in enumerate(frames) if not f['players'][seat]['flags'] & 1), None)


def draw_frame(ax, frame, a_seat, title, space=False):
    from matplotlib.patches import Ellipse, Polygon, Circle
    colors = {a_seat: '#f4d93f', 1 - a_seat: '#80c7e0'}
    floor = SPACE_FLOOR if space else WATER_Y
    ax.set_facecolor('#18263a')
    ax.axhspan(floor - 30, floor, color='#2a1a1a' if space else '#173a52')
    xs = [p['x'] for p in frame['players'] if p['flags'] & 1] or [0.0]
    ys = [p['y'] for p in frame['players'] if p['flags'] & 1] or [0.0]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(90.0, (max(xs) - min(xs)) + 30, 2 * ((max(ys) - min(ys)) + 20))
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 4, cy + span / 4)
    ids = {p['id']: i for i, p in enumerate(frame['players'])}
    for e in frame['entities']:
        owner = ids.get(e['owner'])
        if e['kind'] == 0 and owner is not None and math.hypot(frame['players'][owner]['x'] - e['x'], frame['players'][owner]['y'] - e['y']) < 2:
            continue
        color = '#ff8a3d' if e['flags'] & (1 << 11) else colors.get(owner, '#274c64' if e['layer'] == 4 else '#8b9e87')
        if e['kind'] in (1, 2):
            corners = [(e['x'] + a * e['rx'] + b * e['ux'], e['y'] + a * e['ry'] + b * e['uy']) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            ax.add_patch(Polygon(corners, closed=True, color=color, lw=max(0.5, e['radius'] * 2), joinstyle='round'))
        elif e['kind'] == 0:
            ax.add_patch(Circle((e['x'], e['y']), max(abs(e['radius']), 0.3), color=color))
        else:
            ax.plot([e['x'], e['x'] + e['rx']], [e['y'], e['y'] + e['ry']], color=color, lw=1.5)
    for i, p in enumerate(frame['players']):
        if p['flags'] & 1:
            ax.add_patch(Ellipse((p['x'], p['y']), 2.66 * p['scale'], 1.62 * p['scale'], color=colors[i], ec='k', lw=0.5))
            ax.annotate('A' if i == a_seat else 'B', (p['x'], p['y'] + 1.6 * p['scale']), color='w', fontsize=6, ha='center')
    ax.axhline(floor, color='#e0655a' if space else '#5fb8d9', lw=0.8)
    ax.set_title(title, fontsize=6, color='w')
    ax.set_xticks([]); ax.set_yticks([])


SLOTS = ('dash_activations', 'bow_activations', 'platform_activations')  # telemetry names are legacy: ability slots 1-3


def activations(frame, seat):
    return [frame['telemetry'][seat][k] for k in SLOTS]


def behaviour(frames, a):
    """Per-seat activations per minute, distance while both alive, share of time within 12u, A minus B height."""
    alive = [f for f in frames if f['players'][0]['flags'] & 1 and f['players'][1]['flags'] & 1]
    minutes = max(1, len(alive)) * 4 / 3600
    dist = [math.hypot(f['players'][0]['x'] - f['players'][1]['x'], f['players'][0]['y'] - f['players'][1]['y']) for f in alive]
    end = alive[-1] if alive else frames[0]

    def idle(seat):
        """Longest both-alive stretch (s) without an activation, and the first activation's time and distance."""
        times, first = [0], None
        for f, g, d in zip(alive, alive[1:], dist[1:]):
            if sum(activations(g, seat)) > sum(activations(f, seat)):
                times.append(g['t'])
                first = first or (round(g['t'] / 60, 1), round(d, 1))
        times.append(end['t'])
        return round(max(b - a for a, b in zip(times, times[1:])) / 60, 1), first
    a_idle, a_first = idle(a)
    b_idle, b_first = idle(1 - a)
    return dict(minutes_alive=round(minutes, 2), a_idle_s=a_idle, b_idle_s=b_idle, a_first=a_first, b_first=b_first,
                a_act_pm=round(sum(activations(end, a)) / minutes, 1), b_act_pm=round(sum(activations(end, 1 - a)) / minutes, 1),
                a_act=activations(end, a), b_act=activations(end, 1 - a),
                mean_dist=round(float(np.mean(dist)), 1) if dist else None,
                close_frac=round(float(np.mean([d < 12 for d in dist])), 3) if dist else None,
                a_above_b=round(float(np.mean([f['players'][a]['y'] - f['players'][1 - a]['y'] for f in alive])), 1) if alive else None)


def render(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ability_names()
    spaces = {m for m, label in map_names().items() if ' space' in label}
    files = sorted(args.dir.glob('*.json.gz')) if args.dir.is_dir() else [args.dir]
    summaries = []
    for path in files:
        with gzip.open(path, 'rt') as f:
            visual = json.load(f)
        frames, res = visual['frames'], visual['result']
        a = res['side']
        space = res['map'] in spaces
        kits = res['lineup_ids'][:3], res['lineup_ids'][3:]
        da, db = first_death(frames, a), first_death(frames, 1 - a)
        end = min(i for i in (da, db, len(frames) - 1) if i is not None)
        summary = dict(name=path.name.removesuffix('.json.gz'), outcome=res['outcome'], ticks=frames[-1]['t'],
                       a_kit=kit_name(kits[a], names), b_kit=kit_name(kits[1 - a], names),
                       a_death_s=None if da is None else round(frames[da]['t'] / 60, 2),
                       b_death_s=None if db is None else round(frames[db]['t'] / 60, 2), **behaviour(frames, a))
        summaries.append(summary)
        if args.no_png:
            print(json.dumps(summary), flush=True)
            continue
        start = 0
        even = list(np.linspace(start, end, args.frames // 2, dtype=int))
        tail = list(np.linspace(max(start, end - 30), end, args.frames - len(even), dtype=int))
        picks = sorted(set(even + tail))
        cols = 4
        rows_n = math.ceil(len(picks) / cols) + 1
        fig = plt.figure(figsize=(cols * 3.2, rows_n * 1.8), facecolor='#0e1723')
        for k, i in enumerate(picks):
            ax = fig.add_subplot(rows_n, cols, k + 1)
            f = frames[i]
            act = f.get('actions')
            pressed = ''
            if act:
                buttons = int(act[a][1]) if len(act[a]) > 1 else 0
                pressed = ' '.join(n for bit, n in ((1, 'J'), (2, '1'), (4, '2'), (8, '3')) if buttons & bit)
            draw_frame(ax, f, a, space=space, title=f"t={f['t'] / 60:.1f}s  A:{pressed}  used A{activations(f, a)} B{activations(f, 1 - a)}")
        ax = fig.add_subplot(rows_n, 1, rows_n)
        t = np.array([f['t'] / 60 for f in frames[:end + 1]])
        for seat, label, c in ((a, 'A', '#f4d93f'), (1 - a, 'B', '#80c7e0')):
            ax.plot(t, [f['players'][seat]['y'] for f in frames[:end + 1]], color=c, lw=0.8, label=f'{label} y')
            ax.plot(t, [f['players'][seat]['x'] / 4 for f in frames[:end + 1]], color=c, lw=0.6, ls=':', label=f'{label} x/4')
        ax.plot(t, [math.hypot(f['players'][0]['x'] - f['players'][1]['x'], f['players'][0]['y'] - f['players'][1]['y']) for f in frames[:end + 1]],
                color='#e0655a', lw=0.8, label='distance')
        ax.axhline(SPACE_FLOOR if space else WATER_Y, color='#5fb8d9', lw=0.5)
        ax.set_facecolor('#18263a'); ax.tick_params(colors='w', labelsize=6); ax.legend(fontsize=5, ncol=5, loc='upper left')
        died = ['-' if i is None else f"{frames[i]['t'] / 60:.1f}s" for i in (da, db)]
        fig.suptitle(f"{summary['name']}  A {summary['a_kit']}  vs  B {summary['b_kit']}  outcome {res['outcome']}  "
                     f"A died {died[0]}  B died {died[1]}  activations/min A {summary['a_act_pm']} B {summary['b_act_pm']}  "
                     f"mean distance {summary['mean_dist']}", color='w', fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        target = path.with_name(summary['name'] + '.png')
        fig.savefig(target, dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
        summary['png'] = str(target)
        print(json.dumps(summary), flush=True)
    if args.dir.is_dir():
        (args.dir / 'summary.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in summaries))
    if len(summaries) >= 10:
        rng = np.random.default_rng(0)
        print(f'{len(summaries)} games: A vs B, mean [bootstrap 95%] over games')
        for label, sub in (('all', summaries), ('A won', [s for s in summaries if s['outcome'] == 'a']),
                           ('A lost', [s for s in summaries if s['outcome'] == 'b'])):
            for key in ('act_pm', 'idle_s'):
                diff = np.array([s['a_' + key] - s['b_' + key] for s in sub])
                boot = [rng.choice(diff, diff.size).mean() for _ in range(2000)]
                print(f"  {label:<7} n {len(sub):>3}  {key:<7} A {np.mean([s['a_' + key] for s in sub]):6.1f}  "
                      f"B {np.mean([s['b_' + key] for s in sub]):6.1f}  A-B {diff.mean():+6.1f} "
                      f"[{np.quantile(boot, .025):+.1f}, {np.quantile(boot, .975):+.1f}]")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('report')
    r.add_argument('games', nargs='+', type=Path)
    r.add_argument('--confirm', nargs='*', type=Path, default=[], help='independent-seed rows to re-test the weak pockets on')
    r.add_argument('--out', type=Path)
    r.add_argument('--json', type=Path)
    r.add_argument('--min-games', type=int, default=40)
    r.add_argument('--pairs-min', type=int, default=0, help='also test A-ability x opp-ability pockets with this many games')
    r.add_argument('--fdr', type=float, default=0.05)
    r.add_argument('--confirm-top', type=int, default=20)
    r.add_argument('--list', type=int, default=20, help='rows in the worst-setups table')
    r.set_defaults(func=report)
    q = sub.add_parser('replay')
    q.add_argument('games', type=Path)
    q.add_argument('--pick', nargs='+', required=True, metavar='SEED:SIDE')
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--tag', default='')
    q.set_defaults(func=replay)
    d = sub.add_parser('render')
    d.add_argument('dir', type=Path)
    d.add_argument('--frames', type=int, default=12)
    d.add_argument('--no-png', action='store_true', help='only the per-game behaviour summary')
    d.set_defaults(func=render)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == '__main__':
    raise SystemExit(main())
