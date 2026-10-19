"""Offense metrics from game_diag rows (sweeps/game-diag/eval/game_diag.py or offense_diag.py).

    python offense_metrics.py NAME=FILE[,FILE...] ...      [--json out.json]

Per game, seat A (the subject). Fixed in advance (OFFENSE-2026-10.md section 2):
  len_s        game length in seconds (episode ticks / 60)
  sd           decided (or undecided) at/after sudden death (header tick >= 7200)
  KH_PRE       PRIMARY: A's credited kills per minute of both-alive play before sudden death
               (ratio estimator: sum of kills / sum of exposure; Missile/Spike count once credited)
  ckpm         credited kills by A per minute of game, pooled ratio
  free_win     A won and the loser's death was uncredited with nothing of A's within 3u (env-unforced)
  apm          A ability-button decisions per minute (buttons[1:] fraction x 900)
  idle         A pressed abilities on < 0.5% of decisions, among games >= 750 decisions
  engage_s     seconds until bodies first within 12u (offense_diag rows only)
  first_hit_s  seconds to A's first threat (own entity within 3u of the living opponent), median, censored
  close_frac   fraction of both-alive decisions within 12u (offense_diag rows only)
all_combos.py --games-out rows (deaths = per-seat telemetry counts, no ticks or distances) give the
fields marked *: KH_PRE*, ckpm*, len_s*, sd*, score*, win_credited*; the end tick stands in for the death tick.
CIs: bootstrap over seed pairs (both seats of a deal are one unit), 2000 resamples; proportions Wilson.
"""
import json, math, sys
from collections import Counter, defaultdict
import numpy as np

SUDDEN, WARMUP, NEAR = 7200, 240, 3.0
RNG = np.random.default_rng(0)


def seat(g, who): return g['side'] if who == 'a' else 1 - g['side']
def pts(g): return 1.0 if g['outcome'] == 'a' else .5 if g['outcome'] == 'draw' else 0.


def loser_death(g):
    if g['outcome'] not in ('a', 'b'): return None
    loser = seat(g, 'b' if g['outcome'] == 'a' else 'a')
    ds = [d for d in g['deaths'] if d.get('victim') == loser]
    return ds[-1] if ds else None


def how(d):
    if d is None: return 'no-event'
    if d.get('actor_is') == 'opponent': return 'credited'
    if d.get('actor_is') == 'self': return 'self'
    return 'env-near' if min(d.get('dmin_opp_3s', 1e9), d.get('dmin_oppent_3s', 1e9)) < NEAR else 'env-unforced'


def end_tick(g):
    d = [x['tick'] for x in g['deaths']]
    return max(d) if d else g['ticks'] + WARMUP


def per_game_allcombos(g):
    a, b = g['deaths']
    mins = max(g['ticks'], 1) / 3600; t = g['ticks'] + WARMUP
    ka = int(g['outcome'] == 'a' and b.get('opponent', 0) > 0); kb = int(g['outcome'] == 'b' and a.get('opponent', 0) > 0)
    return dict(score=pts(g), len_s=g['ticks'] / 60, sd=float(t >= SUDDEN), _kpre=ka * (t < SUDDEN),
                _xpre=(min(t, SUDDEN) - WARMUP) / 3600, _k=ka, _kb=kb, _min=mins, win_credited=float(ka))


def per_game(g):
    if 'players' not in g:
        return per_game_allcombos(g)
    a = seat(g, 'a'); b = 1 - a
    mins = max(g['ticks'], 1) / 3600
    ld = loser_death(g)
    credited_a = sum(1 for d in g['deaths'] if d.get('victim') == b and d.get('actor_is') == 'opponent')
    credited_b = sum(1 for d in g['deaths'] if d.get('victim') == a and d.get('actor_is') == 'opponent')
    t = end_tick(g)
    kpre = sum(1 for d in g['deaths'] if d.get('victim') == b and d.get('actor_is') == 'opponent' and d['tick'] < SUDDEN)
    r = dict(score=pts(g), len_s=g['ticks'] / 60, sd=float(t >= SUDDEN),
             _kpre=kpre, _xpre=(min(t, SUDDEN) - WARMUP) / 3600, _k=credited_a, _kb=credited_b, _min=mins,
             win_credited=float(g['outcome'] == 'a' and how(ld) == 'credited'),
             free_win=float(g['outcome'] == 'a' and how(ld) == 'env-unforced'),
             free_loss=float(g['outcome'] == 'b' and how(ld) == 'env-unforced'),
             apm=sum(g['players'][a]['buttons'][1:]) * 900, apm_b=sum(g['players'][b]['buttons'][1:]) * 900,
             long=float(g['decisions'] >= 750),
             idle=float(g['decisions'] >= 750 and sum(g['players'][a]['buttons'][1:]) < 0.005))
    off = g.get('offense')
    if off:
        r['engage_s'] = off['engage_tick'][a] / 60 if off['engage_tick'][a] is not None else g['ticks'] / 60
        r['close_frac'] = off['close_frac'][a]
        r['presses_pm'] = off['presses'][a] / mins
        r['mean_dist'] = off['mean_dist'][a]
        tt = off.get('threat_tick', [None, None])[a]
        r['first_hit_s'] = tt / 60 if tt is not None else g['ticks'] / 60   # censored at game end
    return r


def boot(rows, stat, b=2000):
    units = defaultdict(list)
    for g in rows: units[(g['seed'], g.get('salt', 0), g.get('map'))].append(g)
    keys = list(units)
    est = stat(rows); vals = []
    for _ in range(b):
        pick = RNG.integers(len(keys), size=len(keys))
        vals.append(stat([g for i in pick for g in units[keys[i]]]))
    return est, float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def summarize(rows):
    for g in rows: g['_m'] = per_game(g)
    keys = list(rows[0]['_m'])
    ratio = lambda num, den: (lambda r: sum(g['_m'][num] for g in r) / max(sum(g['_m'][den] for g in r), 1e-9))
    out = {'n': len(rows), 'KH_PRE': boot(rows, ratio('_kpre', '_xpre')) + (len(rows),),
           'ckpm': boot(rows, ratio('_k', '_min')) + (len(rows),), 'ckpm_b': boot(rows, ratio('_kb', '_min')) + (len(rows),)}
    for k in keys:
        if k in ('long',) or k.startswith('_'): continue
        sub = [g for g in rows if k in g['_m'] and (k != 'idle' or g['_m']['long'])]
        if not sub: continue
        fn = (lambda r, k=k: float(np.median([g['_m'][k] for g in r]))) if k in ('len_s', 'engage_s', 'first_hit_s') else \
             (lambda r, k=k: float(np.mean([g['_m'][k] for g in r])))
        if k in ('apm', 'apm_b', 'presses_pm'):   # per-minute rates: pooled over game time
            fn = lambda r, k=k: sum(g['_m'][k] * g['_m']['_min'] for g in r) / sum(g['_m']['_min'] for g in r)
        out[k] = boot(sub, fn) + (len(sub),)
    dec = [g for g in rows if g['outcome'] != 'draw' and 'players' in g]
    out['loser_death'] = dict(Counter(how(loser_death(g)) for g in dec))
    return out


def main(argv):
    js = None
    if '--json' in argv:
        i = argv.index('--json'); js = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    res = {}
    for spec in argv:
        name, files = spec.split('=', 1)
        rows = [json.loads(l) for f in files.split(',') for l in open(f)]
        s = summarize(rows); res[name] = s
        print(f'\n== {name} (n {s["n"]})')
        for k, v in s.items():
            if isinstance(v, tuple): print(f'  {k:12s} {v[0]:8.3f} [{v[1]:.3f},{v[2]:.3f}] n {v[3]}')
        print('  loser death', s['loser_death'])
    if js: json.dump(res, open(js, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
