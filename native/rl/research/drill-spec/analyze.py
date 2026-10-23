"""Summarise drill_look conditions: score, drill usage, episodes, kills/deaths, emergence, what the row shows.

    python3 analyze.py LOOKDIR [LOOKDIR ...] [--json out.json]
CIs: score by a pair bootstrap (2000 draws, the two side-swapped games of a seed pair resampled together);
proportions Wilson; per-episode medians bootstrap over games.
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

DT = 4 / 60.0   # one decision


def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def boot(values, groups, stat=np.mean, n=2000, seed=0):
    values = np.asarray(values, float); groups = np.asarray(groups)
    if len(values) == 0: return (float('nan'),) * 3
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug)); sel = np.concatenate([idx[g] for g in pick])
        out.append(stat(values[sel]))
    return float(stat(values)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def episodes(decs):
    eps = []; cur = []
    for r in decs:
        if cur and (r['g'] != cur[-1]['g'] or r['d'] != cur[-1]['d'] + 1):
            eps.append(cur); cur = []
        cur.append(r)
    if cur: eps.append(cur)
    return eps


def runs(flags):
    out = []; n = 0
    for f in flags:
        if f: n += 1
        elif n: out.append(n); n = 0
    if n: out.append(n)
    return out


def analyze(d):
    d = Path(d)
    games = [json.loads(l) for l in open(d / 'games.jsonl')]
    decs = [json.loads(l) for l in open(d / 'decisions.jsonl')]
    gi = {g['g']: g for g in games}
    decs = [r for r in decs if r['g'] in gi]
    score = [1.0 if g['outcome'] == 'a' else 0.5 if g['outcome'] == 'draw' else 0.0 for g in games]
    pairs = [g['pair'] for g in games]
    res = dict(name=d.name, n=len(games), score=boot(score, pairs))
    res['sd_share'] = wilson(sum(g['sd'] for g in games), len(games))
    minutes = sum(g['decisions'] for g in games) * DT / 60
    res['minutes'] = minutes
    res['casts_per_min'] = sum(g['drill_casts'] for g in games) / minutes
    res['drill_time_share'] = len(decs) / max(1, sum(g['decisions'] for g in games))
    eps = episodes(decs)
    E = []
    for ep in eps:
        g = gi[ep[0]['g']]
        ug = [r['ug'] for r in ep]
        pts = np.array([(r['x'], r['y']) for r in ep])
        steps = np.hypot(*np.diff(pts, axis=0).T) if len(pts) > 1 else np.zeros(0)
        ugstep = np.array([ug[i] and ug[i + 1] for i in range(len(ug) - 1)], bool)
        d0, d1 = ep[0]['d'], ep[-1]['d']
        a_died = [x for x in g['deaths'] if x['victim'] == 'A' and d0 <= x['d'] <= d1 + 1]
        b_drill = [x for x in g['deaths'] if x['victim'] == 'B' and x['killer'] == 'A' and x['ability'] == 26 and d0 <= x['d'] <= d1 + 1]
        entries = sum(1 for i in range(1, len(ug)) if ug[i] and not ug[i - 1]) + (1 if ug and ug[0] else 0)
        emerges = [i for i in range(1, len(ug)) if ug[i - 1] and not ug[i]]
        em = []
        for i in emerges:
            r = ep[i]
            em.append(dict(dist=math.hypot(r['x'] - r['ox'], r['y'] - r['oy']) if r['oalive'] else None,
                           kill_within_1s=any(d0 + i <= x['d'] <= d0 + i + 15 for x in b_drill)))
        # approach: distance to the opponent at episode start vs end (both alive)
        dist0 = math.hypot(ep[0]['x'] - ep[0]['ox'], ep[0]['y'] - ep[0]['oy']) if ep[0]['oalive'] else None
        distmin = min((math.hypot(r['x'] - r['ox'], r['y'] - r['oy']) for r in ep if r['oalive']), default=None)
        E.append(dict(g=ep[0]['g'], pair=g['pair'], dur=len(ep) * DT, ug_dec=sum(ug), ug_s=sum(ug) * DT,
                      longest_ug_s=max(runs(ug), default=0) * DT, entries=entries, path=float(steps.sum()),
                      ug_path=float(steps[ugstep].sum()) if len(steps) else 0.0, a_died=bool(a_died),
                      a_died_unforced=any(x['killer'] in ('none', 'self') for x in a_died),
                      a_cause=[x['cause'] for x in a_died], kill=bool(b_drill), emerges=em, dist0=dist0, distmin=distmin,
                      sd=ep[0]['t'] >= 6960, kill_ug=bool(b_drill) and any(ug[-16:]),
                      # P2 (DRILL-SPEC Section 6): the dive's longest underground run before the kill >= 0.5 s
                      # (8 decisions = 0.53 s), per drill kill
                      kill_ug05=[max(runs(ug[:max(0, x['d'] - d0) + 1]), default=0) >= 8 for x in b_drill]))
    res['episodes'] = len(E)
    if E:
        grp = [e['pair'] for e in E]
        res['ep_dur_s'] = boot([e['dur'] for e in E], grp, np.median)
        res['ep_dur_mean_s'] = boot([e['dur'] for e in E], grp)
        res['ep_ug_share'] = wilson(sum(e['ug_dec'] > 0 for e in E), len(E))
        ugE = [e for e in E if e['ug_dec'] > 0]
        res['ug_ep_longest_ug_s'] = boot([e['longest_ug_s'] for e in ugE], [e['pair'] for e in ugE], np.median) if ugE else None
        res['ug_ep_ug_path'] = boot([e['ug_path'] for e in ugE], [e['pair'] for e in ugE], np.median) if ugE else None
        res['ug_ep_entries_mean'] = float(np.mean([e['entries'] for e in ugE])) if ugE else None
        res['ep_path_median'] = float(np.median([e['path'] for e in E]))
        res['ep_ug_s_p90'] = float(np.percentile([e['ug_s'] for e in E], 90))
        res['ep_ug_s_max'] = float(max(e['ug_s'] for e in E))
        res['ep_ug_ge_1s'] = wilson(sum(e['longest_ug_s'] >= 1.0 for e in E), len(E))
        res['ep_kill'] = wilson(sum(e['kill'] for e in E), len(E))
        res['ep_kill_after_ug'] = wilson(sum(e['kill_ug'] for e in E), len(E))
        k05 = [x for e in E for x in e['kill_ug05']]
        res['drill_kills_after_ug05'] = wilson(sum(k05), len(k05)) if k05 else None
        res['ep_death'] = wilson(sum(e['a_died'] for e in E), len(E))
        res['ep_death_unforced'] = wilson(sum(e['a_died_unforced'] for e in E), len(E))
        causes = defaultdict(int)
        for e in E:
            for c in e['a_cause']: causes[c] += 1
        res['death_causes_in_drill'] = dict(causes)
        em = [x for e in E for x in e['emerges'] if x['dist'] is not None]
        res['emerges'] = len(em)
        res['emerge_dist_median'] = float(np.median([x['dist'] for x in em])) if em else None
        res['emerge_within_6u'] = wilson(sum(x['dist'] < 6 for x in em), len(em)) if em else None
        res['emerge_kill_1s'] = wilson(sum(x['kill_within_1s'] for x in em), len(em)) if em else None
        app = [e['dist0'] - e['distmin'] for e in E if e['dist0'] is not None and e['distmin'] is not None]
        res['ep_closing_median_u'] = float(np.median(app)) if app else None
        res['ep_dist0_median'] = float(np.median([e['dist0'] for e in E if e['dist0'] is not None]))
        res['ep_sd_share'] = sum(e['sd'] for e in E) / len(E)
    # kills and deaths
    akills = [x for g in games for x in g['deaths'] if x['victim'] == 'B' and x['killer'] == 'A']
    res['a_kills'] = len(akills); res['a_drill_kills'] = sum(x['ability'] == 26 for x in akills)
    res['drill_kill_share'] = wilson(res['a_drill_kills'], len(akills)) if akills else None
    wins = [g for g in games if g['outcome'] == 'a']
    res['wins_by_drill'] = wilson(sum(any(x['victim'] == 'B' and x['killer'] == 'A' and x['ability'] == 26 and x['last'] for x in g['deaths']) for g in wins), len(wins)) if wins else None
    adeaths = [x for g in games for x in g['deaths'] if x['victim'] == 'A']
    res['a_deaths'] = len(adeaths); res['a_deaths_in_drill'] = sum(x['a_in_drill'] for x in adeaths)
    pre = [g for g in games]
    res['a_kills_per_min'] = len(akills) / minutes
    # what the row shows when truly underground
    U = [r for r in decs if r['ug']]
    res['ug_decisions'] = len(U)
    if U:
        res['ug_rays_all_zero'] = wilson(sum(r['rays0'] == 36 for r in U), len(U))
        res['ug_obs_sdf_neg'] = wilson(sum(r['obs_sdf'] < 0 for r in U), len(U))
        aheads = [r['ahead'] for r in U if r['ahead'] is not None and r['ahead'] >= 0]
        res['ug_ground_ahead_median'] = float(np.median(aheads)) if aheads else None
        res['ug_ground_ahead_p90'] = float(np.percentile(aheads, 90)) if aheads else None
        res['ug_body_rot_zero'] = wilson(sum(abs(r['body'][3]) < 1e-6 and abs(r['body'][4] - 1) < 1e-6 for r in U), len(U))
    res['_E'] = E
    return res


def main():
    p = argparse.ArgumentParser(); p.add_argument('dirs', nargs='+'); p.add_argument('--json', type=Path)
    a = p.parse_args()
    allres = []
    for d in a.dirs:
        r = analyze(d); allres.append(r)
        show = {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else (round(v, 3) if isinstance(v, float) else v))
                for k, v in r.items() if k != '_E'}
        print(json.dumps(show))
    if a.json:
        a.json.write_text(json.dumps([{k: v for k, v in r.items()} for r in allres], default=str, indent=0))


if __name__ == '__main__':
    main()
