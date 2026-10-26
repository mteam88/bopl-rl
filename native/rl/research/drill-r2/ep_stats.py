"""Round-2 diagnosis on round 1's drill_look logs: what underground travel looks like, how safe it is, what it
earns, and what the round-1 style actually charged for it.

    ep_stats.py LOOKDIR [LOOKDIR ...] [--json out.json]

Per drill episode (a run of consecutive logged drill decisions of seat A): underground runs by the undelayed
SDF, speeds, deaths/kills, surfacings, and a replay of styles/drill_burrow.toml's two per-decision terms
(burrow_approach, drill_retreat) with the SDF as the underground bit (the sim's flag is not in the log; the
two disagree on ~6% of form rows at the surface, DRILL-SPEC 4.1).
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

DT = 4 / 60.0


def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def episodes(decs):
    eps, cur = [], []
    for r in decs:
        if cur and (r['g'] != cur[-1]['g'] or r['d'] != cur[-1]['d'] + 1):
            eps.append(cur); cur = []
        cur.append(r)
    if cur: eps.append(cur)
    return eps


def runs(flags):
    out, n, start = [], 0, 0
    for i, f in enumerate(flags):
        if f:
            if not n: start = i
            n += 1
        elif n:
            out.append((start, n)); n = 0
    if n: out.append((start, n))
    return out


def analyze(d):
    d = Path(d)
    games = [json.loads(l) for l in open(d / 'games.jsonl')]
    decs = [json.loads(l) for l in open(d / 'decisions.jsonl')]
    gi = {g['g']: g for g in games}
    eps = episodes(decs)
    out = dict(name=str(d), games=len(games), episodes=len(eps))
    # --- per-decision safety: deaths of A while in drill, by whether the last logged decision was underground
    dec_by_g = defaultdict(dict)
    for r in decs: dec_by_g[r['g']][r['d']] = r
    ug_dec = sum(r['ug'] for r in decs); air_dec = len(decs) - ug_dec
    all_dec = sum(g['decisions'] for g in games)
    deaths_ug = deaths_air = deaths_out = 0; deaths_ug_cause = defaultdict(int); deaths_air_cause = defaultdict(int)
    for g in games:
        for x in g['deaths']:
            if x['victim'] != 'A': continue
            r = dec_by_g[g['g']].get(x['d']) or dec_by_g[g['g']].get(x['d'] - 1)
            if x['a_in_drill'] and r is not None:
                if r['ug']: deaths_ug += 1; deaths_ug_cause[x['cause'] + '/' + x['killer']] += 1
                else: deaths_air += 1; deaths_air_cause[x['cause'] + '/' + x['killer']] += 1
            else: deaths_out += 1
    out['deaths'] = dict(in_drill_ug=deaths_ug, in_drill_air=deaths_air, out_of_drill=deaths_out,
                         per_1000dec_ug=1000 * deaths_ug / max(1, ug_dec), per_1000dec_drill_air=1000 * deaths_air / max(1, air_dec),
                         per_1000dec_out=1000 * deaths_out / max(1, all_dec - len(decs)),
                         ug_causes=dict(deaths_ug_cause), air_causes=dict(deaths_air_cause))
    out['decisions'] = dict(all=all_dec, drill=len(decs), ug=ug_dec)
    # --- per episode
    E = []
    for ep in eps:
        g = gi[ep[0]['g']]; d0, d1 = ep[0]['d'], ep[-1]['d']
        ug = [r['ug'] for r in ep]
        P = np.array([(r['x'], r['y']) for r in ep]); step = np.hypot(*np.diff(P, axis=0).T) if len(P) > 1 else np.zeros(0)
        dist = np.array([math.hypot(r['x'] - r['ox'], r['y'] - r['oy']) if r['oalive'] else np.nan for r in ep])
        kills = [x for x in g['deaths'] if x['victim'] == 'B' and x['killer'] == 'A' and x['ability'] == 26 and d0 <= x['d'] <= d1 + 1]
        died = [x for x in g['deaths'] if x['victim'] == 'A' and d0 <= x['d'] <= d1 + 1]
        rr = runs(ug)
        longest = max((n for _, n in rr), default=0)
        # speeds: underground steps (both ends ug) vs in-form air steps
        ugs = np.array([ug[i] and ug[i + 1] for i in range(len(ug) - 1)], bool)
        sp_ug = step[ugs] / DT if ugs.any() else np.zeros(0); sp_air = step[~ugs] / DT if (~ugs).any() else np.zeros(0)
        # round-1 style replay: closer = prev(min(dist,30)) - min(dist,30); approach while ug&prev(ug), retreat on
        # off-underground decisions of a form that has been under (since(ug) < since_form_entry)
        dm = np.minimum(np.nan_to_num(dist, nan=30.0), 30.0)
        appr = retr = 0.0; retr_air_after = 0.0; been_ug = False; sd = ep[0]['t'] >= 6960
        for i in range(1, len(ep)):
            if not ep[i]['oalive'] or sd: continue
            closer = dm[i - 1] - dm[i]
            if ug[i - 1]: been_ug = True
            if ug[i] and ug[i - 1]: appr += 0.006 * min(closer, 2.0); been_ug = True
            elif been_ug: retr += 0.006 * min(closer, 0.0)
        # surfacings
        surf = []
        for (s, n) in rr:
            i = s + n
            if i < len(ep):
                r = ep[i]
                surf.append(dict(run=n, dist=dist[i], kill_1s=any(d0 + i <= x['d'] <= d0 + i + 15 for x in kills),
                                 died_1s=any(d0 + i <= x['d'] <= d0 + i + 15 for x in died),
                                 kill_any=bool(kills) and any(x['d'] >= d0 + i for x in kills),
                                 dist_entry=dist[s - 1] if s > 0 else np.nan, speed_out=(step[i - 1] / DT) if i - 1 < len(step) else np.nan))
        E.append(dict(g=ep[0]['g'], pair=g['pair'], n=len(ep), ug_n=sum(ug), longest=longest, runs=len(rr), kill=bool(kills), died=bool(died),
                      died_unforced=any(x['killer'] in ('none', 'self') for x in died), appr=appr, retr=retr, been_ug=any(ug), sd=sd,
                      sp_ug=sp_ug, sp_air=sp_air, surf=surf, min_sdf=min(r['sdf'] for r in ep), dist0=dist[0], distmin=np.nanmin(dist) if np.isfinite(dist).any() else np.nan,
                      kill_ug05=[max((n for s, n in runs(ug[:x['d'] - d0 + 1])), default=0) >= 8 for x in kills]))
    out['ep'] = dict(n=len(E), went_ug=wilson(sum(e['been_ug'] for e in E), len(E)), ge_1s=wilson(sum(e['longest'] >= 15 for e in E), len(E)),
                     ge_05s=wilson(sum(e['longest'] >= 8 for e in E), len(E)), kill=wilson(sum(e['kill'] for e in E), len(E)),
                     died=wilson(sum(e['died'] for e in E), len(E)), runs_per_ug_ep=float(np.mean([e['runs'] for e in E if e['been_ug']])))
    k05 = [x for e in E for x in e['kill_ug05']]; out['ep']['kills_after_ug05'] = wilson(sum(k05), len(k05))
    # speed
    su = np.concatenate([e['sp_ug'] for e in E]) if E else np.zeros(0); sa = np.concatenate([e['sp_air'] for e in E]) if E else np.zeros(0)
    out['speed'] = dict(ug_median=float(np.median(su)), ug_p90=float(np.percentile(su, 90)), ug_max=float(su.max()) if len(su) else 0,
                        air_median=float(np.median(sa)), air_p90=float(np.percentile(sa, 90)))
    # speed vs run position: speed at the k-th underground decision of a run
    byk = defaultdict(list)
    for ep in eps:
        ug = [r['ug'] for r in ep]; P = np.array([(r['x'], r['y']) for r in ep])
        for s, n in runs(ug):
            for k in range(1, n):
                byk[min(k, 30)].append(math.hypot(*(P[s + k] - P[s + k - 1])) / DT)
    out['speed_by_run_decision'] = {k: (float(np.median(v)), len(v)) for k, v in sorted(byk.items()) if len(v) >= 10}
    # outcome by longest-run bucket
    buckets = [(0, 0, 'never_ug'), (1, 7, '<0.5s'), (8, 14, '0.5-1s'), (15, 10 ** 6, '>=1s')]
    out['by_longest'] = {}
    for lo, hi, name in buckets:
        sel = [e for e in E if lo <= e['longest'] <= hi]
        if not sel: continue
        out['by_longest'][name] = dict(n=len(sel), kill=wilson(sum(e['kill'] for e in sel), len(sel)), died=wilson(sum(e['died'] for e in sel), len(sel)),
                                       died_unforced=wilson(sum(e['died_unforced'] for e in sel), len(sel)),
                                       style_appr=float(np.mean([e['appr'] for e in sel])), style_retr=float(np.mean([e['retr'] for e in sel])),
                                       style_net=float(np.mean([e['appr'] + e['retr'] for e in sel])), dist0=float(np.nanmedian([e['dist0'] for e in sel])),
                                       distmin=float(np.nanmedian([e['distmin'] for e in sel])), min_sdf=float(np.median([e['min_sdf'] for e in sel])))
    # surfacing outcomes by run length
    S = [s for e in E for s in e['surf']]
    out['surfacings'] = {}
    for lo, hi, name in [(1, 3, '1-3dec'), (4, 7, '4-7dec'), (8, 14, '8-14dec'), (15, 10 ** 6, '>=15dec')]:
        sel = [s for s in S if lo <= s['run'] <= hi]
        if not sel: continue
        dd = np.array([s['dist'] for s in sel]); fin = np.isfinite(dd)
        out['surfacings'][name] = dict(n=len(sel), within6=wilson(int((dd[fin] <= 6).sum()), int(fin.sum())), within10=wilson(int((dd[fin] <= 10).sum()), int(fin.sum())),
                                       kill_1s=wilson(sum(s['kill_1s'] for s in sel), len(sel)), died_1s=wilson(sum(s['died_1s'] for s in sel), len(sel)),
                                       closing=float(np.nanmedian([s['dist_entry'] - s['dist'] for s in sel])), speed_out=float(np.nanmedian([s['speed_out'] for s in sel])))
    # style totals, which decisions
    out['style_replay'] = dict(appr_sum=float(sum(e['appr'] for e in E)), retr_sum=float(sum(e['retr'] for e in E)),
                               retr_sum_ug_eps=float(sum(e['retr'] for e in E if e['been_ug'])),
                               net_ug_eps_mean=float(np.mean([e['appr'] + e['retr'] for e in E if e['been_ug']])),
                               net_ug_eps_median=float(np.median([e['appr'] + e['retr'] for e in E if e['been_ug']])),
                               share_ug_eps_net_neg=float(np.mean([(e['appr'] + e['retr']) < -1e-6 for e in E if e['been_ug']])))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('dirs', nargs='+'); p.add_argument('--json', type=Path)
    a = p.parse_args()
    res = [analyze(d) for d in a.dirs]
    for r in res:
        print('==', r['name'])
        print(json.dumps({k: v for k, v in r.items() if k != 'name'}, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))
    if a.json: a.json.write_text(json.dumps(res, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))


if __name__ == "__main__":
    main()
