"""Does burrowing build speed, and do launches at a far opponent kill? Round 1's drill_look logs, per surfacing.

    launch_stats.py LOOKDIR [LOOKDIR ...] [--json out.json]

Per drill episode (one form): every underground run and the surfacing that ends it; the entry speed (the step
before the run), the speed of the last underground step and of the first air step (finite differences of the
player position over one decision, 4 ticks), the aim at exit (cos of the angle between the first air step and
the bearing to the opponent), the distance to the opponent at exit, the flight after it (air decisions until
re-entry or the form's end: path length, min distance to the opponent, time to the drill kill if one follows
in this form). Underground = undelayed SDF < 0 (the sim's flag is not logged).
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ep_stats import episodes, runs  # noqa: E402

DT = 4 / 60.0
LAST = None   # (surfacings, kills) of the last analyze() call


def pct(v, q): return float(np.percentile(v, q)) if len(v) else float('nan')


def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def analyze(d):
    d = Path(d)
    games = [json.loads(l) for l in open(d / 'games.jsonl')]
    decs = [json.loads(l) for l in open(d / 'decisions.jsonl')]
    gi = {g['g']: g for g in games}
    S = []   # surfacings
    K = []   # drill kills: (dist at the form's last surfacing before the kill or None, ...)
    for ep in episodes(decs):
        g = gi[ep[0]['g']]; d0 = ep[0]['d']; n = len(ep)
        ug = [r['ug'] for r in ep]
        P = np.array([(r['x'], r['y']) for r in ep]); O = np.array([(r['ox'], r['oy']) for r in ep])
        alive = np.array([r['oalive'] for r in ep])
        V = np.diff(P, axis=0) / DT   # V[i] = velocity over the step i -> i+1
        dist = np.hypot(*(O - P).T)
        kills = sorted(x['d'] - d0 for x in g['deaths'] if x['victim'] == 'B' and x['killer'] == 'A' and x['ability'] == 26 and d0 <= x['d'] <= d0 + n)
        died = sorted(x['d'] - d0 for x in g['deaths'] if x['victim'] == 'A' and d0 <= x['d'] <= d0 + n)
        rr = runs(ug)
        for ri, (s, m) in enumerate(rr):
            i = s + m   # the surfacing decision (first air decision), if inside the episode
            if i >= n: continue
            v_entry = np.hypot(*V[s - 1]) if s >= 1 else np.nan
            v_last_ug = np.hypot(*V[i - 1])          # the step that leaves the ground
            v_air = np.hypot(*V[i]) if i < n - 1 else np.nan   # the first air step
            vx, vy = V[i] if i < n - 1 else V[i - 1]
            bx, by = O[i] - P[i]; bd = math.hypot(bx, by)
            sp = math.hypot(vx, vy)
            cos = (vx * bx + vy * by) / (sp * bd) if sp > 1e-6 and bd > 1e-6 else np.nan
            # what the style can see at the surfacing decision: the step into it (the last underground step)
            ux, uy = V[i - 1]; su = math.hypot(ux, uy)
            cos_ug = (ux * bx + uy * by) / (su * bd) if su > 1e-6 and bd > 1e-6 else np.nan
            # flight: air decisions from i until re-entry or the episode's end
            j = i
            while j < n and not ug[j]: j += 1
            flight = P[i:j + 1] if j < n else P[i:]
            path = float(np.hypot(*np.diff(flight, axis=0).T).sum()) if len(flight) > 1 else 0.0
            mind = float(np.nanmin(np.where(alive[i:j + 1 if j < n else n], dist[i:j + 1 if j < n else n], np.nan))) if alive[i:j + 1 if j < n else n].any() else np.nan
            kill_after = [k for k in kills if k >= i]
            kill_in_flight = [k for k in kill_after if k <= j + 1]
            death_after = [k for k in died if i <= k <= j + 1]
            S.append(dict(pair=g['pair'], run=m, first=(ri == 0), v_entry=v_entry, v_last_ug=v_last_ug, v_air=v_air, speed=sp, cos=cos, cos_ug=cos_ug,
                          dist=bd if alive[i] else np.nan, flight_dec=j - i, flight_path=path, min_dist=mind,
                          kill_form=bool(kill_after), kill_flight=bool(kill_in_flight),
                          t_kill=(kill_after[0] - i) * DT if kill_after else np.nan, died=bool(death_after), ep_runs=len(rr)))
        for k in kills:   # each drill kill: the last surfacing before it in this form, if any
            prior = [(s, m) for (s, m) in rr if s + m <= k]
            if prior:
                s, m = prior[-1]; i = s + m
                K.append(dict(pair=g['pair'], run=m, dist_surf=dist[i] if alive[i] else np.nan, t_after=(k - i) * DT,
                              ug_total=sum(ug[:k + 1]), ug_last3s=sum(ug[max(0, k - 44):k + 1]), ug_last6s=sum(ug[max(0, k - 89):k + 1]),
                              speed=float(np.hypot(*V[i])) if i < n - 1 else np.nan, airborne=not ug[min(k, n - 1)]))
            else:
                K.append(dict(pair=g['pair'], run=0, dist_surf=np.nan, t_after=np.nan, ug_total=sum(ug[:k + 1]), ug_last3s=sum(ug[max(0, k - 44):k + 1]),
                              ug_last6s=sum(ug[max(0, k - 89):k + 1]), speed=np.nan, airborne=not ug[min(k, n - 1)]))
    global LAST
    LAST = (S, K)
    out = dict(name=str(d), surfacings=len(S), drill_kills=len(K))
    # 1. exit speed vs run length
    out['speed_by_run'] = {}
    for lo, hi, name in [(1, 3, '1-3'), (4, 7, '4-7'), (8, 14, '8-14'), (15, 22, '15-22'), (23, 10 ** 6, '>=23')]:
        sel = [s for s in S if lo <= s['run'] <= hi]
        if not sel: continue
        out['speed_by_run'][name] = dict(n=len(sel), v_entry=float(np.nanmedian([s['v_entry'] for s in sel])),
                                         v_last_ug=float(np.nanmedian([s['v_last_ug'] for s in sel])), v_air=float(np.nanmedian([s['v_air'] for s in sel])),
                                         gain=float(np.nanmedian([s['v_last_ug'] - s['v_entry'] for s in sel])),
                                         v_air_p90=pct([s['v_air'] for s in sel if np.isfinite(s['v_air'])], 90),
                                         dist=float(np.nanmedian([s['dist'] for s in sel])), flight_path=float(np.median([s['flight_path'] for s in sel])),
                                         flight_s=float(np.median([s['flight_dec'] for s in sel])) * DT,
                                         kill_form=wilson(sum(s['kill_form'] for s in sel), len(sel))[0], died=wilson(sum(s['died'] for s in sel), len(sel))[0])
    # 2. kills: where did the last surfacing happen, how long after
    kk = [k for k in K if k['run'] > 0]
    ds = np.array([k['dist_surf'] for k in kk]); ta = np.array([k['t_after'] for k in kk])
    out['kills'] = dict(n=len(K), no_surfacing_before=len(K) - len(kk), airborne_at_kill=sum(k['airborne'] for k in K),
                        dist_surf_p=[pct(ds[np.isfinite(ds)], q) for q in (10, 25, 50, 75, 90)], share_dist_gt16=float(np.nanmean(ds > 16)),
                        share_dist_gt25=float(np.nanmean(ds > 25)),
                        t_after_p=[pct(ta, q) for q in (10, 25, 50, 75, 90)], share_t_gt1s=float(np.mean(ta > 1.0)), share_t_gt3s=float(np.mean(ta > 3.0)),
                        ug_run_ge8=sum(k['run'] >= 8 for k in kk),
                        window3s_loses=sum(1 for k in kk if k['run'] >= 8 and k['ug_last3s'] < 8), window6s_loses=sum(1 for k in kk if k['run'] >= 8 and k['ug_last6s'] < 8),
                        window3s_undercredits=sum(1 for k in kk if k['ug_last3s'] < min(k['ug_total'], 30)))
    # 3. launches: fast exits at a far opponent, by aim
    def launch_table(sel):
        if not sel: return None
        return dict(n=len(sel), kill_form=wilson(sum(s['kill_form'] for s in sel), len(sel)), kill_flight=wilson(sum(s['kill_flight'] for s in sel), len(sel))[0],
                    died=wilson(sum(s['died'] for s in sel), len(sel))[0], min_dist=float(np.nanmedian([s['min_dist'] for s in sel])),
                    reach6=float(np.nanmean([s['min_dist'] <= 6 for s in sel])), flight_path=float(np.median([s['flight_path'] for s in sel])),
                    t_kill=float(np.nanmedian([s['t_kill'] for s in sel if s['kill_form']])) if any(s['kill_form'] for s in sel) else None)
    far = [s for s in S if np.isfinite(s['dist']) and s['dist'] > 16 and np.isfinite(s['cos'])]
    near = [s for s in S if np.isfinite(s['dist']) and s['dist'] <= 16 and np.isfinite(s['cos'])]
    out['far_surfacings'] = dict(n=len(far), share=len(far) / max(1, len(S)))
    out['far_by_aim_speed'] = {}
    for an, alo in (('aimed<30deg', 0.866), ('30-60deg', 0.5), ('off', -1.0)):
        ahi = 1.01 if an == 'aimed<30deg' else 0.866 if an == '30-60deg' else 0.5
        for sn, slo, shi in (('fast>=30', 30, 1e9), ('20-30', 20, 30), ('slow<20', 0, 20)):
            sel = [s for s in far if alo <= s['cos'] < ahi and slo <= s['speed'] < shi]
            t = launch_table(sel)
            if t: out['far_by_aim_speed'][f'{an}|{sn}'] = t
    out['near_by_aim'] = {an: launch_table([s for s in near if alo <= s['cos'] < ahi]) for an, alo, ahi in (('aimed<30deg', 0.866, 1.01), ('off', -1.0, 0.866))}
    # 4. today's launch habit: first surfacings that are fast and aimed at an opponent > 16 u
    first = [s for s in S if s['first'] and np.isfinite(s['cos']) and np.isfinite(s['dist'])]
    out['launch_rate'] = dict(first_surfacings=len(first),
                              far_fast_aimed=wilson(sum(1 for s in first if s['dist'] > 16 and s['speed'] >= 30 and s['cos'] >= 0.866), len(first)),
                              far_aimed=wilson(sum(1 for s in first if s['dist'] > 16 and s['cos'] >= 0.866), len(first)),
                              far_fast_aimed_after_ge8=sum(1 for s in first if s['dist'] > 16 and s['speed'] >= 30 and s['cos'] >= 0.866 and s['run'] >= 8))
    # 5. flight length vs speed: how far does a launch fly, by exit speed
    out['flight_by_speed'] = {}
    for sn, slo, shi in (('<20', 0, 20), ('20-30', 20, 30), ('30-40', 30, 40), ('>=40', 40, 1e9)):
        sel = [s for s in S if slo <= s['speed'] < shi]
        if sel: out['flight_by_speed'][sn] = dict(n=len(sel), path=float(np.median([s['flight_path'] for s in sel])), path_p90=pct([s['flight_path'] for s in sel], 90),
                                                 flight_s=float(np.median([s['flight_dec'] for s in sel])) * DT, kill_flight=wilson(sum(s['kill_flight'] for s in sel), len(sel))[0])
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('dirs', nargs='+'); p.add_argument('--json', type=Path)
    a = p.parse_args()
    res = [analyze(d) for d in a.dirs]
    for r in res:
        print('==', r['name']); print(json.dumps({k: v for k, v in r.items() if k != 'name'}, indent=1, default=float))
    if a.json: a.json.write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
