"""The kill rate the offense arms are adopted on, and their pre-registered pass rule.

    venv/bin/python eval/kill_rate.py TREATED_DIR [...] --control CONTROL_DIR [...] [--endpoint anneal]
    venv/bin/python eval/kill_rate.py --factorial PROGRAM_DIR      (I1/I3/I7 as factors: their main effects)

Each DIR is an arm directory or results.json of train/research/arm.py; the panel's games come from
the all_combos --games-out files results.json names. In a factorial (PLAN-2026-10.md 2026-10-19
replan: I1 and I3 are factors of the offense factorial, "judged on its main effect"; off1 adds I7), a
factor is I1, I3 or I7 when its high level sets styles/offense_kill*.toml, styles/offense_time.toml or
styles/offense_sd_discount.toml; its runs at the high level are the treated group and the runs at the
low level the control (each run one unit, the other factors balanced across both sides). I7 takes the
same rule and, since Matthew's 2026-10-20 sign-off, is auto-adopted like I1 and I3.

Kill rate (OFFENSE-2026-10.md section 2 KH_PRE; "kills/min" in PLAN-2026-10.md's 2026-10-19 sign-off):
  per game, seat A = the arm's net:
    kill      1 if B's death is credited to A: B's telemetry deaths_opponent (`credit='game'`), or with
              `credit='cast'` the v4.2 count all_combos records as `credited` (eval/kill_credit.py,
              which adds Missile blasts and cast Spikes), only where every compared row carries it
    pre       the game's last tick is before sudden death (episode tick SUDDEN = 6960, level tick 7200)
    exposure  min(ticks, SUDDEN) / 3600: minutes both were alive before sudden death (one life: the
              first death ends the round)
  per seed:   KH_PRE = sum(kill * pre) / sum(exposure) over all panel games of the endpoint (the four
              held-out nets, equal games each), pooled ratio
  per group:  the mean of its seeds' KH_PRE
Rule (thresholds fixed in advance, unchanged here):
  1. ratio treated / control >= 1.20 with the seed-level bootstrap 95% interval above 1.0
     (seeds resampled within each group; Holm across the arms tested, one-sided 0.025)
  2. panel non-inferiority: the one-sided 95% lower bound of (treated - control) panel score, Welch over
     seed means, above -0.02
  vetoes: ckpm_b (credited kills of A per game minute) up more than 50%; unforced deaths up > 0.03,
  measured by the proxy env_loss (share of games A lost to an environment death: ring-out, fall or
  squish credited to nobody; offense_diag's 3u "near" split needs positions the panel rows lack);
  idle needs button rates (offense_diag rows) and is reported as unmeasured.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

SUDDEN = 6960
RATIO, MARGIN, VETO_CKPM_B, VETO_UNFORCED = 1.20, -0.02, 1.5, 0.03


def game(g, credit='game'):
    """Seat A's (kill before sudden death, exposure minutes, kill, killed, minutes) for one --games-out row."""
    if credit == 'cast':
        ka, kb = g['credited']
    else:
        ka, kb = g['deaths'][1].get('opponent', 0), g['deaths'][0].get('opponent', 0)
    t = g['ticks']
    return (ka if t < SUDDEN else 0), min(t, SUDDEN) / 3600, ka, kb, max(t, 1) / 3600


def points(g):
    return 1.0 if g['outcome'] == 'a' else 0.5 if g['outcome'] == 'draw' else 0.0


def summarize(rows, credit='game'):
    """Kill rates, plus the human-regime scores: pre_score over the games decided before sudden death,
    score_sd_draw with every game that reaches sudden death counted as a draw."""
    x = np.array([game(g, credit) for g in rows], float).reshape(-1, 5)
    s = x.sum(axis=0)
    pre = [points(g) for g in rows if g['ticks'] < SUDDEN]
    return dict(n=len(rows), kh_pre=float(s[0] / max(s[1], 1e-9)), ckpm=float(s[2] / max(s[4], 1e-9)),
                ckpm_b=float(s[3] / max(s[4], 1e-9)),
                score=float(np.mean([points(g) for g in rows])) if rows else None,
                pre_score=float(np.mean(pre)) if pre else None,
                score_sd_draw=float(np.mean([points(g) if g['ticks'] < SUDDEN else 0.5 for g in rows])) if rows else None,
                sd=float(np.mean([g['ticks'] >= SUDDEN for g in rows])) if rows else None,
                len_s=float(np.median([g['ticks'] / 60 for g in rows])) if rows else None,
                env_loss=float(np.mean([g['outcome'] == 'b' and g['deaths'][0].get('environment', 0) > 0 for g in rows]))
                if rows else None, credit=credit)


def load(spec):
    p = Path(spec)
    return json.loads((p / 'results.json' if p.is_dir() else p).read_text())


def panel_rows(result, endpoint):
    row = result['endpoints'].get(endpoint) or {}
    files = [row.get('opponents', {}).get(m, {}).get('games_out') for m in result['panel']]
    if not all(files) or not all(Path(f).exists() for f in files):
        return None
    return [json.loads(line) for f in files for line in Path(f).read_text().splitlines()]


def seed_stats(results, endpoint, credit):
    out = []
    for r in results:
        rows = panel_rows(r, endpoint)
        panel = r['endpoints'].get(endpoint, {}).get('panel')
        if rows is None or panel is None:
            continue
        if credit == 'cast' and any(g.get('credited') is None for g in rows):
            return None
        out.append(dict(seed=r['seed'], arm=r['arm'], panel=panel['score'], **summarize(rows, credit)))
    return out


def boot_ratio(t, c, key, n=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    t, c = np.array([s[key] for s in t]), np.array([s[key] for s in c])
    num = t[rng.integers(len(t), size=(n, len(t)))].mean(axis=1)
    den = c[rng.integers(len(c), size=(n, len(c)))].mean(axis=1)
    ratios = num / np.maximum(den, 1e-12)
    return float(t.mean() / max(c.mean(), 1e-12)), [float(np.quantile(ratios, .025)), float(np.quantile(ratios, .975))], \
        float(np.mean(ratios <= 1.0))


def noninferiority(t, c):
    a, b = [np.array([s['panel'] for s in x]) for x in (t, c)]
    diff = a.mean() - b.mean()
    if min(len(a), len(b)) < 2:
        return dict(diff=float(diff), lower95=None, basis='needs >= 2 seeds a side')
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se else math.inf
    return dict(diff=float(diff), se=se, lower95=float(diff - stats.t.ppf(0.95, df) * se), basis='Welch over seed means')


def rule(arms, control, endpoint='anneal', credit='game'):
    """{arm group: verdict} for treated groups {name: [results]} against one control group [results]."""
    ctrl = seed_stats(control, endpoint, credit)
    out = {}
    for name, results in arms.items():
        t = seed_stats(results, endpoint, credit)
        if not t or not ctrl:
            out[name] = dict(error=f'no complete panel games at {endpoint}' + (' with cast credit' if credit == 'cast' else ''))
            continue
        ratio, ci, p = boot_ratio(t, ctrl, 'kh_pre')
        ckpm_b = boot_ratio(t, ctrl, 'ckpm_b')[0]
        env_loss = float(np.mean([s['env_loss'] for s in t]) - np.mean([s['env_loss'] for s in ctrl]))
        ni = noninferiority(t, ctrl)
        out[name] = dict(endpoint=endpoint, credit=credit, seeds=[s['seed'] for s in t], control_seeds=[s['seed'] for s in ctrl],
                         kh_pre=float(np.mean([s['kh_pre'] for s in t])), kh_pre_control=float(np.mean([s['kh_pre'] for s in ctrl])),
                         ratio=ratio, ratio_ci=ci, p_one_sided=p, noninferiority=ni, ckpm_b_ratio=ckpm_b, env_loss_diff=env_loss,
                         len_s=float(np.mean([s['len_s'] for s in t])), len_s_control=float(np.mean([s['len_s'] for s in ctrl])),
                         sd=float(np.mean([s['sd'] for s in t])), sd_control=float(np.mean([s['sd'] for s in ctrl])),
                         per_seed=t, control_per_seed=ctrl)
    # Holm across the arms tested on the one-sided bootstrap p of ratio <= 1
    tested = sorted((v['p_one_sided'], k) for k, v in out.items() if 'ratio' in v)
    holm_ok = True
    for rank, (p, k) in enumerate(tested):
        holm_ok = holm_ok and p <= 0.025 / (len(tested) - rank)
        v = out[k]
        v['holm'] = holm_ok
        v['kill_rate_pass'] = v['ratio'] >= RATIO and v['ratio_ci'][0] > 1.0 and holm_ok
        v['strength_pass'] = v['noninferiority']['lower95'] is not None and v['noninferiority']['lower95'] > MARGIN
        v['vetoes'] = dict(ckpm_b_up_50pct=v['ckpm_b_ratio'] > VETO_CKPM_B, env_loss_up=v['env_loss_diff'] > VETO_UNFORCED,
                           idle='unmeasured')
        v['adopt'] = v['kill_rate_pass'] and v['strength_pass'] and not v['vetoes']['ckpm_b_up_50pct'] \
            and not v['vetoes']['env_loss_up']
    return out


OFFENSE_STYLES = {'I1': 'offense_kill', 'I3': 'offense_time', 'I7': 'offense_sd_discount'}
SIGN_OFF = ()   # factors whose passes wait for Matthew's sign-off; I7 left it on 2026-10-20 (PLAN)


def factorial_rule(results, endpoint='anneal', credit='game'):
    """{'I1 (factor D)': verdict, ...}: the rule on each offense factor's high vs low runs, Holm across them."""
    runs = [r for r in results if r.get('factors')]
    arms, controls = {}, {}
    for letter in (runs[0]['factors'] if runs else {}):
        hi = [r for r in runs if r['factors'][letter] > 0]
        lo = [r for r in runs if r['factors'][letter] < 0]
        for name, style in OFFENSE_STYLES.items():
            if hi and all(style in json.dumps(r['flags']) for r in hi) and not any(style in json.dumps(r['flags']) for r in lo):
                arms[f'{name} (factor {letter})'], controls[f'{name} (factor {letter})'] = hi, lo
    verdicts = {}
    for name in arms:   # each factor has its own control half; Holm needs them together
        verdicts.update(rule({name: arms[name]}, controls[name], endpoint, credit))
    tested = sorted((v['p_one_sided'], k) for k, v in verdicts.items() if 'ratio' in v)
    ok = True
    for rank, (p, k) in enumerate(tested):
        ok = ok and p <= 0.025 / (len(tested) - rank)
        v = verdicts[k]
        v['holm'] = ok
        v['kill_rate_pass'] = v['ratio'] >= RATIO and v['ratio_ci'][0] > 1.0 and ok
        passed = v['kill_rate_pass'] and v['strength_pass'] and not v['vetoes']['ckpm_b_up_50pct'] \
            and not v['vetoes']['env_loss_up']
        v['sign_off'] = k.split()[0] in SIGN_OFF
        v['adopt'] = passed and not v['sign_off']
        v['passed'] = passed
    return verdicts


def report(verdicts, control_name):
    lines = []
    for name, v in verdicts.items():
        if 'error' in v:
            lines.append(f'{name} vs {control_name}: {v["error"]}')
            continue
        ni = v['noninferiority']
        lower = f"{ni['lower95']:+.4f}" if ni['lower95'] is not None else 'n/a'
        lines.append(
            f"{name} vs {control_name} at {v['endpoint']} ({v['credit']} credit, seeds {v['seeds']} vs {v['control_seeds']}):\n"
            f"  KH_PRE {v['kh_pre']:.4f} vs {v['kh_pre_control']:.4f}/min, ratio {v['ratio']:.3f} "
            f"[{v['ratio_ci'][0]:.3f}, {v['ratio_ci'][1]:.3f}] (need >= {RATIO} and > 1; Holm {'ok' if v['holm'] else 'fail'})"
            f" -> {'PASS' if v['kill_rate_pass'] else 'fail'}\n"
            f"  panel {ni['diff']:+.4f}, one-sided 95% lower {lower} (need > {MARGIN}) -> {'PASS' if v['strength_pass'] else 'fail'}\n"
            f"  ckpm_b ratio {v['ckpm_b_ratio']:.2f} (veto > {VETO_CKPM_B}); env_loss {v['env_loss_diff']:+.3f} (veto > "
            f"+{VETO_UNFORCED}); len_s {v['len_s']:.1f} vs {v['len_s_control']:.1f}; sd {v['sd']:.3f} vs {v['sd_control']:.3f}; "
            f"idle veto unmeasured\n"
            + (f"  PASSED, NOT ADOPTED: needs Matthew's sign-off" if v.get('sign_off') and v.get('passed') else
               f"  ADOPT: {'yes' if v['adopt'] else 'no'}" + (' (sign-off arm)' if v.get('sign_off') else '')))
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('treated', nargs='*')
    p.add_argument('--control', nargs='+')
    p.add_argument('--factorial', help='a program directory (or results.json files\' parent) holding a factorial')
    p.add_argument('--endpoint', default='anneal')
    p.add_argument('--credit', choices=('game', 'cast'), default='game')
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    if a.factorial:
        results = [json.loads(f.read_text()) for f in sorted(Path(a.factorial).glob('*/results.json'))]
        verdicts = factorial_rule(results, a.endpoint, a.credit)
        print(json.dumps(verdicts, indent=1) if a.json else report(verdicts, 'the low level') or 'no I1/I3 factor found')
        return
    if not (a.treated and a.control):
        p.error('give TREATED and --control, or --factorial')
    treated = {}
    for spec in a.treated:
        r = load(spec)
        treated.setdefault(f"{r['program']}/{r['group']}", []).append(r)
    control = [load(s) for s in a.control]
    verdicts = rule(treated, control, a.endpoint, a.credit)
    name = f"{control[0]['program']}/{control[0]['group']}"
    print(json.dumps(verdicts, indent=1) if a.json else report(verdicts, name))


if __name__ == '__main__':
    main()
