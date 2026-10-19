"""Research arm results (train/research/arm.py): a table, training-seed variance and factorial effects.

    venv/bin/python eval/research_report.py [DIR_OR_RESULTS_JSON ...] [--endpoint anneal] [--sigma PROGRAM/GROUP] [--json]

Default input: every native/artifacts/rl/research/*/*/results.json. The primary metric is the
equal-weight mean score against the held-out panel, with a 95% interval from the per-net side-swapped
pair-clustered standard errors.

Seed variance, for each group of >= 2 replicate seeds and each endpoint: the observed sd across seeds
and s_seed = sqrt(max(0, sd_obs^2 - mean eval variance)). Its 95% interval applies the chi-square
interval with k-1 df to sd_obs^2 and subtracts the eval variance from both ends; the 80% upper bound
is what later designs should be sized from. Within-group contrasts (EMA - raw, anneal - raw,
anneal - EMA) are paired across seeds. A group whose arms name `compare = "<group>"` (or
"<program>/<group>" for a control trained by another program, e.g. "e0/obsfix") is tested against that
group on their common seeds (difference of seed means, Welch), and on the offense arms' pre-registered
kill-rate rule (eval/kill_rate.py: KH_PRE ratio and panel non-inferiority at --endpoint, Holm across the
groups sharing a control). Arms whose training
resumed mid-run are starred and listed as tainted; they stay in the statistics.

Factorial (arms with `factors`): at --endpoint, each main effect and two-factor interaction is
mean(y | +) - mean(y | -), aliased columns reported once. Its se is 2 sigma / sqrt(N) with sigma the
observed between-seed sd of --sigma's group (k-1 df); without --sigma, Lenth's pseudo standard error
over the effect columns (m/3 df). The EMA - raw contrast over all runs joins the main effects in the
Benjamini-Hochberg family at q 0.10.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

from scipy import stats

import kill_rate

ROOT = Path(__file__).resolve().parents[1]
Q = 0.10


def load(inputs):
    files = []
    for x in inputs or [ROOT / 'native/artifacts/rl/research']:
        x = Path(x)
        files += [x] if x.is_file() else sorted(x.glob('results.json')) or sorted(x.glob('*/results.json')) \
            or sorted(x.glob('*/*/results.json'))
    return [json.loads(f.read_text()) for f in files]


def panel(r, endpoint):
    return r['endpoints'].get(endpoint, {}).get('panel')


def seed_variance(rows, endpoint):
    cells = [c for r in rows if (c := panel(r, endpoint))]
    k = len(cells)
    if k < 2:
        return None
    mean = sum(c['score'] for c in cells) / k
    var_obs = sum((c['score'] - mean) ** 2 for c in cells) / (k - 1)
    var_eval = sum(c['se'] ** 2 for c in cells) / k
    lo, hi, up80 = ((k - 1) * var_obs / stats.chi2.ppf(q, k - 1) for q in (0.975, 0.025, 0.2))
    root = lambda v: math.sqrt(max(0.0, v))
    return dict(endpoint=endpoint, k=k, mean=mean, sd_obs=math.sqrt(var_obs), se_eval=math.sqrt(var_eval),
                s_seed=root(var_obs - var_eval), s_seed_ci=[root(lo - var_eval), root(hi - var_eval)],
                s_seed_up80=root(up80 - var_eval))


def paired(rows, x, y):
    diffs = [panel(r, x)['score'] - panel(r, y)['score'] for r in rows if panel(r, x) and panel(r, y)]
    if len(diffs) < 2:
        return None
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs))
    p = 2 * stats.t.sf(abs(mean) / se, len(diffs) - 1) if se else 0.0
    return dict(effect=f'{x} - {y}', k=len(diffs), estimate=mean, se=se, p=p, df=len(diffs) - 1)


def compare(treated, control, endpoint):
    """Treatment group minus its control group on their common seeds: Welch over the between-seed sds
    (eval noise included); with one seed a side, the eval se alone (seed variance missing)."""
    seeds = {r['seed'] for r in treated} & {r['seed'] for r in control}
    sides = [[c for r in rows if r['seed'] in seeds and (c := panel(r, endpoint))] for rows in (treated, control)]
    if not all(sides):
        return None
    means = [sum(c['score'] for c in s) / len(s) for s in sides]
    if min(map(len, sides)) >= 2:
        v = [sum((c['score'] - m) ** 2 for c in s) / (len(s) - 1) / len(s) for s, m in zip(sides, means)]
        df = sum(v) ** 2 / sum(x ** 2 / (len(s) - 1) for x, s in zip(v, sides))
        basis = 'Welch over seeds'
    else:
        v = [sum(c['se'] ** 2 for c in s) / len(s) ** 2 for s in sides]
        df, basis = math.inf, 'eval se only: needs >= 2 seeds a side for seed variance'
    se, diff = math.sqrt(sum(v)), means[0] - means[1]
    t = stats.t.ppf(0.975, df)
    return dict(endpoint=endpoint, seeds=sorted(seeds), estimate=diff, se=se, ci=[diff - t * se, diff + t * se],
                p=2 * stats.t.sf(abs(diff) / se, df) if se else 0.0, basis=basis)


def bh(ps, q=Q):
    """Benjamini-Hochberg: which of ps are discoveries at false-discovery rate q."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    passed = max((rank for rank, i in enumerate(order, 1) if ps[i] <= rank / len(ps) * q), default=0)
    return [order.index(i) < passed for i in range(len(ps))]


def factorial(rows, endpoint, sigma):
    runs = [(r['factors'], panel(r, endpoint)['score']) for r in rows if r.get('factors') and panel(r, endpoint)]
    if len(runs) < 4:
        return None
    letters = sorted(runs[0][0])
    n = len(runs)
    effects, seen = [], {}
    for word in [(f,) for f in letters] + list(itertools.combinations(letters, 2)):
        column = tuple(math.prod(x[f] for f in word) for x, _ in runs)
        name = ''.join(word)
        if column in seen or tuple(-c for c in column) in seen:
            seen.get(column, seen.get(tuple(-c for c in column))).setdefault('aliases', []).append(name)
            continue
        estimate = 2 * sum(c * y for c, (_, y) in zip(column, runs)) / n
        seen[column] = row = dict(effect=name, estimate=estimate)
        effects.append(row)
    if sigma:
        se, df, source = 2 * sigma['sd_obs'] / math.sqrt(n), sigma['k'] - 1, f"between-seed sd of {sigma['group']}"
    else:
        s0 = 1.5 * stats.scoreatpercentile([abs(e['estimate']) for e in effects], 50)
        se = 1.5 * stats.scoreatpercentile([abs(e['estimate']) for e in effects if abs(e['estimate']) < 2.5 * s0], 50)
        df, source = len(effects) / 3, f'Lenth PSE over {len(effects)} columns'
    t = stats.t.ppf(0.975, df)
    for e in effects:
        e.update(se=se, df=df, ci=[e['estimate'] - t * se, e['estimate'] + t * se],
                 p=2 * stats.t.sf(abs(e['estimate']) / se, df) if se else 0.0)
    family = [e for e in effects if len(e['effect']) == 1]
    if ema := paired(rows, 'ema', 'raw'):
        t = stats.t.ppf(0.975, ema['df'])
        ema['ci'] = [ema['estimate'] - t * ema['se'], ema['estimate'] + t * ema['se']]
        family.append(ema)
    for e, passed in zip(family, bh([e['p'] for e in family])):
        e['bh'] = passed
    return dict(endpoint=endpoint, runs=n, se_source=source, effects=effects, ema=ema)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='*', type=Path)
    p.add_argument('--endpoint', default='anneal', help='the endpoint factorial effects are estimated at')
    p.add_argument('--sigma', help='PROGRAM/GROUP whose between-seed sd sets the factorial effect se (e.g. e0/l2)')
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    results = load(a.inputs)
    if not results:
        raise SystemExit('no results.json found')
    groups, programs = {}, {}
    for r in results:
        groups.setdefault((r['program'], r['group']), []).append(r)
        programs.setdefault(r['program'], []).append(r)
    endpoints = lambda rows: list(dict.fromkeys(e for r in rows for e in r['endpoints']))
    variance = {f'{prog}/{group}': [v for e in endpoints(rows) if (v := seed_variance(rows, e))]
                for (prog, group), rows in groups.items()}
    contrasts = {f'{prog}/{group}': [c for x, y in (('ema', 'raw'), ('anneal', 'raw'), ('anneal', 'ema'))
                                     if (c := paired(rows, x, y))] for (prog, group), rows in groups.items()}
    control_of = lambda prog, rows: tuple(rows[0]['compare'].split('/', 1)) if '/' in rows[0]['compare'] \
        else (prog, rows[0]['compare'])
    treated = {(prog, group): control_of(prog, rows) for (prog, group), rows in groups.items()
               if rows[0].get('compare') and control_of(prog, rows) in groups}
    versus = {f"{prog}/{group} vs {'/'.join(ctl)}": [c for e in endpoints(groups[(prog, group)])
                                                   if (c := compare(groups[(prog, group)], groups[ctl], e))]
              for (prog, group), ctl in treated.items()}
    offense = {}
    for ctl in dict.fromkeys(treated.values()):
        arms = {f'{prog}/{group}': groups[(prog, group)] for (prog, group), c in treated.items() if c == ctl}
        offense['/'.join(ctl)] = kill_rate.rule(arms, groups[ctl], a.endpoint)
    sigma = None
    if a.sigma:
        sigma = next((dict(v, group=a.sigma) for v in variance.get(a.sigma, []) if v['endpoint'] == a.endpoint), None)
        if not sigma:
            raise SystemExit(f'--sigma {a.sigma}: no replicate group with >= 2 seeds at {a.endpoint}')
    designs = {prog: d for prog, rows in programs.items() if (d := factorial(rows, a.endpoint, sigma))}
    if a.json:
        print(json.dumps(dict(arms=results, seed_variance=variance, contrasts=contrasts, compare=versus,
                              offense_rule=offense, factorial=designs), indent=1, default=float))
        return 0

    print(f"{'program':<8} {'arm':<17} {'seed':>5} {'endpoint':<9} {'panel':>6}  {'95% CI':<15} {'n':>6}  per opponent (score/n)")
    for r in results:
        for endpoint, row in r['endpoints'].items():
            c = row.get('panel')
            head = f"{c['score']:6.3f}  [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] {c['n']:>6}" if c else f"{'-':>6}  {'':<15} {'':>6}"
            each = '  '.join(f"{o}={v['score']:.3f}/{v['n']}" for o, v in row['opponents'].items())
            print(f"{r['program']:<8} {r['arm'] + ('*' if r.get('tainted') else ''):<17} {r['seed']:>5} {endpoint:<9} "
                  f"{head}  {each}")
    if tainted := [r for r in results if r.get('tainted')]:
        print('* TAINTED, resumed mid-run (beta schedule restarted; findings log): '
              + ', '.join(f"{r['program']}/{r['arm']} ({'+'.join(r['tainted'])})" for r in tainted))
    for name, rows in variance.items():
        if not rows:
            continue
        print(f'\n{name}: training-seed variance of the panel score')
        for v in rows:
            print(f"  {v['endpoint']:<9} k {v['k']}  mean {v['mean']:.3f}  sd across seeds {v['sd_obs']:.4f}  "
                  f"eval se {v['se_eval']:.4f}  s_seed {v['s_seed']:.4f} [{v['s_seed_ci'][0]:.4f}, {v['s_seed_ci'][1]:.4f}]"
                  f"  80% upper {v['s_seed_up80']:.4f}")
        for c in contrasts[name]:
            print(f"  {c['effect']:<15} {c['estimate']:+.4f} +/- {c['se']:.4f} (se, {c['k']} paired seeds)")
    for name, rows in versus.items():
        print(f"\n{name}: panel score difference on seeds {rows[0]['seeds']} ({rows[0]['basis']})")
        for c in rows:
            print(f"  {c['endpoint']:<9} {c['estimate']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]  p {c['p']:.3f}")
    for ctl, verdicts in offense.items():
        print(f'\noffense rule (eval/kill_rate.py) against {ctl}:')
        print(kill_rate.report(verdicts, ctl))
    for prog, d in designs.items():
        print(f"\n{prog}: factorial effects at {d['endpoint']} ({d['runs']} runs; se from {d['se_source']}; "
              f"BH q {Q} over the main effects and EMA - raw)")
        for e in d['effects'] + ([d['ema']] if d['ema'] else []):
            alias = f" (= {', '.join(e['aliases'])})" if e.get('aliases') else ''
            mark = '' if 'bh' not in e else ('  BH pass' if e['bh'] else '  BH fail')
            print(f"  {e['effect'] + alias:<20} {e['estimate']:+.4f} [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}]  p {e['p']:.3f}{mark}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
