"""Turn a research program's results into decisions (PLAN-2026-10.md, REVIEW-2026-10-cursor.md, POWER-2026-10.md).

    venv/bin/python train/research/analyze.py e0 [--dir DIR] [--prior-sseed 0.015] [--json OUT]

--json also writes `decisions`: the pre-registered verdicts a queue script acts on (optimizer: `followup`, the
groups to rerun with `arm.py optimizer.toml --followup ...`; x2d-drill: `case`, `stop_optimizer`; drill: `showcase`,
`p1_pass`, `insanely_good`, `adopt_v43`; drill-r2: `showcase`, `p1_pass`, `insanely_good`, `ridiculously_good`, `vetoes`, `d1`-`d4`).

Reads train/research/arms/<program>.toml (or a path) for the expected arms, then every
<out>/<arm>/{constant,anneal}/status.json and evals/*.games.jsonl. Only complete games files enter
the estimates; everything missing is listed at the top. Read-only and idempotent.

Model, per endpoint and opponent: pair-mean outcome Y[seed i, setup j] = mu + a_i + b_j + e_ij. All
arms of a program play the same setups per (endpoint, opponent), so setups are crossed with training
seeds: a two-way ANOVA gives the setup component b (common random numbers: it cancels in every arm
contrast) and the residual e (eval noise that does not cancel). The panel score is the equal-weight
mean over the panel; s_seed^2 = var_i(panel_i) - sum_o MS_resid_o / (P^2 J_o). The endpoints of one
arm play different seed bases, so within-arm endpoint contrasts carry the full within-net eval variance.
"""
import argparse
import itertools
import json
import math
import sys
import tomllib
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm as harness   # noqa: E402

ROOT = harness.ROOT
PRIOR_GRID = (0.01, 0.015, 0.02, 0.03)
Z80 = stats.norm.ppf(0.80)

# Pre-registered rules, quoted from their source. Keys: (program, group).
RULES = {
    ('e0', 'obsfix'): dict(
        kind='noninferior_point', endpoint='anneal', margin=-0.01, seeds=[301, 302, 303, 304],
        source='REVIEW-2026-10-cursor.md Sec 4 "adopt the fix if the held-out mean is >= control - 0.01 and '
               'missile/spike-kill credit works" (annealed endpoint decides, PLAN Sec 2; vs control seeds 301-304, e0.toml)',
        extra='missile/spike kill credit: separate check, not in panel results'),
    ('e8', 'w512'): dict(
        kind='band', endpoint='anneal', hypothesis=0.04, band=0.02,
        source='PLAN E8 "beats the 256 control by >= 0.04; falsified if within +-0.02 at 3 seeds"',
        extra='also falsified if sps drops > 40% (not in panel results)'),
}
FACTORIAL_RULE = dict(q=0.10, min_effect=0.02, exploit_loss=0.05,
                      source='REVIEW Sec 4: BH q 0.10 over the main effects + the EMA contrast, effect >= +0.02, '
                             'exploit panel loses <= 0.05; pairs jointly only if interaction >= +0.02 at |z| >= 2')


# ---- expected layout ----------------------------------------------------------

def program_spec(program):
    config = Path(program)
    if not config.suffix:
        config = ROOT / 'train/research/arms' / f'{program}.toml'
    raw = tomllib.loads(config.read_text())
    top = {**harness.DEFAULTS, **{k: v for k, v in raw.items() if k not in ('arm', 'factorial')}}
    tables = list(raw.get('arm', {}).items()) + (harness.factorial_arms(raw['factorial']) if 'factorial' in raw else [])
    out = harness.path(top.get('out', f'native/artifacts/rl/research/{raw["name"]}'))
    arms = []
    for name, table in tables:
        cfg = {**top, **table}
        started = any((out / f'{name}-s{s}').exists() for s in cfg.get('followup_seeds', [])) and 'seeds' in table
        seeds = harness.expand_seeds(name, table, cfg, (table.get('group', name),) if started else ())
        cfg.pop('seeds', None)
        for seed in seeds:
            a = dict(cfg, name=f'{name}-s{seed}' if 'seeds' in table else name,
                     group=table.get('group', name), seed=seed)
            a['endpoint_names'] = [*harness.ENDPOINTS, *(f'raw{s // 1_000_000}m' for s in a['raw_at'])]
            a['endpoint_names'] = harness.scored_endpoints(a)
            a['evals'] = {o: dict(games=spec.get('games', a['games']),
                                  endpoints=[e for e in spec.get('endpoints', a['endpoint_names']) if e in a['endpoint_names']])
                          for o, spec, _ in harness.evals(a)}
            arms.append(a)
    return raw['name'], top, arms


def stage_status(run):
    try:
        s = json.loads((run / 'status.json').read_text())
    except (OSError, ValueError):
        return 'not started'
    if s.get('status') == 'complete':
        return 'complete'
    done, target = s.get('learner_steps', 0), s.get('target_learner_steps') or 1
    return f"{s.get('status', '?')} {done / 1e9:.2f}/{target / 1e9:.0f}B"


def load_games(path):
    """Pair-mean outcome by game seed (the setup); a pair is the side-swapped two games."""
    pairs = {}
    for line in path.read_text().splitlines():
        if line.startswith('{'):
            g = json.loads(line)
            pairs.setdefault(g['seed'], []).append({'a': 1.0, 'draw': 0.5}.get(g['outcome'], 0.0))
    return pairs


class Data:
    def __init__(self, out, arms):
        self.out = out
        self.cells, self.files, self.missing, self.status = {}, {}, [], {}
        for a in arms:
            d = out / a['name']
            self.status[a['name']] = (stage_status(d / 'constant'), stage_status(d / 'anneal'))
            for o, spec in a['evals'].items():
                for e in spec['endpoints']:
                    f = d / 'evals' / f'{e}-{o}.games.jsonl'
                    pairs = load_games(f) if f.exists() else {}
                    n = sum(map(len, pairs.values()))
                    if n == spec['games'] and all(len(v) == 2 for v in pairs.values()):
                        self.cells[a['name'], e, o] = pairs
                        self.files[a['name'], e, o] = f
                    else:
                        self.missing.append((a['name'], e, o, n, spec['games']))


# ---- estimators ---------------------------------------------------------------

def within_var(pairs):
    """Variance of the score of one net on fresh setups: var(pair means) / J."""
    y = np.array([np.mean(v) for v in pairs.values()])
    return y.var(ddof=1) / len(y), y.var(ddof=1)


def two_way(cells):
    """cells: list of pair dicts (one per net) on common setups -> MS_resid, setup var, J."""
    common = sorted(set.intersection(*(set(c) for c in cells)))
    y = np.array([[np.mean(c[s]) for s in common] for c in cells])
    k, j = y.shape
    resid = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
    ms_resid = (resid ** 2).sum() / ((k - 1) * (j - 1))
    ms_pair = k * y.mean(0).var(ddof=1)
    return ms_resid, max(0.0, (ms_pair - ms_resid) / k), j


def panel(data, a, e):
    """Panel score and its fresh-setup eval variance for one arm endpoint, or None if incomplete."""
    cells = [data.cells.get((a['name'], e, o)) for o in a['panel']]
    if any(c is None for c in cells):
        return None
    p = len(cells)
    score = np.mean([np.mean([x for v in c.values() for x in v]) for c in cells])
    return score, sum(within_var(c)[0] for c in cells) / p ** 2


def crn_correlation(data, arms):
    """Share of the within-net pair variance that is common to nets on the same setup (pooled two-way ANOVAs)."""
    by_cell = {}
    for (name, e, o), pairs in data.cells.items():
        by_cell.setdefault((e, o), []).append(pairs)
    num = den = 0.0
    for cells in by_cell.values():
        if len(cells) >= 2:
            ms_resid, setup, j = two_way(cells)
            num, den = num + setup * j, den + (setup + ms_resid) * j
    return (num / den if den else None), sum(len(c) >= 2 for c in by_cell.values())


def seed_components(data, group_arms, e, r_crn):
    """s_seed for one group at one endpoint from the crossed design. Returns a dict."""
    done = [a for a in group_arms if panel(data, a, e)]
    out = dict(k=len(done), scores=[panel(data, a, e)[0] for a in done], seeds=[a['seed'] for a in done])
    if not done:
        return out
    a0 = done[0]
    w = np.mean([panel(data, a, e)[1] for a in done])       # fresh-setup eval var per arm
    out['eval_var_fresh'] = w
    if len(done) >= 2:
        resid = 0.0
        for o in a0['panel']:
            ms_resid, _, j = two_way([data.cells[a['name'], e, o] for a in done])
            resid += ms_resid / j
        resid /= len(a0['panel']) ** 2
    else:
        resid = w * (1 - (r_crn or 0.0))
    out['eval_var_crn'] = resid
    if len(done) >= 2:
        k, v = len(done), np.var(out['scores'], ddof=1)
        out['var_scores'] = v
        out['s2'] = v - resid
        out['s2_ci'] = [(k - 1) * v / stats.chi2.ppf(0.975, k - 1) - resid, (k - 1) * v / stats.chi2.ppf(0.025, k - 1) - resid]
        out['s2_ub80'] = (k - 1) * v / stats.chi2.ppf(0.20, k - 1) - resid
    return out


def sd(x):
    return math.copysign(math.sqrt(abs(x)), x)


def fmt_s(s2):
    return f'{sd(s2):.4f}' if s2 >= 0 else f'0 (raw {-sd(-s2):.4f})'


# ---- report -------------------------------------------------------------------

def report(program, top, arms, data, prior, lines, ref=None, reference=None):
    p = lines.append
    groups = {}
    for a in arms:
        groups.setdefault(a['group'], []).append(a)
    endpoints = arms[0]['endpoint_names']

    # status
    p(f'== {program}: {len(arms)} arms, panel {", ".join(top["panel"])}')
    trained = sum(s == ('complete', 'complete') for s in data.status.values())
    p(f'trained {trained}/{len(arms)}; eval cells complete {len(data.cells)}/{len(data.cells) + len(data.missing)}')
    for a in arms:
        miss = [m for m in data.missing if m[0] == a['name']]
        c, an = data.status[a['name']]
        if not miss:
            continue
        part = [f'{e}:{o.split("-")[0]} {n}/{g}' for _, e, o, n, g in miss if n]
        todo = len(miss) - len(part)
        p(f'  missing {a["name"]:<14} train {c} | anneal {an} | evals: {todo} not started'
          + (f', partial {", ".join(part)}' if part else ''))

    # eval noise
    r_crn, n_cells = crn_correlation(data, arms)
    p('')
    p('-- eval noise (per net, panel mean)')
    for e in endpoints:
        vals = [panel(data, a, e) for a in arms if panel(data, a, e)]
        if vals:
            w = np.mean([v[1] for v in vals])
            n = sum(a['evals'][o]['games'] for o in a['panel'])
            p(f'  {e:<9} se {math.sqrt(w):.4f} at {n} games ({len(vals)} nets); binomial {0.5 / math.sqrt(n):.4f}')
    side = []
    for pairs in data.cells.values():
        g = np.array([x for v in pairs.values() for x in v])
        side.append(2 * within_var(pairs)[1] / g.var(ddof=1) - 1)
    if side:
        p(f'  side-swap pair correlation {np.mean(side):+.3f} (>0 inflates se above binomial; {len(side)} cells)')
    p(f'  CRN: setup share of pair variance shared across nets r = '
      + (f'{r_crn:.3f} ({n_cells} cells with >= 2 nets)' if r_crn is not None else 'n/a (no cell has 2 nets yet)'))

    # seed variance
    p('')
    p('-- training-seed variance (panel score; s2 = var(scores) - CRN eval var)')
    comps = {}
    for gname, g in groups.items():
        for e in endpoints:
            c = seed_components(data, g, e, r_crn)
            comps[gname, e] = c
            if not c['k']:
                continue
            sc = ' '.join(f'{x:.3f}' for x in c['scores'])
            line = f'  {gname:<7}{e:<9} k {c["k"]} [{sc}] eval se {math.sqrt(c["eval_var_crn"]):.4f}'
            if 's2' in c:
                lo, hi = c['s2_ci']
                line += (f' | s_seed {fmt_s(c["s2"])} 95% [{sd(max(lo, 0)):.4f}, {sd(max(hi, 0)):.4f}]'
                         f' 80%UB {sd(max(c["s2_ub80"], 0)):.4f} seed share {max(c["s2"], 0) / c["var_scores"]:.2f}')
            else:
                line += ' | s_seed needs k >= 2'
            p(line)

    # within-seed endpoint contrasts
    p('')
    p('-- within-seed endpoint effects (mean of per-seed differences; t CI across seeds, eval-only CI if k = 1)')
    contrasts = [(x, 'raw') for x in ('ema', 'anneal') if x in endpoints] + [('anneal', 'ema')] * ('ema' in endpoints) \
        + [('raw', x) for x in endpoints if x.startswith('raw') and x != 'raw']
    effects = {}
    for gname, g in groups.items():
        for hi, lo in contrasts:
            d, v = [], []
            for a in g:
                ph, pl = panel(data, a, hi), panel(data, a, lo)
                if ph and pl:
                    d.append(ph[0] - pl[0])
                    v.append(ph[1] + pl[1])
            if not d:
                continue
            k, m = len(d), float(np.mean(d))
            eval_se = math.sqrt(np.mean(v) / k)
            if k >= 2:
                se = np.std(d, ddof=1) / math.sqrt(k)
                t = stats.t.ppf(0.975, k - 1)
                ci, how = (m - t * se, m + t * se), f'seed sd {np.std(d, ddof=1):.4f}'
            else:
                se, ci, how = eval_se, (m - 1.96 * eval_se, m + 1.96 * eval_se), 'eval only'
            effects[gname, hi, lo] = dict(k=k, mean=m, se=se, ci=ci)
            p(f'  {gname:<7}{hi + "-" + lo:<14}{m:+.4f} 95% [{ci[0]:+.4f}, {ci[1]:+.4f}] k {k} ({how}; eval se {eval_se:.4f})')

    # pre-registered decisions
    p('')
    p('-- pre-registered decisions')
    ctrl = next((g for g in groups if not any(a.get('compare') for a in groups[g]) and 'factors' not in groups[g][0]), None)
    if program == 'e0' and ctrl:
        c = comps.get((ctrl, 'anneal'), {})
        if 's2_ub80' in c:
            ub = sd(max(c['s2_ub80'], 0))
            rule = 'k 3, n 4000' if ub <= 0.015 else 'k 4-6 or n 8000' if ub <= 0.03 else 'shrink program (E2, E3 at 6 seeds)'
            p(f'  E0 s_seed(anneal) 80%UB {ub:.4f} (k {c["k"]}/{len(groups[ctrl])}) -> {rule}'
              f'{"" if c["k"] == len(groups[ctrl]) else "  [interim]"}   (PLAN E0 decision on the REVIEW 80% UB)')
        else:
            p(f'  E0 s_seed(anneal): PENDING, {c.get("k", 0)}/{len(groups[ctrl])} control seeds evaluated')
    if ctrl and ('ema' in endpoints) and not top.get('controls'):
        er, ea = effects.get((ctrl, 'ema', 'raw')), effects.get((ctrl, 'anneal', 'ema'))
        if er and ea:
            tag = '' if er['k'] == len(groups[ctrl]) else '  [interim]'
            v = ('EMA replaces end anneals' if -ea['mean'] >= 0.02 else
                 'ship EMA as gated weights' if er['mean'] >= 0.03 and -ea['mean'] >= -0.02 else
                 'falsified (EMA - raw < +0.015)' if er['mean'] < 0.015 else 'neither threshold met')
            p(f'  E2 EMA-raw {er["mean"]:+.4f}, EMA-anneal {-ea["mean"]:+.4f} (k {er["k"]}) -> {v}{tag}   (PLAN E2, point estimates)')
    for gname, g in groups.items():
        cmp = g[0].get('compare')
        if not cmp:
            continue
        rule = RULES.get((program, gname), dict(kind='report', endpoint='anneal', source='no rule registered: report only'))
        e, seeds = rule['endpoint'], rule.get('seeds') or sorted({a['seed'] for a in g} & {a['seed'] for a in groups[cmp]})
        arm_s = [panel(data, a, e)[0] for a in g if a['seed'] in seeds and panel(data, a, e)]
        ctl_s = [panel(data, a, e)[0] for a in groups[cmp] if a['seed'] in seeds and panel(data, a, e)]
        p(f'  {gname} vs {cmp} @{e}: arm k {len(arm_s)}/{len(seeds)}, control k {len(ctl_s)}/{len(seeds)}')
        p(f'    rule: {rule["source"]}')
        if arm_s and ctl_s:
            diff = np.mean(arm_s) - np.mean(ctl_s)
            se, df = contrast_se(arm_s, ctl_s, comps.get((cmp, e), {}), prior, len(arm_s), len(ctl_s))
            lo1 = diff - stats.t.ppf(0.95, df) * se
            final = len(arm_s) == len(ctl_s) == len(seeds)
            verdict = decide(rule, diff) if final else f'PENDING (interim {decide(rule, diff)})'
            p(f'    diff {diff:+.4f} se {se:.4f} one-sided 95% LB {lo1:+.4f} -> {verdict}')
        else:
            p('    PENDING: no complete panel on one side')
        if rule.get('extra'):
            p(f'    also: {rule["extra"]}')
    fac = [a for a in arms if a.get('factors')]
    if fac and program == 'off1':
        offense_factorial_report(data, fac, reference, p)
    elif fac:
        factorial_report(data, fac, ref or {}, prior, p)
    if program == 'anneal-ctl':
        anneal_ctl_report(data, arms, reference, p)
    if program == 'optimizer':
        DECISIONS['optimizer'] = optimizer_report(data, arms, top, p)
    if program == 'x2d-drill':
        DECISIONS['x2d-drill'] = x2d_report(data, arms, top, p)
    if program == 'drill':
        DECISIONS['drill'] = drill_report(arms, data.out, p)
    if program == 'drill-r2':
        DECISIONS['drill-r2'] = drill_r2_report(arms, data.out, p)

    # secondary opponents
    sec = [o for o in arms[0]['evals'] if o not in arms[0]['panel']]
    if sec:
        p('')
        p('-- exploit panel (report only)')
        for gname, g in groups.items():
            row = []
            for o in sec:
                s = [np.mean([x for v in data.cells[a['name'], e, o].values() for x in v])
                     for a in g for e in arms[0]['evals'][o]['endpoints'] if (a['name'], e, o) in data.cells]
                if s:
                    row.append(f'{o.split("-1500m")[0].split("-2000m")[0]} {np.mean(s):.3f} (k {len(s)})')
            if row:
                p(f'  {gname:<12}' + ', '.join(row))
    human_regime_report(data, {g: v for g, v in groups.items() if not v[0].get('factors')}, p)
    return comps, r_crn


def contrast_se(arm_s, ctl_s, ctl_comp, prior, ka, kc):
    """se of a difference of group means: seed-level Welch when both sides have k >= 2, else prior s_seed."""
    if ka >= 2 and kc >= 2:
        va, vc = np.var(arm_s, ddof=1) / ka, np.var(ctl_s, ddof=1) / kc
        df = (va + vc) ** 2 / (va ** 2 / (ka - 1) + vc ** 2 / (kc - 1))
        return math.sqrt(va + vc), df
    unit = prior ** 2 + ctl_comp.get('eval_var_crn', 0.0)
    return math.sqrt(unit / ka + unit / kc), 1e9


def decide(rule, diff):
    if rule['kind'] == 'noninferior_point':
        return 'ADOPT (strength part)' if diff >= rule['margin'] else 'REJECT'
    if rule['kind'] == 'band':
        return ('supported' if diff >= rule['hypothesis'] else 'falsified' if abs(diff) <= rule['band']
                else 'inconclusive')
    return 'report only'


def bh(pv, q):
    """The keys Benjamini-Hochberg passes at level q."""
    order = sorted(pv, key=pv.get)
    for rank in range(len(order), 0, -1):
        if pv[order[rank - 1]] <= q * rank / len(order):
            return set(order[:rank])
    return set()


# ---- offense and the human regime (eval/kill_rate.py) ------------------------------

HUMAN_KEYS = ('kh_pre', 'kh_pre_cast', 'sd', 'pre_score', 'score_sd_draw', 'ckpm_b', 'env_loss')
OFFENSE_RULE = dict(q=0.10, source='PLAN-2026-10.md 2026-10-19 "pre-registration: off1 and anneal-ctl"')
ANNEAL_CTL = dict(reference_group='l2', q=0.10, min_effect=0.02, source=OFFENSE_RULE['source'],
                  contrasts={'D1 LR decay (e0 anneal - const1b)': ('e0-anneal', 'const1b'),
                             'D2 2B anneal (ann2b - e0 anneal)': ('ann2b', 'e0-anneal'),
                             'D3 floor 2.4e-6 (ann1b-f24 - e0 anneal)': ('ann1b-f24', 'e0-anneal')})


def as_result(data, a, e):
    """The results.json fields eval/kill_rate.py reads, for an endpoint whose panel is complete (else None)."""
    v = panel(data, a, e)
    if not v:
        return None
    return dict(arm=a['name'], group=a['group'], seed=a['seed'], panel=a['panel'], factors=a.get('factors'),
                flags=a.get('flags', {}), endpoints={e: dict(panel=dict(score=v[0]), opponents={
                    m: dict(games_out=str(data.files[a['name'], e, m])) for m in a['panel']})})


def regime(data, a, e='anneal'):
    """kill_rate.summarize over an endpoint's panel games, plus the v4.2 (cast credit) KH_PRE where rows carry it."""
    r = as_result(data, a, e)
    if not r:
        return None
    kr = harness.kill_rate()
    rows = kr.panel_rows(r, e)
    s = kr.summarize(rows)
    s['kh_pre_cast'] = kr.summarize(rows, 'cast')['kh_pre'] if all(g.get('credited') is not None for g in rows) else None
    s['panel'] = r['endpoints'][e]['panel']['score']
    return s


def human_regime_report(data, groups, p, e='anneal'):
    lines = []
    for gname, g in groups.items():
        rows = [s for s in (regime(data, a, e) for a in g) if s]
        if rows:
            vals = [[s[k] for s in rows if s.get(k) is not None] for k in HUMAN_KEYS]
            lines.append(f'  {gname:<12}{len(rows):>3} ' + ''.join(f'{np.mean(v):>14.4f}' if v else f'{"-":>14}' for v in vals))
    if lines:
        p('')
        p(f'-- human regime at {e} (eval/kill_rate.py on the panel games, mean over seeds; kh_pre_cast is v4.2 credit, '
          'secondary; score_sd_draw counts every game reaching sudden death as a draw)')
        p(f'  {"group":<12}{"k":>3} ' + ''.join(f'{k:>14}' for k in HUMAN_KEYS))
        for line in lines:
            p(line)


def factor_label(fac, f):
    hi, lo = [a for a in fac if a['factors'][f] > 0], [a for a in fac if a['factors'][f] < 0]
    for name, style in harness.kill_rate().OFFENSE_STYLES.items():
        if all(style in json.dumps(a['flags']) for a in hi) and not any(style in json.dumps(a['flags']) for a in lo):
            return name
    items = lambda a: {f'{k}={v}' for k, v in a['flags'].items()}
    return ' '.join(sorted(set.intersection(*map(items, hi)) - set.union(*map(items, lo)))) or f


def offense_factorial_report(data, fac, reference, p):
    """off1: BH q 0.10 on the KH_PRE main effects screens factors into round 2 (kill_rate.rule per factor, high vs
    low runs, strength non-inferiority > -0.02, its vetoes); kill_rate.factorial_rule is the I1/I3 auto-adopt rule
    (I7 too since 2026-10-20); secondary main effects and interactions are reported, no decision."""
    kr = harness.kill_rate()
    letters = list(fac[0]['factors'])
    results = [r for r in (as_result(data, a, 'anneal') for a in fac) if r]
    p(f'  offense factorial: {len(results)}/{len(fac)} runs with an annealed panel; rules: {OFFENSE_RULE["source"]}')
    if len(results) < len(fac):
        p('    PENDING: effects need the full balanced design')
        return
    labels = {f: factor_label(fac, f) for f in letters}
    verdicts = {f: kr.rule({f: [r for r in results if r['factors'][f] > 0]},
                           [r for r in results if r['factors'][f] < 0])[f] for f in letters}
    pv = {f: 2 * min(v['p_one_sided'], 1 - v['p_one_sided']) for f, v in verdicts.items()}
    passed = bh(pv, OFFENSE_RULE['q'])
    p(f'    primary: KH_PRE main effect = mean(high runs) / mean(low runs), {len(results) // 2} vs {len(results) // 2}; '
      f'BH q {OFFENSE_RULE["q"]} over the {len(letters)} two-sided bootstrap p')
    for f in sorted(letters, key=pv.get):
        v = verdicts[f]
        ni, veto = v['noninferiority'], v['vetoes']['ckpm_b_up_50pct'] or v['vetoes']['env_loss_up']
        screen = f in passed and v['ratio'] > 1 and ni['lower95'] > kr.MARGIN and not veto
        p(f'    {f} {labels[f]:<38} KH_PRE x{v["ratio"]:.3f} [{v["ratio_ci"][0]:.3f}, {v["ratio_ci"][1]:.3f}] p {pv[f]:.4f} '
          f'BH {"pass" if f in passed else "fail"} | panel {ni["diff"]:+.4f} LB {ni["lower95"]:+.4f} | ckpm_b x{v["ckpm_b_ratio"]:.2f} '
          f'env_loss {v["env_loss_diff"]:+.3f}{" VETO" if veto else ""} -> {"ROUND 2" if screen else "no"}')
    p('    I1/I3/I7 auto-adopt rule (eval/kill_rate.py factorial_rule: ratio >= 1.20, CI > 1, Holm over I1/I3/I7):')
    adopt = kr.factorial_rule(results)
    for line in kr.report(adopt, 'the low level').splitlines():
        p('      ' + line)
    if sum(v['adopt'] for k, v in adopt.items() if k.split()[0] in ('I1', 'I3', 'I7')) >= 2:
        p('    two or more of I1/I3/I7 pass: run the passers combined against each alone before adopting them together '
          '(sign-off 2026-10-19, I7 2026-10-20)')
    runs = {a['name']: regime(data, a) for a in fac}
    keys = ('panel', 'pre_score', 'score_sd_draw', 'sd', 'kh_pre_cast')
    p('    secondary main effects (high - low, no decision): ' + ', '.join(keys))
    for f in letters:
        diff = []
        for k in keys:
            hi = [runs[a['name']][k] for a in fac if a['factors'][f] > 0 and runs[a['name']].get(k) is not None]
            lo = [runs[a['name']][k] for a in fac if a['factors'][f] < 0 and runs[a['name']].get(k) is not None]
            diff.append(f'{np.mean(hi) - np.mean(lo):+.4f}' if hi and lo else '-')
        p(f'    {f} {labels[f]:<38} ' + '  '.join(diff))
    x = np.array([[a['factors'][f] for f in letters] for a in fac], float)
    for metric, y in (('log KH_PRE', np.log([max(runs[a['name']]['kh_pre'], 1e-6) for a in fac])),
                      ('panel', np.array([runs[a['name']]['panel'] for a in fac]))):
        n, effects, alias = len(y), {}, {}
        for i, j in itertools.combinations(range(len(letters)), 2):
            effects[letters[i] + letters[j]] = float(y @ (x[:, i] * x[:, j])) * 2 / n
        for c in itertools.combinations(range(len(letters)), 3):
            col = x[:, c[0]] * x[:, c[1]] * x[:, c[2]]
            if tuple(-col) not in alias:
                alias.setdefault(tuple(col), float(y @ col) * 2 / n)
        se = math.sqrt(np.mean(np.square(list(alias.values())))) if alias else float('nan')
        top = sorted(effects.items(), key=lambda kv: -abs(kv[1]))[:4]
        p(f'    exploratory 2fi on {metric} (se {se:.4f} from {len(alias)} 3fi alias contrasts): '
          + ', '.join(f'{k} {v:+.4f} (|z| {abs(v) / se:.1f})' for k, v in top))
    if reference:
        ref_data, ref_arms = reference
        ctl = [s for s in (regime(ref_data, a) for a in ref_arms if a['group'] == 'obsfix') if s]
        low = next((runs[a['name']] for a in fac if all(v < 0 for v in a['factors'].values())), None)
        if ctl and low:
            p(f'    check: all-low run vs e0/obsfix (k {len(ctl)}): KH_PRE {low["kh_pre"]:.4f} vs '
              f'{np.mean([s["kh_pre"] for s in ctl]):.4f} sd {np.std([s["kh_pre"] for s in ctl], ddof=1) if len(ctl) > 1 else 0:.4f}; '
              f'panel {low["panel"]:.4f} vs {np.mean([s["panel"] for s in ctl]):.4f}')


def anneal_ctl_report(data, arms, reference, p):
    """anneal-ctl: each cell vs E0's own anneal on the same 6B checkpoints and setups; seeds x cells two-way ANOVA
    residual as the error (df (cells-1)(seeds-1)), BH q 0.10 over D1-D3; the restart effect is exploratory."""
    rule = ANNEAL_CTL
    if not reference:
        p('  anneal-ctl: needs --reference e0')
        return
    ref_data, ref_arms = reference
    e0 = {a['seed']: a for a in ref_arms if a['group'] == rule['reference_group']}
    groups = {}
    for a in arms:
        groups.setdefault(a['group'], {})[a['seed']] = a
    cells = {'e0-anneal': {s: (ref_data, a) for s, a in e0.items()}, **{g: {s: (data, a) for s, a in v.items()} for g, v in groups.items()}}
    seeds = sorted(s for s in set.intersection(*(set(c) for c in cells.values()))
                   if all(panel(d, a, 'anneal') for d, a in (c[s] for c in cells.values())))
    p(f'  anneal-ctl: {len(seeds)} seeds with every cell scored (reference e0/{rule["reference_group"]}); rule: {rule["source"]}')
    if len(seeds) < 2:
        p('    PENDING: needs >= 2 complete seeds')
        return
    names = list(cells)
    y = np.array([[panel(*cells[c][s], 'anneal')[0] for s in seeds] for c in names])
    resid = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
    df = (len(names) - 1) * (len(seeds) - 1)
    se = math.sqrt(2 * (resid ** 2).sum() / df / len(seeds))
    p('    cells (panel at anneal): ' + ', '.join(f'{c} {y[i].mean():.4f}' for i, c in enumerate(names)) + f'; contrast se {se:.4f}, t df {df}')
    est, pv = {}, {}
    for name, (hi, lo) in rule['contrasts'].items():
        if hi in names and lo in names:
            est[name] = float(y[names.index(hi)].mean() - y[names.index(lo)].mean())
            pv[name] = float(2 * stats.t.sf(abs(est[name]) / se, df))
    passed = bh(pv, rule['q'])
    t = stats.t.ppf(0.975, df)
    for name in est:
        ok = name in passed and est[name] >= rule['min_effect']
        verdict = ('LR decay adds to the restart' if ok else 'no evidence the decay adds to the restart') if name.startswith('D1') \
            else ('ADOPT as the program endpoint' if ok else 'keep the 1B anneal to 1.2e-5')
        p(f'    {name:<40} {est[name]:+.4f} 95% [{est[name] - t * se:+.4f}, {est[name] + t * se:+.4f}] p {pv[name]:.4f} '
          f'BH {"pass" if name in passed else "fail"} -> {verdict}')
    raw = [panel(ref_data, e0[s], 'raw') for s in seeds]
    if 'const1b' in names and all(raw):
        d = y[names.index('const1b')] - np.array([r[0] for r in raw])
        sd_ = np.std(d, ddof=1)
        p(f'    exploratory restart effect (const1b - e0 raw 6B, paired over seeds, other setups): {d.mean():+.4f} '
          f'95% [{d.mean() - stats.t.ppf(0.975, len(d) - 1) * sd_ / math.sqrt(len(d)):+.4f}, '
          f'{d.mean() + stats.t.ppf(0.975, len(d) - 1) * sd_ / math.sqrt(len(d)):+.4f}]')
    p('    human regime per cell (mean over those seeds): ' + ', '.join(HUMAN_KEYS[:5]))
    for c in names:
        rows = [regime(*cells[c][s]) for s in seeds]
        p(f'      {c:<12}' + ''.join(f'{np.mean([r[k] for r in rows if r.get(k) is not None]):>10.4f}'
                                     if any(r.get(k) is not None for r in rows) else f'{"-":>10}' for k in HUMAN_KEYS[:5]))


def factorial_report(data, fac, ctl, prior, p):
    letters = list(fac[0]['factors'])
    done = [a for a in fac if panel(data, a, 'anneal')]
    p(f'  factorial: {len(done)}/{len(fac)} runs with an annealed panel; rule: {FACTORIAL_RULE["source"]}')
    if not any(o not in fac[0]['panel'] for o in fac[0]['evals']):
        p('    NOTE: no exploit-panel opponents configured, so the "exploit panel loses <= 0.05" condition cannot be checked')
    if len(done) < len(fac):
        p('    PENDING: effects need the full balanced design')
        return
    x = np.array([[a['factors'][f] for f in letters] for a in done], float)
    y = np.array([panel(data, a, 'anneal')[0] for a in done])
    n = len(y)
    cols = {f: x[:, i] for i, f in enumerate(letters)}
    for i, f in enumerate(letters):
        for g in letters[i + 1:]:
            cols[f + g] = x[:, i] * x[:, letters.index(g)]
    three = {}
    for i, f in enumerate(letters):
        for j in range(i + 1, len(letters)):
            for k in range(j + 1, len(letters)):
                col = x[:, i] * x[:, j] * x[:, k]
                key = tuple(col)
                if key not in three and tuple(-col) not in three:
                    three[key] = col
    effect = {name: float(y @ c) * 2 / n for name, c in cols.items()}
    pseudo = [float(y @ c) * 2 / n for c in three.values()]
    se_int = math.sqrt(np.mean(np.square(pseudo))) if pseudo else float('nan')
    s2 = max(ctl['s2'], 0) if 's2' in ctl else prior ** 2
    eval_var = seed_components(data, done, 'anneal', None)['eval_var_crn']
    se_ext = 2 * math.sqrt(s2 + eval_var) / math.sqrt(n)
    df_ext = ctl['k'] - 1 if 's2' in ctl else 0
    df = len(pseudo) + df_ext
    se = math.sqrt((len(pseudo) * se_int ** 2 + df_ext * se_ext ** 2) / df)
    p(f'    effect se {se:.4f} (t df {df}) = df-weighted pool of internal {se_int:.4f} ({len(pseudo)} 3fi alias '
      f'contrasts) and external {se_ext:.4f} (s_seed from {ctl.get("program", "prior")} k {ctl.get("k", 0)}, '
      f'{"df " + str(df_ext) if df_ext else "prior, df 0"})')
    d = [a for a in done if panel(data, a, 'ema') and panel(data, a, 'raw')]
    ema = [panel(data, a, 'ema')[0] - panel(data, a, 'raw')[0] for a in d]
    family = {f: (effect[f], se, df) for f in letters}
    if len(ema) >= 2:
        family['EMA-raw'] = (float(np.mean(ema)), float(np.std(ema, ddof=1) / math.sqrt(len(ema))), len(ema) - 1)
    pv = {k: 2 * stats.t.sf(abs(m / s), df_) for k, (m, s, df_) in family.items()}
    order = sorted(pv, key=pv.get)
    m = len(order)
    passed = set()
    for rank in range(m, 0, -1):
        if pv[order[rank - 1]] <= FACTORIAL_RULE['q'] * rank / m:
            passed = set(order[:rank])
            break
    for k in order:
        mean, s, _ = family[k]
        adopt = k in passed and mean >= FACTORIAL_RULE['min_effect']
        p(f'    {k:<8} {mean:+.4f} se {s:.4f} p {pv[k]:.3f} BH {"pass" if k in passed else "fail"} -> '
          f'{"ADOPT -> confirm 4 vs 4 on 221001" if adopt else "no"}')
    for name in cols:
        if len(name) == 2 and effect[name] >= 0.02 and abs(effect[name] / se) >= 2:
            p(f'    interaction {name} {effect[name]:+.4f} (|z| {abs(effect[name] / se):.1f}) -> adopt pair jointly')


# ---- recipe stage 3 optimizer screen and stage 2d (PLAN-2026-10.md 2026-10-22 pre-registration) ----------------

DECISIONS = {}
OPTIMIZER = dict(
    source='PLAN-2026-10.md 2026-10-22 "pre-registration: x2d-drill and optimizer"',
    endpoint='anneal',
    sd_floor=0.0068,            # per-run sd floor: E0 s_seed(anneal) point 0.0030 (+) CRN eval se 0.0061 at 4300 games
    switch=('adam-3e4', 'adam-matched', 'muon-adamw'),   # candidates to replace Muon
    harmless=('muon-wd',),      # adopted if non-inferior
    screen_promote=0.005,       # switch candidate: follow up if screen diff >= +0.005
    screen_margin=-0.02,        # muon-wd: follow up if screen diff >= -0.02
    confirm_min=0.01,           # switch: diff >= +0.01 and Holm one-sided 0.05 over the promoted candidates
    alpha=0.05, ni_margin=-0.02,   # muon-wd: one-sided 95% LB > -0.02
    kh_veto=0.85)               # no switch/adopt if KH_PRE (game credit) < 0.85 x control
X2D = dict(
    source='EXPLOITER-GAPS-2026-10.md Section 3, restated in PLAN-2026-10.md 2026-10-22',
    defect=0.10, learn=0.15, intrinsic=0.05, guard=-0.05)


def control_runs(top):
    """The scored control replicates named by `controls` ([{program, group}]): [(label, data, arm)]."""
    out = []
    for c in top.get('controls', []):
        name, ctop, carms = program_spec(c['program'])
        g = [a for a in carms if a['group'] == c['group']]
        data = Data(harness.path(ctop.get('out', f'native/artifacts/rl/research/{name}')), g)
        out += [(f"{name}/{a['name']}", data, a) for a in g]
    return out


def pooled(arm_s, ctl_s, floor):
    """diff, se, df of a pooled-variance t (per-run sd floored), arm k may be 1."""
    ka, kc = len(arm_s), len(ctl_s)
    ss = (np.var(arm_s, ddof=1) * (ka - 1) if ka > 1 else 0.0) + np.var(ctl_s, ddof=1) * (kc - 1)
    df = ka + kc - 2
    s = max(math.sqrt(ss / df), floor)
    return float(np.mean(arm_s) - np.mean(ctl_s)), s * math.sqrt(1 / ka + 1 / kc), df


def optimizer_report(data, arms, top, p):
    """Screen (k 1) -> follow-up seeds -> confirm (k 3) against the v4.2 base replicates (top `controls`)."""
    rule, e = OPTIMIZER, OPTIMIZER['endpoint']
    ctl = control_runs(top)
    p(f'  optimizer: rules {rule["source"]}')
    ctl_s = [(lab, panel(d, a, e)[0]) for lab, d, a in ctl if panel(d, a, e)]
    if len(ctl_s) < 2:
        p(f'    PENDING: {len(ctl_s)}/{len(ctl)} control runs scored at {e}')
        return dict(stage='pending')
    cs = [v for _, v in ctl_s]
    p(f'    control @{e} k {len(cs)}: ' + ', '.join(f'{lab} {v:.4f}' for lab, v in ctl_s)
      + f'; mean {np.mean(cs):.4f} sd {np.std(cs, ddof=1):.4f} (per-run sd floor {rule["sd_floor"]})')
    ctl_kh = [r['kh_pre'] for r in (regime(d, a, e) for _, d, a in ctl) if r]
    groups = {}
    for a in arms:
        groups.setdefault(a['group'], []).append(a)
    out, confirm = dict(stage='screen', followup=[], verdicts={}), {}
    for g, members_ in groups.items():
        done = [a for a in members_ if panel(data, a, e)]
        if len(done) < len(members_):
            p(f'    {g:<13} PENDING: {len(done)}/{len(members_)} runs scored at {e}')
            out['stage'] = 'pending'
            continue
        s = [panel(data, a, e)[0] for a in done]
        diff, se, df = pooled(s, cs, rule['sd_floor'])
        kh = [r['kh_pre'] for r in (regime(data, a, e) for a in done) if r]
        khr = np.mean(kh) / np.mean(ctl_kh) if kh and ctl_kh else float('nan')
        sec = []
        for x in ('raw', 'ema'):
            xa = [panel(data, a, x)[0] for a in done if panel(data, a, x)]
            xc = [panel(d, a, x)[0] for _, d, a in ctl if panel(d, a, x)]
            if xa and xc:
                sec.append(f'{x} {np.mean(xa) - np.mean(xc):+.4f}')
        ff = [v for lab, v in ctl_s if lab.startswith('off1/')]
        if ff:
            sec.append(f'vs off1/ff-1 alone {np.mean(s) - np.mean(ff):+.4f}')
        line = (f'    {g:<13} k {len(s)} [{" ".join(f"{x:.4f}" for x in s)}] diff {diff:+.4f} se {se:.4f} df {df} '
                f'| KH_PRE x{khr:.3f} | secondary: {", ".join(sec)}')
        if len(s) == 1:
            if g in rule['switch']:
                go = diff >= rule['screen_promote']
                v = 'FOLLOW UP (+302, +303)' if go else 'drop: no evidence it beats Muon' + (
                    ' (Muon better by >= 0.03)' if diff <= -0.03 else '')
            elif g in rule['harmless']:
                go = diff >= rule['screen_margin']
                v = 'FOLLOW UP (+302, +303)' if go else 'drop: harmful at the screen (diff < -0.02)'
            else:
                go, v = False, 'no rule: report only'
            if go:
                out['followup'].append(g)
            out['verdicts'][g] = v
            p(line + f' -> screen: {v}')
        else:
            confirm[g] = (diff, se, df, khr)
            p(line)
    sw = {g: v for g, v in confirm.items() if g in rule['switch']}
    if sw:
        out['stage'] = 'confirm'
        pv = {g: float(stats.t.sf(d / se, df)) for g, (d, se, df, _) in sw.items()}
        holm, alive = {}, True
        for i, g in enumerate(sorted(pv, key=pv.get)):
            alive = alive and pv[g] <= rule['alpha'] / (len(pv) - i)
            holm[g] = alive
        passed = [g for g in sw if holm[g] and sw[g][0] >= rule['confirm_min'] and not sw[g][3] < rule['kh_veto']]
        best = max(passed, key=lambda g: sw[g][0]) if passed else None
        for g, (d, se, df, khr) in sw.items():
            v = ('SWITCH to ' + g if g == best else 'passes, but another candidate is larger') if g in passed else \
                'KH_PRE veto' if holm[g] and d >= rule['confirm_min'] else 'keep Muon'
            out['verdicts'][g] = v
            p(f'    confirm {g:<13} diff {d:+.4f} one-sided p {pv[g]:.4f} Holm {"pass" if holm[g] else "fail"} '
              f'(m {len(pv)}) -> {v}')
        out['switch'] = best
    for g, (d, se, df, khr) in confirm.items():
        if g in rule['harmless']:
            out['stage'] = 'confirm'
            lb = d - stats.t.ppf(1 - rule['alpha'], df) * se
            ok = lb > rule['ni_margin'] and not khr < rule['kh_veto']
            out['verdicts'][g] = v = 'ADOPT for the long run and the loop' if ok else 'do not adopt'
            out['adopt_wd'] = ok
            p(f'    confirm {g:<13} diff {d:+.4f} one-sided 95% LB {lb:+.4f} (margin {rule["ni_margin"]}) -> {v}')
    if out['followup']:
        p(f'    follow-up: arm.py train/research/arms/optimizer.toml --followup {",".join(out["followup"])}')
    return out


def exploit_pairs(path, subject):
    """{seed: mean score} of one subject's games in an exploit_eval games.jsonl."""
    pairs = {}
    if path.exists():
        for line in path.read_text().splitlines():
            g = json.loads(line)
            if g['subject'] == subject:
                pairs.setdefault(g['seed'], []).append(g['score'])
    return {k: np.mean(v) for k, v in pairs.items() if len(v) == 2}


def paired(a, b):
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    d = np.array([a[s] - b[s] for s in common])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return float(d.mean()), float(h), len(d)


def x2d_report(data, arms, top, p):
    """Stage 2d: tuned (1B) - base (step 0) vs me31 on the arm's own kit schedule decides; panel guard raw - raw0m."""
    rule, out = X2D, dict(case='pending', stop_optimizer=False)
    p(f'  x2d-drill: rules {rule["source"]}')
    own = {}
    for a in arms:
        d = harness.path(top.get('out', 'native/artifacts/rl/research/x2d-drill')) / a['name']
        base = {k: exploit_pairs(d / k / 'games.jsonl', 'base') for k in ('x2d-eval-final', 'x2d-eval-on-iid')}
        r = {k: paired(exploit_pairs(d / k / 'games.jsonl', 'tuned'), base[k]) for k in base}
        curve = [(f't{s}', paired(exploit_pairs(d / 'x2d-eval-curve' / 'games.jsonl', f't{s}'), base['x2d-eval-final']))
                 for s in (250, 500, 750)]
        g = (panel(data, a, 'raw'), panel(data, a, 'raw0m'))
        guard = g[0][0] - g[1][0] if all(g) else None
        if r['x2d-eval-final'] is None:
            p(f'    {a["name"]:<7} PENDING: no matchup eval (x2d-drill.sh eval {a["name"]})')
            continue
        own[a['name']] = r['x2d-eval-final'][0]
        bm = np.mean(list(base['x2d-eval-final'].values()))
        fmt = lambda x: f'{x[0]:+.3f} ±{x[1]:.3f} (n {x[2]})' if x else '-'
        p(f'    {a["name"]:<7} base {bm:.3f}; tuned - base own schedule {fmt(r["x2d-eval-final"])}, iid schedule '
          f'{fmt(r["x2d-eval-on-iid"])}; curve ' + ', '.join(f'{k} {x[0]:+.3f}' if x else f'{k} -' for k, x in curve)
          + (f'; guard panel raw - raw0m {guard:+.4f}' + (' -> CASE 4: overfitting costs general play' if guard < rule['guard'] else '')
             if guard is not None else '; guard pending'))
    m, i = own.get('mirror'), own.get('iid')
    if m is not None:
        if m < rule['defect']:
            out.update(case='1 learner or observation defect on defence', stop_optimizer=True)
        elif i is None:
            out['case'] = 'pending (iid)'
        elif m >= rule['learn'] and i >= rule['learn']:
            out['case'] = '2 learnable; the league starves it'
        elif m >= rule['learn'] and i <= rule['intrinsic']:
            out['case'] = '3 a well-piloted drill beats random kits (run counter)'
        else:
            out['case'] = 'none of 1-3 (between thresholds): report, no stop'
    p(f'    -> case {out["case"]}' + ('; STOP the optimizer program and debug (Section 3 case 1)' if out['stop_optimizer'] else ''))
    return out


# ---- drill showcase specialist (PLAN-2026-10.md 2026-10-23 pre-registration) -------------------------------

DRILL = dict(
    source='DRILL-SPEC-2026-10.md Section 6 as decided by Matthew 2026-10-23, pre-registered in PLAN-2026-10.md '
           '"2026-10-23 pre-registration: drill specialist"',
    games=1200,                  # per drill_look eval: 600 seed pairs, both seats
    p1_point=0.75, p1_lb=0.70,   # P1 (primary): score vs l2-main-136002m on iid random kits, pair-bootstrap 95% LB
    p2_ug1s=0.10, p2_kill05=0.25,   # P2: >= 10% of drill episodes with >= 1 s continuous underground, and
                                    # >= 25% of drill kills after >= 0.5 s underground in that dive
    p3=0.50,                     # P3: score vs me31 on its declared kits
    p4_unforced=0.03, p4_sd=0.30,   # P4 guard: unforced drill deaths <= 3% of drill episodes, SD share <= 30%
    intent=0.20,                 # surfacings within 6 u of the opponent (Section 4 "intent")
    v43=0.05)                    # adopt v4.3 for the specialist if SO - S >= +0.05 on P1 (seed means)


def drill_look_analyze():
    import importlib.util
    spec = importlib.util.spec_from_file_location('drill_look_analyze', ROOT / 'native/rl/research/drill-spec/analyze.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def drill_pairs(d):
    """{pair: mean score of A} of a complete drill_look games.jsonl, else None."""
    f = d / 'games.jsonl'
    if not f.exists():
        return None
    games = [json.loads(l) for l in f.read_text().splitlines()]
    if len(games) != DRILL['games']:
        return None
    pairs = {}
    for g in games:
        pairs.setdefault(g['pair'], []).append(1.0 if g['outcome'] == 'a' else 0.5 if g['outcome'] == 'draw' else 0.0)
    return {k: float(np.mean(v)) for k, v in pairs.items() if len(v) == 2}


def drill_runs(arms, base, p, rule, script, intent_mark=False):
    """Per run at the annealed endpoint, from its `<script> eval` drill_look games: P1 (primary) vs main136 on random
    kits, P2 and P4 on the same games, P3 vs me31 on its kits, intent. Prints one line per run; returns
    (runs, {run: {pair: score}}, {group: [runs]}, {run: drill_look analysis of the P1 games})."""
    A = drill_look_analyze()
    runs, pairs, groups, looks = {}, {}, {}, {}
    for a in arms:
        d = base / a['name']
        main_p, me31_p = drill_pairs(d / 'drill-eval-main'), drill_pairs(d / 'drill-eval-me31')
        if main_p is None:
            p(f'    {a["name"]:<9} PENDING: no complete P1 eval (train/research/arms/{script} eval {a["name"]})')
            continue
        m = A.analyze(d / 'drill-eval-main')
        m3 = A.analyze(d / 'drill-eval-me31') if me31_p is not None else None
        pairs[a['name']], looks[a['name']] = main_p, m
        groups.setdefault(a['group'], []).append(a['name'])
        sc, k05, ug1, unf, sd, em = m['score'], m.get('drill_kills_after_ug05'), m.get('ep_ug_ge_1s'), \
            m.get('ep_death_unforced'), m['sd_share'], m.get('emerge_within_6u')
        r = dict(p1=sc, p1_pass=sc[0] >= rule['p1_point'] and sc[1] >= rule['p1_lb'],
                 p2_pass=bool(ug1 and k05 and ug1[0] >= rule['p2_ug1s'] and k05[0] >= rule['p2_kill05']),
                 p3=m3['score'] if m3 else None, p3_pass=bool(m3 and m3['score'][0] >= rule['p3']),
                 p4_pass=bool(unf and unf[0] <= rule['p4_unforced'] and sd[0] <= rule['p4_sd']),
                 intent=em[0] if em else None, intent_pass=bool(em and em[0] >= rule['intent']),
                 ug1s=ug1, kill05=k05, unforced=unf, sd=sd, group=a['group'], episodes=m.get('episodes', 0))
        r['all_pass'] = r['p1_pass'] and r['p2_pass'] and r['p3_pass'] and r['p4_pass']
        runs[a['name']] = r
        f = lambda x: '-' if not x else f'{x[0]:.3f} [{x[1]:.3f},{x[2]:.3f}]'
        mark = f' {"PASS" if r["intent_pass"] else "fail"}' if intent_mark else ''
        p(f'    {a["name"]:<9} P1 {f(sc)} {"PASS" if r["p1_pass"] else "fail"} | P2 ug>=1s {f(ug1)}, kills after 0.5 s ug '
          f'{f(k05)} {"PASS" if r["p2_pass"] else "fail"} | P3 {f(r["p3"])} {"PASS" if r["p3_pass"] else "fail" if m3 else "pending"} | '
          f'P4 unforced {f(unf)}, SD {f(sd)} {"PASS" if r["p4_pass"] else "fail"} | intent {f(em)}{mark}')
    return runs, pairs, groups, looks


def drill_group_pairs(pairs, groups, g):
    """A group's seed mean per eval seed pair (its runs' common pairs) and its number of runs."""
    names = groups.get(g, [])
    common = set.intersection(*(set(pairs[n]) for n in names)) if names else set()
    return {k: float(np.mean([pairs[n][k] for n in names])) for k in common}, len(names)


def drill_report(arms, base, p):
    """P1 (primary) and P2-P4 per run at the annealed endpoint (drill.sh eval: drill_look vs main136 on random kits
    and vs me31 on its kits); the showcase is the styled run (S or SO) with the best P1 point; group contrasts on seed
    means paired by seed pair: S - C and SO - C (the burrow style's strength cost, report only) and SO - S (v4.3)."""
    rule, out = DRILL, dict(showcase=None, p1_pass=False, insanely_good=False, adopt_v43=None, runs={})
    p(f'  drill: rules {rule["source"]}')
    out['runs'], pairs, groups, _ = drill_runs(arms, base, p, rule, 'drill.sh')
    styled = {n: r for n, r in out['runs'].items() if r['group'] in ('S', 'SO')}
    if styled:
        best = max(styled, key=lambda n: styled[n]['p1'][0])
        out.update(showcase=best, p1_pass=styled[best]['p1_pass'], insanely_good=styled[best]['all_pass'])
        p(f'    showcase (best styled run by P1 point, chosen after the fact among {len(styled)}): {best}: '
          f'P1 {"PASS" if out["p1_pass"] else "FAIL"}; "insanely good" (P1-P4) {"YES" if out["insanely_good"] else "NO"}')
    for t, c, what in (('S', 'C', 'style strength cost, report only'), ('SO', 'C', 'style + v4.3 vs control, report only'),
                       ('SO', 'S', 'v4.3 effect on P1')):
        (gt, kt), (gc, kc) = drill_group_pairs(pairs, groups, t), drill_group_pairs(pairs, groups, c)
        r = paired(gt, gc) if kt and kc else None
        if r is None:
            p(f'    {t} - {c} ({what}): pending')
            continue
        p(f'    {t} - {c} ({what}): {r[0]:+.3f} ±{r[1]:.3f} (eval-only 95% over {r[2]} seed pairs; seeds {kt} vs {kc}, '
          f'seed variance not in the interval)')
        out[f'{t}-{c}'] = r[0]
        if (t, c) == ('SO', 'S'):
            so_p2 = any(out['runs'][n]['p2_pass'] and out['runs'][n]['intent_pass'] for n in groups['SO'])
            s_p2 = any(out['runs'][n]['p2_pass'] and out['runs'][n]['intent_pass'] for n in groups['S'])
            out['adopt_v43'] = r[0] >= rule['v43'] or (so_p2 and not s_p2)
            p(f'    -> v4.3 for the specialist: {"ADOPT" if out["adopt_v43"] else "no"} (SO - S >= +{rule["v43"]} on P1, '
              f'or SO passes P2 + intent where S does not)')
    return out


# ---- drill specialist round 2 (DRILL-R2-2026-10.md Section 2.4) ----------------------------------------------------

DRILL_R2 = dict(
    DRILL,   # P1-P4 and intent: the round-1 targets, unchanged (Section 2.4 table)
    source='DRILL-R2-2026-10.md Section 2.4 (P1-P4 and intent unchanged from spec 6; L, D1-D4 and the hack vetoes), '
           'pre-registered in PLAN-2026-10.md "2026-10-26 Drill round 2 pre-registration"',
    control='K', design='R2', dose='R2x2', ablation='R2v42',
    showcase_groups=('R2', 'R2x2'),   # "the styled v4.3 run with the best P1 point among the unvetoed, chosen after the fact"
    l_share=0.10,                     # L: >= 10% of first surfacings are launches: run >= 8 decisions, last underground
    l_run=8, l_speed=30.0, l_cos=0.866,   # step >= 30 u/s within 30 deg of the opponent (any distance); and the share of
    l_kill_x=2.0,                     # drill kills after such a launch within the form >= 2 x its share at init
    d1=-0.04,                         # D1: the style stays if R2 - K >= -0.04 (point), else rejected; "within noise" flag
    d2_p1=-0.03,                      # D2: x2 only if R2x2 passes P2 where neither R2 seed does and P1 >= R2 mean - 0.03
    d3_keep_p2=0.03, d3_keep_p1=0.03,   # D3 on the gains net of the inits: keep v4.3 if G_P2 >= +3 points or G_P1 >= +0.03;
    d3_near_p2=0.015, d3_near_p1=0.03,  # "the senses are not the bottleneck" if |G_P2| <= 1.5 points and |G_P1| <= 0.03
    tail=0.10,                        # training vetoes: telemetry over the last 10% of the constant stage
    drill_id=26,                      # ability id of Drill (metrics ability_activations, drill_look's kills)
    surface_term='burrow_surface',
    veto_kpm=0.8,                     # kills per minute >= 0.8 x K's
    veto_style_share=0.5,             # style net <= 50% of outcome + kill + death reward
    veto_unforced=2.0,                # eval games: unforced drill deaths per drill episode <= 2 x K's
    far_dist=16.0, veto_arrive=0.5)   # eval games: far launches (> 16 u, first air step >= 30 u/s within 30 deg) that
                                      # reach 6 u or kill in the form >= 0.5 x the rate at init


def train_tail(run, tail):
    """Sums over the last `tail` (by learner steps) of a training stage's metrics.jsonl rows: learner decisions,
    seconds and kills (behavior.totals), the outcome + kill + death reward, Drill slot presses (ability_activations)
    and, per style term, paid and fires (style.terms; fires is a count per row). A row whose style read covers no
    decisions is dropped (the last row of a run repeats the read), so the style and the outcome sums cover the same
    updates. None if the stage is not complete."""
    if stage_status(run) != 'complete' or not (run / 'metrics.jsonl').exists():
        return None
    rows = [json.loads(l) for l in (run / 'metrics.jsonl').read_text().splitlines() if l.strip()]
    rows = [r for r in rows if (r.get('behavior') or {}).get('decisions')]
    if not rows:
        return None
    lo, hi = rows[0]['learner_steps'], rows[-1]['learner_steps']
    rows = [r for r in rows if r['learner_steps'] > lo + (1 - tail) * (hi - lo)
            and ('style' not in r or (r['style'] or {}).get('decisions'))]
    s = dict(rows=len(rows), decisions=0.0, seconds=0.0, kills=0.0, outcome=0.0, drill_presses=0.0,
             style_decisions=0.0, terms={})
    for r in rows:
        b, t = r['behavior'], r['behavior']['totals']
        s['decisions'] += b['decisions']
        s['seconds'] += b['seconds']
        s['kills'] += t['kills']
        s['outcome'] += t['reward_outcome'] + t['reward_kills'] + t['reward_deaths']
        s['drill_presses'] += (r.get('ability_activations') or {}).get(str(DRILL_R2['drill_id']), 0)
        if r.get('style'):
            s['style_decisions'] += r['style']['decisions']
            for name, term in r['style']['terms'].items():
                acc = s['terms'].setdefault(name, dict(paid=0.0, fires=0.0))
                acc['paid'] += term['paid']
                acc['fires'] += term['fires']
    s['kpm'] = s['kills'] / (s['seconds'] / 60) if s['seconds'] else None
    s['style_net'] = sum(t['paid'] for t in s['terms'].values()) if s['terms'] else None
    return s


def drill_surfacings(d):
    """native/rl/research/drill-r2/launch_stats.py's per-surfacing pass, ported: per drill form (a run of consecutive
    decisions of one game in a drill_look decisions.jsonl), every underground run (the logged undelayed SDF bit) and
    the surfacing that ends it. Velocities are finite differences of the player position over one decision (4/60 s).
    Returns (surfacings, drill kills by A). A surfacing has run (decisions underground), first (the form's first),
    v_ug / cos_ug (the last underground step's speed and its cos to the bearing to the opponent: what the style
    sees), v_air / cos_air (the first air step's), dist (to the living opponent at the surfacing, else nan), reach6
    (the flight, until re-entry, comes within 6 u of the living opponent) and kills_form (A's drill kills from the
    surfacing to the end of the form)."""
    dt = 4 / 60.0
    games = {g['g']: g for g in map(json.loads, (d / 'games.jsonl').read_text().splitlines())}
    eps, cur = [], []
    for r in map(json.loads, (d / 'decisions.jsonl').read_text().splitlines()):
        if r['g'] not in games:
            continue
        if cur and (r['g'] != cur[-1]['g'] or r['d'] != cur[-1]['d'] + 1):
            eps.append(cur)
            cur = []
        cur.append(r)
    if cur:
        eps.append(cur)
    out, nkills = [], 0
    for ep in eps:
        g, d0, n = games[ep[0]['g']], ep[0]['d'], len(ep)
        ug = [r['ug'] for r in ep]
        P = np.array([(r['x'], r['y']) for r in ep], dtype=float)
        O = np.array([(r['ox'], r['oy']) for r in ep], dtype=float)
        alive = np.array([bool(r['oalive']) for r in ep])
        V = np.diff(P, axis=0) / dt
        dist = np.hypot(*(O - P).T)
        kills = [x['d'] - d0 for x in g['deaths'] if x['victim'] == 'B' and x['killer'] == 'A'
                 and x['ability'] == DRILL_R2['drill_id'] and d0 <= x['d'] <= d0 + n]
        nkills += len(kills)
        runs, m, start = [], 0, 0
        for i, f in enumerate(ug):
            if f:
                start, m = (i, 1) if not m else (start, m + 1)
            elif m:
                runs.append((start, m))
                m = 0
        if m:
            runs.append((start, m))
        for ri, (s0, m) in enumerate(runs):
            i = s0 + m
            if i >= n or i == 0:
                continue
            bear = O[i] - P[i]
            bd = float(math.hypot(*bear))

            def cos(v):
                nv = float(math.hypot(*v))
                return float(v @ bear) / (nv * bd) if nv > 1e-6 and bd > 1e-6 else float('nan')
            va = V[i] if i < n - 1 else V[i - 1]
            j = i
            while j < n and not ug[j]:
                j += 1
            k = min(j + 1, n)
            span = alive[i:k]
            mind = float(np.min(dist[i:k][span])) if span.any() else float('nan')
            out.append(dict(run=m, first=ri == 0, v_ug=float(math.hypot(*V[i - 1])), cos_ug=cos(V[i - 1]),
                            v_air=float(math.hypot(*va)), cos_air=cos(va), dist=bd if alive[i] else float('nan'),
                            reach6=mind <= 6, kills_form=sum(1 for x in kills if x >= i)))
    return out, nkills


_LAUNCH = {}


def drill_launch(d, rule):
    """L and the launch-arrival rate of one drill_look P1 eval (DRILL-R2 Section 2.4), from drill_surfacings. L's
    denominator is the first surfacings with a living opponent (launch_stats.py's launch_rate); a launch is a first
    surfacing after a run >= l_run decisions whose last underground step is >= l_speed u/s within acos(l_cos) of the
    opponent, any distance; its kill share counts A's drill kills from the launch to the end of the form over all
    drill kills. Far launches (the veto's, launch_stats.py's table) are surfacings > far_dist u from the living
    opponent whose first air step is >= l_speed u/s within acos(l_cos); arrive = they reach 6 u or kill in the form."""
    key = (str(d), id(rule))
    if key in _LAUNCH:
        return _LAUNCH[key]
    if not (d / 'decisions.jsonl').exists():
        return None
    S, nkills = drill_surfacings(d)
    first = [s for s in S if s['first'] and np.isfinite(s['cos_ug']) and np.isfinite(s['dist'])]
    launches = [s for s in first if s['run'] >= rule['l_run'] and s['v_ug'] >= rule['l_speed'] and s['cos_ug'] >= rule['l_cos']]
    far = [s for s in S if np.isfinite(s['dist']) and s['dist'] > rule['far_dist'] and np.isfinite(s['cos_air'])
           and s['cos_air'] >= rule['l_cos'] and s['v_air'] >= rule['l_speed']]
    lk = sum(s['kills_form'] for s in launches)
    frac = lambda xs: sum(xs) / len(far) if far else None
    r = _LAUNCH[key] = dict(
        first=len(first), launches=len(launches), share=len(launches) / len(first) if first else None,
        drill_kills=nkills, launch_kills=lk, kill_share=lk / nkills if nkills else None, far=len(far),
        far_arrive=frac([s['reach6'] or s['kills_form'] > 0 for s in far]),
        far_reach6=frac([s['reach6'] for s in far]), far_kill=frac([s['kills_form'] > 0 for s in far]))
    return r


def drill_r2_vetoes(arm, k_tail, k_run, run, init_launch, rule, base):
    """Section 2.4's hack vetoes for one styled run: training telemetry of its constant stage against K's (kills/min,
    style share, burrow_surface fires <= Drill slot presses), and its P1 eval games (unforced drill deaths per drill
    episode against K's, the far-launch arrival rate against its init's). Each check is (status, text), status
    'pass', 'FAIL', 'pending' or 'not computable'."""
    v = {}
    tail = train_tail(base / arm['name'] / 'constant', rule['tail'])
    if tail is None:
        v['train'] = ('pending', 'constant stage not complete')
    else:
        terms, st = tail['terms'], rule['surface_term']
        if st not in terms:
            v['surface'] = ('not computable', f'{st} is not in this run\'s style.terms')
        else:
            n, presses = terms[st]['fires'], tail['drill_presses']
            v['surface'] = ('pass' if n <= presses else 'FAIL',
                            f'{st} fires {n:.0f} <= Drill slot presses {presses:.0f} (sanity check: presses >= forms)')
        if k_tail is None or not k_tail['kpm']:
            v['kpm'] = ('pending', 'K\'s constant stage not complete')
        else:
            v['kpm'] = ('pass' if tail['kpm'] >= rule['veto_kpm'] * k_tail['kpm'] else 'FAIL',
                        f'kills/min {tail["kpm"]:.3f} >= {rule["veto_kpm"]} x K {k_tail["kpm"]:.3f}')
        if tail['style_net'] is None:
            v['share'] = ('not computable', 'no style.terms logged')
        else:
            v['share'] = ('pass' if tail['style_net'] <= rule['veto_style_share'] * tail['outcome'] else 'FAIL',
                          f'style net {tail["style_net"]:+.1f} <= {rule["veto_style_share"]} x outcome+kill+death '
                          f'{tail["outcome"]:+.1f}')
    if run is None:
        v['eval'] = ('pending', 'no complete P1 eval')
        return tail, v
    unf, k_unf = run['unforced'], k_run and k_run['unforced']
    if not unf:
        v['unforced'] = ('not computable', 'no drill episodes in the P1 eval')
    elif not k_unf:
        v['unforced'] = ('pending', 'no K P1 eval')
    else:
        v['unforced'] = ('pass' if unf[0] <= rule['veto_unforced'] * k_unf[0] else 'FAIL',
                         f'eval unforced drill deaths/drill episode {unf[0]:.4f} <= {rule["veto_unforced"]} x K {k_unf[0]:.4f}')
    L = run.get('launch')
    if not L or L['far_arrive'] is None:
        v['arrive'] = ('not computable', 'no far launches in the P1 eval')
    elif not init_launch or init_launch['far_arrive'] is None:
        v['arrive'] = ('not computable', 'no init P1 eval to set the rate')
    else:
        ok = L['far_arrive'] >= rule['veto_arrive'] * init_launch['far_arrive']
        v['arrive'] = ('pass' if ok else 'FAIL',
                       f'far launches reaching 6 u or killing in the form {L["far_arrive"]:.3f} (reach {L["far_reach6"]:.3f}, '
                       f'kill {L["far_kill"]:.3f}, n {L["far"]}) >= {rule["veto_arrive"]} x init {run["init_run"]} '
                       f'{init_launch["far_arrive"]:.3f}')
    return tail, v


def drill_r2_report(arms, base, p):
    """DRILL-R2 Section 2.4: P1-P4, intent and L per run (drill-r2.sh eval), each run against its init's round-1 eval
    (same seed pairs), the hack vetoes, the showcase ("insanely good" = P1-P4; "ridiculously good with drill" = P1-P4
    plus intent or L), and decisions D1-D4."""
    rule = DRILL_R2
    K, R2, X2, V42 = rule['control'], rule['design'], rule['dose'], rule['ablation']
    out = dict(showcase=None, p1_pass=False, insanely_good=False, ridiculously_good=False, vetoes={}, runs={},
               d1=None, d2=None, d3=None, d4=None)
    p(f'  drill-r2: rules {rule["source"]}')
    out['runs'], pairs, groups, looks = drill_runs(arms, base, p, rule, 'drill-r2.sh', intent_mark=True)
    runs = out['runs']
    f3 = lambda x: '-' if x is None else f'{x:+.3f}'
    pc = lambda x: '-' if x is None else f'{x:.3f}'

    # Before -> after: each run against the round-1 run it was initialised from (same 600 seed pairs, maps, kits).
    before, before_ug, init_launch = {}, {}, {}
    for a in arms:
        init = harness.path(a['init'])
        src = next((q.parent for q in (init, *init.parents) if q.name == 'anneal'), None)
        if src is None or a['name'] not in pairs:
            continue
        runs[a['name']]['init_run'] = src.name
        bp = drill_pairs(src / 'drill-eval-main')
        if bp is None:
            p(f'    {a["name"]:<9} vs init {src.name}: no complete round-1 P1 eval at {src / "drill-eval-main"}')
            continue
        before[a['name']] = bp
        if src.name not in before_ug:
            before_ug[src.name] = drill_look_analyze().analyze(src / 'drill-eval-main').get('ep_ug_ge_1s')
        init_launch[a['name']] = drill_launch(src / 'drill-eval-main', rule)
        b_ug = before_ug[src.name]
        r = paired(pairs[a['name']], bp)
        ug = runs[a['name']]['ug1s']
        p(f'    {a["name"]:<9} - init {src.name} (round 1): P1 {f3(r and r[0])} ±{r[1] if r else 0:.3f} over '
          f'{r[2] if r else 0} seed pairs; P2 ug>=1s {b_ug[0] if b_ug else float("nan"):.3f} -> '
          f'{ug[0] if ug else float("nan"):.3f}')
        runs[a['name']].update(p1_vs_init=r and r[0], ug1s_init=b_ug and b_ug[0])

    # L (launch intent), from the P1 eval's per-surfacing pass; its kill-share bar is 2 x the run's own init's.
    p(f'  L (launch intent): >= {rule["l_share"]:.0%} of first surfacings are launches (run >= {rule["l_run"]} decisions, '
      f'last underground step >= {rule["l_speed"]:.0f} u/s within 30 deg of the opponent) and their drill-kill share '
      f'>= {rule["l_kill_x"]:.0f} x the init\'s')
    for n, r in runs.items():
        L = r['launch'] = drill_launch(base / n / 'drill-eval-main', rule)
        I = init_launch.get(n)
        if L is None or L['share'] is None:
            r['l_pass'] = False
            p(f'    {n:<9} not computable (no decisions.jsonl or no first surfacings)')
            continue
        c1 = L['share'] >= rule['l_share']
        bar = rule['l_kill_x'] * I['kill_share'] if I and I['kill_share'] is not None else None
        c2 = bar is not None and L['kill_share'] is not None and L['kill_share'] >= bar
        r.update(l_share=L['share'], l_kill_share=L['kill_share'], l_pass=c1 and c2)
        p(f'    {n:<9} launches {L["launches"]}/{L["first"]} first surfacings = {L["share"]:.3f} '
          f'{"pass" if c1 else "fail"}; kill share {L["launch_kills"]}/{L["drill_kills"]} = {pc(L["kill_share"])} vs '
          + (f'{rule["l_kill_x"]:.0f} x init {r.get("init_run")} {I["kill_share"]:.3f} = {bar:.3f} {"pass" if c2 else "fail"}'
             if bar is not None else 'init not computable') + f' -> L {"PASS" if r["l_pass"] else "fail"}')

    # Hack vetoes (any one FAIL fails the run as a showcase; "not computable" does not veto).
    p(f'  hack vetoes (training: metrics.jsonl over the last {rule["tail"]:.0%} of the constant stage vs {K}; eval: the '
      f'P1 games vs {K} and vs the init; any FAIL fails the run as a showcase; "not computable" does not veto)')
    k_arm = next((a for a in arms if a['group'] == K), None)
    k_tail = train_tail(base / k_arm['name'] / 'constant', rule['tail']) if k_arm else None
    k_run = runs.get(k_arm['name']) if k_arm else None
    if k_tail:
        p(f'    {K:<9} kills/min {k_tail["kpm"]:.3f} over {k_tail["rows"]} rows, {k_tail["decisions"]:.0f} learner decisions')
    for a in arms:
        if a['group'] == K:
            continue
        _, v = drill_r2_vetoes(a, k_tail, k_run, runs.get(a['name']), init_launch.get(a['name']), rule, base)
        out['vetoes'][a['name']] = {c: s for c, (s, _) in v.items()}
        p(f'    {a["name"]:<9} ' + ' | '.join(f'{c} {s}: {t}' for c, (s, t) in v.items()))
    vetoed = {n for n, v in out['vetoes'].items() if 'FAIL' in v.values()}
    pending = {n for n, v in out['vetoes'].items() if 'pending' in v.values()}

    # Showcase: the styled v4.3 run with the best P1 point among the unvetoed.
    styled = {n: r for n, r in runs.items() if r['group'] in rule['showcase_groups']}
    eligible = {n: r for n, r in styled.items() if n not in vetoed and n not in pending}
    if eligible:
        best = max(eligible, key=lambda n: eligible[n]['p1'][0])
        b = eligible[best]
        out.update(showcase=best, p1_pass=b['p1_pass'], insanely_good=b['all_pass'],
                   ridiculously_good=b['all_pass'] and (b['intent_pass'] or b['l_pass']))
        p(f'    showcase (best styled v4.3 run by P1 point, chosen after the fact among {len(eligible)} unvetoed of '
          f'{len(styled)}): {best}: P1 {"PASS" if out["p1_pass"] else "FAIL"}; "insanely good" (P1-P4) '
          f'{"YES" if out["insanely_good"] else "NO"}; "ridiculously good with drill" (P1-P4 + intent or L) '
          f'{"YES" if out["ridiculously_good"] else "NO"} (intent {"pass" if b["intent_pass"] else "fail"}, '
          f'L {"pass" if b["l_pass"] else "fail"})'
          + (f'; vetoed: {", ".join(sorted(vetoed & set(styled)))}' if vetoed & set(styled) else ''))
    elif styled:
        p(f'    showcase: none (styled v4.3 runs {", ".join(sorted(styled))}: vetoed or vetoes pending)')

    def contrast(t, c):
        (gt, kt), (gc, kc) = drill_group_pairs(pairs, groups, t), drill_group_pairs(pairs, groups, c)
        r = paired(gt, gc) if kt and kc else None
        return r, kt, kc

    def show(t, c, what, r, kt, kc):
        p(f'    {t} - {c} ({what}): ' + ('pending' if r is None else
          f'{r[0]:+.3f} ±{r[1]:.3f} (eval-only 95% over {r[2]} seed pairs; seeds {kt} vs {kc}, seed variance not in the interval)'))

    p('  decisions (seed means paired by seed pair)')
    # D1: the style's strength cost, a veto on the point.
    r, kt, kc = contrast(R2, K)
    show(R2, K, 'D1, style strength cost', r, kt, kc)
    rejected = False
    if r is not None:
        rejected = r[0] < rule['d1']
        noise = abs(r[0]) <= r[1]
        verdict = 'REJECTED (R2x2 and R2v42 informational only)' if rejected else 'stays'
        out['d1'] = dict(diff=r[0], half_width=r[1], verdict=verdict, rejected=rejected, within_noise=noise)
        p(f'    -> D1: the style {verdict} (stays if R2 - K >= {rule["d1"]} on the point)'
          + ('; within noise (the eval-only interval contains 0)' if noise else ''))
    info = ' [informational only: D1 rejected the style]' if rejected else ''
    r2_runs, x2_runs, v42_runs = groups.get(R2, []), groups.get(X2, []), groups.get(V42, [])

    # D2: dose.
    if r2_runs and x2_runs:
        r2_mean = float(np.mean([runs[n]['p1'][0] for n in r2_runs]))
        x2 = runs[x2_runs[0]]
        adopt = x2['p2_pass'] and not any(runs[n]['p2_pass'] for n in r2_runs) and x2['p1'][0] >= r2_mean + rule['d2_p1']
        out['d2'] = dict(adopt_x2=adopt, r2_p1_mean=r2_mean, x2_p1=x2['p1'][0], x2_p2=x2['p2_pass'],
                         r2_p2=[runs[n]['p2_pass'] for n in r2_runs])
        p(f'    -> D2: {"ADOPT the x2 scale" if adopt else "base scale"}{info} ({X2} P2 {"PASS" if x2["p2_pass"] else "fail"}, '
          f'{R2} seeds P2 {"/".join("PASS" if runs[n]["p2_pass"] else "fail" for n in r2_runs)}; {X2} P1 {x2["p1"][0]:.3f} vs '
          f'{R2} mean {r2_mean:.3f} {rule["d2_p1"]:+.2f})')
    else:
        p('    -> D2: pending')

    # D3: the senses, on the paired gains net of each arm's init (R2 from SO-s7121, R2v42 from S-s7111).
    r, kt, kc = contrast(R2, V42)
    show(R2, V42, 'raw contrast, report only: carries the init gap', r, kt, kc)
    d_r2, d_v = [n for n in r2_runs if n in before], [n for n in v42_runs if n in before]
    if d_r2 and d_v:
        common = set.intersection(*(set(pairs[n]) & set(before[n]) for n in d_r2 + d_v))
        delta = lambda names: {k: float(np.mean([pairs[n][k] - before[n][k] for n in names])) for k in common}
        dd = paired(delta(d_r2), delta(d_v))
        ug = lambda n: (runs[n]['ug1s'] or (float('nan'),))[0] - (runs[n].get('ug1s_init') if runs[n].get('ug1s_init') is not None else float('nan'))
        g2 = float(np.mean([ug(n) for n in d_r2])) - float(np.mean([ug(n) for n in d_v]))
        g1 = dd[0] if dd else float('nan')
        keep = g2 >= rule['d3_keep_p2'] or g1 >= rule['d3_keep_p1']
        not_bn = not keep and abs(g2) <= rule['d3_near_p2'] and abs(g1) <= rule['d3_near_p1']
        verdict = ('v4.3 KEPT for the specialist' if keep else
                   'the senses are not the bottleneck (report; v4.3 stays for the main-bot program)' if not_bn else
                   'undecided: round 3 carries both obs')
        out['d3'] = dict(g_p1=g1, g_p1_half_width=dd and dd[1], g_p2=g2, raw=r and r[0], keep_v43=keep,
                         senses_not_bottleneck=not_bn, verdict=verdict)
        p(f'    G_P1 = (R2 - init) - (R2v42 - init) on P1: {g1:+.3f} ±{dd[1] if dd else 0:.3f} over {dd[2] if dd else 0} '
          f'seed pairs (winner\'s curse: biased about -0.01, against keeping); G_P2 (P2 ug>=1s fraction): {g2:+.4f}')
        p(f'    -> D3: {verdict}{info} (keep if G_P2 >= +{rule["d3_keep_p2"]} or G_P1 >= +{rule["d3_keep_p1"]}; not the '
          f'bottleneck if |G_P2| <= {rule["d3_near_p2"]} and |G_P1| <= {rule["d3_near_p1"]})')
    else:
        p('    -> D3: pending (needs R2 and R2v42 P1 evals and their inits\' round-1 P1 evals)')

    # D4: what round 3 is (P2's >= 1 s fraction; binomial se over drill episodes); L's first clause alongside.
    pt = lambda n: runs[n]['ug1s'][0] if runs[n]['ug1s'] else float('nan')
    se2 = lambda n: pt(n) * (1 - pt(n)) / max(1, runs[n]['episodes'])
    k_runs = groups.get(K, [])
    if k_runs and r2_runs and x2_runs:
        k_ug, x_ug = pt(k_runs[0]), pt(x2_runs[0])
        r2_ug = float(np.mean([pt(n) for n in r2_runs]))
        diff = x_ug - r2_ug
        half = 1.96 * math.sqrt(se2(x2_runs[0]) + sum(se2(n) for n in r2_runs) / len(r2_runs) ** 2)
        above = diff - half > 0
        if not above:
            case = ('the dose is not the lever: round 3 is the sense in Section 2.2 under bit 64 against v4.3 with the '
                    'same reward' if k_ug < r2_ug else
                    f'the dose is not the lever and {R2} is not above {K}: the admission in Section 1.3')
        elif x_ug >= rule['p2_ug1s']:
            case = f'{X2} above {R2} but reaches {rule["p2_ug1s"]:.0%}: no dose-sweep case (the dose reaches the P2 bar)'
        elif k_ug < r2_ug:
            case = 'the reward is the lever: round 3 is a dose sweep (x4) with the same hack tests'
        else:
            case = f'no D4 case fires ({X2} above {R2} but {K} >= {R2})'
        ls = lambda names: [runs[n].get('l_share') for n in names]
        lk, lr, lx = ls(k_runs), ls(r2_runs), ls(x2_runs)
        l_r2 = float(np.mean(lr)) if lr and None not in lr else None
        out['d4'] = dict(case=case, k=k_ug, r2=r2_ug, x2=x_ug, x2_minus_r2=diff, half_width=half,
                         l_share=dict(k=lk[0], r2=l_r2, x2=lx[0]))
        p(f'    -> D4: P2 ug>=1s {K} {k_ug:.3f}, {R2} {r2_ug:.3f}, {X2} {x_ug:.3f}; {X2} - {R2} {diff:+.3f} ±{half:.3f} '
          f'(binomial over drill episodes; "above" = interval excludes 0): {case}; L launch share alongside: '
          f'{K} {pc(lk[0])}, {R2} {pc(l_r2)}, {X2} {pc(lx[0])}')
    else:
        p('    -> D4: pending')
    return out


# ---- power --------------------------------------------------------------------

def mde_two_group(s2, ev_a, ka, ev_c, kc, alpha, sided=2):
    se = math.sqrt((s2 + ev_a) / ka + (s2 + ev_c) / kc)
    return (stats.norm.ppf(1 - alpha / sided) + Z80) * se, se


def power_report(comps, r_crn, data, arms, prior, p):
    ctl = next(iter(comps))[0]
    per_game = []
    for a in arms:
        for e in a['endpoint_names']:
            v = panel(data, a, e)
            if v:
                per_game.append(v[1] * sum(a['evals'][o]['games'] for o in a['panel']))
    if not per_game:
        p('-- power: no complete panel yet')
        return
    c_fresh = float(np.mean(per_game))
    _, n_cells = crn_correlation(data, arms)
    r_plan = r_crn if r_crn is not None and n_cells >= 4 else 0.0   # credit CRN once a whole panel has 2 nets
    c = c_fresh * (1 - r_plan)
    ev = lambda n: c / n
    an = comps.get((ctl, 'anneal'), {})
    plan = (sd(max(an['s2_ub80'], 0)), f'80% UB of s_seed(anneal), k {an["k"]}') if 's2_ub80' in an else \
        (prior, f'prior (k {an.get("k", 0)} < 2)')
    grid = sorted({round(plan[0], 4), *PRIOR_GRID})
    p('')
    p(f'-- power: eval var per game {c:.3f} (fresh setups {c_fresh:.3f}, binomial 0.25; CRN r credited {r_plan:.2f}'
      + (f', measured {r_crn:.2f} on {n_cells} cells' if r_crn is not None else '') + ')')
    p(f'   planning s_seed {plan[0]:.4f} = {plan[1]}; columns are s_seed; MDE at 80% power')
    p('   ' + ' ' * 48 + ''.join(f'{s:>7.3f}' for s in grid))
    alpha_bh = (0.10 / 7, 0.10)

    def row(label, fn):
        vals = [fn(s * s) for s in grid]
        p(f'   {label:<48}' + ''.join(f'{v:>7d}' if isinstance(v, int) else f'{v:>7.3f}' for v in vals))

    for lab, al in zip(('1 true of 7, a q/m', 'all true, a q'), alpha_bh):
        row(f'ff1 32x2000 g main effect ({lab})',
            lambda s2, al=al: ff_mde(s2, ev(2000), 32, al))
    row('ff1 power at true +0.02 (BH q/m and >= +0.02)', lambda s2: ff_power(0.02, s2, ev(2000), 32, 0.10 / 7))
    row('ff1 power at true +0.03', lambda s2: ff_power(0.03, s2, ev(2000), 32, 0.10 / 7))
    row('confirm 4v4 x 4300 g, MDE one-sided 0.05', lambda s2: mde_two_group(s2, ev(4300), 4, ev(4300), 4, 0.05, 1)[0])
    row('confirm 4v4, Holm 2 candidates (0.025)', lambda s2: mde_two_group(s2, ev(4300), 4, ev(4300), 4, 0.025, 1)[0])
    row('confirm 4v4 x 8600 g', lambda s2: mde_two_group(s2, ev(8600), 4, ev(8600), 4, 0.05, 1)[0])
    row('confirm 8v8 x 4300 g', lambda s2: mde_two_group(s2, ev(4300), 8, ev(4300), 8, 0.05, 1)[0])
    for ka, kc in ((4, 4), (4, 6)):
        row(f'I1/I3 NI -0.02 power at 0, {ka}v{kc} (2000/4300 g)',
            lambda s2, ka=ka, kc=kc: ni_power(0.0, -0.02, s2, ev(2000), ka, ev(4300), kc))
        row(f'I1/I3 NI margin for 80% power at 0, {ka}v{kc}',
            lambda s2, ka=ka, kc=kc: mde_two_group(s2, ev(2000), ka, ev(4300), kc, 0.05, 1)[0])
    row('I1/I3 NI -0.02 power at 0, 6v6 x 4300 g', lambda s2: ni_power(0.0, -0.02, s2, ev(4300), 6, ev(4300), 6))
    row('E8 3v3 x 4300 g, MDE two-sided 0.05', lambda s2: mde_two_group(s2, ev(4300), 3, ev(4300), 3, 0.05)[0])
    row('seeds/arm for confirm MDE 0.02 at 4300 g', lambda s2: seeds_needed(0.02, s2, ev(4300)))
    row('seeds/arm for confirm MDE 0.02 at inf games', lambda s2: seeds_needed(0.02, s2, 0.0))
    if r_crn and not r_plan:
        ev = lambda n: c_fresh * (1 - r_crn) / n
        p(f'   if CRN r {r_crn:.2f} holds:')
        row('  ff1 MDE (one true of 7)', lambda s2: ff_mde(s2, ev(2000), 32, 0.10 / 7))
        row('  confirm 4v4 x 4300 g', lambda s2: mde_two_group(s2, ev(4300), 4, ev(4300), 4, 0.05, 1)[0])
        row('  I1/I3 NI -0.02 power at 0, 4v6', lambda s2: ni_power(0.0, -0.02, s2, ev(2000), 4, ev(4300), 6))


def ff_mde(s2, ev, n, alpha):
    """True main effect with 80% power to pass both BH (as a per-test alpha) and the >= +0.02 estimate gate."""
    se = 2 * math.sqrt(s2 + ev) / math.sqrt(n)
    return max(0.02, stats.norm.ppf(1 - alpha / 2) * se) + Z80 * se


def ff_power(delta, s2, ev, n, alpha):
    se = 2 * math.sqrt(s2 + ev) / math.sqrt(n)
    return stats.norm.sf((max(0.02, stats.norm.ppf(1 - alpha / 2) * se) - delta) / se)


def ni_power(delta, margin, s2, ev_a, ka, ev_c, kc):
    se = math.sqrt((s2 + ev_a) / ka + (s2 + ev_c) / kc)
    return stats.norm.cdf((delta - margin) / se - stats.norm.ppf(0.95))


def seeds_needed(delta, s2, ev):
    return math.ceil(2 * (stats.norm.ppf(0.95) + Z80) ** 2 * (s2 + ev) / delta ** 2)


def reference_components(program, directory):
    """The reference program's control group at the annealed endpoint (E0 sizes everything after it)."""
    name, top, arms = program_spec(program)
    data = Data(directory or harness.path(top.get('out', f'native/artifacts/rl/research/{name}')), arms)
    r_crn, _ = crn_correlation(data, arms)
    groups = {}
    for a in arms:
        groups.setdefault(a['group'], []).append(a)
    ctrl = next(g for g in groups if not groups[g][0].get('compare') and not groups[g][0].get('factors'))
    return dict(seed_components(data, groups[ctrl], 'anneal', r_crn), program=name, group=ctrl), data, arms, r_crn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('program', help='program name (train/research/arms/<name>.toml) or a TOML path')
    ap.add_argument('--dir', type=Path, help='results directory (default: the TOML out, under this checkout)')
    ap.add_argument('--prior-sseed', type=float, default=0.015,
                    help='s_seed used until >= 2 control seeds land (PLAN E0 hypothesis boundary)')
    ap.add_argument('--reference', default='e0', help='program whose control seeds give s_seed for power and '
                                                      'the factorial external se (default e0)')
    ap.add_argument('--reference-dir', type=Path)
    ap.add_argument('--json', type=Path, help='also write the components and effects here')
    a = ap.parse_args(argv)
    program, top, arms = program_spec(a.program)
    out = a.dir or harness.path(top.get('out', f'native/artifacts/rl/research/{program}'))
    data = Data(out, arms)
    ref = None
    if a.reference and a.reference != program:
        ref = reference_components(a.reference, a.reference_dir)
    lines = []
    comps, r_crn = report(program, top, arms, data, a.prior_sseed, lines, ref and ref[0], ref and (ref[1], ref[2]))
    if ref:   # power is planned from the reference program's control seeds and eval noise
        comps, r_crn, data, arms = {(ref[0]['group'], 'anneal'): ref[0]}, ref[3], ref[1], ref[2]
        lines.append(f'\n(power from reference program {ref[0]["program"]}, group {ref[0]["group"]})')
    power_report(comps, r_crn, data, arms, a.prior_sseed, lines.append)
    print('\n'.join(lines))
    if a.json:
        a.json.write_text(json.dumps(dict(program=program, r_crn=r_crn, missing=data.missing, decisions=DECISIONS.get(program),
                                          components={f'{g}:{e}': c for (g, e), c in comps.items()}),
                                     indent=1, default=float) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
