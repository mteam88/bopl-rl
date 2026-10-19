"""Turn a research program's results into decisions (PLAN-2026-10.md, REVIEW-2026-10-cursor.md, POWER-2026-10.md).

    venv/bin/python train/research/analyze.py e0 [--dir DIR] [--prior-sseed 0.015] [--json OUT]

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
    arms = []
    for name, table in tables:
        cfg = {**top, **table}
        for seed in cfg.pop('seeds', None) or [cfg['seed']]:
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
    if ctrl and ('ema' in endpoints):
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
    (I7: sign-off); secondary main effects and interactions are reported, no decision."""
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
    p('    I1/I3 auto-adopt rule (eval/kill_rate.py factorial_rule: ratio >= 1.20, CI > 1, Holm over I1/I3/I7; I7 needs sign-off):')
    adopt = kr.factorial_rule(results)
    for line in kr.report(adopt, 'the low level').splitlines():
        p('      ' + line)
    if sum(v['adopt'] for k, v in adopt.items() if k.split()[0] in ('I1', 'I3')) == 2:
        p('    I1 and I3 both pass: run I1+I3 combined against each alone before adopting both (sign-off 2026-10-19)')
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
        a.json.write_text(json.dumps(dict(program=program, r_crn=r_crn, missing=data.missing,
                                          components={f'{g}:{e}': c for (g, e), c in comps.items()}),
                                     indent=1, default=float) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
