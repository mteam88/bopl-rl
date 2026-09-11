"""Behaviour flags: labels derived from an evaluation's behaviour totals.

Flags describe *how* a policy plays (never fires, spams platforms, stalls to
the timeout) so a high score can be read with its behaviour next to it. They
never enter the score: a flagged trial keeps its rating, Protein never sees
them. Thresholds live in the sweep TOML:

    [scoring.flags]
    no-offense = "arrows_per_game < 0.05"
    platform-spam = "platforms_per_game > 8"
    stalemate = "timeout_rate > 0.5"

Metrics come from evaluate.py's per-group behavior_totals (research/telemetry.py
FIELDS) pooled over every non-selfplay group on the evaluation maps.
"""
import json
import re
from pathlib import Path

DEFAULT_RULES = {
    'no-offense': 'arrows_per_game < 0.05',
    'platform-spam': 'platforms_per_game > 8',
    'stalemate': 'timeout_rate > 0.5',
}

METRICS = ('arrows_per_game', 'arrow_hits_per_game', 'arrow_hit_rate', 'platforms_per_game',
           'dashes_per_game', 'bows_per_game', 'kills_per_game', 'deaths_per_game',
           'timeout_rate', 'win_rate', 'mean_duration_seconds', 'games')

_RULE = re.compile(r'\s*([a-z_]+)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)\s*$')


def parse_rule(text):
    """'metric < 0.05' -> (metric, op, value); ValueError for unknown metrics."""
    match = _RULE.fullmatch(str(text))
    if not match or match.group(1) not in METRICS:
        raise ValueError(f'Flag rule {text!r} must be "<metric> <|<=|>|>= <number>" with a metric from {METRICS}')
    return match.group(1), match.group(2), float(match.group(3))


def parse_rules(rules):
    return {name: parse_rule(text) for name, text in (rules or DEFAULT_RULES).items()}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def evaluation_metrics(report, include_heldout=False):
    """Behaviour rates pooled over the report's anchor groups (not selfplay)."""
    groups = [g for g in report.get('groups', ())
              if g['opponent'] != 'selfplay' and (include_heldout or g.get('map_role') != 'untrained')]
    totals = {}
    for group in groups:
        for key, value in group.get('behavior_totals', {}).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    games = sum(g['games'] for g in groups)
    kills = sum(g.get('mean_kills', 0.0) * g['games'] for g in groups)
    deaths = sum(g.get('mean_deaths', 0.0) * g['games'] for g in groups)
    duration = sum(g.get('mean_duration', 0.0) * g['games'] for g in groups)
    return dict(
        games=games,
        arrows_per_game=_ratio(totals.get('arrows_fired', 0.0), games),
        arrow_hits_per_game=_ratio(totals.get('arrow_hits', 0.0), games),
        arrow_hit_rate=_ratio(totals.get('arrow_hits', 0.0), totals.get('arrows_fired', 0.0)),
        platforms_per_game=_ratio(totals.get('platform_activations', 0.0), games),
        dashes_per_game=_ratio(totals.get('dash_activations', 0.0), games),
        bows_per_game=_ratio(totals.get('bow_activations', 0.0), games),
        kills_per_game=_ratio(kills, games),
        deaths_per_game=_ratio(deaths, games),
        timeout_rate=_ratio(sum(g.get('timeouts', 0) for g in groups), games),
        win_rate=_ratio(sum(g['wins'] for g in groups), games),
        mean_duration_seconds=_ratio(duration, games),
    )


def raised(metrics, rules):
    """Names of the rules whose comparison holds; no games raises nothing."""
    if not metrics.get('games'):
        return []
    compare = {'<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
               '>': lambda a, b: a > b, '>=': lambda a, b: a >= b}
    return [name for name, (metric, op, value) in parse_rules(rules).items()
            if compare[op](metrics[metric], value)]


def annotate(report, rules=None, include_heldout=False):
    """flags/flag_metrics/flag_rules for one evaluation report (not written)."""
    metrics = evaluation_metrics(report, include_heldout)
    return dict(flags=raised(metrics, rules), flag_metrics=metrics,
                flag_rules=dict(rules or DEFAULT_RULES))


def annotate_run(run_dir, rules=None, include_heldout=False, write=True):
    """Flag every evaluations/*/result.json of a run; with write, store the
    flags in each report and in the run's result.json. Returns
    [(evaluation directory name, annotation)] in step order."""
    run_dir = Path(run_dir)
    rows = []
    for result_file in sorted(run_dir.glob('evaluations/*/result.json'),
                              key=lambda p: int(re.sub(r'\D', '', p.parent.name) or 0)):
        try:
            report = json.loads(result_file.read_text())
        except ValueError:
            continue
        annotation = annotate(report, rules, include_heldout)
        rows.append((result_file.parent.name, annotation))
        if write and any(report.get(key) != value for key, value in annotation.items()):
            report.update(annotation)
            _atomic_json(result_file, report)
    if write and rows:
        result_file = run_dir / 'result.json'
        try:
            result = json.loads(result_file.read_text())
        except (FileNotFoundError, ValueError):
            result = {}
        result.update(rows[-1][1], flags_by_evaluation={name: a['flags'] for name, a in rows})
        _atomic_json(result_file, result)
    return rows


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, allow_nan=False) + '\n')
    temporary.replace(path)
