"""Read-only one-screen summary of a training run's learning signals.

Reads <run>/status.json, config.json, metrics.jsonl and evaluations/*/result.json
(a torn final metrics line is skipped) and prints milestone rows every
--every learner decisions (physical decisions for legacy runs without learner counts)
plus every evaluation. Never touches the run, the
GPU or the game. Written for the sparse-reward null-hypothesis run: the
watch rule is "joint entropy still near uniform and evaluation at chance",
so the last row prints both against the schedule's coefficient.

  python native/rl/autoresearch/watch_run.py <run> [--every 5000000] [--json]
"""
import argparse
import json
import math
from pathlib import Path

MAX_ENTROPY = math.log(256) + math.log(16)  # aim x buttons, nats


def load_metrics(path):
    rows = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1: raise
    for row in rows:
        row['summary_decisions'] = row.get('learner_steps', row['step'])
        row['summary_sps'] = row.get('learner_sps', row.get('sps', float('nan')))
    return rows


def milestones(rows, every):
    """First update at or above each crossed milestone, plus the last update."""
    out, target = [], every
    for row in rows:
        if row['summary_decisions'] >= target:
            out.append(row); target = (row['summary_decisions'] // every + 1) * every
    if not out or out[-1] is not rows[-1]: out.append(rows[-1])
    return out


def window(rows, index, size=50):
    lo = max(0, index-size+1); block = rows[lo:index+1]
    def mean(key):
        values = [r[key] for r in block if isinstance(r.get(key), (int, float))]
        return sum(values)/len(values) if values else float('nan')
    return {k: mean(k) for k in ('entropy', 'entropy_coefficient', 'explained_variance', 'gradient_norm',
                                 'advantage_std', 'kl', 'clip_fraction', 'summary_sps')}


def evaluations(run, config):
    for result in sorted((run/'evaluations').glob('*/result.json')) if (run/'evaluations').exists() else []:
        data = json.loads(result.read_text())
        groups = data.get('groups', [])
        def pooled(opponent, maps=None):
            rows = [g for g in groups if g['opponent'] == opponent and (maps is None or g['map'] in maps)]
            games = sum(g['games'] for g in rows); wins = sum(g['wins'] for g in rows)
            return (wins/games if games else float('nan'), games)
        training = set(data.get('training_maps') or data.get('evaluation_maps') or config.get('training_maps') or [])
        yield dict(step=data.get('step'), games=len(data.get('games', [])),
                   random=pooled('random', training or None), initial=pooled('initial', training or None),
                   heldout=pooled('random', {g['map'] for g in groups}-training) if training else (float('nan'), 0),
                   timeouts=sum(g.get('timeouts', 0) for g in groups))


def summarize(run, every):
    if every <= 0: raise ValueError('--every must be positive')
    run = Path(run)
    status = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
    config = json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else {}
    rows = load_metrics(run/'metrics.jsonl')
    if not rows: raise SystemExit(f'{run}: no metrics yet')
    index = {id(r): i for i, r in enumerate(rows)}
    table = [dict(decisions=r['summary_decisions'], step=r['step'], update=r['update'], elapsed=r['elapsed'],
                  **window(rows, index[id(r)])) for r in milestones(rows, every)]
    last = rows[-1]
    rate = last['summary_decisions']/last['elapsed'] if last['elapsed'] else float('nan')
    has_learner_counts = all('learner_steps' in row for row in rows)
    unit = 'learner' if has_learner_counts else 'physical'
    target = ((status.get('target_learner_steps') or config.get('learner_steps')) if has_learner_counts
              else (status.get('target_steps') or config.get('steps')))
    return dict(run=str(run), status=status.get('status'), decisions=last['summary_decisions'], decision_unit=unit, target=target,
                elapsed_hours=last['elapsed']/3600, decisions_per_second=rate,
                hours_remaining=(max(0, target-last['summary_decisions'])/rate/3600) if target and rate else None,
                learner_passes=config.get('learner_passes'), entropy_half_life=config.get('entropy_half_life'),
                learning_rate_schedule=config.get('learning_rate_schedule'), max_entropy=MAX_ENTROPY,
                milestones=table, evaluations=list(evaluations(run, config)))


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)): return '-'
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path)
    parser.add_argument('--every', type=int, default=5_000_000, help='Milestone spacing (learner decisions, or physical for legacy runs)')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    s = summarize(args.run, args.every)
    if args.json: print(json.dumps(json_safe(s), indent=1, allow_nan=False)); return
    print(f"{s['run']}: {s['status']}, {s['decisions']:,} of {s['target'] or '?'} {s['decision_unit']} decisions, "
          f"{s['elapsed_hours']:.2f} h, {s['decisions_per_second']:,.0f}/s"
          + (f", ~{s['hours_remaining']:.1f} h left" if s['hours_remaining'] else ''))
    print(f"learner passes {fmt(s['learner_passes'])} (at 1, pre-step kl/clip do not measure the resulting policy change), entropy half-life "
          f"{fmt(s['entropy_half_life'])}, lr {fmt(s['learning_rate_schedule'])}; uniform entropy = {MAX_ENTROPY:.3f} nats")
    print(f"{s['decision_unit']:>12} {'entropy':>8} {'coef':>7} {'ev':>6} {'gnorm':>6} {'advstd':>6} {'kl':>7} {'clip':>5} {'dps':>6}")
    print('Signals are trailing 50-update means; evaluation steps are physical decisions.')
    for m in s['milestones']:
        print(f"{m['decisions']:>12,} {fmt(m['entropy'])} {fmt(m['entropy_coefficient'], 5)} {fmt(m['explained_variance'])}"
              f" {fmt(m['gradient_norm'])} {fmt(m['advantage_std'])} {fmt(m['kl'], 5)} {fmt(m['clip_fraction'])} {m['summary_sps']:>6,.0f}")
    for e in s['evaluations']:
        print(f"eval physical {e['step']:>12,}: vs random {fmt(e['random'][0])} ({e['random'][1]}g), vs initial {fmt(e['initial'][0])}"
              f" ({e['initial'][1]}g), held-out vs random {fmt(e['heldout'][0])} ({e['heldout'][1]}g), timeouts {e['timeouts']}, {e['games']} games")
    m = s['milestones'][-1]
    near_uniform = m['entropy'] > MAX_ENTROPY-0.1
    print(f"watch: entropy {'near uniform' if near_uniform else 'moving'} ({m['entropy']:.3f}), "
          f"coefficient {fmt(m['entropy_coefficient'], 5)}"
          + (f", last eval vs random {fmt(s['evaluations'][-1]['random'][0])}" if s['evaluations'] else ''))


if __name__ == '__main__':
    main()
