"""Read-only one-screen summary of a training run's learning signals.

Reads <run>/status.json, config.json, metrics.jsonl and evaluations/*/result.json
(a torn final metrics line is skipped) and prints milestone rows every
--every learner decisions plus every evaluation. Never touches the run, the
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
    with path.open() as stream:
        for line in stream:
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue  # torn final line
    for row in rows:  # historical torch runs counted only physical decisions
        row.setdefault('learner_steps', row['step'])
        row.setdefault('learner_sps', row.get('sps', float('nan')))
    return rows


def milestones(rows, every):
    """Last update at or below each multiple of `every` learner decisions, plus the last update."""
    out, target = [], every
    for row in rows:
        if row['learner_steps'] >= target:
            out.append(row); target += every
    if not out or out[-1] is not rows[-1]: out.append(rows[-1])
    return out


def window(rows, index, size=50):
    lo = max(0, index-size+1); block = rows[lo:index+1]
    def mean(key):
        values = [r[key] for r in block if isinstance(r.get(key), (int, float))]
        return sum(values)/len(values) if values else float('nan')
    return {k: mean(k) for k in ('entropy', 'entropy_coefficient', 'explained_variance', 'gradient_norm',
                                 'advantage_std', 'kl', 'clip_fraction', 'learner_sps')}


def evaluations(run):
    for result in sorted((run/'evaluations').glob('*/result.json')) if (run/'evaluations').exists() else []:
        data = json.loads(result.read_text())
        groups = data.get('groups', [])
        def pooled(opponent, maps=None):
            rows = [g for g in groups if g['opponent'] == opponent and (maps is None or g['map'] in maps)]
            games = sum(g['games'] for g in rows); wins = sum(g['wins'] for g in rows)
            return (wins/games if games else float('nan'), games)
        training = set(data.get('training_maps') or [])
        yield dict(step=data.get('step'), games=len(data.get('games', [])),
                   random=pooled('random', training), initial=pooled('initial', training),
                   heldout=pooled('random', {g['map'] for g in groups}-training) if training else (float('nan'), 0),
                   timeouts=sum(g.get('timeouts', 0) for g in groups))


def summarize(run, every):
    run = Path(run)
    status = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
    config = json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else {}
    rows = load_metrics(run/'metrics.jsonl')
    if not rows: raise SystemExit(f'{run}: no metrics yet')
    index = {id(r): i for i, r in enumerate(rows)}
    table = [dict(learner_steps=r['learner_steps'], step=r['step'], update=r['update'], elapsed=r['elapsed'],
                  **window(rows, index[id(r)])) for r in milestones(rows, every)]
    last = rows[-1]
    rate = last['learner_steps']/last['elapsed'] if last['elapsed'] else float('nan')
    target = status.get('target_learner_steps') or config.get('learner_steps')
    return dict(run=str(run), status=status.get('status'), learner_steps=last['learner_steps'], target=target,
                elapsed_hours=last['elapsed']/3600, learner_per_second=rate,
                hours_remaining=((target-last['learner_steps'])/rate/3600) if target and rate else None,
                learner_passes=config.get('learner_passes'), entropy_half_life=config.get('entropy_half_life'),
                learning_rate_schedule=config.get('learning_rate_schedule'), max_entropy=MAX_ENTROPY,
                milestones=table, evaluations=list(evaluations(run)))


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)): return '-'
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path)
    parser.add_argument('--every', type=int, default=5_000_000, help='Milestone spacing in learner decisions')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    s = summarize(args.run, args.every)
    if args.json: print(json.dumps(s, indent=1)); return
    print(f"{s['run']}: {s['status']}, {s['learner_steps']:,} of {s['target'] or '?'} learner decisions, "
          f"{s['elapsed_hours']:.2f} h, {s['learner_per_second']:,.0f}/s"
          + (f", ~{s['hours_remaining']:.1f} h left" if s['hours_remaining'] else ''))
    print(f"learner passes {s['learner_passes']} (kl/clip are 0 by construction at 1), entropy half-life "
          f"{s['entropy_half_life']}, lr {s['learning_rate_schedule']}; uniform entropy = {MAX_ENTROPY:.3f} nats")
    print(f"{'learner':>12} {'entropy':>8} {'coef':>7} {'ev':>6} {'gnorm':>6} {'advstd':>6} {'kl':>7} {'clip':>5} {'lps':>6}")
    for m in s['milestones']:
        print(f"{m['learner_steps']:>12,} {fmt(m['entropy'])} {fmt(m['entropy_coefficient'], 5)} {fmt(m['explained_variance'])}"
              f" {fmt(m['gradient_norm'])} {fmt(m['advantage_std'])} {fmt(m['kl'], 5)} {fmt(m['clip_fraction'])} {m['learner_sps']:>6,.0f}")
    for e in s['evaluations']:
        print(f"eval {e['step']:>12,}: vs random {fmt(e['random'][0])} ({e['random'][1]}g), vs initial {fmt(e['initial'][0])}"
              f" ({e['initial'][1]}g), held-out vs random {fmt(e['heldout'][0])} ({e['heldout'][1]}g), timeouts {e['timeouts']}, {e['games']} games")
    m = s['milestones'][-1]
    near_uniform = m['entropy'] > MAX_ENTROPY-0.1
    print(f"watch: entropy {'near uniform' if near_uniform else 'moving'} ({m['entropy']:.3f}), "
          f"coefficient {m['entropy_coefficient']:.5f}"
          + (f", last eval vs random {fmt(s['evaluations'][-1]['random'][0])}" if s['evaluations'] else ''))


if __name__ == '__main__':
    main()
