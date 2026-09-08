"""Read-only one-screen summary of a training run's learning signals.

Reads <run>/status.json, config.json, metrics.jsonl and evaluations/*/result.json
(a torn final metrics line is skipped) and prints milestone rows every
--every learner decisions (physical decisions for legacy runs without learner counts)
plus every evaluation. Never touches the run, the
GPU or the game. Written for the sparse-reward null-hypothesis run: the
watch rule is "joint entropy still near uniform and evaluation at chance",
so the last row prints both against the schedule's coefficient.

  python native/rl/autoresearch/watch_run.py <run> [--every 5000000] [--json] [--drift]

--drift also reads <run>/checkpoints/*.bin (native float32 weights, CPU numpy only)
and prints, per weight slice, how far the weights moved between consecutive
checkpoints and how persistent that motion is: `lag1` is the mean cosine
between consecutive increments, `net/rw` the net displacement over the whole
series divided by the random-walk expectation sqrt(n)*mean|increment|. Near
1 and ~0 means the slice is doing a random walk (noise-dominated gradient);
well above 1 with a high cosine means it is moving in a consistent direction.
`row` is the mean per-row norm of the slice (decoder slices only; rows are
hidden-wide), first and last checkpoint: Muon whitens the whole 273-row
decoder, so a row whose gradient is consistent grows while noise rows keep
their initial norm (notes.md 2026-09-07 22:02).
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


def weight_slices(meta):
    """Named slices of the flat native weight vector (layout: encoder, decoder [aim 256 | buttons 16 | value 1] x hidden, MinGRU)."""
    hidden, obs, layers = meta['hidden'], meta['obs_size'], meta['num_layers']
    if meta.get('architecture') == 'bopl-set-mingru-v1':
        encoder = hidden*905 + 64*49 + hidden*(hidden+128)
    else:
        encoder = hidden*obs
    decoder = hidden*273
    total = encoder + decoder + layers*3*hidden**2
    return total, [('encoder', 0, encoder), ('decoder.aim', encoder, encoder+256*hidden),
                   ('decoder.buttons', encoder+256*hidden, encoder+272*hidden),
                   ('decoder.value', encoder+272*hidden, encoder+decoder), ('mingru', encoder+decoder, total)]


def drift(run):
    """Per-slice displacement statistics over the run's checkpoints, oldest to newest."""
    import os
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(var, '1')  # never compete with a live training run's workers
    import numpy as np
    files = sorted(Path(run, 'checkpoints').glob('policy-*.bin'))
    if len(files) < 3: raise SystemExit(f'{run}: need at least three native checkpoints for drift')
    metas = [json.loads(Path(str(f)+'.json').read_text()) for f in files]
    total, slices = weight_slices(metas[0])
    hidden = metas[0]['hidden']
    weights = []
    for f in files:
        w = np.fromfile(f, dtype='<f4')
        if w.size != total: raise SystemExit(f'{f}: {w.size} weights, expected {total}')
        weights.append(w)
    rows = []
    for name, lo, hi in slices:
        inc = np.stack([weights[i][lo:hi]-weights[i-1][lo:hi] for i in range(1, len(weights))])
        norms = np.linalg.norm(inc, axis=1)
        unit = inc/np.maximum(norms, 1e-12)[:, None]
        lag1 = float(np.mean(np.sum(unit[:-1]*unit[1:], axis=1)))
        net = float(np.linalg.norm(weights[-1][lo:hi]-weights[0][lo:hi]))
        walk = math.sqrt(len(inc))*float(norms.mean())
        row0 = row1 = float('nan')
        if name.startswith('decoder'):
            row0 = float(np.linalg.norm(weights[0][lo:hi].reshape(-1, hidden), axis=1).mean())
            row1 = float(np.linalg.norm(weights[-1][lo:hi].reshape(-1, hidden), axis=1).mean())
        rows.append(dict(slice=name, size=hi-lo, initial_norm=float(np.linalg.norm(weights[0][lo:hi])),
                         final_norm=float(np.linalg.norm(weights[-1][lo:hi])), increments=len(inc),
                         mean_increment=float(norms.mean()), lag1_cosine=lag1, net=net, net_over_random_walk=net/walk if walk else float('nan'),
                         row_norm_initial=row0, row_norm_final=row1))
    return dict(checkpoints=len(files), first_learner_steps=metas[0].get('learner_steps'),
                last_learner_steps=metas[-1].get('learner_steps'), slices=rows)


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
    parser.add_argument('--drift', action='store_true', help='Also report per-slice weight drift across checkpoints')
    args = parser.parse_args()
    s = summarize(args.run, args.every)
    if args.drift: s['drift'] = drift(args.run)
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
    if 'drift' in s:
        d = s['drift']
        print(f"drift over {d['checkpoints']} checkpoints ({d['first_learner_steps']:,} to {d['last_learner_steps']:,} learner decisions):")
        print(f"{'slice':>16} {'|w0|':>7} {'|w|':>7} {'mean|dw|':>8} {'lag1':>6} {'net':>6} {'net/rw':>6} {'row0':>6} {'row':>6}")
        for r in d['slices']:
            print(f"{r['slice']:>16} {r['initial_norm']:7.3f} {r['final_norm']:7.3f} {r['mean_increment']:8.4f} {r['lag1_cosine']:6.3f} {r['net']:6.3f} {r['net_over_random_walk']:6.2f} {fmt(r['row_norm_initial']):>6} {fmt(r['row_norm_final']):>6}")


if __name__ == '__main__':
    main()
