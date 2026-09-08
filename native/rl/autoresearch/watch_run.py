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
series divided by sqrt(sum(|increment|²)), the RMS displacement of independent,
zero-mean directions with those lengths. A ratio near 1 and cosine near 0 is
consistent with a random walk; it does not identify the gradient's source.
Higher ratios and cosines indicate directional persistence, not policy strength.
"""
import argparse
import hashlib
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
    for key in ('hidden', 'obs_size', 'num_layers'):
        if type(meta.get(key)) is not int or meta[key] < 1:
            raise ValueError(f'Invalid checkpoint dimension: {key}')
    hidden, obs, layers = meta['hidden'], meta['obs_size'], meta['num_layers']
    if hidden < 8 or hidden % 8:
        raise ValueError('Native hidden size must be divisible by eight')
    if meta.get('architecture') == 'bopl-set-mingru-v1':
        if obs != 3257: raise ValueError('SET requires 3257 observation features')
        encoder = hidden*905 + 64*49 + hidden*(hidden+128)
    elif meta.get('architecture') == 'puffernet-mingru':
        encoder = hidden*obs
    else:
        raise ValueError(f"Unsupported architecture: {meta.get('architecture')}")
    decoder = hidden*273
    total = encoder + decoder + layers*3*hidden**2
    return total, [('encoder', 0, encoder), ('decoder.aim', encoder, encoder+256*hidden),
                   ('decoder.buttons', encoder+256*hidden, encoder+272*hidden),
                   ('decoder.value', encoder+272*hidden, encoder+decoder), ('mingru', encoder+decoder, total)]


def drift(run):
    """Stream validated checkpoints using elementwise reductions, without BLAS calls."""
    import os
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'  # Cap this summary process even when the shell specifies more.
    import numpy as np
    files = sorted(Path(run, 'checkpoints').glob('policy-*.bin'))
    # Checkpoint publishing writes the .bin before its sidecar. Ignore only an
    # unfinished newest checkpoint; a missing interior sidecar is corruption.
    if files and not Path(str(files[-1])+'.json').exists(): files.pop()
    if len(files) < 3: raise SystemExit(f'{run}: need at least three native checkpoints for drift')
    metas = [json.loads(Path(str(f)+'.json').read_text()) for f in files]
    total, slices = weight_slices(metas[0])
    expected = dict(schema=1, backend='pufferlib-native', precision='float32', action_sizes=[256, 16],
                    **{k: metas[0][k] for k in ('architecture', 'hidden', 'obs_size', 'num_layers')})
    steps = []
    for f, meta in zip(files, metas):
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError(f'{f}: incompatible checkpoint metadata')
        step = meta.get('learner_steps')
        if type(step) is not int or step < 0 or (steps and step <= steps[-1]):
            raise ValueError(f'{f}: learner counts must increase strictly')
        steps.append(step)
    rows = [dict(slice=name, size=hi-lo, increments=len(files)-1) for name, lo, hi in slices]
    lengths = [[] for _ in slices]
    cosines = [[] for _ in slices]
    first = previous = previous_delta = None
    def norm(v): return math.sqrt(float(np.sum(v*v, dtype=np.float64)))
    for f, meta in zip(files, metas):
        data = f.read_bytes()
        if len(data) != total*4 or hashlib.sha256(data).hexdigest() != meta.get('weights_sha256'):
            raise ValueError(f'{f}: checkpoint size or hash mismatch')
        w = np.frombuffer(data, dtype='<f4').astype(np.float64)
        if not np.isfinite(w).all(): raise ValueError(f'{f}: nonfinite weights')
        if first is None:
            first = previous = w
            continue
        delta = w-previous
        for i, (_, lo, hi) in enumerate(slices):
            length = norm(delta[lo:hi])
            if previous_delta is not None and length and lengths[i][-1]:
                cosines[i].append(float(np.sum(delta[lo:hi]*previous_delta[lo:hi])) / (length*lengths[i][-1]))
            lengths[i].append(length)
        previous, previous_delta = w, delta
    for i, (_, lo, hi) in enumerate(slices):
        net = norm(previous[lo:hi]-first[lo:hi])
        walk = math.sqrt(sum(v*v for v in lengths[i]))
        rows[i].update(initial_norm=norm(first[lo:hi]), final_norm=norm(previous[lo:hi]),
                       mean_increment=sum(lengths[i])/len(lengths[i]),
                       lag1_cosine=sum(cosines[i])/len(cosines[i]) if cosines[i] else float('nan'),
                       net=net, net_over_random_walk=net/walk if walk else float('nan'))
    return dict(checkpoints=len(files), first_learner_steps=steps[0], last_learner_steps=steps[-1],
                interval_learner_steps=[b-a for a, b in zip(steps, steps[1:])], slices=rows)


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
        print('Directional persistence only; does not identify gradient sources or establish strength.')
        print(f"{'slice':>16} {'|w0|':>7} {'|w|':>7} {'mean|dw|':>8} {'lag1':>6} {'net':>6} {'net/rw':>6}")
        for r in d['slices']:
            print(f"{r['slice']:>16} {r['initial_norm']:7.3f} {r['final_norm']:7.3f} {r['mean_increment']:8.4f} {r['lag1_cosine']:6.3f} {r['net']:6.3f} {r['net_over_random_walk']:6.2f}")


if __name__ == '__main__':
    main()
