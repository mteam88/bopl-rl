"""Bin a training run's metrics.jsonl into curves small enough to plot.

    venv/bin/python eval/run_curves.py RUN_DIR --bins 400 --out curves.json

metrics.jsonl has one row per optimizer update (11.7 GB for allkit-1's 325k
updates), too big to load. This streams it once and, per bin of learner steps,
averages the optimizer scalars and sums the behavior and overflow counters, so
rates (win rate, deaths by cause, arrow hit rate, overflow drop rate) are
computed over thousands of episodes rather than one update's two dozen.
"""
import argparse
import json
from pathlib import Path

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads

SCALARS = ['sps', 'learner_sps', 'policy_loss', 'value_loss', 'entropy', 'entropy_aim', 'entropy_buttons',
           'entropy_hold', 'kl', 'clip_fraction', 'gradient_norm', 'explained_variance', 'advantage_std',
           'rollout_seconds', 'train_seconds', 'wall_seconds', 'entropy_coefficient', 'learning_rate',
           'cuda_memory_mb']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--bins', type=int, default=400)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)

    config = json.loads((a.run / 'config.json').read_text())
    target = config.get('learner_steps') or config['steps']
    width = target / a.bins
    bins = {}
    with open(a.run / 'metrics.jsonl', 'rb') as f:
        for line in f:
            try:
                row = loads(line)
            except ValueError:
                continue
            b = bins.setdefault(int(row['learner_steps'] // width), dict(
                n=0, learner_steps=0, step=0, elapsed=0, scalars={}, behavior={}, actions={}, abilities={},
                overflow={}, set_rows=[0.0] * 4, set_overflow=[0] * 4, opponents={}))
            b['n'] += 1
            b['learner_steps'], b['step'], b['elapsed'] = row['learner_steps'], row['step'], row['elapsed']
            for k in SCALARS:
                if row.get(k) is not None:
                    b['scalars'][k] = b['scalars'].get(k, 0.0) + row[k]
            for k, v in (row.get('behavior') or {}).get('totals', {}).items():
                b['behavior'][k] = b['behavior'].get(k, 0.0) + v
            b['behavior']['decisions'] = b['behavior'].get('decisions', 0) + (row.get('behavior') or {}).get('decisions', 0)
            for k, v in (row.get('action_rates') or {}).items():
                b['actions'][k] = b['actions'].get(k, 0.0) + v
            for k, v in (row.get('ability_activations') or {}).items():
                b['abilities'][k] = b['abilities'].get(k, 0) + v
            for k, v in (row.get('entity_overflow') or {}).items():
                o = b['overflow'].setdefault(k, [0, 0])
                o[0] += v['rows']
                o[1] += v['dropped']
            for i, v in enumerate(row.get('set_rows') or []):
                b['set_rows'][i] += v
            for i, v in enumerate(row.get('set_overflow') or []):
                b['set_overflow'][i] += v
            for name in row.get('pool_opponents') or []:
                kind = 'self' if name.startswith('self@') else name
                b['opponents'][kind] = b['opponents'].get(kind, 0) + 1
            if 'pool' in row and row['pool']:
                b['pool'] = row['pool']
    out = []
    for i in sorted(bins):
        b = bins[i]
        n = b['n']
        out.append(dict(bin=i, updates=n, learner_steps=b['learner_steps'], step=b['step'], elapsed=b['elapsed'],
                        scalars={k: v / n for k, v in b['scalars'].items()}, behavior=b['behavior'],
                        actions={k: v / n for k, v in b['actions'].items()}, abilities=b['abilities'],
                        overflow=b['overflow'], set_rows=[v / n for v in b['set_rows']],
                        set_overflow=b['set_overflow'], opponents=b['opponents'], pool=b.get('pool')))
    a.out.write_text(json.dumps(dict(run=str(a.run), config=config, bins=out)))
    print(f'{len(out)} bins from {sum(b["updates"] for b in out)} updates -> {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
