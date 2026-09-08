"""Summarise a Gate 0 sweep directory (gate0.py output) as a markdown table.

Recomputes CPU-core medians from samples.jsonl (dropping the negative-rate
samples that gate0.py < 2b21359 let through) so old points report correctly.

    python native/rl/many_worlds/gate0_report.py native/artifacts/rl/many-worlds/gate0
"""
import json, statistics, sys
from pathlib import Path


def med(samples, key, lo=0.0):
    values = [s[key] for s in samples if key in s and s[key] == s[key] and s[key] >= lo]
    return statistics.median(values) if values else float('nan')


def main(root):
    root = Path(root)
    results = [json.loads(l) for l in (root / 'results.jsonl').read_text().splitlines() if l.strip()]
    rows = []
    for r in results:
        if r.get('skipped'):
            rows.append((r['point'], r['workers'], r['buffers'], None, r['reason'])); continue
        point = root / r['point']
        samples = [json.loads(l) for l in (point / 'samples.jsonl').read_text().splitlines() if l.strip()]
        n = len(samples); steady = samples[n // 5: max(n // 5 + 1, n - 2)]
        metrics = [json.loads(l) for l in (point / 'run/metrics.jsonl').read_text().splitlines() if l.strip()]
        m = metrics[len(metrics) // 5:]
        rows.append((r['point'], r['workers'], r['buffers'], dict(
            sps=r['learner_sps_steady'], sps_all=r['learner_sps_wall_all'],
            mps=r['matches'] / r['elapsed_training_seconds'],
            games=r['workers'] * 2, pss=r.get('pss_games_gb_max', float('nan')),
            pss_each=r.get('pss_game_gb_median', float('nan')),
            cores_g=med(steady, 'cores_games'), cores_t=med(steady, 'cores_trainer'),
            gpu=med(steady, 'gpu_util'), mem_min=min(s['mem_available_gb'] for s in samples),
            rollout=statistics.median(x['rollout_seconds'] for x in m),
            train=statistics.median(x['train_seconds'] for x in m),
            high=r['cgroup_memory_high_events'], oom=r['cgroup_memory_oom_kills'],
            startup=r['startup_seconds'], rc=r['returncode']), None))
    print('| point | workers | buffers | games | learner sps (steady) | matches/s | PSS games GB (max) | PSS/game GB | game cores | trainer cores | GPU util % | MemAvail min GB | rollout s/update | train s/update | mem.high / oom_kill | rc |')
    print('|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|')
    for name, w, b, d, reason in rows:
        if d is None:
            print(f'| {name} | {w} | {b} | {w*2} | skipped | | | | | | | | | | | | {reason} |'); continue
        print(f"| {name} | {w} | {b} | {d['games']} | {d['sps']:,.0f} | {d['mps']:.1f} | {d['pss']:.1f} | {d['pss_each']:.3f} | "
              f"{d['cores_g']:.2f} | {d['cores_t']:.2f} | {d['gpu']:.0f} | {d['mem_min']:.1f} | {d['rollout']:.3f} | {d['train']:.4f} | "
              f"{d['high']} / {d['oom']} | {d['rc']} |")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'native/artifacts/rl/many-worlds/gate0')
