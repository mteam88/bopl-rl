"""Qualify ordinary game setups with an explicit delayed-input Bow baseline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
from bow_task import ChargeScript, Task, Trial, point_at_opponent
from environment import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maps', default='0')
    parser.add_argument('--seeds', default=','.join(map(str, range(1, 65))))
    parser.add_argument('--hold-ticks', type=int, default=64)
    parser.add_argument('--offsets', default='0', help='Signed angle offsets in stick codes')
    parser.add_argument('--limit', type=int, default=240)
    parser.add_argument('--start-move-ticks', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--controller', choices=('script', 'neutral', 'random'), default='script')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure(args.output / 'workers')
    jobs = [(m, s, side, offset) for m in map(int, args.maps.split(',')) for s in map(int, args.seeds.split(',')) for side in (0, 1) for offset in map(int, args.offsets.split(','))]

    def run_batch(batch):
        results = []
        with Trial(Task(limit=args.limit, start_move_ticks=args.start_move_ticks)) as trial:
            for map_id, seed, side, offset in batch:
                task = Task(map_id=map_id, side=side, limit=args.limit, hold_ticks=args.hold_ticks,
                            start_move_ticks=args.start_move_ticks)
                trial.task = task
                obs = trial.reset(seed)
                script = ChargeScript(task)
                rng = np.random.default_rng(seed * 2 + side)
                while not trial.done:
                    action = ((point_at_opponent(obs) - 1 + offset) % 255 + 1, script.buttons(obs)) if args.controller == 'script' else (0, 0) if args.controller == 'neutral' else (rng.integers(256), 4*rng.integers(2))
                    obs, _, _, result, _ = trial.step(action)
                result['controller'] = args.controller
                result['angle_offset'] = offset
                results.append(result)
                if result['success']:
                    trial.replay(args.output / f'm{map_id}-s{seed}-p{side+1}-a{offset}.rep')
        return results

    with ThreadPoolExecutor(args.workers) as executor:
        batches = list(executor.map(run_batch, [jobs[i::args.workers] for i in range(args.workers)]))
    results = sorted([x for batch in batches for x in batch], key=lambda x: (x['task']['map_id'], x['seed'], x['task']['side'], x['angle_offset']))
    (args.output / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(dict(trials=len(results), successes=sum(x['success'] for x in results), successful_setups=[(x['task']['map_id'], x['seed'], x['task']['side'], x['angle_offset']) for x in results if x['success']]), indent=2))


if __name__ == '__main__':
    main()
