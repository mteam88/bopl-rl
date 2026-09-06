"""Evaluate a saved Bow learner; task overrides are explicit transfer probes."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import torch
from bow_task import Task, Trial
from environment import configure
from policy import load
from train_bow import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maps', help='Defaults to the training map')
    parser.add_argument('--sides', help='Defaults to the training side')
    parser.add_argument('--assistance', choices=('none', 'timing'), help='Defaults to the checkpoint task')
    parser.add_argument('--start-move-ticks', type=int, help='Physical setup variation before control: positive approaches, negative retreats')
    parser.add_argument('--opponent', choices=('idle', 'approach'), help='Defaults to the checkpoint opponent')
    parser.add_argument('--seed', type=int, default=300001)
    parser.add_argument('--trials', type=int, default=128)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('At least one sampled trial is required')
    args.output.mkdir(parents=True, exist_ok=False)
    configure(args.output / 'workers')
    torch.set_num_threads(1)
    policy, checkpoint = load(args.checkpoint)
    if checkpoint.get('task') != 'bow':
        raise ValueError('Expected a Bow curriculum checkpoint')
    policy.eval()
    trained = Task(**checkpoint['config']['task'])
    maps = list(map(int, args.maps.split(','))) if args.maps else [trained.map_id]
    sides = list(map(int, args.sides.split(','))) if args.sides else [trained.side]
    results = []
    for map_id in maps:
        for side in sides:
            task = replace(trained, map_id=map_id, side=side,
                           assistance=args.assistance or trained.assistance,
                           start_move_ticks=trained.start_move_ticks if args.start_move_ticks is None else args.start_move_ticks,
                           opponent=args.opponent or trained.opponent)
            media = args.output / f'm{map_id}-p{side+1}'
            media.mkdir()
            with Trial(task) as trial:
                results.extend(evaluate(policy, trial, args.seed, count=args.trials,
                                        timing=task.assistance == 'timing', output=media))
    report = dict(checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  training_task=asdict(trained), games=results)
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    for map_id in maps:
        for side in sides:
            for mode in ('sample', 'argmax'):
                games = [r for r in results if r['task']['map_id'] == map_id and r['task']['side'] == side and r['mode'] == mode]
                print(json.dumps(dict(map=map_id, side=side, mode=mode, trials=len(games),
                                      successes=sum(r['success'] for r in games))), flush=True)


if __name__ == '__main__':
    main()
