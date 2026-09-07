"""Full-match evaluation of scripted and learned controllers on frozen tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import shutil
from pathlib import Path
import time
import numpy as np
import torch
from environment import Environment, configure
from policy import load
from scripted_opponents import BotSpec, ScriptedOpponent
from telemetry import FIELDS, summarize
from train import atomic_json


class Controller:
    """One player's state, RNG and delayed-observation interface per match."""
    def __init__(self, description, seed, models):
        self.kind = description['kind']
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator().manual_seed(seed)
        self.model = models.get(description.get('path'))
        self.state = self.model.initial(1, 'cpu') if self.model else None
        self.bot = ScriptedOpponent(BotSpec(**description['spec']), seed) if self.kind == 'script' else None

    def act(self, observation):
        if self.bot:
            return self.bot.act(observation)
        if self.kind == 'idle':
            return np.zeros(2, dtype=np.int64)
        if self.kind == 'random':
            return np.array([self.rng.integers(256), self.rng.integers(16)])
        if self.model is None:
            raise ValueError('Unknown controller')
        with torch.no_grad():
            logits, _, self.state = self.model.step(torch.from_numpy(observation.copy()).view(1, -1), self.state)
            return np.array([int(torch.multinomial(p.softmax(-1), 1, generator=self.generator))
                             for p in logits[0].split([256, 16])])


def run(tasks, out, workers=4, limit=3600):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    configure(out / 'workers')
    source_hashes={}
    for path in Path(__file__).parent.glob('*.py'):
        target=out/'source'/path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path,target)
        source_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    models = {}
    for task in tasks:
        for key in ('learner', 'opponent'):
            desc = task[key]
            if desc['kind'] == 'policy' and desc['path'] not in models:
                path = Path(desc['path'])
                if hashlib.sha256(path.read_bytes()).hexdigest() != desc['sha256']:
                    raise ValueError('Checkpoint identity changed')
                model, checkpoint = load(path)
                if checkpoint.get('task') == 'bow':
                    raise ValueError('Convert restricted Bow weights before full-game evaluation')
                models[desc['path']] = model.eval()
    atomic_json(out / 'schedule.json', dict(schema=1, limit=limit, tasks=tasks, source_sha256=source_hashes))
    started = time.monotonic()
    buckets = [tasks[i::workers] for i in range(min(workers, len(tasks)))]

    def work(index_and_bucket):
        index,bucket=index_and_bucket
        results = []
        with Environment(seed=90001+index, map_id=bucket[0]['map'], limit=limit) as env:
            for task in bucket:
                env.reset(task['seed'], task['map'])
                side = task['side']
                controllers = [None, None]
                controllers[side] = Controller(task['learner'], task['seed'] * 7 + side, models)
                controllers[1-side] = Controller(task['opponent'], task['seed'] * 11 + 1-side, models)
                events = []
                moves = [0., 0.]
                previous = env.obs[:, 8:10].copy() * 64
                counts = np.zeros((2, 4), dtype=np.int64)
                decisions = 0
                initial = env.frame()
                while not env.ended:
                    actions = np.stack([controller.act(env.obs[i]) for i, controller in enumerate(controllers)])
                    env.step(actions)
                    events.append(env.telemetry.copy())
                    positions = env.obs[:, 8:10].copy() * 64
                    # Diagnostic distance includes respawn discontinuities.
                    delta = np.linalg.norm(positions-previous, axis=1)
                    for i in range(2):
                        if delta[i] < 8:
                            moves[i] += float(delta[i])
                        counts[i] += (actions[i, 1] & np.array([1, 2, 4, 8])) != 0
                    previous = positions
                    decisions += 1
                frame = env.frame()
                timeout = bool(frame['flags'] & 2)
                me = frame['players'][side]
                result = {**task, 'win':bool(me['flags'] & 2) and not timeout,
                          'loss':bool(frame['players'][1-side]['flags'] & 2) and not timeout,
                          'timeout':timeout, 'ticks':frame['t'], 'decisions':decisions,
                          'kills':me['kills'], 'deaths':me['deaths'], 'step_mode':env.step_mode,
                          'initial_players':initial['players'], 'final_players':frame['players'],
                          'behavior':summarize(np.asarray(events)[:, side]),
                          'opponent_behavior':summarize(np.asarray(events)[:, 1-side]),
                          'movement':moves[side], 'opponent_movement':moves[1-side],
                          'button_rates':(counts[side]/max(1, decisions)).tolist()}
                result['script_stats'] = controllers[side].bot.stats if controllers[side].bot else None
                result['opponent_script_stats'] = controllers[1-side].bot.stats if controllers[1-side].bot else None
                if task.get('save'):
                    replay = out / (task['name'] + '.rep')
                    env.replay(replay)
                    result['replay'] = replay.name
                    result['replay_sha256'] = hashlib.sha256(replay.read_bytes()).hexdigest()
                atomic_json(out / (task['name'] + '.json'), result)
                results.append(result)
                print(json.dumps(dict(name=task['name'], win=result['win'], timeout=timeout,
                                      hits=result['behavior']['totals']['arrow_hits'])), flush=True)
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = [r for bucket in executor.map(work, enumerate(buckets)) for r in bucket]
    groups = []
    for group in sorted({r['group'] for r in results}):
        selected = [r for r in results if r['group'] == group]
        groups.append(dict(group=group, games=len(selected), wins=sum(r['win'] for r in selected),
                           losses=sum(r['loss'] for r in selected), timeouts=sum(r['timeout'] for r in selected),
                           kills=sum(r['kills'] for r in selected), deaths=sum(r['deaths'] for r in selected),
                           behavior_totals={f:sum(r['behavior']['totals'][f] for r in selected) for f in FIELDS},
                           movement=sum(r['movement'] for r in selected)))
    report = dict(schema=1, elapsed=time.monotonic()-started, groups=groups, games=results)
    atomic_json(out / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    run(schedule['tasks'], args.output, args.workers, schedule['limit'])


if __name__ == '__main__':
    main()
