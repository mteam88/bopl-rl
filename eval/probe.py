"""Play a checkpoint on the all-combo benchmark's games and keep the evidence.

    venv/bin/python eval/probe.py A.bin {B.bin|random} --pairs 200 --out DIR [--save losses|all|none]

league/all_combos.py reports one score. This plays its schedule (same seeds,
kits and maps; the action sampling stream differs, so scores agree within noise,
not game for game) and writes, per game, the
outcome and how it ended: A's kills and deaths by cause and A's behavior
counters. Saved games get the native replay (.rep) and the physics-state visual
(.json, the dashboard's evaluation format) under DIR/evaluations/<step>/, so a
loss can be watched rather than guessed at.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'league'), str(HERE.parent / 'native/tools')]
import registry  # noqa: E402
import games as league_games  # noqa: E402
import elo_ladder  # noqa: E402
import map_catalogue  # noqa: E402
import all_combos  # noqa: E402
from common import atomic_json  # noqa: E402
from evaluation_protocol import DEFAULT_LOADOUT  # noqa: E402
from telemetry import summarize  # noqa: E402


def play(built, tasks, workers, seed_base, save, out):
    """games.play's loop with frames and telemetry kept. Tasks carry their pinned kit row."""
    from environment import Environment
    rows = sorted({tuple(t['kits']) for t in tasks})
    if len(rows) > 256:
        raise ValueError('more than 256 kit rows: run fewer pairs')
    index = {r: i for i, r in enumerate(rows)}
    loadouts = dict(mode='set', fixed=list(DEFAULT_LOADOUT), pool=[], lineups=[list(r) for r in rows],
                    set_file=None, no_dup=False, mirror_share=0.0)

    def run(bucket):
        results = []
        with Environment(seed=seed_base + len(tasks), limit=registry.EPISODE_TICKS, delay=registry.DELAY,
                         loadouts=loadouts) as env:
            for task in bucket:
                env.configure_loadouts(dict(loadouts, lineups=[list(rows[index[tuple(task['kits'])]])]))
                env.reset(task['seed'], task['map'])
                rng = np.random.default_rng((task['seed'], task['map'], task['side'], 0))
                env.bot(0, 0); env.bot(1, 0)
                seats = {task['side']: 'A', 1 - task['side']: 'B'}
                for side, name in seats.items():
                    attach = getattr(built[name], 'attach', None)
                    if attach:
                        attach(env, side)
                actors = {side: built[name].start(task, side, rng) for side, name in seats.items()}
                frames = [env.frame()]
                events = []
                while not env.ended:
                    observation = env.rows.copy(); mask = env.mask.copy()
                    actions = np.stack([actors[0](observation, mask), actors[1](observation, mask)])
                    env.step(actions)
                    events.append(env.telemetry[task['side']].copy())
                    frames[-1]['actions'] = actions.tolist()
                    frames.append(env.frame())
                final = frames[-1]
                timeout = bool(final['flags'] & 2)
                won = [bool(p['flags'] & 2) for p in final['players']]
                a_won, b_won = won[task['side']] and not timeout, won[1 - task['side']] and not timeout
                me = final['players'][task['side']]
                behavior = summarize(np.asarray(events))
                result = dict(task, ticks=final['t'],
                              outcome='draw' if timeout or a_won == b_won else ('a' if a_won else 'b'),
                              kills=me['kills'] - behavior['totals']['deaths_self'], deaths=me['deaths'],
                              lineup_ids=env.loadouts[0] + env.loadouts[1], behavior=behavior)
                keep = save == 'all' or (save == 'losses' and result['outcome'] != 'a')
                result['name'] = f"seed{task['seed']}-side{task['side']}-map{task['map']}-{result['outcome']}"
                if keep:
                    name = result['name']
                    env.replay(out / (name + '.rep'))
                    result['replay'] = name + '.rep'
                    atomic_json(out / (name + '.json'), dict(schema=1, step=0, task=dict(task, name=name), result=result,
                                                            frames=frames, rendering='physics state replay',
                                                            tick_rate=60, decision_rate=15,
                                                            actions_timing='observation-before-action'))
                    result['visual'] = name + '.json'
                results.append(result)
        return results

    buckets = [tasks[i::workers] for i in range(min(workers, len(tasks)))]
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        return [g for bucket in pool.map(run, buckets) for g in bucket]


def dashboard(a, model, played, games_dir):
    """Lay DIR out as a run board/serve.py lists (a checkpoints dir makes it one), with the
    probe as its one evaluation, so the saved games open in the dashboard's arena."""
    (a.out / 'checkpoints').mkdir(exist_ok=True)
    config = dict(model.meta.get('config') or {}, schema=1, algorithm='recurrent PPO',
                  probe=dict(a=str(a.a), b=str(a.b), pairs=a.pairs, seed_base=a.seed_base))
    atomic_json(a.out / 'config.json', config)
    atomic_json(a.out / 'status.json', dict(status='complete', pid=None, step=0, learner_steps=0, elapsed=0,
                                            matches=len(played), target_steps=0, target_learner_steps=0))
    games = [dict(name=g['name'], opponent=Path(str(a.b)).stem, map=g['map'], seed=g['seed'], side=g['side'],
                  save='visual' in g, win=g['outcome'] == 'a', timeout=g['outcome'] == 'draw', kills=g['kills'],
                  deaths=g['deaths'], ticks=g['ticks'], **{k: g[k] for k in ('replay', 'visual') if k in g})
             for g in played]
    atomic_json(games_dir / 'result.json', dict(schema=2, step=0, suite='probe', games=games))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a', type=Path)
    p.add_argument('b', help='checkpoint path, or "random" for the uniform random anchor')
    p.add_argument('--pairs', type=int, default=200)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed-base', type=int, default=131001)
    p.add_argument('--save', choices=('losses', 'all', 'none'), default='losses')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)

    built = {'A': elo_ladder.Model('A', a.a.resolve(), 'candidate'),
             'B': elo_ladder.Random() if a.b == 'random' else elo_ladder.Model('B', Path(a.b).resolve(), 'candidate')}
    games_dir = a.out / 'evaluations' / '000000000'
    games_dir.mkdir(parents=True, exist_ok=True)
    league_games.configure(a.out / 'workers')
    todo = all_combos.tasks(a.pairs, a.seed_base, map_catalogue.parse_map_ids(registry.MAPS))
    # The environment holds at most 256 kit rows, so a larger schedule is played in chunks (as games.play does).
    chunks, rows = [[]], [set()]
    for task in todo:
        row = tuple(task['kits'])
        if row not in rows[-1] and len(rows[-1]) == 256:
            chunks.append([]), rows.append(set())
        rows[-1].add(row)
        chunks[-1].append(task)
    played = [g for chunk in chunks for g in play(built, chunk, a.workers, a.seed_base, a.save, games_dir)]
    played.sort(key=lambda g: (g['seed'], g['side']))
    with open(a.out / 'games.jsonl', 'w') as f:
        for g in played:
            f.write(json.dumps(g) + '\n')
    dashboard(a, built['A'], played, games_dir)
    score = np.mean([1.0 if g['outcome'] == 'a' else 0.5 if g['outcome'] == 'draw' else 0.0 for g in played])
    counts = {k: sum(g['outcome'] == k for g in played) for k in ('a', 'b', 'draw')}
    print(f'{a.a.name} vs {a.b}: A scores {score:.3f} over {len(played)} games {counts} -> {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
