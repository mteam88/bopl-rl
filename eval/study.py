"""Play two checkpoints against each other on one kit and save every game for study:
native .rep (renderable with render_video.py), physics frames .json, and a decision
dump for BOTH sides (.mind-a.json.gz / .mind-b.json.gz). Nothing is rated here; the
point is to watch.

    python study.py --out DIR --a name=path --b name=path --loadout 25,24,23 [--maps 0,1,2] [--pairs 3]

DIR is written as a run directory, so pointing --out at artifacts/rl/study-<name>
puts every game on the research dashboard (serve.py), playable in its arena viewer.

`b` may also be a scripted bot name (archer, pursuer, evasive, mixed[-v2][-hard]) or
`random`. Prints one line per game: name, winner, ticks, how each player died.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import Environment
from native_policy import load, decide, writer_ablation
from common import atomic_json
import decision_dump
import layout
from evaluation_protocol import checkpoint_identity, resolve_loadouts, scripted_opponent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'league'))
import games as league_games

CAUSE = {0: 'other', 1: 'squished', 2: 'accident', 3: 'eaten', 4: 'beamed', 5: 'alive'}


def participant(spec):
    name, _, path = spec.partition('=')
    if path:
        return name, Path(path)
    return name, None


def rows_for(model, rows, layouts):
    batch = rows[:, :model.obs_size].copy()
    mine = layout.layout_of(model.architecture)
    for p in range(2):
        if layouts[p] != mine:
            batch[p] = 0
    return torch.from_numpy(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--a', required=True, help='name=checkpoint.bin')
    parser.add_argument('--b', required=True, help='name=checkpoint.bin, a scripted bot name, or random')
    parser.add_argument('--loadout', required=True, help='three ability ids, e.g. 25,24,23 (mirrored)')
    parser.add_argument('--maps', default='0,1,2')
    parser.add_argument('--pairs', type=int, default=2, help='side-swapped seed pairs per map')
    parser.add_argument('--seed-base', type=int, default=70001)
    parser.add_argument('--limit', type=int, default=7200)
    parser.add_argument('--delay', type=int, default=9)
    args = parser.parse_args()
    torch.set_num_threads(1)

    run = args.out.resolve(); run.mkdir(parents=True, exist_ok=True)
    league_games.configure(run / 'workers')
    loadouts = resolve_loadouts([int(x) for x in args.loadout.split(',')])
    maps = [int(x) for x in args.maps.split(',')]

    a_name, a_path = participant(args.a)
    b_name, b_path = participant(args.b)
    a_model, a_meta = load(a_path)
    models = {a_name: a_model}
    identities = {a_name: checkpoint_identity(a_path)}
    scripted = None
    if b_path is not None:
        models[b_name] = load(b_path)[0]; identities[b_name] = checkpoint_identity(b_path)
    elif b_name != 'random':
        scripted = scripted_opponent(b_name)
        if scripted is None:
            parser.error(f'unknown scripted bot {b_name}')

    # The dashboard discovers a directory as a run when it holds metrics.jsonl or
    # checkpoints/, and reads config.json, status.json and evaluations/<step>/result.json.
    step = int(a_meta.get('step') or 0)
    out = run / 'evaluations' / f'{step:09d}'
    out.mkdir(parents=True, exist_ok=True)
    (run / 'checkpoints').mkdir(exist_ok=True)
    ids = [int(x) for x in args.loadout.split(',')]

    tasks = []
    for map_id in maps:
        for pair in range(args.pairs):
            for side in (0, 1):
                tasks.append(dict(map=map_id, seed=args.seed_base + pair, side=side,
                                  name=f'map{map_id}-s{pair}-{"ab" if side == 0 else "ba"}'))

    games = []
    with Environment(seed=args.seed_base, limit=args.limit, delay=args.delay, loadouts=loadouts) as env:
        for task in tasks:
            env.reset(task['seed'], task['map'])
            sides = {task['side']: a_name, 1 - task['side']: b_name}
            env.bot(0, 0); env.bot(1, 0)
            if scripted is not None:
                env.bot(1 - task['side'], *scripted)
            for p in (0, 1):
                model = models.get(sides[p])
                env.set_layout(p, layout.layout_of(model.architecture) if model else 3,
                               ablate=writer_ablation(model) if model else -1)
            layouts = [env.layout_of(0), env.layout_of(1)]
            states = {p: models[sides[p]].initial(2, 'cpu') for p in (0, 1) if sides[p] in models}
            generator = torch.Generator().manual_seed(task['seed'] * 7 + task['side'])
            rng = np.random.default_rng(task['seed'] * 11 + task['side'])
            frames = [env.frame()]; minds = {0: [], 1: []}
            while not env.ended:
                rows = env.rows.copy(); mask = env.mask.copy()
                actions = torch.zeros(2, layout.NUM_ATNS, dtype=torch.long)
                with torch.no_grad():
                    for p in (0, 1):
                        model = models.get(sides[p])
                        if model is None:
                            if sides[p] == 'random':
                                actions[p] = torch.as_tensor(np.array([rng.integers(n) for n in layout.ACTION_SIZES]))
                            continue
                        logits, values, acts, states[p] = decide(model, rows_for(model, rows, layouts), mask, states[p], generator)
                        actions[p] = acts[p]
                        minds[p].append(decision_dump.decision(env.state.header.episode_ticks, logits[p], values[p], acts[p], mask[p]))
                frames[-1]['actions'] = actions.tolist()
                env.step(actions.numpy())
                frames.append(env.frame())
            final = env.frame()
            timeout = bool(final['flags'] & 2)
            winner = next((sides[p] for p in (0, 1) if final['players'][p]['flags'] & 2), None) if not timeout else None
            if not timeout and all(final['players'][p]['flags'] & 2 for p in (0, 1)):
                winner = 'both'
            deaths = {sides[p]: CAUSE.get(final['players'][p]['cause_of_death'], '?') for p in (0, 1)}
            result = dict(task, a=a_name, b=b_name, sides={str(p): sides[p] for p in (0, 1)},
                          winner=winner if winner else ('timeout' if timeout else 'both-dead'),
                          ticks=final['t'], deaths=deaths, final_players=final['players'])
            env.replay(out / (task['name'] + '.rep'))
            atomic_json(out / (task['name'] + '.json'),
                        dict(schema=1, step=step, task=task, result=result, frames=frames,
                             rendering='physics state replay', tick_rate=60, decision_rate=15,
                             actions_timing='observation-before-action'))
            for p in (0, 1):
                if not minds[p]:
                    continue
                # The dashboard's policy panel reads <game>.mind.json.gz; give it side a's.
                suffix = '.mind.json.gz' if sides[p] == a_name else f'.mind-{sides[p]}.json.gz'
                decision_dump.write(out / (task['name'] + suffix), step, p, minds[p],
                                    identities[sides[p]], args.delay)
            games.append(result)
            print(f'{task["name"]:16s} winner={result["winner"]:>14s} ticks={result["ticks"]:5d} '
                  f'deaths={json.dumps(deaths)}', flush=True)
    rows = [dict(opponent=g['sides']['1'] if g['side'] == 0 else g['sides']['0'],
                 map=g['map'], seed=g['seed'], side=g['side'], save=True, name=g['name'],
                 win=g['winner'] == a_name, timeout=g['winner'] == 'timeout',
                 kills=g['final_players'][g['side']]['kills'],
                 deaths=g['final_players'][g['side']]['deaths'],
                 ticks=g['ticks'], final_players=g['final_players'],
                 outcome=g['winner'], cause_of_death=g['deaths'],
                 replay=g['name'] + '.rep', visual=g['name'] + '.json',
                 mind=g['name'] + '.mind.json.gz') for g in games]
    atomic_json(out / 'result.json', dict(
        schema=2, decision_dump_schema=1, map_catalogue='identity-124', step=step,
        suite='diagnostic', checkpoint=a_path.name, evaluation_maps=maps,
        heldout_maps=[m for m in (3, 4) if m not in maps], training_maps=maps,
        frozen_opponents=[], scripted_opponents=[], groups=[],
        note=f'study: {a_name} vs {b_name} on loadout {args.loadout} (rl/eval/study.py)',
        games=rows))
    atomic_json(run / 'status.json', dict(status='complete', step=step, learner_steps=0))
    atomic_json(run / 'config.json', dict(
        label=f'{a_name} vs {b_name} study', training_maps=maps,
        heldout_maps=[m for m in (3, 4) if m not in maps],
        episode_ticks=args.limit, observation_delay_ticks=args.delay,
        loadouts=dict(mode='fixed', fixed=ids, pool=[], lineups=[], set_file=None,
                      no_dup=False, mirror_share=0.0)))
    print('summary', dict(Counter(g['winner'] for g in games)))
    print('dashboard:', run.name)


if __name__ == '__main__':
    main()
