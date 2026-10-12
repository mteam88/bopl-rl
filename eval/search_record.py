"""Expert-iteration data: play a checkpoint through the decision-time search on random
any-3 kits and maps, both seats searched, and record what a distillation needs.

    venv/bin/python eval/search_record.py policy.bin --out DIR --games 200 --workers 4 --budget 8,1,8

One compressed .npz per game, DIR/g<seed>.npz:

    obs        (T, 2, 4812) float16   both players' observation rows (layout v4), decision t
    mask       (T, 2, 277)  uint8     the action mask the decision was sampled with (1 = allowed)
    cand       (T, 2, K, 8) float32   the search's candidates: aim, buttons, hold, visits, mean
                                      return, prior logp, chosen, return variance (search_policy.Handle.stats)
    action     (T, 2, 3)    int16     the action played (the chosen candidate)
    root_value (T, 2)       float32   the critic's value of the player's real row at the root
    reward     (T, 2)       float32   the training reward both players earned over the decision
    outcome    (2,)         float32   +1 win, -1 loss, 0 draw (timeout or trade), per seat
    meta       json         seed, map, kits, ticks, budget, checkpoint

Everything a teacher-forced replay needs (the recurrent state is recomputed from obs
in order), plus the search's per-candidate returns to build the improved-policy
target (train/distill.py).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'league'), str(HERE.parent / 'native/tools')]
import environment  # noqa: E402
import layout  # noqa: E402
from evaluation_protocol import resolve_loadouts  # noqa: E402
from native_policy import load, writer_ablation  # noqa: E402
import search_policy  # noqa: E402

POOL = [a for a in range(1, 39) if a not in (12, 15)]   # all_combos.POOL: every ability but 12 and Revival
TRAINING_REWARD = dict(win=1, loss=-1, timeout=-1.0, kill=search_policy.TRAINING_KILL, death=-search_policy.TRAINING_KILL)


def maps_all():
    import map_catalogue
    import registry
    return map_catalogue.parse_map_ids(registry.MAPS)


def record_game(env, handle, policy, seed, map_id, k, budget_text, checkpoint):
    env.reset(seed, map_id)
    env.bot(0, 0); env.bot(1, 0)
    handle.reset(seed)
    kits = env.loadouts
    obs, masks, cands, actions, values, rewards = [], [], [], [], [], []
    while not env.ended:
        row = env.rows[:, :layout.V4_OBS_SIZE].astype(np.float16)
        mask = env.mask.copy().astype(np.uint8)
        act = np.zeros((2, 3), dtype=np.int16)
        cand = np.zeros((2, k, search_policy.STAT_WIDTH), dtype=np.float32)
        for side in (0, 1):
            act[side] = handle.act(side)
            stats = handle.stats()
            cand[side, :len(stats)] = stats[:k]
        counters = handle.counters()
        value = np.array([counters['root_value_0'], counters['root_value_1']], dtype=np.float32)
        players = env.state.players
        before = [(players[p].kills, players[p].deaths) for p in range(2)]
        env.step(act.astype(np.float32))
        players = env.state.players
        # Training reward: kill shaping over the decision (terminal win/loss/timeout is added at the end).
        reward = np.array([TRAINING_REWARD['kill'] * (players[p].kills - before[p][0])
                           + TRAINING_REWARD['death'] * (players[p].deaths - before[p][1]) for p in range(2)], dtype=np.float32)
        obs.append(row); masks.append(mask); cands.append(cand); actions.append(act); values.append(value); rewards.append(reward)
    final = env.frame()
    timeout = bool(final['flags'] & 2)
    won = [bool(final['players'][p]['flags'] & 2) and not timeout for p in range(2)]
    outcome = np.zeros(2, dtype=np.float32)
    if won[0] != won[1]:
        outcome[:] = [1.0 if won[0] else -1.0, 1.0 if won[1] else -1.0]
    rewards[-1] += np.where(timeout, TRAINING_REWARD['timeout'], outcome)
    meta = dict(seed=seed, map=map_id, kits=kits, ticks=final['t'], timeout=timeout, outcome=outcome.tolist(),
                budget=budget_text, checkpoint=str(checkpoint), decisions=len(obs))
    return dict(obs=np.stack(obs), mask=np.stack(masks), cand=np.stack(cands), action=np.stack(actions),
                root_value=np.stack(values), reward=np.stack(rewards), outcome=outcome,
                meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)), meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--seed', type=int, default=1, help='first game seed; game i plays seed+i')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--budget', default='8,1,8')
    p.add_argument('--limit', type=int, default=10800, help='episode ticks')
    p.add_argument('--workspace', type=Path, default=Path('/tmp/bopl-search-record'))
    a = p.parse_args(argv)
    import games as league_games
    league_games.configure(a.workspace)
    a.out.mkdir(parents=True, exist_ok=True)
    policy, meta = load(a.checkpoint.resolve())
    budget = search_policy.parse_budget(a.budget)
    maps = maps_all()
    seeds = [a.seed + i for i in range(a.games)]
    seeds = [s for s in seeds if not (a.out / f'g{s}.npz').exists()]
    lock = threading.Lock()
    done = dict(games=0, decisions=0, seconds=0.0, started=time.time())
    log = (a.out / 'games.jsonl').open('a')

    def worker(mine):
        loadouts = resolve_loadouts(pool=POOL)
        with environment.Environment(seed=mine[0], limit=a.limit, delay=9, loadouts=loadouts) as env:
            for side in (0, 1):
                env.set_layout(side, 4, ablate=writer_ablation(policy))
            handle = search_policy.Handle(env, policy, budget['k'] * budget['n'], seed=mine[0])
            handle.configure(**budget)
            try:
                for seed in mine:
                    map_id = int(np.random.default_rng(seed).choice(maps))
                    t0 = time.time()
                    data, info = record_game(env, handle, policy, seed, map_id, budget['k'], a.budget, a.checkpoint)
                    info['seconds'] = round(time.time() - t0, 1)
                    tmp = a.out / f'.g{seed}.npz.tmp'
                    with tmp.open('wb') as f:
                        np.savez_compressed(f, **data)
                    tmp.rename(a.out / f'g{seed}.npz')
                    with lock:
                        done['games'] += 1; done['decisions'] += info['decisions']; done['seconds'] += info['seconds']
                        log.write(json.dumps(info) + '\n'); log.flush()
                        if done['games'] % 10 == 0 or done['games'] == len(seeds):
                            wall = time.time() - done['started']
                            print(f"{done['games']}/{len(seeds)} games, {done['decisions']} decisions, "
                                  f"{done['seconds'] / done['games']:.0f} s/game, {done['decisions'] / wall:.0f} decisions/s wall",
                                  flush=True)
            finally:
                handle.close()

    threads = [threading.Thread(target=worker, args=(seeds[i::a.workers],)) for i in range(a.workers) if seeds[i::a.workers]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.close()


if __name__ == '__main__':
    main()
