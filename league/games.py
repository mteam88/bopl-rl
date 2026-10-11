"""Scheduling and playing league games: the one place a game's map and kits are chosen.

Both `league play` (member against member) and `league candidate` (a checkpoint
against the members) come through here, so a candidate is always measured under
exactly the condition the members' ratings were fitted on.
"""
import os
from pathlib import Path
import zlib

import numpy as np

import registry


def configure(workspace):
    """Point the research C environment at the tree's Rust sim, as train.py does."""
    os.environ.update(BOPL_RL_BACKEND='rust', BOPL_RL_INPROC_SPARE='0',
                      BOPL_RL_WORLDS_PER_RUNTIME=str(1 << 20), OMP_NUM_THREADS='1')
    from environment import configure
    configure(Path(workspace), registry.SIM)


def participant(row):
    """The elo_ladder participant that plays a member."""
    import elo_ladder
    if row['kind'] == 'human':
        raise ValueError(f"{row['name']} is a human member: their games are recorded, not played")
    if row['kind'] == 'random':
        return elo_ladder.Random()
    if row['kind'] == 'scripted':
        return elo_ladder.Scripted(row['name'])
    return elo_ladder.Model(row['name'], registry.ROOT / row['path'], row['kind'])


def map_order(maps, seed_base):
    """The rating condition's maps in one fixed random order: a matchup walks it
    from its own offset, so its maps are random but do not repeat until each has
    been played."""
    return [int(m) for m in np.random.default_rng(seed_base).permutation(sorted(maps))]


def matchup(ra, rb, maps, pairs, seed_base, start=0):
    """`pairs` side-swapped seed pairs between two members, continuing at `start`.

    Each seat plays a kit its member was built for (registry.kits), and the two
    may differ, as in a real match; a member with no kits of its own (random)
    takes its opponent's. Seed pair i plays kit combination i mod n and map i of
    games.map_order, both shifted by a per-matchup offset so that a checkpoint
    meeting forty members for one pair each does not play them all on the first
    kit and the first map. Only maps
    both members can play are used (registry.plays_map). `start` is the number of
    seed pairs already recorded for this matchup: games are deterministic in
    their seed, so replaying from 0 would record the same games again as new data.
    Tasks name members by id, the key payoff.jsonl uses, and carry `kits`, the
    six ids [seat 0 | seat 1] the game must be played on.
    """
    ka, kb = registry.kits(ra), registry.kits(rb)
    if ka is None and kb is None:
        return []
    if ka is None:
        combos = [(k, k) for k in kb]
    elif kb is None:
        combos = [(k, k) for k in ka]
    else:
        combos = [(x, y) for x in ka for y in kb]
    order = [m for m in map_order(maps, seed_base) if registry.plays_map(ra, m) and registry.plays_map(rb, m)]
    if not order:
        return []
    shift = zlib.crc32(f"{ra['id']}|{rb['id']}".encode())
    tasks = []
    for pair in range(start, start + pairs):
        x, y = combos[(pair + shift) % len(combos)]
        for side in (0, 1):
            tasks.append(dict(a=ra['id'], b=rb['id'], map=order[(pair + shift) % len(order)], seed=seed_base + pair,
                              side=side, kits=list(x + y) if side == 0 else list(y + x)))
    return tasks


def play(built, tasks, workers, seed_base, style=None):
    """Play tasks from `matchup` and check every game ran on the kits it was scheduled on.
    `style` (elo_ladder.play_games) measures a compiled style in every game without
    changing it."""
    import elo_ladder
    from evaluation_protocol import DEFAULT_LOADOUT
    # The environment pins one [seat 0 | seat 1] row per task (play_games) out of a
    # set of at most 256, so a larger schedule is played in chunks.
    chunks, rows = [[]], [{}]
    for task in tasks:
        row = tuple(task['kits'])
        if row not in rows[-1] and len(rows[-1]) == 256:
            chunks.append([]), rows.append({})
        task['lineup'] = rows[-1].setdefault(row, len(rows[-1]))
        chunks[-1].append(task)
    games = []
    for chunk, lineups in zip(chunks, rows):
        loadouts = dict(mode='set', fixed=list(DEFAULT_LOADOUT), pool=[], lineups=[list(r) for r in lineups],
                        set_file=None, no_dup=False, mirror_share=0.0)
        games += elo_ladder.play_games(built, chunk, registry.EPISODE_TICKS, workers, seed_base,
                                       delay=registry.DELAY, loadouts=loadouts, style=style)
    # lineup_ids is what the game actually played.
    for game in games:
        if list(game.get('lineup_ids') or []) != game['kits']:
            raise RuntimeError(f"kit violation, {game['a']} vs {game['b']} map {game['map']} side {game['side']}:"
                               f" played {game.get('lineup_ids')}, scheduled {game['kits']}")
    return games
