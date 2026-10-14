"""Team games (docs/TEAM_PLAY.md): 2v2 between native checkpoints on the research environment.

    venv/bin/python eval/team_games.py A.bin B.bin --pairs 250 --workers 4 --out team.jsonl
    venv/bin/python eval/team_games.py A.bin B.bin --partner P.bin ...   # A+P vs B+P: who is the better ally of P

Seats 0,1 are team 0 and seats 2,3 team 1 (research_open_teams). Seed pair i deals four
random kits (the all-combos pool, league/all_combos.POOL) and plays them twice with the
teams swapped: A's team holds kits 0,1 on team 0, then kits 2,3 on team 1, so neither a
kit nor a spawn favours a side. Without --partner both seats of a team are one model
(A+A vs B+B); with it, seat 0 of a team is A or B and seat 1 the partner. Maps, ticks
and delay are the league rating condition's (league/registry.py).

A team wins when its players carry the game's win flag and the other's do not; a timeout
or a flag on both teams is a draw. Prints A's score (wins + draws/2) with a seed-pair
bootstrap interval, and per model the team kills (deaths of a teammate the model's
seats caused, from the game's death events) per game.
"""
import argparse
import ctypes as C
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'league'), str(HERE.parent / 'native/tools')]
import registry  # noqa: E402
import games as league_games  # noqa: E402
import map_catalogue  # noqa: E402
import layout  # noqa: E402

POOL = [a for a in range(1, 39) if a not in (12, 15)]   # league/all_combos.POOL
TEAMS = (0, 0, 1, 1)
DEATH = 1


class Event(C.Structure):
    """BoplRlEvent (sim/abi/bopl_rl.h)."""
    _fields_ = [(name, C.c_uint32) for name in ('kind', 'tick', 'subject', 'actor', 'entity', 'source_ability', 'cause',
                                                'flags', 'related', 'related_actor', 'related_tick', 'reserved')] + \
               [(name, C.c_int64) for name in ('x', 'y', 'vx', 'vy', 'ax', 'ay', 'avx', 'avy')]


def bind():
    import environment
    lib = environment.LIB
    lib.research_open_teams.argtypes = [C.c_uint32, C.c_uint32, C.c_uint32, C.c_int, C.c_int, C.POINTER(C.c_uint32)]
    lib.research_open_teams.restype = C.c_void_p
    lib.research_set_kits.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.c_int]
    lib.research_set_kits.restype = C.c_int
    lib.research_events.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_int)]
    lib.research_events.restype = C.POINTER(Event)
    return environment, lib


class TeamEnvironment:
    """research_open_teams with every per-player buffer `players` rows (environment.Environment is two)."""
    def __init__(self, seed, limit, delay, teams=TEAMS):
        self.environment, self.lib = bind()
        self.players = len(teams)
        self.handle = self.lib.research_open_teams(seed, 0, limit, delay, self.players, (C.c_uint32 * self.players)(*teams))
        if not self.handle:
            raise RuntimeError('Could not open the team evaluator')
        stride = self.environment.OBS_STRIDE
        self.rows = np.ctypeslib.as_array(self.lib.research_obs(self.handle), shape=(self.players * stride,)).reshape(self.players, stride)
        self.mask = np.ctypeslib.as_array(self.lib.research_mask(self.handle), shape=(self.players * layout.LOGITS,)).reshape(self.players, layout.LOGITS)
        self.check()

    def check(self):
        if self.lib.research_check(self.handle):
            raise RuntimeError('Game error during evaluation')

    def set_kits(self, kits):
        if self.lib.research_set_kits(self.handle, (C.c_uint32 * len(kits))(*kits), len(kits)):
            raise ValueError(f'bad kits {kits}')

    def reset(self, seed, map_id):
        self.lib.research_reset(self.handle, seed, map_id); self.check()

    def set_layout(self, player, version, ablate):
        if self.lib.research_set_ablation(self.handle, player, int(ablate)) or self.lib.research_set_layout(self.handle, player, int(version)):
            raise ValueError('Bad observation layout selection')

    @property
    def state(self):
        return self.lib.research_state(self.handle).contents

    def step(self, actions):
        actions = np.ascontiguousarray(actions, dtype=np.float32).reshape(self.players, layout.NUM_ATNS)
        self.lib.research_step(self.handle, actions.ctypes.data_as(C.POINTER(C.c_float)))
        self.check()

    def deaths(self):
        """(actor id, subject id) of every death event of the last step."""
        count, support = C.c_uint32(), C.c_int()
        events = self.lib.research_events(self.handle, C.byref(count), C.byref(support))
        if not support.value:
            raise RuntimeError('team games need a backend with death events (the Rust sim)')
        return [(events[i].actor, events[i].subject) for i in range(count.value) if events[i].kind == DEATH]

    def close(self):
        if self.handle:
            self.lib.research_close(self.handle); self.handle = None


HEAD_BOUNDS = [sum(layout.ACTION_SIZES[:i]) for i in range(layout.NUM_ATNS + 1)]


def actor(model, seats, rng):
    """One model's decision for all its seats in one batch, sampled as elo_ladder.Model does."""
    import torch
    policy = model.policy
    state = policy.initial(len(seats), 'cpu')

    def head(row, index):
        part = row[HEAD_BOUNDS[index]:HEAD_BOUNDS[index + 1]]
        weights = np.exp(part - part.max()); weights /= weights.sum()
        return rng.choice(len(part), p=weights)

    def act(rows, mask):
        nonlocal state
        batch = torch.from_numpy(np.ascontiguousarray(rows[seats, :policy.obs_size]))
        with torch.no_grad():
            out, state = policy.step_raw(batch, state)
            logits, _ = policy.game_logits(out)
        logits = logits.numpy()
        actions = []
        for k, seat in enumerate(seats):
            allowed = mask[seat] != 0
            row = np.where(allowed, logits[k], -np.inf)
            action = [0, head(row, 1), 0]
            if policy.aim_ar:
                combos = np.zeros(len(seats), dtype=np.int32); combos[k] = action[1]
                with torch.no_grad():
                    again, _ = policy.game_logits(out, combos)
                row = np.where(allowed, again.numpy()[k], -np.inf)
            action[0] = head(row, 0); action[2] = head(row, 2)
            actions.append(action)
        return actions
    return act


def play_game(env, models, task):
    """task: seats (4 model names), kits (12 ids), map, seed. Returns the result row."""
    env.set_kits(task['kits'])
    env.reset(task['seed'], task['map'])
    names = task['seats']
    for seat, name in enumerate(names):
        env.set_layout(seat, models[name].layout, models[name].ablate)
    rng = np.random.default_rng((task['seed'], task['map'], task['game'],
                                 int(hashlib.sha256('|'.join(names).encode()).hexdigest()[:8], 16)))
    actors = [(seats, actor(models[name], seats, rng))
              for name in dict.fromkeys(names) for seats in [[s for s in range(4) if names[s] == name]]]
    ids = [env.state.players[s].id for s in range(4)]
    team_kills = [0] * 4
    while not env.state.header.flags & 1:
        rows, mask = env.rows.copy(), env.mask.copy()
        actions = np.zeros((4, layout.NUM_ATNS), dtype=np.float32)
        for seats, act in actors:
            for seat, action in zip(seats, act(rows, mask)):
                actions[seat] = action
        env.step(actions)
        for killer, victim in env.deaths():
            if killer and killer != victim and killer in ids and victim in ids:
                k, v = ids.index(killer), ids.index(victim)
                if TEAMS[k] == TEAMS[v]:
                    team_kills[k] += 1
    s = env.state
    timeout = bool(s.header.flags & 2)
    won = [any(s.players[p].flags & 2 for p in range(4) if TEAMS[p] == t) for t in (0, 1)]
    winner = None if timeout or won[0] == won[1] else (0 if won[0] else 1)
    return dict(task, ticks=int(s.header.episode_ticks), winner=winner, timeout=timeout, team_kills=team_kills,
                kills=[int(s.players[p].kills) for p in range(4)], deaths=[int(s.players[p].deaths) for p in range(4)])


def play(models, tasks, workers, seed):
    def run(bucket):
        env = TeamEnvironment(seed, registry.EPISODE_TICKS, registry.DELAY)
        try:
            return [play_game(env, models, task) for task in bucket]
        finally:
            env.close()
    buckets = [tasks[i::workers] for i in range(min(workers, len(tasks)))]
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        return [game for bucket in pool.map(run, buckets) for game in bucket]


def tasks(pairs, seed_base, maps, partner):
    """Seed pair i: four kits and a map, A's team on team 0 then team 1 (kits follow the team slot)."""
    rng = np.random.default_rng(seed_base)
    order = league_games.map_order(maps, seed_base)
    out = []
    for pair in range(pairs):
        kits = [int(v) for v in rng.choice(POOL, 12)]
        for game, a_team in enumerate((0, 1)):
            team = {a_team: ['A', 'P' if partner else 'A'], 1 - a_team: ['B', 'P' if partner else 'B']}
            out.append(dict(map=order[pair % len(order)], seed=seed_base + pair, game=game, a_team=a_team,
                            seats=team[0] + team[1], kits=kits))
    return out


def summarize(played, label):
    points = lambda g: 0.5 if g['winner'] is None else float(g['winner'] == g['a_team'])
    per_pair = {}
    for g in played:
        per_pair.setdefault(g['seed'], []).append(points(g))
    pairs = np.array([np.mean(v) for v in per_pair.values()])
    rng = np.random.default_rng(0)
    boot = [rng.choice(pairs, pairs.size).mean() for _ in range(2000)]
    tk = {}
    for g in played:
        for seat, name in enumerate(g['seats']):
            tk.setdefault(name, [0, 0])
            tk[name][0] += g['team_kills'][seat]
        for name in set(g['seats']):
            tk[name][1] += 1
    # Team-kill games: games where any seat of A's team killed a teammate.
    tk_games = sum(any(g['team_kills'][s] for s in range(4) if TEAMS[s] == g['a_team']) for g in played)
    tk_games_b = sum(any(g['team_kills'][s] for s in range(4) if TEAMS[s] != g['a_team']) for g in played)
    return dict(label=label, games=len(played), score=float(pairs.mean()),
                ci=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                draws=sum(g['winner'] is None for g in played), timeouts=sum(g['timeout'] for g in played),
                team_kills_per_game={name: round(v[0] / v[1], 4) for name, v in tk.items()},
                a_team_teamkill_game_rate=tk_games / len(played), b_team_teamkill_game_rate=tk_games_b / len(played),
                mean_ticks=float(np.mean([g['ticks'] for g in played])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a', type=Path)
    p.add_argument('b', type=Path)
    p.add_argument('--partner', type=Path, help='seat 1 of both teams plays this checkpoint (A+P vs B+P)')
    p.add_argument('--pairs', type=int, default=250)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed-base', type=int, default=171001)
    p.add_argument('--label')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--games-out', type=Path, help='every game as one JSON line')
    a = p.parse_args(argv)
    import elo_ladder
    league_games.configure(a.out.parent / 'team-workers')
    models = {'A': elo_ladder.Model('A', a.a.resolve(), 'candidate'), 'B': elo_ladder.Model('B', a.b.resolve(), 'candidate')}
    if a.partner:
        models['P'] = elo_ladder.Model('P', a.partner.resolve(), 'candidate')
    todo = tasks(a.pairs, a.seed_base, map_catalogue.parse_map_ids(registry.MAPS), a.partner is not None)
    played = play(models, todo, a.workers, a.seed_base)
    label = a.label or f"{a.a} vs {a.b}" + (f" with {a.partner}" if a.partner else '')
    row = dict(summarize(played, label), a=str(a.a), b=str(a.b), partner=str(a.partner) if a.partner else None,
               seed_base=a.seed_base)
    print(f"{row['label']}: A's team scores {row['score']:.3f} [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}] over {row['games']} "
          f"games ({row['draws']} draws, {row['timeouts']} timeouts); team kills per game {row['team_kills_per_game']}; "
          f"games with a team kill: A's team {row['a_team_teamkill_game_rate']:.3f}, B's {row['b_team_teamkill_game_rate']:.3f}")
    with open(a.out, 'a') as f:
        f.write(json.dumps(row) + '\n')
    if a.games_out:
        with open(a.games_out, 'w') as f:
            for g in played:
                f.write(json.dumps(g) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
