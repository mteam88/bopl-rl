"""eval/aggression.py's movement and attack rates, read from human-vs-bot replays.

    BOPL_RL_BACKEND=rust venv/bin/python eval/aggression_replays.py native/artifacts/rl/human-mac --out DIR

DIR/games.jsonl is the play_vs_agent_results.py archive (games.jsonl beside the .rep files).
Each game's replay is re-stepped in the bit-exact (q32) Rust sim; every 4 ticks after the 240-tick warmup
(the bots' decision cadence), for each seat: distance to the nearest live enemy (another
team) while both are alive, and whether its own movement closed on that enemy by more
than aggression.CLOSE. Ability presses per minute are rising edges of the three ability
buttons in the recorded inputs. One row per seat per game, tagged bot or human, and
`faithful` when the sim ended the round on the recorded tick with the recorded winners
(the sim has changed since some recordings; a diverged replay's positions are not the game's).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from aggression import CLOSE, FIX  # noqa: E402

WARMUP = 240
# The bit-exact Q32.32 build (make rust-q32): the f64 training build rounds differently from
# the game, and a long round's positions drift away from the recording (Makefile, RUST_ROLLOUT).
SIM = HERE.parent / 'native/target-q32/release/libbopl_sim.so'


def seats(game):
    """{player id: bot name or None} for a games.jsonl row."""
    bots = {a['player']: a.get('bot') or Path(a['agent']).stem for a in game.get('agents') or []}
    if not bots and game.get('agent_player'):
        bots = {game['agent_player']: game.get('bot') or game['agent'].split('/sweeps/')[-1]}
    return {p['id']: bots.get(p['id']) for p in game['players']}


def measure(job):
    from replay_data import decode, ReplayWorld
    game, rep, logs, sim = job
    meta, inputs = decode(rep)
    who = seats(game)
    count = len(meta['players'])
    teams = [p['team'] for p in meta['players']]
    dist = [[] for _ in range(count)]
    closing = [0] * count
    moves = [0] * count
    with ReplayWorld(meta, Path(logs) / (Path(rep).stem + '.log'), game=sim, check_identity=False) as world:
        before = None
        end = meta['ticks']
        for t in range(meta['ticks']):
            if world.ended:
                end = t
                break
            world.step(list(zip(inputs[t]['angle'], inputs[t]['buttons'])))
            if t < WARMUP or t % 4:
                continue
            s = world.state
            now = [(s.players[i].x / FIX, s.players[i].y / FIX, bool(s.players[i].flags & 1)) for i in range(count)]
            if before:
                for i in range(count):
                    x0, y0, alive0 = before[i]
                    enemies = [j for j in range(count) if teams[j] != teams[i] and before[j][2] and now[j][2]]
                    if not (alive0 and now[i][2] and enemies):
                        continue
                    j = min(enemies, key=lambda j: math.hypot(before[j][0] - x0, before[j][1] - y0))
                    gap = math.hypot(before[j][0] - x0, before[j][1] - y0)
                    dist[i].append(gap)
                    moves[i] += 1
                    if gap > 1e-6:
                        toward = ((now[i][0] - x0) * (before[j][0] - x0) + (now[i][1] - y0) * (before[j][1] - y0)) / gap
                        closing[i] += toward > CLOSE
            before = now
        final = world.state
    minutes = max(meta['ticks'] - WARMUP, 1) / 3600
    recorded = {q['id']: q['won_round'] for q in game['players']}
    # The sim must end the round where the game did with the same winners, else the replay
    # diverged (sim changes since the recording) and its positions are not the game's.
    faithful = end >= meta['ticks'] - 4 and all(bool(final.players[i].flags & 2) == recorded.get(q['id'])
                                                for i, q in enumerate(meta['players']))
    rows = []
    for i, p in enumerate(meta['players']):
        buttons = inputs[WARMUP:, i]['buttons'].astype(np.int64)
        presses = 0
        for b in (2, 4, 8):
            on = (buttons & b) != 0
            presses += int(np.count_nonzero(on[1:] & ~on[:-1]))
        rows.append(dict(session=game['session'], round=game.get('round'), replay=Path(rep).name, players=count, seat=p['id'],
                         bot=who.get(p['id']), human=who.get(p['id']) is None, abilities=p['abilities'],
                         seconds=minutes * 60, faithful=faithful, end_tick=end, won=bool(final.players[i].flags & 2),
                         distance=float(np.mean(dist[i])) if dist[i] else None,
                         closing=closing[i] / moves[i] if moves[i] else None,
                         presses_pm=presses / minutes, kills=int(final.players[i].kills)))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('archive', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--sim', type=Path, default=SIM, help='Rust sim library (default: the q32 build)')
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for line in open(a.archive / 'games.jsonl'):
        game = json.loads(line)
        rep = a.archive / Path(game.get('replay') or '').name
        if not game.get('failed') and game.get('replay') and rep.exists() and game.get('players'):
            jobs.append((game, str(rep), str(a.out / 'logs'), str(a.sim)))
    done, failed = 0, []
    with ProcessPoolExecutor(a.jobs) as pool, open(a.out / 'seats.jsonl', 'w') as f:
        for job, result in zip(jobs, pool.map(measure_safe, jobs)):
            if isinstance(result, str):
                failed.append((job[1], result)); continue
            for row in result:
                f.write(json.dumps(row) + '\n')
            done += 1
    print(f'{done} replays measured, {len(failed)} failed')
    for rep, error in failed[:10]:
        print(' ', Path(rep).name, error)


def measure_safe(job):
    try:
        return measure(job)
    except Exception as error:  # a replay the sim cannot load is reported, not fatal
        return f'{type(error).__name__}: {error}'


if __name__ == '__main__':
    main()
