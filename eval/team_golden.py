"""1v1 golden for team play (docs/TEAM_PLAY.md): f5 vs A20 on fixed seeds through research_open,
hashing every decision's observation rows, masks, telemetry (reward columns included) and game
state; the last three games put a scripted bot on player 1. Run once with
BOPL_RESEARCH_LIBRARY=<library built at the base commit> and once with the current one
(BOPL_RL_BACKEND=rust); the printed digests must match.

    venv/bin/python eval/team_golden.py [workers-dir] > digests.json
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'eval'), str(ROOT/'league'), str(ROOT/'native/tools')]
import registry, layout, elo_ladder, games as league_games
import ctypes as C
from team_games import actor
league_games.configure(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'native/artifacts/rl/team-golden')
import environment
CK = ROOT/'native/artifacts/rl/sweeps'
models = [elo_ladder.Model('f5', (CK/'final-5/run/checkpoints/policy-18666684416.bin').resolve(), 'candidate'),
          elo_ladder.Model('a20', (CK/'final-1/anneal4b/checkpoints/policy-26666729472.bin').resolve(), 'candidate')]
def state_bytes(s):
    # Header, players and counters; the entity and slime pointers differ per process.
    return bytes(s.header) + bytes(s.players) + bytes(s.telemetry)
out = {}
import os
# Last three: player 1 is a scripted bot (mixed, mixed-v2, archer-v2 at hard), for bopl_bots.h's foe choice.
GAMES=[(9001, 0, 0), (9002, 17, 0), (9003, 55, 0), (9004, 101, 0), (9005, 123, 0), (9006, 3, 0),
       (9101, 8, 4), (9102, 40, 9), (9103, 77, 6)][:int(os.environ.get('GOLDEN_GAMES', 9))]
TRACE=os.environ.get('GOLDEN_TRACE')
trace=open(TRACE,'w') if TRACE else None
for game, (seed, map_id, bot) in enumerate(GAMES):
    env = environment.Environment(seed, map_id, registry.EPISODE_TICKS, registry.DELAY,
                                  reward=dict(kill=0.288, timeout=-1.0, win=1, loss=-1, death=-0.288))
    for p, m in enumerate(models): env.set_layout(p, m.layout, m.ablate)
    if bot: env.bot(1, bot, 2)
    rng = np.random.default_rng(seed)
    acts = [actor(m, [p], rng) for p, m in enumerate(models)]
    h = hashlib.sha256(); steps = 0
    while not env.state.header.flags & 1:
        rows, mask = env.rows.copy(), env.mask.copy()
        h.update(rows.tobytes()); h.update(mask.tobytes()); h.update(env.telemetry.tobytes())
        if trace: trace.write(' '.join(hashlib.sha256(x).hexdigest()[:12] for x in (rows.tobytes(), mask.tobytes(), env.telemetry.tobytes(), state_bytes(env.state)))+'\n')
        actions = np.array([acts[p](rows, mask)[0] for p in range(2)], dtype=np.float32)
        env.step(actions); steps += 1
    h.update(env.rows.tobytes()); h.update(env.telemetry.tobytes()); h.update(state_bytes(env.state))
    out[f'{seed}/{map_id}/{bot}'] = dict(steps=steps, digest=h.hexdigest())
    env.close()
print(json.dumps(out, indent=1))
