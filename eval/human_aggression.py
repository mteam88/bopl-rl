"""eval/aggression.py's movement metrics for the human seats of the faithful converted replays.

    venv/bin/python eval/human_aggression.py --data native/artifacts/rl/human-bc/data-2p [--kit 2,4,14] --out FILE.json

Re-steps each replay with the recorded inputs and, at every decision tick (every 4 ticks
from the 240-tick warmup, as the research environment decides), measures for the human
seat exactly what aggression.py measures for a subject: mean distance while both bodies
live, and the share of decisions whose own movement takes it toward (closing) or away
from (retreating) the opponent by more than CLOSE units. Also the human's ability press
rate (rising edges of each ability-slot button per minute; presses, not activations,
since replays carry no activation counter). Means per game with a session-cluster
bootstrap 95% interval. eval/aggression_replays.py reads the same from the Mac archive
(human-vs-bot only, all seats); this one covers the human seats of both corpora, the
faithful ones the BC net trains on, filtered to one kit.
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

FIX = 4294967296.
CLOSE = 0.05
WARMUP = 240


def one(job):
    import os
    os.environ.setdefault('BOPL_RL_BACKEND', 'rust')
    from replay_data import decode, ReplayWorld
    from human_dataset import SIM
    path, seat, log = job
    meta, inputs = decode(path)
    w = ReplayWorld(meta, Path(log), game=str(SIM), check_identity=False)
    me, them = seat, 1 - seat
    ang, btn = inputs['angle'].astype(int), inputs['buttons'].astype(int)
    dist, closing, retreating, dec = [], 0, 0, 0
    prev = None
    try:
        for f in range(meta['ticks']):
            if w.ended:
                break
            s = w.state
            if f >= WARMUP and (f - WARMUP) % 4 == 0:
                p, q = s.players[me], s.players[them]
                cur = (p.x / FIX, p.y / FIX, q.x / FIX, q.y / FIX, bool(p.flags & 1 and q.flags & 1), bool(p.flags & 1))
                if prev is not None and prev[4] and cur[5]:
                    px, py, qx, qy = prev[:4]
                    gap = math.hypot(qx - px, qy - py)
                    dist.append(gap); dec += 1
                    if gap > 1e-6:
                        toward = ((cur[0] - px) * (qx - px) + (cur[1] - py) * (qy - py)) / gap
                        closing += toward > CLOSE; retreating += toward < -CLOSE
                prev = cur
            w.step([(int(ang[f, i]), int(btn[f, i])) for i in range(len(meta['players']))])
        end = f
    finally:
        w.close()
    b = btn[WARMUP:end, me]
    presses = sum(int(((b[1:] >> k) & 1 & ~((b[:-1] >> k) & 1)).sum()) for k in (1, 2, 3))
    minutes = max(1e-9, (end - WARMUP) / 3600)
    return dict(distance=float(np.mean(dist)) if dist else None, closing=closing / dec if dec else None,
                retreating=retreating / dec if dec else None, presses_pm=presses / minutes, jumps_pm=int(((b[1:] & 1) & ~(b[:-1] & 1)).sum()) / minutes)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--kit', default='2,4,14', help='only seats with this kit (sorted compare); "all" for every kit')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    from human_dataset import corpus
    reps = {Path(j['replay']).name: j['replay'] for j in corpus({2})}
    rows = [json.loads(l) for l in (a.data / 'games.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['faithful'] and (a.kit == 'all' or sorted(r['kit']) == sorted(int(x) for x in a.kit.split(',')))]
    jobs = [(reps[r['replay']], r['seat'], f'/tmp/human-aggression-{r["replay"]}.log') for r in rows]
    with ProcessPoolExecutor(a.workers) as pool:
        res = list(pool.map(one, jobs))
    rng = np.random.default_rng(0)
    out = dict(kit=a.kit, games=len(res))
    for k in ('distance', 'closing', 'retreating', 'presses_pm', 'jumps_pm'):
        by = {}
        for r, x in zip(rows, res):
            if x[k] is not None:
                by.setdefault(r['session'], []).append(x[k])
        keys = list(by)
        f = lambda ks: float(np.mean([v for s in ks for v in by[s]]))
        bs = [f(rng.choice(keys, len(keys))) for _ in range(2000)]
        out[k] = [round(f(keys), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
