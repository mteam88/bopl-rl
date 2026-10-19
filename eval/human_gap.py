"""Human 1v1 replays as offense_diag rows, so humans and bots are measured by one definition.

    BOPL_RL_BACKEND=rust venv/bin/python eval/human_gap.py --out FILE.jsonl [--workers 3]

Re-steps every 2-player replay of the human corpora (eval/human_dataset.corpus: the Mac
human-vs-bot archive and the older human-human / human-vs-agent recordings) in the bit-exact
q32 sim, tick by tick, and writes one row per seat in native/rl/research/offense/offense_diag.py's
format with that seat as A (`side`), so offense_metrics.py summarizes humans, bots playing
humans, and bot-vs-bot games with the same code (KH_PRE, ckpm, len_s, sd, free_win, apm, idle,
engage_s, close_frac, first_hit_s, loser-death classes).

Decisions are every 4 ticks from the 240-tick warmup, as the bots decide. A seat's buttons at a
decision are the OR of its 4 recorded ticks (eval/human_dataset.py's "or" rule), so presses and
button rates are counted at the bots' granularity. Deaths come from the sim's events with the
v4.2 credit (eval/kill_credit.py); `tick` is the level tick, as in offense_diag (episode tick
+ 240). Extra fields: who (human / bot name), session (the bootstrap unit: `seed` is set to a
session hash), stocks, faithful (the sim ended the round on the recorded tick with the recorded
winners; analyses keep faithful rows only), presses_by_ability {ability id: rising edges}, and per
player `tel`: the sim's RlTelemetry counters (eval/telemetry.py EVENT_FIELDS: Dash/Bow/Platform
activations, arrows fired, lethal arrow hits, deaths by class), as eval/aggression.py reads them.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

FIX = 4294967296.
WARMUP, REPEAT = 240, 4
WINDOW, CLOSE, THREAT = 45, 12.0, 3.0   # offense_diag: 3 s of decisions, striking range, threat radius
DEATH_CAUSE = {0: 'unknown', 1: 'contact', 2: 'projectile', 3: 'explosion', 4: 'beam', 5: 'squished', 6: 'eaten',
               7: 'ring-out', 8: 'self', 9: 'other'}


def entities(state):
    import ctypes as C
    from environment import Entity
    n = state.header.entity_count
    rows = C.cast(state.entities, C.POINTER(Entity))
    return [(e.owner, e.x / FIX, e.y / FIX) for e in (rows[i] for i in range(n)) if e.flags & 1 and not e.flags & 4]


def measure(job):
    import os
    os.environ.setdefault('BOPL_RL_BACKEND', 'rust')
    from replay_data import decode, ReplayWorld
    from human_dataset import SIM
    import kill_credit
    from telemetry import EVENT_FIELDS
    meta, inputs = decode(job['replay'])
    if len(meta['players']) != 2:
        return []
    btn = inputs['buttons'].astype(np.int64)
    ids = [p['id'] for p in meta['players']]
    hist = [[], []]
    deaths, engage, threat = [], [None, None], [None, None]
    close_n = both_n = 0
    dist_sum = 0.
    decisions, presses = 0, np.zeros(2, np.int64)
    bits = np.zeros((2, 4), np.int64)
    prev_b = [0, 0]
    by_ability = [{}, {}]
    log = Path('/tmp/human-gap-logs') / (Path(job['replay']).stem + '.log')
    with ReplayWorld(meta, log, game=str(SIM), check_identity=False) as world:
        end = meta['ticks']
        for f in range(meta['ticks']):
            if world.ended:
                end = f
                break
            world.step([(int(inputs['angle'][f, i]), int(btn[f, i])) for i in range(2)])
            s = world.state
            for e in world.events():
                if e.kind != kill_credit.DEATH or e.subject not in ids:
                    continue
                v = ids.index(e.subject)
                h = hist[v] or [(0, 0, 0, 1e9, 1e9, 1)]
                deaths.append(dict(tick=int(e.tick), victim=v, actor=int(e.actor), creator=int(e.creator),
                                   cause=DEATH_CAUSE.get(e.cause, e.cause), source_ability=int(e.source_ability),
                                   actor_is=kill_credit.credit(e.actor, e.creator, e.subject),
                                   game_actor_is=kill_credit.game_credit(e.actor, e.subject),
                                   dmin_opp_3s=round(min(r[3] for r in h), 2), dmin_oppent_3s=round(min(r[4] for r in h), 2)))
            if f + 1 < WARMUP or (f + 1 - WARMUP) % REPEAT:
                continue
            # a decision: the state after f+1 ticks; the seat's buttons are the OR of the window it closes
            decisions += 1
            pl = s.players
            xy = [(pl[i].x / FIX, pl[i].y / FIX) for i in (0, 1)]
            alive = [bool(pl[i].flags & 1) for i in (0, 1)]
            ents = entities(s)
            tick = f + 1 - WARMUP
            for i in (0, 1):
                o = 1 - i
                b = int(np.bitwise_or.reduce(btn[max(f + 1 - REPEAT, 0):f + 1, i])) & 0b1111
                for k in range(4):
                    bits[i, k] += (b >> k) & 1
                rising = b & ~prev_b[i] & 0b1110
                presses[i] += bin(rising).count('1')
                for k in (1, 2, 3):
                    if rising >> k & 1 and k - 1 < len(meta['players'][i]['abilities']):
                        a = meta['players'][i]['abilities'][k - 1]
                        by_ability[i][a] = by_ability[i].get(a, 0) + 1
                prev_b[i] = b
                d_opp = math.hypot(xy[i][0] - xy[o][0], xy[i][1] - xy[o][1])
                theirs = [math.hypot(x - xy[i][0], y - xy[i][1]) for owner, x, y in ents if owner == ids[o]]
                d_ent = min(theirs) if theirs else 1e9
                if d_ent < THREAT and threat[o] is None and alive[i]:
                    threat[o] = tick
                hist[i].append((tick, xy[i][0], xy[i][1], d_opp, d_ent, alive[i]))
                del hist[i][:-WINDOW]
            if all(alive):
                d = math.hypot(xy[0][0] - xy[1][0], xy[0][1] - xy[1][1])
                both_n += 1; dist_sum += d
                if d < CLOSE:
                    close_n += 1
                    engage = [x if x is not None else tick for x in engage]
        final = world.state
        won = [bool(final.players[i].flags & 2) for i in (0, 1)]
        tel = [{k: int(getattr(final.telemetry[i], k)) for k in EVENT_FIELDS} for i in (0, 1)]
    recorded = job.get('winners')
    faithful = end >= meta['ticks'] - 4 and (recorded is None or won == [bool(w) for w in recorded])
    ticks = max(end - WARMUP, 0)
    who = job['who']
    rows = []
    for a in (0, 1):
        b = 1 - a
        outcome = 'draw' if won[a] == won[b] else 'a' if won[a] else 'b'
        rows.append(dict(seed=int(hashlib.sha256(job['session'].encode()).hexdigest()[:8], 16), map=meta['level'], side=a,
                         session=job['session'], replay=Path(job['replay']).name, source=job['source'], who=who[a], opponent=who[b],
                         kits=[meta['players'][0]['abilities'], meta['players'][1]['abilities']], stocks=meta['stocks'],
                         faithful=faithful, ticks=ticks, outcome=outcome, decisions=decisions, deaths=deaths,
                         players=[dict(won=won[i], buttons=(bits[i] / max(decisions, 1)).round(4).tolist(),
                                       presses_by_ability=by_ability[i], tel=tel[i]) for i in (0, 1)],
                         offense=dict(engage_tick=engage, close_frac=[round(close_n / max(both_n, 1), 4)] * 2,
                                      mean_dist=[round(dist_sum / max(both_n, 1), 2)] * 2, presses=presses.tolist(),
                                      threat_tick=threat)))
    return rows


def safe(job):
    try:
        return measure(job)
    except Exception as error:   # a replay the sim cannot load is reported, not fatal
        return f'{type(error).__name__}: {error}'


def jobs():
    """corpus() rows with a name per seat: 'human', the bot (Mac archive), or 'agent' (old corpus)."""
    from human_dataset import corpus, MAC
    agents = {}
    for line in open(MAC / 'games.jsonl'):
        g = json.loads(line)
        if g.get('replay'):
            agents[Path(g['replay']).name] = {
                i: (g.get('bot') or Path(str(g.get('agent') or 'agent')).name) if p.get('agent') else 'keyboard'
                for i, p in enumerate(g.get('players') or []) if p.get('agent') or p.get('keyboard')}
    out = []
    for j in corpus({2}):
        names = agents.get(Path(j['replay']).name, {})
        who = ['human' if i in j['seats'] else names.get(i, 'agent' if 'agent' in j['source'] else 'keyboard')
               for i in range(2)]
        out.append(dict(replay=j['replay'], source=j['source'], session=j['session'], winners=j['winners'], who=who))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--limit', type=int)
    a = p.parse_args(argv)
    todo = jobs()[:a.limit]
    done, failed = 0, []
    with ProcessPoolExecutor(a.workers) as pool, open(a.out, 'w') as f:
        for job, rows in zip(todo, pool.map(safe, todo, chunksize=4)):
            if isinstance(rows, str):
                failed.append((job['replay'], rows)); continue
            for row in rows:
                f.write(json.dumps(row) + '\n')
            done += 1
    print(f'{done} replays, {len(failed)} failed')
    for rep, error in failed[:10]:
        print(' ', Path(rep).name, error)


if __name__ == '__main__':
    main()
