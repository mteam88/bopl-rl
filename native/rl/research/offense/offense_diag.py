"""Offense diagnostics (OFFENSE-2026-10.md): sweeps/game-diag/eval/game_diag.py plus, per game, an `offense` dict
(per player: engage_tick = first episode tick with both alive and bodies < 12u apart, close_frac = share of
both-alive decisions < 12u, mean_dist, presses = rising edges of the three ability buttons,
threat_tick = first episode tick one of the player's entities is within 3u of the living opponent: the
time-to-first-hit proxy, as Bopl is one-hit-kill; owner-0 entities (guided missiles, cast platforms) are missed) and --kits fixed
(--kit X,Y,Z on both sides, e.g. the 2,4,14 mirror for the Sept specialist v4-full).

Game-level diagnostics: play a matchup and write one JSON row per game with how it ended.

    venv/bin/python eval/game_diag.py --a PATH --b {PATH|random|scripted:NAME|member:NAME}
        --kits {any|league|bot} --pairs N [--salt S] [--temp-a T] --out games.jsonl

kits: any = all_combos.py deal (random 3-kits, side-swapped, same seed_base => same games);
league = league/games.py matchup of the two registry members (exploiter set kits, mirrored);
bot = A random kit, B (scripted) on its fixed 2,4,14, as in training.
--salt changes only the action-sampling RNG (same map, kits, sim seed), for noise studies.
Per game: ticks, outcome, draw_kind, each side's death (sim cause, telemetry class, death-event
cause/actor/source ability, position, peak y, min distance to the opponent / its entities in the
last 1s and 3s), kills, observation-overflow decisions and what overflowed, button rates.
Plays exactly like elo_ladder.play_games (same participants, masks, per-game RNG when salt=0).
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import os as _os
HERE = Path(__file__).resolve().parent
ROOT = Path(_os.environ.get('BOPL_ROOT') or HERE.parents[3])
sys.path[:0] = [str(ROOT / 'league'), str(HERE), str(ROOT / 'eval'), str(ROOT / 'native/tools')]
import registry  # noqa: E402
import games as league_games  # noqa: E402
import elo_ladder  # noqa: E402
import kill_credit  # noqa: E402
import map_catalogue  # noqa: E402

POOL = [a for a in range(1, 39) if a not in (12, 15)]
FIX = 1 / 4294967296.
ENT_DT = np.dtype([(n, '<u4') for n in ('id', 'kind', 'layer', 'owner', 'flags', 'hierarchy', 'r0', 'r1')]
                  + [(n, '<i8') for n in ('x', 'y', 'vx', 'vy', 'rx', 'ry', 'ux', 'uy', 'radius', 'rot', 'av', 'im', 'cr')])
WINDOW = 45  # decisions = 3 s at 15 Hz
CLOSE = 12.0  # units: styles/pressure.toml's striking range
THREAT = 3.0  # units: an entity of the player's within this of the living opponent = a threat (first hit proxy)


DEATH_CAUSE = {0: 'unknown', 1: 'contact', 2: 'projectile', 3: 'explosion', 4: 'beam', 5: 'squished', 6: 'eaten',
               7: 'ring-out', 8: 'self', 9: 'other'}
SIM_CAUSE = {0: 'other', 1: 'squished', 2: 'accident', 3: 'eaten', 4: 'beamed', 5: 'not-dead-yet'}


def entities(state):
    n = state.header.entity_count
    if n == 0:
        return np.zeros(0, ENT_DT)
    addr = C.cast(state.entities, C.c_void_p).value
    raw = (C.c_char * (n * ENT_DT.itemsize)).from_address(addr)
    e = np.frombuffer(raw, ENT_DT, count=n).copy()
    return e[(e['flags'] & 1 != 0) & (e['flags'] & 4 == 0)]


def play_games(by_name, tasks, limit, workers, seed_base, delay=9, on_result=None, loadouts=None, style=None, salt=0):
    """elo_ladder.play_games plus per-game diagnostics (see module doc)."""
    from environment import Environment, LIB
    import environment
    assert C.sizeof(environment.Entity) == ENT_DT.itemsize
    rows = (loadouts or {}).get('lineups') or []

    def play(bucket):
        import torch
        torch.set_num_threads(1)
        results = []
        with Environment(seed=seed_base + len(tasks), limit=limit, delay=delay, loadouts=loadouts) as env:
            pinned = False
            for task in bucket:
                index = task.get('lineup')
                if index is not None:
                    env.configure_loadouts(dict(loadouts, lineups=[rows[index]], mirror_share=0.)); pinned = True
                elif pinned:
                    env.configure_loadouts(loadouts); pinned = False
                env.reset(task['seed'], task['map'])
                key = (task['seed'], task['map'], task['side'],
                       int(hashlib.sha256((task['a'] + task['b']).encode()).hexdigest()[:8], 16))
                rng = np.random.default_rng(key if not salt else key + (salt,))
                env.bot(0, 0); env.bot(1, 0)
                for name, side in ((task['a'], task['side']), (task['b'], 1 - task['side'])):
                    attach = getattr(by_name[name], 'attach', None)
                    if attach: attach(env, side)
                actors = {task['side']: by_name[task['a']].start(task, task['side'], rng),
                          1 - task['side']: by_name[task['b']].start(task, 1 - task['side'], rng)}
                ab0 = env.ability_telemetry()
                ov0 = env.observation_overflow
                hist = [[], []]  # per player: (tick, x, y, d_opp, d_oppent, alive)
                deaths = []
                buttons = np.zeros((2, 4), np.int64)
                moved = np.zeros(2)
                decisions = 0
                ov_dec = 0
                max_ent = 0
                ov_kinds, ov_src, ov_far_proj, ov_far_oppproj = {}, {}, 0, 0
                last_xy = None
                engage = [None, None]; close_n = np.zeros(2); both_n = 0; dist_sum = 0.
                presses = np.zeros(2, np.int64); prev_b = [0, 0]; threat = [None, None]
                while not env.ended:
                    observation = env.rows.copy(); mask = env.mask.copy()
                    acts = np.stack([actors[0](observation, mask), actors[1](observation, mask)])
                    env.step(acts)
                    decisions += 1
                    for p in (0, 1):
                        b = int(acts[p][1])
                        for bit in range(4):
                            buttons[p, bit] += (b >> bit) & 1
                        presses[p] += bin((b & ~prev_b[p]) & 0b1110).count('1'); prev_b[p] = b
                    s = env.state
                    pl = s.players
                    xy = [(pl[i].x * FIX, pl[i].y * FIX) for i in (0, 1)]
                    if last_xy is not None:
                        for i in (0, 1):
                            moved[i] += abs(xy[i][0] - last_xy[i][0]) + abs(xy[i][1] - last_xy[i][1])
                    last_xy = xy
                    ents = entities(s)
                    max_ent = max(max_ent, len(ents))
                    for i in (0, 1):
                        o = 1 - i
                        d_opp = float(np.hypot(xy[i][0] - xy[o][0], xy[i][1] - xy[o][1]))
                        mine = ents[ents['owner'] == pl[o].id] if len(ents) else ents
                        d_ent = float(np.min(np.hypot(mine['x'] * FIX - xy[i][0], mine['y'] * FIX - xy[i][1]))) if len(mine) else 1e9
                        h = hist[i]
                        if d_ent < THREAT and threat[o] is None and (pl[i].flags & 1): threat[o] = int(s.header.episode_ticks)
                        h.append((s.header.episode_ticks, xy[i][0], xy[i][1], d_opp, d_ent, pl[i].flags & 1))
                        if len(h) > WINDOW: del h[0]
                    if (pl[0].flags & 1) and (pl[1].flags & 1):
                        d01 = float(np.hypot(xy[0][0] - xy[1][0], xy[0][1] - xy[1][1]))
                        both_n += 1; dist_sum += d01
                        if d01 < CLOSE:
                            close_n += 1
                            for i in (0, 1):
                                if engage[i] is None: engage[i] = int(s.header.episode_ticks)
                    ov = env.observation_overflow
                    if ov > ov0:
                        ov_dec += 1
                        for k in np.unique(ents['kind']):
                            ov_kinds[int(k)] = ov_kinds.get(int(k), 0) + int((ents['kind'] == k).sum())
                        for k in np.unique(ents['r0']):
                            ov_src[int(k)] = ov_src.get(int(k), 0) + int((ents['r0'] == k).sum())
                        # Projectiles beyond the 48 nearest (center distance - radius) of either player.
                        for i in (0, 1):
                            d = np.hypot(ents['x'] * FIX - xy[i][0], ents['y'] * FIX - xy[i][1]) - ents['radius'] * FIX
                            pick = (ents['flags'] & 128) != 0
                            d = np.where(pick, d - 1e12, d)
                            order = np.argsort(d)
                            far = ents[order[48:]]
                            proj = (far['flags'] & 256) != 0
                            ov_far_proj += int(proj.sum())
                            ov_far_oppproj += int((proj & (far['owner'] == pl[1 - i].id)).sum())
                        ov0 = ov
                    for e in env.events():
                        if e.kind != 1:
                            continue
                        victim = next((i for i in (0, 1) if pl[i].id == e.subject), None)
                        rec = dict(tick=int(e.tick), victim=victim, actor=int(e.actor), creator=int(e.creator), cause=DEATH_CAUSE.get(e.cause, e.cause),
                                   source_ability=int(e.source_ability), flags=int(e.flags),
                                   x=round(e.x * FIX, 2), y=round(e.y * FIX, 2), vx=round(e.vx * FIX, 2), vy=round(e.vy * FIX, 2))
                        if victim is not None:
                            h = hist[victim]
                            # v4.2 credit (eval/kill_credit.py): a Missile/Spike kill is its caster's
                            rec['actor_is'] = kill_credit.credit(e.actor, e.creator, e.subject)
                            rec['game_actor_is'] = kill_credit.game_credit(e.actor, e.subject)
                            rec['dmin_opp_1s'] = round(min(r[3] for r in h[-15:]), 2)
                            rec['dmin_opp_3s'] = round(min(r[3] for r in h), 2)
                            rec['dmin_oppent_1s'] = round(min(r[4] for r in h[-15:]), 2)
                            rec['dmin_oppent_3s'] = round(min(r[4] for r in h), 2)
                            rec['peak_y_3s'] = round(max(r[2] for r in h), 2)
                        deaths.append(rec)
                final = env.frame()
                timeout = bool(final['flags'] & 2)
                a, b = task['side'], 1 - task['side']
                a_won = bool(final['players'][a]['flags'] & 2) and not timeout
                b_won = bool(final['players'][b]['flags'] & 2) and not timeout
                result = dict(task, salt=salt, ticks=final['t'],
                              outcome='draw' if timeout or a_won == b_won else ('a' if a_won else 'b'))
                if result['outcome'] == 'draw':
                    result['draw_kind'] = 'timeout' if timeout else ('both-dead' if a_won else 'neither-won')
                ab1 = env.ability_telemetry()
                result['players'] = [dict(
                    alive=bool(p['flags'] & 1), won=bool(p['flags'] & 2), cause=SIM_CAUSE.get(p['cause_of_death'], p['cause_of_death']),
                    kills=p['kills'], deaths=p['deaths'], x=p['x'], y=p['y'],
                    tel={k: final['telemetry'][i][k] for k in ('deaths_environment', 'deaths_opponent', 'deaths_self', 'deaths_other')},
                    buttons=(buttons[i] / max(decisions, 1)).round(3).tolist(), moved_per_dec=round(moved[i] / max(decisions, 1), 3))
                    for i, p in enumerate(final['players'][:2])]
                result['deaths'] = deaths
                result['decisions'] = decisions
                result['offense'] = dict(engage_tick=engage, close_frac=(close_n / max(both_n, 1)).round(4).tolist(),
                                         mean_dist=[round(dist_sum / max(both_n, 1), 2)] * 2, presses=presses.tolist(), threat_tick=threat)
                result['overflow'] = dict(decisions=ov_dec, max_entities=max_ent, kinds=ov_kinds, sources=ov_src,
                                          far_projectiles=ov_far_proj, far_opp_projectiles=ov_far_oppproj,
                                          by_ability={i: [ab1['entity_rows'][i] - ab0['entity_rows'][i],
                                                          ab1['entity_overflow'][i] - ab0['entity_overflow'][i]]
                                                      for i in range(39) if ab1['entity_overflow'][i] - ab0['entity_overflow'][i] > 0})
                if loadouts is not None:
                    played = env.loadouts
                    result['lineup_ids'] = played[0] + played[1]
                if on_result: on_result(result)
                results.append(result)
                if len(results) % 25 == 0:
                    print(f'bucket progress {len(results)}/{len(bucket)}', flush=True)
        return results
    if not tasks: return []
    from concurrent.futures import ThreadPoolExecutor
    buckets = [tasks[i::workers] for i in range(min(workers, len(tasks)))]
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        return [g for bucket in pool.map(play, buckets) for g in bucket]


def participant(spec, name):
    if spec == 'random':
        r = elo_ladder.Random(); r.name = name; return r, None
    if spec.startswith('scripted:'):
        return elo_ladder.Scripted(spec.split(':', 1)[1]), registry.by_name(registry.load(verify=False))[spec.split(':', 1)[1]]
    if spec.startswith('member:'):
        row = registry.by_name(registry.load(verify=False))[spec.split(':', 1)[1]]
        return elo_ladder.Model(name, registry.ROOT / row['path'], 'candidate'), row
    return elo_ladder.Model(name, Path(spec).resolve(), 'candidate'), None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True); p.add_argument('--b', required=True)
    p.add_argument('--kits', choices=('any', 'league', 'bot', 'fixed'), default='any')
    p.add_argument('--kit', default='2,4,14', help='--kits fixed: both sides play this kit')
    p.add_argument('--pairs', type=int, default=500); p.add_argument('--start', type=int, default=0)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed-base', type=int, default=131001)
    p.add_argument('--salt', type=int, default=0)
    p.add_argument('--temp-a', type=float); p.add_argument('--temp-b', type=float)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    A, ra = participant(a.a, 'A'); B, rb = participant(a.b, 'B')
    A.name, B.name = 'A', 'B'
    if a.temp_a: A.temperature = (a.temp_a,) * 3
    if a.temp_b: B.temperature = (a.temp_b,) * 3
    built = {'A': A, 'B': B}
    league_games.configure(a.out.parent / 'workers')
    maps = map_catalogue.parse_map_ids(registry.MAPS)
    if a.kits == 'league':
        ra = ra or dict(id='A', kind='main', loadouts=None)
        ra = dict(ra, id='A'); rb = dict(rb, id='B')
        todo = league_games.matchup(ra, rb, maps, a.pairs, a.seed_base, a.start)
    else:
        rng = np.random.default_rng(a.seed_base)
        order = league_games.map_order(maps, a.seed_base)
        if a.kits == 'bot':  # training never pins the bot on a space map (bopl_pin_bot)
            import map_set
            space = set(map_set.space_maps())
            order = [m for m in order if m not in space]
        todo = []
        for pair in range(a.start + a.pairs):
            x, y = [int(v) for v in rng.choice(POOL, 3)], [int(v) for v in rng.choice(POOL, 3)]
            if a.kits == 'bot':
                y = [2, 4, 14]
            if a.kits == 'fixed':
                x = y = [int(v) for v in a.kit.split(',')]
            if pair < a.start:
                continue
            for side in (0, 1):
                todo.append(dict(a='A', b='B', map=order[pair % len(order)], seed=a.seed_base + pair, side=side,
                                 kits=x + y if side == 0 else y + x))
    orig = elo_ladder.play_games
    elo_ladder.play_games = lambda *args, **kw: play_games(*args, salt=a.salt, **kw)
    try:
        played = league_games.play(built, todo, a.workers, a.seed_base)
    finally:
        elo_ladder.play_games = orig
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'a') as f:
        for g in played:
            g.update(label=f'{a.a} vs {a.b}', temp_a=a.temp_a, temp_b=a.temp_b, kits_mode=a.kits)
            g['a_kit'] = g['kits'][:3] if g['side'] == 0 else g['kits'][3:]
            g['b_kit'] = g['kits'][3:] if g['side'] == 0 else g['kits'][:3]
            f.write(json.dumps(g) + '\n')
    pts = [1.0 if g['outcome'] == 'a' else .5 if g['outcome'] == 'draw' else 0. for g in played]
    print(f'{len(played)} games, A score {np.mean(pts):.3f}, median ticks {np.median([g["ticks"] for g in played]):.0f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
