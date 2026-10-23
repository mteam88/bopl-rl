"""How a checkpoint uses Drill: per-decision log of every drill-form decision of seat A, per-episode
summaries, per-game outcomes, and (optionally) the v4 flat block of drill frames and full visual frames.

    drill_look.py --a CKPT --a-kit 26,2,2 --b CKPT --b-kit random|ids --games N --seed-base S --out DIR
                  [--rows-every K] [--visual K]

Seat A plays both sides over seed pairs (pair p: side 0 and side 1, same seed, map and kits).
Ground truth "underground" = the terrain SDF (rounded rects on layers 11/16/21, the v4 writer's walls) at
A's player position < 0 at the decision (undelayed); the row's terrain[9] is the 9-tick-delayed SDF the
policy sees. Kits for B with 'random': 3 iid draws from 2..38 minus {12, 15} (the trainer's pool 'all',
duplicates allowed).
"""
import argparse, gzip, json, math, sys, time
from pathlib import Path
import numpy as np

T = Path(__file__).resolve().parents[4]   # repo root
sys.path[:0] = [str(T / 'eval'), str(T / 'league'), str(T / 'native/tools')]
import registry, elo_ladder, map_catalogue, games as league_games  # noqa
from environment import Environment  # noqa
import layout as L  # noqa

FIX = 4294967296.
DRILL = 26
POOL = [i for i in range(2, 39) if i not in (12, 15)]
DEATH_CAUSE = ['unknown', 'contact', 'projectile', 'explosion', 'beam', 'squished', 'eaten', 'ring_out', 'self', 'other']
LEAGUE = {x['name']: x for x in json.loads((T / 'league/league.json').read_text())['members']}


def ckpt(name):
    return T / LEAGUE[name]['path'] if name in LEAGUE else Path(name)


def walls(st):
    out = []
    for i in range(st.header.entity_count):
        e = st.entities[i]
        if not (e.flags & 1) or (e.flags & 4) or e.kind != 2 or e.layer not in (11, 16, 21):
            continue
        rx, ry, ux, uy = e.right_x / FIX, e.right_y / FIX, e.up_x / FIX, e.up_y / FIX
        ex, ey = math.hypot(rx, ry), math.hypot(ux, uy)
        if ex > 1e-6: rx, ry = rx / ex, ry / ex
        else: rx, ry = 1.0, 0.0
        if ey > 1e-6: ux, uy = ux / ey, uy / ey
        else: ux, uy = -ry, rx
        out.append((e.x / FIX, e.y / FIX, rx, ry, ux, uy, ex, ey, e.radius / FIX, e.layer))
    return np.array(out) if out else np.zeros((0, 10))


def sdf(W, x, y):
    if not len(W): return 1e9
    px, py = x - W[:, 0], y - W[:, 1]
    lx, ly = np.abs(px * W[:, 2] + py * W[:, 3]) - W[:, 6], np.abs(px * W[:, 4] + py * W[:, 5]) - W[:, 7]
    out = np.hypot(np.maximum(lx, 0), np.maximum(ly, 0)) + np.minimum(np.maximum(lx, ly), 0) - W[:, 8]
    return float(out.min())


def thickness(W, x, y, dx, dy, step=0.5, limit=60.0):
    """Distance along (dx, dy) from (x, y) until the SDF turns positive (inside) or the ground ahead
    ends; for a body underground: how much ground is left along the heading. -1 if it never exits."""
    t = 0.0
    while t < limit:
        if sdf(W, x + dx * t, y + dy * t) > 0: return t
        t += step
    return -1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True); p.add_argument('--a-kit', required=True)
    p.add_argument('--b', required=True); p.add_argument('--b-kit', default='random')
    p.add_argument('--games', type=int, default=200); p.add_argument('--seed-base', type=int, default=4100001)
    p.add_argument('--maps', default='training-111'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rows-every', type=int, default=1, help='keep the flat block of every k-th drill decision (0 = none)')
    p.add_argument('--visual', type=int, default=0, help='keep full frames for the first K games')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    league_games.configure(a.out / 'work')
    A = elo_ladder.Model('A', ckpt(a.a), 'candidate'); B = elo_ladder.Model('B', ckpt(a.b), 'candidate')
    akit = [int(v) for v in a.a_kit.split(',')]
    maps = map_catalogue.parse_map_ids(a.maps)
    rng_k = np.random.default_rng(a.seed_base)
    meta = dict(a=str(ckpt(a.a)), b=str(ckpt(a.b)), a_kit=akit, b_kit=a.b_kit, games=a.games, seed_base=a.seed_base,
                maps=a.maps, a_ablate=A.ablate, b_ablate=B.ablate, a_layout=A.layout)
    (a.out / 'meta.json').write_text(json.dumps(meta, indent=1))
    fg = open(a.out / 'games.jsonl', 'w'); fd = open(a.out / 'decisions.jsonl', 'w')
    rows, rowmeta = [], []
    FO, TO, RO = 0, L.V4_TERRAIN_OFFSET, L.V4_RAY_OFFSET
    pairs = (a.games + 1) // 2
    with Environment(seed=a.seed_base, limit=registry.EPISODE_TICKS, delay=registry.DELAY) as env:
        for pair in range(pairs):
            bkit = [int(v) for v in a.b_kit.split(',')] if a.b_kit != 'random' else [int(v) for v in rng_k.choice(POOL, 3)]
            mp = maps[pair % len(maps)] if pair < len(maps) else maps[int(rng_k.integers(len(maps)))]
            seed = a.seed_base + pair
            for side in (0, 1):
                if pair * 2 + side >= a.games: break
                t0 = time.time()
                kits = akit + bkit if side == 0 else bkit + akit
                lo = dict(mode='set', fixed=[2, 4, 14], pool=[], lineups=[kits], set_file=None, no_dup=False, mirror_share=0.0)
                env.configure_loadouts(lo)
                env.reset(seed, mp)
                env.bot(0, 0); env.bot(1, 0)
                sa, sb = side, 1 - side
                A.attach(env, sa); B.attach(env, sb)
                task = dict(seed=seed, map=mp, side=side, kits=kits)
                rng = np.random.default_rng((seed, mp, side, 7))
                acts = {sa: A.start(task, sa, rng), sb: B.start(task, sb, rng)}
                ida, idb = env.state.players[sa].id, env.state.players[sb].id
                frames = [] if pair * 2 + side < a.visual else None
                d = 0; deaths = []; casts = 0; prev_active = False
                W = walls(env.state); wall_n = len(W)
                while not env.ended:
                    st = env.state
                    if st.header.entity_count and d % 15 == 0:
                        W = walls(st)   # moving/destroyed platforms: refresh 1/s
                    obs = env.rows.copy(); mask = env.mask.copy()
                    pa, pb = st.players[sa], st.players[sb]
                    slimes = [st.slimes[i] for i in range(st.header.slime_count) if st.slimes[i].owner == ida]
                    active = False; dslot = -1
                    for s in slimes[:1]:
                        for k in range(3):
                            if s.abilities[k] == DRILL and s.ability_flags[k] & 8:
                                active = True; dslot = k
                    if active and not prev_active: casts += 1
                    if active:
                        x, y = pa.x / FIX, pa.y / FIX
                        form = [st.entities[i] for i in range(st.header.entity_count)
                                if st.entities[i].owner == ida and st.entities[i].layer == 8 and st.entities[i].kind == 1
                                and st.entities[i].flags & 1 and not st.entities[i].flags & 4]
                        ux = uy = None
                        if form:
                            ux, uy = form[0].up_x / FIX, form[0].up_y / FIX
                            n = math.hypot(ux, uy) or 1; ux, uy = ux / n, uy / n
                        g = sdf(W, x, y)
                        hd = (-ux, -uy) if ux is not None else None
                        row = env.obs_row(sa)
                        rec = dict(g=pair * 2 + side, d=d, t=st.header.episode_ticks, slot=dslot, x=round(x, 2), y=round(y, 2),
                                   hx=None if hd is None else round(hd[0], 3), hy=None if hd is None else round(hd[1], 3),
                                   sdf=round(g, 3), ug=g < 0,
                                   ahead=(round(thickness(W, x, y, hd[0], hd[1]), 2) if (hd and g < 0) else None),
                                   ox=round(pb.x / FIX, 2), oy=round(pb.y / FIX, 2), oalive=bool(pb.flags & 1),
                                   obs_sdf=round(float(row[TO + 9]), 3), obs_grad=[round(float(row[TO + 10]), 3), round(float(row[TO + 11]), 3)],
                                   rays0=int((row[RO:RO + 36 * 8].reshape(36, 8)[:, 0] < 1e-6).sum()),
                                   opp_rays=int(row[RO:RO + 36 * 8].reshape(36, 8)[:, 5].sum()),
                                   body=[round(float(v), 3) for v in row[L.V4_SELF_OFFSET + 20 + 1:L.V4_SELF_OFFSET + 20 + 6]],
                                   opp_rel=[round(float(v), 3) for v in row[L.V4_OPPONENT_OFFSET:L.V4_OPPONENT_OFFSET + 4]])
                        fd.write(json.dumps(rec) + '\n')
                        if a.rows_every and (d % a.rows_every == 0):
                            rows.append(row[:L.V4_FLAT].astype(np.float32).copy())
                            rowmeta.append((pair * 2 + side, d, g, x, y, pb.x / FIX, pb.y / FIX,
                                            hd[0] if hd else 0.0, hd[1] if hd else 0.0, rec['ahead'] if rec['ahead'] is not None else -2.0))
                    prev_active = active
                    if frames is not None: frames.append(env.frame())
                    acts_np = np.stack([acts[0](obs, mask), acts[1](obs, mask)])
                    env.step(acts_np)
                    for e in env.events():
                        if e.kind == 1 and e.subject in (ida, idb):
                            who = e.actor or e.creator
                            deaths.append(dict(d=d, t=e.tick, victim='A' if e.subject == ida else 'B',
                                               killer=('none' if not who or who == 0xFFFFFFFF else 'self' if who == e.subject else
                                                       'A' if who == ida else 'B' if who == idb else 'other'),
                                               ability=int(e.source_ability), cause=DEATH_CAUSE[e.cause] if e.cause < len(DEATH_CAUSE) else int(e.cause),
                                               a_in_drill=active, last=bool(e.flags & 2)))
                    d += 1
                st = env.state
                timeout = bool(st.header.flags & 2)
                aw = bool(st.players[sa].flags & 2) and not timeout; bw = bool(st.players[sb].flags & 2) and not timeout
                outcome = 'draw' if timeout or aw == bw else ('a' if aw else 'b')
                tel = env.frame()['telemetry']
                gr = dict(g=pair * 2 + side, pair=pair, seed=seed, map=mp, side=side, kits=kits, b_kit=bkit, outcome=outcome,
                          ticks=st.header.episode_ticks, decisions=d, drill_casts=casts, deaths=deaths, a_tel=tel[sa], b_tel=tel[sb],
                          sd=st.header.episode_ticks >= 6960, secs=round(time.time() - t0, 2))
                fg.write(json.dumps(gr) + '\n'); fg.flush(); fd.flush()
                if frames is not None:
                    with gzip.open(a.out / f'frames-g{pair * 2 + side}.json.gz', 'wt', compresslevel=4) as f:
                        json.dump(dict(game=gr, a_seat=sa, frames=frames), f, separators=(',', ':'))
    if rows:
        np.savez_compressed(a.out / 'rows.npz', flat=np.stack(rows), meta=np.array(rowmeta, dtype=np.float64))
    print('done', a.out)


main()
