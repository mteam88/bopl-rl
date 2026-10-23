"""Probe: what the wire state and the v4 row show while a seat is in Drill form."""
import sys, json
from pathlib import Path
import numpy as np
T = Path(__file__).resolve().parents[4]   # repo root
sys.path[:0] = [str(T / 'eval'), str(T / 'league'), str(T / 'native/tools')]
import registry, elo_ladder, games as league_games
from environment import Environment
import layout as L

FIX = 4294967296.
OUT = Path(sys.argv[1])
league_games.configure(OUT / 'work')
m = {x['name']: x for x in json.loads((T / 'league/league.json').read_text())['members']}
A = elo_ladder.Model('me31', T / m['l2-me31-drill-grenade-2000m']['path'], 'candidate')
B = elo_ladder.Model('main', T / m['l2-main-136002m']['path'], 'candidate')
kits = [26, 2, 3, 34, 20, 9]
lo = dict(mode='set', fixed=[2, 4, 14], pool=[], lineups=[kits], set_file=None, no_dup=False, mirror_share=0.0)
SO, TO, RO, EO = L.V4_SELF_OFFSET, L.V4_TERRAIN_OFFSET, L.V4_RAY_OFFSET, L.V4_ENTITY_OFFSET
shown = 0
with Environment(seed=5, limit=registry.EPISODE_TICKS, delay=registry.DELAY, loadouts=lo) as env:
    for game in range(6):
        env.configure_loadouts(lo)
        env.reset(1000 + game, [5, 20, 40, 60, 80, 100][game])
        env.bot(0, 0); env.bot(1, 0)
        A.attach(env, 0); B.attach(env, 1)
        rng = np.random.default_rng(game)
        task = dict(seed=1000 + game, map=5, side=0, kits=kits)
        acts = {0: A.start(task, 0, rng), 1: B.start(task, 1, rng)}
        n = 0; prev = None
        while not env.ended:
            obs = env.rows.copy(); mask = env.mask.copy()
            st = env.state
            pid = st.players[0].id
            px, py = st.players[0].x / FIX, st.players[0].y / FIX
            s0 = [st.slimes[i] for i in range(st.header.slime_count) if st.slimes[i].owner == pid]
            if s0 and s0[0].ability_flags[0] & 8 and shown < 60:
                s = s0[0]
                row = env.obs_row(0)
                own = [st.entities[i] for i in range(st.header.entity_count) if st.entities[i].owner == pid and st.entities[i].flags & 1 and not st.entities[i].flags & 4]
                ents = [(e.kind, e.layer, round(e.x / FIX, 1), round(e.y / FIX, 1), round(e.vx / FIX, 1), round(e.vy / FIX, 1), round(e.rotation / FIX, 2), round(e.up_x / FIX, 2), round(e.up_y / FIX, 2)) for e in own]
                body = row[SO + 20:SO + 44]
                ter = row[TO:TO + 12]
                rays = row[RO:RO + 36 * 8].reshape(36, 8)
                er = row[EO:EO + 48 * 48].reshape(48, 48)
                own_rows = [(round(float(r[1]), 2), round(float(r[2]), 2), int(r[47])) for r in er if r[0] != 0][:6]
                print(f'g{game} d{n} player=({px:.1f},{py:.1f}) dpos/dec={(None if prev is None else (round(px-prev[0],2), round(py-prev[1],2)))}')
                print('   own entities', ents)
                print('   body v/|v|/sin/cos/inab', np.round(body[[1, 2, 3, 4, 5, 17]], 3).tolist(), 'state1hot', int(np.argmax(body[6:17])))
                print('   terrain', np.round(ter, 3).tolist())
                print('   rays dist(min,mean)', round(float(rays[:, 0].min()), 3), round(float(rays[:, 0].mean()), 3), 'n zero', int((rays[:, 0] < 1e-6).sum()), 'opp-hit rays', int(rays[:, 5].sum()))
                print('   prev', np.round(row[L.V4_PREVIOUS_OFFSET:L.V4_PREVIOUS_OFFSET + 26], 2).tolist())
                shown += 1
            prev = (px, py)
            a = np.stack([acts[0](obs, mask), acts[1](obs, mask)])
            env.step(a)
            n += 1
