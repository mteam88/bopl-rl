"""Decode check of writer revision bit 32 (layout v4.3 drill) against ground truth.

Three research environments run in lockstep (same seed, map, kits and actions; delay 0 so the row
describes the current state):
  A: BOPL_RL_FORM_BODY=1, both seats at the checkpoint's bits (0 or 16)  -- the actors read A
  B: BOPL_RL_FORM_BODY=1, seat 0 at 48 (v4.3), seat 1 at the checkpoint's -- decoded
  C: BOPL_RL_FORM_BODY unset, both seats at the checkpoint's bits         -- identity control
Checks:
  identity: A's rows == C's rows bit for bit on every decision (the export is invisible to v4.2),
            and A's state == B's state (the bit changes rows only).
  flag:     the form row's UNDERGROUND flag vs the terrain SDF < 0 at the player (drill_look walls) and at
            the form row; confusion keys are flag, truth.
  body:     B's self body vx, vy == form row velocity / 100, sin/cos == the form row's rotation;
            the form velocity vs the finite difference of the form row position over the decision.
  rays:     on underground decisions, each around ray's distance in B vs the exit distance along its
            direction from the SDF (0.01 u steps), where no nearer circle is in the way.
  opponent: seat 1 in B reads the checkpoint's bits; seat 0's opponent block in B when seat 1 drills (both seats drill).

    decode_v43.py --ckpt NAME|PATH --games N [--seed-base S] [--out FILE]
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np

T = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(T / 'eval'), str(T / 'league'), str(T / 'native/tools'), str(Path(__file__).resolve().parent)]
import registry, elo_ladder, map_catalogue, games  # noqa
from environment import Environment  # noqa
import layout as L  # noqa

FIX = 4294967296.
UG, FORM = 1 << 24, 1 << 25
LEAGUE = {x['name']: x for x in json.loads((T / 'league/league.json').read_text())['members']}
BODY = L.V4_SELF_OFFSET + L.PLAYER_FEATURES
OBODY = L.V4_OPPONENT_OFFSET + L.PLAYER_FEATURES
RO = L.V4_RAY_OFFSET


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
        out.append((e.x / FIX, e.y / FIX, rx, ry, ux, uy, ex, ey, e.radius / FIX))
    return np.array(out) if out else np.zeros((0, 9))


def sdf(W, x, y):
    x, y = np.atleast_1d(x)[:, None], np.atleast_1d(y)[:, None]
    if not len(W): return np.full(x.shape[0], 1e9)
    px, py = x - W[None, :, 0], y - W[None, :, 1]
    lx = np.abs(px * W[:, 2] + py * W[:, 3]) - W[:, 6]; ly = np.abs(px * W[:, 4] + py * W[:, 5]) - W[:, 7]
    out = np.hypot(np.maximum(lx, 0), np.maximum(ly, 0)) + np.minimum(np.maximum(lx, ly), 0) - W[:, 8]
    return out.min(1)


def exit_distance(W, x, y, dx, dy, limit=40.0, step=0.01):
    t = np.arange(0, limit, step)
    d = sdf(W, x + dx * t, y + dy * t)
    k = np.nonzero(d >= 0)[0]
    return float(t[k[0]]) if len(k) else limit


def form_row(st, pid):
    for i in range(st.header.entity_count):
        e = st.entities[i]
        if e.owner == pid and e.flags & FORM and e.flags & 1 and not e.flags & 4:
            return e
    return None


def snapshot(st):
    """Bytes of everything the state exposes (header, players, entities, slimes)."""
    h = st.header
    ents = b''.join(bytes(st.entities[i]) for i in range(h.entity_count))
    sl = b''.join(bytes(st.slimes[i]) for i in range(h.slime_count))
    return bytes(h) + bytes(st.players) + ents + sl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='l2-me31-drill-grenade-2000m')
    p.add_argument('--kit', default='26,2,2'); p.add_argument('--games', type=int, default=12)
    p.add_argument('--seed-base', type=int, default=7300001); p.add_argument('--maps', default='training-111')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    games.configure(Path('/tmp/drill-decode-work'))
    path = T / LEAGUE[a.ckpt]['path'] if a.ckpt in LEAGUE else Path(a.ckpt)
    M = elo_ladder.Model('M', path, 'candidate')
    assert M.layout == 4 and not M.ablate & 32, (M.layout, M.ablate)
    kit = [int(v) for v in a.kit.split(',')]
    maps = map_catalogue.parse_map_ids(a.maps)
    stats = dict(decisions=0, row_mismatch_AC=0, state_mismatch_AB=0, form_rows=0, flag_agree=0, flag_n=0,
                 flag_ug_truth=0, flag_confusion={'00': 0, '01': 0, '10': 0, '11': 0},
                 flag_confusion_formpos={'00': 0, '01': 0, '10': 0, '11': 0}, flag_disagree_sdf=[], ug_rows=0, body_exact=0, body_n=0, rot_nonzero=0, rot_nonzero_v42=0,
                 fd_err=[], ray_err=[], ray_zero_v42=0, ray_zero_v43=0, ray_n=0, ray_circle_before_exit=0, opp_body_exact=0, opp_body_n=0,
                 bitoff_seat1_equal=0)
    rng0 = np.random.default_rng(a.seed_base)
    envs = []
    for name, on in (('A', True), ('B', True), ('C', False)):
        envs.append(Environment(seed=a.seed_base, limit=registry.EPISODE_TICKS, delay=0))
    A, B, C = envs
    for g in range(a.games):
        seed = a.seed_base + g; mp = maps[int(rng0.integers(len(maps)))]
        kits = kit + kit   # both seats drill: the opponent block is exercised too
        lo = dict(mode='set', fixed=[2, 4, 14], pool=[], lineups=[kits], set_file=None, no_dup=False, mirror_share=0.0)
        for env, on in ((A, True), (B, True), (C, False)):
            if on: os.environ['BOPL_RL_FORM_BODY'] = '1'
            else: os.environ.pop('BOPL_RL_FORM_BODY', None)
            env.configure_loadouts(lo); env.reset(seed, mp); env.bot(0, 0); env.bot(1, 0)
        os.environ['BOPL_RL_FORM_BODY'] = '1'
        for side in (0, 1):
            M.attach(A, side); C.set_layout(side, 4, ablate=M.ablate)
        B.set_layout(0, 4, ablate=48); B.set_layout(1, 4, ablate=M.ablate)
        task = dict(seed=seed, map=mp, side=0, kits=kits)
        rng = np.random.default_rng((seed, mp, 7))
        acts = {s: M.start(task, s, rng) for s in (0, 1)}
        W = walls(A.state); d = 0
        while not A.ended:
            st = A.state
            if d % 15 == 0: W = walls(st)
            stats['decisions'] += 1
            ra, rb, rc = A.rows, B.rows, C.rows
            w = L.V4_OBS_SIZE
            if not np.array_equal(ra[:, :w].view(np.uint32), rc[:, :w].view(np.uint32)): stats['row_mismatch_AC'] += 1
            if snapshot(st) != snapshot(B.state): stats['state_mismatch_AB'] += 1
            if np.array_equal(ra[1, :w].view(np.uint32), rb[1, :w].view(np.uint32)): stats['bitoff_seat1_equal'] += 1
            for seat in (0, 1):
                pid = st.players[seat].id
                f = form_row(st, pid)
                if f is None: continue
                stats['form_rows'] += 1
                x, y = st.players[seat].x / FIX, st.players[seat].y / FIX
                truth = bool(sdf(W, x, y)[0] < 0); flag = bool(f.flags & UG)
                ftruth = bool(sdf(W, f.x / FIX, f.y / FIX)[0] < 0)
                stats['flag_n'] += 1; stats['flag_agree'] += truth == flag; stats['flag_ug_truth'] += truth
                stats['flag_confusion'][f'{int(flag)}{int(truth)}'] += 1
                stats['flag_confusion_formpos'][f'{int(flag)}{int(ftruth)}'] += 1
                if flag != truth: stats['flag_disagree_sdf'].append(float(sdf(W, x, y)[0]))
                vx, vy = f.rotation / FIX, f.angular_velocity / FIX
                rot = math.atan2(f.right_y / FIX, f.right_x / FIX)
                body = rb[0, OBODY:OBODY + 6] if seat == 1 else rb[0, BODY:BODY + 6]
                want = np.array([1, np.clip(vx / 100, -1, 1), np.clip(vy / 100, -1, 1), np.clip(math.hypot(vx, vy) / 100, -1, 1),
                                 math.sin(rot), math.cos(rot)], dtype=np.float32)
                ok = np.allclose(body, want, atol=2e-6)
                if seat == 0:
                    stats['body_n'] += 1; stats['body_exact'] += ok
                    stats['rot_nonzero'] += abs(float(body[4])) > 1e-6
                    stats['rot_nonzero_v42'] += abs(float(ra[0, BODY + 4])) > 1e-6
                else:
                    stats['opp_body_n'] += 1; stats['opp_body_exact'] += ok
                if seat == 0 and flag and truth:
                    stats['ug_rows'] += 1
                    rays_b = rb[0, RO:RO + 36 * 8].reshape(36, 8); rays_a = ra[0, RO:RO + 36 * 8].reshape(36, 8)
                    stats['ray_zero_v42'] += int((rays_a[:32, 0] < 1e-6).all())
                    stats['ray_zero_v43'] += int((rays_b[:32, 0] < 1e-6).all())
                    for k in range(32):
                        ang = 2 * math.pi * k / 32
                        ex = exit_distance(W, x, y, math.cos(ang), math.sin(ang))
                        got = float(rays_b[k, 0]) * 40
                        if got + 0.3 < ex:   # a circle (body, projectile) nearer than the exit: not a wall reading
                            stats['ray_circle_before_exit'] += 1
                            continue
                        stats['ray_n'] += 1; stats['ray_err'].append(got - ex)
                if seat == 0:
                    f0 = (f.x / FIX, f.y / FIX, st.header.episode_ticks)
            prev = {}
            for seat in (0, 1):
                f = form_row(st, st.players[seat].id)
                if f is not None: prev[seat] = (f.x / FIX, f.y / FIX, f.rotation / FIX, f.angular_velocity / FIX, st.header.episode_ticks)
            obs, mask = ra.copy(), A.mask.copy()
            act = np.stack([acts[0](obs, mask), acts[1](obs, mask)])
            for env in envs: env.step(act)
            st2 = A.state
            for seat, (px, py, vx, vy, t0) in prev.items():
                f = form_row(st2, st2.players[seat].id)
                dt = (st2.header.episode_ticks - t0) / 60.0
                if f is not None and dt > 0 and f.flags & UG:
                    fdx, fdy = (f.x / FIX - px) / dt, (f.y / FIX - py) / dt
                    nvx, nvy = f.rotation / FIX, f.angular_velocity / FIX
                    stats['fd_err'].append((fdx - (vx + nvx) / 2, fdy - (vy + nvy) / 2, math.hypot(fdx, fdy)))
            d += 1
        print(f'game {g} map {mp} decisions {d} form {stats["form_rows"]} ug {stats["ug_rows"]} rays {stats["ray_n"]}', flush=True)
    for env in envs: env.close()
    re = np.abs(np.array(stats.pop('ray_err') or [0.0]))
    fd = np.array(stats.pop('fd_err') or [(0, 0, 0)])
    ds = np.abs(np.array(stats.pop('flag_disagree_sdf') or [0.0]))
    stats['flag_disagree_abs_sdf_median'] = float(np.median(ds)); stats['flag_disagree_abs_sdf_p90'] = float(np.percentile(ds, 90))
    out = dict(stats, ray_abs_err_median=float(np.median(re)), ray_abs_err_p95=float(np.percentile(re, 95)),
               ray_within_0p1=float((re <= 0.1).mean()), ray_within_0p5=float((re <= 0.5).mean()),
               fd_n=len(fd), fd_speed_median=float(np.median(fd[:, 2])),
               fd_err_median=float(np.median(np.hypot(fd[:, 0], fd[:, 1]))),
               fd_err_p90=float(np.percentile(np.hypot(fd[:, 0], fd[:, 1]), 90)))
    print(json.dumps(out, indent=1))
    if a.out: a.out.write_text(json.dumps(out, indent=1))


main()
