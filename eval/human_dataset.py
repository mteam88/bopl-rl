"""Human behaviour-cloning data: re-step human replays in the bit-exact q32 sim and record,
at every decision of each human seat, the policy observation in f5's layout and the human's
inputs over that decision converted into the policy's action space (docs/HUMAN_BC.md).

    BOPL_RL_BACKEND=rust venv/bin/python eval/human_dataset.py --out DIR --workers 4 \\
        [--players 2] [--aim mode] [--buttons or] [--limit N]

Observation: the in-game agent shim (train/bopl_agent.c, built with bopl_agent_force into
native/artifacts/rl/human-bc/libbopl_agent.so) is handed every tick's state of the human's
seat exactly as RlAgent.cs hands it the game's, from frame 240 (the training warmup) on with
episode tick = frame - 240. Its decision loop (bopl_puffer.h's, 9-tick observation delay,
4-tick decisions) builds the row the net reads; the human's converted decision is forced
in place of a sample, so the own-action tail, latch state and masks are what the net would
have had playing those actions. The checkpoint (f5) still runs on every row: its logits give
f5's per-head log-likelihood of the human's action and its value, teacher-forced.

Conversion of the 4 recorded ticks [f, f+4) of a decision into (aim, buttons, hold):
  aim      --aim mode (default): the most frequent angle byte (ties: earliest); first; last.
           Angles are the game's own byte (0 neutral, 1..255), so no spatial quantisation,
           only temporal: the net holds one angle for 4 ticks.
  buttons  jump/ability bits (1, 2, 4, 8) of the window: --buttons or (default: down on any
           tick, so 1-3 tick taps survive as a 4-tick press), majority (>= 2 ticks), first.
           start/select/WASD bits are dropped (keyboard seats are skipped).
  hold     0 (one decision) everywhere except a press of the bow: the mask forbids pressing
           the bow on a free decision while it is still down from the last one, so a charged
           shot must be one latch. A bow run of n decisions becomes the latch length in
           {1,2,4,8,16} nearest to n (ties: shorter); the latch freezes the other buttons too.
           Hindsight latches on other buttons would label "commit for k decisions" from the
           human's future, which the human never committed to.

Validation (same job, second pass): forks of the recorded world at every --fork-every-th
decision, stepped H ticks with the seat's converted inputs (everyone else as recorded),
against the recording: the seat's position error at +4 and +32 ticks; the same with the
seat's recorded inputs delayed by one tick (calibration: how far a 1-tick timing change
moves it); and an open-loop world playing the converted inputs for the whole round
(position error over time, same winner).

Output: DIR/g<replay-stem>-s<seat>.npz per human seat
    obs (T, 4812) f16, mask (T, 277) u8, action (T, 3) i16 (as applied, after the shim's
    legalisation), latched (T,) bool, fixed (T,) u8 (bopl_agent_forced_fixed),
    f5_logp (T, 3) f32 (f5's log-prob of the action per head), f5_value (T,) f32,
    outcome () f32 (+1 won, -1 lost, 0 draw/unknown), meta json
and DIR/games.jsonl, one row per seat (conversion and fidelity stats).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE)]
os.environ.setdefault('BOPL_AGENT_LIBRARY', str(ROOT / 'native/artifacts/rl/human-bc/libbopl_agent.so'))

WARMUP = 240
REPEAT = 4
HOLD_LENGTHS = (1, 2, 4, 8, 16)
FIX = 4294967296.
SIM = ROOT / 'native/target-q32/release/libbopl_sim.so'
F5 = ROOT / 'native/artifacts/rl/sweeps/final-5/run/checkpoints/policy-18666684416.bin'
MAC = ROOT / 'native/artifacts/rl/human-mac'
OLD = ROOT / 'native/artifacts/human-replays'
LOGITS = 277
HEADS = ((0, 256), (256, 272), (272, 277))


# ---------------------------------------------------------------- corpus

def sessions_by_mtime(paths, gap=2700):
    """Old corpus: a session is a run of files whose mtimes are < gap seconds apart."""
    order = sorted(paths, key=lambda p: p.stat().st_mtime)
    out, session, last = {}, None, None
    for p in order:
        t = p.stat().st_mtime
        if last is None or t - last > gap:
            session = 'old-' + time.strftime('%Y%m%d-%H%M%S', time.localtime(t))
        out[p.name] = session
        last = t
    return out


def corpus(players):
    """[(replay path, source, session, human seat indices, recorded winners or None)], deduplicated by sha256."""
    jobs, seen = [], set()
    for line in open(MAC / 'games.jsonl'):
        g = json.loads(line)
        rep = MAC / Path(g.get('replay') or '').name
        if g.get('failed') or not g.get('replay') or not rep.exists() or not g.get('players'):
            continue
        sha = hashlib.sha256(rep.read_bytes()).hexdigest()
        if sha in seen:
            continue
        seen.add(sha)
        seats = [i for i, p in enumerate(g['players']) if not p.get('agent') and not p.get('keyboard')]
        winners = [bool(p['won_round']) for p in g['players']]
        agent = str(g.get('agent') or '')
        agent_seats = [i for i, q in enumerate(g['players']) if q.get('agent')]
        local = ROOT / 'native/artifacts/rl' / agent.split('/native/artifacts/rl/')[-1] if '/native/artifacts/rl/' in agent else None
        jobs.append(dict(replay=str(rep), source='mac', session='mac-' + g['session'], seats=seats, winners=winners,
                         players=len(g['players']), agent_seats=agent_seats if len(agent_seats) == 1 else [],
                         agent_checkpoint=str(local) if local and local.exists() and agent.endswith('.bin') else None))
    rows = [json.loads(l) for l in open(OLD / 'index.jsonl')]
    sess = sessions_by_mtime([OLD / r['file'] for r in rows])
    for r in rows:
        rep = OLD / r['file']
        sha = hashlib.sha256(rep.read_bytes()).hexdigest()
        if sha in seen:
            continue
        seen.add(sha)
        if r['kind'] == 'vs_agent':
            seats = [r['agent']['human_side']]
            out = r['agent'].get('outcome')
            winners = None
            if out in ('a', 'b') and r['players'] == 2:
                h = r['agent']['human_side']
                winners = [False, False]; winners[h if out == 'a' else 1 - h] = True
        else:
            seats = list(range(r['players']))
            winners = None
        seats = [s for s in seats if not r['keyboard'][s]]
        jobs.append(dict(replay=str(rep), source='old-' + r['kind'], session=sess[r['file']], seats=seats,
                         winners=winners, players=r['players']))
    return [j for j in jobs if j['players'] in players and j['seats']]


# ---------------------------------------------------------------- conversion

def aim_of(window, rule):
    if rule == 'first':
        return int(window[0])
    if rule == 'last':
        return int(window[-1])
    values, counts = np.unique(window, return_counts=True)
    best = counts.max()
    for v in window:   # earliest of the most frequent
        if counts[values == v][0] == best:
            return int(v)


def buttons_of(window, rule):
    w = window.astype(np.int64) & 15
    if rule == 'first':
        return int(w[0])
    out = 0
    for bit in range(4):
        down = int(((w >> bit) & 1).sum())
        r = rule if rule != 'hybrid' else ('majority' if bit == 0 else 'or')
        if (r == 'or' and down > 0) or (r == 'majority' and 2 * down >= len(w)):
            out |= 1 << bit
    return out


def nearest_hold(n):
    return min(range(len(HOLD_LENGTHS)), key=lambda h: (abs(HOLD_LENGTHS[h] - n), HOLD_LENGTHS[h]))


def plan(angles, buttons, start, end, bow, aim_rule, button_rule):
    """Per decision k (frames start+4k ..): (angle, buttons, hold, latched), and bow-run stats.
    Mirrors the shim's latch (bopl_latch) and bow mask so the forced decisions are legal."""
    frames = list(range(start, end, REPEAT))
    aims = [aim_of(angles[f:min(f + REPEAT, end)], aim_rule) for f in frames]
    combos = [buttons_of(buttons[f:min(f + REPEAT, end)], button_rule) for f in frames]
    out, runs = [], []
    prev = latch = held = 0
    strip_until = -1

    def run(k):
        n = 0
        while k + n < len(combos) and combos[k + n] & bow:
            n += 1
        return n
    for k, a in enumerate(aims):
        if latch > 0:
            latch -= 1
            out.append((a, held, 0, True)); prev = held
            continue
        c = combos[k]
        if bow and c & bow:
            if k < strip_until:
                c &= ~bow
            elif prev & bow:          # still down from a latch that ended: the mask releases it
                strip_until = k + run(k); c &= ~bow
        h = 0
        if bow and c & bow:
            n = run(k)
            h = nearest_hold(n)
            latch, held, strip_until = HOLD_LENGTHS[h] - 1, c, k + n
            runs.append((n, HOLD_LENGTHS[h]))
        out.append((a, c, h, False)); prev = c
    return frames, out, runs


# ---------------------------------------------------------------- sim helpers

def setup_lib():
    from environment import LIB
    LIB.bopl_rl_fork.argtypes = [C.c_void_p, C.c_void_p]; LIB.bopl_rl_fork.restype = C.c_void_p
    LIB.bopl_rl_state.argtypes = [C.c_void_p]
    LIB.bopl_rl_close.argtypes = [C.c_void_p]
    return LIB


def positions(state, n):
    return [(state.players[i].x / FIX, state.players[i].y / FIX, bool(state.players[i].flags & 1)) for i in range(n)]


def resolved_kit(state, seat):
    pid = state.players[seat].id
    for k in range(state.header.slime_count):
        s = state.slimes[k]
        if s.owner == pid and s.flags & 1:
            return [int(a) for a in s.abilities]
    return None


def masked_logp(logits, mask):
    out = np.full(LOGITS, -np.inf, np.float64)
    for lo, hi in HEADS:
        x = logits[lo:hi].astype(np.float64)
        m = mask[lo:hi] != 0
        y = np.where(m, x, -np.inf)
        y = y - y.max()
        out[lo:hi] = y - np.log(np.exp(y).sum())
    return out


def angle_deg(a, b):
    """Angular difference in degrees between two aim bytes (both directional)."""
    d = abs(a - b) * 360.0 / 255.0
    return min(d, 360.0 - d)


# ---------------------------------------------------------------- one replay

def convert(job, a):
    from replay_data import decode, ReplayWorld, action_array
    from agent_shim import Agent
    LIB = setup_lib()
    meta, inputs = decode(job['replay'])
    n = len(meta['players'])
    ticks = meta['ticks']
    stem = Path(job['replay']).stem
    log = Path(a.out) / 'logs' / f'{stem}.log'
    angles = inputs['angle'].astype(np.int64)
    btns = inputs['buttons'].astype(np.int64)
    # ---- pass 1: recorded world, a shim per human seat
    world = ReplayWorld(meta, log, game=str(SIM), check_identity=False)
    agents = {}
    for seat in job['seats']:
        ag = Agent(str(job.get('checkpoint') or a.checkpoint), players=n, player=seat)
        ag.lib.bopl_agent_force.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]
        ag.lib.bopl_agent_forced_fixed.argtypes = [C.c_void_p]; ag.lib.bopl_agent_forced_fixed.restype = C.c_int
        ag.lib.bopl_agent_set_map.argtypes = [C.c_void_p, C.c_uint32]
        ag.lib.bopl_agent_set_map(ag.handle, meta['level'])
        ag.reset(meta['seed'])
        agents[seat] = ag
    pos = []
    end = ticks
    rec = {s: dict(obs=[], mask=[], action=[], latched=[], fixed=[], logits=[], frames=[]) for s in agents}
    packets = {s: np.zeros((ticks, 2), np.int64) for s in agents}
    plans, kits, bow_runs, decided = {}, {}, {}, {}
    status_end = {}
    try:
        for f in range(ticks):
            if world.ended:
                end = f
                break
            state = world.state
            if state.header.episode_ticks != f:
                raise RuntimeError(f'episode tick {state.header.episode_ticks} at frame {f}')
            pos.append(positions(state, n))
            if f == WARMUP:
                for seat in agents:
                    kit = resolved_kit(state, seat) or meta['players'][seat]['abilities']
                    kits[seat] = kit
                    bow = sum(1 << (slot + 1) for slot in range(3) if kit[slot] == 4)
                    frames, pl, runs = plan(angles[:, seat], btns[:, seat], WARMUP, ticks, bow, a.aim, a.buttons)
                    plans[seat] = dict(zip(frames, pl)); bow_runs[seat] = runs
            if f >= WARMUP:
                copy = type(state).from_buffer_copy(state)
                copy.header.episode_ticks = f - WARMUP
                for seat, ag in agents.items():
                    if seat in status_end:
                        continue
                    p = plans[seat].get(f)
                    if p is not None:
                        ag.lib.bopl_agent_force(ag.handle, p[0], p[1], p[2])
                    status, packet = ag.tick(copy)
                    if status == 1:
                        status_end[seat] = f; continue
                    packets[seat][f] = packet
                    if p is not None:
                        if ag.decisions != len(rec[seat]['frames']) + 1:
                            raise RuntimeError(f'seat {seat}: no decision at frame {f}')
                        r = rec[seat]
                        r['frames'].append(f)
                        r['obs'].append(ag.observation.astype(np.float16))
                        r['mask'].append(ag.mask.copy())
                        r['action'].append(ag.decision)
                        r['latched'].append(p[3])
                        r['fixed'].append(ag.lib.bopl_agent_forced_fixed(ag.handle))
                        r['logits'].append(ag.logits.copy())
            acts = [(int(angles[f, i]), int(btns[f, i])) for i in range(n)]
            world.step(acts)
        final = world.state
        final_pos = positions(final, n)
        won = [bool(final.players[i].flags & 2) for i in range(n)]
    finally:
        world.close()
        for ag in agents.values():
            ag.close()
    pos.append(final_pos)
    pos = np.array([[p[:2] for p in row] for row in pos]), np.array([[p[2] for p in row] for row in pos])
    faithful = end >= ticks - 4 and (job['winners'] is None or won == job['winners'])
    # ---- pass 2: validation forks and open-loop worlds
    val = validate(meta, inputs, log, packets, end, pos, list(agents), a, LIB) if a.validate else {}
    rows = []
    for seat in agents:
        r = rec[seat]
        T = len(r['frames'])
        if T == 0:
            continue
        obs = np.stack(r['obs']); mask = np.stack(r['mask']).astype(np.uint8)
        action = np.array(r['action'], np.int16); logits = np.stack(r['logits'])
        logp = np.zeros((T, 3), np.float32); agree = np.zeros((T, 3), bool); aim_err = []
        for t in range(T):
            lp = masked_logp(logits[t, :LOGITS], mask[t])
            for h, (lo, hi) in enumerate(HEADS):
                logp[t, h] = lp[lo + action[t, h]]
                agree[t, h] = int(np.argmax(lp[lo:hi])) == action[t, h]
            am = int(np.argmax(lp[0:256]))
            if action[t, 0] and am:
                aim_err.append(angle_deg(am, int(action[t, 0])))
        outcome = 0.0
        if faithful and any(won):
            outcome = 1.0 if won[seat] else -1.0
        info = dict(replay=Path(job['replay']).name, source=job['source'], session=job['session'], seat=seat, players=n,
                    map=meta['level'], kit=kits.get(seat), kit_recorded=meta['players'][seat]['abilities'], ticks=ticks,
                    end=end, faithful=bool(faithful), won=won[seat], outcome=outcome, decisions=T,
                    aim=a.aim, buttons=a.buttons)
        path = Path(a.out) / f'g{Path(job["replay"]).stem}-s{seat}.npz'
        np.savez_compressed(path, obs=obs, mask=mask, action=action, latched=np.array(r['latched'], bool),
                            fixed=np.array(r['fixed'], np.uint8), f5_logp=logp, f5_value=logits[:, LOGITS].astype(np.float32),
                            outcome=np.float32(outcome), meta=json.dumps(info))
        lat = np.array(r['latched'], bool)
        free = ~lat
        # within-window loss of the conversion
        fr = np.array(r['frames'])
        win_a = np.stack([angles[np.minimum(fr + j, ticks - 1), seat] for j in range(4)], 1)
        win_b = np.stack([btns[np.minimum(fr + j, ticks - 1), seat] & 15 for j in range(4)], 1)
        applied_b = action[:, 1].astype(np.int64)
        tick_bits_mismatch = int(sum(((win_b[:, j] ^ applied_b) & 15 != 0).sum() for j in range(4)))
        tick_aim_mismatch = int(sum((win_a[:, j] != action[:, 0]).sum() for j in range(4)))
        runs = bow_runs.get(seat, [])
        info.update(path=path.name, latched=int(lat.sum()), fixed=int((np.array(r['fixed']) != 0).sum()),
                    tick_aim_mismatch=tick_aim_mismatch / (4 * T), tick_button_mismatch=tick_bits_mismatch / (4 * T),
                    bow_runs=len(runs), bow_abs_err=float(np.mean([abs(L - m) for m, L in runs])) if runs else None,
                    f5_nll=(-logp.sum(1)).mean().item(), f5_nll_aim=float(-logp[:, 0].mean()),
                    f5_nll_buttons=float(-logp[free, 1].mean()) if free.any() else None,
                    f5_nll_hold=float(-logp[free, 2].mean()) if free.any() else None,
                    f5_agree_aim=float(agree[:, 0].mean()), f5_agree_buttons=float(agree[free, 1].mean()) if free.any() else None,
                    f5_agree_hold=float(agree[free, 2].mean()) if free.any() else None,
                    f5_aim_err_deg=float(np.median(aim_err)) if aim_err else None,
                    neutral_share=float((action[:, 0] == 0).mean()), press_share=float((action[:, 1] != 0).mean()),
                    **val.get(seat, {}))
        rows.append(info)
    return rows


def validate(meta, inputs, log, packets, end, pos, seats, a, LIB):
    """Forks of the recorded world: converted vs recorded inputs for one seat over H ticks."""
    from replay_data import ReplayWorld, action_array
    n = len(meta['players'])
    xy, alive = pos
    angles = inputs['angle'].astype(np.int64); btns = inputs['buttons'].astype(np.int64)
    out = {}
    H = (4, 32)
    errs = {s: {('conv', h): [] for h in H} | {('lag1', h): [] for h in H} for s in seats}
    world = ReplayWorld(meta, log, game=str(SIM), check_identity=False)
    fork = None
    try:
        for f in range(end):
            if f >= WARMUP and (f - WARMUP) % (REPEAT * a.fork_every) == 0 and f + max(H) < end:
                for s in seats:
                    for kind in ('conv', 'lag1'):
                        fork = LIB.bopl_rl_fork(world.handle, fork)
                        if not fork:
                            raise RuntimeError('fork failed')
                        for j in range(max(H)):
                            g = f + j
                            acts = [(int(angles[g, i]), int(btns[g, i])) for i in range(n)]
                            if kind == 'conv':
                                acts[s] = (int(packets[s][g, 0]), int(packets[s][g, 1]))
                            else:
                                acts[s] = (int(angles[g - 1, s]), int(btns[g - 1, s]))
                            if LIB.bopl_rl_state(fork).contents.header.flags & 1:
                                break   # the fork's round ended (a death the recording did not have)
                            if LIB.bopl_rl_step(fork, action_array(acts), n, 1):
                                raise RuntimeError('fork step: ' + LIB.bopl_rl_error(fork).decode())
                            if j + 1 in H:
                                st = LIB.bopl_rl_state(fork).contents
                                p = st.players[s]
                                if p.flags & 1 and alive[g + 1, s]:
                                    errs[s][(kind, j + 1)].append(math.hypot(p.x / FIX - xy[g + 1, s, 0], p.y / FIX - xy[g + 1, s, 1]))
            world.step([(int(angles[f, i]), int(btns[f, i])) for i in range(n)])
    finally:
        if fork:
            LIB.bopl_rl_close(fork)
        world.close()
    # open loop: the whole round with the seat on converted inputs
    for s, kind in [(s, k) for s in seats for k in ('conv', 'lag1')]:
        w = ReplayWorld(meta, log, game=str(SIM), check_identity=False)
        diverged = None; err_at = {}
        try:
            for f in range(end):
                if w.ended:
                    break
                acts = [(int(angles[f, i]), int(btns[f, i])) for i in range(n)]
                if f >= WARMUP:
                    acts[s] = (int(packets[s][f, 0]), int(packets[s][f, 1])) if kind == 'conv' else (int(angles[f - 1, s]), int(btns[f - 1, s]))
                w.step(acts)
                p = w.state.players[s]
                if f + 1 < len(xy) and p.flags & 1 and alive[f + 1, s]:
                    e = math.hypot(p.x / FIX - xy[f + 1, s, 0], p.y / FIX - xy[f + 1, s, 1])
                    if diverged is None and e > 1.0:
                        diverged = f + 1 - WARMUP
                    for sec in (1, 2, 5, 10):
                        if f + 1 - WARMUP == 60 * sec:
                            err_at[sec] = e
            open_end = w.state.header.episode_ticks
            open_won = [bool(w.state.players[i].flags & 2) for i in range(n)]
        finally:
            w.close()
        e = errs[s]
        if kind == 'conv':
            out[s] = {f'err_{k}_{h}': (float(np.median(v)) if v else None) for (k, h), v in e.items()}
            out[s].update({f'err_{k}_{h}_p90': (float(np.percentile(v, 90)) if v else None) for (k, h), v in e.items()})
            out[s].update(open_diverge_ticks=diverged, open_err_s={str(k): v for k, v in err_at.items()},
                          open_end=int(open_end), open_won=open_won[s], open_any_won=any(open_won), forks=len(e[('conv', 4)]))
        else:
            out[s].update(lag1_diverge_ticks=diverged, lag1_end=int(open_end), lag1_won=open_won[s])
    return out


def run(job_a):
    job, a = job_a
    try:
        return convert(job, a)
    except Exception as error:
        return f'{Path(job["replay"]).name}: {type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--players', default='2', help='comma list of player counts to include')
    p.add_argument('--aim', choices=('mode', 'first', 'last'), default='mode')
    p.add_argument('--buttons', choices=('or', 'majority', 'first', 'hybrid'), default='or')
    p.add_argument('--fork-every', type=int, default=8, help='validation fork every N decisions')
    p.add_argument('--no-validate', dest='validate', action='store_false')
    p.add_argument('--checkpoint', type=Path, default=F5, help='the net whose layout the rows use and whose logits are logged (f5)')
    p.add_argument('--self-check', help='convert the bot seats of games against checkpoints whose path contains this, with that checkpoint')
    p.add_argument('--limit', type=int, help='first N replays only (quick checks)')
    p.add_argument('--only', help='comma list of replay file names')
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = corpus({int(x) for x in a.players.split(',')})
    if a.only:
        names = set(a.only.split(','))
        jobs = [j for j in jobs if Path(j['replay']).name in names]
    if a.self_check:
        # The pipeline on the bot seats with the bot's own checkpoint: a policy's own inputs
        # convert without loss and it should assign them its usual likelihood.
        jobs = [dict(j, seats=j['agent_seats'], checkpoint=j['agent_checkpoint']) for j in jobs
                if j.get('agent_checkpoint') and j.get('agent_seats') and j['agent_checkpoint'].find(a.self_check) >= 0]
    if a.limit:
        jobs = jobs[:a.limit]
    done = {json.loads(l)['replay'] for l in open(a.out / 'games.jsonl')} if (a.out / 'games.jsonl').exists() else set()
    jobs = [j for j in jobs if Path(j['replay']).name not in done]
    print(f'{len(jobs)} replays to convert ({len(done)} already done)', flush=True)
    failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool, open(a.out / 'games.jsonl', 'a') as out, open(a.out / 'failed.txt', 'a') as bad:
        for k, result in enumerate(pool.map(run, [(j, a) for j in jobs])):
            if isinstance(result, str):
                failed += 1; bad.write(result + '\n'); bad.flush(); continue
            for row in result:
                out.write(json.dumps(row) + '\n')
            out.flush()
            if k % 20 == 0:
                print(f'{k + 1}/{len(jobs)} replays, {failed} failed, {time.time() - t0:.0f}s', flush=True)
    print(f'done: {len(jobs) - failed} converted, {failed} failed', flush=True)


if __name__ == '__main__':
    main()
