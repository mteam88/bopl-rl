"""Prove per-player routing through PufferLib's agent permutation and the historical-opponent slots.

Builds a vector of self-play and historical environments, applies a non-identity
permutation that packs every buffer as [primary rows | bank 0 rows | bank 1 rows]
(the layout the native trainer's frozen banks slice), and checks after real game
steps that:
  * each environment's slots point at exactly the permuted physical rows;
  * the two rows of a match describe two different players who see each other;
  * idle opponent rows (the bank not chosen this match) stay all-zero;
  * the previous-action features of every active row echo the action written to
    that physical row, so actions route per row and not per slot order;
  * sides change between matches while each historical environment keeps its
    bank, an opponent row is terminated and blanked when its match ends, and
    the alignment protocol (tags/count_aligned) reports each historical
    environment once per match boundary and drops mid-swap matches from the
    per-bank pool statistics.
Runs on the CUDA build of the binding but steps on the host path, so it needs a
game player and no GPU work beyond module load.
"""
import argparse, ctypes, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'native/tools'))
import reference  # noqa: E402  (player selection: explicit > BOPL_RL_GAME > IL2CPP)
sys.path.insert(0, str(HERE))
import counters  # noqa: E402
sys.path.insert(0, str(ROOT / 'eval'))
import layout  # noqa: E402  (bopl_layout.h mirror: observation offsets and action heads)

OBS = layout.OBS_SIZE
# Row pitch is BOPL_OBS_STRIDE, not OBS_SIZE: row memory is sized for the wider
# layout v4 so both fit (train/bopl_layout_v4.h:5), and the v3 columns come first
# within each row (train/check_set_encoder.py:126). So every offset below stays a
# layout-v3 offset, but rows are ROW floats apart. Reshaping at OBS instead reads
# row 0 correctly and every later row at a sliding offset.
ROW = layout.OBS_STRIDE
PLAYER0 = layout.PLAYER0
PLAYER = layout.PLAYER_FEATURES
NUM_ATNS = layout.NUM_ATNS  # aim, buttons, hold length index
PREVIOUS = layout.PREVIOUS_FEATURES  # (the layout() helper below shadows the module)
STRIDE = 8 + 14  # bopl_vec_layout row: num_agents, players, banks, tag, side, bank, matches, stepped, slots[14]

def native_handle(lib):
    """The native vector's own handle, which is NOT a PufferLib VecEnv attribute.

    `bopl_vec_*` address the native vector that train/binding.c owns; PufferLib's
    VecEnv exposes only obs_ptr/rewards_ptr/terminals_ptr. train/native_vector.py
    stores the handle as `gpu_obs_ptr` on itself and every consumer (counters.py)
    reads that name, so wrap it the same way train/smoke.py does. Reading
    `gpu_obs_ptr` off the VecEnv instead is an AttributeError.
    """
    handles = (ctypes.c_void_p * 8)()
    if lib.bopl_vec_handles(handles, 8) != 1:
        raise RuntimeError('expected exactly one native vector')
    return SimpleNamespace(gpu_obs_ptr=handles[0])


def host_rows(lib, native, total):
    """obs/rewards/terminals as host arrays, via the binding's own mirror.

    This is a gpu vec (`create_vec(..., True)`), so PufferLib's obs_ptr is device
    memory and reading it with `from_address` yields garbage - which shows up as
    every observation-content check failing while every layout and counter check
    passes. `bopl_vec_host` hands back the host mirror instead; train/smoke.py
    reads the same rows the same way.
    """
    out = [ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()]
    n = lib.bopl_vec_host(ctypes.c_void_p(native.gpu_obs_ptr), *[ctypes.byref(h) for h in out])
    if n != total:
        raise RuntimeError(f'native vector exposes {n} host rows, expected {total}')
    obs = np.ctypeslib.as_array((ctypes.c_float*(total*ROW)).from_address(out[0].value)).reshape(total, ROW)
    rewards = np.ctypeslib.as_array((ctypes.c_float*total).from_address(out[1].value))
    terminals = np.ctypeslib.as_array((ctypes.c_float*total).from_address(out[2].value))
    return obs, rewards, terminals


def layout(lib, native, count):
    out = (ctypes.c_int * (count*STRIDE))()
    n = lib.bopl_vec_layout(ctypes.c_void_p(native.gpu_obs_ptr), out, count*STRIDE)
    assert n == count, n
    rows = np.frombuffer(out, dtype=np.int32).reshape(count, STRIDE)
    return [dict(num_agents=int(r[0]), players=int(r[1]), banks=int(r[2]), tag=int(r[3]), side=int(r[4]),
                 bank=int(r[5]), matches=int(r[6]), stepped=int(r[7]), slots=[int(x) for x in r[8:8+r[0]]]) for r in rows]

def build_perm(envs, num_buffers, banks):
    """Slot order is physical env order; per buffer pack primary rows first, then each bank's rows."""
    total = sum(e['num_agents'] for e in envs); apb = total // num_buffers
    perm = [-1]*total; slot = 0; buf = 0; buf_agents = 0; env_in_buf = []
    groups = []
    for e in envs:
        env_in_buf.append(e); buf_agents += e['num_agents']
        if buf_agents >= apb and buf < num_buffers-1:
            groups.append(env_in_buf); env_in_buf = []; buf += 1; buf_agents = 0
    groups.append(env_in_buf)
    slot = 0
    for b, group in enumerate(groups):
        base = b*apb
        primary = []; bank_rows = [[] for _ in range(banks)]
        for e in group:
            e_slots = list(range(slot, slot+e['num_agents'])); slot += e['num_agents']
            if e['banks']:
                primary.append(e_slots[0])
                for k in range(e['banks']): bank_rows[k].append(e_slots[1+k])
            else: primary += e_slots
        order = primary + [s for rows in bank_rows for s in rows]
        assert len(order) == apb, (len(order), apb)
        for phys, s in enumerate(order): perm[s] = base + phys
        groups[b] = dict(primary=len(primary), bank=[len(r) for r in bank_rows])
    assert -1 not in perm
    return perm, groups

def expected_tags(every, historical):
    """(tag, active bank) per environment: `every - 1` self-play games, then historical game j on bank j % 2."""
    return [pair for j in range(historical) for pair in [(0, -1)]*(every-1) + [(1 + j % 2, j % 2)]]


def mutual(obs, a, b):
    """Rows a and b describe two different players who see each other (see main's pair_ok)."""
    ra, rb = obs[a], obs[b]
    if not (ra[PLAYER0] == 1 and ra[PLAYER0+1] == 1 and rb[PLAYER0] == 1 and rb[PLAYER0+1] == 1): return False
    other_a = ra[PLAYER0+PLAYER:PLAYER0+2*PLAYER]; other_b = rb[PLAYER0+PLAYER:PLAYER0+2*PLAYER]
    if not (other_a[0] == 1 and other_a[1] == 0 and other_b[0] == 1 and other_b[1] == 0): return False
    return abs(other_a[9] + other_b[9]) < 1e-5 and abs(other_a[10] + other_b[10]) < 1e-5


def trainer_layout(lib, _C, check, name, every, workers, num_buffers, banks, env_kwargs, steps):
    """The permutation train.py installs (NativeVector.permutation) on a real vector.

    Checks that every buffer's rows are [learner rows | bank 0 | ... | bank n-1 | idle]
    exactly as the trainer's vec kwargs (primary_rows, frozen_bank_pct -> bank_layout)
    slice them: learner rows (both self-play rows, historical slot 0) fill the primary
    slice; each historical environment's opponent row for bank k lies in bank k's slice
    (compact: only the active bank, the others in the never-uploaded idle tail); the
    idle tail stays blank while the games step; and the per-update accounting train.py
    asserts (learner_rows learner and 2*workers physical decisions per step) holds. The
    stepping checks need one buffer (the host step path drives one); multi-buffer and
    legacy (non-compact) layouts get the routing and region checks only.
    """
    from native_vector import NativeVector
    nv = NativeVector(_C, workers, num_buffers, banks, every)
    apb, primary, live = nv.agents_per_buffer, nv.primary_rows, nv.live_per_bank
    vec = _C.create_vec(dict(vec=dict(total_agents=nv.total_agents, num_buffers=num_buffers), env=env_kwargs), True)
    nv.gpu_obs_ptr = native_handle(lib).gpu_obs_ptr
    native = SimpleNamespace(gpu_obs_ptr=nv.gpu_obs_ptr)
    vec.reset()
    before = nv.layout()
    perm = nv.permutation(before)
    assert lib.bopl_vec_set_perm(ctypes.c_void_p(nv.gpu_obs_ptr), (ctypes.c_int*nv.total_agents)(*perm.tolist())) == 0
    tags = np.ascontiguousarray(before[:, 3], dtype=np.int32)
    assert lib.bopl_vec_set_tags(ctypes.c_void_p(nv.gpu_obs_ptr), tags.ctypes.data_as(ctypes.POINTER(ctypes.c_int))) == 0
    nv.reset()
    envs = layout(lib, native, workers)
    expected, slot = [], 0
    for e in envs: expected.append(perm[slot:slot+e['num_agents']].tolist()); slot += e['num_agents']
    check(f'{name}: slots follow the trainer permutation', [e['slots'] for e in envs] == expected)
    learner, active, idle, regions_ok = [], [], [], True
    per_env = workers // num_buffers
    for i, e in enumerate(envs):
        buffer = i // per_env
        off = [s - buffer*apb for s in e['slots']]
        if not all(0 <= o < apb for o in off): regions_ok = False
        own = off if not e['banks'] else off[:1]
        learner += [s for s in (e['slots'] if not e['banks'] else e['slots'][:1])]
        if not all(o < primary for o in own): regions_ok = False
        for k in range(e['banks']):
            o = off[1+k]
            in_bank = primary + k*live <= o < primary + (k+1)*live
            if nv.compact and k != e['bank']:
                idle.append(e['slots'][1+k]); in_bank = o >= nv.live_rows
            elif k == e['bank']:
                active.append((e['slots'][0], e['slots'][1+k]))
            if not in_bank: regions_ok = False
    check(f'{name}: learner rows fill the primary slice, bank k rows bank k\'s slice, idle rows the tail',
          regions_ok and len(learner) == nv.learner_rows == primary*num_buffers
          and sorted(r % apb for r in learner) == sorted(list(range(primary))*num_buffers),
          compact=nv.compact, primary=primary, live=live, apb=apb)
    hist = [e['bank'] for e in envs if e['banks']]
    check(f'{name}: historical games per bank match the pool',
          all(hist.count(b) == nv.historical_per_bank for b in range(banks)) and len(hist) == workers // every,
          hist=hist, historical_per_bank=nv.historical_per_bank)
    if num_buffers > 1:   # the host step path (cpu_vec_step) drives one buffer only
        vec.close(); return
    obs, _, terminals = host_rows(lib, native, nv.total_agents)
    actions = np.zeros((nv.total_agents, NUM_ATNS), dtype=np.float32)
    pairs_ok, idle_blank, last = True, True, envs
    for step in range(steps):
        for r in range(nv.total_agents): actions[r] = (1 + (r*37+step) % 255, (r+step) % 16, 0)
        vec.cpu_step(actions.ctypes.data)
        cur = layout(lib, native, workers)
        for i, e in enumerate(cur):
            rows = e['slots'] if not e['banks'] else [e['slots'][0], e['slots'][1+e['bank']]]
            if e['matches'] == last[i]['matches'] and not mutual(obs, *rows): pairs_ok = False
        if any(obs[r].any() for r in idle): idle_blank = False
        last = cur
    check(f'{name}: learner and active-bank rows stay two mutually visible players', pairs_ok)
    check(f'{name}: idle rows stay blank', idle_blank, idle=len(idle))
    snap = counters.snapshot(lib, native)
    check(f'{name}: counters match the trainer accounting (learner_rows learner, 2*workers physical per step)',
          snap['learner_steps'] == nv.learner_rows*steps and snap['game_steps'] == 2*workers*steps
          and snap['opponent_game_decisions'] == len(hist)*steps, snap={k: snap[k] for k in ('learner_steps', 'game_steps', 'opponent_game_decisions')},
          learner_rows=nv.learner_rows, steps=steps)
    vec.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'native/artifacts/rl/native-routing')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--episode-ticks', type=int, default=160)
    parser.add_argument('--game', type=Path)
    args = parser.parse_args()
    os.environ['BOPL_RL_GAME'] = str(reference.resolve_game(args.game))
    os.environ.setdefault('BOPL_RL_BOOTSTRAP', str(ROOT/'native/fixtures/movement.rep'))
    log_dir = args.output/'logs'; log_dir.mkdir(parents=True, exist_ok=True)
    os.environ['BOPL_RL_LOG_DIR'] = str(log_dir)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from pufferlib import _C
    lib = ctypes.CDLL(_C.__file__)
    lib.bopl_vec_layout.restype = ctypes.c_int; lib.bopl_vec_layout.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.bopl_vec_handles.restype = ctypes.c_int; lib.bopl_vec_handles.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.bopl_vec_host.restype = ctypes.c_int
    lib.bopl_vec_host.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_void_p)]*3
    lib.bopl_vec_reset.argtypes = [ctypes.c_void_p]; lib.bopl_vec_reset.restype = ctypes.c_int
    lib.bopl_vec_set_perm.restype = ctypes.c_int; lib.bopl_vec_set_perm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.bopl_vec_set_tags.restype = ctypes.c_int; lib.bopl_vec_set_tags.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.bopl_vec_count_aligned.restype = ctypes.c_int; lib.bopl_vec_count_aligned.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    counters.bind(lib)
    report = dict(schema=1, game=os.environ['BOPL_RL_GAME'], checks=[])
    def check(name, ok, **info):
        report['checks'].append(dict(name=name, ok=bool(ok), **info)); print(('PASS ' if ok else 'FAIL ') + name, info if not ok else '')
        return ok

    banks = 2
    for every in (2, 1):
        # Four environments per buffer. every=2 (main agents): self-play (2 slots), historical
        # (3 slots), self-play, historical = 10 slots. every=1 (exploiters): four historical
        # environments of 3 slots = 12 slots. Check names carry an 'e1 ' prefix at every=1.
        tag = '' if every == 2 else f'e{every} '
        hist_pb = 4 // every
        per_buffer = 2*(4-hist_pb) + (1+banks)*hist_pb
        env_kwargs = dict(players=2, banks=banks, historical_every=every, seed=73, observation_delay=9, action_repeat=4,
                          max_episode_ticks=args.episode_ticks, warmup_ticks=60, map=0, training_map_mask=0)
        for num_buffers in (1, 2):
            total = per_buffer*num_buffers
            vec = _C.create_vec(dict(vec=dict(total_agents=total, num_buffers=num_buffers), env=env_kwargs), True)
            native = native_handle(lib)
            envs = layout(lib, native, 4*num_buffers)
            perm, groups = build_perm(envs, num_buffers, banks)
            check(f'{tag}b{num_buffers}: layout counts', [e['num_agents'] for e in envs] == ([2]*(every-1) + [1+banks])*(hist_pb*num_buffers)
                  and all(g == dict(primary=2*(4-hist_pb)+hist_pb, bank=[hist_pb]*banks) for g in groups),
                  envs=[e['num_agents'] for e in envs], groups=groups)
            check(f'{tag}b{num_buffers}: permutation is not identity', perm != list(range(total)), perm=perm)
            # Order mirrors the trainer: PufferLib resets with identity routing inside
            # create_pufferl, the permutation and tags are installed on top, and the native
            # reset then restarts every game with that routing in place. Resetting after the
            # permutation instead would restart the games twice and double-count a match.
            vec.reset()
            assert lib.bopl_vec_set_perm(ctypes.c_void_p(native.gpu_obs_ptr), (ctypes.c_int*total)(*perm)) == 0
            tags = (ctypes.c_int*(4*num_buffers))(*[e['tag'] for e in envs])
            assert lib.bopl_vec_set_tags(ctypes.c_void_p(native.gpu_obs_ptr), tags) == 0
            # PufferLib's reset routes with identity and drops the per-match terminal flags, so
            # train/native_vector.py:91 follows it with bopl_vec_reset, which restarts the games
            # under the installed routing and raises a fresh-episode flag on every row. Checking
            # routing after the upstream reset alone inspects a state the trainer never trains from.
            assert lib.bopl_vec_reset(ctypes.c_void_p(native.gpu_obs_ptr)) == 0
            envs = layout(lib, native, 4*num_buffers)
            # Matches started before any stepping. The native reset aborts the match the
            # upstream reset began and starts another, so this is 2, not 1: counting
            # completions as matches-1 would credit the aborted one.
            base_matches = [e['matches'] for e in envs]
            expected = []; slot = 0
            for e in envs: expected.append(perm[slot:slot+e['num_agents']]); slot += e['num_agents']
            check(f'{tag}b{num_buffers}: slots follow the permutation', [e['slots'] for e in envs] == expected,
                  got=[e['slots'] for e in envs], expected=expected)
            check(f'{tag}b{num_buffers}: historical envs are dealt to the banks with tag 1+bank',
                  [(e['tag'], e['bank']) for e in envs] == expected_tags(every, hist_pb*num_buffers), got=[(e['tag'], e['bank']) for e in envs])
            check(f'{tag}b{num_buffers}: reset aligns every historical env of each bank once',
                  all(lib.bopl_vec_count_aligned(ctypes.c_void_p(native.gpu_obs_ptr), 1+b, 1) == hist_pb*num_buffers//banks for b in range(banks))
                  and all(lib.bopl_vec_count_aligned(ctypes.c_void_p(native.gpu_obs_ptr), 1+b, 0) == 0 for b in range(banks)))
            obs, rewards, terminals = host_rows(lib, native, total)
            actions = np.zeros((total, NUM_ATNS), dtype=np.float32)  # hold stays 0 (length 1): no latch overrides the button echo

            def active_rows(e):
                if not e['banks']: return e['slots']
                return [e['slots'][0], e['slots'][1+e['bank']]]
            def idle_rows(e):
                if not e['banks']: return []
                return [e['slots'][1+k] for k in range(e['banks']) if k != e['bank']]
            def pair_ok(a, b):
                """Rows a and b describe two different players who see each other: each row's own-player
                flag is set, the other player is present and not self, and the relative positions are
                antisymmetric (before spawn both are zero, so distinctness is checked separately)."""
                ra, rb = obs[a], obs[b]
                if not (ra[PLAYER0] == 1 and ra[PLAYER0+1] == 1 and rb[PLAYER0] == 1 and rb[PLAYER0+1] == 1): return False
                other_a = ra[PLAYER0+PLAYER:PLAYER0+2*PLAYER]; other_b = rb[PLAYER0+PLAYER:PLAYER0+2*PLAYER]
                if not (other_a[0] == 1 and other_a[1] == 0 and other_b[0] == 1 and other_b[1] == 0): return False
                return abs(other_a[9] + other_b[9]) < 1e-5 and abs(other_a[10] + other_b[10]) < 1e-5
            def separated(a, b):
                return abs(obs[a, PLAYER0+PLAYER+9]) > 1e-6 or abs(obs[a, PLAYER0+PLAYER+10]) > 1e-6
            ok_pairs = all(pair_ok(*active_rows(e)) for e in envs)
            ok_idle = all(not obs[r].any() for e in envs for r in idle_rows(e))
            check(f'{tag}b{num_buffers}: active rows are two mutually visible players', ok_pairs)
            check(f'{tag}b{num_buffers}: idle bank rows are blank', ok_idle)
            # bopl_vec_reset flags every row as a fresh episode, so the first rollout step zeroes
            # the recurrent state of every bank before consuming observation 0 (binding.c:238-241).
            # Opponent-row-only flagging is c_reset's per-match behaviour, covered by the
            # match-end check below; at reset there is no state worth keeping on any row.
            check(f'{tag}b{num_buffers}: reset flags every row as a fresh episode',
                  all(terminals[r] == 1 for e in envs for r in e['slots'][:e['num_agents']]),
                  terminals=[float(t) for t in terminals])
            if num_buffers > 1:
                vec.close(); continue
            # Step on the host path with a distinct action per physical row.
            sides = {i: set() for i, e in enumerate(envs) if e['banks']}; bank_seen = {i: set() for i in sides}
            # Pool statistics mirror the sticky boundary flag: a match counts toward hist_n_bank only
            # if no reset (a swap) cleared the flag while it was in flight. The reset check above
            # cleared it during every env's first match; a second reset hits bank 0 part way through.
            flag = {i: 0 for i in sides}; finished = {i: 0 for i in sides}; counted = {i: 0 for i in sides}
            skipped = 0; flags_ok = True
            seen_apart = [False]*len(envs)
            matches_before = [e['matches'] for e in envs]; last = envs
            # Expected cumulative counters: every step decides for every player; learner rows are the
            # self-play rows and slot 0 of historical environments; buttons are the low four action bits.
            learner_rows = [r for e in envs for r in (e['slots'][:1] if e['banks'] else e['slots'])]
            expect = dict(steps=0, buttons=[0]*4, learner_episodes=0)
            def account():
                expect['steps'] += 1
                for r in learner_rows:
                    for b in range(4): expect['buttons'][b] += (int(actions[r, 1]) >> b) & 1
                expect['learner_episodes'] += sum(int(terminals[r]) for r in learner_rows)
            action_echo_ok = True; blank_after_ok = True; terminal_pairs_ok = True; previous_active = {i: active_rows(e) for i, e in enumerate(envs)}
            for step in range(args.steps):
                for r in range(total): actions[r] = (1 + (r*37+step) % 255, (r+step) % 16, 0)
                vec.cpu_step(actions.ctypes.data); account()
                cur = layout(lib, native, len(envs))
                for i, e in enumerate(cur):
                    rows = active_rows(e)
                    if e['matches'] == last[i]['matches']:
                        if not pair_ok(*rows): action_echo_ok = False
                        seen_apart[i] |= separated(*rows)
                        for k, r in enumerate(rows):
                            # previous-action tail: [angle==0, cos, sin, buttons bits, latch left, 4 button runs]
                            tail = obs[r, OBS-PREVIOUS:OBS]
                            angle, buttons, _hold = actions[r]
                            want_bits = [(int(buttons) >> b) & 1 for b in range(4)]
                            if terminals[r] == 0 and (tail[0] != (angle == 0) or list(tail[3:7]) != want_bits or tail[7] != 0
                                                      or abs(tail[1]-np.cos((angle-1)*2*np.pi/255)) > 1e-5): action_echo_ok = False
                    else:
                        # A match ended this step: both previous active rows were terminated with equal-magnitude
                        # outcome rewards, the previously active opponent row is blank, the new match started.
                        prev = previous_active[i]
                        if not all(terminals[r] == 1 for r in prev): terminal_pairs_ok = False
                        if e['banks']:
                            old_opp = prev[1]
                            new_rows = active_rows(e)
                            if old_opp not in new_rows and obs[old_opp].any(): blank_after_ok = False
                            if terminals[new_rows[1]] != 1: terminal_pairs_ok = False
                            sides[i].add(e['side']); bank_seen[i].add(e['bank'])
                        if not pair_ok(*active_rows(e)): action_echo_ok = False
                    previous_active[i] = active_rows(e)
                for i in sides:
                    if cur[i]['matches'] != last[i]['matches']:
                        finished[i] += 1; counted[i] += flag[i]; skipped += 1-flag[i]; flag[i] = 1
                aligned = lambda tag, reset=0: lib.bopl_vec_count_aligned(ctypes.c_void_p(native.gpu_obs_ptr), tag, reset)
                for b in range(banks):
                    if aligned(1+b) != sum(flag[i] for i in sides if cur[i]['bank'] == b): flags_ok = False
                if step == args.steps//3:
                    # The trainer swaps bank 0 mid-match: the reset clears bank 0's flags only.
                    before_other = aligned(2)
                    aligned(1, 1)
                    for i in sides:
                        if cur[i]['bank'] == 0: flag[i] = 0
                    if aligned(1) != 0 or aligned(2) != before_other: flags_ok = False
                last = cur
            matches = [e['matches']-m for e, m in zip(cur, matches_before)]
            check(f'{tag}b1: actions echo per physical row and pairs stay mutually visible', action_echo_ok)
            check(f'{tag}b1: paired players separate after spawn', all(seen_apart), seen_apart=seen_apart)
            check(f'{tag}b1: match end terminates both active rows and restarts opponent state', terminal_pairs_ok)
            check(f'{tag}b1: finished opponent row is blanked', blank_after_ok)
            check(f'{tag}b1: several matches per env', min(matches) >= 3, matches=matches)
            check(f'{tag}b1: sides vary per historical env', all(len(s) == 2 for s in sides.values()), sides={k: sorted(v) for k, v in sides.items()})
            check(f'{tag}b1: each historical env keeps its bank', all(s == {cur[i]['bank']} for i, s in bank_seen.items()), banks={k: sorted(v) for k, v in bank_seen.items()})
            check(f'{tag}b1: count_aligned counts envs that started a match since their bank was last swapped', flags_ok)
            log = vec.log()
            expected_n = {b: sum(counted[i] for i in sides if cur[i]['bank'] == b) for b in range(banks)}
            check(f'{tag}b1: per-bank pool statistics skip the matches a reset interrupted',
                  skipped == len(sides)+sum(1 for i in sides if cur[i]['bank'] == 0) and sum(expected_n.values()) > 0
                  and all(round(log.get(f'hist_n_bank_{b}', 0)*log['n']) == expected_n[b] for b in range(banks))
                  and all(0 <= log.get(f'hist_score_bank_{b}', 0) <= log.get(f'hist_n_bank_{b}', 0) for b in range(banks)),
                  skipped=skipped, finished=finished, expected=expected_n, log={k: v for k, v in log.items() if 'bank' in k or k == 'n'})
            # Cumulative counters (what the metrics helper differences): physical and learner decisions,
            # held buttons, completed matches (once each, per map), learner-only telemetry, kills.
            snap = counters.snapshot(lib, native)
            completed = sum(e['matches']-b for e, b in zip(cur, base_matches))
            n = expect['steps']
            check(f'{tag}b1: counters count physical and learner decisions', snap['game_steps'] == 2*len(envs)*n and snap['learner_steps'] == len(learner_rows)*n
                  and snap['opponent_game_decisions'] == len(sides)*n, snap=snap, steps=n)
            check(f'{tag}b1: counters count learner held buttons', snap['buttons'] == expect['buttons'], got=snap['buttons'], want=expect['buttons'])
            check(f'{tag}b1: counters count completed matches once per map', snap['matches'] == completed and snap['map_matches'] == {'0': completed},
                  matches=snap['matches'], completed=completed, maps=snap['map_matches'])
            tel = snap['telemetry']
            check(f'{tag}b1: counters keep learner-only telemetry', tel[12] == expect['learner_episodes'] and tel[13]+tel[14] <= tel[12]
                  and tel[15] > 0 and abs(tel[10] - 0.1*snap['kills']) < 1e-6, telemetry=tel, kills=snap['kills'], episodes=expect['learner_episodes'])
            report['log' + tag.strip()] = log; report['counters' + tag.strip()] = snap
            vec.close()
        # The trainer's own routing (train/native_vector.py, compact and legacy layouts) on the
        # host path: the rows its bank_layout slices hold exactly the learner and active-bank rows.
        for workers, num_buffers in ((4, 1), (8, 2), (6 if every == 1 else 12, 2)):
            trainer_layout(lib, _C, check, f'{tag}trainer w{workers} b{num_buffers}', every, workers, num_buffers, banks,
                           env_kwargs, args.steps//2)
    # Final-trainer reward: no kill shaping, timeout -0.5 paid by both players of a truncated match.
    vec = _C.create_vec(dict(vec=dict(total_agents=4, num_buffers=1), env=dict(env_kwargs, banks=0, historical_every=0,
                             reward_kill_milli=0, reward_timeout_milli=-500)), True)
    native = native_handle(lib)
    vec.reset()
    _, rewards, terminals = host_rows(lib, native, 4)
    actions = np.zeros((4, NUM_ATNS), dtype=np.float32); seen = 0; both = True; nonterminal_zero = True
    for step in range(60):
        vec.cpu_step(actions.ctypes.data)
        for pair in ((0, 1), (2, 3)):
            t = [terminals[r] for r in pair]
            if any(t):
                seen += 1
                if not (all(t) and all(abs(rewards[r] + 0.5) < 1e-6 for r in pair)): both = False
            elif any(rewards[r] != 0 for r in pair): nonterminal_zero = False
    snap = counters.snapshot(lib, native)
    check('timeout reward: both players of a truncated match receive -0.5 and nothing otherwise', seen >= 2 and both and nonterminal_zero, seen=seen)
    check('timeout reward: telemetry reports timeouts, outcome -0.5 each, no kill reward',
          snap['telemetry'][14] == snap['telemetry'][12] == 2*snap['matches'] > 0 and abs(snap['telemetry'][9] + 0.5*snap['telemetry'][12]) < 1e-6
          and snap['telemetry'][10] == 0 and snap['telemetry'][11] == 0, telemetry=snap['telemetry'], matches=snap['matches'])
    vec.close()
    report['ok'] = all(c['ok'] for c in report['checks'])
    (args.output/'result.json').write_text(json.dumps(report, indent=2, default=str))
    print('ROUTING', 'PASS' if report['ok'] else 'FAIL', args.output/'result.json')
    sys.exit(0 if report['ok'] else 1)

if __name__ == '__main__':
    main()
