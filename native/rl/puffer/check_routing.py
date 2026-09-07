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
  * sides and banks change between matches, an opponent row is terminated and
    blanked when its match ends, and the alignment protocol (tags/count_aligned)
    reports each historical environment once per match boundary.
Runs on the CUDA build of the binding but steps on the host path, so it needs a
game player and no GPU work beyond module load.
"""
import argparse, ctypes, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'native/tools'))
import reference  # noqa: E402  (player selection: explicit > BOPL_RL_GAME > IL2CPP)

OBS = 10 + 6*148 + 96*21 + 12*28 + 7
PLAYER0 = 10
STRIDE = 8 + 14  # bopl_vec_layout row: num_agents, players, banks, tag, side, bank, matches, stepped, slots[14]

def layout(lib, vec, count):
    out = (ctypes.c_int * (count*STRIDE))()
    n = lib.bopl_vec_layout(ctypes.c_void_p(vec.gpu_obs_ptr), out, count*STRIDE)
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'native/artifacts/rl/autoresearch/native-routing')
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
    lib.bopl_vec_set_perm.restype = ctypes.c_int; lib.bopl_vec_set_perm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.bopl_vec_set_tags.restype = ctypes.c_int; lib.bopl_vec_set_tags.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.bopl_vec_count_aligned.restype = ctypes.c_int; lib.bopl_vec_count_aligned.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.bopl_set_valid_banks.argtypes = [ctypes.c_uint]
    lib.bopl_vec_step_mode.restype = ctypes.c_int; lib.bopl_vec_step_mode.argtypes = [ctypes.c_void_p]
    report = dict(schema=1, game=os.environ['BOPL_RL_GAME'], checks=[])
    def check(name, ok, **info):
        report['checks'].append(dict(name=name, ok=bool(ok), **info)); print(('PASS ' if ok else 'FAIL ') + name, info if not ok else '')
        return ok

    banks = 2
    # Two buffers, each: self-play (2 slots), historical (3 slots), self-play, historical = 10 slots.
    env_kwargs = dict(players=2, banks=banks, historical_every=2, seed=73, observation_delay=9, action_repeat=4,
                      max_episode_ticks=args.episode_ticks, warmup_ticks=60, map=0, training_map_mask=0)
    for num_buffers in (1, 2):
        total = 10*num_buffers
        vec = _C.create_vec(dict(vec=dict(total_agents=total, num_buffers=num_buffers), env=env_kwargs), True)
        envs = layout(lib, vec, 4*num_buffers)
        perm, groups = build_perm(envs, num_buffers, banks)
        check(f'b{num_buffers}: layout counts', [e['num_agents'] for e in envs] == [2, 3]*(2*num_buffers)
              and all(g == dict(primary=6, bank=[2, 2]) for g in groups), envs=[e['num_agents'] for e in envs], groups=groups)
        check(f'b{num_buffers}: permutation is not identity', perm != list(range(total)), perm=perm)
        assert lib.bopl_vec_set_perm(ctypes.c_void_p(vec.gpu_obs_ptr), (ctypes.c_int*total)(*perm)) == 0
        tags = (ctypes.c_int*(4*num_buffers))(*[e['tag'] for e in envs])
        assert lib.bopl_vec_set_tags(ctypes.c_void_p(vec.gpu_obs_ptr), tags) == 0
        lib.bopl_set_valid_banks(0b11)
        vec.reset()
        envs = layout(lib, vec, 4*num_buffers)
        expected = []; slot = 0
        for e in envs: expected.append(perm[slot:slot+e['num_agents']]); slot += e['num_agents']
        check(f'b{num_buffers}: slots follow the permutation', [e['slots'] for e in envs] == expected,
              got=[e['slots'] for e in envs], expected=expected)
        check(f'b{num_buffers}: tags', [e['tag'] for e in envs] == [0, 1]*(2*num_buffers))
        check(f'b{num_buffers}: reset aligns every historical env once',
              lib.bopl_vec_count_aligned(ctypes.c_void_p(vec.gpu_obs_ptr), 1, 1) == 2*num_buffers
              and lib.bopl_vec_count_aligned(ctypes.c_void_p(vec.gpu_obs_ptr), 1, 0) == 0)
        obs = np.ctypeslib.as_array((ctypes.c_float*(total*OBS)).from_address(vec.obs_ptr)).reshape(total, OBS)
        rewards = np.ctypeslib.as_array((ctypes.c_float*total).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float*total).from_address(vec.terminals_ptr))
        actions = np.zeros((total, 2), dtype=np.float32)

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
            other_a = ra[PLAYER0+148:PLAYER0+2*148]; other_b = rb[PLAYER0+148:PLAYER0+2*148]
            if not (other_a[0] == 1 and other_a[1] == 0 and other_b[0] == 1 and other_b[1] == 0): return False
            return abs(other_a[9] + other_b[9]) < 1e-5 and abs(other_a[10] + other_b[10]) < 1e-5
        def separated(a, b):
            return abs(obs[a, PLAYER0+148+9]) > 1e-6 or abs(obs[a, PLAYER0+148+10]) > 1e-6
        ok_pairs = all(pair_ok(*active_rows(e)) for e in envs)
        ok_idle = all(not obs[r].any() for e in envs for r in idle_rows(e))
        check(f'b{num_buffers}: active rows are two mutually visible players', ok_pairs)
        check(f'b{num_buffers}: idle bank rows are blank', ok_idle)
        # Opponent rows start each match with a terminal flag (fresh recurrent state); learner rows do not.
        check(f'b{num_buffers}: first match terminates opponent rows only',
              all(terminals[e['slots'][1+e['bank']]] == 1 for e in envs if e['banks'])
              and all(terminals[r] == 0 for e in envs for r in ([e['slots'][0]] if e['banks'] else e['slots'])))
        if num_buffers > 1:
            vec.close(); continue
        # Step on the host path with a distinct action per physical row.
        sides = {i: set() for i, e in enumerate(envs) if e['banks']}; bank_seen = {i: set() for i in sides}
        seen_apart = [False]*len(envs)
        aligned_total = 0; matches_before = [e['matches'] for e in envs]; last = envs
        action_echo_ok = True; blank_after_ok = True; terminal_pairs_ok = True; previous_active = {i: active_rows(e) for i, e in enumerate(envs)}
        for step in range(args.steps):
            for r in range(total): actions[r] = (1 + (r*37+step) % 255, (r+step) % 16)
            vec.cpu_step(actions.ctypes.data)
            cur = layout(lib, vec, 4)
            for i, e in enumerate(cur):
                rows = active_rows(e)
                if e['matches'] == last[i]['matches']:
                    if not pair_ok(*rows): action_echo_ok = False
                    seen_apart[i] |= separated(*rows)
                    for k, r in enumerate(rows):
                        # previous-action tail: [angle==0, cos, sin, buttons bits]
                        tail = obs[r, OBS-7:]
                        angle, buttons = actions[r]
                        want_bits = [(int(buttons) >> b) & 1 for b in range(4)]
                        if terminals[r] == 0 and (tail[0] != (angle == 0) or list(tail[3:7]) != want_bits
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
            aligned_total += lib.bopl_vec_count_aligned(ctypes.c_void_p(vec.gpu_obs_ptr), 1, 1)
            last = cur
        matches = [e['matches']-m for e, m in zip(cur, matches_before)]
        check('b1: actions echo per physical row and pairs stay mutually visible', action_echo_ok)
        check('b1: paired players separate after spawn', all(seen_apart), seen_apart=seen_apart)
        check('b1: match end terminates both active rows and restarts opponent state', terminal_pairs_ok)
        check('b1: finished opponent row is blanked', blank_after_ok)
        check('b1: several matches per env', min(matches) >= 3, matches=matches)
        check('b1: sides vary per historical env', all(len(s) == 2 for s in sides.values()), sides={k: sorted(v) for k, v in sides.items()})
        check('b1: banks vary per historical env', all(len(s) == 2 for s in bank_seen.values()), banks={k: sorted(v) for k, v in bank_seen.items()})
        check('b1: count_aligned reports each historical match boundary once', aligned_total == sum(matches[i] for i in sides), aligned=aligned_total, matches=matches)
        check('b1: sampled step transport', lib.bopl_vec_step_mode(ctypes.c_void_p(vec.gpu_obs_ptr)) == 2)
        # Restrict the valid banks: new matches must only pick bank 1.
        lib.bopl_set_valid_banks(0b10)
        picked = set()
        for step in range(args.steps//2):
            vec.cpu_step(actions.ctypes.data)
            cur = layout(lib, vec, 4)
            for i, e in enumerate(cur):
                if e['banks'] and e['matches'] != last[i]['matches']: picked.add(e['bank'])
            last = cur
        check('b1: valid-bank mask restricts new matches', picked == {1}, picked=sorted(picked))
        report['log'] = vec.log()
        vec.close()
    report['ok'] = all(c['ok'] for c in report['checks'])
    (args.output/'result.json').write_text(json.dumps(report, indent=2, default=str))
    print('ROUTING', 'PASS' if report['ok'] else 'FAIL', args.output/'result.json')
    sys.exit(0 if report['ok'] else 1)

if __name__ == '__main__':
    main()
