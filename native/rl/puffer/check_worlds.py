"""Many-worlds parity at the trainer binding: N worlds per process vs one process per game.

Builds two native vectors with identical environment kwargs except `worlds`
(1: one game process per environment plus a spare; N: N environments share
one process), installs the same non-identity agent permutation and tags on
both, resets both and steps both with the same scripted actions for --steps
vector steps, spanning many match boundaries (short episodes) on the training
maps. After every step the host observation, reward and terminal rows and the
cumulative counters must be identical. Runs on the host stepping path.
"""
import argparse, ctypes, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'native/tools'))
import reference  # noqa: E402
sys.path.insert(0, str(HERE))
import counters  # noqa: E402
from check_routing import layout, build_perm, OBS  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'native/artifacts/rl/autoresearch/native-worlds')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--games', type=int, default=8, help='games per vector (multiple of 2*buffers*worlds)')
    parser.add_argument('--buffers', type=int, default=2)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--episode-ticks', type=int, default=240)
    parser.add_argument('--maps', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--game', type=Path)
    args = parser.parse_args()
    if args.games % (2*args.buffers*args.worlds) or args.worlds < 2:
        parser.error('games must be a multiple of 2*buffers*worlds and worlds >= 2')
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
    counters.bind(lib)
    report = dict(schema=1, game=os.environ['BOPL_RL_GAME'], worlds=args.worlds, games=args.games, buffers=args.buffers,
                  steps=args.steps, maps=args.maps, checks=[])
    def check(name, ok, **info):
        report['checks'].append(dict(name=name, ok=bool(ok), **info)); print(('PASS ' if ok else 'FAIL ') + name, info if not ok else '', flush=True)
        return ok
    banks = 2
    games = args.games
    total = (games//2)*(2+1+banks)
    base = dict(players=2, banks=banks, historical_every=2, seed=73, observation_delay=9, action_repeat=4,
                max_episode_ticks=args.episode_ticks, warmup_ticks=60, map=args.maps[0], training_map_mask=sum(1 << m for m in args.maps),
                random_loadouts=1, reward_kill_milli=0, reward_timeout_milli=-500)
    vecs = {}
    for label, worlds in (('single', 1), ('many', args.worlds)):
        vec = _C.create_vec(dict(vec=dict(total_agents=total, num_buffers=args.buffers), env=dict(base, worlds=worlds)), True)
        envs = layout(lib, vec, games)
        perm, _ = build_perm(envs, args.buffers, banks)
        assert lib.bopl_vec_set_perm(ctypes.c_void_p(vec.gpu_obs_ptr), (ctypes.c_int*total)(*perm)) == 0
        assert lib.bopl_vec_set_tags(ctypes.c_void_p(vec.gpu_obs_ptr), (ctypes.c_int*games)(*[e['tag'] for e in envs])) == 0
        vec.reset()
        vecs[label] = dict(vec=vec, perm=perm,
                           obs=np.ctypeslib.as_array((ctypes.c_float*(total*OBS)).from_address(vec.obs_ptr)).reshape(total, OBS),
                           rewards=np.ctypeslib.as_array((ctypes.c_float*total).from_address(vec.rewards_ptr)),
                           terminals=np.ctypeslib.as_array((ctypes.c_float*total).from_address(vec.terminals_ptr)))
    single, many = vecs['single'], vecs['many']
    check('same permutation', single['perm'] == many['perm'])
    logs = sorted(p.name for p in log_dir.iterdir())
    check(f'{args.worlds} worlds share one process log per group', sum('-p' in n for n in logs) == games//args.worlds, logs=logs)
    def same(label):
        s_layout, m_layout = layout(lib, single['vec'], games), layout(lib, many['vec'], games)
        ok = check(f'{label}: layout (sides, banks, matches, steps)', s_layout == m_layout)
        ok &= check(f'{label}: observations identical', np.array_equal(single['obs'], many['obs']),
                    rows=[int(r) for r in np.nonzero((single['obs'] != many['obs']).any(axis=1))[0][:8]])
        ok &= check(f'{label}: rewards identical', np.array_equal(single['rewards'], many['rewards']))
        ok &= check(f'{label}: terminals identical', np.array_equal(single['terminals'], many['terminals']))
        cs, cm = counters.snapshot(lib, single['vec']), counters.snapshot(lib, many['vec'])
        ok &= check(f'{label}: counters identical', cs == cm, single=cs, many=cm)
        return ok, cs
    ok, count = same('after reset')
    actions = np.zeros((total, 2), dtype=np.float32)
    matches_at = count['matches']; first_failure = None
    for step in range(args.steps):
        for r in range(total):
            actions[r] = (1 + (r*37 + step*11) % 255, ((r + step) % 16) | (2 if step % 5 == 0 else 0) | (4 if step % 7 == 0 else 0) | (8 if step % 11 == 0 else 0))
        single['vec'].cpu_step(actions.ctypes.data)
        many['vec'].cpu_step(actions.ctypes.data)
        step_ok = (np.array_equal(single['obs'], many['obs']) and np.array_equal(single['rewards'], many['rewards'])
                   and np.array_equal(single['terminals'], many['terminals']))
        if not step_ok and first_failure is None:
            first_failure = step
            same(f'step {step}')
            break
    ok, count = same(f'after {args.steps} steps')
    check('matches completed across resets', count['matches'] - matches_at >= games*2, matches=count['matches'] - matches_at)
    check('every training map played', all(count['map_matches'].get(str(m), 0) > 0 for m in args.maps), map_matches=count['map_matches'])
    check('no first failure', first_failure is None, first_failure=first_failure)
    report['counters'] = count
    for v in vecs.values(): v['vec'].close()
    report['ok'] = all(c['ok'] for c in report['checks'])
    (args.output/'result.json').write_text(json.dumps(report, indent=2, default=str))
    print('MANY_WORLDS_TRAINER', 'PASS' if report['ok'] else 'FAIL', args.output/'result.json')
    sys.exit(0 if report['ok'] else 1)

if __name__ == '__main__':
    main()
