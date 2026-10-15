"""train.py --loadout-learner-first through the real binding, on the host step path (no GPU work).

Builds a vector of frozen-bank environments (historical_every 1, four banks) on a
two-row lineup table whose learner halves share no ability with the opponent
halves, steps it with random actions through many episodes (both the spare-world
takeover and the direct reset path, whichever the build uses), and after every
step reads each environment's side (bopl_vec_layout) and dealt kits
(bopl_vec_kits). With the flag the learner's seat must always hold a first half
and the other seat a second half, and both sides must occur; without it the
learner must hold a second half in some episodes (the check would catch a no-op).

  PYTHONPATH=<puffer> python train/check_learner_first.py [--steps 3000]
"""
import argparse, ctypes, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / 'native/tools'), str(ROOT / 'eval'), str(HERE)]
import reference  # noqa: E402
import layout  # noqa: E402

ROWS = [(2, 2, 14, 4, 3, 28), (2, 26, 14, 9, 36, 4)]
LEARNER = {r[:3] for r in ROWS}
OPPONENT = {r[3:] for r in ROWS}


def run(lib, _C, flag, steps, envs):
    import train as trainer
    trainer.install_loadout_set(_C, [list(r) for r in ROWS])
    banks = 4
    total = envs * (1 + banks)
    env_kwargs = dict(players=2, seed=73, observation_delay=9, action_repeat=4, max_episode_ticks=600,
                      warmup_ticks=60, map=0, training_map_mask=0, random_maps=1, reward_kill_milli=100,
                      reward_timeout_milli=-500, banks=banks, historical_every=1,
                      **(dict(loadout_learner_first=1) if flag else {}))
    vec = _C.create_vec(dict(vec=dict(total_agents=total, num_buffers=1), env=env_kwargs), True)
    handles = (ctypes.c_void_p * 8)()
    assert lib.bopl_vec_handles(handles, 8) == 1
    native = ctypes.c_void_p(handles[0])
    vec.reset()
    assert lib.bopl_vec_reset(native) == 0
    stride = 8 + 14   # bopl_vec_layout: 8 + BOPL_MAX_SLOTS (bopl_puffer.h)
    lay = (ctypes.c_int * (envs * 64 * stride))()
    kits = (ctypes.c_int * (envs * 64 * 6))()
    rng = np.random.default_rng(11)
    actions = np.zeros((total, layout.NUM_ATNS), dtype=np.float32)
    seen, sides, learner_second, episodes = set(), [0, 0], 0, 0
    for step in range(steps):
        actions[:, 0] = rng.integers(0, 256, total)
        actions[:, 1] = rng.integers(0, 16, total)
        vec.cpu_step(actions.ctypes.data)
        n = lib.bopl_vec_layout(native, lay, len(lay))
        assert lib.bopl_vec_kits(native, kits, len(kits)) == n
        for i in range(n):
            if lay[i * stride + 2] == 0:
                continue
            side, matches = lay[i * stride + 4], lay[i * stride + 6]
            mine = tuple(kits[i * 6 + side * 3 + k] for k in range(3))
            theirs = tuple(kits[i * 6 + (1 - side) * 3 + k] for k in range(3))
            if (i, matches) in seen:
                continue
            seen.add((i, matches)); sides[side] += 1
            if flag:
                assert mine in LEARNER and theirs in OPPONENT, f'env {i} match {matches} side {side}: {mine} | {theirs}'
            learner_second += mine in OPPONENT
    vec.close()
    return dict(matches=len(seen), sides=sides, learner_second=learner_second)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--game', type=Path)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--envs', type=int, default=8)
    args = parser.parse_args()
    os.environ['BOPL_RL_GAME'] = str(reference.resolve_game(args.game))
    os.environ.setdefault('BOPL_RL_BOOTSTRAP', str(ROOT / 'native/fixtures/movement.rep'))
    os.environ.setdefault('BOPL_RL_LOG_DIR', '/tmp/check-learner-first')
    Path(os.environ['BOPL_RL_LOG_DIR']).mkdir(parents=True, exist_ok=True)
    from pufferlib import _C
    lib = ctypes.CDLL(_C.__file__)
    lib.bopl_vec_handles.restype = ctypes.c_int; lib.bopl_vec_handles.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.bopl_vec_reset.argtypes = [ctypes.c_void_p]
    lib.bopl_vec_layout.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.bopl_vec_kits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    on = run(lib, _C, True, args.steps, args.envs)
    off = run(lib, _C, False, args.steps, args.envs)
    print(dict(learner_first=on, default=off))
    assert on['matches'] >= 50 and min(on['sides']) > 0.3 * on['matches'], on
    assert on['learner_second'] == 0
    assert off['learner_second'] > 0.3 * off['matches'], off
    print('ok: with --loadout-learner-first the learner always held its own half, on both sides')


if __name__ == '__main__':
    main()
