"""The style bonus through the real binding, on the host step path (no GPU work).

Steps a self-play vector with fixed pseudo-random actions and hashes every
step's host reward and observation rows. Run it on two builds without
--style: equal hashes prove the no-style reward is bit-identical. With
--style it installs the program, and because rewards never feed back into
the game, the trajectory is the same: each step's reward minus the no-style
reward must equal scale * the style payout, and bopl_vec_style's per-term
counters must sum to it.

  PYTHONPATH=<puffer> python train/check_style_binding.py --game <lib.so> [--style s.toml] [--scale 0.5] --json out.json
"""
import argparse, ctypes, hashlib, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'native/tools'))
import reference  # noqa: E402
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(HERE))
import layout  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--game', type=Path)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--episode-ticks', type=int, default=1200)
    parser.add_argument('--style', type=Path, nargs='*', default=[])
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--rewards', type=Path, help='save the per-step reward rows (npy)')
    parser.add_argument('--json', type=Path)
    parser.add_argument('--log-dir', type=Path, default=Path('/tmp/check-style-binding'))
    args = parser.parse_args()
    os.environ['BOPL_RL_GAME'] = str(reference.resolve_game(args.game))
    os.environ.setdefault('BOPL_RL_BOOTSTRAP', str(ROOT/'native/fixtures/movement.rep'))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    os.environ['BOPL_RL_LOG_DIR'] = str(args.log_dir)
    from pufferlib import _C
    lib = ctypes.CDLL(_C.__file__)
    lib.bopl_vec_handles.restype = ctypes.c_int; lib.bopl_vec_handles.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.bopl_vec_host.restype = ctypes.c_int; lib.bopl_vec_host.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_void_p)]*3
    lib.bopl_vec_reset.argtypes = [ctypes.c_void_p]; lib.bopl_vec_reset.restype = ctypes.c_int
    program = None
    if args.style:
        import style_compile
        program = style_compile.compile_files(args.style)
    # The trainer's own loader and counters (train.py StyleBinding); without
    # --style it installs nothing, as train.py does.
    import train as trainer_module
    style = trainer_module.StyleBinding(_C, program)
    if program is not None:
        style.lib.bopl_style_set_scale(args.scale)
        style.current = args.scale
    total = 2*args.envs
    env_kwargs = dict(players=2, seed=73, observation_delay=9, action_repeat=4, max_episode_ticks=args.episode_ticks,
                      warmup_ticks=60, map=0, training_map_mask=0, random_maps=1, reward_kill_milli=100,
                      reward_timeout_milli=-500)
    vec = _C.create_vec(dict(vec=dict(total_agents=total, num_buffers=1), env=env_kwargs), True)
    handles = (ctypes.c_void_p*8)()
    assert lib.bopl_vec_handles(handles, 8) == 1
    native = ctypes.c_void_p(handles[0])
    vec.reset()
    assert lib.bopl_vec_reset(native) == 0
    obs_p, rew_p, term_p = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    assert lib.bopl_vec_host(native, ctypes.byref(obs_p), ctypes.byref(rew_p), ctypes.byref(term_p)) == total
    obs = np.ctypeslib.as_array(ctypes.cast(obs_p, ctypes.POINTER(ctypes.c_float)), shape=(total*layout.OBS_STRIDE,))
    rewards = np.ctypeslib.as_array(ctypes.cast(rew_p, ctypes.POINTER(ctypes.c_float)), shape=(total,))
    terminals = np.ctypeslib.as_array(ctypes.cast(term_p, ctypes.POINTER(ctypes.c_float)), shape=(total,))
    rng = np.random.default_rng(11)
    actions = np.zeros((total, layout.NUM_ATNS), dtype=np.float32)
    reward_hash, obs_hash = hashlib.sha256(), hashlib.sha256()
    stream = np.zeros((args.steps, total), dtype=np.float32)
    episodes = 0
    for step in range(args.steps):
        actions[:, 0] = rng.integers(0, 256, total)
        actions[:, 1] = rng.integers(0, 16, total)
        actions[:, 2] = 0
        vec.cpu_step(actions.ctypes.data)
        stream[step] = rewards
        reward_hash.update(rewards.tobytes()); reward_hash.update(terminals.tobytes())
        obs_hash.update(obs.tobytes())
        episodes += int(terminals.sum())
    out = dict(steps=args.steps, rows=total, episodes=episodes, reward_sha256=reward_hash.hexdigest(),
               obs_sha256=obs_hash.hexdigest(), reward_sum=float(stream.astype(np.float64).sum()),
               nonzero_rewards=int((stream != 0).sum()))
    if program is not None:
        record = style.read(SimpleNamespace(gpu_obs_ptr=handles[0]))
        out['style'] = dict(sha256=program.sha256, scale=args.scale, decisions=record['decisions'],
                            pay={name: t['pay'] for name, t in record['terms'].items()},
                            paid={name: t['paid'] for name, t in record['terms'].items()},
                            fires={name: t['fires'] for name, t in record['terms'].items()})
    vec.close()
    if args.rewards:
        np.save(args.rewards, stream)
    text = json.dumps(out, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text)


if __name__ == '__main__':
    main()
