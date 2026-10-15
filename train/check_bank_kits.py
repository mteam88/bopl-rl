"""train.py --pinned-opponents (kit pinning) through the real binding, on the host step path (no GPU work).

Part A, unchanged behaviour: the same vector without bank_kits and with
bank_kits=1 but no rows installed must deal every match the same (side,
both kits, map). With --golden OLD_SO the flag-off run is also compared match
for match against a binding built before kit pinning existed.

Part B, pinning: four banks on the all-kits pool (historical_every 1, so every
environment sits in a bank).
  bank 0: no rows (the run's own draw; pin src must stay 0)
  bank 1: scripted rusher (kind 11) at share 0.5, camper rows for its frozen seat
  bank 2: punisher rows (league.json)
  bank 3: camper rows, swapped to punisher rows halfway through
Every match on a pinned bank must hand the frozen seat one of the kits of the
member whose table it was dealt from, on both sides; bot matches hold the bot
kit (2,4,14) and never a space map, and space-map matches on the rusher bank go
to the frozen camper. After the swap, new matches hold punisher kits. The
learner keeps its own varied draw. Both the spare-world takeover path (where a
setup drawn before the swap is re-dealt: redraws > 0) and the direct reset path
(BOPL_RL_INPROC_SPARE=0, the one train.py uses) are run.

  PYTHONPATH=<puffer> python train/check_bank_kits.py [--steps 4000] [--golden /path/_C-old.so]
"""
import argparse, ctypes, importlib.util, json, os, subprocess, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / 'native/tools'), str(ROOT / 'eval'), str(HERE), str(ROOT)]
import reference  # noqa: E402
import layout  # noqa: E402

BANKS = 4
CAMPER, PUNISHER, BOT = 'anticamp-camper-1000m', 'anticamp-punisher-4000m', 'rusher-hard'


def env_kwargs(pool_mask, **extra):
    return dict(players=2, seed=73, observation_delay=9, action_repeat=4, max_episode_ticks=600,
                warmup_ticks=60, map=0, training_map_mask=0, random_maps=1, reward_kill_milli=100,
                reward_timeout_milli=-500, banks=BANKS, historical_every=1,
                loadout_pool=pool_mask & 0x7fffffff, loadout_pool_hi=(pool_mask >> 31) & 0x7fffffff, **extra)


def handles(lib):
    lib.bopl_vec_handles.restype = ctypes.c_int
    lib.bopl_vec_handles.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.bopl_vec_reset.argtypes = [ctypes.c_void_p]
    for name in ('bopl_vec_layout', 'bopl_vec_kits'):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]


def clear_globals(_C, lib):
    import train as trainer
    trainer.install_loadout_set(_C, [])
    trainer.install_loadout_weights(_C, [])
    trainer.set_scripted_curriculum(_C, 0.0, [], [])
    if hasattr(lib, 'bopl_bank_kits_install'):
        import bank_kits
        kl = bank_kits.library(_C)
        for bank in range(BANKS):
            bank_kits.install(kl, bank, [])
        bank_kits.install_bot_kit(kl, [0, 0, 0])


def steps_of(_C, lib, kwargs, steps, envs, on_step):
    """Step a vector with random actions; on_step(step, native, n) after every step."""
    total = envs * (1 + BANKS)
    vec = _C.create_vec(dict(vec=dict(total_agents=total, num_buffers=1), env=kwargs), True)
    hs = (ctypes.c_void_p * 8)()
    assert lib.bopl_vec_handles(hs, 8) == 1
    native = ctypes.c_void_p(hs[0])
    vec.reset()
    assert lib.bopl_vec_reset(native) == 0
    rng = np.random.default_rng(11)
    actions = np.zeros((total, layout.NUM_ATNS), dtype=np.float32)
    for step in range(steps):
        actions[:, 0] = rng.integers(0, 256, total)
        actions[:, 1] = rng.integers(0, 16, total)
        vec.cpu_step(actions.ctypes.data)
        on_step(step, native)
    vec.close()


def record(_C, lib, kwargs, steps, envs):
    """Every match as (env, match) -> [side, kits, map]: the flag-off/no-rows comparison.

    The map comes from bopl_vec_pins, which a binding older than kit pinning
    lacks (map None then; the golden comparison drops it).
    """
    stride = 8 + 14
    lay = (ctypes.c_int * (envs * 64 * stride))()
    kits = (ctypes.c_int * (envs * 64 * 6))()
    pins = (ctypes.c_int * (envs * 64 * 4))() if hasattr(lib, 'bopl_vec_pins') else None
    if pins is not None:
        lib.bopl_vec_pins.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    seen = {}

    def on_step(step, native):
        n = lib.bopl_vec_layout(native, lay, len(lay))
        assert lib.bopl_vec_kits(native, kits, len(kits)) == n
        assert pins is None or lib.bopl_vec_pins(native, pins, len(pins)) == n
        for i in range(n):
            if not lay[i * stride + 2]:
                continue
            key = f'{i}:{lay[i * stride + 6]}'
            if key not in seen:
                seen[key] = [lay[i * stride + 4], [kits[i * 6 + k] for k in range(6)],
                             None if pins is None else pins[i * 4]]
    steps_of(_C, lib, kwargs, steps, envs, on_step)
    return seen


def pinning(_C, lib, kwargs, steps, envs, spare):
    import bank_kits, train as trainer
    from league import registry
    import scripted_roster
    from evaluation_protocol import SCRIPTED_KINDS
    members = registry.by_name(registry.load(verify=False))
    camper = [tuple(k) for k in registry.kits(members[CAMPER])]
    punisher = [tuple(k) for k in registry.kits(members[PUNISHER])]
    bot_kit = tuple(registry.kits(members[BOT])[0])
    space = set(registry._space_maps())
    kl = bank_kits.library(_C)
    tables = {}   # (bank, gen) -> allowed frozen kits (None = unpinned)

    def put(bank, kits_):
        gen = bank_kits.install(kl, bank, bank_kits.member_rows(kits_) if kits_ else [])
        tables[(bank, gen)] = set(kits_) if kits_ else None
    put(0, None); put(1, camper); put(2, punisher); put(3, camper)
    bank_kits.install_bot_kit(kl, bot_kit)
    trainer.set_scripted_curriculum(_C, 0.5, scripted_roster.entries_from_names([BOT], {BOT: 0.0}), [1.0])
    kwargs = dict(kwargs, bank_kits=1, scripted_kinds=SCRIPTED_KINDS['rusher'] << 4)
    seen, stats = set(), dict(matches=0, sides=[0, 0], bot=0, space_frozen=0, earlier_after_swap=0,
                               after_swap_punisher=0, unpinned=0, learner_kits=Counter(), frozen_kits={})
    redraws0 = kl.bopl_bank_kit_redraw_count()
    swap_at = steps // 2
    scratch = (ctypes.c_int * (envs * 64 * bank_kits.LAYOUT_STRIDE))()

    def on_step(step, native):
        if step == swap_at:
            put(3, punisher)
        lay, kits, pins = bank_kits.read(kl, native.value, lib.bopl_vec_layout(native, scratch, len(scratch)))
        for env, bank, match, side, mine, frozen, map_id, bot, src, gen in bank_kits.frozen_matches(lay, kits, pins):
            if (env, match) in seen:
                continue
            seen.add((env, match))
            stats['matches'] += 1; stats['sides'][side] += 1
            stats['learner_kits'][mine] += 1
            stats['frozen_kits'].setdefault(bank, Counter())[frozen] += 1
            where = f'spare={spare} step {step} env {env} bank {bank} match {match} side {side} map {map_id}'
            if bot:
                assert bank == 1 and bot == SCRIPTED_KINDS['rusher'], f'{where}: bot {bot}'
                assert frozen == bot_kit and map_id not in space, f'{where}: bot holds {frozen}'
                stats['bot'] += 1
                continue
            assert (bank, gen) in tables, f'{where}: gen {gen} never installed'
            allowed = tables[(bank, gen)]
            if allowed is None:
                assert src == 0, f'{where}: unpinned bank src {src}'
                stats['unpinned'] += 1
            else:
                assert src == 1 and frozen in allowed, f'{where}: frozen {frozen} src {src} not in gen {gen} table'
            if bank == 1 and map_id in space:
                stats['space_frozen'] += 1
            if bank == 3 and step > swap_at:
                if allowed == set(punisher):
                    stats['after_swap_punisher'] += 1
                else:
                    stats['earlier_after_swap'] += 1
    steps_of(_C, lib, kwargs, steps, envs, on_step)
    stats['redraws'] = int(kl.bopl_bank_kit_redraw_count() - redraws0)
    stats['learner_distinct'] = len(stats['learner_kits'])
    stats['frozen_distinct'] = {b: len(c) for b, c in sorted(stats['frozen_kits'].items())}
    del stats['learner_kits'], stats['frozen_kits']
    return stats


def load_module(path):
    if not path:
        from pufferlib import _C
        return _C
    spec = importlib.util.spec_from_file_location('pufferlib._C', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['pufferlib._C'] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--game', type=Path)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--golden', type=Path, help='a binding built before kit pinning, for a flag-off comparison')
    parser.add_argument('--record', type=Path, help=argparse.SUPPRESS)   # subprocess half of --golden
    parser.add_argument('--module', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--direct', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    # The game train.py runs: the in-process Rust sim under BOPL_RL_BACKEND=rust
    # (spare world optional), else the reference build (always a spare process).
    rust = os.environ.get('BOPL_RL_BACKEND', '').lower() == 'rust'
    game = (args.game or ROOT / 'native/target/release/libbopl_sim.so').resolve() if rust else reference.resolve_game(args.game)
    os.environ['BOPL_RL_GAME'] = str(game)
    if rust:
        os.environ.setdefault('BOPL_RL_WORLDS_PER_RUNTIME', str(1 << 20))
    os.environ.setdefault('BOPL_RL_BOOTSTRAP', str(ROOT / 'native/fixtures/movement.rep'))
    os.environ.setdefault('BOPL_RL_LOG_DIR', '/tmp/check-bank-kits')
    Path(os.environ['BOPL_RL_LOG_DIR']).mkdir(parents=True, exist_ok=True)
    from evaluation_protocol import loadout_pool
    pool_mask = sum(1 << a for a in loadout_pool('all'))
    _C = load_module(args.module)
    lib = ctypes.CDLL(_C.__file__)
    handles(lib)

    def path(direct):
        if direct:
            os.environ['BOPL_RL_INPROC_SPARE'] = '0'
        else:
            os.environ.pop('BOPL_RL_INPROC_SPARE', None)
        clear_globals(_C, lib)
    if args.record:
        path(args.direct)
        args.record.write_text(json.dumps(record(_C, lib, env_kwargs(pool_mask), args.steps, args.envs)))
        return

    # train.py's path (direct under the Rust backend) and the spare-world takeover;
    # the reference build cannot drop its spare, so it runs that path only.
    report = {}
    for direct in ((True, False) if rust else (False,)):
        name = 'direct' if direct else 'spare'
        path(direct)
        off = record(_C, lib, env_kwargs(pool_mask), args.steps, args.envs)
        path(direct)
        norows = record(_C, lib, env_kwargs(pool_mask, bank_kits=1), args.steps, args.envs)
        assert len(off) >= 50 and off == norows, (name, len(off), len(norows),
                                                  [k for k in off if off[k] != norows.get(k)][:5])
        report[f'{name}_unchanged'] = dict(matches=len(off), sides=Counter(v[0] for v in off.values()))
        if args.golden:
            out = Path(os.environ['BOPL_RL_LOG_DIR']) / f'golden-{name}.json'
            subprocess.run([sys.executable, __file__, '--record', str(out), '--module', str(args.golden),
                            '--steps', str(args.steps), '--envs', str(args.envs), *(['--direct'] if direct else []),
                            *(['--game', str(args.game)] if args.game else [])], check=True)
            old = {k: v[:2] for k, v in json.loads(out.read_text()).items()}
            plain = {k: v[:2] for k, v in off.items()}
            assert old == plain, (name, len(old), len(plain), [k for k in plain if plain[k] != old.get(k)][:5])
            report[f'{name}_golden'] = dict(matches=len(old), identical=True)

        path(direct)
        s = pinning(_C, lib, env_kwargs(pool_mask), args.steps, args.envs, not direct)
        report[name] = s
        assert s['matches'] >= 100 and min(s['sides']) > 0.3 * s['matches'], s
        assert s['bot'] > 0 and s['space_frozen'] > 0 and s['unpinned'] > 0, s
        assert s['after_swap_punisher'] > 0, s
        assert s['learner_distinct'] > 0.5 * s['matches'], s
        # A setup the spare drew before the swap is re-dealt at takeover; the
        # direct path draws at reset and never holds a stale one.
        assert (s['redraws'] > 0) if not direct else (s['redraws'] == 0), s
    os.environ.pop('BOPL_RL_INPROC_SPARE', None)
    print(json.dumps(report, default=str))
    print('ok: pinned frozen seats always held their member\'s kits on both sides (bot: its own kit, no space maps); '
          'without the flag or rows every match is dealt as before')


if __name__ == '__main__':
    main()
