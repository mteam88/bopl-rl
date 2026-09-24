"""Report the RL ABI identity of one or more game backends (bopl_rl.h).

Two libraries can only host the same evaluation if they agree on the ABI
version, the ability/map catalogue and the catalogue hash; a difference in any
of them means the policies are reading a different observation stream or the
loadout ids mean different abilities. This opens each game through the very
same host path the evaluation uses (research/environment.py's libbopl_research.so,
which links native/rl/bopl_rl.c), so what it reports is what an evaluation would
actually get -- not a static guess from `nm`.

    python game_abi.py <game.so> [<game2.so> ...]

Exit status 1 when the games disagree on version/catalog/map/ability counts.
"""
import ctypes as C
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))


class PlayerConfig(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('color', 'team', 'keyboard')] + [('abilities', C.c_uint32 * 3)]


class Config(C.Structure):
    _fields_ = ([(name, C.c_uint32) for name in
                 ('seed', 'map', 'players', 'slots', 'stocks', 'mode', 'warmup_ticks', 'max_episode_ticks')]
                + [('player', PlayerConfig * 6)])


def identity(game, log_dir):
    """Open one world on `game` in a fresh process and report what it says about itself."""
    import environment
    from environment import LIB
    environment.configure(log_dir, game)
    LIB.bopl_rl_open.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, C.c_uint32, C.c_char_p, C.c_size_t]
    LIB.bopl_rl_open.restype = C.c_void_p
    LIB.bopl_rl_close.argtypes = [C.c_void_p]
    LIB.bopl_rl_fork.argtypes = [C.c_void_p, C.c_void_p]; LIB.bopl_rl_fork.restype = C.c_void_p
    LIB.bopl_rl_default_config.argtypes = [C.POINTER(Config), C.c_uint32]
    LIB.bopl_rl_reset.argtypes = [C.c_void_p, C.POINTER(Config)]; LIB.bopl_rl_reset.restype = C.c_int
    for name in ('source_hash', 'catalog_hash', 'error'):
        fn = getattr(LIB, 'bopl_rl_' + name); fn.argtypes = [C.c_void_p]; fn.restype = C.c_char_p
    for name in ('map_count', 'ability_count'):
        fn = getattr(LIB, 'bopl_rl_' + name); fn.argtypes = [C.c_void_p]; fn.restype = C.c_uint32
    error = C.create_string_buffer(4096)
    world = LIB.bopl_rl_open(os.fsencode(str(Path(game).resolve())),
                             os.fsencode(str(ROOT / 'native/fixtures/movement.rep')),
                             os.fsencode(str(Path(log_dir).resolve() / 'game-abi.log')), 120000, error, len(error))
    if not world:
        raise SystemExit(f'{game}: open failed: {error.value.decode(errors="replace")}')
    # bopl_rl_fork refuses a world with no episode (abi.rs: `!source.world.valid`),
    # so a fork probed straight after open reports false even for the Rust sim.
    # Reset first, then ask -- otherwise this tool would report a falsehood.
    config = Config()
    LIB.bopl_rl_default_config(C.byref(config), 2)
    if LIB.bopl_rl_reset(world, C.byref(config)):
        raise SystemExit(f'{game}: reset failed: {LIB.bopl_rl_error(world).decode(errors="replace")}')
    fork = LIB.bopl_rl_fork(world, None)
    if fork:
        LIB.bopl_rl_close(fork)
    # The host refuses to load a library whose bopl_rl_abi_version() is not 3
    # (bopl_rl.c load_inproc_lib), so a successful in-process open proves it.
    info = dict(game=str(Path(game).resolve()),
                abi_version=3 if str(game).endswith('.so') else None,
                source_hash=LIB.bopl_rl_source_hash(world).decode(),
                catalog_hash=LIB.bopl_rl_catalog_hash(world).decode(),
                map_count=int(LIB.bopl_rl_map_count(world)),
                ability_count=int(LIB.bopl_rl_ability_count(world)),
                forks=bool(fork),
                identity=environment.game_identity(game))
    LIB.bopl_rl_close(world)
    return info


def main():
    games = sys.argv[1:]
    if not games:
        raise SystemExit(__doc__)
    logs = ROOT / 'native/artifacts/rl/logs/game-abi'
    rows = []
    for game in games:
        # A NativeAOT runtime cannot be unloaded, and two different games would
        # both be resident in one process; ask a child process about each.
        out = __import__('subprocess').run(
            [sys.executable, __file__, '--one', game, str(logs)],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent))
        if out.returncode:
            raise SystemExit(f'{game}: {out.stdout}{out.stderr}')
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
    for row in rows:
        print(json.dumps(row, indent=2))
    keys = ('abi_version', 'catalog_hash', 'map_count', 'ability_count')
    disagree = [k for k in keys if len({r[k] for r in rows}) > 1]
    print('\nAGREE on', ', '.join(keys)) if not disagree else print('\nDISAGREE on', ', '.join(disagree))
    print('source_hash:', {r['game'].rsplit('/', 1)[-1]: r['source_hash'] for r in rows})
    print('forks:', {r['game'].rsplit('/', 1)[-1]: r['forks'] for r in rows})
    return 1 if disagree else 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--one':
        print(json.dumps(identity(sys.argv[2], sys.argv[3])))
    else:
        sys.exit(main())
