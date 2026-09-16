"""Recover full per-tick state of the human-vs-agent rounds (games.jsonl + .rep) by
replaying the recorded inputs through the Unity Linux build.

    python human_replays.py extract --out DIR [--workers 4] [--log native/artifacts/rl/human/mac/games.jsonl]

The Mac recordings (BOPLREP2, source hash d0562ac8...) carry only the per-tick input
stream; the Linux Mono build (Builds/Linux) is built from the same simulation source
and reproduces every round tick for tick (checked: the world ends at the recorded
level tick with the recorded winner and cause of death). That build speaks RL
protocol / state ABI v2; HEAD's bopl_rl.c was moved to v3 for the .NET/Rust sims
(3721f8a) without a Unity rebuild, so this module compiles the pre-v3 bopl_rl.c from
git into native/artifacts/rl/libbopl_rl_v2.so and reads the v2 state layout itself.

Output: DIR/games/<session>-r<round>.json.gz, one per round, with the games.jsonl row,
the replay header and one frame per level tick (players, slimes, arrows, platforms,
telemetry and the inputs applied that tick); DIR/games.jsonl is the manifest with the
fidelity check per round.
"""
import argparse
import ctypes as C
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

V2_COMMIT = '3721f8a^'  # last bopl_rl.c that speaks the Unity build's protocol
V2_LIB = ROOT / 'native/artifacts/rl/libbopl_rl_v2.so'
UNITY = ROOT / 'Builds/Linux/BoplBattle.x86_64'
FIX = 4294967296.0
TELEMETRY_FIELDS = ('dash_activations', 'bow_activations', 'platform_activations', 'arrows_fired', 'arrow_hits',
                    'deaths_environment', 'deaths_opponent', 'deaths_self', 'deaths_other')


def build_v2():
    if V2_LIB.exists():
        return V2_LIB
    work = V2_LIB.parent / 'bopl_rl_v2_src'
    work.mkdir(parents=True, exist_ok=True)
    for name in ('bopl_rl.c', 'bopl_rl.h'):
        src = subprocess.run(['git', 'show', f'{V2_COMMIT}:native/rl/{name}'], cwd=ROOT, check=True, capture_output=True).stdout
        (work / name).write_bytes(src)
    subprocess.run(['cc', '-O2', '-shared', '-fPIC', '-o', str(V2_LIB), str(work / 'bopl_rl.c'), '-ldl', '-lpthread'], check=True)
    return V2_LIB


class Player(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('id', 'team', 'flags', 'alive_bodies', 'stocks', 'kills', 'deaths', 'cause_of_death')] + \
               [(n, C.c_int64) for n in ('x', 'y', 'scale', 'invisible_time', 'ability_time')]


class Entity(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('id', 'kind', 'layer', 'owner', 'flags', 'hierarchy', 'reserved0', 'reserved1')] + \
               [(n, C.c_int64) for n in ('x', 'y', 'vx', 'vy', 'right_x', 'right_y', 'up_x', 'up_y', 'radius', 'rotation', 'angular_velocity', 'inverse_mass')]


class Slime(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('id', 'owner', 'flags', 'state')] + [(n, C.c_int64) for n in ('x', 'y', 'vx', 'vy', 'rotation', 'scale')] + \
               [('abilities', C.c_uint32 * 3), ('ability_flags', C.c_uint32 * 3), ('cooldowns', C.c_int64 * 3), ('timers', C.c_int64 * 3)]


class Header(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('version', 'tick', 'episode_ticks', 'flags', 'player_count', 'entity_count', 'warmup_ticks', 'slime_count')] + \
               [('simulation_time', C.c_int64), ('player_time_scale', C.c_int64)]


class Telemetry(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in TELEMETRY_FIELDS]


class State(C.Structure):
    _fields_ = [('header', Header), ('players', Player * 6), ('entities', C.POINTER(Entity)), ('slimes', C.POINTER(Slime)), ('telemetry', Telemetry * 6)]


class PlayerConfig(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('color', 'team', 'keyboard')] + [('abilities', C.c_uint32 * 3)]


class Config(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('seed', 'map', 'players', 'slots', 'stocks', 'mode', 'warmup_ticks', 'max_episode_ticks')] + [('player', PlayerConfig * 6)]


class Action(C.Structure):
    _fields_ = [('angle', C.c_uint8), ('reserved', C.c_uint8), ('buttons', C.c_uint16)]


def load_lib():
    lib = C.CDLL(str(build_v2()))
    lib.bopl_rl_open.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, C.c_uint32, C.c_char_p, C.c_size_t]
    lib.bopl_rl_open.restype = C.c_void_p
    lib.bopl_rl_reset.argtypes = [C.c_void_p, C.POINTER(Config)]
    lib.bopl_rl_step.argtypes = [C.c_void_p, C.POINTER(Action), C.c_size_t, C.c_uint32]
    lib.bopl_rl_state.argtypes = [C.c_void_p]
    lib.bopl_rl_state.restype = C.POINTER(State)
    lib.bopl_rl_close.argtypes = [C.c_void_p]
    for name in ('error', 'source_hash', 'catalog_hash'):
        f = getattr(lib, 'bopl_rl_' + name)
        f.argtypes = [C.c_void_p]
        f.restype = C.c_char_p
    return lib


def fix(v, digits=3):
    return round(v / FIX, digits)


def frame(state, inputs=None):
    """Compact per-tick record. Lists, not dicts, to keep 300k+ frames small; see FRAME_KEYS."""
    h = state.header
    players = [[p.id, p.flags, p.cause_of_death, p.kills, p.deaths, fix(p.x), fix(p.y)] for p in state.players[:h.player_count]]
    slimes = [[s.owner, s.flags, s.state, fix(s.x), fix(s.y), fix(s.vx), fix(s.vy), list(s.abilities), list(s.ability_flags),
               [fix(v) for v in s.cooldowns], [fix(v) for v in s.timers]] for s in state.slimes[:h.slime_count]]
    arrows, platforms, others = [], [], []
    for e in state.entities[:h.entity_count]:
        if not e.flags & 1 or e.flags & 4:
            continue
        if e.kind == 0 and e.layer == 9:
            arrows.append([e.id, e.owner, fix(e.x), fix(e.y), fix(e.vx), fix(e.vy), e.flags])
        elif e.kind == 2:
            platforms.append([e.id, fix(e.x), fix(e.y), fix(e.right_x), fix(e.right_y), fix(e.up_x), fix(e.up_y), fix(e.radius), fix(e.vx), fix(e.vy), e.flags, e.hierarchy])
        elif e.kind in (0, 1) and e.layer == 8 and e.owner:
            continue  # player bodies/hurtboxes: the slime record carries them
        elif e.kind == 1 and e.layer == 4:
            continue  # water box
        else:
            others.append([e.id, e.kind, e.layer, e.owner, fix(e.x), fix(e.y), fix(e.vx), fix(e.vy), fix(e.radius), e.flags])
    telemetry = [[getattr(state.telemetry[i], f) for f in TELEMETRY_FIELDS] for i in range(h.player_count)]
    out = dict(t=h.tick, flags=h.flags, players=players, slimes=slimes, arrows=arrows, platforms=platforms, telemetry=telemetry)
    if others:
        out['others'] = others
    if inputs is not None:
        out['inputs'] = inputs
    return out


FRAME_KEYS = dict(
    players=('id', 'flags', 'cause_of_death', 'kills', 'deaths', 'x', 'y'),
    slimes=('owner', 'flags', 'state', 'x', 'y', 'vx', 'vy', 'abilities', 'ability_flags', 'cooldowns', 'timers'),
    arrows=('id', 'owner', 'x', 'y', 'vx', 'vy', 'flags'),
    platforms=('id', 'x', 'y', 'rx', 'ry', 'ux', 'uy', 'radius', 'vx', 'vy', 'flags', 'hierarchy'),
    others=('id', 'kind', 'layer', 'owner', 'x', 'y', 'vx', 'vy', 'radius', 'flags'),
    telemetry=TELEMETRY_FIELDS,
    inputs=('angle', 'buttons'),
)


class World:
    def __init__(self, lib, meta, log, game=UNITY):
        self.lib = lib
        cfg = self.config = Config()
        cfg.seed, cfg.map, cfg.players = meta['seed'], meta['level'], len(meta['players'])
        cfg.slots = len(meta['players'][0]['abilities'])
        cfg.stocks, cfg.mode, cfg.warmup_ticks, cfg.max_episode_ticks = meta['stocks'], meta['mode'], 0, meta['ticks'] + 1
        for i, p in enumerate(meta['players']):
            cfg.player[i] = PlayerConfig(p['color'], p['team'], p['keyboard'], (C.c_uint32 * 3)(*p['abilities']))
        error = C.create_string_buffer(512)
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        self.handle = lib.bopl_rl_open(os.fsencode(game), os.fsencode(ROOT / 'native/fixtures/movement.rep'), os.fsencode(log), 60000, error, 512)
        if not self.handle:
            raise RuntimeError(error.value.decode())
        for key in ('source', 'catalog'):
            have = getattr(lib, 'bopl_rl_' + key + '_hash')(self.handle).decode()
            if have != meta[key]:
                self.close()
                raise ValueError(f'replay {key} hash {meta[key][:8]} differs from the game build {have[:8]}')
        if lib.bopl_rl_reset(self.handle, C.byref(cfg)):
            raise RuntimeError(lib.bopl_rl_error(self.handle).decode())

    @property
    def state(self):
        return self.lib.bopl_rl_state(self.handle).contents

    def step(self, actions):
        acts = (Action * len(actions))(*(Action(int(a), 0, int(b)) for a, b in actions))
        status = self.lib.bopl_rl_step(self.handle, acts, len(actions), 1)
        if status not in (0, 4):
            raise RuntimeError(f'step status {status}: {self.lib.bopl_rl_error(self.handle).decode()}')
        return status

    def close(self):
        if self.handle:
            self.lib.bopl_rl_close(self.handle)
            self.handle = None


def replay_round(lib, row, replay_dir, log):
    from replay_data import decode
    meta, inputs = decode(replay_dir / os.path.basename(row['replay']))
    world = World(lib, meta, log)
    frames = []
    ended_at = None
    try:
        frames.append(frame(world.state))
        for t in range(meta['ticks']):
            acts = list(zip(inputs[t]['angle'].tolist(), inputs[t]['buttons'].tolist()))
            world.step(acts)
            frames.append(frame(world.state, [list(a) for a in acts]))
            if world.state.header.flags & 1:
                ended_at = t + 1
                break
    finally:
        world.close()
    final = frames[-1]
    alive = [p[1] & 1 for p in final['players']]
    winner = next((i for i, p in enumerate(final['players']) if p[1] & 2), None)
    outcome = 'a' if winner == 0 else 'b' if winner == 1 else 'draw'
    causes = [p[2] for p in final['players']]
    check = dict(ended=ended_at is not None, ended_at=ended_at, recorded_level_ticks=row['level_ticks'],
                 ticks_match=ended_at == row['level_ticks'], outcome=outcome, outcome_match=outcome == row['outcome'],
                 causes=causes, causes_match=causes == [p['cause_of_death'] for p in row['players']], alive=alive,
                 replay_ticks=meta['ticks'])
    return meta, frames, check


def _worker(args):
    out, rows, replay_dir, index = args
    lib = load_lib()
    results = []
    for row in rows:
        name = f"{row['session']}-r{row['round']}"
        started = time.monotonic()
        try:
            meta, frames, check = replay_round(lib, row, replay_dir, out / 'workers' / f'w{index}.log')
            meta = {k: v for k, v in meta.items() if k != 'path'}
            with gzip.open(out / 'games' / (name + '.json.gz'), 'wt', compresslevel=4) as handle:
                json.dump(dict(schema=1, name=name, row=row, meta=meta, keys=FRAME_KEYS, check=check, frames=frames), handle, separators=(',', ':'))
            result = dict(name=name, file=name + '.json.gz', seconds=round(time.monotonic() - started, 2), **check)
        except Exception as error:  # keep going, report
            result = dict(name=name, error=repr(error), seconds=round(time.monotonic() - started, 2))
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def extract(args):
    out = args.out.resolve()
    (out / 'games').mkdir(parents=True, exist_ok=True)
    (out / 'workers').mkdir(exist_ok=True)
    rows = [json.loads(l) for l in args.log.read_text().splitlines() if l.strip()]
    replay_dir = args.log.parent
    manifest = out / 'games.jsonl'
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                r = json.loads(line)
                if 'error' not in r:
                    done.add(r['name'])
            except ValueError:
                pass
    todo = [r for r in rows if f"{r['session']}-r{r['round']}" not in done]
    print(f'{len(rows)} rounds, {len(todo)} to replay, {args.workers} workers', flush=True)
    build_v2()
    buckets = [todo[i::args.workers] for i in range(args.workers)]
    ctx = mp.get_context('spawn')
    started = time.monotonic()
    with ctx.Pool(args.workers) as pool, manifest.open('a') as handle:
        for results in pool.imap_unordered(_worker, [(out, b, replay_dir, i) for i, b in enumerate(buckets) if b]):
            for r in results:
                handle.write(json.dumps(r) + '\n')
            handle.flush()
    print(f'done in {time.monotonic() - started:.0f}s', flush=True)


def load_game(path):
    """Read one extracted round back as dict-per-record frames."""
    with gzip.open(path, 'rt') as handle:
        game = json.load(handle)
    keys = game['keys']
    for f in game['frames']:
        for field, names in keys.items():
            if field in f:
                f[field] = [dict(zip(names, rec)) for rec in f[field]]
    return game


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--log', type=Path, default=ROOT / 'native/artifacts/rl/human/mac/games.jsonl')
    p.add_argument('--workers', type=int, default=4)
    p.set_defaults(func=extract)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
