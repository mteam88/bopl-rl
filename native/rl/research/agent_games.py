"""Play checkpoint-vs-bot / checkpoint-vs-checkpoint rounds in the Unity Linux build with
the in-game agent shim (libbopl_agent.so) on BOTH sides, saving frames in the
human_replays.py format so human_taxonomy.py classifies them with the same code.

    python agent_games.py play --out DIR --agent NAME=PATH ... --opponent SPEC ... [--maps 0,1,2,...] [--seeds 4] [--workers 4]

Why Unity and not the Rust sim: the human rounds were played against layout-v2
checkpoints (o7-*), HEAD's research library and Rust sim are layout v3, but the
Sep-14 libbopl_agent.so is still v2 and runs those exact weights (and the scripted
bots) against the same Unity build that reproduces the human replays. This also
allows the unseen levels the human played on; the search bots cannot run here
(they fork the in-process sim).

Player 0 = opponent (the "human" seat of the taxonomy), player 1 = the agent under study.
"""
import argparse
import ctypes as C
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import human_replays as hr  # noqa: E402

AGENT_LIB = Path(os.environ.get('BOPL_AGENT_LIBRARY') or ROOT / 'native/artifacts/rl/libbopl_agent.so')
STANDARD = [2, 4, 14]
WARMUP = 240
LIMIT = 7200


class AgentAction(C.Structure):
    _fields_ = [('angle', C.c_uint8), ('reserved', C.c_uint8), ('buttons', C.c_uint16)]


def agent_lib():
    lib = C.CDLL(str(AGENT_LIB))
    lib.bopl_agent_open.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_char_p, C.c_size_t]
    lib.bopl_agent_open.restype = C.c_void_p
    lib.bopl_agent_reset.argtypes = [C.c_void_p, C.c_uint32]
    lib.bopl_agent_tick.argtypes = [C.c_void_p, C.POINTER(hr.State), C.POINTER(AgentAction)]
    lib.bopl_agent_tick.restype = C.c_int
    lib.bopl_agent_close.argtypes = [C.c_void_p]
    lib.bopl_agent_id.argtypes = [C.c_void_p]
    lib.bopl_agent_id.restype = C.c_char_p
    lib.bopl_agent_ability.argtypes = [C.c_void_p, C.c_int]
    lib.bopl_agent_ability.restype = C.c_int
    lib.bopl_agent_obs_size.restype = C.c_int
    return lib


class Shim:
    def __init__(self, lib, spec, player):
        error = C.create_string_buffer(1024)
        self.lib = lib
        self.handle = lib.bopl_agent_open(os.fsencode(str(spec)), 2, player, error, 1024)
        if not self.handle:
            raise ValueError(error.value.decode())
        self.id = lib.bopl_agent_id(self.handle).decode()
        self.abilities = [lib.bopl_agent_ability(self.handle, s) for s in range(3)]
        self.player = player

    def reset(self, seed):
        self.lib.bopl_agent_reset(self.handle, seed)

    def tick(self, state):
        out = AgentAction()
        status = self.lib.bopl_agent_tick(self.handle, C.byref(state), C.byref(out))
        return status, (out.angle, out.buttons)

    def close(self):
        if self.handle:
            self.lib.bopl_agent_close(self.handle)
            self.handle = None


class GameWorld:
    """One Unity process (v2 protocol) reused across rounds."""

    def __init__(self, lib, log):
        self.lib = lib
        error = C.create_string_buffer(512)
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        self.handle = lib.bopl_rl_open(os.fsencode(hr.UNITY), os.fsencode(ROOT / 'native/fixtures/movement.rep'), os.fsencode(log), 60000, error, 512)
        if not self.handle:
            raise RuntimeError(error.value.decode())
        self.source = lib.bopl_rl_source_hash(self.handle).decode()

    def reset(self, seed, level, abilities):
        cfg = hr.Config()
        cfg.seed, cfg.map, cfg.players, cfg.slots = seed, level, 2, 3
        # the shim requires header.episode_ticks == 0 on its first call, so the game runs the warmup itself
        cfg.stocks, cfg.mode, cfg.warmup_ticks, cfg.max_episode_ticks = 0, 0, WARMUP, LIMIT + 600
        for i in range(2):
            cfg.player[i] = hr.PlayerConfig(i, i, 0, (C.c_uint32 * 3)(*abilities[i]))
        if self.lib.bopl_rl_reset(self.handle, C.byref(cfg)):
            raise RuntimeError(self.lib.bopl_rl_error(self.handle).decode())

    @property
    def state(self):
        return self.lib.bopl_rl_state(self.handle).contents

    def step(self, actions):
        acts = (hr.Action * 2)(*(hr.Action(int(a), 0, int(b)) for a, b in actions))
        status = self.lib.bopl_rl_step(self.handle, acts, 2, 1)
        if status not in (0, 4):
            raise RuntimeError(f'step status {status}: {self.lib.bopl_rl_error(self.handle).decode()}')

    def close(self):
        if self.handle:
            self.lib.bopl_rl_close(self.handle)
            self.handle = None


def play_round(world, opponent, agent, seed, level):
    world.reset(seed, level, [opponent.abilities, agent.abilities])
    opponent.reset(seed)
    agent.reset(seed)
    first = hr.frame(world.state)
    assert first['t'] == WARMUP, first['t']
    # the taxonomy indexes frames by level tick: pad the warmup the game ran internally
    frames = [dict(first, t=t) for t in range(WARMUP)] + [first]
    ended = timeout = False
    packets = [(0, 0), (0, 0)]
    for t in range(LIMIT):
        s0, packets[0] = opponent.tick(world.state)
        s1, packets[1] = agent.tick(world.state)
        if s0 == 1 or s1 == 1:
            ended = True
            break
        world.step(packets)
        frames.append(hr.frame(world.state, [list(p) for p in packets]))
        if world.state.header.flags & 1:
            ended = True
            break
    else:
        timeout = True
    return frames, ended, timeout


def _worker(args):
    out, agents, opponents, tasks, index = args
    lib = hr.load_lib()
    alib = agent_lib()
    world = GameWorld(lib, out / 'workers' / f'w{index}.log')
    shims = {}
    results = []
    for task in tasks:
        name = task['name']
        started = time.monotonic()
        try:
            if ('agent', task['agent']) not in shims:
                shims[('agent', task['agent'])] = Shim(alib, agents[task['agent']], 1)
            if ('opp', task['opponent']) not in shims:
                shims[('opp', task['opponent'])] = Shim(alib, opponents[task['opponent']], 0)
            agent, opp = shims[('agent', task['agent'])], shims[('opp', task['opponent'])]
            frames, ended, timeout = play_round(world, opp, agent, task['seed'], task['map'])
            final = frames[-1]
            winner = next((i for i, p in enumerate(final['players']) if p[1] & 2), None)
            outcome = 'draw' if timeout or winner is None else ('a' if winner == 0 else 'b')
            row = dict(a=opp.id, b=agent.id, agent=agents[task['agent']], agent_name=task['agent'], opponent=task['opponent'], opponent_id=opp.id,
                       map=task['map'], seed=task['seed'], side=0, ticks=final['t'] - WARMUP, level_ticks=final['t'], outcome=outcome, timeout=timeout,
                       human_abilities=[opp.abilities], agent_abilities=agent.abilities, session='unity-' + task['agent'], round=0, time='',
                       replay=name + '.rep', players=[dict(id=p[0], cause_of_death=p[2], deaths=p[4], kills=p[3]) for p in final['players']],
                       stocks=0, game_mode=0, roster=None)
            check = dict(ended=ended, ended_at=final['t'], recorded_level_ticks=final['t'], ticks_match=True, outcome=outcome, outcome_match=True,
                         causes=[p[2] for p in final['players']], causes_match=True, alive=[p[1] & 1 for p in final['players']], replay_ticks=final['t'])
            meta = dict(seed=task['seed'], level=task['map'], mode=0, stocks=0, ticks=final['t'], source=world.source,
                        players=[dict(id=1, color=0, team=0, keyboard=False, abilities=opp.abilities), dict(id=2, color=1, team=1, keyboard=False, abilities=agent.abilities)])
            with gzip.open(out / 'games' / (name + '.json.gz'), 'wt', compresslevel=4) as handle:
                json.dump(dict(schema=1, name=name, row=row, meta=meta, keys=hr.FRAME_KEYS, check=check, frames=frames), handle, separators=(',', ':'))
            result = dict(name=name, agent=task['agent'], opponent=task['opponent'], map=task['map'], seed=task['seed'], outcome=outcome, timeout=timeout,
                          ticks=final['t'] - WARMUP, causes=check['causes'], seconds=round(time.monotonic() - started, 2))
        except Exception as error:
            result = dict(name=name, error=repr(error), seconds=round(time.monotonic() - started, 2))
        results.append(result)
        print(json.dumps(result), flush=True)
    for s in shims.values():
        s.close()
    world.close()
    return results


def play(args):
    out = args.out.resolve()
    (out / 'games').mkdir(parents=True, exist_ok=True)
    (out / 'workers').mkdir(exist_ok=True)
    agents = dict(item.split('=', 1) for item in args.agent)
    opponents = {}
    for spec in args.opponent:
        name, _, path = spec.partition('=')
        opponents[name] = path or name
    maps = [int(m) for m in args.maps.split(',')]
    tasks = []
    for a in agents:
        for o in opponents:
            for m in maps:
                for k in range(args.seeds):
                    seed = args.seed_base + 1000 * m + k
                    tasks.append(dict(name=f'{a}-vs-{o}-m{m}-s{k}', agent=a, opponent=o, map=m, seed=seed))
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
    todo = [t for t in tasks if t['name'] not in done]
    print(f'{len(tasks)} rounds, {len(todo)} to play, {args.workers} workers', flush=True)
    hr.build_v2()
    # group by agent+opponent so each worker opens few shims
    todo.sort(key=lambda t: (t['agent'], t['opponent'], t['map'], t['seed']))
    buckets = [todo[i::args.workers] for i in range(args.workers)]
    ctx = mp.get_context('spawn')
    started = time.monotonic()
    with ctx.Pool(args.workers) as pool, manifest.open('a') as handle:
        for results in pool.imap_unordered(_worker, [(out, agents, opponents, b, i) for i, b in enumerate(buckets) if b]):
            for r in results:
                handle.write(json.dumps(r) + '\n')
            handle.flush()
    print(f'done in {time.monotonic() - started:.0f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('play')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--agent', action='append', default=[], metavar='NAME=PATH')
    p.add_argument('--opponent', action='append', default=[], metavar='NAME[=PATH]', help='bot name or NAME=checkpoint path')
    p.add_argument('--maps', default='0,1,2,3,4')
    p.add_argument('--seeds', type=int, default=4)
    p.add_argument('--seed-base', type=int, default=610001)
    p.add_argument('--workers', type=int, default=4)
    p.set_defaults(func=play)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
