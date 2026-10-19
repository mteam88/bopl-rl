"""Decision-time search participant: a checkpoint plus a policy+value guided lookahead
on forks of the live game (train/bopl_search.h, built into libbopl_research.so).

    Search(name, path, kind, k=8, n=2, depth=8, halving=False)

is a drop-in for elo_ladder.Model in league/games.py tasks: attach() sets the row
layout as Model does and remembers the environment, start() opens (once per
environment) a C search handle that reads the environment's rows and masks
directly, and act() returns the searched action. Both players' rows must be
layout v4, because the search models the opponent with the same checkpoint on
the opponent's real row.

    venv/bin/python eval/search_policy.py --bench policy.bin

times the primitives (fork, decision step, policy forward at batch 2 and larger,
full shadow decision) and a few searched decisions at some budgets.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import environment  # noqa: E402
from environment import LIB  # noqa: E402
import elo_ladder  # noqa: E402
import layout  # noqa: E402
from native_policy import V4_SHAPE_KEYS, writer_ablation  # noqa: E402

LIB.research_search_open.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_long, C.c_int, C.c_int, C.c_int, C.c_int,
                                     C.POINTER(C.c_int), C.c_int, C.c_uint64]
LIB.research_search_open.restype = C.c_void_p
LIB.research_search_configure.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_float, C.c_float, C.c_float,
                                          C.c_float, C.c_float]
LIB.research_search_configure.restype = C.c_int
LIB.research_search_reset.argtypes = [C.c_void_p, C.c_uint64]
LIB.research_search_act.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int)]
LIB.research_search_act.restype = C.c_int
LIB.research_search_stats.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int]
LIB.research_search_stats.restype = C.c_int
LIB.research_search_stat_width.restype = C.c_int
LIB.research_search_counters.argtypes = [C.c_void_p, C.POINTER(C.c_double)]
LIB.research_search_error.argtypes = [C.c_void_p]
LIB.research_search_error.restype = C.c_char_p
LIB.research_search_bench.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]
LIB.research_search_bench.restype = C.c_double
LIB.research_search_close.argtypes = [C.c_void_p]

STAT_WIDTH = LIB.research_search_stat_width()
COUNTERS = ('decisions', 'searched', 'rollouts', 'steps', 'forward_rows', 'seconds', 'root_value_0', 'root_value_1')
TRAINING_KILL = 0.2883693237829292


def parse_budget(text):
    """'k,n,depth[,h][,mix=M][,vs=V][,latched=0]' -> Search keyword arguments."""
    parts = text.split(',')
    out = dict(k=int(parts[0]), n=int(parts[1]), depth=int(parts[2]))
    for part in parts[3:]:
        if part == 'h':
            out['halving'] = True
        elif part.startswith('mix='):
            out['prior_mix'] = float(part[4:])
        elif part.startswith('vs='):
            out['value_scale'] = float(part[3:])
        elif part.startswith('latched='):
            out['search_latched'] = bool(int(part[8:]))
        else:
            raise ValueError(f'unknown search budget part {part!r}')
    return out


class Handle:
    """A C search handle over one environment and one checkpoint."""
    def __init__(self, env, policy, rollouts, seed=1):
        if policy.architecture != layout.V4_ARCHITECTURE:
            raise ValueError('The search needs a v4 checkpoint')
        shape = (C.c_int * 8)(*[int(policy.v4_shape[key]) for key in V4_SHAPE_KEYS])
        weights = np.ascontiguousarray(policy.weights, dtype=np.float32)
        self.ptr = LIB.research_search_open(env.handle, weights.ctypes.data_as(C.POINTER(C.c_float)), weights.size,
                                            policy.hidden, policy.num_layers, policy.aim_fourier, policy.aim_ar,
                                            shape, rollouts, seed)
        if not self.ptr:
            raise RuntimeError('Could not open the search')
        self.env = env
        self.out = (C.c_int * 3)()
        self.searched = C.c_int()
        self._counters = (C.c_double * len(COUNTERS))()

    def configure(self, k, n, depth, halving=False, search_latched=True, gamma=0.995, kill=TRAINING_KILL, timeout=-1.0,
                  prior_mix=0.0, value_scale=1.0):
        """The chosen candidate maximises value_scale*mean return + prior_mix*(prior logp + Gumbel):
        prior_mix 0 is the best mean, prior_mix 1 Gumbel-top-k's improved policy."""
        if LIB.research_search_configure(self.ptr, k, n, depth, int(halving), int(search_latched), gamma, kill, timeout,
                                         prior_mix, value_scale):
            raise ValueError('Bad search budget (k*n must fit the rollouts the handle was opened with)')

    def reset(self, seed):
        LIB.research_search_reset(self.ptr, (int(seed) & ((1 << 64) - 1)) or 1)

    def act(self, side):
        if LIB.research_search_act(self.ptr, side, self.out, self.searched):
            raise RuntimeError(LIB.research_search_error(self.ptr).decode())
        return np.asarray(self.out[:], dtype=np.int64)

    def stats(self):
        """Per-candidate rows: aim, buttons, hold, visits, mean return, prior logp, chosen, return variance."""
        out = (C.c_float * (64 * STAT_WIDTH))()
        count = LIB.research_search_stats(self.ptr, out, 64 * STAT_WIDTH)
        return np.asarray(out[:count * STAT_WIDTH], dtype=np.float32).reshape(count, STAT_WIDTH)

    def counters(self):
        LIB.research_search_counters(self.ptr, self._counters)
        return dict(zip(COUNTERS, self._counters[:]))

    def bench(self, what, iterations, batch=1):
        return LIB.research_search_bench(self.ptr, what, iterations, batch)

    def close(self):
        if self.ptr:
            LIB.research_search_close(self.ptr)
            self.ptr = None


class Search(elo_ladder.Model):
    """A checkpoint played through the search. Timing is accumulated over games in
    self.totals (decisions, searched, rollouts, steps, forward rows, seconds); call
    finish() after the last game to harvest the open handles, then summary()."""
    needs_value = True

    def __init__(self, name, path, kind, k=8, n=2, depth=8, halving=False, search_latched=True,
                 gamma=0.995, kill=TRAINING_KILL, timeout=-1.0, prior_mix=0.0, value_scale=1.0, step=None, library=None):
        super().__init__(name, path, kind, step=step, library=library)
        self.budget = dict(k=int(k), n=int(n), depth=int(depth), halving=bool(halving), search_latched=bool(search_latched),
                           gamma=float(gamma), kill=float(kill), timeout=float(timeout),
                           prior_mix=float(prior_mix), value_scale=float(value_scale))
        self.local = threading.local()
        self.lock = threading.Lock()
        self.totals = {key: 0.0 for key in COUNTERS[:6]}
        self.games = 0
        self.open = []

    def attach(self, env, side):
        super().attach(env, side)
        # play_games runs one environment per worker thread; start() finds it here.
        self.local.env = env

    def _harvest(self, handle):
        counters = handle.counters()
        if counters['decisions']:
            with self.lock:
                for key in self.totals:
                    self.totals[key] += counters[key]
                self.games += 1

    def handle_for(self, env):
        handle = getattr(self.local, 'handle', None)
        if handle is None or handle.env is not env:
            if handle is not None:
                self._harvest(handle); handle.close()
            handle = Handle(env, self.policy, self.budget['k'] * self.budget['n'])
            handle.configure(**self.budget)
            self.local.handle = handle
            with self.lock:
                self.open.append(handle)
        return handle

    def start(self, task, side, rng):
        env = getattr(self.local, 'env', None)
        if env is None:
            raise RuntimeError('Search.start before attach')
        handle = self.handle_for(env)
        self._harvest(handle)
        handle.reset(int(rng.integers(1, 2 ** 62)))
        return lambda observation, mask: handle.act(side)

    def finish(self):
        with self.lock:
            handles, self.open = self.open, []
        for handle in handles:
            if handle.ptr:
                self._harvest(handle); handle.close()

    def summary(self):
        t = self.totals
        return dict(budget=self.budget, games=self.games, decisions=t['decisions'], searched=t['searched'],
                    ms_per_decision=1000 * t['seconds'] / max(t['decisions'], 1),
                    rollouts_per_searched=t['rollouts'] / max(t['searched'], 1),
                    forward_rows=t['forward_rows'], seconds=t['seconds'])


def bench(path, budgets, iterations=200, workspace='/tmp/bopl-search-bench'):
    from native_policy import load
    sys.path.insert(0, str(HERE.parent / 'league'))
    import games as league_games
    league_games.configure(workspace)
    policy, meta = load(Path(path))
    env = environment.Environment(seed=5, map_id=3, limit=10800, delay=9)
    for side in (0, 1):
        env.set_layout(side, 4, ablate=writer_ablation(policy))
    handle = Handle(env, policy, rollouts=512)
    handle.configure(k=1, n=1, depth=1)
    # Play 40 raw decisions first so the rows are a mid-game state.
    for _ in range(40):
        env.step(np.stack([handle.act(0), handle.act(1)]))
    out = {}
    out['fork_us'] = 1e6 * handle.bench(0, iterations)
    out['decision_step_us'] = 1e6 * handle.bench(1, iterations)
    out['forward_batch2_us_per_row'] = 1e6 * handle.bench(2, iterations) / 2
    for batch in (1, 4, 16, 64):
        out[f'forward_batch{batch}_us_per_row'] = 1e6 * handle.bench(4, max(iterations // batch, 5), batch)
    out['shadow_decision_us'] = 1e6 * handle.bench(3, iterations)
    for k, n, depth, halving in budgets:
        handle.configure(k=k, n=n, depth=depth, halving=halving)
        handle.reset(7)
        t0 = time.perf_counter()
        decisions = 0
        for _ in range(30):
            env.step(np.stack([handle.act(0), handle.act(1)]))
            decisions += 2
            if env.ended:
                break
        c = handle.counters()
        out[f'k{k}n{n}d{depth}{"h" if halving else ""}'] = dict(
            ms_per_decision=1000 * (time.perf_counter() - t0) / decisions, ms_in_c=1000 * c['seconds'] / max(c['decisions'], 1),
            rollouts_per_searched=c['rollouts'] / max(c['searched'], 1), searched=c['searched'], decisions=c['decisions'],
            steps=c['steps'], forward_rows=c['forward_rows'], stats=handle.stats()[:4].round(3).tolist())
    handle.close(); env.close()
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bench', type=Path, help='checkpoint to time')
    p.add_argument('--iterations', type=int, default=200)
    a = p.parse_args()
    if a.bench:
        budgets = [(4, 1, 4, False), (8, 1, 8, False), (8, 2, 8, False), (16, 1, 8, True), (16, 2, 8, True)]
        print(json.dumps(bench(a.bench, budgets, a.iterations), indent=1))
