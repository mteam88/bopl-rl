"""Does the terminal state still carry the episode's telemetry counters?

bopl_rl.h describes BoplRlTelemetry as "episode-cumulative event counters,
excluding warmup". evaluate.py relies on that at exactly one moment -- the final
frame -- to charge a player's suicides back out of its Kills:

    kills = me['kills'] - final['telemetry'][side]['deaths_self']

(the game credits a suicide to the victim's own Kills). If a backend clears the
counters when the episode ends, that subtraction silently becomes a no-op and the
player keeps credit for killing itself.

This steps real episodes to termination on each backend and compares the counters
in the terminal frame against the last frame before it. Nothing is inferred from
stored results: both backends are driven here, in this process, through the same
research API the evaluation uses.

    python check_terminal_telemetry.py <lib-a.so> [<lib-b.so> ...] [--episodes N]

Reported per backend: how many finished episodes had a nonzero counter on the last
pre-terminal frame, and how many of those still had it on the terminal frame. An
episode that never accumulated anything proves nothing and is counted separately,
so a vacuous run cannot read as a pass.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FIELDS = ('deaths_self', 'deaths_opponent', 'deaths_environment', 'arrows_fired',
          'dash_activations', 'bow_activations', 'platform_activations')


def probe(game, episodes, seed0):
    import numpy as np
    import environment
    from telemetry import EVENT_FIELDS, FIELDS as TELEMETRY_FIELDS
    environment.configure(ROOT / 'native/artifacts/rl/logs/terminal-telemetry', game)
    from environment import Environment
    from layout import ACTION_SIZES

    rows = []
    rng = np.random.default_rng(7)
    with Environment(seed=seed0, map_id=0, limit=3600) as env:
        for episode in range(episodes):
            env.reset(seed0 + episode, map_id=episode % 3)
            env.bot(1, 4, 2)  # a scripted opponent, so episodes actually end
            previous = None
            # evaluate.py sums research_telemetry over the episode's decisions
            # (events -> summarize -> behavior.totals); the ABI struct is already
            # cumulative. Accumulate the same way, or the two are not comparable.
            accumulator = np.zeros((2, len(TELEMETRY_FIELDS)))
            while not env.ended:
                # Discrete heads (aim 256, buttons 16, hold 5), sampled inside the
                # live action mask so every decision is one the trainer could make.
                mask = env.mask
                actions = np.zeros((2, len(ACTION_SIZES)), dtype=np.float32)
                for player in range(2):
                    offset = 0
                    for head, size in enumerate(ACTION_SIZES):
                        allowed = np.flatnonzero(mask[player, offset:offset + size])
                        actions[player, head] = rng.choice(allowed) if len(allowed) else 0
                        offset += size
                before = env.frame()
                env.step(actions)
                accumulator += env.telemetry[:2]
                previous = before
            final = env.frame()
            if previous is None:
                continue
            last = {f: sum(_total(p[f]) for p in previous['telemetry']) for f in FIELDS}
            end = {f: sum(_total(p[f]) for p in final['telemetry']) for f in FIELDS}
            # The two telemetry sources evaluate.py mixes in one expression:
            # research_telemetry (its own per-decision accumulator, which feeds
            # behavior.totals) and BoplRlState.telemetry (the backend's own ABI
            # struct, which the kills correction subtracts). They are supposed to
            # be the same counters.
            abi = {f: [int(p[f]) for p in final['telemetry'][:2]] for f in FIELDS}
            research = {f: [int(round(accumulator[i][TELEMETRY_FIELDS.index(f)])) for i in range(2)]
                        for f in FIELDS}
            rows.append(dict(episode=episode, ticks=final['t'], last=last, terminal=end,
                             abi=abi, research=research))
    return rows


def _total(value):
    return sum(value) if isinstance(value, (list, tuple)) else value


def summarise(game, rows):
    accumulated = [r for r in rows if any(r['last'][f] for f in FIELDS)]
    kept = [r for r in accumulated if all(r['terminal'][f] >= r['last'][f] for f in FIELDS)]
    cleared = [r for r in accumulated if all(r['terminal'][f] == 0 for f in FIELDS)]
    print(f"\n{Path(game).name}")
    print(f"  episodes run                         {len(rows)}")
    print(f"  episodes that accumulated counters   {len(accumulated)}  (the only informative ones)")
    print(f"  terminal frame KEPT the counters     {len(kept)}")
    print(f"  terminal frame CLEARED them to zero  {len(cleared)}")
    print(f"  {'counter':22s} {'pre-term':>9s} {'terminal':>9s} | "
          f"{'ABI struct':>12s} {'research acc':>13s}  agree?")
    disagree = []
    for f in FIELDS:
        a = sum(r['last'][f] for r in accumulated)
        b = sum(r['terminal'][f] for r in accumulated)
        abi = sum(sum(r['abi'][f]) for r in accumulated)
        res = sum(sum(r['research'][f]) for r in accumulated)
        if abi != res:
            disagree.append(f)
        print(f"    {f:22s} {a:9d} {b:9d} | {abi:12d} {res:13d}  {'yes' if abi == res else 'NO'}")
    if disagree:
        print(f"  !! the two telemetry sources disagree on: {', '.join(disagree)}")
    return dict(game=str(game), episodes=len(rows), informative=len(accumulated),
                kept=len(kept), cleared=len(cleared), sources_disagree=disagree,
                totals={f: [sum(r['last'][f] for r in accumulated),
                            sum(r['terminal'][f] for r in accumulated)] for f in FIELDS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('games', nargs='+')
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--seed', type=int, default=90001)
    a = p.parse_args()
    if len(a.games) > 1:
        out = []
        for game in a.games:
            # One NativeAOT runtime per process: ask a child about each backend.
            r = subprocess.run([sys.executable, __file__, game, '--episodes', str(a.episodes),
                                '--seed', str(a.seed)], cwd=str(HERE), text=True, capture_output=True)
            print(r.stdout, end='')
            if r.returncode:
                raise SystemExit(r.stderr)
            out.append(r.stdout)
        return
    rows = probe(a.games[0], a.episodes, a.seed)
    summary = summarise(a.games[0], rows)
    if not summary['informative']:
        raise SystemExit('no episode accumulated any counter: this check proves nothing')


if __name__ == '__main__':
    main()
