"""Replay the same evaluation N times on one backend and count how often it disagrees.

sim/PORTING.md carries an open TODO: a determinism flake seen once in ~17 runs of
sim/Oracle/checkpoint.sh, suspected to be a `Dictionary<string,_>` iteration-order
leak (.NET randomizes string hash seeds per process). SIM_OVERHAUL.md:229 later
guessed it was "likely a socket-path artefact". Both statements rest on a single
observation that 16 retries failed to reproduce, so nobody knows the rate on the
in-process path the RL evaluation actually uses.

That matters for any oracle-hosted measurement: if the backend is not reproducible
across processes, per-game comparisons are noise and only large-N averages mean
anything. This measures it. Each repeat is a SEPARATE evaluate.py process -- which
is the unit the hypothesis is about, since a string hash seed is drawn once per
process -- playing the identical schedule, and every repeat is compared to the first
on win/timeout/kills/deaths/ticks.

    python backend_determinism.py --game LIB.so --kit KIT --repeats 16 [--games 2]

Run it on the Rust sim too: that arm must come back perfectly identical, and if it
does not, the harness is nondeterministic and the oracle numbers mean nothing. A
repeat count and a shared-game count are printed so a vacuous run cannot look clean.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_transfer import MATCH_FIELDS, evaluate, final_checkpoint, read_games

SCRIPTED = ['mixed-hard', 'mixed-v2-hard', 'archer-v2-hard', 'pursuer-v2-hard', 'evasive-v2-hard']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kit', default='dash-bow-platform')
    p.add_argument('--repeats', type=int, default=16)
    p.add_argument('--games', type=int, default=2)
    p.add_argument('--heldout-games', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--label', default='backend')
    a = p.parse_args()

    out = a.out.resolve()
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    runs = []
    for i in range(a.repeats):
        started = time.monotonic()
        result = evaluate(out / 'repeats' / f'{i:02d}', a.kit, a.game, SCRIPTED,
                          a.games, a.heldout_games, a.workers,
                          out / 'logs' / f'repeat-{i:02d}.log')
        runs.append(read_games(result))
        print(f'repeat {i:02d}: {len(runs[-1])} games in {time.monotonic()-started:.1f}s', flush=True)

    reference = runs[0]
    if not reference:
        raise SystemExit('no games played: this measurement is vacuous')
    rows, per_key = [], {}
    for i, games in enumerate(runs[1:], start=1):
        shared = sorted(set(reference) & set(games))
        if len(shared) != len(reference):
            raise SystemExit(f'repeat {i} played a different schedule ({len(shared)} shared)')
        bad = [k for k in shared if any(reference[k][f] != games[k][f] for f in MATCH_FIELDS)]
        for k in bad:
            per_key.setdefault(k, set()).add(i)
        rows.append(dict(repeat=i, shared=len(shared), differing_games=len(bad),
                         examples=[list(k) for k in bad[:3]]))
        print(f'vs repeat 00: repeat {i:02d} differs on {len(bad):4d}/{len(shared)} games', flush=True)

    # Distinct outcome signatures per game, over all repeats: 1 means fully reproducible.
    distinct = {}
    for k in reference:
        distinct[k] = len({tuple(g[k][f] for f in MATCH_FIELDS) for g in runs})
    flaky = {k: v for k, v in distinct.items() if v > 1}
    processes = sum(1 for r in rows if r['differing_games'])
    summary = dict(label=a.label, game=str(Path(a.game).resolve()), kit=a.kit,
                   repeats=a.repeats, games_per_repeat=len(reference),
                   processes_that_differed=processes,
                   game_slots_ever_flaky=len(flaky),
                   max_distinct_outcomes_for_one_game=max(distinct.values()),
                   comparisons=rows)
    (out / 'determinism.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(f"\n{a.label}: {a.repeats} processes x {len(reference)} games "
          f"({(a.repeats-1)*len(reference)} paired comparisons)")
    print(f"  processes disagreeing with the first: {processes}/{a.repeats-1}")
    print(f"  game slots that ever varied:          {len(flaky)}/{len(reference)}")
    print(f"  max distinct outcomes for one game:   {max(distinct.values())}")
    print('  VERDICT: reproducible across processes' if not flaky else
          '  VERDICT: NOT reproducible across processes')


if __name__ == '__main__':
    main()
