"""How aggressively each bot plays: the same opponents, the same kit, per-bot movement and attack rates.

    venv/bin/python eval/aggression.py --subject f5=F5.bin --subject v4=V4.bin \
        --opponent v4=V4.bin --opponent mixed-v2-hard --kit 2,4,14 --pairs 100 --out DIR

Every subject plays every opponent on the same seeds, maps and kit (mirror, side-swapped, as
league/combo_set.py schedules them), so the subjects' rows differ only by the subject. Per game
it records, for the subject's seat, from the research state after every decision:

  distance    mean distance between the two bodies while both are alive (game units)
  closing     share of decisions where the subject's own movement takes it toward the
              opponent by more than CLOSE units; retreating the share moving away
  actions     Dash / Bow / Platform activations, arrows fired and arrow hits per minute
              (eval/telemetry.py), kills per minute
  first_kill  seconds to the subject's first kill (games without one count the game length)

and the outcome. The summary averages per game with bootstrap 95% intervals over seed pairs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'league'), str(HERE.parent / 'native/tools')]
import registry  # noqa: E402
import games as league_games  # noqa: E402
import elo_ladder  # noqa: E402
import map_catalogue  # noqa: E402
from combo_set import tasks as combo_tasks  # noqa: E402
from evaluation_protocol import DEFAULT_LOADOUT  # noqa: E402
from telemetry import FIELDS  # noqa: E402

FIX = 4294967296.
CLOSE = 0.05   # game units per decision: below this a move counts as neither closing nor retreating
COUNTERS = ('dash_activations', 'bow_activations', 'platform_activations', 'arrows_fired', 'arrow_hits')
METRICS = ('win', 'seconds', 'distance', 'closing', 'retreating', 'kills_pm', 'first_kill') + \
    tuple(c + '_pm' for c in COUNTERS)


def participant(spec, name):
    if spec.endswith('.bin'):
        return elo_ladder.Model(name, Path(spec).resolve(), 'candidate')
    return elo_ladder.Scripted(spec)


def play_one(env, built, task, kit_row):
    """One game; the subject is 'A' on task['side']. Returns the subject's metrics."""
    env.configure_loadouts(kit_row)
    env.reset(task['seed'], task['map'])
    rng = np.random.default_rng((task['seed'], task['map'], task['side'], 0))
    env.bot(0, 0); env.bot(1, 0)
    me, them = task['side'], 1 - task['side']
    seats = {me: 'A', them: 'B'}
    for side, name in seats.items():
        attach = getattr(built[name], 'attach', None)
        if attach:
            attach(env, side)
    actors = {side: built[name].start(task, side, rng) for side, name in seats.items()}
    counts = np.zeros(len(FIELDS))
    distances, closing, retreating, decisions = [], 0, 0, 0
    first_kill = None
    kills0 = None
    while not env.ended:
        s = env.state
        p, q = s.players[me], s.players[them]
        before = (p.x / FIX, p.y / FIX, q.x / FIX, q.y / FIX, bool(p.flags & 1 and q.flags & 1))
        if kills0 is None:
            kills0 = p.kills
        observation = env.rows.copy(); mask = env.mask.copy()
        env.step(np.stack([actors[0](observation, mask), actors[1](observation, mask)]))
        counts += np.asarray(env.telemetry[me], dtype=np.float64)
        s = env.state
        p = s.players[me]
        if first_kill is None and p.kills > kills0:
            first_kill = s.header.episode_ticks / 60
        px, py, qx, qy, alive = before
        if alive and p.flags & 1:
            gap = math.hypot(qx - px, qy - py)
            distances.append(gap)
            decisions += 1
            if gap > 1e-6:
                toward = ((p.x / FIX - px) * (qx - px) + (p.y / FIX - py) * (qy - py)) / gap
                closing += toward > CLOSE
                retreating += toward < -CLOSE
    totals = dict(zip(FIELDS, counts.tolist()))
    final = env.state
    seconds = final.header.episode_ticks / 60
    timeout = bool(final.header.flags & 2)
    won = [bool(final.players[i].flags & 2) for i in (0, 1)]
    outcome = 'draw' if timeout or won[me] == won[them] else ('a' if won[me] else 'b')
    kills = final.players[me].kills - kills0 - totals['deaths_self']
    minutes = seconds / 60 if seconds else 1
    row = dict(win=1.0 if outcome == 'a' else 0.5 if outcome == 'draw' else 0.0, seconds=seconds,
               distance=float(np.mean(distances)) if distances else None,
               closing=closing / decisions if decisions else None,
               retreating=retreating / decisions if decisions else None,
               kills_pm=kills / minutes, first_kill=first_kill if first_kill is not None else seconds)
    for c in COUNTERS:
        row[c + '_pm'] = totals[c] / minutes
    return row


def run(built, todo, workers, seed_base):
    from environment import Environment
    rows = sorted({tuple(t['kits']) for t in todo})
    loadouts = dict(mode='set', fixed=list(DEFAULT_LOADOUT), pool=[], lineups=[list(r) for r in rows],
                    set_file=None, no_dup=False, mirror_share=0.0)

    def bucket_run(bucket):
        out = []
        with Environment(seed=seed_base + len(todo), limit=registry.EPISODE_TICKS, delay=registry.DELAY,
                         loadouts=loadouts) as env:
            for task in bucket:
                row = play_one(env, built, task, dict(loadouts, lineups=[list(task['kits'])]))
                out.append(dict(seed=task['seed'], side=task['side'], map=task['map'], **row))
        return out

    buckets = [todo[i::workers] for i in range(min(workers, len(todo)))]
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        return [g for b in pool.map(bucket_run, buckets) for g in b]


def summarize(games, rng):
    """{metric: [mean, low, high]} bootstrapped over seed pairs."""
    by_pair = {}
    for g in games:
        by_pair.setdefault(g['seed'], []).append(g)
    pairs = list(by_pair.values())
    out = {}
    for m in METRICS:
        per = np.array([np.mean([g[m] for g in p if g[m] is not None]) for p in pairs
                        if any(g[m] is not None for g in p)])
        boot = [rng.choice(per, per.size).mean() for _ in range(2000)]
        out[m] = [round(float(per.mean()), 4), round(float(np.percentile(boot, 2.5)), 4),
                  round(float(np.percentile(boot, 97.5)), 4)]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subject', action='append', required=True, help='name=checkpoint.bin')
    p.add_argument('--opponent', action='append', required=True, help='name=checkpoint.bin, or a scripted bot name')
    p.add_argument('--kit', default='2,4,14')
    p.add_argument('--pairs', type=int, default=100)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--seed-base', type=int, default=181001)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--temperature', type=float, nargs='+',
                   help="subjects' sampling temperature: one value for all heads or aim buttons hold (default 1)")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    league_games.configure(a.out / 'workers')
    kit = tuple(int(v) for v in a.kit.split(','))
    todo = [{k: v for k, v in t.items() if k != 'pair_kits'}
            for t in combo_tasks([kit], a.pairs, a.seed_base, map_catalogue.parse_map_ids(registry.MAPS))]
    split = lambda s: s.split('=', 1) if '=' in s else (s, s)
    opponents = [split(o) for o in a.opponent]
    summary = {}
    for sname, spath in (split(s) for s in a.subject):
        subject = participant(spath, 'A')
        if a.temperature:
            subject.temperature = tuple(a.temperature * 3 if len(a.temperature) == 1 else a.temperature)
        for oname, ospec in opponents:
            games = run({'A': subject, 'B': participant(ospec, 'B')}, todo, a.workers, a.seed_base)
            with open(a.out / 'games.jsonl', 'a') as f:
                for g in games:
                    f.write(json.dumps(dict(subject=sname, opponent=oname, kit=list(kit), **g)) + '\n')
            summary.setdefault(sname, {})[oname] = summarize(games, np.random.default_rng(0))
            row = summary[sname][oname]
            print(f'{sname:>8} vs {oname:<14} ' + '  '.join(f'{m} {row[m][0]:.3g}' for m in METRICS), flush=True)
        everyone = [json.loads(l) for l in open(a.out / 'games.jsonl') if json.loads(l)['subject'] == sname]
        summary[sname]['all'] = summarize(everyone, np.random.default_rng(0))
    (a.out / 'summary.json').write_text(json.dumps(dict(kit=list(kit), pairs=a.pairs, seed_base=a.seed_base, temperature=a.temperature,
                                                        close=CLOSE, summary=summary), indent=1))


if __name__ == '__main__':
    main()
