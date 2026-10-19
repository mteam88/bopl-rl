"""How aggressively each bot plays: the same opponents, the same kit, per-bot movement and attack rates.

    venv/bin/python eval/aggression.py --subject f5=F5.bin --subject v4=V4.bin \
        --opponent v4=V4.bin --opponent mixed-v2-hard --kit 2,4,14 --pairs 100 --out DIR

Every subject plays every opponent on the same seeds, maps and kit (mirror, side-swapped, as
league/combo_set.py schedules them), so the subjects' rows differ only by the subject. Per game
it records, for the subject's seat, from the research state after every decision:

  distance    mean distance between the two bodies while both are alive (game units)
  closing     share of decisions where the subject's own movement takes it toward the
              opponent by more than CLOSE units; retreating the share moving away
  actions     Dash / Bow / Platform activations and arrows fired per minute (eval/telemetry.py);
              arrow_kills: RlTelemetry.ArrowHit, which counts only an arrow that kills
              (PlayerCollision.cs:455), so in a Bow-only kit it equals the credited kills
  kills       credited kills (eval/kill_credit.py: the game's killer, or the caster of a
              Missile blast or cast Spike, as v4.2 training credits them); kill_win 1 if the
              subject won by a credited kill; alive_s the seconds both players were alive
  first_kill  seconds to the subject's first credited kill, None without one (censored)

and the outcome. The summary bootstraps 95% intervals over seed pairs: per-game means for most
columns; kills_pm and the *_pm rates as pooled ratios (sum of events / sum of minutes, kills_pm
over both-alive minutes); first_kill as the Kaplan-Meier median with games that end without a
subject kill censored at their length (Bopl rounds are one life, so a kill ends the game: the
plain mean over games would just be the game length).
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
import kill_credit  # noqa: E402

FIX = 4294967296.
CLOSE = 0.05   # game units per decision: below this a move counts as neither closing nor retreating
# RlTelemetry's first three counters are per kit slot (sim/abi/bopl_rl.h ability_activations[3]);
# eval/telemetry.py names them for the 2,4,14 slot order, so columns() renames them by the kit.
COUNTERS = ('dash_activations', 'bow_activations', 'platform_activations', 'arrows_fired', 'arrow_hits')
BASE = ('win', 'seconds', 'distance', 'closing', 'retreating', 'kill_win', 'kills_pm', 'first_kill')


def columns(kit):
    """{telemetry field: column} for a kit: slot activations named by the slot's ability; arrow_hits is
    arrow_kills (the counter counts lethal arrows only)."""
    from evaluation_protocol import ability_names
    names = ability_names()
    slug = [names.get(a, str(a)).lower().replace(' ', '_') for a in kit]
    slug = [n if slug.count(n) == 1 else f'{n}{i + 1}' for i, n in enumerate(slug)]
    return {**{c: f'{slug[i]}_activations' for i, c in enumerate(COUNTERS[:3])},
            'arrows_fired': 'arrows_fired', 'arrow_hits': 'arrow_kills'}


def metrics(kit):
    return BASE + tuple(c + '_pm' for c in columns(kit).values())


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
    kills, alive_ticks = 0, 0
    ids = [s.id for s in env.state.players[:2]]
    while not env.ended:
        s = env.state
        p, q = s.players[me], s.players[them]
        before = (p.x / FIX, p.y / FIX, q.x / FIX, q.y / FIX, bool(p.flags & 1 and q.flags & 1))
        tick0 = s.header.episode_ticks
        observation = env.rows.copy(); mask = env.mask.copy()
        env.step(np.stack([actors[0](observation, mask), actors[1](observation, mask)]))
        counts += np.asarray(env.telemetry[me], dtype=np.float64)
        s = env.state
        p = s.players[me]
        new = kill_credit.credited_kills(env.events(), ids)[me]
        if new and first_kill is None:
            first_kill = s.header.episode_ticks / 60
        kills += new
        px, py, qx, qy, alive = before
        if alive:
            alive_ticks += s.header.episode_ticks - tick0
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
    minutes = seconds / 60 if seconds else 1
    row = dict(win=1.0 if outcome == 'a' else 0.5 if outcome == 'draw' else 0.0, seconds=seconds,
               distance=float(np.mean(distances)) if distances else None,
               closing=closing / decisions if decisions else None,
               retreating=retreating / decisions if decisions else None,
               kills=kills, kill_win=float(outcome == 'a' and kills > 0), alive_s=alive_ticks / 60,
               kills_pm=kills / max(alive_ticks / 3600, 1e-9), first_kill=first_kill)
    for c, name in columns(task['kits'][3 * me:3 * me + 3]).items():
        row[name + '_pm'] = totals[c] / minutes
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


def km_median(times, events):
    """Kaplan-Meier median of event times (events[i] False = censored at times[i]); None if S never reaches 0.5."""
    order = np.lexsort((~np.asarray(events, bool), np.asarray(times)))
    t, e = np.asarray(times, float)[order], np.asarray(events, bool)[order]
    at_risk, surv = len(t), 1.0
    for ti, ei in zip(t, e):
        if ei:
            surv *= 1 - 1 / at_risk
            if surv <= 0.5:
                return float(ti)
        at_risk -= 1
    return None


def statistic(m, games):
    if m == 'kills_pm':   # credited kills per both-alive minute, pooled
        return sum(g['kills'] for g in games) / max(sum(g['alive_s'] for g in games) / 60, 1e-9)
    if m.endswith('_pm'):  # events per game minute, pooled
        return sum(g[m] * g['seconds'] for g in games) / max(sum(g['seconds'] for g in games), 1e-9)
    if m == 'first_kill':
        return km_median([g['first_kill'] if g['first_kill'] is not None else g['seconds'] for g in games],
                         [g['first_kill'] is not None for g in games])
    vals = [g[m] for g in games if g[m] is not None]
    return float(np.mean(vals)) if vals else None


def summarize(games, rng):
    """{metric: [estimate, low, high]}, the interval bootstrapped over seed pairs (None where undefined)."""
    by_pair = {}
    for g in games:
        by_pair.setdefault(g['seed'], []).append(g)
    pairs = list(by_pair.values())
    picks = [rng.integers(len(pairs), size=len(pairs)) for _ in range(2000)]
    out = {}
    for m in BASE + tuple(k for k in games[0] if k.endswith('_pm') and k != 'kills_pm'):
        est = statistic(m, games)
        boot = [b for b in (statistic(m, [g for i in pick for g in pairs[i]]) for pick in picks) if b is not None]
        rnd = lambda x: None if x is None else round(float(x), 4)
        out[m] = [rnd(est), rnd(np.percentile(boot, 2.5)) if boot else None, rnd(np.percentile(boot, 97.5)) if boot else None]
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
            print(f'{sname:>8} vs {oname:<14} ' + '  '.join(f'{m} {v[0]:.3g}' if v[0] is not None else f'{m} -'
                                                            for m, v in row.items()), flush=True)
        everyone = [json.loads(l) for l in open(a.out / 'games.jsonl') if json.loads(l)['subject'] == sname]
        summary[sname]['all'] = summarize(everyone, np.random.default_rng(0))
    (a.out / 'summary.json').write_text(json.dumps(dict(kit=list(kit), pairs=a.pairs, seed_base=a.seed_base, temperature=a.temperature,
                                                        close=CLOSE, summary=summary), indent=1))


if __name__ == '__main__':
    main()
