"""Do the eight specialist checkpoints still perform in the corrected simulator?

An A/B of one frozen checkpoint against itself: same weights, same opponents, same
maps, same seeds, same side-swaps -- only the sim library differs. `run` plays the
grid through research/evaluate.py (which is the training-era evaluation protocol);
`report` pairs the games and prints per-kit deltas with cluster-bootstrap intervals.

    python checkpoint_transfer.py run    --out DIR --libs base=A.so main=B.so [--kits ...]
    python checkpoint_transfer.py report --out DIR --base base --arm main

Pairing is on (opponent, map, seed, side): the same starting condition under both
libraries. The trajectories diverge -- that is the point -- but the matchup, map and
spawn are held fixed, so the paired delta is far tighter than two independent Wilson
intervals on the same games. Clusters for the bootstrap are side-swapped seed pairs
(opponent, map, seed), the unit evaluate.py schedules.

Two loud checks guard against measuring nothing, in the manner this project has been
bitten before:
  * `run` re-plays, under the training-era library, the exact schedule stored in each
    run's own evaluations/<step>/result.json and requires every game to match on
    win/timeout/kills/deaths/ticks. A library that is not the training-era build, or a
    harness that is not sampling the policy, cannot pass this.
  * `report` requires a nonzero fraction of paired games to differ in tick count. If
    the two libraries produced identical trajectories everywhere, the A/B is vacuous
    and the win-rate delta of 0.000 would be an artefact, not a finding.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SWEEPS = ROOT / 'native/artifacts/rl/sweeps'
VENV = ROOT / 'native/artifacts/rl/venv/bin/python'

KITS = ['dash-bow-platform', 'roll-grenade-smoke', 'teleport-mine-tesla', 'web-mjolnir-freeze',
        'dash-missile-platform', 'grapple-beam-rock', 'duplicator-blink-blackhole', 'drill-throw-magnet']

# The five bots the runs were evaluated against while training, plus the two `search`
# bots, which are ~200 Elo stronger (pool v7) and pull the measurement off the ceiling
# the training-era suite sits on (0.95-1.00 win rate on trained maps).
TRAINING_ERA_SCRIPTED = ['mixed-hard', 'mixed-v2-hard', 'archer-v2-hard', 'pursuer-v2-hard', 'evasive-v2-hard']
HARD_SCRIPTED = ['search-medium', 'search-hard']
TRAINING_MAPS = [0, 1, 2]
HELDOUT_MAPS = [3, 4]

KEY = ('opponent', 'map', 'seed', 'side')
MATCH_FIELDS = ('win', 'timeout', 'kills', 'deaths', 'ticks')


def kit_run(kit):
    return SWEEPS / f'{kit}-long/000'


def final_checkpoint(kit):
    run = kit_run(kit)
    step = json.loads((run / 'result.json').read_text())['step']
    return run / f'checkpoints/policy-{step:09d}.bin', step


def game_key(game):
    return tuple(game[k] for k in KEY)


def read_games(path):
    return {game_key(g): g for g in json.loads(Path(path).read_text())['games']}


def env_for(library):
    env = dict(os.environ)
    env.update(BOPL_RL_BACKEND='rust', BOPL_RL_INPROC_SPARE='0',
               BOPL_RL_WORLDS_PER_RUNTIME=str(1 << 20), OMP_NUM_THREADS='1',
               BOPL_RL_GAME=str(Path(library).resolve()))
    return env


def evaluate(run_dir, kit, library, scripted, games, heldout_games, workers, log):
    checkpoint, step = final_checkpoint(kit)
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    result = run_dir / 'evaluations' / f'{step:09d}' / 'result.json'
    if result.is_file():
        return result
    command = [str(VENV), str(Path(__file__).resolve().parent / 'evaluate.py'),
               '--run', str(run_dir), '--checkpoint', str(checkpoint),
               '--initial', str(kit_run(kit) / 'checkpoints/policy-000000000.bin'),
               '--evaluation-maps', ','.join(map(str, TRAINING_MAPS)),
               '--heldout-maps', ','.join(map(str, HELDOUT_MAPS)),
               '--games-per-opponent', str(games), '--heldout-games', str(heldout_games),
               '--workers', str(workers), '--game', str(Path(library).resolve()),
               '--scripted-opponents', ','.join(scripted), '--no-pool-record']
    with open(log, 'w') as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT,
                       env=env_for(library), cwd=str(Path(__file__).resolve().parent))
    return result


# ---- sanity ------------------------------------------------------------------

def reproduce_training_era(out, kit, library, workers=2):
    """Replay the run's own stored evaluation schedule and require every game to match.

    This is the check that would fail loudly if the harness were measuring nothing:
    a wrong library, an unloaded checkpoint or a zeroed action head all break it.
    """
    checkpoint, step = final_checkpoint(kit)
    stored = kit_run(kit) / 'evaluations' / f'{step:09d}' / 'result.json'
    record = json.loads(stored.read_text())
    run_dir = Path(out) / 'sanity' / kit
    result = evaluate(run_dir, kit, library, record['scripted_opponents'], 8, 4, workers,
                      Path(out) / 'logs' / f'sanity-{kit}.log')
    want, got = read_games(stored), read_games(result)
    if set(want) != set(got):
        raise SystemExit(f'{kit}: replayed schedule differs from the stored one')
    bad = [k for k in want if any(want[k][f] != got[k][f] for f in MATCH_FIELDS)]
    replays = [(k, want[k]['replay_sha256']) for k in want if want[k].get('replay_sha256')]
    identical = sum(got[k].get('replay_sha256') == sha for k, sha in replays)
    return dict(kit=kit, games=len(want), mismatched=len(bad),
                stored_wins=sum(g['win'] for g in want.values()),
                replayed_wins=sum(g['win'] for g in got.values()),
                saved_replays=len(replays), identical_replays=identical,
                ok=not bad and identical == len(replays))


# ---- outcomes -------------------------------------------------------------------

# The grid was played by the evaluation scripts at 3994e0c3, whose `win` field counts an
# undecided round (a trade) as a win for whichever side the report was written from.
# Main has since redefined it (25b093f8 "evaluation: report undecided rounds and score
# them as draws"): a round carrying the win flag for more than one team is a win for
# nobody. Every stored game row carries `final_players` flags, so main's definition is
# recovered here exactly -- `undecided` and the corrected `kills` are recomputed from the
# stored rows rather than inherited from the 3994e0c3 `win` field, and the legacy field
# is never read. Both arms would have carried the same bias, but the trade *rate* itself
# moves between libraries, so the legacy field would have hidden a real effect.

def classify(game):
    """main's outcome for the reporting side: win / loss / undecided / timeout / both-dead."""
    flagged = [bool(p['flags'] & 2) for p in game['final_players']]
    if game['timeout']:
        return 'timeout'
    if sum(flagged) > 1:
        return 'undecided'
    if not any(flagged):
        return 'both-dead'
    return 'win' if flagged[game['side']] else 'loss'


def is_win(game):
    return classify(game) == 'win'


def score(game):
    """Win 1, loss 0, every kind of draw 0.5 -- the honest scalar when trades are common.

    A suite that ends in trades reads as a collapse under a strict win rate and as a
    clean sweep under the legacy trade-as-win field; neither is true, and this is the
    number the per-kit verdicts below are taken from.
    """
    outcome = classify(game)
    return 1.0 if outcome == 'win' else 0.0 if outcome == 'loss' else 0.5


def corrected_kills(game):
    """Kills of somebody else: the game credits a suicide to the victim's own Kills."""
    return game['kills'] - game['behavior']['totals'].get('deaths_self', 0.0)


# ---- statistics ----------------------------------------------------------------

def wilson(wins, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, mid - half), min(1.0, mid + half))


def paired(base_games, arm_games, keep, value, resamples=20000, seed=0):
    """Paired base-vs-main delta of `value`, bootstrapped over side-swapped seed pairs.

    Pairing is on (opponent, map, seed, side): the same starting condition under both
    libraries. Resampling whole (opponent, map, seed) clusters keeps the two sides of a
    swap together, which is the correlation evaluate.py's own Wilson note says it does
    not adjust for.
    """
    clusters = {}
    for key, b in base_games.items():
        m = arm_games.get(key)
        if m is None or not keep(b):
            continue
        clusters.setdefault((b['opponent'], b['map'], b['seed']), []).append((value(b), value(m)))
    keys = list(clusters)
    if not keys:
        return dict(n=0, base=float('nan'), arm=float('nan'), delta=float('nan'),
                    ci=(float('nan'), float('nan')), clusters=0)
    sums = np.array([[sum(b for b, _ in clusters[k]), sum(m for _, m in clusters[k]), len(clusters[k])]
                     for k in keys], dtype=float)
    n = int(sums[:, 2].sum())
    base, arm = sums[:, 0].sum() / n, sums[:, 1].sum() / n
    rng = np.random.default_rng(seed)
    totals = sums[rng.integers(0, len(keys), size=(resamples, len(keys)))].sum(axis=1)
    deltas = (totals[:, 1] - totals[:, 0]) / np.maximum(1, totals[:, 2])
    return dict(n=n, clusters=len(keys), base=float(base), arm=float(arm), delta=float(arm - base),
                ci=(float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))))


def outcome_mix(games, keep):
    kinds = ('win', 'loss', 'undecided', 'timeout', 'both-dead')
    sel = [g for g in games.values() if keep(g)]
    if not sel:
        return {k: 0.0 for k in kinds}
    counts = {k: 0 for k in kinds}
    for g in sel:
        counts[classify(g)] += 1
    return {k: counts[k] / len(sel) for k in kinds}


def churn_of(base_games, arm_games, keep):
    both = [(b, arm_games[k]) for k, b in base_games.items() if k in arm_games and keep(b)]
    return (sum(b['ticks'] != m['ticks'] for b, m in both) / len(both)) if both else 0.0


def flips(base_games, arm_games, keep):
    lost = gained = 0
    for key, b in base_games.items():
        m = arm_games.get(key)
        if m is None or not keep(b):
            continue
        lost += is_win(b) and not is_win(m)
        gained += is_win(m) and not is_win(b)
    return lost, gained


# Pool v7 members that appear in this schedule, with their frozen ratings. A
# fixed-anchor Elo (elo_ladder.bradley_terry_fixed, OpenAI Five's frozen reference
# rating) turns saturated win rates into one scalar that is not at a ceiling. The
# schedule is identical under both libraries, so unlike reference_pool.py's band
# allocation there is no adaptive-opponent confound between the arms. Draws --
# timeout, undecided and both-dead alike -- are draws in the fit, as in
# reference_pool.evaluation_games after 25b093f8.
POOL_ANCHORS = {'random': 0.0, 'mixed-hard': 378.2, 'search-hard': 580.7, 'search-medium': 588.8}


def anchored_elo(games):
    import elo_ladder
    records = {}
    for opponent, outcome in games:
        w, l, d = records.get(opponent, (0, 0, 0))
        records[opponent] = (w + (outcome == 'win'), l + (outcome == 'loss'),
                             d + (outcome in ('undecided', 'timeout', 'both-dead')))
    records = {k: list(v) for k, v in records.items() if k in POOL_ANCHORS}
    return elo_ladder.bradley_terry_fixed(records, POOL_ANCHORS) if records else float('nan')


def elo_pair(base_games, arm_games, keep, resamples=2000, seed=1):
    clusters = {}
    for key, b in base_games.items():
        m = arm_games.get(key)
        if m is None or not keep(b) or b['opponent'] not in POOL_ANCHORS:
            continue
        clusters.setdefault((b['opponent'], b['map'], b['seed']), []).append(
            ((b['opponent'], classify(b)), (m['opponent'], classify(m))))
    keys = list(clusters)
    if not keys:
        return dict(base=float('nan'), arm=float('nan'), delta=float('nan'),
                    ci=(float('nan'), float('nan')), games=0)
    flat_b = [g for k in keys for g, _ in clusters[k]]
    flat_m = [g for k in keys for _, g in clusters[k]]
    base, arm = anchored_elo(flat_b), anchored_elo(flat_m)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        picked = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        deltas.append(anchored_elo([g for k in picked for _, g in clusters[k]])
                      - anchored_elo([g for k in picked for g, _ in clusters[k]]))
    return dict(base=base, arm=arm, delta=arm - base, games=len(flat_b),
                ci=(float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))))


def means(games, keep):
    sel = [g for g in games.values() if keep(g)]
    if not sel:
        return {}
    return dict(kills=float(np.mean([corrected_kills(g) for g in sel])),
                deaths=float(np.mean([g['deaths'] for g in sel])),
                seconds=float(np.mean([g['ticks'] / 60 for g in sel])))


def divergence_rates(path):
    rows = {}
    if not Path(path).is_file():
        return rows
    lines = Path(path).read_text().splitlines()
    header = lines[0].split('\t')
    ki, di = header.index('kit'), header.index('diverged')
    for line in lines[1:]:
        f = line.split('\t')
        kit = f[ki].removesuffix('-long')
        n, d = rows.get(kit, (0, 0))
        rows[kit] = (n + 1, d + int(f[di]))
    return {k: d / n for k, (n, d) in rows.items()}


# ---- commands --------------------------------------------------------------------

def command_run(args):
    out = Path(args.out).resolve()
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    libs = dict(item.split('=', 1) for item in args.libs)
    import hashlib
    manifest = dict(libraries={name: dict(path=str(Path(p).resolve()),
                                          sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
                               for name, p in libs.items()},
                    kits=args.kits, scripted=args.scripted, games=args.games,
                    heldout_games=args.heldout_games,
                    training_maps=TRAINING_MAPS, heldout_maps=HELDOUT_MAPS)
    path = out / f'manifest-{"-".join(sorted(libs))}.json'
    path.write_text(json.dumps(manifest, indent=1) + '\n')
    for kit in args.kits:
        for name, library in libs.items():
            evaluate(out / 'runs' / kit / name, kit, library, args.scripted,
                     args.games, args.heldout_games, args.workers,
                     out / 'logs' / f'{kit}-{name}.log')
            print(f'done {kit} {name}', flush=True)


def command_sanity(args):
    out = Path(args.out).resolve()
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    libs = dict(item.split('=', 1) for item in args.libs)
    rows = [reproduce_training_era(out, kit, libs[args.base], args.workers) for kit in args.kits]
    (out / 'sanity.json').write_text(json.dumps(rows, indent=1) + '\n')
    for r in rows:
        print(f"{r['kit']:28s} games={r['games']:4d} mismatched={r['mismatched']:3d} "
              f"wins {r['stored_wins']}->{r['replayed_wins']} replays {r['identical_replays']}/{r['saved_replays']} "
              f"{'OK' if r['ok'] else 'FAIL'}", flush=True)
    if not all(r['ok'] for r in rows):
        raise SystemExit('training-era reproduction failed: the harness is not measuring what it claims')


SUITES = dict(
    training_era_suite=lambda g: g['opponent'] in (['random', 'initial'] + TRAINING_ERA_SCRIPTED),
    hard_bots=lambda g: g['opponent'] in HARD_SCRIPTED,
    all_bots=lambda g: g['opponent'] not in ('selfplay',),
)
ROLES = (('trained', TRAINING_MAPS), ('heldout', HELDOUT_MAPS), ('both', None))


def command_crosscheck(args):
    """Are two evaluations of the same library interchangeable on the games they share?

    The oracle cannot host the two `search` bots (they decide by rollouts on a
    bopl_rl_fork, which only the Rust sim provides), so an oracle arm must be
    scheduled without them. evaluation_protocol.schedule derives a game's seed
    from (seed_base, map, game index) alone -- never from the opponent's position
    in the list -- so dropping an opponent should leave every other game's
    (opponent, map, seed, side) key AND its outcome untouched, which is what
    makes an existing 7-opponent arm reusable as the comparison arm.

    "Should" is the part that has bitten this project before: the workers reuse
    one game world across tasks, so if any state leaked across resets, changing
    the task list would change the results. This replays that claim against real
    data instead of asserting it: it requires a nonzero number of shared games
    (a comparator that silently compared nothing would otherwise "pass") and
    reports exact agreement on win/timeout/kills/deaths/ticks.

        checkpoint_transfer.py crosscheck --out DIR --reference OTHER_DIR \
            --base main --arm main --kits ...
    """
    out, reference = Path(args.out).resolve(), Path(args.reference).resolve()
    rows, failed = [], False
    for kit in args.kits:
        _, step = final_checkpoint(kit)
        mine = out / 'runs' / kit / args.arm / 'evaluations' / f'{step:09d}' / 'result.json'
        theirs = reference / 'runs' / kit / args.base / 'evaluations' / f'{step:09d}' / 'result.json'
        if not mine.is_file() or not theirs.is_file():
            rows.append(dict(kit=kit, shared=0, mismatched=0, ok=False,
                             note=f'missing {"mine" if not mine.is_file() else "reference"}'))
            failed = True
            continue
        a, b = read_games(mine), read_games(theirs)
        shared = sorted(set(a) & set(b))
        bad = [k for k in shared if any(a[k][f] != b[k][f] for f in MATCH_FIELDS)]
        # Opponents present on only one side are the point of the exercise; count
        # them so the report shows the comparison was not vacuously empty.
        only_mine = sorted({k[0] for k in set(a) - set(b)})
        only_theirs = sorted({k[0] for k in set(b) - set(a)})
        ok = bool(shared) and not bad
        failed |= not ok
        rows.append(dict(kit=kit, mine=len(a), reference=len(b), shared=len(shared),
                         mismatched=len(bad), opponents_only_mine=only_mine,
                         opponents_only_reference=only_theirs, ok=ok,
                         examples=[list(k) for k in bad[:5]]))
    for r in rows:
        print(f"{r['kit']:28s} shared={r.get('shared',0):5d} mismatched={r.get('mismatched',0):4d} "
              f"only-reference={','.join(r.get('opponents_only_reference') or ['-'])} "
              f"{'OK' if r['ok'] else 'FAIL ' + str(r.get('note') or r.get('examples'))}", flush=True)
    (out / 'crosscheck.json').write_text(json.dumps(rows, indent=1) + '\n')
    if failed:
        raise SystemExit('crosscheck failed: the two schedules are NOT interchangeable on their shared games')
    print('\ncrosscheck OK: every shared game agrees on ' + '/'.join(MATCH_FIELDS))


def command_report(args):
    out = Path(args.out).resolve()
    diverge = divergence_rates(args.divergence)
    report = {}
    for kit in args.kits:
        _, step = final_checkpoint(kit)
        base = read_games(out / 'runs' / kit / args.base / 'evaluations' / f'{step:09d}' / 'result.json')
        arm = read_games(out / 'runs' / kit / args.arm / 'evaluations' / f'{step:09d}' / 'result.json')
        entry = dict(step=step, divergence=diverge.get(kit))
        for suite, keep in SUITES.items():
            for role, maps in ROLES:
                sel = (lambda g, keep=keep, maps=maps: keep(g) and (maps is None or g['map'] in maps))
                stat = dict(score=paired(base, arm, sel, score, args.resamples),
                            win=paired(base, arm, sel, lambda g: float(is_win(g)), args.resamples),
                            undecided=paired(base, arm, sel,
                                             lambda g: float(classify(g) == 'undecided'), args.resamples),
                            churn=churn_of(base, arm, sel),
                            base_mix=outcome_mix(base, sel), arm_mix=outcome_mix(arm, sel),
                            base_means=means(base, sel), arm_means=means(arm, sel))
                stat['lost'], stat['gained'] = flips(base, arm, sel)
                # A suite can be empty by construction: the oracle backend cannot host
                # the two `search` bots (no bopl_rl_fork), so `hard_bots` has no paired
                # games there and `paired` reports n=0 with NaN rates. Report the empty
                # suite as empty rather than crashing on round(NaN).
                n = stat['win']['n']
                stat['base_win_ci'] = wilson(round(stat['win']['base'] * n), n) if n else (float('nan'),) * 2
                stat['arm_win_ci'] = wilson(round(stat['win']['arm'] * n), n) if n else (float('nan'),) * 2
                entry[f'{suite}:{role}'] = stat
        for role, maps in ROLES:
            entry[f'elo:{role}'] = elo_pair(base, arm, lambda g, m=maps: m is None or g['map'] in m)
        entry['by_opponent'] = {o: paired(base, arm, lambda g, o=o: g['opponent'] == o, score, 4000)
                                for o in sorted({g['opponent'] for g in base.values()} - {'selfplay'})}
        report[kit] = entry

    overall_churn = float(np.mean([e['all_bots:both']['churn'] for e in report.values()]))
    if overall_churn <= 0:
        raise SystemExit('no paired game differed between the libraries: the A/B measured nothing')
    (out / f'report-{args.base}-vs-{args.arm}.json').write_text(json.dumps(report, indent=1) + '\n')

    print(f'# {args.base} -> {args.arm}   evaluation semantics: main @25b093f8 '
          f'(undecided = draw), recomputed from stored final_players flags')
    print(f'# trajectory churn: {overall_churn:.1%} of paired games end on a different tick\n')
    for suite in SUITES:
        print(f'## {suite}: score (win 1, draw .5, loss 0), paired delta, 95% cluster bootstrap')
        print(f"{'kit':28s} {'role':8s} {'n':>6s} {'pairs':>6s} {'base':>7s} {'main':>7s} "
              f"{'delta':>8s} {'95% CI':>18s} {'win%':>15s} {'undec%':>15s} {'lose/gain':>10s}")
        for kit, entry in report.items():
            for role, _ in ROLES:
                s = entry[f'{suite}:{role}']
                sc, w, u = s['score'], s['win'], s['undecided']
                print(f"{kit:28s} {role:8s} {sc['n']:6d} {sc['clusters']:6d} {sc['base']:7.3f} {sc['arm']:7.3f} "
                      f"{sc['delta']:+8.3f} [{sc['ci'][0]:+.3f},{sc['ci'][1]:+.3f}] "
                      f"{w['base']:6.3f}->{w['arm']:<6.3f} {u['base']:6.3f}->{u['arm']:<6.3f} "
                      f"{s['lost']:4d}/{s['gained']:<5d}")
        print()

    print('## fixed-anchor Elo vs pool v7 (random / mixed-hard / search-medium / search-hard)')
    print(f"{'kit':28s} {'role':8s} {'games':>6s} {'base':>8s} {'main':>8s} {'delta':>8s} {'95% CI':>20s}")
    for kit, entry in report.items():
        for role, _ in ROLES:
            e = entry[f'elo:{role}']
            print(f"{kit:28s} {role:8s} {e['games']:6d} {e['base']:8.1f} {e['arm']:8.1f} "
                  f"{e['delta']:+8.1f} [{e['ci'][0]:+7.1f},{e['ci'][1]:+7.1f}]")

    print('\n## per-opponent paired score delta (all maps)')
    kits = list(report)
    print(' ' * 19 + ' '.join(f'{k[:11]:>12s}' for k in kits))
    for opponent in sorted({o for e in report.values() for o in e['by_opponent']}):
        cells = [f"{report[k]['by_opponent'][opponent]['delta']:+12.3f}"
                 if report[k]['by_opponent'].get(opponent, {}).get('n') else '           -' for k in kits]
        print(f'{opponent:18s} ' + ' '.join(cells))

    print('\n## continuous metrics, all bots, all maps (base -> main)')
    print(f"{'kit':28s} {'kills':>18s} {'deaths':>18s} {'seconds':>18s}")
    for kit, entry in report.items():
        s = entry['all_bots:both']
        cells = ' '.join(f"{s['base_means'][f]:8.3f}->{s['arm_means'][f]:<9.3f}"
                         for f in ('kills', 'deaths', 'seconds'))
        print(f'{kit:28s} {cells}')

    print('\n# observation divergence (blast-radius rows-tol1) vs performance change')
    xs, ys, es = [], [], []
    for kit, entry in report.items():
        d = entry['divergence']
        if d is None:
            continue
        s = entry['all_bots:both']
        xs.append(d); ys.append(s['score']['delta']); es.append(entry['elo:both']['delta'])
        print(f"{kit:28s} obs_divergence={d:.3f} score_delta={ys[-1]:+.3f} "
              f"elo_delta={es[-1]:+7.1f} churn={s['churn']:.3f}")
    if len(xs) > 2:
        print(f'pearson r (divergence, score delta) = {np.corrcoef(xs, ys)[0,1]:+.3f}  (n={len(xs)} kits)')
        print(f'pearson r (divergence, elo delta)   = {np.corrcoef(xs, es)[0,1]:+.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('run', 'sanity', 'report', 'crosscheck'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--libs', nargs='*', default=[], help='name=path/to/libbopl_sim.so')
    parser.add_argument('--kits', nargs='*', default=KITS)
    parser.add_argument('--scripted', nargs='*', default=TRAINING_ERA_SCRIPTED + HARD_SCRIPTED)
    parser.add_argument('--games', type=int, default=192, help='games per opponent per trained map')
    parser.add_argument('--heldout-games', type=int, default=288)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--base', default='base')
    parser.add_argument('--arm', default='main')
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--reference', type=Path,
                        help='crosscheck: another run directory to compare shared games against')
    parser.add_argument('--divergence', type=Path,
                        default=Path('/home/mte/.local/share/bopl-null-hypothesis/blast-radius-2026-09-23/rows-tol1.tsv'))
    args = parser.parse_args()
    dict(run=command_run, sanity=command_sanity, report=command_report,
         crosscheck=command_crosscheck)[args.command](args)


if __name__ == '__main__':
    main()
