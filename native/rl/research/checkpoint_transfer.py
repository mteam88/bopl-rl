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


# ---- statistics ----------------------------------------------------------------

def wilson(wins, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, mid - half), min(1.0, mid + half))


def paired_delta(pairs, resamples=20000, seed=0):
    """Mean of (main win - base win) over games, with a cluster bootstrap CI.

    `pairs` is a list of (cluster, base_win, main_win). Clusters are side-swapped
    seed pairs; resampling whole clusters keeps the two games of a pair together,
    which is the correlation evaluate.py's own Wilson note says it does not adjust for.
    """
    if not pairs:
        return dict(n=0, base=float('nan'), main=float('nan'), delta=float('nan'),
                    ci=(float('nan'), float('nan')), n01=0, n10=0, churn=None)
    clusters = {}
    for cluster, base, main in pairs:
        clusters.setdefault(cluster, []).append((base, main))
    keys = list(clusters)
    base_all = np.array([b for _, b, _ in pairs], dtype=float)
    main_all = np.array([m for _, _, m in pairs], dtype=float)
    # Per-cluster sums, so a bootstrap draw is a sum over sampled clusters.
    sums = np.array([[sum(b for b, _ in clusters[k]), sum(m for _, m in clusters[k]), len(clusters[k])]
                     for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(resamples, len(keys)))
    picked = sums[draws]                     # (resamples, clusters, 3)
    totals = picked.sum(axis=1)
    deltas = (totals[:, 1] - totals[:, 0]) / np.maximum(1, totals[:, 2])
    n01 = int(((base_all == 0) & (main_all == 1)).sum())
    n10 = int(((base_all == 1) & (main_all == 0)).sum())
    return dict(n=len(pairs), base=float(base_all.mean()), main=float(main_all.mean()),
                delta=float(main_all.mean() - base_all.mean()),
                ci=(float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
                n01=n01, n10=n10, clusters=len(keys))


# Pool v7 members that appear in this schedule, with their frozen ratings. A
# fixed-anchor Elo (elo_ladder.bradley_terry_fixed, OpenAI Five's frozen reference
# rating) turns the saturated win rates into one scalar that is not at a ceiling.
# The schedule is identical under both libraries, so unlike reference_pool.py's
# band allocation there is no adaptive-opponent confound between the two arms.
POOL_ANCHORS = {'random': 0.0, 'mixed-hard': 378.2, 'search-hard': 580.7, 'search-medium': 588.8}


def records_from(games):
    records = {}
    for opponent, win, timeout in games:
        w, l, d = records.get(opponent, (0, 0, 0))
        records[opponent] = (w + (win and not timeout), l + (not win and not timeout), d + bool(timeout))
    return records


def anchored_elo(games):
    import elo_ladder
    records = {k: list(v) for k, v in records_from(games).items() if k in POOL_ANCHORS}
    if not records:
        return float('nan')
    return elo_ladder.bradley_terry_fixed(records, POOL_ANCHORS)


def elo_pair(base_games, arm_games, keep, resamples=2000, seed=1):
    """Fixed-anchor Elo under each library and the delta, bootstrapped over seed pairs."""
    clusters = {}
    for key, b in base_games.items():
        m = arm_games.get(key)
        if m is None or not keep(b) or b['opponent'] not in POOL_ANCHORS:
            continue
        clusters.setdefault((b['opponent'], b['map'], b['seed']), []).append(
            ((b['opponent'], b['win'], b['timeout']), (m['opponent'], m['win'], m['timeout'])))
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
        gb = [g for k in picked for g, _ in clusters[k]]
        gm = [g for k in picked for _, g in clusters[k]]
        deltas.append(anchored_elo(gm) - anchored_elo(gb))
    return dict(base=base, arm=arm, delta=arm - base, games=len(flat_b),
                ci=(float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))))


def means(base_games, arm_games, keep):
    """Continuous metrics, which unlike a saturated win rate can still move."""
    out = {}
    for label, source in (('base', base_games), ('arm', arm_games)):
        sel = [g for k, g in source.items() if keep(base_games.get(k, g))]
        if not sel:
            continue
        out[label] = dict(kills=float(np.mean([g['kills'] for g in sel])),
                          deaths=float(np.mean([g['deaths'] for g in sel])),
                          seconds=float(np.mean([g['ticks'] / 60 for g in sel])),
                          timeout_rate=float(np.mean([g['timeout'] for g in sel])))
    return out


def collect(base_games, arm_games, keep):
    pairs = []
    churn = same = 0
    for key, b in base_games.items():
        m = arm_games.get(key)
        if m is None or not keep(b):
            continue
        pairs.append(((b['opponent'], b['map'], b['seed']), int(b['win']), int(m['win'])))
        churn += b['ticks'] != m['ticks']
        same += 1
    return pairs, (churn / same if same else 0.0)


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
    manifest = dict(libraries={name: dict(path=str(Path(p).resolve()),
                                          sha256=__import__('hashlib').sha256(Path(p).read_bytes()).hexdigest())
                               for name, p in libs.items()},
                    kits=args.kits, scripted=args.scripted, games=args.games,
                    heldout_games=args.heldout_games,
                    training_maps=TRAINING_MAPS, heldout_maps=HELDOUT_MAPS)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
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


def command_report(args):
    out = Path(args.out).resolve()
    diverge = divergence_rates(args.divergence)
    suites = dict(
        all=lambda g: True,
        training_era_suite=lambda g: g['opponent'] in (['random', 'initial'] + TRAINING_ERA_SCRIPTED),
        hard_bots=lambda g: g['opponent'] in HARD_SCRIPTED,
    )
    report = {}
    for kit in args.kits:
        _, step = final_checkpoint(kit)
        base = read_games(out / 'runs' / kit / args.base / 'evaluations' / f'{step:09d}' / 'result.json')
        arm = read_games(out / 'runs' / kit / args.arm / 'evaluations' / f'{step:09d}' / 'result.json')
        entry = dict(step=step, divergence=diverge.get(kit))
        for suite, keep in suites.items():
            for role, maps in (('trained', TRAINING_MAPS), ('heldout', HELDOUT_MAPS), ('both', None)):
                sel = (lambda g, keep=keep, maps=maps: keep(g) and (maps is None or g['map'] in maps))
                pairs, churn = collect(base, arm, sel)
                stat = paired_delta(pairs, args.resamples)
                stat['churn'] = churn
                stat['base_ci'] = wilson(sum(p[1] for p in pairs), len(pairs))
                stat['arm_ci'] = wilson(sum(p[2] for p in pairs), len(pairs))
                stat['means'] = means(base, arm, sel)
                entry[f'{suite}:{role}'] = stat
        for role, maps in (('trained', TRAINING_MAPS), ('heldout', HELDOUT_MAPS), ('both', None)):
            sel = (lambda g, maps=maps: maps is None or g['map'] in maps)
            entry[f'elo:{role}'] = elo_pair(base, arm, sel)
        by_opponent = {}
        for opponent in sorted({g['opponent'] for g in base.values()}):
            sel = (lambda g, o=opponent: g['opponent'] == o)
            pairs, churn = collect(base, arm, sel)
            s2 = paired_delta(pairs, 4000)
            s2['churn'] = churn
            by_opponent[opponent] = s2
        entry['by_opponent'] = by_opponent
        report[kit] = entry
    overall_churn = np.mean([e['all:both']['churn'] for e in report.values()])
    if overall_churn <= 0:
        raise SystemExit('no paired game differed between the libraries: the A/B measured nothing')
    (out / f'report-{args.base}-vs-{args.arm}.json').write_text(json.dumps(report, indent=1) + '\n')
    print(f'# {args.base} -> {args.arm};  trajectory churn {overall_churn:.1%} of paired games')
    for suite in suites:
        print(f'\n## {suite}')
        print(f"{'kit':28s} {'role':8s} {'n':>6s} {'base':>7s} {'main':>7s} {'delta':>8s} "
              f"{'95% CI':>18s} {'lose/gain':>10s}")
        for kit, entry in report.items():
            for role in ('trained', 'heldout', 'both'):
                s = entry[f'{suite}:{role}']
                print(f"{kit:28s} {role:8s} {s['n']:6d} {s['base']:7.3f} {s['main']:7.3f} "
                      f"{s['delta']:+8.3f} [{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}] {s['n10']:4d}/{s['n01']:<5d}")
    print('\n## fixed-anchor Elo vs pool v7 (random / mixed-hard / search-medium / search-hard)')
    print(f"{'kit':28s} {'role':8s} {'games':>6s} {'base':>8s} {'main':>8s} {'delta':>8s} {'95% CI':>20s}")
    for kit, entry in report.items():
        for role in ('trained', 'heldout', 'both'):
            e = entry[f'elo:{role}']
            print(f"{kit:28s} {role:8s} {e['games']:6d} {e['base']:8.1f} {e['arm']:8.1f} "
                  f"{e['delta']:+8.1f} [{e['ci'][0]:+7.1f},{e['ci'][1]:+7.1f}]")

    print('\n## per-opponent paired delta (all maps)')
    kits = list(report)
    opponents = sorted({o for e in report.values() for o in e['by_opponent']})
    print(f"{'opponent':18s} " + ' '.join(f'{k[:11]:>12s}' for k in kits))
    for opponent in opponents:
        cells = []
        for kit in kits:
            s2 = report[kit]['by_opponent'].get(opponent)
            cells.append('           -' if not s2 or not s2['n'] else f"{s2['delta']:+12.3f}")
        print(f'{opponent:18s} ' + ' '.join(cells))

    print('\n## continuous metrics, all opponents, all maps (base -> main)')
    print(f"{'kit':28s} {'kills':>16s} {'deaths':>16s} {'seconds':>16s} {'timeout':>16s}")
    for kit, entry in report.items():
        m = entry['all:both']['means']
        cells = ' '.join(f"{m['base'][f]:7.3f}->{m['arm'][f]:<8.3f}" for f in ('kills', 'deaths', 'seconds', 'timeout_rate'))
        print(f'{kit:28s} {cells}')

    print('\n# divergence (blast-radius rows-tol1) vs delta')
    xs, ys = [], []
    for kit, entry in report.items():
        d = entry['divergence']
        if d is None:
            continue
        xs.append(d); ys.append(entry['all:both']['delta'])
        print(f"{kit:28s} obs_divergence={d:.3f} win_delta={ys[-1]:+.3f} "
              f"elo_delta={entry['elo:both']['delta']:+7.1f} churn={entry['all:both']['churn']:.3f}")
    if len(xs) > 2:
        elo = [report[k]['elo:both']['delta'] for k in report if report[k]['divergence'] is not None]
        print(f'pearson r (divergence, win delta) = {np.corrcoef(xs, ys)[0,1]:+.3f}  (n={len(xs)} kits)')
        print(f'pearson r (divergence, elo delta) = {np.corrcoef(xs, elo)[0,1]:+.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('run', 'sanity', 'report'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--libs', nargs='*', default=[], help='name=path/to/libbopl_sim.so')
    parser.add_argument('--kits', nargs='*', default=KITS)
    parser.add_argument('--scripted', nargs='*', default=TRAINING_ERA_SCRIPTED + HARD_SCRIPTED)
    parser.add_argument('--games', type=int, default=128, help='games per opponent per trained map')
    parser.add_argument('--heldout-games', type=int, default=192)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--base', default='base')
    parser.add_argument('--arm', default='main')
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--divergence', type=Path,
                        default=Path('/home/mte/.local/share/bopl-null-hypothesis/blast-radius-2026-09-23/rows-tol1.tsv'))
    args = parser.parse_args()
    dict(run=command_run, sanity=command_sanity, report=command_report)[args.command](args)


if __name__ == '__main__':
    main()
