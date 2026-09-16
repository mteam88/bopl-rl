"""Frozen reference pool: build it, verify it, rate checkpoints against it.

The pool is a fixed set of reference agents (the random policy, scripted bots,
frozen checkpoints) whose Elo ratings come from one full round robin fitted
by Bradley-Terry maximum likelihood (research/elo_ladder.py, random = 0 Elo,
timeouts = draws). Once written, pool.json is immutable: every checkpoint's
SHA-256 and a hash over (name, kind, sha256, rating) are recorded, and both
are re-checked whenever the pool is loaded. New references are appended by
building a new pool file; members are never removed or re-rated in place.

A candidate is rated OpenAI Five style (Dota 2 paper, appendix J): it plays
pool members whose ratings stay fixed, and its own rating is the 1-D MLE of
those games (elo_ladder.bradley_terry_fixed). The band-allocated rater spends
a small first pass against everyone, then the remaining games on 15-85%
matchups. Uncertainty is a cluster bootstrap over side-swapped seed pairs
(95% CI). A run's score is the interquartile mean of its last k rated
evaluation checkpoints (rliable / scipy trim_mean 0.25).

Rating files are derived data on the run: <run>/reference/<pool_sha8>/<checkpoint>.json.
A rating on another pool lives in another <pool_sha8>/ directory. Legacy files
at <run>/reference/<checkpoint>.json (overnight-4's driver) are still read.
Cache key: (pool sha, checkpoint sha, games). A fuller band rating is reused
for a cheaper request; an evaluation-fit rating is not.

    python native/rl/sweep/reference_pool.py build --out DIR/pool.json ...
    python native/rl/sweep/reference_pool.py verify DIR/pool.json
    python native/rl/sweep/reference_pool.py rate --pool DIR/pool.json --checkpoint policy.bin
    python native/rl/sweep/reference_pool.py rate-runs --pool DIR/pool.json --run DIR [--sweep DIR]
    python native/rl/sweep/reference_pool.py human --pool DIR/pool.json
"""
import argparse
import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'native/rl/research'), str(ROOT / 'native/tools')]

import elo_ladder  # noqa: E402  (pure until games are played)
from evaluation_protocol import checkpoint_identity, opponents, scripted_names  # noqa: E402

SCHEMA = 1
RUST_SIM = ROOT / 'native/target/release/libbopl_sim.so'
# The pool evaluate.py rates in-run evaluations on when --pool is not given.
# Pool v7 since 2026-09-16 (v4-lite onwards); runs before that (024, the
# seed-024-s7x floor, v4-full) logged pool v5 (sha8 ec5bd5ae): research/
# readout.py refits their stored games on v7 so curves stay comparable.
DEFAULT_POOL = ROOT / 'native/artifacts/rl/sweeps/anchors/pool-v7/pool.json'
DEFAULT_HUMAN = ROOT / 'native/artifacts/rl/human/mac'
CANDIDATE = 'candidate'


# ---- files -------------------------------------------------------------------

def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=1) + '\n')
    temporary.replace(path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def game_library(game=None):
    """The sim the games run on: explicit path, else the Rust sim under
    BOPL_RL_BACKEND=rust, else the .NET/Unity resolution (train.py's rule)."""
    if game:
        return Path(game).resolve()
    if os.environ.get('BOPL_RL_BACKEND', '').lower() == 'rust':
        return RUST_SIM.resolve()
    from reference import resolve_game
    return resolve_game()


def configure_games(workspace, game):
    """Point the research C environment at the sim (environment.configure)."""
    game = Path(game)
    if game.resolve() == RUST_SIM.resolve():
        # As train.py: the Rust sim needs no spare world and hosts every world in one runtime.
        os.environ.setdefault('BOPL_RL_INPROC_SPARE', '0')
        os.environ.setdefault('BOPL_RL_WORLDS_PER_RUNTIME', str(1 << 20))
    from environment import configure
    configure(Path(workspace) / 'workers', game)


# ---- pool identity -------------------------------------------------------------

def pool_hash(members):
    """Content hash of the frozen pool: names, kinds, weights and ratings."""
    rows = sorted((m['name'], m['kind'], m.get('sha256'), round(float(m['rating']), 3)) for m in members)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def load_pool(path, verify=True):
    """Read pool.json; with verify, re-check every checkpoint hash and the pool hash."""
    path = Path(path).resolve()
    pool = json.loads(path.read_text())
    if pool.get('schema') != SCHEMA or not pool.get('members'):
        raise ValueError(f'{path} is not a schema {SCHEMA} reference pool')
    pool['path'] = str(path)
    if verify:
        verify_pool(pool)
    return pool


def verify_pool(pool):
    """Raise when any checkpoint or the pool as a whole differs from what was rated."""
    directory = Path(pool['path']).parent
    for member in pool['members']:
        if member['kind'] != 'checkpoint':
            continue
        identity = checkpoint_identity(directory / member['path'])
        if identity['sha256'] != member['sha256'] or identity['metadata_sha256'] != member['metadata_sha256']:
            raise ValueError(f"Reference {member['name']} changed since the pool was rated")
    if pool_hash(pool['members']) != pool['pool_sha256']:
        raise ValueError('Reference pool members or ratings were edited after rating')


def pool_sha8(pool):
    return pool['pool_sha256'][:8]


def rating_dir(run_dir, pool):
    """<run>/reference/<pool_sha8>/ — the only place new rating files are written."""
    return Path(run_dir) / 'reference' / pool_sha8(pool)


def rating_name(checkpoint):
    return Path(checkpoint).stem + '.json'


def _read_rating(path, pool):
    try:
        payload = json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return None
    if payload.get('schema') != SCHEMA or payload.get('pool_sha256') != pool['pool_sha256']:
        return None
    return payload


def find_rating(run_dir, pool, checkpoint):
    """Return (payload, path) for a cached rating, preferring the sha8 directory.

    Falls back to the overnight-4 location <run>/reference/<stem>.json.
    path is always the sha8 location when nothing is cached (where to write).
    """
    name = rating_name(checkpoint)
    primary = rating_dir(run_dir, pool) / name
    payload = _read_rating(primary, pool)
    if payload:
        return payload, primary
    legacy = Path(run_dir) / 'reference' / name
    payload = _read_rating(legacy, pool)
    if payload:
        return payload, legacy
    return None, primary


def load_run_ratings(run_dir, pool):
    """{checkpoint stem: rating payload} for this pool, sha8 files overriding legacy."""
    run_dir = Path(run_dir)
    files = {}
    for directory in (run_dir / 'reference', rating_dir(run_dir, pool)):
        if not directory.is_dir():
            continue
        for path in directory.glob('*.json'):
            if path.parent != directory:
                continue
            payload = _read_rating(path, pool)
            if payload:
                files[path.stem] = payload
    return files


def cache_hit(previous, identity, pool, games):
    """True when previous is the same (pool, checkpoint sha, games), or a fuller band rating."""
    if not previous or previous.get('schema') != SCHEMA:
        return False
    if previous.get('pool_sha256') != pool['pool_sha256']:
        return False
    sha = (previous.get('checkpoint_identity') or {}).get('sha256')
    if identity and sha and sha != identity.get('sha256'):
        return False
    budget = previous.get('budget')
    if budget == games:
        return True
    if previous.get('source') == 'evaluation':
        return False
    n = budget if isinstance(budget, int) else previous.get('games')
    return isinstance(n, int) and n >= games


def rate_command(run_dir, pool_path=None):
    pool_path = pool_path or DEFAULT_POOL
    python = ROOT / 'native/artifacts/rl/venv/bin/python'
    script = ROOT / 'native/rl/sweep/reference_pool.py'
    return (f'BOPL_RL_BACKEND=rust {python} {script} rate-runs '
            f'--pool {pool_path} --run {run_dir} --workers 3')


def participants(pool, library=None):
    """Playable elo_ladder participants for every pool member, keyed by name."""
    directory = Path(pool['path']).parent
    result = {}
    for member in pool['members']:
        if member['kind'] == 'random':
            result[member['name']] = elo_ladder.Random()
        elif member['kind'] == 'scripted':
            result[member['name']] = elo_ladder.Scripted(member['name'])
        else:
            result[member['name']] = elo_ladder.Model(member['name'], directory / member['path'],
                                                      'anchor', library=library)
    return result


# ---- statistics ----------------------------------------------------------------

def expected_score(rating, opponent):
    return 1 / (1 + 10 ** ((opponent - rating) / 400))


def records_against(games, candidate):
    """{member: [wins, losses, draws]} from the candidate's side of play_games results."""
    records = {}
    for game in games:
        member = game['b'] if game['a'] == candidate else game['a']
        row = records.setdefault(member, [0, 0, 0])
        if game['outcome'] == 'draw':
            row[2] += 1
        elif (game['outcome'] == 'a') == (game['a'] == candidate):
            row[0] += 1
        else:
            row[1] += 1
    return records


def bootstrap_interval(games, candidate, ratings, samples=1000, seed=0, level=0.95):
    """Percentile CI of the fixed-pool rating, resampling side-swapped seed pairs
    (the two games of a pair share a seed and are not independent)."""
    clusters = {}
    for game in games:
        clusters.setdefault((game['a'], game['b'], game['map'], game['seed']), []).append(game)
    clusters = list(clusters.values())
    if len(clusters) < 2:
        rating = elo_ladder.bradley_terry_fixed(records_against(games, candidate), ratings)
        return [rating, rating]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        picks = rng.integers(len(clusters), size=len(clusters))
        resampled = [game for index in picks for game in clusters[index]]
        estimates.append(elo_ladder.bradley_terry_fixed(records_against(resampled, candidate), ratings))
    tail = (1 - level) / 2
    return [float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail))]


def pool_intervals(names, records, anchor='random', samples=200, seed=0, level=0.95):
    """Percentile CIs of round-robin ratings: each pair's W-L-D counts are
    resampled as a multinomial with the observed proportions (parametric
    bootstrap over games), then refit with elo_ladder.bradley_terry."""
    rng = np.random.default_rng(seed)
    estimates = {name: [] for name in names}
    for _ in range(samples):
        resampled = {}
        for pair, counts in records.items():
            total = sum(counts)
            resampled[pair] = list(rng.multinomial(total, np.asarray(counts) / total)) if total else [0, 0, 0]
        for name, value in elo_ladder.bradley_terry(names, resampled, anchor).items():
            estimates[name].append(value)
    tail = (1 - level) / 2
    return {name: [float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))]
            for name, values in estimates.items()}


def iqm(values):
    """Interquartile mean as rliable computes it (scipy trim_mean with 0.25):
    int(0.25 * n) values are cut from each end, so for n < 4 it is the mean."""
    values = sorted(float(v) for v in values)
    if not values:
        raise ValueError('IQM of no values')
    cut = int(0.25 * len(values))
    kept = values[cut:len(values) - cut]
    return sum(kept) / len(kept)


def iqm_interval(rows):
    """IQM of point ratings and of the per-rating 95% interval bounds."""
    score = iqm(row['rating'] for row in rows)
    return score, [iqm(row['interval'][0] for row in rows), iqm(row['interval'][1] for row in rows)]


def fit_rating(games, candidate, pool, bootstrap=1000, seed=0):
    """Fixed-member Bradley-Terry MLE + bootstrap CI from already-played games."""
    ratings = {m['name']: float(m['rating']) for m in pool['members']}
    records = records_against(games, candidate)
    rating = elo_ladder.bradley_terry_fixed(records, ratings)
    interval = bootstrap_interval(games, candidate, ratings, samples=bootstrap, seed=seed)
    return rating, interval, records, ratings


def rating_records(records, rating, ratings):
    return {member: dict(wins=w, losses=l, draws=d, games=w + l + d,
                         expected=expected_score(rating, ratings[member]),
                         observed=(w + d / 2) / (w + l + d) if w + l + d else 0.0)
            for member, (w, l, d) in sorted(records.items())}


def games_from_evaluation(report, pool):
    """play_games-shaped rows for eval games against pool members on pool maps."""
    members = {m['name'] for m in pool['members']}
    maps = set(pool.get('maps') or ())
    games = []
    for game in report.get('games') or ():
        if game.get('opponent') not in members:
            continue
        if maps and game.get('map') not in maps:
            continue
        if game.get('timeout'):
            outcome = 'draw'
        elif game.get('win'):
            outcome = 'a'
        else:
            outcome = 'b'
        games.append(dict(a=CANDIDATE, b=game['opponent'], map=game['map'],
                          seed=game['seed'], side=game.get('side', 0), outcome=outcome))
    return games


def rating_from_evaluation(report, pool, checkpoint, game=None, bootstrap=1000):
    """Fit a cheap in-run Elo from evaluate.py games (no extra play)."""
    games = games_from_evaluation(report, pool)
    if not games:
        return None
    rating, interval, records, ratings = fit_rating(
        games, CANDIDATE, pool, bootstrap=bootstrap, seed=int(report.get('seed_base') or 0))
    game_path = game_library(game)
    game_sha = sha256_file(game_path) if Path(game_path).is_file() else None
    pool_sha = (pool.get('game') or {}).get('sha256')
    return dict(
        schema=SCHEMA, source='evaluation', checkpoint=str(checkpoint),
        checkpoint_identity=report.get('checkpoint_identity'),
        pool=pool['path'], pool_sha256=pool['pool_sha256'], budget=len(games),
        maps=sorted({g['map'] for g in games}), seed_base=report.get('seed_base'),
        game=str(game_path), game_sha256=game_sha,
        game_mismatch=bool(game_sha and pool_sha and game_sha != pool_sha),
        rating=rating, interval=interval, bootstrap=bootstrap, games=len(games),
        draws=sum(1 for g in games if g['outcome'] == 'draw'),
        records=rating_records(records, rating, ratings),
        note='Bradley-Terry MLE vs fixed pool ratings from evaluate.py games on pool maps; '
             'timeouts = draws; one virtual draw per member; 95% percentile bootstrap over seed pairs')


def write_evaluation_rating(run_dir, report, pool, checkpoint, game=None):
    """Write <run>/reference/<sha8>/<stem>.json unless a fuller band rating exists."""
    result = rating_from_evaluation(report, pool, checkpoint, game=game)
    if not result:
        return None
    identity = result.get('checkpoint_identity') or {}
    previous, _ = find_rating(run_dir, pool, checkpoint)
    if previous and cache_hit(previous, identity, pool, result['budget']):
        return previous
    path = rating_dir(run_dir, pool) / rating_name(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, result)
    return result


@lru_cache(maxsize=4096)
def _sidecar_learner_steps(path, inode, modified, size):
    meta = json.loads(Path(path).read_text())
    if meta.get('learner_steps') is not None:
        return int(meta['learner_steps'])
    return None


def _checkpoint_learner_steps(run_dir, stem):
    """learner_steps from the checkpoint sidecar, else the policy-* step number."""
    sidecar = Path(run_dir) / 'checkpoints' / (stem + '.bin.json')
    try:
        stat = sidecar.stat()
        value = _sidecar_learner_steps(str(sidecar), stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if value is not None:
            return value
    except (FileNotFoundError, ValueError, OSError):
        pass
    try:
        return int(stem.split('-', 1)[1])
    except (ValueError, IndexError):
        return None


def _curve_row(stem, payload, learner_steps):
    return dict(
        checkpoint=stem + '.bin', learner_steps=learner_steps,
        rating=payload['rating'],
        interval=payload.get('interval') or [payload['rating'], payload['rating']],
        games=payload.get('games'), source=payload.get('source'),
        game_mismatch=payload.get('game_mismatch', False),
    )


def rated_curve(run_dir, pool):
    """Rated checkpoints in learner-step order from cached rating files (no eval JSON)."""
    run_dir = Path(run_dir)
    rows = []
    for stem, payload in load_run_ratings(run_dir, pool).items():
        if not stem.startswith('policy-'):
            continue
        learner_steps = _checkpoint_learner_steps(run_dir, stem)
        if learner_steps is None or payload.get('rating') is None:
            continue
        rows.append(_curve_row(stem, payload, learner_steps))
    rows.sort(key=lambda row: row['learner_steps'])
    return rows


def run_iqm(run_dir, pool, last_k=3):
    """IQM over the last k rated evaluation checkpoints, or None if none are rated."""
    rows = rated_curve(run_dir, pool)
    if not rows:
        return None
    window = rows[-max(1, int(last_k)):]
    score, interval = iqm_interval(window)
    return dict(score=score, interval=interval, last_k=window,
                game_mismatch=any(r.get('game_mismatch') for r in window))


# ---- candidate rating ----------------------------------------------------------

def candidate_tasks(candidate, member, maps, pairs, seed_base, start_pair=0):
    """`pairs` side-swapped seed pairs vs one member, cycling maps so every
    map gets an equal share; start_pair continues a member's sequence."""
    tasks = []
    for pair in range(start_pair, start_pair + pairs):
        map_id = maps[pair % len(maps)]
        seed = seed_base + pair // len(maps)
        tasks.append(dict(a=candidate, b=member, map=map_id, seed=seed, side=0))
        tasks.append(dict(a=candidate, b=member, map=map_id, seed=seed, side=1))
    return tasks


def plan_round(rating, ratings, remaining_games, band, round_games):
    """Seed pairs per member for the next round: every in-band member gets an
    equal share of min(round_games, remaining); with nobody in band the two
    members nearest 50% expected score keep the estimate anchored."""
    if remaining_games < 2:
        return {}
    in_band = [name for name, value in ratings.items()
               if band[0] <= expected_score(rating, value) <= band[1]]
    if not in_band:
        in_band = sorted(ratings, key=lambda name: abs(expected_score(rating, ratings[name]) - .5))[:2]
    pairs_total = min(round_games, remaining_games) // 2
    if pairs_total < 1:
        return {}
    per_member = max(1, pairs_total // len(in_band))
    plan = {}
    for name in in_band:
        if sum(plan.values()) + per_member > pairs_total:
            break
        plan[name] = per_member
    return plan


def rate_checkpoint(checkpoint, pool, out=None, games=240, first_pass_games=6, band=(.15, .85),
                    round_games=48, workers=4, seed_base=70001, maps=None, library=None,
                    game=None, bootstrap=1000, name=CANDIDATE, log=None):
    """Rate one checkpoint against the frozen pool; cached under out or the run's sha8 dir."""
    checkpoint = Path(checkpoint).resolve()
    identity = checkpoint_identity(checkpoint)
    maps = list(maps or pool['maps'])
    if out is None:
        if checkpoint.parent.name != 'checkpoints':
            raise ValueError('out is required when the checkpoint is not in a run directory')
        out = rating_dir(checkpoint.parent.parent, pool)
        previous, _ = find_rating(checkpoint.parent.parent, pool, checkpoint)
        if cache_hit(previous, identity, pool, games):
            return previous
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / (checkpoint.stem + '.json')
    previous = _read_rating(cache, pool)
    if previous is None and out.name == pool_sha8(pool):
        previous = _read_rating(out.parent / cache.name, pool)
    if cache_hit(previous, identity, pool, games):
        return previous
    game = game_library(game)
    game_sha256 = sha256_file(game)
    configure_games(out, game)
    # A pool member that is this very checkpoint (pool v6 anchors overnight-7
    # finals) would only contribute 50% games pulling the candidate toward its
    # own frozen rating; rate against the others.
    ratings = {m['name']: float(m['rating']) for m in pool['members']
               if m.get('sha256') is None or m['sha256'] != identity.get('sha256')}
    by_name = participants(pool, library=library)
    by_name[name] = elo_ladder.Model(name, checkpoint, 'candidate', library=library)
    limit = int(pool['episode_ticks'])
    delay = int(pool.get('observation_delay_ticks', 9))
    started = time.monotonic()
    played_pairs = {member: 0 for member in ratings}
    all_games = []
    rounds = []

    def play(plan):
        tasks = []
        for member, pairs in plan.items():
            tasks += candidate_tasks(name, member, maps, pairs, seed_base, played_pairs[member])
            played_pairs[member] += pairs
        all_games.extend(elo_ladder.play_games(by_name, tasks, limit, workers, seed_base, delay=delay))
        rating = elo_ladder.bradley_terry_fixed(records_against(all_games, name), ratings)
        rounds.append(dict(rating=rating, games=len(all_games), allocated={m: 2 * p for m, p in plan.items()}))
        if log:
            log(f'{checkpoint.name}: round {len(rounds)} {len(all_games)} games, rating {rating:.1f}')
        return rating

    first = max(1, first_pass_games // 2)
    rating = play({member: first for member in ratings})
    while True:
        plan = plan_round(rating, ratings, games - len(all_games), band, round_games)
        if not plan:
            break
        rating = play(plan)
    records = records_against(all_games, name)
    interval = bootstrap_interval(all_games, name, ratings, samples=bootstrap, seed=seed_base)
    result = dict(
        schema=SCHEMA, checkpoint=str(checkpoint), checkpoint_identity=identity,
        pool=pool['path'], pool_sha256=pool['pool_sha256'], budget=games, maps=maps,
        band=list(band), first_pass_games=first_pass_games, round_games=round_games,
        seed_base=seed_base, episode_ticks=limit, observation_delay_ticks=delay,
        game=str(game), game_sha256=game_sha256,
        game_mismatch=game_sha256 != pool.get('game', {}).get('sha256'),
        rating=rating, interval=interval, bootstrap=bootstrap, games=len(all_games),
        draws=sum(1 for g in all_games if g['outcome'] == 'draw'),
        source='band', records=rating_records(records, rating, ratings),
        rounds=rounds, elapsed=time.monotonic() - started,
        note='Bradley-Terry MLE vs fixed pool ratings; timeouts = draws; one virtual draw per member; '
             '95% percentile bootstrap over side-swapped seed pairs')
    atomic_json(cache, result)
    return result


# ---- pool construction ---------------------------------------------------------

def build_pool(out, scripted=(), checkpoints=(), maps=(0, 1, 2), games_per_pair=48, seed_base=50001,
               episode_ticks=7200, delay=9, workers=4, slow_workers=None, library=None, game=None, note='', log=None):
    """Full round robin over random + scripted + checkpoint members; writes pool.json.

    checkpoints: [(name, path)]; paths are recorded relative to the pool file.
    games_per_pair is the total per pair, split evenly over maps and sides.
    """
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'{out} exists; a pool is immutable, build a new file instead')
    maps = list(maps)
    if games_per_pair % (2 * len(maps)) or games_per_pair < 2 * len(maps):
        raise ValueError(f'games_per_pair must be a positive multiple of {2 * len(maps)} (sides x maps)')
    out.parent.mkdir(parents=True, exist_ok=True)
    game = game_library(game)
    configure_games(out.parent, game)
    members = [dict(name='random', kind='random')]
    members += [dict(name=name, kind='scripted') for name in scripted]
    by_name = {'random': elo_ladder.Random(), **{name: elo_ladder.Scripted(name) for name in scripted}}
    for name, path in checkpoints:
        path = Path(path).resolve()
        if name in by_name:
            raise ValueError(f'Duplicate pool member {name}')
        identity = checkpoint_identity(path)
        members.append(dict(name=name, kind='checkpoint', path=os.path.relpath(path, out.parent), **identity))
        by_name[name] = elo_ladder.Model(name, path, 'anchor', library=library)
    names = sorted(by_name)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    started = time.monotonic()
    games, timing = play_pairs(by_name, pairs, maps, games_per_pair, seed_base, episode_ticks, delay,
                               workers, slow_workers, log)
    records = elo_ladder.tally(games)
    ratings = elo_ladder.bradley_terry(names, records, 'random')
    intervals = pool_intervals(names, records)
    counts = tally_counts(names, records)
    for member in members:
        wins, losses, draws = counts[member['name']]
        member.update(rating=round(ratings[member['name']], 3), interval=intervals[member['name']],
                      games=wins + losses + draws, wins=wins, losses=losses, draws=draws)
    pool = dict(schema=SCHEMA, created=time.time(), note=note, game=dict(path=str(game), sha256=sha256_file(game)),
                maps=maps, games_per_pair=games_per_pair, seed_base=seed_base, episode_ticks=episode_ticks,
                observation_delay_ticks=delay, games=len(games), elapsed=time.monotonic() - started, timing=timing,
                fit='Bradley-Terry MLE, timeouts as draws, one virtual draw per pair, random anchored at 0 Elo',
                members=members,
                matches=[dict(a=a, b=b, wins_a=wa, wins_b=wb, draws=d) for (a, b), (wa, wb, d) in sorted(records.items())])
    pool['pool_sha256'] = pool_hash(members)
    atomic_json(out, pool)
    pool['path'] = str(out)
    return pool


def play_pairs(by_name, pairs, maps, games_per_pair, seed_base, episode_ticks, delay, workers, slow_workers, log):
    """Side-swapped fixed-seed games for every (a, b) pair, games_per_pair split
    evenly over maps and sides. Rollout-search bots (bopl_rl_fork, ~4 ms per
    decision) dominate the wall time; their games go through a separate, wider
    thread pool and are timed on their own."""
    from environment import Environment
    tasks = []
    for a, b in pairs:
        tasks += elo_ladder.pair_tasks(a, b, maps, games_per_pair // len(maps), seed_base)
    slow = {name for name, member in by_name.items()
            if isinstance(member, elo_ladder.Scripted) and Environment.bot_forks(member.bot[0])}
    slow_tasks = [task for task in tasks if task['a'] in slow or task['b'] in slow]
    fast_tasks = [task for task in tasks if task['a'] not in slow and task['b'] not in slow]
    if log:
        log(f'{len(by_name)} participants, {len(tasks)} games')
    timing = {}
    games = []
    for label, batch, count in (('fast', fast_tasks, workers), ('slow', slow_tasks, slow_workers or workers)):
        if not batch:
            continue
        phase = time.monotonic()
        games += elo_ladder.play_games(by_name, batch, episode_ticks, count, seed_base, delay=delay)
        timing[label] = dict(games=len(batch), workers=count, elapsed=time.monotonic() - phase, members=sorted(slow) if label == 'slow' else None)
        if log:
            log(f"{label} phase {len(batch)} games, {count} workers, {timing[label]['elapsed']:.0f} s")
    return games, timing


def tally_counts(names, records):
    """{name: [wins, losses, draws]} from a tally."""
    counts = {name: [0, 0, 0] for name in names}
    for (a, b), (wa, wb, d) in records.items():
        counts[a][0] += wa; counts[a][1] += wb; counts[a][2] += d
        counts[b][0] += wb; counts[b][1] += wa; counts[b][2] += d
    return counts


def excluded_rounds(path):
    if not path or not Path(path).is_file():
        return set()
    pairs = set()
    for item in json.loads(Path(path).read_text()):
        for round_id in item.get('rounds') or ():
            pairs.add((item['session'], int(round_id)))
    return pairs


def resolve_member(row, pool):
    """Map a human-game row onto a pool member name, or None."""
    names = {m['name'] for m in pool['members']}
    by_sha = {m['sha256']: m['name'] for m in pool['members'] if m.get('sha256')}
    agent = row.get('agent')
    if isinstance(agent, str) and agent in names:
        return agent
    if isinstance(agent, str) and agent.startswith('bot:') and agent[4:] in names:
        return agent[4:]
    sha = row.get('agent_sha256')
    if sha in by_sha:
        return by_sha[sha]
    b = row.get('b') or ''
    if b.startswith('bot:') and b[4:] in names:
        return b[4:]
    if b.startswith('policy:') and b[7:] in by_sha:
        return by_sha[b[7:]]
    return None


def load_human_rows(directory):
    """games.jsonl plus .session-*.jsonl kind=round rows, deduped on (session, round)."""
    directory = Path(directory)
    rows, seen = [], set()
    games_path = directory / 'games.jsonl'
    if games_path.is_file():
        for line in games_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seen.add((row.get('session'), row.get('round')))
            rows.append(row)
    for path in sorted(directory.glob('.session-*.jsonl')):
        session = path.name[len('.session-'):-len('.jsonl')]
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('kind') not in (None, 'round'):
                continue
            key = (session, row.get('round'))
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(row, session=session))
    return rows


def rate_human(pool, directory, excluded=None, bootstrap=1000, seed=0):
    """Matthew's Elo vs pool members with ratings held fixed.

    agent_won means the bot won; games.jsonl outcome 'a' is a human win (a=human).
    Games against non-members are dropped. Maps 0-123 vs the pool's 0-2.
    """
    directory = Path(directory)
    skip = excluded_rounds(directory / 'excluded.json' if excluded is None else excluded)
    games, maps, skipped = [], [], 0
    for row in load_human_rows(directory):
        if (row.get('session'), row.get('round')) in skip:
            continue
        member = resolve_member(row, pool)
        if not member:
            skipped += 1
            continue
        if 'agent_won' in row:
            if row.get('draw'):
                outcome = 'draw'
            elif row.get('agent_won'):
                outcome = 'b'
            else:
                outcome = 'a'
            map_id = int(row.get('level', row.get('map', 0)))
        else:
            outcome = row.get('outcome', 'draw')
            map_id = int(row.get('map', 0))
        maps.append(map_id)
        games.append(dict(a='human', b=member, map=map_id, seed=int(row.get('seed') or 0),
                          side=int(row.get('side') or 0), outcome=outcome))
    if not games:
        raise ValueError(f'No human games against pool members in {directory}')
    rating, interval, records, ratings = fit_rating(games, 'human', pool, bootstrap=bootstrap, seed=seed)
    pool_sha = (pool.get('game') or {}).get('sha256')
    return dict(
        schema=SCHEMA, source='human', name='human', pool=pool['path'], pool_sha256=pool['pool_sha256'],
        budget=len(games), maps=sorted(set(maps)), map_min=min(maps), map_max=max(maps),
        skipped_unknown=skipped, excluded=len(skip),
        game_mismatch=True,  # Unity, maps 0-123
        pool_game_sha256=pool_sha, rating=rating, interval=interval, bootstrap=bootstrap,
        games=len(games), draws=sum(1 for g in games if g['outcome'] == 'draw'),
        records=rating_records(records, rating, ratings),
        caveats=['The game rotates maps 0-123 while the pool plays 0-2.',
                 f'n={len(games)} games after exclusions.'],
        note='Matthew vs pool members; agent_won = bot won; maps rotate 0-123 while the pool plays 0-2; n is small')


def sweep_run_dirs(sweep_dir):
    sweep_dir = Path(sweep_dir)
    state_path = sweep_dir / 'sweep.json'
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        return [sweep_dir / trial['id'] for trial in state.get('trials') or []
                if (sweep_dir / trial['id']).is_dir()]
    return sorted(p for p in sweep_dir.iterdir() if p.is_dir() and (p / 'config.json').is_file())


def rate_runs(pool, runs, last_k=3, games=240, first_pass_games=6, band=(.15, .85),
              round_games=48, workers=4, seed_base=70001, log=None):
    """Band-rate the last k evaluation checkpoints of each run into the run dir."""
    import flags
    import scoring
    rows = []
    for run_dir in runs:
        run_dir = Path(run_dir)
        curve = scoring.reference_curve(run_dir, pool, last_k=last_k, games=games,
                                        first_pass_games=first_pass_games, band=band,
                                        round_games=round_games, workers=workers,
                                        seed_base=seed_base, log=log)
        if not curve:
            if log:
                log(f'{run_dir.name}: no evaluated checkpoints, skipped')
            continue
        summary = run_iqm(run_dir, pool, last_k=last_k)
        annotated = flags.annotate_run(run_dir, write=False)
        final = summary['last_k'][-1]
        row = dict(run=str(run_dir), name=run_dir.name, rating=final['rating'], interval=final['interval'],
                   games=final['games'], iqm=summary['score'], iqm_interval=summary['interval'],
                   last_k=[dict(checkpoint=r['checkpoint'], learner_steps=r['learner_steps'],
                                rating=r['rating'], interval=r['interval'], games=r['games'])
                           for r in summary['last_k']],
                   flags=annotated[-1][1]['flags'] if annotated else [],
                   game_mismatch=summary['game_mismatch'])
        rows.append(row)
        if log:
            log(f"{run_dir.name}: {row['rating']:.0f} [{row['interval'][0]:.0f}, {row['interval'][1]:.0f}]  "
                f"IQM{last_k} {row['iqm']:.0f}"
                + (f"; flags {', '.join(row['flags'])}" if row['flags'] else '')
                + ('  game_mismatch' if row['game_mismatch'] else ''))
    return rows


def format_table(members):
    lines = [f"{'rating':>8}  {'95% CI':>12}  {'kind':10}  {'W-L-D':>12}  name"]
    for member in sorted(members, key=lambda m: -m['rating']):
        low, high = member.get('interval', (member['rating'], member['rating']))
        wins, losses, draws = member.get('wins', 0), member.get('losses', 0), member.get('draws', 0)
        if 'wins' not in member and member.get('records'):
            wins = sum(r['wins'] for r in member['records'].values())
            losses = sum(r['losses'] for r in member['records'].values())
            draws = sum(r['draws'] for r in member['records'].values())
        lines.append(f"{member['rating']:8.1f}  [{low:5.0f},{high:5.0f}]  {member.get('kind', ''):10}  "
                     f"{wins:>4}-{losses}-{draws:<3}  {member['name']}")
    return '\n'.join(lines)


def format_run_table(rows, last_k=3):
    lines = [f"{'IQM'+str(last_k):>8}  {'95% CI':>12}  {'final':>8}  {'mismatch':8}  flags  name"]
    for row in sorted(rows, key=lambda r: -r['iqm']):
        low, high = row['iqm_interval']
        lines.append(f"{row['iqm']:8.1f}  [{low:5.0f},{high:5.0f}]  {row['rating']:8.1f}  "
                     f"{'yes' if row['game_mismatch'] else '':8}  {' '.join(row['flags']):20}  {row['name']}")
    return '\n'.join(lines)


def _band(value):
    return tuple(float(v) for v in value.split(','))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Round robin + BT fit -> pool.json')
    build.add_argument('--out', type=Path, required=True)
    build.add_argument('--scripted', type=scripted_names, default='',
                       help='Comma-separated scripted bots (kind-level)')
    build.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH')
    build.add_argument('--manifest', type=Path, help='Evaluation-opponents manifest (schema 2) to include')
    build.add_argument('--maps', default='0,1,2')
    build.add_argument('--games-per-pair', type=int, default=48)
    build.add_argument('--seed-base', type=int, default=50001)
    build.add_argument('--episode-ticks', type=int, default=7200)
    build.add_argument('--workers', type=int, default=4)
    build.add_argument('--slow-workers', type=int, default=None,
                       help='Threads for games involving rollout-search bots (default: --workers)')
    build.add_argument('--library', type=Path, default=None, help='CPU policy library (native_policy.load)')
    build.add_argument('--game', type=Path, default=None)
    build.add_argument('--note', default='')
    verify = commands.add_parser('verify', help='Check checkpoint hashes and the pool hash')
    verify.add_argument('pool', type=Path)
    rate = commands.add_parser('rate', help='Rate one checkpoint against a pool')
    rate.add_argument('--pool', type=Path, required=True)
    rate.add_argument('--checkpoint', type=Path, required=True)
    rate.add_argument('--out', type=Path, default=None,
                      help='Directory for the rating file (default: <run>/reference/<pool_sha8>/)')
    rate.add_argument('--games', type=int, default=240)
    rate.add_argument('--first-pass-games', type=int, default=6)
    rate.add_argument('--band', default='0.15,0.85')
    rate.add_argument('--workers', type=int, default=4)
    rate.add_argument('--seed-base', type=int, default=70001)
    rate.add_argument('--library', type=Path, default=None)
    rate.add_argument('--game', type=Path, default=None)
    runs = commands.add_parser('rate-runs', help='Rate run directories into <run>/reference/<pool_sha8>/')
    runs.add_argument('--pool', type=Path, required=True)
    runs.add_argument('--run', action='append', default=[], type=Path, help='Run directory (repeatable)')
    runs.add_argument('--sweep', type=Path, help='Sweep directory: rate every trial with a run dir')
    runs.add_argument('--last-k', type=int, default=3)
    runs.add_argument('--games', type=int, default=240)
    runs.add_argument('--first-pass-games', type=int, default=6)
    runs.add_argument('--band', default='0.15,0.85')
    runs.add_argument('--round-games', type=int, default=48)
    runs.add_argument('--workers', type=int, default=4)
    runs.add_argument('--seed-base', type=int, default=70001)
    human = commands.add_parser('human', help="Matthew's Elo vs the pool from human/mac games")
    human.add_argument('--pool', type=Path, required=True)
    human.add_argument('--dir', type=Path, default=DEFAULT_HUMAN)
    human.add_argument('--excluded', type=Path, default=None)
    args = parser.parse_args()
    log = lambda message: print(f'[reference {time.strftime("%H:%M:%S")}] {message}', flush=True)

    def named_checkpoints():
        checkpoints = []
        for item in args.checkpoint:
            name, separator, path = item.partition('=')
            if not separator:
                parser.error('--checkpoint needs NAME=PATH')
            checkpoints.append((name, path))
        return checkpoints
    if args.command == 'build':
        checkpoints = named_checkpoints()
        for item in opponents(args.manifest) if args.manifest else []:
            checkpoints.append((item['name'], item['path']))
        pool = build_pool(args.out, scripted=args.scripted, checkpoints=checkpoints,
                          maps=[int(m) for m in args.maps.split(',')], games_per_pair=args.games_per_pair,
                          seed_base=args.seed_base, episode_ticks=args.episode_ticks, workers=args.workers,
                          slow_workers=args.slow_workers, library=args.library, game=args.game, note=args.note, log=log)
        print(format_table(pool['members']))
        print(f"pool {pool['pool_sha256'][:16]}  {pool['games']} games in {pool['elapsed']:.0f}s -> {pool['path']}")
    elif args.command == 'verify':
        pool = load_pool(args.pool)
        print(format_table(pool['members']))
        print(f"verified pool {pool['pool_sha256']}")
    elif args.command == 'rate-runs':
        pool = load_pool(args.pool)
        directories = list(args.run)
        if args.sweep:
            directories += sweep_run_dirs(args.sweep)
        if not directories:
            parser.error('rate-runs needs --run and/or --sweep')
        lo, hi = _band(args.band)
        rows = rate_runs(pool, directories, last_k=args.last_k, games=args.games,
                         first_pass_games=args.first_pass_games, band=(lo, hi),
                         round_games=args.round_games, workers=args.workers,
                         seed_base=args.seed_base, log=log)
        print(format_run_table(rows, last_k=args.last_k))
    elif args.command == 'human':
        pool = load_pool(args.pool)
        result = rate_human(pool, args.dir, excluded=args.excluded)
        print(f"{result['rating']:.1f} Elo  95% CI [{result['interval'][0]:.1f}, {result['interval'][1]:.1f}]  "
              f"{result['games']} games ({result['draws']} draws)")
        for line in result['caveats']:
            print(f'  {line}')
        for member, row in sorted(result['records'].items(), key=lambda kv: -kv[1]['games']):
            print(f"  {member:20} {row['wins']:>3}-{row['losses']}-{row['draws']:<3} expected {row['expected']:.2f} observed {row['observed']:.2f}")
    else:
        pool = load_pool(args.pool)
        lo, hi = _band(args.band)
        result = rate_checkpoint(args.checkpoint, pool, args.out, games=args.games,
                                 first_pass_games=args.first_pass_games, band=(lo, hi), workers=args.workers,
                                 seed_base=args.seed_base, library=args.library, game=args.game, log=log)
        print(f"{result['rating']:.1f} Elo  95% CI [{result['interval'][0]:.1f}, {result['interval'][1]:.1f}]  "
              f"{result['games']} games ({result['draws']} draws)"
              + (f" in {result['elapsed']:.0f}s" if 'elapsed' in result else ''))
        for member, row in sorted(result['records'].items(), key=lambda kv: -kv[1]['games']):
            print(f"  {member:20} {row['wins']:>3}-{row['losses']}-{row['draws']:<3} expected {row['expected']:.2f} observed {row['observed']:.2f}")


if __name__ == '__main__':
    main()
