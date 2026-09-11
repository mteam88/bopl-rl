"""Frozen reference pool: build it, verify it, rate checkpoints against it.

The pool is a fixed set of reference agents (the random policy, scripted bots,
frozen checkpoints) whose Elo ratings come from one full round robin fitted
by Bradley-Terry maximum likelihood (research/elo_ladder.py, random = 0 Elo,
timeouts = draws). Once written, pool.json is immutable: every checkpoint's
SHA-256 and a hash over (name, kind, sha256, rating) are recorded, and both
are re-checked whenever the pool is loaded. New references are appended by
building a new pool file; members are never removed or re-rated in place.

A candidate checkpoint is rated OpenAI Five style (Dota 2 paper, appendix J):
it plays pool members whose ratings stay fixed, and its own rating is the 1-D
MLE of those games (elo_ladder.bradley_terry_fixed). Game budget follows the
same recipe: a small first pass against everyone, then the remaining games go
to members whose expected win rate against the candidate is inside the band
(default 15-85%); lopsided matchups carry almost no rating information.
Uncertainty is a cluster bootstrap over side-swapped seed pairs (95% CI).
Aggregation over a trial's last k evaluations is the interquartile mean of
rliable (Agarwal et al. 2021, scipy.stats.trim_mean(x, 0.25)).

    python native/rl/sweep/reference_pool.py build --out DIR/pool.json \
        --scripted mixed-hard,archer-hard --checkpoint o3-012-final=DIR/o3-012-final.bin ...
    python native/rl/sweep/reference_pool.py verify DIR/pool.json
    python native/rl/sweep/reference_pool.py rate --pool DIR/pool.json --checkpoint policy.bin --out DIR2
"""
import argparse
import hashlib
import json
import math
import os
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


def iqm(values):
    """Interquartile mean as rliable computes it (scipy trim_mean with 0.25):
    int(0.25 * n) values are cut from each end, so for n < 4 it is the mean."""
    values = sorted(float(v) for v in values)
    if not values:
        raise ValueError('IQM of no values')
    cut = int(0.25 * len(values))
    kept = values[cut:len(values) - cut]
    return sum(kept) / len(kept)


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


def rate_checkpoint(checkpoint, pool, out, games=240, first_pass_games=6, band=(.15, .85),
                    round_games=48, workers=4, seed_base=70001, maps=None, library=None,
                    game=None, bootstrap=1000, name='candidate', log=None):
    """Rate one checkpoint against the frozen pool; cached in <out>/<checkpoint>.json."""
    checkpoint = Path(checkpoint).resolve()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = checkpoint_identity(checkpoint)
    maps = list(maps or pool['maps'])
    cache = out / (checkpoint.stem + '.json')
    if cache.exists():
        previous = json.loads(cache.read_text())
        if (previous.get('schema') == SCHEMA and previous.get('checkpoint_identity') == identity
                and previous.get('pool_sha256') == pool['pool_sha256']
                and previous.get('budget') == games and previous.get('maps') == maps
                and previous.get('band') == list(band) and previous.get('seed_base') == seed_base):
            return previous
    game = game_library(game)
    game_sha256 = sha256_file(game)
    configure_games(out, game)
    ratings = {m['name']: float(m['rating']) for m in pool['members']}
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
        records={member: dict(wins=w, losses=l, draws=d, games=w + l + d,
                              expected=expected_score(rating, ratings[member]),
                              observed=(w + d / 2) / (w + l + d))
                 for member, (w, l, d) in sorted(records.items())},
        rounds=rounds, elapsed=time.monotonic() - started,
        note='Bradley-Terry MLE vs fixed pool ratings; timeouts = draws; one virtual draw per member; '
             '95% percentile bootstrap over side-swapped seed pairs')
    atomic_json(cache, result)
    return result


# ---- pool construction ---------------------------------------------------------

def build_pool(out, scripted=(), checkpoints=(), maps=(0, 1, 2), games_per_pair=48, seed_base=50001,
               episode_ticks=7200, delay=9, workers=4, library=None, game=None, note='', log=None):
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
    tasks = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            tasks += elo_ladder.pair_tasks(a, b, maps, games_per_pair // len(maps), seed_base)
    started = time.monotonic()
    if log:
        log(f'pool: {len(names)} members, {len(tasks)} games on {game.name}')
    games = elo_ladder.play_games(by_name, tasks, episode_ticks, workers, seed_base, delay=delay)
    records = elo_ladder.tally(games)
    ratings = elo_ladder.bradley_terry(names, records, 'random')
    counts = {name: [0, 0, 0] for name in names}
    for (a, b), (wa, wb, d) in records.items():
        counts[a][0] += wa; counts[a][1] += wb; counts[a][2] += d
        counts[b][0] += wb; counts[b][1] += wa; counts[b][2] += d
    for member in members:
        wins, losses, draws = counts[member['name']]
        member.update(rating=round(ratings[member['name']], 3), games=wins + losses + draws,
                      wins=wins, losses=losses, draws=draws)
    pool = dict(schema=SCHEMA, created=time.time(), note=note, game=dict(path=str(game), sha256=sha256_file(game)),
                maps=maps, games_per_pair=games_per_pair, seed_base=seed_base, episode_ticks=episode_ticks,
                observation_delay_ticks=delay, games=len(games), elapsed=time.monotonic() - started,
                fit='Bradley-Terry MLE, timeouts as draws, one virtual draw per pair, random anchored at 0 Elo',
                members=members,
                matches=[dict(a=a, b=b, wins_a=wa, wins_b=wb, draws=d) for (a, b), (wa, wb, d) in sorted(records.items())])
    pool['pool_sha256'] = pool_hash(members)
    atomic_json(out, pool)
    pool['path'] = str(out)
    return pool


def format_table(pool):
    lines = [f"{'rating':>8}  {'kind':10}  {'W-L-D':>12}  name"]
    for member in sorted(pool['members'], key=lambda m: -m['rating']):
        lines.append(f"{member['rating']:8.1f}  {member['kind']:10}  "
                     f"{member['wins']:>4}-{member['losses']}-{member['draws']:<3}  {member['name']}")
    return '\n'.join(lines)


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
    build.add_argument('--library', type=Path, default=None, help='CPU policy library (native_policy.load)')
    build.add_argument('--game', type=Path, default=None)
    build.add_argument('--note', default='')
    verify = commands.add_parser('verify', help='Check checkpoint hashes and the pool hash')
    verify.add_argument('pool', type=Path)
    rate = commands.add_parser('rate', help='Rate one checkpoint against a pool')
    rate.add_argument('--pool', type=Path, required=True)
    rate.add_argument('--checkpoint', type=Path, required=True)
    rate.add_argument('--out', type=Path, required=True)
    rate.add_argument('--games', type=int, default=240)
    rate.add_argument('--first-pass-games', type=int, default=6)
    rate.add_argument('--band', default='0.15,0.85')
    rate.add_argument('--workers', type=int, default=4)
    rate.add_argument('--seed-base', type=int, default=70001)
    rate.add_argument('--library', type=Path, default=None)
    rate.add_argument('--game', type=Path, default=None)
    args = parser.parse_args()
    log = lambda message: print(f'[reference {time.strftime("%H:%M:%S")}] {message}', flush=True)
    if args.command == 'build':
        checkpoints = []
        for item in args.checkpoint:
            name, separator, path = item.partition('=')
            if not separator:
                parser.error('--checkpoint needs NAME=PATH')
            checkpoints.append((name, path))
        for item in opponents(args.manifest) if args.manifest else []:
            checkpoints.append((item['name'], item['path']))
        pool = build_pool(args.out, scripted=args.scripted, checkpoints=checkpoints,
                          maps=[int(m) for m in args.maps.split(',')], games_per_pair=args.games_per_pair,
                          seed_base=args.seed_base, episode_ticks=args.episode_ticks, workers=args.workers,
                          library=args.library, game=args.game, note=args.note, log=log)
        print(format_table(pool))
        print(f"pool {pool['pool_sha256'][:16]}  {pool['games']} games in {pool['elapsed']:.0f}s -> {pool['path']}")
    elif args.command == 'verify':
        pool = load_pool(args.pool)
        print(format_table(pool))
        print(f"verified pool {pool['pool_sha256']}")
    else:
        pool = load_pool(args.pool)
        lo, hi = (float(v) for v in args.band.split(','))
        result = rate_checkpoint(args.checkpoint, pool, args.out, games=args.games,
                                 first_pass_games=args.first_pass_games, band=(lo, hi), workers=args.workers,
                                 seed_base=args.seed_base, library=args.library, game=args.game, log=log)
        print(f"{result['rating']:.1f} Elo  95% CI [{result['interval'][0]:.1f}, {result['interval'][1]:.1f}]  "
              f"{result['games']} games ({result['draws']} draws) in {result['elapsed']:.0f}s")
        for member, row in sorted(result['records'].items(), key=lambda kv: -kv[1]['games']):
            print(f"  {member:20} {row['wins']:>3}-{row['losses']}-{row['draws']:<3} expected {row['expected']:.2f} observed {row['observed']:.2f}")


if __name__ == '__main__':
    main()
