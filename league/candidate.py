"""Rating a checkpoint against the league: how a training run is measured.

A candidate plays league members whose ratings stay fixed - fitted from
payoff.jsonl under registry.rating_condition - and its own rating is the 1-D
Bradley-Terry MLE of those games (OpenAI Five's frozen reference rating, Dota 2
paper appendix J). Games go where they carry information: a small first pass
against every member, then rounds against the members it has a 15-85% chance
against. The interval is a bootstrap over side-swapped seed pairs.

The games are the league's own (games.matchup): the candidate on the kits its
run trained on, each member on its own, on the condition's random maps. So the
number answers "how strong is this checkpoint on its real setup", and the file
carries a rating per kit beside it.

<run>/league/<checkpoint stem>.json keeps the games, not only the number. The
members' ratings move as the league grows, so `load` refits the stored games
against the current ratings; the stored rating is the one at play time.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'eval'), str(ROOT / 'native/tools')]

import registry  # noqa: E402
import rating  # noqa: E402
import games as league_games  # noqa: E402
import elo_ladder  # noqa: E402
import map_catalogue  # noqa: E402
from evaluation_protocol import ability_names, checkpoint_identity  # noqa: E402

SCHEMA = 1
CANDIDATE = 'candidate'
OUTCOME = {'a': 0, 'b': 1, 'draw': 2}


def reference(league=None):
    """(condition key, {member id: Elo}) fitted on the league's current games."""
    key, _ = registry.rating_condition()
    counts = registry.read_payoff(condition=key)
    known = registry.by_id(league or registry.load())
    ids = sorted({i for pair in counts for i in pair if i in known})
    if registry.ANCHOR not in ids:
        raise SystemExit(f'the league has no games under {key}; run `league play` first')
    return key, rating.ratings(ids, {pair: value for pair, value in counts.items()
                                     if pair[0] in known and pair[1] in known})['ratings']


def ratings_sha256(ratings):
    return hashlib.sha256(json.dumps(sorted((k, round(v, 3)) for k, v in ratings.items())).encode()).hexdigest()


# ---- statistics ----------------------------------------------------------------

def records(played):
    """{member id: [wins, losses, draws]} from the candidate's side."""
    out = {}
    for game in played:
        out.setdefault(game['b'], [0, 0, 0])[OUTCOME[game['outcome']]] += 1
    return out


def interval(played, ratings, samples=1000, seed=0, level=0.95):
    """Percentile interval, resampling side-swapped seed pairs (the two games of a
    pair share a seed and a map and are not independent)."""
    clusters = {}
    for game in played:
        clusters.setdefault((game['b'], game['map'], game['seed']), []).append(game)
    clusters = list(clusters.values())
    if len(clusters) < 2:
        value = elo_ladder.bradley_terry_fixed(records(played), ratings)
        return [value, value]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        picks = rng.integers(len(clusters), size=len(clusters))
        estimates.append(elo_ladder.bradley_terry_fixed(
            records([game for index in picks for game in clusters[index]]), ratings))
    tail = (1 - level) / 2
    return [float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail))]


def expected(value, opponent):
    return 1 / (1 + 10 ** ((opponent - value) / 400))


def own_kit(game):
    return game['kits'][:3] if game['side'] == 0 else game['kits'][3:]


def fit(played, ratings, bootstrap=1000, seed=0):
    """The candidate's rating from its games against members rated `ratings`, overall and per kit."""
    played = [game for game in played if game['b'] in ratings]

    def one(rows):
        counts = records(rows)
        value = elo_ladder.bradley_terry_fixed(counts, ratings)
        draws = {}
        for game in rows:
            if game['outcome'] == 'draw':
                draws[game['draw_kind']] = draws.get(game['draw_kind'], 0) + 1
        return dict(rating=value, interval=interval(rows, ratings, bootstrap, seed), games=len(rows),
                    draws=sum(draws.values()), draw_kinds=draws,
                    records={member: dict(wins=w, losses=l, draws=d, expected=expected(value, ratings[member]),
                                          observed=(w + d / 2) / (w + l + d))
                             for member, (w, l, d) in sorted(counts.items())})
    result = one(played)
    by_kit = {}
    for game in played:
        by_kit.setdefault(tuple(own_kit(game)), []).append(game)
    if len(by_kit) > 1:
        names = ability_names()
        result['kits'] = {','.join(map(str, kit)): dict(one(rows), label=' / '.join(names[a] for a in kit))
                          for kit, rows in sorted(by_kit.items())}
    return result


def iqm(values):
    """Interquartile mean as rliable computes it (scipy trim_mean 0.25): the mean below n = 4."""
    values = sorted(float(v) for v in values)
    if not values:
        raise ValueError('IQM of no values')
    cut = int(0.25 * len(values))
    kept = values[cut:len(values) - cut]
    return sum(kept) / len(kept)


def plan_round(value, ratings, remaining, band, round_games):
    """Seed pairs per member for the next round: an equal share of the round for
    every member in the band; with nobody in it, the two nearest an even game."""
    if remaining < 2:
        return {}
    in_band = [m for m, r in ratings.items() if band[0] <= expected(value, r) <= band[1]]
    if not in_band:
        in_band = sorted(ratings, key=lambda m: abs(expected(value, ratings[m]) - .5))[:2]
    pairs = min(round_games, remaining) // 2
    if pairs < 1:
        return {}
    each = max(1, pairs // len(in_band))
    plan = {}
    for member in in_band:
        if sum(plan.values()) + each > pairs:
            break
        plan[member] = each
    return plan


# ---- files ---------------------------------------------------------------------

def result_path(checkpoint):
    """<run>/league/<stem>.json for a checkpoint in a run's checkpoints/ directory."""
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent.name != 'checkpoints':
        raise ValueError(f'{checkpoint} is not in a run\'s checkpoints/ directory')
    return checkpoint.parent.parent / 'league' / (checkpoint.stem + '.json')


def read(path, key=None):
    """A stored rating under condition `key` (the current one when omitted), else None."""
    try:
        payload = json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return None
    if payload.get('schema') != SCHEMA or payload.get('condition') != (key or registry.rating_condition()[0]):
        return None
    return payload


def load(path, reference_ratings=None, bootstrap=1000):
    """A stored rating refitted on the members' current ratings; None if absent or another condition."""
    key, ratings = reference_ratings or reference()
    payload = read(path, key)
    if payload is None:
        return None
    if payload['reference_sha256'] != ratings_sha256(ratings):
        payload.update(fit(payload['played'], ratings, bootstrap, payload['seed_base']),
                       reference_sha256=ratings_sha256(ratings))
    return payload


def learner_steps(checkpoint):
    meta = json.loads(Path(str(checkpoint) + '.json').read_text())
    return meta.get('learner_steps')


def curve(run_dir, reference_ratings=None, bootstrap=1000):
    """Rated checkpoints of a run in learner-step order, refitted on the current ratings."""
    run_dir = Path(run_dir)
    reference_ratings = reference_ratings or reference()
    rows = []
    for path in sorted((run_dir / 'league').glob('policy-*.json')):
        payload = load(path, reference_ratings, bootstrap)
        checkpoint = run_dir / 'checkpoints' / (path.stem + '.bin')
        if payload is None or not checkpoint.is_file():
            continue
        rows.append(dict(payload, learner_steps=learner_steps(checkpoint)))
    return sorted(rows, key=lambda row: row['learner_steps'] or 0)


# ---- playing -------------------------------------------------------------------

def rate(checkpoint, games=240, first_pass_games=6, band=(.15, .85), round_games=48, workers=4,
         seed_base=70001, source='band', bootstrap=1000, log=print):
    """Rate one checkpoint against the league, reusing a stored rating of at least `games` games.

    source labels the record: 'in-run' for the trainer's small budget on every
    evaluation checkpoint, 'band' for the scorer's full one, which replaces it.
    """
    checkpoint = Path(checkpoint).resolve()
    identity = checkpoint_identity(checkpoint)
    path = result_path(checkpoint)
    league = registry.load()
    key, ratings = reference(league)
    previous = load(path, (key, ratings), bootstrap)
    if (previous and previous['checkpoint_identity']['sha256'] == identity['sha256']
            and previous['budget'] >= games):
        return previous
    config = json.loads(Path(str(checkpoint) + '.json').read_text()).get('config') or {}
    me = dict(id=CANDIDATE, name=CANDIDATE, kind='candidate', loadouts=registry.trained_loadouts(config))
    rows = registry.by_id(league)
    maps = map_catalogue.parse_map_ids(registry.MAPS)
    # Every rated member the candidate can meet, except a member that is this very
    # checkpoint: its games would only pull the candidate toward its own rating.
    used = {member: value for member, value in ratings.items()
            if member != identity['sha256'] and league_games.matchup(me, rows[member], maps, 1, seed_base)}
    built = {member: league_games.participant(rows[member]) for member in used}
    built[CANDIDATE] = elo_ladder.Model(CANDIDATE, checkpoint, 'candidate')
    league_games.configure(path.parent / 'workers')
    started, played, pairs, rounds = time.monotonic(), [], {member: 0 for member in used}, []

    def play(plan):
        tasks = []
        for member, count in plan.items():
            tasks += league_games.matchup(me, rows[member], maps, count, seed_base, pairs[member])
            pairs[member] += count
        for game in league_games.play(built, tasks, workers, seed_base):
            played.append(dict(b=game['b'], map=game['map'], seed=game['seed'], side=game['side'],
                               kits=game['kits'], outcome=game['outcome'], ticks=game['ticks'],
                               **({'draw_kind': game['draw_kind']} if 'draw_kind' in game else {})))
        value = elo_ladder.bradley_terry_fixed(records(played), used)
        rounds.append(dict(rating=value, games=len(played), allocated={m: 2 * c for m, c in plan.items()}))
        log(f'{checkpoint.name}: round {len(rounds)}, {len(played)} games, rating {value:.0f}')
        return value

    value = play({member: max(1, first_pass_games // 2) for member in used})
    while plan := plan_round(value, used, games - len(played), band, round_games):
        value = play(plan)
    result = dict(schema=SCHEMA, condition=key, checkpoint=str(checkpoint), checkpoint_identity=identity,
                  source=source, budget=games, seed_base=seed_base, band=list(band),
                  first_pass_games=first_pass_games, round_games=round_games,
                  reference_sha256=ratings_sha256(ratings),
                  **fit(played, used, bootstrap, seed_base),
                  names={member: rows[member]['name'] for member in used},
                  rounds=rounds, elapsed=time.monotonic() - started, played=played)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, allow_nan=False) + '\n')
    temporary.replace(path)
    return result
