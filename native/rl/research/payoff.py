"""The league: members, the all-pairs payoff matrix, and what it says.

Reads `native/rl/league/` — `league.json` (members) and `payoff.jsonl`
(append-only aggregated counts, one row per ordered pair per condition).

The hard rule here is that counts are NEVER pooled across `condition` values.
A condition pins the simulator library hash, the reward semantics and the
loadout/map setup; mixing two of them is what once made a kit look like it won
92.6% of its games when it actually won 62.2% and traded mutual deaths in
another 30.4%. Every number this module returns belongs to exactly one
condition, and `conditions()` reports what else was on disk so the UI can say
the view is partial.

Scoring is draw-aware throughout: a win is 1, a draw is 0.5, a loss is 0.

Ratings come from `elo_ladder.bradley_terry_fit`, the same draw-aware fit with
the same `random`-anchored scale and the same one-virtual-draw regularizer the
rest of the project uses, so a league rating and a pool rating mean the same
thing. What is new here is everything a scalar rating cannot express: the Nash
mixture, the cycles, and the saturation signals.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import elo_ladder

NASH_ITERATIONS = 4000
# Below this many games a pair is drawn but never counted as a cycle edge: a
# 1-0 record is not evidence that A beats B.
CYCLE_MIN_GAMES = 20
CYCLE_LIMIT = 24
ANCHOR = 'random'
KINDS = ('main', 'exploiter', 'specialist', 'scripted', 'random')


def read(directory):
    """Members and payoff rows off disk. A missing league is an empty one."""
    directory = Path(directory)
    members = []
    raw = _read_json(directory / 'league.json')
    if isinstance(raw, dict):
        raw = raw.get('members', [])
    for entry in raw or []:
        if isinstance(entry, dict) and entry.get('id'):
            members.append(entry)
    rows = []
    payoff = directory / 'payoff.jsonl'
    if payoff.is_file():
        for line in payoff.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a torn last line while the writer appends
            if isinstance(row, dict) and row.get('a') and row.get('b'):
                rows.append(row)
    return members, rows


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _counts(row):
    return (int(row.get('wins') or 0), int(row.get('losses') or 0),
            int(row.get('draws') or 0))


def conditions(rows):
    """Every condition seen, most-played first, with its game count."""
    totals = defaultdict(int)
    for row in rows:
        wins, losses, draws = _counts(row)
        totals[row.get('condition') or 'unspecified'] += wins + losses + draws
    return [{'condition': name, 'games': games}
            for name, games in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))]


def mirrored(rows, condition):
    """True when every ordered pair has an exact mirror row.

    The contract says one row per *ordered* pair, which this module reads as
    "each game is recorded once, under the side it was played from" — so (a, b)
    and (b, a) are different games and their counts add. If the writer instead
    mirrors every game into both rows, that reading doubles every total. The
    two schemes are indistinguishable from a single row, so we look at the
    whole file: a perfect mirror everywhere is the signature of the second, and
    the UI says so rather than quietly reporting twice the games played.
    """
    seen = {}
    for row in rows:
        if (row.get('condition') or 'unspecified') != condition:
            continue
        seen[(row['a'], row['b'])] = _counts(row)
    if not seen:
        return False
    for (a, b), (wins, losses, draws) in seen.items():
        other = seen.get((b, a))
        if other is None or other != (losses, wins, draws):
            return False
    return True


def pairs(rows, condition):
    """{(a, b): [wins_a, wins_b, draws]} for one condition, both orientations.

    `payoff.jsonl` is append-only, so several rows may describe the same
    ordered pair; they sum. A row (b, a) is the same match-up seen from the
    other side, so its losses are a's wins.
    """
    tally = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        if (row.get('condition') or 'unspecified') != condition:
            continue
        a, b = row['a'], row['b']
        if a == b:
            continue
        wins, losses, draws = _counts(row)
        if not (wins + losses + draws):
            continue
        key = (a, b) if (a, b) in tally or (b, a) not in tally else (b, a)
        entry = tally[key]
        if key == (a, b):
            entry[0] += wins
            entry[1] += losses
        else:
            entry[0] += losses
            entry[1] += wins
        entry[2] += draws
    return dict(tally)


def matrix(members, rows, condition):
    """The all-pairs table for one condition."""
    ids = [member['id'] for member in members]
    index = {member_id: i for i, member_id in enumerate(ids)}
    folded = pairs(rows, condition)
    size = len(ids)
    cells = [[None] * size for _ in range(size)]
    for (a, b), (wins_a, wins_b, draws) in folded.items():
        i, j = index.get(a), index.get(b)
        if i is None or j is None:
            continue  # a result against something no longer in the league
        games = wins_a + wins_b + draws
        if not games:
            continue
        cells[i][j] = {'score': (wins_a + 0.5 * draws) / games, 'games': games,
                       'wins': wins_a, 'losses': wins_b, 'draws': draws}
        cells[j][i] = {'score': (wins_b + 0.5 * draws) / games, 'games': games,
                       'wins': wins_b, 'losses': wins_a, 'draws': draws}
    played = sum(1 for i in range(size) for j in range(size) if i != j and cells[i][j])
    possible = size * (size - 1)
    return {
        'condition': condition,
        'ids': ids,
        'cells': cells,
        'records': folded,
        'coverage': (played / possible) if possible else 0.0,
        'pairs': played // 2,
        'games': sum(value[0] + value[1] + value[2] for value in folded.values()),
        'mirrored': mirrored(rows, condition),
    }


def totals(cells):
    """Each member's overall score and game count in this condition."""
    out = []
    for row in cells:
        score = sum(cell['score'] * cell['games'] for cell in row if cell)
        games = sum(cell['games'] for cell in row if cell)
        out.append({'score': (score / games) if games else None, 'games': games,
                    'opponents': sum(1 for cell in row if cell)})
    return out


def ratings(ids, records):
    """Draw-aware Bradley-Terry with 95% intervals, anchored at `random`.

    Population-composition sensitive by construction: add five near-copies of
    one agent and everyone else's rating moves. That is exactly why the Nash
    scalar sits next to it in the UI.
    """
    if not ids:
        return {}, {}
    anchor = ANCHOR if ANCHOR in ids else ids[0]
    if not records:
        return {}, {}
    fit = elo_ladder.bradley_terry_fit(list(ids), records, anchor)
    return fit.get('ratings') or {}, fit.get('intervals') or {}


def nash(cells, played=None):
    """A maximum-entropy-ish Nash mixture of the population, and its values.

    The payoff matrix is made antisymmetric (`score - 0.5`), which turns the
    league into a symmetric zero-sum game. Averaged multiplicative-weights
    self-play converges to a Nash equilibrium of such a game; starting from the
    uniform mixture keeps it near the maximum-entropy one, which is the
    equilibrium worth reporting because it is the one that does not depend on
    an arbitrary choice among equivalent agents.

    Unlike Bradley-Terry this does not care how the population is composed:
    duplicating an agent splits its mass instead of re-weighting everyone. When
    the two disagree, the league is telling you something.

    Returns each member's mass in the mixture, and its expected score against
    the mixture back on the 0..1 scale.
    """
    total_size = len(cells)
    if not total_size:
        return [], []
    # Only members that have actually played. An empty cell is a zero
    # advantage, and a zero advantage in a symmetric zero-sum game reads as a
    # draw against everyone -- so an unplayed member is scored as an unbeatable
    # strategy and is handed mass for never having been tested. With 26 of 36
    # members unplayed that was 71% of the mixture sitting on agents with no
    # evidence at all. They come back None, which is what the UI expects.
    if played is None:
        played = [i for i, row in enumerate(cells) if any(row)]
    played = list(played)
    size = len(played)
    if not size:
        return [None] * total_size, [None] * total_size
    advantage = [[0.0] * size for _ in range(size)]
    for x, i in enumerate(played):
        for y, j in enumerate(played):
            cell = cells[i][j]
            if i != j and cell:
                advantage[x][y] = cell['score'] - 0.5
            # A missing cell between two members that have each played others
            # is still a zero here; the UI reports coverage so a sparse matrix
            # is visibly sparse.
    weights = [1.0 / size] * size
    accumulated = [0.0] * size
    for step in range(NASH_ITERATIONS):
        payoffs = [sum(advantage[i][j] * weights[j] for j in range(size))
                   for i in range(size)]
        peak = max(payoffs)
        updated = [weights[i] * math.exp(payoffs[i] - peak) for i in range(size)]
        total = sum(updated) or 1.0
        weights = [value / total for value in updated]
        if step >= NASH_ITERATIONS // 2:
            for i in range(size):
                accumulated[i] += weights[i]
    total = sum(accumulated) or 1.0
    mixture = [value / total for value in accumulated]
    values = [0.5 + sum(advantage[i][j] * mixture[j] for j in range(size))
              for i in range(size)]
    mixture_out = [None] * total_size
    values_out = [None] * total_size
    for x, i in enumerate(played):
        mixture_out[i] = mixture[x]
        values_out[i] = values[x]
    return mixture_out, values_out


def cycles(cells, ids, min_games=CYCLE_MIN_GAMES, limit=CYCLE_LIMIT):
    """Triples where A beats B beats C beats A.

    This is the thing a single number cannot say. A league can have a clean
    rating order and still be riddled with rock-paper-scissors underneath, and
    when it is, "who is best" is the wrong question to ask of it.
    """
    size = len(cells)
    beats = [[False] * size for _ in range(size)]
    margin = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            cell = cells[i][j]
            if i != j and cell and cell['games'] >= min_games and cell['score'] > 0.5:
                beats[i][j] = True
                margin[i][j] = cell['score'] - 0.5
    found = []
    for i in range(size):
        for j in range(i + 1, size):
            if not beats[i][j]:
                continue
            for k in range(i + 1, size):
                # i is the smallest index of the triple, so each cycle is
                # found once rather than three times.
                if k == j or not beats[j][k] or not beats[k][i]:
                    continue
                found.append({
                    'ids': [ids[i], ids[j], ids[k]],
                    'scores': [cells[i][j]['score'], cells[j][k]['score'], cells[k][i]['score']],
                    'strength': min(margin[i][j], margin[j][k], margin[k][i]),
                })
    found.sort(key=lambda item: -item['strength'])
    edges = sum(1 for i in range(size) for j in range(size) if beats[i][j])
    return {'count': len(found), 'edges': edges, 'top': found[:limit]}


def saturation(members, cells, rated):
    """Signs that the league has stopped telling us anything new.

    Two of them: the spread of win rates collapsing toward 0.5 (nobody can
    exploit anybody, so new members are not adding strategies), and the
    membership skewing toward one origin run (the league has become a record of
    the current learner rather than a spread of opponents).
    """
    scores = [row['score'] for row in totals(cells) if row['score'] is not None]
    spread = None
    if len(scores) > 1:
        mean = sum(scores) / len(scores)
        spread = math.sqrt(sum((value - mean) ** 2 for value in scores) / (len(scores) - 1))
    origins = defaultdict(int)
    for member in members:
        origin = member.get('origin') or {}
        if origin.get('run'):
            origins[origin['run']] += 1
    top_run, top_count = max(origins.items(), key=lambda kv: kv[1]) if origins else (None, 0)
    kinds = defaultdict(int)
    for member in members:
        kinds[member.get('kind') or 'unknown'] += 1
    values = [value for value in rated if value is not None]
    return {
        'spread': spread,
        'range': (max(scores) - min(scores)) if len(scores) > 1 else None,
        'rating_range': (max(values) - min(values)) if len(values) > 1 else None,
        'members': len(members),
        'top_run': top_run,
        'top_run_share': (top_count / len(members)) if members else None,
        'kinds': dict(kinds),
    }


def summary(members, rows, condition):
    """Everything the league page needs for one condition, in one object."""
    table = matrix(members, rows, condition)
    cells = table['cells']
    ids = table['ids']
    played = [i for i, row in enumerate(cells) if any(row)]
    rating, interval = ratings([ids[i] for i in played], table.pop('records'))
    weights, values = nash(cells, played)
    overall = totals(cells)
    table['members'] = [
        dict(member,
             rating=rating.get(member['id']),
             interval=interval.get(member['id']),
             nash_weight=weights[i] if i < len(weights) else None,
             nash_value=values[i] if i < len(values) else None,
             score=overall[i]['score'], games=overall[i]['games'],
             opponents=overall[i]['opponents'])
        for i, member in enumerate(members)
    ]
    table['rated'] = len(played)
    table['unrated'] = len(ids) - len(played)
    table['cycles'] = cycles(cells, ids)
    table['saturation'] = saturation(members, cells,
                                     [rating.get(member_id) for member_id in ids])
    table['anchor'] = ANCHOR if ANCHOR in ids else (ids[0] if ids else None)
    return table
