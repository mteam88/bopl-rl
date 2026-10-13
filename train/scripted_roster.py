"""Elo-matchmade scripted-bot roster for train.py (--scripted-roster).

Names are the public `kind-level` strings in evaluation_protocol.
The default roster is the four v2 kinds at easy/medium/hard (12 entries). The
search bots are opt-in (`--scripted-roster default,search-medium,search-hard`):
at ~1.2 ms per decision they straggle the shared env pool and cut learner sps
by 84% even at 5-12% of the scripted games (2026-09-13, w128 b4). Entries
start at their league ratings (random = 0, the level an untrained learner plays
at) and move with the learner from there.
"""
from pathlib import Path
import sys

from evaluation_protocol import SCRIPTED_KINDS, SCRIPTED_LEVELS, scripted_opponent

DEFAULT_ROSTER = tuple(
    f'{kind}-{level}'
    for kind in ('archer-v2', 'pursuer-v2', 'evasive-v2', 'mixed-v2')
    for level in ('easy', 'medium', 'hard')
)
ROSTER_MAX = 32
WEIGHT_FLOOR = 0.1


def parse_roster(value):
    """Comma-separated kind-level names, or 'default', or empty. Distinct, <= ROSTER_MAX."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        names = [str(item) for item in value if str(item)]
    else:
        text = str(value).strip()
        if not text or text == 'none':
            return []
        names = []
        for item in text.split(','):
            if item == 'default':
                names.extend(DEFAULT_ROSTER)
            elif item:
                names.append(item)
    if len(names) > ROSTER_MAX:
        raise ValueError(f'At most {ROSTER_MAX} roster entries, got {len(names)}')
    if len(set(names)) != len(names):
        raise ValueError(f'Duplicate roster names: {names}')
    parsed = []
    for name in names:
        ids = scripted_opponent(name)
        if ids is None:
            raise ValueError(f'Unknown scripted roster name {name!r}')
        parsed.append(name)
    return parsed


def split_name(name):
    ids = scripted_opponent(name)
    if ids is None:
        raise ValueError(f'Unknown scripted roster name {name!r}')
    kind_name, _, level_name = name.rpartition('-')
    return kind_name, level_name, ids[0], ids[1]


def league_ratings():
    """Scripted name -> league Elo under the league's condition (league/candidate.reference)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'league'))
    import candidate
    _, ratings = candidate.reference()
    return ratings


def initial_elo(name, ratings):
    """The bot's league rating: relative to LEARNER_ELO, the fixed origin."""
    if split_name(name)[0] == 'uniform':
        return float(ratings.get('random', 0.0))  # the random anchor itself, pinned at 0 in the league
    if name not in ratings:
        raise ValueError(f'{name} has no league rating yet; run `league play` first')
    return float(ratings[name])


# The roster rates its bots RELATIVE TO THE LEARNER, with the learner fixed at
# the origin: every consumer below reads only win_prob's difference, so an
# entry's Elo is "how far above the current learner this bot is" and the whole
# drift lands on the entries. This number used to come from the self-play
# pool's online Elo -- the learner rated on games against frozen snapshots, then
# spent on scripted matchmaking, two conditions conflated into one rating.
# Cross-condition ratings are league/rating.py's job, measured post-hoc.
LEARNER_ELO = 0.0
# One K-update per log window, not per game: a window holds dozens of matches.
ELO_K = 16.0


def win_prob(learner_elo, entry_elo):
    """P(learner beats entry) on the pool's 400-point Elo scale."""
    return 1.0 / (1.0 + 10.0 ** ((float(entry_elo) - float(learner_elo)) / 400.0))


def draw_weights(entry_elos, learner_elo):
    """w_i = p_i (1-p_i) + 0.1/n. All positive so every entry keeps getting rated."""
    n = len(entry_elos)
    if n < 1:
        return []
    eps = WEIGHT_FLOOR / n
    return [win_prob(learner_elo, elo) * (1.0 - win_prob(learner_elo, elo)) + eps for elo in entry_elos]


def informativeness(entry_elos, learner_elo):
    """I = max_i 4 p (1-p): 1 at a coin flip, 0 when every entry is beaten or hopeless."""
    if not entry_elos:
        return 0.0
    best = 0.0
    for elo in entry_elos:
        p = win_prob(learner_elo, elo)
        best = max(best, 4.0 * p * (1.0 - p))
    return best


def scripted_share(entry_elos, learner_elo, share_min=0.0, share_max=1.0):
    """clamp(share_min, share_max * I)."""
    value = float(share_max) * informativeness(entry_elos, learner_elo)
    lo, hi = float(share_min), float(share_max)
    if lo > hi:
        lo, hi = hi, lo
    return min(hi, max(lo, value))


def rate_entry(entry_elo, learner_elo, score, games, k):
    """One log-window Elo update, opponent side.

    `score` is draw-aware from the learner's side — a win 1, a draw 0.5, a loss
    0 — and `games` is every match played, timeouts and mutual deaths included.
    Dropping undecided games from `games` instead would rate an entry only on
    the matches it managed to decide, and a bot that stalls the learner would
    read as never having been played. One K-update per log interval, not K per
    game: a window has dozens. The learner is the fixed origin (LEARNER_ELO);
    only entries move.
    """
    games = int(games)
    if games <= 0:
        return float(entry_elo)
    score_rate = float(score) / games
    expected = win_prob(learner_elo, entry_elo)
    return float(entry_elo) - float(k) * (score_rate - expected)


def entries_from_names(names, ratings):
    out = []
    for name in names:
        kind_name, level_name, kind, level = split_name(name)
        out.append(dict(name=name, kind_name=kind_name, level_name=level_name,
                        kind=kind, level=level, elo=initial_elo(name, ratings),
                        games=0, wins=0, draws=0))
    return out


def apply_window(entries, window, learner_elo, k):
    """Add (games, wins, draws) per entry and rate each that played.

    `games` counts every match, so `games - wins - draws` is losses. A draw is
    worth half a win in the rating and in the reported rate.
    """
    for entry, (games, wins, draws) in zip(entries, window):
        games, wins, draws = int(games), int(wins), int(draws)
        if games < 0 or wins < 0 or draws < 0 or wins + draws > games:
            raise ValueError(f'Bad window counts games={games} wins={wins} draws={draws}')
        if games:
            entry['elo'] = rate_entry(entry['elo'], learner_elo, wins + 0.5 * draws, games, k)
            entry['games'] += games
            entry['wins'] += wins
            entry['draws'] += draws
    return entries


def score_rate(entry):
    """Draw-aware score rate from the learner's side, or None if unplayed."""
    games = int(entry['games'])
    if not games:
        return None
    return (int(entry['wins']) + 0.5 * int(entry['draws'])) / games


def state_dict(entries, *, share, learner_elo):
    return dict(schema=1, share=float(share), learner_elo=float(learner_elo),
                entries=[dict(name=e['name'], elo=e['elo'], games=e['games'],
                              wins=e['wins'], draws=e['draws'])
                         for e in entries])


def restore_counts(entries, record):
    """Copy persisted Elo/games/wins onto a live entry list (same names, in order)."""
    saved = record.get('entries') or []
    if len(saved) != len(entries):
        raise ValueError(f'roster.json has {len(saved)} entries, run has {len(entries)}')
    for entry, row in zip(entries, saved):
        if row.get('name') != entry['name']:
            raise ValueError(f"roster.json order/name mismatch: {row.get('name')!r} vs {entry['name']!r}")
        entry['elo'] = float(row['elo'])
        entry['games'] = int(row['games'])
        entry['wins'] = int(row['wins'])
        # Rosters written before draws were counted carry none; their games
        # field held decided matches only, so 0 is the honest reading of it.
        entry['draws'] = int(row.get('draws', 0))
    return entries
