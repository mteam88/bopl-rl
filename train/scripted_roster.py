"""Elo-matchmade scripted-bot roster for train.py (--scripted-roster).

Names are the public `kind-level` strings in evaluation_protocol.
The default roster is the four v2 kinds at easy/medium/hard (12 entries). The
search bots are opt-in (`--scripted-roster default,search-medium,search-hard`):
at ~1.2 ms per decision they straggle the shared env pool and cut learner sps
by 84% even at 5-12% of the scripted games (2026-09-13, w128 b4). Anchor Elos
are pool-v5 ratings (random = 0); unrated levels start at that kind's hard
rating minus 75 per step down.
"""
from pathlib import Path

from evaluation_protocol import SCRIPTED_KINDS, SCRIPTED_LEVELS, scripted_opponent

DEFAULT_ROSTER = tuple(
    f'{kind}-{level}'
    for kind in ('archer-v2', 'pursuer-v2', 'evasive-v2', 'mixed-v2')
    for level in ('easy', 'medium', 'hard')
)
ROSTER_MAX = 32
LEVEL_GAP = 75.0
WEIGHT_FLOOR = 0.1
# pool-v5 scripted members (native/artifacts/rl/sweeps/anchors/pool-v5/pool.json).
# random is 0 on that scale, so these are the trainer offsets directly.
POOL_V5_RATINGS = {
    'archer-hard': 379.235,
    'pursuer-hard': 375.653,
    'evasive-hard': 428.505,
    'mixed-hard': 424.589,
    'mixed-medium': 349.295,
    'archer-v2-hard': 625.266,
    'pursuer-v2-hard': 632.449,
    'evasive-v2-hard': 651.931,
    'mixed-v2-hard': 598.377,
    'search-hard': 668.328,
    'search-medium': 660.603,
}
ANCHOR_POOL = Path(__file__).resolve().parents[1] / 'native/artifacts/rl/sweeps/anchors/pool-v5/pool.json'


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


def load_anchor_ratings(path=None):
    """Scripted name -> pool rating. File wins over the embedded pool-v5 table."""
    ratings = dict(POOL_V5_RATINGS)
    path = Path(path) if path is not None else ANCHOR_POOL
    if path.is_file():
        import json
        payload = json.loads(path.read_text())
        for member in payload.get('members', []):
            if member.get('kind') == 'scripted' and isinstance(member.get('rating'), (int, float)):
                ratings[member['name']] = float(member['rating'])
    return ratings


def initial_elo(name, ratings, elo_init=0.0):
    """elo_init + pool rating, or hard rating minus LEVEL_GAP per step for unrated levels."""
    if name in ratings:
        return float(elo_init) + float(ratings[name])
    kind_name, level_name, _, _ = split_name(name)
    hard = f'{kind_name}-hard'
    if hard not in ratings:
        raise ValueError(f'No pool-v5 rating for {name!r} or {hard!r}')
    steps = {'hard': 0, 'medium': 1, 'easy': 2}[level_name]
    return float(elo_init) + float(ratings[hard]) - LEVEL_GAP * steps


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


def rate_entry(entry_elo, learner_elo, wins, games, k):
    """One log-window Elo update, opponent side.

    Timeouts are not games: `games` is decided matches only. One K-update per
    log interval, not K per game — a window has dozens. The learner is the fixed
    origin (LEARNER_ELO); only entries move.
    """
    games = int(games)
    if games <= 0:
        return float(entry_elo)
    score_rate = float(wins) / games
    expected = win_prob(learner_elo, entry_elo)
    return float(entry_elo) - float(k) * (score_rate - expected)


def entries_from_names(names, ratings, elo_init=0.0):
    out = []
    for name in names:
        kind_name, level_name, kind, level = split_name(name)
        out.append(dict(name=name, kind_name=kind_name, level_name=level_name,
                        kind=kind, level=level, elo=initial_elo(name, ratings, elo_init),
                        games=0, wins=0))
    return out


def apply_window(entries, window, learner_elo, k):
    """Add decided (games, wins) per entry and rate each that played."""
    for entry, (games, wins) in zip(entries, window):
        games, wins = int(games), int(wins)
        if games < 0 or wins < 0 or wins > games:
            raise ValueError(f'Bad window counts games={games} wins={wins}')
        if games:
            entry['elo'] = rate_entry(entry['elo'], learner_elo, wins, games, k)
            entry['games'] += games
            entry['wins'] += wins
    return entries


def state_dict(entries, *, share, learner_elo):
    return dict(schema=1, share=float(share), learner_elo=float(learner_elo),
                entries=[dict(name=e['name'], elo=e['elo'], games=e['games'], wins=e['wins'])
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
    return entries
