"""The league: a permanent registry of members and the payoff between them.

Two committed files beside this module are the whole artifact:

  league.json   members, and the conditions payoff rows are keyed by
  payoff.jsonl  append-only aggregated win/loss/draw counts

A member is immutable once added. Its `id` is the sha256 of its weights for a
checkpoint, the bot name for a scripted bot, and `random` for the anchor, so a
member cannot change under its name without the id changing; `league_sha256`
hashes the member list so the file cannot be edited after it was written. Both
are re-checked on every load. The precedent is sweep/reference_pool.py, which
froze a rated pool the same way.

payoff.jsonl stores counts, never per-game rows. Bradley-Terry and Nash read
only counts, and a league that grows for months would otherwise commit millions
of lines of per-game detail to git permanently. Per-game detail is derived data
and belongs under native/artifacts/.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'native/rl/research'), str(ROOT / 'native/tools')]

LEAGUE = HERE / 'league.json'
PAYOFF = HERE / 'payoff.jsonl'
SCHEMA = 1
ANCHOR = 'random'
KINDS = ('main', 'exploiter', 'specialist', 'scripted', 'random')


# ---- conditions --------------------------------------------------------------

def loadout_tag(loadouts):
    """Short human name for a resolved loadout distribution (evaluation_protocol)."""
    if loadouts['mode'] == 'set':
        return Path(loadouts['set_file']).stem
    if loadouts['mode'] == 'pool':
        return 'pool-' + '-'.join(str(a) for a in loadouts['pool'][:4])
    return 'fixed-' + '-'.join(str(a) for a in loadouts['fixed'])


def loadout_sha256(loadouts):
    return hashlib.sha256(json.dumps(loadouts, sort_keys=True).encode()).hexdigest()


def condition(sim_sha256, loadouts, maps, episode_ticks=7200, delay=9):
    """A measurement condition: everything that can change who wins a game.

    Its key names the simulator library, the loadout distribution, the map set,
    the episode limit and the observation delay. Counts from two conditions are
    never summed -- the port's fidelity fixes move win rates, so a library hash
    that differs is a different measurement, not more data on the same one.

    The reward function is deliberately NOT part of this key, although it is
    recorded on every member. Reward decides what a policy was trained to
    maximise, not who wins a game that is played: `play` reads the winner off
    the simulator's own end-of-episode flags and never consults a reward.
    f65c33b7 ("reward: stop paying for trades") changed only the training
    binding (native/rl/puffer/bopl_agent.c, bopl_puffer.h) and states of the
    simulator "Nothing in native/src/rollout or sim/ moves". Games played
    either side of it under one library are therefore directly comparable, and
    keying them apart would split this matrix into halves that could never be
    pooled and would cost twice the CPU to fill.
    """
    maps = sorted(int(m) for m in maps)
    key = (f"sim{sim_sha256[:8]}.{loadout_tag(loadouts)}"
           f".map{''.join(str(m) for m in maps)}.t{int(episode_ticks)}.d{int(delay)}")
    return key, dict(key=key, sim_sha256=sim_sha256, maps=maps,
                     episode_ticks=int(episode_ticks), delay=int(delay),
                     loadouts=loadouts, loadout_sha256=loadout_sha256(loadouts))


# ---- members -----------------------------------------------------------------

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def member(name, kind, *, id, parent=None, origin=None, sim=None, reward=None,
           loadouts=None, added=None, path=None, metadata_sha256=None):
    """One immutable league member. Field order is the order it is written in."""
    if kind not in KINDS:
        raise ValueError(f'{kind} is not one of {KINDS}')
    row = dict(id=id, name=name, kind=kind, parent=parent, origin=origin, sim=sim,
               reward=reward, loadouts=loadouts, added=added)
    if path is not None:
        row['path'] = str(path)
    if metadata_sha256 is not None:
        row['metadata_sha256'] = metadata_sha256
    return row


def checkpoint_member(name, kind, weights, *, parent=None, origin=None, sim=None,
                      reward=None, loadouts=None, added=None):
    """A trained member, identified by the sha256 of the weights it froze at."""
    weights = Path(weights)
    sidecar = Path(str(weights) + '.json')
    return member(name, kind, id=sha256_file(weights), parent=parent, origin=origin,
                  sim=sim, reward=reward, loadouts=loadouts, added=added,
                  path=str(weights.relative_to(ROOT)),
                  metadata_sha256=sha256_file(sidecar) if sidecar.is_file() else None)


def league_hash(members):
    """Content hash over the member list; any edit after writing changes it."""
    return hashlib.sha256(json.dumps(members, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(members, conditions, path=None):
    path = Path(path or LEAGUE)
    payload = dict(schema=SCHEMA, anchor=ANCHOR, members=list(members),
                   conditions=dict(conditions), league_sha256=league_hash(list(members)))
    temporary = Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=1, allow_nan=False) + '\n')
    temporary.replace(path)
    return payload


def load(path=None, verify=True):
    """Read league.json and re-check every hash it recorded.

    verify=False is for `seed`, which writes the file it would otherwise check.
    """
    path = Path(path or LEAGUE)
    league = json.loads(path.read_text())
    if league.get('schema') != SCHEMA:
        raise ValueError(f'{path} is not a schema {SCHEMA} league')
    league['path'] = str(Path(path).resolve())
    if verify:
        checked = verify_league(league)
        league['verified'] = checked
    return league


def verify_league(league):
    """Raise if the file was edited or any member's weights changed on disk.

    Returns {checked, weights_rehashed, missing} so a caller can report what was
    actually re-verified rather than asserting that something was.
    """
    if league_hash(league['members']) != league.get('league_sha256'):
        raise ValueError('league.json members were edited after they were written')
    rehashed, missing, names = 0, [], set()
    for row in league['members']:
        if row['name'] in names:
            raise ValueError(f"duplicate member name {row['name']}")
        names.add(row['name'])
        if not row.get('path'):
            continue
        weights = ROOT / row['path']
        if not weights.is_file():
            missing.append(row['name'])
            continue
        if sha256_file(weights) != row['id']:
            raise ValueError(f"member {row['name']} weights changed since it was added")
        sidecar = Path(str(weights) + '.json')
        if row.get('metadata_sha256') and sidecar.is_file() \
                and sha256_file(sidecar) != row['metadata_sha256']:
            raise ValueError(f"member {row['name']} metadata changed since it was added")
        rehashed += 1
    return dict(checked=len(league['members']), weights_rehashed=rehashed, missing=missing)


def by_name(league):
    return {row['name']: row for row in league['members']}


# ---- payoff ------------------------------------------------------------------

def append(rows, path=None):
    """Append aggregated counts. One row per (a, b, condition) per play session;
    the loader sums duplicates, so the file stays append-only and small."""
    path = Path(path or PAYOFF)
    with path.open('a') as handle:
        for row in rows:
            handle.write(json.dumps({k: row[k] for k in ('a', 'b', 'condition', 'wins', 'losses', 'draws')}) + '\n')


def read_payoff(path=None, condition=None):
    """{(a, b): [wins_a, losses_a, draws]} summed over rows, one condition only.

    Rows are ordered pairs as they were played; (b, a) is folded into (a, b) with
    wins and losses swapped, so a pair is one entry however it was scheduled.
    """
    path = Path(path or PAYOFF)
    counts = {}
    if not path.is_file():
        return counts
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if condition is not None and row['condition'] != condition:
            continue
        a, b = row['a'], row['b']
        wins, losses = row['wins'], row['losses']
        if a > b:
            a, b, wins, losses = b, a, losses, wins
        entry = counts.setdefault((a, b), [0, 0, 0])
        entry[0] += wins
        entry[1] += losses
        entry[2] += row['draws']
    return counts


def conditions_present(path=None):
    """{condition: rows} actually in the payoff file."""
    path = Path(path or PAYOFF)
    found = {}
    if not path.is_file():
        return found
    for line in path.read_text().splitlines():
        if line.strip():
            key = json.loads(line)['condition']
            found[key] = found.get(key, 0) + 1
    return found
