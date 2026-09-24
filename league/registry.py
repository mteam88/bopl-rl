"""The league: a permanent registry of members and the payoff between them.

Two committed files beside this module are the whole artifact:

  league.json   members, and the conditions payoff rows are keyed by
  payoff.jsonl  append-only aggregated win/loss/draw counts

A member is immutable once added. Its `id` is the sha256 of its weights for a
checkpoint, the bot name for a scripted bot, and `random` for the anchor, so a
member cannot change under its name without the id changing; `league_sha256`
hashes the member list so the file cannot be edited after it was written. Both
are re-checked on every load.

payoff.jsonl stores counts, never per-game rows. Bradley-Terry and Nash read
only counts, and a league that grows for months would otherwise commit millions
of lines of per-game detail to git permanently. Per-game detail is derived data
and belongs under native/artifacts/.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT / 'native/tools')]

import map_catalogue   # noqa: E402  (after sys.path)
import map_set   # noqa: E402

# Map sets up to this size are spelled out in a condition key; larger ones hash.
MAP_TAG_SPELLED = 8

LEAGUE = HERE / 'league.json'
PAYOFF = HERE / 'payoff.jsonl'
SIM = ROOT / 'native/target/release/libbopl_sim.so'
SCHEMA = 1
ANCHOR = 'random'
KINDS = ('main', 'exploiter', 'specialist', 'scripted', 'random', 'human')

# The one condition the league measures under, per simulator: every map, drawn at
# random (games.map_order); each seat on a kit its member plays (OWN_KIT); games
# long enough to be decided. The game starts sudden death at 7200 ticks (120 s at
# 60 Hz, GameSessionHandler TimeBeforeSuddenDeath) and it ends nearly every game
# by ~8000, so 10800 is a backstop. A 7200 limit scored 17.5% of games between
# strong members in tournament-20260917 as draws at the moment they were decided.
MAPS = 'all-124'
EPISODE_TICKS = 10800
DELAY = 9
# Each seat on a kit its member declares (kits), the two free to differ.
OWN_KIT = dict(mode='own')


# ---- conditions --------------------------------------------------------------

def loadout_tag(loadouts):
    """Short human name for a resolved loadout distribution (evaluation_protocol)."""
    if loadouts['mode'] == 'own':
        return 'own-kit'
    if loadouts['mode'] == 'set':
        return Path(loadouts['set_file']).stem
    if loadouts['mode'] == 'pool':
        return 'pool-' + '-'.join(str(a) for a in loadouts['pool'][:4])
    return 'fixed-' + '-'.join(str(a) for a in loadouts['fixed'])


def loadout_sha256(loadouts):
    """Fingerprint of a loadout distribution, independent of where it was read from.

    `set_file` is excluded: it is an absolute path into whichever checkout ran
    the measurement, so hashing it would make two identical conditions measured
    from two clones disagree, and `play`'s guard against one condition key naming
    two different loadout sets would fire on a move rather than on a change.
    Nothing is lost - `lineups` already holds the file's expanded contents, which
    is what actually decides who plays what.
    """
    content = {key: value for key, value in loadouts.items() if key != 'set_file'}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def map_tag(maps):
    """Short, unambiguous label for a map set: the named set, or the IDs, or a hash.

    The first form was the IDs concatenated with no separator, which is ambiguous
    the moment any ID reaches two digits - [1, 12] and [11, 2] both read "112" -
    and two conditions that collided on one key would be pooled, which is the one
    thing a condition exists to prevent. It also does not survive scale: the 111
    training maps would put a 250-character key on every payoff row.

    So: a set that is exactly one of native/data/map-sets.json's named sets is
    named (maptraining-111), a short set is spelled out with separators
    (map0-1-2), and anything larger is the sha256 of the canonical list
    (map8a1b2c3d) with the full list recorded beside it in the condition.
    """
    maps = sorted(int(m) for m in maps)
    for name, ids in map_catalogue.named_sets().items():
        if sorted(int(m) for m in ids) == maps:
            return 'map' + name
    if len(maps) <= MAP_TAG_SPELLED:
        return 'map' + '-'.join(str(m) for m in maps)
    return 'map' + hashlib.sha256(','.join(str(m) for m in maps).encode()).hexdigest()[:8]


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
    binding (train/bopl_agent.c, bopl_puffer.h) and states of the
    simulator "Nothing in native/src/rollout or sim/ moves". Games played
    either side of it under one library are therefore directly comparable, and
    keying them apart would split this matrix into halves that could never be
    pooled and would cost twice the CPU to fill.
    """
    maps = sorted(int(m) for m in maps)
    key = (f"sim{sim_sha256[:8]}.{loadout_tag(loadouts)}"
           f".{map_tag(maps)}.t{int(episode_ticks)}.d{int(delay)}")
    return key, dict(key=key, sim_sha256=sim_sha256, maps=maps,
                     episode_ticks=int(episode_ticks), delay=int(delay),
                     loadouts=loadouts, loadout_sha256=loadout_sha256(loadouts))


def rating_condition():
    """(key, detail) of the league's condition on the tree's simulator."""
    return condition(sha256_file(SIM), OWN_KIT, map_catalogue.parse_map_ids(MAPS), EPISODE_TICKS, DELAY)


# ---- members -----------------------------------------------------------------

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def member(name, kind, *, id, parent=None, target=None, origin=None, sim=None, reward=None,
           loadouts=None, added=None, path=None, metadata_sha256=None):
    """One immutable league member. Field order is the order it is written in.

    `parent` and `target` are different questions and neither substitutes for the
    other. `parent` is where the weights came from and is null for anything that
    started from random init, which is every exploiter here by design. `target`
    is the member the exploiter was trained to beat. Without it, "how long did
    this exploiter take to beat its target" has to guess from the schedule, and
    an even schedule makes that guess arbitrary.
    """
    if kind not in KINDS:
        raise ValueError(f'{kind} is not one of {KINDS}')
    if target is not None and kind not in ('exploiter', 'main'):
        raise ValueError(f'a {kind} member has no target; only exploiters and mains are aimed at one')
    row = dict(id=id, name=name, kind=kind, parent=parent, target=target, origin=origin, sim=sim,
               reward=reward, loadouts=loadouts, added=added)
    if path is not None:
        row['path'] = str(path)
    if metadata_sha256 is not None:
        row['metadata_sha256'] = metadata_sha256
    return row


def checkpoint_member(name, kind, weights, *, parent=None, target=None, origin=None, sim=None,
                      reward=None, loadouts=None, added=None):
    """A trained member, identified by the sha256 of the weights it froze at."""
    weights = Path(weights)
    sidecar = Path(str(weights) + '.json')
    return member(name, kind, id=sha256_file(weights), parent=parent, target=target, origin=origin,
                  sim=sim, reward=reward, loadouts=loadouts, added=added,
                  path=str(weights.relative_to(ROOT)),
                  metadata_sha256=sha256_file(sidecar) if sidecar.is_file() else None)


def kits(row):
    """The kits a member can meaningfully play, as 3-tuples; None means any kit.

    Read from the member's `loadouts`, which records what it was built for:
    `fixed` is one kit, `set` lists its `kits`, and null is kit-general (only the
    random anchor). A scripted bot is `fixed` 2,4,14 because bopl_bots.h indexes
    BOT_SLOT_DASH 0, BOT_SLOT_BOW 1, BOT_SLOT_PLATFORM 2 - handed any other kit it
    presses whatever sits in the slot it wants. A checkpoint plays what it trained
    on. Scheduling only ever pairs members on a kit both can play, so a game
    between two members is always a game both of them were built for.
    """
    loadouts = row.get('loadouts')
    if loadouts is None:
        return None
    if loadouts['mode'] == 'fixed':
        return [tuple(loadouts['fixed'])]
    return [tuple(kit) for kit in loadouts['kits']]


def plays_map(row, map_id):
    """Whether a member can meaningfully play a map.

    Scripted bots cannot play space maps: bopl_bots.h hard-codes grass physics,
    down to "death: water y < -11.3 (non-space maps)", and nothing in either bot
    file checks which kind of map it is on.
    """
    return row['kind'] != 'scripted' or map_id not in _space_maps()


@functools.cache
def _space_maps():
    return frozenset(map_set.space_maps())


def trained_loadouts(config):
    """A run's config.json `loadouts` as a member declaration (see kits).

    A run with no `loadouts` trained on DEFAULT_LOADOUT, the same reading
    evaluate.py makes. A `set` run declares every kit either seat was dealt; a
    free `pool` draw is not a kit list and is refused rather than read as "any".
    """
    from evaluation_protocol import DEFAULT_LOADOUT
    loadouts = config.get('loadouts')
    if not loadouts:
        return dict(mode='fixed', fixed=list(DEFAULT_LOADOUT))
    if loadouts['mode'] == 'fixed':
        return dict(mode='fixed', fixed=list(loadouts['fixed']))
    if loadouts['mode'] == 'set':
        halves = {tuple(row[i:i + 3]) for row in loadouts['lineups'] for i in (0, 3) if row[i:i + 3]}
        return dict(mode='set', kits=[list(kit) for kit in sorted(halves)])
    raise ValueError(f"cannot declare kits for a {loadouts['mode']!r} loadout draw")


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


def opponents(league, names=None, root=None, policy=None):
    """League members a training run can load as frozen opponents.

    Every member whose weights exist on disk, as dict(name, path, kind) in league
    order. Scripted members and the random anchor are excluded: they carry no
    weights, so a bank cannot be loaded with one (the environment plays them by
    kind and level instead). `names` restricts the set - that is how an exploiter
    aims at one member without a second mechanism.

    A member whose path is recorded but missing on disk is an error, not a
    silent omission: a run that quietly trained against a narrower population
    than its config named would be impossible to interpret afterwards.

    `policy` is the loading run's architecture (POLICY_SHAPE keys, as written in
    a checkpoint's .json sidecar). Every member must match it exactly or this
    raises. The bank holds a flat float32 array and the GPU reads it with the
    learner's own kernel and shapes, so a mismatch is not a load error: a 4.4 MB
    v4/256 checkpoint read into a 620 KB set/128 policy is accepted and plays at
    chance, and the run reports an even score against a member rated 592 Elo
    above random. Found exactly that way, 2026-09-24.
    """
    root = Path(root or ROOT)
    wanted = None if names is None else list(names)
    if wanted is not None:
        known = {m['name'] for m in league['members']}
        unknown = [n for n in wanted if n not in known]
        if unknown:
            raise ValueError(f'Not league members: {unknown}')
    out, missing = [], []
    for m in league['members']:
        if wanted is not None and m['name'] not in wanted:
            continue
        if not m.get('path'):
            continue
        path = Path(m['path'])
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            missing.append(f"{m['name']} -> {path}")
            continue
        out.append(dict(name=m['name'], path=str(path), kind=m['kind']))
    if missing:
        raise FileNotFoundError('League members have no weights on disk: ' + '; '.join(missing))
    if policy is not None:
        wrong = []
        for row in out:
            theirs = policy_shape(row['path'])
            differs = {key: (theirs.get(key), policy.get(key))
                       for key in POLICY_SHAPE if theirs.get(key) != policy.get(key)}
            if differs:
                wrong.append(f"{row['name']}: " + ', '.join(
                    f'{key} {mine!r} != run {ours!r}' for key, (mine, ours) in differs.items()))
        if wrong:
            raise ValueError('League members were trained with another policy architecture and '
                             'cannot be loaded into this run\'s banks: ' + '; '.join(wrong))
    if wanted is not None:
        empty = [n for n in wanted if not any(o['name'] == n for o in out)]
        if empty:
            raise ValueError(f'Named league members carry no weights and cannot be loaded: {empty}')
    return out


# What has to agree for one member's weights to run in another run's frozen bank.
POLICY_SHAPE = ('architecture', 'obs_size', 'hidden', 'num_layers', 'aim_fourier', 'aim_ar', 'v4_shape')


def policy_shape(path):
    """A checkpoint's architecture, from its .json sidecar. Defaults match the sidecar's."""
    meta = json.loads(Path(str(path) + '.json').read_text())
    return {key: meta.get(key, 0 if key in ('aim_fourier', 'aim_ar') else None) for key in POLICY_SHAPE}


def by_name(league):
    return {row['name']: row for row in league['members']}


def by_id(league):
    return {row['id']: row for row in league['members']}


def ids_for(league, names):
    """Member ids for display names, in the order given.

    Names address members at the CLI and on screen; ids address them in
    payoff.jsonl. Keep the two apart: a name is a mutable label, and the payoff
    log is append-only and outlives its members, so a name that was removed and
    later reused would silently inherit the old agent's record. A sha256 cannot.
    """
    rows = by_name(league)
    missing = [name for name in names if name not in rows]
    if missing:
        raise ValueError(f'not members of this league: {missing}')
    return [rows[name]['id'] for name in names]


# ---- payoff ------------------------------------------------------------------

def canonical(a, b, wins, losses, draws):
    """One ordered pair per matchup: a < b, counts from a's side.

    Both orderings of a matchup describe the same games from opposite sides, so a
    file containing both invites a reader to sum them and double-count. This is
    the single place rows are built, and it swaps (b, a) into (a, b) -- so
    payoff.jsonl never holds both orderings and a consumer may sum rows with
    matching (a, b, condition) without inspecting direction.

    Side-swapping is NOT this: every matchup is played from both seatings and
    both seatings are already inside these counts.
    """
    if a > b:
        return b, a, losses, wins, draws
    return a, b, wins, losses, draws


def append(rows, path=None):
    """Append aggregated counts in canonical (a < b) order.

    One row per (a, b, condition) per play session; the loader sums duplicates,
    so the file stays append-only and small.
    """
    path = Path(path or PAYOFF)
    with path.open('a') as handle:
        for row in rows:
            a, b, wins, losses, draws = canonical(row['a'], row['b'], row['wins'],
                                                  row['losses'], row['draws'])
            handle.write(json.dumps(dict(a=a, b=b, condition=row['condition'],
                                         wins=wins, losses=losses, draws=draws)) + '\n')


def read_payoff(path=None, condition=None):
    """{(a, b): [wins_a, losses_a, draws]} summed over rows, one condition only.

    `append` only ever writes the canonical a < b ordering, so a well-formed file
    holds one ordering per pair. This still folds (b, a) defensively, in case a
    row was hand-written or produced by an older writer.
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
