"""Immutable opponent identities and side-swapped evaluation schedules."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from map_catalogue import MAP_COUNT,parse_map_ids

ABILITY_COUNT = 39
DEFAULT_LOADOUT = (2, 4, 14)
# ABILITY_PORT_GAPS.md: Time Stop still needs per-updatable clock coverage and
# Revival still lacks its full multi-actor model. Explicit pools may opt in.
DEFAULT_POOL_EXCLUSIONS = (12, 15)


def map_ids(value):
    """Map selection syntax shared by training, evaluation and sweep tools.

    Accepts comma-separated IDs, inclusive ranges (``0-7,120-123``), ``all``,
    and names from native/data/map-sets.json. Order is retained and duplicates
    are rejected.
    """
    return parse_map_ids(value)


@lru_cache(maxsize=1)
def ability_names():
    """{ability id: name} from the ripped catalogue the sim and the game share.

    Telemetry and loadouts are ability IDs everywhere; only reports resolve them,
    so this stays out of the hot paths.
    """
    rows=json.loads((Path(__file__).resolve().parents[3]/'native/data/rollout.json').read_text())['abilities']
    return {int(row['id']):row['name'] for row in rows}


def loadout_ids(value):
    """Exactly three ability IDs. Duplicate abilities are game-legal."""
    try:
        result=[int(item.strip()) for item in str(value).split(',')]
    except ValueError as error:
        raise ValueError('Expected three comma-separated ability IDs') from error
    if len(result)!=3 or any(ability<1 or ability>=ABILITY_COUNT for ability in result):
        raise ValueError(f'Expected three ability IDs between 1 and {ABILITY_COUNT-1}')
    return result


def loadout_pool(value):
    """Explicit IDs, ``all``, or ``all-<excluded IDs>``.

    ``all`` always excludes the partial abilities in ABILITY_PORT_GAPS.md;
    ``all-...`` adds exclusions. An explicit list may intentionally include
    those abilities.
    """
    text=str(value).strip().lower()
    if text == 'all' or text.startswith('all-'):
        excluded=set(DEFAULT_POOL_EXCLUSIONS)
        if text != 'all':
            try:
                excluded.update(int(item.strip()) for item in text[4:].split(',') if item.strip())
            except ValueError as error:
                raise ValueError('Expected all or all-<comma-separated ability IDs>') from error
        if any(ability<1 or ability>=ABILITY_COUNT for ability in excluded):
            raise ValueError(f'Excluded ability IDs must be between 1 and {ABILITY_COUNT-1}')
        result=[ability for ability in range(1,ABILITY_COUNT) if ability not in excluded]
    else:
        try:
            result=[int(item.strip()) for item in text.split(',')]
        except ValueError as error:
            raise ValueError('Expected ability IDs, all, or all-<excluded IDs>') from error
    if not result or len(set(result))!=len(result) or any(ability<1 or ability>=ABILITY_COUNT for ability in result):
        raise ValueError(f'Pool needs distinct ability IDs between 1 and {ABILITY_COUNT-1}')
    return result


def loadout_lineups(path):
    """Read explicit two-player lineups: three shared IDs or six player-ordered IDs per line."""
    rows=[]
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        text=raw.split('#',1)[0].strip()
        if not text:
            continue
        try:
            values=[int(item) for item in re.split(r'[\s,|/;]+',text) if item]
        except ValueError as error:
            raise ValueError(f'{path}:{number}: expected ability IDs') from error
        if len(values)==3:
            values=values*2
        if len(values)!=6 or any(ability<1 or ability>=ABILITY_COUNT for ability in values):
            raise ValueError(f'{path}:{number}: expected three shared or six player-ordered IDs in 1..{ABILITY_COUNT-1}')
        rows.append(values)
    if not rows:
        raise ValueError(f'{path}: no lineups')
    if len(rows)>256:
        raise ValueError(f'{path}: at most 256 lineups')
    return rows


def resolve_loadouts(fixed=DEFAULT_LOADOUT, pool=None, lineup_file=None, no_dup=False, mirror_share=0.0):
    fixed=list(fixed)
    if pool is not None and lineup_file is not None:
        raise ValueError('--loadout-pool and --loadout-set are mutually exclusive')
    if not 0 <= mirror_share <= 1:
        raise ValueError('--mirror-share must be in [0, 1]')
    rows=loadout_lineups(lineup_file) if lineup_file is not None else []
    if no_dup:
        candidates=rows or ([fixed] if pool is None else [])
        if any(len(set(row[:3]))<3 or (len(row)>3 and len(set(row[3:6]))<3) for row in candidates):
            raise ValueError('--loadout-no-dup conflicts with a fixed/set lineup containing duplicates')
        if pool is not None and len(pool)<3:
            raise ValueError('--loadout-no-dup needs at least three abilities in the pool')
    mode='set' if rows else ('pool' if pool is not None else 'fixed')
    return dict(mode=mode,fixed=fixed,pool=list(pool or []),lineups=rows,
                set_file=str(Path(lineup_file).resolve()) if lineup_file is not None else None,
                no_dup=bool(no_dup),mirror_share=float(mirror_share))


# Scripted opponents (native/rl/puffer/bopl_bots.h): hand-coded bots that read
# the true game state through the same C environment. Names are `kind-level`;
# the numeric IDs match BOPL_BOT_* in the header.
#
# Kind table — append-only (SCRIPTED_BOTS.md "Versioning"): a bot's behaviour
# is never changed under an existing name, so scores stay comparable across
# time; an improved bot is a new kind with a version suffix (`mixed-v2`) or a
# new family (`search`). The level is always the last token of the public
# name (`mixed-v2-hard`). Ids stay below 16 (train.py packs one per nibble).
SCRIPTED_KINDS={'archer':1,'pursuer':2,'evasive':3,'mixed':4,'search':5,
                'archer-v2':6,'pursuer-v2':7,'evasive-v2':8,'mixed-v2':9}
SCRIPTED_LEVELS={'easy':0,'medium':1,'hard':2}
SCRIPTED_NAMES=tuple(f'{kind}-{level}' for kind in SCRIPTED_KINDS for level in SCRIPTED_LEVELS)
assert len(set(SCRIPTED_KINDS.values()))==len(SCRIPTED_KINDS) and max(SCRIPTED_KINDS.values())<16


def scripted_opponent(name):
    """(kind_id, level_id) for a scripted bot name like 'mixed-hard'; None otherwise."""
    kind,_,level=name.rpartition('-')
    if kind in SCRIPTED_KINDS and level in SCRIPTED_LEVELS:
        return SCRIPTED_KINDS[kind],SCRIPTED_LEVELS[level]
    return None


# Pool-v5 members spanning the 15-85% band for current (~400 Elo) policies.
DEFAULT_EVAL_SCRIPTED = (
    'mixed-hard,mixed-v2-hard,archer-v2-hard,pursuer-v2-hard,evasive-v2-hard,search-medium')


def scripted_names(value):
    names=[item for item in value.split(',') if item]
    if len(set(names))!=len(names) or any(scripted_opponent(name) is None for name in names):
        raise ValueError(f'Scripted opponents must be distinct names from {SCRIPTED_NAMES}')
    return names


def checkpoint_identity(path):
    """Freeze both halves of a native checkpoint, without loading a model."""
    path = Path(path)
    if path.suffix != '.bin':
        raise ValueError('Native checkpoints must use .bin weights and a .bin.json sidecar')
    data = path.read_bytes()
    sidecar = Path(str(path) + '.json').read_bytes()
    meta = json.loads(sidecar)
    digest = hashlib.sha256(data).hexdigest()
    if meta.get('schema') != 1 or meta.get('backend') != 'pufferlib-native':
        raise ValueError('Unsupported native checkpoint metadata')
    if meta.get('weights_sha256') != digest:
        raise ValueError('Native checkpoint weights differ from metadata hash')
    return dict(sha256=digest, metadata_sha256=hashlib.sha256(sidecar).hexdigest())


def opponents(path):
    if path is None:
        return []
    path=Path(path).resolve()
    manifest=json.loads(path.read_text())
    if manifest.get('schema')!=2 or not manifest.get('opponents'):
        raise ValueError('Expected schema 2 and a nonempty native frozen opponent list')
    result=[]; names={'initial','random','selfplay',*SCRIPTED_NAMES}
    for item in manifest['opponents']:
        name=item['name']
        # A level suffix is reserved for scripted bots, present and future
        # (the dashboard labels any `<kind>-<level>` name "Scripted ...").
        if (not re.fullmatch(r'[a-z][a-z0-9-]{0,47}',name) or name in names
                or name.rpartition('-')[2] in SCRIPTED_LEVELS):
            raise ValueError('Frozen opponents need distinct safe names (no scripted-bot level suffix)')
        names.add(name)
        checkpoint=(path.parent/item['path']).resolve()
        identity = checkpoint_identity(checkpoint)
        if any(identity[key] != item.get(key) for key in identity):
            raise ValueError(f'Frozen opponent weights or metadata changed: {name}')
        result.append(dict(name=name,path=str(checkpoint),**identity))
    return result


def schedule(evaluation_maps,heldout_maps,frozen,games,heldout_games,frozen_games,
             seed_base,heldout_seed_base,selfplay_seed,scripted=()):
    if set(evaluation_maps)&set(heldout_maps):
        raise ValueError('Evaluation and held-out map lists overlap')
    if any(n<2 or n%2 for n in (games,heldout_games,frozen_games)):
        raise ValueError('Evaluation requires positive side-swapped seed pairs')
    tasks=[]; groups=[]
    for map_id in evaluation_maps:
        for opponent in ['random','initial',*scripted,*[item['name'] for item in frozen]]:
            count=frozen_games if any(opponent==item['name'] for item in frozen) else games
            groups.append((opponent,map_id))
            for game in range(count):
                name=f'{opponent}-{game}' if map_id==0 else f'{opponent}-map-{map_id}-{game}'
                tasks.append(dict(opponent=opponent,map=map_id,seed=seed_base+100000*map_id+game//2,
                                  side=game%2,save=game==0,name=name))
    for map_id in heldout_maps:
        for opponent in ['random',*scripted]:
            groups.append((opponent,map_id))
            prefix='' if opponent=='random' else f'{opponent}-'
            for game in range(heldout_games):
                tasks.append(dict(opponent=opponent,map=map_id,seed=heldout_seed_base+game//2,
                                  side=game%2,save=game==0,name=f'{prefix}map-{map_id}-{game}'))
    tasks.append(dict(opponent='selfplay',map=evaluation_maps[0],seed=selfplay_seed,side=0,save=True,name='selfplay'))
    return tasks,groups
