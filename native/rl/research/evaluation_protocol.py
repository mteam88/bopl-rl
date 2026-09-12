"""Immutable opponent identities and side-swapped evaluation schedules."""
import hashlib
import json
from pathlib import Path
import re


def map_ids(value):
    maps=[int(item) for item in value.split(',')]
    if not maps or len(set(maps))!=len(maps) or any(m<0 or m>30 for m in maps):
        raise ValueError('Distinct map IDs between 0 and 30 required')
    return maps


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
