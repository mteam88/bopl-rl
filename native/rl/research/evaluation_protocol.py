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


def opponents(path):
    if path is None:
        return []
    path=Path(path).resolve()
    manifest=json.loads(path.read_text())
    if manifest.get('schema')!=1 or not manifest.get('opponents'):
        raise ValueError('Expected schema 1 and a nonempty frozen opponent list')
    result=[]; names={'initial','random','selfplay'}
    for item in manifest['opponents']:
        name=item['name']
        if not re.fullmatch(r'[a-z][a-z0-9-]{0,47}',name) or name in names:
            raise ValueError('Frozen opponents need distinct safe names')
        names.add(name)
        checkpoint=(path.parent/item['path']).resolve()
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest!=item['sha256']:
            raise ValueError(f'Frozen opponent changed: {name}')
        result.append(dict(name=name,path=str(checkpoint),sha256=digest))
    return result


def schedule(evaluation_maps,heldout_maps,frozen,games,heldout_games,frozen_games,
             seed_base,heldout_seed_base,selfplay_seed):
    if set(evaluation_maps)&set(heldout_maps):
        raise ValueError('Evaluation and held-out map lists overlap')
    if any(n<2 or n%2 for n in (games,heldout_games,frozen_games)):
        raise ValueError('Evaluation requires positive side-swapped seed pairs')
    tasks=[]; groups=[]
    for map_id in evaluation_maps:
        for opponent in ['random','initial',*[item['name'] for item in frozen]]:
            count=games if opponent in ('random','initial') else frozen_games
            groups.append((opponent,map_id))
            for game in range(count):
                name=f'{opponent}-{game}' if map_id==0 else f'{opponent}-map-{map_id}-{game}'
                tasks.append(dict(opponent=opponent,map=map_id,seed=seed_base+100000*map_id+game//2,
                                  side=game%2,save=game==0,name=name))
    for map_id in heldout_maps:
        groups.append(('random',map_id))
        for game in range(heldout_games):
            tasks.append(dict(opponent='random',map=map_id,seed=heldout_seed_base+game//2,
                              side=game%2,save=game==0,name=f'map-{map_id}-{game}'))
    tasks.append(dict(opponent='selfplay',map=evaluation_maps[0],seed=selfplay_seed,side=0,save=True,name='selfplay'))
    return tasks,groups
