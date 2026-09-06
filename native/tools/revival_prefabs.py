"""Pinned revival effect prefab and animation asset contract."""
import json
from functools import lru_cache
from actor_stream import ROOT
PREFAB="ee3ca83d3a7c0264abd0950f3ec10ea7"
PARTICLE="eee6dbb9aeac3284a9114099780e9e17"

@lru_cache
def effect_config():
    assets=json.loads((ROOT/'native/reference-export/assets.json').read_text())
    parts={p['type']:p['properties'] for p in assets[PREFAB]['data']['components']}
    animation=assets[parts['ReviveEffect']['animData'].split('-')[2]]['data']['properties']
    sprites={};result=dict(particle=PARTICLE)
    for name in ['wisp','shoot']:
        key=next(f'animations.Array.data[{i}]' for i in range(animation['animations.Array.size']) if animation[f'animations.Array.data[{i}].name']==name)
        frames=[]
        for i in range(animation[key+'.animation.Array.size']):
            prefix=f'{key}.animation.Array.data[{i}]';sprite=animation[prefix+'.sprite']
            if sprite not in sprites:sprites[sprite]=len(sprites)
            frames.append(dict(sprite=sprites[sprite],duration=animation[prefix+'.frameDuration']))
        result[name]=frames
    initial=parts['UnityEngine.SpriteRenderer']['m_Sprite']
    if initial not in sprites:sprites[initial]=len(sprites)
    result['initial_sprite']=sprites[initial]
    return result
