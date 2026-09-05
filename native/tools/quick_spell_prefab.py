"""Pinned QuickSpell offsets and AirWave animations shared by Gust and Revival."""
import copy
import gzip
import json
from functools import lru_cache
from actor_stream import ROOT
from verify_invisibility import info


@lru_cache
def assets():
    animations = {}
    for line in gzip.open(ROOT/'native/fixtures/animator-assets.jsonl.gz', 'rt'):
        q = json.loads(line).get('query')
        if q and q['asset']['path'].endswith('/AirWave.asset') and 'frame' not in q['asset']:
            animations[q['asset']['animation']] = q['actions'][0]['frames']
    assert set(animations) == {'castAir', 'exit'}
    source = json.loads((ROOT/'native/reference-export/assets.json').read_text())
    result = {}
    for name, guid in [('gust','05feec16b8416224fb842a3774879cf4'),('revival','54cfd620d8ce77046b9051f0be61dfaa')]:
        parts = {c['type']: c['properties'] for c in source[guid]['data']['components']}
        c = parts['QuickSpell']
        assert c['spellFunc.m_PersistentCalls.m_Calls.Array.size'] == 0
        result[name] = dict(animations=copy.deepcopy(animations),
            enter_offset={k:c[f'EnterAbilityOffset.{k}.m_rawValue'] for k in ['x','y']},
            exit_offset={k:c[f'ExitAbilityOffset.{k}.m_rawValue'] for k in ['x','y']},
            shockwave_offset={k:c[f'shockwaveOffset.{k}.m_rawValue'] for k in ['x','y']},
            audio_enter=c['AudioToPlayOnEnter'], audio_cast=c['AudioToPlayOnCast'],
            spell=None, particle_prefab=True, particle_present=False,
            scale_particle_size=c['scaleParticleSize'], info=info(), player=None)
    return result
