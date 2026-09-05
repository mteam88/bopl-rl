"""Pinned MC animation frames and empty-player eating growth constant."""
import gzip
import json
from functools import lru_cache
from actor_stream import ROOT


@lru_cache
def assets():
    animations = {}
    for line in gzip.open(ROOT/'native/fixtures/animator-assets.jsonl.gz', 'rt'):
        q = json.loads(line).get('query')
        if not q:
            continue
        asset = q['asset']
        if (asset['path'].endswith('/MCAnimData.asset') and 'frame' not in asset
                and asset.get('animation') in ['bite', 'chew', 'swallow']):
            animations[asset['animation']] = q['actions'][0]['frames']
    assert len(animations) == 3
    source = json.loads((ROOT/'native/reference-export/assets.json').read_text())
    slime = next(c['properties'] for c in source['91b35711383743b4e9b73623dc72879b']['data']['components'] if c['type'] == 'SlimeController')
    return animations, slime['scaleIncreaseOnNom.m_rawValue']

