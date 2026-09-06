"""Original revival particle duplicate updates, travel, arrival and clone respawn."""
import argparse
import copy
import json
from functools import lru_cache
from actor_stream import ROOT,capture,verify,promote,golden,records
from verify_revival_marker import interactions as marker_queries
from verify_actors import ONE,vec
from revival_coverage import coverage
from revival_prefabs import effect_config, PREFAB as EFFECT

PREFAB='eee6dbb9aeac3284a9114099780e9e17'

@lru_cache
def config():
    assets=json.loads((ROOT/'native/reference-export/assets.json').read_text())
    parts={p['type']:p['properties'] for p in assets[PREFAB]['data']['components']}
    particle=parts['ReviveParticle'];animation=assets[particle['spawnEffect'].split('-')[2]]['data']['properties']
    key=next(f'animations.Array.data[{i}]' for i in range(animation['animations.Array.size']) if animation[f'animations.Array.data[{i}].name']=='spawn')
    sprites={};frames=[]
    for i in range(animation[key+'.animation.Array.size']):
        prefix=f'{key}.animation.Array.data[{i}]';sprite=animation[prefix+'.sprite']
        if sprite not in sprites:sprites[sprite]=len(sprites)
        frames.append(dict(sprite=sprites[sprite],duration=animation[prefix+'.frameDuration']))
    initial=parts['UnityEngine.SpriteRenderer']['m_Sprite']
    if initial not in sprites:sprites[initial]=len(sprites)
    return dict(speed=particle['speed.m_rawValue'],spawn=frames,initial_sprite=sprites[initial])

def queries(seed):
    for base in marker_queries(seed):
        if base['revival_case']!='interaction_clone_after':continue
        n=len(base['runtime']['scheduler']['players'])
        for mode in ['ordinary','null_slime','time_stop','protected','zero_speed','negative_speed','zero_dt','null_target','end_before_arrival','destroy_slime','zero_distance','exact_distance','moving_target','reinit','shared_marker','destroy_particle','null_slime_stopped','null_slime_protected','no_marker_particle','reinit_after_arrival','negative_dt']:
            q=copy.deepcopy(base);q['particle_case']=mode;r=q['runtime']
            r['spells']['prefabs'][PREFAB]=dict(kind='revive_particle',config=copy.deepcopy(config()))
            r['spells']['prefabs'][EFFECT]=dict(kind='revive_effect',config=copy.deepcopy(effect_config()))
            r['combat'].update(clone_prefab=copy.deepcopy(q['clone_prefab']),revive_effect=EFFECT)
            if mode=='zero_speed':r['spells']['prefabs'][PREFAB]['config']['speed']=0
            if mode=='negative_speed':r['spells']['prefabs'][PREFAB]['config']['speed']=-30*ONE
            if mode=='no_marker_particle':
                for key in r['spells']['prefabs']:
                    if r['spells']['prefabs'][key]['kind']=='marker':r['spells']['prefabs'][key]['config']['particle']=False
            if mode in ['protected','null_slime_protected']:
                for p in r['scheduler']['players']:p['protected']=True
            for a in q['actions']:
                a['commands']=[c for c in a.get('commands',[]) if c['op'] not in ['actor_spawn','destroy_actor']]
            q['actions'] += [copy.deepcopy(q['actions'][-1]) for _ in range(100)]
            for tick,action in enumerate(q['actions']):
                if tick==12:action['commands']=[dict(op='revive_particle',prefab=PREFAB,actor=i,target=None if mode=='null_target' else i,slime=None if mode.startswith('null_slime') else 2*n+i,clone_id=i,position=vec(-20,30)) for i in range(n)]
                if mode in ['zero_distance','exact_distance'] and tick==12:
                    for command in action['commands']:command['position']=vec(0 if mode=='zero_distance' else -60,30)
                    action['commands']=[dict(op='spell_command',spell=i,command=dict(kind='position',value=vec(0,30))) for i in range(n)]+action['commands']
                    if mode=='exact_distance':action['dt']=ONE
                if mode=='shared_marker' and tick==12:
                    for command in action['commands']:command['target']=0
                if mode=='moving_target' and tick==20:action['commands']=[dict(op='spell_command',spell=i,command=dict(kind='position',value=vec(40,30))) for i in range(n)]
                if mode in ['reinit','reinit_after_arrival'] and tick==(20 if mode=='reinit' else 60):action['commands']=[dict(op='revive_particle_init',actor=i,spell=n+i,target=i,slime=2*n+i,clone_id=100+i) for i in range(n)]
                if mode=='destroy_particle' and tick==20:action['commands']=[dict(op='spell_command',spell=n+i,command=dict(kind='destroy')) for i in range(n)]
                if mode=='zero_dt' and tick>=12:action['dt']=0
                if mode=='negative_dt' and 12<=tick<45:action['dt']=-71582788
                if mode in ['time_stop','protected','null_slime_stopped','null_slime_protected'] and 12<=tick<60:action['scale']=0
                if mode=='end_before_arrival' and tick==20:action['commands']=[dict(op='spell_command',spell=i,command=dict(kind='end')) for i in range(n)]
                if mode=='destroy_slime' and tick==20:action['commands']=[dict(op='destroy_actor',actor=2*n+i) for i in range(n)]
            yield q

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=725110);p.add_argument('--cases',type=int,default=0);p.add_argument('--directory',default='revive-particle');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');a=p.parse_args()
    fixture=ROOT/'native/fixtures/revive-particle.jsonl.xz'
    if a.golden:report=golden(fixture)
    else:
        qs=list(queries(a.seed));qs=qs[:a.cases] if a.cases else qs
        directory=ROOT/f'native/artifacts/{a.directory}'
        inputs,hashes=capture(directory,qs,a.reuse_unity);report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:
            try:report['revival_coverage']=coverage(directory,inputs,'particle')
            except ValueError as error:report['coverage_error']=str(error)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report and not a.cases:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(report,indent=2));return int(not report['match'] or 'coverage_error' in report)
if __name__=='__main__':raise SystemExit(main())
