"""Actual deathEffect prefab birth, original animation callbacks and cleanup."""
import argparse,copy,json
from functools import lru_cache
from actor_stream import ROOT,capture,verify,promote,golden
from verify_actor_deaths import queries as death_queries,contacts,handoffs
from verify_actors import ONE

@lru_cache
def prefab():
    assets=json.loads((ROOT/'native/reference-export/assets.json').read_text())
    p=assets['77371654cd9e205439d519a3ddcccefe']['data']['properties']
    raw={}
    for i in range(p['animations.Array.size']):
        key=f'animations.Array.data[{i}]'
        raw[p[key+'.name']]=[(p[f'{key}.animation.Array.data[{j}].sprite'],p[f'{key}.animation.Array.data[{j}].frameDuration']) for j in range(p[key+'.animation.Array.size'])]
    sprites={};out={}
    for name in ['anticipation','start','end']:
        out[name]=[]
        for sprite,duration in raw[name]:
            if sprite not in sprites:sprites[sprite]=len(sprites)
            out[name].append(dict(sprite=sprites[sprite],duration=duration))
    c={v['type']:v['properties'] for v in assets['b413ae31575d4db46bf682451f1b8dd5']['data']['components']}
    e=c['DeathEffect2']
    out.update(initial_sprite=sprites[c['UnityEngine.SpriteRenderer']['m_Sprite']],gibbs=e['numberGibbs'],distance=e['gibbMaxDistance.m_rawValue'],hitstop=e['hitStopFrames'])
    return out

def queries(seed):
    bases=[q for q in death_queries(seed) if q['death_case']=='death']
    for base in bases:
        for mode in ['direct','time_stop','simultaneous','camera_fault']:
            q=copy.deepcopy(base);q['registration_state']=True;q['death_case']=mode;r=q['runtime'];r['combat']['death_effects']=dict(prefab=prefab(),instances=[])
            while len(q['actions'])<125:q['actions'].append(copy.deepcopy(q['actions'][-1]))
            request=q['actions'][5]['commands'][0]['request'];request['spawn_effect']=True
            if mode=='simultaneous':q['actions'][5]['commands']=[dict(op='kill_actor',actor=i,request=copy.deepcopy(request)) for i in range(len(r['actors']))]
            if mode=='camera_fault':r['combat']['camera_present']=False
            if mode=='time_stop':
                for i,a in enumerate(q['actions']):
                    if 4<=i<35:a['scale']=0
            yield q
    for q in handoffs(seed):
        q['registration_state']=True;q['death_case']='effect_'+q['death_case'];q['runtime']['combat']['death_effects']=dict(prefab=prefab(),instances=[])
        q['actions'][5]['commands'][0]['request']['spawn_effect']=True
        while len(q['actions'])<125:q['actions'].append(copy.deepcopy(q['actions'][-1]))
        yield q
    for base in contacts(seed):
        if base['death_case']!='missing_effect':continue
        base['registration_state']=True;base['death_case']='contact';base['runtime']['combat']['death_effects']=dict(prefab=prefab(),instances=[])
        while len(base['actions'])<105:base['actions'].append(copy.deepcopy(base['actions'][-1]))
        yield base

def coverage(directory,queries):
    last={};failed=set()
    with (directory/'unity.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if 'state' in row:last[row['case']]=row['state']
            elif isinstance(row.get('result'),dict) and 'failed' in row['result']:
                last[row['case']]=row['result']['failed'];failed.add(row['case'])
    total=dict(instances=0,removed=0,remaining_at_source_error=0,splash_hits=0,accounted_body_deaths=0,final_death_events=0,normal_contact_body_deaths=0)
    for case,state in last.items():
        initial=queries[case]['runtime']['combat'];combat=state['combat'];effects=combat['death_effects']['instances']
        total['instances']+=len(effects);total['removed']+=sum(not e['present'] for e in effects)
        total['remaining_at_source_error']+=sum(e['present'] for e in effects) if case in failed else 0
        total['splash_hits']+=sum(len(e['splashes']) for e in effects)
        deaths=sum(a['bodies_alive']-b['bodies_alive'] for a,b in zip(initial['book']['players'],combat['book']['players']))
        total['accounted_body_deaths']+=deaths;total['final_death_events']+=len(combat['events'])-len(initial['events'])
        if queries[case]['death_case']=='contact':total['normal_contact_body_deaths']+=deaths
    return total

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=724548);p.add_argument('--golden',action='store_true');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--directory',default='death-effects');p.add_argument('--cases',type=int,default=0);a=p.parse_args()
    fixture=ROOT/'native/fixtures/death-effects.jsonl.gz'
    if a.golden:result=golden(fixture)
    else:
        directory=ROOT/f'native/artifacts/{a.directory}';qs=list(queries(a.seed));qs=qs[:a.cases] if a.cases else qs
        inputs,hashes=capture(directory,qs,a.reuse_unity);result=dict(verify(directory,inputs),provenance=hashes)
        if result['match']:result['death_effect_coverage']=coverage(directory,inputs)
        (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['match'] and not a.cases:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(result,indent=2));return int(not result['match'])
if __name__=='__main__':raise SystemExit(main())
