"""Original map-boundary deaths, actual water/death prefabs and scheduled cleanup."""
import argparse,copy,json,struct
from collections import Counter
from functools import lru_cache
from actor_stream import ROOT,capture,verify,promote,golden
from verify_actor_deaths import queries as death_queries
from verify_death_effects import prefab as death_prefab
from verify_actors import ONE,vec,f32

@lru_cache
def assets():
    source=json.loads((ROOT/'native/reference-export/assets.json').read_text())
    def components(guid):return {c['type']:c['properties'] for c in source[guid]['data']['components']}
    c=components('78aeacd5a5409014fa3c41329e0f093f');p=source['491ed5108ce2fdc4f8bedf9ea90cf736']['data']['properties'];sprites={};frames=[]
    for i in range(p['animations.Array.size']):
        k=f'animations.Array.data[{i}]'
        if p[k+'.name']!='enter':continue
        for j in range(p[k+'.animation.Array.size']):
            key=f'{k}.animation.Array.data[{j}]';sprite=p[key+'.sprite']
            if sprite not in sprites:sprites[sprite]=10000+len(sprites)
            frames.append(dict(sprite=sprites[sprite],duration=p[key+'.frameDuration']))
    water=dict(frames=frames,initial_sprite=sprites[c['UnityEngine.SpriteRenderer']['m_Sprite']])
    def effect(guid,kind,material):
        c=components(guid);bits=c['UnityEngine.Transform']['m_LocalScale.x']['f32']
        # FixTransform.position is nonserialized; prefab assets have not run
        # Awake. Unity's live field is zero despite a saved water offset of -1.47.
        # BoundPrefab checks this against the actual loaded prefab every capture.
        return dict(kind=kind,offset=vec(),scale=struct.unpack('<f',struct.pack('<I',bits&0xffffffff))[0],material=material)
    death=effect('b413ae31575d4db46bf682451f1b8dd5','death',1000000);down=effect('78aeacd5a5409014fa3c41329e0f093f','water',1000001)
    p=components('91b35711383743b4e9b73623dc72879b')['DestroyIfOutsideSceneBounds']
    bounds=dict(x=p['DestroyIfTooFarX'],y=p['DestroyIfTooFarY'],no_up=p['DoNotDestroyUp'],transform_present=True,x_effect=death,up_effect=death,down_effect=down,use_x_material=p['UsePlayerMaterialForXEffect'],use_up_material=p['UsePlayerMaterialForUpEffect'],use_down_material=p['UsePlayerMaterialForDownEffect'])
    return bounds,water

def bind(q):
    bounds,water=assets();r=q['runtime'];r['combat']['death_effects']=dict(prefab=death_prefab(),water_prefab=water,instances=[]);q['registration_state']=True;q['scripts']={}
    for i,a in enumerate(r['actors']):
        a['bounds']=copy.deepcopy(bounds);cid=len(r['components']);r['components'].append(dict(actor=i,role='bounds'))
        r['scheduler']['components'].append(dict(owner=a['entity']['node'],kind='updatable',present=True,destroyed=False,hierarchy=4,enabled=True,ignore_activity=False));r['scheduler']['live'].append(cid)
    for action in q['actions']:action['commands']=[]
    while len(q['actions'])<190:q['actions'].append(copy.deepcopy(q['actions'][-1]))
    for action in q['actions']:action['enabled']=[c['role']=='bounds' for c in r['components']]

def queries(seed):
    for base in death_queries(seed):
        if base['death_case']!='death':continue
        for mode in ['water','up','left','right','water_equal','up_equal','left_equal','right_equal','space_bottom','space_equal','priority','no_up','no_x','no_y','inactive','null_effect','null_transform','axes_off_null','time_stop','no_pc','water_material','death_material','water_x','multi_body','reviving']:
            q=copy.deepcopy(base);q['bounds_case']=mode;bind(q);r=q['runtime'];victim=len(r['actors'])-1;a=r['actors'][victim];b=a['bounds'];r['space']=mode.startswith('space_')
            x,y=0,f32(-11.3)-1
            if mode.startswith('up') or mode=='no_up':y=f32(72.5)+(0 if mode=='up_equal' else 1)
            if mode.startswith('left'):x,y=f32(-131.25)-(0 if mode=='left_equal' else 1),0
            if mode.startswith('right') or mode in ['no_x','water_x']:x,y=f32(131.25)+(0 if mode=='right_equal' else 1),0
            if mode=='water_equal':y=f32(-11.3)
            if mode.startswith('space_'):y=-26*ONE-(0 if mode=='space_equal' else 1)
            if mode=='priority':x=f32(131.25)+ONE
            if mode=='no_up':b['no_up']=True
            if mode=='no_x':b['x']=False
            if mode=='no_y':b['y']=False
            if mode=='inactive':a['entity']['active']=False
            if mode=='null_effect':b['down_effect']=None
            if mode in ['null_transform','axes_off_null']:b['transform_present']=False
            if mode=='axes_off_null':b.update(x=False,y=False)
            if mode=='no_pc':del a['collision']
            if mode=='multi_body':r['combat']['book']['players'][victim]['bodies_alive']=2
            if mode=='reviving':r['combat']['revival'][victim]['status']=2
            if mode=='water_material':b['use_down_material']=True
            if mode=='death_material':b['use_up_material']=False;y=f32(72.5)+1
            if mode=='water_x':b['x_effect']=copy.deepcopy(b['down_effect']);b['use_x_material']=False
            if mode=='time_stop':
                for action in q['actions'][:30]:action['scale']=0
            a['controller']['physics']['motion']['position']=dict(x=x,y=y)
            # The boundary checker reads FixTransform.Scale, independent of
            # whether PlayerPhysics or PlayerBody will run this frame.
            a['controller']['physics']['scale']=[ONE//4,ONE//2,ONE,2*ONE,3*ONE,ONE+12345][victim]
            yield q

def coverage(directory,qs):
    last={};counts=Counter()
    with (directory/'unity.jsonl').open() as source:
        for line in source:
            row=json.loads(line);state=row.get('state',row.get('result',{}).get('failed'))
            if state is not None:last[row['case']]=state
    for case,state in last.items():
        counts['effect_cases']+=bool(state['combat']['death_effects']['instances'])
        for e in state['combat']['death_effects']['instances']:
            counts['water_births' if e.get('kind')=='water' else 'death_births']+=1
            counts['removed']+=not e['present'];counts['present_at_end']+=e['present'];counts['splash_hits']+=len(e['splashes'])
        counts['final_deaths']+=len(state['combat']['events'])-len(qs[case]['runtime']['combat']['events'])
        counts['accounted_body_deaths']+=sum(a['bodies_alive']-b['bodies_alive'] for a,b in zip(qs[case]['runtime']['combat']['book']['players'],state['combat']['book']['players']))
    return dict(counts)

def interactions(seed):
    from verify_actor_spawn import queries as spawn_queries
    from verify_actor_deaths import handoffs
    for base in spawn_queries(seed):
        if base['spawn_case']!='permanent':continue
        for stopped in [False,True]:
            q=copy.deepcopy(base);q['bounds_case']='permanent_time_stop' if stopped else 'permanent';bind(q);r=q['runtime'];victim=len(r['actors'])-1;r['space']=False
            for i,a in enumerate(r['actors']):
                if i!=victim:a['bounds'].update(x=False,y=False)
            r['actors'][victim]['controller']['physics']['motion']['position']=dict(x=0,y=f32(-11.3)-1)
            for tick,action in enumerate(q['actions']):
                action['enabled']=[c['role'] in ['bounds','animation'] for c in r['components']]
                if stopped and tick<30:action['scale']=0
            yield q
    for base in handoffs(seed):
        if base['death_case'] not in ['dash','invulnerable_owner']:continue
        q=copy.deepcopy(base);q['bounds_case']='dash' if base['death_case']=='dash' else 'dash_invulnerable_owner';bind(q);r=q['runtime'];n=len(r['scheduler']['players']);victim=n-1
        for i,a in enumerate(r['actors']):
            if i!=n+victim:a['bounds'].update(x=False,y=False)
            if a.get('ordinary'):a['ordinary']['dash']['pause']=10*ONE
        # SyncPhysicsTo honors a grounded PlayerInfo by placing the dash on its
        # platform. Start airborne so every roster actually reaches the bound.
        r['actors'][victim]['controller']['physics']['motion'].update(position=dict(x=f32(131.25)+ONE,y=0),grounded=False,ground=None,local=0)
        for tick,action in enumerate(q['actions']):
            action['enabled']=[c['role'] in ['motor','ability','dash','bounds','animation'] for c in r['components']]
            for i,inp in enumerate(action['routing_inputs']):inp.update(can_use=True,buttons=[i==victim and tick==0,False,False])
        yield q

def failures(seed):
    for base in queries(seed):
        if base['bounds_case']!='water':continue
        for mode in ['water_renderer','up_renderer','camera','no_color_missing_renderer']:
            q=copy.deepcopy(base);q['bounds_case']=mode;r=q['runtime'];a=r['actors'][-1]
            if mode=='camera':r['combat']['camera_present']=False
            else:
                a['slime']['animator']['render']=None
                if mode=='up_renderer':a['controller']['physics']['motion']['position']=dict(x=0,y=f32(72.5)+1)
                elif mode=='water_renderer':a['bounds']['use_down_material']=True
            yield q

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=726001);p.add_argument('--golden',action='store_true');p.add_argument('--reuse-unity',action='store_true');group=p.add_mutually_exclusive_group();group.add_argument('--interactions',action='store_true');group.add_argument('--faults',action='store_true');p.add_argument('--directory');p.add_argument('--cases',type=int,default=0);a=p.parse_args()
    name='actor-bounds-failures' if a.faults else 'actor-bounds-interactions' if a.interactions else 'actor-bounds';fixture=ROOT/f'native/fixtures/{name}.jsonl.xz'
    if a.golden:result=golden(fixture)
    else:
        directory=ROOT/f'native/artifacts/{a.directory or name}';qs=list(failures(a.seed) if a.faults else interactions(a.seed) if a.interactions else queries(a.seed));qs=qs[:a.cases] if a.cases else qs
        inputs,hashes=capture(directory,qs,a.reuse_unity);result=dict(verify(directory,inputs),provenance=hashes)
        if result['match']:result['bounds_coverage']=coverage(directory,inputs)
        if result['match'] and a.interactions and result['bounds_coverage']['effect_cases']!=len(inputs):result['coverage_error']='Not every interaction created a boundary effect'
        (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['match'] and not a.cases and 'coverage_error' not in result:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(result,indent=2));return int(not result['match'] or 'coverage_error' in result)
if __name__=='__main__':raise SystemExit(main())
