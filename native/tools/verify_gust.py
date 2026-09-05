"""Original Gust QuickSpell creates a real Shockwave in the composed scheduler."""
import argparse
import copy
import json
from functools import lru_cache
from collections import Counter
from actor_stream import ROOT, capture, verify, promote, golden, records
from verify_quick_spell import queries as quick_queries
from verify_actors import ONE, DT, vec

PREFAB='7c5ec508494ffd640870203e5c6a447e'


@lru_cache
def config():
    source=json.loads((ROOT/'native/reference-export/assets.json').read_text())
    p=next(c['properties'] for c in source[PREFAB]['data']['components'] if c['type']=='Shockwave')
    c=p['forceCurve01.Anim']
    curve=dict(pre_wrap=c['preWrap'],post_wrap=c['postWrap'],keys=[dict(time=k['time'],value=k['value'],in_tangent=k['inTangent'],out_tangent=k['outTangent']) for k in c['keys']])
    return dict(platform_force=p['platformForce.m_rawValue'],default_force=p['defaultForce.m_rawValue'],grounded_multiplier=p['groundedPlayerForceMultiplier.m_rawValue'],radius=p['radius.m_rawValue'],mask=p['collisionMask'],activate_on_start=p['ActivateOnStart'],every_frame=p['CallEveryFrame'],curve=curve,kill_after_frame=p['killAfterOnFrame'],gust=p['isGustAbility'])


def queries(seed):
    for base in quick_queries(seed):
        if base['quick_case'] not in ['gust','time_stop','protected','zero_dt','scales']:
            continue
        q=copy.deepcopy(base)
        q.update(gust_case=q['quick_case'],registration_state=True)
        r=q['runtime'];n=len(r['scheduler']['players'])
        r['spells']=dict(prefabs={PREFAB:dict(kind='shockwave',config=copy.deepcopy(config()))},instances=[])
        for actor in r['actors'][n:]:
            actor['ordinary']['quick_spell']['spell']=PREFAB
            actor['ordinary']['is_pushable']=False
        yield q


def contacts(seed):
    modes = ['airborne','grounded','inactive','disabled_collider','pushable_ability','unpushable_ability','time_stop','half_scale','double_scale']
    for base in queries(seed):
        if base['gust_case'] != 'gust':continue
        n=len(base['runtime']['scheduler']['players'])
        if n<2:continue
        for mode in modes:
            q=copy.deepcopy(base);q['gust_case']='contact_'+mode
            r=q['runtime'];q['victim']=1
            for i,actor in enumerate(r['actors']):
                motion=actor['controller']['physics']['motion'];motion.update(position=vec(-8+4*(i%n),2),velocity=vec(),external_velocity=vec(),grounded=mode=='grounded' and i==1,ground=None,rotation=0,right=vec(1,0),up=vec(0,1))
                actor['controller']['apply_gravity']=False;actor['entity']['body_simulated']=False
                if actor.get('slime'):actor['slime']['state']['suppress']=i!=0
                obj=next(o for o in r['scene']['objects'] if o['id']==actor['object']);obj['transform'].update(position=copy.deepcopy(motion['position']),offset=vec(),rotation=0,right=vec(1,0));obj['collider']['shape']['center']=copy.deepcopy(motion['position'])
            if mode in ['pushable_ability','unpushable_ability']:
                r['actors'][1]['entity']['active']=False;r['actors'][n+1]['entity']['active']=True
                r['actors'][n+1]['ordinary']['is_pushable']=mode=='pushable_ability';q['victim']=n+1
            if mode=='inactive':r['actors'][1]['entity']['active']=False
            if mode=='disabled_collider':
                next(o for o in r['scene']['objects'] if o['id']==r['actors'][1]['object'])['enabled']=False
            if mode in ['half_scale','double_scale']:r['scheduler']['players'][0]['scale']=ONE//2 if mode=='half_scale' else 2*ONE
            r['scheduler']['players'][0]['protected']=mode=='time_stop'
            for tick,action in enumerate(q['actions']):
                action['enabled']=[c['role'] in ['ability','animation'] or c['role']=='motor' and c['actor']==0 for c in r['components']]
                action['commands']=[]
                action['inputs']=[dict(direction=vec(),jump=False) for _ in range(n)]
                for i,v in enumerate(action['routing_inputs']):v.update(can_use=True,buttons=[i==0 and tick==0,False,False])
                if mode=='time_stop' and tick>0:action['scale']=0
                # Inactive noncasting QuickSpell bodies have no current animation.
                if mode in ['pushable_ability','unpushable_ability']:
                    for j,c in enumerate(r['components']):
                        if c['actor']==n+1 and c['role']=='animation':action['enabled'][j]=False
            yield q


def clones(seed):
    from verify_actor_clones import queries as clone_queries
    bases={len(q['runtime']['actors']):q for q in clone_queries(seed) if q['clone_case']=='before'}
    for base in queries(seed):
        if base['gust_case']!='gust':continue
        n=len(base['runtime']['scheduler']['players']);cloned=bases[n]
        for mode in ['capacity','death_and_recast','pending_removal']:
            q=copy.deepcopy(base);q.update(gust_case='clone_'+mode,dynamic_actors=True,clone_prefab=copy.deepcopy(cloned['clone_prefab']))
            r=q['runtime'];r['combat']=copy.deepcopy(cloned['runtime']['combat']);count=16 if mode=='capacity' else 2
            for i,actor in enumerate(r['actors']):
                actor['entity'].update(present=True,destroyed=False)
                actor['collision']=dict(invulnerable=0,was_killed=False,death_trauma=.5)
                motion=actor['controller']['physics']['motion'];motion.update(position=vec(-8+4*(i%n),20),velocity=vec(),external_velocity=vec(),grounded=False,ground=None)
                actor['controller']['apply_gravity']=False;actor['entity']['body_simulated']=False
                if actor.get('slime'):actor['slime']['managed_destroyed']=False
                obj=next(o for o in r['scene']['objects'] if o['id']==actor['object']);obj['transform']['position']=copy.deepcopy(motion['position']);obj['collider']['shape']['center']=copy.deepcopy(motion['position'])
            q['actions']=q['actions'][:60]
            for tick,action in enumerate(q['actions']):
                action['enabled']=[c['role'] in ['ability','animation'] or c['role']=='motor' and c['actor']==0 for c in r['components']]
                action['commands']=[];action['inputs']=[dict(direction=vec(),jump=False) for _ in range(n)]
                for i,v in enumerate(action['routing_inputs']):v.update(can_use=True,buttons=[i==0 and tick==20,False,False])
            births=[]
            for player in range(n):
                for j in range(count):
                    k=player*count+j
                    births.append(dict(op='clone_actor',actor=player,player=player,position=vec(-15+(k%16)*1.5,18+(k//16)*1.5)))
            q['actions'][1]['commands']=births
            q['actions'][3]['commands']=[dict(op='actor_spawn',actor=2*n+i,action='fast',position=copy.deepcopy(birth['position'])) for i,birth in enumerate(births) if not (i==0 and mode=='pending_removal')]
            if mode=='pending_removal':q['actions'][1]['commands'].append(dict(op='destroy_actor',actor=2*n))
            if mode=='death_and_recast':
                q['actions'][30]['commands']=[dict(op='kill_actor',actor=2*n,request=dict(killer=1,spawn_effect=True,ignore_invulnerability=True,cause=0)),dict(op='quick_cast',actor=n)]
            q['actions'][40]['commands']=[dict(op='destroy_actor',actor=2*n+i) for i in range(n*count) if not (i==0 and mode!='capacity')]
            q['expected_births']=n*count
            yield q


def custom(seed):
    modes=['single_activation','kill_after_frame','inactive_start_flag','empty_curve','single_key_curve','shifted_curve','null_particle_after_spawn','missing_exit_after_spawn']
    for base in queries(seed):
        if base['gust_case']!='gust':continue
        for mode in modes:
            q=copy.deepcopy(base);q['gust_case']='custom_'+mode;r=q['runtime'];c=r['spells']['prefabs'][PREFAB]['config'];n=len(r['scheduler']['players'])
            if mode=='single_activation':c['every_frame']=False
            elif mode=='kill_after_frame':c['kill_after_frame']=True
            elif mode=='inactive_start_flag':c['activate_on_start']=False
            elif mode=='empty_curve':c['curve']['keys']=[]
            elif mode=='single_key_curve':c['curve']['keys']=c['curve']['keys'][:1]
            elif mode=='shifted_curve':
                for k in c['curve']['keys']:k['time']+=ONE
            elif mode=='null_particle_after_spawn':r['actors'][n]['ordinary']['quick_spell']['particle_prefab']=False
            elif mode=='missing_exit_after_spawn':r['actors'][n]['ordinary']['quick_spell']['animations'].pop('exit')
            yield q


def interactions(seed):
    modes=['ignore_player','ignore_platform','requery_inactive','requery_destroyed','zero_scale','negative_scale_requery','destroy']
    for base in contacts(seed):
        if base['gust_case']!='contact_airborne':continue
        for mode in modes:
            q=copy.deepcopy(base);q['gust_case']='interaction_'+mode
            r=q['runtime'];victim=q['victim'];target=r['actors'][victim]['object'];platform=r['bindings'][0]['object'];q['ignored_target']=platform if mode=='ignore_platform' else target
            r['scene']['layers']=[0]*32
            def command(kind,**kwargs):return dict(op='spell_command',spell=0,command=dict(kind=kind,**kwargs))
            if mode.startswith('ignore'):
                q['actions'][7]['commands']=[command('ignore',object=q['ignored_target'])]
                if mode=='ignore_player':q['actions'][12]['commands']=[command('activate',dt=0)]
            elif mode=='requery_inactive':
                q['actions'][12]['commands']=[dict(op='actor_active',actor=victim,value=False),command('activate',dt=0)]
            elif mode=='requery_destroyed':
                q['actions'][12]['commands']=[dict(op='destroy_actor',actor=victim),command('activate',dt=0)]
            elif mode=='zero_scale':q['actions'][10]['commands']=[command('scale',value=0)]
            elif mode=='negative_scale_requery':q['actions'][12]['commands']=[command('scale',value=-ONE),command('activate',dt=0)]
            elif mode=='destroy':q['actions'][10]['commands']=[command('destroy')]
            yield q


def coverage(directory,inputs):
    counts=Counter();last={};results={};initial_push={}
    for row in records((directory/'unity.jsonl').open()):
        if 'state' in row:
            last[row['case']]=row['state']
            q=inputs[row['case']]
            if row['action']==8 and q['gust_case'].startswith('interaction_'):initial_push[row['case']]=row['state']['actors'][q['victim']]['controller']['physics']['motion']['external_velocity']
        else:
            results[row['case']]=row['result']
            if 'failed' in row['result']:last[row['case']]=row['result']['failed']
    for case,q in enumerate(inputs):
        expected={'custom_empty_curve':'index_out_of_range','custom_null_particle_after_spawn':'instantiate_null','custom_missing_exit_after_spawn':'animation_missing'}.get(q['gust_case'])
        if results[case].get('error')!=expected:raise ValueError(f'Case {case}: unexpected error {results[case].get("error")}')
        if expected:
            counts['failed_prefix_waves']+=len(last[case]['spells']['instances'])
            if not last[case]['spells']['instances']:raise ValueError(f'Case {case}: failure preceded spell allocation')
            continue
        n=(2 if q['gust_case']=='clone_death_and_recast' else 1) if q['gust_case'].startswith(('contact_','clone_','interaction_')) else len(q['runtime']['scheduler']['players']);states=last[case]['spells']['instances']
        if len(states)!=n:raise ValueError(f'Case {case}: expected {n} waves, got {len(states)}')
        if q['gust_case'].startswith('contact_'):
            victim=last[case]['actors'][q['victim']]
            velocity=victim['controller']['physics']['motion']['external_velocity']
            pushed=velocity != vec()
            expected=q['gust_case'] not in ['contact_inactive','contact_disabled_collider','contact_unpushable_ability']
            if pushed != expected:raise ValueError(f'Case {case}: target push gate was not exercised')
            counts['pushed_victims']+=pushed;counts['guarded_victims']+=not pushed
        if q['gust_case'].startswith('clone_'):
            original=len(q['runtime']['actors']);born=last[case]['actors'][original:]
            if len(born)!=q['expected_births'] or any(a['entity']['present'] for a in born):raise ValueError(f'Case {case}: clone allocation/removal coverage failed')
            counts['clones_born']+=len(born)
            if q['gust_case']=='clone_capacity' and len(q['runtime']['scheduler']['players'])>=3:
                if states[0]['state']['state']['hit_count']!=50:raise ValueError(f'Case {case}: 50-target limit was not reached')
                counts['full_hit_buffers']+=1
        if q['gust_case'].startswith('interaction_'):
            wave=states[0]['state']['state'];mode=q['gust_case'];target=q['runtime']['actors'][q['victim']]['object']
            if mode=='interaction_ignore_player':
                velocity=last[case]['actors'][q['victim']]['controller']['physics']['motion']['external_velocity']
                if velocity!={k:v*2 for k,v in initial_push[case].items()} or velocity==vec():raise ValueError('Ignoring a player did not preserve repeat activation push')
            if mode=='interaction_ignore_platform' and q['ignored_target'] not in wave['hit_objects']:raise ValueError('Ignored platform was outside the wave')
            if mode in ['interaction_requery_inactive','interaction_requery_destroyed'] and target in wave['hits'][:wave['hit_count']]:raise ValueError('Requery retained an inactive/destroyed target')
            if mode=='interaction_zero_scale' and wave['scale']!=0:raise ValueError('Zero scale was not retained')
            counts[mode]+=1
        for wave in states:
            counts['waves']+=1;counts['hits']+=wave['state']['state']['hit_count']
            counts['removed']+=not wave['present'];counts['retained_targets']+=len(wave['state']['state']['hit_objects'])
            if q['gust_case'] in ['zero_dt','custom_single_activation']:
                if not wave['present'] or wave['destroyed']:raise ValueError('Zero-delta wave expired')
            elif wave['present'] or not wave['destroyed']:raise ValueError('Timed wave failed to expire')
    return dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=543286)
    p.add_argument('--cases',type=int,default=0)
    p.add_argument('--directory',default='gust')
    p.add_argument('--reuse-unity',action='store_true')
    p.add_argument('--golden',action='store_true')
    group=p.add_mutually_exclusive_group()
    for suite in ['contacts','clones','custom','interactions']:group.add_argument('--'+suite,dest='suite',action='store_const',const=suite)
    args=p.parse_args();fixture=ROOT/('native/fixtures/gust'+('-'+args.suite if args.suite else '')+'.jsonl.xz')
    if args.golden:report=golden(fixture)
    else:
        qs=list({None:queries,'contacts':contacts,'clones':clones,'custom':custom,'interactions':interactions}[args.suite](args.seed));qs=qs[:args.cases] if args.cases else qs
        directory=ROOT/f'native/artifacts/{args.directory}'
        inputs,hashes=capture(directory,qs,args.reuse_unity)
        report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:
            try:report['gust_coverage']=coverage(directory,inputs)
            except ValueError as e:report['coverage_error']=str(e)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report and not args.cases:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(report,indent=2));return int(not report['match'] or 'coverage_error' in report)


if __name__=='__main__':raise SystemExit(main())
