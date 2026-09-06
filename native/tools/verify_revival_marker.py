"""Original Revival QuickSpell allocation, physical markers and replacement lifecycle."""
import argparse
import copy
import json
from collections import Counter
from functools import lru_cache
from actor_stream import ROOT, capture, verify, promote, golden, records
from verify_quick_spell import queries as quick_queries
from verify_actor_clones import queries as clone_queries
from verify_actors import ONE, DT, vec

PREFAB='ebf9efaac968c41419a17acfca82d901'


@lru_cache
def config():
    source=json.loads((ROOT/'native/reference-export/assets.json').read_text())[PREFAB]['data']
    parts={c['type']:c['properties'] for c in source['components']}
    b=parts['BoplBody'];c=parts['DPhysicsCircle'];t=parts['FixTransform']
    body={name:b[field+'.m_rawValue'] for name,field in dict(mass='mass',inertia='momentOfInertia',bounciness='bounciness',friction='friction',dynamic_friction='dynamicFriction',gravity='gravityScale',air_drag='airFriction01',rotational_drag='rotationalDrag01',angular_velocity='StartAngularVelocity').items()}
    body['velocity']={k:b[f'StartVelocity.{k}.m_rawValue'] for k in ['x','y']}
    return dict(body=body,circle=dict(scale=ONE,min_scale=c['MinScale.m_rawValue'],max_scale=c['MaxScale.m_rawValue'],start_radius=c['startRadius.m_rawValue'],initialized=False,manually_call_init=c['manuallyCallInit']),layer=source['layer'],offset={k:t[f'offset.{k}.m_rawValue'] for k in ['x','y']},particle=True,children=len(source['children']))


def queries(seed):
    combat={len(q['runtime']['actors']):q['runtime']['combat'] for q in clone_queries(seed) if q['clone_case']=='before'}
    for base in quick_queries(seed):
        if base['quick_case']!='revival':continue
        n=len(base['runtime']['scheduler']['players'])
        for mode in ['ordinary','water','space','bounds','scales','zero_dt','time_stop','protected','recast','end','add_scale','manual_init']:
            q=copy.deepcopy(base);q.update(revival_case=mode,registration_state=True)
            r=q['runtime'];r['combat']=copy.deepcopy(combat[n]);r['spells']=dict(prefabs={PREFAB:dict(kind='marker',config=copy.deepcopy(config()))},instances=[])
            for p in r['combat']['revival']:p['positions']=[]
            for i,actor in enumerate(r['actors'][n:]):
                quick=actor['ordinary']['quick_spell'];quick.update(spell=PREFAB,revival=dict(player=None,indicator=None))
                index=len(r['components']);r['components'].append(dict(actor=n+i,role='revival'));r['scheduler']['components'].append(dict(owner=n+i,kind='updatable',present=True,destroyed=False,hierarchy=index,enabled=True,ignore_activity=False))
            total=len(r['components']);r['scheduler'].update(live=[],added=total,pending=dict(items=list(range(total)),length=total))
            for player in r['scheduler']['players']:player['protected']=mode=='protected'
            if mode=='space':r['space']=True
            if mode=='manual_init':r['spells']['prefabs'][PREFAB]['config']['circle']['manually_call_init']=True
            for tick,action in enumerate(q['actions']):
                action['enabled'] += [True]*n
                if mode=='zero_dt':action['dt']=0
                if mode in ['time_stop','protected'] and 2<=tick<30:action['scale']=0
                if mode in ['water','space','bounds']:
                    # Place the caster through existing actor commands before the initial button press.
                    if tick==0:
                        for actor in r['actors']:
                            position=vec(0,-20) if mode in ['water','space'] else vec(120,60)
                            actor['controller']['physics']['motion'].update(position=position,grounded=False,ground=None)
                            actor['controller']['apply_gravity']=False;actor['entity']['body_simulated']=False
                if mode=='scales' and tick==0:action['commands']=[dict(op='player_scale',player=i,value=2*ONE) for i in range(n)]
                if mode=='recast' and tick==35:action['commands']=[dict(op='quick_cast',actor=n+i) for i in range(n)]
                if mode=='end' and tick==35:action['commands']=[dict(op='spell_command',spell=i,command=dict(kind='end')) for i in range(n)]
                if mode=='add_scale' and tick in [20,40]:action['commands']=[dict(op='spell_command',spell=i,command=dict(kind='add_scale',value=ONE if tick==20 else -2*ONE)) for i in range(n)]
                if mode=='manual_init' and tick==20:action['commands']=[dict(op='spell_command',spell=i,command=dict(kind='manual_init')) for i in range(n)]
            yield q



def custom(seed):
    for base in queries(seed):
        if base['revival_case']!='ordinary':continue
        n=len(base['runtime']['scheduler']['players'])
        for mode in ['velocity','gravity','offset','scale_before_init','destroy_before_init','end_before_init','manual_forever','no_particle_end','end_then_recast','null_particle','null_spell','missing_exit']:
            q=copy.deepcopy(base);q['revival_case']='custom_'+mode;r=q['runtime'];c=r['spells']['prefabs'][PREFAB]['config']
            if mode=='velocity':c['body']['velocity']=vec(20,-10);c['body']['angular_velocity']=ONE
            if mode=='gravity':c['body']['gravity']=ONE
            if mode=='offset':c['offset']=vec(2,1)
            if mode=='manual_forever':c['circle']['manually_call_init']=True
            if mode=='no_particle_end':c['particle']=False
            if mode in ['null_particle','null_spell','missing_exit']:
                quick=r['actors'][n]['ordinary']['quick_spell']
                if mode=='null_particle':quick['particle_prefab']=False
                if mode=='null_spell':quick['spell']=None
                if mode=='missing_exit':del quick['animations']['exit']
            for tick,action in enumerate(q['actions']):
                if mode in ['scale_before_init','destroy_before_init','end_before_init'] and tick==7:
                    cmd=dict(kind={'scale_before_init':'add_scale','destroy_before_init':'destroy','end_before_init':'end'}[mode])
                    if mode=='scale_before_init':cmd['value']=ONE
                    action['commands']=[dict(op='spell_command',spell=i,command=copy.deepcopy(cmd)) for i in range(n)]
                if mode in ['no_particle_end','end_then_recast'] and tick==35:action['commands']=[dict(op='spell_command',spell=i,command=dict(kind='end')) for i in range(n)]
                if mode=='end_then_recast' and tick==50:action['commands']=[dict(op='quick_cast',actor=n+i) for i in range(n)]
            yield q


def interactions(seed):
    cloned={len(q['runtime']['actors']):q for q in clone_queries(seed) if q['clone_case']=='before'}
    for base in queries(seed):
        if base['revival_case']!='ordinary':continue
        n=len(base['runtime']['scheduler']['players'])
        for mode in ['clone_before','clone_after','clone_recast','marker_without_revive','end_twice']:
            q=copy.deepcopy(base);q['revival_case']='interaction_'+mode;r=q['runtime']
            if mode.startswith('clone_'):
                q.update(dynamic_actors=True,clone_prefab=copy.deepcopy(cloned[n]['clone_prefab']))
                for a in r['actors']:
                    a['collision']=dict(invulnerable=0,was_killed=False,death_trauma=.5)
                    a['entity'].update(present=True,destroyed=False)
                    if a.get('slime'):a['slime']['managed_destroyed']=False
            if mode=='marker_without_revive':
                for a in r['actors'][n:]:del a['ordinary']['quick_spell']['revival']
            birth_ticks=[2] if mode=='clone_before' else [10] if mode=='clone_after' else [2,10] if mode=='clone_recast' else []
            for tick,action in enumerate(q['actions']):
                if tick in birth_ticks:action['commands']=[dict(op='clone_actor',actor=i,player=i,position=vec(i*3,20)) for i in range(n)]
                if tick-2 in birth_ticks:
                    start=2*n+birth_ticks.index(tick-2)*n
                    action['commands']=[dict(op='actor_spawn',actor=start+i,action='fast',position=vec(i*3,20)) for i in range(n)]
                if mode.startswith('clone_') and tick==45:action['commands']=[dict(op='destroy_actor',actor=2*n+i) for i in range(n*len(birth_ticks))]
                if mode=='clone_recast' and tick==35:action['commands']=[dict(op='quick_cast',actor=n+i) for i in range(n)]
                if mode=='end_twice' and tick in [35,45]:action['commands']=[dict(op='spell_command',spell=i,command=dict(kind='end')) for i in range(n)]
                if mode=='marker_without_revive':
                    action['enabled']=[flag and c['role']!='revival' for flag,c in zip(action['enabled'],r['components'])]
                    if tick==0:action['commands']=[dict(op='player_scale',player=i,value=2*ONE) for i in range(n)]
            yield q

def coverage(directory,inputs):
    last={};errors={};counts=Counter()
    with (directory/'unity.jsonl').open() as stream:
        for row in records(stream):
            if 'state' in row:last[row['case']]=row['state']
            if 'error' in row.get('result',{}):errors[row['case']]=row['result'];last[row['case']]=row['result']['failed']
    for i,q in enumerate(inputs):
        frame=last[i];n=len(q['runtime']['scheduler']['players']);mode=q['revival_case'];instances=frame['spells']['instances']
        error={'custom_null_particle':'instantiate_null','custom_null_spell':'null_reference','custom_missing_exit':'animation_missing'}.get(mode)
        if error:
            if errors.get(i,{}).get('error')!=error:raise ValueError('Unexpected marker failure')
            expected=0 if mode=='custom_null_spell' else 1
            if len(instances)!=expected or sum(p['respawn_count'] for p in frame['combat']['revival'])!=expected:raise ValueError('Failure lost allocated marker or flag')
            counts['intentional_errors']+=1
            continue
        if i in errors:raise ValueError('Marker corpus did not complete')
        if len(instances)!=(2*n if mode in ['recast','custom_end_then_recast','interaction_clone_recast'] else n):raise ValueError('Expected marker was not created')
        alive=sum(not v['destroyed'] for v in instances)
        if alive!=(0 if mode in ['end','custom_destroy_before_init','custom_end_before_init','custom_no_particle_end','interaction_end_twice'] else n):raise ValueError('Unexpected marker removal/replacement')
        positions=0 if mode=='interaction_marker_without_revive' else 2 if mode=='custom_end_then_recast' else 1
        # Completing a clone spawn clears ExtraLife after the first cast.
        status=0 if mode in ['interaction_marker_without_revive','interaction_clone_before','interaction_clone_after'] else 1
        if any(p['status']!=status or p['respawn_count']!=positions for p in frame['combat']['revival']):raise ValueError('Revival flag or positions were not set')
        for v in instances:
            state=v['state']['state'];initialized=mode not in ['custom_destroy_before_init','custom_end_before_init','custom_manual_forever']
            if state['circle']['circle']['initialized']!=initialized:raise ValueError('Unexpected marker collider initialization')
            counts['markers']+=1;counts['destroyed']+=v['destroyed'];counts['initialized']+=initialized
        if mode.startswith('interaction_clone_'):
            expected=n*(2 if mode=='interaction_clone_recast' else 1)
            clones=frame['actors'][2*n:]
            if len(clones)!=expected or any(a['entity']['present'] for a in clones):raise ValueError('Clones were not created/removed')
            counts['clones']+=expected
        if mode=='interaction_marker_without_revive' and any(v['state']['state']['circle']['circle']['scale']!=2*ONE for v in instances):raise ValueError('Plain QuickSpell failed to scale the marker')
        counts[mode]+=1
    return dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=830506);p.add_argument('--cases',type=int,default=0);p.add_argument('--directory',default='revival-marker');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--custom',action='store_true');p.add_argument('--interactions',action='store_true')
    args=p.parse_args();fixture=ROOT/('native/fixtures/revival-marker'+('-custom' if args.custom else '-interactions' if args.interactions else '')+'.jsonl.xz')
    if args.golden:report=golden(fixture)
    else:
        qs=list((custom if args.custom else interactions if args.interactions else queries)(args.seed));qs=qs[:args.cases] if args.cases else qs
        directory=ROOT/'native/artifacts'/args.directory;inputs,hashes=capture(directory,qs,args.reuse_unity);report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:
            try:report['marker_coverage']=coverage(directory,inputs)
            except ValueError as e:report['coverage_error']=str(e)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report and not args.cases:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(report,indent=2));return int(not report['match'] or 'coverage_error' in report)

if __name__=='__main__':raise SystemExit(main())
