"""Empty-callback InstantAbility methods and the shipped NullAbility clone failure."""
import argparse
import copy
import json
from actor_stream import ROOT,capture,verify,promote,golden,records
from verify_invisibility import queries as invisibility, info
from verify_clone_abilities import queries as equipped
from verify_actors import ONE,vec


def null_prefab():
    a=json.loads((ROOT/'native/reference-export/assets.json').read_text())['c6184b2da376bc0418d15724e93d435f']['data']
    parts={p['type']:p['properties'] for p in a['components']};ia=parts['InstantAbility']
    assert 'FixTransform' not in parts and not a['children'] and ia['funcOnEnter.m_PersistentCalls.m_Calls.Array.size']==0 and ia['EffectOnEnter'] is None and not ia['isHookshot'] and not ia['isInvisibilty']
    return dict(present=True,destroyed=False,cooldown=ia['cooldown.m_rawValue'],stopped_during_time_stop=ia['cooldownStoppedDuringTimeStop'],ground_only=ia['OnlyCastableOnGround'],castable=True,ignore_next_cooldown=False,handler=dict(kind='instant',transform_present=False,position=vec(),active=a['active'],info=info(),slime=None,audio=ia['AudioOnCast']))


def queries(seed):
    asset=null_prefab()
    for base in invisibility(6,seed,150):
        for mode in ['empty','castable_metadata_false','audio','null_transform']:
            q=copy.deepcopy(base);q['instant_case']=mode;q['entry_events']=True
            for actor in q['runtime']['actors']:
                for ability in actor['slime']['abilities']['pool']:
                    ability.update(copy.deepcopy(asset));ability['handler'].update(active=False,transform_present=mode!='null_transform')
                    if mode=='castable_metadata_false':ability['castable']=False
                    if mode=='audio':ability['handler']['audio']='invisibility'
            yield q


def clone_faults(seed):
    asset=null_prefab()
    for base in equipped(seed):
        if base['equipped_clone_case']!='single':continue
        for mode,icons in [('null_prefab',[0]),('partial_allocation',[19,0,19]),('null_first',[0,19])]:
            q=copy.deepcopy(base);q['instant_case']=mode;n=len(q['runtime']['actors']);q['clone_prefab']['abilities'][0]=copy.deepcopy(asset)
            source=q['runtime']['actors'][-1]['slime']['abilities'];source['slots']=[0]*len(icons);source['indicators']=[dict(seconds_since_reset=10000.,primary_icon=i,lifecycle=dict(active=False,follow=True)) for i in icons]
            q['actions']=q['actions'][:1];q['actions'][0]['commands']=[dict(op='full_clone_actor',actor=n-1,player=n-1,position=vec(10,35))]
            yield q


def coverage(directory,qs,cloning):
    finals={};results={};casts=0
    with (directory/'unity.jsonl').open() as stream:
        for row in records(stream):
            if 'state' in row:
                finals[row['case']]=row['state'];casts+=sum(e['event']['kind']=='ability_entry' for e in row['state']['report']['events'])
                if any(p['invisible'] for p in row['state']['players']):raise ValueError('Empty callback made a player invisible')
            else:
                results[row['case']]=row['result']
                if 'failed' in row['result']:finals[row['case']]=row['result']['failed']
    allocated=0
    for case,q in enumerate(qs):
        mode=q['instant_case'];r=results[case];error='null_reference' if cloning or mode=='null_transform' else None
        if r.get('error')!=error:raise ValueError(f'{case}: unexpected result')
        if cloning:
            n=len(q['runtime']['actors']);s=finals[case]
            if len(s['actors'])!=n+1 or r['completed']!=0:raise ValueError('Missing clone failure prefix')
            a=s['actors'][-1]['slime']['abilities'];count=2 if mode=='partial_allocation' else 1
            if a['slots'] or len(a['pool'])!=count or a['indicators']:raise ValueError('Premature slot/indicator assignment')
            bad=a['pool'][-1]
            if bad['registration']['transform'] is not None or not bad['handler']['active'] or bad['handler']['transform_present']:raise ValueError('Missing-transform prefab was changed')
            allocated+=count
    return dict(casts=casts,allocated_failure_prefabs=allocated)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=731249);p.add_argument('--directory',default='instant-ability');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--clone-faults',action='store_true');a=p.parse_args();fixture=ROOT/('native/fixtures/null-ability-clones.jsonl.xz' if a.clone_faults else 'native/fixtures/instant-ability.jsonl.xz')
    if a.golden:report=golden(fixture)
    else:
        qs=list(clone_faults(a.seed) if a.clone_faults else queries(a.seed));directory=ROOT/f'native/artifacts/{a.directory}';inputs,hashes=capture(directory,qs,a.reuse_unity);report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:
            try:report['coverage']=coverage(directory,inputs,a.clone_faults)
            except ValueError as error:report['coverage_error']=str(error)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(report,indent=2));return int(not report['match'] or 'coverage_error' in report)
if __name__=='__main__':raise SystemExit(main())
