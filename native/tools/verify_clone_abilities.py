"""Original SpawnClone allocates fresh equipped abilities and indicator objects."""
import argparse
import copy
import json
from actor_stream import ROOT,capture,verify,promote,golden,records
from collections import Counter
from verify_actor_clones import queries as clones
from verify_invisibility import asset_config,info
from verify_actors import ONE,vec

def queries(seed):
    asset=asset_config()
    ability=dict(present=True,destroyed=False,cooldown=asset['cooldown'],stopped_during_time_stop=asset['stop'],ground_only=asset['ground_only'],castable=True,ignore_next_cooldown=False,handler=dict(kind='invisibility',restore_cooldown=asset['restore'],gate_invisible=asset['gate'],component_present=True,linked=True,transform_present=True,position=vec(),active=False,info=info(),slime=None,audio=asset['audio']))
    for base in clones(seed):
        if base['clone_case']!='before':continue
        for mode in ['single','three','alias','null_slot','zero','death','pending_destroy','clone_again','short_indicators','null_indicator','extra_indicator']:
            q=copy.deepcopy(base);r=q['runtime'];n=len(r['actors']);q['equipped_clone_case']=mode;q['clone_prefab']['abilities']=[None]*39;q['clone_prefab']['abilities'][19]=copy.deepcopy(ability)
            q['clone_prefab']['actor']['slime']['entry']=dict(info=info(),rope_target=None,was_invisible=False)
            for actor in r['actors']:
                count=3 if mode in ['three','alias','short_indicators'] else 0 if mode=='zero' else 1
                routing=actor['slime']['abilities'];routing.update(slots=list(range(count)),pool=[copy.deepcopy(ability) for _ in range(count)],indicators=[dict(seconds_since_reset=10000.,primary_icon=19,lifecycle=dict(active=False,follow=True)) for _ in range(count)],timers=[10000*ONE]*max(count,3))
                for a in routing['pool']:a['cooldown']=10000*ONE
                if mode=='alias':routing['slots']=[0]*3
                if mode=='null_slot':routing['slots']=[None]
                if mode=='short_indicators':routing['indicators']=routing['indicators'][:1]
                if mode=='null_indicator':routing['indicators']=[None]
                if mode=='extra_indicator':routing['indicators']*=4
            q['actions'] += [copy.deepcopy(q['actions'][-1]) for _ in range(150)]
            for tick,action in enumerate(q['actions']):
                action['commands']=[]
                action['routing_inputs']=[dict(can_use=True,buttons=[tick>=80]*3) for _ in range(n)]
                if tick==0:action['commands']=[dict(op='full_clone_actor',actor=i,player=i,position=vec(i*4,35)) for i in range(n)]
                if tick==10 and mode!='pending_destroy':action['commands']=[dict(op='actor_spawn',actor=n+i,action='respawn',position=vec(i*4,35)) for i in range(n)]
                if tick==0 and mode=='pending_destroy':action['commands'] += [dict(op='kill_actor',actor=n+i,request=dict(killer=1,spawn_effect=False,ignore_invulnerability=True,cause=0)) for i in range(n)]
                if tick==260 and mode=='death':action['commands']=[dict(op='kill_actor',actor=n+i,request=dict(killer=1,spawn_effect=False,ignore_invulnerability=True,cause=0)) for i in range(n)]
                if tick==250 and mode=='clone_again':action['commands']=[dict(op='full_clone_actor',actor=n+i,player=i,position=vec(i*4,35)) for i in range(n)]
            yield q

def coverage(directory,queries):
    first={};last={};results={};counts=Counter()
    with (directory/'unity.jsonl').open() as stream:
        for row in records(stream):
            case=row['case']
            if 'state' in row:
                first.setdefault(case,row['state']);last[case]=row['state']
                counts['casts']+=sum(e['event']==dict(kind='sound',value='invisibility') for e in row['state']['report']['events'])
            else:
                results[case]=row['result']
                if 'failed' in row['result']:last[case]=row['result']['failed'];first.setdefault(case,last[case])
    for case,q in enumerate(queries):
        mode=q['equipped_clone_case'];n=len(q['runtime']['actors']);f=first[case];l=last[case];result=results[case]
        error={'short_indicators':'index_out_of_range','null_indicator':'null_reference','extra_indicator':'index_out_of_range'}.get(mode)
        if result.get('error')!=error:raise ValueError(f'{case} {mode}: unexpected result {result.get("error")}')
        if error:
            if result['completed']!=0 or len(l['actors'])!=n+1:raise ValueError(f'{case}: bad failed clone allocation prefix')
            counts['faults']+=1;continue
        expected=3 if mode in ['three','alias'] else 0 if mode=='zero' else 1
        for a in f['actors'][n:]:
            routing=a['slime']['abilities']
            if len(routing['slots'])!=(0 if mode=='pending_destroy' else expected) or len(routing['pool'])!=expected or len(routing['indicators'])!=3:raise ValueError(f'{case}: bad copied ability count')
            if mode!='pending_destroy' and any(a['cooldown']!=2*ONE for a in routing['pool']):raise ValueError(f'{case}: copied mutable cooldown instead of prefab')
            counts['fresh_abilities']+=len(routing['pool'])
        if mode in ['death','pending_destroy']:
            for a in l['actors'][n:]:
                if a['entity']['present'] or a['slime']['abilities']['slots'] or any(p['present'] or not p['destroyed'] for p in a['slime']['abilities']['pool']):raise ValueError(f'{case}: incomplete pooled removal')
                counts['removed_clones']+=1
        elif mode=='clone_again':
            if len(l['actors'])!=3*n:raise ValueError(f'{case}: missing second generation')
            for a in l['actors'][2*n:]:
                if a['slime']['abilities']['pool'][0]['cooldown']!=2*ONE:raise ValueError(f'{case}: second generation inherited old cooldown')
            counts['second_generation']+=n
        counts['complete']+=1
    if counts['casts']==0:raise ValueError('No actual clone casts')
    return dict(counts)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=725112);p.add_argument('--cases',type=int,default=0);p.add_argument('--directory',default='clone-abilities');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--long',action='store_true');a=p.parse_args();fixture=ROOT/('native/fixtures/clone-abilities-long.jsonl.xz' if a.long else 'native/fixtures/clone-abilities.jsonl.xz')
    if a.golden:report=golden(fixture)
    else:
        qs=list(queries(a.seed))
        if a.long:
            qs=[q for q in qs if q['equipped_clone_case'] in ['single','three']]
            for q in qs:q['actions'] += [copy.deepcopy(q['actions'][-1]) for _ in range(600)]
        qs=qs[:a.cases] if a.cases else qs;directory=ROOT/f'native/artifacts/{a.directory}';inputs,hashes=capture(directory,qs,a.reuse_unity);report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:
            try:report['coverage']=coverage(directory,inputs)
            except ValueError as error:report['coverage_error']=str(error)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report and not a.cases:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(report,indent=2));return int(not report['match'] or 'coverage_error' in report)
if __name__=='__main__':raise SystemExit(main())
