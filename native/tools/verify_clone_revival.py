"""Marker death copies equipped Invisibility again and completes its replacement spawn."""
import argparse
import copy
import json
from actor_stream import ROOT,capture,verify,promote,golden,records
from verify_revive_particle import queries as particles
from verify_clone_abilities import queries as equipped
from verify_actors import ONE

def queries(seed):
    template=next(equipped(seed))['clone_prefab']
    for base in particles(seed):
        if base['particle_case']!='ordinary':continue
        q=copy.deepcopy(base);r=q['runtime'];n=len(r['scheduler']['players'])
        q['clone_prefab']=copy.deepcopy(template);r['combat']['clone_prefab']=copy.deepcopy(template)
        q['equipped_revival_case']='indicator_selected_invisibility'
        for actor in r['actors'][:n]:
            for indicator in actor['slime']['abilities']['indicators']:
                if indicator is not None:indicator['primary_icon']=19
        for action in q['actions']:
            action['commands']=[c for c in action.get('commands',[]) if c['op']!='revive_particle']
            for c in action['commands']:
                if c['op']=='clone_actor':c['op']='full_clone_actor'
        q['actions'] += [copy.deepcopy(q['actions'][-1]) for _ in range(160)]
        q['actions'][15]['commands']=[dict(op='kill_actor',actor=2*n+i,request=dict(killer=1,spawn_effect=True,ignore_invulnerability=True,cause=0)) for i in range(n)]
        yield q

def coverage(directory,qs,require_casts=False):
    final={};counts=dict(replacements=0,removed_abilities=0,particles=0,casts=0,uncast_removed_replacements=0,replacement_deaths=0)
    with (directory/'unity.jsonl').open() as stream:
        for row in records(stream):
            if 'state' in row:
                final[row['case']]=row['state']
                counts['casts']+=sum(e['event']==dict(kind='sound',value='invisibility') for e in row['state']['report']['events'])
            elif 'error' in row['result']:raise ValueError('Unexpected equipped revival failure')
    for case,q in enumerate(qs):
        n=len(q['runtime']['scheduler']['players']);s=final[case]
        if len(s['actors'])!=4*n:raise ValueError(f'{case}: replacement count')
        for old in s['actors'][2*n:3*n]:
            if old['entity']['present'] or old['slime']['abilities']['slots'] or any(a['present'] or not a['destroyed'] for a in old['slime']['abilities']['pool']):raise ValueError(f'{case}: old equipped clone not removed')
            counts['removed_abilities']+=len(old['slime']['abilities']['pool'])
        for new in s['actors'][3*n:]:
            if not new['slime']['state']['has_spawned'] or new['slime']['abilities']['pool'][0]['handler']['kind']!='invisibility':raise ValueError(f'{case}: replacement not equipped and spawned')
            counts['replacements']+=1
            ability=new['slime']['abilities']['pool'][0]
            if not new['entity']['present']:counts['replacement_deaths']+=1
            if require_casts and ability['handler']['info']['player']==0:
                if new['entity']['present'] or ability['present'] or not ability['destroyed']:raise ValueError(f'{case}: live replacement did not cast')
                counts['uncast_removed_replacements']+=1
        ps=[i for i in s['spells']['instances'] if i['state']['kind']=='revive_particle']
        if len(ps)!=n or any(i['present'] or not i['state']['state']['arrived'] for i in ps):raise ValueError(f'{case}: particles did not finish')
        counts['particles']+=len(ps)
    return counts

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=725110);p.add_argument('--cases',type=int,default=0);p.add_argument('--directory',default='clone-revival');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--casts',action='store_true');a=p.parse_args();fixture=ROOT/('native/fixtures/clone-revival-casts.jsonl.xz' if a.casts else 'native/fixtures/clone-revival.jsonl.xz')
    if a.golden:report=golden(fixture)
    else:
        qs=list(queries(a.seed))
        if a.casts:
            for q in qs:
                n=len(q['runtime']['scheduler']['players'])
                q['actions'] += [copy.deepcopy(q['actions'][-1]) for _ in range(200)]
                q['actions'][200]['commands']=[dict(op='actor_active',actor=i,value=False) for i in range(n)]
                for action in q['actions'][200:]:
                    for i in action['routing_inputs']:i['buttons']=[True,False,False]
        qs=qs[:a.cases] if a.cases else qs;directory=ROOT/f'native/artifacts/{a.directory}';inputs,hashes=capture(directory,qs,a.reuse_unity);report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:
            try:
                report['coverage']=coverage(directory,inputs,a.casts)
            except ValueError as error:report['coverage_error']=str(error)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report and not a.cases:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(report,indent=2));return int(not report['match'] or 'coverage_error' in report)
if __name__=='__main__':raise SystemExit(main())
