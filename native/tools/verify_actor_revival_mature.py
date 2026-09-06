"""Revival of fully spawned bodies composes normal death effects with replacement allocation."""
import argparse
import copy
import json
from actor_stream import ROOT,capture,verify,promote,golden,records
from verify_revive_particle import queries as particles
from verify_actors import vec

def queries(seed):
    for base in particles(seed):
        if base['particle_case']!='ordinary':continue
        n=len(base['runtime']['scheduler']['players'])
        q=copy.deepcopy(base)
        q['actions'] += [copy.deepcopy(q['actions'][-1]) for _ in range(100)]
        q['actions'][12]['commands']=[dict(op='actor_spawn',actor=2*n+i,action='fast',position=vec(i*2,30)) for i in range(n)]
        q['actions'][80]['commands']=[dict(op='kill_actor',actor=2*n+i,request=dict(killer=1,spawn_effect=True,ignore_invulnerability=True,cause=0)) for i in range(n)]
        yield q

def coverage(directory,queries):
    last={};before={};errors={};counts=dict(deaths=0,replacements=0,particles=0,death_effects=0)
    with (directory/'unity.jsonl').open() as stream:
        for row in records(stream):
            if 'state' in row:
                last[row['case']]=row['state']
                if row['action']==79:before[row['case']]=row['state']
            else:errors[row['case']]=row['result'].get('error')
    for case,q in enumerate(queries):
        n=len(q['runtime']['scheduler']['players']);s=last[case]
        if errors[case] is not None:raise ValueError(f'Case {case}: {errors[case]}')
        bodies=before[case]['actors'][2*n:3*n]
        if len(bodies)!=n or any(not a['slime']['state']['has_spawned'] or not a['entity']['present'] for a in bodies):raise ValueError('Bodies were not fully spawned before death')
        if any(a['entity']['present'] for a in s['actors'][2*n:3*n]) or len(s['actors'])!=4*n:raise ValueError('Death did not replace every body')
        particles=[e for e in s['spells']['instances'] if e['state']['kind']=='revive_particle']
        if len(particles)!=n or any(not p['state']['state']['arrived'] or p['present'] for p in particles):raise ValueError('Replacement particles did not finish')
        effects=s['combat']['death_effects']['instances']
        if len(effects)<n or any(e.get('kind')=='water' for e in effects[:n]):raise ValueError('Normal death effects were not allocated first')
        if any(r['positions'] or r['respawn_count'] for r in s['combat']['revival']):raise ValueError('Death retained its marker list')
        counts['deaths']+=n;counts['replacements']+=n;counts['particles']+=n;counts['death_effects']+=len(effects)
    return counts

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=725110);p.add_argument('--directory',default='actor-revival-mature');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');a=p.parse_args()
    fixture=ROOT/'native/fixtures/actor-revival-mature.jsonl.xz'
    if a.golden:report=golden(fixture)
    else:
        directory=ROOT/f'native/artifacts/{a.directory}'
        inputs,hashes=capture(directory,list(queries(a.seed)),a.reuse_unity);report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:
            try:report['coverage']=coverage(directory,inputs)
            except ValueError as error:report['coverage_error']=str(error)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(report,indent=2));return int(not report['match'] or 'coverage_error' in report)
if __name__=='__main__':raise SystemExit(main())
