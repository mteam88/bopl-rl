"""Revival holder failures and mixed per-player protection during time stop."""
import argparse
import copy
import json
from actor_stream import ROOT,capture,verify,promote,golden,records
from verify_revive_particle import queries as particles,PREFAB as PARTICLE
from revival_prefabs import PREFAB as EFFECT

def queries(seed):
    for base in particles(seed):
        if base['particle_case']!='ordinary':continue
        n=len(base['runtime']['scheduler']['players'])
        for prefab in [PARTICLE,EFFECT]:
            q=copy.deepcopy(base);q['cast_fault']=prefab;q['edge_case']='holder'
            for actor in q['runtime']['actors'][n:]:actor['ordinary']['quick_spell']['spell']=prefab
            q['actions']=q['actions'][:10]
            yield q
        for mode in ['mixed_protection','null_slime_first','null_slime_last']:
            q=copy.deepcopy(base);q['edge_case']=mode
            for i,player in enumerate(q['runtime']['scheduler']['players']):
                player['protected']=i % 2==0 if mode=='mixed_protection' else i==(0 if mode=='null_slime_first' else n-1)
            for tick,action in enumerate(q['actions']):
                if 12<=tick<60:action['scale']=0
                if tick==12 and mode.startswith('null_slime'):
                    for command in action['commands']:command['slime']=None
            yield q

def coverage(directory,inputs):
    counts=dict(allocated=0,flags_unchanged=0,mixed_cases=0,mixed_particles=0)
    last={}
    with (directory/'unity.jsonl').open() as stream:
        for row in records(stream):
            if 'state' in row:last[row['case']]=row['state']
            if 'result' not in row:continue
            result=row['result'];q=inputs[row['case']]
            if q['edge_case']!='holder':
                if 'error' in result:raise ValueError('Mixed protection case failed')
                particles=[e for e in last[row['case']]['spells']['instances'] if e['state']['kind']=='revive_particle']
                n=len(q['runtime']['scheduler']['players'])
                if len(particles)<n or any(not p['state']['state']['arrived'] or p['present'] for p in particles[:n]):raise ValueError('Mixed protection particles did not complete')
                counts['mixed_cases']+=1;counts['mixed_particles']+=n
                continue
            expected='not_implemented' if q['cast_fault']==PARTICLE else 'cannot_set_revive_effect_player'
            if result.get('error')!=expected:raise ValueError(f'Case {row["case"]}: wrong cast failure')
            s=result['failed'];instances=s['spells']['instances']
            if len(instances)!=1 or not instances[0]['present']:raise ValueError('Cast did not preserve its original allocation')
            if any(p['respawn_count'] or p['status'] for p in s['combat']['revival']):raise ValueError('Failed holder assignment set the revival flag')
            counts['allocated']+=1;counts['flags_unchanged']+=1
    return counts

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=725110);p.add_argument('--directory',default='revival-edges');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');a=p.parse_args()
    fixture=ROOT/'native/fixtures/revival-edges.jsonl.xz'
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
