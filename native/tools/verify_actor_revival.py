"""PlayerCollision revival deaths allocate real clone bodies, effects and particles."""
import argparse
import copy
import json
from actor_stream import ROOT,capture,verify,promote,golden,records
from verify_revive_particle import queries as particles
from revival_prefabs import PREFAB as EFFECT
from verify_actors import ONE,vec
from revival_coverage import coverage

def queries(seed):
    for base in particles(seed):
        if base['particle_case']!='ordinary':continue
        n=len(base['runtime']['scheduler']['players'])
        for mode in ['ordinary','simultaneous','water','limit','time_stop','protected','multiple_markers','null_particle']:
            q=copy.deepcopy(base);q['death_revival_case']=mode;r=q['runtime']
            for a in q['actions']:
                a['commands']=[c for c in a.get('commands',[]) if c['op']!='revive_particle']
            q['actions'] += [copy.deepcopy(q['actions'][-1]) for _ in range(160)]
            if mode=='limit':
                for p in r['combat']['book']['players']:p['bodies_alive']=16
            if mode=='protected':
                for p in r['scheduler']['players']:p['protected']=True
            if mode=='null_particle':r['spells']['prefabs'][EFFECT]['config']['particle']=None
            for tick,action in enumerate(q['actions']):
                if tick==10 and mode=='water':
                    for c in action['commands']:c['position']=vec(0,-12)
                if tick==(60 if mode=='multiple_markers' else 15) and mode!='water':
                    action['commands']=[dict(op='kill_actor',actor=2*n+i,request=dict(killer=1,spawn_effect=True,ignore_invulnerability=True,cause=0)) for i in range(n if mode=='simultaneous' else 1)]
                if mode in ['time_stop','protected'] and 15<=tick<60:action['scale']=0
                if mode=='multiple_markers' and tick==35:action['commands']=[dict(op='spell_command',spell=i,command=dict(kind='end')) for i in range(n)]
                if mode=='multiple_markers' and tick==50:action['commands']=[dict(op='quick_cast',actor=n+i) for i in range(n)]
            yield q

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=725110);p.add_argument('--cases',type=int,default=0);p.add_argument('--directory',default='actor-revival');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--water',action='store_true');a=p.parse_args()
    fixture=ROOT/('native/fixtures/actor-revival-water.jsonl.xz' if a.water else 'native/fixtures/actor-revival.jsonl.xz')
    if a.golden:report=golden(fixture)
    else:
        qs=list(queries(a.seed))
        if a.water:
            qs=[q for q in qs if q['death_revival_case']=='water']
            for q in qs:q['runtime']['space']=False
        qs=qs[:a.cases] if a.cases else qs
        directory=ROOT/f'native/artifacts/{a.directory}'
        inputs,hashes=capture(directory,qs,a.reuse_unity);report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:
            try:report['revival_coverage']=coverage(directory,inputs,'death')
            except ValueError as error:report['coverage_error']=str(error)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report and not a.cases:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(report,indent=2));return int(not report['match'] or 'coverage_error' in report)
if __name__=='__main__':raise SystemExit(main())
