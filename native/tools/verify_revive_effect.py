"""Actual revival effect animations launch original particles into shared clone lifecycles."""
import argparse
import copy
import json
from functools import lru_cache
from actor_stream import ROOT,capture,verify,promote,golden,records
from verify_revive_particle import queries as particle_queries,PREFAB as PARTICLE
from verify_actors import ONE,vec
from revival_coverage import coverage

PREFAB='ee3ca83d3a7c0264abd0950f3ec10ea7'

from revival_prefabs import effect_config as config

def queries(seed):
    for base in particle_queries(seed):
        if base['particle_case']!='ordinary':continue
        n=len(base['runtime']['scheduler']['players'])
        for mode in ['ordinary','multiple_targets','empty','null_targets','time_stop','protected','null_player','null_player_stopped','null_particle','null_marker','destroy_marker','destroy_slime','destroy_effect','zero_direction','null_slime']:
            q=copy.deepcopy(base);q['effect_case']=mode;r=q['runtime']
            r['spells']['prefabs'][PREFAB]=dict(kind='revive_effect',config=copy.deepcopy(config()))
            if mode=='null_particle':r['spells']['prefabs'][PREFAB]['config']['particle']=None
            if mode=='protected':
                for p in r['scheduler']['players']:p['protected']=True
            q['actions'] += [copy.deepcopy(q['actions'][-1]) for _ in range(100)]
            for tick,action in enumerate(q['actions']):
                if tick==12:
                    commands=[]
                    for i in range(1 if mode=='multiple_targets' else n):
                        targets=[dict(marker=j,slime=2*n+j) for j in range(n)] if mode=='multiple_targets' else [dict(marker=None if mode=='null_marker' else i,slime=None if mode=='null_slime' else 2*n+i)]
                        if mode=='empty':targets=[]
                        if mode=='null_targets':targets=None
                        commands.append(dict(op='revive_effect',prefab=PREFAB,actor=i,player=None if mode.startswith('null_player') else i,position=vec(-20,30),targets=targets))
                    if mode=='zero_direction':
                        for command in commands:command['position']=vec(0,30)
                        commands=[dict(op='spell_command',spell=i,command=dict(kind='position',value=vec(0,30))) for i in range(n)]+commands
                    action['commands']=commands
                if mode in ['time_stop','protected','null_player_stopped'] and 12<=tick<60:action['scale']=0
                if tick==20 and mode=='destroy_marker':action['commands']=[dict(op='spell_command',spell=i,command=dict(kind='destroy')) for i in range(n)]
                if tick==20 and mode=='destroy_slime':action['commands']=[dict(op='destroy_actor',actor=2*n+i) for i in range(n)]
                if tick==20 and mode=='destroy_effect':action['commands']=[dict(op='spell_command',spell=n+i,command=dict(kind='destroy')) for i in range(n)]
            yield q

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=725110);p.add_argument('--cases',type=int,default=0);p.add_argument('--directory',default='revive-effect');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');a=p.parse_args()
    fixture=ROOT/'native/fixtures/revive-effect.jsonl.xz'
    if a.golden:report=golden(fixture)
    else:
        qs=list(queries(a.seed));qs=qs[:a.cases] if a.cases else qs
        directory=ROOT/f'native/artifacts/{a.directory}'
        inputs,hashes=capture(directory,qs,a.reuse_unity);report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:
            try:report['revival_coverage']=coverage(directory,inputs,'effect')
            except ValueError as error:report['coverage_error']=str(error)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report and not a.cases:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(report,indent=2));return int(not report['match'] or 'coverage_error' in report)
if __name__=='__main__':raise SystemExit(main())
