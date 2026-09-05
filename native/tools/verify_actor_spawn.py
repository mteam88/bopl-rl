"""Scheduled SlimeController initialization, revival, animation completion and movement."""
import argparse,copy,json
from actor_stream import ROOT,capture,verify,promote,golden
from verify_actor_deaths import queries as deaths
from verify_death_effects import prefab as death_prefab
from verify_spawn import animation_assets
from verify_actors import ONE,vec

def queries(seed):
    animation,_,_=animation_assets()
    bases=[q for q in deaths(seed) if q['death_case']=='death']
    for base in bases:
        for mode in ['init','respawn','fast','permanent','time_stop','tutorial','physics_respawn','body_fast','missing_animation','missing_renderer','null_ability','missing_ability','null_indicator']:
            q=copy.deepcopy(base);q['spawn_case']=mode;q['player_state']=True;r=q['runtime'];q['scripts']={};q['routing_checks']=False
            n=len(r['actors']);victim=n-1
            for i,a in enumerate(r['actors']):
                s=a['slime'];s['spawn']=dict(physics_enabled=True,renderer_enabled=True,fast=False,next_ability=2,abilityless=mode not in ['init','time_stop','tutorial','missing_animation','missing_renderer'],tutorial=mode=='tutorial')
                s['animations']['spawn']=copy.deepcopy(animation['spawn'])
                count=1+(i%3);s['abilities'].update(slots=[0]*count,pool=[dict(present=True,destroyed=False,cooldown=10*ONE,stopped_during_time_stop=False,ground_only=False,castable=True,ignore_next_cooldown=False)],timers=[12*ONE]*3,indicators=[dict(seconds_since_reset=10000.0,lifecycle=dict(active=True,follow=False)) for _ in range(count)],can_use=False,buttons=[False]*3)
                r['scheduler']['players'][i]['flags']=dict(spark=True,macho=True,platform=True)
                r['combat']['revival'][i]['spawn_position']=vec(2*i-5,12)
            for a in q['actions']:a['commands']=[]
            while len(q['actions'])<210:q['actions'].append(copy.deepcopy(q['actions'][-1]))
            def command(action):return dict(op='actor_spawn',actor=victim,action=action,position=vec(2*victim-5,12))
            if mode in ['respawn','fast']:
                q['actions'][5]['commands']=[command('prepare')];q['actions'][15]['commands']=[command(mode)]
            if mode in ['physics_respawn','body_fast']:
                phase='physics:*' if mode=='physics_respawn' else 'late:'+str(next(i for i,c in enumerate(r['components']) if c['actor']==victim and c['role']=='body'))
                q['scripts'][phase]=[[] for _ in range(5)]+[[command('respawn' if mode=='physics_respawn' else 'fast')]]
            if mode=='fast':
                q['actions'][110]['commands']=[command('prepare')];q['actions'][130]['commands']=[command('fast')]
            if mode=='permanent':
                r['combat']['revival'][victim]['status']=3;r['combat']['death_effects']=dict(prefab=death_prefab(),instances=[]);q['registration_state']=True
                q['actions'][5]['commands']=[dict(op='kill_actor',actor=victim,request=dict(killer=1,spawn_effect=True,ignore_invulnerability=True,cause=0))]
            if mode=='time_stop':
                for a in q['actions'][:40]:a['scale']=0
                r['scheduler']['players'][0]['protected']=True
            if mode=='missing_animation':del r['actors'][victim]['slime']['animations']['spawn']
            if mode=='missing_renderer':r['actors'][victim]['slime']['animator']['render']=None
            if mode in ['null_ability','missing_ability','null_indicator']:
                s=r['actors'][victim]['slime']
                if mode=='null_ability':s['abilities']['slots'][0]=None
                if mode=='missing_ability':s['abilities']['pool'][0]['present']=False
                if mode=='null_indicator':s['abilities']['indicators'][0]=None
                q['actions'][0]['commands']=[command('spawn' if mode=='null_indicator' else 'prepare')]
            yield q

def initial_queries(seed):
    for q in queries(seed):
        if q['spawn_case']!='init':continue
        q['spawn_case']='first_spawn'
        for actor in q['runtime']['actors']:actor['slime']['state']['has_spawned']=False
        yield q


def coverage(directory,qs):
    previous={};counts=dict(spawn_completions=0,body_simulation_stopped=0,body_simulation_started=0,physics_disabled=0,physics_enabled=0,indicator_activations=0,cleared_player_flag_sets=0)
    def observe(case,state):
        before=previous.get(case,qs[case]['runtime'])
        for a,b in zip(before['actors'],state['actors']):
            counts['spawn_completions']+=not a['slime']['state']['has_spawned'] and b['slime']['state']['has_spawned']
            counts['body_simulation_stopped']+=a['entity']['body_simulated'] and not b['entity']['body_simulated']
            counts['body_simulation_started']+=not a['entity']['body_simulated'] and b['entity']['body_simulated']
            counts['physics_disabled']+=a['slime']['spawn']['physics_enabled'] and not b['slime']['spawn']['physics_enabled']
            counts['physics_enabled']+=not a['slime']['spawn']['physics_enabled'] and b['slime']['spawn']['physics_enabled']
            for x,y in zip(a['slime']['abilities']['indicators'],b['slime']['abilities']['indicators']):
                if x and y:counts['indicator_activations']+=not x['lifecycle']['active'] and y['lifecycle']['active']
        for a,b in zip(before.get('players',before.get('scheduler',{}).get('players',[])),state['players']):
            counts['cleared_player_flag_sets']+=any(a['flags'].values()) and not any(b['flags'].values())
        previous[case]=state
    with (directory/'unity.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if 'state' in row:observe(row['case'],row['state'])
            elif 'failed' in row.get('result',{}):observe(row['case'],row['result']['failed'])
    return counts

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=725105);p.add_argument('--golden',action='store_true');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--initial',action='store_true');p.add_argument('--directory');p.add_argument('--cases',type=int,default=0);a=p.parse_args()
    name='actor-spawn-initial' if a.initial else 'actor-spawn';fixture=ROOT/f'native/fixtures/{name}.jsonl.gz'
    if a.golden:result=golden(fixture)
    else:
        directory=ROOT/f'native/artifacts/{a.directory or name}';qs=list(initial_queries(a.seed) if a.initial else queries(a.seed));qs=qs[:a.cases] if a.cases else qs
        inputs,hashes=capture(directory,qs,a.reuse_unity);result=dict(verify(directory,inputs),provenance=hashes)
        if result['match']:result['spawn_coverage']=coverage(directory,inputs)
        (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['match'] and not a.cases:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(result,indent=2));return int(not result['match'])
if __name__=='__main__':raise SystemExit(main())
