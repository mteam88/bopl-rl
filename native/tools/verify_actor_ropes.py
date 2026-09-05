"""Original scheduled slime/body/physics with players sharing live scene ropes."""
import argparse
import copy
import json
import random
from collections import Counter
from actor_stream import ROOT, capture, verify, promote, golden
from verify_slime_actors import queries as slime_queries
from verify_actors import ONE, DT, vec


def queries(count, seed, steps):
    rng = random.Random(seed + 273)
    for case, q in enumerate(slime_queries(count, seed, steps)):
        runtime = q['runtime']; actors = runtime['actors']; n = len(actors)
        runtime['scheduler']['loaded'] = False
        bodies = []; metadata = []
        for i in range(0, n, 2):
            first = actors[i]; second = actors[i + 1] if i + 1 < n else None
            start = copy.deepcopy(first['controller']['physics']['motion']['position'])
            end = copy.deepcopy(second['controller']['physics']['motion']['position']) if second else dict(x=start['x']+3*ONE,y=start['y']+6*ONE)
            size = rng.choice([3, 8, 16, 40]); segments = [dict(x=start['x']+(end['x']-start['x'])*j//(size-1), y=start['y']+(end['y']-start['y'])*j//(size-1)) for j in range(size)]
            segments += [vec()] * (40-size)
            previous = copy.deepcopy(segments)
            r = dict(segment=segments, previous=previous, count=size, enabled=case%13!=12,
                force_start=vec(), force_end=vec(), mass_start=ONE//2, mass_end=ONE//2 if second else ONE,
                separation=rng.choice([ONE//4, ONE//2, ONE]), time_since_hook_arrived=[ONE,0,1,-1][case%4],
                detached=case%17==16, colliding_rects=[0]*32, collision_count=0)
            index=len(bodies);bodies.append(r)
            metadata.append(dict(owner=i+1, time_air_start=rng.choice([0,ONE//4,-ONE]), time_air_end=0))
            first['body'].update(rope=index, top=True)
            if second:second['body'].update(rope=index, top=False)
        runtime['actor_ropes']=dict(metadata=metadata,player_gravity=10*ONE,max_stretch=(1000000 if case<6 else rng.choice([1,100,1000000]))*ONE)
        runtime['scene']['rope_system']=dict(bodies=bodies,gravity=10*ONE,space=runtime['space'],iterations=3,stretchiness=429496729)
        for actor in actors:
            actor['controller']['physics']['previous_scale']=ONE
            actor['controller']['skip_update']=False
            actor['config']['physics']['rope_strength']=120*ONE
            actor['body']['ignore_gravity']=case%7==6
            actor['body']['rope_mass']=rng.choice([0,ONE//4,ONE//2,ONE])
            actor['slime']['state'].update(has_spawned=True,suppress=False)
        for tick, action in enumerate(q['actions']):
            if case<6:
                action.update(dt=DT,scale=ONE,enabled=[True]*len(runtime['components']))
                for inp in action['inputs']:inp['jump']=tick==60
            elif tick<30:
                for inp in action['inputs']:inp['jump']=False
        yield q


def handoffs(count, seed, steps):
    from verify_invisibility import queries as invisibility_queries
    from verify_dash_ability import queries as dash_queries
    for kind, source in [('invisibility',invisibility_queries),('dash',dash_queries)]:
        for case,(q,rope_query) in enumerate(zip(source(count,seed,steps),queries(count,seed,steps))):
            r=q['runtime'];template=rope_query['runtime'];n=len(r['scheduler']['players'])
            q['rope_ability']=kind;r['actor_ropes']=template['actor_ropes'];r['scene']['rope_system']=template['scene']['rope_system'];r['scheduler']['loaded']=False
            r['actor_ropes']['max_stretch']=1000000*ONE
            for i in range(n):
                for key in ['rope','top','ignore_gravity','rope_mass']:r['actors'][i]['body'][key]=template['actors'][i]['body'][key]
            for actor in r['actors']:
                actor['config']['physics']['rope_strength']=120*ONE
                actor['controller']['physics']['previous_scale']=ONE
            for tick,action in enumerate(q['actions']):
                action['scale']=ONE;action['enabled']=[True]*len(r['components'])
                for inp in action['inputs']:inp['jump']=tick==90
            yield q


def faults(seed):
    base=list(queries(2,seed,6))[1]
    for name in ['late_pull','late_pull_renderer_missing','late_pull_animation_missing','late_pull_cannot_unground']:
        q=copy.deepcopy(base);q['fault']=name;r=q['runtime'];a=r['actors'][0];m=a['controller']['physics']['motion'];rope=r['scene']['rope_system']['bodies'][0]
        m.update(grounded=True,velocity=vec(),external_velocity=vec());a['body']['rope']=0;a['body']['top']=True;a['controller']['apply_gravity']=False
        a['slime']['state'].update(mode=0,has_spawned=True,suppress=False)
        rope.update(mass_start=ONE,mass_end=ONE,time_since_hook_arrived=ONE)
        for j in range(rope['count']):rope['segment'][j]=dict(x=m['position']['x']+j*ONE,y=m['position']['y']+10*ONE)
        rope['previous']=copy.deepcopy(rope['segment'])
        for action in q['actions']:
            action.update(dt=DT,scale=ONE,enabled=[c['actor']==0 and c['role']=='physics' for c in r['components']])
            for inp in action['inputs']:inp.update(direction=vec(),jump=False)
        if name=='late_pull_renderer_missing':a['slime']['animator']['render']=None
        elif name=='late_pull_animation_missing':del a['slime']['animations']['fallLeftLoop']
        elif name=='late_pull_cannot_unground':a['controller']['physics']['cannot_unground']=True
        yield q


def coverage(directory, queries):
    counts=Counter(); previous={}
    with (directory/'unity.jsonl').open() as stream:
        for row in map(json.loads,stream):
            if 'state' not in row:continue
            case=row['case'];frame=row['state'];before=previous.get(case,queries[case]['runtime']);actors=frame['actors'];ropes=frame['scene']['rope_system']['bodies']
            attached=Counter(a['body']['rope'] for a in actors if a['body']['rope'] is not None)
            counts['frames_with_shared_rope']+=any(n>1 for n in attached.values())
            for a,old in zip(actors,before['actors']):
                body=a['body'];index=body['rope'];counts['attached_player_frames']+=index is not None
                counts['detachments']+=old['body']['rope'] is not None and index is None
                if index is not None:
                    counts['grounded_attached_frames' if body['physics']['grounded'] else 'airborne_attached_frames']+=1
                    counts['arrived_attached_frames']+=ropes[index]['time_since_hook_arrived']>0
            counts['retained_rope_contact_count_frames']+=sum(r['collision_count']>0 for r in ropes)
            counts['release_sounds']+=sum(e['event']==dict(kind='sound',value='hookshotLetGo') for e in frame['report']['events'])
            for e in frame['report']['events']:
                source=actors[e['actor']].get('slime')
                if source and source.get('entry',{}).get('info',{}).get('rope') is not None:
                    if e['event']['kind']=='ability_entry':counts['ability_entries_with_rope']+=1
                    if e['event']==dict(kind='sound',value='invisibility'):counts['invisibility_casts_with_rope']+=1
            previous[case]=frame
    return dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=60);p.add_argument('--steps',type=int,default=240);p.add_argument('--seed',type=int,default=572191);p.add_argument('--abilities',action='store_true');p.add_argument('--faults',action='store_true');p.add_argument('--golden',action='store_true');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--name',default='actor-ropes');a=p.parse_args();fixture=ROOT/f'native/fixtures/{a.name}.jsonl.gz'
    if a.golden:report=golden(fixture)
    else:
        directory=ROOT/f'native/artifacts/{a.name}';inputs,hashes=capture(directory,list(faults(a.seed) if a.faults else (handoffs if a.abilities else queries)(a.cases,a.seed,a.steps)),a.reuse_unity)
        report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:report['rope_coverage']=coverage(directory,inputs)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:promote(directory,fixture,inputs,set(range(min(12,len(inputs))))|set(report['error_cases']),hashes,'Original SlimeController, PlayerBody and PlayerPhysics with shared live DetPhysics ropes' + (' and Invisibility/Dash handoffs' if a.abilities else '') + '; excludes hookshot projectile lifecycle')
    print(json.dumps(report,indent=2));return int(not report['match'])

if __name__=='__main__':raise SystemExit(main())
