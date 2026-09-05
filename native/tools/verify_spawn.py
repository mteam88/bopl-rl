"""Compare SlimeController revival/spawning with real animator and ground callbacks."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import random
import subprocess
from reference import ROOT,first_difference
from verify_beam import encoded,reference
from verify_player_ground import queries as grounds,vec,ONE

FIXTURE=ROOT/'native/fixtures/spawn.jsonl.gz'
SCHEMA='bopl-spawn-v1'


def animation_assets():
    path=ROOT/'native/fixtures/animator-assets.jsonl.gz';raw=path.read_bytes();data={};template=None
    for line in gzip.decompress(raw).decode().splitlines()[1:]:
        q=json.loads(line)['query'];a=q['asset']
        if not a['path'].endswith('/MCAnimData.asset') or 'frame' in a:continue
        template=q
        if a.get('animation') in ['spawn','fallLeftLoop']:data[a['animation']]=q['actions'][0]['frames']
        if a.get('velocity_animation')=='fallLeft':data['fallLeft']=q['state']['velocity_maps'][0]
    return data,template,hashlib.sha256(raw).hexdigest()


def queries(count,seed):
    rng=random.Random(seed);assets,template,_=animation_assets()
    for case,g in enumerate(grounds(count,seed)):
        q=deepcopy(template);q.pop('asset');q['operation']='dynamics_spawn'
        q['state'].update(kind=0,current=None,velocity_maps=None,callback=None,body=0)
        q['state']['render']=dict(sprite=None if case%3 else 1,flip_x=bool(case%2),flip_y=bool(case%3))
        if case%89==0:q['state']['render']=None
        q['bodies']=[dict(velocity=vec(rng.randint(-40,40),rng.randint(-20,20)),rotation=rng.randint(-ONE,ONE)) for _ in range(3)]
        pool=[bool(rng.randrange(2)) for _ in range(6)];pool[0]=True
        revives=[i for i,v in enumerate(pool) if v]
        q['players']=[dict(id=i+1,protected=bool((case+i)%2),extra_life=(case+i)%4,spark=True,macho=True,platform=True,revive=rng.choice([None,*revives])) for i in range(case%6+1)]
        player=case%len(q['players'])+1
        holder=None if case%41==0 else player
        ability_count=case%7 if case%9==0 else rng.randrange(1,4)
        ability_ids=[rng.randrange(6) for _ in range(ability_count)]
        if case%43==0 and ability_ids:ability_ids[-1]=None
        refs=[rng.randrange(6) for _ in range(ability_count)]
        if case%47==0 and refs:refs[-1]=None
        if case%53==0:refs=refs[:1]
        cooldowns=[rng.randint(-4*ONE,4*ONE) for _ in range(ability_count if case%7 else 1)]
        q['spawn']=dict(motion=g['state'],physics_enabled=bool(case%2),simulated=bool(case%3),hurtbox_enabled=bool(case%2),renderer_enabled=bool(case%2),mode=case%11,
            suppress=bool(case%2),has_spawned=bool(case%3),fast=bool(case%7==0),invulnerable=rng.getrandbits(32),next_ability=rng.randrange(-5,10),abilityless=bool(case%7==0),
            abilities=None if case%71==0 else ability_ids,ability_pool=pool,cooldowns=None if case%73==0 else cooldowns,
            indicator_refs=None if case%79==0 else refs,indicators=[bool(rng.randrange(2)) for _ in range(6)])
        frames=deepcopy(assets['spawn']) if case%5==0 else [dict(sprite=rng.randrange(5),duration=rng.choice([-7,-3,0,1,3,7])) for _ in range(rng.randrange(1,6))]
        data=dict(spawn=frames,fall_loop=assets['fallLeftLoop'],fall_map=assets['fallLeft'])
        if case%83==0:data['spawn']=None
        if case%97==0:data['fall_map']=None
        if case%101==0:data['fall_loop']=None
        if case%107==0:data['spawn']=[]
        if case%109==0:data['spawn']=[dict(sprite=0,duration=-(1<<31))]
        q['config']=dict(animation_speed=case%3,player=99 if case%37==0 else player,holder=holder,jump_strength=rng.choice([-30*ONE,0,ONE,30*ONE,80*ONE]),tutorial=case%6==0,
            animations=None if case%113==0 else data,detach_offset=vec(rng.randrange(-2,3),rng.randrange(-2,3)),detach_velocity=vec(rng.randrange(-5,6),rng.randrange(-5,6)))
        q['holder']=holder;q['surfaces']=g['surfaces'];q['callbacks']={'1':[]}
        if case%4==0:q['state'].update(current=frames,position=rng.randrange(9),callback=1,callback_last=True)
        def position():return vec(rng.randrange(-30,31),rng.randrange(-20,21))
        actions=[dict(op='pop',direction=position()),dict(op=['init','before','prepare','respawn','fast'][case%5],position=position())]
        def ticks(count):
            for i in range(count):
                actions.append(dict(op='animate') if i%7==0 else dict(op='update',scale=0 if i%5==0 else ONE,dt=rng.choice([0,ONE//60,ONE])))
        prefix=actions;actions=[]
        if q['state']['callback'] is not None:ticks(5)
        actions.extend(prefix)
        ticks(130 if case%5==0 else 45)
        actions.append(dict(op='prepare',position=position()));ticks(4)
        actions.append(dict(op='fast',position=position()));ticks(30)
        actions.append(dict(op='respawn',position=position()));ticks(130 if case%5==0 else 45)
        actions.extend([dict(op='init'),dict(op='spawn'),dict(op='prepare',position=position()),dict(op='spawn')]);q['actions']=actions
        yield q


def check(records):
    p=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=encoded(r['query'] for r in records),text=True,capture_output=True,check=True,timeout=300)
    outputs=[json.loads(line) for line in p.stdout.splitlines()]
    if len(outputs)!=len(records):raise ValueError('Incomplete spawn output')
    counts,actions,errors,events,players=(Counter() for _ in range(5));failures=[]
    for i,(r,out) in enumerate(zip(records,outputs)):
        expected=r['expected'];q=r['query']
        players[len(q['players'])]+=1
        counts['inflight_scenarios']+=q['state']['callback'] is not None
        ids=q['spawn']['abilities'];refs=q['spawn']['indicator_refs']
        counts['duplicate_abilities']+=ids is not None and len(ids)!=len(set(ids))
        counts['duplicate_indicators']+=refs is not None and len(refs)!=len(set(refs))
        counts['over_three_abilities']+=ids is not None and len(ids)>3
        if isinstance(expected,list):frames=expected;counts['complete_scenarios']+=1
        else:
            frames=expected['completed'];errors[expected['error']]+=1
            counts['failed_fast_respawns']+=q['actions'][expected['action']]['op']=='fast' and expected['failed']['spawn']['fast']
        counts['completed_actions']+=len(frames);actions.update(a['op'] for a in r['query']['actions'][:len(frames)])
        previous=r['query']['spawn'];previous_anim=q['state']
        for a,f in zip(q['actions'],frames):
            counts['animation_spawn_callbacks']+=a['op'] in ['animate','update'] and previous_anim['callback']==1 and f['animator']['callback']!=1
            previous_anim=f['animator']
            events.update(e['kind'] for e in f['events'])
            counts['became_spawned']+=not previous['has_spawned'] and f['spawn']['has_spawned']
            previous=f['spawn']
        if out!=expected:
            counts['mismatches']+=1
            if len(failures)<10:
                diff=first_difference(expected,out);compact=lambda v:dict(type=type(v).__name__,length=len(v)) if isinstance(v,(dict,list)) else v
                failures.append(dict(case=i,difference=[diff[0],compact(diff[1]),compact(diff[2])]))
    return dict(match=not counts['mismatches'],cases=len(records),**counts,actions=dict(actions),events=dict(events),player_counts=dict(players),errors=dict(errors),first_failures=failures)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=600);p.add_argument('--seed',type=int,default=81749)
    p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');args=p.parse_args()
    if args.golden:
        with gzip.open(FIXTURE,'rt') as stream:header=json.loads(next(stream));records=[json.loads(line) for line in stream]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid spawn fixture')
        report=check(records)
    else:
        if args.cases<1:p.error('--cases must be positive')
        directory=ROOT/'native/artifacts/spawn';records,hashes=reference(directory,list(queries(args.cases,args.seed)),args.reuse_unity)
        hashes=dict(hashes,animation_fixture=animation_assets()[2]);report=dict(check(records),provenance=hashes)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            selected=set(range(min(24,len(records))))|set(random.Random(args.seed).sample(range(len(records)),min(128,len(records))))
            selected.update(i for i,r in enumerate(records) if isinstance(r['expected'],dict))
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,
                scope='Actual SlimeController Init/BeforeSpawnInit/Spawn/ReSpawn/ReSpawnFast/PrepareToRevive/PopIntoAir with SpriteAnimator and own PlayerPhysics ground-loss listener; no cloning, stocks or full world scheduler')
            FIXTURE.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])


if __name__=='__main__':raise SystemExit(main())
