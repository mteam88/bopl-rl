"""Original onExitAbility with continued scheduled motion and exact failure state."""
import argparse
import copy
import json
import random
from actor_stream import ROOT, capture, verify, promote, golden
from verify_actor_entity import queries as actor_queries
from verify_actors import ONE, vec


def exit_info(actor, rng, grounds):
    grounded = bool(rng.randrange(2))
    return dict(dead=False, killer=2, position=vec(rng.randrange(-12,13), rng.randrange(2,18)), external_velocity=vec(rng.randrange(-3,4),rng.randrange(-3,4)), velocity=vec(rng.randrange(-8,9),rng.randrange(-8,9)), facing_right=bool(rng.randrange(2)), grounded=grounded, grounded_speed=rng.randrange(-5,6)*ONE, sprite=None, ground=rng.randrange(grounds), local=rng.randrange(0,9)*ONE//2, just_landed=grounded and bool(rng.randrange(2)), cause=0, radius=actor['controller']['physics']['motion']['radius'], enter_turn=bool(rng.randrange(2)), enter_jump=grounded and bool(rng.randrange(2)), rope=None, top=bool(rng.randrange(2)))


def queries(count, seed, steps):
    rng = random.Random(seed + 181)
    for case, q in enumerate(actor_queries(count, seed, steps)):
        r = q['runtime'];actors = r['actors'];n = len(actors)
        for i, actor in enumerate(actors):
            actor['entity']['active'] = False
            slime = actor['slime'];slime['abilities']['in_ability'] = True
            slime['entry']['invisible_on_entry'] = (case+i)%3==0
            slime['abilities']['indicators'] = [dict(seconds_since_reset=rng.choice([-2.,0.,3.,10000.]), lifecycle=dict(active=bool(rng.randrange(2)),follow=True)) for _ in slime['abilities']['slots']]
        for tick, action in enumerate(q['actions']):
            if tick%37==0:
                actor=tick//37%n
                action['commands'].append(dict(op='slime_exit',actor=actor,info=exit_info(actors[actor],rng,len(r['bindings']))))
        yield q


def faults(seed):
    base=next(queries(1,seed,1))
    for name in ('dead_missing_collision','indicator_null','indicator_missing_follow','indicator_array_short','invalid_ground_landing','invalid_ground_jump','expiry_null_slot','grounded','air'):
        q=copy.deepcopy(base);q['fault']=name;q['scripts']={}
        actor=q['runtime']['actors'][0];s=actor['slime'];s['entry']['invisible_on_entry']=False
        action=q['actions'][0];action['enabled']=[False]*len(action['enabled'])
        info=exit_info(actor,random.Random(seed),len(q['runtime']['bindings']));info.update(grounded=True,ground=0,just_landed=False,enter_jump=False,enter_turn=False)
        action['commands']=[dict(op='slime_exit',actor=0,info=info)]
        if name=='dead_missing_collision':info['dead']=True
        elif name=='indicator_null':s['abilities']['indicators'][0]=None
        elif name=='indicator_missing_follow':s['abilities']['indicators'][0]['lifecycle']['follow']=False
        elif name=='indicator_array_short':s['abilities']['indicators']=[]
        elif name=='invalid_ground_landing':info.update(ground=None,just_landed=True)
        elif name=='invalid_ground_jump':info.update(ground=None,enter_jump=True)
        elif name=='expiry_null_slot':s['entry']['invisible_on_entry']=True;s['abilities']['slots'][0]=None
        elif name=='air':info.update(grounded=False,ground=None)
        yield q


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=int,default=48);p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=781438)
    p.add_argument('--golden',action='store_true');p.add_argument('--faults',action='store_true');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--name',default='slime-exit')
    a=p.parse_args();fixture=ROOT/f'native/fixtures/{a.name}.jsonl.gz'
    if a.golden:report=golden(fixture)
    else:
        directory=ROOT/f'native/artifacts/{a.name}';inputs,hashes=capture(directory,list(faults(a.seed) if a.faults else queries(a.cases,a.seed,a.steps)),a.reuse_unity)
        report=dict(verify(directory,inputs),provenance=hashes);(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:promote(directory,fixture,inputs,set(range(min(12,len(inputs))))|set(report['error_cases']),hashes,'Original SlimeController ability return callback and continued actor motion; invocation supplied explicitly, no ordinary ability entry or PlayerCollision bound')
    print(json.dumps(report,indent=2));return int(not report['match'])

if __name__=='__main__':raise SystemExit(main())
