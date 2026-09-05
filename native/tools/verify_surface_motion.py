"""Verify blocked platform traversal and hop queries against actual Unity methods."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
import random
import subprocess
from reference import ROOT, first_difference
from verify_beam import encoded, reference
from verify_player_rays import queries as ray_queries
from verify_player_ground import vec
from verify_dynamics import ONE

FIXTURE = ROOT / 'native/fixtures/surface-motion.jsonl.gz'
SCHEMA = 'bopl-surface-motion-v1'


def queries(count, seed):
    rng = random.Random(seed)
    for case, base in enumerate(ray_queries(count, seed)):
        scene = base['scene']
        if case % 2 == 0:
            scene['composites'] = []
            scene['objects'] = scene['objects'][:1]
            scene['objects'][0]['composite'] = None
        prototype = deepcopy(scene['objects'][0])
        root = deepcopy(prototype)
        root.update(id=10000, active=case % 17 != 0, destroyed=case % 29 == 0, enabled=case % 7 != 0,
            composite=None, relative_pos=vec(), relative_orientation=0)
        def rectangle(center, x, y, radius):
            return dict(kind='rounded_rect', shape=dict(inner=dict(center=center,
                right=vec(x,0),up=vec(0,y),inverse_extents=vec(1/x,1/y)), radius=int(radius*ONE)))
        center = vec(rng.randint(-8,8),rng.randint(-6,6)) if case%4==3 else vec()
        right = [vec(1,0),vec(0,1),vec(.6,.8),vec(-1,0)][(case//4)%4] if case%4==3 else vec(1,0)
        up = dict(x=-right['y'],y=right['x'])
        width,height,corner = (rng.choice([.125,3,10]),rng.choice([.125,1,5]),rng.choice([.25,.5,4])) if case%4==3 else (3,1,.5)
        root['collider'] = rectangle(center,width,height,corner)
        root['collider']['shape']['inner'].update(right={k:int(v*width) for k,v in right.items()},up={k:int(v*height) for k,v in up.items()})
        root['transform'].update(position=deepcopy(center),rotation=0,offset=vec())
        scene['objects'].append(root)
        for i in range(case % 6 + 1):
            obj = deepcopy(root)
            obj.update(id=11000+i, active=i % 5 != 4, destroyed=case % 37 == 0, enabled=(case+i) % 3 != 0)
            center = vec((-1 if i%2 else 1)*(5+i), 2.5 if i%3 else -2)
            obj['collider'] = rectangle(center,.5,3,.25)
            obj['transform']['position'] = deepcopy(center)
            scene['objects'].append(obj)
        rng.shuffle(scene['objects'])
        by_id = {t['object']: base['platforms'][t['ground']] for t in base['targets'] if t['ground'] is not None}
        surface = dict(center=deepcopy(root['collider']['shape']['inner']['center']),right=right,up=up,extents=vec(width,height),radius=int(corner*ONE))
        bindings = []
        for obj in scene['objects']:
            if obj['collider']['kind'] != 'rounded_rect': continue
            platform = deepcopy(by_id.get(obj['id'], dict(surface=surface, destroyed=False, has_shape=True)))
            if obj['id'] == 10000:
                platform = dict(surface=deepcopy(surface),destroyed=bool(case%11==0),has_shape=case%31!=0)
                if case % 43 == 0: platform['surface'].update(extents=vec(),radius=0)
            elif obj['id'] >= 11000 and case % 13 == 0: platform = None
            bindings.append(dict(object=obj['id'],platform=platform,
                transform_destroyed=bool((case+obj['id'])%11==0),active_self=obj['active']))
        radius = 0 if case%43==0 else ONE
        actions = []
        for i, dt in enumerate([ONE, 0, ONE//10, -ONE, 4*ONE]):
            request = dict(position=vec(0,2.5),radius=radius,speed=(20 if i%2==0 else -20)*ONE,
                dt=dt,local=[0,ONE//4,ONE//2,ONE-1,-ONE//4][i],blocked=True)
            if i==0:
                actions.append(dict(op='move',object=10000,request=dict(request,blocked=False)))
            # Hopping runs before cleanup to exercise its distinct inclusion rules.
            for disallowed in [[],[-1]*10,[10000,-1]]:
                actions.append(dict(op='hop',object=10000,request=request,disallowed=disallowed))
            actions.append(dict(op='limit',object=10000,request=request,velocity=vec(20 if i%2==0 else -20, 0)))
            actions.append(dict(op='move',object=10000,request=request))
            actions.append(dict(op='multiple',object=10000,request=request,second=[vec(0,-2.5),ONE//4]))
            request = dict(request,position=vec(rng.randint(-10,10),rng.randint(-8,8)),
                radius=rng.choice([0,ONE//8,ONE,3*ONE]),speed=rng.randint(-40,40)*ONE,local=rng.randint(-ONE,2*ONE),blocked=bool(i%2))
            actions.append(dict(op='move',object=10000,request=request))
        yield dict(operation='dynamics_surface_motion',scene=scene,bindings=bindings,actions=actions)


def check(records):
    result = subprocess.run([str(ROOT / 'native/target/release/dynamics-probe')],
        input=encoded(r['query'] for r in records),text=True,capture_output=True,check=True)
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    if len(outputs) != len(records): raise ValueError('Incomplete surface motion output')
    counts, actions, errors = (Counter() for _ in range(3))
    failures = []
    for i,(record,output) in enumerate(zip(records,outputs)):
        expected = record['expected']
        if isinstance(expected,list):
            frames=expected;counts['complete_scenarios']+=1
        else:
            frames=expected['completed'];errors[expected['error']]+=1
        counts['completed_actions']+=len(frames)
        for action,frame in zip(record['query']['actions'],frames):
            actions[action['op']]+=1
            value=frame['result']
            if action['op']=='hop':
                counts['hop_blockers']+=value['blocking'] is not None
                counts['hop_remainders']+=value['remaining']!=0
            if action['op']=='limit': counts['limit_blockers']+=value[1] is not None
        if output!=expected:
            counts['mismatches']+=1
            if len(failures)<10:
                diff=first_difference(expected,output)
                compact=lambda v: dict(type=type(v).__name__,length=len(v)) if isinstance(v,(dict,list)) else v
                failures.append(dict(case=i,difference=[diff[0],compact(diff[1]),compact(diff[2])]))
    return dict(match=not counts['mismatches'],cases=len(records),**counts,actions=dict(actions),errors=dict(errors),first_failures=failures)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=int,default=1000);p.add_argument('--seed',type=int,default=92734)
    p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true')
    args=p.parse_args()
    if args.golden:
        with gzip.open(FIXTURE,'rt') as stream:
            header=json.loads(next(stream));records=[json.loads(line) for line in stream]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid surface motion fixture')
        report=check(records)
    else:
        if args.cases<1:p.error('--cases must be positive')
        directory=ROOT/'native/artifacts/surface-motion'
        records,hashes=reference(directory,list(queries(args.cases,args.seed)),args.reuse_unity)
        report=dict(check(records),provenance=hashes)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            selected=set(range(min(32,len(records))))|set(random.Random(args.seed).sample(range(len(records)),min(128,len(records))))
            selected.update(i for i,r in enumerate(records) if isinstance(r['expected'],dict))
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,
                scope='Actual StickyRoundedRectangle traversal, multiple-body limits and direct hop queries; supplied caches, non-null collider transforms; excludes recursive player hopping')
            FIXTURE.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])


if __name__=='__main__':raise SystemExit(main())
