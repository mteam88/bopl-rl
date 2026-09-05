"""Real actor GameObjects, original DPhysicsBox and scheduled component activity."""
import argparse
import copy
import json
import random
from collections import Counter
from actor_stream import ROOT, capture, verify, promote, golden, records
from verify_invisibility import queries as actor_queries
from verify_actors import ONE, vec


def queries(count, seed, steps, scale=True):
    rng = random.Random(seed)
    for case, q in enumerate(actor_queries(count, seed, steps)):
        q['scripts'] = {}
        r = q['runtime']
        n = len(r['actors'])
        for i, actor in enumerate(r['actors']):
            actor['push_strength'] = None  # Variable sizes need the bite integration.
            actor['entity'] = dict(node=i, active=True, body_simulated=True, box=dict(scale=ONE, min_scale=ONE//10, max_scale=5*ONE, minimum_extents=4294967, start_extents=vec(1, 1), initialized=True))
            actor['controller']['physics']['scale'] = ONE
            actor['controller']['physics']['previous_scale'] = ONE
            for c, binding in zip(r['scheduler']['components'], r['components']):
                if binding['actor'] == i:
                    c['ignore_activity'] = False
            for a in actor['slime']['abilities']['pool']:
                a['cooldown'] = 10000*ONE
        for player in r['scheduler']['players']:
            player['scale'] = ONE
        for tick, action in enumerate(q['actions']):
            for entry in action['routing_inputs']:
                entry.update(can_use=True, buttons=[False]*3)
            action['commands'] = []
            if tick % 11 == 0:
                action['commands'].append(dict(op='actor_active', actor=rng.randrange(n), value=bool((tick//11)%2)))
            if tick % 7 == 0:
                action['commands'].append(dict(op='body_simulated', actor=rng.randrange(n), value=bool((tick//7)%2)))
            if scale and tick % 5 == 0:
                action['commands'].append(dict(op='box_scale', actor=rng.randrange(n), value=rng.choice([ONE//4, ONE//2, ONE, ONE*2, ONE*3, ONE*8, -ONE])))
            if tick % 17 == 0:
                action['commands'].append(dict(op='player_scale', player=rng.randrange(n), value=rng.choice([ONE//8, ONE//2, ONE, ONE*2, ONE*8])))
            if tick % 19 == 0:
                action['commands'].append(dict(op='box_enabled', actor=rng.randrange(n), value=bool((tick//19)%2)))
            action['commands'].append(dict(op='box_sync', actor=rng.randrange(n)))
            q['scripts'].setdefault('magnet:*', []).append([dict(op='actor_active', actor=rng.randrange(n), value=True)])
        yield q


def faults(seed):
    base = next(queries(1, seed, 1))
    for name in ('old_box_zero', 'old_transform_zero', 'new_scale_zero', 'uninitialized_zero', 'sync_zero_x', 'sync_zero_y', 'uninitialized_sync', 'normal_scale'):
        q = copy.deepcopy(base)
        q['fault'] = name
        actor = q['runtime']['actors'][0]
        box = actor['entity']['box']
        obj = next(o for o in q['runtime']['scene']['objects'] if o['id'] == actor['object'])
        obj['transform']['offset'] = vec(.75, -.375)
        a = q['actions'][0]
        a['enabled'] = [False] * len(a['enabled'])
        a['commands'] = [dict(op='box_scale', actor=0, value=ONE//2)]
        q['scripts'] = {}
        if name == 'old_box_zero':
            box['scale'] = 0
        elif name == 'old_transform_zero':
            actor['controller']['physics']['scale'] = 0
        elif name in ('new_scale_zero', 'uninitialized_zero'):
            box.update(min_scale=0, max_scale=0, initialized=name != 'uninitialized_zero')
        elif name.startswith('sync_zero'):
            obj['collider']['shape']['inverse_extents'][name[-1]] = 0
            a['commands'] = [dict(op='box_sync', actor=0)]
        elif name == 'uninitialized_sync':
            box['initialized'] = False
            a['commands'] = [dict(op='box_sync', actor=0)]
        yield q


def coverage(directory):
    counts, commands = Counter(), Counter()
    with (directory / 'unity.jsonl').open() as stream:
        for row in records(stream):
            if 'state' not in row:
                continue
            frame = row['state']
            for actor in frame['actors']:
                entity = actor['entity']
                counts['actor_snapshots'] += 1
                counts['active' if entity['active'] else 'inactive'] += 1
                counts['body_simulated' if entity['body_simulated'] else 'body_unsimulated'] += 1
                counts['non_unit_box_scale'] += entity['box']['scale'] != ONE
            for event in frame['report']['events']:
                counts['detached_while_unsimulated'] += event['event']['kind'] == 'detached' and not frame['actors'][event['actor']]['entity']['body_simulated']
            commands.update(c['op'] for c in frame['effects'])
    return dict(entity=dict(counts), commands=dict(commands))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=int, default=48)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--seed', type=int, default=961287)
    p.add_argument('--golden', action='store_true')
    p.add_argument('--reuse-unity', action='store_true')
    p.add_argument('--no-scale', action='store_true')
    p.add_argument('--faults', action='store_true')
    p.add_argument('--name', default='actor-entity')
    args=p.parse_args();fixture=ROOT/f'native/fixtures/{args.name}.jsonl.gz'
    if args.golden:
        report=golden(fixture)
    else:
        directory=ROOT/f'native/artifacts/{args.name}'
        inputs,hashes=capture(directory,list(faults(args.seed) if args.faults else queries(args.cases,args.seed,args.steps,not args.no_scale)),args.reuse_unity)
        report=dict(verify(directory,inputs),provenance=hashes,**coverage(directory))
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            promote(directory,fixture,inputs,set(range(min(12,len(inputs))))|set(report['error_cases']),hashes,'Scheduled real GameObject activity, body simulation gates and DPhysicsBox scale/position updates; no ordinary ability exit yet')
    print(json.dumps(report,indent=2));return int(not report['match'])

if __name__=='__main__':raise SystemExit(main())
