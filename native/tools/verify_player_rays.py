"""Compare ordered rounded-rectangle and player raycasts with actual Unity methods."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
import random
import subprocess
from reference import ROOT, first_difference
from verify_beam import encoded, reference
from verify_player_physics import queries as player_queries
from verify_physics_scene import queries as scene_queries
from verify_player_ground import vec
from verify_dynamics import ONE

FIXTURE = ROOT / 'native/fixtures/player-rays.jsonl.gz'
SCHEMA = 'bopl-player-rays-v1'


def queries(count, seed):
    rng = random.Random(seed)
    for case, (player, world) in enumerate(zip(player_queries(count, seed), scene_queries(count, seed + 1))):
        scene = world['scene']
        state = player['state']
        motion = state['motion']
        platforms = []
        targets = []
        if case % 4 != 3:
            # First, negative-side and positive-side hits, with exact duplicate
            # geometry to expose registration-order tie breaking.
            scene['composites'] = []
            for obj in scene['objects']: obj['composite'] = None
            prototype = deepcopy(scene['objects'][0])
            scene['objects'] = [o for o in scene['objects'] if o['collider']['kind'] != 'rounded_rect']
            y = [0, -1, 1][case % 3]
            for i in range(case % 6 + 1):
                obj = deepcopy(prototype)
                center = vec(0 if i < 2 else 5 * i, y)
                obj.update(id=100 + i, active=i < 2 or bool(i % 2), destroyed=case % 17 == 0 and i == 0,
                    layer=(case + i) % 32, composite=None, relative_pos=vec(), relative_orientation=0)
                obj['collider'] = dict(kind='rounded_rect', shape=dict(inner=dict(center=center,
                    right=vec(1, 0), up=vec(0, .125), inverse_extents=vec(1, 8)), radius=ONE // 4))
                obj['transform'].update(position=deepcopy(center), rotation=0, offset=vec())
                scene['objects'].append(obj)
            rng.shuffle(scene['objects'])
            motion.update(position=vec(-8, 0), radius=ONE, velocity=vec(9, 0), external_velocity=vec(3, 0))
        for obj in scene['objects']:
            if obj['collider']['kind'] != 'rounded_rect': continue
            # The cached platform surface is deliberately independent of the
            # ray shape in random scenes, just as component caches can be.
            surface = deepcopy(player['platforms'][len(platforms) % 2]['surface'])
            if case % 4 != 3:
                surface.update(center=deepcopy(obj['collider']['shape']['inner']['center']),
                    right=vec(1, 0), up=vec(0, 1), extents=vec(1, .125), radius=ONE // 4)
            index = len(platforms)
            platforms.append(dict(surface=surface, destroyed=bool((case + index) % 13 == 0), has_shape=case % 23 != 0))
            targets.append(dict(object=obj['id'], ground=None if case % 19 == 0 else index))
        motion['ground'] = 0 if platforms and case % 2 else None
        players = [dict(id=i + 1, protected=bool((case + i) % 3 == 0)) for i in range(case % 6 + 1)]
        holder = None if case % 11 == 0 else 99 if case % 13 == 0 else case % len(players) + 1
        scale = [0, ONE // 4, ONE, 2 * ONE, -ONE][case % 5]
        actions = []
        # Exact distance cutoffs (including equality), zero directions and rays
        # from inside a shape test the complete near/far hit record.
        center = vec(0, [0, -1, 1][case % 3])
        for distance in [-ONE, 0, 27 * ONE // 4 - 1, 27 * ONE // 4, 27 * ONE // 4 + 1, 100 * ONE]:
            actions.append(dict(op='query', origin=dict(x=-8 * ONE, y=center['y']), direction=vec(1, 0), distance=distance))
        actions.extend([dict(op='query', origin=center, direction=direction, distance=20 * ONE)
                        for direction in [vec(), vec(1, 0), vec(0, -1), vec(.6, .8)]])
        for dt in [0, ONE // 60, ONE, -ONE]:
            actions.append(dict(op='velocity', attach=False, dt=dt))
        for i in range(8):
            velocity = vec(12, 0) if i < 3 else vec(rng.randint(-30, 30), rng.randint(-30, 30))
            actions.append(dict(op='collision', velocity=velocity, attach=i >= 2, dt=ONE if i < 3 else rng.choice([0, ONE // 60, ONE, 4 * ONE])))
        actions.append(dict(op='velocity', attach=True, dt=ONE))
        yield dict(operation='dynamics_player_rays', state=state, scene=scene, platforms=platforms,
            targets=targets, players=players, holder=holder, time_scale=scale,
            event_offset=vec(.01, -.02) if case % 3 == 0 else vec(), actions=actions)


def check(records):
    result = subprocess.run([str(ROOT / 'native/target/release/dynamics-probe')],
        input=encoded(r['query'] for r in records), text=True, capture_output=True, check=True)
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    if len(outputs) != len(records): raise ValueError('Incomplete player raycast output')
    counts, actions, errors, layers, player_counts = (Counter() for _ in range(5))
    failures = []
    for i, (record, output) in enumerate(zip(records, outputs)):
        expected = record['expected']
        if isinstance(expected, list):
            frames = expected
            counts['complete_scenarios'] += 1
        else:
            frames = expected['completed']
            errors[expected['error']] += 1
        player_counts[len(record['query']['players'])] += 1
        counts['completed_actions'] += len(frames)
        for action, frame in zip(record['query']['actions'], frames):
            actions[action['op']] += 1
            counts['attachment_events'] += len(frame['events'])
            hit = frame['result']
            if isinstance(hit, dict) and hit['object'] is not None:
                counts['query_hits'] += 1
                layers[hit['layer']] += 1
            if hit is True: counts['player_hits'] += 1
        if output != expected:
            counts['mismatches'] += 1
            if len(failures) < 10:
                diff = first_difference(expected, output)
                compact = lambda v: dict(type=type(v).__name__, length=len(v)) if isinstance(v, (dict, list)) else v
                failures.append(dict(case=i, difference=[diff[0], compact(diff[1]), compact(diff[2])]))
    return dict(match=not counts['mismatches'], cases=len(records), **counts, actions=dict(actions),
        errors=dict(errors), hit_layers=dict(layers), registered_player_counts=dict(player_counts), first_failures=failures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=int, default=1000)
    p.add_argument('--seed', type=int, default=48631)
    p.add_argument('--reuse-unity', action='store_true')
    p.add_argument('--golden', action='store_true')
    args = p.parse_args()
    if args.golden:
        with gzip.open(FIXTURE, 'rt') as stream:
            header = json.loads(next(stream))
            records = [json.loads(line) for line in stream]
        if header['schema'] != SCHEMA or header['cases'] != len(records): raise ValueError('Invalid player raycast fixture')
        report = check(records)
    else:
        if args.cases < 1: p.error('--cases must be positive')
        directory = ROOT / 'native/artifacts/player-rays'
        records, hashes = reference(directory, list(queries(args.cases, args.seed)), args.reuse_unity)
        report = dict(check(records), provenance=hashes)
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if report['match']:
            selected = set(range(min(64, len(records)))) | set(random.Random(args.seed).sample(range(len(records)), min(128, len(records))))
            selected.update(i for i, r in enumerate(records) if isinstance(r['expected'], dict))
            header = dict(schema=SCHEMA, cases=len(selected), provenance=hashes,
                scope='Actual rounded-rectangle and player raycasts, cleanup order, attachment and listener effects; component oracle, not complete player movement')
            FIXTURE.write_bytes(gzip.compress(encoded([header, *[records[i] for i in sorted(selected)]]).encode(), mtime=0))
    print(json.dumps(report, indent=2))
    return int(not report['match'])


if __name__ == '__main__': raise SystemExit(main())
