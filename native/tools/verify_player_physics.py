"""Verify ability physics handoff, scaling, jumps and grounded rope coupling."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
import math
import random
import subprocess
from reference import ROOT, first_difference
from verify_beam import encoded, reference
from verify_player_ground import queries as ground_queries, vec
from verify_rope import queries as rope_queries
from verify_dynamics import ONE, sample

FIXTURE = ROOT / 'native/fixtures/player-physics.jsonl.gz'
SCHEMA = 'bopl-player-physics-v1'


def queries(count, seed):
    rng = random.Random(seed)
    grounds = iter(ground_queries(count, seed))
    ropes = iter(rope_queries(count, seed + 1))
    for case in range(count):
        base = next(grounds)
        motion = base['state']
        platforms = [dict(surface=s, destroyed=bool((case + i) % 13 == 0), has_shape=(case + i) % 19 != 0) for i, s in enumerate(base['surfaces'])]
        # Keep positive shape radii; zero scale and missing shapes have dedicated branches.
        for p in platforms: p['surface']['radius'] = max(p['surface']['radius'], ONE // 4)
        state = dict(motion=motion, scale=rng.choice([0, ONE // 4, ONE, 2 * ONE]),
            previous_scale=0 if case % 29 == 0 else ONE, jumped=bool(case % 2),
            sync_rotation=bool(case % 3), cannot_unground=bool(case % 7 == 0))
        config = dict(gravity=sample(rng, -2, 8), max_fall_speed=0 if case % 31 == 0 else 27 * ONE,
            gravity_modifier=rng.choice([0, ONE // 2, ONE, 2 * ONE]), air_accel=10 * ONE,
            max_speed=-10 * ONE if case % 37 == 0 else 19 * ONE, inverse_mass=sample(rng, 0, 70),
            jump_strength=30 * ONE, jump_extra_x=int(.6 * ONE), jump_momentum=int(.65 * ONE),
            jump_normal_scale=0 if case % 41 == 0 else ONE, jump_teleport=int(.02 * ONE), rope_strength=sample(rng, 0, 160))
        r = next(ropes)['rope']
        r['collision_count'] = 0
        r['colliding_rects'] = [0] * 32
        if case % 43 == 0: r['count'] = 0
        rope = None if case % 11 == 0 else dict(physics=r, owner=case % 6 + 1, time_air_start=0, time_air_end=0)
        top = bool(case % 2)
        if case % 10 == 0:
            platforms[0] = dict(surface=dict(center=vec(), right=vec(1, 0), up=vec(0, 1), extents=vec(3, 2), radius=ONE), destroyed=False, has_shape=True)
            motion.update(grounded=True, ground=0, local=0, radius=ONE, external_velocity=vec())
            r['enabled'] = True
            r['time_since_hook_arrived'] = ONE
            r['count'] = 3
            end = 0 if top else 2
            root = math.isqrt(int(.07 * ONE) * ONE) + [-1, 0, 1][(case // 10) % 3]
            r['segment'][end] = dict(x=-3 * ONE, y=4 * ONE + root)
            rope = dict(physics=r, owner=case % 6 + 1, time_air_start=0, time_air_end=0)
            config['rope_strength'] = 120 * ONE
            state['cannot_unground'] = False
        def info(i):
            return dict(grounded=bool((case + i) % 3) or case % 17 == 0, ground=[None, 0, 1][(case + i) % 3],
                local=rng.choice([0, ONE // 4, ONE // 2, ONE - 1, ONE, ONE + 1]), radius=rng.choice([0, ONE // 4, ONE, 3 * ONE]),
                grounded_speed=sample(rng, -40, 40), position=vec(rng.randint(-8, 8), rng.randint(-8, 8)),
                velocity=vec(rng.randint(-20, 20), rng.randint(-20, 20)), external_velocity=vec(rng.randint(-20, 20), rng.randint(-20, 20)))
        actions = [dict(op='late'), dict(op='scale', scale=state['scale']), dict(op='gravity'), dict(op='air', input=rng.choice([-ONE, 0, ONE])),
            dict(op='force', force=vec(.5, -.25)), dict(op='offset', offset=vec() if case % 5 == 0 else vec(.01, -.02))]
        for i in range(3):
            actions.extend([dict(op='sync', exit=bool((case + i) % 2), info=info(i)), dict(op='late'), dict(op='jump'), dict(op='align')])
        yield dict(operation='dynamics_player_physics', state=deepcopy(state), config=config, platforms=platforms,
            rope=rope, top=top, owner=case % 6 + 1, space=bool(case % 2), event_offset=vec(.01, -.02) if case % 3 == 0 else vec(), actions=actions)


def check(records):
    p = subprocess.run([str(ROOT / 'native/target/release/dynamics-probe')], input=encoded(r['query'] for r in records), text=True, capture_output=True, check=True)
    outputs = [json.loads(line) for line in p.stdout.splitlines()]
    if len(outputs) != len(records): raise ValueError('Incomplete native player physics output')
    counts = Counter()
    actions = Counter()
    errors = Counter()
    failures = []
    for i, (r, output) in enumerate(zip(records, outputs)):
        expected = r['expected']
        if isinstance(expected, list):
            frames = expected
            counts['complete_scenarios'] += 1
        else:
            frames = expected['completed']
            errors[expected['error']] += 1
        counts['completed_actions'] += len(frames)
        counts['detach_events'] += sum(len(f['events']) for f in frames)
        actions.update(a['op'] for a in r['query']['actions'][:len(frames)])
        if output != expected:
            counts['mismatches'] += 1
            if len(failures) < 10:
                diff = first_difference(expected, output)
                compact = lambda v: dict(type=type(v).__name__, length=len(v)) if isinstance(v, (dict, list)) else v
                failures.append(dict(case=i, difference=[diff[0], compact(diff[1]), compact(diff[2])]))
    return dict(match=not counts['mismatches'], cases=len(records), **counts, actions=dict(actions), errors=dict(errors), first_failures=failures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=int, default=1000)
    p.add_argument('--seed', type=int, default=83621)
    p.add_argument('--reuse-unity', action='store_true')
    p.add_argument('--golden', action='store_true')
    args = p.parse_args()
    if args.golden:
        with gzip.open(FIXTURE, 'rt') as f:
            header = json.loads(next(f))
            records = [json.loads(line) for line in f]
        if header['schema'] != SCHEMA or header['cases'] != len(records): raise ValueError('Invalid player physics fixture')
        report = check(records)
    else:
        if args.cases < 1: p.error('--cases must be positive')
        directory = ROOT / 'native/artifacts/player-physics'
        records, hashes = reference(directory, list(queries(args.cases, args.seed)), args.reuse_unity)
        report = dict(check(records), provenance=hashes)
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if report['match']:
            selected = set(range(min(24, len(records)))) | set(random.Random(args.seed).sample(range(len(records)), min(128, len(records))))
            selected.update(i for i, r in enumerate(records) if isinstance(r['expected'], dict))
            header = dict(schema=SCHEMA, cases=len(selected), provenance=hashes,
                scope='Actual PlayerPhysics state transitions and grounded rope late update; configurable detachment listener; no raycast or surface movement controller')
            FIXTURE.write_bytes(gzip.compress(encoded([header, *[records[i] for i in sorted(selected)]]).encode(), mtime=0))
    print(json.dumps(report, indent=2))
    return int(not report['match'])


if __name__ == '__main__': raise SystemExit(main())
