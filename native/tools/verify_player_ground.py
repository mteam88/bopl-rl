"""Differential tests for real PlayerPhysics attachment and detachment methods."""
import argparse
import gzip
import json
import random
import subprocess
from reference import ROOT, first_difference
from verify_beam import encoded, reference

ONE = 1 << 32
FIXTURE = ROOT / 'native/fixtures/player-ground.jsonl.gz'
SCHEMA = 'bopl-player-ground-v1'


def vec(x=0, y=0):
    return dict(x=int(x * ONE), y=int(y * ONE))


def queries(count, seed):
    rng = random.Random(seed)
    for case in range(count):
        surfaces = []
        for i in range(2):
            right = [vec(1, 0), vec(0, 1), vec(-1, 0), vec(.6, .8)][(case + i) % 4]
            surfaces.append(dict(center=vec(rng.randint(-30, 30), rng.randint(-20, 20)),
                right=right, up=dict(x=-right['y'], y=right['x']),
                extents=vec(rng.choice([.125, 1, 4, 20]), rng.choice([.125, 1, 3])),
                radius=rng.choice([0, ONE // 10, ONE, 5 * ONE])))
        radius = rng.choice([0, ONE // 8, int(.76 * ONE), ONE, 4 * ONE])
        if case % 64 == 0:
            surfaces[0]['extents'] = vec()
            surfaces[0]['radius'] = radius = 0
        state = dict(position=vec(rng.randint(-35, 35), rng.randint(-25, 25)),
            rotation=rng.randint(-7 * ONE, 7 * ONE), up=vec(.8, .6), right=vec(.6, -.8),
            velocity=vec(rng.randint(-100, 100), rng.randint(-100, 100)),
            external_velocity=vec(rng.randint(-50, 50), rng.randint(-50, 50)),
            radius=radius, grounded=bool(case % 2), ground=[None, 0, 1][case % 3],
            local=rng.randint(-2 * ONE, 2 * ONE), grounded_speed=rng.randint(-30 * ONE, 30 * ONE))
        actions = [dict(op='unground', null_rotation=bool(case % 2))] if case % 4 == 0 else []
        for i in range(4):
            s = surfaces[i % 2]
            # Include edge/corner equality as well as unrelated requested positions.
            position = dict(x=s['center']['x'] + s['extents']['x'], y=s['center']['y'] + s['extents']['y']) if i == 0 and case % 8 == 0 else vec(rng.randint(-40, 40), rng.randint(-30, 30))
            actions.extend([dict(op='attach', surface=i % 2, position=position),
                dict(op='unground', null_rotation=bool((case + i) % 2)),
                dict(op='unground', null_rotation=bool((case + i + 1) % 2))])
        yield dict(operation='dynamics_player_ground', state=state, surfaces=surfaces, actions=actions)


def check(records):
    result = subprocess.run([str(ROOT / 'native/target/release/dynamics-probe')],
        input=encoded(r['query'] for r in records), text=True, capture_output=True, check=True)
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    if len(outputs) != len(records):
        raise ValueError('Incomplete native ground output')
    failures = []
    actions = errors = events = attachments = 0
    for i, (r, output) in enumerate(zip(records, outputs)):
        expected = r['expected']
        frames = expected if isinstance(expected, list) else expected['completed']
        actions += len(frames)
        errors += isinstance(expected, dict)
        events += sum(frame['event'] is not None for frame in frames)
        attachments += sum(action['op'] == 'attach' for action in r['query']['actions'][:len(frames)])
        if output != expected and len(failures) < 10:
            diff = first_difference(expected, output)
            compact = lambda v: dict(type=type(v).__name__, length=len(v)) if isinstance(v, (dict, list)) else v
            failures.append(dict(case=i, difference=[diff[0], compact(diff[1]), compact(diff[2])]))
    return dict(match=not failures, cases=len(records), completed_actions=actions,
        complete_scenarios=len(records) - errors, failing_actions=errors, attachments=attachments,
        detach_events=events, first_failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=72931)
    parser.add_argument('--golden', action='store_true')
    parser.add_argument('--reuse-unity', action='store_true')
    args = parser.parse_args()
    if args.golden:
        with gzip.open(FIXTURE, 'rt') as stream:
            header = json.loads(next(stream))
            records = [json.loads(line) for line in stream]
        if header['schema'] != SCHEMA or header['cases'] != len(records):
            raise ValueError('Invalid ground fixture')
        report = check(records)
    else:
        if args.cases < 1:
            parser.error('--cases must be positive')
        directory = ROOT / 'native/artifacts/player-ground'
        records, hashes = reference(directory, list(queries(args.cases, args.seed)), args.reuse_unity)
        report = dict(check(records), provenance=hashes)
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if report['match']:
            selected = set(range(min(24, len(records)))) | set(random.Random(args.seed).sample(range(len(records)), min(128, len(records))))
            selected.update(i for i, r in enumerate(records) if isinstance(r['expected'], dict))
            header = dict(schema=SCHEMA, cases=len(selected), provenance=hashes,
                scope='Actual attachment and detachment methods; records listener calls; excludes rendering and jump sound')
            FIXTURE.write_bytes(gzip.compress(encoded([header, *[records[i] for i in sorted(selected)]]).encode(), mtime=0))
    print(json.dumps(report, indent=2))
    return int(not report['match'])


if __name__ == '__main__':
    raise SystemExit(main())
