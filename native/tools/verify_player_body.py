"""Compare actual PlayerBody methods and coupled rope integration with Rust."""
import argparse
from collections import Counter
import gzip
import json
import random
import subprocess
from pathlib import Path
from reference import ROOT, first_difference, provenance, read_trace
from verify_beam import encoded, reference
from verify_rope import queries as rope_queries
from verify_dynamics import ONE, sample, vec

FIXTURE = ROOT / 'native/fixtures/player-body.jsonl.gz'
SCHEMA = 'bopl-player-body-v1'


def queries(count, seed):
    rng = random.Random(seed)
    sources = iter(rope_queries(count * 2, seed))
    for case in range(count):
        ids = list(range(1, case % 6 + 2))
        holder = rng.choice([*ids, 1000]) if case % 17 else None
        players = [dict(id=i, protected=bool((case + i) % 3 == 0), jump=bool(case % 4 == 0)) for i in ids]
        ropes = []
        for i in range(2):
            r = next(sources)['rope']
            if case % 83 == 0 and i == 0:
                r['count'] = 1
                r['enabled'] = True
            r['collision_count'] = 0
            r['colliding_rects'] = [0] * 32
            r['time_since_hook_arrived'] = rng.choice([-1, 0, 1, ONE // 4, ONE, 2 * ONE])
            r['detached'] = case % 11 == i
            ropes.append(dict(physics=r, owner=rng.choice(ids), time_air_start=sample(rng, -.1, 1), time_air_end=sample(rng, -.1, 1)))
        context = dict(players=players, time_scale=rng.choice([0, ONE // 100, ONE // 2, ONE, 2 * ONE]),
            rope_gravity=sample(rng, -3, 15), player_gravity=sample(rng, -3, 15),
            max_stretch=rng.choice([ONE // 100, 100 * ONE, 1000000 * ONE]))
        state = dict(position=vec(rng, -5, 5), velocity=vec(rng, -10, 10), external_velocity=vec(rng, -10, 10),
            previous_position=vec(rng, -5, 5), previous_velocity=vec(rng, -10, 10),
            previous_imposed=vec(rng, -10, 10), previous_external=vec(rng, -10, 10), extra_swing_force=vec(rng, -5, 5),
            rope=[None, 0, 1][case % 3], top=bool(case % 2), ignore_gravity=bool(case % 3 == 0),
            rope_mass=rng.choice([0, ONE // 4, ONE // 2, ONE, -ONE // 2, 3 * ONE // 2]), holder=holder,
            physics=None if case % 4 == 0 else dict(grounded=bool(case % 3), jumped=bool(case % 5 == 0)))
        dt = rng.choice([0, 71582788, 35791394, 143165576])
        actions = [dict(op='late'), dict(op='init'), dict(op='force', force=vec(rng, -1, 1))]
        for i in range(4):
            actions.extend([dict(op='update', dt=dt), dict(op='rope_step', dt=dt), dict(op='late')])
            if i == 0: actions.append(dict(op='attach', rope=1, top=not state['top']))
            if i == 1: actions.extend([dict(op='clear'), dict(op='attach', rope=0, top=state['top'])])
            if i == 2: actions.append(dict(op='attach', rope=None, top=not state['top']))
        yield dict(operation='dynamics_player_body', state=state, context=context, ropes=ropes, actions=actions)


def check(records, output_path=None):
    process = subprocess.run([str(ROOT / 'native/target/release/dynamics-probe')],
        input=encoded(r['query'] for r in records), text=True, capture_output=True, check=True)
    if output_path: output_path.write_text(process.stdout)
    outputs = [json.loads(line) for line in process.stdout.splitlines()]
    if len(outputs) != len(records): raise ValueError('Incomplete native body output')
    failures = []
    counts = Counter()
    errors = Counter()
    actions = Counter()
    for i, (r, actual) in enumerate(zip(records, outputs)):
        expected = r['expected']
        if isinstance(expected, list):
            frames = expected
            counts['complete_scenarios'] += 1
        else:
            frames = expected['completed']
            errors[expected['error']] += 1
        counts['completed_actions'] += len(frames)
        counts['release_sounds'] += sum(frame['released'] for frame in frames)
        actions.update(action['op'] for action in r['query']['actions'][:len(frames)])
        before = dict(state=r['query']['state'], ropes=r['query']['ropes'])
        for action, frame in zip(r['query']['actions'], frames):
            index = before['state']['rope']
            if action['op'] in ('update', 'late') and index is not None:
                counts['attached_body_actions'] += 1
                rope = before['ropes'][index]['physics']
                counts['enabled_rope_body_actions'] += rope['enabled']
                counts['arrived_rope_body_actions'] += rope['enabled'] and rope['time_since_hook_arrived'] > 0
            before = frame
        if actual != expected:
            counts['mismatches'] += 1
            if len(failures) < 10:
                diff = first_difference(expected, actual)
                compact = lambda v: dict(type=type(v).__name__, length=len(v)) if isinstance(v, (list, dict)) else v
                failures.append(dict(case=i, difference=[diff[0], compact(diff[1]), compact(diff[2])]))
    return dict(match=not counts['mismatches'], cases=len(records), **counts,
        error_categories=dict(errors), actions=dict(actions), first_failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=600)
    parser.add_argument('--seed', type=int, default=97245)
    parser.add_argument('--golden', action='store_true')
    parser.add_argument('--reuse-unity', action='store_true')
    parser.add_argument('--traces', type=Path, nargs='+')
    parser.add_argument('--gameplay', action='store_true')
    args = parser.parse_args()
    fixture = ROOT / 'native/fixtures/player-body-gameplay.jsonl.gz' if args.traces or args.gameplay else FIXTURE
    if args.traces:
        records = []
        sources = []
        for path in args.traces:
            source = provenance(path)
            start = len(records)
            for row in read_trace(path):
                if row['kind'] != 'state': continue
                for stage in row['value'].get('playerBodyStages', []):
                    if 'capture_error' in stage: raise ValueError(stage['capture_error'])
                    records.append(dict(stage, source=len(sources), tick=row['value']['tick']))
            if len(records) == start: raise ValueError('No player body stages captured')
            sources.append(source)
        directory = ROOT / 'native/artifacts/player-body-gameplay'
        directory.mkdir(parents=True, exist_ok=True)
        report = dict(check(records, directory / 'native.jsonl'), provenance=sources)
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if report['match']:
            selected = set(range(min(24, len(records)))) | set(random.Random(args.seed).sample(range(len(records)), min(128, len(records))))
            selected.update(i for i, r in enumerate(records) if r['query']['state']['rope'] is not None or isinstance(r['expected'], dict))
            header = dict(schema=SCHEMA, cases=len(selected), provenance=sources,
                scope='Independent actual PlayerBody update and late-update calls; Unity supplies each pre-state')
            fixture.write_bytes(gzip.compress(encoded([header, *[records[i] for i in sorted(selected)]]).encode(), mtime=0))
    elif args.golden:
        with gzip.open(fixture, 'rt') as stream:
            header = json.loads(next(stream))
            records = [json.loads(line) for line in stream]
        if header['schema'] != SCHEMA or len(records) != header['cases']: raise ValueError('Invalid body fixture')
        report = check(records)
    else:
        if args.cases < 1: parser.error('--cases must be positive')
        directory = ROOT / 'native/artifacts/player-body'
        records, hashes = reference(directory, list(queries(args.cases, args.seed)), args.reuse_unity)
        report = dict(check(records, directory / 'native.jsonl'), provenance=hashes)
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if report['match']:
            selected = set(range(min(24, len(records)))) | set(random.Random(args.seed).sample(range(len(records)), min(96, len(records))))
            selected.update(i for i, r in enumerate(records) if isinstance(r['expected'], dict))
            header = dict(schema=SCHEMA, cases=len(selected), provenance=hashes,
                scope='Actual PlayerBody methods and coupled rope solver; supplied player and component context; no whole-game scheduler')
            FIXTURE.write_bytes(gzip.compress(encoded([header, *[records[i] for i in sorted(selected)]]).encode(), mtime=0))
    print(json.dumps(report, indent=2))
    return int(not report['match'])


if __name__ == '__main__': raise SystemExit(main())
