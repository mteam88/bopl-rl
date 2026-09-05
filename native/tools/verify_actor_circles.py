"""Original DPhysicsCircle scaling, radius writes and synchronization in live actors."""
import argparse
import copy
import json
from collections import Counter
from actor_stream import ROOT, capture, verify, promote, golden, records
from verify_actor_entity import queries as entity_queries
from verify_actors import ONE, vec


def convert(q, mixed=False):
    for i, actor in enumerate(q['runtime']['actors']):
        if mixed and i % 2:
            continue
        old = actor['entity'].pop('box')
        actor['entity']['circle'] = {k: old[k] for k in ['scale', 'min_scale', 'max_scale', 'initialized']}
        actor['entity']['circle'].update(start_radius=ONE, manually_call_init=False)
        obj = next(o for o in q['runtime']['scene']['objects'] if o['id'] == actor['object'])
        obj['collider'] = dict(kind='circle', shape=dict(center=copy.deepcopy(obj['collider']['shape']['center']), radius=ONE))
    return q


def queries(seed, count=48, steps=240):
    for case, q in enumerate(entity_queries(count, seed, steps)):
        q = convert(q, mixed=bool(case % 2))
        q['circle_case'] = 'mixed' if case % 2 else 'circles'
        for i, actor in enumerate(q['runtime']['actors']):
            # Keep normal scenarios out of the upstream generator's deliberate divide fault.
            actor['controller']['physics']['previous_scale'] = ONE
            obj = next(o for o in q['runtime']['scene']['objects'] if o['id'] == actor['object'])
            obj['transform']['offset'] = vec(.125*(i+1), -.0625*i)
        ids = [i for i, a in enumerate(q['runtime']['actors']) if 'circle' in a['entity']]
        for tick, action in enumerate(q['actions']):
            if tick % 13 == 4:
                action['commands'].append(dict(op='circle_radius', actor=ids[(tick//13) % len(ids)], value=[ONE//4, ONE, 2*ONE][(tick//13)%3]))
        yield q


def faults(seed):
    modes = ['old_circle_zero', 'old_transform_zero', 'new_scale_zero', 'zero_then_scale',
             'uninitialized_zero', 'uninitialized_scale', 'uninitialized_sync', 'negative_radius',
             'radius_write', 'uninitialized_radius', 'normal_scale', 'reversed_clamps']
    for base in entity_queries(6, seed, 1):
        base = convert(base)
        for mode in modes:
            q = copy.deepcopy(base)
            q.update(circle_case=mode, scripts={})
            actor = q['runtime']['actors'][0]
            c = actor['entity']['circle']
            obj = next(o for o in q['runtime']['scene']['objects'] if o['id'] == actor['object'])
            obj['transform']['offset'] = vec(.75, -.375)
            action = q['actions'][0]
            action['enabled'] = [False]*len(action['enabled'])
            action['commands'] = [dict(op='box_scale', actor=0, value=ONE//2)]
            if mode == 'old_circle_zero':
                c['scale'] = 0
            elif mode == 'old_transform_zero':
                actor['controller']['physics']['scale'] = 0
            elif mode in ['new_scale_zero', 'zero_then_scale', 'uninitialized_zero']:
                c.update(min_scale=0, max_scale=0, initialized=mode != 'uninitialized_zero')
                if mode == 'zero_then_scale':
                    action['commands'] *= 2
            elif mode.startswith('uninitialized'):
                c['initialized'] = False
            elif mode == 'reversed_clamps':
                c.update(min_scale=2*ONE, max_scale=ONE//4)
            if mode == 'uninitialized_sync':
                action['commands'] = [dict(op='box_sync', actor=0)]
            if mode in ['negative_radius', 'radius_write', 'uninitialized_radius']:
                action['commands'] = [dict(op='circle_radius', actor=0, value=-ONE if mode == 'negative_radius' else ONE//8)]
            yield q


def handoffs(seed):
    from verify_ordinary_ability import queries as ordinary_queries
    for q in ordinary_queries(24, seed, 180):
        q = convert(q)
        q['circle_case'] = 'ordinary_handoff'
        for actor in q['runtime']['actors']:
            actor['controller']['physics']['previous_scale'] = ONE
        yield q


def lifecycle(seed):
    from verify_actor_lifecycle import queries as lifecycle_queries
    for q in lifecycle_queries(seed):
        q = convert(q)
        q['circle_case'] = 'lifecycle_'+q['lifecycle_case']
        yield q


def coverage(directory, inputs):
    counts = Counter()
    results, last, entries = {}, {}, Counter()
    with (directory/'unity.jsonl').open() as stream:
        for row in records(stream):
            if 'state' not in row:
                results[row['case']] = row
                continue
            last[row['case']] = row['state']
            entries[row['case']] += sum(e['event']['kind'] == 'ability_entry' for e in row['state']['report']['events'])
            for actor in row['state']['actors']:
                c = actor['entity'].get('circle')
                if c:
                    counts['circle_snapshots'] += 1
                    counts['non_unit_scale'] += c['scale'] != ONE
                    counts['uninitialized'] += not c['initialized']
                else:
                    counts['box_snapshots'] += 1
                counts['inactive'] += not actor['entity']['active']
    for case, q in enumerate(inputs):
        error = results[case]['result'].get('error')
        expected = 'division_by_zero' if q['circle_case'] in ['old_circle_zero', 'old_transform_zero', 'zero_then_scale'] else 'null_reference' if q['circle_case'] == 'lifecycle_missing' else None
        if error != expected:
            raise ValueError(f'Case {case} {q["circle_case"]}: expected {expected}, got {error}')
        mode = q['circle_case']
        if mode == 'ordinary_handoff':
            if not entries[case]:
                raise ValueError(f'Case {case}: ordinary ability never entered')
            counts['ordinary_entries'] += entries[case]
        if mode.startswith('lifecycle_'):
            state = results[case]['result'].get('failed', last.get(case))
            removed = sum(a['entity']['present'] is False for a in state['actors'])
            expected_removed = len(q['runtime']['actors']) if mode == 'lifecycle_children' else 1
            if removed != expected_removed:
                raise ValueError(f'Case {case}: expected {expected_removed} removals, got {removed}')
            counts['removed_actors'] += removed
        counts[mode] += 1
    if not counts['circle_snapshots']:
        raise ValueError('No circle states observed')
    return dict(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=513215)
    p.add_argument('--cases', type=int, default=0)
    group = p.add_mutually_exclusive_group()
    for suite in ['faults', 'handoffs', 'lifecycle']:
        group.add_argument('--'+suite, dest='suite', action='store_const', const=suite)
    p.add_argument('--golden', action='store_true')
    p.add_argument('--reuse-unity', action='store_true')
    p.add_argument('--directory')
    args = p.parse_args()
    name = 'actor-circles'+('-'+args.suite if args.suite else '')
    fixture = ROOT/f'native/fixtures/{name}.jsonl.xz'
    if args.golden:
        report = golden(fixture)
    else:
        qs = list({None: queries, 'faults': faults, 'handoffs': handoffs, 'lifecycle': lifecycle}[args.suite](args.seed))
        if args.cases:
            qs = qs[:args.cases]
        directory = ROOT/f'native/artifacts/{args.directory or name}'
        inputs, hashes = capture(directory, qs, args.reuse_unity)
        report = dict(verify(directory, inputs), provenance=hashes)
        if report['match']:
            try:
                report['circle_coverage'] = coverage(directory, inputs)
            except ValueError as error:
                report['coverage_error'] = str(error)
        (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report and not args.cases:
            promote(directory, fixture, inputs, set(range(len(inputs))), hashes, __doc__)
    print(json.dumps(report, indent=2))
    return int(not report['match'] or 'coverage_error' in report)


if __name__ == '__main__':
    raise SystemExit(main())
