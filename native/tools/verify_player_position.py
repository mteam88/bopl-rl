"""Shared player position arbitration through original SlimeController and Updater."""
import argparse
import json
import random
from collections import Counter
from actor_stream import ROOT, capture, verify, promote, golden, records
from verify_invisibility import queries as ability_queries
from verify_actors import ONE, vec


def queries(count, seed, steps):
    rng = random.Random(seed)
    for case, q in enumerate(ability_queries(count, seed, steps)):
        q['scripts'] = {}
        runtime = q['runtime']
        actors = runtime['actors']
        n = len(actors)
        # Several distinct controller identities can own the same Player.
        owners = [i if case % 4 == 0 else i % max(1, n // 2) for i in range(n)]
        q['position_owners'] = owners
        for i, actor in enumerate(actors):
            actor['body']['holder'] = runtime['scheduler']['players'][owners[i]]['id']
            actor['slime']['managed_destroyed'] = case % 7 == 0 and i == n - 1
            routing = actor['slime']['abilities']
            slots = (case + i) % 5 if case % 3 else 2
            routing['slots'] = [0] * slots
            routing['timers'] = [0] * slots
            routing['indicators'] = []
            routing['buttons'] = [False] * 3
        for player in runtime['scheduler']['players']:
            player['position'] = dict(value=vec(123, 456), current=rng.choice([None, *range(n)]), previous=rng.choice([None, *range(n)]))
            player.update(invisible=False, time_invisible=0, time_ability=0)
        # Exercise arbitration before reset, between component updates, and
        # after physics; commands call real SetPosition or set its life flag.
        for tick, action in enumerate(q['actions']):
            for i, owner in enumerate(owners):
                action['inputs'][i] = dict(action['inputs'][owner])
            for value in action['routing_inputs']:
                value.update(can_use=True, buttons=[False] * 3)
            if case < 6:
                action['enabled'] = [True] * len(runtime['components'])
            action['commands'] = []
            for _ in range(2):
                cmd = dict(op='set_position', player=rng.randrange(n), controller=rng.choice([None, *range(n)]), value=vec(rng.randrange(-100, 100), rng.randrange(-100, 100)))
                action['commands'].append(cmd)
            if tick % 13 == 0:
                action['commands'].append(dict(op='controller_destroyed', actor=rng.randrange(n), value=bool((tick // 13) % 2)))
            for key in ('player_update:0', 'update:1', 'magnet:*', 'rope_late:*'):
                q['scripts'].setdefault(key, []).append([dict(op='set_position', player=rng.randrange(n), controller=rng.choice([None, *range(n)]), value=vec(rng.randrange(-100, 100), rng.randrange(-100, 100)))])
        if case % 3 == 0:
            q['scripts'] = {}
            for action in q['actions']:
                action['commands'] = [c for c in action['commands'] if c['op'] == 'controller_destroyed']
        yield q


def coverage(path, queries):
    counts = Counter()
    previous = {}
    with path.open() as stream:
        for row in records(stream):
            if 'state' not in row:
                continue
            for i, player in enumerate(row['state']['players']):
                state = player['position']
                counts['player_snapshots'] += 1
                counts['selected'] += state['current'] is not None
                counts['previous_retained'] += state['current'] is not None and state['current'] == state['previous']
                key = (row['case'], i)
                counts['winner_changes'] += key in previous and state['current'] != previous[key]
                previous[key] = state['current']
            counts['commands'] += len(row['state']['effects'])
    return dict(counts, shared_owner_cases=sum(len(set(q['position_owners'])) < len(q['position_owners']) for q in queries))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=int, default=48)
    p.add_argument('--steps', type=int, default=240)
    p.add_argument('--seed', type=int, default=849172)
    p.add_argument('--golden', action='store_true')
    p.add_argument('--reuse-unity', action='store_true')
    p.add_argument('--name', default='player-position')
    args = p.parse_args()
    fixture = ROOT / f'native/fixtures/{args.name}.jsonl.gz'
    if args.golden:
        report = golden(fixture)
    else:
        directory = ROOT / f'native/artifacts/{args.name}'
        inputs, hashes = capture(directory, list(queries(args.cases, args.seed, args.steps)), args.reuse_unity)
        report = dict(verify(directory, inputs), provenance=hashes, position=coverage(directory / 'unity.jsonl', inputs))
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if report['match']:
            selected = set(range(min(12, len(inputs)))) | set(report['error_cases'])
            promote(directory, fixture, inputs, selected, hashes, 'Original Player.SetPosition and SlimeController updates, shared controller ownership, managed destruction and arbitrary phase writes; no full spawning or ability object lifecycle')
    print(json.dumps(report, indent=2))
    return int(not report['match'])

if __name__ == '__main__':
    raise SystemExit(main())
