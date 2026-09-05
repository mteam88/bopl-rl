"""Dynamic original clone prefab allocation, scheduling, revival and body lifecycle."""
import argparse
import copy
import json
from collections import Counter
from actor_stream import ROOT, capture, verify, promote, golden
from clone_prefab import prefab
from verify_actor_spawn import queries as spawn_queries
from verify_actor_bounds import assets as boundary_assets
from verify_death_effects import prefab as death_prefab
from verify_actors import ONE, vec


def queries(seed):
    for base in spawn_queries(seed):
        if base['spawn_case'] != 'init':
            continue
        for mode in ['before', 'init', 'update', 'physics', 'late', 'twice', 'limit', 'null', 'delete_pending']:
            q = copy.deepcopy(base)
            q.update(clone_case=mode, dynamic_actors=True, clone_prefab=copy.deepcopy(prefab()),
                     registration_state=True, player_state=True, routing_checks=False, scripts={})
            r = q['runtime']
            n = len(r['actors'])
            r['combat']['death_effects'] = dict(prefab=death_prefab(), water_prefab=boundary_assets()[1], instances=[])
            for i, actor in enumerate(r['actors']):
                actor['controller']['physics']['motion'].update(position=vec(20*i-50, 15), velocity=vec(),
                    external_velocity=vec(), grounded=False, ground=None, local=0)
                actor['slime']['state'].update(mode=8, suppress=True, has_spawned=False)
                actor['slime']['abilities']['can_use'] = True
                actor['entity']['body_simulated'] = False
                r['scheduler']['players'][i]['flags'] = dict(spark=False, macho=False, platform=False)
            action = dict(dt=71582788, scale=ONE, enabled=[False]*len(r['components']),
                          inputs=[dict(direction=vec(), jump=False) for _ in range(n)], commands=[])
            q['actions'] = [copy.deepcopy(action) for _ in range(150)]
            command = dict(op='clone_actor', actor=n-1, player=n-1, position=vec(10, 35))
            if mode in ['init', 'update', 'late', 'physics']:
                driver = r['scheduler']['pending']['items'][0] if mode == 'init' else next(i for i, c in enumerate(r['components']) if c['role'] == 'collision')
                key = 'physics:*' if mode == 'physics' else mode+':'+str(driver)
                q['scripts'][key] = [[command]]
                if mode in ['update', 'late']:
                    for a in q['actions']:
                        a['enabled'][driver] = True
            else:
                q['actions'][0]['commands'] = [command]
            if mode == 'limit':
                r['combat']['book']['players'][n-1]['bodies_alive'] = 17
            elif mode == 'null':
                command['null_prefab'] = True
            elif mode == 'delete_pending':
                q['actions'][0]['commands'].append(dict(op='destroy_actor', actor=n))
            else:
                q['actions'][10]['commands'] = [dict(op='actor_spawn', actor=n, action='respawn', position=vec(10, 35))]
                q['actions'][135]['commands'] = [dict(op='destroy_actor', actor=n)]
                if mode == 'twice':
                    q['actions'][0]['commands'].append(dict(command, position=vec(-10, 35)))
                    q['actions'][12]['commands'] = [dict(op='actor_spawn', actor=n+1, action='fast', position=vec(-10, 35))]
                    q['actions'][135]['commands'].append(dict(op='destroy_actor', actor=n+1))
            yield q


def saturation(seed):
    for q in queries(seed):
        if q['clone_case'] != 'before':
            continue
        n = len(q['runtime']['actors'])
        q.update(clone_case='saturation', routing_checks=True)
        q['actions'] = q['actions'][:3]
        commands = []
        for player in range(n):
            for clone in range(17):
                index = player*17+clone
                commands.append(dict(op='clone_actor', actor=player, player=player,
                    position=vec(-60+(index % 32)*4, 20+(index//32)*5)))
        q['actions'][0]['commands'] = commands
        q['actions'][2]['commands'] = [dict(op='destroy_actor', actor=n+i) for i in range(n*16)]
        yield q


def interactions(seed):
    for base in queries(seed):
        if base['clone_case'] != 'before':
            continue
        for mode in ['kill', 'kill_then_clone', 'effect_init_clone', 'kill_original',
                     'water_then_clone', 'shared_input', 'time_stop', 'protected', 'pending_scale', 'player_scale']:
            q = copy.deepcopy(base)
            q.update(clone_case=mode, routing_checks=True)
            n = len(q['runtime']['actors'])
            q['runtime']['combat']['collisions'] = True
            q['actions'] = [copy.deepcopy(base['actions'][0]) for _ in range(240)]
            for action in q['actions']:
                action['commands'] = []
            q['actions'][0]['commands'] = copy.deepcopy(base['actions'][0]['commands'])
            def birth(caller=n-1):
                return dict(op='clone_actor', actor=caller, player=n-1, position=vec(-10, 35))
            def kill(actor, effect=True):
                return dict(op='kill_actor', actor=actor, request=dict(killer=1, spawn_effect=effect,
                            ignore_invulnerability=True, cause=0))
            def respawn(actor, position, fast=False):
                return dict(op='actor_spawn', actor=actor, action='fast' if fast else 'respawn', position=position)
            q['actions'][10]['commands'] = [respawn(n, vec(10, 35))]
            q['actions'][40]['commands'] = [kill(n)]
            if mode == 'pending_scale':
                q['actions'][0]['commands'].append(dict(op='box_scale', actor=n, value=ONE//2))
            if mode == 'player_scale':
                q['runtime']['scheduler']['players'][n-1]['scale'] = 2*ONE if n % 2 else ONE//2
                for player in q['runtime']['combat']['book']['players']:
                    player['team'] = 0
            if mode in ['pending_scale', 'player_scale']:
                q['actions'][40]['commands'] = []
                q['actions'][130]['commands'] = [kill(n)]
            if mode in ['kill_then_clone', 'effect_init_clone', 'kill_original', 'water_then_clone']:
                if mode == 'kill_then_clone':
                    q['actions'][41]['commands'] = [birth()]
                elif mode == 'effect_init_clone':
                    component = len(q['runtime']['scheduler']['components'])+9
                    q['scripts']['init:'+str(component)] = [[birth()]]
                elif mode == 'kill_original':
                    q['runtime']['actors'][n-1]['slime']['abilities'].update(slots=[], pool=[], indicators=[])
                    q['actions'][20]['commands'] = [kill(n-1)]
                    q['actions'][21]['commands'] = [birth(n)]
                else:
                    q['runtime']['space'] = False
                    q['actions'][0]['commands'][0]['position'] = vec(10, -12)
                    # Bounds explicitly ignores spawn invulnerability: the first
                    # clone dies in frame zero, leaving its water effect alive.
                    q['actions'][1]['commands'] = [birth()]
                    q['actions'][10]['commands'] = []
                    q['actions'][40]['commands'] = []
                q['actions'][60]['commands'] = [respawn(n+1, vec(-10, 35), True)]
                q['actions'][100]['commands'] = [kill(n+1)]
            if mode == 'shared_input':
                q['actions'][0]['commands'].append(birth())
                q['actions'][12]['commands'] = [respawn(n+1, vec(-10, 35), True)]
                q['actions'][40]['commands'] = []
                q['actions'][130]['commands'] = [kill(n), kill(n+1)]
                for tick, action in enumerate(q['actions']):
                    action['inputs'][n-1].update(direction=vec(1 if tick//20 % 2 else -1), jump=tick % 19 == 0)
                    action['routing_inputs'] = [dict(can_use=tick % 7 != 0, buttons=[tick % 3 == 0, False, tick % 5 == 0]) for _ in range(n)]
            if mode in ['time_stop', 'protected']:
                q['runtime']['scheduler']['players'][n-1]['protected'] = mode == 'protected'
                for action in q['actions'][:35]:
                    action['scale'] = 0
            yield q


def orphan_inputs(seed):
    for q in interactions(seed):
        if q['clone_case'] != 'kill_original':
            continue
        n = len(q['runtime']['actors'])
        for tick, action in enumerate(q['actions']):
            action['routing_inputs'] = [dict(can_use=tick % 2 == 1,
                buttons=[tick % 3 == 0, False, tick % 5 == 0]) for _ in range(n)]
        yield q


def coverage(directory, queries):
    counts = Counter()
    last = {}
    errors = {}
    with (directory/'unity.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            case = row['case']
            state = row.get('state', row.get('result', {}).get('failed'))
            if 'result' in row:
                errors[case] = row['result'].get('error')
            if state is None:
                continue
            initial = len(queries[case]['runtime']['actors'])
            counts['spawn_calls'] += sum(c['op'] == 'clone_actor' for c in state['effects'])
            for actor in state['actors'][initial:]:
                counts['pending_snapshots'] += 'pending_object' in actor and actor['entity']['present']
                counts['null_timer_snapshots'] += actor['slime']['abilities']['timers'] is None
            last[case] = state
    for case, state in last.items():
        q = queries[case]
        initial = len(q['runtime']['actors'])
        clones = state['actors'][initial:]
        expected = (16*initial if q['clone_case'] == 'saturation' else
                    0 if q['clone_case'] in ['limit', 'null'] else
                    2 if q['clone_case'] in ['twice', 'kill_then_clone', 'effect_init_clone', 'kill_original', 'water_then_clone', 'shared_input'] else 1)
        if len(clones) != expected:
            raise ValueError(f'Case {case}: expected {expected} real clone births, observed {len(clones)}')
        expected_error = 'null_reference' if q['clone_case'] == 'null' else None
        if errors[case] != expected_error:
            raise ValueError(f'Case {case}: unexpected source failure {errors[case]}')
        counts['births'] += len(clones)
        counts['initialized'] += sum(a['entity']['box']['initialized'] for a in clones)
        counts['removed'] += sum(not a['entity']['present'] for a in clones)
        counts['spawned_clones'] += sum(a['slime']['state']['has_spawned'] for a in clones)
        balance = len(clones)+sum(p['bodies_alive'] for p in q['runtime']['combat']['book']['players'])-sum(p['bodies_alive'] for p in state['combat']['book']['players'])
        counts['accounted_body_deaths'] += max(0, balance)
        counts['reserved_without_actor'] += max(0, -balance)
        counts['final_player_deaths'] += len(state['combat']['events'])-len(q['runtime']['combat']['events'])
        effects = state['combat']['death_effects']['instances']
        counts['effects_created'] += len(effects)
        counts['effects_removed'] += sum(not e['present'] for e in effects)
        if len(q['actions']) >= 240:
            mode = q['clone_case']
            expected_deaths = expected + (mode == 'kill_original')
            expected_spawned = (2 if mode == 'shared_input' else
                                1 if mode in ['kill_then_clone', 'effect_init_clone', 'kill_original',
                                              'water_then_clone', 'pending_scale', 'player_scale'] else 0)
            actual = (balance, len(effects), sum(a['slime']['state']['has_spawned'] for a in clones),
                      sum(not a['entity']['present'] for a in clones))
            wanted = (expected_deaths, expected, expected_spawned, expected)
            if actual != wanted:
                raise ValueError(f'Case {case}: deaths/effects/spawn completions/removals {actual}, expected {wanted}')
            if any(e['present'] for e in effects):
                raise ValueError(f'Case {case}: effects remain after the cleanup horizon')
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=725105)
    parser.add_argument('--cases', type=int, default=0)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--directory', default='actor-clones')
    parser.add_argument('--saturation', action='store_true')
    parser.add_argument('--interactions', action='store_true')
    parser.add_argument('--orphan-inputs', action='store_true')
    parser.add_argument('--golden', action='store_true')
    parser.add_argument('--reuse-unity', action='store_true')
    args = parser.parse_args()
    name = ('actor-clones-orphan-inputs' if args.orphan_inputs else 'actor-clones-saturation'
            if args.saturation else 'actor-clones-interactions' if args.interactions else 'actor-clones')
    fixture = ROOT/f'native/fixtures/{name}.jsonl.xz'
    if args.golden:
        report = golden(fixture)
    else:
        qs = list(orphan_inputs(args.seed) if args.orphan_inputs else saturation(args.seed)
                  if args.saturation else interactions(args.seed) if args.interactions else queries(args.seed))
        if args.cases:
            qs = qs[:args.cases]
        if args.steps:
            for q in qs:
                q['actions'] = q['actions'][:args.steps]
        directory = ROOT/f'native/artifacts/{args.directory}'
        inputs, hashes = capture(directory, qs, args.reuse_unity)
        report = dict(verify(directory, inputs), provenance=hashes)
        if report['match']:
            try:
                report['clone_coverage'] = coverage(directory, inputs)
            except ValueError as error:
                report['coverage_error'] = str(error)
        (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        if report['match'] and 'coverage_error' not in report and not args.cases and not args.steps:
            promote(directory, fixture, inputs, set(range(len(inputs))), hashes, __doc__)
    print(json.dumps(report, indent=2))
    return int(not report['match'] or 'coverage_error' in report)


if __name__ == '__main__':
    raise SystemExit(main())
