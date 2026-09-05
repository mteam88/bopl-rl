"""Original bite/chew/swallow callbacks, delayed deaths and shared player growth."""
import argparse
import copy
import json
from collections import Counter
from actor_stream import ROOT, capture, verify, promote, golden
from verify_actor_deaths import queries as death_queries
from eating_prefab import assets
from verify_actors import ONE, vec




def queries(seed):
    animations, growth = assets()
    for base in death_queries(seed):
        if base['death_case'] != 'death':
            continue
        for mode in ['bite', 'invulnerable', 'clear_guard', 'guard_after_bite', 'destroy_victim',
                     'disable_victim', 'chew_without_target', 'repeat_bite', 'target_lost_rebite',
                     'time_stop', 'protected', 'limit_scale', 'negative_growth', 'counter_overflow']:
            q = copy.deepcopy(base)
            q.update(eating_case=mode, player_state=True, routing_checks=False, scripts={})
            r = q['runtime']
            n = len(r['actors'])
            victim = n-1
            for i, actor in enumerate(r['actors']):
                actor['controller']['physics']['motion'].update(position=vec(i*12-30, 35),
                    grounded=False, ground=None, local=0, velocity=vec(), external_velocity=vec())
                actor['entity']['body_simulated'] = False
                actor['slime']['state'].update(mode=0, has_spawned=True, suppress=True)
                actor['slime']['animations'].update(copy.deepcopy(animations))
                actor['slime']['eating'] = dict(target=None, swallowing=False, scale_increase=growth)
                actor['slime']['animator'].update(current=copy.deepcopy(actor['slime']['animations']['idle']),
                    next=None, callback=None, index=0, position=0, kind=0, callback_last=False)
                actor['push_strength'] = None
                r['scheduler']['players'][i].update(eaten_count=0, scale=ONE, protected=False)
            action = copy.deepcopy(q['actions'][0])
            action['enabled'] = [c['role'] in ['motor', 'animation', 'collision'] for c in r['components']]
            action['commands'] = []
            for v in action['routing_inputs']:
                v.update(can_use=True, buttons=[False]*3)
            q['actions'] = [copy.deepcopy(action) for _ in range(120)]
            bite = dict(op='bite_actor', actor=0, victim=victim)
            q['actions'][0]['commands'] = [bite]
            guard = lambda value: dict(op='collision_invulnerable', actor=victim, value=value, mask=1)
            if mode in ['invulnerable', 'clear_guard']:
                r['actors'][victim]['collision']['invulnerable'] = 1
                if mode == 'clear_guard':
                    q['actions'][5]['commands'] = [guard(False), copy.deepcopy(bite)]
            if mode in ['guard_after_bite', 'counter_overflow']:
                q['actions'][5]['commands'] = [guard(True)]
            if mode == 'counter_overflow':
                r['scheduler']['players'][0]['eaten_count'] = 2147483647
            if mode == 'destroy_victim':
                q['actions'][5]['commands'] = [dict(op='destroy_actor', actor=victim)]
            if mode == 'disable_victim':
                q['actions'][5]['commands'] = [dict(op='actor_active', actor=victim, value=False)]
            if mode == 'chew_without_target':
                q['actions'][0]['commands'] = [dict(op='chew_actor', actor=0)]
            if mode == 'repeat_bite':
                q['actions'][5]['commands'] = [copy.deepcopy(bite)]
            if mode == 'target_lost_rebite':
                q['actions'][5]['commands'] = [dict(op='controller_destroyed', actor=victim, value=True), copy.deepcopy(bite)]
            if mode in ['time_stop', 'protected']:
                for action in q['actions'][22:52]:
                    action['scale'] = 0
                r['scheduler']['players'][0]['protected'] = mode == 'protected'
            if mode == 'limit_scale':
                r['scheduler']['players'][0]['scale'] = 3*ONE-1
            if mode == 'negative_growth':
                r['actors'][0]['slime']['eating']['scale_increase'] = -ONE
            yield q


def contacts(seed):
    for base in queries(seed):
        if base['eating_case'] != 'bite' or len(base['runtime']['actors']) < 2:
            continue
        for mode in ['threshold', 'below_threshold', 'same_team', 'airborne', 'invulnerable',
                     'time_stop', 'protected', 'already_biting', 'multiple_contacts']:
            q = copy.deepcopy(base)
            q['eating_case'] = 'contact_'+mode
            r = q['runtime']
            victim = len(r['actors'])-1
            r['combat']['collisions'] = True
            r['scene']['layers'] = [0]*32
            r['scene']['layers'][8] = 1 << 8
            for action in q['actions']:
                action['commands'] = []
            actor = r['actors'][0]
            actor['push_strength'] = 21474836
            actor['bite_threshold'] = ONE
            actor['controller']['physics'].update(scale=2*ONE, previous_scale=2*ONE)
            actor['controller']['physics']['motion'].update(grounded=True, ground=0, local=0)
            actor['entity']['box']['scale'] = 2*ONE
            r['scheduler']['players'][0]['scale'] = 2*ONE
            r['combat']['book']['players'][0]['team'] = 0
            r['combat']['book']['players'][victim]['team'] = 1
            r['actors'][victim]['controller']['physics']['motion']['position'] = vec(-29, 35)
            for i, component in enumerate(r['components']):
                if component['role'] == 'motor':
                    q['actions'][0]['enabled'][i] = False
            if mode == 'below_threshold':
                actor['bite_threshold'] += 1
            if mode == 'same_team':
                r['combat']['book']['players'][victim]['team'] = 0
            if mode == 'airborne':
                actor['controller']['physics']['motion'].update(grounded=False, ground=None)
            if mode == 'invulnerable':
                r['actors'][victim]['collision']['invulnerable'] = 1
            if mode in ['time_stop', 'protected']:
                for action in q['actions'][:45]:
                    action['scale'] = 0
                r['scheduler']['players'][0]['protected'] = mode == 'protected'
            if mode == 'already_biting':
                actor['slime']['state']['mode'] = 9
            if mode == 'multiple_contacts' and victim > 1:
                r['actors'][1]['controller']['physics']['motion']['position'] = vec(-30, 35)
                r['combat']['book']['players'][1]['team'] = 1
            yield q


def handoffs(seed):
    import random
    from verify_actor_deaths import handoffs as death_handoffs
    from verify_slime_exit import exit_info
    animations, growth = assets()
    for base in death_handoffs(seed):
        n = len(base['runtime']['scheduler']['players'])
        if base['death_case'] != 'dash' or n < 2:
            continue
        for mode in ['dash_target', 'slime_target', 'guarded_ability', 'guarded_owner', 'ability_exits', 'ability_destroyed']:
            q = copy.deepcopy(base)
            q.update(eating_case='handoff_'+mode, player_state=True, scripts={})
            r = q['runtime']
            victim = n-1
            for i, actor in enumerate(r['actors'][:n]):
                actor['slime']['animations'].update(copy.deepcopy(animations))
                actor['slime']['eating'] = dict(target=None, swallowing=False, scale_increase=growth)
                r['scheduler']['players'][i].update(eaten_count=0, protected=False, scale=ONE)
                if i != victim:
                    actor['slime']['abilities'].update(slots=[], pool=[], indicators=[])
            eater = r['actors'][0]
            eater['entity']['body_simulated'] = False
            eater['controller']['apply_gravity'] = False
            eater['controller']['physics']['motion'].update(position=vec(-40, 35),
                grounded=False, ground=None, velocity=vec(), external_velocity=vec())
            eater['slime']['state']['suppress'] = True
            for action in q['actions']:
                for i, component in enumerate(r['components']):
                    if component['actor'] == 0 and component['role'] in ['body', 'physics']:
                        action['enabled'][i] = False
            if mode == 'guarded_ability':
                r['actors'][n+victim]['collision']['invulnerable'] = 1
            if mode == 'guarded_owner':
                r['actors'][victim]['collision']['invulnerable'] = 1
            for tick, action in enumerate(q['actions']):
                action['commands'] = []
                for i, routing in enumerate(action['routing_inputs']):
                    routing['buttons'] = [i == victim and tick == 0, False, False]
            while len(q['actions']) < 120:
                q['actions'].append(copy.deepcopy(q['actions'][-1]))
            q['actions'][2]['commands'] = [dict(op='bite_actor', actor=0,
                victim=victim if mode == 'slime_target' else n+victim)]
            if mode == 'ability_exits':
                q['actions'][5]['commands'] = [dict(op='ordinary_exit', actor=n+victim,
                    info=exit_info(r['actors'][victim], random.Random(seed), len(r['bindings'])))]
            if mode == 'ability_destroyed':
                q['actions'][5]['commands'] = [dict(op='destroy_actor', actor=n+victim)]
            yield q


def clones(seed):
    from verify_actor_clones import queries as clone_queries
    animations, growth = assets()
    for base in clone_queries(seed):
        if base['clone_case'] != 'before':
            continue
        for mode in ['pending_target', 'spawned_target', 'shared_growth', 'replacement']:
            q = copy.deepcopy(base)
            q.update(eating_case='clone_'+mode, routing_checks=True)
            r = q['runtime']
            n = len(r['actors'])
            # The eater owns these clones, including the one-player case.
            q['actions'][0]['commands'][0]['player'] = 0
            for i, actor in enumerate(r['actors']):
                actor['slime']['animations'].update(copy.deepcopy(animations))
                actor['slime']['eating'] = dict(target=None, swallowing=False, scale_increase=growth)
                r['scheduler']['players'][i].update(eaten_count=0, protected=False, scale=ONE)
            q['actions'] = q['actions'][:120]
            for tick, action in enumerate(q['actions']):
                if tick > 0:
                    action['commands'] = []
                for i, component in enumerate(r['components']):
                    action['enabled'][i] = component['actor'] == 0 and component['role'] in ['motor', 'animation']
            def spawn(actor):
                return dict(op='actor_spawn', actor=actor, action='fast', position=vec(10, 35))
            def birth():
                return dict(op='clone_actor', actor=0, player=0, position=vec(-10, 35))
            if mode == 'pending_target':
                q['actions'][0]['commands'].append(dict(op='bite_actor', actor=0, victim=n))
                q['actions'][90]['commands'] = [dict(op='destroy_actor', actor=n)]
            else:
                q['actions'][10]['commands'] = [spawn(n)]
                q['actions'][20]['commands'] = [dict(op='bite_actor', actor=0, victim=n)]
                if mode in ['shared_growth', 'replacement']:
                    q['actions'][0 if mode == 'shared_growth' else 40]['commands'].append(birth())
                    q['actions'][12 if mode == 'shared_growth' else 50]['commands'] = [spawn(n+1)]
                    q['actions'][100]['commands'] = [dict(op='destroy_actor', actor=n+1)]
            yield q


def timing(seed):
    from verify_ragdoll import ragdoll_animations
    for base in queries(seed):
        if base['eating_case'] != 'guard_after_bite':
            continue
        for mode in ['zero_dt', 'slow_time', 'interrupted_swallow']:
            q = copy.deepcopy(base)
            q['eating_case'] = mode
            if mode == 'zero_dt':
                for action in q['actions']:
                    action['dt'] = 0
            elif mode == 'slow_time':
                for action in q['actions']:
                    action['scale'] = ONE//4
            else:
                slime = q['runtime']['actors'][0]['slime']
                slime['ragdoll'] = dict(duration=0, time=0, speed=vec(), magnet_holder=None)
                slime['animations'].update(ragdoll_animations())
                q['runtime']['scheduler']['players'][0]['scale'] = 5*ONE//2
                q['actions'][25]['commands'] = [dict(op='ragdoll', actor=0,
                    request=dict(duration=ONE//10, speed=vec(), fast=False, reset_animation=True, magnet=None))]
            yield q


def failures(seed):
    for base in queries(seed):
        if base['eating_case'] != 'guard_after_bite':
            continue
        for animation in ['bite', 'chew', 'swallow', 'idle']:
            q = copy.deepcopy(base)
            q['eating_case'] = 'missing_'+animation
            del q['runtime']['actors'][0]['slime']['animations'][animation]
            yield q


def coverage(directory, queries):
    counts = Counter()
    last, previous, errors, sounds, first_bite, callbacks = {}, {}, {}, {}, {}, {}
    for line in (directory/'unity.jsonl').open():
        row = json.loads(line)
        case = row['case']
        if 'result' in row:
            errors[case] = row['result'].get('error')
        state = row.get('state', row.get('result', {}).get('failed'))
        if state is None:
            continue
        sounds.setdefault(case, Counter())
        callbacks[case] = callbacks.get(case, 0) + len((state['report']['physics'] or {}).get('callbacks', []))
        for e in state['report']['events']:
            if e['event']['kind'] == 'sound':
                sound = e['event']['value']
                sounds[case][sound] += 1
                counts[sound] += 1
                if sound == 'bite':
                    first_bite.setdefault(case, row.get('action', 0))
        if case in previous:
            old = previous[case]['players'][0]['scale']
            new = state['players'][0]['scale']
            counts['growth_frames'] += new > old
            counts['shrinking_frames'] += new < old
            if new != old and queries[case]['actions'][row.get('action', 0)]['scale'] == 0:
                counts['scale_changes_during_time_stop'] += 1
                if not state['players'][0]['protected']:
                    raise ValueError(f'Case {case}: unprotected growth during time stop')
        previous[case] = state
        last[case] = state
    for case, state in last.items():
        q = queries[case]
        initial = q['runtime']
        mode = q['eating_case']
        expected_error = 'animation_missing' if mode.startswith('missing_') else None
        if errors.get(case) != expected_error:
            raise ValueError(f'Case {case}: expected {expected_error}, got {errors.get(case)}')
        births = len(state['actors'])-len(initial['actors'])
        counts['clone_births'] += births
        deaths = births + sum(a['bodies_alive']-b['bodies_alive'] for a,b in zip(initial['combat']['book']['players'],state['combat']['book']['players']))
        increments = (state['players'][0]['eaten_count']-initial['scheduler']['players'][0]['eaten_count']) % (1 << 32)
        counts['eaten_count_increments'] += increments
        counts['counted_without_body_death'] += max(0, increments-deaths)
        counts['eaten_final_deaths'] += sum(e['cause'] == 3 for e in state['combat']['events'])
        counts['removed_actors'] += sum(not a['entity']['present'] for a in state['actors'])
        counts['body_deaths'] += deaths
        counts['still_swallowing'] += state['actors'][0]['slime']['eating']['swallowing']
        if expected_error:
            continue
        if state['actors'][0]['slime']['eating']['swallowing'] != (mode == 'interrupted_swallow'):
            raise ValueError(f'Case {case}: swallowing did not complete')
        if mode.startswith('contact_'):
            if callbacks[case] == 0:
                raise ValueError(f'Case {case}: intended contact did not happen')
            intended = mode in ['contact_threshold', 'contact_time_stop', 'contact_protected', 'contact_multiple_contacts']
            if bool(sounds[case]['bite']) != intended or bool(deaths) != intended:
                raise ValueError(f'Case {case}: contact bite/death gate was not exercised as intended')
            if mode == 'contact_time_stop' and first_bite[case] < 45:
                raise ValueError(f'Case {case}: bite started during time stop')
            if mode == 'contact_protected' and first_bite[case] != 0:
                raise ValueError(f'Case {case}: protected bite did not start during time stop')
        elif mode.startswith('handoff_'):
            expected_death = mode in ['handoff_dash_target', 'handoff_slime_target', 'handoff_ability_exits']
            expected_increment = expected_death or mode == 'handoff_guarded_owner'
            expected_bite = mode != 'handoff_guarded_ability'
            if (deaths, increments, sounds[case]['bite']) != (int(expected_death), int(expected_increment), int(expected_bite)):
                raise ValueError(f'Case {case}: intended ability handoff was not exercised')
        elif mode.startswith('clone_'):
            expected_births = 2 if mode in ['clone_shared_growth', 'clone_replacement'] else 1
            expected_eaten = int(mode != 'clone_pending_target')
            removed = sum(not a['entity']['present'] for a in state['actors'][len(initial['actors']):])
            if (births, removed, deaths, increments, sounds[case]['bite']) != (expected_births, expected_births, expected_eaten, expected_eaten, expected_eaten):
                raise ValueError(f'Case {case}: intended clone birth/eating/removal did not happen')
        elif not mode.startswith(('handoff_', 'clone_')):
            expected_death = mode in ['bite', 'clear_guard', 'repeat_bite', 'time_stop', 'protected', 'limit_scale', 'negative_growth']
            expected_increment = expected_death or mode in ['guard_after_bite', 'counter_overflow', 'zero_dt', 'slow_time', 'interrupted_swallow']
            expected_bites = 0 if mode in ['invulnerable', 'chew_without_target'] else 2 if mode == 'repeat_bite' else 1
            if (deaths, increments, sounds[case]['bite']) != (int(expected_death), int(expected_increment), expected_bites):
                raise ValueError(f'Case {case}: unexpected body deaths/eaten count/bites {(deaths, increments, sounds[case]["bite"])}')
            if mode == 'interrupted_swallow' and state['players'][0]['scale'] != 3*ONE:
                raise ValueError(f'Case {case}: interrupted swallowing did not reach the scale clamp')
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=725411)
    parser.add_argument('--cases', type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--faults', dest='suite', action='store_const', const='failures')
    group.add_argument('--contacts', dest='suite', action='store_const', const='contacts')
    group.add_argument('--handoffs', dest='suite', action='store_const', const='handoffs')
    group.add_argument('--clones', dest='suite', action='store_const', const='clones')
    group.add_argument('--timing', dest='suite', action='store_const', const='timing')
    parser.add_argument('--golden', action='store_true')
    parser.add_argument('--reuse-unity', action='store_true')
    parser.add_argument('--directory')
    args = parser.parse_args()
    name = 'actor-eating'+('-'+args.suite if args.suite else '')
    generator = {None: queries, 'failures': failures, 'contacts': contacts,
                 'handoffs': handoffs, 'clones': clones, 'timing': timing}[args.suite]
    fixture = ROOT/f'native/fixtures/{name}.jsonl.xz'
    if args.golden:
        report = golden(fixture)
    else:
        qs = list(generator(args.seed))
        qs = qs[:args.cases] if args.cases else qs
        directory = ROOT/f'native/artifacts/{args.directory or name}'
        inputs, hashes = capture(directory, qs, args.reuse_unity)
        report = dict(verify(directory, inputs), provenance=hashes)
        if report['match']:
            try:
                report['eating_coverage'] = coverage(directory, inputs)
            except ValueError as error:
                report['coverage_error'] = str(error)
        (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        if report['match'] and not args.cases and 'coverage_error' not in report:
            promote(directory, fixture, inputs, set(range(len(inputs))), hashes, __doc__)
    print(json.dumps(report, indent=2))
    return int(not report['match'] or 'coverage_error' in report)


if __name__ == '__main__':
    raise SystemExit(main())
