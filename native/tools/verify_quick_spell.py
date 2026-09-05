"""Original QuickSpell casting and exit callbacks with the optional spell reference empty."""
import argparse
import copy
import json
from collections import Counter
from actor_stream import ROOT, capture, verify, promote, golden, records
from quick_spell_prefab import assets
from verify_ordinary_ability import queries as ordinary_queries
from verify_actor_circles import convert
from verify_actors import ONE, DT, vec


def queries(seed):
    for case, base in enumerate(ordinary_queries(6, seed, 100)):
        n = len(base['runtime']['scheduler']['players'])
        for mode in ['gust', 'revival', 'offsets', 'time_stop', 'protected', 'zero_dt', 'scales', 'disabled_animation']:
            q = convert(copy.deepcopy(base))
            q.update(quick_case=mode, scripts={}, player_state=True)
            r = q['runtime']
            r['scheduler']['loaded'] = False
            for i, actor in enumerate(r['actors']):
                actor['controller']['physics']['previous_scale'] = ONE
                if i<n:
                    actor['slime']['abilities']['timers'] = [ONE]
                    actor['entity']['active'] = True
                    continue
                quick = copy.deepcopy(assets()['revival' if mode == 'revival' else 'gust'])
                quick['animator'] = copy.deepcopy(r['actors'][i-n]['slime']['animator'])
                quick['animator'].update(current=None, next=None, body=None, callback=None, index=0, position=0, kind=0, swapped=False, callback_last=False)
                quick['animator']['render'] = dict(sprite=None, flip_x=False, flip_y=False)
                if mode == 'offsets':
                    quick.update(enter_offset=vec(2,-1), exit_offset=vec(-3,2), audio_cast='testQuickCast')
                actor['ordinary']['quick_spell'] = quick
                index = len(r['components'])
                r['components'].append(dict(actor=i,role='animation'))
                r['scheduler']['components'].append(dict(owner=i,kind='updatable',present=True,destroyed=False,hierarchy=index,enabled=True,ignore_activity=False))
            total = len(r['components'])
            r['scheduler'].update(live=[], added=total, pending=dict(items=list(range(total)),length=total))
            for player in r['scheduler']['players']:
                player.update(invisible=False, protected=mode == 'protected')
            for tick, action in enumerate(q['actions']):
                action.update(commands=[], enabled=[True]*total, dt=0 if mode=='zero_dt' else DT, scale=ONE)
                for v in action['routing_inputs']:
                    v.update(can_use=True,buttons=[tick == 0,False,False])
                if mode in ['time_stop','protected'] and 2<=tick<30:
                    action['scale'] = 0
                if mode == 'disabled_animation' and 2<=tick<30:
                    action['enabled'] = [c['role'] != 'animation' or c['actor'] < n for c in r['components']]
                if mode == 'scales' and tick%5==0:
                    action['commands'] = [dict(op='player_scale',player=i,value=[ONE//2,ONE,2*ONE][(tick//5)%3]) for i in range(n)]
            yield q



def faults(seed):
    modes = ['missing_cast', 'missing_exit', 'null_cast', 'empty_cast', 'null_exit',
             'empty_exit', 'null_renderer', 'null_particle', 'cast_without_player']
    for base in queries(seed):
        if base['quick_case'] != 'gust':
            continue
        for mode in modes:
            q = copy.deepcopy(base)
            q['quick_case'] = mode
            n = len(q['runtime']['scheduler']['players'])
            quick = q['runtime']['actors'][n]['ordinary']['quick_spell']
            if mode == 'missing_cast':
                quick['animations'].pop('castAir')
            elif mode == 'missing_exit':
                quick['animations'].pop('exit')
            elif mode in ['null_cast', 'empty_cast', 'null_exit', 'empty_exit']:
                quick['animations']['castAir' if mode.endswith('cast') else 'exit'] = None if mode.startswith('null') else []
            elif mode == 'null_renderer':
                quick['animator']['render'] = None
            elif mode == 'null_particle':
                quick['particle_prefab'] = False
            elif mode == 'cast_without_player':
                q['actions'][0]['commands'] = [dict(op='quick_cast', actor=n)]
            yield q


def coverage(directory, inputs):
    counts, entries, casts, exits, previous, results = Counter(), Counter(), Counter(), Counter(), {}, {}
    for row in records((directory/'unity.jsonl').open()):
        case = row['case']
        if 'state' not in row:
            results[case] = row['result']
            continue
        frame = row['state']
        n = len(inputs[case]['runtime']['scheduler']['players'])
        entries[case] += sum(e['event']['kind']=='ability_entry' for e in frame['report']['events'])
        for i,actor in enumerate(frame['actors'][n:]):
            quick=actor['ordinary']['quick_spell']
            before=previous.get((case,i), inputs[case]['runtime']['actors'][n+i])
            casts[case] += quick['particle_present'] and not before['ordinary']['quick_spell']['particle_present']
            exits[case] += before['entity']['active'] and not actor['entity']['active']
            previous[case,i]=actor
    for case,q in enumerate(inputs):
        mode=q['quick_case'];n=len(q['runtime']['scheduler']['players'])
        expected={'missing_cast':'animation_missing','missing_exit':'animation_missing',
                  'null_cast':'null_reference','null_exit':'null_reference','empty_cast':'index_out_of_range',
                  'empty_exit':'index_out_of_range','null_renderer':'null_reference','null_particle':'instantiate_null',
                  'cast_without_player':'null_reference'}.get(mode)
        if results[case].get('error') != expected:
            raise ValueError(f'Case {case}: expected {expected}, got {results[case].get("error")}')
        if expected is None and (entries[case],casts[case],exits[case]) != (n,n,n):
            raise ValueError(f'Case {case}: expected {n} entries, casts and returns, got {(entries[case],casts[case],exits[case])}')
        counts['entries']+=entries[case];counts['first_cast_particles']+=casts[case];counts['returns']+=exits[case]
        counts[mode]+=1
    return dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=629851)
    p.add_argument('--cases',type=int,default=0)
    p.add_argument('--directory',default='quick-spell')
    p.add_argument('--reuse-unity',action='store_true')
    p.add_argument('--golden',action='store_true')
    p.add_argument('--faults',action='store_true')
    args=p.parse_args();fixture=ROOT/('native/fixtures/quick-spell'+('-faults' if args.faults else '')+'.jsonl.xz')
    if args.golden:
        report=golden(fixture)
    else:
        qs=list((faults if args.faults else queries)(args.seed));qs=qs[:args.cases] if args.cases else qs
        directory=ROOT/f'native/artifacts/{args.directory}'
        inputs,hashes=capture(directory,qs,args.reuse_unity)
        report=dict(verify(directory,inputs),provenance=hashes)
        if report['match']:
            try:report['quick_coverage']=coverage(directory,inputs)
            except ValueError as e:report['coverage_error']=str(e)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match'] and not args.cases and 'coverage_error' not in report:
            promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(report,indent=2));return int(not report['match'] or 'coverage_error' in report)


if __name__=='__main__':raise SystemExit(main())
