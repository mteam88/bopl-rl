"""Semantic completion guards for revival captures; parity alone is insufficient."""
from collections import Counter
from actor_stream import records

def coverage(directory,queries,family):
    counts=Counter();last={};errors={};first_arrival={}
    with (directory/'unity.jsonl').open() as stream:
        for row in records(stream):
            case=row['case'];state=row.get('state',row.get('result',{}).get('failed'))
            if 'result' in row:errors[case]=row['result'].get('error')
            if state is None:continue
            last[case]=state
            for event in state['report']['events']:
                counts['arrival_sounds']+=event['event']==dict(kind='sound',value='reviveDing')
                counts['death_revival_sounds']+=event['event']==dict(kind='sound',value='reviveOnDeath')
            updates=Counter(p['id'] for p in state['report']['phases'] if p['kind']=='update')
            for index,e in enumerate(state['spells']['instances']):
                if e['state']['kind']!='revive_particle':continue
                counts['extra_particle_updates']+=max(0,updates[e['components'][0]]-1)
                if e['state']['state']['arrived']:first_arrival.setdefault((case,index),row.get('action',row.get('result',{}).get('action')))
    def require(condition,case,message):
        if not condition:raise ValueError(f'Case {case}: {message}')
    for case,q in enumerate(queries):
        s=last[case];n=len(q['runtime']['scheduler']['players']);instances=s['spells']['instances']
        particles=[e for e in instances if e['state']['kind']=='revive_particle'];effects=[e for e in instances if e['state']['kind']=='revive_effect']
        clones=s['actors'][len(q['runtime']['actors']):]
        mode=q[dict(particle='particle_case',effect='effect_case',death='death_revival_case')[family]]
        expected_error=('null_reference' if family=='particle' and mode=='null_target' or family=='effect' and mode in ['null_player_stopped','null_marker'] else 'instantiate_null' if family in ['effect','death'] and mode=='null_particle' else None)
        require(errors[case]==expected_error,case,f'expected error {expected_error}, got {errors[case]}')
        counts['particles']+=len(particles);counts['arrived']+=sum(p['state']['state']['arrived'] for p in particles)
        counts['particles_removed']+=sum(not p['present'] for p in particles);counts['effects']+=len(effects)
        counts['effects_removed']+=sum(not e['present'] for e in effects);counts['clones']+=len(clones)
        counts['full_clone_births']+=sum(len(a['slime']['abilities']['indicators'])==3 for a in clones)
        counts['clones_finished_spawn']+=sum(a['slime']['state']['has_spawned'] for a in clones)
        counts['clones_removed']+=sum(not a['entity']['present'] for a in clones)
        if family=='particle':
            expected=1 if mode=='null_target' else n
            require(len(particles)>=expected,case,'missing initial particle allocation')
            initial=particles[:expected]
            should_arrive=mode not in ['null_target','zero_speed','zero_dt','destroy_particle']
            require(all(p['state']['state']['arrived']==should_arrive for p in initial),case,'unexpected initial particle arrival state')
            if should_arrive or mode=='destroy_particle':require(all(not p['present'] for p in initial),case,'initial particle cleanup did not complete')
            if mode in ['zero_distance','exact_distance']:
                require(all(first_arrival[(case,instances.index(p))]==12 for p in initial),case,'zero/exact distance did not arrive on the birth tick')
            if mode=='null_target':require(not initial[0]['state']['state']['arrived'],case,'null target completed arrival')
        elif family=='effect':
            expected=1 if mode=='multiple_targets' else n
            require(len(effects)>=expected,case,'missing original revival effect allocation')
            if expected_error is None:require(all(not e['present'] for e in effects[:expected]),case,'initial effect cleanup did not complete')
            wants_particles=expected_error is None and mode not in ['empty','null_targets','destroy_effect']
            require(len(particles)>=n if wants_particles else len(particles)==0,case,'unexpected particle allocation from effect')
            if wants_particles:require(all(p['state']['state']['arrived'] and not p['present'] for p in particles[:n]),case,'initial particles did not finish')
        else:
            deaths=n if mode in ['simultaneous','water'] else 1
            if mode=='water' and q['runtime']['space']:
                deaths=0
                require(all(a['entity']['present'] for a in clones[:n]),case,'space water-height control lost a body')
                counts['space_boundary_controls']+=1
            require(len(effects)==deaths,case,f'expected {deaths} death-triggered effects, got {len(effects)}')
            expected_particles=0 if expected_error else deaths*(2 if mode=='multiple_markers' else 1)
            require(len(particles)==expected_particles,case,'incorrect death-triggered particle count')
            expected_clones=n+(0 if mode=='limit' else deaths*(2 if mode=='multiple_markers' else 1))
            require(len(clones)==expected_clones,case,'incorrect replacement body count')
            for i in range(deaths):
                require(not s['actors'][2*n+i]['entity']['present'],case,'dying clone was not removed')
                require(s['combat']['revival'][i]['respawn_count']==0 and s['combat']['revival'][i]['positions']==[],case,'death did not clear marker list')
            if not expected_error:
                require(all(p['state']['state']['arrived'] and not p['present'] for p in particles),case,'death particles did not finish')
                require(all(not e['present'] for e in effects),case,'death effects did not finish')
            if mode=='limit':require(all(t['slime'] is None for e in effects for t in e['state']['state']['targets']),case,'clone cap allocated a body')
    return dict(counts)
