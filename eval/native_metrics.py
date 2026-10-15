"""Translate native logs and cumulative learner counters for Learning Lab.

Counters are sampled at the rollout barrier, before reset or log aggregation can
discard unfinished episodes. ``game_steps`` counts physical player decisions;
``learner_steps`` excludes frozen players. Padded native rows count in neither.
Telemetry follows telemetry.FIELDS, buttons are jump/dash/arrow/platform, and
matches/map_matches count completed games once (not once per player).
"""
import math

from telemetry import EVENT_FIELDS, FIELDS


def empty_counters():
    return dict(learner_steps=0, game_steps=0, matches=0, map_matches={},
                buttons=[0]*4, telemetry=[0.0]*len(FIELDS), kills=0,
                opponent_game_decisions=0, ability_activations=[0]*39,
                entity_rows=[0]*39, entity_overflow=[0]*39)


def update(log, counters, previous, *, elapsed, previous_elapsed,
           entropy_coefficient, learning_rate, pool_opponents=(), rollout_seconds=None,
           train_seconds=None):
    """Return one update record; never infer learner work from padded SPS.

    Call once after each update with a fresh snapshot. Keep ``previous`` from
    the preceding call (or empty_counters at startup). Unknown native loss
    diagnostics remain JSON null, rather than appearing as measured zeroes.
    """
    def delta(key):
        value = counters[key] - previous[key]
        if value < 0:
            raise ValueError(f'Cumulative counter moved backwards: {key}')
        return value

    decisions, game_decisions = delta('learner_steps'), delta('game_steps')
    dt = elapsed - previous_elapsed
    if decisions <= 0 or game_decisions < decisions or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Expected positive elapsed time and learner decisions within physical decisions')
    if len(counters['telemetry']) != len(FIELDS) or len(previous['telemetry']) != len(FIELDS):
        raise ValueError('Unexpected native telemetry width')
    totals = {key: float(now - before) for key, now, before in
              zip(FIELDS, counters['telemetry'], previous['telemetry'])}
    totals['kills'] = delta('kills')
    if not all(math.isfinite(value) for value in totals.values()):
        raise FloatingPointError('Nonfinite cumulative learner telemetry')
    seconds = totals['ticks']/60
    if seconds < 0:
        raise ValueError('Cumulative learner ticks moved backwards')
    behavior = dict(schema=1, decisions=decisions, seconds=seconds, totals=totals,
                    per_minute={key: totals[key]*60/seconds if seconds else 0 for key in EVENT_FIELDS},
                    reward_per_1000_decisions={key: totals[key]*1000/decisions
                                              for key in ('reward_outcome','reward_kills','reward_deaths')})
    if len(counters['buttons']) != 4 or len(previous['buttons']) != 4:
        raise ValueError('Expected four cumulative button counts')
    buttons = [now-before for now,before in zip(counters['buttons'],previous['buttons'])]
    if any(n < 0 or n > decisions for n in buttons):
        raise ValueError('Held-button counts must be within learner decisions')
    ability_activations=[now-before for now,before in zip(counters['ability_activations'],previous['ability_activations'])]
    entity_rows=[now-before for now,before in zip(counters['entity_rows'],previous['entity_rows'])]
    entity_overflow=[now-before for now,before in zip(counters['entity_overflow'],previous['entity_overflow'])]
    if any(value<0 for values in (ability_activations,entity_rows,entity_overflow) for value in values):
        raise ValueError('Ability/entity counters moved backwards')
    if any(dropped>rows for dropped,rows in zip(entity_overflow,entity_rows)):
        raise ValueError('Entity overflow exceeds candidate rows')
    loss, perf, util = (log.get(key, {}) for key in ('loss','perf','util'))
    result = dict(schema=1, backend='pufferlib-native', step_unit='physical_player_decisions',
                  step=counters['game_steps'], learner_steps=counters['learner_steps'],
                  native_agent_steps=log.get('agent_steps'), update=log.get('epoch'),
                  elapsed=elapsed, sps=game_decisions/dt, learner_sps=decisions/dt,
                  learner_fraction=decisions/game_decisions,
                  matches=counters['matches'], map_matches=dict(counters['map_matches']),
                  entropy_coefficient=entropy_coefficient, learning_rate=learning_rate,
                  action_rates=dict(zip(('jump','dash','arrow','platform'),
                                        (n/decisions for n in buttons))), behavior=behavior,
                  ability_activations={str(ability):count for ability,count in enumerate(ability_activations) if count},
                  entity_overflow={str(ability):dict(rows=entity_rows[ability],dropped=entity_overflow[ability],
                                                     rate=(entity_overflow[ability]/entity_rows[ability]
                                                           if entity_rows[ability] else None))
                                   for ability in range(1,39)},
                  mean_reward=sum(totals[k] for k in ('reward_outcome','reward_kills','reward_deaths'))/decisions,
                  pool_opponents=list(pool_opponents), opponent_game_decisions=counters['opponent_game_decisions'],
                  script_assignments={}, env=dict(log.get('env', {})),
                  cuda_memory_mb=util.get('vram_used_gb',0)*1024 if 'vram_used_gb' in util else None,
                  rollout_seconds=rollout_seconds, train_seconds=train_seconds if train_seconds is not None else perf.get('train'))
    for dest, src in dict(policy_loss='policy', value_loss='value', entropy='entropy',
                          kl='kl', clip_fraction='clipfrac', gradient_norm='gradient_norm',
                          explained_variance='explained_variance', entropy_aim='entropy_aim',
                          entropy_buttons='entropy_buttons', entropy_hold='entropy_hold',
                          advantage_std='advantage_std').items():
        value = loss.get(src)
        if value is not None and not math.isfinite(value):
            # Explained variance is undefined when return variance is zero.
            # Nonfinite optimization diagnostics instead indicate a failed update.
            if src != 'explained_variance':
                raise FloatingPointError(f'Nonfinite native training diagnostic: {src}')
            value = None
        result[dest] = float(value) if value is not None else None
    # KL-to-reference runs only (train.py --kl-reference): the sampled
    # KL(pi || reference) per learner decision and its reward coefficient.
    for key in ('kl_reference', 'kl_coef'):
        if loss.get(key) is not None:
            result[key] = float(loss[key])
    return result
