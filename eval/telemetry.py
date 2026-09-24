"""Research-only event columns. No telemetry is supplied to the policy."""
import numpy as np

EVENT_FIELDS = (
    'dash_activations', 'bow_activations', 'platform_activations',
    'arrows_fired', 'arrow_hits', 'deaths_environment', 'deaths_opponent',
    'deaths_self', 'deaths_other',
)
FIELDS = EVENT_FIELDS + (
    'reward_outcome', 'reward_kills', 'reward_deaths',
    'episodes', 'wins', 'timeouts', 'ticks',
)

def summarize(samples):
    """Sum only the caller-selected agent decisions; rates use their game time."""
    values=np.asarray(samples).reshape(-1,len(FIELDS)).sum(axis=0,dtype=np.float64)
    totals=dict(zip(FIELDS,values.tolist()))
    seconds=totals['ticks']/60
    return dict(schema=1,decisions=len(samples),seconds=seconds,totals=totals,
                per_minute={k:totals[k]*60/seconds if seconds else 0 for k in EVENT_FIELDS},
                reward_per_1000_decisions={k:totals[k]*1000/len(samples) if len(samples) else 0
                                          for k in ('reward_outcome','reward_kills','reward_deaths')})
