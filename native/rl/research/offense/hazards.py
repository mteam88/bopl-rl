"""Decisive-death hazard per minute, pre-sudden-death vs sudden death, by how the loser died.
    python hazards.py NAME=FILE[,FILE...] ...
Exposure: minutes both players were alive (game start to the decisive death or end), split at tick 7200.
"""
import json, sys, math
from collections import Counter
sys.path.insert(0, __import__('os').path.dirname(__file__))
from offense_metrics import loser_death, how, end_tick, SUDDEN, WARMUP

for spec in sys.argv[1:]:
    name, files = spec.split('=', 1)
    rows = [json.loads(l) for f in files.split(',') for l in open(f)]
    exp = {'pre': 0., 'sd': 0.}; ev = {'pre': Counter(), 'sd': Counter()}
    for g in rows:
        t = end_tick(g)
        exp['pre'] += (min(t, SUDDEN) - WARMUP) / 3600
        exp['sd'] += max(t - SUDDEN, 0) / 3600
        if g['outcome'] in ('a', 'b'):
            ev['sd' if t >= SUDDEN else 'pre'][how(loser_death(g))] += 1
    print(f'== {name} n {len(rows)}')
    for ph in ('pre', 'sd'):
        tot = sum(ev[ph].values())
        print(f'  {ph:3s} exposure {exp[ph]:7.1f} min, decisive deaths {tot}, hazard {tot/exp[ph]:.3f}/min '
              f'(95% ±{1.96*math.sqrt(tot)/exp[ph]:.3f}); ' +
              ', '.join(f'{k} {v/exp[ph]:.3f}' for k, v in ev[ph].most_common()))
