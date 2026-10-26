"""Kill rate by the style-visible aim and speed at a surfacing (round 1 P1 logs): the data behind drill_burrow2 launch ramps.

    aim_bins.py LOOKDIR [LOOKDIR ...]
"""
import sys, math, numpy as np
sys.path.insert(0, 'native/rl/research/drill-r2')
import launch_stats as L
def w(k, n): 
    p = k / n if n else float('nan'); return f'{100*p:5.1f}% (n={n})'
for d in sys.argv[1:]:
    L.analyze(d); S, K = L.LAST
    S = [s for s in S if np.isfinite(s['dist']) and np.isfinite(s['cos_ug'])]
    print('==', d, len(S))
    print('agreement of the style-visible aim (last ug step) with the first-air-step aim: corr',
          np.corrcoef([s['cos'] for s in S if np.isfinite(s['cos'])], [s['cos_ug'] for s in S if np.isfinite(s['cos'])])[0,1].round(3),
          'median |diff|', np.nanmedian([abs(s['cos']-s['cos_ug']) for s in S]).round(3))
    print('-- kill in form by aim bin (cos_ug) x distance; speed = v_last_ug')
    bins = [(-1.01, 0, '>90'), (0, 0.5, '60-90'), (0.5, 0.707, '45-60'), (0.707, 0.866, '30-45'), (0.866, 0.966, '15-30'), (0.966, 1.01, '<15')]
    for dn, dlo, dhi in (('near<=6', 0, 6), ('6-16', 6, 16), ('far>16', 16, 1e9)):
        row = []
        for lo, hi, nm in bins:
            sel = [s for s in S if lo <= s['cos_ug'] < hi and dlo < s['dist'] <= dhi]
            row.append(f'{nm}: {w(sum(x["kill_form"] for x in sel), len(sel))}')
        print(f'  {dn:8s}', ' | '.join(row))
    print('-- kill in form by last-ug speed bin x aimed (cos_ug>=0.866), far > 16 u')
    for lo, hi in ((0,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,45),(45,1e9)):
        sel = [s for s in S if s['dist'] > 16 and lo <= s['v_last_ug'] < hi]
        a = [s for s in sel if s['cos_ug'] >= 0.866]; o = [s for s in sel if s['cos_ug'] < 0.5]
        print(f'  v_ug {lo:>3.0f}-{hi:<4.0f} aimed {w(sum(x["kill_form"] for x in a), len(a))} reach6 {np.mean([x["min_dist"]<=6 for x in a]) if a else float("nan"):.2f}   off {w(sum(x["kill_form"] for x in o), len(o))}')
    print('-- near (<=6) by last-ug speed, aimed vs off')
    for lo, hi in ((0,15),(15,30),(30,1e9)):
        sel = [s for s in S if s['dist'] <= 6 and lo <= s['v_last_ug'] < hi]
        a = [s for s in sel if s['cos_ug'] >= 0.5]; o = [s for s in sel if s['cos_ug'] < 0.5]
        print(f'  v_ug {lo:>3.0f}-{hi:<4.0f} aimed(<60) {w(sum(x["kill_form"] for x in a), len(a))}  off {w(sum(x["kill_form"] for x in o), len(o))}')
    # run length vs launch quality today
    q = lambda s: max(min(1,max(0,(6-s['dist'])/0+0)) if False else 0, 0)
    print('-- surfacings by run with cos_ug>=0.866 and v_last_ug>=25 (aimed+fast at style thresholds)')
    for lo, hi in ((1,3),(4,7),(8,14),(15,1e9)):
        sel = [s for s in S if lo <= s['run'] <= hi]
        af = [s for s in sel if s['cos_ug'] >= 0.866 and s['v_last_ug'] >= 25]
        print(f'  run {lo}-{hi}: n {len(sel)} aimed+fast {len(af)} ({100*len(af)/max(1,len(sel)):.0f}%) kill {w(sum(x["kill_form"] for x in af), len(af))}')
