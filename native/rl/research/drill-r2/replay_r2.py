"""Replay styles/drill_burrow2.toml's terms (approximately, in Python) over round 1's drill_look logs: what the
style would pay a round-1 policy at init, per term, against the outcome reward it already gets.

    replay_r2.py LOOKDIR [LOOKDIR ...]

Approximations: the underground bit is the undelayed SDF (the sim's flag is not in the log; they disagree on
~6% of form rows at the surface), a logged drill episode is one form, the exit velocity is the player's
position over the last decision (as the style computes it), the running minimum distance for the flight term
is taken within the episode (the style's is a 10 s window across forms), and kills are the games' death events.
"""
import json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ep_stats import episodes, runs  # noqa: E402

KILL = 0.288
U = 15.0   # u/s per u a decision


def near(d): return min(1.0, max(0.0, (16.0 - d) / 10.0))


def launch(d, step, cos):
    aimed = min(1.0, max(0.0, (cos - 0.5) / 0.366))
    fast = min(1.0, max(0.0, (step * U - 15.0) / 20.0))
    return aimed * max(near(d), fast)


def replay(d, scale=1.0):
    d = Path(d)
    games = [json.loads(l) for l in open(d / 'games.jsonl')]
    decs = [json.loads(l) for l in open(d / 'decisions.jsonl')]
    gi = {g['g']: g for g in games}
    names = ['burrow_approach', 'burrow_surface', 'burrow_flight', 'burrow_kill', 'drill_unforced']
    tot = {k: 0.0 for k in names}; fires = {k: 0 for k in names}
    surf_run, surf_q, far_launch, episodes_paid = [], [], 0, 0
    for ep in episodes(decs):
        g = gi[ep[0]['g']]; d0 = ep[0]['d']; n = len(ep)
        if ep[0]['t'] >= 6960: continue
        ug = [r['ug'] for r in ep]
        P = [(r['x'], r['y']) for r in ep]
        dist = [math.hypot(r['x'] - r['ox'], r['y'] - r['oy']) if r['oalive'] else None for r in ep]
        rr = runs(ug)
        paid = 0.0
        if rr:
            s, m = rr[0]
            for i in range(s + 1, s + m):   # first run: approach, gains capped at 2 u, losses in full
                if dist[i] is None or dist[i - 1] is None: continue
                closer = min(dist[i - 1], 30) - min(dist[i], 30)
                v = 0.006 * min(closer, 2.0) * scale; tot['burrow_approach'] += v; paid += v; fires['burrow_approach'] += 1
            i = s + m
            if i < n and dist[i] is not None and i >= 1:   # first surfacing: the run's length times the launch quality
                vx, vy = P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]
                bx, by = ep[i]['ox'] - P[i][0], ep[i]['oy'] - P[i][1]
                step = math.hypot(vx, vy)
                cos = (vx * bx + vy * by) / (max(step, 1 / U) * max(dist[i], 1.0))
                q = launch(dist[i], step, cos)
                pay = 0.075 * min(m * 4 / 60, 2.0) * q * scale
                if q > 0: fires['burrow_surface'] += 1; surf_run.append(m); surf_q.append(q); far_launch += dist[i] > 16
                tot['burrow_surface'] += pay; paid += pay
            # flight: after the first surfacing, in the air, every u of new minimum distance (within the episode), 2 u cap, scaled by the run (full at 2 s)
            runmin = min(min(x, 30) for x in dist[:i + 1] if x is not None) if any(x is not None for x in dist[:i + 1]) else 30.0
            for j in range(i + 1, n):
                if dist[j] is None: continue
                if not ug[j]:
                    prog = runmin - min(dist[j], 30)
                    if prog > 0:
                        v = 0.006 * min(prog, 2.0) * min(m * 4 / 60, 2.0) / 2 * scale; tot['burrow_flight'] += v; paid += v; fires['burrow_flight'] += 1
                runmin = min(runmin, min(dist[j], 30))
        for x in g['deaths']:
            if x['victim'] == 'B' and x['killer'] == 'A' and x['ability'] == 26 and d0 <= x['d'] <= d0 + len(ep):
                k = x['d'] - d0; c = min(sum(ug[max(0, k - 89):k + 1]), k + 1)
                if c >= 8: v = 0.2 * min(1.0, c / 30) * scale; tot['burrow_kill'] += v; paid += v; fires['burrow_kill'] += 1
            if x['victim'] == 'A' and x['a_in_drill'] and x['killer'] in ('none', 'self') and x['cause'] in ('ring_out', 'self') and d0 <= x['d'] <= d0 + len(ep):
                tot['drill_unforced'] -= 0.08; paid -= 0.08; fires['drill_unforced'] += 1
        episodes_paid += paid != 0
    kills = sum(1 for g in games for x in g['deaths'] if x['victim'] == 'B' and x['killer'] == 'A')
    dec = sum(g['decisions'] for g in games)
    neps = sum(1 for ep in episodes(decs) if ep[0]['t'] < 6960)
    print(f'== {d}  games {len(games)} decisions {dec} A kills {kills} (kill reward {KILL * kills:.0f}); drill episodes {neps}, paid or charged {episodes_paid}')
    for k in tot:
        print(f'   {k:16s} total {tot[k]:+8.2f}  fires {fires[k]:5d}  fire rate {fires[k] / dec:.2e}  per fire {tot[k] / max(1, fires[k]):+.4f}')
    s = sum(tot.values())
    print(f'   style total {s:+.2f} = {100 * s / (KILL * kills):.1f}% of the kill reward; surfacings paid: {len(surf_run)}, {far_launch} beyond 16 u, '
          f'run median {np.median(surf_run):.0f} decisions, p90 {np.percentile(surf_run, 90):.0f}, launch quality median {np.median(surf_q):.2f}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        replay(p)
