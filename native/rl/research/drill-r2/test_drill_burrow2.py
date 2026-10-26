"""Unit and anti-hack tests of styles/drill_burrow2.toml on synthetic games, through both interpreters.

Same harness as native/rl/research/drill-spec/test_drill_burrow.py (its dump() writes the replay-dump format;
player 1 holds Drill/Dash/Dash and its Drill form row carries BOPL_ENTITY_UNDERGROUND while scripted
underground). The C VM (train/bopl_style.h through libstyle_check.so) and the numpy interpreter
(eval/style_score.py) must agree bit for bit on every decision, and player 1's per-term totals must match
each scenario's hand-computed expectation. Every scenario starts with two out-of-form decisions, as a real
game does (spawn), so since_form_entry is defined.

Geometry: the opponent (player 2) sits at the origin and the driller (player 1) moves, so the style's exit
velocity (the player's position over one decision; 1 u a decision = 15 u/s) is what the script says. In
`frames` the driller moves radially, so a closing step is aimed (cos 1) and a retreating one is not; `path`
scripts explicit positions for off-axis exits. Both are rotated by 0.37 rad so nothing is axis-aligned.

    python native/rl/research/drill-r2/test_drill_burrow2.py [--lib native/artifacts/rl/libstyle_check.so] [--style S.toml] [--scale 2]
"""
import argparse, ctypes, math, sys, tempfile
from pathlib import Path
import numpy as np

T = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(T / 'train'), str(T / 'eval'), str(T / 'native/rl/research/drill-spec')]
import style_compile as sc  # noqa: E402
import style_score as ss  # noqa: E402
import style_check as chk  # noqa: E402
from test_drill_burrow import dump, DRILL, DASH, BOW, CONTACT, RING_OUT, SELF  # noqa: E402

PRE = 2   # out-of-form decisions before each scripted form
TH = 0.37   # rotation of every scenario, rad
U = 15.0   # u/s per u a decision (4 ticks)


def rot(x, y):
    return tuple(float(np.float32(v)) for v in (round(x * math.cos(TH) - y * math.sin(TH), 3), round(x * math.sin(TH) + y * math.cos(TH), 3)))


def path(pts, **extra):
    """pts: list of ((x, y), form, ug) with the opponent at the origin; the driller's positions, rotated."""
    full = [(pts[0][0], False, False)] * PRE + list(pts)
    return [dict(p1=rot(*xy), p2=(0.0, 0.0), form=form, ug=ug, **extra) for xy, form, ug in full]


def frames(seq, **extra):
    """seq: list of (dist, form, ug); the driller moves radially toward (or away from) the opponent."""
    return path([((-dd, 0.0), form, ug) for dd, form, ug in seq], **extra)


def near(d):
    return min(1.0, max(0.0, (16.0 - d) / 10.0))


def launch(d, step, cos=1.0):
    """launch quality at a surfacing: step = u moved over the surfacing decision, cos = exit angle to the opponent."""
    aimed = min(1.0, max(0.0, (cos - 0.5) / 0.366))
    fast = min(1.0, max(0.0, (step * U - 15.0) / 20.0))
    return aimed * max(near(d), fast)


def surface_pay(run_dec, d, step=1.0, cos=1.0):
    return 0.075 * min(run_dec * 4 / 60.0, 2.0) * launch(d, step, cos)


def flight_pay(steps, run_dec=30):
    """progress steps (u of new minimum distance each), 2 u cap a decision, scaled by the run behind the surfacing (full at 2 s)"""
    return sum(0.006 * min(max(s, 0.0), 2.0) for s in steps) * min(run_dec * 4 / 60.0, 2.0) / 2.0


def scenarios():
    S = []
    # 1. a straight underground approach 29 -> 6 u (1 u a decision), surfacing on a 6 -> 5 step, then an air retreat 2 u a decision.
    seq = [(30, True, False)] + [(30 - j, True, True) for j in range(1, 25)] + [(5, True, False)] + [(5 + 2 * j, True, False) for j in range(1, 5)]
    S.append(('approach_surface', frames(seq), [], dict(burrow_approach=(0.006 * 23, 1e-4), burrow_surface=(surface_pay(24, 5), 1e-4), burrow_flight=(0.0, 1e-4))))
    # 2. circling at a constant distance far from the opponent, never surfacing: nothing
    S.append(('circle_far', frames([(25, True, False)] + [(25, True, True)] * 200), [], dict(total=(0.0, 1e-4))))
    # 3. the same, then popping out far away with no exit motion: nothing (not near, not fast, not aimed)
    S.append(('circle_far_surface', frames([(25, True, False)] + [(25, True, True)] * 200 + [(25, True, False)] * 3), [], dict(total=(0.0, 1e-4))))
    # 4. circling 13 s under the opponent (8 u) then surfacing toward it (8 -> 7, 15 u/s): the run pay is capped at 2 s, so the extra 11 s buy nothing
    S.append(('circle_near_surface', frames([(8, True, False)] + [(8, True, True)] * 200 + [(7, True, False)] * 3), [],
              dict(burrow_approach=(0.0, 1e-4), burrow_surface=(surface_pay(30, 7), 1e-4), total=(surface_pay(30, 7), 1e-4))))
    # 5. hop-chain dip farm: dip 4 u closer underground, pop out one more u closer, back off 4 u in the air, repeat 20 times in one form.
    #    Only the first run's approach and the first surfacing pay; the air never sets a new minimum; the loop cannot be farmed.
    seq = []
    for c in range(20):
        seq += [(14 - j, True, True) for j in range(5)] + [(9 + j, True, False) for j in range(5)]
    S.append(('hop_chain_dip', frames(seq), [], dict(burrow_approach=(0.006 * 4, 1e-4), burrow_surface=(surface_pay(5, 9), 1e-4), burrow_flight=(0.0, 1e-4),
                                                   total=(0.006 * 4 + surface_pay(5, 9), 1e-4))))
    # 6. fast underground oscillation (+-5 u): gains count 2 u, losses 5 u: < 0
    S.append(('oscillate_fast', frames([(15 + (5 if j % 2 else 0), True, True) for j in range(200)]), [], dict(burrow_approach=('<=', -0.5), total=('<=', -0.5))))
    # 7. underground, approach slowly (1 u a decision) and retreat fast (5 u in one): loops net 0, the run pays its net 20 -> 15 only
    S.append(('slow_in_fast_out', frames([([20, 19, 18, 17, 16, 15][j % 6], True, True) for j in range(180)]), [],
              dict(burrow_approach=(0.03, 1e-4), total=(0.03, 1e-4))))
    # 8. an underground loop 20 -> 10 -> 20 inside one run telescopes to <= 0: the entry step (air -> ground) is not paid,
    #    the whole retreat is charged, so the loop nets exactly -0.006 (one unpaid 1 u step)
    S.append(('ug_loop', frames([(20, True, False)] + [(20 - j, True, True) for j in range(1, 11)] + [(10 + j, True, True) for j in range(1, 11)]), [],
              dict(burrow_approach=(-0.006, 1e-4), total=('<=', 0.0))))
    # 9. a dive that clips a platform and overshoots: the air retreat afterwards costs nothing (round 1 charged it)
    seq = [(20, True, False), (12, True, True), (8, True, True), (5, True, True), (4, True, False)] + [(4 + 3 * j, True, False) for j in range(1, 10)]
    S.append(('air_overshoot', frames(seq), [], dict(burrow_approach=(0.024, 1e-4), burrow_surface=(surface_pay(3, 4), 1e-4), burrow_flight=(0.0, 1e-4), total=('>=', 0.0))))
    # 10. a second surfacing in the same form is not paid (first surfacing only)
    seq = [(10, True, False)] + [(9, True, True)] * 10 + [(8, True, False)] * 3 + [(7, True, True)] * 30 + [(6, True, False)] * 3
    S.append(('second_surfacing', frames(seq), [], dict(burrow_surface=(surface_pay(10, 8), 1e-4), burrow_approach=(0.0, 1e-4), burrow_flight=(0.0, 1e-4))))
    # 11. a new form (after out-of-form decisions) pays its own first surfacing again
    one = [(10, True, False)] + [(9, True, True)] * 10 + [(8, True, False)] * 3
    S.append(('new_form', frames(one + [(8, False, False)] * 5 + one), [], dict(burrow_surface=(2 * surface_pay(10, 8), 1e-4), burrow_approach=(0.0, 1e-4))))
    # 12. the underground flag flickers at the surface (one air decision inside a 21-decision run): the first run pays, the second does not;
    #     the pay is below the unflickered run's, never above
    seq = [(10, True, False)] + [(9, True, True)] * 10 + [(8, True, False)] + [(7, True, True)] * 10 + [(6, True, False)] * 3
    S.append(('flicker', frames(seq), [], dict(burrow_surface=(surface_pay(10, 8), 1e-4), total=(surface_pay(10, 8), 1e-4))))
    # 13-14. sudden death and a dead opponent: nothing
    seq = [(30, True, False)] + [(30 - j, True, True) for j in range(1, 25)] + [(5, True, False)] * 3
    S.append(('sudden_death', frames(seq, sd=1), [], dict(total=(0.0, 1e-4))))
    S.append(('opp_dead', frames(seq, p2_alive=False), [], dict(total=(0.0, 1e-4))))
    # 15. a teleport-sized jump in one decision is clamped to 2 u
    S.append(('clamp', frames([(30, True, True), (10, True, True), (10, True, True)]), [], dict(burrow_approach=(0.012, 1e-4))))
    # 16. other entity flag bits next to UNDERGROUND do not change the reading
    seq = [(30, True, False)] + [(30 - j, True, True) for j in range(1, 25)] + [(5, True, False)] * 3
    S.append(('flag_noise', frames(seq, flag_noise=2 | 16 | (1 << 20)), [], dict(burrow_approach=(0.006 * 23, 1e-4), burrow_surface=(surface_pay(24, 5), 1e-4))))
    # 17. burrow kill after 31 underground decisions: the full premium 0.2
    S.append(('burrow_kill', frames([(3, True, True)] * 31), [(PRE + 30, 2, 1, DRILL, CONTACT)], dict(burrow_kill=(0.2, 1e-6))))
    # 18. kill after 13 underground decisions: 0.2 * 13 / 30
    S.append(('short_burrow_kill', frames([(3, True, j >= 18) for j in range(31)]), [(PRE + 30, 2, 1, DRILL, CONTACT)], dict(burrow_kill=(0.2 * 13 / 30, 1e-5))))
    # 19. a hop chain's underground time adds up for the kill: 3 dips of 6 decisions in the last 6 s = 18 -> 0.2 * 19 / 30
    #     (the chain's surfacings have no exit motion, so they pay nothing)
    seq = []
    for c in range(3):
        seq += [(3, True, True)] * 6 + [(3, True, False)] * 4
    S.append(('hop_chain_kill', frames(seq + [(3, True, True)]), [(PRE + len(seq), 2, 1, DRILL, CONTACT)],
              dict(burrow_kill=(0.2 * 19 / 30, 1e-5), burrow_surface=(0.0, 1e-4))))
    # 20. kill after only 4 underground decisions: nothing
    S.append(('dip_kill', frames([(3, True, j >= 27) for j in range(31)]), [(PRE + 30, 2, 1, DRILL, CONTACT)], dict(burrow_kill=(0.0, 1e-4))))
    # 21. kill with another ability after burrowing: nothing
    S.append(('dash_kill', frames([(3, True, True)] * 31), [(PRE + 30, 2, 1, DASH, CONTACT)], dict(burrow_kill=(0.0, 1e-4))))
    # 22. burrow kill in sudden death: nothing
    S.append(('sd_kill', frames([(3, True, True)] * 31, sd=1), [(PRE + 30, 2, 1, DRILL, CONTACT)], dict(burrow_kill=(0.0, 1e-4))))
    # 23. underground long ago (a 20-decision run, then 100 decisions = 6.7 s in the air, outside the 360-tick window): nothing
    S.append(('stale_kill', frames([(3, True, j < 20) for j in range(120)]), [(PRE + 119, 2, 1, DRILL, CONTACT)],
              dict(burrow_kill=(0.0, 1e-4), burrow_surface=(0.0, 1e-4))))
    # 24-26. unforced drill deaths: ring-out and self cost 0.08; being killed costs nothing extra
    S.append(('ring_out', frames([(20, True, False)] * 10), [(PRE + 9, 1, 0, 0, RING_OUT)], dict(drill_unforced=(-0.08, 1e-6))))
    S.append(('self_death', frames([(20, True, True)] * 10), [(PRE + 9, 1, 1, DRILL, SELF)], dict(drill_unforced=(-0.08, 1e-6))))
    S.append(('killed', frames([(20, True, True)] * 10), [(PRE + 9, 1, 2, BOW, CONTACT)], dict(drill_unforced=(0.0, 1e-4))))
    # 27. ring-out out of form: nothing
    S.append(('ring_out_no_form', frames([(20, False, False)] * 10), [(PRE + 9, 1, 0, 0, RING_OUT)], dict(drill_unforced=(0.0, 1e-4))))

    # -- launches (round 2 revision: a far surfacing pays for being aimed and fast, the flight for closing, the kill for the form)
    # 28. a 2 s burrow 30 u away, launched at the opponent at 45 u/s (30 -> 27), flying 27 -> 3 at 3 u a decision, then the kill:
    #     surface 0.15 (full run, aimed, fast), flight 8 x 0.012 (3 u steps capped at 2), kill 0.2 (30 underground decisions in the form)
    seq = [(30, True, False)] + [(30, True, True)] * 30 + [(27 - 3 * j, True, False) for j in range(9)]
    S.append(('launch_far_aimed', frames(seq), [(PRE + len(seq) - 1, 2, 1, DRILL, CONTACT)],
              dict(burrow_approach=(0.0, 1e-4), burrow_surface=(surface_pay(30, 27, 3), 1e-4), burrow_flight=(flight_pay([3] * 8), 1e-4),
                   burrow_kill=(0.2, 1e-5), total=(0.15 + 0.096 + 0.2, 1e-4))))
    # 29. the same burrow, surfacing at 15 u/s (30 -> 29) toward a far opponent: not fast, not near: no surface pay
    seq = [(30, True, False)] + [(30, True, True)] * 30 + [(29, True, False)] * 5
    S.append(('launch_far_slow', frames(seq), [], dict(burrow_surface=(0.0, 1e-4), total=(0.0, 1e-4))))
    # 30. the same burrow, launched at 45 u/s perpendicular to the opponent: aimed = 0: nothing; the sideways flight sets no new minimum
    pts = [((-30, 0), True, False)] + [((-30, 0), True, True)] * 30 + [((-30, 3 * j), True, False) for j in range(1, 6)]
    S.append(('launch_far_off', path(pts), [], dict(burrow_surface=(0.0, 1e-4), burrow_flight=(0.0, 1e-4), total=(0.0, 1e-4))))
    # 31. launched at 64 u/s diagonally (45 deg off the bearing at the exit point; 51 deg off the bearing from where it lands, which is what
    #     the style measures: cos 0.625, aimed 0.34 of the full pay); the diagonal flight closes 30 -> 27.2 -> 24.7 -> 22.8 -> 21.6 -> 21.2,
    #     new minima of 2.4, 1.9, 1.2, 0.4 after the surfacing step
    pts = [((-30, 0), True, False)] + [((-30, 0), True, True)] * 30 + [((-30 + 3 * j, 3 * j), True, False) for j in range(1, 6)]
    d = [math.hypot(-30 + 3 * j, 3 * j) for j in range(0, 6)]
    cos1 = (3 * 27 + 3 * -3) / (math.hypot(3, 3) * d[1])
    S.append(('launch_far_diag', path(pts), [], dict(burrow_surface=(surface_pay(30, d[1], 3 * math.sqrt(2), cos1), 2e-4),
                                                    burrow_flight=(flight_pay([d[j - 1] - d[j] for j in range(2, 6)]), 2e-4))))
    # 32. launch farm: launch at the far opponent, fly to 15 u, go back underground to 30 u, launch again, 20 times in one form:
    #     the first launch (run 10 decisions) and the first flight's 12 u of new minimum (at a third of the rate: a 0.67 s run) pay; nothing else does
    seq = [(30, True, False)]
    for c in range(20):
        seq += [(30, True, True)] * 10 + [(27 - 3 * j, True, False) for j in range(5)] + [(18 + 3 * j, True, True) for j in range(5)]
    S.append(('launch_farm', frames(seq), [], dict(burrow_approach=(0.0, 1e-4), burrow_surface=(surface_pay(10, 27, 3), 1e-4), burrow_flight=(flight_pay([3] * 4, 10), 1e-4),
                                                 total=(surface_pay(10, 27, 3) + flight_pay([3] * 4, 10), 1e-4))))
    # 32b. the same dive after a 2-decision clip instead of a 2 s burrow: the launch pays 0.1 of the rate, so does the flight (0.0096 for 24 u)
    seq = [(30, True, False)] + [(30, True, True)] * 2 + [(27 - 3 * j, True, False) for j in range(9)]
    S.append(('clip_flight', frames(seq), [], dict(burrow_surface=(surface_pay(2, 27, 3), 1e-4), burrow_flight=(flight_pay([3] * 8, 2), 1e-4),
                                                 total=(surface_pay(2, 27, 3) + flight_pay([3] * 8, 2), 1e-4))))
    # 33. aim and never arrive: launch at 45 u/s toward the far opponent, then veer away and hover; burrow again and launch again:
    #     the first launch's 0.15 and nothing more (no flight progress, no second surfacing, no kill)
    seq = [(30, True, False)] + [(30, True, True)] * 30 + [(27, True, False)] + [(28 + 2 * j, True, False) for j in range(10)] + [(46, True, False)] * 20 \
        + [(46, True, True)] * 30 + [(43, True, False)] * 5
    S.append(('launch_no_arrive', frames(seq), [], dict(burrow_surface=(surface_pay(30, 27, 3), 1e-4), burrow_flight=(0.0, 1e-4), burrow_approach=(0.0, 1e-4),
                                                      total=(surface_pay(30, 27, 3), 1e-4))))
    # 34. a form that never touches the ground, diving 30 -> 3 in the air: the flight term needs a surfacing in the form: nothing
    S.append(('air_only_dive', frames([(30 - 3 * j, True, False) for j in range(10)]), [], dict(total=(0.0, 1e-4))))
    # 35. air oscillation after a launch (27 -> 25 -> 27 -> 25 ...): the first 2 u of new minimum pay once, the loop pays and costs nothing
    seq = [(30, True, False)] + [(30, True, True)] * 30 + [(27, True, False)] + [(25 if j % 2 == 0 else 27, True, False) for j in range(40)]
    S.append(('flight_oscillation', frames(seq), [], dict(burrow_surface=(surface_pay(30, 27, 3), 1e-4), burrow_flight=(flight_pay([2]), 1e-4),
                                                        total=(surface_pay(30, 27, 3) + flight_pay([2]), 1e-4))))
    # 36. a long flight before the kill: 2 s underground, launch, 8 decisions to 3 u, 42 decisions (2.8 s) at 3 u, kill 3.4 s after surfacing:
    #     the 6 s form window credits the 30 underground decisions (a 3 s window would have paid 0)
    seq = [(30, True, False)] + [(30, True, True)] * 30 + [(27 - 3 * j, True, False) for j in range(9)] + [(3, True, False)] * 42
    S.append(('long_flight_kill', frames(seq), [(PRE + len(seq) - 1, 2, 1, DRILL, CONTACT)],
              dict(burrow_kill=(0.2, 1e-5), burrow_surface=(surface_pay(30, 27, 3), 1e-4), burrow_flight=(flight_pay([3] * 8), 1e-4))))
    # 37. the previous form's underground time does not count: 30 underground decisions, out of form, a new form with 4 underground
    #     decisions and a kill: ug_form = min(34 in the window, 4 in this form) < 8: nothing
    seq = [(3, True, True)] * 30 + [(3, False, False)] * 2 + [(3, True, True)] * 4
    S.append(('prev_form_kill', frames(seq), [(PRE + len(seq) - 1, 2, 1, DRILL, CONTACT)], dict(burrow_kill=(0.0, 1e-4))))
    # 38-39. popping out next to the opponent pays only when aimed: 10 decisions under it, exit 6 -> 5 toward it (15 u/s, near): 0.05;
    #     exit at 45 u/s sideways past it (aim < 0): nothing
    S.append(('pop_out_aimed', frames([(6, True, False)] + [(6, True, True)] * 10 + [(5, True, False)] * 3), [],
              dict(burrow_surface=(surface_pay(10, 5, 1), 1e-4))))
    pts = [((-5, 0), True, False)] + [((-5, 0), True, True)] * 10 + [((-5, 3 * j), True, False) for j in range(1, 4)]
    S.append(('pop_out_off', path(pts), [], dict(burrow_surface=(0.0, 1e-4), burrow_flight=(0.0, 1e-4))))
    return S


def main():
    p = argparse.ArgumentParser(); p.add_argument('--lib', type=Path, default=T / 'native/artifacts/rl/libstyle_check.so')
    p.add_argument('--style', type=Path, default=T / 'styles/drill_burrow2.toml')
    p.add_argument('--scale', type=float, default=1.0, help='expected multiple of the positive terms (drill_burrow2_x2.toml: 2)')
    a = p.parse_args()
    lib = ctypes.CDLL(str(a.lib))
    lib.style_check_run.restype = ctypes.c_int
    lib.style_check_run.argtypes = [ctypes.c_char_p, ctypes.c_long, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 9 \
        + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    program = sc.compile_files([a.style])
    prog = ss.Decoded(program.data); names = program.term_names; n = len(names)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, fr, events, expect in scenarios():
            path_ = Path(tmp) / f'{name}.npz'
            ev = dump(path_, fr, events)
            game = ss.Game(path_, events=ev)
            c_terms, c_totals = chk.run_c(lib, program.data, game, ev if len(ev) else None, 1, n)
            ok = True; got = {}
            for pl in range(game.P):
                pays, total = ss.Player(prog, game, pl, 1).run()
                if not (np.array_equal(c_terms[pl].view(np.uint32), pays.view(np.uint32))
                        and np.array_equal(c_totals[pl].view(np.uint32), total.view(np.uint32))):
                    ok = False; print(f'{name}: C/Python disagree for player {pl + 1}')
            sums = c_terms[0].astype(np.float64).sum(0)
            for t, term in enumerate(names + ['total']):
                v = float(sums.sum()) if term == 'total' else float(sums[t])
                got[term] = round(v, 6)
                if term not in expect and term == 'total': continue
                want, tol = expect.get(term, (0.0, 1e-4))
                if term != 'drill_unforced' and a.scale != 1.0:   # the dose variant scales every positive term
                    if want in ('<=', '>='): tol *= a.scale
                    else: want *= a.scale; tol *= a.scale
                if want == '<=': bad = v > tol
                elif want == '>=': bad = v < tol
                else: bad = abs(v - want) > tol
                if bad: ok = False
            if c_terms[1].any(): ok = False; print(f'{name}: player 2 (no drill) was paid')
            failures += not ok
            print(f'{"PASS" if ok else "FAIL"} {name:20s} {got}')
    print('DRILL_BURROW2 TESTS', 'PASS' if not failures else f'FAIL ({failures})')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
