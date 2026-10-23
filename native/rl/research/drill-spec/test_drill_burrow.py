"""Unit and anti-hack tests of styles/drill_burrow.toml on synthetic games, through both interpreters.

Each scenario is a two-player game written as a replay dump (eval/style_score.py's format, one row per
4-tick decision). Player 1 holds Drill/Dash/Dash; its Drill form row (entity kind 1, layer 8,
BOPL_ENTITY_FORM_BODY) carries BOPL_ENTITY_UNDERGROUND while scripted underground. The C VM
(train/bopl_style.h through libstyle_check.so) and the numpy interpreter (eval/style_score.py) must agree
bit for bit on every decision, and player 1's per-term totals must match the scenario's expectation.

    python native/rl/research/drill-spec/test_drill_burrow.py [--lib native/artifacts/rl/libstyle_check.so] [--style S.toml]
"""
import argparse, ctypes, json, math, sys, tempfile
from pathlib import Path
import numpy as np

T = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(T / 'train'), str(T / 'eval')]
import style_compile as sc  # noqa: E402
import style_score as ss  # noqa: E402
import style_check as chk  # noqa: E402

ACTIVE, FORM, UG = 1, 1 << 25, 1 << 24
DRILL, DASH, BOW = 26, 2, 4
DEATH, CONTACT, RING_OUT, SELF = 1, 1, 7, 8


def dump(path, frames, events=()):
    """frames: per decision dict(p1=(x, y), p2=(x, y), form=bool, ug=bool, sd=0|1, p2_alive=True, flag_noise=0)."""
    D = len(frames)
    H = np.dtype([('tick', '<u4'), ('episode_ticks', '<u4'), ('flags', '<u4'), ('entity_count', '<u4'), ('slime_count', '<u4'),
                  ('sudden_death', '<u4')])
    PL = np.dtype([('tick', '<u4'), ('player', 'u1')] + [(n, '<u4') for n in ('id', 'team', 'flags', 'alive_bodies', 'stocks', 'kills',
                                                                            'deaths', 'cause_of_death')]
                  + [(n, '<f4') for n in ('x', 'y', 'scale', 'invisible_time', 'ability_time')])
    SL = np.dtype([('tick', '<u4')] + [(n, '<u4') for n in ('id', 'owner', 'flags', 'state')] + [(n, '<f4') for n in ('x', 'y', 'vx', 'vy', 'rotation', 'scale')]
                  + [('abilities', '<u4', (3,)), ('ability_flags', '<u4', (3,)), ('cooldowns', '<f4', (3,)), ('timers', '<f4', (3,))])
    EN = np.dtype([('tick', '<u4'), ('id', '<u4'), ('kind', 'u1')] + [(n, '<u4') for n in ('layer', 'owner', 'flags', 'hierarchy', 'source_ability', 'timer')]
                  + [(n, '<f4') for n in ('x', 'y', 'vx', 'vy', 'right_x', 'right_y', 'up_x', 'up_y', 'radius', 'rotation', 'angular_velocity',
                                          'inverse_mass', 'clock_rate')])
    TE = np.dtype([('tick', '<u4'), ('player', 'u1')] + [(n, '<u4') for n in sc.TELEMETRY_FIELDS])
    header, players, slimes, ents, tel = [], [], [], [], []
    for k, f in enumerate(frames):
        n_ent = 0
        for p, key in ((0, 'p1'), (1, 'p2')):
            x, y = f[key]
            alive = 1 if (p == 0 or f.get('p2_alive', True)) else 0
            players.append((k, p, p + 1, p, alive, alive, 0, 0, 0, 0, x, y, 1.0, 0.0, 0.0))
            tel.append((k, p) + (0,) * len(sc.TELEMETRY_FIELDS))
            kit = (DRILL, DASH, DASH) if p == 0 else (BOW, DASH, DASH)
            aflags = (1 | (8 if (p == 0 and f.get('form')) else 0), 1, 1)
            slimes.append((k, 100 + p, p + 1, ACTIVE | (4 if (p == 0 and f.get('form')) else 0), 0, x, y, 0.0, 0.0, 0.0, 1.0,
                           kit, aflags, (0.0,) * 3, (0.0,) * 3))
        if f.get('form'):
            x, y = f['p1']
            flags = ACTIVE | FORM | (UG if f.get('ug') else 0) | f.get('flag_noise', 0)
            ents.append((k, 500, 1, 8, 1, flags, 0, DRILL, 0, x, y, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 1.0)); n_ent += 1
        header.append((4 * k + 1, 4 * k, 0, n_ent, 2, f.get('sd', 0)))
    meta = json.dumps(dict(events=True, players=[dict(id=1, abilities=[DRILL, DASH, DASH]), dict(id=2, abilities=[BOW, DASH, DASH])]))
    np.savez(path, header=np.array(header, H), players=np.array(players, PL), slimes=np.array(slimes, SL),
             entities=np.array(ents, EN), telemetry=np.array(tel, TE), meta=np.array(meta))
    ev = np.zeros(len(events), ss.EVENT_DTYPE)
    for i, (dec, subject, actor, source, cause) in enumerate(events):
        ev[i]['tick'] = 4 * dec + 1; ev[i]['kind'] = DEATH; ev[i]['subject'] = subject; ev[i]['actor'] = actor
        ev[i]['source_ability'] = source; ev[i]['cause'] = cause; ev[i]['flags'] = 2
    return ev


def path_frames(n, dist_of, ug_of, form_of=lambda k: True, **extra):
    out = []
    for k in range(n):
        dd = dist_of(k); a = 0.37 * k   # the opponent's bearing turns: the geometry is not axis-aligned
        out.append(dict(p1=(0.0, 0.0), p2=(round(dd * math.cos(a), 3), round(dd * math.sin(a), 3)), form=form_of(k), ug=ug_of(k), **extra))
    for f in out:   # float32-exact positions (Q32.32 round trip)
        f['p2'] = tuple(float(np.float32(v)) for v in f['p2'])
    return out


def scenarios():
    S = []
    # 1. a straight underground approach 30 -> 6 u over 24 decisions (1 u a decision): 0.006 * 24
    S.append(('approach', path_frames(30, lambda k: max(30.0 - k, 6.0), lambda k: True), [], dict(burrow_approach=(0.144, 1e-4))))
    # 2. circling at a constant distance: pays nothing
    S.append(('circle', path_frames(200, lambda k: 12.0, lambda k: True), [], dict(burrow_approach=(0.0, 1e-4))))
    # 3. oscillating approach/retreat (+-1.5 u): telescopes to the net change, here 0
    S.append(('oscillate', path_frames(201, lambda k: 15.0 + (1.5 if k % 2 else 0.0), lambda k: True), [], dict(burrow_approach=(0.0, 1e-4))))
    # 4. the hop-chain farm: approach 4 u underground, back off 4 u above ground in the same form, repeat.
    #    burrow_approach pays each dip, drill_retreat charges each retreat: the loop nets 0.
    def dip(k): return k % 10 < 5
    def dip_dist(k): return 20.0 - (k % 10) if k % 10 < 5 else 16.0 + (k % 10 - 5)
    S.append(('dip_cycle', path_frames(200, dip_dist, dip), [], dict(burrow_approach=(0.006 * 4 * 20, 1e-3),
              drill_retreat=(-0.006 * 4 * 20, 1e-3), total=(0.0, 1e-3))))
    # 4b. the same loop with a fast retreat (5 u a decision) above ground: <= 0
    S.append(('dip_fast_retreat', path_frames(180, lambda k: [20.0, 19.0, 18.0, 17.0, 16.0, 20.0][k % 6], lambda k: k % 6 < 5), [],
              dict(burrow_approach=(0.006 * 4 * 30, 1e-3), drill_retreat=(-0.006 * 4 * 30, 1e-3), total=('<=', 1e-6))))
    # 4c. fast underground oscillation (+-5 u): gains count 2 u, losses 5 u: < 0
    S.append(('oscillate_fast', path_frames(200, lambda k: 15.0 + (5.0 if k % 2 else 0.0), lambda k: True), [],
              dict(burrow_approach=('<=', -0.5), total=('<=', -0.5))))
    # 4e. underground, approach slowly (1 u a decision) and retreat fast (5 u in one): a symmetric clamp
    #     would charge the retreat 2 u and farm 3 u a loop; the loss counts in full: the loops net 0 and the
    #     run pays its net approach only (20 -> 15 u: 0.03)
    S.append(('slow_in_fast_out', path_frames(180, lambda k: [20.0, 19.0, 18.0, 17.0, 16.0, 15.0][k % 6], lambda k: True), [],
              dict(burrow_approach=(0.03, 1e-4), total=(0.03, 1e-4))))
    # 4d. retreat in a form that never went underground is not charged
    S.append(('air_retreat', path_frames(30, lambda k: 6.0 + k, lambda k: False), [], dict(drill_retreat=(0.0, 1e-4))))
    # 5. sudden death: nothing
    S.append(('sudden_death', path_frames(30, lambda k: max(30.0 - k, 6.0), lambda k: True, sd=1), [], dict(burrow_approach=(0.0, 1e-4))))
    # 6. above ground (form, no underground bit): nothing
    S.append(('above_ground', path_frames(30, lambda k: max(30.0 - k, 6.0), lambda k: False), [], dict(burrow_approach=(0.0, 1e-4))))
    # 7. opponent dead: nothing
    S.append(('opp_dead', path_frames(30, lambda k: max(30.0 - k, 6.0), lambda k: True, p2_alive=False), [], dict(burrow_approach=(0.0, 1e-4))))
    # 8. a teleport-sized jump in one decision is clamped to 2 u
    S.append(('clamp', path_frames(3, lambda k: [30.0, 10.0, 10.0][k], lambda k: True), [], dict(burrow_approach=(0.012, 1e-4))))
    # 9. other entity flag bits next to UNDERGROUND (bits 0-23 noise) do not change the reading
    S.append(('flag_noise', path_frames(30, lambda k: max(30.0 - k, 6.0), lambda k: True, flag_noise=2 | 16 | (1 << 20)), [],
              dict(burrow_approach=(0.144, 1e-4))))
    # 10. burrow kill after 30 underground decisions: the full premium 0.12
    S.append(('burrow_kill', path_frames(31, lambda k: 3.0, lambda k: True), [(30, 2, 1, DRILL, CONTACT)], dict(burrow_kill=(0.12, 1e-6))))
    # 11. kill after 13 underground decisions in the window (decisions 18-30): 0.12 * 13 / 22 (count_over(90) holds 23 decisions)
    S.append(('short_burrow_kill', path_frames(31, lambda k: 3.0, lambda k: k >= 18), [(30, 2, 1, DRILL, CONTACT)],
              dict(burrow_kill=(0.12 * 13 / 22, 1e-5))))
    # 12. kill after only 4 underground decisions: nothing
    S.append(('dip_kill', path_frames(31, lambda k: 3.0, lambda k: k >= 26), [(30, 2, 1, DRILL, CONTACT)], dict(burrow_kill=(0.0, 1e-4))))
    # 13. kill with another ability after burrowing: nothing
    S.append(('dash_kill', path_frames(31, lambda k: 3.0, lambda k: True), [(30, 2, 1, DASH, CONTACT)], dict(burrow_kill=(0.0, 1e-4))))
    # 14. burrow kill in sudden death: nothing
    S.append(('sd_kill', path_frames(31, lambda k: 3.0, lambda k: True, sd=1), [(30, 2, 1, DRILL, CONTACT)], dict(burrow_kill=(0.0, 1e-4))))
    # 15. underground long ago (outside the 90-tick window): nothing
    S.append(('stale_kill', path_frames(60, lambda k: 3.0, lambda k: k < 20), [(59, 2, 1, DRILL, CONTACT)], dict(burrow_kill=(0.0, 1e-4))))
    # 16-18. unforced drill deaths: ring-out and self cost 0.08; being killed costs nothing extra
    S.append(('ring_out', path_frames(10, lambda k: 20.0, lambda k: False), [(9, 1, 0, 0, RING_OUT)], dict(drill_unforced=(-0.08, 1e-6))))
    S.append(('self_death', path_frames(10, lambda k: 20.0, lambda k: True), [(9, 1, 1, DRILL, SELF)], dict(drill_unforced=(-0.08, 1e-6))))
    S.append(('killed', path_frames(10, lambda k: 20.0, lambda k: True), [(9, 1, 2, BOW, CONTACT)], dict(drill_unforced=(0.0, 1e-4))))
    # 19. ring-out out of form: nothing
    S.append(('ring_out_no_form', path_frames(10, lambda k: 20.0, lambda k: False, form_of=lambda k: False), [(9, 1, 0, 0, RING_OUT)],
             dict(drill_unforced=(0.0, 1e-4))))
    return S


def main():
    p = argparse.ArgumentParser(); p.add_argument('--lib', type=Path, default=T / 'native/artifacts/rl/libstyle_check.so')
    p.add_argument('--style', type=Path, default=T / 'styles/drill_burrow.toml')
    a = p.parse_args()
    lib = ctypes.CDLL(str(a.lib))
    lib.style_check_run.restype = ctypes.c_int
    lib.style_check_run.argtypes = [ctypes.c_char_p, ctypes.c_long, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 9 \
        + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    program = sc.compile_files([a.style])
    prog = ss.Decoded(program.data); names = program.term_names; n = len(names)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, frames, events, expect in scenarios():
            path = Path(tmp) / f'{name}.npz'
            ev = dump(path, frames, events)
            game = ss.Game(path, events=ev)
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
                want, tol = expect.get(term, (0.0, 1e-4))   # float32 distances: ~1e-6 a decision of rounding
                if (v > tol) if want == '<=' else (abs(v - want) > tol): ok = False
            if c_terms[1].any(): ok = False; print(f'{name}: player 2 (no drill) was paid')
            failures += not ok
            print(f'{"PASS" if ok else "FAIL"} {name:18s} {got}')
    print('DRILL_BURROW TESTS', 'PASS' if not failures else f'FAIL ({failures})')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
