"""Prove `league style-rates` measures without touching play, and measures right.

  python eval/style_rates_check.py [--target l1-main-8000m] [--members 8] [--styles ...] [--json out.json]

Plays the target's first candidate games (league/candidate.py: games.matchup,
one side-swapped pair per member) twice, without and with the styles installed
(eval/environment.py install_style), and checks:

1. identity: every decision's observation rows, action mask and state header
   and players, and each game's exported replay, hash the same both ways, and
   so do the results (outcome, ticks, kits);
2. known answers (eval/style_rates_check.toml): the decision counter fires on
   every decision, the kill/death event terms equal the game's own kill and
   death counters;
3. agreement: each decision's state and events, captured in the dump format
   (eval/replay_dump.py), scored offline by eval/style_score.py's numpy
   interpreter, give the same per-term pay and fires as the in-game C tallies.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'league'), str(ROOT / 'eval'), str(ROOT / 'native/tools'), str(ROOT / 'train')]

import registry  # noqa: E402
import candidate  # noqa: E402
import games as league_games  # noqa: E402
import style_rates  # noqa: E402

CHECK = ROOT / 'eval/style_rates_check.toml'


def key(task):
    return (task['b'], task['map'], task['seed'], task['side'])


def instrument(capture):
    """An Environment subclass hashing every decision (and capturing dump tables when
    `capture`), installed where elo_ladder.play_games imports it. Returns the
    per-thread holder whose .env is the game in progress."""
    import environment
    from replay_data import Event
    from replay_dump import PLAYER, SLIME, ENTITY, EVENT, TELEMETRY, event_row, Q
    lib = environment.LIB
    lib.research_events.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_int)]
    lib.research_events.restype = C.POINTER(Event)
    local = threading.local()
    head = C.sizeof(environment.Header) + C.sizeof(environment.Player * 6)

    base = getattr(environment, 'PlainEnvironment', environment.Environment)
    environment.PlainEnvironment = base

    class Hashing(base):
        def reset(self, seed, map_id=0):
            super().reset(seed, map_id)
            local.env = self
            self.digest, self.steps = hashlib.sha256(), 0
            self.tables = dict(header=[], players=[], slimes=[], entities=[], telemetry=[], events=[])

        def step(self, actions):
            super().step(actions)
            self.digest.update(self.rows.tobytes()); self.digest.update(self.mask.tobytes())
            self.digest.update(C.string_at(C.addressof(self.state), head))
            t = self.steps
            self.steps += 1
            if not capture:
                return
            count, support = C.c_uint32(), C.c_int()
            events = lib.research_events(self.handle, C.byref(count), C.byref(support))
            self.support = support.value
            self.tables['events'] += [event_row(t, events[k]) for k in range(count.value)]
            s = self.state
            h = s.header
            # replay_dump.dump's rows, one per decision; `tick` is the row index, so
            # style_score delivers a step's events to that step's decision.
            self.tables['header'].append((t, h.episode_ticks, h.flags, h.entity_count, h.slime_count, h.sudden_death))
            for i in range(h.player_count):
                p = s.players[i]
                self.tables['players'].append((t, i, p.id, p.team, p.flags, p.alive_bodies, p.stocks, p.kills, p.deaths,
                                               p.cause_of_death, p.x * Q, p.y * Q, p.scale * Q, p.invisible_time * Q,
                                               p.ability_time * Q))
                self.tables['telemetry'].append(tuple([t, i] + [getattr(s.telemetry[i], n)
                                                                for n, _ in s.telemetry[i]._fields_]))
            for k in range(h.slime_count):
                m = s.slimes[k]
                self.tables['slimes'].append((t, m.id, m.owner, m.flags, m.state, m.x * Q, m.y * Q, m.vx * Q, m.vy * Q,
                                              m.rotation * Q, m.scale * Q, tuple(m.abilities), tuple(m.ability_flags),
                                              tuple(c * Q for c in m.cooldowns), tuple(c * Q for c in m.timers)))
            for k in range(h.entity_count):
                e = s.entities[k]
                self.tables['entities'].append((t, e.id, e.kind, e.layer, e.owner, e.flags, e.hierarchy, e.reserved0,
                                                e.reserved1, e.x * Q, e.y * Q, e.vx * Q, e.vy * Q, e.right_x * Q,
                                                e.right_y * Q, e.up_x * Q, e.up_y * Q, e.radius * Q, e.rotation * Q,
                                                e.angular_velocity * Q, e.inverse_mass * Q, e.clock_rate * Q))

        def finish(self, directory, name):
            """(decision digest, replay sha256, dump path or None) of the game just played."""
            path = Path(directory) / f'{name}.rep'
            self.replay(path)
            replay = hashlib.sha256(path.read_bytes()).hexdigest()
            path.unlink()
            dump = None
            if capture:
                dump = Path(directory) / f'{name}.npz'
                t = self.tables
                np.savez(dump, meta=np.array(json.dumps({'events': bool(self.support)})),
                         header=np.array(t['header'], dtype=[('tick', 'u4'), ('episode_ticks', 'u4'), ('flags', 'u4'),
                                                             ('entity_count', 'u4'), ('slime_count', 'u4'),
                                                             ('sudden_death', 'u4')]),
                         players=np.array(t['players'], dtype=PLAYER), slimes=np.array(t['slimes'], dtype=SLIME),
                         entities=np.array(t['entities'], dtype=ENTITY),
                         telemetry=np.array(t['telemetry'], dtype=[('tick', 'u4'), ('player', 'u1')]
                                            + [(f, 'u4') for f in TELEMETRY]),
                         events=np.array(t['events'], dtype=EVENT))
            final = self.state
            counters = [dict(kills=final.players[i].kills, deaths=final.players[i].deaths) for i in range(2)]
            return dict(digest=self.digest.hexdigest(), replay=replay, dump=str(dump) if dump else None,
                        steps=self.steps, counters=counters)

    environment.Environment = Hashing
    return local


def play(built, tasks, style, directory, capture, seed_base):
    local = instrument(capture)
    out = {}

    def on_result(result):
        env = local.env
        name = '-'.join(str(v) for v in key(result))
        out[key(result)] = dict(result=result, **env.finish(directory, name))
    # games.play's call (one chunk: a check schedule is far below 256 kit rows), plus on_result.
    import elo_ladder
    from evaluation_protocol import DEFAULT_LOADOUT
    rows = {}
    for task in tasks:
        task['lineup'] = rows.setdefault(tuple(task['kits']), len(rows))
    loadouts = dict(mode='set', fixed=list(DEFAULT_LOADOUT), pool=[], lineups=[list(r) for r in rows],
                    set_file=None, no_dup=False, mirror_share=0.0)
    elo_ladder.play_games(built, tasks, registry.EPISODE_TICKS, 4, seed_base, delay=registry.DELAY, loadouts=loadouts,
                          on_result=on_result, style=style)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', default='l1-main-8000m')
    parser.add_argument('--members', type=int, default=8, help='members to meet (one side-swapped pair each)')
    parser.add_argument('--styles', nargs='+', type=Path,
                        default=sorted(p for p in (ROOT / 'styles').glob('*.toml') if p.name != 'categories.toml'))
    parser.add_argument('--seed-base', type=int, default=70001)
    parser.add_argument('--workspace', default='/tmp/league-style-rates-check')
    parser.add_argument('--condition', help='ratings condition for the band (league style-rates --condition)')
    parser.add_argument('--json', type=Path)
    parser.add_argument('--keep', type=Path, help='keep the per-game dumps here')
    args = parser.parse_args()

    league = registry.load()
    label, checkpoint, me, identity = style_rates.target(args.target, league)
    _, _, rows, maps, used = candidate.opponents(me, identity['sha256'], args.seed_base, league, args.condition)
    # A spread of the band: members in rating order, every k-th.
    order = sorted(used, key=lambda m: used[m])
    chosen = [order[int(i * (len(order) - 1) / max(1, args.members - 1))] for i in range(args.members)]
    chosen = list(dict.fromkeys(chosen))
    tasks = [t for m in chosen for t in league_games.matchup(me, rows[m], maps, 1, args.seed_base)]
    import elo_ladder
    built = {m: league_games.participant(rows[m]) for m in chosen}
    built[candidate.CANDIDATE] = elo_ladder.Model(candidate.CANDIDATE, checkpoint, 'candidate')
    league_games.configure(Path(args.workspace))
    style = style_rates.compile_styles(list(args.styles) + [CHECK])
    terms = style['terms']
    print(f'{label}: {len(tasks)} games against {len(chosen)} members, {len(terms)} terms', flush=True)

    with tempfile.TemporaryDirectory(prefix='style-rates-check-') as directory:
        if args.keep:
            args.keep.mkdir(parents=True, exist_ok=True); directory = str(args.keep)
        plain = play(built, [dict(t) for t in tasks], None, directory, False, args.seed_base)
        styled = play(built, [dict(t) for t in tasks], style, directory, True, args.seed_base)

        # 1. identity
        fields = ('outcome', 'ticks', 'lineup_ids', 'draw_kind')
        identical = dict(games=len(plain), results=0, decisions=0, replays=0)
        for k, a in plain.items():
            b = styled[k]
            identical['results'] += all(a['result'].get(f) == b['result'].get(f) for f in fields)
            identical['decisions'] += a['digest'] == b['digest'] and a['steps'] == b['steps']
            identical['replays'] += a['replay'] == b['replay']

        # 2. known answers
        index = {name: t for t, name in enumerate(terms)}
        known = dict(seats=0, decision=0, kill_event=0, death_event=0, event_support=0, kills=0, deaths=0,
                     mismatches=[])
        for k, game in styled.items():
            result = game['result']
            known['event_support'] += result['event_support'] == 1
            for seat in (0, 1):
                tally = result['style'][seat]
                known['seats'] += 1
                pay, fires = tally['style_rates_check.decision']
                known['decision'] += pay == fires == result['style_decisions'][seat] == game['steps']
                kills, deaths = tally['style_rates_check.kill_event'][0], tally['style_rates_check.death_event'][0]
                known['kill_event'] += kills == game['counters'][seat]['kills']
                known['death_event'] += deaths == game['counters'][seat]['deaths']
                if kills != game['counters'][seat]['kills'] or deaths != game['counters'][seat]['deaths']:
                    known['mismatches'].append(dict(game=list(k), seat=seat, events=dict(kills=kills, deaths=deaths),
                                                    counters=game['counters'][seat], dump=game['dump']))
                known['kills'] += game['counters'][seat]['kills']; known['deaths'] += game['counters'][seat]['deaths']

        # 3. agreement with the offline interpreter
        import style_score
        prog = style_score.Decoded(style['data'])
        agree = dict(seat_terms=0, fires_equal=0, pay_equal=0, pay_max_abs_diff=0.0, decisions_equal=0,
                     mismatches=[])
        per_term = {name: dict(fires_equal=0, pay_equal=0, n=0, fires_c=0, fires_py=0) for name in terms}
        for k, game in styled.items():
            dump = style_score.Game(game['dump'])
            scored = style_score.score_game(prog, dump, support=dump.event_support)
            result = game['result']
            for seat in (0, 1):
                pays, _ = scored[seat]
                agree['decisions_equal'] += dump.D == result['style_decisions'][seat]
                for t, name in enumerate(terms):
                    c_pay, c_fires = result['style'][seat][name]
                    py_pay = float(pays[:, t].astype(np.float64).sum()); py_fires = int((pays[:, t] != 0).sum())
                    row = per_term[name]
                    row['n'] += 1; row['fires_c'] += c_fires; row['fires_py'] += py_fires
                    agree['seat_terms'] += 1
                    agree['fires_equal'] += c_fires == py_fires; row['fires_equal'] += c_fires == py_fires
                    agree['pay_equal'] += c_pay == py_pay; row['pay_equal'] += c_pay == py_pay
                    agree['pay_max_abs_diff'] = max(agree['pay_max_abs_diff'], abs(c_pay - py_pay))
                    if c_fires != py_fires or c_pay != py_pay:
                        agree['mismatches'].append(dict(game=list(k), seat=seat, term=name, c=[c_pay, c_fires],
                                                        py=[py_pay, py_fires]))
    ok = (identical['results'] == identical['decisions'] == identical['replays'] == identical['games']
          and known['decision'] == known['kill_event'] == known['death_event'] == known['seats']
          and known['event_support'] == len(styled)
          and agree['fires_equal'] == agree['pay_equal'] == agree['seat_terms']
          and agree['decisions_equal'] == known['seats'])
    report = dict(target=label, members=[rows[m]['name'] for m in chosen], games=len(tasks), terms=terms,
                  program_sha256=style['sha256'], identical=identical, known=known,
                  agreement=dict(agree, mismatches=agree['mismatches'][:50]), per_term=per_term, ok=ok)
    print(json.dumps({k: v for k, v in report.items() if k not in ('per_term',)}, indent=1))
    for name, row in per_term.items():
        print(f"  {name:32s} fires C {row['fires_c']:6d} py {row['fires_py']:6d}  "
              f"fires equal {row['fires_equal']}/{row['n']}  pay equal {row['pay_equal']}/{row['n']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + '\n')
    print('OK' if ok else 'FAILED')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
