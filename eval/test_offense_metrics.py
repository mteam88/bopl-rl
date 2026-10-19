"""Offense metrics: kill credit, aggression.py's kill columns, and the offense arms' kill-rate rule.

    BOPL_RL_BACKEND=rust venv/bin/python -m unittest eval/test_offense_metrics.py

The regression it guards (2026-10-19): aggression.py reported kills_pm equal to arrow_hits_pm and
first_kill equal to the game length in every row. Neither was a copy bug: RlTelemetry.ArrowHit
counts only lethal arrows, so in the Bow-only 2,4,14 kit every credited kill is an arrow hit, and a
Bopl round has one life, so the first kill ends the game and "seconds to first kill, else the game
length" is the game length in every game. The live test plays a few scripted games and checks the
columns now mean something different from those identities.
"""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import numpy as np  # noqa: E402

import kill_credit  # noqa: E402
import kill_rate  # noqa: E402


class Ev:
    def __init__(self, subject, actor=0, creator=0, kind=1):
        self.kind, self.subject, self.actor, self.creator = kind, subject, actor, creator


class KillCredit(unittest.TestCase):
    def test_credit(self):
        self.assertEqual(kill_credit.credit(2, 0, 1), 'opponent')
        self.assertEqual(kill_credit.credit(0, 2, 1), 'opponent')       # Missile blast / cast Spike
        self.assertEqual(kill_credit.game_credit(0, 1), 'none')         # the game's counters miss it
        self.assertEqual(kill_credit.credit(0, 1, 1), 'self')           # own Missile
        self.assertEqual(kill_credit.credit(0xFFFFFFFF, 0, 1), 'none')  # environment
        self.assertEqual(kill_credit.credit(0, 0, 1), 'none')

    def test_credited_kills(self):
        events = [Ev(2, actor=1), Ev(1, creator=2), Ev(2, actor=2), Ev(1, kind=3, actor=2)]
        self.assertEqual(kill_credit.credited_kills(events, [1, 2]), [1, 1])


class Aggression(unittest.TestCase):
    def setUp(self):
        import aggression
        self.agg = aggression

    def test_km_median(self):
        km = self.agg.km_median
        self.assertEqual(km([1, 2, 3, 4, 5], [True] * 5), 3)
        self.assertIsNone(km([10, 20, 30], [True, False, False]))  # S(t) stays at 2/3
        self.assertEqual(km([1, 2, 3, 4], [True, False, True, True]), 3)

    def test_summary_kill_columns(self):
        def game(seed, seconds, kill):
            row = dict(seed=seed, win=float(kill), seconds=seconds, distance=10., closing=.5, retreating=.2,
                       kills=int(kill), kill_win=float(kill), alive_s=seconds, kills_pm=int(kill) / (seconds / 60),
                       first_kill=seconds if kill else None)
            row.update({c + '_pm': 0. for c in ('dash_activations', 'bow_activations', 'platform_activations',
                                                'arrows_fired', 'arrow_kills')})
            return row
        games = [game(1, 6, True), game(1, 114, False), game(2, 60, True), game(2, 120, False)]
        s = self.agg.summarize(games, np.random.default_rng(0))
        self.assertAlmostEqual(s['kills_pm'][0], 2 / 5, places=4)   # pooled: 2 kills in 300 s, not mean(10, 0, 1, 0)
        self.assertEqual(s['first_kill'][0], 60)                   # KM median with the no-kill games censored
        self.assertNotEqual(s['first_kill'][0], s['seconds'][0])
        self.assertEqual(s['kill_win'][0], .5)

    def test_live_columns(self):
        import os
        if os.environ.get('BOPL_RL_BACKEND') != 'rust':
            self.skipTest('set BOPL_RL_BACKEND=rust')
        import tempfile
        agg = self.agg
        agg.league_games.configure(Path(tempfile.mkdtemp(prefix='test-offense-')))
        todo = [{k: v for k, v in t.items() if k != 'pair_kits'}
                for t in agg.combo_tasks([(2, 4, 14)], 6, 181001, agg.map_catalogue.parse_map_ids(agg.registry.MAPS))]
        games = agg.run({'A': agg.participant('mixed-v2-hard', 'A'), 'B': agg.participant('mixed-v2-hard', 'B')},
                        todo, 2, 181001)
        self.assertEqual(len(games), 12)
        for g in games:
            self.assertIn(g['kills'], (0, 1))
            self.assertEqual(g['first_kill'] is None, g['kills'] == 0)
            if g['kills']:
                self.assertLessEqual(g['first_kill'], g['seconds'])
            if g['kill_win']:
                self.assertEqual(g['kills'], 1)
            # Bow-only kit: a credited kill is a lethal arrow, so the two counts agree game by game
            # (kills from the step events, arrow kills from telemetry: a dropped event shows here)
            self.assertAlmostEqual(g['arrow_kills_pm'] * g['seconds'] / 60, g['kills'], places=6)
            self.assertIn('bow_activations_pm', g)
        self.assertEqual(agg.columns((4, 14, 2))['dash_activations'], 'bow_activations')   # slot 1 of that kit
        self.assertTrue(any(g['kills'] == 0 for g in games) and any(g['kills'] for g in games))


class KillRate(unittest.TestCase):
    @staticmethod
    def row(ticks, a_kills, b_kills, cast=None):
        return dict(ticks=ticks, deaths=[dict(opponent=b_kills), dict(opponent=a_kills)], credited=cast,
                    outcome='a' if a_kills else 'b' if b_kills else 'draw')

    def test_game_and_summary(self):
        rows = [self.row(3600, 1, 0), self.row(7200, 1, 0), self.row(1800, 0, 1), self.row(10800, 0, 0)]
        s = kill_rate.summarize(rows)
        exposure = (3600 + 6960 + 1800 + 6960) / 3600
        self.assertAlmostEqual(s['kh_pre'], 1 / exposure)           # the kill after sudden death does not count
        self.assertAlmostEqual(s['ckpm'], 2 / ((3600 + 7200 + 1800 + 10800) / 3600))
        self.assertAlmostEqual(s['ckpm_b'], 1 / ((3600 + 7200 + 1800 + 10800) / 3600))
        self.assertEqual(s['sd'], .5)
        cast = kill_rate.summarize([self.row(3600, 0, 0, [1, 0])], 'cast')
        self.assertAlmostEqual(cast['kh_pre'], 1.0)

    def test_rule(self):
        def results(seed, kh, panel, group):
            return dict(seed=seed, arm=f'{group}-s{seed}', _kh=kh, _panel=panel)

        def stats(rs):
            return [dict(seed=r['seed'], arm=r['arm'], panel=r['_panel'], kh_pre=r['_kh'], ckpm_b=.1, len_s=100., sd=.5,
                         env_loss=.2) for r in rs]
        ctrl = [results(s, .10 + .002 * i, .60 + .003 * i, 'c') for i, s in enumerate((301, 302, 303, 304))]
        good = [results(s, .14 + .002 * i, .605 + .003 * i, 't') for i, s in enumerate((301, 302, 303, 304))]
        weak = [results(s, .11 + .002 * i, .60 + .003 * i, 'w') for i, s in enumerate((301, 302, 303, 304))]
        saved = kill_rate.seed_stats
        kill_rate.seed_stats = lambda rs, endpoint, credit: stats(rs)
        try:
            v = kill_rate.rule({'good': good, 'weak': weak}, ctrl)
        finally:
            kill_rate.seed_stats = saved
        self.assertTrue(v['good']['kill_rate_pass'] and v['good']['strength_pass'] and v['good']['adopt'])
        self.assertFalse(v['weak']['kill_rate_pass'])
        self.assertGreater(v['good']['ratio'], 1.2)

    def test_factorial_detects_offense_factors(self):
        def run(i, d, e, kh):
            return dict(seed=400 + i, arm=f'ff-{i}', factors=dict(A=1 if i % 2 else -1, D=d, E=e), _kh=kh, _panel=.6,
                        flags=dict(style='styles/offense_kill.toml') if d > 0 else
                        dict(style='styles/offense_time.toml') if e > 0 else {})
        runs = [run(i, d, e, .15 if d > 0 else .10) for i, (d, e) in enumerate([(1, -1), (-1, 1), (1, -1), (-1, -1)] * 2)]
        saved = kill_rate.seed_stats
        kill_rate.seed_stats = lambda rs, endpoint, credit: [
            dict(seed=r['seed'], arm=r['arm'], panel=r['_panel'] + .001 * (r['seed'] % 3), kh_pre=r['_kh'] + .001 * (r['seed'] % 3),
                 ckpm_b=.1, len_s=100., sd=.5, env_loss=.2) for r in rs]
        try:
            v = kill_rate.factorial_rule(runs)
        finally:
            kill_rate.seed_stats = saved
        self.assertEqual(sorted(v), ['I1 (factor D)', 'I3 (factor E)'])
        self.assertTrue(v['I1 (factor D)']['kill_rate_pass'])
        self.assertFalse(v['I3 (factor E)']['kill_rate_pass'])


if __name__ == '__main__':
    unittest.main()
