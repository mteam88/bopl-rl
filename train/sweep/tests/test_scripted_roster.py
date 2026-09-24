"""Roster name parsing, Elo init, draw weights and share (stdlib only)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'train'), str(ROOT / 'eval'), str(ROOT / 'train/sweep')]
import scripted_roster
import sweep_config


class ParseRoster(unittest.TestCase):
    def test_default_is_the_twelve_v2_levels(self):
        names = scripted_roster.parse_roster('default')
        self.assertEqual(len(names), 12)
        self.assertEqual(names, list(scripted_roster.DEFAULT_ROSTER))
        self.assertIn('mixed-v2-easy', names)
        self.assertNotIn('search-hard', names)   # opt-in: it straggles the env pool
        self.assertNotIn('mixed-hard', names)
        with_search = scripted_roster.parse_roster('default,search-medium,search-hard')
        self.assertEqual(with_search, names + ['search-medium', 'search-hard'])

    def test_explicit_and_v1(self):
        self.assertEqual(scripted_roster.parse_roster('mixed-v2-hard,search-medium'),
                         ['mixed-v2-hard', 'search-medium'])
        self.assertEqual(scripted_roster.parse_roster('archer-hard'), ['archer-hard'])
        self.assertEqual(scripted_roster.parse_roster(''), [])
        self.assertEqual(scripted_roster.parse_roster('none'), [])
        self.assertEqual(scripted_roster.parse_roster(['evasive-v2-medium']), ['evasive-v2-medium'])

    def test_rejects_bad_names(self):
        with self.assertRaises(ValueError):
            scripted_roster.parse_roster('mixed-v2-hard,mixed-v2-hard')
        with self.assertRaises(ValueError):
            scripted_roster.parse_roster('not-a-bot')
        with self.assertRaises(ValueError):
            scripted_roster.parse_roster('mixed')  # kind without level


class EloMath(unittest.TestCase):
    def test_rated_and_unrated_init(self):
        ratings = scripted_roster.POOL_V5_RATINGS
        self.assertEqual(scripted_roster.initial_elo('evasive-v2-hard', ratings), ratings['evasive-v2-hard'])
        self.assertEqual(scripted_roster.initial_elo('mixed-medium', ratings), ratings['mixed-medium'])
        self.assertAlmostEqual(scripted_roster.initial_elo('mixed-v2-medium', ratings),
                               ratings['mixed-v2-hard'] - 75)
        self.assertAlmostEqual(scripted_roster.initial_elo('archer-v2-easy', ratings),
                               ratings['archer-v2-hard'] - 150)

    def test_coin_flip_weight_and_share(self):
        # Learner Elo = entry Elo -> p=0.5, I=1, share=share_max.
        elos = [100.0, 100.0]
        weights = scripted_roster.draw_weights(elos, 100.0)
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(weights[0], 0.25 + 0.1 / 2)
        self.assertAlmostEqual(scripted_roster.informativeness(elos, 100.0), 1.0)
        self.assertAlmostEqual(scripted_roster.scripted_share(elos, 100.0, 0.0, 1.0), 1.0)
        self.assertAlmostEqual(scripted_roster.scripted_share(elos, 100.0, 0.05, 0.4), 0.4)

    def test_hopeless_share_hits_min(self):
        # Every entry far above the learner: p~0, I~0, share = share_min.
        elos = [2000.0, 2500.0]
        self.assertLess(scripted_roster.informativeness(elos, 0.0), 1e-4)
        self.assertAlmostEqual(scripted_roster.scripted_share(elos, 0.0, 0.01, 1.0), 0.01)
        weights = scripted_roster.draw_weights(elos, 0.0)
        self.assertGreater(min(weights), 0.0)
        self.assertEqual(len(weights), 2)

    def test_window_elo_matches_pool_sign(self):
        # Learner wins the window against an equal opponent: entry Elo falls.
        after = scripted_roster.rate_entry(100.0, 100.0, score=8, games=10, k=16)
        self.assertLess(after, 100.0)
        # No games played is a no-op.
        self.assertEqual(scripted_roster.rate_entry(100.0, 50.0, score=0, games=0, k=16), 100.0)

    def test_a_drawn_window_is_half_a_win(self):
        # Ten draws against an equal opponent is a 0.5 score rate, which is what
        # the entry was already expected to hold, so its Elo must not move. A
        # timeout counted as an absence instead would leave games=0 and look
        # identical -- these two differ only when the expectation is not 0.5.
        equal = scripted_roster.apply_window(
            [dict(elo=0.0, games=0, wins=0, draws=0)], [(10, 0, 10)], learner_elo=0.0, k=16)[0]
        self.assertAlmostEqual(equal['elo'], 0.0)
        self.assertEqual((equal['games'], equal['wins'], equal['draws']), (10, 0, 10))
        self.assertAlmostEqual(scripted_roster.score_rate(equal), 0.5)
        # Against a bot the learner is expected to lose to, ten draws are an
        # over-performance and must pull the bot's rating down.
        strong = scripted_roster.apply_window(
            [dict(elo=400.0, games=0, wins=0, draws=0)], [(10, 0, 10)], learner_elo=0.0, k=16)[0]
        self.assertLess(strong['elo'], 400.0)

    def test_resume_state_roundtrip(self):
        ratings = scripted_roster.POOL_V5_RATINGS
        names = ['mixed-v2-hard', 'search-hard']
        entries = scripted_roster.entries_from_names(names, ratings)
        scripted_roster.apply_window(entries, [(10, 3, 2), (4, 4, 0)], learner_elo=0.0, k=16)
        payload = scripted_roster.state_dict(entries, share=0.4, learner_elo=12.5)
        again = scripted_roster.entries_from_names(names, ratings)
        scripted_roster.restore_counts(again, payload)
        self.assertEqual([e['games'] for e in again], [10, 4])
        self.assertEqual([e['wins'] for e in again], [3, 4])
        self.assertEqual([e['draws'] for e in again], [2, 0])
        self.assertEqual([e['elo'] for e in again], [e['elo'] for e in entries])
        scripted_roster.apply_window(again, [(2, 1, 0), (0, 0, 0)], learner_elo=12.5, k=16)
        self.assertEqual(again[0]['games'], 12)
        self.assertEqual(again[1]['games'], 4)


class Overnight7(unittest.TestCase):
    def test_config_validates_without_decay(self):
        path = ROOT / 'train/sweep/configs/overnight-7.toml'
        spec = sweep_config.load(path)
        self.assertNotIn('scripted_decay', spec.fixed)
        self.assertNotIn('scripted_decay', spec.params)
        self.assertEqual(spec.fixed.get('scripted_roster'), 'default')
        self.assertIn('scripted_share_max', spec.params)
        self.assertIn('scripted_share_min', spec.params)
        defaults = spec.defaults()
        self.assertTrue(spec.params['scripted_share_min']['min']
                        <= defaults['scripted_share_min']
                        <= spec.params['scripted_share_min']['max'])
        self.assertTrue(spec.params['scripted_share_max']['min']
                        <= defaults['scripted_share_max']
                        <= spec.params['scripted_share_max']['max'])
        self.assertLessEqual(defaults['scripted_share_min'], defaults['scripted_share_max'])
        argv = spec.trial_arguments(defaults, Path('/tmp/o7'))
        self.assertEqual(argv[argv.index('--scripted-roster') + 1], 'default')
        self.assertNotIn('--scripted-decay', argv)
        self.assertIn('--scripted-share-max', argv)
        self.assertIn('--scripted-share-min', argv)


if __name__ == '__main__':
    unittest.main()
