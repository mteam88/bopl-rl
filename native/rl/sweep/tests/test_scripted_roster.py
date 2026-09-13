"""Roster name parsing, Elo init, draw weights and share (stdlib only)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / 'native/rl/puffer'), str(ROOT / 'native/rl/research')]
import scripted_roster
import sweep_config


class ParseRoster(unittest.TestCase):
    def test_default_is_fourteen(self):
        names = scripted_roster.parse_roster('default')
        self.assertEqual(len(names), 14)
        self.assertEqual(names, list(scripted_roster.DEFAULT_ROSTER))
        self.assertIn('mixed-v2-easy', names)
        self.assertIn('search-hard', names)
        self.assertNotIn('search-easy', names)
        self.assertNotIn('mixed-hard', names)

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
        self.assertEqual(scripted_roster.initial_elo('evasive-v2-hard', ratings, 0), ratings['evasive-v2-hard'])
        self.assertEqual(scripted_roster.initial_elo('mixed-medium', ratings, 10), 10 + ratings['mixed-medium'])
        self.assertAlmostEqual(scripted_roster.initial_elo('mixed-v2-medium', ratings),
                               ratings['mixed-v2-hard'] - 75)
        self.assertAlmostEqual(scripted_roster.initial_elo('archer-v2-easy', ratings, 5),
                               5 + ratings['archer-v2-hard'] - 150)

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
        # Every entry 800 Elo above the learner: p~0, I~0, share = share_min.
        elos = [800.0, 900.0]
        self.assertLess(scripted_roster.informativeness(elos, 0.0), 0.02)
        self.assertAlmostEqual(scripted_roster.scripted_share(elos, 0.0, 0.01, 1.0), 0.01)
        weights = scripted_roster.draw_weights(elos, 0.0)
        self.assertGreater(min(weights), 0.0)
        self.assertEqual(len(weights), 2)

    def test_window_elo_matches_pool_sign(self):
        # Learner wins the window against an equal opponent: entry Elo falls.
        after = scripted_roster.rate_entry(100.0, 100.0, wins=8, games=10, k=16)
        self.assertLess(after, 100.0)
        # Timeouts are not games: empty window is a no-op.
        self.assertEqual(scripted_roster.rate_entry(100.0, 50.0, wins=0, games=0, k=16), 100.0)

    def test_resume_state_roundtrip(self):
        ratings = scripted_roster.POOL_V5_RATINGS
        names = ['mixed-v2-hard', 'search-hard']
        entries = scripted_roster.entries_from_names(names, ratings, elo_init=0)
        scripted_roster.apply_window(entries, [(10, 3), (4, 4)], learner_elo=0.0, k=16)
        payload = scripted_roster.state_dict(entries, share=0.4, learner_elo=12.5)
        again = scripted_roster.entries_from_names(names, ratings, elo_init=0)
        scripted_roster.restore_counts(again, payload)
        self.assertEqual([e['games'] for e in again], [10, 4])
        self.assertEqual([e['wins'] for e in again], [3, 4])
        self.assertEqual([e['elo'] for e in again], [e['elo'] for e in entries])
        scripted_roster.apply_window(again, [(2, 1), (0, 0)], learner_elo=12.5, k=16)
        self.assertEqual(again[0]['games'], 12)
        self.assertEqual(again[1]['games'], 4)


class Overnight7(unittest.TestCase):
    def test_config_validates_without_decay(self):
        path = ROOT / 'native/rl/sweep/configs/overnight-7.toml'
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
