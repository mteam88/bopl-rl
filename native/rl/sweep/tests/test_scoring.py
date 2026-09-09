"""Scorers against the fixture run directory (copied from run 007)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scoring

FIXTURE = Path(__file__).resolve().parent / 'fixtures/run'


def expected_anchor_score(result_file, include_heldout=False):
    report = json.loads(result_file.read_text())
    groups = [g for g in report['groups'] if g['opponent'] != 'selfplay'
              and (include_heldout or g.get('map_role') != 'untrained')]
    return sum(g['win_rate'] for g in groups) / len(groups)


class AnchorCurve(unittest.TestCase):
    def test_curve_matches_fixture(self):
        curve = scoring.anchor_curve(FIXTURE)
        self.assertEqual([p.learner_steps for p in curve], [5001216, 10002432])
        for point, directory in zip(curve, ('006668288', '013336576')):
            expected = expected_anchor_score(FIXTURE / 'evaluations' / directory / 'result.json')
            self.assertAlmostEqual(point.score, expected, places=9)
        self.assertGreater(curve[0].elapsed, 0)
        self.assertGreater(curve[1].elapsed, curve[0].elapsed)

    def test_heldout_toggle_changes_group_set(self):
        trained = scoring.anchor_curve(FIXTURE)[0].score
        with_heldout = scoring.anchor_curve(FIXTURE, include_heldout=True)[0].score
        expected = expected_anchor_score(
            FIXTURE / 'evaluations/006668288/result.json', include_heldout=True)
        self.assertAlmostEqual(with_heldout, expected, places=9)
        self.assertNotAlmostEqual(trained, with_heldout, places=9)

    def test_opponent_filter(self):
        only_random = scoring.anchor_curve(FIXTURE, opponents=('random',))
        report = json.loads((FIXTURE / 'evaluations/006668288/result.json').read_text())
        groups = [g for g in report['groups']
                  if g['opponent'] == 'random' and g['map_role'] == 'trained']
        self.assertAlmostEqual(only_random[0].score,
                               sum(g['win_rate'] for g in groups) / len(groups), places=9)

    def test_missing_run_is_empty(self):
        self.assertEqual(scoring.anchor_curve(FIXTURE / 'does-not-exist'), [])


class BehaviourCurve(unittest.TestCase):
    def test_kills_per_episode(self):
        curve = scoring.behaviour_curve(FIXTURE, 'kills_per_episode')
        self.assertGreater(len(curve), 4)
        steps = [p.learner_steps for p in curve]
        self.assertEqual(steps, sorted(steps))
        for point in curve:
            self.assertGreaterEqual(point.score, 0)

    def test_arrow_hit_rate_bounded(self):
        for point in scoring.behaviour_curve(FIXTURE, 'arrow_hit_rate'):
            self.assertTrue(0 <= point.score <= 1)

    def test_unknown_metric(self):
        with self.assertRaises(ValueError):
            scoring.behaviour_curve(FIXTURE, 'nonsense')


class Downsample(unittest.TestCase):
    def test_passthrough_when_short(self):
        points = [scoring.Point(i * 100, i, i) for i in range(1, 4)]
        self.assertEqual(len(scoring.downsample(points, 5)), 3)

    def test_bins_and_exact_final(self):
        points = [scoring.Point(i * 1000, float(i), float(i)) for i in range(1, 21)]
        sampled = scoring.downsample(points, 5)
        self.assertLessEqual(len(sampled), 5)
        self.assertEqual(sampled[-1].learner_steps, 20000)
        self.assertEqual(sampled[-1].score, 20.0)
        steps = [p.learner_steps for p in sampled]
        self.assertEqual(steps, sorted(steps))

    def test_dispatch(self):
        by_config = scoring.curve(FIXTURE, dict(scorer='anchors', opponents=[], include_heldout=False))
        self.assertEqual(len(by_config), 2)


class FinalCheckpoint(unittest.TestCase):
    def test_none_without_weights(self):
        # The fixture only carries sidecars, not .bin weights.
        self.assertIsNone(scoring.final_checkpoint(FIXTURE))


if __name__ == '__main__':
    unittest.main()
