"""Curve helpers against the fixture run directory (copied from run 007)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scoring

FIXTURE = Path(__file__).resolve().parent / 'fixtures/run'


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

    def test_dispatch_rejects_unknown_scorer(self):
        with self.assertRaises(ValueError):
            scoring.curve(FIXTURE, dict(scorer='anchors'))


class FinalCheckpoint(unittest.TestCase):
    def test_none_without_weights(self):
        # The fixture only carries sidecars, not .bin weights.
        self.assertIsNone(scoring.final_checkpoint(FIXTURE))


if __name__ == '__main__':
    unittest.main()
