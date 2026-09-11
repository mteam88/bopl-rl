"""Behaviour flags from evaluation reports (labels, never score)."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flags

FIXTURE = Path(__file__).resolve().parent / 'fixtures/run'
REPORT = FIXTURE / 'evaluations/006668288/result.json'


def report():
    return json.loads(REPORT.read_text())


class Rules(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(flags.parse_rule('arrows_per_game < 0.05'), ('arrows_per_game', '<', 0.05))
        self.assertEqual(flags.parse_rule('timeout_rate>=0.5'), ('timeout_rate', '>=', 0.5))
        for bad in ('movement < 1', 'arrows_per_game = 1', 'arrows_per_game < x', ''):
            with self.assertRaises(ValueError):
                flags.parse_rule(bad)

    def test_defaults_parse(self):
        self.assertEqual(set(flags.parse_rules(None)), set(flags.DEFAULT_RULES))


class Metrics(unittest.TestCase):
    def test_pooled_over_anchor_groups(self):
        data = report()
        metrics = flags.evaluation_metrics(data)
        groups = [g for g in data['groups'] if g['opponent'] != 'selfplay' and g['map_role'] != 'untrained']
        games = sum(g['games'] for g in groups)
        arrows = sum(g['behavior_totals']['arrows_fired'] for g in groups)
        self.assertEqual(metrics['games'], games)
        self.assertAlmostEqual(metrics['arrows_per_game'], arrows / games)
        self.assertAlmostEqual(metrics['timeout_rate'], sum(g['timeouts'] for g in groups) / games)
        self.assertTrue(0 <= metrics['win_rate'] <= 1)

    def test_raised_by_threshold(self):
        metrics = flags.evaluation_metrics(report())
        arrows = metrics['arrows_per_game']
        self.assertEqual(flags.raised(metrics, {'quiet': f'arrows_per_game < {arrows + 1}'}), ['quiet'])
        self.assertEqual(flags.raised(metrics, {'quiet': f'arrows_per_game > {arrows + 1}'}), [])
        self.assertEqual(flags.raised(dict(games=0), None), [])

    def test_camper_report_flags(self):
        # A synthetic zero-arrow, platform-heavy, timeout-heavy report raises all three defaults.
        data = report()
        for group in data['groups']:
            totals = group['behavior_totals']
            totals['arrows_fired'] = 0
            totals['platform_activations'] = 20 * group['games']
            group['timeouts'] = group['games']
        self.assertEqual(sorted(flags.annotate(data)['flags']), ['no-offense', 'platform-spam', 'stalemate'])


class AnnotateRun(unittest.TestCase):
    def test_writes_reports_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            shutil.copytree(FIXTURE, run)
            rules = {'always': 'games >= 1', 'never': 'games < 0'}
            rows = flags.annotate_run(run, rules)
            self.assertEqual([name for name, _ in rows], ['006668288', '013336576'])
            for name, annotation in rows:
                self.assertEqual(annotation['flags'], ['always'])
                stored = json.loads((run / 'evaluations' / name / 'result.json').read_text())
                self.assertEqual(stored['flags'], ['always'])
                self.assertEqual(stored['flag_rules'], rules)
                self.assertEqual(stored['games'], json.loads((FIXTURE / 'evaluations' / name / 'result.json').read_text())['games'])
            result = json.loads((run / 'result.json').read_text())
            self.assertEqual(result['flags'], ['always'])
            self.assertEqual(result['flags_by_evaluation'], {'006668288': ['always'], '013336576': ['always']})
            self.assertEqual(result['status'], 'complete')  # existing keys survive

    def test_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            shutil.copytree(FIXTURE, run)
            before = (run / 'result.json').read_bytes()
            flags.annotate_run(run, None, write=False)
            self.assertEqual((run / 'result.json').read_bytes(), before)
            self.assertNotIn('flags', json.loads((run / 'evaluations/006668288/result.json').read_text()))


if __name__ == '__main__':
    unittest.main()
