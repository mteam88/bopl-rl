"""Read an evaluation's ability and lineup telemetry without starting a game."""
import json
from pathlib import Path
import tempfile
import unittest

import lineup_report


def report(loadouts):
    """An evaluate.py result.json shaped like the fields the report reads."""
    games = []
    for index, row in enumerate(loadouts):
        for opponent, win in (('mixed-hard', index == 0), ('random', True), ('selfplay', False)):
            games.append(dict(opponent=opponent, map=0, seed=index, side=0, win=win,
                              timeout=False, loadouts=row))
    return dict(step=42, loadout_selection=dict(mode='set', set_file='/tmp/lineups-8.txt'),
                ability_activations={'2': 100, '4': 0, '14': 7},
                entity_overflow={'4': dict(rows=200, dropped=20, rate=.1), '9': dict(rows=0, dropped=0, rate=None)},
                groups=[dict(games=8, timeouts=1), dict(games=8, timeouts=0)], games=games)


class LineupReport(unittest.TestCase):
    ROWS = [[[2, 4, 14], [2, 4, 14]], [[16, 3, 13], [16, 3, 13]]]

    def test_abilities_resolve_to_names_and_keep_silent_slots(self):
        rows = lineup_report.ability_rows(report(self.ROWS))
        # Ability 4 activated nothing yet still has entity rows; 9 has neither and is dropped.
        self.assertEqual([(i, n, a, r, d) for i, n, a, r, d in rows],
                         [(2, 'Dash', 100, 0, 0), (4, 'Bow', 0, 200, 20), (14, 'Platform', 7, 0, 0)])

    def test_timeout_rate_is_over_every_group(self):
        self.assertAlmostEqual(lineup_report.timeout_rate(report(self.ROWS)), 1 / 16)
        self.assertEqual(lineup_report.timeout_rate({}), 0.0)

    def test_lineup_rows_split_scripted_from_random_and_drop_selfplay(self):
        rows = lineup_report.lineup_rows(report(self.ROWS))
        self.assertEqual(sorted(rows), [('16,3,13', 'random'), ('16,3,13', 'scripted'),
                                        ('2,4,14', 'random'), ('2,4,14', 'scripted')])
        self.assertEqual(rows[('2,4,14', 'scripted')], dict(games=1, wins=1, timeouts=0, undecided=0))
        self.assertEqual(rows[('16,3,13', 'scripted')], dict(games=1, wins=0, timeouts=0, undecided=0))

    def test_asymmetric_episodes_keep_both_halves(self):
        rows = lineup_report.lineup_rows(report([[[2, 4, 14], [16, 3, 13]]]))
        self.assertEqual(sorted({key for key, _ in rows}), ['2,4,14|16,3,13'])

    def test_evaluations_are_read_in_step_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            for step in (2000, 1000):
                directory = run / 'evaluations' / f'{step:09d}'
                directory.mkdir(parents=True)
                (directory / 'result.json').write_text(json.dumps(dict(report(self.ROWS), step=step)))
            (run / 'evaluations' / '000003000').mkdir()
            (run / 'evaluations' / '000003000' / 'result.json').write_text('{"step": 3')  # still being written
            rows = lineup_report.evaluations(run)
            self.assertEqual([step for step, _ in rows], [1000, 2000])
            self.assertIn('Dash/Bow/Platform', lineup_report.format_report(*rows[0]))


if __name__ == '__main__':
    unittest.main()
