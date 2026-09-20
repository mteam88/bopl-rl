"""What the dashboard says about a run's loadouts, off the evaluation files alone."""
import json
from pathlib import Path
import tempfile
import unittest

import lineup_report
import serve

FIXTURE = Path(__file__).resolve().parents[1] / 'sweep/tests/fixtures/run'
LINEUPS = [[2, 4, 14, 2, 4, 14], [16, 3, 13, 16, 3, 13]]


def evaluation(step, wins, activations, present=((2, 4, 14), (16, 3, 13))):
    """An evaluate.py result.json with two lineups and one frozen opponent."""
    games = []
    for triple in present:
        for opponent in ('random', 'mixed-hard', 'mixed-v2-hard', 'rival-a', 'selfplay'):
            games.append(dict(opponent=opponent, map=0, seed=1, side=0, timeout=opponent == 'selfplay',
                              win=opponent in wins.get(tuple(triple), ()),
                              loadouts=[list(triple), list(triple)]))
    return dict(
        schema=2, step=step, suite='evaluation',
        loadout_selection=dict(mode='set', fixed=[2, 4, 14], pool=[], lineups=LINEUPS,
                               set_file='/tmp/lineups-2.txt', no_dup=False, mirror_share=1.0),
        ability_activations={str(k): v for k, v in activations.items()},
        entity_overflow={'4': dict(rows=100, dropped=10, rate=.1), '13': dict(rows=0, dropped=0, rate=None)},
        groups=[dict(games=10, timeouts=2)], games=games)


def write_run(root, ratings=None):
    run = Path(root) / 'set-run'
    (run / 'evaluations').mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps(dict(schema=1, loadouts=dict(
        mode='set', fixed=[2, 4, 14], pool=[], lineups=LINEUPS,
        set_file='/tmp/lineups-2.txt', no_dup=False, mirror_share=1.0))))
    (run / 'status.json').write_text('{"status":"running","pid":0,"learner_steps":2000}')
    wins = {(2, 4, 14): ('random',), (16, 3, 13): ('random', 'mixed-hard', 'rival-a')}
    for step, activations in ((1000, {2: 40, 4: 0, 14: 20, 16: 5, 3: 5, 13: 0}),
                              (2000, {2: 60, 4: 0, 14: 30, 16: 9, 3: 7, 13: 0})):
        directory = run / 'evaluations' / f'{step:09d}'
        directory.mkdir()
        (directory / 'result.json').write_text(json.dumps(evaluation(step, wins, activations)))
    if ratings:
        directory = run / 'reference' / 'abc12345' / 'loadout-def67890'
        directory.mkdir(parents=True)
        (directory / 'policy-2000.json').write_text(json.dumps(ratings))
    return run


RATING = dict(schema=1, rating=310.0, interval=[290.0, 330.0], games=120, source='band',
              lineups={'2,4,14': dict(rating=350.0, interval=[330.0, 370.0], games=60, draws=1),
                       '16,3,13': dict(rating=270.0, interval=[250.0, 290.0], games=60, draws=0)})


class Selections(unittest.TestCase):
    def test_badges_name_the_draw(self):
        badge = lambda **kw: lineup_report.selection_badge(dict(lineup_report.DEFAULT_SELECTION, **kw))
        self.assertEqual(badge(), '2·4·14')
        self.assertEqual(badge(mode='set', lineups=LINEUPS, set_file='/data/lineups-8.txt', mirror_share=1.0),
                         'set lineups-8 ×2 (mirror 100%)')
        self.assertEqual(badge(mode='pool', pool=list(range(2, 38)), mirror_share=.5), 'pool 36 (mirror 50%)')

    def test_missing_block_is_the_production_triple(self):
        self.assertEqual(lineup_report.selection(None)['fixed'], [2, 4, 14])
        brief = lineup_report.selection_brief(lineup_report.selection({}))
        self.assertEqual([item['name'] for item in brief['fixed']], ['Dash', 'Bow', 'Platform'])
        self.assertEqual(brief['badge'], '2·4·14')


class LegacyRun(unittest.TestCase):
    def test_legacy_fixture_collapses_to_one_fixed_lineup(self):
        payload = serve.loadout_payload(FIXTURE)
        self.assertEqual(payload['loadouts']['mode'], 'fixed')
        self.assertEqual(payload['loadouts']['badge'], '2·4·14')
        # No ability counters existed when this ran: reporting zeroes would invent a finding.
        self.assertFalse(payload['loadouts']['telemetry'])
        self.assertEqual(payload['abilities'], [])
        self.assertEqual([row['key'] for row in payload['lineups']['rows']], ['2,4,14'])
        self.assertTrue(payload['lineups']['rows'][0]['comparable'])
        self.assertEqual(payload['lineups']['ratings'], [])


class SetRun(unittest.TestCase):
    def test_abilities_keep_the_ones_that_never_fired(self):
        with tempfile.TemporaryDirectory() as root:
            payload = serve.loadout_payload(write_run(root))
            rows = {row['name']: row for row in payload['abilities']}
            self.assertEqual(sorted(rows), ['Bow', 'Dash', 'Grenade', 'Platform', 'Roll', 'Smoke'])
            self.assertEqual(rows['Bow']['activations'], 0)
            self.assertEqual(rows['Bow']['games'], 10)
            self.assertEqual(rows['Bow']['dropped'], 20)
            self.assertAlmostEqual(rows['Bow']['drop_rate'], .1)
            self.assertAlmostEqual(rows['Dash']['per_game'], 100 / 10)
            self.assertEqual(rows['Smoke']['rows'], 0)
            self.assertEqual([point['step'] for point in rows['Dash']['trend']], [1000, 2000])

    def test_lineups_split_opponents_and_mark_the_pool_row(self):
        with tempfile.TemporaryDirectory() as root:
            payload = serve.loadout_payload(write_run(root))
            rows = {row['key']: row for row in payload['lineups']['rows']}
            self.assertEqual(sorted(rows), ['16,3,13', '2,4,14'])
            self.assertEqual(rows['2,4,14']['label'], 'Dash/Bow/Platform')
            self.assertTrue(rows['2,4,14']['comparable'])
            self.assertFalse(rows['16,3,13']['comparable'])
            self.assertEqual(rows['16,3,13']['families']['scripted']['games'], 4)
            self.assertAlmostEqual(rows['16,3,13']['families']['scripted']['win_rate'], .5)
            self.assertAlmostEqual(rows['2,4,14']['families']['scripted']['win_rate'], 0)
            self.assertAlmostEqual(rows['16,3,13']['families']['frozen']['win_rate'], 1)
            self.assertAlmostEqual(rows['2,4,14']['families']['random']['win_rate'], 1)
            self.assertAlmostEqual(rows['2,4,14']['timeout_rate'], .2)   # selfplay times out
            self.assertEqual([point['step'] for point in rows['2,4,14']['trend']], [1000, 2000])
            # Strongest scripted record first: the lineup that is failing sorts last.
            self.assertEqual([row['key'] for row in payload['lineups']['rows']], ['16,3,13', '2,4,14'])

    def test_per_lineup_reference_ratings_are_discovered(self):
        with tempfile.TemporaryDirectory() as root:
            run = write_run(root, RATING)
            payload = serve.loadout_payload(run)
            self.assertTrue(payload['lineups']['rated'])
            rating = payload['lineups']['ratings'][0]
            self.assertEqual((rating['pool_sha8'], rating['loadout']), ('abc12345', 'loadout-def67890'))
            self.assertEqual(rating['learner_steps'], 2000)
            self.assertEqual({row['key']: row['rating'] for row in rating['rows']},
                             {'2,4,14': 350.0, '16,3,13': 270.0})
            self.assertEqual({row['key']: row['rating'] for row in payload['lineups']['rows']},
                             {'2,4,14': 350.0, '16,3,13': 270.0})

    def test_ability_win_rate_is_over_the_games_it_was_drawn_in(self):
        with tempfile.TemporaryDirectory() as root:
            rows = {row['name']: row for row in serve.loadout_payload(write_run(root))['abilities']}
            self.assertEqual((rows['Dash']['wins'], rows['Dash']['games']), (2, 10))
            self.assertAlmostEqual(rows['Dash']['win_rate'], .2)
            self.assertAlmostEqual(rows['Roll']['win_rate'], .6)

    def test_evaluations_are_digested_once_per_file(self):
        with tempfile.TemporaryDirectory() as root:
            run = write_run(root)
            serve._evaluation_digest.cache_clear()
            serve.loadout_payload(run)
            misses = serve._evaluation_digest.cache_info().misses
            serve.loadout_payload(run)
            self.assertEqual(serve._evaluation_digest.cache_info().misses, misses)
            self.assertEqual(misses, 2)


class PoolRun(unittest.TestCase):
    """A free draw makes nearly every lineup unique; the table must not become one row per game."""

    def payload(self, root):
        run = Path(root) / 'pool-run'
        directory = run / 'evaluations' / '000001000'
        directory.mkdir(parents=True)
        pool = list(range(2, 38))
        selection = dict(mode='pool', fixed=[2, 4, 14], pool=pool, lineups=[], set_file=None,
                         no_dup=True, mirror_share=0.5)
        (run / 'config.json').write_text(json.dumps(dict(schema=1, loadouts=selection)))
        games = [dict(opponent='mixed-hard', map=0, seed=i, side=0, win=i % 2 == 0, timeout=False,
                      name=f'mixed-hard-{i}', loadouts=[[pool[i % 34], pool[(i + 1) % 34], pool[(i + 2) % 34]]] * 2)
                 for i in range(80)]
        (directory / 'result.json').write_text(json.dumps(dict(
            schema=2, step=1000, loadout_selection=selection, ability_activations={'2': 10},
            entity_overflow={}, groups=[dict(games=80, timeouts=0)], games=games)))
        return serve.loadout_payload(run)

    def test_only_the_most_played_lineups_are_listed(self):
        with tempfile.TemporaryDirectory() as root:
            payload = self.payload(root)
            self.assertEqual(payload['loadouts']['badge'], 'pool 36 (mirror 50%) no-dup')
            self.assertEqual(payload['lineups']['distinct'], 34)
            self.assertEqual(len(payload['lineups']['rows']), serve.LINEUP_ROWS)
            self.assertEqual(payload['lineups']['hidden'], 34 - serve.LINEUP_ROWS)
            games = [row['games'] for row in payload['lineups']['rows']]
            self.assertEqual(max(games), 3)   # the most-played rows survive the cut
            self.assertGreaterEqual(min(games), 2)


if __name__ == '__main__':
    unittest.main()
