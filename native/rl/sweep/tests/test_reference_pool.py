"""Reference pool: fixed-pool Bradley-Terry, game allocation, pool identity, IQM curve."""
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference_pool as rp  # adds research/ to sys.path for elo_ladder
import elo_ladder
import scoring

FIXTURE = Path(__file__).resolve().parent / 'fixtures/run'


def synthetic_games(true_rating, ratings, pairs_per_member, seed=0):
    """Side-swapped games of a candidate at true_rating against fixed members."""
    rng = np.random.default_rng(seed)
    games = []
    for member, rating in ratings.items():
        p = rp.expected_score(true_rating, rating)
        for pair in range(pairs_per_member):
            for side in (0, 1):
                games.append(dict(a='candidate', b=member, map=pair % 3, seed=pair // 3, side=side,
                                  outcome='a' if rng.random() < p else 'b'))
    return games


class FixedFit(unittest.TestCase):
    RATINGS = {'random': 0.0, 'bot-a': 150.0, 'bot-b': 300.0, 'ref-c': 450.0, 'ref-d': 700.0}

    def test_recovers_rating(self):
        games = synthetic_games(400.0, self.RATINGS, 400)
        records = rp.records_against(games, 'candidate')
        rating = elo_ladder.bradley_terry_fixed(records, self.RATINGS)
        self.assertLess(abs(rating - 400.0), 25.0)

    def test_pool_ratings_never_move(self):
        # Only the candidate is estimated: the members' ratings are inputs, untouched.
        ratings = dict(self.RATINGS)
        elo_ladder.bradley_terry_fixed({'random': [10, 0, 0], 'ref-d': [0, 10, 0]}, ratings)
        self.assertEqual(ratings, self.RATINGS)

    def test_undefeated_stays_finite(self):
        rating = elo_ladder.bradley_terry_fixed({'random': [20, 0, 0], 'bot-a': [20, 0, 0]}, self.RATINGS)
        self.assertTrue(math.isfinite(rating))
        self.assertGreater(rating, 600)

    def test_draws_count_half(self):
        even = elo_ladder.bradley_terry_fixed({'bot-b': [0, 0, 20]}, self.RATINGS)
        self.assertAlmostEqual(even, 300.0, places=6)

    def test_no_games(self):
        self.assertEqual(elo_ladder.bradley_terry_fixed({}, self.RATINGS), 0.0)

    def test_bootstrap_interval_brackets_estimate(self):
        games = synthetic_games(400.0, self.RATINGS, 12)
        rating = elo_ladder.bradley_terry_fixed(rp.records_against(games, 'candidate'), self.RATINGS)
        low, high = rp.bootstrap_interval(games, 'candidate', self.RATINGS, samples=200)
        self.assertLess(low, rating)
        self.assertGreater(high, rating)
        self.assertLess(high - low, 400)


class Allocation(unittest.TestCase):
    RATINGS = {'random': 0.0, 'weak': 100.0, 'peer': 400.0, 'peer2': 450.0, 'strong': 1200.0}

    def test_band_concentrates_games(self):
        plan = rp.plan_round(420.0, self.RATINGS, remaining_games=48, band=(.15, .85), round_games=48)
        self.assertEqual(set(plan), {'peer', 'peer2'})   # random: 0.92, weak: 0.87, strong: 0.01
        self.assertEqual(sum(plan.values()), 24)

    def test_empty_band_falls_back_to_nearest_two(self):
        plan = rp.plan_round(-2000.0, self.RATINGS, remaining_games=8, band=(.15, .85), round_games=48)
        self.assertEqual(set(plan), {'random', 'weak'})

    def test_budget_exhausted(self):
        self.assertEqual(rp.plan_round(420.0, self.RATINGS, remaining_games=1, band=(.15, .85), round_games=48), {})
        plan = rp.plan_round(420.0, self.RATINGS, remaining_games=2, band=(.15, .85), round_games=48)
        self.assertEqual(sum(plan.values()), 1)

    def test_candidate_tasks_cycle_maps_and_continue_seeds(self):
        first = rp.candidate_tasks('c', 'm', [0, 1, 2], 3, 1000)
        self.assertEqual([t['map'] for t in first], [0, 0, 1, 1, 2, 2])
        self.assertEqual({t['seed'] for t in first}, {1000})
        self.assertEqual([t['side'] for t in first], [0, 1] * 3)
        later = rp.candidate_tasks('c', 'm', [0, 1, 2], 2, 1000, start_pair=3)
        self.assertEqual([(t['map'], t['seed']) for t in later], [(0, 1001), (0, 1001), (1, 1001), (1, 1001)])
        keys = {(t['map'], t['seed'], t['side']) for t in first + later}
        self.assertEqual(len(keys), len(first) + len(later))  # no repeated game


class Loadouts(unittest.TestCase):
    """The head-to-head rater's loadout plumbing: no flags must change nothing."""
    LINEUPS = '2,4,14\n16,3,13\n25 24 23\n'

    def lineup_file(self, directory):
        path = Path(directory) / 'lineups.txt'
        path.write_text(self.LINEUPS)
        return path

    def test_no_flags_leaves_the_sampler_alone(self):
        # None, not an explicit fixed 2,4,14: fixed mode draws no randomness of its
        # own, so an unflagged rating replays exactly the games it always did.
        self.assertIsNone(rp.rater_loadouts(None, None))
        self.assertEqual(rp.lineup_indices(None), ())
        self.assertEqual(rp.candidate_tasks('c', 'm', [0, 1, 2], 4, 1000),
                         rp.candidate_tasks('c', 'm', [0, 1, 2], 4, 1000, lineups=()))
        self.assertEqual(elo_ladder.pair_tasks('a', 'b', [0, 1], 4, 500),
                         elo_ladder.pair_tasks('a', 'b', [0, 1], 4, 500, lineups=()))
        self.assertNotIn('lineup', rp.candidate_tasks('c', 'm', [0], 1, 1000)[0])

    def test_flags_resolve_to_fixed_and_set_selections(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixed = rp.rater_loadouts([9, 9, 3], None)
            self.assertEqual((fixed['mode'], fixed['fixed']), ('fixed', [9, 9, 3]))
            chosen = rp.rater_loadouts(None, self.lineup_file(tmp))
            self.assertEqual(chosen['mode'], 'set')
            # A three-id line is already mirrored by loadout_lineups, so the rater
            # never needs mirror_share on top of it.
            self.assertEqual(chosen['lineups'][0], [2, 4, 14, 2, 4, 14])
            self.assertEqual(chosen['mirror_share'], 0.0)
            self.assertEqual(rp.lineup_indices(chosen), (0, 1, 2))

    def test_candidate_tasks_cycle_lineups_fastest(self):
        # The band rater hands out a few pairs per member per round; a lineup that
        # only came round once per map cycle would go unplayed at a small budget.
        tasks = rp.candidate_tasks('c', 'm', [0, 1, 2], 6, 1000, lineups=(0, 1, 2))
        pairs = [(t['lineup'], t['map'], t['seed']) for t in tasks[::2]]
        self.assertEqual(pairs, [(0, 0, 1000), (1, 0, 1000), (2, 0, 1000),
                                 (0, 1, 1000), (1, 1, 1000), (2, 1, 1000)])
        later = rp.candidate_tasks('c', 'm', [0, 1, 2], 2, 1000, start_pair=9, lineups=(0, 1, 2))
        self.assertEqual([(t['lineup'], t['map'], t['seed']) for t in later[::2]], [(0, 0, 1001), (1, 0, 1001)])
        self.assertEqual([t['side'] for t in later], [0, 1, 0, 1])

    def test_pair_tasks_repeat_the_schedule_per_lineup(self):
        plain = elo_ladder.pair_tasks('a', 'b', [0, 1], 4, 500)
        with_sets = elo_ladder.pair_tasks('a', 'b', [0, 1], 4, 500, lineups=(0, 1, 2))
        self.assertEqual(len(with_sets), 3 * len(plain))
        self.assertEqual(sorted({t['lineup'] for t in with_sets}), [0, 1, 2])
        keys = {(t['map'], t['seed'], t['side'], t['lineup']) for t in with_sets}
        self.assertEqual(len(keys), len(with_sets))   # no repeated game

    def test_lineup_keys_name_mirror_and_split_rows(self):
        self.assertEqual(rp.lineup_key([2, 4, 14, 2, 4, 14]), '2,4,14')
        self.assertEqual(rp.lineup_key([2, 4, 14, 3, 5, 6]), '2,4,14|3,5,6')
        self.assertEqual(rp.lineup_label('2,4,14'), 'Dash/Bow/Platform')

    def test_ratings_split_per_lineup(self):
        ratings = {'bot': 200.0}
        games = []
        for lineup, wins in (([2, 4, 14] * 2, 8), ([16, 3, 13] * 2, 0)):
            for index in range(8):
                games.append(dict(a='candidate', b='bot', map=0, seed=index, side=index % 2,
                                  outcome='a' if index < wins else 'b', lineup_ids=lineup))
        self.assertEqual(sorted(rp.group_lineups(games)), ['16,3,13', '2,4,14'])
        rows = rp.lineup_ratings(games, 'candidate', ratings, bootstrap=20)
        self.assertGreater(rows['2,4,14']['rating'], 200.0)
        self.assertLess(rows['16,3,13']['rating'], 200.0)
        self.assertEqual(rows['2,4,14']['records']['bot']['wins'], 8)
        self.assertEqual(rows['16,3,13']['label'], 'Roll/Grenade/Smoke')
        # Games that never said what they played (the default fixed draw) get no split.
        self.assertEqual(rp.lineup_ratings([dict(g, lineup_ids=None) for g in games], 'candidate', ratings), {})

    def test_a_lineup_rating_never_shadows_the_default_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = rp.load_pool(fake_pool(tmp))
            run = Path(tmp) / 'run'
            loadouts = rp.rater_loadouts(None, self.lineup_file(tmp))
            directory = rp.rating_dir(run, pool, loadouts)
            self.assertEqual(directory.parent, rp.rating_dir(run, pool))
            directory.mkdir(parents=True)
            payload = dict(schema=rp.SCHEMA, pool_sha256=pool['pool_sha256'], rating=111.0,
                           interval=[100.0, 120.0], games=96, source='band', loadouts=loadouts,
                           lineups={'2,4,14': dict(rating=111.0)})
            (directory / 'policy-000000009.json').write_text(json.dumps(payload))
            (run / 'checkpoints').mkdir(parents=True)
            (run / 'checkpoints' / 'policy-000000009.bin.json').write_text(json.dumps({'learner_steps': 9000}))
            self.assertEqual(rp.run_iqm(run, pool, loadouts=loadouts)['score'], 111.0)
            self.assertIsNone(rp.run_iqm(run, pool))   # the non-recursive glob never sees it
            found, _ = rp.find_rating(run, pool, 'policy-000000009.bin', loadouts)
            self.assertEqual(found['rating'], 111.0)
            self.assertIsNone(rp.find_rating(run, pool, 'policy-000000009.bin')[0])

    def test_cache_rejects_a_different_selection(self):
        identity = dict(sha256='abc')
        pool = dict(pool_sha256='ee' * 32)
        with tempfile.TemporaryDirectory() as tmp:
            loadouts = rp.rater_loadouts(None, self.lineup_file(tmp))
            previous = dict(schema=rp.SCHEMA, pool_sha256=pool['pool_sha256'], checkpoint_identity=identity,
                            budget=240, games=240, source='band', loadouts=loadouts)
            self.assertTrue(rp.cache_hit(previous, identity, pool, 240, loadouts))
            self.assertFalse(rp.cache_hit(previous, identity, pool, 240))
            self.assertFalse(rp.cache_hit(previous, identity, pool, 240, rp.rater_loadouts([2, 4, 14], None)))
            # Ratings written before loadouts existed are the default selection.
            legacy = {key: value for key, value in previous.items() if key != 'loadouts'}
            self.assertTrue(rp.cache_hit(legacy, identity, pool, 240))
            self.assertFalse(rp.cache_hit(legacy, identity, pool, 240, loadouts))


class RatingLayout(unittest.TestCase):
    def test_sha8_directory_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = rp.load_pool(fake_pool(tmp))
            run = Path(tmp) / 'run'
            (run / 'reference').mkdir(parents=True)
            payload = dict(schema=rp.SCHEMA, pool_sha256=pool['pool_sha256'], rating=321.0,
                           checkpoint_identity=dict(sha256='abc'), budget=240, games=240, source='band')
            (run / 'reference' / 'policy-000000001.json').write_text(json.dumps(payload))
            found, path = rp.find_rating(run, pool, 'policy-000000001.bin')
            self.assertEqual(found['rating'], 321.0)
            self.assertEqual(path, run / 'reference' / 'policy-000000001.json')
            sha_dir = rp.rating_dir(run, pool)
            sha_dir.mkdir(parents=True)
            newer = dict(payload, rating=400.0, budget=480, games=480)
            (sha_dir / 'policy-000000001.json').write_text(json.dumps(newer))
            found, path = rp.find_rating(run, pool, 'policy-000000001.bin')
            self.assertEqual(found['rating'], 400.0)
            self.assertEqual(path, sha_dir / 'policy-000000001.json')

    def test_cache_prefers_fuller_band_rating(self):
        identity = dict(sha256='abc')
        pool = dict(pool_sha256='ee' * 32)
        previous = dict(schema=rp.SCHEMA, pool_sha256=pool['pool_sha256'],
                        checkpoint_identity=identity, budget=480, games=480, source='band')
        self.assertTrue(rp.cache_hit(previous, identity, pool, 240))
        self.assertTrue(rp.cache_hit(previous, identity, pool, 480))
        cheap = dict(previous, source='evaluation', budget=80, games=80)
        self.assertFalse(rp.cache_hit(cheap, identity, pool, 240))

    def test_iqm_falls_back_to_rated_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = rp.load_pool(fake_pool(tmp))
            run = Path(tmp) / 'run'
            (run / 'checkpoints').mkdir(parents=True)
            (run / 'reference' / rp.pool_sha8(pool)).mkdir(parents=True)
            (run / 'checkpoints' / 'policy-000000009.bin.json').write_text(
                json.dumps({'learner_steps': 9000}))
            payload = dict(schema=rp.SCHEMA, pool_sha256=pool['pool_sha256'], rating=410.0,
                           interval=[400.0, 420.0], games=240, source='band')
            (run / 'reference' / rp.pool_sha8(pool) / 'policy-000000009.json').write_text(json.dumps(payload))
            summary = rp.run_iqm(run, pool, last_k=3)
            self.assertAlmostEqual(summary['score'], 410.0)
            self.assertEqual(summary['last_k'][0]['checkpoint'], 'policy-000000009.bin')
            self.assertEqual(summary['last_k'][0]['learner_steps'], 9000)


class EvaluationFit(unittest.TestCase):
    def test_fit_from_eval_games_ignores_non_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = rp.load_pool(fake_pool(tmp))
            report = dict(
                seed_base=10001,
                checkpoint_identity=dict(sha256='cafebabe', metadata_sha256='deadbeef'),
                games=[
                    dict(opponent='random', map=0, seed=1, side=0, win=True, timeout=False),
                    dict(opponent='random', map=0, seed=1, side=1, win=True, timeout=False),
                    dict(opponent='mixed-hard', map=0, seed=2, side=0, win=False, timeout=False),
                    dict(opponent='mixed-hard', map=0, seed=2, side=1, win=False, timeout=True),
                    dict(opponent='initial', map=0, seed=3, side=0, win=True, timeout=False),
                    dict(opponent='mixed-hard', map=9, seed=4, side=0, win=True, timeout=False),  # not a pool map
                ])
            result = rp.rating_from_evaluation(report, pool, 'policy.bin')
            self.assertEqual(result['source'], 'evaluation')
            self.assertEqual(result['games'], 4)
            self.assertEqual(result['records']['random']['wins'], 2)
            self.assertEqual(result['records']['mixed-hard']['losses'], 1)
            self.assertEqual(result['records']['mixed-hard']['draws'], 1)
            self.assertNotIn('initial', result['records'])
            run = Path(tmp) / 'run'
            written = rp.write_evaluation_rating(run, report, pool, run / 'checkpoints' / 'policy.bin')
            self.assertEqual(written['rating'], result['rating'])
            self.assertTrue((rp.rating_dir(run, pool) / 'policy.json').is_file())
            # A later band rating with more games is not overwritten.
            band = dict(written, source='band', budget=240, games=240, rating=999.0,
                        checkpoint_identity=report['checkpoint_identity'])
            (rp.rating_dir(run, pool) / 'policy.json').write_text(json.dumps(band))
            kept = rp.write_evaluation_rating(run, report, pool, run / 'checkpoints' / 'policy.bin')
            self.assertEqual(kept['rating'], 999.0)


class HumanRating(unittest.TestCase):
    def test_agent_won_is_a_bot_win_and_exclusions_drop(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = rp.load_pool(fake_pool(tmp))
            directory = Path(tmp) / 'human'
            directory.mkdir()
            rows = [
                dict(session='s1', round=0, agent='mixed-hard', agent_won=True, draw=False, seed=1, level=0),
                dict(session='s1', round=1, agent='mixed-hard', agent_won=False, draw=False, seed=2, level=1),
                dict(session='s1', round=2, agent='mixed-hard', agent_won=False, draw=True, seed=3, level=2),
                dict(session='s1', round=3, agent='stranger', agent_won=False, draw=False, seed=4, level=0),
                dict(session='s1', round=4, a='human', b='bot:mixed-hard', outcome='a', map=5, seed=5, agent='mixed-hard'),
            ]
            (directory / 'games.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            (directory / 'excluded.json').write_text(json.dumps([dict(session='s1', rounds=[1], reason='friend')]))
            result = rp.rate_human(pool, directory, bootstrap=50)
            self.assertEqual(result['games'], 3)  # round 1 excluded, stranger skipped
            self.assertEqual(result['records']['mixed-hard']['losses'], 1)  # agent_won
            self.assertEqual(result['records']['mixed-hard']['draws'], 1)
            self.assertEqual(result['records']['mixed-hard']['wins'], 1)  # outcome a
            self.assertEqual(result['skipped_unknown'], 1)
            self.assertTrue(result['game_mismatch'])
            self.assertEqual(result['map_max'], 5)


class Iqm(unittest.TestCase):
    def test_small_samples_are_means(self):
        self.assertEqual(rp.iqm([1, 2, 3]), 2.0)
        self.assertEqual(rp.iqm([10]), 10.0)

    def test_trims_quartiles(self):
        self.assertEqual(rp.iqm([-100, 1, 2, 3, 4, 5, 6, 1000]), 3.5)

    def test_empty(self):
        with self.assertRaises(ValueError):
            rp.iqm([])


def fake_pool(directory, checkpoint_name='ref.bin'):
    """A pool.json with one real-looking checkpoint file (fixture sidecar contract)."""
    directory = Path(directory)
    weights = b'\x00' * 16
    (directory / checkpoint_name).write_bytes(weights)
    import hashlib
    digest = hashlib.sha256(weights).hexdigest()
    sidecar = json.dumps(dict(schema=1, backend='pufferlib-native', weights_sha256=digest)).encode()
    (directory / (checkpoint_name + '.json')).write_bytes(sidecar)
    members = [dict(name='random', kind='random', rating=0.0),
               dict(name='mixed-hard', kind='scripted', rating=250.0),
               dict(name='ref', kind='checkpoint', path=checkpoint_name, rating=400.0,
                    sha256=digest, metadata_sha256=hashlib.sha256(sidecar).hexdigest())]
    pool = dict(schema=rp.SCHEMA, maps=[0, 1, 2], episode_ticks=7200, members=members,
                pool_sha256=rp.pool_hash(members))
    (directory / 'pool.json').write_text(json.dumps(pool))
    return directory / 'pool.json'


class PoolIdentity(unittest.TestCase):
    def test_verify_accepts_frozen_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = rp.load_pool(fake_pool(tmp))
            self.assertEqual([m['name'] for m in pool['members']], ['random', 'mixed-hard', 'ref'])

    def test_edited_rating_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fake_pool(tmp)
            pool = json.loads(path.read_text())
            pool['members'][1]['rating'] = 999.0
            path.write_text(json.dumps(pool))
            with self.assertRaises(ValueError):
                rp.load_pool(path)

    def test_swapped_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fake_pool(tmp)
            (Path(tmp) / 'ref.bin').write_bytes(b'\x01' * 16)
            with self.assertRaises(ValueError):
                rp.load_pool(path)

    def test_hash_ignores_member_order(self):
        members = json.loads(fake_pool(tempfile.mkdtemp()).read_text())['members']
        self.assertEqual(rp.pool_hash(members), rp.pool_hash(list(reversed(members))))


class ReferenceCurve(unittest.TestCase):
    def test_iqm_over_last_k_with_stub_rater(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            shutil.copytree(FIXTURE, run)
            # The fixture ships sidecars only; give the two evaluated checkpoints weights.
            for name in ('policy-006668288.bin', 'policy-013336576.bin'):
                (run / 'checkpoints' / name).write_bytes(b'')
            pool = rp.load_pool(fake_pool(tmp))
            rated = []

            def rate(checkpoint, pool_, workspace, **options):
                rated.append((Path(checkpoint).name, Path(workspace), options['games']))
                return dict(rating=100.0 * len(rated))

            curve = scoring.reference_curve(run, pool, last_k=3, games=64, workspace=Path(tmp) / 'ws', rate=rate)
            self.assertEqual([p.learner_steps for p in curve], [5001216, 10002432])
            self.assertEqual([p.score for p in curve], [100.0, 150.0])   # IQM of [100], [100, 200]
            self.assertEqual([r[0] for r in rated], ['policy-006668288.bin', 'policy-013336576.bin'])
            self.assertEqual(rated[0][1], Path(tmp) / 'ws')
            self.assertEqual(rated[0][2], 64)
            only_last = scoring.reference_curve(run, pool, last_k=1, rate=rate)
            self.assertEqual([p.learner_steps for p in only_last], [10002432])
            # Default workspace is <run>/reference/<pool_sha8>
            default_dir = []

            def rate_default(checkpoint, pool_, workspace, **options):
                default_dir.append(Path(workspace))
                return dict(rating=1.0)

            scoring.reference_curve(run, pool, last_k=1, rate=rate_default)
            self.assertEqual(default_dir[0], rp.rating_dir(run, pool))

    def test_no_checkpoints_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = rp.load_pool(fake_pool(tmp))
            self.assertEqual(scoring.reference_curve(Path(tmp) / 'missing', pool, rate=lambda *a, **k: None), [])


class PoolIntervals(unittest.TestCase):
    def test_bootstrap_covers_anchor_and_orders_members(self):
        names = ['random', 'strong', 'weak']
        records = {('random', 'strong'): [8, 40, 0], ('random', 'weak'): [20, 28, 0], ('strong', 'weak'): [36, 10, 2]}
        intervals = rp.pool_intervals(names, records, samples=50)
        self.assertEqual(intervals['random'], [0.0, 0.0])
        self.assertLess(intervals['weak'][1], intervals['strong'][0])


if __name__ == '__main__':
    unittest.main()
