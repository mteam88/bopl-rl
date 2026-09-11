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

    def test_no_checkpoints_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = rp.load_pool(fake_pool(tmp))
            self.assertEqual(scoring.reference_curve(Path(tmp) / 'missing', pool, rate=lambda *a, **k: None), [])


if __name__ == '__main__':
    unittest.main()


class RescoreComparison(unittest.TestCase):
    def test_ranks_over_shared_trials(self):
        import rescore
        row = lambda i, iqm: dict(id=i, iqm=iqm, old_score=0.5, rating=iqm, interval=[iqm - 10, iqm + 10], flags=[], last_k=[])
        previous = dict(pool='p1', pool_sha256='a' * 64, trials=[row('000', 300), row('001', 350), row('002', 320)])
        report = dict(sweep='/s', pool='p2', pool_sha256='b' * 64, games=1, first_pass_games=1, band=[.15, .85],
                      last_k=3, flag_rules={}, trials=[row('000', 340), row('001', 330), row('003', 900)])
        lines = rescore.comparison(report, previous)
        table = [l for l in lines if l.startswith('| 0')]
        self.assertEqual([l.split(' | ')[0][2:] for l in table], ['000', '001'])  # 003 not shared
        self.assertIn('| 000 | 300 | 2 | 340 | 1 | +1 | +40 |', lines)
        self.assertIn('| 001 | 350 | 1 | 330 | 2 | -1 | -20 |', lines)
        self.assertTrue(lines[-1].endswith('000, 001.'))


class PoolIntervals(unittest.TestCase):
    def test_bootstrap_covers_anchor_and_orders_members(self):
        names = ['random', 'strong', 'weak']
        records = {('random', 'strong'): [8, 40, 0], ('random', 'weak'): [20, 28, 0], ('strong', 'weak'): [36, 10, 2]}
        intervals = rp.pool_intervals(names, records, samples=50)
        self.assertEqual(intervals['random'], [0.0, 0.0])
        self.assertLess(intervals['weak'][1], intervals['strong'][0])
