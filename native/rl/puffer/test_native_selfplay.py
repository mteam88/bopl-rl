"""CPU tests: the trainer's pool state drives pufferlib.selfplay.step as intended."""
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import native_selfplay

from pufferlib import selfplay

UPSTREAM_SAMPLE, UPSTREAM_EVICT = selfplay.sample_opponent, selfplay.evict


class FakeBackend:
    """Records the calls the pool makes; alignment is scripted per bank."""
    def __init__(self, banks):
        self.loaded = {b: [] for b in range(banks)}
        self.saved = []
        self.aligned = {b: 0 for b in range(banks)}
        self.resets = []

    def save_weights(self, trainer, path):
        Path(path).write_bytes(b'w')
        self.saved.append(path)

    def load_frozen_bank(self, trainer, bank, path):
        self.loaded[bank].append(path)

    def count_aligned(self, trainer, tag, reset):
        value = self.aligned[tag - 1]
        if reset:
            self.resets.append(tag - 1)
            self.aligned[tag - 1] = 0
        return value


def logs(n, **per_bank):
    """Native env log: n-averaged per-bank counts, as static_vec_log emits them."""
    flat = {'env/n': float(n)}
    for bank, (games, wins) in per_bank.items():
        flat[f'env/hist_n_bank_{bank}'] = games / n
        flat[f'env/hist_score_bank_{bank}'] = wins / n
    return flat


class PoolFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.pool_dir = Path(self.directory.name)
        self.bootstrap = str(self.pool_dir / 'initial.bin')
        Path(self.bootstrap).write_bytes(b'w')
        self.trainer = types.SimpleNamespace(global_step=0)
        self.backend = FakeBackend(2)
        self.pool = native_selfplay.state(self.pool_dir, self.bootstrap, 2, 3, 0, max_size=4, swap_winrate=.6,
                                          min_games=4, snapshot_interval=1000, opp_timeout_steps=5000,
                                          elo_init=0., elo_k=16., seed=1)
        self.ledger = native_selfplay.members(self.bootstrap)

    def tearDown(self):
        self.directory.cleanup()

    def step(self, flat, epoch):
        native_selfplay.credit(self.pool, flat, self.ledger)
        native_selfplay.step(selfplay.step, self.trainer, self.backend, self.pool, flat, epoch)
        native_selfplay.register(self.pool, self.ledger, self.trainer.global_step * 4, self.trainer.global_step * 3)

    def summary(self):
        return native_selfplay.summary(self.pool, self.ledger)


class NativeSelfplayTests(PoolFixture):
    def test_state_matches_selfplay_step_contract(self):
        self.assertEqual(self.pool['num_banks'], 2)
        self.assertEqual([b['num_hist_envs'] for b in self.pool['banks']], [3, 3])
        self.assertEqual(self.pool['pool'], [dict(path=self.bootstrap, elo=0.)])
        with self.assertRaises(ValueError):
            native_selfplay.state(self.pool_dir, self.bootstrap, 2, 3, 0, max_size=4, swap_winrate=1.5, min_games=4,
                                  snapshot_interval=1, opp_timeout_steps=1, elo_init=0., elo_k=16., seed=1)
        summary = self.summary()
        self.assertEqual(summary, dict(schema=1, elo=0., size=1, winrate=None,
                                       entries=[dict(step=0, learner_steps=0, elo=0., games=0, winrate=None)],
                                       banks=[dict(opponent_step=0, opponent_elo=0., games=0, winrate=None, pending_step=None,
                                                   winrate_at_swap=0., epochs_to_align=0)]*2))

    def test_winrate_swap_waits_for_alignment_and_is_reported(self):
        before = self.summary()
        # Two updates of 3 games per bank: bank 0 wins 5/6 (>= 0.6 over >= 4 games), bank 1 loses.
        self.trainer.global_step = 100
        self.step(logs(4, **{'0': (3, 3), '1': (3, 0)}), 0)
        self.assertLess(self.pool['banks'][0]['cur_opp_elo'], 0.)
        self.assertGreater(self.pool['banks'][1]['cur_opp_elo'], 0.)
        self.assertIsNone(self.pool['banks'][0]['pending_opp_path'])
        self.trainer.global_step = 200
        self.step(logs(4, **{'0': (3, 2), '1': (3, 0)}), 1)
        after = self.summary()
        bank = self.pool['banks'][0]
        self.assertEqual(bank['hist_n'], 6.)
        self.assertIsNotNone(bank['pending_opp_path'])
        self.assertEqual(len(self.pool['pool']), 2, 'a winrate swap snapshots while the pool is small')
        self.assertEqual(self.backend.loaded[0], [], 'no bank loads before the bank aligned')
        events = native_selfplay.events(before, after, self.ledger, learner_steps=600)
        self.assertEqual([e['event'] for e in events], ['snapshot', 'armed'])
        self.assertEqual(events[1]['bank'], 0)
        self.assertEqual(events[1]['games'], 6)
        self.assertAlmostEqual(events[1]['winrate'], 5 / 6)
        self.assertEqual(events[0]['step'], 800)
        self.assertTrue(all(e['learner_steps'] == 600 for e in events))
        # A partial alignment does not swap; the full one does, and resets bank 0's flags and stats.
        self.backend.aligned[0] = 2
        self.trainer.global_step = 300
        self.step(logs(4), 2)
        self.assertEqual(self.backend.loaded[0], [])
        self.backend.aligned[0] = 3
        before = self.summary()
        self.trainer.global_step = 400
        self.step(logs(4), 3)
        after = self.summary()
        self.assertEqual(len(self.backend.loaded[0]), 1)
        self.assertEqual(self.backend.resets, [0])
        self.assertEqual(bank['hist_n'], 0.)
        self.assertEqual(bank['cur_opp_path'], self.backend.loaded[0][0])
        self.assertEqual(bank['opp_started_step'], 400)
        events = native_selfplay.events(before, after, self.ledger, learner_steps=1200)
        self.assertEqual([(e['event'], e['bank'], e['epochs_to_align']) for e in events], [('loaded', 0, 2)])
        # Both banks rate the shared bootstrap separately; the retired record keeps bank 0's view.
        self.assertEqual(events[0]['retired'], dict(step=0, elo=before['banks'][0]['opponent_elo'], games=12, winrate=5/12))
        self.assertEqual([e['games'] for e in after['entries']], [12, 0])
        self.assertEqual(after['entries'][1]['learner_steps'], 600)
        self.assertIn(after['banks'][0]['opponent_step'], {0, 800})
        # The metrics summary reports the bank state after the swap.
        self.assertEqual(after['banks'][0]['games'], 0)
        self.assertEqual(after['banks'][1]['games'], 6)
        self.assertEqual(after['banks'][1]['winrate'], 0.)
        self.assertEqual(after['size'], 2)

    def test_interval_snapshots_and_eviction(self):
        self.pool['opp_timeout_steps'] = 0
        kinds = []
        for epoch in range(6):
            before = self.summary()
            self.trainer.global_step = 1000 * (epoch + 1)
            self.step(logs(1), epoch)
            after = self.summary()
            kinds += [(e['event'], e['step']) for e in native_selfplay.events(before, after, self.ledger, learner_steps=0)]
        self.assertEqual(after['size'], 4, 'pool is capped at max_size')
        self.assertEqual([e['step'] for e in after['entries']], [0, 16000, 20000, 24000])
        self.assertEqual(kinds, [('snapshot', 4000), ('snapshot', 8000), ('snapshot', 12000), ('snapshot', 16000),
                                 ('evicted', 4000), ('snapshot', 20000), ('evicted', 8000), ('snapshot', 24000), ('evicted', 12000)])
        self.assertEqual(after['entries'][0]['step'], 0, 'stride eviction never drops the bootstrap anchor')

    def test_simultaneous_bank_swaps_snapshot_once(self):
        # Both banks beat the shared bootstrap in the same update: upstream would
        # append the same {global_step}.bin twice; the pool must list it once.
        self.trainer.global_step = 100
        before = self.summary()
        self.step(logs(4, **{'0': (4, 4), '1': (4, 4)}), 0)
        after = self.summary()
        paths = [e['path'] for e in self.pool['pool']]
        self.assertEqual(len(paths), len(set(paths)), 'no duplicate pool entries')
        self.assertEqual(after['size'], 2)
        self.assertEqual([e['event'] for e in native_selfplay.events(before, after, self.ledger, learner_steps=0)],
                         ['snapshot', 'armed', 'armed'])
        self.assertTrue(all(b['pending_opp_path'] is not None for b in self.pool['banks']))

    def test_timeout_swap_without_snapshot(self):
        self.pool['snapshot_interval'] = 0
        self.trainer.global_step = 5000
        before = self.summary()
        self.step(logs(1), 0)
        after = self.summary()
        self.assertEqual(after['size'], 1)
        self.assertTrue(all(b['pending_step'] == 0 for b in after['banks']))
        self.assertEqual([e['event'] for e in native_selfplay.events(before, after, self.ledger, learner_steps=0)], ['armed', 'armed'])


class LeagueTests(PoolFixture):
    """Pinned members and the PFSP draw (POLICY_V4_PLAN.md 7e) on the same fake backend."""
    def setUp(self):
        super().setUp()
        self.pinned = []
        for name in ('aim-s73', 'aim-s74', 'aim-maps'):
            path = self.pool_dir / f'{name}.bin'
            path.write_bytes(b'p')
            self.pinned.append((name, str(path)))
        native_selfplay.add_pinned(self.pool, self.ledger, self.pinned)
        native_selfplay.install_sampler(selfplay, self.pool, self.ledger, mode='pfsp')

    def tearDown(self):
        # The upstream module must be restored for the other test class.
        selfplay.sample_opponent, selfplay.evict = UPSTREAM_SAMPLE, UPSTREAM_EVICT
        super().tearDown()

    def test_pinned_are_members_with_negative_steps(self):
        summary = self.summary()
        self.assertEqual(summary['size'], 4)
        self.assertEqual([e['step'] for e in summary['entries']], [-1, -2, -3, 0])
        self.assertEqual([r['name'] for r in summary['pinned']], ['aim-s73', 'aim-s74', 'aim-maps'])
        self.assertEqual(summary['sampler'], 'pfsp')
        self.assertIsNone(summary['pinned_draw_share'])
        self.assertEqual(summary['pinned_banks'], 0, 'banks start on the bootstrap as before')
        with self.assertRaises(ValueError):
            native_selfplay.add_pinned(self.pool, self.ledger, self.pinned[:1])

    def test_pinned_survive_eviction_and_max_size_bounds_own_snapshots(self):
        self.pool['opp_timeout_steps'] = 0
        for epoch in range(8):
            self.trainer.global_step = 1000 * (epoch + 1)
            self.step(logs(1), epoch)
        entries = self.pool['pool']
        self.assertEqual([e.get('name') for e in entries[:3]], ['aim-s73', 'aim-s74', 'aim-maps'])
        own = [e for e in entries if not e.get('pinned')]
        self.assertEqual(len(own), 4, 'max_size caps the run\'s own snapshots, not the pinned members')
        self.assertEqual(self.ledger[own[0]['path']]['step'], 0, 'the bootstrap anchor is still kept')

    def test_pfsp_weights_follow_the_ledger(self):
        # Unplayed: 0.25. A member beaten 9/10 (smoothed 10/12): (1/6)^2. One never beaten: (11/12)^2.
        paths = [p for _, p in self.pinned]
        self.ledger[paths[0]].update(games=10, wins=9)
        self.ledger[paths[1]].update(games=10, wins=0)
        weights = native_selfplay.pfsp_weights(self.pool['pool'][:3], self.ledger)
        self.assertAlmostEqual(weights[0], (1 - 10 / 12) ** 2)
        self.assertAlmostEqual(weights[1], (1 - 1 / 12) ** 2)
        self.assertAlmostEqual(weights[2], 0.25)
        # A member beaten 1000/1000 keeps the floor.
        self.ledger[paths[0]].update(games=1000, wins=1000)
        self.assertEqual(native_selfplay.pfsp_weights(self.pool['pool'][:1], self.ledger)[0], native_selfplay.PFSP_FLOOR)

    def test_draws_prefer_the_unbeaten_and_are_counted(self):
        paths = [p for _, p in self.pinned]
        self.ledger[paths[0]].update(games=200, wins=200)   # beaten: floor
        self.ledger[paths[1]].update(games=200, wins=0)     # never beaten: ~1
        self.ledger[paths[2]].update(games=200, wins=100)   # even: ~0.25
        self.ledger[self.bootstrap].update(games=200, wins=200)
        draws = {}
        for _ in range(400):
            entry = selfplay.sample_opponent(self.pool['pool'], self.pool['rng'])
            draws[entry.get('name', 'own')] = draws.get(entry.get('name', 'own'), 0) + 1
        self.assertGreater(draws['aim-s74'], 0.6 * 400)
        self.assertGreater(draws['aim-maps'], 0.1 * 400)
        self.assertLess(draws.get('aim-s73', 0) + draws.get('own', 0), 0.05 * 400)
        summary = self.summary()
        self.assertEqual(sum(r['draws'] for r in summary['pinned']) + self.pool['pool'][3].get('draws', 0), 400)
        self.assertGreater(summary['pinned_draw_share'], 0.95)
        self.assertAlmostEqual(summary['pinned_game_share'], 600 / 800)

    def test_swap_can_arm_a_pinned_member_and_summary_counts_it(self):
        # Bank 0 beats the bootstrap; the draw (PFSP, all unplayed at 0.25 vs a
        # beaten bootstrap at the floor) lands on a pinned member.
        self.trainer.global_step = 100
        self.step(logs(4, **{'0': (4, 4)}), 0)
        bank = self.pool['banks'][0]
        self.assertIsNotNone(bank['pending_opp_path'])
        self.assertTrue(self.ledger[bank['pending_opp_path']].get('pinned'))
        self.backend.aligned[0] = 3
        self.trainer.global_step = 200
        before = self.summary()
        self.step(logs(4), 1)
        after = self.summary()
        self.assertEqual(after['pinned_banks'], 1)
        self.assertLess(after['banks'][0]['opponent_step'], 0)
        events = native_selfplay.events(before, after, self.ledger, learner_steps=0)
        self.assertEqual([e['event'] for e in events], ['loaded'])
        # Games against the pinned member are credited to it.
        self.step(logs(4, **{'0': (4, 1)}), 2)
        row = next(r for r in self.summary()['pinned'] if r['name'] == self.ledger[bank['cur_opp_path']]['name'])
        self.assertEqual((row['games'], row['wins']), (4, 1))

    def test_index_mode_keeps_upstream_weights_but_protects_pinned(self):
        native_selfplay.install_sampler(selfplay, self.pool, self.ledger, mode='index')
        self.assertEqual(self.pool['sampler'], 'index')
        self.pool['opp_timeout_steps'] = 0
        for epoch in range(8):
            self.trainer.global_step = 1000 * (epoch + 1)
            self.step(logs(1), epoch)
        self.assertEqual(sum(1 for e in self.pool['pool'] if e.get('pinned')), 3)
        self.assertEqual(sum(1 for e in self.pool['pool'] if not e.get('pinned')), 4)


if __name__ == '__main__':
    unittest.main()
