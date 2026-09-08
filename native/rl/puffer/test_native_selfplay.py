"""CPU tests: the trainer's pool state drives pufferlib.selfplay.step as intended."""
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import native_selfplay

from pufferlib import selfplay


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


class NativeSelfplayTests(unittest.TestCase):
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
        selfplay.step(self.trainer, self.backend, self.pool, flat, epoch)
        native_selfplay.register(self.pool, self.ledger, self.trainer.global_step * 4, self.trainer.global_step * 3)

    def summary(self):
        return native_selfplay.summary(self.pool, self.ledger)

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

    def test_timeout_swap_without_snapshot(self):
        self.pool['snapshot_interval'] = 0
        self.trainer.global_step = 5000
        before = self.summary()
        self.step(logs(1), 0)
        after = self.summary()
        self.assertEqual(after['size'], 1)
        self.assertTrue(all(b['pending_step'] == 0 for b in after['banks']))
        self.assertEqual([e['event'] for e in native_selfplay.events(before, after, self.ledger, learner_steps=0)], ['armed', 'armed'])


if __name__ == '__main__':
    unittest.main()
