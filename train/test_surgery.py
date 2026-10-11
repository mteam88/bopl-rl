"""Checkpoint surgery: the offset arithmetic, against a real trained checkpoint.

    native/artifacts/rl/venv/bin/python -m unittest discover -s train -p 'test_surgery.py'

The identity case is the load-bearing one: surgery from a checkpoint to its own
architecture must reproduce the published blob byte for byte, because every other
case rides on the same offsets.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'train'), str(ROOT / 'eval')]
import layout
import policy_v4
import surgery
from native_checkpoint import write_checkpoint

CHECKPOINT = ROOT / 'native/artifacts/rl/sweeps/v4-full/000/checkpoints/policy-2313256960.bin'
CONFIG = dict(training_maps=[0, 1, 2], episode_ticks=7200)


def published(directory, name, weights, arch):
    """A real checkpoint on disk, written the way training publishes one."""
    path = Path(directory) / name
    write_checkpoint(path, weights.tofile, obs_size=layout.V4_OBS_SIZE, hidden=arch['hidden'],
                     num_layers=arch['num_layers'], architecture=layout.V4_ARCHITECTURE,
                     game_steps=4096, learner_steps=2048, config=dict(CONFIG),
                     aim_fourier=arch['aim_fourier'], aim_ar=arch['aim_ar'], v4_shape=surgery.v4_shape(arch))
    return path


def synthetic(directory, name, **changes):
    arch = surgery.architecture(32, 1, 4, 0, changes)
    weights = policy_v4.init_weights(arch['hidden'], arch['num_layers'], arch['aim_fourier'], arch['aim_ar'],
                                     seed=7, **surgery.v4_shape(arch))
    return published(directory, name, weights, arch), arch


def actions(report):
    return {row[0]: row[3] for row in report}


@unittest.skipUnless(CHECKPOINT.exists(), f'needs the checkpoint {CHECKPOINT}')
class RealCheckpointTests(unittest.TestCase):
    """Surgery on native/artifacts/rl/sweeps/v4-full/000/checkpoints/policy-2313256960.bin."""

    @classmethod
    def setUpClass(cls):
        cls.arch, cls.metadata, cls.weights = surgery.read_checkpoint(CHECKPOINT)
        cls.tensors = surgery.tensors(cls.arch)

    def source(self, name):
        offset, shape = self.tensors[name]
        return self.weights[offset:offset + shape[0] * shape[1]].reshape(shape)

    def grafted(self, **changes):
        target_arch = dict(self.arch, **changes)
        weights, report = surgery.surgery(self.weights, self.arch, target_arch)
        view = lambda name: (lambda o, s: weights[o:o + s[0] * s[1]].reshape(s))(*surgery.tensors(target_arch)[name])
        return target_arch, weights, report, view

    def test_identity_reproduces_the_published_blob_byte_for_byte(self):
        weights, report = surgery.surgery(self.weights, self.arch, dict(self.arch))
        self.assertEqual(set(actions(report).values()), {'copied'})
        self.assertEqual(weights.tobytes(), CHECKPOINT.read_bytes())

    def test_growing_hidden_keeps_every_concatenated_axis_in_register(self):
        """The failure this rules out: old[:rows, :cols] on an axis that is a
        concatenation, which slides one part of it into another."""
        old, new = self.arch['hidden'], 384
        target_arch, _, report, view = self.grafted(hidden=new)
        self.assertEqual(actions(report)['mingru0'], 'resized')

        source, grown = self.source('mingru0'), view('mingru0')
        for gate in range(3):   # candidate, update, highway: stacked rows, not one rectangle
            np.testing.assert_array_equal(grown[gate * new:gate * new + old, :old],
                                          source[gate * old:(gate + 1) * old, :old])
        self.assertFalse(np.array_equal(grown[:old * 3, :old], source[:, :old]))

        source, grown = self.source('critic_in'), view('critic_in')   # [trunk, privileged]
        np.testing.assert_array_equal(grown[:, :old], source[:, :old])
        np.testing.assert_array_equal(grown[:, new:new + layout.V4_PRIVILEGED],
                                      source[:, old:old + layout.V4_PRIVILEGED])

        source, grown = self.source('fusion'), view('fusion')   # [flat, four (max || mean) pools]
        np.testing.assert_array_equal(grown[:old], source)       # set_dim unchanged: whole rows carry over
        self.assertEqual(surgery.tensors(target_arch)['fusion'][1], (new, source.shape[1]))

    def test_widening_the_aim_head_preserves_the_aim_distribution(self):
        """K 4 -> 8 rescales the shared harmonics by the basis' sqrt(K) and zeroes
        the new ones, so the trained policy comes out unchanged and only then
        learns the finer harmonics. The button and hold rows move by 8 rows."""
        target_arch, weights, report, _ = self.grafted(aim_fourier=8)
        self.assertEqual(actions(report), dict(actions(report), actor='resized'))
        self.assertEqual([row[3] for row in report if row[0] != 'actor'], ['copied'] * (len(report) - 1))

        shape = surgery.v4_shape(self.arch)
        before = policy_v4.PolicyV4(self.weights, self.arch['hidden'], self.arch['num_layers'], 4, 0, **shape)
        after = policy_v4.PolicyV4(weights, target_arch['hidden'], target_arch['num_layers'], 8, 0, **shape)
        observations = torch.from_numpy(np.random.default_rng(3).normal(size=(4, layout.V4_OBS_SIZE)).astype(np.float32))
        state = before.initial(4)
        old_logits, old_value, _ = before.step(observations, state)
        new_logits, new_value, _ = after.step(observations, state)
        # Not bitwise: a wider actor is a differently shaped GEMM, and the
        # coefficients are stored rescaled in fp32 before the basis re-divides
        # them. 1e-4 on aim logits that run past 9 leaves the distribution alone.
        np.testing.assert_allclose(new_logits, old_logits, atol=1e-4)
        np.testing.assert_allclose(new_value, old_value, atol=1e-5)
        self.assertGreater(old_logits[:, :layout.ACT_AIM].abs().max().item(), 1.0)

    def test_turning_on_autoregressive_aim_gives_every_combo_the_trained_head(self):
        target_arch, weights, _, _ = self.grafted(aim_ar=1)
        shape = surgery.v4_shape(self.arch)
        after = policy_v4.PolicyV4(weights, target_arch['hidden'], target_arch['num_layers'], 4, 1, **shape)
        before = policy_v4.PolicyV4(self.weights, self.arch['hidden'], self.arch['num_layers'], 4, 0, **shape)
        observations = torch.from_numpy(np.random.default_rng(5).normal(size=(2, layout.V4_OBS_SIZE)).astype(np.float32))
        old_rows, _ = before.step_raw(observations, before.initial(2))
        new_rows, _ = after.step_raw(observations, after.initial(2))
        for combo in range(layout.ACT_BUTTONS):
            np.testing.assert_allclose(*[p.game_logits(rows, [combo] * 2)[0]
                                         for p, rows in ((after, new_rows), (before, old_rows))], atol=1e-5)


class ArchitectureChangeTests(unittest.TestCase):
    """Tensors that appear and disappear, on checkpoints written by write_checkpoint."""

    def test_added_and_removed_tensors_are_reported_and_the_rest_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            path, arch = synthetic(directory, 'no-ffn.bin', ffn_dim=0)
            _, _, weights = surgery.read_checkpoint(path)
            target_arch = dict(arch, ffn_dim=256, critic_dim=0)
            grown, report = surgery.surgery(weights, arch, target_arch)
            self.assertEqual(actions(report), {
                'player_w1': 'copied', 'player_w2': 'copied', 'slot_w1': 'copied', 'slot_w2': 'copied',
                'entity_w1': 'copied', 'entity_w2': 'copied', 'slime_w1': 'copied', 'slime_w2': 'copied',
                'embedding': 'copied', 'flat': 'copied', 'fusion': 'copied', 'mingru0': 'copied',
                'ffn_in': 'initialised', 'ffn_out': 'initialised', 'actor': 'resized',
                'critic_in': 'dropped', 'critic_b1': 'dropped', 'critic_b2': 'dropped', 'critic_out': 'dropped'})

            # critic_dim 0 appends a shared value row to the actor; the aim, button
            # and hold rows above it must not shift into it.
            old = surgery.tensors(arch)['actor']
            new = surgery.tensors(target_arch)['actor']
            self.assertEqual((old[1][0], new[1][0]), (30, 31))
            np.testing.assert_array_equal(grown[new[0]:new[0] + 30 * arch['hidden']],
                                          weights[old[0]:old[0] + 30 * arch['hidden']])

    def test_dropping_the_fourier_head_refuses_to_reuse_the_coefficients(self):
        with tempfile.TemporaryDirectory() as directory:
            path, arch = synthetic(directory, 'fourier.bin')
            _, _, weights = surgery.read_checkpoint(path)
            _, report = surgery.surgery(weights, arch, dict(arch, aim_fourier=0))
            actor = next(row for row in report if row[0] == 'actor')
            self.assertEqual(actor[3], 'resized')
            self.assertEqual(actor[4], (layout.ACT_BUTTONS + layout.ACT_HOLD) * arch['hidden'])

    def test_the_written_checkpoint_is_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source, arch = synthetic(directory, 'source.bin')
            target = Path(directory) / 'grown.bin'
            surgery.main([str(source), str(target), '--hidden', '64', '--set-inner', '64'])
            grown_arch, metadata, weights = surgery.read_checkpoint(target)   # re-reads and rechecks the sha
            self.assertEqual(grown_arch, dict(arch, hidden=64, set_inner=64))
            self.assertEqual(metadata['config']['surgery']['changed'],
                             {'hidden': [arch['hidden'], 64], 'set_inner': [arch['set_inner'], 64]})
            self.assertEqual(weights.size, layout.policy_v4_weights(
                64, 1, aim_fourier=4, aim_ar=0, **dict(surgery.v4_shape(arch), set_inner=64)))
            self.assertTrue(np.isfinite(weights).all())

    def test_widening_by_duplication_keeps_the_policy_over_recurrent_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            for changes in ({}, dict(ffn_dim=48), dict(critic_dim=0)):
                source, arch = synthetic(directory, f'source-{len(changes)}-{list(changes.values())}.bin', **changes)
                _, _, weights = surgery.read_checkpoint(source)
                grown, grown_arch, _ = surgery.widen(weights, arch, 3, noise=0.0)
                self.assertEqual(grown_arch, dict(arch, hidden=3 * arch['hidden']))
                before = policy_v4.PolicyV4(weights, **arch)
                after = policy_v4.PolicyV4(grown, **grown_arch)
                observations = torch.from_numpy(np.random.default_rng(9).normal(
                    size=(5, 3, layout.V4_OBS_SIZE)).astype(np.float32))
                old_state, new_state = before.initial(3), after.initial(3)
                for step in observations:
                    old_rows, old_state = before.step_raw(step, old_state)
                    new_rows, new_state = after.step_raw(step, new_state)
                    width = before.width     # the critic's layer norm is the one documented departure
                    torch.testing.assert_close(new_rows[:, :width], old_rows[:, :width], atol=1e-4, rtol=1e-4)
                    torch.testing.assert_close(new_state[:, :, :arch['hidden']], old_state, atol=1e-5, rtol=1e-5)

    def test_a_v3_checkpoint_is_refused_rather_than_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v3.bin'
            weights = np.zeros(layout.policy_weights(8, 1, layout.ARCHITECTURE, 4, 0), np.float32)
            write_checkpoint(path, weights.tofile, obs_size=layout.OBS_SIZE, hidden=8, num_layers=1,
                             architecture=layout.ARCHITECTURE, game_steps=8, learner_steps=4,
                             config=dict(CONFIG), aim_fourier=4)
            with self.assertRaises(ValueError):
                surgery.read_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
