"""Native checkpoint publication and failure behavior, using real CPU loading."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import layout
from native_checkpoint import write_checkpoint
from native_policy import load


class NativeCheckpointTests(unittest.TestCase):
    def test_checkpoint_loads_with_physical_and_learner_counters(self):
        for architecture, obs in (("puffernet-mingru", 11), (layout.ARCHITECTURE, layout.OBS_SIZE)):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                hidden = 8
                encoder = hidden * obs if obs == 11 else layout.set_encoder_weights(hidden)
                weights = np.linspace(-.02, .02, encoder + layout.DECODER * hidden + 3 * hidden**2, dtype=np.float32)
                path = Path(directory) / "policy-000000032.bin"
                kwargs = dict(obs_size=obs, hidden=hidden, num_layers=1, architecture=architecture,
                              game_steps=32, learner_steps=24, config=dict(training_maps=[0, 1, 2], episode_ticks=7200))
                result = write_checkpoint(path, weights.tofile, **kwargs)
                policy, metadata = load(path)
                self.assertEqual(result, metadata)
                self.assertEqual(metadata["step"], 32)
                self.assertEqual(metadata["learner_steps"], 24)
                self.assertEqual(metadata["action_sizes"], layout.ACTION_SIZES)
                logits, value, state = policy.step(torch.ones(1, obs), policy.initial(1))
                self.assertEqual(tuple(logits.shape), (1, layout.LOGITS))
                for output in (logits, value, state):
                    self.assertTrue(torch.isfinite(output).all())
                with self.assertRaises(ValueError):
                    write_checkpoint(Path(directory) / "v1.bin", weights.tofile, **dict(kwargs, architecture="bopl-set-mingru-v1"))
                with self.assertRaises(ValueError):
                    write_checkpoint(Path(directory) / "v2.bin", weights.tofile, **dict(kwargs, architecture="bopl-set-mingru-v2"))
                with self.assertRaises(FileExistsError):
                    write_checkpoint(path, weights.tofile, **kwargs)
                self.assertEqual(sorted(p.name for p in path.parent.iterdir()), [path.name, path.name + ".json"])

    def test_aim_parameterisation_is_recorded_and_sizes_the_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            hidden = 8
            for aim_fourier, aim_ar in ((0, 1), (16, 0), (16, 1)):
                with self.subTest(aim_fourier=aim_fourier, aim_ar=aim_ar):
                    count = hidden * (11 + layout.decoder_logits(aim_fourier, aim_ar) + 1) + 3 * hidden**2
                    weights = np.linspace(-.02, .02, count, dtype=np.float32)
                    path = Path(directory) / f"policy-{aim_fourier}-{aim_ar}.bin"
                    kwargs = dict(obs_size=11, hidden=hidden, num_layers=1, architecture="puffernet-mingru",
                                  game_steps=32, learner_steps=24, config=dict(training_maps=[0], episode_ticks=7200),
                                  aim_fourier=aim_fourier, aim_ar=aim_ar)
                    metadata = write_checkpoint(path, weights.tofile, **kwargs)
                    self.assertEqual((metadata["aim_fourier"], metadata["aim_ar"]), (aim_fourier, aim_ar))
                    policy, _ = load(path)
                    self.assertEqual((policy.aim_fourier, policy.aim_ar), (aim_fourier, aim_ar))
                    logits, value, state = policy.step(torch.ones(1, 11), policy.initial(1))
                    self.assertEqual(tuple(logits.shape), (1, layout.LOGITS))
                    with self.assertRaises(ValueError):  # the plain decoder does not fit these weights
                        write_checkpoint(Path(directory) / "plain.bin", weights.tofile, **dict(kwargs, aim_fourier=0, aim_ar=0))
                    with self.assertRaises(ValueError):
                        write_checkpoint(Path(directory) / "bad.bin", weights.tofile, **dict(kwargs, aim_ar=2))

    def test_failed_save_never_publishes_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.bin"
            kwargs = dict(obs_size=11, hidden=8, num_layers=1, architecture="puffernet-mingru",
                          game_steps=32, learner_steps=24, config=dict(training_maps=[0], episode_ticks=7200))
            def failed(name):
                Path(name).write_bytes(b"partial")
                raise RuntimeError("Native save failed")
            with self.assertRaises(RuntimeError):
                write_checkpoint(path, failed, **kwargs)
            count = 8 * (11 + layout.DECODER) + 3 * 8**2
            for weights in (np.zeros(count - 1, np.float32), np.full(count, np.nan, np.float32)):
                with self.assertRaises(ValueError):
                    write_checkpoint(path, weights.tofile, **kwargs)
            self.assertEqual(list(path.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
