"""Native checkpoint publication and failure behavior, using real CPU loading."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from native_checkpoint import write_checkpoint
from native_policy import load


class NativeCheckpointTests(unittest.TestCase):
    def test_checkpoint_loads_with_physical_and_learner_counters(self):
        for architecture, obs in (("puffernet-mingru", 11), ("bopl-set-mingru-v1", 3257)):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                hidden = 8
                encoder = hidden * obs if obs == 11 else hidden * 905 + 64 * 49 + hidden * (hidden + 128)
                weights = np.linspace(-.02, .02, encoder + 273 * hidden + 3 * hidden**2, dtype=np.float32)
                path = Path(directory) / "policy-000000032.bin"
                kwargs = dict(obs_size=obs, hidden=hidden, num_layers=1, architecture=architecture,
                              game_steps=32, learner_steps=24, config=dict(training_maps=[0, 1, 2], episode_ticks=7200))
                result = write_checkpoint(path, weights.tofile, **kwargs)
                policy, metadata = load(path)
                self.assertEqual(result, metadata)
                self.assertEqual(metadata["step"], 32)
                self.assertEqual(metadata["learner_steps"], 24)
                for output in policy.step(torch.ones(1, obs), policy.initial(1)):
                    self.assertTrue(torch.isfinite(output).all())
                with self.assertRaises(FileExistsError):
                    write_checkpoint(path, weights.tofile, **kwargs)
                self.assertEqual(sorted(p.name for p in path.parent.iterdir()), [path.name, path.name + ".json"])

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
            count = 8 * (11 + 273) + 3 * 8**2
            for weights in (np.zeros(count - 1, np.float32), np.full(count, np.nan, np.float32)):
                with self.assertRaises(ValueError):
                    write_checkpoint(path, weights.tofile, **kwargs)
            self.assertEqual(list(path.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
