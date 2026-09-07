"""CPU-only checkpoint ABI, upstream-model parity, and game-state isolation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from native_policy import NativePolicy, ROOT, REVISION, load


class NativePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        source = ROOT / "native/artifacts/rl/dependencies" / f"PufferLib-{REVISION}" / "pufferlib/models.py"
        spec = importlib.util.spec_from_file_location("upstream_policy_reference", source)
        cls.upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.upstream)

    def fixture(self, obs=11, hidden=8, layers=1):
        rng = np.random.default_rng(73)
        count = hidden * (obs + 273) + layers * 3 * hidden**2
        weights = (rng.standard_normal(count) * .025).astype(np.float32)
        policy = NativePolicy(weights, obs, hidden, layers)
        return policy, weights

    def test_upstream_parity_with_caller_resets(self):
        # Compare to upstream's independent Torch MinGRU, including its highway
        # connection. Exercise real Bopl dimensions and multiple recurrent layers.
        for obs, hidden, layers in ((11, 8, 1), (3257, 128, 1), (11, 16, 2)):
            with self.subTest(obs=obs, hidden=hidden, layers=layers):
                policy, weights = self.fixture(obs, hidden, layers)
                encoder_end = hidden * obs
                decoder_end = encoder_end + 273 * hidden
                encoder = torch.from_numpy(weights[:encoder_end].reshape(hidden, obs))
                decoder = torch.from_numpy(weights[encoder_end:decoder_end].reshape(273, hidden))
                recurrent = self.upstream.MinGRU(hidden, layers).eval()
                with torch.no_grad():
                    offset = decoder_end
                    for layer in recurrent.layers:
                        size = layer.weight.numel()
                        layer.weight.copy_(torch.from_numpy(weights[offset:offset + size].reshape(3 * hidden, hidden)))
                        offset += size
                actual_state = policy.initial(3)
                reference_state = recurrent.initial_state(3, "cpu")
                generator = torch.Generator().manual_seed(19)
                for tick in range(24):
                    # Deliberately stagger resets, as independently ending games do.
                    if tick in (5, 11, 17):
                        row = (tick // 6) % 3
                        actual_state[:, row] = 0
                        reference_state[0][:, row] = 0
                    observations = torch.randn(3, obs, generator=generator)
                    preserved = actual_state.clone()
                    logits, values, actual_next = policy.step(observations, actual_state)
                    torch.testing.assert_close(actual_state, preserved, rtol=0, atol=0)
                    with torch.no_grad():
                        out, reference_state = recurrent.forward_eval(F.linear(observations, encoder), reference_state)
                        decoded = F.linear(out, decoder)
                    torch.testing.assert_close(logits, decoded[:, :272], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(values, decoded[:, 272], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(actual_next, reference_state[0], rtol=2e-5, atol=2e-6)
                    actual_state = actual_next

    def test_parallel_games_have_independent_context_and_state(self):
        policy, _ = self.fixture()
        streams = [torch.randn(15, 2, 11, generator=torch.Generator().manual_seed(seed)) for seed in (10, 20)]

        def run(stream, barrier=None):
            state = policy.initial(2)
            outputs = []
            for obs in stream:
                if barrier:
                    barrier.wait(timeout=10)
                logits, _, state = policy.step(obs, state)
                outputs.append(logits)
            return torch.stack(outputs), state

        expected = [run(stream) for stream in streams]
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(run, stream, barrier) for stream in streams]
            actual = [future.result() for future in futures]
        for a, b in zip(actual, expected):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_checkpoint_metadata_and_weight_identity(self):
        policy, weights = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.bin"
            weights.tofile(path)
            meta = dict(schema=1, backend="pufferlib-native", architecture="puffernet-mingru",
                        precision="float32", obs_size=11, hidden=8, num_layers=1,
                        action_sizes=[256, 16], step=42, weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            sidecar = Path(str(path) + ".json")
            sidecar.write_text(json.dumps(meta))
            restored, saved = load(path)
            self.assertEqual(saved["step"], 42)
            obs = torch.ones(1, 11)
            for actual, expected in zip(restored.step(obs, restored.initial(1)), policy.step(obs, policy.initial(1))):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "precision"):
                load(path, metadata=dict(meta, precision="bfloat16"))
            path.write_bytes(path.read_bytes()[:-4])
            with self.assertRaisesRegex(ValueError, "metadata hash"):
                load(path)
            meta["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "finite native weights"):
                load(path, metadata=meta)

    def test_rejects_bad_shapes_and_nonfinite_inputs(self):
        policy, weights = self.fixture()
        for obs, state in ((torch.zeros(1, 12), policy.initial(1)),
                           (torch.zeros(1, 11), policy.initial(2)),
                           (torch.full((1, 11), float("nan")), policy.initial(1))):
            with self.assertRaises(ValueError):
                policy.step(obs, state)
        with self.assertRaisesRegex(ValueError, "divisible by eight"):
            NativePolicy(weights, 11, hidden=7)


if __name__ == "__main__":
    unittest.main()
