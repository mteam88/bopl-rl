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

import layout
from layout import ACTION_SIZES, ARCHITECTURE, DECODER, LOGITS, OBS_SIZE, SET_FLAT
from native_policy import NativePolicy, ROOT, REVISION, load

# Set-encoder slices of one observation row: (start, stop, rows, width).
ENTITY_SLICE = (layout.OBS_ENTITY_OFFSET, layout.OBS_SLIME_OFFSET, layout.OBS_ENTITIES, layout.ENTITY_FEATURES)
SLIME_SLICE = (layout.OBS_SLIME_OFFSET, layout.OBS_PREVIOUS_OFFSET, layout.OBS_SLIMES, layout.SLIME_FEATURES)
SET_SLICES = (ENTITY_SLICE, SLIME_SLICE)


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
        count = hidden * (obs + DECODER) + layers * 3 * hidden**2
        weights = (rng.standard_normal(count) * .025).astype(np.float32)
        policy = NativePolicy(weights, obs, hidden, layers)
        return policy, weights

    def test_upstream_parity_with_caller_resets(self):
        # Compare to upstream's independent Torch MinGRU, including its highway
        # connection. Exercise real Bopl dimensions and multiple recurrent layers.
        for obs, hidden, layers in ((11, 8, 1), (OBS_SIZE, 128, 1), (11, 16, 2)):
            with self.subTest(obs=obs, hidden=hidden, layers=layers):
                policy, weights = self.fixture(obs, hidden, layers)
                encoder_end = hidden * obs
                decoder_end = encoder_end + DECODER * hidden
                encoder = torch.from_numpy(weights[:encoder_end].reshape(hidden, obs))
                decoder = torch.from_numpy(weights[encoder_end:decoder_end].reshape(DECODER, hidden))
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
                    torch.testing.assert_close(logits, decoded[:, :LOGITS], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(values, decoded[:, LOGITS], rtol=2e-5, atol=2e-6)
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
                        action_sizes=list(ACTION_SIZES), step=42, weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            sidecar = Path(str(path) + ".json")
            sidecar.write_text(json.dumps(meta))
            restored, saved = load(path)
            self.assertEqual(saved["step"], 42)
            obs = torch.ones(1, 11)
            for actual, expected in zip(restored.step(obs, restored.initial(1)), policy.step(obs, policy.initial(1))):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "precision"):
                load(path, metadata=dict(meta, precision="bfloat16"))
            with self.assertRaisesRegex(ValueError, "action_sizes"):
                load(path, metadata=dict(meta, action_sizes=[256, 16]))  # layout v1 checkpoints are refused
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

    def test_set_policy_is_invariant_to_slot_order_and_absent_features(self):
        # These are properties of the game's set observation contract; the
        # independent full encoder oracle is kept outside the shipped code.
        hidden, batch = 16, 4
        rng = np.random.default_rng(91)
        shapes = ((hidden, SET_FLAT), (64, layout.ENTITY_FEATURES), (64, layout.SLIME_FEATURES), (hidden, hidden + 128),
                  (DECODER, hidden), (3 * hidden, hidden))
        weights = np.concatenate([(rng.standard_normal(shape) * .025).astype(np.float32).ravel()
                                  for shape in shapes])
        policy = NativePolicy(weights, OBS_SIZE, hidden, architecture=ARCHITECTURE)
        obs = rng.standard_normal((batch, OBS_SIZE)).astype(np.float32)
        for lo, hi, rows, width in SET_SLICES:
            slots = obs[:, lo:hi].reshape(batch, rows, width)
            slots[:, :, 0] = rng.integers(2, size=(batch, rows))
            slots[0, :, 0] = 0
            slots[1, :, 0] = 1

        def step(value):
            return policy.step(torch.from_numpy(value.copy()), policy.initial(batch))

        expected = step(obs)
        permuted, poisoned, empty = (obs.copy() for _ in range(3))
        for lo, hi, rows, width in SET_SLICES:
            slots = obs[:, lo:hi].reshape(batch, rows, width)
            for row in range(batch):
                permuted[row, lo:hi] = slots[row, rng.permutation(rows)].ravel()
            poison = poisoned[:, lo:hi].reshape(batch, rows, width)
            poison[:, :, 1:][slots[:, :, 0] == 0] = 1e9
            empty[:, lo:hi] = 0
        for variant in (permuted, poisoned):
            for actual, reference in zip(step(variant), expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        # Reusing activations across populated and empty sets must not retain
        # an entity's previous pooled output.
        fresh = NativePolicy(weights, OBS_SIZE, hidden, architecture=ARCHITECTURE)
        empty_result = fresh.step(torch.from_numpy(empty.copy()), fresh.initial(batch))
        step(obs)
        for actual, reference in zip(step(empty), empty_result):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_set_policy_recurrent_outputs_match_independent_reference(self):
        # Qualify the complete checkpoint layout, including SET -> MinGRU ->
        # decoder, nonzero carried state, staggered resets and empty sets.
        for hidden, layers in ((16, 2), (128, 1)):
            with self.subTest(hidden=hidden, layers=layers):
                batch = 3
                rng = np.random.default_rng(193)
                shapes = [(hidden, SET_FLAT), (64, layout.ENTITY_FEATURES), (64, layout.SLIME_FEATURES),
                          (hidden, hidden + 128), (DECODER, hidden)]
                shapes += [(3 * hidden, hidden)] * layers
                matrices = [(rng.standard_normal(shape) * .03).astype(np.float32)
                            for shape in shapes]
                weights = np.concatenate([matrix.ravel() for matrix in matrices])
                self.assertEqual(weights.size, layout.policy_weights(hidden, layers, ARCHITECTURE))
                policy = NativePolicy(weights, OBS_SIZE, hidden, layers, architecture=ARCHITECTURE)
                flat, entity, slime, fusion, decoder = map(torch.from_numpy, matrices[:5])
                recurrent = self.upstream.MinGRU(hidden, layers).eval()
                with torch.no_grad():
                    for layer, matrix in zip(recurrent.layers, matrices[5:]):
                        layer.weight.copy_(torch.from_numpy(matrix))
                state = torch.from_numpy(rng.standard_normal((layers, batch, hidden)).astype(np.float32))
                reference = recurrent.initial_state(batch, "cpu")
                reference[0].copy_(state)

                def pooled(rows, projection):
                    present = rows[:, :, 0] != 0
                    safe = rows.masked_fill(~present[:, :, None], 0)
                    projected = F.linear(safe, projection).relu()
                    projected = projected.masked_fill(~present[:, :, None], -float("inf"))
                    return torch.where(present.any(1, keepdim=True),
                                       projected.max(1).values, 0)

                for tick in range(20):
                    if tick in (3, 8, 15):
                        row = tick % batch
                        state[:, row] = 0
                        reference[0][:, row] = 0
                    obs = rng.standard_normal((batch, OBS_SIZE)).astype(np.float32)
                    for lo, hi, count, width in SET_SLICES:
                        rows = obs[:, lo:hi].reshape(batch, count, width)
                        rows[:, :, 0] = 0 if tick == 10 else rng.integers(2, size=(batch, count))
                        rows[:, :, 1:][rows[:, :, 0] == 0] = 1e9
                    observations = torch.from_numpy(obs)
                    logits, values, state = policy.step(observations, state)
                    with torch.no_grad():
                        (e_lo, e_hi, e_rows, e_width), (s_lo, s_hi, s_rows, s_width) = SET_SLICES
                        flat_input = torch.cat((observations[:, :e_lo], observations[:, s_hi:]), 1)
                        joined = torch.cat((F.linear(flat_input, flat).relu(),
                                            pooled(observations[:, e_lo:e_hi].reshape(batch, e_rows, e_width), entity),
                                            pooled(observations[:, s_lo:s_hi].reshape(batch, s_rows, s_width), slime)), 1)
                        encoded = F.linear(joined, fusion).relu()
                        out, reference = recurrent.forward_eval(encoded, reference)
                        decoded = F.linear(out, decoder)
                    torch.testing.assert_close(logits, decoded[:, :LOGITS], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(values, decoded[:, LOGITS], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(state, reference[0], rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
