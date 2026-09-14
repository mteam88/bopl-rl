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
from native_policy import NativePolicy, ROOT, REVISION, decide, load, masked

# Set-encoder slices of one observation row: (start, stop, rows, width, id column).
PLAYER_SLICE = (layout.OBS_PLAYER_OFFSET, layout.OBS_SLOT_OFFSET, layout.OBS_PLAYERS, layout.PLAYER_FEATURES, None)
SLOT_SLICE = (layout.OBS_SLOT_OFFSET, layout.OBS_ENTITY_OFFSET, layout.OBS_SLOTS, layout.SLOT_FEATURES,
              layout.SLOT_ABILITY_ID)
ENTITY_SLICE = (layout.OBS_ENTITY_OFFSET, layout.OBS_SLIME_OFFSET, layout.OBS_ENTITIES,
                layout.ENTITY_FEATURES, layout.ENTITY_ABILITY_ID)
SLIME_SLICE = (layout.OBS_SLIME_OFFSET, layout.OBS_PREVIOUS_OFFSET, layout.OBS_SLIMES,
               layout.SLIME_FEATURES, None)
SET_SLICES = (PLAYER_SLICE, SLOT_SLICE, ENTITY_SLICE, SLIME_SLICE)


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
        shapes = ((hidden, SET_FLAT), (64, layout.PLAYER_FEATURES),
                  (64, layout.SLOT_FEATURES - 1), (64, layout.ENTITY_FEATURES - 1),
                  (64, layout.SLIME_FEATURES), (layout.EMBEDDING_IDS, 64), (hidden, hidden + 256),
                  (DECODER, hidden), (3 * hidden, hidden))
        weights = np.concatenate([(rng.standard_normal(shape) * .025).astype(np.float32).ravel()
                                  for shape in shapes])
        policy = NativePolicy(weights, OBS_SIZE, hidden, architecture=ARCHITECTURE)
        obs = rng.standard_normal((batch, OBS_SIZE)).astype(np.float32)
        for lo, hi, rows, width, id_column in SET_SLICES:
            slots = obs[:, lo:hi].reshape(batch, rows, width)
            slots[:, :, 0] = rng.integers(2, size=(batch, rows))
            if id_column is not None:
                slots[:, :, id_column] = rng.integers(layout.EMBEDDING_IDS, size=(batch, rows))
            slots[0, :, 0] = 0
            slots[1, :, 0] = 1

        def step(value):
            return policy.step(torch.from_numpy(value.copy()), policy.initial(batch))

        expected = step(obs)
        permuted, poisoned, empty = (obs.copy() for _ in range(3))
        for lo, hi, rows, width, id_column in SET_SLICES:
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
                shapes = [(hidden, SET_FLAT), (64, layout.PLAYER_FEATURES),
                          (64, layout.SLOT_FEATURES - 1), (64, layout.ENTITY_FEATURES - 1),
                          (64, layout.SLIME_FEATURES), (layout.EMBEDDING_IDS, 64),
                          (hidden, hidden + 256), (DECODER, hidden)]
                shapes += [(3 * hidden, hidden)] * layers
                matrices = [(rng.standard_normal(shape) * .03).astype(np.float32)
                            for shape in shapes]
                weights = np.concatenate([matrix.ravel() for matrix in matrices])
                self.assertEqual(weights.size, layout.policy_weights(hidden, layers, ARCHITECTURE))
                policy = NativePolicy(weights, OBS_SIZE, hidden, layers, architecture=ARCHITECTURE)
                flat, player, slot, entity, slime, embedding, fusion, decoder = map(torch.from_numpy, matrices[:8])
                recurrent = self.upstream.MinGRU(hidden, layers).eval()
                with torch.no_grad():
                    for layer, matrix in zip(recurrent.layers, matrices[8:]):
                        layer.weight.copy_(torch.from_numpy(matrix))
                state = torch.from_numpy(rng.standard_normal((layers, batch, hidden)).astype(np.float32))
                reference = recurrent.initial_state(batch, "cpu")
                reference[0].copy_(state)

                def pooled(rows, projection, id_column=None):
                    present = rows[:, :, 0] != 0
                    safe = rows.masked_fill(~present[:, :, None], 0)
                    if id_column is None:
                        projected = F.linear(safe, projection)
                    else:
                        ids = safe[:, :, id_column]
                        valid = (ids >= 0) & (ids < layout.EMBEDDING_IDS) & (ids == ids.trunc())
                        ids = torch.where(valid, ids, 0).long()
                        projected = F.linear(safe[:, :, :id_column], projection) + embedding[ids]
                    projected = projected.relu()
                    projected = projected.masked_fill(~present[:, :, None], -float("inf"))
                    return torch.where(present.any(1, keepdim=True),
                                       projected.max(1).values, 0)

                for tick in range(20):
                    if tick in (3, 8, 15):
                        row = tick % batch
                        state[:, row] = 0
                        reference[0][:, row] = 0
                    obs = rng.standard_normal((batch, OBS_SIZE)).astype(np.float32)
                    for lo, hi, count, width, id_column in SET_SLICES:
                        rows = obs[:, lo:hi].reshape(batch, count, width)
                        rows[:, :, 0] = 0 if tick == 10 else rng.integers(2, size=(batch, count))
                        if id_column is not None:
                            rows[:, :, id_column] = rng.integers(layout.EMBEDDING_IDS, size=(batch, count))
                        rows[:, :, 1:][rows[:, :, 0] == 0] = 1e9
                    observations = torch.from_numpy(obs)
                    logits, values, state = policy.step(observations, state)
                    with torch.no_grad():
                        flat_input = torch.cat((observations[:, :layout.OBS_PLAYER_OFFSET],
                                                observations[:, layout.OBS_PREVIOUS_OFFSET:]), 1)
                        joined = torch.cat((F.linear(flat_input, flat).relu(),
                                            pooled(observations[:, PLAYER_SLICE[0]:PLAYER_SLICE[1]].reshape(batch, PLAYER_SLICE[2], PLAYER_SLICE[3]), player),
                                            pooled(observations[:, SLOT_SLICE[0]:SLOT_SLICE[1]].reshape(batch, SLOT_SLICE[2], SLOT_SLICE[3]), slot, SLOT_SLICE[4]),
                                            pooled(observations[:, ENTITY_SLICE[0]:ENTITY_SLICE[1]].reshape(batch, ENTITY_SLICE[2], ENTITY_SLICE[3]), entity, ENTITY_SLICE[4]),
                                            pooled(observations[:, SLIME_SLICE[0]:SLIME_SLICE[1]].reshape(batch, SLIME_SLICE[2], SLIME_SLICE[3]), slime)), 1)
                        encoded = F.linear(joined, fusion).relu()
                        out, reference = recurrent.forward_eval(encoded, reference)
                        decoded = F.linear(out, decoder)
                    torch.testing.assert_close(logits, decoded[:, :LOGITS], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(values, decoded[:, LOGITS], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(state, reference[0], rtol=2e-5, atol=2e-6)

    def aim_fixture(self, aim_fourier, aim_ar, obs=11, hidden=8, seed=5):
        rng = np.random.default_rng(seed)
        count = layout.policy_weights(hidden, 1, "puffernet-mingru", aim_fourier, aim_ar) - hidden * (OBS_SIZE - obs)
        weights = (rng.standard_normal(count) * .3).astype(np.float32)
        return NativePolicy(weights, obs, hidden, 1, aim_fourier=aim_fourier, aim_ar=aim_ar), weights, rng

    @staticmethod
    def fourier_reference(block, harmonics):
        """The 256 aim logits of one Fourier aim block (bopl_aim.h), in float64."""
        theta = 2 * np.pi * np.arange(255) / 255
        k = np.arange(1, harmonics + 1)[:, None]
        angles = (np.cos(k * theta) * block[1::2][:, None] + np.sin(k * theta) * block[2::2][:, None]).sum(0)
        return np.concatenate(([block[0]], angles / np.sqrt(harmonics)))

    def test_aim_parameterisations_materialise_the_game_logits(self):
        # The raw row is the decoder's output; the game logits follow bopl_aim.h
        # for every button combo, and the width/weight count follow layout.py.
        for aim_fourier, aim_ar in ((0, 1), (4, 0), (16, 1), (32, 1)):
            with self.subTest(aim_fourier=aim_fourier, aim_ar=aim_ar):
                policy, weights, rng = self.aim_fixture(aim_fourier, aim_ar)
                width = layout.decoder_logits(aim_fourier, aim_ar)
                self.assertEqual(policy.width, width)
                hidden, obs = policy.hidden, policy.obs_size
                encoder = torch.from_numpy(weights[:hidden * obs].reshape(hidden, obs))
                decoder = torch.from_numpy(weights[hidden * obs:hidden * (obs + width + 1)].reshape(width + 1, hidden))
                recurrent = self.upstream.MinGRU(hidden, 1).eval()
                with torch.no_grad():
                    recurrent.layers[0].weight.copy_(torch.from_numpy(weights[hidden * (obs + width + 1):].reshape(3 * hidden, hidden)))
                observations = torch.from_numpy(rng.standard_normal((5, obs)).astype(np.float32))
                rows, state = policy.step_raw(observations, policy.initial(5))
                with torch.no_grad():
                    out, _ = recurrent.forward_eval(F.linear(observations, encoder), recurrent.initial_state(5, "cpu"))
                    expected_rows = F.linear(out, decoder)
                torch.testing.assert_close(rows, expected_rows, rtol=2e-5, atol=2e-6)
                self.assertEqual(tuple(rows.shape), (5, width + 1))
                combos = np.array([0, 3, 15, 7, 8], dtype=np.int32) if aim_ar else np.zeros(5, dtype=np.int32)
                logits, values = policy.game_logits(rows, combos)
                torch.testing.assert_close(values, rows[:, width], rtol=0, atol=0)
                block_width = layout.aim_block_width(aim_fourier)
                buttons_offset = layout.decoder_buttons_offset(aim_fourier, aim_ar)
                row64 = rows.double().numpy()
                for i, combo in enumerate(combos):
                    start = layout.aim_block_offset(aim_fourier, aim_ar, int(combo))
                    block = row64[i, start:start + block_width]
                    aim = self.fourier_reference(block, aim_fourier) if aim_fourier else block
                    np.testing.assert_allclose(logits[i, :layout.ACT_AIM].numpy(), aim, rtol=1e-4, atol=1e-5)
                    np.testing.assert_array_equal(logits[i, layout.ACT_AIM:].numpy(),
                                                  row64[i, buttons_offset:buttons_offset + 21].astype(np.float32))
                # step() is combo 0; other combos differ only in the aim head when autoregressive.
                step_logits, step_values, _ = policy.step(observations, policy.initial(5))
                torch.testing.assert_close(step_logits, policy.game_logits(rows)[0], rtol=0, atol=0)
                if aim_ar:
                    self.assertFalse(torch.equal(step_logits[1, :layout.ACT_AIM], logits[1, :layout.ACT_AIM]))
                    torch.testing.assert_close(step_logits[:, layout.ACT_AIM:], logits[:, layout.ACT_AIM:], rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    policy.game_logits(rows, np.array([0, 0, 0, 0, 16], dtype=np.int32))

    def test_decide_samples_buttons_before_aim(self):
        # With aim_ar the aim head comes from the sampled combo's block: feed a
        # policy whose combos have wildly different aim blocks and check that
        # the aim draws follow the combo (and the mask), reproducibly.
        policy, weights, rng = self.aim_fixture(0, 1, seed=8)
        hidden, obs, width = policy.hidden, policy.obs_size, policy.width
        editable = weights.copy()
        decoder = editable[hidden * obs:hidden * (obs + width + 1)].reshape(width + 1, hidden)
        decoder[:layout.ACT_BUTTONS * layout.ACT_AIM] = 0
        for combo in range(layout.ACT_BUTTONS):  # combo c aims only at bin 10 * c + 1
            decoder[combo * layout.ACT_AIM + 10 * combo + 1] = 40
        policy = NativePolicy(editable, obs, hidden, 1, aim_ar=1)
        observations = torch.from_numpy(rng.standard_normal((64, obs)).astype(np.float32))
        mask = np.ones((64, LOGITS), dtype=np.uint8)
        mask[:32, layout.ACT_AIM:layout.ACT_AIM + layout.ACT_BUTTONS] = 0
        mask[:32, layout.ACT_AIM + 5] = 1  # rows 0..31 are latched to combo 5
        generator = torch.Generator().manual_seed(3)
        logits, values, actions, state = decide(policy, observations, mask, policy.initial(64), generator)
        self.assertEqual(tuple(actions.shape), (64, layout.NUM_ATNS))
        self.assertTrue((actions[:32, 1] == 5).all())
        self.assertTrue(torch.isneginf(logits[:32, layout.ACT_AIM]).all())
        # Where the sampled combo's block is peaked (the peak's sign follows the hidden state), aim follows it.
        rows, _ = policy.step_raw(observations, policy.initial(64))
        peaked = 0
        for i in range(64):
            block = rows[i, actions[i, 1] * layout.ACT_AIM:(actions[i, 1] + 1) * layout.ACT_AIM]
            if block.max() > 20:
                self.assertEqual(int(actions[i, 0]), 10 * int(actions[i, 1]) + 1)
                peaked += 1
        self.assertGreater(peaked, 8)
        # The returned logits are the sampled combo's, masked.
        expected, expected_values = policy.game_logits(rows, actions[:, 1])
        torch.testing.assert_close(logits, masked(expected, mask), rtol=0, atol=0)
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
        again = decide(policy, observations, mask, policy.initial(64), torch.Generator().manual_seed(3))
        torch.testing.assert_close(again[2], actions, rtol=0, atol=0)
        # Without aim_ar, decide is step + per-head sampling from the same generator stream.
        plain, _ = self.fixture()
        plain_mask = np.ones((3, LOGITS), dtype=np.uint8)
        plain_obs = torch.randn(3, 11, generator=torch.Generator().manual_seed(1))
        got = decide(plain, plain_obs, plain_mask, plain.initial(3), torch.Generator().manual_seed(9))
        reference_logits, _, _ = plain.step(plain_obs, plain.initial(3))
        torch.testing.assert_close(got[0], reference_logits, rtol=0, atol=0)
        self.assertTrue(((got[2] >= 0) & (got[2] < torch.tensor(ACTION_SIZES))).all())

    def test_aim_metadata_selects_the_parameterisation(self):
        policy, weights, _ = self.aim_fixture(8, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.bin"
            weights.tofile(path)
            meta = dict(schema=1, backend="pufferlib-native", architecture="puffernet-mingru",
                        precision="float32", obs_size=11, hidden=8, num_layers=1, aim_fourier=8, aim_ar=1,
                        action_sizes=list(ACTION_SIZES), step=1, weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            restored, _ = load(path, metadata=meta)
            self.assertEqual((restored.aim_fourier, restored.aim_ar, restored.width), (8, 1, layout.decoder_logits(8, 1)))
            obs = torch.ones(2, 11)
            for actual, expected in zip(restored.step(obs, restored.initial(2)), policy.step(obs, policy.initial(2))):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "finite native weights"):  # the row width no longer fits
                load(path, metadata=dict(meta, aim_ar=0))
            with self.assertRaisesRegex(ValueError, "aim"):
                load(path, metadata=dict(meta, aim_fourier=65))


if __name__ == "__main__":
    unittest.main()
