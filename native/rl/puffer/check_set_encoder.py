"""Qualify layout-v3 CUDA set forward and winner-only weight gradients.

The Torch implementation is an independent reference. The crowded rows come
from check_observations.c, so this also runs the fused encoder on the exact
synthetic state used to qualify controlling bodies, truncation and visibility.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "native/rl/research"))
import layout

OBS = layout.OBS_SIZE
FLAT = layout.SET_FLAT
E = layout.SET_DIM
REVISION = "42f70d6932c30ac977736f861006809c50168ba9"
SET_SPECS = (
    ("player", layout.OBS_PLAYER_OFFSET, layout.OBS_SLOT_OFFSET,
     layout.OBS_PLAYERS, layout.PLAYER_FEATURES, layout.SET_PLAYER_PROJECTION, None),
    ("slot", layout.OBS_SLOT_OFFSET, layout.OBS_ENTITY_OFFSET,
     layout.OBS_SLOTS, layout.SLOT_FEATURES, layout.SET_SLOT_PROJECTION, layout.SLOT_ABILITY_ID),
    ("entity", layout.OBS_ENTITY_OFFSET, layout.OBS_SLIME_OFFSET,
     layout.OBS_ENTITIES, layout.ENTITY_FEATURES, layout.SET_ENTITY_PROJECTION, layout.ENTITY_ABILITY_ID),
    ("slime", layout.OBS_SLIME_OFFSET, layout.OBS_PREVIOUS_OFFSET,
     layout.OBS_SLIMES, layout.SLIME_FEATURES, layout.SET_SLIME_PROJECTION, None),
)


def weight_sizes(hidden):
    return [hidden * FLAT,
            E * layout.SET_PLAYER_PROJECTION,
            E * layout.SET_SLOT_PROJECTION,
            E * layout.SET_ENTITY_PROJECTION,
            E * layout.SET_SLIME_PROJECTION,
            layout.EMBEDDING_IDS * E,
            hidden * (hidden + 4 * E)]


def weight_count(hidden):
    count = sum(weight_sizes(hidden))
    assert count == layout.set_encoder_weights(hidden)
    return count


def build(output, source, compiler):
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    cuda = nvidia / "cu13"
    obj = output.with_suffix(".o")
    overlay = ROOT / "native/artifacts/rl/puffer-overlay"
    if not (overlay / "kernels.cu").exists():
        raise SystemExit(f"{overlay} missing: run build.py --device cuda first")
    subprocess.run([str(cuda / "bin/nvcc"), "-O3", "-std=c++17", "-arch=sm_120",
                    "-ccbin", compiler, "-Xcompiler=-fPIC", "-DPRECISION_FLOAT",
                    *os.environ.get("BS_EXTRA", "").split(), "-I" + str(overlay),
                    "-I" + str(source / "src"), "-I" + str(cuda / "include"),
                    "-c", str(HERE / "check_set_encoder.cu"), "-o", str(obj)], check=True)
    subprocess.run([compiler, "-shared", "-Wl,--no-undefined", str(obj),
                    str(cuda / "lib/libcudart.so.13"), str(cuda / "lib/libcublas.so.13"),
                    str(cuda / "lib/libcublasLt.so.13"), str(cuda / "lib/libcurand.so.10"),
                    "-lcuda", "-Wl,--disable-new-dtags,-rpath," + str(cuda / "lib"),
                    "-o", str(output)], check=True)


def unpack_weights(weights, hidden, dtype):
    tensor = torch.tensor(weights, dtype=dtype, requires_grad=True)
    parts = tensor.split(weight_sizes(hidden))
    shapes = ((hidden, FLAT), (E, layout.SET_PLAYER_PROJECTION),
              (E, layout.SET_SLOT_PROJECTION), (E, layout.SET_ENTITY_PROJECTION),
              (E, layout.SET_SLIME_PROJECTION), (layout.EMBEDDING_IDS, E),
              (hidden, hidden + 4 * E))
    return tensor, [part.reshape(shape) for part, shape in zip(parts, shapes)]


def oracle(observations, weights, hidden, incoming, dtype=torch.float32):
    """Independent max-pool reference with exact integer ability-id gathers."""
    all_weights, (flat, player, slot, entity, slime, embedding, fusion) = unpack_weights(weights, hidden, dtype)
    obs = torch.from_numpy(np.ascontiguousarray(observations)).to(dtype)
    flat_input = torch.cat((obs[:, :layout.OBS_PLAYER_OFFSET],
                            obs[:, layout.OBS_PREVIOUS_OFFSET:]), dim=1)

    def pool(spec, projection):
        _, lo, hi, count, width, projected_width, id_column = spec
        rows = obs[:, lo:hi].reshape(-1, count, width)
        present = rows[:, :, 0] != 0
        safe = rows.masked_fill(~present[:, :, None], 0)
        values = safe[:, :, :projected_width] @ projection.T
        if id_column is not None:
            ids = safe[:, :, id_column]
            valid = (ids >= 0) & (ids < layout.EMBEDDING_IDS) & (ids == ids.trunc())
            values = values + embedding[torch.where(valid, ids, 0).long()]
        values = values.relu().masked_fill(~present[:, :, None], -float("inf"))
        return torch.where(present.any(1, keepdim=True), values.max(1).values, 0)

    joined = torch.cat(((flat_input @ flat.T).relu(),
                        pool(SET_SPECS[0], player), pool(SET_SPECS[1], slot),
                        pool(SET_SPECS[2], entity), pool(SET_SPECS[3], slime)), dim=1)
    output = (joined @ fusion.T).relu()
    (output * torch.from_numpy(incoming).to(dtype)).sum().backward()
    return output.detach().float().numpy(), all_weights.grad.float().numpy()


class Check:
    def __init__(self, library, batch, hidden):
        self.lib = library
        self.batch = batch
        self.hidden = hidden
        self.context = library.bopl_set_check_create(batch, hidden)
        if not self.context:
            raise RuntimeError(library.bopl_set_check_error().decode())
        assert library.bopl_set_check_weight_count(self.context) == weight_count(hidden)

    def run(self, weights, observations, incoming, graph=False):
        output = np.empty((self.batch, self.hidden), np.float32)
        gradients = np.empty_like(weights)
        args = [np.ascontiguousarray(value, dtype=np.float32)
                for value in (weights, observations, incoming)]
        if self.lib.bopl_set_check_run(self.context, *args, output, gradients, int(graph)):
            raise RuntimeError(self.lib.bopl_set_check_error().decode())
        assert np.isfinite(output).all() and np.isfinite(gradients).all()
        return output, gradients

    def gather(self, ticks, total, indices):
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        if self.lib.bopl_set_check_gather(self.context, ticks, total, indices):
            raise RuntimeError(self.lib.bopl_set_check_error().decode())

    def time(self, iterations, forward_only=False):
        return self.lib.bopl_set_check_time(self.context, iterations, int(forward_only))

    def close(self):
        if self.context:
            self.lib.bopl_set_check_close(self.context)
            self.context = None


def random_obs(rng, batch, packed=False):
    obs = rng.normal(size=(batch, OBS)).astype(np.float32)
    for spec in SET_SPECS:
        _, lo, hi, count, width, _, id_column = spec
        rows = obs[:, lo:hi].reshape(batch, count, width)
        if packed:
            counts = rng.integers(0, count + 1, size=batch)
            mask = np.arange(count)[None, :] < counts[:, None]
        else:
            mask = rng.random((batch, count)) < 0.35
        rows[:, :, 0] = mask
        rows[~mask] = 0
        if id_column is not None:
            rows[:, :, id_column] = rng.integers(layout.EMBEDDING_IDS, size=(batch, count))
            rows[:, :, id_column][~mask] = 0
    return obs


def crowded_fixture(output):
    executable = output / "check_observations"
    data = output / "crowded-observations.f32"
    subprocess.run(["cc", "-O2", "-std=gnu11", "-I" + str(ROOT / "native/rl"),
                    "-I" + str(HERE), str(ROOT / "native/rl/research/check_observations.c"),
                    "-lm", "-o", str(executable)], check=True)
    subprocess.run([str(executable), "--dump", str(data)], check=True)
    rows = np.fromfile(data, np.float32)
    assert rows.size == 2 * OBS
    return rows.reshape(2, OBS)


def compare_case(library, observations, hidden, rng, graph=True):
    batch = len(observations)
    weights = (rng.normal(size=weight_count(hidden)) * .03).astype(np.float32)
    incoming = rng.normal(size=(batch, hidden)).astype(np.float32)
    expected, expected_grad = oracle(observations, weights, hidden, incoming)
    check = Check(library, batch, hidden)
    try:
        actual, gradient = check.run(weights, observations, incoming)
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-5)
        np.testing.assert_allclose(gradient, expected_grad, rtol=4e-4, atol=7e-5)
        if graph:
            replay, replay_grad = check.run(weights, observations, incoming, True)
            np.testing.assert_array_equal(replay, actual)
            np.testing.assert_array_equal(replay_grad, gradient)
        return dict(batch=batch, hidden=hidden, weights=len(weights),
                    forward_max_abs=float(np.abs(actual - expected).max()),
                    gradient_max_abs=float(np.abs(gradient - expected_grad).max()),
                    graph_replay=graph)
    finally:
        check.close()


def packed_and_permutation(library, rng):
    batch, hidden = 31, 32
    obs = random_obs(rng, batch, packed=True)
    weights = (rng.normal(size=weight_count(hidden)) * .03).astype(np.float32)
    incoming = rng.normal(size=(batch, hidden)).astype(np.float32)
    check = Check(library, batch, hidden)
    try:
        library.bopl_set_check_set_packed(0)
        every, every_grad = check.run(weights, obs, incoming)
        library.bopl_set_check_set_packed(1)
        packed, packed_grad = check.run(weights, obs, incoming)
        np.testing.assert_array_equal(packed, every)
        np.testing.assert_array_equal(packed_grad, every_grad)
        permuted = obs.copy()
        for _, lo, hi, count, width, _, _ in SET_SPECS:
            rows = permuted[:, lo:hi].reshape(batch, count, width)
            for row in range(batch):
                rows[row] = rows[row, rng.permutation(count)]
        library.bopl_set_check_set_packed(0)
        changed, changed_grad = check.run(weights, permuted, incoming)
        np.testing.assert_allclose(changed, every, rtol=0, atol=1e-6)
        np.testing.assert_allclose(changed_grad, every_grad, rtol=0, atol=2e-5)
        return dict(batch=batch, hidden=hidden, packed_bit_identical=True,
                    permutation_forward_max_abs=float(np.abs(changed - every).max()),
                    permutation_gradient_max_abs=float(np.abs(changed_grad - every_grad).max()))
    finally:
        library.bopl_set_check_set_packed(0)
        check.close()


def gather_case(library, rng):
    ticks, total, segments, hidden = 8, 5, 3, 16
    obs_tb = random_obs(rng, ticks * total).reshape(ticks, total, OBS)
    indices = np.array([3, 1, 3], np.int32)
    contiguous = np.ascontiguousarray(obs_tb[:, indices].transpose(1, 0, 2).reshape(ticks * segments, OBS))
    weights = (rng.normal(size=weight_count(hidden)) * .03).astype(np.float32)
    incoming = rng.normal(size=(ticks * segments, hidden)).astype(np.float32)
    check = Check(library, ticks * segments, hidden)
    try:
        direct, direct_grad = check.run(weights, contiguous, incoming)
        check.gather(ticks, total, indices)
        gathered, gathered_grad = check.run(weights, obs_tb, incoming)
        np.testing.assert_array_equal(gathered, direct)
        np.testing.assert_array_equal(gathered_grad, direct_grad)
        expected, expected_grad = oracle(contiguous, weights, hidden, incoming)
        np.testing.assert_allclose(gathered, expected, rtol=2e-4, atol=1e-5)
        np.testing.assert_allclose(gathered_grad, expected_grad, rtol=4e-4, atol=7e-5)
        return dict(ticks=ticks, total=total, segments=segments, duplicate_segment=True,
                    forward_max_abs=float(np.abs(gathered - expected).max()),
                    gradient_max_abs=float(np.abs(gathered_grad - expected_grad).max()))
    finally:
        check.close()


def finite_differences(library, rng):
    batch, hidden = 2, 8
    obs = random_obs(rng, batch, packed=True)
    weights = rng.uniform(.01, .04, size=weight_count(hidden)).astype(np.float32)
    incoming = rng.uniform(-1, 1, size=(batch, hidden)).astype(np.float32)
    check = Check(library, batch, hidden)
    rows = []
    try:
        _, gradient = check.run(weights, obs, incoming)
        offset = 0
        names = ("flat", "player", "slot", "entity", "slime", "embedding", "fusion")
        for name, size in zip(names, weight_sizes(hidden)):
            count = min(6, size)
            indices = offset + rng.choice(size, size=count, replace=False)
            numerical = []
            for index in indices:
                plus, minus = weights.copy(), weights.copy()
                plus[index] += np.float32(.001)
                minus[index] -= np.float32(.001)
                high, _ = check.run(plus, obs, incoming)
                low, _ = check.run(minus, obs, incoming)
                numerical.append(np.sum((high.astype(np.float64) - low) * incoming)
                                 / (float(plus[index]) - float(minus[index])))
            numerical = np.asarray(numerical)
            np.testing.assert_allclose(gradient[indices], numerical, rtol=.006, atol=.0005,
                                       err_msg=f"{name} native finite difference")
            rows.append(dict(block=name, coordinates=count,
                             max_absolute_error=float(np.abs(gradient[indices] - numerical).max())))
            offset += size
        return rows
    finally:
        check.close()


def timing(library, rng, batch=24576, hidden=128, ticks=128, total=448):
    segments = batch // ticks
    obs_tb = random_obs(rng, ticks * total, packed=True).reshape(ticks, total, OBS)
    indices = rng.choice(total, size=segments, replace=False).astype(np.int32)
    weights = (rng.normal(size=weight_count(hidden)) * .03).astype(np.float32)
    incoming = rng.normal(size=(batch, hidden)).astype(np.float32)
    check = Check(library, batch, hidden)
    try:
        check.gather(ticks, total, indices)
        check.run(weights, obs_tb, incoming)
        return dict(batch=batch, hidden=hidden, forward_ms=check.time(10, True),
                    forward_backward_ms=check.time(10))
    finally:
        check.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "native/artifacts/rl/native-migration-audit/cuda-set-v3")
    parser.add_argument("--puffer", type=Path,
                        default=ROOT / "native/artifacts/rl/dependencies" / f"PufferLib-{REVISION}")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--cuda-host-compiler", default="g++-15")
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    library_path = args.output / "check_set_encoder.so"
    if args.build or args.build_only:
        build(library_path, args.puffer, args.cuda_host_compiler)
    if args.build_only:
        print(library_path)
        return
    os.environ.setdefault("BOPL_TF32", "0")
    library = ctypes.CDLL(str(library_path))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    library.bopl_set_check_create.argtypes = [ctypes.c_int, ctypes.c_int]
    library.bopl_set_check_create.restype = ctypes.c_void_p
    library.bopl_set_check_error.restype = ctypes.c_char_p
    library.bopl_set_check_weight_count.argtypes = [ctypes.c_void_p]
    library.bopl_set_check_weight_count.restype = ctypes.c_long
    library.bopl_set_check_run.argtypes = [ctypes.c_void_p, *([array] * 5), ctypes.c_int]
    library.bopl_set_check_run.restype = ctypes.c_int
    library.bopl_set_check_close.argtypes = [ctypes.c_void_p]
    library.bopl_set_check_gather.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                              np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")]
    library.bopl_set_check_gather.restype = ctypes.c_int
    library.bopl_set_check_time.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.bopl_set_check_time.restype = ctypes.c_double
    library.bopl_set_check_set_packed.argtypes = [ctypes.c_int]
    library.bopl_set_check_set_mma.argtypes = [ctypes.c_int]
    library.bopl_set_check_set_flat_bf16.argtypes = [ctypes.c_int]
    library.bopl_set_check_set_packed(0)
    torch.set_num_threads(1)
    rng = np.random.default_rng(73)
    cases = [compare_case(library, random_obs(rng, batch), hidden, rng, graph=False)
             for batch, hidden in ((1, 8), (3, 16), (7, 128), (257, 32))]
    crowded = compare_case(library, crowded_fixture(args.output), 32, rng, graph=False)
    report = dict(schema=6, pass_all=True, random=cases, crowded=crowded,
                  packed=packed_and_permutation(library, rng), gather=gather_case(library, rng),
                  finite_differences=finite_differences(library, rng),
                  graph=compare_case(library, random_obs(rng, 7), 32, rng, graph=True))
    if args.time:
        report["timing"] = timing(library, rng)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
