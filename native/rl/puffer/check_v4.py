"""Qualify the policy v4 CUDA encoder + decoder (bopl_v4.cu) against the torch
reference (research/policy_v4.py): forward rows, weight gradients, the row
budget's deterministic overflow, gather mode and CUDA-graph replay.

The set path runs in bf16 on tensor cores (fp32 accumulation), so the
comparison is relative to the reference's scale rather than bit-exact; the
flat, fusion, FFN and critic GEMMs are fp32 (BOPL_TF32=0 here).
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
import layout  # noqa: E402
import policy_v4  # noqa: E402

REVISION = "42f70d6932c30ac977736f861006809c50168ba9"
OD = layout.decoder_logits(4, 0)
SHAPE = dict(layout.V4_DEFAULTS)
DEFAULT_BUDGET = (6, 12, 20, 8)   # bopl_v4.cu DEFAULT_BUDGET: packed rows per sample per set


def budget_in_effect():
    value = os.environ.get("BOPL4_ROW_BUDGET")
    return tuple(int(v) for v in value.split(",")) if value else DEFAULT_BUDGET


def build(output, source, compiler):
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    cuda = nvidia / "cu13"
    obj = output.with_suffix(".o")
    overlay = ROOT / "native/artifacts/rl/puffer-overlay"
    if not (overlay / "kernels.cu").exists():
        raise SystemExit(f"{overlay} missing: run build.py --device cuda first")
    subprocess.run([str(cuda / "bin/nvcc"), "-O3", "-std=c++17", "-arch=sm_120",
                    "-ccbin", compiler, "-Xcompiler=-fPIC", "-DPRECISION_FLOAT",
                    "-I" + str(overlay), "-I" + str(source / "src"), "-I" + str(cuda / "include"), "-I" + str(HERE),
                    "-c", str(HERE / "check_v4.cu"), "-o", str(obj)], check=True)
    subprocess.run([compiler, "-shared", "-Wl,--no-undefined", str(obj),
                    str(cuda / "lib/libcudart.so.13"), str(cuda / "lib/libcublas.so.13"),
                    str(cuda / "lib/libcublasLt.so.13"), str(cuda / "lib/libcurand.so.10"),
                    "-lcuda", "-Wl,--disable-new-dtags,-rpath," + str(cuda / "lib"),
                    "-o", str(output)], check=True)


# ---- the two flat layouts: the trainer allocator (16-byte aligned, no MinGRU) and the checkpoint (8-float aligned) ----
def trainer_tensors(hidden):
    return [(n, s) for n, s in layout.policy_v4_tensors(hidden, 1, 4, 0, **SHAPE) if not n.startswith("mingru")]


def trainer_offsets(hidden):
    offsets, total = {}, 0
    for name, (rows, cols) in trainer_tensors(hidden):
        total = (total + 3) & ~3
        offsets[name] = (total, rows, cols)
        total += rows * cols
    return offsets, total


def to_checkpoint(flat_trainer, hidden):
    """Trainer-order buffer -> research/policy_v4.py buffer (with a zero MinGRU)."""
    offsets, _ = trainer_offsets(hidden)
    out = np.zeros(layout.policy_v4_weights(hidden, 1, aim_fourier=4, aim_ar=0, **SHAPE), np.float32)
    pos = 0
    for name, (rows, cols) in layout.policy_v4_tensors(hidden, 1, 4, 0, **SHAPE):
        if name in offsets:
            o, r, c = offsets[name]
            out[pos:pos + rows * cols] = flat_trainer[o:o + r * c]
        pos += layout.aligned(rows * cols)
    return out


def from_checkpoint_grad(grad_ckpt, hidden):
    offsets, total = trainer_offsets(hidden)
    out = np.zeros(total, np.float32)
    pos = 0
    for name, (rows, cols) in layout.policy_v4_tensors(hidden, 1, 4, 0, **SHAPE):
        if name in offsets:
            o, r, c = offsets[name]
            out[o:o + r * c] = grad_ckpt[pos:pos + rows * cols]
        pos += layout.aligned(rows * cols)
    return out


def oracle(obs, flat_trainer, hidden, grad_logits, grad_value, budget=None):
    """(encoder out, decoder rows [logits, value], trainer-order weight gradient).
    `budget`: per-set rows per sample the packed buffers hold; rows past the
    capacity (in sample order) are dropped exactly as the CUDA path drops them."""
    obs = np.array(obs, np.float32, copy=True)
    budget = budget or budget_in_effect()
    if budget is not None:
        for s, (name, offset, count, width, _) in enumerate(layout.V4_SETS):
            cap = min(len(obs) * budget[s], len(obs) * count)
            cap = (cap + 63) // 64 * 64
            rows = obs[:, offset:offset + count * width].reshape(len(obs), count, width).copy()
            valid = rows[:, :, 0] != 0
            rank = np.cumsum(valid.ravel()) - 1
            drop = valid.ravel() & (rank >= cap)
            rows.reshape(-1, width)[drop] = 0
            obs[:, offset:offset + count * width] = rows.reshape(len(obs), -1)
    weights = torch.tensor(to_checkpoint(flat_trainer, hidden), requires_grad=True)
    net = policy_v4.PolicyV4(weights, hidden, 1, 4, 0, **SHAPE)
    net.set_dtype = torch.bfloat16
    x = torch.from_numpy(obs[:, :layout.V4_OBS_SIZE])
    flat = torch.relu(x[:, :layout.V4_FLAT] @ net.w["flat"].t())
    pools = [net.encode_set(x, *spec) for spec in layout.V4_SETS]
    u = policy_v4.layer_norm(torch.cat([flat, *pools], -1) @ net.w["fusion"].t())
    y = u
    if SHAPE["ffn_dim"]:
        y = y + torch.relu(policy_v4.layer_norm(y) @ net.w["ffn_in"].t()) @ net.w["ffn_out"].t()
    logits = y @ net.w["actor"].t()
    value = net.value(y, x)
    rows = torch.cat([logits, value], -1)
    (logits * torch.from_numpy(grad_logits)).sum().add((value[:, 0] * torch.from_numpy(grad_value)).sum()).backward()
    return u.detach().numpy(), rows.detach().numpy(), from_checkpoint_grad(weights.grad.numpy(), hidden)


# ---- synthetic rows ----
def random_obs(rng, batch, density=0.5):
    obs = np.zeros((batch, layout.OBS_STRIDE), np.float32)
    obs[:, :layout.V4_FLAT] = rng.normal(size=(batch, layout.V4_FLAT)) * 0.5
    for name, offset, count, width, id_column in layout.V4_SETS:
        rows = rng.normal(size=(batch, count, width)).astype(np.float32) * 0.5
        present = rng.random((batch, count)) < density
        rows[:, :, 0] = np.where(present, rng.uniform(0.5, 1.5, (batch, count)), 0.0)
        rows[~present] = 0
        if id_column is not None:
            ids = rng.integers(0, layout.EMBEDDING_IDS, (batch, count)).astype(np.float32)
            bad = rng.random((batch, count)) < 0.1
            ids[bad] = rng.choice([-1.0, layout.EMBEDDING_IDS + 2.0, 3.5], bad.sum())
            rows[:, :, id_column] = ids
            rows[:, :, id_column][~present] = 0
        obs[:, offset:offset + count * width] = rows.reshape(batch, -1)
    obs[:, layout.V4_PRIVILEGED_OFFSET:layout.V4_OBS_SIZE] = rng.normal(size=(batch, layout.V4_PRIVILEGED)) * 0.5
    return obs


def crowded_fixture(output):
    executable = output / "check_observations"
    data = output / "crowded-v4.f32"
    subprocess.run(["cc", "-O2", "-std=gnu11", "-I" + str(ROOT / "native/rl"), "-I" + str(HERE),
                    str(ROOT / "native/rl/research/check_observations.c"), "-lm", "-o", str(executable)], check=True)
    subprocess.run([str(executable), "--dump-v4", str(data)], check=True)
    rows = np.fromfile(data, np.float32).reshape(2, layout.V4_OBS_SIZE)
    obs = np.zeros((2, layout.OBS_STRIDE), np.float32)
    obs[:, :layout.V4_OBS_SIZE] = rows
    return obs


def init_weights(rng, hidden):
    offsets, total = trainer_offsets(hidden)
    out = np.zeros(total, np.float32)
    for name, (o, rows, cols) in offsets.items():
        bound = 1.0 / np.sqrt(cols) if name != "embedding" else 0.1
        out[o:o + rows * cols] = rng.uniform(-bound, bound, rows * cols)
    return out


class Check:
    def __init__(self, library, batch, hidden):
        self.lib, self.batch, self.hidden = library, batch, hidden
        self.context = library.bopl_v4_check_create(batch, hidden, OD)
        if not self.context:
            raise RuntimeError(library.bopl_v4_check_error().decode())
        _, total = trainer_offsets(hidden)
        assert library.bopl_v4_check_weight_count(self.context) == total, (library.bopl_v4_check_weight_count(self.context), total)
        shapes = np.zeros(2 * 64, np.int64)
        n = library.bopl_v4_check_param_shapes(self.context, shapes, 64)
        expected = [tuple(s) for _, s in trainer_tensors(hidden)]
        expected[-1] = (expected[-1][1], 1)   # critic_out is registered 1-D
        assert [tuple(shapes[2 * i:2 * i + 2]) for i in range(n)] == expected, list(shapes[:2 * n])
        self.width = library.bopl_v4_check_out_width(self.context)

    def run(self, weights, obs, grad_logits, grad_value, graph=False):
        enc = np.empty((self.batch, self.hidden), np.float32)
        dec = np.empty((self.batch, self.width), np.float32)
        grads = np.empty_like(weights)
        overflow = np.zeros(4, np.int32)
        args = [np.ascontiguousarray(v, dtype=np.float32) for v in (weights, obs, grad_logits, grad_value)]
        if self.lib.bopl_v4_check_run(self.context, *args, enc, dec, grads, overflow, int(graph)):
            raise RuntimeError(self.lib.bopl_v4_check_error().decode())
        assert np.isfinite(enc).all() and np.isfinite(dec).all() and np.isfinite(grads).all()
        return enc, dec[:, :OD + 1], grads, overflow

    def gather(self, ticks, total, indices):
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        if self.lib.bopl_v4_check_gather(self.context, ticks, total, indices):
            raise RuntimeError(self.lib.bopl_v4_check_error().decode())

    def profile(self):
        names = np.zeros(64 * 64, np.uint8); ms = np.zeros(64, np.float64)
        n = self.lib.bopl_v4_check_profile(self.context, names.ctypes.data_as(ctypes.c_char_p), ms, 64)
        out = {}
        for i in range(n):
            name = bytes(names[64 * i:64 * i + 64]).split(b"\0")[0].decode()
            out[name] = out.get(name, 0.0) + float(ms[i])
        return out

    def time(self, iterations, forward_only=False):
        return self.lib.bopl_v4_check_time(self.context, iterations, int(forward_only))

    def close(self):
        self.lib.bopl_v4_check_close(self.context)


def relative(actual, expected):
    scale = max(float(np.abs(expected).max()), 1e-6)
    return float(np.abs(actual - expected).max() / scale)


def relative_q99(actual, expected):
    """99th percentile of the error, relative to the tensor's largest value: a
    ReLU whose pre-activation sits within fp32 ordering noise of zero flips its
    gradient mask between the two implementations and moves one whole row of
    the weight gradient, which the max would report as a failure."""
    scale = max(float(np.abs(expected).max()), 1e-6)
    return float(np.quantile(np.abs(actual - expected), 0.99) / scale)


def compare(library, obs, hidden, rng, graph=False, budget=None, tol=(2e-2, 3e-2)):
    batch = len(obs)
    weights = init_weights(rng, hidden)
    grad_logits = rng.normal(size=(batch, OD)).astype(np.float32)
    grad_value = rng.normal(size=batch).astype(np.float32)
    enc_e, dec_e, grad_e = oracle(obs, weights, hidden, grad_logits, grad_value, budget)
    check = Check(library, batch, hidden)
    try:
        enc, dec, grad, overflow = check.run(weights, obs, grad_logits, grad_value)
        offsets, _ = trainer_offsets(hidden)
        per_tensor = {name: relative_q99(grad[o:o + r * c], grad_e[o:o + r * c]) for name, (o, r, c) in offsets.items()}
        per_tensor_max = {name: relative(grad[o:o + r * c], grad_e[o:o + r * c]) for name, (o, r, c) in offsets.items()}
        result = dict(batch=batch, hidden=hidden, budget=list(budget_in_effect()), encoder_rel=relative(enc, enc_e), logits_rel=relative(dec[:, :OD], dec_e[:, :OD]),
                      value_rel=relative(dec[:, OD], dec_e[:, OD]), grad_rel_q99=per_tensor, grad_rel_max=per_tensor_max, overflow=overflow.tolist())
        worst = max(result["encoder_rel"], result["logits_rel"], result["value_rel"])
        assert worst < tol[0], result
        assert max(per_tensor.values()) < tol[1], result
        assert max(per_tensor_max.values()) < 0.25, result
        if graph:
            for _ in range(2):
                enc2, dec2, grad2, _ = check.run(weights, obs, grad_logits, grad_value, True)
                np.testing.assert_array_equal(enc2, enc); np.testing.assert_array_equal(dec2, dec)
                np.testing.assert_array_equal(grad2, grad)
            result["graph_replay"] = True
        return result
    finally:
        check.close()


def gather_case(library, rng, hidden=64):
    ticks, total, segments = 4, 6, 3
    obs = random_obs(rng, ticks * total)
    indices = rng.choice(total, segments, replace=False)
    gathered = obs.reshape(ticks, total, -1)[:, indices].transpose(1, 0, 2).reshape(segments * ticks, -1)
    weights = init_weights(rng, hidden)
    grad_logits = rng.normal(size=(segments * ticks, OD)).astype(np.float32)
    grad_value = rng.normal(size=segments * ticks).astype(np.float32)
    check = Check(library, segments * ticks, hidden)
    try:
        enc, dec, grad, _ = check.run(weights, gathered, grad_logits, grad_value)
        check.gather(ticks, total, indices)
        enc_g, dec_g, grad_g, _ = check.run(weights, obs, grad_logits, grad_value, True)
        np.testing.assert_array_equal(enc_g, enc); np.testing.assert_array_equal(dec_g, dec)
        np.testing.assert_array_equal(grad_g, grad)
        return dict(ticks=ticks, total=total, segments=segments, identical=True)
    finally:
        check.close()


def overflow_case(library, rng, hidden=64):
    """A tiny row budget: the dropped rows are the same rows the reference drops."""
    budget = (2, 3, 4, 2)
    os.environ["BOPL4_ROW_BUDGET"] = ",".join(map(str, budget))
    try:
        obs = random_obs(rng, 40, density=0.8)
        result = compare(library, obs, hidden, rng, budget=budget)
        assert sum(result["overflow"]) > 0, result
        return result
    finally:
        del os.environ["BOPL4_ROW_BUDGET"]


def timing(library, rng, hidden=256):
    out = {}
    for batch in (4096, 8192):
        obs = random_obs(rng, batch, density=0.4)
        check = Check(library, batch, hidden)
        try:
            weights = init_weights(rng, hidden)
            check.run(weights, obs, np.zeros((batch, OD), np.float32), np.zeros(batch, np.float32))
            out[str(batch)] = dict(forward_ms=check.time(20, True), forward_backward_ms=check.time(20), stages_ms=check.profile())
        finally:
            check.close()
    return out


def cpu_case(rng, hidden=64, layers=2, steps=3, **shape):
    """The CPU evaluator (policy_eval.c + bopl_v4_eval.h, what the pool rater and the
    in-game agent run) against the torch reference, fp32 both, on random weights
    and rows: raw decoder rows, game logits, value and recurrent state."""
    import native_policy
    full = {**layout.V4_DEFAULTS, **shape}
    weights = policy_v4.init_weights(hidden, layers, aim_fourier=4, seed=int(rng.integers(1 << 30)), **full)
    ref = policy_v4.PolicyV4(weights, hidden, layers, 4, 0, **full)
    policy = native_policy.NativePolicy(weights, layout.V4_OBS_SIZE, hidden, layers, architecture=layout.V4_ARCHITECTURE,
                                        aim_fourier=4, v4_shape=full)
    state = torch.zeros(layers, 6, hidden)
    worst = dict(rows=0.0, state=0.0, logits=0.0, value=0.0)
    for _ in range(steps):
        obs = torch.from_numpy(random_obs(rng, 6, 0.4)[:, :layout.V4_OBS_SIZE].copy())
        rows_c, state_c = policy.step_raw(obs, state)
        rows_t, state_t = ref.step_raw(obs, state)
        logits_c, value_c = policy.game_logits(rows_c)
        logits_t, value_t = ref.game_logits(rows_t)
        worst["rows"] = max(worst["rows"], (rows_c - rows_t).abs().max().item())
        worst["state"] = max(worst["state"], (state_c - state_t).abs().max().item())
        worst["logits"] = max(worst["logits"], (logits_c - logits_t).abs().max().item())
        worst["value"] = max(worst["value"], (value_c - value_t).abs().max().item())
        state = state_c
    result = dict(hidden=hidden, layers=layers, shape=full, max_abs=worst, pass_all=max(worst.values()) < 1e-4)
    assert result["pass_all"], result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "native/artifacts/rl/native-migration-audit/cuda-v4")
    parser.add_argument("--puffer", type=Path, default=ROOT / "native/artifacts/rl/dependencies" / f"PufferLib-{REVISION}")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--cuda-host-compiler", default="g++-15")
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    library_path = args.output / "check_v4.so"
    if args.build or args.build_only:
        build(library_path, args.puffer, args.cuda_host_compiler)
    if args.build_only:
        print(library_path)
        return
    os.environ.setdefault("BOPL_TF32", "0")
    library = ctypes.CDLL(str(library_path))
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    library.bopl_v4_check_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    library.bopl_v4_check_create.restype = ctypes.c_void_p
    library.bopl_v4_check_error.restype = ctypes.c_char_p
    library.bopl_v4_check_weight_count.argtypes = [ctypes.c_void_p]
    library.bopl_v4_check_weight_count.restype = ctypes.c_long
    library.bopl_v4_check_out_width.argtypes = [ctypes.c_void_p]
    library.bopl_v4_check_param_shapes.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"), ctypes.c_int]
    library.bopl_v4_check_run.argtypes = [ctypes.c_void_p, *([f32] * 7), i32, ctypes.c_int]
    library.bopl_v4_check_run.restype = ctypes.c_int
    library.bopl_v4_check_gather.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, i32]
    library.bopl_v4_check_profile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), ctypes.c_int]
    library.bopl_v4_check_time.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.bopl_v4_check_time.restype = ctypes.c_double
    library.bopl_v4_check_close.argtypes = [ctypes.c_void_p]
    torch.set_num_threads(4)
    rng = np.random.default_rng(73)
    report = dict(schema=1, pass_all=True)
    report["random"] = [compare(library, random_obs(rng, batch, density), hidden, rng)
                        for batch, hidden, density in ((1, 64, 0.5), (3, 128, 0.5), (65, 64, 0.3), (257, 256, 0.3), (130, 64, 0.6))]
    report["empty_sets"] = compare(library, random_obs(rng, 5, density=0.0), 64, rng)
    report["crowded"] = compare(library, crowded_fixture(args.output), 64, rng)
    report["graph"] = compare(library, random_obs(rng, 33), 128, rng, graph=True)
    report["gather"] = gather_case(library, rng)
    report["overflow"] = overflow_case(library, rng)
    report["cpu"] = [cpu_case(rng), cpu_case(rng, hidden=128, layers=1, set_inner=64, ffn_dim=0, critic_dim=384),
                     cpu_case(rng, critic_privileged=0, mean_pool=0), cpu_case(rng, critic_dim=0)]
    if args.time:
        report["timing"] = timing(library, rng)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
