"""Check the patched CUDA MinGRU training scan against the recurrent reference fixture.

The fixture (Matthew's `recurrent-reference/*.npz`, pinned upstream Torch MinGRU
as test oracle only) stores observations (T, B, O), nonzero initial states
(L, B, H), `reset_before_observation` (T, B), outputs (T, B, logits + 1),
per-step states (T, L, B, H), upstream grad_outputs, all weight gradients and
initial-state gradients. This runs the real training path
(policy_forward_train + policy_backward from the overlay's patched models.cu)
once per case with the fixture's reset flags and compares everything the CUDA
side exposes: outputs, final states, weight gradients, initial-state gradients,
plus the tail-loss initial-state gradient (must be exactly zero when every row
resets somewhere in the sequence).
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sysconfig
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REVISION = "42f70d6932c30ac977736f861006809c50168ba9"


def build(output, overlay, source, compiler):
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    cuda = nvidia / "cu13"
    obj = output.with_suffix(".o")
    subprocess.run([str(cuda / "bin/nvcc"), "-O3", "-std=c++17", "-arch=sm_120",
                    "-ccbin", compiler, "-Xcompiler=-fPIC", "-DPRECISION_FLOAT",
                    "-I" + str(overlay), "-I" + str(source / "src"), "-I" + str(cuda / "include"),
                    "-c", str(HERE / "check_recurrent.cu"), "-o", str(obj)], check=True)
    subprocess.run([compiler, "-shared", "-Wl,--no-undefined", str(obj),
                    str(cuda / "lib/libcudart.so.13"), str(cuda / "lib/libcublas.so.13"), str(cuda / "lib/libcublasLt.so.13"),
                    str(cuda / "lib/libcurand.so.10"),
                    "-Wl,--disable-new-dtags,-rpath," + str(cuda / "lib"),
                    "-o", str(output)], check=True)


class Check:
    def __init__(self, lib, obs, hidden, layers, T, B, logits):
        self.lib = lib
        self.shape = (obs, hidden, layers, T, B, logits)
        self.context = lib.bopl_recurrent_check_create(obs, hidden, layers, T, B, logits)
        if not self.context:
            raise RuntimeError(lib.bopl_recurrent_check_error().decode())
        self.weight_count = lib.bopl_recurrent_check_weight_count(self.context)

    def run(self, weights, x, state, reset, grad_logits, grad_value):
        obs, hidden, layers, T, B, logits = self.shape
        assert weights.shape == (self.weight_count,)
        out = np.empty((B, T, logits + 1), np.float32)
        final_state = np.empty((layers, B, hidden), np.float32)
        weight_grads = np.empty_like(weights)
        state_grads = np.empty((layers, B, hidden), np.float32)
        args = [np.ascontiguousarray(a, dtype=np.float32) for a in (weights, x, state, reset, grad_logits, grad_value)]
        if self.lib.bopl_recurrent_check_run(self.context, *args, out, final_state, weight_grads, state_grads):
            raise RuntimeError(self.lib.bopl_recurrent_check_error().decode())
        return out, final_state, weight_grads, state_grads

    def close(self):
        if self.context:
            self.lib.bopl_recurrent_check_close(self.context)
            self.context = None


SCALE_TOL = 2e-6  # absolute tolerance floor as a fraction of max|expected| (~16 fp32 ulps)


def compare(name, actual, expected, rtol, atol, scale_tol=SCALE_TOL):
    """Elementwise |err| <= max(atol, scale_tol * max|expected|) + rtol * |expected|.

    The scale term is about 16 fp32 ulps of the largest element: gradients are
    fp32 reductions over thousands of terms (cuBLAS vs Torch order), so their
    noise floor is relative to the tensor's magnitude, not a fixed 1e-6.
    """
    actual = np.asarray(actual, np.float64)
    expected = np.asarray(expected, np.float64)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    if not np.isfinite(actual).all():
        return dict(name=name, ok=False, reason="nonfinite")
    err = np.abs(actual - expected)
    scale = float(np.abs(expected).max())
    tol = max(atol, scale_tol * scale) + rtol * np.abs(expected)
    return dict(name=name, ok=bool((err <= tol).all()), max_abs=float(err.max()),
                max_abs_over_scale=float(err.max() / scale) if scale > 0 else 0.0,
                expected_max_abs=scale, abs_tol_applied=float(max(atol, scale_tol * scale)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path,
                        default=ROOT / "native/artifacts/rl/native-migration-audit/recurrent-reference")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "native/artifacts/rl/native-migration-audit/recurrent-cuda")
    parser.add_argument("--overlay", type=Path, default=ROOT / "native/artifacts/rl/puffer-overlay")
    parser.add_argument("--puffer", type=Path, default=ROOT / "native/artifacts/rl/dependencies" / f"PufferLib-{REVISION}")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--cuda-host-compiler", default="g++-15")
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=2e-6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lib_path = args.output / "libcheck_recurrent.so"
    if args.build or not lib_path.exists():
        build(lib_path, args.overlay, args.puffer, args.cuda_host_compiler)
    # The overlay's cuBLAS wrapper defaults to TF32; the fixture comparison is fp32.
    os.environ.setdefault("BOPL_TF32", "0")
    lib = ctypes.CDLL(str(lib_path))
    lib.bopl_recurrent_check_error.restype = ctypes.c_char_p
    lib.bopl_recurrent_check_create.restype = ctypes.c_void_p
    lib.bopl_recurrent_check_create.argtypes = [ctypes.c_int] * 6
    lib.bopl_recurrent_check_weight_count.restype = ctypes.c_long
    lib.bopl_recurrent_check_weight_count.argtypes = [ctypes.c_void_p]
    f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    lib.bopl_recurrent_check_run.restype = ctypes.c_int
    lib.bopl_recurrent_check_run.argtypes = [ctypes.c_void_p] + [f32] * 10
    lib.bopl_recurrent_check_close.argtypes = [ctypes.c_void_p]

    reference = json.loads((args.fixtures / "result.json").read_text())
    cases = []
    all_ok = True
    for case in reference["cases"]:
        d = np.load(args.fixtures / case["file"])
        T, B, obs = d["observations"].shape
        layers, _, hidden = d["initial_state"].shape
        logits = d["outputs"].shape[-1] - 1
        chk = Check(lib, obs, hidden, layers, T, B, logits)
        assert chk.weight_count == d["weights"].shape[0], (chk.weight_count, d["weights"].shape)
        x = np.ascontiguousarray(d["observations"].transpose(1, 0, 2))          # (B, T, O)
        reset = np.ascontiguousarray(d["reset_before_observation"].T)           # (B, T)
        go = d["grad_outputs"].transpose(1, 0, 2)                               # (B, T, logits+1)
        checks = []

        # 1. Full loss: outputs, final state, weight gradients, initial-state gradients.
        out, final_state, wgrad, sgrad = chk.run(d["weights"], x, d["initial_state"], reset,
                                                 go[:, :, :logits], go[:, :, logits])
        checks.append(compare("outputs", out.transpose(1, 0, 2), d["outputs"], args.rtol, args.atol))
        checks.append(compare("final_state", final_state, d["states"][-1], args.rtol, args.atol))
        checks.append(compare("weight_gradients", wgrad, d["weight_gradients"], args.rtol, args.atol))
        checks.append(compare("initial_state_gradients", sgrad, d["initial_state_gradients"], args.rtol, args.atol))

        # 2. Loss on the last timestep only: every row resets inside the sequence,
        #    so no gradient may reach the initial state (exact zero expected).
        tail = np.zeros_like(go)
        tail[:, -1, :] = 1.0
        _, _, _, sgrad_tail = chk.run(d["weights"], x, d["initial_state"], reset,
                                      tail[:, :, :logits], tail[:, :, logits])
        r = compare("tail_loss_initial_state_gradient", sgrad_tail,
                    d["final_output_initial_state_gradient"], 0.0, 0.0, 0.0)
        r["exactly_zero"] = bool(np.count_nonzero(sgrad_tail) == 0)
        r["ok"] = r["ok"] and r["exactly_zero"]
        checks.append(r)

        # 3. Determinism: same inputs twice give identical bytes.
        out2, _, wgrad2, sgrad2 = chk.run(d["weights"], x, d["initial_state"], reset,
                                          go[:, :, :logits], go[:, :, logits])
        checks.append(dict(name="deterministic", ok=bool(np.array_equal(out, out2)
                            and np.array_equal(wgrad, wgrad2) and np.array_equal(sgrad, sgrad2))))
        chk.close()
        ok = all(c["ok"] for c in checks)
        all_ok &= ok
        cases.append(dict(file=case["file"], obs=obs, hidden=hidden, layers=layers, batch=B,
                          timesteps=T, ok=ok, checks=checks))
        print(f"{case['file']}: {'PASS' if ok else 'FAIL'}")
        for c in checks:
            print("   ", json.dumps(c))

    report = dict(schema=1, source_revision=REVISION, fixture=str(args.fixtures),
                  reset_semantics=reference.get("reset_semantics"),
                  rtol=args.rtol, atol=args.atol, scale_tol=SCALE_TOL,
                  tolerance="|err| <= max(atol, scale_tol*max|expected|) + rtol*|expected|",
                  timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  ok=all_ok, cases=cases)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ALL PASS" if all_ok else "FAILURES")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
