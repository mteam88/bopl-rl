"""Evaluate native PufferLib MinGRU checkpoints through upstream CPU PufferNet.

Implements the existing evaluator's initial()/step()/eval() interface. This is
inference only; no Torch model or optimizer duplicates the native trainer.
Each thread and batch size owns a C context; recurrent state belongs to the
caller, so independent games cannot leak memory into one another.

Checkpoint sidecar: <weights.bin>.json, schema 1, backend=pufferlib-native,
architecture=puffernet-mingru, precision=float32, obs_size, hidden, num_layers,
action_sizes=[256,16], weights_sha256, and step (learner_steps recommended).
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import threading

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
REVISION = "42f70d6932c30ac977736f861006809c50168ba9"
LIBRARY = ROOT / "native/artifacts/rl/libbopl_policy_eval.so"


def build(output=LIBRARY):
    """Explicit build: loading a checkpoint never runs a compiler."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    include = ROOT / "native/artifacts/rl/dependencies" / f"PufferLib-{REVISION}" / "src"
    subprocess.run(["cc", "-O3", "-std=gnu11", "-fPIC", "-shared",
                    "-I" + str(include), str(ROOT / "native/rl/puffer/policy_eval.c"),
                    "-lm", "-o", str(output)], check=True)
    return output


def _library(path):
    lib = ctypes.CDLL(str(path))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.bopl_policy_eval_create.argtypes = [array, ctypes.c_size_t, *([ctypes.c_int] * 4)]
    lib.bopl_policy_eval_create.restype = ctypes.c_void_p
    lib.bopl_policy_eval_step.argtypes = [ctypes.c_void_p, array, array, array, array]
    lib.bopl_policy_eval_step.restype = None
    lib.bopl_policy_eval_close.argtypes = [ctypes.c_void_p]
    lib.bopl_policy_eval_close.restype = None
    return lib


class _Context:
    def __init__(self, lib, weights, batch, observations, hidden, layers):
        self.lib = lib
        self.handle = lib.bopl_policy_eval_create(weights, weights.size, batch, observations, hidden, layers)
        if not self.handle:
            raise ValueError("Invalid native policy dimensions or weights")

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.bopl_policy_eval_close(self.handle)


class NativePolicy:
    def __init__(self, weights, obs_size, hidden=128, num_layers=1, library=LIBRARY):
        self.obs_size = int(obs_size)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        if self.obs_size < 1 or self.hidden < 8 or self.hidden % 8 or self.num_layers < 1:
            raise ValueError("Positive dimensions and hidden size divisible by eight are required")
        expected = self.hidden * (self.obs_size + 273) + self.num_layers * 3 * self.hidden**2
        self.weights = np.asarray(weights, dtype=np.float32).reshape(-1).copy()
        if self.weights.size != expected or not np.isfinite(self.weights).all():
            raise ValueError(f"Expected {expected} finite native weights, got {self.weights.size}")
        self.weights.flags.writeable = False
        self.lib = _library(library)
        self.local = threading.local()

    def eval(self):
        return self

    def initial(self, agents, device="cpu"):
        if torch.device(device).type != "cpu" or agents < 1:
            raise ValueError("Native CPU evaluation requires a positive batch on CPU")
        return torch.zeros(self.num_layers, agents, self.hidden)

    def step(self, observations, state):
        if observations.device.type != "cpu" or state.device.type != "cpu":
            raise ValueError("NativePolicy evaluates CPU tensors only")
        if observations.ndim != 2 or observations.shape[1] != self.obs_size:
            raise ValueError("Observation shape differs from native checkpoint")
        batch = observations.shape[0]
        if batch < 1 or tuple(state.shape) != (self.num_layers, batch, self.hidden):
            raise ValueError("Recurrent state shape differs from native checkpoint")
        obs = np.ascontiguousarray(observations.detach().numpy(), dtype=np.float32)
        current = np.ascontiguousarray(state.detach().numpy(), dtype=np.float32)
        if not np.isfinite(obs).all() or not np.isfinite(current).all():
            raise ValueError("Nonfinite observation or recurrent state")
        if not hasattr(self.local, "contexts"):
            self.local.contexts = {}
        if batch not in self.local.contexts:
            self.local.contexts[batch] = _Context(self.lib, self.weights, batch, self.obs_size,
                                                self.hidden, self.num_layers)
        context = self.local.contexts[batch]
        output = np.empty((batch, 273), dtype=np.float32)
        next_state = np.empty_like(current)
        self.lib.bopl_policy_eval_step(context.handle, obs, current, output, next_state)
        if not np.isfinite(output).all() or not np.isfinite(next_state).all():
            raise FloatingPointError("Native checkpoint produced nonfinite policy output")
        return torch.from_numpy(output[:, :272].copy()), torch.from_numpy(output[:, 272].copy()), torch.from_numpy(next_state)


def load(path, device="cpu", metadata=None, library=LIBRARY):
    if torch.device(device).type != "cpu":
        raise ValueError("NativePolicy loading supports CPU evaluation only")
    path = Path(path)
    meta = dict(metadata) if metadata is not None else json.loads(Path(str(path) + ".json").read_text())
    expected = {"schema": 1, "backend": "pufferlib-native", "architecture": "puffernet-mingru",
                "precision": "float32", "action_sizes": [256, 16]}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Unsupported native checkpoint {key}: {meta.get(key)!r}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != meta.get("weights_sha256"):
        raise ValueError("Native checkpoint weights differ from metadata hash")
    if len(data) % 4:
        raise ValueError("Native weights are not a complete float32 array")
    for key in ("obs_size", "hidden", "num_layers", "step"):
        if type(meta.get(key)) is not int or meta[key] < (0 if key == "step" else 1):
            raise ValueError(f"Invalid native checkpoint {key}")
    policy = NativePolicy(np.frombuffer(data, dtype="<f4"), meta["obs_size"], meta["hidden"],
                          meta["num_layers"], library=library)
    return policy, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("Use --build to compile the CPU evaluation binding")
    print(build())
