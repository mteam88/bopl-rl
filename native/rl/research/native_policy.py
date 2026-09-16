"""Evaluate native PufferLib MinGRU checkpoints through upstream CPU PufferNet.

Implements the existing evaluator's initial()/step()/eval() interface. This is
inference only; no Torch model or optimizer duplicates the native trainer.
Each thread and batch size owns a C context; recurrent state belongs to the
caller, so independent games cannot leak memory into one another.

Checkpoint sidecar: <weights.bin>.json, schema 1, backend=pufferlib-native,
architecture=puffernet-mingru or layout.ARCHITECTURE (bopl-set-mingru-v3),
precision=float32, obs_size, hidden, num_layers, action_sizes=layout.ACTION_SIZES
([256,16,5]: aim, buttons, hold length), weights_sha256, and step (physical
player decisions, matching dashboard metrics and evaluation reports). Record
learner_steps separately for checkpoint cadence and the training budget.
Optional aim_fourier (harmonics, 0 = raw bins) and aim_ar (1 = one aim block
per button combo, buttons sampled first) select the aim head parameterisation
(puffer/bopl_aim.h); absent means 0.

step() returns layout.LOGITS policy logits (heads concatenated in
layout.ACTION_SIZES order), the value, and the next recurrent state. With
aim_ar the aim logits of step() are those of button combo 0; decide() samples
the way the trainer does (buttons, then the aim block of the sampled combo,
then hold) and returns the logits it sampled from.
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

import layout

ROOT = Path(__file__).resolve().parents[3]
REVISION = "42f70d6932c30ac977736f861006809c50168ba9"
LIBRARY = ROOT / "native/artifacts/rl/libbopl_policy_eval.so"
ARCHITECTURES = {"puffernet-mingru", layout.ARCHITECTURE, layout.V4_ARCHITECTURE}
V4_SHAPE_KEYS = ("set_dim", "set_inner", "flat_dim", "ffn_dim", "critic_dim", "critic_privileged", "critic_detach", "mean_pool")
HEAD_BOUNDS = [sum(layout.ACTION_SIZES[:i]) for i in range(layout.NUM_ATNS + 1)]  # logits row offsets per head


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
    path = Path(path)
    source = ROOT / "native/rl/puffer/policy_eval.c"
    if not path.exists():
        raise RuntimeError(f"{path} is missing; run native_policy.build() (or puffer/build.py)")
    if source.exists() and source.stat().st_mtime > path.stat().st_mtime:
        # overnight-7 lost its first trials to a copy that predated the Fourier aim head.
        raise RuntimeError(f"{path} is older than {source}; run native_policy.build() (or puffer/build.py)")
    lib = ctypes.CDLL(str(path))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.bopl_policy_eval_create_aim.argtypes = [array, ctypes.c_size_t, *([ctypes.c_int] * 7)]
    lib.bopl_policy_eval_create_aim.restype = ctypes.c_void_p
    lib.bopl_policy_eval_create_v4.argtypes = [array, ctypes.c_size_t, *([ctypes.c_int] * 5), ints]
    lib.bopl_policy_eval_create_v4.restype = ctypes.c_void_p
    lib.bopl_policy_eval_step_raw.argtypes = [ctypes.c_void_p, array, array, array, array]
    lib.bopl_policy_eval_step_raw.restype = None
    lib.bopl_policy_eval_game_logits.argtypes = [ctypes.c_void_p, array, ints, array]
    lib.bopl_policy_eval_game_logits.restype = None
    lib.bopl_policy_eval_close.argtypes = [ctypes.c_void_p]
    lib.bopl_policy_eval_close.restype = None
    return lib


class _Context:
    def __init__(self, lib, weights, batch, observations, hidden, layers, use_set, aim_fourier, aim_ar, v4_shape=None):
        self.lib = lib
        if v4_shape is not None:
            shape = np.array([v4_shape[key] for key in V4_SHAPE_KEYS], dtype=np.int32)
            self.handle = lib.bopl_policy_eval_create_v4(weights, weights.size, batch, hidden, layers, aim_fourier, aim_ar, shape)
        else:
            self.handle = lib.bopl_policy_eval_create_aim(weights, weights.size, batch, observations, hidden, layers,
                                                          use_set, aim_fourier, aim_ar)
        if not self.handle:
            raise ValueError("Invalid native policy dimensions or weights")

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.bopl_policy_eval_close(self.handle)


class NativePolicy:
    def __init__(self, weights, obs_size, hidden=128, num_layers=1, library=LIBRARY,
                 architecture="puffernet-mingru", aim_fourier=0, aim_ar=0, v4_shape=None):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unsupported native checkpoint architecture: {architecture!r}")
        self.architecture = architecture
        self.obs_size = int(obs_size)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.aim_fourier = int(aim_fourier)
        self.aim_ar = int(aim_ar)
        if self.obs_size < 1 or self.hidden < 8 or self.hidden % 8 or self.num_layers < 1:
            raise ValueError("Positive dimensions and hidden size divisible by eight are required")
        if not 0 <= self.aim_fourier <= layout.AIM_MAX_HARMONICS or self.aim_ar not in (0, 1):
            raise ValueError(f"Unsupported aim head parameterisation: aim_fourier={aim_fourier!r} aim_ar={aim_ar!r}")
        if self.architecture == layout.ARCHITECTURE and self.obs_size != layout.OBS_SIZE:
            raise ValueError(f"Bopl set checkpoints require {layout.OBS_SIZE} observation features")
        self.width = layout.decoder_logits(self.aim_fourier, self.aim_ar)  # decoder logit columns; the row is width + 1
        self.v4_shape = None
        if self.architecture == layout.V4_ARCHITECTURE:
            # Policy v4 (POLICY_V4_PLAN.md): layout v4 rows, the sidecar's v4_shape sizes the network.
            if self.obs_size != layout.V4_OBS_SIZE:
                raise ValueError(f"Bopl v4 checkpoints require {layout.V4_OBS_SIZE} observation features")
            self.v4_shape = {**layout.V4_DEFAULTS, **{k: int(v) for k, v in (v4_shape or {}).items() if k in layout.V4_DEFAULTS}}
            expected = layout.policy_v4_weights(self.hidden, self.num_layers, aim_fourier=self.aim_fourier, aim_ar=self.aim_ar,
                                                **self.v4_shape)
        else:
            # layout.policy_weights for the set encoder; the linear encoder accepts any obs width.
            encoder_weights = (layout.set_encoder_weights(self.hidden) if self.architecture == layout.ARCHITECTURE
                               else self.hidden * self.obs_size)
            expected = encoder_weights + self.hidden * (self.width + 1) + self.num_layers * 3 * self.hidden**2
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

    def _context(self, batch):
        if not hasattr(self.local, "contexts"):
            self.local.contexts = {}
        if batch not in self.local.contexts:
            self.local.contexts[batch] = _Context(self.lib, self.weights, batch, self.obs_size, self.hidden,
                                                self.num_layers, int(self.architecture == layout.ARCHITECTURE),
                                                self.aim_fourier, self.aim_ar, self.v4_shape)
        return self.local.contexts[batch]

    def step_raw(self, observations, state):
        """The raw decoder rows (batch, width + 1: aim block(s), buttons, hold, value) and the next state."""
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
        context = self._context(batch)
        rows = np.empty((batch, self.width + 1), dtype=np.float32)
        next_state = np.empty_like(current)
        self.lib.bopl_policy_eval_step_raw(context.handle, obs, current, rows, next_state)
        if not np.isfinite(rows).all() or not np.isfinite(next_state).all():
            raise FloatingPointError("Native checkpoint produced nonfinite policy output")
        return torch.from_numpy(rows), torch.from_numpy(next_state)

    def game_logits(self, rows, buttons=None):
        """The layout.LOGITS game logits and value of each raw row, for its button combo (aim_ar only)."""
        rows = np.ascontiguousarray(rows.detach().numpy() if hasattr(rows, "detach") else rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.width + 1:
            raise ValueError("Decoder row width differs from native checkpoint")
        batch = rows.shape[0]
        combos = np.zeros(batch, dtype=np.int32) if buttons is None else np.ascontiguousarray(
            buttons.detach().numpy() if hasattr(buttons, "detach") else buttons, dtype=np.int32).reshape(-1)
        if combos.shape != (batch,) or (combos < 0).any() or (combos >= layout.ACT_BUTTONS).any():
            raise ValueError("One button combo per row is required")
        out = np.empty((batch, layout.DECODER), dtype=np.float32)
        self.lib.bopl_policy_eval_game_logits(self._context(batch).handle, rows, combos, out)
        if not np.isfinite(out).all():
            raise FloatingPointError("Native checkpoint produced nonfinite policy output")
        return torch.from_numpy(out[:, :layout.LOGITS].copy()), torch.from_numpy(out[:, layout.LOGITS].copy())

    def step(self, observations, state):
        rows, next_state = self.step_raw(observations, state)
        logits, values = self.game_logits(rows)
        return logits, values, next_state


def masked(logits, mask):
    """Apply the environment's action mask (1 = allowed) the way the trainer samples: disallowed logits -> -inf."""
    return logits.masked_fill(torch.from_numpy(np.asarray(mask, dtype=np.uint8) == 0), float("-inf"))


def sample_head(logits, head, generator):
    """One index per row of one head of masked logits (batch, layout.LOGITS)."""
    part = logits[:, HEAD_BOUNDS[head]:HEAD_BOUNDS[head + 1]]
    return torch.multinomial(part.softmax(-1), 1, generator=generator).squeeze(-1)


def sample(logits, generator):
    """One index per head (aim, buttons, hold), shape (batch, layout.NUM_ATNS)."""
    return torch.stack([sample_head(logits, head, generator) for head in range(layout.NUM_ATNS)], -1)


def decide(policy, observations, mask, state, generator):
    """One policy decision sampled as the trainer samples: (masked logits, values, actions, next_state).

    Without aim_ar: aim, buttons, hold in turn from the row's logits. With
    aim_ar: buttons first (their logits are shared by every combo), then aim
    from the sampled combo's block, then hold; the returned logits are those
    of the sampled combo, so their aim head is the distribution aim came from.
    """
    rows, next_state = policy.step_raw(observations, state)
    logits, values = policy.game_logits(rows)
    logits = masked(logits, mask)
    if not policy.aim_ar:
        return logits, values, sample(logits, generator), next_state
    actions = torch.empty(rows.shape[0], layout.NUM_ATNS, dtype=torch.int64)
    actions[:, 1] = sample_head(logits, 1, generator)
    logits, values = policy.game_logits(rows, actions[:, 1])
    logits = masked(logits, mask)
    actions[:, 0] = sample_head(logits, 0, generator)
    actions[:, 2] = sample_head(logits, 2, generator)
    return logits, values, actions, next_state


def load(path, device="cpu", metadata=None, library=LIBRARY):
    if torch.device(device).type != "cpu":
        raise ValueError("NativePolicy loading supports CPU evaluation only")
    path = Path(path)
    meta = dict(metadata) if metadata is not None else json.loads(Path(str(path) + ".json").read_text())
    expected = {"schema": 1, "backend": "pufferlib-native",
                "precision": "float32", "action_sizes": list(layout.ACTION_SIZES)}
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
    for key in ("aim_fourier", "aim_ar"):
        if type(meta.get(key, 0)) is not int or meta.get(key, 0) < 0:
            raise ValueError(f"Invalid native checkpoint {key}")
    policy = NativePolicy(np.frombuffer(data, dtype="<f4"), meta["obs_size"], meta["hidden"],
                          meta["num_layers"], library=library, architecture=meta.get("architecture"),
                          aim_fourier=meta.get("aim_fourier", 0), aim_ar=meta.get("aim_ar", 0),
                          v4_shape=meta.get("v4_shape"))
    return policy, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("Use --build to compile the CPU evaluation binding")
    print(build())
