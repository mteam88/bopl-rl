"""Publish native weights and evaluator metadata as an immutable checkpoint.

The native backend owns serialization. Its callback receives a temporary path;
the final .bin becomes visible only after validated metadata is in place.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

import layout


def write_checkpoint(path, save_weights, *, obs_size, hidden, num_layers,
                     architecture, game_steps, learner_steps, config, aim_fourier=0, aim_ar=0, v4_shape=None):
    """Call save_weights(str_path), validate float32 weights, return metadata.

    Pass lambda name: _C.save_weights(trainer, name) as save_weights. Steps are
    actual environment counters, never PufferLib's padded native agent count.
    aim_fourier/aim_ar (train/bopl_aim.h) size the decoder and are recorded
    for the evaluators.
    """
    dimensions = dict(obs_size=obs_size, hidden=hidden, num_layers=num_layers)
    aim = dict(aim_fourier=aim_fourier, aim_ar=aim_ar)
    for key, value in {**dimensions, **aim, "game_steps": game_steps, "learner_steps": learner_steps}.items():
        if type(value) is not int or value < (0 if key.endswith("steps") or key in aim else 1):
            raise ValueError(f"Invalid native checkpoint {key}")
    if hidden < 8 or hidden % 8 or learner_steps > game_steps:
        raise ValueError("Invalid native hidden size or decision counters")
    if aim_fourier > layout.AIM_MAX_HARMONICS or aim_ar > 1:
        raise ValueError("Invalid native checkpoint aim head parameterisation")
    if architecture == layout.V4_ARCHITECTURE and obs_size == layout.V4_OBS_SIZE:
        # Policy v4: the whole network is described by the shape dict.
        v4_shape = {**layout.V4_DEFAULTS, **(v4_shape or {})}
        expected = layout.policy_v4_weights(hidden, num_layers, aim_fourier=aim_fourier, aim_ar=aim_ar, **v4_shape)
    else:
        if architecture == layout.ARCHITECTURE and obs_size == layout.OBS_SIZE:
            encoder = layout.set_encoder_weights(hidden)
        elif architecture == "puffernet-mingru":
            encoder = hidden * obs_size
        else:
            raise ValueError("Unsupported native checkpoint architecture or observation size")
        expected = encoder + (layout.decoder_logits(aim_fourier, aim_ar) + 1) * hidden + num_layers * 3 * hidden**2
        v4_shape = None
    if not config.get("training_maps") or type(config.get("episode_ticks")) is not int or config["episode_ticks"] < 1:
        raise ValueError("Checkpoint config requires training_maps and positive episode_ticks")
    metadata = dict(schema=1, backend="pufferlib-native", architecture=architecture,
                    precision="float32", **dimensions, **aim, action_sizes=list(layout.ACTION_SIZES),
                    **({"v4_shape": v4_shape} if v4_shape else {}),
                    step=game_steps, learner_steps=learner_steps,
                    step_unit="physical_player_decisions", config=config)
    # Reject unserializable/nonfinite configuration before invoking the backend.
    json.dumps(metadata, allow_nan=False)
    path = Path(path)
    sidecar = Path(str(path) + ".json")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as directory:
        pending = Path(directory) / "weights.bin"
        save_weights(str(pending))
        data = pending.read_bytes()
        if len(data) != expected * 4 or not np.isfinite(np.frombuffer(data, dtype="<f4")).all():
            raise ValueError(f"Expected {expected} finite native float32 weights")
        metadata["weights_sha256"] = hashlib.sha256(data).hexdigest()
        pending_meta = Path(directory) / "metadata.json"
        pending_meta.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        os.replace(pending_meta, sidecar)
        os.replace(pending, path)
    return metadata
