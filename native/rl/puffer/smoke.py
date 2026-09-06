"""Run upstream PufferLib rollout, optimization, checkpoint and reload checks."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
parser.add_argument("--output", type=Path, default=ROOT / "native/artifacts/rl/training")
options = parser.parse_args()
if options.epochs < 1 or options.workers < 1:
    parser.error("epochs and workers must be positive")
options.output = options.output.resolve()
options.output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("BOPL_RL_GAME", str(ROOT / "Builds/Linux/BoplBattle.x86_64"))
os.environ.setdefault("BOPL_RL_BOOTSTRAP", str(ROOT / "native/fixtures/movement.rep"))
os.environ.setdefault("BOPL_RL_LOG_DIR", str(options.output))
os.environ.setdefault("OMP_NUM_THREADS", str(options.workers))

import torch
from pufferlib import _C, pufferl
from pufferlib.torch_pufferl import PuffeRL

if bool(_C.gpu) != (options.device == "cuda"):
    raise RuntimeError(f"PufferLib backend gpu={_C.gpu}; build the requested {options.device} backend first")
if options.device == "cuda":
    torch.cuda.init()
    value = torch.randn(128, 128, device="cuda")
    assert torch.isfinite(value @ value).all().item()
    torch.cuda.synchronize()
torch.set_num_threads(options.workers)
sys.argv = [sys.argv[0]]
args = pufferl.load_config("bopl")
args["vec"].update(total_agents=6 * options.workers, num_threads=options.workers)
args["train"].update(total_timesteps=options.epochs * 64 * 6 * options.workers,
                     minibatch_size=64 * 6 * options.workers)
args["env"]["max_episode_ticks"] = 480
pufferl.validate_config(args)
vec = _C.create_vec(args, _C.gpu)
trainer = None
start = time.monotonic()
try:
    from pufferlib.torch_pufferl import load_policy
    policy = load_policy(args, vec)
    trainer = PuffeRL(args, vec, policy, verbose=False)
    before = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    records = []
    for epoch in range(options.epochs):
        trainer.rollouts()
        if not torch.isfinite(trainer.observations).all().item():
            raise AssertionError("Nonfinite observations")
        trainer.train()
        record = trainer.log()
        if not all(math.isfinite(float(value)) for value in record["loss"].values()):
            raise AssertionError("Nonfinite training loss")
        records.append(record)
        print(json.dumps(record), flush=True)
    changed = sum(not torch.equal(before[key], value) for key, value in policy.state_dict().items())
    if not changed:
        raise AssertionError("Training did not change weights")
    checkpoint = options.output / "policy.pt"
    trainer.save_weights(str(checkpoint))
    trained = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    trainer.load_weights(str(checkpoint))
    if any(not torch.equal(trained[key], value) for key, value in policy.state_dict().items()):
        raise AssertionError("Checkpoint reload changed weights")
    result = dict(status="PUFFER_TRAINING_SMOKE_OK", device=options.device, epochs=options.epochs,
                  workers=options.workers, agent_steps=trainer.global_step, changed_tensors=changed,
                  elapsed_seconds=time.monotonic() - start, checkpoint=str(checkpoint), logs=records)
    (options.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "logs"}), flush=True)
finally:
    if trainer is not None:
        trainer.close()
    else:
        vec.close()
