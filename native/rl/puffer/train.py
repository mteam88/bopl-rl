"""Train Bopl with upstream PufferLib and save local checkpoints and logs."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--players", type=int, choices=range(1, 7), default=6)
parser.add_argument("--seed", type=int, default=73)
parser.add_argument("--backend", choices=("native", "torch"), default="native")
parser.add_argument("--random-maps", action="store_true")
parser.add_argument("--random-loadouts", action="store_true")
parser.add_argument("--load", type=Path)
parser.add_argument("--output", type=Path, required=True)
options = parser.parse_args()
if options.steps < 1 or options.workers < 1:
    parser.error("steps and workers must be positive")
output = options.output.resolve()
output.mkdir(parents=True, exist_ok=False)
os.environ.setdefault("BOPL_RL_GAME", str(ROOT / "Builds/Linux/BoplBattle.x86_64"))
os.environ.setdefault("BOPL_RL_BOOTSTRAP", str(ROOT / "native/fixtures/movement.rep"))
os.environ.setdefault("BOPL_RL_LOG_DIR", str(output))
os.environ.setdefault("OMP_NUM_THREADS", str(options.workers))

import torch
from pufferlib import _C, pufferl

if not _C.gpu:
    raise RuntimeError("Training requires the CUDA backend; run build.py --device cuda")
torch.cuda.init()
torch.manual_seed(options.seed)
torch.set_num_threads(options.workers)
sys.argv = [sys.argv[0]]
args = pufferl.load_config("bopl")
args["seed"] = options.seed
agents = options.workers * options.players
args["vec"].update(total_agents=agents, num_threads=options.workers)
args["train"].update(total_timesteps=options.steps, minibatch_size=agents * 64, seed=options.seed)
args["env"].update(players=options.players, random_maps=int(options.random_maps), random_loadouts=int(options.random_loadouts))
args["load_model_path"] = str(options.load.resolve()) if options.load else None
args["cudagraphs"] = -1  # Variable episode resets can block the CPU worker.
pufferl.validate_config(args)
backend = _C
if options.backend == "torch":
    from pufferlib.torch_pufferl import PuffeRL
    backend = PuffeRL
(output / "config.json").write_text(json.dumps(args, indent=2) + "\n")
trainer = backend.create_pufferl(args)
started = time.monotonic()
try:
    with (output / "metrics.jsonl").open("w") as log:
        while trainer.global_step < options.steps:
            backend.rollouts(trainer)
            backend.train(trainer)
            record = backend.log(trainer)
            if not all(math.isfinite(float(value)) for value in record["loss"].values()):
                raise AssertionError("Nonfinite training loss")
            line = json.dumps(record)
            log.write(line + "\n"); log.flush()
            print(line, flush=True)
            if trainer.epoch % 20 == 0:
                backend.save_weights(trainer, str(output / f"policy-{trainer.global_step}.bin"))
    final = output / "policy-final.bin"
    backend.save_weights(trainer, str(final))
    result = dict(status="TRAINING_COMPLETE", device="cuda", backend=options.backend,
                  agent_steps=trainer.global_step, elapsed_seconds=time.monotonic() - started,
                  checkpoint=str(final))
finally:
    backend.close(trainer)
(output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
