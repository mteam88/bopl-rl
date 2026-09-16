"""CPU cost per observation of the raycast configurations, against the encode and the sim step.

Plays 024-final vs mixed-hard on each training map and, at a spread of mid-game
states, times probe_rays for each (n, extra_down, hits) configuration (one
player, one observation), the full v3 encode of both players (probe_pack), and
the env step, all single-threaded in the C binding. Writes results/bench.json.
"""
import argparse
import json
import time
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import layout  # noqa: E402
from generate import OUT, CONTROLLERS, INDEX, Runner, scripted_opponent, decide  # noqa: E402

CONFIGS = [(8, 0, 1), (8, 4, 1), (16, 0, 1), (16, 4, 1), (32, 0, 1), (32, 4, 1), (64, 4, 1), (16, 4, 2), (32, 4, 2), (16, 4, 3), (32, 4, 3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUT)
    parser.add_argument("--states", type=int, default=12, help="states per map")
    args = parser.parse_args()
    runner = Runner(args.dataset, args.dataset / "workers")
    env = runner.env
    focal = INDEX["024-final"]
    timings = {f"rays{n}+{e}x{h}": [] for n, e, h in CONFIGS}
    timings["pack_both_players"] = []; timings["env_step"] = []
    entity_counts = []
    for m in (0, 1, 2):
        env.reset(700_000 + m, m)
        env.bot(0, 0); env.bot(1, *scripted_opponent("mixed-hard"))
        state = runner.policy(focal).initial(1, "cpu")
        generator = torch.Generator().manual_seed(m)
        decision = 0
        while not env.ended and len(timings["env_step"]) < args.states * (m + 1):
            obs = env.obs.copy(); mask = env.mask.copy()
            with torch.no_grad():
                _, _, action, state = decide(runner.policy(focal), torch.from_numpy(obs[0:1]), mask[0:1], state, generator)
            actions = np.zeros((2, layout.NUM_ATNS), dtype=np.float32); actions[0] = action[0].numpy()
            if decision % 25 == 24:
                live = env.live_state_pointer()
                entity_counts.append(int(live.contents.header.entity_count) if hasattr(live.contents.header, "entity_count") else -1)
                for n, e, h in CONFIGS:
                    timings[f"rays{n}+{e}x{h}"].append(env.rays_bench(live, 0, n, e, h, iterations=1000) / 1000.0)
                timings["pack_both_players"].append(env.pack_bench(iterations=500) / 1000.0)
                t0 = time.perf_counter(); env.step(actions); timings["env_step"].append((time.perf_counter() - t0) * 1e6)
            else:
                env.step(actions)
            decision += 1
    result = {k: dict(mean_us=float(np.mean(v)), median_us=float(np.median(v)), max_us=float(np.max(v)), n=len(v)) for k, v in timings.items()}
    result["_note"] = ("microseconds per call; rays* = one player's observation, pack = the v3 encode of both players, "
                       "env_step = one decision step of the Rust sim incl. hold ticks and the encode (Python-timed). "
                       f"entity counts sampled: {entity_counts}")
    for k, v in result.items():
        if k != "_note":
            print(f"{k:24s} {v['mean_us']:9.2f} us (median {v['median_us']:.2f}, max {v['max_us']:.2f})")
    (args.dataset / "results").mkdir(exist_ok=True)
    (args.dataset / "results" / "bench.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
