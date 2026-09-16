"""Win rate of 024-final with obs[8:10] (absolute position) zeroed at inference, by map set.

The decisive form of the memorisation question: if the policy's use of its
absolute position is a per-map lookup, zeroing it should cost more on the
training maps 0-2 than on the held-out maps 3-4 (or even help there).
Plays 024-final vs scripted and pool opponents, side-swapped, same seeds in
every condition; only the 024 player's observation is modified.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import layout  # noqa: E402
from generate import OUT, CONTROLLERS, INDEX, Runner, scripted_opponent, decide  # noqa: E402

OPPONENTS = ["archer-hard", "pursuer-hard", "mixed-hard", "v3b-007-500m", "v3b-059-final"]


def play(runner, seed, map_id, opponent, side, zero_pos, delay_pos=False):
    env = runner.env
    env.reset(seed, map_id)
    env.bot(0, 0); env.bot(1, 0)
    focal = side; other = 1 - side
    spec = CONTROLLERS[INDEX[opponent]]
    states = {focal: runner.policy(INDEX["024-final"]).initial(1, "cpu")}
    if spec["kind"] == "scripted":
        env.bot(other, *scripted_opponent(opponent))
    else:
        states[other] = runner.policy(INDEX[opponent]).initial(1, "cpu")
    generator = torch.Generator().manual_seed(seed * 7 + 3)
    while not env.ended:
        obs = env.obs.copy(); mask = env.mask.copy()
        if zero_pos:
            obs[focal, 8:10] = 0.0
        actions = np.zeros((2, layout.NUM_ATNS), dtype=np.float32)
        for p, ctrl in ((focal, INDEX["024-final"]), (other, INDEX[opponent] if spec["kind"] == "checkpoint" else None)):
            if ctrl is None:
                continue
            with torch.no_grad():
                _, _, action, states[p] = decide(runner.policy(ctrl), torch.from_numpy(obs[p:p + 1]), mask[p:p + 1], states[p], generator)
            actions[p] = action[0].numpy()
        env.step(actions)
    final = env.state
    timeout = bool(final.header.flags & 2)
    return 0.5 if timeout else float(bool(final.players[focal].flags & 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUT)
    parser.add_argument("--games", type=int, default=40, help="per opponent per map set (side-swapped pairs)")
    args = parser.parse_args()
    torch.set_num_threads(2)
    runner = Runner(args.dataset, args.dataset / "workers")
    result = {}
    for name, maps in (("maps012", (0, 1, 2)), ("maps34", (3, 4))):
        for zero in (False, True):
            scores = {}
            for opp in OPPONENTS:
                s = []
                for i in range(args.games):
                    s.append(play(runner, 800_000 + i // 2, maps[(i // 2) % len(maps)], opp, i % 2, zero))
                scores[opp] = float(np.mean(s))
            key = f"{name}/{'zero_pos' if zero else 'normal'}"
            result[key] = dict(per_opponent=scores, mean=float(np.mean(list(scores.values()))), games=args.games * len(OPPONENTS))
            print(key, round(result[key]["mean"], 3), {k: round(v, 2) for k, v in scores.items()}, flush=True)
    (args.dataset / "results").mkdir(exist_ok=True)
    (args.dataset / "results" / "ablate-position.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
