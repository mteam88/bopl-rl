"""Curve boundaries and beam hitbox initialization/update sequences."""
import random

from verify_dynamics import ONE, sample, vec, vector


def curve(rng, case):
    keys = []
    time = sample(rng, -4, 1)
    for _ in range(case % 9):
        time += sample(rng, .01, 2)
        keys.append({"time": time, "value": sample(rng, -2, 5),
                     "in_tangent": sample(rng, -10, 10), "out_tangent": sample(rng, -10, 10)})
    return {"keys": keys, "pre_wrap": [0, 1, 2, 4, 8][case % 5],
            "post_wrap": [0, 1, 2, 4, 8][case // 5 % 5]}


def curve_queries(count, seed):
    rng = random.Random(seed)
    for case in range(count):
        c = curve(rng, case)
        if case % 17 == 0 and len(c["keys"]) >= 2:
            c["keys"][-1]["time"] = c["keys"][0]["time"]
        times = [sample(rng, -30, 30) for _ in range(24)]
        times.extend([-(1 << 63), (1 << 63) - 1, -1, 0, 1])
        for key in c["keys"]:
            times.extend(key["time"] + offset for offset in (-1, 0, 1))
        if len(c["keys"]) >= 2:
            first, last = c["keys"][0]["time"], c["keys"][-1]["time"]
            times.extend(first + (last - first) * k for k in range(-4, 5))
        yield {"operation": "dynamics_curve", "curve": c, "times": times}


def hitbox_queries(count, seed):
    rng = random.Random(seed)
    for case in range(count):
        config = {"base_width": sample(rng, .01, 1), "extra_length": sample(rng, -.5, 2),
                  "minimum_extents": sample(rng, .0001, .1)}
        actions = []
        for tick in range(14):
            if tick == 2:
                actions.append({"operation": "initialize"})
                continue
            direction = vec(rng, -2, 2)
            if (case + tick) % 17 == 0:
                direction = vector(0, 0)
            elif tick % 4 == 0:
                direction = vector(ONE if tick % 8 == 0 else -ONE, 0)
            distance = sample(rng, 0, 70)
            if tick == 5:
                distance = -2 * config["extra_length"]
            dimensions = {
                "position": vec(rng, -20, 20), "direction": direction,
                "distance": distance, "width": 0 if tick % 7 == 0 else sample(rng, -1, 5),
                "age": sample(rng, -3, 10), "curve": curve(rng, case + tick),
                "player_id": 1 + (case + tick) % 6, "beam_id": case * 20 + tick,
            }
            actions.append({"operation": "dimensions", "dimensions": dimensions})
        yield {"operation": "dynamics_beam_hitbox", "config": config, "actions": actions}
