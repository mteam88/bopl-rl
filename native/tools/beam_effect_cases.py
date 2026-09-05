"""Stateful beam target effects, including stopped owners and source failures."""
import math
import random
from beam_cases import palette
from verify_dynamics import ONE, sample, vec, vector

KINDS = ["wall", "terrain", "lethal", "player", "affector", "orb", "other"]
ROLES = ["Single", "Parent", "Child", "MaybeChild", "ParentAndChild", "ChildWithEffect"]


def queries(count, seed):
    rng = random.Random(seed)
    for case in range(count):
        config = {
            "water": sample(rng, -15, -5),
            "time_scale": rng.choice([0, 42949671, 42949672, 42949673, ONE]),
            "protected": [bool(rng.getrandbits(1)) for _ in range(6)],
            "platform_push": sample(rng, 0, 100), "black_hole_push": sample(rng, 0, 20),
            "body_push": sample(rng, 0, 60),
        }
        targets = [{
            "kind": kind, "destroyed": rng.randrange(10) == 0,
            "component": rng.randrange(5) != 0, "component_destroyed": rng.randrange(10) == 0,
            "force": vec(rng, -3, 3), "velocity": vec(rng, -3, 3),
            "mass": 0 if case % 41 == 0 else sample(rng, -10, 10),
            "orb": {"kind": case % 6, "charge": rng.choice([ONE, ONE - 1, sample(rng, -1, 2)]),
                    "rate": rng.choice([0, 60 * ONE, sample(rng, -1, 10)]),
                    "firing": bool(rng.getrandbits(1)), "colors": palette(rng)},
        } for kind in KINDS]
        frames = []
        for tick in range(8):
            impacts = []
            for i in range(7):
                angle = rng.uniform(-math.pi, math.pi)
                position = vec(rng, -20, 20)
                direction = vector(int(math.cos(angle) * ONE), int(math.sin(angle) * ONE))
                if i < 3 and tick == 0:
                    position["y"] = config["water"] + ONE + (case % 3 - 1)
                    direction = vector(ONE, 0)
                beam = {
                    "id": i + 1, "owner_id": 1000 if i == 6 else i + 1,
                    "position": position, "direction": direction,
                    "time_passed": 0, "distance": sample(rng, 0, 30),
                    "scale": rng.choice([-ONE, 0, ONE, 40 * ONE, 40 * ONE + 1, sample(rng, .1, 60)]),
                    "colors": palette(rng), "ground": None,
                }
                impacts.append({
                    "beam": beam, "role": "Single" if tick == 0 else rng.choice(ROLES),
                    "water_reflection": tick > 0 and rng.randrange(5) == 0,
                    "target": i if tick == 0 else rng.choice([None, *range(7)]),
                })
            frames.append(impacts)
        yield {"operation": "dynamics_beam_effects", "config": config, "targets": targets, "frames": frames}
