"""Deterministic beam geometry, color and multi-frame crossing scenarios."""
import copy
import math
import random
import struct

from verify_dynamics import ONE, sample, vec, vector

COLOR_KEYS = ("white", "yellow", "orange", "red")


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def palette(rng):
    return {key: [bits(rng.uniform(0, 1)) for _ in range(4)] for key in COLOR_KEYS}


def queries(count, seed):
    rng = random.Random(seed)
    colors = random.Random(seed ^ 902)

    def ray(i):
        angle = rng.uniform(-math.pi, math.pi)
        distance = sample(rng, .1, 50)
        return {
            "position": vec(rng, -10, 10),
            "direction": vector(int(math.cos(angle) * ONE), int(math.sin(angle) * ONE)),
            "distance": distance, "reflected": bool(i & 1), "was_reflected": bool(i & 2),
            "reflection_distance": rng.randrange(distance),
        }

    for i in range(count):
        a, b = ray(i), ray(i >> 2)
        if i % 11 == 0:
            b["direction"] = copy.deepcopy(a["direction"])
        if i % 13 == 0:
            b["position"] = copy.deepcopy(a["position"])
        if i % 17 == 0:
            a["direction"], b["direction"] = vector(ONE, 0), vector(0, ONE)
        if i % 19 == 0:
            a["distance"] = a["reflection_distance"] = 0
        yield {"operation": "dynamics_beam_lines", "a": a, "b": b}
        for flag in (False, True):
            yield {"operation": "dynamics_beam_intersection", "a": a, "b": b, "minimize_b": flag}
        yield {"operation": "dynamics_beam_colors", "a": palette(colors), "b": palette(colors)}

    # Exercise achromatic branches and exact hue sectors, absent from random RGB.
    edges = [(0, 0, 0, 0), (1, 1, 1, 1), (.5, .5, .5, .5),
             (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1),
             (1, 1, 0, 1), (0, 1, 1, 1), (1, 0, 1, 1), (2, .1, .5, 1)]
    for a in edges:
        for b in edges:
            yield {
                "operation": "dynamics_beam_colors",
                "a": {key: list(map(bits, a)) for key in COLOR_KEYS},
                "b": {key: list(map(bits, b)) for key in COLOR_KEYS},
            }


def graph_queries(count, seed, *, copies=1, ticks=12):
    from verify_physics_scene import queries as scenes

    rng = random.Random(seed)
    worlds = iter(scenes(count, seed + 13))
    for case in range(count):
        scene = next(worlds)["scene"]
        scene["composites"] = []
        for obj in scene["objects"]:
            obj["composite"] = None
        config = {
            "maximum": sample(rng, 15, 60), "water": sample(rng, -15, -8),
            "space": case % 4 == 0, "mask": rng.getrandbits(32),
            "cross_offset": sample(rng, 0, .2), "environmental_player": 1000,
        }
        players = 1 + case % 6
        beam_count = players * copies
        initial = []
        for i in range(beam_count):
            angle = (i + 1) * math.tau / beam_count
            if case % 3 == 0:
                position = vector(int(math.cos(angle) * 8 * ONE), int(math.sin(angle) * 8 * ONE))
                direction = angle + math.pi
            else:
                position = vec(rng, -10, 10)
                direction = rng.uniform(-math.pi, math.pi) if case % 3 == 2 else .00001 * i
            initial.append({
                "id": i + 1, "owner_id": i % players + 1, "position": position,
                "direction": vector(int(math.cos(direction) * ONE), int(math.sin(direction) * ONE)),
                "time_passed": 0, "scale": sample(rng, .1, 3),
                "distance": config["maximum"], "colors": palette(rng), "ground": None,
            })
        frames = []
        for tick in range(ticks):
            frame = copy.deepcopy(initial)
            if tick in (4, 5):
                frame = frame[:-1]
            if tick == 9:
                frame = []
            for i, beam in enumerate(frame):
                beam["time_passed"] = tick * 71582788
                beam["position"]["x"] += int(math.sin(tick * .2 + i) * ONE * .05)
                beam["position"]["y"] += int(math.cos(tick * .3 + i) * ONE * .05)
                if tick >= 10:
                    beam["id"] += 100
            frames.append(frame)
        yield {"operation": "dynamics_beam_graph", "config": config, "scene": scene, "frames": frames}
