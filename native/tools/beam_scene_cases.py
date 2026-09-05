"""Combined beam/rigid/composite/rope scenes with persistent hitbox pools."""
import math
import itertools
import random

from beam_cases import graph_queries, palette
from verify_dynamics import ONE, sample, vec
from verify_physics_scene import queries as physics_queries, add_ropes

LAYERS = {"wall": 11, "terrain": 21, "lethal": 16, "player": 8, "affector": 23, "orb": 26, "other": 0}


def queries(count, seed, *, stress=False):
    rng = random.Random(seed ^ 13)
    scenes = list(physics_queries(count, seed + 701))
    add_ropes(scenes, seed + 17)
    for case, (world, beams) in enumerate(zip(scenes, graph_queries(count, seed, copies=3 if stress else 1, ticks=36 if stress else 12))):
        scene = world["scene"]
        scene["objects"].sort(key=lambda o: ["circle", "box", "rounded_rect"].index(o["collider"]["kind"]))
        targets = []
        for i, obj in enumerate(scene["objects"]):
            shape = obj["collider"]["kind"]
            kind = ["wall", "terrain", "lethal"][(case + i) % 3] if shape == "rounded_rect" else "player" if shape == "box" else ["affector", "orb", "other"][(case + i) % 3]
            obj["layer"] = LAYERS[kind]
            obj["callback"] = True
            obj["callback_destroyed"] = obj["destroyed"]
            # Explicit cached direction also exercises its distinction from rotation.
            angle = obj["transform"]["rotation"] / ONE
            obj["transform"]["right"] = {"x": int(math.cos(angle) * ONE), "y": int(math.sin(angle) * ONE)}
            if kind != "other":
                obj["active"] = True
            targets.append({"id": obj["id"], "state": {
                "kind": kind, "destroyed": obj["destroyed"], "component": kind != "other",
                "component_destroyed": obj["destroyed"], "force": obj["body"]["force"].copy(),
                "velocity": vec(rng, -1, 1), "mass": sample(rng, .1, 5),
                "orb": {"kind": case % 6, "charge": sample(rng, 0, 1), "rate": sample(rng, .1, 10),
                        "firing": False, "colors": palette(rng)},
            }})
        graph = beams["config"]
        graph["mask"] = sum(1 << layer for layer in LAYERS.values())
        config = {
            "graph": graph,
            "hitbox": {"base_width": sample(rng, .1, 1), "extra_length": sample(rng, .01, 1), "minimum_extents": 4294967},
            "effects": {"water": graph["water"], "time_scale": 0 if case % 7 == 0 else ONE,
                        "protected": [bool(rng.getrandbits(1)) for _ in range(6)],
                        "platform_push": sample(rng, 0, 20), "body_push": sample(rng, 0, 15), "black_hole_push": sample(rng, 0, 5)},
            "width": sample(rng, .05, .5), "parent_offset": sample(rng, 0, .2), "layer": 18,
            "width_curve": {"pre_wrap": 0, "post_wrap": 0, "keys": [
                {"time": 0, "value": ONE // 5, "in_tangent": 0, "out_tangent": ONE},
                {"time": ONE // 3, "value": ONE, "in_tangent": 0, "out_tangent": 0}]},
        }
        frames = [{"dt": dt, "beams": frame} for dt, frame in zip(itertools.cycle(world["steps"]), beams["frames"])]
        frames[3]["destroy_slots"] = [0]
        frames[9]["destroy_slots"] = [min(case % 6, 1)]
        if stress:
            frames[21]["destroy_slots"] = [0]
            frames[27]["destroy_slots"] = [1]
        yield {"operation": "dynamics_beam_scene", "scene": scene, "config": config, "targets": targets, "frames": frames}
