#!/usr/bin/env python3
"""Extract the data the Rust rollout backend needs from native/reference-export.

Sources: maps.json, all map scene graphs, and assets.json prefab/asset entries of the
reconstructed game (reference 67484d0). Values are Q32.32 raw integers exactly as
serialized; nothing here is invented. Output: native/data/rollout.json.

Cross-check: Level1 platform geometry must reproduce the verified
native/data/level1-movement.json colliders (surface extents/radius/center).
"""
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "reference-export"

ONE = 1 << 32


def f32(v):
    if isinstance(v, dict) and "f32" in v:
        return struct.unpack("<f", struct.pack("<I", v["f32"] & 0xFFFFFFFF))[0]
    return v


def load(name):
    with open(EXPORT / name) as fh:
        return json.load(fh)


ASSETS = load("assets.json")
MAPS = load("maps.json")


def find_prefab(path):
    for guid, entry in ASSETS.items():
        if entry.get("path") == path:
            return guid, entry
    raise KeyError(path)


def component(entry, type_name, all_matches=False):
    found = [c for c in entry["data"]["components"] if c["type"] == type_name]
    if all_matches:
        return found
    if not found:
        raise KeyError(f"{entry.get('path')}: {type_name}")
    return found[0]["properties"]


def updater_count(prefab):
    return registering_count(prefab)


# Updater registration is opt-in per class. Updater never sweeps the scene for
# MonoUpdatable; each component calls Updater.RegisterUpdatable(this) from its
# own Awake (InstantAbility.cs:62-71), and EndRegisterPrefab then advances
# prefabsAdded by the number of components that actually registered -
# `prefabsAdded += toBeAddedPrefab.Count` (Updater.cs:143-153). So a serialized
# hierarchyNumber on a class that never registers is dead data, and the
# prefab's span over-counts the cursor advance. Two ability prefabs are in that
# position: SuperStar (Gold mushroom, hierarchyNumber 2, SuperStar.cs:18-21) and
# HookshotInstant (Grappling hook, hierarchyNumber 2). Both derive from
# MonoUpdatable, both are enabled, and neither ever reaches the Updater.
CSHARP = ROOT.parent / "Assets" / "Scripts" / "Assembly-CSharp"


def _class_table():
    table = {}
    for path in sorted(CSHARP.glob("*.cs")):
        text = path.read_text(errors="replace")
        decl = re.search(
            r"^\s*(?:public |internal |abstract |sealed |partial |static )*"
            r"class\s+(\w+)\s*(?::\s*([^{]+))?",
            text,
            re.M,
        )
        if decl is None:
            continue
        bases = [b.strip() for b in (decl.group(2) or "").split(",") if b.strip()]
        awake = re.search(r"\bvoid Awake\s*\(\s*\)", text)
        body = ""
        if awake is not None:
            start = text.index("{", awake.end())
            depth = 0
            end = start
            while end < len(text):
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                end += 1
            body = text[start:end]
        table[decl.group(1)] = (
            bases[0] if bases else None,
            awake is not None,
            "Updater.RegisterUpdatable" in body,
        )
    return table


CLASSES = _class_table()


def registers_with_updater(type_name):
    # Unity dispatches only the most-derived Awake, so the first ancestor that
    # declares one decides whether this component ever reaches the Updater.
    seen = set()
    while type_name in CLASSES and type_name not in seen:
        seen.add(type_name)
        base, declares, registers = CLASSES[type_name]
        if declares:
            return registers
        type_name = base
    return False


def registering_count(prefab):
    def visit(game_object):
        own = sum(
            "hierarchyNumber" in c.get("properties", {})
            and registers_with_updater(c["type"])
            for c in game_object.get("components", [])
        )
        return own + sum(visit(child) for child in game_object.get("children", []))

    return visit(prefab["data"])


EXECUTION_ORDER = json.loads((ROOT.parent / "sim" / "Export" / "execution-order.json").read_text())


def registrations(prefab, copy=False, exclude=()):
    # The prefab's updatables in the order Updater.RegisterUpdatable sees them
    # when it is instantiated, as their local HierarchyNumbers: Awake runs over
    # the active behaviours collected depth-first, stable-sorted by script
    # execution order (UnityShim World.CollectActiveBehaviours and
    # ScriptExecutionOrder.SortStable). An inactive root still registers:
    # InstantiateFixed(GameObject) switches it on and off again inside the
    # prefab window (FixTransform.cs:131-141). SlimeController.Awake registers
    # itself and then AddComponents the RewindRecorder, whose own Awake
    # registers it at once with HierarchyNumber 8 (SlimeController.cs:459-466,
    # RewindRecorder.cs:91-94).
    #
    # `copy`: what ShootDuplicator.DuplicateObject instantiates from a live
    # instance. It first renumbers the source's MonoUpdatables 1.. in
    # HierarchyNumber order (ShootDuplicator.cs:709-717), and a slime's live
    # RewindRecorder sits last on its GameObject, so the copy collects it with
    # the rest and its Awake runs in execution order. `exclude`: components
    # gone from the live source (GrowOnStart destroys itself).
    batch = []

    def collect(game_object, root=False):
        if not root and not game_object.get("active", True):
            return
        components = [c for c in game_object.get("components", []) if c["type"] not in exclude]
        if copy and any(c["type"] == "SlimeController" for c in components):
            components.append({"type": "RewindRecorder", "properties": {"hierarchyNumber": 8}})
        batch.extend(components)
        for child in game_object.get("children", []):
            collect(child)

    collect(prefab["data"], root=True)
    number = {id(c): c["properties"]["hierarchyNumber"] for c in batch if "hierarchyNumber" in c.get("properties", {})}
    if copy:
        ranked = sorted(number, key=number.get)
        number = {key: rank + 1 for rank, key in enumerate(ranked)}
    order = sorted(range(len(batch)), key=lambda i: (EXECUTION_ORDER.get(batch[i]["type"], 0), i))
    out = []
    for i in order:
        c = batch[i]
        if id(c) in number and registers_with_updater(c["type"]):
            out.append(number[id(c)])
            if c["type"] == "SlimeController" and not copy:
                out.append(8)
    return out


def updater_order(node):
    # The node's own components that register with the Updater, in
    # hierarchyNumber order: the order its UpdateSims run within the prefab's
    # block of updatables.
    return [
        typ for _, typ in sorted(
            (c["properties"]["hierarchyNumber"], c["type"])
            for c in node.get("components", [])
            if "hierarchyNumber" in c.get("properties", {}) and registers_with_updater(c["type"])
        )
    ]


def node_component(node, type_name):
    for c in node.get("components", []):
        if c["type"] == type_name:
            return c["properties"]
    return None


def walk(nodes):
    for n in nodes:
        yield n
        yield from walk(n.get("children", [])) 


def collider_child_count(prefab):
    # Arrow.OnCollide parents a landed arrow to the hit collider's own
    # FixTransform and counts that transform's children (Arrow.cs:100-104):
    # the authored children of the one GameObject holding the prefab's collider.
    holders = [
        n for n in [prefab["data"], *walk(prefab["data"].get("children", []))]
        if any(c["type"] in ("DPhysicsBox", "DPhysicsCircle") for c in n.get("components", []))
    ]
    assert len(holders) == 1, prefab.get("path")
    return len(holders[0].get("children", []))


def raw(props, key):
    v = props[key]
    if isinstance(v, dict):
        raise ValueError(f"{key} is not raw: {v}")
    return v


def vec2(props, prefix):
    return {"x": raw(props, f"{prefix}.x.m_rawValue"), "y": raw(props, f"{prefix}.y.m_rawValue")}


def fix(props, key):
    return raw(props, f"{key}.m_rawValue")


def curve(props, prefix):
    """AnimationCurveFixed: runtime wrap modes and exact KeyFrameF raws."""
    n = props.get(f"{prefix}.keys.Array.size", 0)
    keys = []
    for i in range(n):
        base = f"{prefix}.keys.Array.data[{i}]"
        keys.append({
            "time": raw(props, f"{base}.time.m_rawValue"),
            "value": raw(props, f"{base}.value.m_rawValue"),
            "in_tangent": raw(props, f"{base}.inTangent.m_rawValue"),
            "out_tangent": raw(props, f"{base}.outTangent.m_rawValue"),
        })
    # The normalized object records UnityEngine.WrapMode as observed by the
    # runtime. Do not use m_Pre/PostInfinity: those are Unity's serialized
    # internal codes and differ from the public enum values.
    anim = props[f"{prefix}.Anim"]
    return {
        "pre_wrap": anim["preWrap"],
        "post_wrap": anim["postWrap"],
        "keys": keys,
    }


def physics_body(props):
    """BoplBody fields consumed by DetPhysics/PhysTools."""
    return {
        "mass": fix(props, "mass"),
        "bounciness": fix(props, "bounciness"),
        "static_friction": fix(props, "friction"),
        "dynamic_friction": fix(props, "dynamicFriction"),
        "moment_of_inertia": fix(props, "momentOfInertia"),
        "gravity_scale": fix(props, "gravityScale"),
        "air_drag": fix(props, "airFriction01"),
        "rotational_drag": fix(props, "rotationalDrag01"),
        "start_velocity": vec2(props, "StartVelocity"),
        "start_angular_velocity": fix(props, "StartAngularVelocity"),
    }


def animations(guid):
    """AnimationData asset -> {name: [frameDuration...]} (sim ticks, sign = flip)."""
    p = ASSETS[guid]["data"]["properties"]
    anims = {}
    per = {}
    for k, v in p.items():
        m = re.match(r"animations\.Array\.data\[(\d+)\]\.(.*)", k)
        if m:
            per.setdefault(int(m.group(1)), {})[m.group(2)] = v
    for i in sorted(per):
        a = per[i]
        durs = []
        j = 0
        while f"animation.Array.data[{j}].frameDuration" in a:
            durs.append(a[f"animation.Array.data[{j}].frameDuration"])
            j += 1
        anims[a.get("name")] = durs
    return anims


def guid_of(ref):
    """GlobalObjectId_V1-1-<guid>-... -> guid"""
    if ref is None:
        return None
    m = re.match(r"GlobalObjectId_V1-\d+-([0-9a-f]{32})-", ref)
    return m.group(1) if m else None


def extract_map(metadata):
    index = metadata["level"]
    if metadata["file"] != f"map-{index}.json":
        raise ValueError(f"map catalog level/file mismatch: {metadata}")
    scene = load(metadata["file"])
    platforms = []
    water = None
    session = None
    spawners = []
    scene_components = set()
    for node in walk(scene):
        scene_components.update(c["type"] for c in node.get("components", []))
        srr = node_component(node, "StickyRoundedRectangle")
        if srr is not None:
            ft = node_component(node, "FixTransform")
            rr = node_component(node, "DPhysicsRoundedRect")
            av = node_component(node, "AnimateVelocity")
            body = node_component(node, "BoplBody")
            anti = node_component(node, "AntiLockPlatform")
            grow = node_component(node, "GrowOnStart")
            vector_field = node_component(node, "VectorFieldPlatform")
            entry = {
                "name": node.get("name"),
                "intro_order": len(platforms),
                "position": vec2(ft, "_position"),
                "rotation": fix(ft, "_rotation"),
                "scale": ft.get("_scale.m_rawValue", ONE),
                "extents": vec2(rr, "startExtents"),
                "radius": fix(rr, "startRadius") if "startRadius.m_rawValue" in rr else fix(rr, "radius"),
                "min_scale": fix(rr, "MinScale"),
                "max_scale": fix(rr, "MaxScale"),
                "platform_type": srr["platformType"],
                "grow_scale": fix(grow, "scaleUp") if grow is not None else ONE,
                "layer": node.get("layer"),
                "hierarchy": rr["hierarchyNumber"],
                "updaters": updater_order(node),
                # Arrow.OnCollide's childCount > 80 counts the platform's own
                # authored children too (Arrow.cs:104).
                "child_count": len(node.get("children", [])),
                # What DuplicateObject instantiates: GrowOnStart.Init destroys
                # itself before any Duplicator can copy the platform.
                "updatable_count": sum(
                    "hierarchyNumber" in c.get("properties", {}) and c["type"] != "GrowOnStart"
                    for descendant in walk([node])
                    for c in descendant.get("components", [])
                ),
                # The scene's own numbers in registration order, and a copy's
                # (registrations above).
                "registrations": registrations({"data": node}, exclude=("GrowOnStart",)),
                "copy_registrations": registrations({"data": node}, copy=True, exclude=("GrowOnStart",)),
                "duplicate_collider_hierarchy": 1 + [
                    typ for _, typ in sorted(
                        (c["properties"]["hierarchyNumber"], c["type"])
                        for descendant in walk([node])
                        for c in descendant.get("components", [])
                        if "hierarchyNumber" in c.get("properties", {})
                        and c["type"] != "GrowOnStart"
                    )
                ].index("DPhysicsRoundedRect"),
                "animate_velocity": {
                    "speed": fix(av, "speed"),
                    "mu": fix(av, "mu"),
                    "rotation_speed": fix(av, "rotationSpeed"),
                    "rotation_mu": fix(av, "rotationMu"),
                    "interpolate_rotation_home": av["interpolateRotationHome"],
                    "sudden_death_speed": fix(av, "suddenDeathSpeed"),
                    "sudden_death_max_delay": fix(av, "MaxRandomSuddenDeathDelay"),
                    "sudden_death_acceleration": fix(av, "SuddenDeathAcceleration"),
                    "sudden_death_y_speed": curve(av, "suddenDeathYSpeed"),
                },
                "body": physics_body(body),
            }
            if anti is not None:
                n = anti.get("OrbitPath.Array.size", 0)
                entry["orbit"] = {
                    "force": fix(anti, "OrbitForce"),
                    "delay": fix(anti, "DelaySeconds"),
                    # The force-driven motion needs the body dynamics too:
                    # velocity += (spring + force/mass) * dt, then *= airDrag.
                    "mass": fix(body, "mass"),
                    "air_drag": fix(body, "airFriction01"),
                    "path": [vec2(anti, f"OrbitPath.Array.data[{i}]") for i in range(n)],
                }
            if vector_field is not None:
                entry["vector_field"] = {
                    "delay": fix(vector_field, "DelaySeconds"),
                    "orbit_speed": fix(vector_field, "orbitSpeed"),
                    "expand_speed": fix(vector_field, "expandSpeed"),
                    "center": vec2(vector_field, "centerPoint"),
                    "normal_speed_friction": fix(vector_field, "normalSpeedFriction"),
                    "dead_zone_dist_squared": fix(vector_field, "DeadZoneDist"),
                    "orbit_acceleration_multiplier": fix(vector_field, "OrbitAccelerationMulitplier"),
                    "target_radius": fix(vector_field, "targetRadius"),
                    "ovalness": fix(vector_field, "ovalness01"),
                }
            platforms.append(entry)
        if node.get("name") == "WaterHitBox":
            ft = node_component(node, "FixTransform")
            box = node_component(node, "DPhysicsBox")
            water = {
                "position": vec2(ft, "_position"),
                "extents": vec2(box, "startExtents"),
                "layer": node.get("layer"),
                "hierarchy": box["hierarchyNumber"],
            }
        g = node_component(node, "GameSessionHandler")
        if g is not None:
            n = g.get("teamSpawns.Array.size", 0)
            session = {
                "team_spawns": [vec2(g, f"teamSpawns.Array.data[{i}]") for i in range(n)],
                "teammate_spacing": fix(g, "teammateSpawnSpacing"),
                "level_type": g.get("levelType"),
                "time_before_sudden_death": fix(g, "TimeBeforeSuddenDeath"),
                "level_animation_radius": fix(g, "levelAnimationRadius"),
                "max_random_delay": fix(g, "MaxRandomDelay"),
                "level_animation_time": fix(g, "LevelAnimationTime"),
                "start_animation_progress": fix(g, "startProgressOfAnimateInGrounds"),
                "animate_in": curve(g, "AnimateInOut"),
            }
        sp = node_component(node, "AbilitySpawner")
        if sp is not None:
            ft = node_component(node, "FixTransform")
            spawners.append({
                "position": vec2(ft, "_position"),
                "hierarchy": sp["hierarchyNumber"],
            })
    if session is None or not spawners:
        raise ValueError(f"map {index} is missing its session or ability spawner")
    if session["level_type"] != 2 and water is None:
        raise ValueError(f"non-space map {index} is missing WaterHitBox")
    # Updater/DetPhysics registration is serialized hierarchy order, not YAML
    # transform traversal order. Most scenes happen to share both orders, but
    # nested 2v2 scene roots do not.
    platforms.sort(key=lambda platform: platform["hierarchy"])
    spawners.sort(key=lambda spawner: spawner["hierarchy"])
    return {
        "level": index,
        "scene": metadata["scene"],
        "path": metadata["path"],
        "platforms": platforms,
        "water": water,
        "session": session,
        "ability_spawners": spawners,
        "components": sorted(scene_components),
    }


def main():
    out = {"reference": load("manifest.json")["reference"], "maps": [], "prefabs": {}}
    for expected, metadata in enumerate(MAPS):
        if metadata["level"] != expected:
            raise ValueError(f"map catalog is not dense at {expected}: {metadata}")
        out["maps"].append(extract_map(metadata))

    # ---- Player (slime) ----
    _, player = find_prefab("Assets/GameObject/Player.prefab")
    phys = component(player, "PlayerPhysics")
    slime_ctl = component(player, "SlimeController")
    box = component(player, "DPhysicsBox")
    pcol = component(player, "PlayerCollision")
    player_bounds = component(player, "DestroyIfOutsideSceneBounds")
    boundary_effect = ASSETS[guid_of(player_bounds["DownEffect"])]
    anim_guid = guid_of(slime_ctl.get("animData"))
    slime_anims = animations(anim_guid)
    out["prefabs"]["player"] = {
        "layer": player["data"].get("layer"),
        "physics": {k: v for k, v in {
            "max_speed": fix(phys, "maxSpeed"),
            "jump_strength": fix(phys, "jumpStrength"),
            "accel": fix(phys, "accel"),
            "radius": fix(phys, "radius"),
            "gravity_accel": fix(phys, "gravity_accel"),
            "gravity_max_fall_speed": fix(phys, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(phys, "gravity_modifier"),
            "jump_extra_x": fix(phys, "jumpExtraXStrength"),
            "jump_momentum": fix(phys, "jumpKeptMomentum"),
            "jump_normal_scale": fix(phys, "jumpNormalScaleFactor"),
            "jump_teleport": fix(phys, "extraJumpTeleportMultiplier"),
            "air_accel": fix(phys, "airAccel"),
        }.items()},
        "hurtbox_extents": vec2(box, "startExtents"),
        "sync_rotation_to_platform": phys["SyncRotationToPlatform"],
        "spawn_anim_ticks": sum(abs(d) for d in slime_anims["spawn"]),
        # SlimeController.ReSpawnFast uses SpeedUpAnimation(anim, 4): C#
        # integer division per frame, with every duration clamped to >= 1.
        "fast_spawn_anim_ticks": sum(max(1, abs(d) // 4) for d in slime_anims["spawn"]),
        "boundary_effect_updatable_count": updater_count(boundary_effect),
        "land_anim_ticks": sum(abs(d) for d in slime_anims["land"]),
        "anims": {k: v for k, v in slime_anims.items() if k in (
            "spawn", "land", "stop", "runTurnAround", "idle", "run")},
    }

    # ---- Dash ----
    _, dash = find_prefab("Assets/GameObject/Dash.prefab")
    dash_ability = component(dash, "Ability")
    dash_box = component(dash, "DPhysicsBox")
    dash_tf = component(dash, "Dash")
    out["prefabs"]["dash"] = {
        "layer": dash["data"].get("layer"),
        "cooldown": fix(dash_ability, "Cooldown"),
        "hurtbox_extents": vec2(dash_box, "startExtents"),
        "fields": {k: f32(v) if isinstance(v, dict) else v
                   for k, v in dash_tf.items() if not k.startswith("m_")
                   and not isinstance(v, dict) or (isinstance(v, dict) and "f32" in v)},
    }
    # keep exact Fix raws for DashTransform where present
    out["prefabs"]["dash"]["raw_fields"] = {
        k: v for k, v in dash_tf.items() if k.endswith("m_rawValue")}

    # ---- Bow / Arrow ----
    _, bow = find_prefab("Assets/GameObject/Bow.prefab")
    bow_ability = component(bow, "Ability")
    bow_tf = component(bow, "BowTransform")
    bow_box = component(bow, "DPhysicsBox")
    bow_phys = component(bow, "PlayerPhysics")
    out["prefabs"]["bow"] = {
        "layer": bow["data"].get("layer"),
        "cooldown": fix(bow_ability, "Cooldown"),
        "hurtbox_extents": vec2(bow_box, "startExtents"),
        "knockback": bow_tf["KnockBack"],
        "arrow_speed": fix(bow_tf, "ArrowSpeed"),
        "firepoint_offset": vec2(bow_tf, "FirepointOffset"),
        "aireal_control_blend": fix(bow_tf, "airealControlBlend"),
        "time_before_arrows_hurt_owner": fix(bow_tf, "TimeBeforeArrowsHurtOwner"),
        "max_slow_mo_speed": fix(bow_tf, "maxSlowMoSpeed"),
        "max_arrows": bow_tf["maxNumberOfArrows"],
        "physics": {
            "max_speed": fix(bow_phys, "maxSpeed"),
            "accel": fix(bow_phys, "accel"),
            "air_accel": fix(bow_phys, "airAccel"),
            "radius": fix(bow_phys, "radius"),
            "gravity_accel": fix(bow_phys, "gravity_accel"),
            "gravity_max_fall_speed": fix(bow_phys, "gravity_maxFallSpeed"),
            "jump_strength": fix(bow_phys, "jumpStrength"),
        },
        "anims": animations(guid_of(bow_tf.get("animData"))),
    }
    arrow_guid = guid_of(bow_tf.get("Arrow"))
    arrow = ASSETS[arrow_guid]
    arrow_body = component(arrow, "BoplBody")
    arrow_circle = component(arrow, "DPhysicsCircle")
    out["prefabs"]["arrow"] = {
        "layer": arrow["data"].get("layer"),
        "radius": fix(arrow_circle, "startRadius") if "startRadius.m_rawValue" in arrow_circle else fix(arrow_circle, "radius"),
        "collider_hierarchy": arrow_circle["hierarchyNumber"],
        "arrow_hierarchy": component(arrow, "Arrow")["hierarchyNumber"],
        "updatable_count": updater_count(arrow),
        "body": {k[:-len(".m_rawValue")]: v for k, v in arrow_body.items()
                 if k.endswith("m_rawValue")},
    }

    # ---- Dropped-ability pickup (SlimeController.DropAbilities) ----
    pickup = ASSETS[guid_of(slime_ctl.get("abilityPickupPrefab"))]
    pickup_body = component(pickup, "BoplBody")
    pickup_circle = component(pickup, "DPhysicsCircle")
    pickup_tf = component(pickup, "FixTransform")
    out["prefabs"]["ability_pickup"] = {
        "layer": pickup["data"].get("layer"),
        "radius": fix(pickup_circle, "startRadius"),
        "offset": vec2(pickup_tf, "offset"),
        "collider_hierarchy": pickup_circle["hierarchyNumber"],
        "updatable_count": sum(
            "hierarchyNumber" in c.get("properties", {})
            for c in pickup["data"]["components"]
        ),
        "body": physics_body(pickup_body),
        "up_force": fix(component(pickup, "DynamicAbilityPickup"), "UpForce"),
        "wobble_speed": fix(component(pickup, "DynamicAbilityPickup"), "wobbleSpeed"),
        "wobble_amount": fix(component(pickup, "DynamicAbilityPickup"), "wobbleAmount"),
        "launch_force": fix(component(pickup, "DynamicAbilityPickup"), "launchForce"),
        "disabled_seconds": fix(component(pickup, "DynamicAbilityPickup"), "disabledSeconds"),
        "swap_period": fix(component(pickup, "DynamicAbilityPickup"), "SwapAbilityPeriod"),
        "vibration_strength": fix(component(pickup, "DynamicAbilityPickup"), "vibrationStr"),
        "vibration_duration": fix(component(pickup, "DynamicAbilityPickup"), "vibrationDuration"),
        "vibrate_timing_offset": fix(component(pickup, "DynamicAbilityPickup"), "vibrateTimingOffset"),
        "gravity": fix(component(pickup, "DynamicAbilityPickup"), "gravity"),
    }

    # ---- Platform ability ----
    _, plat = find_prefab("Assets/GameObject/Platform.prefab")
    plat_ability = component(plat, "Ability")
    ptf = component(plat, "PlatformTransform")
    plat_circle = component(plat, "DPhysicsCircle")
    out["prefabs"]["platform_ability"] = {
        "layer": plat["data"].get("layer"),
        "cooldown": fix(plat_ability, "Cooldown"),
        "hurtbox_radius": fix(plat_circle, "startRadius") if "startRadius.m_rawValue" in plat_circle else fix(plat_circle, "radius"),
        "time_in_exit": fix(ptf, "TimeInExit"),
        "size_change_speed": fix(ptf, "SizeChangeSpeed"),
        "max_size": fix(ptf, "MaxSize"),
        "min_size": fix(ptf, "MinSize"),
        "rotation_speed": fix(ptf, "rotationSpeed"),
        "max_area": fix(ptf, "maxArea"),
        "max_size_time": fix(ptf, "maxSizeTime"),
        "face_height": fix(ptf, "faceHeight"),
        "ejection_offset": fix(ptf, "EjectionOffset"),
        "ejection_curve": curve(ptf, "EjectionCurve01"),
        "enter_scale_curve": curve(ptf, "EnterScaleAnimation"),
        "anims": animations(guid_of(ptf.get("animData"))),
    }
    rp_guid = guid_of(ptf.get("platformPrefab"))
    rp = ASSETS[rp_guid]
    rp_rr = component(rp, "DPhysicsRoundedRect")
    rp_av = component(rp, "AnimateVelocity")
    rp_res = component(rp, "ResizablePlatform")
    rp_body = component(rp, "BoplBody")
    out["prefabs"]["resizable_platform"] = {
        "layer": rp["data"].get("layer"),
        "extents": vec2(rp_rr, "startExtents"),
        "radius": fix(rp_rr, "startRadius") if "startRadius.m_rawValue" in rp_rr else fix(rp_rr, "radius"),
        "min_scale": fix(rp_rr, "MinScale"),
        "max_scale": fix(rp_rr, "MaxScale"),
        "animate_velocity": {
            "speed": fix(rp_av, "speed"),
            "mu": fix(rp_av, "mu"),
            "rotation_speed": fix(rp_av, "rotationSpeed"),
            "rotation_mu": fix(rp_av, "rotationMu"),
            "interpolate_rotation_home": rp_av["interpolateRotationHome"],
            "sudden_death_speed": fix(rp_av, "suddenDeathSpeed"),
            "sudden_death_max_delay": fix(rp_av, "MaxRandomSuddenDeathDelay"),
            "sudden_death_acceleration": fix(rp_av, "SuddenDeathAcceleration"),
            "sudden_death_y_speed": curve(rp_av, "suddenDeathYSpeed"),
        },
        "body": physics_body(rp_body),
        "collider_hierarchy": rp_rr["hierarchyNumber"],
        "updatable_count": registering_count(rp),
        "updaters": updater_order(rp["data"]),
        "mass_per_area": fix(rp_res, "MassPerArea"),
        "remove_curve": curve(rp_res, "RemoveAnimation"),
        "resizable_keys": sorted(k for k in rp_res if not k.startswith("m_")),
    }

    # ---- Black Hole casting form and persistent rigid-body affector ----
    _, black_form = find_prefab("Assets/GameObject/Black hole.prefab")
    black_clap = component(black_form, "BlackHoleClap")
    black_box = component(black_form, "DPhysicsBox")
    black_fix = component(black_form, "FixTransform")
    black_physics = component(black_form, "PlayerPhysics")
    black_hole = ASSETS[guid_of(black_clap["blackHolePrefab"])]
    black = component(black_hole, "BlackHole")
    black_circle = component(black_hole, "DPhysicsCircle")
    out["prefabs"]["black_hole"] = {
        "hurtbox_extents": vec2(black_box, "startExtents"),
        "collider_hierarchy": black_box["hierarchyNumber"],
        "collider_offset": vec2(black_fix, "offset"),
        "physics": {
            "max_speed": fix(black_physics, "maxSpeed"),
            "accel": fix(black_physics, "accel"),
            "air_accel": fix(black_physics, "airAccel"),
            "radius": fix(black_physics, "radius"),
            "gravity_accel": fix(black_physics, "gravity_accel"),
            "gravity_max_fall_speed": fix(black_physics, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(black_physics, "gravity_modifier"),
            "jump_strength": fix(black_physics, "jumpStrength"),
        },
        "animations": animations(guid_of(black_clap["animData"])),
        "spawn_offset": vec2(black_clap, "blackholeOffset"),
        "hole": {
            "layer": black_hole["data"].get("layer"),
            "updatable_count": updater_count(black_hole),
            "collider_hierarchy": black_circle["hierarchyNumber"],
            "start_radius": fix(black_circle, "startRadius"),
            "starting_mass": fix(black, "startingMass"),
            "gravity": fix(black, "G"),
            "controlled_platform_multiplier": fix(black, "platformMultiplier"),
            "growth": fix(black, "growth"),
            "base_growth": fix(black, "baseGrowth"),
            "merge_growth": fix(black, "ExtraBlackHoleMergeGrowth"),
            "size_curve": curve(black, "SizeGrowthAnim"),
            "min_size": fix(black, "minSize"),
            "drag": fix(black, "drag01"),
            "player_force": fix(black, "playerForce"),
            "influence_radius_multiplier": fix(black, "influenceRadiusMultiplier"),
        },
    }

    # ---- Gust / QuickSpell ----
    _, gust = find_prefab("Assets/GameObject/Gust.prefab")
    gust_quick = component(gust, "QuickSpell")
    gust_circle = component(gust, "DPhysicsCircle")
    gust_fix = component(gust, "FixTransform")
    gust_spell = ASSETS[guid_of(gust_quick["spell"])]
    gust_wave = component(gust_spell, "Shockwave")
    out["prefabs"]["gust"] = {
        "anims": animations(guid_of(gust_quick["animData"])),
        "shockwave_offset": vec2(gust_quick, "shockwaveOffset"),
        "collider_offset": vec2(gust_fix, "offset"),
        "hurtbox_radius": fix(gust_circle, "startRadius"),
        "shockwave_updatable_count": updater_count(gust_spell),
        "platform_force": fix(gust_wave, "platformForce"),
        "default_force": fix(gust_wave, "defaultForce"),
        "grounded_multiplier": fix(gust_wave, "groundedPlayerForceMultiplier"),
        "radius": fix(gust_wave, "radius"),
        "curve": curve(gust_wave, "forceCurve01"),
    }

    # ---- Revival / QuickSpell ----
    _, revival = find_prefab("Assets/GameObject/Revival.prefab")
    revival_quick = component(revival, "QuickSpell")
    revival_circle = component(revival, "DPhysicsCircle")
    revival_fix = component(revival, "FixTransform")
    revival_marker = ASSETS[guid_of(revival_quick["spell"])]
    marker_circle = component(revival_marker, "DPhysicsCircle")
    marker_fix = component(revival_marker, "FixTransform")
    marker_body = component(revival_marker, "BoplBody")
    revival_effect = ASSETS[guid_of(component(revival, "PlayerCollision")["reviveEffectPrefab"])]
    effect_config = component(revival_effect, "ReviveEffect")
    particle = ASSETS[guid_of(effect_config["Particle"])]
    particle_config = component(particle, "ReviveParticle")
    death_effect = ASSETS[guid_of(component(revival, "PlayerCollision")["deathEffect"])]
    def hierarchy_span(prefab):
        def visit(game_object):
            own = (
                c.get("properties", {}).get("hierarchyNumber", 0)
                for c in game_object.get("components", [])
            )
            return max([*own, *(visit(child) for child in game_object.get("children", []))])
        return visit(prefab["data"])
    out["prefabs"]["revival"] = {
        "anims": animations(guid_of(revival_quick["animData"])),
        "shockwave_offset": vec2(revival_quick, "shockwaveOffset"),
        "collider_offset": vec2(revival_fix, "offset"),
        "hurtbox_radius": fix(revival_circle, "startRadius"),
        "marker": {
            "layer": revival_marker["data"].get("layer"),
            "updatable_count": 3,
            "collider_hierarchy": marker_circle["hierarchyNumber"],
            "offset": vec2(marker_fix, "offset"),
            "radius": fix(marker_circle, "startRadius"),
            "body": physics_body(marker_body),
        },
        "death_effect_updatable_count": updater_count(death_effect),
        "effect_updatable_count": updater_count(revival_effect),
        # ReviveParticle registers itself again in InitRevive, after
        # EndRegisterPrefab, so that second entry does not advance prefabsAdded.
        "particle_updatable_count": updater_count(particle),
        "effect_wisp_ticks": sum(abs(t) for t in animations(guid_of(effect_config["animData"]))["wisp"]),
        "particle_speed": fix(particle_config, "speed"),
    }

    # ---- Teleport / persistent portal indicator ----
    _, teleport = find_prefab("Assets/GameObject/Teleport.prefab")
    teleport_spell = component(teleport, "Teleport")
    teleport_marker = ASSETS[guid_of(teleport_spell["tpiPrefab"])]
    teleport_indicator = component(teleport_marker, "TeleportIndicator")
    teleport_circle = component(teleport_marker, "DPhysicsCircle")
    teleport_fix = component(teleport_marker, "FixTransform")
    teleport_body = component(teleport_marker, "BoplBody")
    teleport_effect = ASSETS[guid_of(teleport_indicator["teleportingSpriteEffectPrefab"])]
    teleport_effect_config = component(teleport_effect, "TeleportedObjectEffect")
    out["prefabs"]["teleport"] = {
        "up_cast_offset": fix(teleport_spell, "upCastOffset"),
        "player_radius": fix(teleport_spell, "radius"),
        "teleportable_mask": teleport_indicator["teleportableThings.m_Bits"],
        "marker": {
            "layer": teleport_marker["data"].get("layer"),
            "updatable_count": updater_count(teleport_marker),
            "collider_hierarchy": teleport_circle["hierarchyNumber"],
            "offset": vec2(teleport_fix, "offset"),
            "radius": fix(teleport_circle, "startRadius"),
            "teleport_radius": fix(teleport_indicator, "teleportRadius"),
            "body": physics_body(teleport_body),
        },
        "effect_updatable_count": updater_count(teleport_effect),
        "effect_duration": fix(teleport_effect_config, "duration"),
    }

    # ---- Portal gun, its dynamic shot, and retained linked endpoints ----
    _, portal_gun_prefab = find_prefab("Assets/GameObject/Portal gun.prefab")
    portal_gun = component(portal_gun_prefab, "PortalGun")
    portal_shot_prefab = ASSETS[guid_of(portal_gun["shotPrefab"])]
    portal_shot = component(portal_shot_prefab, "PortalShot")
    portal_shot_body = component(portal_shot_prefab, "BoplBody")
    portal_shot_circle = component(portal_shot_prefab, "DPhysicsCircle")
    portal_prefab = ASSETS[guid_of(portal_gun["portalPrefab"])]
    portal = component(portal_prefab, "Portal")
    out["prefabs"]["portal_gun"] = {
        "shot_speed": fix(portal_gun, "shotSpeed"),
        "spawn_offset": fix(portal_gun, "spawnOffset"),
        "shot_layer": portal_shot_prefab["data"].get("layer"),
        "shot_updatable_count": updater_count(portal_shot_prefab),
        "shot_child_count": collider_child_count(portal_shot_prefab),
        "shot_collider_hierarchy": portal_shot_circle["hierarchyNumber"],
        "shot_radius": fix(portal_shot_circle, "startRadius"),
        "shot_body": physics_body(portal_shot_body),
        "max_flight_seconds": fix(portal_shot, "maxFlightSeconds"),
        "portal_updatable_count": updater_count(portal_prefab),
        "half_width": fix(portal, "halfWidth"),
        "thickness": fix(portal, "thickness"),
        "exit_offset": fix(portal, "exitOffset"),
        "lifetime_seconds": fix(portal, "lifetimeSeconds"),
    }

    # ---- Grenade / shared ThrowItem2 item path ----
    _, grenade_ability = find_prefab("Assets/GameObject/Grenade_0.prefab")
    grenade_throw = component(grenade_ability, "ThrowItem2")
    grenade_box = component(grenade_ability, "DPhysicsBox")
    grenade_phys = component(grenade_ability, "PlayerPhysics")
    grenade_anims = animations(guid_of(grenade_throw["animData"]))
    grenade_item = ASSETS[guid_of(grenade_throw["ItemPrefab"])]
    grenade = component(grenade_item, "Grenade")
    grenade_item_data = component(grenade_item, "Item")
    grenade_circle = component(grenade_item, "DPhysicsCircle")
    grenade_body = component(grenade_item, "BoplBody")
    grenade_explode = component(grenade_item, "GrenadeExplode")
    grenade_explosion = ASSETS[guid_of(grenade_explode["explosion"])]
    explosion = component(grenade_explosion, "Explosion")
    explosion_box = component(grenade_explosion, "DPhysicsBox")
    explosion_waves = [c["properties"]
                       for c in component(grenade_explosion, "Shockwave", all_matches=True)]
    assert len(explosion_waves) == 2
    explosion_anims = animations(guid_of(explosion["explosionAnimation"]))
    out["prefabs"]["grenade"] = {
        "hurtbox_extents": vec2(grenade_box, "startExtents"),
        "throw": {
            "air_position": vec2(grenade_throw, "aThrowPosition"),
            "ground_position": vec2(grenade_throw, "gThrowPosition"),
            "ground_dummy": [vec2(grenade_throw, f"groundedDummyPositions.Array.data[{i}]")
                             for i in range(grenade_throw["groundedDummyPositions.Array.size"])],
            "air_dummy": [vec2(grenade_throw, f"airealDummyPositions.Array.data[{i}]")
                          for i in range(grenade_throw["airealDummyPositions.Array.size"])],
            "hurt_owner_delay": fix(grenade_throw, "DisableGrenadeHitBoxTime"),
            "max_force": fix(grenade_throw, "maxThrowForce"),
            "min_force": fix(grenade_throw, "minThrowForce"),
            "force_gain_speed": fix(grenade_throw, "ThrowForceGainSpeed"),
            "max_slow_mo_speed": fix(grenade_throw, "maxSlowMoSpeed"),
            "knockback": fix(grenade_throw, "knockback"),
            "physics": {
                "max_speed": fix(grenade_phys, "maxSpeed"),
                "jump_strength": fix(grenade_phys, "jumpStrength"),
                "accel": fix(grenade_phys, "accel"),
                "radius": fix(grenade_phys, "radius"),
                "gravity_accel": fix(grenade_phys, "gravity_accel"),
                "gravity_max_fall_speed": fix(grenade_phys, "gravity_maxFallSpeed"),
                "gravity_modifier": fix(grenade_phys, "gravity_modifier"),
                "jump_extra_x": fix(grenade_phys, "jumpExtraXStrength"),
                "jump_momentum": fix(grenade_phys, "jumpKeptMomentum"),
                "jump_normal_scale": fix(grenade_phys, "jumpNormalScaleFactor"),
                "jump_teleport": fix(grenade_phys, "extraJumpTeleportMultiplier"),
                "air_accel": fix(grenade_phys, "airAccel"),
            },
            "animations": {name: grenade_anims[name]
                           for name in ("gEnter", "gExit", "aEnter", "aExit")},
        },
        "item": {
            "layer": grenade_item["data"].get("layer"),
            "updatable_count": updater_count(grenade_item),
            "child_count": collider_child_count(grenade_item),
            "collider_hierarchy": grenade_circle["hierarchyNumber"],
            "radius": fix(grenade_circle, "startRadius"),
            "body": physics_body(grenade_body),
            "detonation_time": fix(grenade, "detonationTime"),
            "clone_count": grenade_item_data["nrOfClonesWhenDuplicated"],
            "duplicate_pattern": grenade_item_data["duplicatePattern"],
            "duplicate_spacing": fix(grenade_item_data, "duplicate_spacing"),
            "duplicate_angle": fix(grenade_item_data, "duplicate_angle"),
        },
        "explosion": {
            "layer": grenade_explosion["data"].get("layer"),
            "updatable_count": updater_count(grenade_explosion),
            "collider_hierarchy": explosion_box["hierarchyNumber"],
            "extents": vec2(explosion_box, "startExtents"),
            "animation_ticks": sum(abs(d) for d in explosion_anims[next(iter(explosion_anims))]),
            "shockwaves": [{
                "platform_force": fix(wave, "platformForce"),
                "default_force": fix(wave, "defaultForce"),
                "grounded_player_multiplier": fix(wave, "groundedPlayerForceMultiplier"),
                "radius": fix(wave, "radius"),
                "collision_mask": wave["collisionMask.m_Bits"],
            } for wave in explosion_waves],
        },
    }

    # ---- Missile / SparkShot guided projectile ----
    _, missile_ability = find_prefab("Assets/GameObject/Missile.prefab")
    spark_shot = component(missile_ability, "SparkShot")
    missile_box = component(missile_ability, "DPhysicsBox")
    missile_phys = component(missile_ability, "PlayerPhysics")
    missile_anims = animations(guid_of(spark_shot["animData"]))
    spark = ASSETS[guid_of(spark_shot["SparkPrefab"])]
    spark_missile = component(spark, "Missile")
    spark_item = component(spark, "Item")
    spark_box = component(spark, "DPhysicsBox")
    spark_body = component(spark, "BoplBody")
    missile_explosion = ASSETS[guid_of(spark_missile["onHitExplosionPrefab"])]
    missile_explosion_component = component(missile_explosion, "Explosion")
    missile_explosion_box = component(missile_explosion, "DPhysicsBox")
    missile_explosion_waves = [c["properties"] for c in
                               component(missile_explosion, "Shockwave", all_matches=True)]
    assert len(missile_explosion_waves) == 1
    missile_explosion_anims = animations(
        guid_of(missile_explosion_component["explosionAnimation"])
    )
    out["prefabs"]["missile"] = {
        "hurtbox_extents": vec2(missile_box, "startExtents"),
        "projectile_offset": vec2(spark_shot, "projectileOffset"),
        "projectile_start_speed": fix(spark_shot, "projectileStartSpeed"),
        "rotation_speed": fix(spark_shot, "SparkRotationSpeed"),
        "physics": {
            "max_speed": fix(missile_phys, "maxSpeed"),
            "jump_strength": fix(missile_phys, "jumpStrength"),
            "accel": fix(missile_phys, "accel"),
            "radius": fix(missile_phys, "radius"),
            "gravity_accel": fix(missile_phys, "gravity_accel"),
            "gravity_max_fall_speed": fix(missile_phys, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(missile_phys, "gravity_modifier"),
            "jump_extra_x": fix(missile_phys, "jumpExtraXStrength"),
            "jump_momentum": fix(missile_phys, "jumpKeptMomentum"),
            "jump_normal_scale": fix(missile_phys, "jumpNormalScaleFactor"),
            "jump_teleport": fix(missile_phys, "extraJumpTeleportMultiplier"),
            "air_accel": fix(missile_phys, "airAccel"),
        },
        "animations": {name: missile_anims[name]
                       for name in ("enter", "enterToIdle", "idle")},
        "spark": {
            "layer": spark["data"].get("layer"),
            "updatable_count": sum(
                "hierarchyNumber" in c.get("properties", {})
                for node in [spark["data"], *walk(spark["data"].get("children", []))]
                for c in node.get("components", [])
            ),
            "collider_hierarchy": spark_box["hierarchyNumber"],
            "child_count": collider_child_count(spark),
            "extents": vec2(spark_box, "startExtents"),
            "body": physics_body(spark_body),
            "minimum_forward_speed": fix(spark_missile, "minimumForwardsSpeed"),
            "accel": fix(spark_missile, "accel"),
            "instability_accel": fix(spark_missile, "instabilityAccel"),
            "clone_count": spark_item["nrOfClonesWhenDuplicated"],
            "duplicate_pattern": spark_item["duplicatePattern"],
            "duplicate_spacing": fix(spark_item, "duplicate_spacing"),
            "duplicate_angle": fix(spark_item, "duplicate_angle"),
        },
        "explosion": {
            "layer": missile_explosion["data"].get("layer"),
            "updatable_count": updater_count(missile_explosion),
            "collider_hierarchy": missile_explosion_box["hierarchyNumber"],
            "extents": vec2(missile_explosion_box, "startExtents"),
            "animation_ticks": sum(abs(d) for d in
                                   missile_explosion_anims[next(iter(missile_explosion_anims))]),
            "shockwave": {
                "platform_force": fix(missile_explosion_waves[0], "platformForce"),
                "default_force": fix(missile_explosion_waves[0], "defaultForce"),
                "grounded_player_multiplier": fix(
                    missile_explosion_waves[0], "groundedPlayerForceMultiplier"
                ),
                "radius": fix(missile_explosion_waves[0], "radius"),
                "collision_mask": missile_explosion_waves[0]["collisionMask.m_Bits"],
            },
        },
    }

    # ---- Mine / original ThrowItem path and autonomous mine ----
    _, mine_ability = find_prefab("Assets/GameObject/Mine.prefab")
    mine_throw = component(mine_ability, "ThrowItem")
    mine_box = component(mine_ability, "DPhysicsBox")
    mine_fix = component(mine_ability, "FixTransform")
    mine_phys = component(mine_ability, "PlayerPhysics")
    mine_anims = animations(guid_of(mine_throw["animData"]))
    mine_item = ASSETS[guid_of(mine_throw["GrenadePrefab"])]
    mine_item_data = component(mine_item, "Item")
    mine = component(mine_item, "Mine")
    mine_circle = component(mine_item, "DPhysicsCircle")
    mine_body = component(mine_item, "BoplBody")
    mine_explosion = ASSETS[guid_of(mine["Explosion"])]
    mine_explosion_component = component(mine_explosion, "Explosion")
    mine_explosion_box = component(mine_explosion, "DPhysicsBox")
    mine_explosion_waves = [c["properties"] for c in
                            component(mine_explosion, "Shockwave", all_matches=True)]
    assert len(mine_explosion_waves) == 1
    mine_explosion_anims = animations(
        guid_of(mine_explosion_component["explosionAnimation"])
    )
    out["prefabs"]["mine"] = {
        "hurtbox_extents": vec2(mine_box, "startExtents"),
        "collider_offset": vec2(mine_fix, "offset"),
        "throw": {
            "throw_position": vec2(mine_throw, "throwPosition"),
            "throw_strength": fix(mine_throw, "throwStrength"),
            "aerial_throw_strength": fix(mine_throw, "airealThrowStrength"),
            "knockback": fix(mine_throw, "KnockBack"),
            "min_throw_strength": fix(mine_throw, "minThrowStrength"),
            "only_throw_up": mine_throw["onlyThrowUp"],
            "physics": {
                "max_speed": fix(mine_phys, "maxSpeed"),
                "jump_strength": fix(mine_phys, "jumpStrength"),
                "accel": fix(mine_phys, "accel"),
                "radius": fix(mine_phys, "radius"),
                "gravity_accel": fix(mine_phys, "gravity_accel"),
                "gravity_max_fall_speed": fix(mine_phys, "gravity_maxFallSpeed"),
                "gravity_modifier": fix(mine_phys, "gravity_modifier"),
                "jump_extra_x": fix(mine_phys, "jumpExtraXStrength"),
                "jump_momentum": fix(mine_phys, "jumpKeptMomentum"),
                "jump_normal_scale": fix(mine_phys, "jumpNormalScaleFactor"),
                "jump_teleport": fix(mine_phys, "extraJumpTeleportMultiplier"),
                "air_accel": fix(mine_phys, "airAccel"),
            },
            "animations": {name: mine_anims[name]
                           for name in ("gEnter", "gLoop", "gExit", "aEnter", "aExit")},
        },
        "item": {
            "layer": mine_item["data"].get("layer"),
            "updatable_count": updater_count(mine_item),
            "child_count": collider_child_count(mine_item),
            "collider_hierarchy": mine_circle["hierarchyNumber"],
            "radius": fix(mine_circle, "startRadius"),
            "body": physics_body(mine_body),
            "clone_count": mine_item_data["nrOfClonesWhenDuplicated"],
            "duplicate_pattern": mine_item_data["duplicatePattern"],
            "duplicate_spacing": fix(mine_item_data, "duplicate_spacing"),
            "duplicate_angle": fix(mine_item_data, "duplicate_angle"),
            "hover_strength": fix(mine, "HoverStr"),
            "hover_period": fix(mine, "HoverPeriod"),
            "time_before_armed": fix(mine, "TimeBeforeArmed"),
            "chase_time": fix(mine, "chaseTime"),
            "igniters": mine["Igniters.m_Bits"],
            "scans_for": mine["ScansFor.m_Bits"],
            "chase_strength": fix(mine, "chaseStrength"),
            "scan_radius": fix(mine, "scanRadius"),
            "shake_force": fix(mine, "shakeForce"),
            "anticipation_duration": fix(mine, "anticipationDuration"),
        },
        "explosion": {
            "layer": mine_explosion["data"].get("layer"),
            "updatable_count": updater_count(mine_explosion),
            "collider_hierarchy": mine_explosion_box["hierarchyNumber"],
            "extents": vec2(mine_explosion_box, "startExtents"),
            "animation_ticks": sum(abs(d) for d in
                                   mine_explosion_anims[next(iter(mine_explosion_anims))]),
            "shockwave": {
                "platform_force": fix(mine_explosion_waves[0], "platformForce"),
                "default_force": fix(mine_explosion_waves[0], "defaultForce"),
                "grounded_player_multiplier": fix(
                    mine_explosion_waves[0], "groundedPlayerForceMultiplier"
                ),
                "radius": fix(mine_explosion_waves[0], "radius"),
                "collision_mask": mine_explosion_waves[0]["collisionMask.m_Bits"],
            },
        },
    }

    # ---- Throw / MachoThrow2 platform-to-boulder conversion ----
    _, macho_ability = find_prefab("Assets/GameObject/Throw.prefab")
    macho = component(macho_ability, "MachoThrow2")
    macho_box = component(macho_ability, "DPhysicsBox")
    macho_fix = component(macho_ability, "FixTransform")
    macho_phys = component(macho_ability, "PlayerPhysics")
    macho_anims = animations(guid_of(macho["animData"]))
    boulder = ASSETS[guid_of(macho["boulderPrefab"])]
    boulder_rr = component(boulder, "DPhysicsRoundedRect")
    boulder_body = component(boulder, "BoplBody")
    boulder_logic = component(boulder, "Boulder")
    boulder_bounds = component(boulder, "DestroyIfOutsideSceneBounds")
    boulder_down_effect = ASSETS[guid_of(boulder_bounds["DownEffect"])]
    macho_smoke = ASSETS[guid_of(macho["smokeEffectPrefab"])]
    out["prefabs"]["macho_throw"] = {
        "hurtbox_extents": vec2(macho_box, "startExtents"),
        "collider_offset": vec2(macho_fix, "offset"),
        "collider_hierarchy": macho_box["hierarchyNumber"],
        "physics": {
            "max_speed": fix(macho_phys, "maxSpeed"),
            "jump_strength": fix(macho_phys, "jumpStrength"),
            "accel": fix(macho_phys, "accel"),
            "radius": fix(macho_phys, "radius"),
            "gravity_accel": fix(macho_phys, "gravity_accel"),
            "gravity_max_fall_speed": fix(macho_phys, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(macho_phys, "gravity_modifier"),
            "jump_extra_x": fix(macho_phys, "jumpExtraXStrength"),
            "jump_momentum": fix(macho_phys, "jumpKeptMomentum"),
            "jump_normal_scale": fix(macho_phys, "jumpNormalScaleFactor"),
            "jump_teleport": fix(macho_phys, "extraJumpTeleportMultiplier"),
            "air_accel": fix(macho_phys, "airAccel"),
        },
        "animations": {name: macho_anims[name] for name in (
            "enter", "idle", "idleToWalk", "walk", "beginThrow", "exit",
            "turn", "pickup", "grab",
        )},
        "dummy_positions": [vec2(macho, f"dummyPositions.Array.data[{i}]")
                            for i in range(macho["dummyPositions.Array.size"])],
        "prepare_positions": [vec2(macho, f"dummyPositionsPrepareThrow.Array.data[{i}]")
                              for i in range(macho["dummyPositionsPrepareThrow.Array.size"])],
        "force_from_boulder": fix(macho, "forceFromBoulder"),
        "smoke_updatable_count": updater_count(macho_smoke),
        "player_radius_hack": fix(macho, "playerRadiusHack"),
        "move_after_throw_frames": macho["framesBeforeYouCanMoveAfterThrowing"],
        "jump_after_throw_frames": macho["framesBeforeYouCanJumpAfterThrowing"],
        "rotation_velocity": fix(macho, "rotationVelOnThrow"),
        "ground_angular_multiplier": fix(macho, "angVelMultiplierOnGround01"),
        "max_attachable_distance": fix(macho, "maxDistToPickupAttachable"),
        "shrink_after_frames": macho["framesUntilPlatformShrinks"],
        "projectile_position": vec2(macho, "projectileSpawnPos"),
        "quick_projectile_position": vec2(macho, "projectileSpawnPos_low"),
        "throw_speed": fix(macho, "throwSpeed"),
        "min_throw_vector": vec2(macho, "minThrowVector"),
        "extra_strength_scale": fix(macho, "proportionOfExtraStrFromScale"),
        "quick_throw_strength": fix(macho, "quickThrowStr"),
        "quick_charge_scaling": fix(macho, "quickThrowChargeScaling"),
        "platform_scale_reduction": fix(macho, "scaleReductionOnGroundPlatform"),
        "ground_to_hands": fix(macho, "distfromGroundToHands"),
        "quick_position_swap_frame": macho["spritesBeforeWeSwapToDummyPosition"],
        "boulder": {
            "layer": boulder["data"].get("layer"),
            "updatable_count": updater_count(boulder),
            "collider_hierarchy": boulder_rr["hierarchyNumber"],
            "updaters": updater_order(boulder["data"]),
            "extents": vec2(boulder_rr, "startExtents"),
            "radius": fix(boulder_rr, "startRadius"),
            "min_scale": fix(boulder_rr, "MinScale"),
            "max_scale": fix(boulder_rr, "MaxScale"),
            "body": physics_body(boulder_body),
            "dropped_bounciness": fix(boulder_logic, "bouncinessWhenDropped01"),
            "down_effect_updatable_count": updater_count(boulder_down_effect),
        },
    }

    # ---- Drill player form and its three dynamic lethal hitboxes ----
    _, drill_ability = find_prefab("Assets/GameObject/Drill.prefab")
    drill = component(drill_ability, "Drill")
    drill_box = component(drill_ability, "DPhysicsBox")
    drill_fix = component(drill_ability, "FixTransform")
    drill_combo = component(drill_ability, "HitboxCombo")
    drill_anims = animations(guid_of(drill["animData"]))
    drill_hitbox = ASSETS[guid_of(drill_combo["hitboxPrefab"])]
    out["prefabs"]["drill"] = {
        "hurtbox_extents": vec2(drill_box, "startExtents"),
        "collider_offset": vec2(drill_fix, "offset"),
        "collider_hierarchy": drill_box["hierarchyNumber"],
        "enter_ticks": sum(abs(d) for d in drill_anims["enter"]),
        "exit_ticks": sum(abs(d) for d in drill_anims["exit"]),
        "drill_offset": vec2(drill, "drillOffset"),
        "drill_hitbox": vec2(drill, "drillHitbox"),
        "gravity": fix(drill, "gravityStr"),
        "drill_strength": fix(drill, "drillstr"),
        "rotation_speed_air": fix(drill, "rotationSpeedAir"),
        "leave_ground_speed_multiplier": fix(drill, "leaveGroundSpeedMultiplier"),
        "enter_ground_speed_multiplier": fix(drill, "EnterGroundSpeedMultiplier"),
        "speed_to_rotation": curve(drill, "SpeedToRotationSpeed01"),
        "on_enter_offset": vec2(drill, "onEnterOffset"),
        "min_underground_speed": fix(drill, "minUndergroundSpeed"),
        "keep_strong_gravity_seconds": fix(drill, "keepStrongGravitySeconds"),
        "strong_gravity": fix(drill, "strongGravity"),
        "max_early_fall_speed": fix(drill, "maxEarlyFallSpeed"),
        "hitboxes": [
            {
                "offset": vec2(drill_combo, f"hitBoxDefinitions.Array.data[{i}].offset"),
                "extents": vec2(drill_combo, f"hitBoxDefinitions.Array.data[{i}].extents"),
            }
            for i in range(drill_combo["hitBoxDefinitions.Array.size"])
        ],
        "hitbox_layer": drill_hitbox["data"].get("layer"),
        "hitbox_updatable_count": hierarchy_span(drill_hitbox),
        "hitbox_collider_hierarchy": component(drill_hitbox, "DPhysicsBox")["hierarchyNumber"],
    }

    # ---- Rock / BounceBall player form ----
    _, rock_ability = find_prefab("Assets/GameObject/Rock.prefab")
    rock = component(rock_ability, "BounceBall")
    rock_circle = component(rock_ability, "DPhysicsCircle")
    rock_body = component(rock_ability, "BoplBody")
    _, rock_no_player = find_prefab("Assets/GameObject/Rock_noPlayer.prefab")
    out["prefabs"]["rock"] = {
        "child_count": collider_child_count(rock_ability),
        # ShootDuplicator.bounceBallPrefab, the Duplicator's copy of a Rock.
        "no_player_child_count": collider_child_count(rock_no_player),
        "radius": fix(rock_circle, "startRadius"),
        "collider_hierarchy": rock_circle["hierarchyNumber"],
        "body": physics_body(rock_body),
        "max_duration": fix(rock, "maxDuration"),
        "enter_scale": curve(rock, "scaleOnEnter"),
        "is_cancellable": rock["IsCancellable"],
        "exit_time": fix(rock, "exitTime"),
        "exit_scale": curve(rock, "ScaleOnExit"),
        "player_collision_bounce": fix(rock, "playerCollisionBounce"),
        "shockwave_force": fix(rock, "shockwaveForce"),
        "min_rope_length": fix(rock, "minRopeLength"),
        "rope_pull_strength": fix(rock, "ropePullStr"),
        "squared_max_rope_separation": fix(rock, "sqrdMaxRopeSeparation"),
        "rope_player_mass": fix(rock, "RopePlayerMass01"),
    }

    # ---- Roll player form ----
    _, roll_ability = find_prefab("Assets/GameObject/Roll.prefab")
    roll = component(roll_ability, "Roll")
    roll_circle = component(roll_ability, "DPhysicsCircle")
    roll_physics = component(roll_ability, "PlayerPhysics")
    out["prefabs"]["roll"] = {
        "radius": fix(roll_circle, "startRadius"),
        "collider_hierarchy": roll_circle["hierarchyNumber"],
        "physics": {
            "max_speed": fix(roll_physics, "maxSpeed"),
            "jump_strength": fix(roll_physics, "jumpStrength"),
            "accel": fix(roll_physics, "accel"),
            "radius": fix(roll_physics, "radius"),
            "gravity_accel": fix(roll_physics, "gravity_accel"),
            "gravity_max_fall_speed": fix(roll_physics, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(roll_physics, "gravity_modifier"),
            "jump_extra_x": fix(roll_physics, "jumpExtraXStrength"),
            "jump_momentum": fix(roll_physics, "jumpKeptMomentum"),
            "jump_normal_scale": fix(roll_physics, "jumpNormalScaleFactor"),
            "jump_teleport": fix(roll_physics, "extraJumpTeleportMultiplier"),
            "air_accel": fix(roll_physics, "airAccel"),
        },
        "max_speed": fix(roll, "MaxSpeed"),
        "min_speed": fix(roll, "MinSpeed"),
        "speed_curve": curve(roll, "speedCurve"),
        "charge_time": fix(roll, "chargeTime"),
        "dash_time": fix(roll, "dashTime"),
        "move_collision_density": fix(roll, "MoveCollisionDensity"),
        "collision_mask": roll["collidesWith.m_Bits"],
        "platform_turn_speed": fix(roll, "platformTurnSpeed"),
        "early_release_jump_strength": fix(roll, "EarlyReleaseJumpStr"),
        "short_jump_strength": fix(roll, "shortJumpStr"),
        # The trail has a ParticleSystem and FollowTransform, but neither is a
        # MonoUpdatable and therefore neither consumes Updater hierarchy.
        "trail_hierarchy_count": 0,
    }

    # ---- Meteor Smash player form ----
    _, meteor_ability = find_prefab("Assets/GameObject/Meteor.prefab")
    meteor = component(meteor_ability, "MeteorSmash")
    meteor_box = component(meteor_ability, "DPhysicsBox")
    meteor_physics = component(meteor_ability, "PlayerPhysics")
    meteor_transform = component(meteor_ability, "FixTransform")
    meteor_anims = animations(guid_of(meteor["animData"]))
    meteor_shockwave = ASSETS[guid_of(meteor["ShockwavePrefab"])]
    meteor_smear = ASSETS[guid_of(meteor["SmearFramePrefab"])]
    shockwave = component(meteor_shockwave, "Shockwave")
    smear_box = component(meteor_smear, "DPhysicsBox")
    smear_transform = component(meteor_smear, "FixTransform")
    smear_lifetime = component(meteor_smear, "MeteorSmear")
    out["prefabs"]["meteor"] = {
        "hurtbox_extents": vec2(meteor_box, "startExtents"),
        "collider_hierarchy": meteor_box["hierarchyNumber"],
        "collider_offset": vec2(meteor_transform, "offset"),
        "physics": {
            "max_speed": fix(meteor_physics, "maxSpeed"),
            "jump_strength": fix(meteor_physics, "jumpStrength"),
            "accel": fix(meteor_physics, "accel"),
            "radius": fix(meteor_physics, "radius"),
            "gravity_accel": fix(meteor_physics, "gravity_accel"),
            "gravity_max_fall_speed": fix(meteor_physics, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(meteor_physics, "gravity_modifier"),
            "jump_extra_x": fix(meteor_physics, "jumpExtraXStrength"),
            "jump_momentum": fix(meteor_physics, "jumpKeptMomentum"),
            "jump_normal_scale": fix(meteor_physics, "jumpNormalScaleFactor"),
            "jump_teleport": fix(meteor_physics, "extraJumpTeleportMultiplier"),
            "air_accel": fix(meteor_physics, "airAccel"),
        },
        "time_before_drop": fix(meteor, "timeBeforeDrop"),
        "jump_speed": fix(meteor, "jumpSpeed"),
        "hold_speed": fix(meteor, "holdSpeed"),
        "max_hold_time": fix(meteor, "maxHoldTime"),
        "min_hold_time": fix(meteor, "minHoldTime"),
        "speed_curve": curve(meteor, "speedCurve"),
        "vibration_strength": fix(meteor, "vibrationStr"),
        "fixed_updates_in_dash": meteor["FixedUpdatesInDash"],
        "ground_force": fix(meteor, "groundForce"),
        "player_push_force": fix(meteor, "playerPushForce"),
        "shake_strength": fix(meteor, "shakeStr"),
        "shake_duration": fix(meteor, "shakeDuration"),
        "spin_curve": curve(meteor, "spinCurve"),
        "raycast_mask": meteor["raycastMask.m_Bits"],
        "ground_enter_ticks": sum(abs(d) for d in meteor_anims["gEnter"]),
        "air_ticks": sum(abs(d) for d in meteor_anims["air"]),
        "air_start_hold_ticks": sum(abs(d) for d in meteor_anims["aStartHold"]),
        "shockwave": {
            "updatable_count": hierarchy_span(meteor_shockwave),
            "offset": vec2(component(meteor_shockwave, "FixTransform"), "_position"),
            "platform_force": fix(shockwave, "platformForce"),
            "default_force": fix(shockwave, "defaultForce"),
            "grounded_player_multiplier": fix(shockwave, "groundedPlayerForceMultiplier"),
            "radius": fix(shockwave, "radius"),
            "collision_mask": shockwave["collisionMask.m_Bits"],
            "curve": curve(shockwave, "forceCurve01"),
        },
        "smear": {
            "layer": meteor_smear["data"]["layer"],
            "updatable_count": hierarchy_span(meteor_smear),
            "collider_hierarchy": smear_box["hierarchyNumber"],
            "extents": vec2(smear_box, "startExtents"),
            "offset": vec2(smear_transform, "offset"),
            "lifespan_ticks": smear_lifetime["LifespanTicks"],
        },
    }

    # ---- Push / ControlPlatform player form ----
    _, push_ability = find_prefab("Assets/GameObject/Push.prefab")
    push = component(push_ability, "ControlPlatform")
    push_box = component(push_ability, "DPhysicsBox")
    push_physics = component(push_ability, "PlayerPhysics")
    push_fix_transform = component(push_ability, "FixTransform")
    push_anims = animations(guid_of(push["animData"]))
    out["prefabs"]["push"] = {
        "hurtbox_extents": vec2(push_box, "startExtents"),
        "collider_hierarchy": push_box["hierarchyNumber"],
        "collider_offset": vec2(push_fix_transform, "offset"),
        "radius": fix(push_physics, "radius"),
        "grab_ground_ticks": sum(abs(d) for d in push_anims["grabGround"]),
        "platform_force": fix(push, "platformForce"),
        "scene_bound_limit_offset": fix(push, "SceneBoundLimitOffset"),
        "boulder_force_multiplier": fix(push, "boulderForceMul01"),
        "home_position_lerp_speed": fix(push, "homePositionLerpSpeed"),
    }

    # ---- Freeze ray / shared GunTransform form ----
    _, freeze_ability = find_prefab("Assets/GameObject/Freeze ray.prefab")
    freeze_gun = component(freeze_ability, "GunTransform")
    freeze_box = component(freeze_ability, "DPhysicsBox")
    freeze_physics = component(freeze_ability, "PlayerPhysics")
    freeze_fix_transform = component(freeze_ability, "FixTransform")
    freeze_payload = component(freeze_ability, "FreezeRayGun")
    freeze_anims = animations(guid_of(freeze_gun["animData"]))
    _, frozen_prefab = find_prefab("Assets/GameObject/FrozenSolid.prefab")
    frozen = component(frozen_prefab, "FrozenSolid")
    out["prefabs"]["freeze"] = {
        "hurtbox_extents": vec2(freeze_box, "startExtents"),
        "collider_hierarchy": freeze_box["hierarchyNumber"],
        "collider_offset": vec2(freeze_fix_transform, "offset"),
        "radius": fix(freeze_physics, "radius"),
        "physics": {
            "max_speed": fix(freeze_physics, "maxSpeed"),
            "jump_strength": fix(freeze_physics, "jumpStrength"),
            "accel": fix(freeze_physics, "accel"),
            "radius": fix(freeze_physics, "radius"),
            "gravity_accel": fix(freeze_physics, "gravity_accel"),
            "gravity_max_fall_speed": fix(freeze_physics, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(freeze_physics, "gravity_modifier"),
            "jump_extra_x": fix(freeze_physics, "jumpExtraXStrength"),
            "jump_momentum": fix(freeze_physics, "jumpKeptMomentum"),
            "jump_normal_scale": fix(freeze_physics, "jumpNormalScaleFactor"),
            "jump_teleport": fix(freeze_physics, "extraJumpTeleportMultiplier"),
            "air_accel": fix(freeze_physics, "airAccel"),
        },
        "air_control_blend": fix(freeze_gun, "airealControlBlend"),
        "firepoint_offset": vec2(freeze_gun, "FirepointOffset"),
        "knockback": fix(freeze_gun, "KnockBack"),
        "max_slow_mo_speed": fix(freeze_gun, "maxSlowMoSpeed"),
        "transform_ticks": sum(abs(d) for d in freeze_anims["gunTransform"]),
        "fire_ticks": sum(abs(d) for d in freeze_anims["gunTransformFire"]),
        "collision_mask": freeze_payload["collisionMask.m_Bits"],
        "collision_mask_no_water": freeze_payload["collisionMaskNoWater.m_Bits"],
        "freeze_seconds": fix(frozen, "freezeSeconds"),
        "frozen_updatable_count": updater_count(frozen_prefab),
    }

    # ---- Blink gun / shared GunTransform form and QuantumTunnel ----
    _, blink_ability = find_prefab("Assets/GameObject/Blink gun.prefab")
    blink_gun = component(blink_ability, "GunTransform")
    blink_box = component(blink_ability, "DPhysicsBox")
    blink_physics = component(blink_ability, "PlayerPhysics")
    blink_fix_transform = component(blink_ability, "FixTransform")
    blink_payload = component(blink_ability, "ShootQuantum")
    blink_anims = animations(guid_of(blink_gun["animData"]))
    blink_tunnel_prefab = ASSETS[guid_of(blink_payload["QuantumTunnelPrefab"])]
    blink_tunnel = component(blink_tunnel_prefab, "QuantumTunnel")
    out["prefabs"]["blink"] = {
        "hurtbox_extents": vec2(blink_box, "startExtents"),
        "collider_hierarchy": blink_box["hierarchyNumber"],
        "collider_offset": vec2(blink_fix_transform, "offset"),
        "radius": fix(blink_physics, "radius"),
        "physics": {
            "max_speed": fix(blink_physics, "maxSpeed"),
            "jump_strength": fix(blink_physics, "jumpStrength"),
            "accel": fix(blink_physics, "accel"),
            "radius": fix(blink_physics, "radius"),
            "gravity_accel": fix(blink_physics, "gravity_accel"),
            "gravity_max_fall_speed": fix(blink_physics, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(blink_physics, "gravity_modifier"),
            "jump_extra_x": fix(blink_physics, "jumpExtraXStrength"),
            "jump_momentum": fix(blink_physics, "jumpKeptMomentum"),
            "jump_normal_scale": fix(blink_physics, "jumpNormalScaleFactor"),
            "jump_teleport": fix(blink_physics, "extraJumpTeleportMultiplier"),
            "air_accel": fix(blink_physics, "airAccel"),
        },
        "air_control_blend": fix(blink_gun, "airealControlBlend"),
        "firepoint_offset": vec2(blink_gun, "FirepointOffset"),
        "knockback": fix(blink_gun, "KnockBack"),
        "max_slow_mo_speed": fix(blink_gun, "maxSlowMoSpeed"),
        "transform_ticks": sum(abs(d) for d in blink_anims["gunTransform"]),
        "fire_ticks": sum(abs(d) for d in blink_anims["gunTransformFire"]),
        "collision_mask": blink_payload["collisionMask.m_Bits"],
        "min_player_duration": fix(blink_payload, "minPlayerDuration"),
        "wall_duration": fix(blink_payload, "WallDuration"),
        "wall_delay": fix(blink_payload, "WallDelay"),
        "wall_shake": fix(blink_payload, "WallShake"),
        "on_player_hit_cooldown": fix(blink_payload, "onPlayerHitCD"),
        "tunnel_release_duration": blink_tunnel["ScaleAnim"]["keys"][-1]["time"],
        "tunnel_updatable_count": updater_count(blink_tunnel_prefab),
    }

    # ---- Growth/Shrink rays / ScaleChanger ----
    scale_rays = {}
    for name, ability_path, effect_path in (
        ("growth", "Assets/GameObject/Growth ray.prefab", "Assets/GameObject/GrowRayEffect.prefab"),
        ("shrink", "Assets/GameObject/Shrink ray.prefab", "Assets/GameObject/ShrinkRayEffect.prefab"),
    ):
        _, ray_ability = find_prefab(ability_path)
        ray_gun = component(ray_ability, "GunTransform")
        ray_box = component(ray_ability, "DPhysicsBox")
        ray_physics = component(ray_ability, "PlayerPhysics")
        ray_fix_transform = component(ray_ability, "FixTransform")
        ray_payload = component(ray_ability, "ShootScaleChange")
        ray_anims = animations(guid_of(ray_gun["animData"]))
        _, effect_prefab = find_prefab(effect_path)
        changer = component(effect_prefab, "ScaleChanger")
        scale_rays[name] = {
            "hurtbox_extents": vec2(ray_box, "startExtents"),
            "collider_hierarchy": ray_box["hierarchyNumber"],
            "collider_offset": vec2(ray_fix_transform, "offset"),
            "radius": fix(ray_physics, "radius"),
            "physics": {
                "max_speed": fix(ray_physics, "maxSpeed"),
                "jump_strength": fix(ray_physics, "jumpStrength"),
                "accel": fix(ray_physics, "accel"),
                "radius": fix(ray_physics, "radius"),
                "gravity_accel": fix(ray_physics, "gravity_accel"),
                "gravity_max_fall_speed": fix(ray_physics, "gravity_maxFallSpeed"),
                "gravity_modifier": fix(ray_physics, "gravity_modifier"),
                "jump_extra_x": fix(ray_physics, "jumpExtraXStrength"),
                "jump_momentum": fix(ray_physics, "jumpKeptMomentum"),
                "jump_normal_scale": fix(ray_physics, "jumpNormalScaleFactor"),
                "jump_teleport": fix(ray_physics, "extraJumpTeleportMultiplier"),
                "air_accel": fix(ray_physics, "airAccel"),
            },
            "air_control_blend": fix(ray_gun, "airealControlBlend"),
            "firepoint_offset": vec2(ray_gun, "FirepointOffset"),
            "knockback": fix(ray_gun, "KnockBack"),
            "max_slow_mo_speed": fix(ray_gun, "maxSlowMoSpeed"),
            "transform_ticks": sum(abs(d) for d in ray_anims["gunTransform"]),
            "fire_ticks": sum(abs(d) for d in ray_anims["gunTransformFire"]),
            "collision_mask": ray_payload["collisionMask.m_Bits"],
            "collision_mask_no_water": ray_payload["collisionMaskNoWater.m_Bits"],
            "scale_curve": curve(changer, "scaleAnimation"),
            "player_multiplier": fix(changer, "PlayerMultiplier"),
            "multiplier": fix(changer, "multiplier"),
            "small_non_players_multiplier": fix(changer, "smallNonPlayersMultiplier"),
            "inverse_mass_before_big": fix(changer, "inverseMassBeforeConsideredBig"),
            "changer_updatable_count": updater_count(effect_prefab),
        }
    out["prefabs"]["scale_rays"] = scale_rays

    # ---- Duplicator / shared GunTransform form ----
    _, duplicator_ability = find_prefab("Assets/GameObject/Duplicator.prefab")
    duplicator_gun = component(duplicator_ability, "GunTransform")
    duplicator_box = component(duplicator_ability, "DPhysicsBox")
    duplicator_physics = component(duplicator_ability, "PlayerPhysics")
    duplicator_fix = component(duplicator_ability, "FixTransform")
    duplicator = component(duplicator_ability, "ShootDuplicator")
    duplicator_anims = animations(guid_of(duplicator_gun["animData"]))
    platform_effect_prefab = ASSETS[guid_of(duplicator["platformEffect"])]
    platform_effect = component(platform_effect_prefab, "DuplicatePlatformEffect")
    out["prefabs"]["duplicator"] = {
        "hurtbox_extents": vec2(duplicator_box, "startExtents"),
        "collider_hierarchy": duplicator_box["hierarchyNumber"],
        "collider_offset": vec2(duplicator_fix, "offset"),
        "radius": fix(duplicator_physics, "radius"),
        "physics": {
            "max_speed": fix(duplicator_physics, "maxSpeed"),
            "jump_strength": fix(duplicator_physics, "jumpStrength"),
            "accel": fix(duplicator_physics, "accel"),
            "radius": fix(duplicator_physics, "radius"),
            "gravity_accel": fix(duplicator_physics, "gravity_accel"),
            "gravity_max_fall_speed": fix(duplicator_physics, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(duplicator_physics, "gravity_modifier"),
            "jump_extra_x": fix(duplicator_physics, "jumpExtraXStrength"),
            "jump_momentum": fix(duplicator_physics, "jumpKeptMomentum"),
            "jump_normal_scale": fix(duplicator_physics, "jumpNormalScaleFactor"),
            "jump_teleport": fix(duplicator_physics, "extraJumpTeleportMultiplier"),
            "air_accel": fix(duplicator_physics, "airAccel"),
        },
        "air_control_blend": fix(duplicator_gun, "airealControlBlend"),
        "firepoint_offset": vec2(duplicator_gun, "FirepointOffset"),
        "knockback": fix(duplicator_gun, "KnockBack"),
        "max_slow_mo_speed": fix(duplicator_gun, "maxSlowMoSpeed"),
        "transform_ticks": sum(abs(d) for d in duplicator_anims["gunTransform"]),
        "fire_ticks": sum(abs(d) for d in duplicator_anims["gunTransformFire"]),
        "triple_raycast_radius": fix(duplicator, "tripleRaycastRadius"),
        "collision_mask": duplicator["collisionMask.m_Bits"],
        "collision_mask_no_water": duplicator["collisionMaskNoWater.m_Bits"],
        "platform_shake_duration": fix(duplicator, "PlatformShakeDuration"),
        "platform_shake": fix(duplicator, "platformShake"),
        "platform_overlap_force": fix(platform_effect, "platformOverlapForce"),
        "platform_scale_loss_speed": fix(platform_effect, "scaleLossSpeed"),
        "platform_effect_updatable_count": updater_count(platform_effect_prefab),
    }

    # ---- Magnet gun player form, pull field and held-object launch ----
    _, magnet_ability = find_prefab("Assets/GameObject/Magnet gun.prefab")
    magnet = component(magnet_ability, "MagnetGun")
    magnet_box = component(magnet_ability, "DPhysicsBox")
    magnet_physics = component(magnet_ability, "PlayerPhysics")
    magnet_fix = component(magnet_ability, "FixTransform")
    magnet_anims = animations(guid_of(magnet["animData"]))
    out["prefabs"]["magnet"] = {
        "hurtbox_extents": vec2(magnet_box, "startExtents"),
        "collider_hierarchy": magnet_box["hierarchyNumber"],
        "collider_offset": vec2(magnet_fix, "offset"),
        "radius": fix(magnet_physics, "radius"),
        "physics": {
            "max_speed": fix(magnet_physics, "maxSpeed"),
            "jump_strength": fix(magnet_physics, "jumpStrength"),
            "accel": fix(magnet_physics, "accel"),
            "radius": fix(magnet_physics, "radius"),
            "gravity_accel": fix(magnet_physics, "gravity_accel"),
            "gravity_max_fall_speed": fix(magnet_physics, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(magnet_physics, "gravity_modifier"),
            "jump_extra_x": fix(magnet_physics, "jumpExtraXStrength"),
            "jump_momentum": fix(magnet_physics, "jumpKeptMomentum"),
            "jump_normal_scale": fix(magnet_physics, "jumpNormalScaleFactor"),
            "jump_teleport": fix(magnet_physics, "extraJumpTeleportMultiplier"),
            "air_accel": fix(magnet_physics, "airAccel"),
        },
        "air_control_blend": fix(magnet, "airealControlBlend"),
        "firepoint_offset": vec2(magnet, "FirepointOffset"),
        "knockback": fix(magnet, "KnockBack"),
        "player_rock_extra_offset": vec2(magnet, "playerRockExtraOffset"),
        "boulder_extra_offset": vec2(magnet, "boulderExtraOffset"),
        "held_lerp": fix(magnet, "heldPosLerpFactor01"),
        "fire_force_mines": fix(magnet, "FireForceMines"),
        "fire_force": fix(magnet, "FireForce"),
        "fire_ragdoll_force": fix(magnet, "FireRagdollForce"),
        "fire_black_hole_force": fix(magnet, "FireBlackHoleForce"),
        "fire_rock_force": fix(magnet, "FireRockForce"),
        "fire_missile_force": fix(magnet, "FireMissileForce"),
        "fire_boulder_force": fix(magnet, "FireBoulderForce"),
        "roped_force_multiplier": fix(magnet, "ropedForceMul"),
        "max_slow_mo_speed": fix(magnet, "maxSlowMoSpeed"),
        "only_pull": magnet["onlyPull.m_Bits"],
        "only_pickup": magnet["onlyPickup.m_Bits"],
        "pickup_radius": fix(magnet, "pickupRadius"),
        "max_lock_distance_squared": fix(magnet, "maxLockDist"),
        "close_pull_multiplier": fix(magnet, "closePullStrMul"),
        "fast_pull_radius_squared": fix(magnet, "FastPullSqrRadius"),
        "pull_width": fix(magnet, "pullWidth"),
        "pull_strength": fix(magnet, "pullStr"),
        "player_pull_strength": fix(magnet, "playerPullStr"),
        "projectile_pull_strength": fix(magnet, "projectilePullStr"),
        "wall_pull_strength": fix(magnet, "wallPullStr"),
        "black_hole_pull_strength": fix(magnet, "blackHolePullStr"),
        "macho_pull_strength": fix(magnet, "MachoSlimePullStr"),
        "player_mass": fix(magnet, "playerMass"),
        "locked_angular_velocity": fix(magnet, "angVelOnLockedItem"),
        "small_ragdoll_duration": fix(magnet, "smallRagdollDuration"),
        "fire_ragdoll_duration": fix(magnet, "fireRagdollDuration"),
        "wall_mask": magnet["wallmask.m_Bits"],
        "raycast_offset": vec2(magnet, "raycastOffset"),
        "max_pull_duration": fix(magnet, "maxPullDuration"),
        "parry_push_strength": fix(magnet, "parryPushStr"),
        "pull_curve": curve(magnet, "pullStrMultiplierCurve"),
        "enter_ticks": sum(abs(d) for d in magnet_anims["enter"]),
        "prepull_ticks": sum(abs(d) for d in magnet_anims["prepull"]),
        "exit_ticks": sum(abs(d) for d in magnet_anims["exit"]),
    }

    # ---- Tack / shared ThrowItem2 item path ----
    _, tack_ability = find_prefab("Assets/GameObject/Banana peel.prefab")
    tack_throw = component(tack_ability, "ThrowItem2")
    tack_box = component(tack_ability, "DPhysicsBox")
    tack_phys = component(tack_ability, "PlayerPhysics")
    tack_anims = animations(guid_of(tack_throw["animData"]))
    tack_item = ASSETS[guid_of(tack_throw["ItemPrefab"])]
    tack_item_data = component(tack_item, "Item")
    tack_circle = component(tack_item, "DPhysicsCircle")
    tack_body = component(tack_item, "BoplBody")
    tack_trap = component(tack_item, "BananaPeel")
    out["prefabs"]["tack"] = {
        "hurtbox_extents": vec2(tack_box, "startExtents"),
        "throw": {
            "air_position": vec2(tack_throw, "aThrowPosition"),
            "ground_position": vec2(tack_throw, "gThrowPosition"),
            "ground_dummy": [vec2(tack_throw, f"groundedDummyPositions.Array.data[{i}]")
                             for i in range(tack_throw["groundedDummyPositions.Array.size"])],
            "air_dummy": [vec2(tack_throw, f"airealDummyPositions.Array.data[{i}]")
                          for i in range(tack_throw["airealDummyPositions.Array.size"])],
            "hurt_owner_delay": fix(tack_throw, "DisableGrenadeHitBoxTime"),
            "max_force": fix(tack_throw, "maxThrowForce"),
            "min_force": fix(tack_throw, "minThrowForce"),
            "force_gain_speed": fix(tack_throw, "ThrowForceGainSpeed"),
            "max_slow_mo_speed": fix(tack_throw, "maxSlowMoSpeed"),
            "knockback": fix(tack_throw, "knockback"),
            "physics": {
                "max_speed": fix(tack_phys, "maxSpeed"),
                "jump_strength": fix(tack_phys, "jumpStrength"),
                "accel": fix(tack_phys, "accel"),
                "radius": fix(tack_phys, "radius"),
                "gravity_accel": fix(tack_phys, "gravity_accel"),
                "gravity_max_fall_speed": fix(tack_phys, "gravity_maxFallSpeed"),
                "gravity_modifier": fix(tack_phys, "gravity_modifier"),
                "jump_extra_x": fix(tack_phys, "jumpExtraXStrength"),
                "jump_momentum": fix(tack_phys, "jumpKeptMomentum"),
                "jump_normal_scale": fix(tack_phys, "jumpNormalScaleFactor"),
                "jump_teleport": fix(tack_phys, "extraJumpTeleportMultiplier"),
                "air_accel": fix(tack_phys, "airAccel"),
            },
            "animations": {name: tack_anims[name]
                           for name in ("gEnter", "gExit", "aEnter", "aExit")},
        },
        "item": {
            "layer": tack_item["data"].get("layer"),
            "updatable_count": updater_count(tack_item),
            "child_count": collider_child_count(tack_item),
            "collider_hierarchy": tack_circle["hierarchyNumber"],
            "radius": fix(tack_circle, "startRadius"),
            "body": physics_body(tack_body),
            "clone_count": tack_item_data["nrOfClonesWhenDuplicated"],
            "duplicate_pattern": tack_item_data["duplicatePattern"],
            "duplicate_spacing": fix(tack_item_data, "duplicate_spacing"),
            "duplicate_angle": fix(tack_item_data, "duplicate_angle"),
            "arm_seconds": fix(tack_trap, "armSeconds"),
            "lifetime_seconds": fix(tack_trap, "lifetimeSeconds"),
        },
    }

    # ---- Poke Ball / shared ThrowItem2 item path ----
    _, poke_ability = find_prefab("Assets/GameObject/Poke ball.prefab")
    poke_throw = component(poke_ability, "ThrowItem2")
    poke_box = component(poke_ability, "DPhysicsBox")
    poke_phys = component(poke_ability, "PlayerPhysics")
    poke_anims = animations(guid_of(poke_throw["animData"]))
    poke_item = ASSETS[guid_of(poke_throw["ItemPrefab"])]
    poke_item_data = component(poke_item, "Item")
    poke_circle = component(poke_item, "DPhysicsCircle")
    poke_body = component(poke_item, "BoplBody")
    poke_payload = component(poke_item, "PokeBall")
    out["prefabs"]["poke_ball"] = {
        "hurtbox_extents": vec2(poke_box, "startExtents"),
        "throw": {
            "air_position": vec2(poke_throw, "aThrowPosition"),
            "ground_position": vec2(poke_throw, "gThrowPosition"),
            "ground_dummy": [vec2(poke_throw, f"groundedDummyPositions.Array.data[{i}]")
                             for i in range(poke_throw["groundedDummyPositions.Array.size"])],
            "air_dummy": [vec2(poke_throw, f"airealDummyPositions.Array.data[{i}]")
                          for i in range(poke_throw["airealDummyPositions.Array.size"])],
            "hurt_owner_delay": fix(poke_throw, "DisableGrenadeHitBoxTime"),
            "max_force": fix(poke_throw, "maxThrowForce"),
            "min_force": fix(poke_throw, "minThrowForce"),
            "force_gain_speed": fix(poke_throw, "ThrowForceGainSpeed"),
            "max_slow_mo_speed": fix(poke_throw, "maxSlowMoSpeed"),
            "knockback": fix(poke_throw, "knockback"),
            "physics": {
                "max_speed": fix(poke_phys, "maxSpeed"),
                "jump_strength": fix(poke_phys, "jumpStrength"),
                "accel": fix(poke_phys, "accel"),
                "radius": fix(poke_phys, "radius"),
                "gravity_accel": fix(poke_phys, "gravity_accel"),
                "gravity_max_fall_speed": fix(poke_phys, "gravity_maxFallSpeed"),
                "gravity_modifier": fix(poke_phys, "gravity_modifier"),
                "jump_extra_x": fix(poke_phys, "jumpExtraXStrength"),
                "jump_momentum": fix(poke_phys, "jumpKeptMomentum"),
                "jump_normal_scale": fix(poke_phys, "jumpNormalScaleFactor"),
                "jump_teleport": fix(poke_phys, "extraJumpTeleportMultiplier"),
                "air_accel": fix(poke_phys, "airAccel"),
            },
            "animations": {name: poke_anims[name]
                           for name in ("gEnter", "gExit", "aEnter", "aExit")},
        },
        "item": {
            "layer": poke_item["data"].get("layer"),
            "updatable_count": updater_count(poke_item),
            "child_count": collider_child_count(poke_item),
            "collider_hierarchy": poke_circle["hierarchyNumber"],
            "radius": fix(poke_circle, "startRadius"),
            "body": physics_body(poke_body),
            "clone_count": poke_item_data["nrOfClonesWhenDuplicated"],
            "duplicate_pattern": poke_item_data["duplicatePattern"],
            "duplicate_spacing": fix(poke_item_data, "duplicate_spacing"),
            "duplicate_angle": fix(poke_item_data, "duplicate_angle"),
            "capture_seconds": fix(poke_payload, "captureSeconds"),
            "fizzle_seconds": fix(poke_payload, "fizzleSeconds"),
            "pop_up_speed": fix(poke_payload, "popUpSpeed"),
        },
    }

    # ---- Mjolnir / shared ThrowItem2 item path ----
    _, mjolnir_ability = find_prefab("Assets/GameObject/Mjolnir.prefab")
    mjolnir_throw = component(mjolnir_ability, "ThrowItem2")
    mjolnir_box = component(mjolnir_ability, "DPhysicsBox")
    mjolnir_phys = component(mjolnir_ability, "PlayerPhysics")
    mjolnir_anims = animations(guid_of(mjolnir_throw["animData"]))
    mjolnir_item = ASSETS[guid_of(mjolnir_throw["ItemPrefab"])]
    mjolnir_item_data = component(mjolnir_item, "Item")
    mjolnir_circle = component(mjolnir_item, "DPhysicsCircle")
    mjolnir_body = component(mjolnir_item, "BoplBody")
    mjolnir = component(mjolnir_item, "Mjolnir")
    out["prefabs"]["mjolnir"] = {
        "hurtbox_extents": vec2(mjolnir_box, "startExtents"),
        "throw": {
            "air_position": vec2(mjolnir_throw, "aThrowPosition"),
            "ground_position": vec2(mjolnir_throw, "gThrowPosition"),
            "ground_dummy": [vec2(mjolnir_throw, f"groundedDummyPositions.Array.data[{i}]")
                             for i in range(mjolnir_throw["groundedDummyPositions.Array.size"])],
            "air_dummy": [vec2(mjolnir_throw, f"airealDummyPositions.Array.data[{i}]")
                          for i in range(mjolnir_throw["airealDummyPositions.Array.size"])],
            "hurt_owner_delay": fix(mjolnir_throw, "DisableGrenadeHitBoxTime"),
            "max_force": fix(mjolnir_throw, "maxThrowForce"),
            "min_force": fix(mjolnir_throw, "minThrowForce"),
            "force_gain_speed": fix(mjolnir_throw, "ThrowForceGainSpeed"),
            "max_slow_mo_speed": fix(mjolnir_throw, "maxSlowMoSpeed"),
            "knockback": fix(mjolnir_throw, "knockback"),
            "physics": {
                "max_speed": fix(mjolnir_phys, "maxSpeed"),
                "jump_strength": fix(mjolnir_phys, "jumpStrength"),
                "accel": fix(mjolnir_phys, "accel"),
                "radius": fix(mjolnir_phys, "radius"),
                "gravity_accel": fix(mjolnir_phys, "gravity_accel"),
                "gravity_max_fall_speed": fix(mjolnir_phys, "gravity_maxFallSpeed"),
                "gravity_modifier": fix(mjolnir_phys, "gravity_modifier"),
                "jump_extra_x": fix(mjolnir_phys, "jumpExtraXStrength"),
                "jump_momentum": fix(mjolnir_phys, "jumpKeptMomentum"),
                "jump_normal_scale": fix(mjolnir_phys, "jumpNormalScaleFactor"),
                "jump_teleport": fix(mjolnir_phys, "extraJumpTeleportMultiplier"),
                "air_accel": fix(mjolnir_phys, "airAccel"),
            },
            "animations": {name: mjolnir_anims[name]
                           for name in ("gEnter", "gExit", "aEnter", "aExit")},
        },
        "item": {
            "layer": mjolnir_item["data"].get("layer"),
            "updatable_count": updater_count(mjolnir_item),
            "child_count": collider_child_count(mjolnir_item),
            "collider_hierarchy": mjolnir_circle["hierarchyNumber"],
            "radius": fix(mjolnir_circle, "startRadius"),
            "body": physics_body(mjolnir_body),
            "clone_count": mjolnir_item_data["nrOfClonesWhenDuplicated"],
            "duplicate_pattern": mjolnir_item_data["duplicatePattern"],
            "duplicate_spacing": fix(mjolnir_item_data, "duplicate_spacing"),
            "duplicate_angle": fix(mjolnir_item_data, "duplicate_angle"),
            "outbound_drag": fix(mjolnir, "outboundDrag"),
            "return_speed_threshold": fix(mjolnir, "returnSpeedThreshold"),
            "max_flight_seconds": fix(mjolnir, "maxFlightSeconds"),
            "recall_accel": fix(mjolnir, "recallAccel"),
            "recall_max_speed": fix(mjolnir, "recallMaxSpeed"),
            "clang_grace_seconds": fix(mjolnir, "clangGraceSeconds"),
            "platform_push_force": fix(mjolnir, "platformPushForce"),
        },
    }

    # ---- Smoke / shared ThrowItem2 item path ----
    _, smoke_ability = find_prefab("Assets/GameObject/Smoke.prefab")
    smoke_throw = component(smoke_ability, "ThrowItem2")
    smoke_box = component(smoke_ability, "DPhysicsBox")
    smoke_phys = component(smoke_ability, "PlayerPhysics")
    smoke_anims = animations(guid_of(smoke_throw["animData"]))
    smoke_item = ASSETS[guid_of(smoke_throw["ItemPrefab"])]
    smoke_item_data = component(smoke_item, "Item")
    smoke_circle = component(smoke_item, "DPhysicsCircle")
    smoke_body = component(smoke_item, "BoplBody")
    smoke_grenade = component(smoke_item, "Grenade")
    smoke_spawner = component(smoke_item, "SmokeGrenadeExplode2")
    cloud = ASSETS[guid_of(smoke_spawner["Smoke"])]
    cloud_item = component(cloud, "Item")
    cloud_circle = component(cloud, "DPhysicsCircle")
    cloud_body = component(cloud, "BoplBody")
    cloud_smoke = component(cloud, "FlammableSmoke")
    cloud_anims = animations(guid_of(cloud_smoke["animData"]))
    cloud_burst = ASSETS[guid_of(smoke_spawner["SmokeExplosion"])]
    smoke_explosion = ASSETS[guid_of(cloud_smoke["ExplosionPrefab"])]
    smoke_explosion_box = component(smoke_explosion, "DPhysicsBox")
    smoke_explosion_config = component(smoke_explosion, "Explosion")
    smoke_explosion_anims = animations(guid_of(smoke_explosion_config["explosionAnimation"]))
    out["prefabs"]["smoke"] = {
        "hurtbox_extents": vec2(smoke_box, "startExtents"),
        "throw": {
            "air_position": vec2(smoke_throw, "aThrowPosition"),
            "ground_position": vec2(smoke_throw, "gThrowPosition"),
            "ground_dummy": [vec2(smoke_throw, f"groundedDummyPositions.Array.data[{i}]")
                             for i in range(smoke_throw["groundedDummyPositions.Array.size"])],
            "air_dummy": [vec2(smoke_throw, f"airealDummyPositions.Array.data[{i}]")
                          for i in range(smoke_throw["airealDummyPositions.Array.size"])],
            "hurt_owner_delay": fix(smoke_throw, "DisableGrenadeHitBoxTime"),
            "max_force": fix(smoke_throw, "maxThrowForce"),
            "min_force": fix(smoke_throw, "minThrowForce"),
            "force_gain_speed": fix(smoke_throw, "ThrowForceGainSpeed"),
            "max_slow_mo_speed": fix(smoke_throw, "maxSlowMoSpeed"),
            "knockback": fix(smoke_throw, "knockback"),
            "physics": {
                "max_speed": fix(smoke_phys, "maxSpeed"),
                "jump_strength": fix(smoke_phys, "jumpStrength"),
                "accel": fix(smoke_phys, "accel"),
                "radius": fix(smoke_phys, "radius"),
                "gravity_accel": fix(smoke_phys, "gravity_accel"),
                "gravity_max_fall_speed": fix(smoke_phys, "gravity_maxFallSpeed"),
                "gravity_modifier": fix(smoke_phys, "gravity_modifier"),
                "jump_extra_x": fix(smoke_phys, "jumpExtraXStrength"),
                "jump_momentum": fix(smoke_phys, "jumpKeptMomentum"),
                "jump_normal_scale": fix(smoke_phys, "jumpNormalScaleFactor"),
                "jump_teleport": fix(smoke_phys, "extraJumpTeleportMultiplier"),
                "air_accel": fix(smoke_phys, "airAccel"),
            },
            "animations": {name: smoke_anims[name]
                           for name in ("gEnter", "gExit", "aEnter", "aExit")},
        },
        "item": {
            "layer": smoke_item["data"].get("layer"),
            "updatable_count": updater_count(smoke_item),
            "child_count": collider_child_count(smoke_item),
            "collider_hierarchy": smoke_circle["hierarchyNumber"],
            "radius": fix(smoke_circle, "startRadius"),
            "body": physics_body(smoke_body),
            "detonation_time": fix(smoke_grenade, "detonationTime"),
            "clone_count": smoke_item_data["nrOfClonesWhenDuplicated"],
            "duplicate_pattern": smoke_item_data["duplicatePattern"],
            "duplicate_spacing": fix(smoke_item_data, "duplicate_spacing"),
            "duplicate_angle": fix(smoke_item_data, "duplicate_angle"),
            "cloud_offsets": [vec2(smoke_spawner, f"cloudOffsets.Array.data[{i}]")
                              for i in range(smoke_spawner["cloudOffsets.Array.size"])],
            "spread": fix(smoke_spawner, "spread"),
            "noise": fix(smoke_spawner, "noise"),
            "burst_updatable_count": updater_count(cloud_burst),
        },
        "cloud": {
            "layer": cloud["data"].get("layer"),
            "updatable_count": updater_count(cloud),
            "collider_hierarchy": cloud_circle["hierarchyNumber"],
            "base_radius": fix(cloud_circle, "startRadius"),
            "body": physics_body(cloud_body),
            "clone_count": cloud_item["nrOfClonesWhenDuplicated"],
            "duplicate_pattern": cloud_item["duplicatePattern"],
            "duplicate_spacing": fix(cloud_item, "duplicate_spacing"),
            "duplicate_angle": fix(cloud_item, "duplicate_angle"),
            "radii": [fix(cloud_smoke, f"SmokeRadius.Array.data[{i}]")
                      for i in range(cloud_smoke["SmokeRadius.Array.size"])],
            "enter_frames": cloud_anims["enter"],
            "min_ignition_delay": fix(cloud_smoke, "minDetonationDelay"),
            "max_ignition_delay": fix(cloud_smoke, "maxDetonationDelay"),
        },
        "explosion": {
            "layer": smoke_explosion["data"].get("layer"),
            "updatable_count": registering_count(smoke_explosion),
            "collider_hierarchy": smoke_explosion_box["hierarchyNumber"],
            "extents": vec2(smoke_explosion_box, "startExtents"),
            "collider_offset": vec2(component(smoke_explosion, "FixTransform"), "offset"),
            "animation_ticks": sum(abs(d) for d in smoke_explosion_anims[next(iter(smoke_explosion_anims))]),
        },
    }

    # ---- Time Stop / CastSpell ----
    _, time_stop_ability = find_prefab("Assets/GameObject/Time stop.prefab")
    time_stop_cast = component(time_stop_ability, "CastSpell")
    time_stop_circle = component(time_stop_ability, "DPhysicsCircle")
    time_stop_fix = component(time_stop_ability, "FixTransform")
    time_stop_anims = animations(guid_of(time_stop_cast["animData"]))
    time_stop_effect = ASSETS[guid_of(time_stop_cast["spell"])]
    time_stop = component(time_stop_effect, "TimeStop")
    exit_keys = time_stop["ExitAlpha.m_Curve.Array.size"]
    out["prefabs"]["time_stop"] = {
        "cast_time": fix(time_stop_cast, "castTime"),
        "center_offset": vec2(time_stop_cast, "centerOffset"),
        "exit_offset": vec2(time_stop_cast, "ExitAbilityOffset"),
        "collider_offset": vec2(time_stop_fix, "offset"),
        "hurtbox_radius": fix(time_stop_circle, "startRadius"),
        "collider_hierarchy": time_stop_circle["hierarchyNumber"],
        "animations": {name: time_stop_anims[name]
                       for name in ("idleToFall", "castSpell", "easeOutSpellCast")},
        "effect_updatable_count": updater_count(time_stop_effect),
        "duration_f32": f32(time_stop["duration"]),
        "exit_seconds_f32": f32(time_stop[
            f"ExitAlpha.m_Curve.Array.data[{exit_keys - 1}].time"]),
    }

    # ---- Spike / platform-attached SpikeAttack ----
    _, spike_ability = find_prefab("Assets/GameObject/Spike.prefab")
    spike = component(spike_ability, "Spike")
    spike_box = component(spike_ability, "DPhysicsBox")
    spike_fix = component(spike_ability, "FixTransform")
    spike_anims = animations(guid_of(spike["animData"]))
    spike_attack = ASSETS[guid_of(spike["spikePrefab"])]
    spike_attack_logic = component(spike_attack, "SpikeAttack")
    spike_attack_box = component(spike_attack, "DPhysicsBox")
    spike_attack_fix = component(spike_attack, "FixTransform")
    spike_attack_body = component(spike_attack, "BoplBody")
    out["prefabs"]["spike"] = {
        "hurtbox_extents": vec2(spike_box, "startExtents"),
        "collider_offset": vec2(spike_fix, "offset"),
        "collider_hierarchy": spike_box["hierarchyNumber"],
        "animations": {name: spike_anims[name] for name in ("cast", "recover")},
        "surface_offset": fix(spike, "surfaceOffset"),
        "cast_force": fix(spike, "castForce"),
        "attack": {
            "layer": spike_attack["data"].get("layer"),
            "updatable_count": updater_count(spike_attack),
            "child_count": collider_child_count(spike_attack),
            "collider_hierarchy": spike_attack_box["hierarchyNumber"],
            "extents": vec2(spike_attack_box, "startExtents"),
            "collider_offset": vec2(spike_attack_fix, "offset"),
            "body": physics_body(spike_attack_body),
            "down_thrust": curve(spike_attack_logic, "DownThrustAnimation"),
            "knock_player": fix(spike_attack_logic, "knockAwayStr"),
            "knock_wall": fix(spike_attack_logic, "knockAwayWallStr"),
            "self_push_multiplier": fix(spike_attack_logic, "selfPushMultiplier"),
            "push_time": fix(spike_attack_logic, "pushTime"),
        },
    }

    # ---- Engine / platform-attached RocketEngine ----
    _, engine_ability = find_prefab("Assets/GameObject/Engine.prefab")
    engine_place = component(engine_ability, "PlantObjectOnPlatform")
    engine_factory = component(engine_ability, "PlaceRocketEngine")
    engine_box = component(engine_ability, "DPhysicsBox")
    engine_fix = component(engine_ability, "FixTransform")
    engine_anims = animations(guid_of(engine_place["animData"]))
    rocket = ASSETS[guid_of(engine_factory["plantablePrefab"])]
    rocket_logic = component(rocket, "RocketEngine")
    rocket_box = component(rocket, "DPhysicsBox")
    rocket_fix = component(rocket, "FixTransform")
    rocket_body = component(rocket, "BoplBody")
    rocket_anims = animations(guid_of(rocket_logic["FireData"]))
    out["prefabs"]["engine"] = {
        "hurtbox_extents": vec2(engine_box, "startExtents"),
        "collider_offset": vec2(engine_fix, "offset"),
        "collider_hierarchy": engine_box["hierarchyNumber"],
        "animations": {name: engine_anims[name] for name in ("cast", "recover")},
        "plant_offset": vec2(engine_factory, "plantOffset"),
        "cast_force": fix(engine_factory, "castForce"),
        "buffer_size": engine_factory["bufferSize"],
        "rocket": {
            "layer": rocket["data"].get("layer"),
            "updatable_count": updater_count(rocket),
            "child_count": collider_child_count(rocket),
            "collider_hierarchy": rocket_box["hierarchyNumber"],
            "extents": vec2(rocket_box, "startExtents"),
            "collider_offset": vec2(rocket_fix, "offset"),
            "body": physics_body(rocket_body),
            "force": curve(rocket_logic, "ForceAnim"),
            "start_ticks": sum(abs(d) for d in rocket_anims["start"]),
            "end_ticks": sum(abs(d) for d in rocket_anims["end"]),
            "fire_offset": vec2(rocket_logic, "offset"),
            "fire_radius": fix(rocket_logic, "radius"),
            "smoke_mask": rocket_logic["smokeOnlyLayerMask.m_Bits"],
        },
    }

    # ---- Tesla coil / paired SimpleSparkNode ----
    _, tesla_ability = find_prefab("Assets/GameObject/Tesla coil.prefab")
    tesla_place = component(tesla_ability, "PlantObjectOnPlatform")
    tesla_factory = component(tesla_ability, "PlaceSparkNode")
    tesla_box = component(tesla_ability, "DPhysicsBox")
    tesla_fix = component(tesla_ability, "FixTransform")
    tesla_anims = animations(guid_of(tesla_place["animData"]))
    spark = ASSETS[guid_of(tesla_factory["plantablePrefab"])]
    spark_logic = component(spark, "SimpleSparkNode")
    spark_circle = component(spark, "DPhysicsCircle")
    spark_fix = component(spark, "FixTransform")
    spark_body = component(spark, "BoplBody")
    lightning = ASSETS[guid_of(spark_logic["lightningPrefab"])]
    lightning_logic = component(lightning, "StretchableLightning")
    lightning_box = component(lightning, "DPhysicsBox")
    lightning_anims = animations(guid_of(lightning_logic["animData"]))
    out["prefabs"]["tesla"] = {
        "hurtbox_extents": vec2(tesla_box, "startExtents"),
        "collider_offset": vec2(tesla_fix, "offset"),
        "collider_hierarchy": tesla_box["hierarchyNumber"],
        "animations": {name: tesla_anims[name] for name in ("cast", "recover")},
        "plant_offset": vec2(tesla_factory, "plantOffset"),
        "cast_force": fix(tesla_factory, "castForce"),
        "buffer_size": 2,
        "node": {
            "layer": spark["data"].get("layer"),
            # SpriteAnimator receives hierarchy 5 even though the normalized
            # runtime inventory does not classify it as an updatable here.
            "updatable_count": hierarchy_span(spark),
            "child_count": collider_child_count(spark),
            "collider_hierarchy": spark_circle["hierarchyNumber"],
            "radius": fix(spark_circle, "startRadius"),
            "collider_offset": vec2(spark_fix, "offset"),
            "body": physics_body(spark_body),
            "line_length": fix(spark_logic, "lightningLineLength"),
            "hitbox_offset_y": fix(spark_logic, "hitBoxOffsetY"),
        },
        "lightning": {
            "layer": lightning["data"].get("layer"),
            "updatable_count": hierarchy_span(lightning),
            "collider_hierarchy": lightning_box["hierarchyNumber"],
            "extents": vec2(lightning_box, "startExtents"),
            "minimum_extent": fix(lightning_box, "minimumExtents"),
            "flicker_period": fix(lightning_logic, "flickerPeriod"),
            "enter_ticks": sum(abs(d) for d in lightning_anims["EnterOrb"]),
            "flicker_ticks": sum(abs(d) for d in lightning_anims["flicker"]),
        },
    }

    # ---- Beam player form, retained orb, and DetPhysics beam pool ----
    _, beam_prefab = find_prefab("Assets/GameObject/Beam.prefab")
    beam = component(beam_prefab, "Beam")
    beam_phys = component(beam_prefab, "PlayerPhysics")
    beam_box = component(beam_prefab, "DPhysicsBox")
    beam_fix = component(beam_prefab, "FixTransform")
    beam_anims = animations(guid_of(beam["animData"]))
    orb_prefab = ASSETS[guid_of(beam["OrbEffectPrefab"])]
    orb_circle = component(orb_prefab, "DPhysicsCircle")
    beam_object = ASSETS[guid_of(next(
        node_component(node, "DetPhysics")["beamObjectPrefab"]
        for node in walk(load("map-0.json"))
        if node_component(node, "DetPhysics") is not None
    ))]
    beam_object_logic = component(beam_object, "BeamObject")
    beam_object_box = component(beam_object, "DPhysicsBox")
    det = next(node_component(node, "DetPhysics")
               for node in walk(load("map-0.json"))
               if node_component(node, "DetPhysics") is not None)
    out["prefabs"]["beam"] = {
        "hurtbox_extents": vec2(beam_box, "startExtents"),
        "collider_offset": vec2(beam_fix, "offset"),
        "collider_hierarchy": beam_box["hierarchyNumber"],
        "physics": {
            "max_speed": fix(beam_phys, "maxSpeed"),
            "accel": fix(beam_phys, "accel"),
            "air_accel": fix(beam_phys, "airAccel"),
            "radius": fix(beam_phys, "radius"),
            "gravity_accel": fix(beam_phys, "gravity_accel"),
            "gravity_max_fall_speed": fix(beam_phys, "gravity_maxFallSpeed"),
            "gravity_modifier": fix(beam_phys, "gravity_modifier"),
            "jump_strength": fix(beam_phys, "jumpStrength"),
        },
        "enter_ground_ticks": sum(abs(t) for t in beam_anims["enterFrame1"])
                              + sum(abs(t) for t in beam_anims["enter"]),
        # Beam.OnLeftFirstFrame runs an extra UpdateSim(0) at this boundary.
        "enter_first_frame_ticks": sum(abs(t) for t in beam_anims["enterFrame1"]),
        "enter_air_ticks": sum(abs(t) for t in beam_anims["enterAir"]),
        # SpriteAnimator's passed-sprite counter becomes visible to Beam's
        # LateUpdateSim two updater positions after each cumulative boundary.
        "air_orb_frame8_ticks": sum(abs(t) for t in beam_anims["enterAir"][:7]) + 2,
        "air_orb_frame9_ticks": sum(abs(t) for t in beam_anims["enterAir"][:8]) + 2,
        "air_orb_frame10_ticks": sum(abs(t) for t in beam_anims["enterAir"][:9]) + 2,
        "exit_ground_ticks": sum(abs(t) for t in beam_anims["exit"]),
        "exit_air_ticks": sum(abs(t) for t in beam_anims["exitAir"]),
        "up_alignment": fix(beam, "upAlignmentUntilUpAnimationIsUsed"),
        "min_grounded_angle": fix(beam, "minGroundedAngle"),
        "exit_scale_curve": curve(beam, "exitScaleAnim"),
        "beam_offset": fix(beam, "beamOffset"),
        "air_extents": vec2(beam, "hurtBoxExtentsAir"),
        "air_offset": vec2(beam, "hurtBoxOffsetAir"),
        "angular_aim_speed": fix(beam, "angularAimSpeed"),
        "angular_aim_curve": curve(beam, "angularAimSpeedAnim01"),
        "exit_up_speed": fix(beam, "onExitUpSpeed"),
        "max_time": fix(beam, "maxTime"),
        "max_time_air": fix(beam, "maxTimeAir"),
        "exit_scale_time": fix(beam, "exitScaleTime"),
        "shake_duration": fix(beam, "shakeDuration"),
        "shake_strength": fix(beam, "shakeStr"),
        "orb_layer": orb_prefab["data"].get("layer"),
        "orb_updatable_count": hierarchy_span(orb_prefab),
        "orb_collider_hierarchy": orb_circle["hierarchyNumber"],
        "orb_radius": fix(orb_circle, "startRadius"),
        "orb_exit_ticks": sum(abs(t) for t in animations(guid_of(beam["orbAnimData"]))["whiteExit"]),
        "maximum_distance": fix(det, "maxBeamDistance"),
        "width": fix(det, "beamWidth"),
        "width_curve": curve(det, "beamWidthAnim"),
        "parent_offset": fix(det, "beamParentOffset"),
        "platform_force": fix(det, "beamPlatformPushForce"),
        "black_hole_force": fix(det, "beamBlackHolePushForce"),
        "rigid_body_force": fix(det, "beamRigidBodyPushFOrce"),
        "cross_offset": fix(det, "crossBeamOffset"),
        "hit_mask": det["beamHitMask"],
        "object_layer": beam_object["data"].get("layer"),
        "object_updatable_count": hierarchy_span(beam_object),
        "object_collider_hierarchy": beam_object_box["hierarchyNumber"],
        "base_width": fix(beam_object_logic, "baseWidth01"),
        "extra_length": fix(beam_object_logic, "extraLength"),
        "minimum_extent": fix(beam_object_box, "minimumExtents"),
    }

    # ---- Shared rope physics and Web Swing ----
    _, web_prefab = find_prefab("Assets/GameObject/Web swing.prefab")
    web = component(web_prefab, "WebSwing")
    rope_prefab = ASSETS[guid_of(web["ropePrefab"])]
    rope = component(rope_prefab, "Rope")
    attachment_prefab = ASSETS[guid_of(web["attachmentPrefab"])]
    attachment = component(attachment_prefab, "RopeAttachment")
    _, grapple_prefab = find_prefab("Assets/GameObject/Grappling hook.prefab")
    grapple = component(grapple_prefab, "HookshotInstant")
    hook_prefab = ASSETS[guid_of(grapple["hookPrefab"])]
    hook_attachment = component(hook_prefab, "RopeAttachment")
    hook_circle = component(hook_prefab, "DPhysicsCircle")
    hook = component(hook_prefab, "RopeHook")
    out["prefabs"]["ropes"] = {
        # RopeBodyList/RopeBody fixed arrays in the pinned source. They are
        # structural capacities rather than authored Unity properties.
        "pool_capacity": 128,
        "max_segments": 40,
        "max_contacts": 32,
        "segments": rope["nrOfSegments"],
        "max_ropes_per_player": rope["maxRopesPerPlayer"],
        "gravity": fix(det, "ropeGravity"),
        "iterations": det["ropeRelaxationIterations"],
        "stretchiness": fix(det, "platformStretchiness"),
        "max_stretch": fix(det, "maxStretchLength"),
        "grounded_strength": fix(det, "groundedPlayerRopeStr"),
        "segment_radius": fix(det, "ropeSegmentRadius"),
        "player_gravity": fix(det, "playerGravity"),
        "platform_mass": fix(attachment, "platformMass01"),
        "platform_heaviness": fix(attachment, "platformHeavinessScaling"),
        "platform_pull": fix(attachment, "platformPullStr"),
        "boulder_pull": fix(attachment, "boulderPullStr"),
        "platform_break_stretch": fix(attachment, "stretchBeforeBreak_platform"),
        "non_platform_mass": fix(attachment, "nonPlatformMass"),
        "extra_swing_force": fix(slime_ctl, "extraSwingForce"),
        "web_max_length": fix(web, "maxWebLength"),
        "web_shorten": fix(web, "ropeShorten"),
        "web_cooldown": fix(web, "cooldown"),
        "web_miss_cooldown": fix(web, "missCooldown"),
        "web_release_boost": fix(web, "releaseBoost"),
        "web_grounded_pop": fix(web, "groundedPop"),
        "grapple_spawn_offset": fix(grapple, "hookSpawnOffset"),
        "grapple_speed": fix(grapple, "hookSpeed"),
        "grapple_reel_frames": grapple["framesBetweenReelIns"],
        "grapple_start_separation": fix(grapple, "startingSegmentSeparation"),
        "grapple_reeled_separation": fix(grapple, "reeledInSegmentSeparation"),
        "grapple_reel_speed": fix(grapple, "ReelInSpeed"),
        "grapple_delete_separation": fix(grapple, "separationBeforeReelinDeletion"),
        "grapple_cooldown": fix(grapple, "cooldown"),
        "grapple_player_cast_radius": fix(slime_ctl, "RopeCircleCastRadius"),
        "hook_layer": hook_prefab["data"]["layer"],
        "hook_radius": fix(hook_circle, "startRadius"),
        "hook_collider_hierarchy": hook_circle["hierarchyNumber"],
        "hook_component_hierarchy": hook["hierarchyNumber"],
        "hook_updatable_count": hierarchy_span(hook_prefab),
        "hook_segments_affected": hook["segmentsAffectedBySpeed"],
        "hook_speed_falloff": fix(hook, "segmentSpeedFalloff01"),
        "hook_fall_time": fix(hook, "timeUntilFall"),
        "hook_pull": fix(hook_attachment, "pullStr"),
        "rope_reel_out_interval": fix(rope, "ReelOutSpeed"),
        "rope_segments_affected": rope["segmentsAffectedByReelOut"],
        "rope_segment_wave_length": rope["segmentWaveLength"],
        "rope_reel_out_segment_speed": fix(rope, "reelOutSegmentSpeed"),
        "rope_sprinkler_speed": fix(rope, "sprinklerSpeed"),
        "rope_max_travel_time": fix(rope, "maxTravelTime"),
    }

    # ---- Complete ability catalog ----
    # The catalog order is the public RL ability id. Pull common routing data
    # from the concrete prefab instead of maintaining a second hand-written
    # cooldown table in Rust. Rewind derives from InstantAbility, so its
    # inherited serialized fields live on the Rewind component itself.
    coverage = json.load(open(ROOT / "data" / "ability-coverage.json"))["abilities"]
    _, icons = find_prefab("Assets/MonoBehaviour/AbilityIcons.asset")
    icon_props = icons["data"]["properties"]
    ability_catalog = []
    for expected_id, item in enumerate(coverage):
        if item["id"] != expected_id:
            raise ValueError(f"ability coverage is not dense at {expected_id}")
        _, prefab = find_prefab(item["prefab"])
        components = prefab["data"]["components"]
        ordinary = next((c for c in components if c["type"] == "Ability"), None)
        if ordinary is not None:
            props = ordinary["properties"]
            cooldown = fix(props, "Cooldown")
            instant = False
            castable = props["isCastable"]
        else:
            inherited = next((c for c in components
                              if "cooldown.m_rawValue" in c.get("properties", {})
                              and "OnlyCastableOnGround" in c.get("properties", {})), None)
            if inherited is None:
                raise KeyError(f"{item['prefab']}: AbilityMonoBehaviour")
            props = inherited["properties"]
            cooldown = fix(props, "cooldown")
            instant = True
            castable = True
        prefix = f"sprites.Array.data[{expected_id}]"
        effect_cooldown = next((fix(c["properties"], "cooldown")
                                for c in components
                                if c["type"] in ("Invisibility", "SuperStar")), None)
        effect_duration = None
        air_jump_strength = None
        form_jump_speed = None
        effect_updatable_count = None
        # SuperStarEffect kicks a form through its own PlayerPhysics, so only
        # a form with both a PlayerBody and a PlayerPhysics has a jumpStrength
        # to kick with.
        types = {c["type"] for c in components}
        form_jump_strength = (fix(component(prefab, "PlayerPhysics"), "jumpStrength")
                              if {"PlayerBody", "PlayerPhysics"} <= types else None)
        # PlayerPhysics.SyncRotationToPlatform: whether its Move and
        # LateUpdateSim turn a grounded body to the surface (PlayerPhysics.cs:
        # 143, 221, 268). None for a form with no PlayerPhysics.
        sync_rotation_to_platform = (component(prefab, "PlayerPhysics")["SyncRotationToPlatform"]
                                     if "PlayerPhysics" in types else None)
        if item["name"] == "Gold mushroom":
            star = next(c for c in components if c["type"] == "SuperStar")
            effect = ASSETS[guid_of(star["properties"]["effectPrefab"])]
            effect_props = component(effect, "SuperStarEffect")
            effect_duration = fix(effect_props, "duration")
            air_jump_strength = fix(effect_props, "airJumpStrength")
            form_jump_speed = fix(effect_props, "formJumpSpeed")
            effect_updatable_count = registering_count(effect)
        ability_catalog.append({
            "id": expected_id,
            "name": item["name"],
            "prefab": item["prefab"],
            "cooldown": cooldown,
            "ground_only": props["OnlyCastableOnGround"],
            "stopped_during_time_stop": props["cooldownStoppedDuringTimeStop"],
            "castable": castable,
            "instant": instant,
            "rope": props.get("isHookshot", False),
            "offensive": icon_props[f"{prefix}.isOffensiveAbility"],
            # Child GameObjects register in the same Updater hierarchy. Roll
            # and Beam each contain one child SpriteAnimator after their root
            # components, so the count has to walk children, not just the root.
            # Counting the components that actually register (rather than the
            # prefab's hierarchyNumber span) is what EndRegisterPrefab advances
            # prefabsAdded by; it derives the Grappling hook special case that
            # used to be hard-coded here and fixes Gold mushroom the same way.
            "updatable_count": registering_count(prefab),
            "effect_cooldown": effect_cooldown,
            "effect_duration": effect_duration,
            "air_jump_strength": air_jump_strength,
            "form_jump_speed": form_jump_speed,
            "form_jump_strength": form_jump_strength,
            "sync_rotation_to_platform": sync_rotation_to_platform,
            "effect_updatable_count": effect_updatable_count,
            "effect_prefab": effect["path"] if item["name"] == "Gold mushroom" else None,
        })
    out["abilities"] = ability_catalog

    # ---- validation against the verified Level1 movement data ----
    ref = json.load(open(ROOT / "data" / "level1-movement.json"))
    ref_platforms = {(p["surface"]["center"]["x"], p["surface"]["center"]["y"]):
                     (p["surface"]["extents"], p["surface"]["radius"]) for p in ref["platforms"]}
    mine = out["maps"][0]["platforms"]
    matched = 0
    for p in mine:
        key = (p["position"]["x"], p["position"]["y"])
        if key in ref_platforms:
            ext, rad = ref_platforms[key]
            scale = p["scale"]
            def scaled(v):
                return (v * scale) >> 32
            # Runtime collider init rounds within a few Q32.32 LSBs; tolerate <= 16 raw.
            if abs(scaled(p["extents"]["x"]) - ext["x"]) <= 16 \
               and abs(scaled(p["extents"]["y"]) - ext["y"]) <= 16 \
               and abs(scaled(p["radius"]) - rad) <= 16:
                matched += 1
            else:
                print(f"MISMATCH {p['name']}: extents {p['extents']} scale {scale} vs {ext} radius {p['radius']} vs {rad}", file=sys.stderr)
        else:
            print(f"UNMATCHED center {p['name']} {key}", file=sys.stderr)
    print(f"Level1 platform cross-check: {matched}/{len(mine)} matched")

    # Physics2D's layer collision matrix, which DetPhysics' SortAndSweep asks
    # through GetIgnoreLayerCollision (DetPhysics.cs:2183, 2252): bit b of row
    # a is set when layers a and b collide.
    with open(ROOT.parent / "sim" / "Export" / "settings.json") as fh:
        out["collision_matrix"] = json.load(fh)["collisionMatrix"]

    # Every prefab's updatables in registration order, by asset name: what
    # each instantiation hands Updater's next append batch.
    prefabs = [entry for entry in sorted(ASSETS.values(), key=lambda entry: entry.get("path", ""))
               if entry.get("path", "").endswith(".prefab") and registrations(entry)]
    out["registrations"] = {Path(entry["path"]).stem: registrations(entry) for entry in prefabs}
    out["copy_registrations"] = {Path(entry["path"]).stem: registrations(entry, copy=True) for entry in prefabs}

    target = ROOT / "data" / "rollout.json"
    with open(target, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {target}")

    def coverage_notes(map_data):
        notes = ["Authored geometry, intro animation, bodies, spawns, and ability spawners are ported."]
        components = set(map_data["components"])
        if "GrowOnStart" in components:
            notes.append("GrowOnStart scale, collider history, mass, and inertia are ported.")
        if "VectorFieldPlatform" in components:
            notes.append("VectorFieldPlatform orbital motion is ported and map-gated.")
        if "AntiLockPlatform" in components:
            notes.append("AntiLockPlatform waypoint motion is ported and map-gated.")
        if map_data["session"]["level_type"] == 2:
            notes.append("Space gravity, bounds, dash, and rope behavior are ported; this scene has no water collider.")
        if len(map_data["ability_spawners"]) > 1:
            notes.append("Both authored ability spawners are preserved in hierarchy order.")
        if "KeepInCameraCorner" in components:
            notes.append("KeepInCameraCorner is camera-only.")
        notes.append("WaterWaves and SceneBoundArrowController are render/HUD-only.")
        return " ".join(notes)

    coverage = {
        "reference": out["reference"],
        "maps": [
            {
                "level": m["level"],
                "scene": m["scene"],
                "folder": str(Path(m["path"]).parent).removeprefix("Assets/Scenes/"),
                "components": m["components"],
                "status": "ported",
                "notes": coverage_notes(m),
            }
            for m in out["maps"]
        ],
    }
    coverage_target = ROOT / "data" / "map-coverage.json"
    with open(coverage_target, "w") as fh:
        json.dump(coverage, fh, indent=2)
        fh.write("\n")
    print(f"wrote {coverage_target}")

    # Dense RL ids now are the authored level ids. Retain the established
    # evaluation split for its old ids and make every newly ported map a
    # training candidate; run configs remain the authority for a given run.
    held_out = {3, 4, 8, 10, 17, 19, 23, 29, 36, 39, 42, 44, 53}
    map_set = {
        "maps": [
            {
                "rl": m["level"],
                "level": m["level"],
                "scene": m["path"],
                "platforms": len(m["platforms"]),
                # LevelType: grass 0, snow 1, space 2. Only space is mechanical
                # (half gravity, no water, Camera_YMin as the lower blast
                # boundary); snow differs from grass only cosmetically, so
                # nothing downstream should branch on 1 versus 0.
                "level_type": m["session"]["level_type"],
                "role": "held-out" if m["level"] in held_out else "training",
            }
            for m in out["maps"]
        ],
        "split": {
            "training": [m["level"] for m in out["maps"] if m["level"] not in held_out],
            "held_out": [m["level"] for m in out["maps"] if m["level"] in held_out],
        },
    }
    map_set_target = ROOT / "data" / "map-set.json"
    with open(map_set_target, "w") as fh:
        # indent=2 matches the committed file. It used to write indent=1, so
        # every regeneration produced a ~2000-line semantically identical diff.
        json.dump(map_set, fh, indent=2)
        fh.write("\n")
    print(f"wrote {map_set_target} ({len(map_set['maps'])} maps)")


if __name__ == "__main__":
    main()
