#!/usr/bin/env python3
"""Extract the data the Rust rollout backend needs from native/reference-export.

Sources: map-{0..4}.json scene graphs and assets.json prefab/asset entries of the
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


def node_component(node, type_name):
    for c in node.get("components", []):
        if c["type"] == type_name:
            return c["properties"]
    return None


def walk(nodes):
    for n in nodes:
        yield n
        yield from walk(n.get("children", [])) 


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


def extract_map(index):
    scene = load(f"map-{index}.json")
    platforms = []
    water = None
    session = None
    spawner = None
    for node in walk(scene):
        srr = node_component(node, "StickyRoundedRectangle")
        if srr is not None:
            ft = node_component(node, "FixTransform")
            rr = node_component(node, "DPhysicsRoundedRect")
            av = node_component(node, "AnimateVelocity")
            body = node_component(node, "BoplBody")
            anti = node_component(node, "AntiLockPlatform")
            entry = {
                "name": node.get("name"),
                "position": vec2(ft, "_position"),
                "rotation": fix(ft, "_rotation"),
                "scale": ft.get("_scale.m_rawValue", ONE),
                "extents": vec2(rr, "startExtents"),
                "radius": fix(rr, "startRadius") if "startRadius.m_rawValue" in rr else fix(rr, "radius"),
                "layer": node.get("layer"),
                "animate_velocity": {
                    "speed": fix(av, "speed"),
                    "mu": fix(av, "mu"),
                    "rotation_speed": fix(av, "rotationSpeed"),
                    "rotation_mu": fix(av, "rotationMu"),
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
            platforms.append(entry)
        if node.get("name") == "WaterHitBox":
            ft = node_component(node, "FixTransform")
            box = node_component(node, "DPhysicsBox")
            water = {
                "position": vec2(ft, "_position"),
                "extents": vec2(box, "startExtents"),
                "layer": node.get("layer"),
            }
        g = node_component(node, "GameSessionHandler")
        if g is not None:
            n = g.get("teamSpawns.Array.size", 0)
            session = {
                "team_spawns": [vec2(g, f"teamSpawns.Array.data[{i}]") for i in range(n)],
                "teammate_spacing": fix(g, "teammateSpawnSpacing"),
                "level_type": g.get("levelType"),
                "time_before_sudden_death": fix(g, "TimeBeforeSuddenDeath"),
            }
        sp = node_component(node, "AbilitySpawner")
        if sp is not None:
            ft = node_component(node, "FixTransform")
            spawner = {"position": vec2(ft, "_position")}
    return {"platforms": platforms, "water": water, "session": session, "ability_spawner": spawner}


def main():
    out = {"reference": load("manifest.json")["reference"], "maps": [], "prefabs": {}}
    for i in range(5):
        out["maps"].append(extract_map(i))

    # ---- Player (slime) ----
    _, player = find_prefab("Assets/GameObject/Player.prefab")
    phys = component(player, "PlayerPhysics")
    slime_ctl = component(player, "SlimeController")
    box = component(player, "DPhysicsBox")
    pcol = component(player, "PlayerCollision")
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
        "spawn_anim_ticks": sum(abs(d) for d in slime_anims["spawn"]),
        # SlimeController.ReSpawnFast uses SpeedUpAnimation(anim, 4): C#
        # integer division per frame, with every duration clamped to >= 1.
        "fast_spawn_anim_ticks": sum(max(1, abs(d) // 4) for d in slime_anims["spawn"]),
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
        "mass": fix(pickup_body, "mass"),
        "offset": vec2(pickup_tf, "offset"),
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
        "animate_velocity": {
            "speed": fix(rp_av, "speed"),
            "mu": fix(rp_av, "mu"),
            "rotation_speed": fix(rp_av, "rotationSpeed"),
            "rotation_mu": fix(rp_av, "rotationMu"),
            "sudden_death_speed": fix(rp_av, "suddenDeathSpeed"),
            "sudden_death_max_delay": fix(rp_av, "MaxRandomSuddenDeathDelay"),
            "sudden_death_acceleration": fix(rp_av, "SuddenDeathAcceleration"),
            "sudden_death_y_speed": curve(rp_av, "suddenDeathYSpeed"),
        },
        "body": physics_body(rp_body),
        "mass_per_area": fix(rp_res, "MassPerArea"),
        "remove_curve": curve(rp_res, "RemoveAnimation"),
        "resizable_keys": sorted(k for k in rp_res if not k.startswith("m_")),
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
        if item["name"] == "Gold mushroom":
            star = next(c for c in components if c["type"] == "SuperStar")
            effect = ASSETS[guid_of(star["properties"]["effectPrefab"])]
            effect_props = component(effect, "SuperStarEffect")
            effect_duration = fix(effect_props, "duration")
            air_jump_strength = fix(effect_props, "airJumpStrength")
            form_jump_speed = fix(effect_props, "formJumpSpeed")
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
            "effect_cooldown": effect_cooldown,
            "effect_duration": effect_duration,
            "air_jump_strength": air_jump_strength,
            "form_jump_speed": form_jump_speed,
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

    target = ROOT / "data" / "rollout.json"
    with open(target, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
