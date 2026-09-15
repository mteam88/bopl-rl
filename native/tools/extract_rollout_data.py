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
                "hierarchy": rr["hierarchyNumber"],
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
    updater_count = lambda prefab: sum(
        "hierarchyNumber" in c.get("properties", {}) for c in prefab["data"]["components"]
    )
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
        # ReviveParticle registers itself in Awake and InitRevive in addition
        # to its SpriteAnimator, consuming three updater hierarchy entries.
        "particle_updatable_count": updater_count(particle) + 1,
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
            "updatable_count": updater_count(smoke_explosion),
            "collider_hierarchy": smoke_explosion_box["hierarchyNumber"],
            "extents": vec2(smoke_explosion_box, "startExtents"),
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
            "updatable_count": max((c.get("properties", {}).get("hierarchyNumber", 0)
                                    for c in components), default=0),
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
