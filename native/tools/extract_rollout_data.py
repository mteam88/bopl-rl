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

    # ---- Rock / BounceBall player form ----
    _, rock_ability = find_prefab("Assets/GameObject/Rock.prefab")
    rock = component(rock_ability, "BounceBall")
    rock_circle = component(rock_ability, "DPhysicsCircle")
    rock_body = component(rock_ability, "BoplBody")
    out["prefabs"]["rock"] = {
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
    }

    # ---- Roll player form ----
    _, roll_ability = find_prefab("Assets/GameObject/Roll.prefab")
    roll = component(roll_ability, "Roll")
    roll_circle = component(roll_ability, "DPhysicsCircle")
    roll_physics = component(roll_ability, "PlayerPhysics")
    roll_trail = ASSETS[guid_of(roll["DashTrailPrefab"])]
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
        # The oracle assigns two reference hierarchy entries when this
        # runtime-instantiated trail wakes: its ParticleSystem and follower.
        "trail_hierarchy_count": sum(
            c["type"] in ("UnityEngine.ParticleSystem", "FollowTransform")
            for c in roll_trail["data"]["components"]
        ),
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
            # Child GameObjects register in the same Updater hierarchy. Roll
            # and Beam each contain one child SpriteAnimator after their root
            # components, so the prefab's span is larger than the root max.
            "updatable_count": hierarchy_span(prefab),
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
