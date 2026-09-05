using System;
using System.Collections.Generic;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Actual PlayerPhysics methods on one body shared with real platform/rope
// objects. The detachment listener can move the body to test callback ordering.
public static class ReferencePlayerPhysics
{
    public static JToken Evaluate(JObject query)
    {
        var objects = new List<GameObject>();
        var oldSpace = Constants.leveltype;
        var oldStrength = DetPhysics.Get().groundedPlayerRopeStr;
        try
        {
            Constants.leveltype = (bool)query["space"] ? LevelType.space : LevelType.grass;
            var platforms = new List<StickyRoundedRectangle>();
            foreach (var value in query["platforms"])
            {
                var platform = ReferencePlayerGround.Platform(value["surface"], objects);
                platform.IsDestroyed = (bool)value["destroyed"];
                if (!(bool)value["has_shape"]) platform.rr = null;
                platforms.Add(platform);
            }
            var go = new GameObject("player physics oracle");
            go.SetActive(false);
            objects.Add(go);
            var state = query["state"];
            var motion = state["motion"];
            var ft = go.AddComponent<FixTransform>();
            ft.position = V(motion["position"]);
            Set(ft, "rotationInner", F(motion["rotation"]));
            Set(ft, "upInner", V(motion["up"]));
            Set(ft, "rightInner", V(motion["right"]));
            Set(ft, "scale", F(state["scale"]));
            var holder = go.AddComponent<Hitbox>();
            holder.SetPlayerId((int)query["owner"]);
            go.AddComponent<DPhysicsBox>();
            var body = go.AddComponent<PlayerBody>();
            Set(body, "fixTransform", ft);
            body.selfImposedVelocity = V(motion["velocity"]);
            body.externalVelocity = V(motion["external_velocity"]);
            RopeBody rope = null;
            if (query["rope"].Type != JTokenType.Null)
            {
                var r = query["rope"];
                rope = ReferenceRope.Read(r["physics"]);
                rope.ownerId = (byte)r["owner"];
                rope.timeInAir_start = F(r["time_air_start"]);
                rope.timeInAir_end = F(r["time_air_end"]);
            }
            body.ropeBody = rope;
            body.topAttachment = (bool)query["top"];
            var physics = go.AddComponent<PlayerPhysics>();
            Set(physics, "body", body);
            Set(physics, "playerIdHolder", holder);
            Set(physics, "isGrounded", (bool)motion["grounded"]);
            Set(physics, "attachedGround", motion["ground"].Type == JTokenType.Null ? null : platforms[(int)motion["ground"]]);
            Set(physics, "<LocalPlatformPos>k__BackingField", F(motion["local"]));
            physics.groundedSpeed = F(motion["grounded_speed"]);
            physics.radius = F(motion["radius"]);
            Set(physics, "prevScal", F(state["previous_scale"]));
            physics.jumpedThisFrame = (bool)state["jumped"];
            Set(physics, "SyncRotationToPlatform", (bool)state["sync_rotation"]);
            physics.cannotBeUngroundedByRopes = (bool)state["cannot_unground"];
            var config = query["config"];
            physics.gravity_accel = F(config["gravity"]);
            physics.gravity_maxFallSpeed = F(config["max_fall_speed"]);
            physics.gravity_modifier = F(config["gravity_modifier"]);
            physics.airAccel = F(config["air_accel"]);
            physics.Speed = F(config["max_speed"]);
            physics.inverseMass01 = F(config["inverse_mass"]);
            physics.jumpStrength = F(config["jump_strength"]);
            physics.jumpExtraXStrength = F(config["jump_extra_x"]);
            physics.jumpKeptMomentum = F(config["jump_momentum"]);
            physics.jumpNormalScaleFactor = F(config["jump_normal_scale"]);
            physics.extraJumpTeleportMultiplier = F(config["jump_teleport"]);
            DetPhysics.Get().groundedPlayerRopeStr = F(config["rope_strength"]);
            JArray events = null;
            physics.OnNoLongerGrounded += angle => { events.Add(angle.m_rawValue); body.position += V(query["event_offset"]); };
            JObject Snapshot()
            {
                var s = (JObject)state.DeepClone();
                s["motion"] = ReferencePlayerGround.Motion(physics, body, platforms);
                s["scale"] = ft.Scale.m_rawValue;
                s["previous_scale"] = Get<Fix>(physics, "prevScal").m_rawValue;
                s["jumped"] = physics.jumpedThisFrame;
                JToken r = rope == null ? JValue.CreateNull() : new JObject {
                    ["physics"] = ReferenceRope.State(rope), ["owner"] = rope.ownerId,
                    ["time_air_start"] = rope.timeInAir_start.m_rawValue, ["time_air_end"] = rope.timeInAir_end.m_rawValue };
                return new JObject { ["state"] = s, ["rope"] = r, ["events"] = events };
            }
            var frames = new JArray();
            foreach (var action in query["actions"])
            {
                events = new JArray();
                string error = null;
                try
                {
                    switch ((string)action["op"])
                    {
                        case "scale": Set(ft, "scale", F(action["scale"])); physics.UpdateScale(); break;
                        case "gravity": physics.AddGravityFactor(); break;
                        case "air": physics.AirealMovement(F(action["input"]), Fix.Zero); break;
                        case "force": physics.AddForce(V(action["force"])); break;
                        case "jump": physics.Jump(); break;
                        case "align": physics.AlignToGround(); break;
                        case "offset": physics.ApplyOffset(V(action["offset"])); break;
                        case "late": physics.LateUpdateSim(Fix.Zero); break;
                        case "sync":
                            var info = action["info"];
                            var ground = info["ground"].Type == JTokenType.Null ? null : platforms[(int)info["ground"]];
                            if ((bool)action["exit"]) physics.SyncPhysicsTo(new AbilityExitInfo {
                                isGrounded = (bool)info["grounded"], currentlyAttachedGround = ground,
                                groundedLocalPosition = F(info["local"]), radius = F(info["radius"]), groundedSpeed = F(info["grounded_speed"]),
                                position = V(info["position"]), selfImposedVelocity = V(info["velocity"]), externalVelocity = V(info["external_velocity"]) });
                            else physics.SyncPhysicsTo(new PlayerInfo {
                                isGrounded = (bool)info["grounded"], currentlyAttachedGround = ground,
                                groundedLocalPosition = F(info["local"]), radius = F(info["radius"]), groundedSpeed = F(info["grounded_speed"]),
                                position = V(info["position"]), selfImposedVelocity = V(info["velocity"]), externalVelocity = V(info["external_velocity"]) });
                            break;
                        default: throw new ArgumentException("Unknown player physics action");
                    }
                }
                catch (NullReferenceException) { error = "null_reference"; }
                catch (DivideByZeroException) { error = "division_by_zero"; }
                catch (IndexOutOfRangeException) { error = "index_out_of_range"; }
                if (error != null) return new JObject { ["error"] = error, ["action"] = frames.Count, ["completed"] = frames, ["failed"] = Snapshot() };
                frames.Add(Snapshot());
            }
            return frames;
        }
        finally
        {
            Constants.leveltype = oldSpace;
            DetPhysics.Get().groundedPlayerRopeStr = oldStrength;
            foreach (var go in objects) if (go != null) UnityEngine.Object.DestroyImmediate(go);
        }
    }
}
