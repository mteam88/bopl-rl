using System;
using System.Collections.Generic;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Actual PlayerBody methods with live RopeBody objects and the original rope
// solver. No movement or rope method is replaced by a test implementation.
public static class ReferencePlayerBody
{
    private static bool active;
    private static bool released;
    internal static void Released() { if (active) released = true; else ReferencePlayerBodyRuntime.Released(); }

    internal static JObject SnapshotBody(PlayerBody body, List<RopeBody> ropes, bool sound)
    {
        var holder = Get<IPlayerIdHolder>(body, "idHolder");
        var physics = Get<PlayerPhysics>(body, "physics");
        if (body.ropeBody != null && !ropes.Contains(body.ropeBody))
            throw new InvalidOperationException("Uncaptured rope reference");
        var state = new JObject {
            ["position"] = V(body.position), ["velocity"] = V(body.selfImposedVelocity),
            ["external_velocity"] = V(body.externalVelocity), ["previous_position"] = V(body.prevPosition),
            ["previous_velocity"] = V(Get<Vec2>(body, "prevPlayerVel")),
            ["previous_imposed"] = V(Get<Vec2>(body, "prevImposedVel")),
            ["previous_external"] = V(Get<Vec2>(body, "prevExternalVel")),
            ["extra_swing_force"] = V(body.extraSwingForce),
            ["rope"] = body.ropeBody == null ? JValue.CreateNull() : new JValue(ropes.IndexOf(body.ropeBody)),
            ["top"] = body.topAttachment, ["ignore_gravity"] = body.ignoreGravityWhenRoped,
            ["rope_mass"] = body.RopePlayerMass01.m_rawValue,
            ["holder"] = holder == null ? JValue.CreateNull() : new JValue(holder.GetPlayerId()),
            ["physics"] = physics == null ? JValue.CreateNull() : new JObject {
                ["grounded"] = physics.IsGrounded(), ["jumped"] = physics.jumpedThisFrame } };
        var rs = new JArray();
        foreach (var r in ropes) rs.Add(new JObject {
            ["physics"] = ReferenceRope.State(r), ["owner"] = r.ownerId,
            ["time_air_start"] = r.timeInAir_start.m_rawValue, ["time_air_end"] = r.timeInAir_end.m_rawValue });
        return new JObject { ["state"] = state, ["ropes"] = rs, ["released"] = sound };
    }

    public static JToken Evaluate(JObject query)
    {
        var context = query["context"];
        var p = DetPhysics.Get();
        var oldPlayers = PlayerHandler.Get().PlayerList();
        var oldScale = GameTime.PlayerTimeScale;
        var oldRopeGravity = p.ropeGravity;
        var oldDt = p.SimDeltaTimeThisFrame;
        var oldPlayerGravity = p.playerGravity;
        var oldStretch = p.maxStretchLength;
        var oldIterations = p.ropeRelaxationIterations;
        var oldStretchiness = p.platformStretchiness;
        var oldRects = p.roundedRects;
        var oldRopes = p.ropes.ropeBodies;
        var levelField = typeof(DetPhysics).GetField("levelTypeThisFrame", BindingFlags.Static | BindingFlags.NonPublic);
        var oldLevel = levelField.GetValue(null);
        GameObject go = null;
        try
        {
            var players = new List<Player>();
            foreach (var value in context["players"])
            {
                var player = new Player((int)value["id"], 0);
                player.isProtectedFromTimeStop = (bool)value["protected"];
                player.GetLocalInputs().jumpButton_DOWN_THIS_FRAME = (bool)value["jump"];
                players.Add(player);
            }
            PlayerHandler.Get().SetPlayerList(players);
            GameTime.PlayerTimeScale = F(context["time_scale"]);
            p.ropeGravity = F(context["rope_gravity"]);
            p.playerGravity = F(context["player_gravity"]);
            p.maxStretchLength = F(context["max_stretch"]);
            p.ropeRelaxationIterations = 3;
            p.platformStretchiness = (Fix)0.1;
            levelField.SetValue(null, LevelType.grass);
            p.roundedRects = new PhysicsBodyList<RoundedRect>();
            var ropes = new List<RopeBody>();
            foreach (var value in query["ropes"])
            {
                var r = ReferenceRope.Read(value["physics"]);
                r.ownerId = (byte)value["owner"];
                r.timeInAir_start = F(value["time_air_start"]);
                r.timeInAir_end = F(value["time_air_end"]);
                ropes.Add(r);
            }
            p.ropes.ropeBodies = ropes.ToArray();
            go = new GameObject("player body oracle");
            go.SetActive(false);
            var input = query["state"];
            var ft = go.AddComponent<FixTransform>();
            ft.position = V(input["position"]);
            var holder = go.AddComponent<Hitbox>();
            if (input["holder"].Type != JTokenType.Null) holder.SetPlayerId((int)input["holder"]);
            go.AddComponent<DPhysicsBox>();
            var body = go.AddComponent<PlayerBody>();
            Set(body, "fixTransform", ft);
            Set(body, "idHolder", input["holder"].Type == JTokenType.Null ? null : holder);
            body.selfImposedVelocity = V(input["velocity"]);
            body.externalVelocity = V(input["external_velocity"]);
            body.prevPosition = V(input["previous_position"]);
            Set(body, "prevPlayerVel", V(input["previous_velocity"]));
            Set(body, "prevImposedVel", V(input["previous_imposed"]));
            Set(body, "prevExternalVel", V(input["previous_external"]));
            body.extraSwingForce = V(input["extra_swing_force"]);
            body.ropeBody = input["rope"].Type == JTokenType.Null ? null : ropes[(int)input["rope"]];
            body.topAttachment = (bool)input["top"];
            body.ignoreGravityWhenRoped = (bool)input["ignore_gravity"];
            body.RopePlayerMass01 = F(input["rope_mass"]);
            if (input["physics"].Type != JTokenType.Null)
            {
                var physics = go.AddComponent<PlayerPhysics>();
                Set(physics, "isGrounded", (bool)input["physics"]["grounded"]);
                physics.jumpedThisFrame = (bool)input["physics"]["jumped"];
                Set(body, "physics", physics);
            }
            JObject Snapshot() => SnapshotBody(body, ropes, released);
            active = true;
            var frames = new JArray();
            foreach (var action in query["actions"])
            {
                released = false;
                string error = null;
                try
                {
                    switch ((string)action["op"])
                    {
                        case "init": body.Init(); break;
                        case "force": body.AddForce(V(action["force"])); break;
                        case "attach": body.AttachRope(action["rope"].Type == JTokenType.Null ? null : ropes[(int)action["rope"]], (bool)action["top"]); break;
                        case "clear": body.EnterAbilityClearRope(); break;
                        case "update": body.UpdateSim(F(action["dt"])); break;
                        case "late": body.LateUpdateSim(Fix.Zero); break;
                        case "rope_step":
                            var dt = F(action["dt"]);
                            p.SimDeltaTimeThisFrame = dt == Fix.Zero ? GameTime.FixedTimeStep * (Fix)0.1 : dt;
                            // Serial dispatch retains a deterministic first-error boundary.
                            // This tests each original solver, not Parallel.ForEach scheduling.
                            foreach (var r in ropes) p.SimulateRope(r, null);
                            break;
                        default: throw new ArgumentException("Unknown player body action");
                    }
                }
                catch (DivideByZeroException) { error = "division_by_zero"; }
                catch (NullReferenceException) { error = "null_reference"; }
                catch (IndexOutOfRangeException) { error = "index_out_of_range"; }
                if (error != null) return new JObject { ["error"] = error, ["action"] = frames.Count,
                    ["completed"] = frames, ["failed"] = Snapshot() };
                frames.Add(Snapshot());
            }
            return frames;
        }
        finally
        {
            active = false;
            if (go != null) UnityEngine.Object.DestroyImmediate(go);
            PlayerHandler.Get().SetPlayerList(oldPlayers);
            GameTime.PlayerTimeScale = oldScale;
            p.ropeGravity = oldRopeGravity;
            p.SimDeltaTimeThisFrame = oldDt;
            p.playerGravity = oldPlayerGravity;
            p.maxStretchLength = oldStretch;
            p.ropeRelaxationIterations = oldIterations;
            p.platformStretchiness = oldStretchiness;
            p.roundedRects = oldRects;
            p.ropes.ropeBodies = oldRopes;
            levelField.SetValue(null, oldLevel);
        }
    }
}
