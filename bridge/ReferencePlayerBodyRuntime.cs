using System;
using System.Collections.Generic;
using BoplFixedMath;
using Newtonsoft.Json.Linq;

// Opt-in per-method replay observations. Unity supplies each initial state;
// this does not replace the real scheduler or execute a native gameplay episode.
internal static class ReferencePlayerBodyRuntime
{
    internal sealed class Capture
    {
        internal PlayerBody body;
        internal List<RopeBody> ropes;
        internal JObject query;
        internal bool released;
        internal string error;
        internal Capture previous;
    }
    private static Capture current;

    internal static Capture Begin(PlayerBody body, string operation, Fix dt)
    {
        if (!ReferencePhysicsCapture.Enabled) return null;
        try
        {
            var ropes = new List<RopeBody>();
            if (body.ropeBody != null) ropes.Add(body.ropeBody);
            var frame = ReferencePlayerBody.SnapshotBody(body, ropes, false);
            var players = new JArray();
            foreach (var player in PlayerHandler.Get().PlayerList()) players.Add(new JObject {
                ["id"] = player.Id, ["protected"] = player.isProtectedFromTimeStop,
                ["jump"] = player.jumpButton_PressedThisFrame() });
            var physics = DetPhysics.Get();
            var context = new JObject {
                ["players"] = players, ["time_scale"] = GameTime.PlayerTimeScale.m_rawValue,
                ["rope_gravity"] = physics.ropeGravity.m_rawValue,
                ["player_gravity"] = physics.playerGravity.m_rawValue,
                ["max_stretch"] = physics.maxStretchLength.m_rawValue };
            var capture = new Capture {
                body = body, ropes = ropes, previous = current,
                query = new JObject {
                    ["operation"] = "dynamics_player_body", ["state"] = frame["state"],
                    ["ropes"] = frame["ropes"], ["context"] = context,
                    ["actions"] = new JArray(new JObject { ["op"] = operation, ["dt"] = dt.m_rawValue }) } };
            current = capture;
            return capture;
        }
        catch (Exception error)
        {
            ReferenceCapture.PlayerBodyStage(new JObject { ["capture_error"] = error.ToString() });
            return null;
        }
    }
    internal static void Released() { if (current != null) current.released = true; }
    internal static void Failed(Capture capture, Exception error)
    {
        if (capture == null) return;
        capture.error = error is NullReferenceException ? "null_reference" :
            error is DivideByZeroException ? "division_by_zero" :
            error is IndexOutOfRangeException ? "index_out_of_range" : error.GetType().FullName;
    }
    internal static void End(Capture capture)
    {
        if (capture == null) return;
        try
        {
            var frame = ReferencePlayerBody.SnapshotBody(capture.body, capture.ropes, capture.released);
            JToken expected = capture.error == null ? (JToken)new JArray(frame) : new JObject {
                ["error"] = capture.error, ["action"] = 0, ["completed"] = new JArray(), ["failed"] = frame };
            ReferenceCapture.PlayerBodyStage(new JObject { ["query"] = capture.query, ["expected"] = expected });
        }
        catch (Exception error)
        {
            ReferenceCapture.PlayerBodyStage(new JObject { ["capture_error"] = error.ToString() });
        }
        finally { current = capture.previous; }
    }
}
