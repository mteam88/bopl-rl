using System;
using System.Collections.Generic;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Executes the actual PlayerPhysics methods. Only presentation (render transform
// and optional jump sound) is excluded; detachment listeners are recorded.
public static class ReferencePlayerGround
{
    private static StickyRoundedRectangle Platform(JToken value, List<GameObject> objects)
    {
        var go = new GameObject("ground oracle platform");
        go.SetActive(false);
        objects.Add(go);
        var ft = go.AddComponent<FixTransform>();
        ft.position = V(value["center"]);
        Set(ft, "rightInner", V(value["right"]));
        Set(ft, "upInner", V(value["up"]));
        go.AddComponent<DPhysicsBox>();
        var body = go.AddComponent<BoplBody>();
        Set(body, "fixTrans", ft);
        var platform = go.AddComponent<StickyRoundedRectangle>();
        Set(platform, "groundBody", body);
        var extent = V(value["extents"]);
        var radius = F(value["radius"]);
        platform.rr = go.AddComponent<DPhysicsRoundedRect>();
        platform.rr.radius = radius;
        Set(platform, "extents", extent);
        var segments = new Fix[] { extent.x * (Fix)2L, radius * Fix.PiOver2,
            extent.y * (Fix)2L, radius * Fix.PiOver2, extent.x * (Fix)2L,
            radius * Fix.PiOver2, extent.y * (Fix)2L, radius * Fix.PiOver2 };
        Set(platform, "circumferences", segments);
        Set(platform, "totalCircum", segments[0] * (Fix)2L + segments[1] * (Fix)4L + segments[2] * (Fix)2L);
        return platform;
    }

    public static JToken Evaluate(JObject query)
    {
        var objects = new List<GameObject>();
        try
        {
            var surfaces = new List<StickyRoundedRectangle>();
            foreach (var value in query["surfaces"]) surfaces.Add(Platform(value, objects));
            var go = new GameObject("ground oracle player");
            go.SetActive(false);
            objects.Add(go);
            var state = query["state"];
            var ft = go.AddComponent<FixTransform>();
            ft.position = V(state["position"]);
            Set(ft, "rotationInner", F(state["rotation"]));
            Set(ft, "upInner", V(state["up"]));
            Set(ft, "rightInner", V(state["right"]));
            go.AddComponent<Hitbox>().SetPlayerId(1);
            go.AddComponent<DPhysicsBox>();
            var body = go.AddComponent<PlayerBody>();
            Set(body, "fixTransform", ft);
            body.selfImposedVelocity = V(state["velocity"]);
            body.externalVelocity = V(state["external_velocity"]);
            var physics = go.AddComponent<PlayerPhysics>();
            Set(physics, "body", body);
            physics.radius = F(state["radius"]);
            physics.groundedSpeed = F(state["grounded_speed"]);
            Set(physics, "isGrounded", (bool)state["grounded"]);
            Set(physics, "<LocalPlatformPos>k__BackingField", F(state["local"]));
            if (state["ground"].Type != JTokenType.Null) Set(physics, "attachedGround", surfaces[(int)state["ground"]]);
            JToken notification = JValue.CreateNull();
            physics.OnNoLongerGrounded += rotation => notification = new JValue(rotation.m_rawValue);
            JObject Snapshot()
            {
                var ground = physics.getAttachedGround();
                var s = new JObject {
                    ["position"] = V(body.position), ["rotation"] = body.rotation.m_rawValue,
                    ["up"] = V(body.up), ["right"] = V(body.right),
                    ["velocity"] = V(body.selfImposedVelocity), ["external_velocity"] = V(body.externalVelocity),
                    ["radius"] = physics.radius.m_rawValue, ["grounded"] = physics.IsGrounded(),
                    ["ground"] = ground == null ? JValue.CreateNull() : new JValue(surfaces.IndexOf(ground)),
                    ["local"] = physics.LocalPlatformPos.m_rawValue, ["grounded_speed"] = physics.groundedSpeed.m_rawValue };
                return new JObject { ["state"] = s, ["event"] = notification };
            }
            var frames = new JArray();
            foreach (var action in query["actions"])
            {
                notification = JValue.CreateNull();
                try
                {
                    switch ((string)action["op"])
                    {
                        case "attach": physics.AttachToGround(surfaces[(int)action["surface"]], V(action["position"])); break;
                        case "unground": physics.UnGround((bool)action["null_rotation"], false); break;
                        default: throw new ArgumentException("Unknown ground action");
                    }
                }
                catch (DivideByZeroException)
                {
                    return new JObject { ["error"] = "DivisionByZero", ["action"] = frames.Count,
                        ["completed"] = frames, ["failed"] = Snapshot() };
                }
                frames.Add(Snapshot());
            }
            return frames;
        }
        finally { foreach (var go in objects) if (go != null) UnityEngine.Object.DestroyImmediate(go); }
    }
}
