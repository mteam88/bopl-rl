using System;
using System.Collections.Generic;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

internal static class ReferencePlayerRays
{
    private static JObject Hit(RaycastInformation h) => new JObject {
        ["object"] = h ? new JValue(h.pp.instanceId) : JValue.CreateNull(),
        ["near_position"] = V(h.nearPos), ["far_position"] = V(h.farPos),
        ["near_distance"] = h.nearDist.m_rawValue, ["far_distance"] = h.farDist.m_rawValue, ["layer"] = h.layer };

    internal static JToken Evaluate(JObject query, DetPhysics scene, List<ReferencePhysicsScene.Adapter> adapters, Func<JObject> snapshot)
    {
        var objects = new List<GameObject>();
        var oldPlayers = PlayerHandler.Get().PlayerList();
        var oldScale = GameTime.PlayerTimeScale;
        try
        {
            var players = new List<Player>();
            foreach (var p in query["players"]) players.Add(new Player((int)p["id"], 0) { isProtectedFromTimeStop = (bool)p["protected"] });
            PlayerHandler.Get().SetPlayerList(players);
            GameTime.PlayerTimeScale = F(query["time_scale"]);
            var platforms = new List<StickyRoundedRectangle>();
            foreach (var p in query["platforms"])
            {
                var source = ReferencePlayerGround.Platform(p["surface"], objects);
                source.IsDestroyed = (bool)p["destroyed"];
                if (!(bool)p["has_shape"]) source.rr = null;
                platforms.Add(source);
            }
            foreach (var target in query["targets"])
            {
                if (target["ground"].Type == JTokenType.Null) continue;
                var a = adapters.Find(v => v.parent.instanceId == (int)target["object"]);
                var go = a.transform.gameObject;
                // These components only provide the original GetComponent lookup.
                // Their Init methods are not run and do not register extra shapes.
                go.AddComponent<DPhysicsRoundedRect>();
                go.AddComponent<BoplBody>();
                var platform = go.AddComponent<StickyRoundedRectangle>();
                var source = platforms[(int)target["ground"]];
                Set(platform, "groundBody", source.GetGroundBody());
                platform.rr = source.rr;
                Set(platform, "extents", Get<Vec2>(source, "extents"));
                Set(platform, "circumferences", Get<Fix[]>(source, "circumferences"));
                Set(platform, "totalCircum", Get<Fix>(source, "totalCircum"));
                platform.IsDestroyed = source.IsDestroyed;
                platforms[(int)target["ground"]] = platform;
            }
            var receiver = new GameObject("player raycast oracle");
            receiver.SetActive(false);
            objects.Add(receiver);
            var state = query["state"];
            var motion = state["motion"];
            var ft = receiver.AddComponent<FixTransform>();
            ft.position = V(motion["position"]);
            Set(ft, "rotationInner", F(motion["rotation"]));
            Set(ft, "rightInner", V(motion["right"]));
            Set(ft, "upInner", V(motion["up"]));
            var holder = receiver.AddComponent<Hitbox>();
            if (query["holder"].Type != JTokenType.Null) holder.SetPlayerId((int)query["holder"]);
            receiver.AddComponent<DPhysicsBox>();
            var body = receiver.AddComponent<PlayerBody>();
            Set(body, "fixTransform", ft);
            body.selfImposedVelocity = V(motion["velocity"]);
            body.externalVelocity = V(motion["external_velocity"]);
            var physics = receiver.AddComponent<PlayerPhysics>();
            Set(physics, "body", body);
            Set(physics, "playerIdHolder", query["holder"].Type == JTokenType.Null ? null : holder);
            physics.radius = F(motion["radius"]);
            physics.groundedSpeed = F(motion["grounded_speed"]);
            Set(physics, "isGrounded", (bool)motion["grounded"]);
            Set(physics, "attachedGround", motion["ground"].Type == JTokenType.Null ? null : platforms[(int)motion["ground"]]);
            Set(physics, "<LocalPlatformPos>k__BackingField", F(motion["local"]));
            JArray events = null;
            physics.OnAttachedToGround += hit => { events.Add(Hit(hit)); body.position += V(query["event_offset"]); };
            JObject Frame(JToken result)
            {
                var s = (JObject)state.DeepClone();
                s["motion"] = ReferencePlayerGround.Motion(physics, body, platforms);
                var world = snapshot();
                world["objects"] = new JArray(world["objects"].OrderBy(o => (int)o["id"]));
                JArray Order(PhysicsParent[] parents, int length) { var order = new JArray(); for (int i=0;i<length;i++) order.Add(parents[i].instanceId); return order; }
                var orders = new JArray(Order(scene.roundedRects.physicsParents, scene.roundedRects.Length),
                    Order(scene.circles.physicsParents, scene.circles.Length), Order(scene.boxes.physicsParents, scene.boxes.Length));
                return new JObject { ["state"] = s, ["scene"] = world, ["orders"] = orders, ["events"] = events, ["result"] = result };
            }
            var frames = new JArray();
            foreach (var action in query["actions"])
            {
                events = new JArray();
                JToken result = JValue.CreateNull();
                string error = null;
                try
                {
                    if ((string)action["op"] == "query") result = Hit(scene.RaycastToClosestRoundedRect(V(action["origin"]), V(action["direction"]), F(action["distance"])));
                    else if ((string)action["op"] == "velocity") result = physics.VelocityBasedRaycasts((bool)action["attach"], F(action["dt"]));
                    else if ((string)action["op"] == "collision") result = physics.CollisionRaycasts((bool)action["attach"], V(action["velocity"]), F(action["dt"]));
                    else throw new ArgumentException("Unknown player raycast action");
                }
                catch (NullReferenceException) { error = "null_reference"; }
                catch (DivideByZeroException) { error = "division_by_zero"; }
                if (error != null) return new JObject { ["error"] = error, ["action"] = frames.Count, ["completed"] = frames, ["failed"] = Frame(JValue.CreateNull()) };
                frames.Add(Frame(result));
            }
            return frames;
        }
        finally
        {
            PlayerHandler.Get().SetPlayerList(oldPlayers);
            GameTime.PlayerTimeScale = oldScale;
            foreach (var go in objects) if (go != null) UnityEngine.Object.DestroyImmediate(go);
        }
    }
}
