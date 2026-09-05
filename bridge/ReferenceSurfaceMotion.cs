using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

internal static class ReferenceSurfaceMotion
{
    internal static Dictionary<int, StickyRoundedRectangle> BindPlatforms(JToken bindings, DetPhysics scene, List<ReferencePhysicsScene.Adapter> adapters, List<GameObject> objects)
    {
        var platforms = new Dictionary<int, StickyRoundedRectangle>();
        foreach (var binding in bindings)
        {
            int id = (int)binding["object"];
            var a = adapters.Find(v => v.sourceId == id);
            a.transform.IsDestroyed = (bool)binding["transform_destroyed"];
            // The fixture's active-self flag is independent of collider enabled.
            if (a.transform.gameObject.activeSelf != (bool)binding["active_self"]) throw new ArgumentException("Inconsistent active-self input");
            if (binding["platform"].Type == JTokenType.Null) continue;
            var source = ReferencePlayerGround.Platform(binding["platform"]["surface"], objects);
            var rr = a.transform.gameObject.AddComponent<DPhysicsRoundedRect>();
            rr.pp = a.parent;
            rr.radius = source.rr.radius;
            rr.enabled = a.enabled;
            rr.IsDestroyed = a.IsDestroyed;
            a.transform.gameObject.AddComponent<BoplBody>();
            var platform = a.transform.gameObject.AddComponent<StickyRoundedRectangle>();
            Set(platform, "groundBody", source.GetGroundBody());
            Set(platform, "extents", Get<Vec2>(source, "extents"));
            Set(platform, "circumferences", Get<Fix[]>(source, "circumferences"));
            Set(platform, "totalCircum", Get<Fix>(source, "totalCircum"));
            platform.rr = (bool)binding["platform"]["has_shape"] ? rr : null;
            platform.IsDestroyed = (bool)binding["platform"]["destroyed"];
            a.parent.monobehaviourCollider = rr;
            int index = scene.roundedRects.ColliderIndex(a.parent.instanceId);
            scene.roundedRects.physicsParents[index] = a.parent;
            platforms.Add(id, platform);
        }
        return platforms;
    }
    internal static JToken Evaluate(JObject query, DetPhysics scene, List<ReferencePhysicsScene.Adapter> adapters, Func<JObject> snapshot)
    {
        var objects = new List<GameObject>();
        var platforms = new Dictionary<int, StickyRoundedRectangle>();
        try
        {
            platforms = BindPlatforms(query["bindings"], scene, adapters, objects);
            var go = new GameObject("surface movement oracle body");
            go.SetActive(false); objects.Add(go);
            var ft = go.AddComponent<FixTransform>();
            var body = go.AddComponent<PlayerBody>();
            Set(body, "fixTransform", ft);
            var frames = new JArray();
            JObject Frame(JToken result)
            {
                var world = snapshot();
                world["objects"] = new JArray(world["objects"].OrderBy(o => (int)o["id"]));
                JArray Order(PhysicsParent[] pp, int count) { var ids = new JArray(); for(int i=0;i<count;i++)ids.Add(pp[i].instanceId);return ids; }
                return new JObject { ["scene"] = world, ["orders"] = new JArray(Order(scene.roundedRects.physicsParents,scene.roundedRects.Length),Order(scene.circles.physicsParents,scene.circles.Length),Order(scene.boxes.physicsParents,scene.boxes.Length)), ["result"] = result };
            }
            JToken Id(StickyRoundedRectangle p) => p == null ? JValue.CreateNull() : new JValue(platforms.First(v => v.Value == p).Key);
            foreach (var action in query["actions"])
            {
                var request = action["request"];
                var platform = platforms[(int)action["object"]];
                body.position = V(request["position"]);
                RaycastHit2D[] buffer = (bool)request["blocked"] ? new RaycastHit2D[20] : null;
                JToken result = JValue.CreateNull();
                string error = null;
                try
                {
                    switch ((string)action["op"])
                    {
                        case "move": result = platform.moveAlongSurface(body,F(request["radius"]),F(request["speed"]),F(request["dt"]),F(request["local"]),ref buffer).m_rawValue; break;
                        case "multiple": result = platform.moveAlongSurfaceMultiple(body,F(request["radius"]),F(request["speed"]),F(request["dt"]),F(request["local"]),V(action["second"][0]),F(action["second"][1]),ref buffer).m_rawValue; break;
                        case "hop":
                            var disallowed = action["disallowed"].ToObject<int[]>();
                            var local = platform.PlatformHopMove(body,F(request["radius"]),F(request["speed"]),F(request["dt"]),F(request["local"]),ref buffer,ref disallowed,out var blocking,out var remaining);
                            result = new JObject { ["local"] = local.m_rawValue, ["remaining"] = remaining.m_rawValue, ["blocking"] = Id(blocking) }; break;
                        case "limit":
                            object[] args = { F(request["radius"]),body.position,V(action["velocity"]),F(request["dt"]),buffer,null };
                            Fix limit = (Fix)typeof(StickyRoundedRectangle).GetMethod("maxMovementAllowed",BindingFlags.NonPublic|BindingFlags.Instance).Invoke(platform,args);
                            result = new JArray(limit.m_rawValue,Id((StickyRoundedRectangle)args[5])); break;
                        default: throw new ArgumentException("Unknown surface motion action");
                    }
                }
                catch (Exception e)
                {
                    if (e is TargetInvocationException) e = e.InnerException;
                    if (e is NullReferenceException) error = "null_reference";
                    else if (e is DivideByZeroException) error = "division_by_zero";
                    else throw;
                }
                if (error != null) return new JObject { ["error"] = error, ["action"] = frames.Count, ["completed"] = frames, ["failed"] = Frame(JValue.CreateNull()) };
                frames.Add(Frame(result));
            }
            return frames;
        }
        finally { foreach(var go in objects)if(go != null)UnityEngine.Object.DestroyImmediate(go); }
    }
}
