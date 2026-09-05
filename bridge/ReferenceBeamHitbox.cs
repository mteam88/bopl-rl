using System;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;

// Runs the real prefab's BeamObject.SetDimensions, Initialize and DPhysicsBox
// methods. Gameplay collision callbacks are a subsequent stage.
public static class ReferenceBeamHitbox
{
    private const BindingFlags Fields = BindingFlags.Instance | BindingFlags.NonPublic;
    internal static JObject Snapshot(BeamObject beam)
    {
        var physics = DetPhysics.Get();
        var hitbox = beam.GetComponent<DPhysicsBox>();
        var transform = beam.GetComponent<FixTransform>();
        JToken collider = JValue.CreateNull();
        JToken orientation = JValue.CreateNull();
        if (hitbox.initHasBeenCalled && physics.boxes.ContainsParent(hitbox.GetPhysicsParent().instanceId))
        {
            int index = physics.boxes.ColliderIndex(hitbox.GetPhysicsParent().instanceId);
            var box = physics.boxes.colliders[index];
            collider = new JObject { ["center"] = V(box.center), ["right"] = V(box.right),
                ["up"] = V(box.up), ["inverse_extents"] = V(box.inverseExtents) };
            orientation = new JValue(physics.boxes.physicsBodies[index].orientation.m_rawValue);
        }
        return new JObject {
            ["position"] = V(transform.position), ["right"] = V(transform.right),
            ["rotation"] = transform.rotation.m_rawValue, ["offset"] = V(transform.offset),
            ["start_extents"] = V((Vec2)typeof(DPhysicsBox).GetField("startExtents", Fields | BindingFlags.Public).GetValue(hitbox)),
            ["owner_id"] = beam.GetComponent<Explosion>().PlayerOwnerId, ["beam_id"] = beam.beamId,
            ["collider"] = collider, ["orientation"] = orientation
        };
    }
    public static JToken Evaluate(JObject query)
    {
        var physics = DetPhysics.Get();
        var oldBoxes = physics.boxes;
        physics.boxes = new PhysicsBodyList<Box>();
        BeamObject beam = null;
        try
        {
            beam = FixTransform.InstantiateFixed(physics.beamObjectPrefab, Vec2.zero);
            var hitbox = beam.GetComponent<DPhysicsBox>();
            var transform = beam.GetComponent<FixTransform>();
            if (beam.GetComponent<BoplBody>() != null) throw new InvalidOperationException("Expected trigger beam prefab");
            var config = query["config"];
            beam.baseWidth01 = F(config["base_width"]);
            beam.extraLength = F(config["extra_length"]);
            hitbox.minimumExtents = F(config["minimum_extents"]);
            var result = new JArray();
            foreach (var action in query["actions"])
            {
                if ((string)action["operation"] == "initialize")
                {
                    beam.Initialize();
                }
                else
                {
                    var d = action["dimensions"];
                    beam.SetDimensions(V(d["position"]), F(d["width"]), V(d["direction"]), F(d["distance"]),
                        F(d["age"]), ReferenceCurve.Read(d["curve"]), (int)d["player_id"], (int)d["beam_id"]);
                }
                result.Add(Snapshot(beam));
            }
            return result;
        }
        finally
        {
            if (beam != null) UnityEngine.Object.DestroyImmediate(beam.gameObject);
            physics.boxes = oldBoxes;
        }
    }
}
