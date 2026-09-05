using System;
using System.Collections.Generic;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

public static class ReferenceBeamEffects
{
    public static JToken Evaluate(JObject query)
    {
        var physics = DetPhysics.Get();
        var oldBoxes = physics.boxes; var oldCircles = physics.circles;
        var oldObjects = Get<BeamObject[]>(physics, "beamObjectPool");
        var oldEffects = Get<BeamHitAnim[]>(physics, "beamHitEffectPool");
        var oldPlayers = PlayerHandler.Get().PlayerList();
        var oldWater = Constants.WATER_HEIGHT; var oldTime = GameTime.PlayerTimeScale;
        var objects = new BeamObject[512]; var effects = new BeamHitAnim[512];
        var targets = new List<ReferenceBeamTargets>();
        physics.boxes = new PhysicsBodyList<Box>(); physics.circles = new PhysicsBodyList<Circle>();
        Set(physics, "beamObjectPool", objects); Set(physics, "beamHitEffectPool", effects);
        var config = query["config"];
        Constants.WATER_HEIGHT = F(config["water"]); GameTime.PlayerTimeScale = F(config["time_scale"]);
        var players = new List<Player>();
        for (int i = 0; i < 6; i++) players.Add(new Player(i + 1, i) { isProtectedFromTimeStop = (bool)config["protected"][i] });
        PlayerHandler.Get().SetPlayerList(players);
        physics.beamPlatformPushForce = F(config["platform_push"]);
        physics.beamBlackHolePushForce = F(config["black_hole_push"]);
        physics.beamRigidBodyPushFOrce = F(config["body_push"]);
        physics.beamWidthAnim = new AnimationCurveFixed { keys = new[] {
            new KeyFrameF { time = Fix.Zero, value = Fix.One },
            new KeyFrameF { time = Fix.One, value = Fix.One }
        }};
        typeof(AnimationCurveFixed).GetField("Anim", Fields).SetValue(physics.beamWidthAnim, new AnimationCurve());
        Set(physics, "keyCountLastFrame", 0);
        try
        {
            foreach (JObject target in query["targets"]) targets.Add(new ReferenceBeamTargets(target));
            var result = new JArray();
            foreach (JArray frame in query["frames"])
            {
                Set(physics, "beamsThisFrame", frame.Count);
                for (int i = 0; i < frame.Count; i++)
                {
                    var impact = frame[i]; var b = impact["beam"];
                    physics.beamBodies[i] = new DetPhysics.BeamBody {
                        id = (int)b["id"], ownerId = (int)b["owner_id"], position = V(b["position"]), direction = V(b["direction"]),
                        scale = F(b["scale"]), distance = F(b["distance"]), timePassed = Fix.Zero,
                        colors = ReferenceBeam.ReadColors(b["colors"])
                    };
                    physics.beamRoles[i] = (DetPhysics.BeamRole)Enum.Parse(typeof(DetPhysics.BeamRole), (string)impact["role"]);
                    Get<bool[]>(physics, "beamIsReflected")[i] = (bool)impact["water_reflection"];
                    Get<FixTransform[]>(physics, "beamRaycastHitRRs")[i] = impact["target"].Type == JTokenType.Null ? null : targets[(int)impact["target"]].transform;
                }
                try { typeof(DetPhysics).GetMethod("renderBeams", Fields).Invoke(physics, new object[] { 0 }); }
                catch (TargetInvocationException error) { throw error.InnerException; }
                var states = new JArray(); foreach (var target in targets) states.Add(target.Snapshot()); result.Add(states);
            }
            return result;
        }
        finally
        {
            foreach (var beam in objects) if (beam != null) UnityEngine.Object.DestroyImmediate(beam.gameObject);
            foreach (var effect in effects) if (effect != null) UnityEngine.Object.DestroyImmediate(effect.gameObject);
            foreach (var target in targets) UnityEngine.Object.DestroyImmediate(target.gameObject);
            physics.boxes = oldBoxes; physics.circles = oldCircles;
            Set(physics, "beamObjectPool", oldObjects); Set(physics, "beamHitEffectPool", oldEffects);
            PlayerHandler.Get().SetPlayerList(oldPlayers); Constants.WATER_HEIGHT = oldWater; GameTime.PlayerTimeScale = oldTime;
        }
    }
}
