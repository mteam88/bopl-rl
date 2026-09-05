using System;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;

// Real gameplay components, initialized only as far as the beam effect methods
// require. Their force/charge methods execute unchanged.
internal sealed class ReferenceBeamTargets
{
    internal readonly GameObject gameObject;
    internal readonly FixTransform transform;
    private readonly JObject source;
    private DPhysicsBox box;
    private BoplBody body;
    private BlackHole blackHole;
    private Orb orb;
    internal const BindingFlags Fields = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
    internal static void Set(object obj, string name, object value) => obj.GetType().GetField(name, Fields).SetValue(obj, value);
    internal static T Get<T>(object obj, string name) => (T)obj.GetType().GetField(name, Fields).GetValue(obj);

    internal ReferenceBeamTargets(JObject input)
    {
        source = input;
        gameObject = new GameObject("Beam effect target");
        transform = gameObject.AddComponent<FixTransform>();
        var sprite = gameObject.AddComponent<SpriteRenderer>();
        string kind = (string)input["kind"];
        // Use DetPhysics's actual cached layer numbers for the effect branches.
        gameObject.layer = kind == "wall" ? DetPhysics.wallLayer : kind == "terrain" ? DetPhysics.terrainLayer :
            kind == "lethal" ? DetPhysics.lethalterrainLayer : kind == "player" ? DetPhysics.playerLayer :
            kind == "affector" ? DetPhysics.affectorLayer : kind == "orb" ? DetPhysics.orbLayer : 0;
        if ((bool)input["component"])
        {
            if (kind == "wall" || kind == "terrain" || kind == "lethal" || kind == "player")
            {
                box = gameObject.AddComponent<DPhysicsBox>();
                box.SetExtents(new Vec2(Fix.One, Fix.One));
                if (kind == "player") body = gameObject.AddComponent<BoplBody>();
                box.ManualInit();
                var p = DetPhysics.Get();
                p.boxes.physicsBodies[p.boxes.ColliderIndex(box.GetPhysicsParent().instanceId)].force = V(input["force"]);
                if (body != null) body.IsDestroyed = (bool)input["component_destroyed"];
                else box.IsDestroyed = (bool)input["component_destroyed"];
            }
            else if (kind == "affector" || kind == "orb")
            {
                var circle = gameObject.AddComponent<DPhysicsCircle>();
                if (kind == "affector")
                {
                    blackHole = gameObject.AddComponent<BlackHole>();
                    Set(blackHole, "mass", F(input["mass"]));
                    Set(blackHole, "velocity", V(input["velocity"]));
                    blackHole.IsDestroyed = (bool)input["component_destroyed"];
                }
                else
                {
                    var arrow = new GameObject("arrow"); arrow.transform.SetParent(gameObject.transform); arrow.AddComponent<SpriteRenderer>();
                    orb = gameObject.AddComponent<Orb>();
                    Set(orb, "innerCircle", sprite);
                    Set(orb, "orbType", (Orb.OrbType)(int)input["orb"]["kind"]);
                    Set(orb, "beamCharge01", F(input["orb"]["charge"]));
                    Set(orb, "isFiringBeam", (bool)input["orb"]["firing"]);
                    Set(orb, "beamColors", ReferenceBeam.ReadColors(input["orb"]["colors"]));
                    orb.beamChargeRate01 = F(input["orb"]["rate"]);
                    orb.IsDestroyed = (bool)input["component_destroyed"];
                }
            }
        }
        transform.IsDestroyed = (bool)input["destroyed"];
    }
    internal JObject Snapshot()
    {
        var state = (JObject)source.DeepClone();
        if (box != null)
        {
            var p = DetPhysics.Get();
            state["force"] = V(p.boxes.physicsBodies[p.boxes.ColliderIndex(box.GetPhysicsParent().instanceId)].force);
        }
        if (blackHole != null) state["velocity"] = V(Get<Vec2>(blackHole, "velocity"));
        if (orb != null)
        {
            state["orb"]["kind"] = (int)Get<Orb.OrbType>(orb, "orbType");
            state["orb"]["charge"] = Get<Fix>(orb, "beamCharge01").m_rawValue;
            state["orb"]["firing"] = Get<bool>(orb, "isFiringBeam");
            state["orb"]["colors"] = ReferenceBeam.ColorsJson(Get<DetPhysics.BeamColors>(orb, "beamColors"));
        }
        return state;
    }
}
