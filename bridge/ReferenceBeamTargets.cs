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
    private readonly IPhysicsCollider sceneCollider;
    private BoplBody body;
    private BlackHole blackHole;
    private Orb orb;
    internal const BindingFlags Fields = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
    internal static void Set(object obj, string name, object value) => obj.GetType().GetField(name, Fields).SetValue(obj, value);
    internal static T Get<T>(object obj, string name) => (T)obj.GetType().GetField(name, Fields).GetValue(obj);

    internal ReferenceBeamTargets(JObject input, GameObject existing = null)
    {
        source = input;
        gameObject = existing != null ? existing : new GameObject("Beam effect target");
        transform = existing != null ? existing.GetComponent<FixTransform>() : gameObject.AddComponent<FixTransform>();
        sceneCollider = existing != null ? existing.GetComponent<IPhysicsCollider>() : null;
        var sprite = gameObject.AddComponent<SpriteRenderer>();
        string kind = (string)input["kind"];
        // Use DetPhysics's actual cached layer numbers for the effect branches.
        if(existing == null) gameObject.layer = kind == "wall" ? DetPhysics.wallLayer : kind == "terrain" ? DetPhysics.terrainLayer :
            kind == "lethal" ? DetPhysics.lethalterrainLayer : kind == "player" ? DetPhysics.playerLayer :
            kind == "affector" ? DetPhysics.affectorLayer : kind == "orb" ? DetPhysics.orbLayer : 0;
        if ((bool)input["component"])
        {
            if (kind == "wall" || kind == "terrain" || kind == "lethal" || kind == "player")
            {
                if (existing == null)
                {
                    box = gameObject.AddComponent<DPhysicsBox>();
                    box.SetExtents(new Vec2(Fix.One, Fix.One));
                    if (kind == "player") body = gameObject.AddComponent<BoplBody>();
                    box.ManualInit();
                    var p = DetPhysics.Get();
                    p.boxes.physicsBodies[p.boxes.ColliderIndex(box.GetPhysicsParent().instanceId)].force = V(input["force"]);
                }
                else if (kind == "player") body = gameObject.AddComponent<BoplBody>();
                if (body != null) body.IsDestroyed = (bool)input["component_destroyed"];
                else (sceneCollider ?? box).IsDestroyed = (bool)input["component_destroyed"];
            }
            else if (kind == "affector" || kind == "orb")
            {
                if(existing == null) gameObject.AddComponent<DPhysicsCircle>();
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
        if (sceneCollider != null)
        {
            var p = DetPhysics.Get(); int id = sceneCollider.GetPhysicsParent().instanceId;
            bool present = sceneCollider.shape == Shape.Circle ? p.circles.ContainsParent(id) : sceneCollider.shape == Shape.Box ? p.boxes.ContainsParent(id) : p.roundedRects.ContainsParent(id);
            if(present) state["force"] = V(sceneCollider.GetBody().force);
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
