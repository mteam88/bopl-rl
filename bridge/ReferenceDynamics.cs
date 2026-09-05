using System;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;

public static class ReferenceDynamics
{
    internal static Fix F(JToken value) => Fix.FromRaw((long)value);
    internal static Vec2 V(JToken value) => new Vec2(F(value["x"]), F(value["y"]));
    internal static JObject V(Vec2 value) => new JObject { ["x"] = value.x.m_rawValue, ["y"] = value.y.m_rawValue };
    internal static Box Box(JToken value) => new Box { center = V(value["center"]), right = V(value["right"]), up = V(value["up"]), inverseExtents = V(value["inverse_extents"]) };
    internal static Circle Circle(JToken value) => new Circle { center = V(value["center"]), radius = F(value["radius"]) };
    internal static RoundedRect Rounded(JToken value) => new RoundedRect { box = Box(value["inner"]), radius = F(value["radius"]) };
    internal static PhysicsBody Body(JToken v) => new PhysicsBody { inverseMass=F(v["inverse_mass"]),velocity=V(v["velocity"]),force=V(v["force"]),
        angularVelocity=F(v["angular_velocity"]),orientation=F(v["orientation"]),inverseMomentOfInertia=F(v["inverse_inertia"]),bounciness=F(v["bounciness"]),
        staticFriction=F(v["static_friction"]),dynamicFriction=F(v["dynamic_friction"]),gravityScale=F(v["gravity_scale"]),airDrag=F(v["air_drag"]),rotationalDrag=F(v["rotational_drag"]) };
    internal static JObject Body(PhysicsBody v) => new JObject { ["inverse_mass"]=v.inverseMass.m_rawValue,["velocity"]=V(v.velocity),["force"]=V(v.force),
        ["angular_velocity"]=v.angularVelocity.m_rawValue,["orientation"]=v.orientation.m_rawValue,["inverse_inertia"]=v.inverseMomentOfInertia.m_rawValue,["bounciness"]=v.bounciness.m_rawValue,
        ["static_friction"]=v.staticFriction.m_rawValue,["dynamic_friction"]=v.dynamicFriction.m_rawValue,["gravity_scale"]=v.gravityScale.m_rawValue,["air_drag"]=v.airDrag.m_rawValue,["rotational_drag"]=v.rotationalDrag.m_rawValue };
    internal static CollisionManifold Contact(JToken v) => new CollisionManifold { penetration=F(v["penetration"]),normal=V(v["normal"]),contactPoint=V(v["point"]) };
    public static JToken Evaluate(JObject q)
    {
        if((string)q["operation"]=="dynamics_player_physics")return ReferencePlayerPhysics.Evaluate(q);
        if((string)q["operation"]=="dynamics_player_body")return ReferencePlayerBody.Evaluate(q);
        if((string)q["operation"]=="dynamics_player_ground")return ReferencePlayerGround.Evaluate(q);
        if((string)q["operation"]=="dynamics_player_collision")return ReferenceCollisionPolicy.Evaluate(q);
        if((string)q["operation"]=="dynamics_player_life")return ReferencePlayerLife.Evaluate(q);
        if((string)q["operation"]=="dynamics_curve")return ReferenceCurve.Evaluate(q);
        string op = (string)q["operation"];
        if(op.StartsWith("dynamics_beam_",StringComparison.Ordinal))return ReferenceBeam.Evaluate(q);
        CollisionManifold contact = default;
        bool hit;
        switch (op)
        {
            case "dynamics_rope": return ReferenceRope.Evaluate(q);
            case "dynamics_scene": return ReferencePhysicsScene.Evaluate(q);
            case "dynamics_player_rays": return ReferencePhysicsScene.Evaluate(q);
            case "dynamics_circle_circle": hit=PhysTools.CollisionTest(Circle(q["a"]),Circle(q["b"]),ref contact);break;
            case "dynamics_circle_box": hit=PhysTools.CollisionTest(Circle(q["a"]),Box(q["b"]),ref contact);break;
            case "dynamics_circle_rounded": hit=PhysTools.CollisionTest(Circle(q["a"]),Rounded(q["b"]),ref contact);break;
            case "dynamics_box_box": hit=PhysTools.CollisionTest(Box(q["a"]),Box(q["b"]),ref contact);break;
            case "dynamics_box_rounded": hit=PhysTools.CollisionTest(Box(q["a"]),Rounded(q["b"]),ref contact);break;
            case "dynamics_rounded_rounded": hit=PhysTools.CollisionTest(Rounded(q["a"]),Rounded(q["b"]),ref contact);break;
            case "dynamics_integrate":
                var physics=DetPhysics.Get();physics.gravityAccel=F(q["gravity"]);
                var args=new object[] {Body(q["body"]),F(q["dt"])};
                Vec2 translation;
                try { translation=(Vec2)typeof(DetPhysics).GetMethod("IntegrateBody",BindingFlags.NonPublic|BindingFlags.Instance).Invoke(physics,args); }
                catch(TargetInvocationException e) { throw e.InnerException; }
                return new JObject { ["translation"]=V(translation),["body"]=Body((PhysicsBody)args[0]) };
            case "dynamics_resolve":
                var a=Body(q["a"]);var b=Body(q["b"]);
                PhysTools.ResolveCollision(Contact(q["contact"]),ref a,ref b,V(q["pos_a"]),V(q["pos_b"]));
                return new JObject {["a"]=Body(a),["b"]=Body(b)};
            case "dynamics_correction":
                var correction=(bool)q["full"] ? PhysTools.FullPositionalCorrection(Contact(q["contact"]),F(q["inverse_a"]),F(q["inverse_b"]))
                    : PhysTools.PositionalCorrection(Contact(q["contact"]),F(q["inverse_a"]),F(q["inverse_b"]));
                return new JArray(V(correction.a),V(correction.b));
            default: throw new ArgumentException("Unknown dynamics query");
        }
        return hit ? new JObject {["penetration"]=contact.penetration.m_rawValue,["normal"]=V(contact.normal),["point"]=V(contact.contactPoint)} : JValue.CreateNull();
    }
}
