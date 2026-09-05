using System;
using System.Collections.Generic;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;

// Optional stage-level evidence taken from real gameplay, before ability callbacks.
// Includes rope physics. Beam subsystem state still needs a replay-stage adapter.
public static class ReferencePhysicsCapture
{
    public static bool Enabled => ReferenceCapture.Enabled && Environment.GetEnvironmentVariable("BOPL_TRACE_PHYSICS")=="1";
    public static JObject LastStep {get;private set;}
    private static readonly Dictionary<int,int> ids=new Dictionary<int,int>();
    private static readonly BindingFlags flags=BindingFlags.NonPublic|BindingFlags.Instance;
    internal static Func<int,int> IdentityOverride;
    private static int Id(int instance) {if(IdentityOverride!=null)return IdentityOverride(instance);if(!ids.TryGetValue(instance,out int id)){id=ids.Count+1;ids.Add(instance,id);}return id;}
    private static JArray Ids(List<int> instances) {var result=new JArray();foreach(int id in instances)result.Add(Id(id));return result;}
    public static void Begin(DetPhysics p, Fix dt)
    {
        if(!Enabled)return;
        int ropes=0;foreach(var r in p.ropes.ropeBodies)if(r.enabled)ropes++;
        LastStep=new JObject {
            ["tick"]=Updater.SimulationTicks,["dt"]=(GameTime.IsTimeStopped()?Fix.Zero:dt).m_rawValue,
            ["ropes"]=ropes,["beams"]=(int)typeof(DetPhysics).GetField("beamsThisFrame",flags).GetValue(p),["before"]=Scene(p)
        };
    }
    public static void End(DetPhysics p)
    {
        if(!Enabled)return;
        if(LastStep==null)throw new InvalidOperationException("Physics capture without begin");
        LastStep["after"]=Scene(p);
        var events=new JArray();
        foreach(var c in (List<CollisionInformation>)typeof(DetPhysics).GetField("ToMakeCallBacks",flags).GetValue(p))
            events.Add(new JObject {["receiver"]=Id(c.pp.instanceId),["other"]=Id(c.colliderPP.instanceId),["collider"]=ReferencePhysicsScene.ShapeJson(c.collider),["layer"]=c.layer,["impact_velocity"]=V(c.colliderImpactVelocity),
                ["contact"]=new JObject {["penetration"]=c.penetration.m_rawValue,["point"]=V(c.contactPoint),["normal"]=V(c.normal)}});
        LastStep["callbacks"]=events;
    }
    public static JObject Take() {var result=LastStep;LastStep=null;return result;}
    internal static JObject Scene(DetPhysics p)
    {
        var objects=new JArray();
        Append(objects,p.circles,p);Append(objects,p.boxes,p);Append(objects,p.roundedRects,p);
        var composites=new JArray();
        foreach(var c in p.composites)composites.Add(new JObject {["body"]=Body(c.combinedBody),["position"]=V(c.position),["destroyed"]=c.IsDestroyed,
            ["circles"]=Ids(c.circleIds),["boxes"]=Ids(c.boxIds),["rounded_rects"]=Ids(c.roundedRectIds)});
        var layers=new JArray();
        for(int i=0;i<32;i++){uint mask=0;for(int j=0;j<32;j++)if(!Physics2D.GetIgnoreLayerCollision(i,j))mask|=1u<<j;layers.Add(mask);}
        var ropes=new JArray();foreach(var r in p.ropes.ropeBodies)ropes.Add(ReferenceRope.State(r));
        return new JObject {["objects"]=objects,["composites"]=composites,["layers"]=layers,["gravity"]=p.gravityAccel.m_rawValue,
            ["rope_system"]=new JObject {["bodies"]=ropes,["gravity"]=p.ropeGravity.m_rawValue,["iterations"]=p.ropeRelaxationIterations,["stretchiness"]=p.platformStretchiness.m_rawValue,
                ["space"]=(LevelType)typeof(DetPhysics).GetField("levelTypeThisFrame",BindingFlags.Static|BindingFlags.NonPublic).GetValue(null)==LevelType.space}};
    }
    private static void Append<T>(JArray result, PhysicsBodyList<T> list, DetPhysics p) where T:struct,IShape
    {
        for(int i=0;i<list.Length;i++)
        {
            var parent=list.physicsParents[i];var ft=parent.fixTrans;var body=list.physicsBodies[i];
            result.Add(new JObject {
                ["id"]=Id(parent.instanceId),["collider"]=ReferencePhysicsScene.ShapeJson(list.colliders[i]),["body"]=Body(body),["layer"]=list.colliders[i].Layer(),
                ["transform"]=new JObject {["position"]=V(ft==null?Vec2.zero:ft.position),["rotation"]=ft==null?0:ft.rotation.m_rawValue,["offset"]=V(ft==null?Vec2.zero:ft.offset),["right"]=V(ft==null?Vec2.right:ft.right)},
                ["active"]=DetPhysics.IsActive(parent),["destroyed"]=ft==null||parent.monobehaviourCollider.IsDestroyed,
                ["callback"]=parent.collisionCallback!=null,["callback_destroyed"]=parent.collisionCallback!=null&&parent.collisionCallback.IsDestroyed,
                ["composite"]=body.compositeParent==null?JValue.CreateNull():new JValue(p.composites.IndexOf(body.compositeParent)),
                ["relative_pos"]=V(body.relativePos),["relative_orientation"]=body.relativeOrientation.m_rawValue
            });
        }
    }
}
