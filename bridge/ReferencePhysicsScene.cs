using System;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;

// Builds isolated inputs and invokes the actual DetPhysics.Simulate implementation.
// The collider adapter only supplies engine object state and records callbacks;
// it contains no integration, collision, ordering or resolution implementation.
public static class ReferencePhysicsScene
{
    public static bool Capturing;
    private static bool sweepX, roundedSweepX;
    private static readonly JArray events = new JArray();
    private static DetPhysics physics;
    private static RopeBody[] originalRopes;
    private static readonly string[] pairFields = { "potentialColliderIds_RR_RR", "potentialColliderIds_BoxRR", "potentialColliderIds_CircleRR", "potentialColliderIds_CircleCircle", "potentialColliderIds_CircleBox", "potentialColliderIds_BoxBox" };
    private static readonly Shape[,] pairShapes = { {Shape.RoundedRect,Shape.RoundedRect}, {Shape.Box,Shape.RoundedRect}, {Shape.Circle,Shape.RoundedRect}, {Shape.Circle,Shape.Circle}, {Shape.Circle,Shape.Box}, {Shape.Box,Shape.Box} };
    private static readonly BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
    public static void SweepAxes(bool x, bool roundedX) { if (Capturing) { sweepX=x;roundedSweepX=roundedX; } }
    private static JObject BoxJson(Box b) => new JObject { ["center"]=V(b.center),["right"]=V(b.right),["up"]=V(b.up),["inverse_extents"]=V(b.inverseExtents) };
    internal static JObject ShapeJson(IShape shape)
    {
        if (shape is Circle c) return new JObject { ["kind"]="circle",["shape"]=new JObject {["center"]=V(c.center),["radius"]=c.radius.m_rawValue} };
        if (shape is Box b) return new JObject { ["kind"]="box",["shape"]=BoxJson(b) };
        var r=(RoundedRect)shape;
        return new JObject { ["kind"]="rounded_rect",["shape"]=new JObject {["inner"]=BoxJson(r.box),["radius"]=r.radius.m_rawValue} };
    }
    private static PhysicsParent Parent(Shape shape, int index) => shape==Shape.Circle ? physics.circles.physicsParents[index] : shape==Shape.Box ? physics.boxes.physicsParents[index] : physics.roundedRects.physicsParents[index];
    private static (IShape,PhysicsBody) State(Adapter a)
    {
        if (a.shape==Shape.Circle) {int i=physics.circles.ColliderIndex(a.parent.instanceId);return (physics.circles.colliders[i],physics.circles.physicsBodies[i]);}
        if (a.shape==Shape.Box) {int i=physics.boxes.ColliderIndex(a.parent.instanceId);return (physics.boxes.colliders[i],physics.boxes.physicsBodies[i]);}
        int j=physics.roundedRects.ColliderIndex(a.parent.instanceId);return (physics.roundedRects.colliders[j],physics.roundedRects.physicsBodies[j]);
    }
    private static bool Contains(Adapter a) => a.shape==Shape.Circle ? physics.circles.ContainsParent(a.parent.instanceId) : a.shape==Shape.Box ? physics.boxes.ContainsParent(a.parent.instanceId) : physics.roundedRects.ContainsParent(a.parent.instanceId);
    private static JObject CompositeJson(CompositeBody c) => new JObject { ["body"]=Body(c.combinedBody),["position"]=V(c.position),["destroyed"]=c.IsDestroyed,["circles"]=new JArray(c.circleIds),["boxes"]=new JArray(c.boxIds),["rounded_rects"]=new JArray(c.roundedRectIds) };
    private static void Command(JToken v, List<Adapter> adapters)
    {
        int id=(int)v["id"];var a=adapters.Find(x=>x.parent.instanceId==id);
        switch((string)v["op"])
        {
            case "attach": if(a.shape==Shape.Circle)physics.AddCircleToRR(id,(int)v["root"]);else physics.AddBoxToRR(id,(int)v["root"]);break;
            case "detach":
                var body=State(a).Item2;
                if(a.shape==Shape.Circle)physics.RemoveCircleFromComposite(id,body.compositeParent);
                else if(a.shape==Shape.Box)physics.RemoveBoxFromComposite(id,body.compositeParent);
                else physics.RemoveRoundedRectFromComposite(id,body.compositeParent);
                break;
            case "destroy":
                if(a.shape==Shape.Circle)physics.DestroyCircle(a.parent);else if(a.shape==Shape.Box)physics.DestroyBox(a.parent);else physics.DestroyRoundedRect(a.parent);
                break;
            case "force":
                if(a.shape==Shape.Circle)physics.circles.physicsBodies[physics.circles.ColliderIndex(id)].force+=V(v["force"]);
                else if(a.shape==Shape.Box)physics.boxes.physicsBodies[physics.boxes.ColliderIndex(id)].force+=V(v["force"]);
                else physics.roundedRects.physicsBodies[physics.roundedRects.ColliderIndex(id)].force+=V(v["force"]);
                break;
            default:throw new ArgumentException("Unknown scene command");
        }
    }
    private static JObject Snapshot(JObject scene, JArray objects, List<Adapter> adapters)
    {
        var snapshot=(JObject)scene.DeepClone();var states=new JArray();
        for(int i=0;i<adapters.Count;i++)
        {
            var a=adapters[i];if(!Contains(a))continue;
            var (shape,body)=State(a);var o=(JObject)objects[i].DeepClone();
            if(a.parent.monobehaviourCollider.enabled)o.Remove("enabled");else o["enabled"]=false;
            o["body"]=Body(body);o["collider"]=ShapeJson(shape);
            o["composite"]=body.compositeParent==null ? JValue.CreateNull() : new JValue(physics.composites.IndexOf(body.compositeParent));
            o["relative_pos"]=V(body.relativePos);o["relative_orientation"]=body.relativeOrientation.m_rawValue;
            o["transform"]=new JObject {["position"]=V(a.transform.position),["rotation"]=a.transform.rotation.m_rawValue,["offset"]=V(a.transform.offset)};
            if(objects[i]["transform"]["right"]!=null)o["transform"]["right"]=V(a.transform.right);
            states.Add(o);
        }
        snapshot["objects"]=states;
        if(scene["rope_system"]!=null){var bodies=new JArray();foreach(var r in physics.ropes.ropeBodies)bodies.Add(ReferenceRope.State(r));snapshot["rope_system"]["bodies"]=bodies;}
        var composites=new JArray();foreach(var c in physics.composites)composites.Add(CompositeJson(c));snapshot["composites"]=composites;
        foreach(var c in composites)foreach(string field in new[]{"circles","boxes","rounded_rects"})
            c[field]=new JArray(c[field].Select(id => adapters.Find(a=>a.parent.instanceId==(int)id)?.sourceId ?? (int)id));
        return snapshot;
    }
    public static JToken Evaluate(JObject query)
    {
        if (physics==null) {physics=new GameObject("Reference physics scene").AddComponent<DetPhysics>();originalRopes=physics.ropes.ropeBodies;}
        var scene=(JObject)query["scene"].DeepClone();
        var objects=(JArray)scene["objects"];
        var adapters=new List<Adapter>();
        physics.circles=new PhysicsBodyList<Circle>();physics.boxes=new PhysicsBodyList<Box>();physics.roundedRects=new PhysicsBodyList<RoundedRect>();
        physics.composites.Clear();
        var ropeSystem=scene["rope_system"];
        if(ropeSystem!=null)
        {
            var bodies=new List<RopeBody>();
            foreach(var v in ropeSystem["bodies"]){var r=ReferenceRope.Read(v);r.indexInArray=bodies.Count;bodies.Add(r);}
            physics.ropes.ropeBodies=bodies.ToArray();
            physics.ropeGravity=F(ropeSystem["gravity"]);physics.ropeRelaxationIterations=(int)ropeSystem["iterations"];physics.platformStretchiness=F(ropeSystem["stretchiness"]);
            typeof(DetPhysics).GetField("levelTypeThisFrame",BindingFlags.Static|BindingFlags.NonPublic).SetValue(null,(bool)ropeSystem["space"]?LevelType.space:LevelType.grass);
        }
        else physics.ropes.ropeBodies=Array.Empty<RopeBody>();
        typeof(DetPhysics).GetField("collisionCount",flags).SetValue(physics,0);
        Array.Clear((AABB[])typeof(DetPhysics).GetField("boundingBoxes",flags).GetValue(physics),0,4096);
        Array.Clear((AABB[])typeof(DetPhysics).GetField("bb_rope_rr",flags).GetValue(physics),0,1024);
        physics.gravityAccel=F(scene["gravity"]);GameTime.PlayerTimeScale=Fix.One;
        for(int i=0;i<32;i++) for(int j=i;j<32;j++) Physics2D.IgnoreLayerCollision(i,j,((uint)scene["layers"][i] & (1u<<j))==0);
        foreach(var v in scene["composites"])
        {
            var c=CompositeBody.CreateNewComposite(physics.composites);
            c.combinedBody=Body(v["body"]);c.position=V(v["position"]);
            foreach(var id in v["circles"])c.circleIds.Add((int)id);
            foreach(var id in v["boxes"])c.boxIds.Add((int)id);
            foreach(var id in v["rounded_rects"])c.roundedRectIds.Add((int)id);
        }
        // Set destroyed flags after allocation so the original reuse helper cannot
        // reuse a slot that was explicitly present in the input scene.
        for(int i=0;i<physics.composites.Count;i++)physics.composites[i].IsDestroyed=(bool)scene["composites"][i]["destroyed"];
        Capturing=true;
        try
        {
            foreach(JObject v in objects)
            {
                var go=new GameObject("Reference collider");
                var ft=go.AddComponent<FixTransform>();
                ft.position=V(v["transform"]["position"]);ft.rotation=F(v["transform"]["rotation"]);ft.offset=V(v["transform"]["offset"]);
                if(v["transform"]["right"]!=null)typeof(FixTransform).GetField("rightInner",flags).SetValue(ft,V(v["transform"]["right"]));
                go.SetActive((bool)v["active"]);
                var a=new Adapter { transform=ft, IsDestroyed=(bool)v["destroyed"], callbackDestroyed=(bool)v["callback_destroyed"], shape=(string)v["collider"]["kind"]=="circle"?Shape.Circle:(string)v["collider"]["kind"]=="box"?Shape.Box:Shape.RoundedRect };
                a.parent=new PhysicsParent { instanceId=(string)query["operation"]=="dynamics_player_controller"?go.GetInstanceID():(int)v["id"],fixTrans=ft,monobehaviourCollider=a,collisionCallback=(bool)v["callback"]?a:null,hierarchyNumber=adapters.Count };
                a.sourceId=(int)v["id"];
                a.enabled=(bool?)v["enabled"] ?? true;
                adapters.Add(a);
                var body=Body(v["body"]);body.relativePos=V(v["relative_pos"]);body.relativeOrientation=F(v["relative_orientation"]);
                if(v["composite"].Type!=JTokenType.Null)body.compositeParent=physics.composites[(int)v["composite"]];
                var raw=v["collider"]["shape"];int layer=(int)v["layer"];
                if(a.shape==Shape.Circle){var shape=Circle(raw);shape.layer=layer;physics.circles.AddBody(a.parent,shape,body);}
                else if(a.shape==Shape.Box){var shape=Box(raw);shape.layer=layer;physics.boxes.AddBody(a.parent,shape,body);}
                else {var shape=Rounded(raw);shape.layer=layer;physics.roundedRects.AddBody(a.parent,shape,body);}
            }
            if ((string)query["operation"] == "dynamics_player_controller")
            {
                foreach(var c in physics.composites)foreach(var ids in new[]{c.circleIds,c.boxIds,c.roundedRectIds})
                    for(int i=0;i<ids.Count;i++)ids[i]=adapters.Find(a=>a.sourceId==ids[i]).parent.instanceId;
                return ReferencePlayerController.Evaluate(query, physics, adapters, () => Snapshot(scene, objects, adapters));
            }
            if ((string)query["operation"] == "dynamics_player_rays")
                return ReferencePlayerRays.Evaluate(query, physics, adapters, () => Snapshot(scene, objects, adapters));
            if ((string)query["operation"] == "dynamics_surface_motion")
                return ReferenceSurfaceMotion.Evaluate(query, physics, adapters, () => Snapshot(scene, objects, adapters));
            var result=new JArray();
            int tick=0;
            foreach(var dt in query["steps"])
            {
                var commands=query["commands"]?[tick++];
                if(commands!=null)foreach(var command in commands)Command(command,adapters);
                events.Clear();sweepX=false;roundedSweepX=false;
                physics.Simulate(F(dt));
                var snapshot=Snapshot(scene,objects,adapters);
                var groups=new JArray();
                for(int g=0;g<pairFields.Length;g++)
                {
                    var pairs=new JArray();
                    if(physics.circles.Length+physics.boxes.Length+physics.roundedRects.Length>0)foreach(IdPair pair in (List<IdPair>)typeof(DetPhysics).GetField(pairFields[g],flags).GetValue(physics))
                        pairs.Add(new JArray(Parent(pairShapes[g,0],pair.id1).instanceId,Parent(pairShapes[g,1],pair.id2).instanceId));
                    groups.Add(pairs);
                }
                result.Add(new JObject {["scene"]=snapshot,["report"]=new JObject {["sweep_x"]=sweepX,["rounded_sweep_x"]=roundedSweepX,["pairs"]=groups,["callbacks"]=events.DeepClone()}});
            }
            return result;
        }
        finally {Capturing=false;physics.ropes.ropeBodies=originalRopes;foreach(var a in adapters)UnityEngine.Object.DestroyImmediate(a.transform.gameObject);}
    }
    internal sealed class Adapter : IPhysicsCollider, ICollisionCallback
    {
        public int sourceId;public FixTransform transform;public PhysicsParent parent;public bool callbackDestroyed;
        bool ICollisionCallback.IsDestroyed {get=>callbackDestroyed;set=>callbackDestroyed=value;}
        public bool IsDestroyed {get;set;}
        public bool enabled {get;set;}=true;
        public bool initHasBeenCalled {get;set;}=true;
        public int HierarchyNumber {get;set;}
        public Shape shape {get;set;}
        public Vec2 position {get=>transform.position;set=>transform.position=value;}
        public Fix rotation {get=>transform.rotation;set=>transform.rotation=value;}
        public Vec2 up {get=>transform.up;set=>transform.up=value;}
        public Vec2 right {get=>transform.right;set=>transform.right=value;}
        public Vec2 velocity {get;set;}
        public Fix angularVelocity {get;set;}
        public Fix inverseMass {get;set;}
        public Fix inverseMomentOfInertia {get;set;}
        public Fix Scale {get;set;}
        public PhysicsParent GetPhysicsParent()=>parent;
        public bool IsEnabled()=>true;
        public void OnCollide(CollisionInformation c) => events.Add(new JObject {
            ["receiver"]=c.pp.instanceId,["other"]=c.colliderPP.instanceId,["layer"]=c.layer,["collider"]=ShapeJson(c.collider),["impact_velocity"]=V(c.colliderImpactVelocity),
            ["contact"]=new JObject {["point"]=V(c.contactPoint),["normal"]=V(c.normal),["penetration"]=c.penetration.m_rawValue} });
        public void RegisterCollisionCallback(ICollisionCallback c)=>throw new NotSupportedException();
        public void UnRegisterCollisionCallback(ICollisionCallback c)=>throw new NotSupportedException();
        public void AddForceAtPosition(Vec2 f,Vec2 p,ForceMode2D mode=ForceMode2D.Force)=>throw new NotSupportedException();
        public PhysicsBody GetBody()=>throw new NotSupportedException();
        public void AddForce(Vec2 f)=>throw new NotSupportedException();
        public void SetRelativePositionOfChildBody(Vec2 p,Vec2 r)=>throw new NotSupportedException();
        public void SetRelativeOrientationOfChildBody(Fix r)=>throw new NotSupportedException();
        public void UpdatePhysicsPositions()=>throw new NotSupportedException();
        public bool IsComposite()=>throw new NotSupportedException();
    }
}
