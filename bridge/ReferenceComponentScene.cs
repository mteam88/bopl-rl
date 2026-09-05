using System;
using System.Collections.Generic;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Real DPhysics components backed by explicit raw scene state, for stage tests.
internal sealed class ReferenceComponentScene : IDisposable
{
    internal readonly Dictionary<int,GameObject> objects = new Dictionary<int,GameObject>();
    internal readonly Dictionary<int,int> identities = new Dictionary<int,int>();
    internal readonly List<ReferenceBeamTargets> targets = new List<ReferenceBeamTargets>();
    private readonly DetPhysics physics;
    private readonly PhysicsBodyList<Circle> oldCircles;
    private readonly PhysicsBodyList<Box> oldBoxes;
    private readonly PhysicsBodyList<RoundedRect> oldRects;
    private readonly RopeBody[] oldRopes;
    private readonly List<CompositeBody> oldComposites;
    internal ReferenceComponentScene(DetPhysics physics, JObject scene, JArray targetInput)
    {
        this.physics = physics;
        oldCircles = physics.circles; oldBoxes = physics.boxes; oldRects = physics.roundedRects;
        oldRopes = physics.ropes.ropeBodies; oldComposites = new List<CompositeBody>(physics.composites);
        physics.circles = new PhysicsBodyList<Circle>(); physics.boxes = new PhysicsBodyList<Box>(); physics.roundedRects = new PhysicsBodyList<RoundedRect>();
        physics.composites.Clear(); physics.gravityAccel = F(scene["gravity"]);
        foreach(var v in scene["composites"])
        {
            var c = CompositeBody.CreateNewComposite(physics.composites);
            c.combinedBody = Body(v["body"]); c.position = V(v["position"]);
        }
        for(int i=0;i<physics.composites.Count;i++) physics.composites[i].IsDestroyed = (bool)scene["composites"][i]["destroyed"];
        for(int i=0;i<32;i++) for(int j=i;j<32;j++) Physics2D.IgnoreLayerCollision(i,j,((uint)scene["layers"][i] & (1u<<j))==0);
        foreach(var v in scene["objects"])
        {
            int id = (int)v["id"]; var go = new GameObject("Beam physics obstacle"); objects.Add(id,go); identities.Add(go.GetInstanceID(),id);
            var ft = go.AddComponent<FixTransform>(); var t = v["transform"];
            ft.position=V(t["position"]); ft.rotation=F(t["rotation"]); ft.offset=V(t["offset"]);
            if(t["right"]!=null) { Set(ft,"rightInner",V(t["right"])); Set(ft,"upInner",Vec2.MulByI(V(t["right"]))); }
            go.layer=(int)v["layer"]; var raw=v["collider"]["shape"]; var body=Body(v["body"]);
            body.relativePos=V(v["relative_pos"]); body.relativeOrientation=F(v["relative_orientation"]);
            if(v["composite"].Type!=JTokenType.Null) body.compositeParent=physics.composites[(int)v["composite"]];
            IPhysicsCollider collider;
            switch((string)v["collider"]["kind"])
            {
                case "circle":
                    var circle=go.AddComponent<DPhysicsCircle>(); circle.ManualInit(); collider=circle;
                    int ci=physics.circles.ColliderIndex(go.GetInstanceID()); var cs=Circle(raw); cs.layer=go.layer;
                    physics.circles.colliders[ci]=cs; physics.circles.physicsBodies[ci]=body; break;
                case "box":
                    var box=go.AddComponent<DPhysicsBox>(); box.ManualInit(); collider=box;
                    int bi=physics.boxes.ColliderIndex(go.GetInstanceID()); var bs=Box(raw); bs.layer=go.layer;
                    physics.boxes.colliders[bi]=bs; physics.boxes.physicsBodies[bi]=body; break;
                default:
                    var rect=go.AddComponent<DPhysicsRoundedRect>(); rect.ManualInit(); collider=rect;
                    int ri=physics.roundedRects.ColliderIndex(go.GetInstanceID()); var rs=Rounded(raw); rs.layer=go.layer;
                    physics.roundedRects.colliders[ri]=rs; physics.roundedRects.physicsBodies[ri]=body; break;
            }
            collider.IsDestroyed=(bool)v["destroyed"]; go.SetActive((bool)v["active"]);
        }
        for(int i=0;i<physics.composites.Count;i++)
        {
            var c=physics.composites[i]; var v=scene["composites"][i];
            foreach(var id in v["circles"]) c.circleIds.Add(objects[(int)id].GetInstanceID());
            foreach(var id in v["boxes"]) c.boxIds.Add(objects[(int)id].GetInstanceID());
            foreach(var id in v["rounded_rects"]) c.roundedRectIds.Add(objects[(int)id].GetInstanceID());
        }
        foreach(JObject target in targetInput) targets.Add(new ReferenceBeamTargets((JObject)target["state"],objects[(int)target["id"]]));
        var ropes=scene["rope_system"]; var bodies=new List<RopeBody>();
        foreach(var r in ropes["bodies"]) { var rope=ReferenceRope.Read(r); rope.indexInArray=bodies.Count; bodies.Add(rope); }
        physics.ropes.ropeBodies=bodies.ToArray(); physics.ropeGravity=F(ropes["gravity"]);
        physics.ropeRelaxationIterations=(int)ropes["iterations"]; physics.platformStretchiness=F(ropes["stretchiness"]);
        typeof(DetPhysics).GetField("levelTypeThisFrame",System.Reflection.BindingFlags.Static|System.Reflection.BindingFlags.NonPublic)
            .SetValue(null,(bool)ropes["space"]?LevelType.space:LevelType.grass);
        Set(physics,"collisionCount",0);
    }
    public void Dispose()
    {
        foreach(var go in objects.Values) UnityEngine.Object.DestroyImmediate(go);
        physics.circles=oldCircles; physics.boxes=oldBoxes; physics.roundedRects=oldRects;
        physics.ropes.ropeBodies=oldRopes; physics.composites.Clear(); physics.composites.AddRange(oldComposites);
    }
}
