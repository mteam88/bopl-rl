using System;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using static ReferenceDynamics;

public static class ReferenceRope
{
    private static JArray Vectors(Vec2[] values) {var a=new JArray();foreach(var v in values)a.Add(V(v));return a;}
    private static JObject State(RopeBody r) => new JObject {
        ["segment"]=Vectors(r.segment),["previous"]=Vectors(r.segmentPrev),["count"]=r.segmentCount,["enabled"]=r.enabled,
        ["force_start"]=V(r.force_start),["force_end"]=V(r.force_end),["mass_start"]=r.mass_start.m_rawValue,["mass_end"]=r.mass_end.m_rawValue,
        ["separation"]=r.segmentSeparation.m_rawValue,["time_since_hook_arrived"]=r.timeSinceHookArrived.m_rawValue,["detached"]=r.hasBeenDettached,
        ["colliding_rects"]=new JArray(r.collidingRoundedRects),["collision_count"]=r.nrOfCollidingRects
    };
    public static JToken Evaluate(JObject q)
    {
        var v=q["rope"];var r=new RopeBody {
            segmentCount=(int)v["count"],enabled=(bool)v["enabled"],force_start=V(v["force_start"]),force_end=V(v["force_end"]),
            mass_start=F(v["mass_start"]),mass_end=F(v["mass_end"]),segmentSeparation=F(v["separation"]),timeSinceHookArrived=F(v["time_since_hook_arrived"]),
            hasBeenDettached=(bool)v["detached"],nrOfCollidingRects=(int)v["collision_count"]
        };
        for(int i=0;i<r.segment.Length;i++){r.segment[i]=V(v["segment"][i]);r.segmentPrev[i]=V(v["previous"][i]);}
        for(int i=0;i<r.collidingRoundedRects.Length;i++)r.collidingRoundedRects[i]=(int)v["colliding_rects"][i];
        var p=DetPhysics.Get();var oldRects=p.roundedRects;var oldRopes=p.ropes.ropeBodies;
        p.roundedRects=new PhysicsBodyList<RoundedRect>();int id=0;
        foreach(var rect in q["rectangles"])p.roundedRects.AddCollider(new PhysicsParent {instanceId=id++},Rounded(rect));
        p.ropes.ropeBodies=new[]{r};
        var config=q["config"];p.ropeGravity=F(config["gravity"]);p.ropeRelaxationIterations=(int)config["iterations"];p.platformStretchiness=F(config["stretchiness"]);
        typeof(DetPhysics).GetField("levelTypeThisFrame",BindingFlags.Static|BindingFlags.NonPublic).SetValue(null,(bool)config["space"]?LevelType.space:LevelType.grass);
        try
        {
            var output=new JArray();
            foreach(var a in q["actions"])
            {
                JToken result=JValue.CreateNull();
                switch((string)a["op"])
                {
                    case "step":
                        try {p.SimulateRopes_parallel(F(config["dt"]));}
                        catch(AggregateException e){throw e.Flatten().InnerExceptions[0];}
                        break;
                    case "bounds":var b=PhysTools.ComputeBoundingBox(r,0);result=new JArray(V(b.min),V(b.max));break;
                    case "spawn":result=r.SpawnAdditionalSegment(V(a["point"]));break;
                    case "reel":result=r.ReelInSegment((int)a["minimum"]);break;
                    case "reverse_reel":result=r.ReverseReelInSegment((int)a["minimum"]);break;
                    case "mass":r.SetMass01(F(a["value"]),(bool)a["top"],F(a["scale"]));break;
                    case "detach":r.Dettach((bool)a["top"]);break;
                    default:throw new ArgumentException("Unknown rope action");
                }
                output.Add(new JObject {["rope"]=State(r),["result"]=result});
            }
            return output;
        }
        finally {p.roundedRects=oldRects;p.ropes.ropeBodies=oldRopes;}
    }
}
