using System;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using static ReferenceDynamics;
using UnityEngine;

public static class ReferenceBeam
{
    internal static bool GraphOnly;
    internal static JObject GraphResult;
    internal static bool BeforeRender(DetPhysics p,int reflected)
    {
        if(!GraphOnly)return false;
        GraphResult=ReferenceBeamGraph.Snapshot(p,reflected);return true;
    }
    internal static Color ReadColor(JToken v)=>new Color(BitConverter.ToSingle(BitConverter.GetBytes((uint)v[0]),0),BitConverter.ToSingle(BitConverter.GetBytes((uint)v[1]),0),BitConverter.ToSingle(BitConverter.GetBytes((uint)v[2]),0),BitConverter.ToSingle(BitConverter.GetBytes((uint)v[3]),0));
    internal static JArray ColorJson(Color c)=>new JArray(BitConverter.ToUInt32(BitConverter.GetBytes(c.r),0),BitConverter.ToUInt32(BitConverter.GetBytes(c.g),0),BitConverter.ToUInt32(BitConverter.GetBytes(c.b),0),BitConverter.ToUInt32(BitConverter.GetBytes(c.a),0));
    internal static DetPhysics.BeamColors ReadColors(JToken v)=>new DetPhysics.BeamColors {white=ReadColor(v["white"]),yellow=ReadColor(v["yellow"]),orange=ReadColor(v["orange"]),red=ReadColor(v["red"])};
    internal static JObject ColorsJson(DetPhysics.BeamColors c)=>new JObject {["white"]=ColorJson(c.white),["yellow"]=ColorJson(c.yellow),["orange"]=ColorJson(c.orange),["red"]=ColorJson(c.red)};
    private static readonly BindingFlags flags=BindingFlags.Instance|BindingFlags.NonPublic;
    private static DetPhysics.BeamBody Ray(JToken v)=>new DetPhysics.BeamBody {position=V(v["position"]),direction=V(v["direction"]),distance=F(v["distance"])};
    public static JToken Evaluate(JObject q)
    {
        if((string)q["operation"]=="dynamics_beam_colors")return ColorsJson(ReadColors(q["a"]).Mix(ReadColors(q["b"])));
        if((string)q["operation"]=="dynamics_beam_graph")return ReferenceBeamGraph.Evaluate(q);
        var a=Ray(q["a"]);var b=Ray(q["b"]);Vec2 distance;bool hit;
        if((string)q["operation"]=="dynamics_beam_lines")hit=PhysTools.LineToLineIntersection(a.position,a.direction,a.distance,b.position,b.direction,b.distance,out distance);
        else
        {
            var p=DetPhysics.Get();p.beamBodies[0]=a;p.beamBodies[1]=b;
            var reflected=(bool[])typeof(DetPhysics).GetField("beamIsReflected",flags).GetValue(p);
            var was=(bool[])typeof(DetPhysics).GetField("beamWasReflected",flags).GetValue(p);
            var lengths=(Fix[])typeof(DetPhysics).GetField("beamReflectionDistance",flags).GetValue(p);
            for(int i=0;i<2;i++){var v=q[i==0?"a":"b"];reflected[i]=(bool)v["reflected"];was[i]=(bool)v["was_reflected"];lengths[i]=F(v["reflection_distance"]);}
            object[] args={0,1,Vec2.zero};
            try{hit=(bool)typeof(DetPhysics).GetMethod((bool)q["minimize_b"]?"IntersectBeams_minimizedAlongB":"IntersectBeams",flags).Invoke(p,args);}
            catch(TargetInvocationException e){throw e.InnerException;}
            distance=(Vec2)args[2];
        }
        return new JObject {["hit"]=hit,["distances"]=V(distance)};
    }
}
