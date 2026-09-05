using System;
using System.Collections.Generic;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;

// Executes UpdateBeams through the point immediately before renderBeams, which
// creates gameplay hitboxes and applies target-specific effects. Those effects
// are a distinct unverified stage, not silently included in this graph oracle.
public static class ReferenceBeamGraph
{
    private static readonly BindingFlags flags=BindingFlags.NonPublic|BindingFlags.Instance;
    private static readonly Dictionary<FixTransform,int> identities=new Dictionary<FixTransform,int>();
    private static T Get<T>(DetPhysics p,string name)=>(T)typeof(DetPhysics).GetField(name,flags).GetValue(p);
    private static void Set(DetPhysics p,string name,object value)=>typeof(DetPhysics).GetField(name,flags).SetValue(p,value);
    internal static JObject Snapshot(DetPhysics p,int reflectedCount)
    {
        int count=Get<int>(p,"beamsThisFrame"),n=count+reflectedCount;
        var bodies=new JArray();var roles=new JArray();var parents=new JArray();var reflected=new JArray();var was=new JArray();var lengths=new JArray();var hits=new JArray();
        for(int i=0;i<n;i++)
        {
            var b=p.beamBodies[i];bodies.Add(new JObject {["id"]=b.id,["owner_id"]=b.ownerId,["position"]=V(b.position),["direction"]=V(b.direction),["time_passed"]=b.timePassed.m_rawValue,
                ["scale"]=b.scale.m_rawValue,["distance"]=b.distance.m_rawValue,["colors"]=ReferenceBeam.ColorsJson(b.colors),["ground"]=JValue.CreateNull()});
            roles.Add(p.beamRoles[i].ToString());parents.Add(Get<short[]>(p,"beamParentInd")[i]);reflected.Add(Get<bool[]>(p,"beamIsReflected")[i]);was.Add(Get<bool[]>(p,"beamWasReflected")[i]);lengths.Add(Get<Fix[]>(p,"beamReflectionDistance")[i].m_rawValue);
            var ft=Get<FixTransform[]>(p,"beamRaycastHitRRs")[i];hits.Add(ft==null?JValue.CreateNull():new JValue(identities[ft]));
        }
        var crossings=new JArray();
        for(int i=0;i<Get<int>(p,"crossingsThisFrameCount");i++){var c=Get<DetPhysics.BeamCrossingRecord[]>(p,"crossingsLastFrame")[i];crossings.Add(new JObject {["a"]=c.idA,["b"]=c.idB,["age"]=c.ageTicks});}
        var mapping=new JArray();var map=Get<HashMap_short>(p,"lineIdtoInd");for(int i=0;i<map.entryCount;i++)mapping.Add(new JArray(map.keys[i],map.values[i]));
        return new JObject {["bodies"]=bodies,["roles"]=roles,["parents"]=parents,["reflected"]=reflected,["was_reflected"]=was,["reflection_distances"]=lengths,["hits"]=hits,["crossings"]=crossings,["total"]=count,["initial"]=Get<int>(p,"initialBeamCount"),["mapping"]=mapping};
    }
    public static JToken Evaluate(JObject q)
    {
        var p=DetPhysics.Get();var oldCircles=p.circles;var oldBoxes=p.boxes;var oldRects=p.roundedRects;
        var water=Constants.WATER_HEIGHT;var level=Constants.leveltype;
        p.circles=new PhysicsBodyList<Circle>();p.boxes=new PhysicsBodyList<Box>();p.roundedRects=new PhysicsBodyList<RoundedRect>();identities.Clear();
        var objects=new List<GameObject>();
        var config=q["config"];p.maxBeamDistance=F(config["maximum"]);p.beamHitMask=(int)(uint)config["mask"];p.crossBeamOffset=F(config["cross_offset"]);
        Constants.WATER_HEIGHT=F(config["water"]);Constants.leveltype=(bool)config["space"]?LevelType.space:LevelType.grass;
        p.beamBodies=new DetPhysics.BeamBody[128];p.beamRoles=new DetPhysics.BeamRole[128];
        Set(p,"beamParentInd",new short[128]);Set(p,"beamIsReflected",new bool[128]);Set(p,"beamWasReflected",new bool[128]);Set(p,"beamReflectionDistance",new Fix[128]);Set(p,"beamRaycastHitRRs",new FixTransform[128]);
        Set(p,"lineIdtoInd",new HashMap_short(128,128));Set(p,"crossingsLastFrame",new DetPhysics.BeamCrossingRecord[64]);Set(p,"crossingsThisFrame",new DetPhysics.BeamCrossing[64]);
        foreach(string field in new[]{"beamsThisFrame","initialBeamCount","totalBeamsLastFrame","crossingsThisFrameCount","crossingsLastFrameCount"})Set(p,field,0);
        ReferenceBeam.GraphOnly=true;
        try
        {
            foreach(var v in q["scene"]["objects"])
            {
                var go=new GameObject("Beam reference obstacle");objects.Add(go);var ft=go.AddComponent<FixTransform>();ft.position=V(v["transform"]["position"]);ft.rotation=F(v["transform"]["rotation"]);go.SetActive((bool)v["active"]);
                var a=new ReferencePhysicsScene.Adapter {transform=ft,IsDestroyed=(bool)v["destroyed"]};a.parent=new PhysicsParent {instanceId=(int)v["id"],fixTrans=ft,monobehaviourCollider=a};identities.Add(ft,(int)v["id"]);
                var data=v["collider"]["shape"];int layer=(int)v["layer"];var body=Body(v["body"]);
                switch((string)v["collider"]["kind"])
                {
                    case "circle":var circle=Circle(data);circle.layer=layer;p.circles.AddBody(a.parent,circle,body);break;
                    case "box":var box=Box(data);box.layer=layer;p.boxes.AddBody(a.parent,box,body);break;
                    case "rounded_rect":var rect=Rounded(data);rect.layer=layer;p.roundedRects.AddBody(a.parent,rect,body);break;
                    default:throw new ArgumentException("Unknown beam obstacle shape");
                }
            }
            var result=new JArray();
            foreach(var frame in q["frames"])
            {
                foreach(var b in frame)p.AddBeamBody(new DetPhysics.BeamBody {id=(int)b["id"],ownerId=(int)b["owner_id"],position=V(b["position"]),direction=V(b["direction"]),distance=F(b["distance"]),
                    scale=F(b["scale"]),timePassed=F(b["time_passed"]),colors=ReferenceBeam.ReadColors(b["colors"])});
                ReferenceBeam.GraphResult=null;p.UpdateBeams();
                if(ReferenceBeam.GraphResult==null)throw new InvalidOperationException("Beam graph capture missing");result.Add(ReferenceBeam.GraphResult);
            }
            return result;
        }
        finally {ReferenceBeam.GraphOnly=false;p.circles=oldCircles;p.boxes=oldBoxes;p.roundedRects=oldRects;Constants.WATER_HEIGHT=water;Constants.leveltype=level;foreach(var go in objects)UnityEngine.Object.DestroyImmediate(go);}
    }
}
