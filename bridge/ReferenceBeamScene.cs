using System;
using System.Collections.Generic;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Full physics stage with real pooled beams and targets. Stops before gameplay
// collision callbacks, whose queued inputs are compared without exclusions.
public static class ReferenceBeamScene
{
    internal static bool Active;
    private static JObject graph;
    private static Dictionary<int,int> identities;
    internal static void BeforeRender(DetPhysics p,int reflected)
    {
        if(Active) graph=ReferenceBeamGraph.Snapshot(p,reflected,ft=>identities[ft.gameObject.GetInstanceID()]);
    }
    public static bool BeforeCallbacks() => Active;
    private static void ResetGraph(DetPhysics p)
    {
        p.beamBodies=new DetPhysics.BeamBody[128]; p.beamRoles=new DetPhysics.BeamRole[128];
        Set(p,"beamParentInd",new short[128]); Set(p,"beamIsReflected",new bool[128]); Set(p,"beamWasReflected",new bool[128]);
        Set(p,"beamReflectionDistance",new Fix[128]); Set(p,"beamRaycastHitRRs",new FixTransform[128]);
        Set(p,"lineIdtoInd",new HashMap_short(128,128)); Set(p,"crossingsLastFrame",new DetPhysics.BeamCrossingRecord[64]); Set(p,"crossingsThisFrame",new DetPhysics.BeamCrossing[64]);
        foreach(string field in new[]{"beamsThisFrame","initialBeamCount","totalBeamsLastFrame","crossingsThisFrameCount","crossingsLastFrameCount","keyCountLastFrame"}) Set(p,field,0);
    }
    private static JArray Callbacks(DetPhysics p)
    {
        var result=new JArray();
        foreach(var c in Get<List<CollisionInformation>>(p,"ToMakeCallBacks"))
            result.Add(new JObject { ["receiver"]=identities[c.pp.instanceId], ["other"]=identities[c.colliderPP.instanceId],
                ["collider"]=ReferencePhysicsScene.ShapeJson(c.collider), ["layer"]=c.layer, ["impact_velocity"]=V(c.colliderImpactVelocity),
                ["contact"]=new JObject { ["penetration"]=c.penetration.m_rawValue, ["point"]=V(c.contactPoint), ["normal"]=V(c.normal) } });
        return result;
    }
    public static JToken Evaluate(JObject query)
    {
        var p=DetPhysics.Get(); var config=query["config"]; var gc=config["graph"]; var ec=config["effects"]; var hc=config["hitbox"];
        var oldObjects=Get<BeamObject[]>(p,"beamObjectPool"); var oldEffects=Get<BeamHitAnim[]>(p,"beamHitEffectPool");
        var pool=new BeamObject[512]; var effects=new BeamHitAnim[512];
        var oldPlayers=PlayerHandler.Get().PlayerList(); var oldTime=GameTime.PlayerTimeScale;
        var oldWater=Constants.WATER_HEIGHT; var oldLevel=Constants.leveltype;
        var prefab=p.beamObjectPrefab; var oldWidth=prefab.baseWidth01; var oldExtra=prefab.extraLength; var oldMinimum=prefab.GetComponent<DPhysicsBox>().minimumExtents; var oldLayer=prefab.gameObject.layer;
        ReferenceComponentScene components=null;
        try
        {
            components=new ReferenceComponentScene(p,(JObject)query["scene"],(JArray)query["targets"]);
            identities=components.identities; int nextId=1; foreach(int id in components.objects.Keys) nextId=Math.Max(nextId,id+1);
            ReferencePhysicsCapture.IdentityOverride=id=>identities[id];
            ResetGraph(p); Set(p,"beamObjectPool",pool); Set(p,"beamHitEffectPool",effects);
            p.maxBeamDistance=F(gc["maximum"]); p.beamHitMask=(int)(uint)gc["mask"]; p.crossBeamOffset=F(gc["cross_offset"]);
            p.beamWidth=F(config["width"]); p.beamParentOffset=F(config["parent_offset"]); p.beamWidthAnim=ReferenceCurve.Read(config["width_curve"]);
            p.beamPlatformPushForce=F(ec["platform_push"]); p.beamBlackHolePushForce=F(ec["black_hole_push"]); p.beamRigidBodyPushFOrce=F(ec["body_push"]);
            Constants.WATER_HEIGHT=F(gc["water"]); Constants.leveltype=(bool)gc["space"]?LevelType.space:LevelType.grass;
            GameTime.PlayerTimeScale=F(ec["time_scale"]);
            var players=new List<Player>(); for(int i=0;i<6;i++) players.Add(new Player(i+1,i){isProtectedFromTimeStop=(bool)ec["protected"][i]}); PlayerHandler.Get().SetPlayerList(players);
            prefab.baseWidth01=F(hc["base_width"]); prefab.extraLength=F(hc["extra_length"]); prefab.GetComponent<DPhysicsBox>().minimumExtents=F(hc["minimum_extents"]); prefab.gameObject.layer=(int)config["layer"];
            var result=new JArray(); Active=true;
            foreach(var frame in query["frames"])
            {
                if(frame["destroy_slots"] is JArray destroyed) foreach(var slot in destroyed)
                    if(pool[(int)slot]!=null) Updater.DestroyFix(pool[(int)slot].gameObject);
                foreach(var b in frame["beams"]) p.AddBeamBody(new DetPhysics.BeamBody { id=(int)b["id"], ownerId=(int)b["owner_id"], position=V(b["position"]), direction=V(b["direction"]),
                    distance=F(b["distance"]), timePassed=F(b["time_passed"]), scale=F(b["scale"]), colors=ReferenceBeam.ReadColors(b["colors"]) });
                graph=null; p.Simulate(F(frame["dt"]));
                var slots=new JArray();
                foreach(var beam in pool)
                {
                    if(beam==null) break;
                    int raw=beam.gameObject.GetInstanceID(); if(!identities.ContainsKey(raw)) identities.Add(raw,nextId++);
                    var state=new JObject { ["id"]=identities[raw], ["hitbox"]=ReferenceBeamHitbox.Snapshot(beam) };
                    if(beam.IsDestroyed) state["destroyed"]=true;
                    slots.Add(state);
                }
                var targets=new JArray(); for(int i=0;i<components.targets.Count;i++) targets.Add(new JObject { ["id"]=query["targets"][i]["id"], ["state"]=components.targets[i].Snapshot() });
                result.Add(new JObject { ["scene"]=ReferencePhysicsCapture.Scene(p), ["graph"]=graph??(JToken)JValue.CreateNull(), ["pool"]=slots, ["targets"]=targets, ["callbacks"]=Callbacks(p) });
            }
            return result;
        }
        finally
        {
            Active=false; ReferencePhysicsCapture.IdentityOverride=null;
            foreach(var beam in pool) if(beam!=null) UnityEngine.Object.DestroyImmediate(beam.gameObject);
            foreach(var effect in effects) if(effect!=null) UnityEngine.Object.DestroyImmediate(effect.gameObject);
            components?.Dispose(); Set(p,"beamObjectPool",oldObjects); Set(p,"beamHitEffectPool",oldEffects);
            prefab.baseWidth01=oldWidth; prefab.extraLength=oldExtra; prefab.GetComponent<DPhysicsBox>().minimumExtents=oldMinimum; prefab.gameObject.layer=oldLayer;
            PlayerHandler.Get().SetPlayerList(oldPlayers); GameTime.PlayerTimeScale=oldTime; Constants.WATER_HEIGHT=oldWater; Constants.leveltype=oldLevel;
        }
    }
}
