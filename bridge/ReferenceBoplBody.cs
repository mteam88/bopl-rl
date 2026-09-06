using System;
using System.Collections.Generic;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Binds real components to supplied registered physics state, then calls original methods.
// bind_initialized changes only the supplied component flag; it is not an Init substitute.
internal static class ReferenceBoplBody
{
    private static readonly string[] ConfigNames={"mass","momentOfInertia","bounciness","friction","dynamicFriction","gravityScale","airFriction01","rotationalDrag01"};
    private static readonly string[] JsonNames={"mass","inertia","bounciness","friction","dynamic_friction","gravity","air_drag","rotational_drag"};
    internal static void Bind(BoplBody body,JToken state)
    {
        for(int i=0;i<ConfigNames.Length;i++)Set(body,ConfigNames[i],F(state["config"][JsonNames[i]]));
        body.StartVelocity=V(state["config"]["velocity"]);body.StartAngularVelocity=F(state["config"]["angular_velocity"]);
        Set(body,"isResizable",(bool)state["resizable"]);Set(body,"prevScale",F(state["prev_scale"]));Set(body,"framesUntilDrop",(long)state["frames_until_drop"]);
        Set(body,"originalSortingLayerName",(string)state["original_sorting"]["layer"]);Set(body,"originalSortingOrder",(int)state["original_sorting"]["order"]);
    }
    internal static JObject State(BoplBody body,Dictionary<int,MagnetGun> magnets)
    {
        var config=new JObject();for(int i=0;i<ConfigNames.Length;i++)config[JsonNames[i]]=Get<Fix>(body,ConfigNames[i]).m_rawValue;
        config["velocity"]=V(body.StartVelocity);config["angular_velocity"]=body.StartAngularVelocity.m_rawValue;
        JToken magnet=JValue.CreateNull();foreach(var pair in magnets)if(pair.Value==body.magnetGunHolder)magnet=pair.Key;
        var sprite=body.GetComponent<SpriteRenderer>();
        return new JObject{["config"]=config,["resizable"]=Get<bool>(body,"isResizable"),["prev_scale"]=Get<Fix>(body,"prevScale").m_rawValue,
            ["frames_until_drop"]=Get<long>(body,"framesUntilDrop"),["magnet"]=magnet,
            ["sorting"]=sprite==null?JValue.CreateNull():(JToken)new JObject{["layer"]=sprite.sortingLayerName,["order"]=sprite.sortingOrder},
            ["original_sorting"]=new JObject{["layer"]=Get<string>(body,"originalSortingLayerName"),["order"]=Get<int>(body,"originalSortingOrder")}};
    }
    internal static JToken Evaluate(JObject query,DetPhysics physics,List<ReferencePhysicsScene.Adapter> adapters,Func<JObject> snapshot)
    {
        var a=adapters.Find(v=>v.sourceId==(int)query["object"]);var go=a.transform.gameObject;go.SetActive(false);
        IPhysicsCollider collider=a.shape==Shape.Circle?(IPhysicsCollider)go.AddComponent<DPhysicsCircle>():a.shape==Shape.Box?go.AddComponent<DPhysicsBox>():go.AddComponent<DPhysicsRoundedRect>();
        var body=go.AddComponent<BoplBody>();var initial=query["state"];
        SpriteRenderer sprite=null;if(initial["sorting"].Type!=JTokenType.Null){sprite=go.AddComponent<SpriteRenderer>();sprite.sortingLayerName=(string)initial["sorting"]["layer"];sprite.sortingOrder=(int)initial["sorting"]["order"];}
        go.SetActive(true);Bind(body,initial);Set(a.transform,"scale",F(query["transform_scale"]));
        collider.initHasBeenCalled=(bool)query["initialized"];collider.enabled=a.enabled;
        var parent=a.parent;parent.monobehaviourCollider=collider;a.parent=parent;
        if(collider is DPhysicsCircle circle){circle.pp=parent;physics.circles.physicsParents[physics.circles.ColliderIndex(parent.instanceId)]=parent;}
        else if(collider is DPhysicsBox box){box.pp=parent;physics.boxes.physicsParents[physics.boxes.ColliderIndex(parent.instanceId)]=parent;}
        else{((DPhysicsRoundedRect)collider).pp=parent;physics.roundedRects.physicsParents[physics.roundedRects.ColliderIndex(parent.instanceId)]=parent;}
        go.SetActive((bool)query["scene"]["objects"][adapters.IndexOf(a)]["active"]);
        var magnets=new Dictionary<int,MagnetGun>();
        MagnetGun Magnet(JToken value){if(value.Type==JTokenType.Null)return null;int id=(int)value;if(!magnets.ContainsKey(id)){var mg=new GameObject("inert magnet identity");mg.SetActive(false);magnets[id]=mg.AddComponent<MagnetGun>();}return magnets[id];}
        body.magnetGunHolder=Magnet(initial["magnet"]);
        var frames=new JArray();
        JObject Frame(JToken result)=>new JObject{["state"]=State(body,magnets),["scene"]=snapshot(),["initialized"]=collider.initHasBeenCalled,["transform_scale"]=a.transform.Scale.m_rawValue,["result"]=result};
        try
        {
            foreach(var action in query["actions"])
            {
                JToken result=JValue.CreateNull();string error=null;
                try
                {
                    switch((string)action["op"])
                    {
                        case "init":body.Init();break;
                        case "update":body.UpdateSim(F(action["dt"]));break;
                        case "hold":body.Hold((long)action["frames"],Magnet(action["magnet"]),(bool)action["change_sorting"]);break;
                        case "drop":body.Drop();break;
                        case "held":result=body.IsBeingHeld();break;
                        case "new_body":result=Body(body.NewPhysicsBody(F(action["orientation"])));break;
                        case "inverse_mass":result=body.InverseMass.m_rawValue;break;
                        case "inverse_inertia":result=body.InverseMomentOfInertia.m_rawValue;break;
                        case "set_mass":body.InverseMass=F(action["value"]);break;
                        case "set_inertia":body.InverseMomentOfInertia=F(action["value"]);break;
                        case "set_velocity":body.velocity=V(action["value"]);break;
                        case "set_angular":body.angularVelocity=F(action["value"]);break;
                        case "velocity":result=V(body.velocity);break;
                        case "angular":result=body.angularVelocity.m_rawValue;break;
                        case "body":result=Body(body.PhysicsBody());break;
                        case "force":body.AddForce(V(action["force"]),(bool)action["impulse"]?ForceMode2D.Impulse:ForceMode2D.Force);break;
                        case "force_at":body.AddForceAtPosition(V(action["force"]),V(action["position"]),(bool)action["impulse"]?ForceMode2D.Impulse:ForceMode2D.Force);break;
                        case "transform_scale":a.transform.SetScale_DPhysicsOnly(F(action["value"]));break;
                        case "bind_initialized":collider.initHasBeenCalled=(bool)action["value"];break;
                        default:throw new ArgumentException("Unknown BoplBody action");
                    }
                }
                catch(DivideByZeroException){error="division_by_zero";}
                if(error!=null)return new JObject{["error"]=error,["action"]=frames.Count,["completed"]=frames,["failed"]=Frame(JValue.CreateNull())};
                frames.Add(Frame(result));
            }
            return frames;
        }
        finally{foreach(var magnet in magnets.Values)if(magnet!=null)UnityEngine.Object.DestroyImmediate(magnet.gameObject);}
    }
}
