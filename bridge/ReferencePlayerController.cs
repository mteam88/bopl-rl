using System;
using System.Collections.Generic;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

internal static class ReferencePlayerController
{
    internal static Action<string> SoundCapture;
    internal static JToken Evaluate(JObject query, DetPhysics scene, List<ReferencePhysicsScene.Adapter> adapters, Func<JObject> snapshot)
    {
        var objects = new List<GameObject>();
        var oldPlayers = PlayerHandler.Get().PlayerList(); var oldScale = GameTime.PlayerTimeScale;
        var oldSpace = Constants.leveltype; var oldDeadzone = Constants.GROUNDALIGNMENTDEADZONE;
        var oldSound = SoundCapture;
        try
        {
            var bound = ReferenceSurfaceMotion.BindPlatforms(query["bindings"],scene,adapters,objects);
            var platforms = query["bindings"].Select(b=>bound.TryGetValue((int)b["object"],out var p)?p:null).ToList();
            for(int i=0;i<platforms.Count;i++)if(platforms[i]!=null)
            {
                var g=query["grounds"][i];var p=platforms[i];p.platformType=(bool)g["ice"]?PlatformType.ice:PlatformType.grass;
                Set(p,"dropPlayersForFrames",(int)g["drop_frames"]);Set(p,"dontDropPlayerId",(int)g["dont_drop"]);Set(p,"dropForce",F(g["drop_force"]));
            }
            PlayerHandler.Get().SetPlayerList(query["players"].Select(p=>new Player((int)p["id"],0){isProtectedFromTimeStop=(bool)p["protected"]}).ToList());
            GameTime.PlayerTimeScale=F(query["time_scale"]);Constants.leveltype=(bool)query["space"]?LevelType.space:LevelType.grass;
            var state=query["state"];var original=state["physics"];var motion=original["motion"];var config=query["config"];var c=config["physics"];
            var go=new GameObject("player controller oracle");go.SetActive(false);objects.Add(go);
            var ft=go.AddComponent<FixTransform>();ft.position=V(motion["position"]);
            Set(ft,"rotationInner",F(motion["rotation"]));Set(ft,"upInner",V(motion["up"]));Set(ft,"rightInner",V(motion["right"]));Set(ft,"scale",F(original["scale"]));
            var holder=go.AddComponent<Hitbox>();if(query["holder"].Type!=JTokenType.Null)holder.SetPlayerId((int)query["holder"]);
            go.AddComponent<DPhysicsBox>();var body=go.AddComponent<PlayerBody>();Set(body,"fixTransform",ft);
            body.selfImposedVelocity=V(motion["velocity"]);body.externalVelocity=V(motion["external_velocity"]);
            var physics=go.AddComponent<PlayerPhysics>();Set(physics,"body",body);Set(physics,"playerIdHolder",query["holder"].Type==JTokenType.Null?null:holder);
            Set(physics,"isGrounded",(bool)motion["grounded"]);Set(physics,"attachedGround",motion["ground"].Type==JTokenType.Null?null:platforms[(int)motion["ground"]]);
            Set(physics,"<LocalPlatformPos>k__BackingField",F(motion["local"]));physics.radius=F(motion["radius"]);physics.groundedSpeed=F(motion["grounded_speed"]);
            Set(physics,"prevScal",F(original["previous_scale"]));physics.jumpedThisFrame=(bool)original["jumped"];Set(physics,"SyncRotationToPlatform",(bool)original["sync_rotation"]);
            physics.cannotBeUngroundedByRopes=(bool)original["cannot_unground"];physics.skipNextUpdate=(bool)state["skip_update"];Set(physics,"ApplyGravity",(bool)state["apply_gravity"]);
            Set(physics,"blockableTerrainHitBuffer",(bool)state["blocked"]?new RaycastHit2D[20]:null);
            physics.gravity_accel=F(c["gravity"]);physics.gravity_maxFallSpeed=F(c["max_fall_speed"]);physics.gravity_modifier=F(c["gravity_modifier"]);
            physics.airAccel=F(c["air_accel"]);physics.Speed=F(c["max_speed"]);physics.accel=F(config["accel"]);
            Set(physics,"PlatformSlipperyness01",F(config["friction"]));Set(physics,"IcePlatformSlipperyness01",F(config["ice_friction"]));Constants.GROUNDALIGNMENTDEADZONE=F(config["alignment_deadzone"]);
            int Canonical(int id)=>adapters.Find(a=>a.parent.instanceId==id)?.sourceId ?? id;
            int Runtime(int id)=>adapters.Find(a=>a.sourceId==id)?.parent.instanceId ?? id;
            var buffer=query["hop_buffer"].Select(id=>Runtime((int)id)).ToArray();
            JArray events=null;
            void Event(string kind,JToken value)=>events.Add(new JObject{["kind"]=kind,["value"]=value});
            physics.OnNoLongerGrounded+=angle=>{Event("detached",angle.m_rawValue);body.position+=V(query["detach_offset"]);body.selfImposedVelocity+=V(query["detach_velocity"]);};
            physics.OnAttachedToGround+=hit=>{
                Event("attached",new JObject{["object"]=Canonical(hit.pp.instanceId),["near_position"]=V(hit.nearPos),["far_position"]=V(hit.farPos),["near_distance"]=hit.nearDist.m_rawValue,["far_distance"]=hit.farDist.m_rawValue,["layer"]=hit.layer});
                body.position+=V(query["attach_offset"]);
            };
            SoundCapture=name=>Event("jump_sound",name);
            if((bool)query["ability"])
            {
                var ability=go.AddComponent<ReferenceControllerAbility>();
                ability.OnExit=info=>{
                    if(info.isDead||info.killerId!=0||info.isFacingRight||info.lastSprite!=null||info.justlanded||info.causeOfDeath!=0||info.enterTurn||info.enterJump||info.ropeAttachment!=null||info.topAttachment)throw new InvalidOperationException("Unexpected drop exit defaults");
                    Event("exit",new JObject{["grounded"]=info.isGrounded,["ground"]=info.currentlyAttachedGround==null?JValue.CreateNull():new JValue(platforms.IndexOf(info.currentlyAttachedGround)),["local"]=info.groundedLocalPosition.m_rawValue,["radius"]=info.radius.m_rawValue,["grounded_speed"]=info.groundedSpeed.m_rawValue,["position"]=V(info.position),["velocity"]=V(info.selfImposedVelocity),["external_velocity"]=V(info.externalVelocity)});
                    body.position+=V(query["exit_offset"]);
                };
            }
            JObject Frame(JToken result)
            {
                var s=(JObject)state.DeepClone();var p=(JObject)original.DeepClone();p["motion"]=ReferencePlayerGround.Motion(physics,body,platforms);
                p["previous_scale"]=Get<Fix>(physics,"prevScal").m_rawValue;p["scale"]=ft.Scale.m_rawValue;s["physics"]=p;s["skip_update"]=physics.skipNextUpdate;
                var world=snapshot();world["objects"]=new JArray(world["objects"].OrderBy(o=>(int)o["id"]));
                JArray Order(PhysicsParent[] pp,int length){var ids=new JArray();for(int i=0;i<length;i++)ids.Add(Canonical(pp[i].instanceId));return ids;}
                return new JObject{["state"]=s,["scene"]=world,["orders"]=new JArray(Order(scene.roundedRects.physicsParents,scene.roundedRects.Length),Order(scene.circles.physicsParents,scene.circles.Length),Order(scene.boxes.physicsParents,scene.boxes.Length)),["hop_buffer"]=new JArray(buffer.Select(Canonical)),["events"]=events,["result"]=result};
            }
            var frames=new JArray();
            foreach(var action in query["actions"])
            {
                events=new JArray();JToken result=JValue.CreateNull();string error=null;
                try
                {
                    switch((string)action["op"])
                    {
                        case "update":physics.UpdateSim(F(action["dt"]));break;
                        case "move":physics.Move(V(action["input"]),F(action["dt"]),(bool)action["surface"]);break;
                        case "macho":physics.MachoMove(V(action["input"]),F(action["dt"]),F(action["multiplier"]));break;
                        case "drop":physics.DropPlatformTest();break;
                        case "align":physics.AlignToGround();break;
                        case "current":physics.MoveAlongCurrentPlatForm(F(action["dt"]));break;
                        case "hop":physics.MoveAlongCurrentPlatFormSwapIfBlocked(body,F(action["radius"]),F(action["speed"]),F(action["dt"]),ref buffer);break;
                        case "alignment":result=physics.InputGroundAlignment(V(action["input"])).m_rawValue;break;
                        case "increment":result=physics.incrementTowards(F(action["current"]),F(action["target"]),F(action["accel"]),F(action["dt"])).m_rawValue;break;
                        case "accelerate":result=physics.accelerateTowards(F(action["current"]),F(action["target"]),F(action["accel"]),F(action["dt"])).m_rawValue;break;
                        default:throw new ArgumentException("Unknown player controller action");
                    }
                }
                catch(NullReferenceException){error="null_reference";}
                catch(DivideByZeroException){error="division_by_zero";}
                if(error!=null)return new JObject{["error"]=error,["action"]=frames.Count,["completed"]=frames,["failed"]=Frame(JValue.CreateNull())};
                frames.Add(Frame(result));
            }
            return frames;
        }
        finally
        {
            SoundCapture=oldSound;PlayerHandler.Get().SetPlayerList(oldPlayers);GameTime.PlayerTimeScale=oldScale;Constants.leveltype=oldSpace;Constants.GROUNDALIGNMENTDEADZONE=oldDeadzone;
            foreach(var go in objects)if(go!=null)UnityEngine.Object.DestroyImmediate(go);
        }
    }
}
