using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Runs the actual SlimeController methods with its actual ground-loss handler
// and SpriteAnimator. Registration, stock selection and cloning are upstream.
internal static class ReferenceSpawn
{
    internal static JToken Evaluate(JObject query,ReferenceAnimator.Context animation)
    {
        var objects=new List<UnityEngine.Object>();var groundObjects=new List<GameObject>();
        var oldTutorial=TutorialGameHandler.isInTutorial;var oldSound=ReferencePlayerController.SoundCapture;
        try
        {
            var config=query["config"];var state=query["spawn"];var motion=state["motion"];
            var go=animation.GameObject;
            var ft=go.AddComponent<FixTransform>();ft.position=V(motion["position"]);
            Set(ft,"rotationInner",F(motion["rotation"]));Set(ft,"upInner",V(motion["up"]));Set(ft,"rightInner",V(motion["right"]));
            var hitbox=go.GetComponent<Hitbox>();var collider=go.AddComponent<DPhysicsBox>();
            var body=go.AddComponent<PlayerBody>();Set(body,"fixTransform",ft);body.simulated=(bool)state["simulated"];
            body.selfImposedVelocity=V(motion["velocity"]);body.externalVelocity=V(motion["external_velocity"]);
            var surfaces=query["surfaces"].Select(s=>ReferencePlayerGround.Platform(s,groundObjects)).ToList();
            var physics=go.AddComponent<PlayerPhysics>();Set(physics,"body",body);
            Set(physics,"playerIdHolder",config["holder"].Type==JTokenType.Null?null:hitbox);
            Set(physics,"isGrounded",(bool)motion["grounded"]);Set(physics,"attachedGround",motion["ground"].Type==JTokenType.Null?null:surfaces[(int)motion["ground"]]);
            Set(physics,"<LocalPlatformPos>k__BackingField",F(motion["local"]));physics.radius=F(motion["radius"]);physics.groundedSpeed=F(motion["grounded_speed"]);
            physics.jumpStrength=F(config["jump_strength"]);physics.enabled=(bool)state["physics_enabled"];collider.enabled=(bool)state["hurtbox_enabled"];
            var collision=go.AddComponent<PlayerCollision>();Set(collision,"isInvulnerableMask",(uint)state["invulnerable"]);
            var slime=go.AddComponent<SlimeController>();slime.body=body;slime.playerPhysics=physics;slime.playerNumber=(int)config["player"];
            slime.suppressInput=(bool)state["suppress"];slime.isAbilitylessClone=(bool)state["abilityless"];
            Set(slime,"playerCollision",collision);Set(slime,"hurtBox",collider);Set(slime,"spriteRen",animation.Animator.spriteRen);Set(slime,"bodyAnimator",animation.Animator);
            Set(slime,"animationSpeedContainer",animation.Bodies[(int)config["animation_speed"]]);Set(slime,"currentState",(SlimeController.State)(int)state["mode"]);
            Set(slime,"hasSpawned",(bool)state["has_spawned"]);Set(slime,"fastSpawnAnim",(bool)state["fast"]);Set(slime,"nextAbiltityToChangeOnPickup",(int)state["next_ability"]);
            Set(slime,"abilityCooldownTimers",state["cooldowns"].Type==JTokenType.Null?null:state["cooldowns"].Select(F).ToArray());
            if(animation.Animator.spriteRen!=null)animation.Animator.spriteRen.enabled=(bool)state["renderer_enabled"];
            if(config["animations"].Type!=JTokenType.Null)
            {
                var data=ScriptableObject.CreateInstance<AnimationData>();objects.Add(data);slime.animData=data;
                var source=config["animations"];var frames=new Dictionary<string,SpriteFrame[]>();var maps=new Dictionary<string,VelocityFrame[]>();
                if(source["spawn"].Type!=JTokenType.Null)frames["spawn"]=animation.Frames(source["spawn"]);
                if(source["fall_loop"].Type!=JTokenType.Null)frames["fallLeftLoop"]=animation.Frames(source["fall_loop"]);
                if(source["fall_map"].Type!=JTokenType.Null)maps["fallLeft"]=animation.Map(source["fall_map"]);
                Set(data,"spriteFrames",frames);Set(data,"velocityFrames",maps);
            }
            var abilities=new List<AbilityMonoBehaviour>();var revives=new List<Revive>();
            foreach(var hasRevive in state["ability_pool"])
            {
                var abilityGo=new GameObject("spawn ability");abilityGo.SetActive(false);objects.Add(abilityGo);
                abilities.Add(abilityGo.AddComponent<Ability>());revives.Add((bool)hasRevive?abilityGo.AddComponent<Revive>():null);
            }
            slime.abilities=state["abilities"].Type==JTokenType.Null?null:state["abilities"].Select(i=>i.Type==JTokenType.Null?null:abilities[(int)i]).ToList();
            var indicators=new List<AbilityReadyIndicator>();
            foreach(var active in state["indicators"])
            {
                var indicatorGo=new GameObject("spawn indicator");indicatorGo.SetActive(false);indicatorGo.transform.SetParent(go.transform);objects.Add(indicatorGo);
                indicators.Add(indicatorGo.AddComponent<AbilityReadyIndicator>());indicatorGo.SetActive((bool)active);
            }
            slime.AbilityReadyIndicators=state["indicator_refs"].Type==JTokenType.Null?null:state["indicator_refs"].Select(i=>i.Type==JTokenType.Null?null:indicators[(int)i]).ToArray();
            var players=PlayerHandler.Get().PlayerList();
            for(int i=0;i<players.Count;i++)
            {
                var p=players[i];var source=query["players"][i];p.ExtraLife=(ReviveStatus)(int)source["extra_life"];
                p.InSparkShot=(bool)source["spark"];p.InMachoThrow=(bool)source["macho"];p.InPlatformTransform=(bool)source["platform"];
                p.ReviveInstance=source["revive"].Type==JTokenType.Null?null:revives[(int)source["revive"]];
            }
            TutorialGameHandler.isInTutorial=(bool)config["tutorial"];
            JArray events=null;
            var detached=(Action<Fix>)Delegate.CreateDelegate(typeof(Action<Fix>),slime,typeof(SlimeController).GetMethod("OnNoLongerGrounded",BindingFlags.NonPublic|BindingFlags.Instance));
            physics.OnNoLongerGrounded+=detached;
            physics.OnNoLongerGrounded+=angle=>{events.Add(new JObject{["kind"]="detached",["value"]=angle.m_rawValue});body.position+=V(config["detach_offset"]);body.selfImposedVelocity+=V(config["detach_velocity"]);};
            ReferencePlayerController.SoundCapture=name=>events.Add(new JObject{["kind"]="jump_sound",["value"]=name});
            animation.RegisterCallback(slime.Spawn,1);
            JObject Snapshot()
            {
                var s=(JObject)state.DeepClone();s["motion"]=ReferencePlayerGround.Motion(physics,body,surfaces);
                s["physics_enabled"]=physics.enabled;s["simulated"]=body.simulated;s["hurtbox_enabled"]=collider.enabled;
                if(animation.Animator.spriteRen!=null)s["renderer_enabled"]=animation.Animator.spriteRen.enabled;
                s["mode"]=(int)slime.getCurrentState();s["suppress"]=slime.suppressInput;s["has_spawned"]=slime.HasSpawned;
                s["fast"]=Get<bool>(slime,"fastSpawnAnim");s["invulnerable"]=Get<uint>(collision,"isInvulnerableMask");s["next_ability"]=slime.NextAbiltityToChangeOnPickup;
                var cooldowns=Get<Fix[]>(slime,"abilityCooldownTimers");s["cooldowns"]=cooldowns==null?JValue.CreateNull():new JArray(cooldowns.Select(f=>f.m_rawValue));
                s["indicators"]=new JArray(indicators.Select(i=>i.gameObject.activeSelf));
                var ps=new JArray();for(int i=0;i<players.Count;i++)
                {
                    var p=players[i];var source=(JObject)query["players"][i].DeepClone();source["extra_life"]=(int)p.ExtraLife;
                    source["spark"]=p.InSparkShot;source["macho"]=p.InMachoThrow;source["platform"]=p.InPlatformTransform;
                    ps.Add(source);
                }
                return new JObject{["spawn"]=s,["animator"]=animation.Snapshot(),["bodies"]=new JArray(animation.Bodies.Select(b=>new JObject{["velocity"]=V(b.velocity),["rotation"]=b.rotation.m_rawValue})),["players"]=ps,["events"]=events};
            }
            var framesOut=new JArray();
            foreach(var a in query["actions"])
            {
                events=new JArray();string error=null;
                try
                {
                    switch((string)a["op"])
                    {
                        case "init":slime.Init();break;
                        case "before":typeof(SlimeController).GetMethod("BeforeSpawnInit",BindingFlags.NonPublic|BindingFlags.Instance).Invoke(slime,null);break;
                        case "prepare":slime.PrepareToRevive(V(a["position"]));break;
                        case "respawn":slime.ReSpawn(V(a["position"]));break;
                        case "fast":slime.ReSpawnFast(V(a["position"]));break;
                        case "spawn":slime.Spawn();break;
                        case "pop":slime.PopIntoAir(V(a["direction"]));break;
                        case "animate":animation.Animator.UpdateAnimations();break;
                        case "update":GameTime.PlayerTimeScale=F(a["scale"]);animation.Animator.UpdateSim(F(a["dt"]));break;
                        default:throw new ArgumentException("Unknown spawn action");
                    }
                }
                catch(Exception e)
                {
                    if(e is TargetInvocationException)e=e.InnerException;
                    if(e is NullReferenceException)error="null_reference";
                    else if(e is IndexOutOfRangeException)error="index_out_of_range";
                    else if(e is OverflowException)error="overflow";
                    else if(e is DivideByZeroException)error="division_by_zero";
                    else if(e.Message.StartsWith("animation '"))error="animation_missing";
                    else if(e.Message.StartsWith("Velocity animation '"))error="velocity_animation_missing";
                    else throw;
                }
                if(error!=null)return new JObject{["error"]=error,["action"]=framesOut.Count,["completed"]=framesOut,["failed"]=Snapshot()};
                framesOut.Add(Snapshot());
            }
            return framesOut;
        }
        finally
        {
            TutorialGameHandler.isInTutorial=oldTutorial;ReferencePlayerController.SoundCapture=oldSound;
            foreach(var o in objects)if(o!=null)UnityEngine.Object.DestroyImmediate(o);
            foreach(var go in groundObjects)if(go!=null)UnityEngine.Object.DestroyImmediate(go);
        }
    }
}
