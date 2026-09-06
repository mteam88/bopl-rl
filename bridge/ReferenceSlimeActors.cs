using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Owns per-actor sprite identities and runs the original SlimeController with
// optional ordinary ability routing. Nested animator contexts construct/reference data.
internal static class ReferenceSlimeActors
{
    internal sealed class Context
    {
        internal SlimeController Slime;
        internal ReferenceAnimator.Context Animation;
        internal FlippingTransform Flip;
        internal Fix TurnInput;
        internal Dictionary<uint,MagnetGun> Magnets=new Dictionary<uint,MagnetGun>();
        internal List<GameObject> Objects;
        internal List<GameObject> Entities;
        internal List<RopeBody> Ropes;
        internal AbilityMonoBehaviour[] RoutingPool;
        internal int Actor;
        internal List<Context> Peers;
        internal List<StickyRoundedRectangle> Platforms;
        internal List<ReferencePhysicsScene.Adapter> Adapters;
        internal Dictionary<int,ReferenceInstantAbilities.Instance> Instants=new Dictionary<int,ReferenceInstantAbilities.Instance>();
        internal void RoutingInputs(JToken value)
        {
            var player=PlayerHandler.Get().GetPlayer(Slime.playerNumber);player.CanUseAbilities=(bool)value["can_use"];
            var inputs=Get<Player.InputProfile>(player,"Inputs");var buttons=value["buttons"];
            inputs.firstAbilityButton_IS_DOWN=buttons.Count()>0&&(bool)buttons[0];inputs.secondAbilityButton_IS_DOWN=buttons.Count()>1&&(bool)buttons[1];inputs.thirdAbilityButton_IS_DOWN=buttons.Count()>2&&(bool)buttons[2];
        }
        internal JArray RoutingChecks()
        {
            JToken Check(string method,int index)
            {
                try{return new JValue((bool)typeof(SlimeController).GetMethod(method,BindingFlags.Instance|BindingFlags.NonPublic).Invoke(Slime,new object[]{index}));}
                catch(Exception e){while(e is TargetInvocationException)e=e.InnerException;return new JObject{["error"]=e is NullReferenceException?"null_reference":e is IndexOutOfRangeException?"index_out_of_range":e is ArgumentOutOfRangeException?"argument_out_of_range":throw e};}
            }
            return new JArray(Enumerable.Range(0,Slime.abilities.Count+1).Select(i=>new JObject{["ready"]=Check("isAbilityReady",i),["castable"]=Check("isAbilityCastable",i)}));
        }
        internal MagnetGun Magnet(JToken id)
        {
            if(id.Type==JTokenType.Null)return null;uint key=(uint)id;
            if(!Magnets.TryGetValue(key,out var magnet))
            {
                var go=new GameObject("ragdoll magnet reference");go.SetActive(false);Objects.Add(go);magnet=go.AddComponent<MagnetGun>();Magnets.Add(key,magnet);
            }
            return magnet;
        }
        internal void Apply(JToken request)=>Slime.ApplyRagDoll(F(request["duration"]),V(request["speed"]),(bool)request["fast"],(bool)request["reset_animation"],Magnet(request["magnet"]));

        internal void TrackCallback()
        {
            var callback=Get<Action>(Animation.Animator,"animationSwappedCallBack");
            if(callback==null){TurnInput=Fix.Zero;return;}
            string name=callback.Method.Name;uint id;
            if(name.Contains("EnterStop")){id=1;TurnInput=Fix.Zero;}
            else if(name.Contains("EnterTurn"))
            {
                id=2;var field=callback.Target.GetType().GetFields(BindingFlags.Instance|BindingFlags.Public|BindingFlags.NonPublic).Single(f=>f.Name=="relativeInput");
                TurnInput=(Fix)field.GetValue(callback.Target);
            }
            else if(name=="setStateToIdle"){id=3;TurnInput=Fix.Zero;}
            else if(name=="Spawn"){id=4;TurnInput=Fix.Zero;}
            else if(name=="Chew"){id=5;TurnInput=Fix.Zero;}
            else if(name=="Swallow"){id=6;TurnInput=Fix.Zero;}
            else if(name=="FinishEating"){id=7;TurnInput=Fix.Zero;}
            else throw new InvalidOperationException("Unmapped slime callback "+name);
            Animation.RegisterCallback(callback,id);
        }
        internal JObject Snapshot(JToken original)
        {
            TrackCallback();var value=(JObject)original.DeepClone();var s=value["state"];
            if(value["spawn"]!=null)value["spawn"]=ReferenceActorSpawn.Snapshot(value["spawn"],this);
            if(value["eating"]!=null)
            {
                var target=Get<SlimeController>(Slime,"playerToEat");
                value["eating"]=new JObject{["target"]=ReferenceEquals(target,null)?JValue.CreateNull():new JValue(Peers.FindIndex(p=>p!=null&&ReferenceEquals(p.Slime,target))),["swallowing"]=Get<bool>(Slime,"swallowing"),["scale_increase"]=Slime.scaleIncreaseOnNom.m_rawValue};
            }
            if(value["managed_destroyed"]!=null||Slime.IsDestroyed)value["managed_destroyed"]=Slime.IsDestroyed;
            s["mode"]=(int)Slime.getCurrentState();s["facing_right"]=Flip.facingRight();s["was_upside_down"]=Get<bool>(Slime,"wasUpsideDown");
            s["input"]=V(Get<Vec2>(Slime,"inputVector"));s["previous_input"]=Get<Fix>(Slime,"prevInput").m_rawValue;s["suppress"]=Slime.suppressInput;s["has_spawned"]=Slime.HasSpawned;
            s["time_moonwalking"]=Get<Fix>(Slime,"timeSpentMoonwalking").m_rawValue;s["turn_input"]=TurnInput.m_rawValue;
            if(value["ragdoll"]!=null)
            {
                var holder=Get<MagnetGun>(Slime,"magnetHolder");
                value["ragdoll"]=new JObject{["duration"]=Get<Fix>(Slime,"ragdollDuration").m_rawValue,["time"]=Get<Fix>(Slime,"timeInRagdoll").m_rawValue,["speed"]=V(Get<Vec2>(Slime,"ragDollSpeed")),["magnet_holder"]=holder==null?JValue.CreateNull():new JValue(Magnets.First(p=>p.Value==holder).Key)};
            }
            if(value["abilities"]!=null)
            {
                var routing=value["abilities"];routing["slots"]=new JArray(Slime.abilities.Select(a=>ReferenceEquals(a,null)?(JToken)JValue.CreateNull():new JValue(Array.IndexOf(RoutingPool,a))));routing["timers"]=Get<Fix[]>(Slime,"abilityCooldownTimers")==null?JValue.CreateNull():new JArray(Get<Fix[]>(Slime,"abilityCooldownTimers").Select(f=>f.m_rawValue));routing["in_ability"]=Get<bool>(Slime,"isInAbility");
                var player=PlayerHandler.Get().GetPlayer(Slime.playerNumber);routing["can_use"]=player.CanUseAbilities;
                var inputs=Get<Player.InputProfile>(player,"Inputs");var buttons=routing["buttons"];
                for(int i=0;i<buttons.Count();i++)buttons[i]=i==0?inputs.firstAbilityButton_IS_DOWN:i==1?inputs.secondAbilityButton_IS_DOWN:i==2?inputs.thirdAbilityButton_IS_DOWN:(bool)buttons[i];
                for(int i=0;i<RoutingPool.Length;i++){routing["pool"][i]["present"]=RoutingPool[i]!=null;if(!ReferenceEquals(RoutingPool[i],null))routing["pool"][i]["destroyed"]=RoutingPool[i].IsDestroyed;}
                foreach(var pair in Instants){var metadata=routing["pool"][pair.Key];metadata["cooldown"]=pair.Value.Ability.GetCooldown().m_rawValue;metadata["handler"]=pair.Value.Snapshot(metadata["handler"]);}
                routing["indicators"]=new JArray(Slime.AbilityReadyIndicators.Select((indicator,i)=>ReferenceSlimeExit.Indicator(indicator,i<original["abilities"]["indicators"].Count()?original["abilities"]["indicators"][i]:JValue.CreateNull())));
            }
            if(value["entry"]!=null){var entry=value["entry"];entry["info"]=ReferenceInstantAbilities.Info(Get<PlayerInfo>(Slime,"playerAbilityInfoPackage"),this);entry["was_invisible"]=Get<bool>(Slime,"wasInvisible");if(entry["invisible_on_entry"]!=null)entry["invisible_on_entry"]=Get<bool>(Slime,"wasInvisibileWhenEnteredAbility");entry["rope_target"]=ReferenceInstantAbilities.Target(Get<PhysicsParent>(Slime,"ropeCastBuffer"),this);}
            value["animator"]=Animation.Snapshot();value["speed"]=new JObject{["velocity"]=V(Animation.Bodies[0].velocity),["rotation"]=Animation.Bodies[0].rotation.m_rawValue};return value;
        }
    }
    internal static JToken Evaluate(JObject query,DetPhysics scene,List<ReferencePhysicsScene.Adapter> adapters,Func<JObject> snapshot)
    {
        var contexts=new List<ReferenceAnimator.Context>();
        JToken Next(int index)
        {
            if(index==query["runtime"]["actors"].Count())return ReferenceActors.Evaluate(query,scene,adapters,snapshot,contexts);
            var actor=query["runtime"]["actors"][index];var slime=actor["slime"]??actor["ordinary"]?["quick_spell"];
            if(slime==null||slime.Type==JTokenType.Null){contexts.Add(null);return Next(index+1);}
            var q=new JObject{["state"]=slime["animator"].DeepClone(),["bodies"]=slime["speed"]==null?new JArray():new JArray(slime["speed"].DeepClone()),["holder"]=actor["body"]["holder"].DeepClone(),
                ["players"]=query["runtime"]["scheduler"]["players"].DeepClone(),["callbacks"]=new JObject()};
            return ReferenceAnimator.Evaluate(q,context=>{contexts.Add(context);return Next(index+1);});
        }
        return Next(0);
    }
    internal static Context Bind(JToken value,SlimeController slime,PlayerBody body,PlayerPhysics physics,IPhysicsCollider hurtbox,ReferenceAnimator.Context animation,List<GameObject> objects,List<UnityEngine.Object> assets,int actor,List<StickyRoundedRectangle> platforms,List<ReferencePhysicsScene.Adapter> adapters,List<GameObject> entities,List<RopeBody> ropes)
    {
        var s=value["state"];slime.abilities=new List<AbilityMonoBehaviour>();slime.AbilityReadyIndicators=Array.Empty<AbilityReadyIndicator>();
        if(value["managed_destroyed"]!=null)slime.IsDestroyed=(bool)value["managed_destroyed"];
        slime.suppressInput=(bool)s["suppress"];Set(slime,"currentState",(SlimeController.State)(int)s["mode"]);Set(slime,"hasSpawned",(bool)s["has_spawned"]);
        Set(slime,"wasUpsideDown",(bool)s["was_upside_down"]);Set(slime,"inputVector",V(s["input"]));Set(slime,"prevInput",F(s["previous_input"]));Set(slime,"timeSpentMoonwalking",F(s["time_moonwalking"]));
        Set(slime,"bodyAnimator",animation.Animator);Set(slime,"spriteRen",animation.Animator.spriteRen);Set(slime,"animationSpeedContainer",animation.Bodies[0]);Set(slime,"hurtBox",hurtbox);
        var go=new GameObject("slime facing");go.SetActive(false);objects.Add(go);var flip=go.AddComponent<FlippingTransform>();flip.alignTransform((bool)s["facing_right"]);Set(slime,"flipTrans",flip);
        var data=ScriptableObject.CreateInstance<AnimationData>();assets.Add(data);slime.animData=data;
        Set(data,"spriteFrames",((JObject)value["animations"]).Properties().ToDictionary(p=>p.Name,p=>animation.Frames(p.Value)));
        Set(data,"velocityFrames",new Dictionary<string,VelocityFrame[]>{{"fallLeft",animation.Map(value["fall_map"])}});
        var flags=BindingFlags.NonPublic|BindingFlags.Instance;
        physics.OnNoLongerGrounded+=(Action<Fix>)Delegate.CreateDelegate(typeof(Action<Fix>),slime,typeof(SlimeController).GetMethod("OnNoLongerGrounded",flags));
        physics.OnAttachedToGround+=(Action<RaycastInformation>)Delegate.CreateDelegate(typeof(Action<RaycastInformation>),slime,typeof(SlimeController).GetMethod("attachToGround",flags,null,new[]{typeof(RaycastInformation)},null));
        var context=new Context{Slime=slime,Animation=animation,Flip=flip,TurnInput=F(s["turn_input"]),Objects=objects,Entities=entities,Ropes=ropes,Actor=actor,Platforms=platforms,Adapters=adapters};
        if(value["ragdoll"]!=null)
        {
            var r=value["ragdoll"];Set(slime,"ragdollDuration",F(r["duration"]));Set(slime,"timeInRagdoll",F(r["time"]));Set(slime,"ragDollSpeed",V(r["speed"]));Set(slime,"magnetHolder",context.Magnet(r["magnet_holder"]));
        }
        if(value["abilities"]!=null)
        {
            var routing=value["abilities"];
            context.RoutingPool=routing["pool"].Select((a,index)=>{
                if((string)a["handler"]?["kind"]=="entity")return (AbilityMonoBehaviour)ReferenceOrdinaryAbilities.Link(a,context);
                if(a["handler"]!=null){var instant=ReferenceInstantAbilities.Bind(a,context);context.Instants.Add(index,instant);return (AbilityMonoBehaviour)instant.Ability;}
                var obj=new GameObject("routing ability");obj.SetActive(false);objects.Add(obj);var ability=obj.AddComponent<ReferenceRoutingAbility>();
                ability.IsDestroyed=(bool)a["destroyed"];ability.Cooldown=F(a["cooldown"]);ability.cooldownStoppedDuringTimeStop=(bool)a["stopped_during_time_stop"];ability.OnlyCastableOnGround=(bool)a["ground_only"];ability.isCastable=(bool)a["castable"];ability.IgnoreCooldown=(bool)a["ignore_next_cooldown"];
                if(!(bool)a["present"])UnityEngine.Object.DestroyImmediate(obj);return (AbilityMonoBehaviour)ability;
            }).ToArray();
            slime.abilities=routing["slots"].Select(id=>id.Type==JTokenType.Null?null:(AbilityMonoBehaviour)context.RoutingPool[(int)id]).ToList();
            Set(slime,"abilityCooldownTimers",routing["timers"].Type==JTokenType.Null?null:routing["timers"].Select(F).ToArray());Set(slime,"isInAbility",(bool)routing["in_ability"]);
            slime.AbilityReadyIndicators=routing["indicators"].Select(i=>ReferenceSlimeExit.Indicator(i,objects)).ToArray();
            context.RoutingInputs(routing);
        }
        if(value["entry"]!=null){var entry=value["entry"];Set(slime,"playerAbilityInfoPackage",ReferenceInstantAbilities.Info(entry["info"],context));Set(slime,"wasInvisible",(bool)entry["was_invisible"]);if(entry["invisible_on_entry"]!=null)Set(slime,"wasInvisibileWhenEnteredAbility",(bool)entry["invisible_on_entry"]);Set(slime,"ropeCastBuffer",ReferenceInstantAbilities.Target(entry["rope_target"],context));}
        if(value["spawn"]!=null)ReferenceActorSpawn.Bind(value["spawn"],context);
        return context;
    }
}
