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
// empty abilities. The nested animator contexts only construct/reference data.
internal static class ReferenceSlimeActors
{
    internal sealed class Context
    {
        internal SlimeController Slime;
        internal ReferenceAnimator.Context Animation;
        internal FlippingTransform Flip;
        internal Fix TurnInput;
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
            else throw new InvalidOperationException("Unmapped slime callback "+name);
            Animation.RegisterCallback(callback,id);
        }
        internal JObject Snapshot(JToken original)
        {
            TrackCallback();var value=(JObject)original.DeepClone();var s=value["state"];
            s["mode"]=(int)Slime.getCurrentState();s["facing_right"]=Flip.facingRight();s["was_upside_down"]=Get<bool>(Slime,"wasUpsideDown");
            s["input"]=V(Get<Vec2>(Slime,"inputVector"));s["previous_input"]=Get<Fix>(Slime,"prevInput").m_rawValue;s["suppress"]=Slime.suppressInput;s["has_spawned"]=Slime.HasSpawned;
            s["time_moonwalking"]=Get<Fix>(Slime,"timeSpentMoonwalking").m_rawValue;s["turn_input"]=TurnInput.m_rawValue;
            value["animator"]=Animation.Snapshot();value["speed"]=new JObject{["velocity"]=V(Animation.Bodies[0].velocity),["rotation"]=Animation.Bodies[0].rotation.m_rawValue};return value;
        }
    }
    internal static JToken Evaluate(JObject query,DetPhysics scene,List<ReferencePhysicsScene.Adapter> adapters,Func<JObject> snapshot)
    {
        var contexts=new List<ReferenceAnimator.Context>();
        JToken Next(int index)
        {
            if(index==query["runtime"]["actors"].Count())return ReferenceActors.Evaluate(query,scene,adapters,snapshot,contexts);
            var actor=query["runtime"]["actors"][index];var slime=actor["slime"];
            var q=new JObject{["state"]=slime["animator"].DeepClone(),["bodies"]=new JArray(slime["speed"].DeepClone()),["holder"]=actor["body"]["holder"].DeepClone(),
                ["players"]=query["runtime"]["scheduler"]["players"].DeepClone(),["callbacks"]=new JObject()};
            return ReferenceAnimator.Evaluate(q,context=>{contexts.Add(context);return Next(index+1);});
        }
        return Next(0);
    }
    internal static Context Bind(JToken value,SlimeController slime,PlayerBody body,PlayerPhysics physics,IPhysicsCollider hurtbox,ReferenceAnimator.Context animation,List<GameObject> objects,List<UnityEngine.Object> assets)
    {
        var s=value["state"];slime.abilities=new List<AbilityMonoBehaviour>();slime.AbilityReadyIndicators=Array.Empty<AbilityReadyIndicator>();
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
        return new Context{Slime=slime,Animation=animation,Flip=flip,TurnInput=F(s["turn_input"])};
    }
}
