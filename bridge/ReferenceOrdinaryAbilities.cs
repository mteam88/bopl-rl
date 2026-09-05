using System;
using System.Collections.Generic;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;
using static ReferenceDynamics;

// Original Ability on a separate physical actor. Concrete IAbilityComponents
// are added separately; the plain component's entry array is genuinely empty.
internal static class ReferenceOrdinaryAbilities
{
    internal sealed class Instance
    {
        internal Ability Ability;
        internal List<ReferenceSlimeActors.Context> Slimes;
        internal int Owner;
        private ReferenceSlimeActors.Context Context => Slimes[Owner];
        private JToken Id(SlimeController slime)=>slime==null?JValue.CreateNull():new JValue(Slimes.FindIndex(s=>s!=null&&s.Slime==slime));
        internal JObject Snapshot(JToken initial)
        {
            var value=(JObject)initial.DeepClone();
            value["info"]=ReferenceInstantAbilities.Info(Ability.GetPlayerInfo(),Context);
            value["slime"]=Id(Ability.GetSlimeController());
            var callback=Get<Action<AbilityExitInfo>>(Ability,"onExitAbility");
            value["callback"]=callback==null?JValue.CreateNull():Id(callback.Target as SlimeController);
            value["previous_scale"]=Get<Fix>(Ability,"prevScale").m_rawValue;
            value["body_linked"]=Get<PlayerBody>(Ability,"playerBody")!=null;
            value["collider_linked"]=Get<IPhysicsCollider>(Ability,"potentialCollider")!=null;
            return value;
        }
        internal void Exit(JToken info)=>Ability.ExitAbility(ReferenceSlimeExit.Info(info,Context));
    }
    internal static Ability Link(JToken metadata,ReferenceSlimeActors.Context owner)
    {
        if(!(bool)metadata["present"]||(bool)metadata["ignore_next_cooldown"])throw new NotSupportedException("Ordinary ability link requires a present base Ability and its fixed IgnoreNextCooldown=false");
        var ability=owner.Entities[(int)metadata["handler"]["actor"]].GetComponent<Ability>();
        ability.Cooldown=F(metadata["cooldown"]);ability.cooldownStoppedDuringTimeStop=(bool)metadata["stopped_during_time_stop"];
        ability.OnlyCastableOnGround=(bool)metadata["ground_only"];ability.isCastable=(bool)metadata["castable"];ability.IsDestroyed=(bool)metadata["destroyed"];
        return ability;
    }
    internal static Instance Bind(JToken state,GameObject go,List<ReferenceSlimeActors.Context> slimes)
    {
        var ability=go.GetComponent<Ability>();int owner=(int)state["owner"];
        Set(ability,"playerInfo",ReferenceInstantAbilities.Info(state["info"],slimes[owner]));
        Set(ability,"slimeController",state["slime"].Type==JTokenType.Null?null:slimes[(int)state["slime"]].Slime);
        Action<AbilityExitInfo> callback=null;
        if(state["callback"].Type!=JTokenType.Null)callback=(Action<AbilityExitInfo>)Delegate.CreateDelegate(typeof(Action<AbilityExitInfo>),slimes[(int)state["callback"]].Slime,typeof(SlimeController).GetMethod("onExitAbility",BindingFlags.Instance|BindingFlags.NonPublic));
        Set(ability,"onExitAbility",callback);Set(ability,"prevScale",F(state["previous_scale"]));
        if(!(bool)state["body_linked"])Set(ability,"playerBody",null);
        if(!(bool)state["collider_linked"])Set(ability,"potentialCollider",null);
        return new Instance{Ability=ability,Slimes=slimes,Owner=owner};
    }
}
