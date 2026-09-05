using System;
using System.Collections.Generic;
using System.Reflection;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

internal static class ReferenceSlimeExit
{
    internal static AbilityExitInfo Info(JToken value,ReferenceSlimeActors.Context owner)
    {
        return new AbilityExitInfo{isDead=(bool)value["dead"],killerId=(int)value["killer"],position=V(value["position"]),externalVelocity=V(value["external_velocity"]),selfImposedVelocity=V(value["velocity"]),
            isFacingRight=(bool)value["facing_right"],isGrounded=(bool)value["grounded"],groundedSpeed=F(value["grounded_speed"]),lastSprite=owner.Animation.Sprite(value["sprite"]),
            currentlyAttachedGround=value["ground"].Type==JTokenType.Null?null:owner.Platforms[(int)value["ground"]],groundedLocalPosition=F(value["local"]),justlanded=(bool)value["just_landed"],
            causeOfDeath=(CauseOfDeath)(int)value["cause"],radius=F(value["radius"]),enterTurn=(bool)value["enter_turn"],enterJump=(bool)value["enter_jump"],ropeAttachment=value["rope"].Type==JTokenType.Null?null:owner.Ropes[(int)value["rope"]],topAttachment=(bool)value["top"]};
    }
    internal static void Apply(JToken info,ReferenceSlimeActors.Context owner)
        =>typeof(SlimeController).GetMethod("onExitAbility",BindingFlags.NonPublic|BindingFlags.Instance).Invoke(owner.Slime,new object[]{Info(info,owner)});
    internal static AbilityReadyIndicator Indicator(JToken value,List<GameObject> objects)
    {
        if(value.Type==JTokenType.Null)return null;
        var go=new GameObject("headless routing indicator");go.SetActive(false);objects.Add(go);
        var indicator=go.AddComponent<AbilityReadyIndicator>();
        if(value["lifecycle"]!=null)
        {
            if((bool)value["lifecycle"]["follow"]){go.AddComponent<FollowTransform>();indicator.Init();}
            go.SetActive(true);go.SetActive(false);go.SetActive((bool)value["lifecycle"]["active"]);
        }
        Set(indicator,"SecondsSinceReset",(float)value["seconds_since_reset"]);
        return indicator;
    }
    internal static JToken Indicator(AbilityReadyIndicator indicator,JToken initial)
    {
        if(indicator==null)return JValue.CreateNull();
        var value=(JObject)initial.DeepClone();value["seconds_since_reset"]=Get<float>(indicator,"SecondsSinceReset");
        if(value["lifecycle"]!=null)value["lifecycle"]=new JObject{["active"]=indicator.gameObject.activeSelf,["follow"]=Get<FollowTransform>(indicator,"followTrans")!=null};
        return value;
    }
}
