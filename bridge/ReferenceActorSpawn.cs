using System;
using System.Reflection;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;
using static ReferenceDynamics;

internal static class ReferenceActorSpawn
{
    internal static void Bind(JToken state,ReferenceSlimeActors.Context owner)
    {
        var s=owner.Slime;s.playerPhysics.enabled=(bool)state["physics_enabled"];s.isAbilitylessClone=(bool)state["abilityless"];
        Set(s,"fastSpawnAnim",(bool)state["fast"]);Set(s,"nextAbiltityToChangeOnPickup",(int)state["next_ability"]);
        if(owner.Animation.Animator.spriteRen!=null)owner.Animation.Animator.spriteRen.enabled=(bool)state["renderer_enabled"];
        owner.Animation.RegisterCallback(s.Spawn,4);
    }
    internal static JObject Snapshot(JToken initial,ReferenceSlimeActors.Context owner)
    {
        var v=(JObject)initial.DeepClone();var s=owner.Slime;v["physics_enabled"]=s.playerPhysics.enabled;v["abilityless"]=s.isAbilitylessClone;
        v["fast"]=Get<bool>(s,"fastSpawnAnim");v["next_ability"]=s.NextAbiltityToChangeOnPickup;
        if(owner.Animation.Animator.spriteRen!=null)v["renderer_enabled"]=owner.Animation.Animator.spriteRen.enabled;return v;
    }
    internal static void Apply(string action,JToken position,ReferenceSlimeActors.Context owner)
    {
        var s=owner.Slime;
        switch(action)
        {
            case "init":s.Init();break;
            case "before":typeof(SlimeController).GetMethod("BeforeSpawnInit",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(s,null);break;
            case "spawn":s.Spawn();break;
            case "prepare":s.PrepareToRevive(V(position));break;
            case "respawn":s.ReSpawn(V(position));break;
            case "fast":s.ReSpawnFast(V(position));break;
            case "unground":s.playerPhysics.UnGround();break;
            default:throw new ArgumentException("Unknown actor spawn action");
        }
    }
}
