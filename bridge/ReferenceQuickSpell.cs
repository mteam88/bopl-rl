using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;
using static ReferenceDynamics;

// Observes the original QuickSpell and SpriteAnimator; cast callbacks execute unchanged.
internal static class ReferenceQuickSpell
{
    internal static void Create(GameObject go)
    {
        go.AddComponent<SpriteRenderer>();go.AddComponent<QuickSpell>();
    }
    internal static void Bind(JToken s,QuickSpell quick,ReferenceSlimeActors.Context owner,ReferenceAnimator.Context animation,List<GameObject> objects,List<UnityEngine.Object> assets)
    {
        var data=ScriptableObject.CreateInstance<AnimationData>();assets.Add(data);
        Set(data,"spriteFrames",((JObject)s["animations"]).Properties().ToDictionary(p=>p.Name,p=>animation.Frames(p.Value)));
        Set(quick,"animData",data);Set(quick,"animator",animation.Animator);Set(animation.Animator,"idHolder",quick.GetComponent<Ability>());
        Set(quick,"playerInfo",ReferenceInstantAbilities.Info(s["info"],owner));
        Set(quick,"player",s["player"].Type==JTokenType.Null?null:PlayerHandler.Get().GetPlayer((int)s["player"]));
        Set(quick,"EnterAbilityOffset",V(s["enter_offset"]));Set(quick,"ExitAbilityOffset",V(s["exit_offset"]));Set(quick,"shockwaveOffset",V(s["shockwave_offset"]));
        Set(quick,"AudioToPlayOnEnter",(string)s["audio_enter"]);Set(quick,"AudioToPlayOnCast",(string)s["audio_cast"]);quick.scaleParticleSize=(bool)s["scale_particle_size"];
        if(s["spell"].Type!=JTokenType.Null)throw new NotSupportedException("QuickSpell prefab binding is not configured");
        Set(quick,"spell",null);Set(quick,"spellFunc",null);
        if((bool)s["particle_prefab"])
        {
            var go=new GameObject("headless cast particle prefab");go.SetActive(false);objects.Add(go);Set(quick,"ParticleCastPrefab",go.AddComponent<ParticleSystem>());
        }
        if((bool)s["particle_present"])
        {
            var go=new GameObject("previous cast particle");go.SetActive(false);objects.Add(go);Set(quick,"ParticleOnCast",go.AddComponent<ParticleSystem>());
        }
        foreach(var pair in new[]{("Cast",1u),("EmptyExitInfo",2u)})
        {
            var callback=(Action)Delegate.CreateDelegate(typeof(Action),quick,typeof(QuickSpell).GetMethod(pair.Item1,BindingFlags.Instance|BindingFlags.NonPublic));
            animation.RegisterCallback(callback,pair.Item2);
        }
    }
    internal static JObject Snapshot(JToken initial,QuickSpell quick,ReferenceSlimeActors.Context owner,ReferenceAnimator.Context animation)
    {
        var value=(JObject)initial.DeepClone();value["info"]=ReferenceInstantAbilities.Info(Get<PlayerInfo>(quick,"playerInfo"),owner);
        var player=Get<Player>(quick,"player");value["player"]=player==null?JValue.CreateNull():new JValue(player.Id);
        var particle=Get<ParticleSystem>(quick,"ParticleOnCast");if(particle!=null&&!owner.Objects.Contains(particle.gameObject))owner.Objects.Add(particle.gameObject);value["particle_present"]=particle!=null;value["animator"]=animation.Snapshot();return value;
    }
    internal static void Cast(QuickSpell quick)=>typeof(QuickSpell).GetMethod("Cast",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(quick,null);
}
