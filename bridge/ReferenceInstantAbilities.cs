using System;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Events;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Constructs the original InstantAbility + Invisibility component graph. Both
// entry and the cooldown restoration callback execute the shipped methods.
internal static class ReferenceInstantAbilities
{
    internal sealed class Instance
    {
        internal InstantAbility Ability;
        internal Invisibility Effect;
        internal FixTransform Transform;
        internal ReferenceSlimeActors.Context Owner;
        internal JObject Snapshot(JToken initial)
        {
            var value=(JObject)initial.DeepClone();
            value["position"]=V(Transform.position);value["active"]=Ability.gameObject.activeSelf;
            value["info"]=Info(Ability.playerInfo,Owner);value["slime"]=Slime(Ability.GetSlimeController(),Owner);
            return value;
        }
    }
    internal static JToken Slime(SlimeController slime,ReferenceSlimeActors.Context owner)
    {
        if(slime==null)return JValue.CreateNull();
        if(slime!=owner.Slime)throw new InvalidOperationException("Unmapped entry slime");
        return new JValue(owner.Actor);
    }
    internal static PhysicsParent Target(JToken id,ReferenceSlimeActors.Context owner)=>id.Type==JTokenType.Null?default:owner.Adapters.Single(a=>a.sourceId==(int)id).parent;
    internal static JToken Target(PhysicsParent p,ReferenceSlimeActors.Context owner)
    {
        if(p.instanceId==0)return JValue.CreateNull();
        return new JValue(owner.Adapters.Single(a=>a.parent.instanceId==p.instanceId).sourceId);
    }
    internal static PlayerInfo Info(JToken v,ReferenceSlimeActors.Context owner)
    {
        if(v["rope"].Type!=JTokenType.Null)throw new NotSupportedException("Actor rope binding is not installed");
        return new PlayerInfo{
            playerId=(int)v["player"],AbilityButtonUsedIndex012=(int)v["slot"],position=V(v["position"]),externalVelocity=V(v["external_velocity"]),selfImposedVelocity=V(v["velocity"]),
            playerMaterial=v["material"].Type==JTokenType.Null?null:owner.Animation.Animator.spriteRen.material,isGrounded=(bool)v["grounded"],groundedSpeed=F(v["grounded_speed"]),rotation=F(v["rotation"]),
            forwardsVector=V(v["forward"]),upVector=V(v["up"]),isFacingRight=(bool)v["facing_right"],currentlyAttachedGround=v["ground"].Type==JTokenType.Null?null:owner.Platforms[(int)v["ground"]],
            groundedLocalPosition=F(v["local"]),radius=F(v["radius"]),EntrySprite=owner.Animation.Sprite(v["sprite"]),ropeBody=null,topAttachment=(bool)v["top"],
            useRopeOnOtherPlayer=(bool)v["use_rope_on_player"],playerToUseRopeOn=Target(v["rope_target"],owner),slimeController=v["slime"].Type==JTokenType.Null?null:owner.Slime};
    }
    internal static JObject Info(PlayerInfo v,ReferenceSlimeActors.Context owner)
    {
        if(v.ropeBody!=null)throw new NotSupportedException("Actor rope binding is not installed");
        return new JObject{
            ["player"]=v.playerId,["slot"]=v.AbilityButtonUsedIndex012,["position"]=V(v.position),["external_velocity"]=V(v.externalVelocity),["velocity"]=V(v.selfImposedVelocity),
            ["material"]=v.playerMaterial==null?JValue.CreateNull():new JValue(owner.Actor),["grounded"]=v.isGrounded,["grounded_speed"]=v.groundedSpeed.m_rawValue,["rotation"]=v.rotation.m_rawValue,
            ["forward"]=V(v.forwardsVector),["up"]=V(v.upVector),["facing_right"]=v.isFacingRight,["ground"]=v.currentlyAttachedGround==null?JValue.CreateNull():new JValue(owner.Platforms.IndexOf(v.currentlyAttachedGround)),
            ["local"]=v.groundedLocalPosition.m_rawValue,["radius"]=v.radius.m_rawValue,["sprite"]=owner.Animation.SpriteId(v.EntrySprite),["rope"]=JValue.CreateNull(),["top"]=v.topAttachment,
            ["use_rope_on_player"]=v.useRopeOnOtherPlayer,["rope_target"]=Target(v.playerToUseRopeOn,owner),["slime"]=Slime(v.slimeController,owner)};
    }
    internal static Instance Bind(JToken metadata,ReferenceSlimeActors.Context owner)
    {
        var state=metadata["handler"];if((string)state["kind"]!="invisibility")throw new ArgumentException("Unknown instant handler");
        var go=new GameObject("reference instant invisibility");go.SetActive(false);owner.Objects.Add(go);
        var transform=go.AddComponent<FixTransform>();var ability=go.AddComponent<InstantAbility>();
        var effect=(bool)state["component_present"]?go.AddComponent<Invisibility>():null;
        // Prime Awake once before supplying state. The caller resets Updater's
        // registration queues after all actor components have been constructed.
        go.SetActive(true);go.SetActive(false);
        ability.OnlyCastableOnGround=(bool)metadata["ground_only"];ability.IsDestroyed=(bool)metadata["destroyed"];
        ability.SetCoolDown(F(metadata["cooldown"]));ability.cooldownStoppedDuringTimeStop=(bool)metadata["stopped_during_time_stop"];ability.ignoreNextCooldown=(bool)metadata["ignore_next_cooldown"];
        ability.isInvisibilty=(bool)state["gate_invisible"];ability.AudioOnCast=(string)state["audio"];ability.EffectOnEnter=null;ability.funcOnEnter=new UnityEvent();
        if(effect!=null){effect.cooldown=F(state["restore_cooldown"]);if(!(bool)state["linked"])Set(effect,"ia",null);ability.funcOnEnter.AddListener(effect.CastInvisibility);}
        if(!(bool)state["transform_present"])Set(ability,"fixTrans",null);
        transform.position=V(state["position"]);ability.playerInfo=Info(state["info"],owner);Set(ability,"slimeController",state["slime"].Type==JTokenType.Null?null:owner.Slime);
        go.SetActive((bool)state["active"]);
        if(!(bool)metadata["present"])throw new NotSupportedException("Destroyed instant component binding is not installed");
        return new Instance{Ability=ability,Effect=effect,Transform=transform,Owner=owner};
    }
}
