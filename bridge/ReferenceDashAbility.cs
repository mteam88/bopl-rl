using System;
using System.Collections.Generic;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;
using static ReferenceDynamics;

// Binds the original Dash component; renderer-only smear geometry is not projected.
internal static class ReferenceDashAbility
{
    internal static void Create(GameObject go)
    {
        go.AddComponent<SpriteRenderer>();go.AddComponent<UnityEngine.Animator>();
        var dash=go.AddComponent<Dash>();
        var smear=new GameObject("headless dash smear");smear.transform.SetParent(go.transform);
        Set(dash,"SmearFrame",smear.AddComponent<SpriteRenderer>());
    }
    internal static RaycastInformation Hit(JToken h,ReferenceSlimeActors.Context owner)
    {
        var parent=h["object"].Type==JTokenType.Null?default(PhysicsParent):owner.Adapters.Find(a=>a.sourceId==(int)h["object"]).parent;
        parent.hierarchyNumber=(int)h["hierarchy"];
        return new RaycastInformation{pp=parent,nearPos=V(h["near_position"]),farPos=V(h["far_position"]),nearDist=F(h["near_distance"]),farDist=F(h["far_distance"]),layer=(int)h["layer"]};
    }
    internal static JObject Hit(RaycastInformation h)=>new JObject{["hierarchy"]=h.pp.hierarchyNumber,["object"]=h.pp.fixTrans==null?JValue.CreateNull():new JValue(h.pp.instanceId),["near_position"]=V(h.nearPos),["far_position"]=V(h.farPos),["near_distance"]=h.nearDist.m_rawValue,["far_distance"]=h.farDist.m_rawValue,["layer"]=h.layer};
    internal static void Bind(JToken s,Dash dash,ReferenceSlimeActors.Context owner)
    {
        dash.pauseLength=F(s["pause"]);dash.dashLength=F(s["length"]);dash.leaveSpeed=F(s["leave_speed"]);dash.FixedUpdatesInDash=(int)s["fixed_updates"];dash.colliders=unchecked((int)(uint)s["mask"]);dash.anticipationSprite=owner.Animation.Sprite(s["anticipation"]);
        Set(dash,"inputVector",V(s["input"]));Set(dash,"playerInfo",ReferenceInstantAbilities.Info(s["info"],owner));
        Set(dash,"player",s["player"].Type==JTokenType.Null?null:PlayerHandler.Get().GetPlayer((int)s["player"]));
        Set(dash,"timePassed",F(s["time"]));Set(dash,"updatesPassedInDash",(int)s["updates"]);Set(dash,"inDash",(bool)s["in_dash"]);
        Set(dash,"collisions",s["hits"].Select(h=>Hit(h,owner)).ToList());
        Set(dash,"hitGround",s["hit_ground"].Type==JTokenType.Null?null:owner.Platforms[(int)s["hit_ground"]]);Set(dash,"HitGroundLocalPosition",F(s["hit_local"]));
        var sprite=Get<SpriteRenderer>(dash,"spriteRen");sprite.enabled=(bool)s["sprite_enabled"];sprite.sprite=owner.Animation.Sprite(s["sprite"]);Get<SpriteRenderer>(dash,"SmearFrame").enabled=(bool)s["smear_enabled"];
    }
    internal static JObject Snapshot(JToken initial,Dash dash,ReferenceSlimeActors.Context owner)
    {
        var s=(JObject)initial.DeepClone();s["input"]=V(Get<Vec2>(dash,"inputVector"));s["info"]=ReferenceInstantAbilities.Info(Get<PlayerInfo>(dash,"playerInfo"),owner);
        var player=Get<Player>(dash,"player");s["player"]=player==null?JValue.CreateNull():new JValue(player.Id);
        s["time"]=Get<Fix>(dash,"timePassed").m_rawValue;s["updates"]=Get<int>(dash,"updatesPassedInDash");s["in_dash"]=Get<bool>(dash,"inDash");
        s["hits"]=new JArray(Get<List<RaycastInformation>>(dash,"collisions").Select(Hit));
        var ground=Get<StickyRoundedRectangle>(dash,"hitGround");s["hit_ground"]=ground==null?JValue.CreateNull():new JValue(owner.Platforms.IndexOf(ground));s["hit_local"]=Get<Fix>(dash,"HitGroundLocalPosition").m_rawValue;
        var sprite=Get<SpriteRenderer>(dash,"spriteRen");s["sprite_enabled"]=sprite.enabled;s["sprite"]=owner.Animation.SpriteId(sprite.sprite);s["smear_enabled"]=Get<SpriteRenderer>(dash,"SmearFrame").enabled;return s;
    }
}
