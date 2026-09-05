using System;
using System.Collections.Generic;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Runs the actual collision-selection method. Only its downstream kill/attach
// handlers are replaced by recording sinks; their implementations are separate.
public static class ReferenceCollisionPolicy
{
    internal static bool Active;
    private static bool killResult;
    private static JArray effects;
    private static Dictionary<int,int> identities;
    internal static bool Kill(int killer,CauseOfDeath cause)
    {
        effects.Add(new JObject { ["kind"]="kill",["killer"]=killer,["cause"]=(int)cause });
        return killResult;
    }
    internal static bool Attach(StickyRoundedRectangle wall, Vec2 location)
    {
        if(!Active){ReferenceCollisionRuntime.Attach(wall,location);return false;}
        effects.Add(new JObject { ["kind"]="attach",["target"]=identities[wall.gameObject.GetInstanceID()],["position"]=V(location) });return true;
    }
    internal static void Lightning() {if(Active)effects.Add(new JObject { ["kind"]="lightning" });else ReferenceCollisionRuntime.Lightning();}
    private static bool Present(JToken value)=>value!=null&&value.Type!=JTokenType.Null;
    private static IPhysicsCollider[] Ignored(JToken values)
    {
        var result=new IPhysicsCollider[6];
        for(int i=0;i<6;i++)if(Present(values[i]))result[i]=new ReferencePhysicsScene.Adapter {
            IsDestroyed=(bool)values[i]["destroyed"],parent=new PhysicsParent {hierarchyNumber=(int)values[i]["hierarchy"]} };
        return result;
    }
    private static JArray IgnoreSnapshot(PlayerCollision receiver)
    {
        var output=new JArray();foreach(var c in Get<IPhysicsCollider[]>(receiver,"ignoredCollider"))
            output.Add(c==null?JValue.CreateNull():new JObject { ["hierarchy"]=c.GetPhysicsParent().hierarchyNumber,["destroyed"]=c.IsDestroyed });
        return output;
    }
    public static JToken Evaluate(JObject query)
    {
        return Run(query);
    }
    private static JToken Run(JObject query)
    {
        var objects=new List<GameObject>();var groundBodies=new Dictionary<int,BoplBody>();
        identities=new Dictionary<int,int>();
        GameObject Make(string name) {var go=new GameObject(name);go.SetActive(false);objects.Add(go);return go;}
        BoplBody GroundBody(int id) {if(!groundBodies.TryGetValue(id,out var body)){var go=Make("ground body");go.AddComponent<DPhysicsBox>();body=go.AddComponent<BoplBody>();groundBodies.Add(id,body);}return body;}
        var globals=query["globals"];var state=query["receiver"];var players=new List<Player>();
        foreach(var id in globals["players"])players.Add(new Player((int)id,0) {playersAndClonesStillAlive=1,IsLocalPlayer=false});
        foreach(var id in globals["local_players"])players.Find(p=>p.Id==(int)id).IsLocalPlayer=true;
        var victim=players.Find(p=>p.Id==(int)state["player_id"]);if(victim!=null)victim.playersAndClonesStillAlive=(bool)state["player_alive"]?1:0;
        var oldPlayers=PlayerHandler.Get().PlayerList();PlayerHandler.Get().SetPlayerList(players);
        var af=typeof(AchievementHandler).GetField("isAchieved",BindingFlags.Static|BindingFlags.NonPublic);var oldAchievements=af.GetValue(null);
        var achieved=new bool[28];achieved[7]=(bool)globals["engine_achievement"];achieved[8]=(bool)globals["arrow_achievement"];achieved[22]=(bool)globals["rock_achievement"];af.SetValue(null,achieved);
        bool oldReplay=GameLobby.isPlayingAReplay;GameLobby.isPlayingAReplay=(bool)globals["replay"];
        try
        {
            var receiverGo=Make("collision receiver");var ft=receiverGo.AddComponent<FixTransform>();ft.position=V(state["position"]);Set(ft,"rightInner",V(state["right"]));
            var holder=receiverGo.AddComponent<Hitbox>();holder.SetPlayerId((int)state["player_id"]);receiverGo.AddComponent<DPhysicsBox>();
            var body=receiverGo.AddComponent<PlayerBody>();Set(body,"fixTransform",ft);
            PlayerPhysics physics=null;
            if(Present(state["physics"])) {
                physics=receiverGo.AddComponent<PlayerPhysics>();Set(physics,"isGrounded",(bool)state["physics"]["grounded"]);
                if(Present(state["physics"]["attached"])) { var attached=Make("attached ground").AddComponent<StickyRoundedRectangle>();
                    Set(attached,"groundBody",GroundBody((int)state["physics"]["attached"]["body"]));attached.IsDestroyed=(bool)state["physics"]["attached"]["destroyed"];Set(physics,"attachedGround",attached); }
            }
            var receiver=receiverGo.AddComponent<PlayerCollision>();Set(receiver,"playerIdHolder",holder);Set(receiver,"body",body);Set(receiver,"physics",physics);
            receiver.isSquishable=(bool)state["squishable"];receiver.platformSquishPointOffset=V(state["squish_offset"]);
            Set(receiver,"isInvulnerableMask",(uint)state["invulnerable"]);Set(receiver,"wasKilledThisFrame",(bool)state["was_killed"]);
            receiver.IgnoreBeamId((int)state["ignored_beam"]);Set(receiver,"ignoredCollider",Ignored(state["ignored"]));
            var hits=new List<CollisionInformation>();var rocks=new List<BounceBall>();
            foreach(var h in query["hits"])
            {
                var go=Make("collision target");identities.Add(go.GetInstanceID(),(int)h["id"]);go.layer=(int)h["layer"];
                var trans=go.AddComponent<FixTransform>();trans.IsDestroyed=(bool)h["destroyed"];
                if((bool)h["chain_lightning"])go.tag="ChainLightning";
                if(Present(h["platform"])) {var platform=go.AddComponent<StickyRoundedRectangle>();Set(platform,"groundBody",GroundBody((int)h["platform"]["body"]));Set(platform,"pickupPlayerFrames",(bool)h["platform"]["pickup"]?1:0);
                    if(Present(h["platform"]["engine"])) {var e=h["platform"]["engine"];var engine=go.AddComponent<RocketEngine>();engine.ownerId=(int)e["owner"];engine.IsDestroyed=(bool)e["destroyed"];Set(engine,"isEngineOn",(bool)e["on"]);} }
                if(Present(h["orb_kind"])) {var orb=go.AddComponent<Orb>();Set(orb,"orbType",(Orb.OrbType)(int)h["orb_kind"]);}
                if(Present(h["explosion"])) {var e=h["explosion"];var explosion=go.AddComponent<Explosion>();explosion.PlayerOwnerId=(int)e["owner"];explosion.canHurtOwner=(bool)e["hurt_owner"];if((bool)e["beam"])go.tag="beam";
                    if(Present(e["beam_id"]))go.AddComponent<BeamObject>().beamId=(int)e["beam_id"]; }
                if(Present(h["hitbox_owner"]))go.AddComponent<Hitbox>().SetPlayerId((int)h["hitbox_owner"]);
                if(Present(h["projectile"])) {var p=h["projectile"];var projectile=go.AddComponent<Projectile>();projectile.SetPlayerId((int)p["owner"]);projectile.canHurtOwner=(bool)p["hurt_owner"];projectile.timePassed=F(p["age"]);
                    if(Present(p["arrow_destroyed"]))go.AddComponent<Arrow>().IsDestroyed=(bool)p["arrow_destroyed"]; }
                if(Present(h["holder_owner"]))go.AddComponent<Hitbox>().SetPlayerId((int)h["holder_owner"]);
                BounceBall rock=null;
                if(Present(h["ability"])) {var a=h["ability"];var ability=go.AddComponent<Ability>();ability.SetPlayerId((int)a["owner"]);ability.KillPlayersOnContact=(bool)a["kills_on_contact"];
                    if(Present(a["rock"])) {rock=go.AddComponent<BounceBall>();rock.playersKilledThisUse=(int)a["rock"]["hits"];
                        if(Present(a["rock"]["player_local"]))Set(rock,"player",new Player(1000,0){IsLocalPlayer=(bool)a["rock"]["player_local"]});} }
                else if(Present(h["parent_ability"])) {var parent=Make("contact ability parent");go.transform.SetParent(parent.transform);var a=h["parent_ability"];var ability=parent.AddComponent<Ability>();ability.SetPlayerId((int)a["owner"]);ability.KillPlayersOnContact=(bool)a["kills_on_contact"];parent.SetActive(true);}
                rocks.Add(rock);
                var shape=h["collider"];IShape collider=(string)shape["kind"]=="circle"?(IShape)Circle(shape["shape"]):(string)shape["kind"]=="box"?(IShape)Box(shape["shape"]):Rounded(shape["shape"]);
                hits.Add(new CollisionInformation {layer=(int)h["collision_layer"],collider=collider,colliderPP=new PhysicsParent {hierarchyNumber=(int)h["hierarchy"],fixTrans=(bool)h["transform_present"]?trans:null}});
            }
            Active=true;var output=new JArray();
            JObject Snapshot(JToken result)
            {
                var rs=(JObject)state.DeepClone();rs["was_killed"]=Get<bool>(receiver,"wasKilledThisFrame");rs["invulnerable"]=Get<uint>(receiver,"isInvulnerableMask");rs["ignored_beam"]=Get<int>(receiver,"ignoredBeamId");rs["ignored"]=IgnoreSnapshot(receiver);
                var gs=(JObject)globals.DeepClone();gs["engine_achievement"]=achieved[7];gs["arrow_achievement"]=achieved[8];gs["rock_achievement"]=achieved[22];
                var hs=(JArray)query["hits"].DeepClone();for(int i=0;i<rocks.Count;i++)if(rocks[i]!=null)hs[i]["ability"]["rock"]["hits"]=rocks[i].playersKilledThisUse;
                return new JObject { ["receiver"]=rs,["globals"]=gs,["hits"]=hs,["effects"]=effects,["result"]=result };
            }
            foreach(var action in query["actions"])
            {
                effects=new JArray();JToken result=JValue.CreateNull();
                try { switch((string)action["op"]) {
                    case "update":receiver.UpdateSim(Fix.Zero);break;
                    case "invulnerable":receiver.SetInvulnerable((bool)action["enabled"],(uint)action["mask"]);break;
                    case "ignore_beam":receiver.IgnoreBeamId((int)action["id"]);break;
                    case "copy_ignored":var copyGo=Make("copy source");copyGo.AddComponent<DPhysicsBox>();var copy=copyGo.AddComponent<PlayerCollision>();Set(copy,"ignoredCollider",Ignored(action["ignored"]));receiver.CopyIgnoredColliderFrom(copy);break;
                    case "collide":killResult=(bool)action["kill_result"];if((bool)action["callback"])receiver.OnCollide(hits[(int)action["hit"]]);else result=receiver.KillPlayerOnCollision(hits[(int)action["hit"]]);break;
                    default:throw new ArgumentException("Unknown collision policy operation");
                } }
                catch(NullReferenceException) {return new JObject { ["error"]="null_reference",["action"]=output.Count,["completed"]=output,["failed"]=Snapshot(JValue.CreateNull()) };}
                catch(OverflowException) {return new JObject { ["error"]="overflow",["action"]=output.Count,["completed"]=output,["failed"]=Snapshot(JValue.CreateNull()) };}
                output.Add(Snapshot(result));
            }
            return output;
        }
        finally {Active=false;PlayerHandler.Get().SetPlayerList(oldPlayers);af.SetValue(null,oldAchievements);GameLobby.isPlayingAReplay=oldReplay;foreach(var go in objects)if(go!=null)UnityEngine.Object.DestroyImmediate(go);}
    }
}
