using System;
using System.Collections.Generic;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Captures OnCollide decisions without replacing the real downstream handlers.
// Their state changes are excluded from this rule-level projection.
internal static class ReferenceCollisionRuntime
{
    internal sealed class Capture
    {
        internal PlayerCollision receiver;
        internal FixTransform target;
        internal JObject query;
        internal JArray effects=new JArray();
        internal JObject expected;
        internal bool frozen;
    }
    private static readonly Stack<Capture> stack=new Stack<Capture>();
    private static int Id(UnityEngine.Object value)=>ReferencePhysicsCapture.Id(value.GetInstanceID());
    private static JToken NullableId(BoplBody value)=>value==null?JValue.CreateNull():new JValue(Id(value));
    private static JObject Globals()
    {
        var players=new JArray();var local=new JArray();foreach(var p in PlayerHandler.Get().PlayerList()) {players.Add(p.Id);if(p.IsLocalPlayer)local.Add(p.Id);}
        return new JObject { ["players"]=players,["local_players"]=local,["replay"]=GameLobby.isPlayingAReplay,
            ["engine_achievement"]=AchievementHandler.IsAchieved(AchievementHandler.AchievementEnum.GetAKillWithTheEngineAbility),
            ["arrow_achievement"]=AchievementHandler.IsAchieved(AchievementHandler.AchievementEnum.LandALongRangeArrow),
            ["rock_achievement"]=AchievementHandler.IsAchieved(AchievementHandler.AchievementEnum.Get2KillsWith1UseOfRock) };
    }
    private static JObject AbilityState(Ability ability,IPlayerIdHolder holder,BounceBall rock)
    {
        if(ability==null)return null;
        if(holder==null)throw new InvalidOperationException("Contact ability has no owner holder");
        JToken r=JValue.CreateNull();
        if(rock!=null)r=new JObject { ["hits"]=rock.playersKilledThisUse,["player_local"]=rock.getPlayer()==null?JValue.CreateNull():new JValue(rock.getPlayer().IsLocalPlayer) };
        return new JObject { ["owner"]=holder.GetPlayerId(),["kills_on_contact"]=ability.KillPlayersOnContact,["rock"]=r };
    }
    private static JObject Hit(CollisionInformation c)
    {
        var ft=c.colliderPP.fixTrans;var go=ft==null?null:ft.gameObject;
        var h=new JObject { ["id"]=go==null?0:Id(go),["layer"]=go==null?0:go.layer,["collision_layer"]=c.layer,["hierarchy"]=c.colliderPP.hierarchyNumber,
            ["transform_present"]=ft!=null,["destroyed"]=ft!=null&&ft.IsDestroyed,["collider"]=ReferencePhysicsScene.ShapeJson(c.collider),
            ["platform"]=JValue.CreateNull(),["orb_kind"]=JValue.CreateNull(),["explosion"]=JValue.CreateNull(),["hitbox_owner"]=JValue.CreateNull(),
            ["projectile"]=JValue.CreateNull(),["holder_owner"]=JValue.CreateNull(),["ability"]=JValue.CreateNull(),["parent_ability"]=JValue.CreateNull(),["chain_lightning"]=false };
        if(go==null)return h;
        h["chain_lightning"]=go.CompareTag("ChainLightning");
        var platform=go.GetComponent<StickyRoundedRectangle>();
        if(platform!=null) {
            if(platform.GetGroundBody()==null)throw new InvalidOperationException("Platform has no ground body");
            var engine=platform.GetComponentInChildren<RocketEngine>();
            h["platform"]=new JObject { ["body"]=Id(platform.GetGroundBody()),["pickup"]=platform.pickUpPlayersInsteadOfSquish,
                ["engine"]=engine==null?JValue.CreateNull():new JObject { ["owner"]=engine.ownerId,["destroyed"]=engine.IsDestroyed,["on"]=engine.IsEngineOn() } };
        }
        var orb=go.GetComponent<Orb>();if(orb!=null)h["orb_kind"]=(int)Get<Orb.OrbType>(orb,"orbType");
        var explosion=go.GetComponent<Explosion>();var beam=go.GetComponent<BeamObject>();
        if(explosion!=null)h["explosion"]=new JObject { ["owner"]=explosion.PlayerOwnerId,["hurt_owner"]=explosion.canHurtOwner,["beam"]=explosion.CompareTag("beam"),["beam_id"]=beam==null?JValue.CreateNull():new JValue(beam.beamId) };
        var hitbox=go.GetComponent<Hitbox>();if(hitbox!=null)h["hitbox_owner"]=hitbox.GetPlayerId();
        var projectile=go.GetComponent<Projectile>();var arrow=go.GetComponent<Arrow>();
        if(projectile!=null)h["projectile"]=new JObject { ["owner"]=projectile.GetPlayerId(),["hurt_owner"]=projectile.canHurtOwner,["age"]=projectile.timePassed.m_rawValue,["arrow_destroyed"]=arrow==null?JValue.CreateNull():new JValue(arrow.IsDestroyed) };
        var holder=go.GetComponent<IPlayerIdHolder>();if(holder!=null)h["holder_owner"]=holder.GetPlayerId();
        var ability=go.GetComponent<Ability>();
        h["ability"]=AbilityState(ability,holder,go.GetComponent<BounceBall>())??(JToken)JValue.CreateNull();
        if(ability==null)h["parent_ability"]=AbilityState(go.GetComponentInParent<Ability>(),go.GetComponentInParent<IPlayerIdHolder>(),null)??(JToken)JValue.CreateNull();
        return h;
    }
    internal static Capture Begin(PlayerCollision receiver,CollisionInformation collision)
    {
        if(!ReferencePhysicsCapture.Enabled)return null;
        try
        {
            int playerId=receiver.GetPlayerId();var player=PlayerHandler.Get().GetPlayer(playerId);var body=Get<PlayerBody>(receiver,"body");var physics=Get<PlayerPhysics>(receiver,"physics");
            JToken ground=JValue.CreateNull();
            if(physics!=null) {var attached=physics.getAttachedGround();if(attached!=null&&attached.GetGroundBody()==null)throw new InvalidOperationException("Attached platform has no ground body");
                ground=new JObject { ["grounded"]=physics.IsGrounded(),["attached"]=attached==null?JValue.CreateNull():new JObject { ["body"]=NullableId(attached.GetGroundBody()),["destroyed"]=attached.IsDestroyed } };}
            var ignored=new JArray();foreach(var c in Get<IPhysicsCollider[]>(receiver,"ignoredCollider"))ignored.Add(c==null?JValue.CreateNull():new JObject { ["hierarchy"]=c.GetPhysicsParent().hierarchyNumber,["destroyed"]=c.IsDestroyed });
            var state=new JObject { ["player_id"]=playerId,["player_alive"]=player==null?JValue.CreateNull():new JValue(player.IsAlive),["invulnerable"]=Get<uint>(receiver,"isInvulnerableMask"),
                ["was_killed"]=Get<bool>(receiver,"wasKilledThisFrame"),["position"]=V(body.position),["right"]=V(body.right),["squish_offset"]=V(receiver.platformSquishPointOffset),
                ["squishable"]=receiver.isSquishable,["physics"]=ground,["ignored"]=ignored,["ignored_beam"]=Get<int>(receiver,"ignoredBeamId") };
            var capture=new Capture {receiver=receiver,target=collision.colliderPP.fixTrans,query=new JObject { ["operation"]="dynamics_player_collision",["receiver"]=state,["globals"]=Globals(),["hits"]=new JArray(Hit(collision)),
                ["actions"]=new JArray(new JObject { ["op"]="collide",["hit"]=0,["callback"]=true,["kill_result"]=false }) }};
            stack.Push(capture);return capture;
        }
        catch(Exception error) {ReferenceCapture.PlayerCollisionStage(new JObject { ["capture_error"]=error.ToString() });return null;}
    }
    private static void Freeze(Capture capture)
    {
        if(capture.frozen)return;
        var receiver=(JObject)capture.query["receiver"].DeepClone();receiver["was_killed"]=Get<bool>(capture.receiver,"wasKilledThisFrame");
        var hits=(JArray)capture.query["hits"].DeepClone();
        var rock=capture.target==null?null:capture.target.GetComponent<BounceBall>();
        if(rock!=null&&hits[0]["ability"].Type!=JTokenType.Null&&hits[0]["ability"]["rock"].Type!=JTokenType.Null)hits[0]["ability"]["rock"]["hits"]=rock.playersKilledThisUse;
        capture.expected=new JObject { ["receiver"]=receiver,["globals"]=Globals(),["hits"]=hits,["effects"]=capture.effects,["result"]=JValue.CreateNull() };capture.frozen=true;
    }
    internal static void Kill(int killer,CauseOfDeath cause)
    {
        if(stack.Count==0)return;var c=stack.Peek();if(c.frozen)return;
        c.effects.Add(new JObject { ["kind"]="kill",["killer"]=killer,["cause"]=(int)cause });Freeze(c);
    }
    internal static void Attach(StickyRoundedRectangle wall, Vec2 location)
    {
        if(stack.Count!=0&&!stack.Peek().frozen)stack.Peek().effects.Add(new JObject { ["kind"]="attach",["target"]=Id(wall.gameObject),["position"]=V(location) });
    }
    internal static void Lightning()
    {
        if(stack.Count!=0&&!stack.Peek().frozen)stack.Peek().effects.Add(new JObject { ["kind"]="lightning" });
    }
    internal static void End(Capture capture)
    {
        if(capture==null)return;
        try {Freeze(capture);ReferenceCapture.PlayerCollisionStage(new JObject { ["query"]=capture.query,["expected"]=new JArray(capture.expected) });}
        catch(Exception error) {ReferenceCapture.PlayerCollisionStage(new JObject { ["capture_error"]=error.ToString() });}
        finally {if(stack.Pop()!=capture)throw new InvalidOperationException("Collision capture stack mismatch");}
    }
}
