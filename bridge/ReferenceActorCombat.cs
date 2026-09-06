using System;
using System.Collections.Generic;
using System.Reflection;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;

// Binds the real PlayerCollision death path to the composed actor scene.
internal sealed class ReferenceActorCombat : IDisposable
{
    private const BindingFlags Flags=BindingFlags.Static|BindingFlags.NonPublic;
    private readonly JToken initial;
    private readonly List<Player> players;
    private readonly JArray events=new JArray();
    private readonly FieldInfo kills=typeof(AchievementHandler).GetField("killsThisFrame",Flags);
    private readonly FieldInfo initialized=typeof(GameSession).GetField("isInitialized",Flags);
    private readonly object oldKills,oldInitialized;
    private readonly int oldStocks;
    private readonly ShakableCamera camera;
    private readonly float oldTrauma,oldMax;
    private readonly string oldTag;
    internal ReferenceActorCombat(JToken value,List<Player> roster)
    {
        initial=value;players=roster;oldKills=kills.GetValue(null);oldInitialized=initialized.GetValue(null);oldStocks=Settings.Get().Stocks;
        if((bool)value["stocks_enabled"]||(bool)value["drop_enabled"])throw new NotSupportedException("Actor stock/drop scene binding is not installed");
        initialized.SetValue(null,false);Settings.Get().Stocks=0;
        kills.SetValue(null,value["book"]["registered_kills"].ToObject<int[]>());
        foreach(var e in value["events"])events.Add(e.DeepClone());
        for(int i=0;i<players.Count;i++)
        {
            var p=players[i];var v=value["book"]["players"][i];var r=value["revival"][i];
            if((int)v["id"]!=p.Id)throw new ArgumentException("Combat player identity");
            p.Team=(int)v["team"];p.playersAndClonesStillAlive=(int)v["bodies_alive"];p.WonThisRound=(bool)v["won_round"];p.Kills=(int)v["kills"];p.Deaths=(int)v["deaths"];p.stillAliveThisRound=(bool)v["still_alive_round"];p.CauseOfDeath=(CauseOfDeath)(int)v["cause"];p.RumbleState=(Rumble)(int)v["rumble"];
            typeof(Player).GetProperty("TimestampWhenKilled").SetValue(p,(long)v["killed_at"]);
            if(r["spawn_position"]!=null)p.SpawnPosition=ReferenceDynamics.V(r["spawn_position"]);
            p.ExtraLife=(ReviveStatus)(int)r["status"];p.StocksRemaining=(int)r["stocks"];
            if((int)r["respawn_count"]!=0)throw new NotSupportedException("Actor revive indicators are not installed");
            p.playerDied+=(id,cause)=>events.Add(new JObject{["player"]=id,["cause"]=(int)cause});
        }
        camera=GameObject.FindGameObjectWithTag("ShakableCamera").GetComponent<ShakableCamera>();oldTrauma=camera.GetTrauma();oldMax=camera.maxTrauma;oldTag=camera.gameObject.tag;
        Set(camera,"trauma",(float)value["trauma"]);camera.maxTrauma=(float)value["max_trauma"];
        if(!(bool)value["camera_present"])camera.gameObject.tag="Untagged";
    }
    internal static void Bind(PlayerCollision collision,JToken value)
    {
        collision.deathEffect=ReferenceDeathEffects.Active?.Prefab;collision.SetInvulnerable(true,(uint)value["invulnerable"]);Set(collision,"wasKilledThisFrame",(bool)value["was_killed"]);collision.deathTrauma=(float)value["death_trauma"];
    }
    internal static JObject Collision(PlayerCollision collision)=>new JObject{["invulnerable"]=Get<uint>(collision,"isInvulnerableMask"),["was_killed"]=Get<bool>(collision,"wasKilledThisFrame"),["death_trauma"]=(double)collision.deathTrauma};
    internal JObject Snapshot()
    {
        var result=(JObject)initial.DeepClone();var book=ReferencePlayerLife.Snapshot(players,events);result["events"]=book["events"];book.Remove("events");result["book"]=book;
        for(int i=0;i<players.Count;i++){var r=result["revival"][i];r["status"]=(int)players[i].ExtraLife;r["stocks"]=players[i].StocksRemaining;r["respawn_count"]=players[i].RespawnPositions.Count;if(r["spawn_position"]!=null)r["spawn_position"]=ReferenceDynamics.V(players[i].SpawnPosition);if(r["positions"]!=null)r["positions"]=new JArray(players[i].RespawnPositions.ConvertAll(v=>ReferenceSpellWorld.Active.MarkerIndex(v)));if(players[i].ReviveInstance!=null)r["ability"]=ReferenceSpellWorld.Active.ActorId(players[i].ReviveInstance.GetComponent<QuickSpell>());}
        result["trauma"]=(double)camera.GetTrauma();result["max_trauma"]=(double)camera.maxTrauma;return result;
    }
    public void Dispose()
    {
        kills.SetValue(null,oldKills);initialized.SetValue(null,oldInitialized);Settings.Get().Stocks=oldStocks;
        Set(camera,"trauma",oldTrauma);camera.maxTrauma=oldMax;camera.gameObject.tag=oldTag;
    }
}
