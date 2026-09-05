using System;
using System.Collections.Generic;
using System.Reflection;
using Newtonsoft.Json.Linq;

// Calls the real Player.Kill, with event listeners that record its dispatch.
// Spawning, ability exits and body destruction are separate lifecycle stages.
public static class ReferencePlayerLife
{
    private const BindingFlags Fields=BindingFlags.Static|BindingFlags.NonPublic;
    private static JObject Snapshot(List<Player> players, JArray events)
    {
        var states=new JArray();
        foreach(var p in players)states.Add(new JObject {
            ["id"]=p.Id,["team"]=p.Team,["bodies_alive"]=p.playersAndClonesStillAlive,["won_round"]=p.WonThisRound,
            ["kills"]=p.Kills,["deaths"]=p.Deaths,["still_alive_round"]=p.stillAliveThisRound,
            ["killed_at"]=p.TimestampWhenKilled,["cause"]=(int)p.CauseOfDeath,["rumble"]=(int)p.RumbleState });
        var counts=(int[])typeof(AchievementHandler).GetField("killsThisFrame",Fields).GetValue(null);
        return new JObject { ["players"]=states,["registered_kills"]=new JArray(counts),["events"]=events.DeepClone() };
    }
    internal static JObject Begin(Player victim, int killer, long timestamp, CauseOfDeath cause)
    {
        if(!ReferenceCapture.Recording)return null;
        var players=PlayerHandler.Get().PlayerList();int index=players.IndexOf(victim);
        if(index<0)throw new InvalidOperationException("Unregistered Player.Kill receiver in reference capture");
        return new JObject { ["operation"]="dynamics_player_life",["state"]=Snapshot(players,new JArray()),
            ["actions"]=new JArray(new JObject { ["victim"]=index,["killer"]=killer,["timestamp"]=timestamp,["cause"]=(int)cause }) };
    }
    internal static void End(JObject query, bool dispatchDeath)
    {
        if(query==null)return;
        var events=new JArray();var action=query["actions"][0];
        if(dispatchDeath)events.Add(new JObject { ["player"]=query["state"]["players"][(int)action["victim"]]["id"], ["cause"]=action["cause"] });
        ReferenceCapture.PlayerLifeStage(new JObject { ["query"]=query,["expected"]=new JArray(Snapshot(PlayerHandler.Get().PlayerList(),events)) });
    }
    public static JToken Evaluate(JObject query)
    {
        var oldPlayers=PlayerHandler.Get().PlayerList();
        var field=typeof(AchievementHandler).GetField("killsThisFrame",Fields);
        var oldKills=field.GetValue(null);
        var players=new List<Player>();var events=new JArray();
        foreach(var v in query["state"]["players"])
        {
            var player=new Player((int)v["id"],(int)v["team"]) {
                playersAndClonesStillAlive=(int)v["bodies_alive"],WonThisRound=(bool)v["won_round"],
                Kills=(int)v["kills"],Deaths=(int)v["deaths"],stillAliveThisRound=(bool)v["still_alive_round"],
                CauseOfDeath=(CauseOfDeath)(int)v["cause"],RumbleState=(Rumble)(int)v["rumble"]
            };
            typeof(Player).GetProperty("TimestampWhenKilled").SetValue(player,(long)v["killed_at"]);
            player.playerDied+=(id,cause)=>events.Add(new JObject { ["player"]=id,["cause"]=(int)cause });
            players.Add(player);
        }
        foreach(var e in query["state"]["events"])events.Add(e.DeepClone());
        var counts=query["state"]["registered_kills"].ToObject<int[]>();
        PlayerHandler.Get().SetPlayerList(players);field.SetValue(null,counts);
        try
        {
            var result=new JArray();
            foreach(var action in query["actions"])
            {
                var player=players[(int)action["victim"]];
                player.Kill((int)action["killer"],(long)action["timestamp"],(CauseOfDeath)(int)action["cause"]);
                result.Add(Snapshot(players,events));
            }
            return result;
        }
        finally {PlayerHandler.Get().SetPlayerList(oldPlayers);field.SetValue(null,oldKills);}
    }
}
