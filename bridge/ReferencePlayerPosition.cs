using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Bind identities; arbitration itself always runs original Player.SetPosition.
internal static class ReferencePlayerPosition
{
    private static SlimeController Controller(JToken id,List<ReferenceSlimeActors.Context> slimes)
        =>id.Type==JTokenType.Null?null:slimes[(int)id].Slime;
    private static JToken Id(SlimeController value,List<ReferenceSlimeActors.Context> slimes)
        =>value==null?JValue.CreateNull():new JValue(slimes.FindIndex(s=>s.Slime==value));
    internal static void Bind(JToken initial,List<Player> players,List<ReferenceSlimeActors.Context> slimes)
    {
        for(int i=0;i<players.Count;i++)
        {
            var state=initial[i]["position"];if(state==null)continue;
            players[i].Position=V(state["value"]);
            Set(players[i],"positionSetterThisFrame",Controller(state["current"],slimes));
            Set(players[i],"positionSetterLastFrame",Controller(state["previous"],slimes));
        }
    }
    internal static JObject Snapshot(Player player,List<ReferenceSlimeActors.Context> slimes)
        =>new JObject{["value"]=V(player.Position),["current"]=Id(Get<SlimeController>(player,"positionSetterThisFrame"),slimes),["previous"]=Id(Get<SlimeController>(player,"positionSetterLastFrame"),slimes)};
    internal static bool Command(JToken command,List<Player> players,List<ReferenceSlimeActors.Context> slimes)
    {
        switch((string)command["op"])
        {
            case "set_position":players[(int)command["player"]].SetPosition(V(command["value"]),Controller(command["controller"],slimes));return true;
            case "controller_destroyed":slimes[(int)command["actor"]].Slime.IsDestroyed=(bool)command["value"];return true;
            default:return false;
        }
    }
}
