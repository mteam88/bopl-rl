using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Data binding and observation only; UpdateSim executes the original recorder.
internal static class ReferenceActorRewind
{
    internal static void Bind(JToken value, SlimeController slime)
    {
        var recorder=slime.GetComponent<RewindRecorder>()??slime.gameObject.AddComponent<RewindRecorder>();
        Set(recorder,"slime",slime);
        Set(recorder,"ring",value["ring"].Select(s=>new RewindRecorder.Sample {
            position=V(s["position"]),selfImposedVelocity=V(s["velocity"]),externalVelocity=V(s["external_velocity"]),rotation=F(s["rotation"]),
            cooldown0=F(s["cooldowns"][0]),cooldown1=F(s["cooldowns"][1]),cooldown2=F(s["cooldowns"][2])
        }).ToArray());
        Set(recorder,"head",(int)value["head"]);Set(recorder,"count",(int)value["count"]);
        if((bool)value["rewinding"])throw new System.NotSupportedException("Rewind playback binding not implemented");
        Set(recorder,"isRewinding",false);
    }
    internal static JObject Snapshot(RewindRecorder recorder)=>new JObject {
        ["head"]=Get<int>(recorder,"head"),["count"]=Get<int>(recorder,"count"),["rewinding"]=Get<bool>(recorder,"isRewinding"),
        ["ring"]=new JArray(Get<RewindRecorder.Sample[]>(recorder,"ring").Select(s=>new JObject {
            ["position"]=V(s.position),["velocity"]=V(s.selfImposedVelocity),["external_velocity"]=V(s.externalVelocity),["rotation"]=s.rotation.m_rawValue,
            ["cooldowns"]=new JArray(s.cooldown0.m_rawValue,s.cooldown1.m_rawValue,s.cooldown2.m_rawValue)
        }))
    };
}
