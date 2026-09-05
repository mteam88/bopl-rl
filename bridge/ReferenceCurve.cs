using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;

public static class ReferenceCurve
{
    internal static AnimationCurveFixed Read(JToken value)
    {
        var curve = new AnimationCurveFixed();
        var animation = new AnimationCurve();
        animation.preWrapMode = (WrapMode)(int)value["pre_wrap"];
        animation.postWrapMode = (WrapMode)(int)value["post_wrap"];
        typeof(AnimationCurveFixed).GetField("Anim", BindingFlags.Instance | BindingFlags.NonPublic).SetValue(curve, animation);
        var keys = (JArray)value["keys"];
        curve.keys = new KeyFrameF[keys.Count];
        for (int i = 0; i < keys.Count; i++)
        {
            var key = keys[i];
            curve.keys[i] = new KeyFrameF {
                time = F(key["time"]), value = F(key["value"]),
                inTangent = F(key["in_tangent"]), outTangent = F(key["out_tangent"])
            };
        }
        return curve;
    }
    public static JToken Evaluate(JObject query)
    {
        var curve = Read(query["curve"]);
        var output = new JArray();
        foreach (var time in query["times"]) output.Add(curve.Evaluate(F(time)).m_rawValue);
        return output;
    }
}
