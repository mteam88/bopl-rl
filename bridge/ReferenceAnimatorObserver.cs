using System;
using System.Collections.Generic;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;

// Reads a prefab's real animator without installing supplied animation state.
internal static class ReferenceAnimatorObserver
{
    internal static ReferenceAnimator.Context Create(SlimeController slime,JToken template)
    {
        var animator=slime.GetComponentInChildren<SpriteAnimator>();
        var speed=Get<SpeedContainer>(slime,"animationSpeedContainer");
        var sprites=new Dictionary<Sprite,uint>();var callbacks=new Dictionary<Action,uint>();
        void Bind(Sprite sprite,uint id)
        {
            if(sprites.TryGetValue(sprite,out var previous)&&previous!=id||sprites.Any(p=>p.Value==id&&p.Key!=sprite))throw new ArgumentException("Clone sprite identity mismatch");
            sprites[sprite]=id;
        }
        foreach(var entry in ((JObject)template["animations"]).Properties())
        {
            var actual=slime.animData.GetAnimation(entry.Name);var expected=entry.Value;
            if(actual.Length!=expected.Count())throw new ArgumentException("Clone animation length");
            for(int i=0;i<actual.Length;i++){if(actual[i].frameDuration!=(int)expected[i]["duration"])throw new ArgumentException("Clone animation duration");Bind(actual[i].sprite,(uint)expected[i]["sprite"]);}
        }
        var fall=slime.animData.GetVelocityMap("fallLeft");
        if(fall.Length!=template["fall_map"].Count())throw new ArgumentException("Clone fall map length");
        for(int i=0;i<fall.Length;i++){var expected=template["fall_map"][i];if(fall[i].speedFrom.m_rawValue!=(long)expected["from"]||fall[i].speedTo.m_rawValue!=(long)expected["to"])throw new ArgumentException("Clone fall map bounds");Bind(fall[i].sprite,(uint)expected["sprite"]);}
        JToken Id(Sprite s)=>s==null?JValue.CreateNull():new JValue(sprites[s]);
        JToken Frames(SpriteFrame[] frames)=>frames==null?JValue.CreateNull():new JArray(frames.Select(f=>new JObject{["sprite"]=Id(f.sprite),["duration"]=f.frameDuration}));
        JToken Maps(VelocityFrame[][] maps)=>maps==null?JValue.CreateNull():new JArray(maps.Select(row=>row==null?JValue.CreateNull():(JToken)new JArray(row.Select(f=>new JObject{["sprite"]=Id(f.sprite),["from"]=f.speedFrom.m_rawValue,["to"]=f.speedTo.m_rawValue}))));
        JObject Snapshot()
        {
            var r=animator.spriteRen;var callback=Get<Action>(animator,"animationSwappedCallBack");var body=Get<SpeedContainer>(animator,"animatedBody");
            if(body!=null&&!ReferenceEquals(body,speed))throw new ArgumentException("Clone animation body");
            var tracks=Get<Animation2DFrame[]>(animator,"animation2D");var speeds=Get<Fix[]>(animator,"XSpeeds");
            return new JObject {
                ["render"]=r==null?JValue.CreateNull():new JObject{["sprite"]=Id(r.sprite),["flip_x"]=r.flipX,["flip_y"]=r.flipY},
                ["current"]=Frames(Get<SpriteFrame[]>(animator,"currentAnim")),["next"]=Frames(Get<SpriteFrame[]>(animator,"nextAnim")),
                ["index"]=Get<int>(animator,"currentSpriteIndex"),["position"]=Get<int>(animator,"currentSpritePos"),["kind"]=(int)Get<SpriteAnimator.AnimationType>(animator,"animType"),["swapped"]=animator.swappedAnimationThisUpdate,
                ["callback"]=callback==null?JValue.CreateNull():new JValue(callbacks[callback]),["body"]=body==null?JValue.CreateNull():new JValue(0),
                ["velocity_maps"]=Maps(Get<VelocityFrame[][]>(animator,"currentVAnim")),["tracks"]=tracks==null?JValue.CreateNull():new JArray(tracks.Select(t=>new JObject{["frames"]=Frames(t.animation),["duration"]=t.frameDuration})),
                ["position_y"]=Get<int>(animator,"currentSpritePosY"),["going_right"]=Get<bool>(animator,"anim2DGoingRight"),["target"]=Get<int>(animator,"anim2DtargetAnim"),
                ["x_speeds"]=speeds==null?JValue.CreateNull():new JArray(speeds.Select(f=>f.m_rawValue)),["min_x"]=Get<Fix>(animator,"minXSpeed").m_rawValue,["swap_timing"]=Get<int>(animator,"animationSwapTiming"),
                ["callback_last"]=Get<bool>(animator,"callBackOnLastFrame"),["always_stop"]=animator.AlwaysStopDuringTimeStop,["position_2d"]=Get<int>(animator,"curr2DAnimPos"),["animation_index"]=Get<int>(animator,"currAnimIndex"),["child_suffix"]=animator.childAnimationSuffix
            };
        }
        return new ReferenceAnimator.Context{GameObject=animator.gameObject,Animator=animator,Bodies=new[]{speed},SpriteId=Id,Sprite=value=>value.Type==JTokenType.Null?null:sprites.Single(p=>p.Value==(uint)value).Key,Snapshot=Snapshot,RegisterCallback=(callback,id)=>callbacks[callback]=id};
    }
}
