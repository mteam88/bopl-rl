using System;
using System.Collections.Generic;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;
using static ReferenceDynamics;

// Binds identities to the actual death-revival effect prefab and animation assets.
internal sealed class ReferenceReviveEffect
{
    internal readonly ReviveEffect Effect;
    internal readonly FixTransform Transform;
    internal readonly SpriteAnimator Animator;
    private readonly Dictionary<Sprite,uint> sprites=new Dictionary<Sprite,uint>();
    internal static GameObject Prefab()
    {
        var source=Resources.FindObjectsOfTypeAll<PlayerCollision>().First(p=>p.reviveEffectPrefab!=null).reviveEffectPrefab;
        var go=UnityEngine.Object.Instantiate(source.gameObject);go.name="reference revival effect prefab";return go;
    }
    internal ReferenceReviveEffect(GameObject go,JToken config)
    {
        Effect=go.GetComponent<ReviveEffect>();Transform=go.GetComponent<FixTransform>();Animator=go.GetComponent<SpriteAnimator>();
        void Bind(Sprite sprite,uint id)
        {
            if(sprites.TryGetValue(sprite,out var old)&&old!=id||sprites.Any(p=>p.Value==id&&p.Key!=sprite))throw new ArgumentException("Effect sprite identity mismatch");sprites[sprite]=id;
        }
        var data=Get<AnimationData>(Effect,"animData");
        foreach(string name in new[]{"wisp","shoot"})
        {
            var frames=data.GetAnimation(name);if(frames.Length!=config[name].Count())throw new ArgumentException("Effect animation length");
            for(int i=0;i<frames.Length;i++){if(frames[i].frameDuration!=(int)config[name][i]["duration"])throw new ArgumentException("Effect animation duration");Bind(frames[i].sprite,(uint)config[name][i]["sprite"]);}
        }
        Bind(Animator.spriteRen.sprite,(uint)config["initial_sprite"]);
    }
    private JToken Id(Sprite s)=>s==null?JValue.CreateNull():new JValue(sprites[s]);
    private JToken Frames(SpriteFrame[] frames)=>frames==null?JValue.CreateNull():new JArray(frames.Select(f=>new JObject{["sprite"]=Id(f.sprite),["duration"]=f.frameDuration}));
    private JObject Animation()
    {
        var a=Animator;var r=a.spriteRen;var cb=Get<Action>(a,"animationSwappedCallBack");
        if(cb!=null&&cb.Method.DeclaringType!=typeof(ReviveEffect))throw new ArgumentException("Unexpected revival effect callback");
        return new JObject{["render"]=new JObject{["sprite"]=Id(r.sprite),["flip_x"]=r.flipX,["flip_y"]=r.flipY},["current"]=Frames(Get<SpriteFrame[]>(a,"currentAnim")),["next"]=Frames(Get<SpriteFrame[]>(a,"nextAnim")),["index"]=Get<int>(a,"currentSpriteIndex"),["position"]=Get<int>(a,"currentSpritePos"),["kind"]=(int)Get<SpriteAnimator.AnimationType>(a,"animType"),["swapped"]=a.swappedAnimationThisUpdate,["callback"]=cb==null?JValue.CreateNull():new JValue(cb.Method.Name=="Shoot"?1:cb.Method.Name=="SelfDestruct"?2:throw new ArgumentException("Unexpected effect callback")),["body"]=null,["velocity_maps"]=null,["tracks"]=null,["position_y"]=Get<int>(a,"currentSpritePosY"),["going_right"]=Get<bool>(a,"anim2DGoingRight"),["target"]=Get<int>(a,"anim2DtargetAnim"),["x_speeds"]=null,["min_x"]=Get<Fix>(a,"minXSpeed").m_rawValue,["swap_timing"]=Get<int>(a,"animationSwapTiming"),["callback_last"]=Get<bool>(a,"callBackOnLastFrame"),["always_stop"]=a.AlwaysStopDuringTimeStop,["position_2d"]=Get<int>(a,"curr2DAnimPos"),["animation_index"]=Get<int>(a,"currAnimIndex"),["child_suffix"]=a.childAnimationSuffix};
    }
    internal JObject Snapshot(Func<RevivePositionIndicator,JToken> marker,Func<SlimeController,JToken> slime)
    {
        var player=Get<Player>(Effect,"player");var targets=Get<List<ReviveEffect.SlimeControllerRevival>>(Effect,"playersToRevive");
        return new JObject{["player"]=player==null?JValue.CreateNull():new JValue(player.Id),["targets"]=targets==null?JValue.CreateNull():new JArray(targets.Select(t=>new JObject{["slime"]=slime(t.sm),["marker"]=marker(t.reviveIndicator)})),["right"]=V(Transform.right),["animator"]=Animation()};
    }
}
