using System;
using System.Collections.Generic;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;
using static ReferenceDynamics;

// Reads the original particle and its animation, without replacing their methods.
internal sealed class ReferenceReviveParticle
{
    internal readonly ReviveParticle Particle;
    internal readonly FixTransform Transform;
    internal readonly SpriteAnimator Animator;
    private readonly Dictionary<Sprite,uint> sprites=new Dictionary<Sprite,uint>();
    internal static GameObject Prefab(JToken config)
    {
        var effect=Resources.FindObjectsOfTypeAll<PlayerCollision>().First(p=>p.reviveEffectPrefab!=null).reviveEffectPrefab;
        var source=Get<ReviveParticle>(effect,"Particle");
        if(source==null)throw new InvalidOperationException("Actual revival particle unavailable");
        var go=UnityEngine.Object.Instantiate(source.gameObject);go.name="reference revival particle prefab";
        Set(go.GetComponent<ReviveParticle>(),"speed",F(config["speed"]));
        return go;
    }
    internal ReferenceReviveParticle(GameObject go,JToken config)
    {
        Particle=go.GetComponent<ReviveParticle>();Transform=go.GetComponent<FixTransform>();Animator=go.GetComponent<SpriteAnimator>();
        void Bind(Sprite sprite,uint id)
        {
            if(sprites.TryGetValue(sprite,out var old)&&old!=id||sprites.Any(p=>p.Value==id&&p.Key!=sprite))throw new ArgumentException("Particle sprite identity mismatch");sprites[sprite]=id;
        }
        var frames=Get<AnimationData>(Particle,"spawnEffect").GetAnimation("spawn");
        if(frames.Length!=config["spawn"].Count())throw new ArgumentException("Particle animation length");
        for(int i=0;i<frames.Length;i++){if(frames[i].frameDuration!=(int)config["spawn"][i]["duration"])throw new ArgumentException("Particle animation duration");Bind(frames[i].sprite,(uint)config["spawn"][i]["sprite"]);}
        Bind(Animator.spriteRen.sprite,(uint)config["initial_sprite"]);
    }
    private JToken Id(Sprite s)=>s==null?JValue.CreateNull():new JValue(sprites[s]);
    private JToken Frames(SpriteFrame[] frames)=>frames==null?JValue.CreateNull():new JArray(frames.Select(f=>new JObject{["sprite"]=Id(f.sprite),["duration"]=f.frameDuration}));
    private JObject Animation()
    {
        var a=Animator;var r=a.spriteRen;var cb=Get<Action>(a,"animationSwappedCallBack");
        if(cb!=null&&cb.Method.DeclaringType!=typeof(ReviveParticle))throw new ArgumentException("Unexpected particle callback");
        return new JObject{["render"]=new JObject{["sprite"]=Id(r.sprite),["flip_x"]=r.flipX,["flip_y"]=r.flipY},["current"]=Frames(Get<SpriteFrame[]>(a,"currentAnim")),["next"]=Frames(Get<SpriteFrame[]>(a,"nextAnim")),["index"]=Get<int>(a,"currentSpriteIndex"),["position"]=Get<int>(a,"currentSpritePos"),["kind"]=(int)Get<SpriteAnimator.AnimationType>(a,"animType"),["swapped"]=a.swappedAnimationThisUpdate,["callback"]=cb==null?JValue.CreateNull():new JValue(1),["body"]=null,["velocity_maps"]=null,["tracks"]=null,["position_y"]=Get<int>(a,"currentSpritePosY"),["going_right"]=Get<bool>(a,"anim2DGoingRight"),["target"]=Get<int>(a,"anim2DtargetAnim"),["x_speeds"]=null,["min_x"]=Get<Fix>(a,"minXSpeed").m_rawValue,["swap_timing"]=Get<int>(a,"animationSwapTiming"),["callback_last"]=Get<bool>(a,"callBackOnLastFrame"),["always_stop"]=a.AlwaysStopDuringTimeStop,["position_2d"]=Get<int>(a,"curr2DAnimPos"),["animation_index"]=Get<int>(a,"currAnimIndex"),["child_suffix"]=a.childAnimationSuffix};
    }
    internal JObject Snapshot(Func<RevivePositionIndicator,JToken> marker,Func<SlimeController,JToken> slime)
    {
        return new JObject{["target"]=marker(Get<RevivePositionIndicator>(Particle,"targetIndicator")),["slime"]=slime(Get<SlimeController>(Particle,"slimeContToRevive")),["clone_id"]=Get<int>(Particle,"playerCloneId"),["distance"]=Get<Fix>(Particle,"distanceToTravel").m_rawValue,["travelled"]=Get<Fix>(Particle,"distanceTravelled").m_rawValue,["arrived"]=Get<bool>(Particle,"hasArrived"),["speed"]=Get<Fix>(Particle,"speed").m_rawValue,["right"]=V(Transform.right),["animator"]=Animation()};
    }
}
