using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

internal static class ReferenceAnimator
{
    internal sealed class Context
    {
        internal GameObject GameObject;
        internal SpriteAnimator Animator;
        internal SpeedContainer[] Bodies;
        internal Func<JObject> Snapshot;
        internal Func<JToken,SpriteFrame[]> Frames;
        internal Func<JToken,VelocityFrame[]> Map;
        internal Action<Action,uint> RegisterCallback;
    }
    internal static JToken Evaluate(JObject query, Func<Context,JToken> operation=null)
    {
        var objects=new List<UnityEngine.Object>();
        var oldPlayers=PlayerHandler.Get().PlayerList();var oldScale=GameTime.PlayerTimeScale;
        try
        {
            var texture=new Texture2D(1,1);objects.Add(texture);
            var sprites=new Dictionary<uint,Sprite>();var ids=new Dictionary<Sprite,uint>();
            Sprite Sprite(JToken v)
            {
                if(v.Type==JTokenType.Null)return null;uint id=(uint)v;
                if(!sprites.TryGetValue(id,out var s)){s=UnityEngine.Sprite.Create(texture,new Rect(0,0,1,1),Vector2.zero);sprites.Add(id,s);ids.Add(s,id);objects.Add(s);}
                return s;
            }
            JToken Id(Sprite s)=>s==null?JValue.CreateNull():new JValue(ids[s]);
            SpriteFrame[] Frames(JToken v)=>v.Type==JTokenType.Null?null:v.Select(f=>new SpriteFrame(Sprite(f["sprite"]),(int)f["duration"])).ToArray();
            JToken FrameValues(SpriteFrame[] fs)=>fs==null?JValue.CreateNull():new JArray(fs.Select(f=>new JObject{["sprite"]=Id(f.sprite),["duration"]=f.frameDuration}));
            VelocityFrame[] Map(JToken v)=>v.Type==JTokenType.Null?null:v.Select(f=>new VelocityFrame(Sprite(f["sprite"]),F(f["from"]),F(f["to"]))).ToArray();
            VelocityFrame[][] Maps(JToken v)=>v.Type==JTokenType.Null?null:v.Select(Map).ToArray();
            JToken MapValues(VelocityFrame[][] maps)=>maps==null?JValue.CreateNull():new JArray(maps.Select(row=>row==null?JValue.CreateNull():(JToken)new JArray(row.Select(f=>new JObject{["sprite"]=Id(f.sprite),["from"]=f.speedFrom.m_rawValue,["to"]=f.speedTo.m_rawValue}))));
            Animation2DFrame[] Tracks(JToken v)=>v.Type==JTokenType.Null?null:v.Select(t=>new Animation2DFrame("",Frames(t["frames"]),(int)t["duration"])).ToArray();
            JToken TrackValues(Animation2DFrame[] tracks)=>tracks==null?JValue.CreateNull():new JArray(tracks.Select(t=>new JObject{["frames"]=FrameValues(t.animation),["duration"]=t.frameDuration}));
            var bodies=query["bodies"].Select(b=>new SpeedContainer(V(b["velocity"]),F(b["rotation"]))).ToArray();
            SpeedContainer Body(JToken index)=>index.Type==JTokenType.Null?null:bodies[(int)index];
            PlayerHandler.Get().SetPlayerList(query["players"].Select(p=>new Player((int)p["id"],0){isProtectedFromTimeStop=(bool)p["protected"]}).ToList());
            var go=new GameObject("animator oracle");go.SetActive(false);objects.Add(go);
            var holder=go.AddComponent<Hitbox>();if(query["holder"].Type!=JTokenType.Null)holder.SetPlayerId((int)query["holder"]);
            var animator=go.AddComponent<SpriteAnimator>();Set(animator,"idHolder",query["holder"].Type==JTokenType.Null?null:holder);
            var state=query["state"];
            string[,] fields={{"index","currentSpriteIndex"},{"position","currentSpritePos"},{"position_y","currentSpritePosY"},{"target","anim2DtargetAnim"},{"swap_timing","animationSwapTiming"},{"position_2d","curr2DAnimPos"},{"animation_index","currAnimIndex"}};
            for(int i=0;i<fields.GetLength(0);i++)Set(animator,fields[i,1],(int)state[fields[i,0]]);
            Set(animator,"animType",(SpriteAnimator.AnimationType)(int)state["kind"]);Set(animator,"swappedAnim",(bool)state["swapped"]);
            Set(animator,"anim2DGoingRight",(bool)state["going_right"]);Set(animator,"callBackOnLastFrame",(bool)state["callback_last"]);
            animator.AlwaysStopDuringTimeStop=(bool)state["always_stop"];animator.childAnimationSuffix=(string)state["child_suffix"];
            Set(animator,"currentAnim",Frames(state["current"]));Set(animator,"nextAnim",Frames(state["next"]));Set(animator,"animatedBody",Body(state["body"]));
            Set(animator,"currentVAnim",Maps(state["velocity_maps"]));Set(animator,"animation2D",Tracks(state["tracks"]));
            Set(animator,"XSpeeds",state["x_speeds"].Type==JTokenType.Null?null:state["x_speeds"].Select(F).ToArray());Set(animator,"minXSpeed",F(state["min_x"]));
            animator.spriteRen=state["render"].Type==JTokenType.Null?null:go.GetComponent<SpriteRenderer>();
            if(animator.spriteRen!=null){var r=state["render"];animator.spriteRen.sprite=Sprite(r["sprite"]);animator.spriteRen.flipX=(bool)r["flip_x"];animator.spriteRen.flipY=(bool)r["flip_y"];}
            var callbacks=new Dictionary<uint,Action>();var callbackIds=new Dictionary<Action,uint>();JArray events=null;
            Action Callback(JToken v)
            {
                if(v.Type==JTokenType.Null)return null;uint id=(uint)v;
                if(!callbacks.TryGetValue(id,out var callback))
                {
                    callback=()=>{events.Add(new JObject{["id"]=id,["state"]=Snapshot()});foreach(var command in query["callbacks"][id.ToString()])Start(command);};
                    callbacks.Add(id,callback);callbackIds.Add(callback,id);
                }
                return callback;
            }
            JObject Snapshot()
            {
                var s=(JObject)state.DeepClone();for(int i=0;i<fields.GetLength(0);i++)s[fields[i,0]]=Get<int>(animator,fields[i,1]);
                s["kind"]=(int)Get<SpriteAnimator.AnimationType>(animator,"animType");s["swapped"]=animator.swappedAnimationThisUpdate;
                s["going_right"]=Get<bool>(animator,"anim2DGoingRight");s["callback_last"]=Get<bool>(animator,"callBackOnLastFrame");s["always_stop"]=animator.AlwaysStopDuringTimeStop;
                s["child_suffix"]=animator.childAnimationSuffix;var callback=Get<Action>(animator,"animationSwappedCallBack");s["callback"]=callback==null?JValue.CreateNull():new JValue(callbackIds[callback]);
                var body=Get<SpeedContainer>(animator,"animatedBody");s["body"]=body==null?JValue.CreateNull():new JValue(Array.IndexOf(bodies,body));
                s["current"]=FrameValues(Get<SpriteFrame[]>(animator,"currentAnim"));s["next"]=FrameValues(Get<SpriteFrame[]>(animator,"nextAnim"));s["velocity_maps"]=MapValues(Get<VelocityFrame[][]>(animator,"currentVAnim"));s["tracks"]=TrackValues(Get<Animation2DFrame[]>(animator,"animation2D"));
                var speeds=Get<Fix[]>(animator,"XSpeeds");s["x_speeds"]=speeds==null?JValue.CreateNull():new JArray(speeds.Select(f=>f.m_rawValue));s["min_x"]=Get<Fix>(animator,"minXSpeed").m_rawValue;
                var r=animator.spriteRen;s["render"]=r==null?JValue.CreateNull():new JObject{["sprite"]=Id(r.sprite),["flip_x"]=r.flipX,["flip_y"]=r.flipY};return s;
            }
            void Start(JToken a)
            {
                switch((string)a["op"])
                {
                    case "begin":
                        var f=Frames(a["frames"]);var callback=Callback(a["callback"]);
                        switch((int)a["style"]){case 0:animator.beginAnimation(f);break;case 1:animator.beginAnimThenPlayAnim(f,Frames(a["next"]),callback);break;case 2:animator.beginAnim(f,callback);break;case 3:animator.beginAnimThenDoAction(f,callback);break;default:throw new ArgumentException("Unknown begin style");}break;
                    case "y":animator.beginYSpeedBasedAnimation(Map(a["map"]),Frames(a["looping"]),Sprite(a["anticipation"]),Body(a["body"]));break;
                    case "xy":
                        var maps=Maps(a["maps"]);var speeds=a["speeds"].Type==JTokenType.Null?null:a["speeds"].Select(F).ToArray();
                        if((bool)a["immediate"])animator.beginImmediateXYSpeedBasedAnimation(maps,speeds,F(a["min_x"]),(int)a["timing"],Body(a["body"]),(bool)a["flip"]);
                        else animator.beginXYSpeedBasedAnimation(maps,speeds,F(a["min_x"]),(int)a["timing"],Sprite(a["anticipation"]),Body(a["body"]),(bool)a["flip"]);break;
                    case "two_d":
                        if((bool)a["override_sprite"])animator.beginAnimation2D(Tracks(a["tracks"]),(int)a["start_x"],(int)a["start_y"],(bool)a["right"],(int)a["target"],Sprite(a["anticipation"]));
                        else animator.beginAnimation2D(Tracks(a["tracks"]),(int)a["start_x"],(int)a["start_y"],(bool)a["right"],(int)a["target"]);break;
                    case "rotation":animator.beginRotationAnimation(Frames(a["frames"]),Body(a["body"]));break;
                    case "skip":animator.SkipToSpriteOfIndex((int)a["index"]);break;
                    case "target":animator.Setanim2DtargetAnim((int)a["index"]);break;
                    case "child":animator.StartChildAnimations((string)a["suffix"]);break;
                    default:throw new ArgumentException("Unknown animator start");
                }
            }
            Set(animator,"animationSwappedCallBack",Callback(state["callback"]));
            if(operation!=null)return operation(new Context{GameObject=go,Animator=animator,Bodies=bodies,Snapshot=Snapshot,Frames=Frames,Map=Map,
                RegisterCallback=(callback,id)=>{
                    var current=Get<Action>(animator,"animationSwappedCallBack");
                    if(current!=null&&callbackIds[current]==id)Set(animator,"animationSwappedCallBack",callback);
                    callbackIds[callback]=id;callbacks[id]=callback;
                }});
            JObject Frame(JToken result)=>new JObject{["state"]=Snapshot(),["bodies"]=new JArray(bodies.Select(b=>new JObject{["velocity"]=V(b.velocity),["rotation"]=b.rotation.m_rawValue})),["events"]=events,["result"]=result};
            var frames=new JArray();
            foreach(var a in query["actions"])
            {
                events=new JArray();JToken result=JValue.CreateNull();string error=null;
                try
                {
                    switch((string)a["op"])
                    {
                        case "speed_up":
                            result=FrameValues((SpriteFrame[])typeof(SlimeController).GetMethod("SpeedUpAnimation",BindingFlags.Static|BindingFlags.NonPublic).Invoke(null,new object[]{Frames(a["frames"]),(int)a["factor"]}));break;
                        case "animate":animator.UpdateAnimations();break;
                        case "update":GameTime.PlayerTimeScale=F(a["scale"]);animator.UpdateSim(F(a["dt"]));break;
                        case "speed":bodies[(int)a["index"]].velocity=V(a["velocity"]);bodies[(int)a["index"]].rotation=F(a["rotation"]);break;
                        case "current_sprite":result=Id(animator.getCurrentSprite());break;
                        case "passed":result=animator.getNumberOfPassedSprites();break;
                        case "current_index":result=animator.CurrentSpriteIndex();break;
                        case "current_animation":result=FrameValues(animator.GetCurrentAnimation());break;
                        case "left":result=animator.XYAnimIsLeftOfMidPoint();break;
                        default:Start(a);break;
                    }
                }
                catch(Exception e)
                {
                    if(e is TargetInvocationException)e=e.InnerException;
                    if(e is NullReferenceException)error="null_reference";
                    else if(e is IndexOutOfRangeException)error="index_out_of_range";
                    else if(e is OverflowException)error="overflow";
                    else if(e is DivideByZeroException)error="division_by_zero";
                    else throw;
                }
                if(error!=null)return new JObject{["error"]=error,["action"]=frames.Count,["completed"]=frames,["failed"]=Frame(JValue.CreateNull())};
                frames.Add(Frame(result));
            }
            return frames;
        }
        finally
        {
            PlayerHandler.Get().SetPlayerList(oldPlayers);GameTime.PlayerTimeScale=oldScale;
            foreach(var o in objects)if(o!=null)UnityEngine.Object.DestroyImmediate(o);
        }
    }
}
