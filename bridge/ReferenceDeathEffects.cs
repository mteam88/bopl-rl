using System;
using System.Collections.Generic;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;
using static ReferenceDynamics;

// Observes original spawned prefab components; never replaces their methods.
internal sealed class ReferenceDeathEffects : IDisposable
{
    internal static ReferenceDeathEffects Active;
    internal static Action<string,MonoUpdatable> ComponentPhase;
    internal readonly GameObject Prefab;
    private readonly JToken initial;
    private readonly Dictionary<Sprite,uint> sprites=new Dictionary<Sprite,uint>();
    private readonly List<Entry> entries=new List<Entry>();
    private int nextNode,nextComponent;
    internal Func<PlayerBody,int> ActorId;
    internal Func<Material,int> MaterialId;
    private sealed class Entry {internal GameObject Go;internal DeathEffect2 Effect;internal SpriteAnimator Animator;internal int Node;internal int[] Components;internal int Actor;internal Material Material;internal int? MaterialIdentity;internal JArray Splashes=new JArray();internal JObject Removed;internal JArray Registration;}
    internal ReferenceDeathEffects(JToken value,JToken schedule)
    {
        initial=value;nextNode=schedule["nodes"].Count();nextComponent=schedule["components"].Count();
        Prefab=Resources.FindObjectsOfTypeAll<PlayerCollision>().Select(p=>p.deathEffect).FirstOrDefault(p=>p!=null&&p.GetComponent<DeathEffect2>()!=null);
        if(Prefab==null)throw new InvalidOperationException("Actual player death prefab unavailable");
        var e=Prefab.GetComponent<DeathEffect2>();var p=value["prefab"];
        if(e.numberGibbs!=(int)p["gibbs"]||e.gibbMaxDistance.m_rawValue!=(long)p["distance"]||e.hitStopFrames!=(int)p["hitstop"])throw new ArgumentException("Death prefab configuration mismatch");
        foreach(string name in new[]{"anticipation","start","end"})
        {
            var frames=e.animData.GetAnimation(name);var expected=p[name];if(frames.Length!=expected.Count())throw new ArgumentException("Death animation length");
            for(int i=0;i<frames.Length;i++)
            {
                if(frames[i].frameDuration!=(int)expected[i]["duration"])throw new ArgumentException("Death animation duration");
                uint id=(uint)expected[i]["sprite"];if(sprites.Any(v=>v.Value==id&&v.Key!=frames[i].sprite))throw new ArgumentException("Death sprite identity collision");if(sprites.TryGetValue(frames[i].sprite,out uint old)&&old!=id)throw new ArgumentException("Death sprite alias");sprites[frames[i].sprite]=id;
            }
        }
        if((uint)p["initial_sprite"]!=sprites[Prefab.GetComponent<SpriteRenderer>().sprite])throw new ArgumentException("Death initial sprite");
        Active=this;
    }
    internal static void Spawned(GameObject go,PlayerBody body)
    {
        var self=Active;if(self==null)return;
        var e=go.GetComponent<DeathEffect2>();if(e==null)throw new NotSupportedException("Unbound death prefab type");
        var material=Get<IPlayerIdHolder>(body.GetComponent<PlayerCollision>(),"playerIdHolder").GetPlayerMaterial();
        if(go.GetComponent<SpriteRenderer>().sharedMaterial!=material)throw new ArgumentException("Death material assignment mismatch");
        self.entries.Add(new Entry{Go=go,Effect=e,Animator=go.GetComponent<SpriteAnimator>(),Node=self.nextNode++,Components=new[]{self.nextComponent++,self.nextComponent++,self.nextComponent++},Actor=self.ActorId(body),Material=material,MaterialIdentity=material==null?(int?)null:self.MaterialId(material)});
    }
    internal static void Splash(DeathEffect2 e,Vec2 p)
    {
        var self=Active;if(self==null)return;self.entries.First(v=>v.Effect==e).Splashes.Add(V(p));
    }
    internal int? Id(MonoUpdatable component)
    {
        foreach(var e in entries){if(ReferenceEquals(e.Effect,component))return e.Components[0];if(ReferenceEquals(e.Animator,component))return e.Components[1];}return null;
    }
    internal int Actor(MonoUpdatable component)=>entries.First(e=>ReferenceEquals(e.Effect,component)||ReferenceEquals(e.Animator,component)).Actor;
    internal int? Node(GameObject go)=>entries.Where(e=>ReferenceEquals(e.Go,go)).Select(e=>(int?)e.Node).FirstOrDefault();
    internal void Removing(GameObject go)
    {
        foreach(var e in entries)if(ReferenceEquals(e.Go,go)){e.Removed=Snapshot(e);e.Removed["present"]=false;e.Registration=Components(e);foreach(var c in e.Registration)c["present"]=false;}
    }
    private JToken Id(Sprite sprite)=>sprite==null?JValue.CreateNull():new JValue(sprites[sprite]);
    private JToken Frames(SpriteFrame[] frames)=>frames==null?JValue.CreateNull():new JArray(frames.Select(f=>new JObject{["sprite"]=Id(f.sprite),["duration"]=f.frameDuration}));
    private JObject Animator(SpriteAnimator a)
    {
        var r=a.spriteRen;var cb=Get<Action>(a,"animationSwappedCallBack");
        return new JObject{["render"]=new JObject{["sprite"]=Id(r.sprite),["flip_x"]=r.flipX,["flip_y"]=r.flipY},["current"]=Frames(Get<SpriteFrame[]>(a,"currentAnim")),["next"]=Frames(Get<SpriteFrame[]>(a,"nextAnim")),["index"]=Get<int>(a,"currentSpriteIndex"),["position"]=Get<int>(a,"currentSpritePos"),["kind"]=(int)Get<SpriteAnimator.AnimationType>(a,"animType"),["swapped"]=a.swappedAnimationThisUpdate,["callback"]=cb==null?JValue.CreateNull():new JValue(cb.Method.Name=="startExplode"?1:cb.Method.Name=="OnExplode"?2:cb.Method.Name=="SelfDestruct"?3:throw new ArgumentException("Unexpected death callback")),["body"]=null,["velocity_maps"]=null,["tracks"]=null,["position_y"]=Get<int>(a,"currentSpritePosY"),["going_right"]=Get<bool>(a,"anim2DGoingRight"),["target"]=Get<int>(a,"anim2DtargetAnim"),["x_speeds"]=null,["min_x"]=Get<Fix>(a,"minXSpeed").m_rawValue,["swap_timing"]=Get<int>(a,"animationSwapTiming"),["callback_last"]=Get<bool>(a,"callBackOnLastFrame"),["always_stop"]=a.AlwaysStopDuringTimeStop,["position_2d"]=Get<int>(a,"curr2DAnimPos"),["animation_index"]=Get<int>(a,"currAnimIndex"),["child_suffix"]=a.childAnimationSuffix};
    }
    private JObject Snapshot(Entry e)
    {
        if(e.Removed!=null)return e.Removed;
        // SpriteRenderer.material clones the assigned material on first splash.
        // Preserve its owning player's identity across that visual-only clone.
        return new JObject{["actor"]=e.Actor,["node"]=e.Node,["components"]=new JArray(e.Components),["position"]=V(e.Go.GetComponent<FixTransform>().position),["scale"]=(double)e.Go.transform.localScale.x,["material"]=e.MaterialIdentity.HasValue?new JValue(e.MaterialIdentity.Value):JValue.CreateNull(),["vibrate"]=Get<bool>(e.Effect,"vibrate"),["present"]=true,["destroyed"]=e.Effect.IsDestroyed,["animator"]=Animator(e.Animator),["splashes"]=e.Splashes.DeepClone()};
    }
    private JArray Components(Entry e)=>e.Registration??new JArray(new MonoUpdatable[]{e.Effect,e.Animator}.Select((c,i)=>new JObject{["id"]=e.Components[i],["hierarchy"]=c.HierarchyNumber,["destroyed"]=c.IsDestroyed,["present"]=true}));
    internal JObject Registration(List<ReferenceActorDriver> drivers)
    {
        object Field(string name)=>typeof(Updater).GetField(name,System.Reflection.BindingFlags.Static|System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Public).GetValue(null);
        JToken Ref(MonoUpdatable c){if(ReferenceEquals(c,null))return JValue.CreateNull();var id=Id(c);if(id.HasValue)return new JValue(id.Value);return new JValue(drivers.FindIndex(d=>ReferenceEquals(d,c)));}
        JArray Refs(IEnumerable<MonoUpdatable> values)=>new JArray(values.Select(Ref));
        return new JObject{["added"]=(int)Field("addedUpdatables"),["prefabs_added"]=(int)Field("prefabsAdded"),["registering_prefab"]=(bool)Field("registeringAPrefab"),["pending"]=Refs((List<MonoUpdatable>)Field("toBeAddedUpdatables")),["live"]=Refs(Updater.updatables),["hitstop"]=(int)Field("hitStopFrames"),["dynamic"]=new JArray(entries.SelectMany(e=>Components(e)))};
    }
    internal JObject Snapshot()=>new JObject{["prefab"]=initial["prefab"].DeepClone(),["instances"]=new JArray(entries.Select(Snapshot))};
    public void Dispose(){ComponentPhase=null;Active=null;foreach(var e in entries)if(e.Go!=null)UnityEngine.Object.DestroyImmediate(e.Go);}
}
