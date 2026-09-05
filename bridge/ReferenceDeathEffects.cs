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
    internal readonly GameObject WaterPrefab;
    private readonly JToken initial;
    private readonly Dictionary<Sprite,uint> sprites=new Dictionary<Sprite,uint>();
    private readonly List<Entry> entries=new List<Entry>();
    private int nextNode,nextComponent;
    internal Func<PlayerBody,int> ActorId;
    internal Func<Material,int> MaterialId;
    private readonly Dictionary<Material,int> defaultMaterials=new Dictionary<Material,int>();
    private sealed class Entry {internal GameObject Go;internal MonoUpdatable Effect;internal bool Water;internal SpriteAnimator Animator;internal int Node;internal int[] Components;internal int Actor;internal Material Material;internal int? MaterialIdentity;internal JArray Splashes=new JArray();internal JObject Removed;internal JArray Registration;}
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
        if(value["water_prefab"]!=null)
        {
            WaterPrefab=Resources.FindObjectsOfTypeAll<DestroyIfOutsideSceneBounds>().Select(b=>b.DownEffect).FirstOrDefault(g=>g!=null&&g.name=="WaterDeathEffect 1");
            if(WaterPrefab==null)throw new InvalidOperationException("Actual water death prefab unavailable");
            var play=WaterPrefab.GetComponent<PlayAnimation>();var w=value["water_prefab"];
            if(!play.SelfDestructOnAnimationEnd||play.SelfDeactivateOnAnimationEnd||play.animationToPlay!="enter"||!string.IsNullOrEmpty(play.animationOnEnd))throw new ArgumentException("Water animation configuration");
            var frames=play.data.GetAnimation(play.animationToPlay);if(frames.Length!=w["frames"].Count())throw new ArgumentException("Water animation length");
            for(int i=0;i<frames.Length;i++)
            {
                if(frames[i].frameDuration!=(int)w["frames"][i]["duration"])throw new ArgumentException("Water animation duration");
                uint id=(uint)w["frames"][i]["sprite"];if(sprites.Any(v=>v.Value==id&&v.Key!=frames[i].sprite))throw new ArgumentException("Water sprite identity collision");if(sprites.TryGetValue(frames[i].sprite,out uint old)&&old!=id)throw new ArgumentException("Water sprite alias");sprites[frames[i].sprite]=id;
            }
            if((uint)w["initial_sprite"]!=sprites[WaterPrefab.GetComponent<SpriteRenderer>().sprite])throw new ArgumentException("Water initial sprite");
        }
        Active=this;
    }
    internal GameObject BoundPrefab(JToken value)
    {
        if(value==null||value.Type==JTokenType.Null)return null;
        var go=(string)value["kind"]=="water"?WaterPrefab:Prefab;
        if(go==null)throw new ArgumentException("Missing boundary prefab");
        if(go.GetComponent<FixTransform>().position!=V(value["offset"])||go.transform.localScale.x!=(float)value["scale"])throw new ArgumentException("Boundary prefab transform mismatch: "+go.name+" runtime="+V(go.GetComponent<FixTransform>().position)+" serialized="+V(go.GetComponent<FixTransform>().serializedPosition)+" scale="+go.transform.localScale.x+" expected="+value);
        var material=go.GetComponent<SpriteRenderer>().sharedMaterial;int? id=(int?)value["material"];
        if((material==null)!=(!id.HasValue))throw new ArgumentException("Boundary prefab material mismatch");
        if(material!=null){if(defaultMaterials.Any(p=>p.Value==id.Value&&p.Key!=material)||defaultMaterials.TryGetValue(material,out int old)&&old!=id.Value)throw new ArgumentException("Boundary material identity collision");defaultMaterials[material]=id.Value;}
        return go;
    }
    internal static void BoundarySpawned(GameObject go,DestroyIfOutsideSceneBounds owner)
    {
        var self=Active;if(self==null)return;
        var death=go.GetComponent<DeathEffect2>();bool water=death==null;
        var effect=water?(MonoUpdatable)go.GetComponent<PlayAnimation>():death;
        if(effect==null)throw new NotSupportedException("Unbound boundary effect");
        var material=go.GetComponent<SpriteRenderer>().sharedMaterial;
        int? id=material==null?(int?)null:self.defaultMaterials.TryGetValue(material,out int found)?found:self.MaterialId(material);
        self.entries.Add(new Entry{Go=go,Effect=effect,Water=water,Animator=go.GetComponent<SpriteAnimator>(),Node=self.nextNode++,Components=new[]{self.nextComponent++,self.nextComponent++,self.nextComponent++},Actor=self.ActorId(owner.GetComponent<PlayerBody>()),Material=material,MaterialIdentity=id});
    }
    internal static void BoundaryColored(GameObject go)
    {
        var self=Active;if(self==null)return;var entry=self.entries.First(e=>ReferenceEquals(e.Go,go));var material=go.GetComponent<SpriteRenderer>().sharedMaterial;
        entry.Material=material;entry.MaterialIdentity=material==null?(int?)null:self.defaultMaterials.TryGetValue(material,out int found)?found:self.MaterialId(material);
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
        var value=new JObject{["actor"]=e.Actor,["node"]=e.Node,["components"]=new JArray(e.Components),["position"]=V(e.Go.GetComponent<FixTransform>().position),["scale"]=(double)e.Go.transform.localScale.x,["material"]=e.MaterialIdentity.HasValue?new JValue(e.MaterialIdentity.Value):JValue.CreateNull(),["vibrate"]=!e.Water&&Get<bool>(e.Effect,"vibrate"),["present"]=true,["destroyed"]=e.Effect.IsDestroyed,["animator"]=Animator(e.Animator),["splashes"]=e.Splashes.DeepClone()};
        if(e.Water)value["kind"]="water";return value;
    }
    private JArray Components(Entry e)=>e.Registration??new JArray(new MonoUpdatable[]{e.Effect,e.Animator}.Select((c,i)=>new JObject{["id"]=e.Components[i],["hierarchy"]=c.HierarchyNumber,["destroyed"]=c.IsDestroyed,["present"]=true}));
    internal JObject Registration(List<ReferenceActorDriver> drivers)
    {
        object Field(string name)=>typeof(Updater).GetField(name,System.Reflection.BindingFlags.Static|System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Public).GetValue(null);
        JToken Ref(MonoUpdatable c){if(ReferenceEquals(c,null))return JValue.CreateNull();var id=Id(c);if(id.HasValue)return new JValue(id.Value);int index=drivers.FindIndex(d=>ReferenceEquals(d,c));if(index<0)throw new NotSupportedException("Unbound registered component "+c.GetType().Name+" on "+c.gameObject.name);return new JValue(index);}
        JArray Refs(IEnumerable<MonoUpdatable> values)=>new JArray(values.Select(Ref));
        return new JObject{["added"]=(int)Field("addedUpdatables"),["prefabs_added"]=(int)Field("prefabsAdded"),["registering_prefab"]=(bool)Field("registeringAPrefab"),["pending"]=Refs((List<MonoUpdatable>)Field("toBeAddedUpdatables")),["live"]=Refs(Updater.updatables),["hitstop"]=(int)Field("hitStopFrames"),["dynamic"]=new JArray(entries.SelectMany(e=>Components(e)))};
    }
    internal JObject Snapshot(){var value=new JObject{["prefab"]=initial["prefab"].DeepClone(),["instances"]=new JArray(entries.Select(Snapshot))};if(initial["water_prefab"]!=null)value["water_prefab"]=initial["water_prefab"].DeepClone();return value;}
    public void Dispose(){ComponentPhase=null;Active=null;foreach(var e in entries)if(e.Go!=null)UnityEngine.Object.DestroyImmediate(e.Go);}
}
