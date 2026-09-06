using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;
using static ReferenceDynamics;

// Records original spell prefab instances and their real Updater registration.
internal sealed class ReferenceSpellWorld : IDisposable
{
    internal static ReferenceSpellWorld Active;
    private readonly JObject initial;
    private readonly ReferenceDeathEffects effects;
    private readonly List<Entry> entries=new List<Entry>();
    private readonly Dictionary<string,GameObject> prefabs=new Dictionary<string,GameObject>();
    private int nextNode,nextComponent;
    private readonly List<ReferencePhysicsScene.Adapter> adapters;
    internal Func<QuickSpell,int> ActorId;
    internal Func<int,int> ObjectId=id=>id;
    internal Func<int,int> RuntimeObjectId=id=>id;
    private sealed class Entry
    {
        internal GameObject Go;internal Shockwave Wave;internal string Prefab;internal int Actor,Node;
        internal int[] Components;internal JObject Removed,Cached;internal JArray Registration;internal ReferenceRevivalMarker Marker;internal MonoBehaviour[] Behaviours;
    }
    internal ReferenceSpellWorld(JToken state,JToken schedule,ReferenceDeathEffects effects,List<ReferencePhysicsScene.Adapter> adapters)
    {
        this.adapters=adapters;
        initial=(JObject)state.DeepClone();this.effects=effects;nextNode=schedule["nodes"].Count();nextComponent=schedule["components"].Count();
        if(state["instances"].Count()!=0)throw new NotSupportedException("Initial spell instances are not bound");
        foreach(var p in ((JObject)state["prefabs"]).Properties())
        {
            if((string)p.Value["kind"]=="marker"){prefabs.Add(p.Name,ReferenceRevivalMarker.Prefab(p.Value["config"]));continue;}
            if((string)p.Value["kind"]!="shockwave")throw new NotSupportedException("Unbound spell kind");
            var source=Resources.FindObjectsOfTypeAll<QuickSpell>().FirstOrDefault(q=>q.name=="Gust"&&q.SpellPrefab!=null)?.SpellPrefab;
            if(source==null||source.GetComponent<Shockwave>()==null)throw new InvalidOperationException("Actual Gust shockwave prefab unavailable");
            var prefab=UnityEngine.Object.Instantiate(source);prefab.name="reference Gust spell prefab";
            var wave=prefab.GetComponent<Shockwave>();var c=p.Value["config"];
            wave.platformForce=F(c["platform_force"]);wave.defaultForce=F(c["default_force"]);wave.groundedPlayerForceMultiplier=F(c["grounded_multiplier"]);wave.radius=F(c["radius"]);wave.collisionMask=unchecked((int)(uint)c["mask"]);
            wave.ActivateOnStart=(bool)c["activate_on_start"];wave.CallEveryFrame=(bool)c["every_frame"];wave.forceCurve01=ReferenceCurve.Read(c["curve"]);wave.killAfterOnFrame=(bool)c["kill_after_frame"];wave.isGustAbility=(bool)c["gust"];
            prefabs.Add(p.Name,prefab);
        }
        Active=this;
    }
    internal GameObject Prefab(string id)=>prefabs[id];
    internal static void Spawned(GameObject go,GameObject prefab,QuickSpell caster)
    {
        var self=Active;if(self==null)return;
        string key=self.prefabs.Single(p=>ReferenceEquals(p.Value,prefab)).Key;
        int node=self.effects!=null?self.effects.ReserveNodes(1):self.nextNode++;
        bool marker=go.GetComponent<RevivePositionIndicator>()!=null;int count=marker?4:2;
        int component=self.effects!=null?self.effects.ReserveComponents(count):self.nextComponent;
        if(self.effects==null)self.nextComponent+=count;
        var entry=new Entry{Go=go,Wave=go.GetComponent<Shockwave>(),Prefab=key,Actor=self.ActorId(caster),Node=node,Components=Enumerable.Range(component,count).ToArray()};
        if(marker){entry.Marker=new ReferenceRevivalMarker(go,self.adapters);entry.Behaviours=new MonoBehaviour[]{entry.Marker.Indicator,entry.Marker.Transform,entry.Marker.BodyComponent,entry.Marker.CircleComponent};}
        else entry.Behaviours=new MonoBehaviour[]{entry.Wave,go.GetComponent<FixTransform>()};
        self.entries.Add(entry);
    }
    internal JToken MarkerIndex(RevivePositionIndicator marker)
    {
        if(ReferenceEquals(marker,null))return JValue.CreateNull();int index=entries.FindIndex(e=>e.Marker!=null&&ReferenceEquals(e.Marker.Indicator,marker));
        if(index<0)throw new NotSupportedException("Unbound marker identity");return new JValue(index);
    }
    internal void CircleCollision(DPhysicsCircle circle,CollisionInformation info)
    {
        var entry=entries.Find(e=>e.Marker!=null&&ReferenceEquals(e.Marker.CircleComponent,circle));entry?.Marker.Adapter.Observe(info);
    }
    internal void Command(JToken command)
    {
        var e=entries[(int)command["spell"]];var c=command["command"];
        switch((string)c["kind"])
        {
            case "activate":e.Wave.ActivateShockWave(F(c["dt"]));break;
            case "ignore":e.Wave.AddObjectToIgnore(RuntimeObjectId((int)c["object"]));break;
            case "scale":e.Wave.SetScale(F(c["value"]));break;
            case "destroy":Updater.DestroyFix(e.Go);break;
            case "end":e.Marker.Indicator.End();break;
            case "add_scale":e.Marker.Indicator.AddScale(F(c["value"]));break;
            case "position":e.Marker.BodyComponent.position=V(c["value"]);break;
            case "manual_init":e.Marker.CircleComponent.ManualInit();break;
            default:throw new NotSupportedException("Unknown spell command");
        }
    }
    internal int? ComponentId(MonoBehaviour c)
    {
        foreach(var e in entries)for(int i=0;i<e.Behaviours.Length;i++)if(ReferenceEquals(e.Behaviours[i],c))return e.Components[i];return null;
    }
    internal int? Id(MonoUpdatable c)=>ComponentId(c);
    internal int Actor(MonoUpdatable c)=>entries.First(e=>e.Behaviours.Any(b=>ReferenceEquals(b,c))).Actor;
    internal int? Node(GameObject go)=>entries.Where(e=>ReferenceEquals(e.Go,go)).Select(e=>(int?)e.Node).FirstOrDefault();
    private JObject Snapshot(Entry e)
    {
        if(e.Removed!=null)return e.Removed;
        if(e.Cached!=null)return e.Cached;
        if(e.Marker!=null)return new JObject{["prefab"]=e.Prefab,["actor"]=e.Actor,["node"]=e.Node,["components"]=new JArray(e.Components),["position"]=V(e.Marker.Transform.position),["present"]=true,["destroyed"]=e.Marker.Indicator.IsDestroyed,["state"]=new JObject{["kind"]="marker",["state"]=e.Marker.Snapshot()}};
        var w=e.Wave;var hits=Get<PhysicsParent[]>(w,"hits");
        var state=new JObject{["scale"]=Get<Fix>(w,"scale").m_rawValue,["time"]=Get<Fix>(w,"timeAlive").m_rawValue,["owner"]=w.GetPlayerId(),["hit_count"]=Get<int>(w,"hitCount"),
            ["hits"]=new JArray(hits.Select(h=>ReferenceEquals(h.fixTrans,null)?JValue.CreateNull():(JToken)new JValue(ObjectId(h.instanceId)))),
            ["hit_objects"]=new JArray(Get<HashSet<int>>(w,"gameObjectIdsHit").Select(ObjectId).OrderBy(x=>x)),["ignored"]=new JArray(Get<HashSet<int>>(w,"extraObjectsToIgnore").Select(ObjectId).OrderBy(x=>x))};
        return new JObject{["prefab"]=e.Prefab,["actor"]=e.Actor,["node"]=e.Node,["components"]=new JArray(e.Components),["position"]=V(e.Go.GetComponent<FixTransform>().position),["present"]=true,["destroyed"]=w.IsDestroyed,["state"]=new JObject{["kind"]="shockwave",["state"]=state}};
    }
    internal void RemovingMono(MonoBehaviour mono)
    {
        foreach(var e in entries)if(e.Behaviours.Any(b=>ReferenceEquals(b,mono))&&e.Cached==null)e.Cached=Snapshot(e);
    }
    internal void Removing(GameObject go)
    {
        foreach(var e in entries)if(ReferenceEquals(e.Go,go))
        {
            e.Removed=Snapshot(e);e.Removed["present"]=false;e.Registration=Components(e);foreach(var c in e.Registration)c["present"]=false;
        }
    }
    private JArray Components(Entry e)=>e.Registration??new JArray(e.Behaviours.Select((b,i)=>(b,i)).Where(pair=>pair.b is MonoUpdatable).Select(pair=>new JObject{["id"]=e.Components[pair.i],["hierarchy"]=((MonoUpdatable)pair.b).HierarchyNumber,["destroyed"]=((MonoUpdatable)pair.b).IsDestroyed,["present"]=pair.b!=null}));
    internal JArray Components()=>new JArray(entries.SelectMany(Components));
    internal JObject Snapshot()=>new JObject{["prefabs"]=initial["prefabs"].DeepClone(),["instances"]=new JArray(entries.Select(Snapshot))};
    internal void JoinRegistration()
    {
        if(effects==null)return;var previousId=effects.AdditionalId;var previousComponents=effects.AdditionalComponents;
        effects.AdditionalId=c=>Id(c)??previousId?.Invoke(c);
        effects.AdditionalComponents=()=>new JArray(Components().Concat(previousComponents?.Invoke()??new JArray()));
    }
    internal JObject Registration(List<ReferenceActorDriver> drivers)
    {
        if(effects!=null)return effects.Registration(drivers);
        object Field(string name)=>typeof(Updater).GetField(name,BindingFlags.Static|BindingFlags.NonPublic|BindingFlags.Public).GetValue(null);
        JToken Ref(MonoUpdatable c)
        {
            if(ReferenceEquals(c,null))return JValue.CreateNull();var id=Id(c);if(id.HasValue)return new JValue(id.Value);
            int index=drivers.FindIndex(d=>ReferenceEquals(d,c));if(index<0)throw new NotSupportedException("Unbound registered component "+c.GetType().Name);return new JValue(index);
        }
        JArray Refs(IEnumerable<MonoUpdatable> values)=>new JArray(values.Select(Ref));
        return new JObject{["added"]=(int)Field("addedUpdatables"),["prefabs_added"]=(int)Field("prefabsAdded"),["registering_prefab"]=(bool)Field("registeringAPrefab"),["pending"]=Refs((List<MonoUpdatable>)Field("toBeAddedUpdatables")),["live"]=Refs(Updater.updatables),["hitstop"]=(int)Field("hitStopFrames"),["dynamic"]=new JArray(Components().OrderBy(c=>(int)c["id"]))};
    }
    public void Dispose(){Active=null;foreach(var e in entries)if(e.Go!=null)UnityEngine.Object.DestroyImmediate(e.Go);foreach(var p in prefabs.Values)UnityEngine.Object.DestroyImmediate(p);}
}
