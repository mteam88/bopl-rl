using System;
using System.Collections.Generic;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Observe original prefab instances before SpawnAbilityLessClone prepares them.
// This class adds identities and readers, never replaces prefab component state.
internal sealed class ReferenceActorClones : IDisposable
{
    internal static Action<SlimeController> Created;
    internal static Action<SlimeController,Vec2> Pushed;
    internal int CreatingPlayer;
    private readonly Dictionary<MonoUpdatable,(int Id,int Actor)> components=new Dictionary<MonoUpdatable,(int,int)>();
    private readonly Dictionary<int,ReferencePhysicsScene.Adapter> clones=new Dictionary<int,ReferencePhysicsScene.Adapter>();
    private readonly Action<DPhysicsBox,CollisionInformation> oldCollision;
    internal int? Id(MonoUpdatable c)=>components.TryGetValue(c,out var value)?(int?)value.Id:null;
    internal int Actor(MonoUpdatable c)=>components[c].Actor;
    internal ReferenceActorClones(JObject query,JArray actorData,List<ReferenceSlimeActors.Context> slimes,
        List<PlayerBody> bodies,List<PlayerPhysics> physics,List<FixTransform> transforms,List<GameObject> entities,List<GameObject> objects,
        List<StickyRoundedRectangle> platforms,List<ReferencePhysicsScene.Adapter> adapters,List<RopeBody> ropes,
        ReferenceDeathEffects effects,Action<int,string,JToken> emit,Action<int> select,Action<int,Vec2> push)
    {
        if(effects==null)throw new ArgumentException("Clone capture requires the shared prefab registry");
        var prefab=query["clone_prefab"];if(prefab==null)throw new ArgumentException("Missing clone prefab contract");
        Pushed=(slime,delta)=>{if(delta!=Vec2.zero&&components.TryGetValue(slime,out var value))push(value.Actor,delta);};
        foreach(string key in new[]{"x_effect","up_effect","down_effect"})effects.BoundPrefab(prefab["actor"]["bounds"][key]);
        oldCollision=ReferencePhysicsScene.BoxCollision;
        ReferencePhysicsScene.BoxCollision=(box,hit)=>{oldCollision?.Invoke(box,hit);if(components.TryGetValue(box,out var value)){select(value.Actor);clones[value.Actor].Observe(hit);}};
        var revive=Resources.FindObjectsOfTypeAll<PlayerCollision>().First(p=>p.reviveEffectPrefab!=null&&p.reviveEffectPrefab.emptyPlayerPrefab!=null).reviveEffectPrefab;
        foreach(var body in bodies){var collision=body.GetComponent<PlayerCollision>();if(collision!=null)collision.reviveEffectPrefab=revive;}
        Created=slime=>
        {
            int actor=actorData.Count;select(actor);var go=slime.gameObject;objects.Add(go);var body=go.GetComponent<PlayerBody>();var p=go.GetComponent<PlayerPhysics>();var ft=go.GetComponent<FixTransform>();var box=go.GetComponent<DPhysicsBox>();
            var tree=go.GetComponentsInChildren<Transform>(true);
            if(tree.Length!=prefab["nodes"].Count())throw new ArgumentException("Clone prefab node count");
            int nodeBase=effects.ReserveNodes(tree.Length);int componentBase=effects.ReserveComponents(prefab["components"].Count());
            for(int i=0;i<tree.Length;i++)
            {
                int parent=Array.IndexOf(tree,tree[i].parent);var expected=prefab["nodes"][i]["parent"];
                if(parent!=(expected.Type==JTokenType.Null?-1:(int)expected))throw new ArgumentException("Clone prefab hierarchy");
            }
            var updatables=go.GetComponentsInChildren<MonoUpdatable>(true);
            var all=new List<MonoBehaviour>(updatables);all.Add(ft);
            if(all.Count!=prefab["components"].Count())throw new ArgumentException("Clone prefab component count");
            string Role(MonoBehaviour c)=>c is PlayerPhysics?"physics":c is SlimeController?"motor":c is PlayerCollision?"collision":c is DestroyIfOutsideSceneBounds?"bounds":c is PlayerBody?"body":c is DPhysicsBox?"collider":c is RewindRecorder?"rewind":c is SpriteAnimator?"animation":null;
            for(int i=0;i<all.Count;i++)
            {
                var expected=prefab["components"][i];if((string)expected["role"]!=Role(all[i])||(int)expected["state"]["owner"]!=Array.IndexOf(tree,all[i].transform))throw new ArgumentException("Clone component identity");
                if(all[i] is MonoUpdatable updatable)components.Add(updatable,(componentBase+i,actor));
            }
            var value=(JObject)prefab["actor"].DeepClone();int id=adapters.Select(a=>a.sourceId).Concat(actorData.Select(a=>(int)a["object"])).DefaultIfEmpty(0).Max()+1;
            value["input"]=actorData.First(a=>(int?)a["body"]["holder"]==CreatingPlayer)["input"].DeepClone();
            value["object"]=id;value["entity"]["node"]=(int)value["entity"]["node"]+nodeBase;value["pending_object"]["id"]=id;
            var adapter=new ReferencePhysicsScene.Adapter{sourceId=id,shape=Shape.Box,transform=ft,parent=box.pp,realBox=box,template=(JObject)value["pending_object"].DeepClone()};
            clones.Add(actor,adapter);adapters.Add(adapter);actorData.Add(value);entities.Add(go);bodies.Add(body);physics.Add(p);transforms.Add(ft);
            var animation=ReferenceAnimatorObserver.Create(slime,value["slime"]);
            slimes.Add(new ReferenceSlimeActors.Context{Slime=slime,Animation=animation,Flip=go.GetComponentInChildren<FlippingTransform>(),TurnInput=Fix.Zero,Objects=objects,Entities=entities,Ropes=ropes,Actor=actor,Peers=slimes,Platforms=platforms,Adapters=adapters,RoutingPool=Array.Empty<AbilityMonoBehaviour>()});
            p.OnNoLongerGrounded+=angle=>emit(actor,"detached",angle.m_rawValue);
            p.OnAttachedToGround+=hit=>emit(actor,"attached",new JObject{["object"]=hit.pp.instanceId,["near_position"]=V(hit.nearPos),["far_position"]=V(hit.farPos),["near_distance"]=hit.nearDist.m_rawValue,["far_distance"]=hit.farDist.m_rawValue,["layer"]=hit.layer});
        };
        effects.AdditionalId=Id;
        effects.AdditionalComponents=()=>new JArray(components.OrderBy(p=>p.Value.Id).Select(p=>new JObject{["id"]=p.Value.Id,["hierarchy"]=p.Key.HierarchyNumber,["destroyed"]=p.Key.IsDestroyed,["present"]=p.Key!=null}));
    }
    internal void Snapshot(int actor,JObject value)
    {
        if(!clones.TryGetValue(actor,out var adapter))return;
        var box=adapter.realBox;value["collision_hierarchy"]=box.pp.hierarchyNumber;
        if(box.initHasBeenCalled){value.Remove("pending_object");return;}
        var pending=(JObject)value["pending_object"];
        pending["transform"]["position"]=V(adapter.transform.position);pending["transform"]["rotation"]=adapter.transform.rotation.m_rawValue;pending["transform"]["right"]=V(adapter.transform.right);pending["transform"]["offset"]=V(adapter.transform.offset);
        pending["active"]=box.gameObject.activeInHierarchy;if(box.enabled)pending.Remove("enabled");else pending["enabled"]=false;pending["destroyed"]=box.IsDestroyed;
    }
    public void Dispose(){Created=null;Pushed=null;ReferencePhysicsScene.BoxCollision=oldCollision;}
}
