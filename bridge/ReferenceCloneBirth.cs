using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;

// Diagnostic capture of the actual empty-player prefab's Awake and clone-birth
// path. No Init/Update is run here; scheduled clone execution has its own probe.
internal static class ReferenceCloneBirth
{
    private const BindingFlags Fields=BindingFlags.Instance|BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.DeclaredOnly;
    private const BindingFlags Statics=BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic;
    private static JToken Value(object value)
    {
        if(value==null)return JValue.CreateNull();
        if(value is Fix f)return new JValue(f.m_rawValue);
        if(value is Vec2 v)return V(v);
        if(value is Enum)return new JValue(Convert.ToInt32(value));
        if(value is string||value.GetType().IsPrimitive)return JToken.FromObject(value);
        if(value is UnityEngine.Object obj)return obj==null?JValue.CreateNull():new JObject{["type"]=obj.GetType().Name,["name"]=obj.name};
        if(value is Delegate callback)return new JArray(callback.GetInvocationList().Select(d=>d.Method.Name));
        if(value is IEnumerable items){var array=new JArray();foreach(var item in items)array.Add(Value(item));return array;}
        // Do not traverse arbitrary scene references or cyclic ownership graphs.
        return new JObject{["type"]=value.GetType().Name};
    }
    private static JObject Component(Component component)
    {
        var fields=new JObject();
        for(var type=component.GetType();type!=typeof(MonoBehaviour)&&type!=typeof(Component)&&type!=typeof(UnityEngine.Object);type=type.BaseType)
            foreach(var field in type.GetFields(Fields).OrderBy(f=>f.Name))fields[type.Name+"."+field.Name]=Value(field.GetValue(component));
        var result=new JObject{["type"]=component.GetType().Name,["fields"]=fields};
        if(component is Behaviour behavior)result["enabled"]=behavior.enabled;
        return result;
    }
    internal static JToken Evaluate(JObject query)
    {
        var saved=typeof(Updater).GetFields(Statics).Where(f=>!f.IsLiteral&&!f.IsInitOnly).ToDictionary(f=>f,f=>f.GetValue(null));
        var oldPlayers=PlayerHandler.Get().PlayerList();var oldSound=ReferencePlayerController.SoundCapture;
        var born=new List<SlimeController>();GameObject ownerObject=null;string stage="source";
        object Field(string name)=>typeof(Updater).GetField(name,Statics).GetValue(null);
        void Set(string name,object value)=>typeof(Updater).GetField(name,Statics).SetValue(null,value);
        try
        {
            var source=Resources.FindObjectsOfTypeAll<PlayerCollision>().First(p=>p.reviveEffectPrefab!=null&&p.reviveEffectPrefab.emptyPlayerPrefab!=null);
            stage="players";
            var players=new List<Player>();
            foreach(var p in query["players"])players.Add(new Player((int)p["id"],0){playersAndClonesStillAlive=(int)p["bodies"],ExtraLife=(ReviveStatus)(int)p["revival"],Color=source.reviveEffectPrefab.emptyPlayerPrefab.GetComponentInChildren<SpriteRenderer>(true).sharedMaterial});
            PlayerHandler.Get().SetPlayerList(players);
            stage="scheduler";
            Set("addedUpdatables",(int?)query["added"]??0);Set("prefabsAdded",(int?)query["prefabs_added"]??0);Set("registeringAPrefab",false);
            Set("toBeAddedUpdatables",new List<MonoUpdatable>());Set("toBeAddedPrefab",new List<MonoUpdatable>());Updater.updatables=new List<MonoUpdatable>();
            ownerObject=new GameObject("clone birth caller");ownerObject.SetActive(false);
            stage="caller";
            ownerObject.AddComponent<FixTransform>();ownerObject.AddComponent<PlayerBody>();ownerObject.AddComponent<DPhysicsBox>();
            var owner=ownerObject.AddComponent<PlayerCollision>();owner.reviveEffectPrefab=source.reviveEffectPrefab;
            var sounds=new JArray();ReferencePlayerController.SoundCapture=name=>sounds.Add(name);
            var results=new JArray();
            foreach(var action in query["actions"])
            {
                stage="birth";
                var player=players[(int)action["player"]];SlimeController clone=null;string error=null;
                owner.reviveEffectPrefab=(bool?)action["null_prefab"]==true?null:source.reviveEffectPrefab;
                try{clone=owner.SpawnAbilityLessClone(player,null,V(action["position"]));}
                catch(NullReferenceException){error="null_reference";}
                if(clone!=null)born.Add(clone);
                stage="snapshot";
                JToken Snapshot(SlimeController s)
                {
                    var box=s.GetComponent<DPhysicsBox>();var renderer=s.GetPlayerSprite();
                    return new JObject{["player"]=s.playerNumber,["active"]=s.gameObject.activeInHierarchy,["position"]=V(s.GetComponent<FixTransform>().position),["scale"]=s.GetComponent<FixTransform>().Scale.m_rawValue,
                        ["material_matches"]=renderer.sharedMaterial==players.Single(p=>p.Id==s.playerNumber).Color,["sprite_null"]=renderer.sprite==null,["renderer_enabled"]=renderer.enabled,
                        ["physics_registered"]=ReferenceBeamTargets.Get<Dictionary<int,int>>(DetPhysics.Get().boxes,"instanceIdToColliderIndex").ContainsKey(box.pp.instanceId),
                        ["components"]=new JArray(s.GetComponentsInChildren<MonoBehaviour>(true).Where(c=>c is MonoUpdatable||c is FixTransform).Select(Component))};
                }
                JArray Registered(string field)=>new JArray(((List<MonoUpdatable>)Field(field)).Select(c=>new JObject{["clone"]=born.FindIndex(s=>c.transform==s.transform||c.transform.IsChildOf(s.transform)),["type"]=c.GetType().Name,["hierarchy"]=c.HierarchyNumber}));
                results.Add(new JObject{["returned"]=clone!=null,["error"]=error==null?JValue.CreateNull():new JValue(error),["players"]=new JArray(players.Select(p=>new JObject{["id"]=p.Id,["bodies"]=p.playersAndClonesStillAlive,["revival"]=(int)p.ExtraLife})),
                    ["clones"]=new JArray(born.Select(Snapshot)),["added"]=(int)Field("addedUpdatables"),["prefabs_added"]=(int)Field("prefabsAdded"),["registering_prefab"]=(bool)Field("registeringAPrefab"),["pending"]=Registered("toBeAddedUpdatables"),["prefab_pending"]=Registered("toBeAddedPrefab"),["sounds"]=sounds.DeepClone()});
            }
            return results;
        }
        catch(Exception error){throw new InvalidOperationException("Clone birth capture failed at "+stage,error);}
        finally
        {
            foreach(var clone in born)if(clone!=null)UnityEngine.Object.DestroyImmediate(clone.gameObject);
            if(ownerObject!=null)UnityEngine.Object.DestroyImmediate(ownerObject);
            PlayerHandler.Get().SetPlayerList(oldPlayers);ReferencePlayerController.SoundCapture=oldSound;
            foreach(var entry in saved)entry.Key.SetValue(null,entry.Value);
        }
    }
}
