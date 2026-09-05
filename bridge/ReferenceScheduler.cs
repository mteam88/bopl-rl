using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

internal static class ReferenceScheduler
{
    internal static Action<string,object> Capture;
    private static readonly BindingFlags statics=BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic;
    private static FieldInfo Field(string name)=>typeof(Updater).GetField(name,statics);
    internal static JToken Evaluate(JObject query)
    {
        var saved=typeof(Updater).GetFields(statics).Where(f=>!f.IsLiteral&&!f.IsInitOnly).ToDictionary(f=>f,f=>f.GetValue(null));
        var oldPlayers=PlayerHandler.Get().PlayerList();var oldScale=GameTime.PlayerTimeScale;var oldCapture=Capture;var oldRecord=Host.recordReplay;
        var nodes=new List<GameObject>();var components=new List<MonoBehaviour>();
        try
        {
            var state=(JObject)query["state"].DeepClone();var metadata=(JArray)state["components"];var nodeData=(JArray)state["nodes"];
            foreach(var n in nodeData){var go=new GameObject("scheduled node");go.SetActive(false);nodes.Add(go);}
            for(int i=0;i<nodes.Count;i++)if(nodeData[i]["parent"].Type!=JTokenType.Null)nodes[i].transform.SetParent(nodes[(int)nodeData[i]["parent"]].transform);
            for(int i=0;i<nodes.Count;i++)foreach(var child in nodeData[i]["children"])nodes[(int)child].transform.SetAsLastSibling();
            for(int i=0;i<metadata.Count;i++)
            {
                var c=metadata[i];var go=nodes[(int)c["owner"]];MonoBehaviour component;
                switch((string)c["kind"])
                {
                    case "updatable":
                        var u=go.AddComponent<ReferenceScheduledComponent>();u.CanonicalId=i;u.EnabledFlag=(bool)c["enabled"];u.IgnoreActivity=(bool)c["ignore_activity"];
                        u.HierarchyNumber=(int)c["hierarchy"];u.IsDestroyed=(bool)c["destroyed"];component=u;break;
                    case "fix":var ft=go.AddComponent<FixTransform>();ft.IsDestroyed=(bool)c["destroyed"];component=ft;break;
                    case "plain":component=go.AddComponent<ReferenceControllerAbility>();break;
                    default:throw new ArgumentException("Unknown scheduled component");
                }
                components.Add(component);
            }
            for(int i=0;i<nodes.Count;i++)nodes[i].SetActive((bool)nodeData[i]["active"]);
            var objectIds=nodes.Select((o,i)=>(o,i)).ToDictionary(x=>(UnityEngine.Object)x.o,x=>x.i);
            var componentIds=components.Select((o,i)=>(o,i)).ToDictionary(x=>(UnityEngine.Object)x.o,x=>x.i);
            for(int i=0;i<components.Count;i++)if(!(bool)metadata[i]["present"])UnityEngine.Object.DestroyImmediate(components[i]);
            for(int i=0;i<nodes.Count;i++)if(!(bool)nodeData[i]["present"])UnityEngine.Object.DestroyImmediate(nodes[i]);
            JToken Ref(UnityEngine.Object o,bool gameObject=false)=>ReferenceEquals(o,null)?JValue.CreateNull():new JValue((gameObject?objectIds:componentIds)[o]);
            MonoUpdatable Up(JToken id)=>id.Type==JTokenType.Null?null:(MonoUpdatable)components[(int)id];
            MonoBehaviour Mono(JToken id)=>id.Type==JTokenType.Null?null:components[(int)id];
            GameObject Node(JToken id)=>id.Type==JTokenType.Null?null:nodes[(int)id];
            List<MonoUpdatable> List(JToken ids)=>ids.Select(Up).ToList();
            JArray Refs(IEnumerable<MonoUpdatable> list)=>new JArray(list.Select(c=>Ref(c)));
            var pending=new List<MonoUpdatable>(state["pending"]["items"].Count());
            foreach(var id in state["pending"]["items"].Take((int)state["pending"]["length"]))pending.Add(Up(id));
            var initialStorage=(MonoUpdatable[])typeof(List<MonoUpdatable>).GetField("_items",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(pending);
            for(int i=0;i<initialStorage.Length;i++)initialStorage[i]=Up(state["pending"]["items"][i]);
            Field("toBeAddedUpdatables").SetValue(null,pending);Updater.updatables=List(state["live"]);
            Field("toBeAddedPrefab").SetValue(null,List(state["prefab"]));Field("toDestroyBuffer").SetValue(null,List(state["destroy_buffer"]));
            Field("destroyedObjects").SetValue(null,new Queue<GameObject>(state["destroyed_objects"].Select(Node)));
            Field("destroyedMonos").SetValue(null,new Queue<MonoBehaviour>(state["destroyed_monos"].Select(Mono)));
            string[,] fields={{"added","addedUpdatables"},{"prefabs_added","prefabsAdded"},{"registering_prefab","registeringAPrefab"},{"loaded","LoadedALevelThisUpdate"},{"stopped","gameHasStopped"},{"ticks","SimulationTicks"},{"hitstop","hitStopFrames"}};
            for(int i=0;i<fields.GetLength(0);i++){var f=Field(fields[i,1]);f.SetValue(null,state[fields[i,0]].ToObject(f.FieldType));}
            Updater.SimTimePassed=F(state["time"]);Updater.SimTimeSinceLevelLoaded=F(state["level_time"]);GameTime.PlayerTimeScale=F(state["time_scale"]);Host.recordReplay=false;
            var inputs=state["inputs"].Select(p=>new Player.InputProfile{jumpButton_IS_DOWN=(bool)p["jump"],jumpButton_DOWN_THIS_FRAME=(bool)p["pressed"],jumpButton_DOWN_LAST_FRAME=(bool)p["previous"]}).ToList();
            var players=state["players"].Select(p=>{
                var player=new Player((int)p["id"],0){isProtectedFromTimeStop=(bool)p["protected"],isInvisible=(bool)p["invisible"],timeSpentInvisible=F(p["time_invisible"]),timeSinceEnteredAbility=F(p["time_ability"])};
                Set(player,"Inputs",p["input"].Type==JTokenType.Null?null:inputs[(int)p["input"]]);return player;
            }).ToList();PlayerHandler.Get().SetPlayerList(players);
            var calls=new Dictionary<string,int>();JArray events=null;
            void Notify(string kind,int? id)
            {
                events.Add(new JObject{["kind"]=kind,["id"]=id.HasValue?new JValue(id.Value):JValue.CreateNull()});
                string key=kind+":"+(id.HasValue?id.Value.ToString():"*");int index=calls.TryGetValue(key,out int previous)?previous:0;calls[key]=index+1;
                var script=query["scripts"][key];if(script!=null&&index<script.Count())foreach(var command in script[index])Command(command);
            }
            foreach(var u in components.OfType<ReferenceScheduledComponent>())u.Notify=(kind,id)=>Notify(kind,id);
            Capture=(kind,target)=>{
                int? id=target==null?null:target is int integer?integer:kind=="destroy_object"?objectIds[(UnityEngine.Object)target]:componentIds[(UnityEngine.Object)target];
                Notify(kind,id);
            };
            void Command(JToken a)
            {
                switch((string)a["op"])
                {
                    case "tick":typeof(Updater).GetMethod("TickSimulationInner",statics).Invoke(null,new object[]{F(a["dt"])});break;
                    case "register":Updater.RegisterUpdatable(Up(a["id"]));break;
                    case "begin":Updater.BeginRegisterPrefab();break;
                    case "end":Updater.EndRegisterPrefab();break;
                    case "reinit":Updater.ReInit();break;
                    case "pre_level":Updater.PreLevelLoad();break;
                    case "post_level":Updater.PostLevelLoad();break;
                    case "destroy_object":Updater.DestroyFix(Node(a["id"]));break;
                    case "destroy_mono":Updater.DestroyFix(Mono(a["id"]));break;
                    case "sort":((List<MonoUpdatable>)Field("toBeAddedUpdatables").GetValue(null)).Sort(Updater.updateComparer);break;
                    case "hitstop":Updater.HitStop((int)a["value"]);break;
                    case "scale":GameTime.PlayerTimeScale=F(a["value"]);break;
                    case "enable":((ReferenceScheduledComponent)Mono(a["id"])).EnabledFlag=(bool)a["value"];break;
                    case "mark":Up(a["id"]).IsDestroyed=(bool)a["value"];break;
                    case "hierarchy":Up(a["id"]).HierarchyNumber=(int)a["value"];break;
                    case "active":Node(a["id"]).SetActive((bool)a["value"]);nodeData[(int)a["id"]]["active"]=(bool)a["value"];break;
                    case "live_add":Updater.updatables.Add(Up(a["id"]));break;
                    case "live_remove":Updater.updatables.RemoveAt((int)a["index"]);break;
                    case "live_clear":Updater.updatables.Clear();break;
                    case "input":inputs[(int)a["id"]].jumpButton_IS_DOWN=(bool)a["value"];break;
                    default:throw new ArgumentException("Unknown scheduler command");
                }
            }
            JObject Snapshot()
            {
                var s=(JObject)state.DeepClone();var cs=new JArray();
                for(int i=0;i<components.Count;i++)
                {
                    var c=(JObject)metadata[i].DeepClone();var mb=components[i];c["present"]=mb!=null;
                    if(mb is ReferenceScheduledComponent u){c["destroyed"]=u.IsDestroyed;c["hierarchy"]=u.HierarchyNumber;c["enabled"]=u.EnabledFlag;}
                    if(mb is FixTransform ft)c["destroyed"]=ft.IsDestroyed;cs.Add(c);
                }
                s["components"]=cs;var ns=new JArray();for(int i=0;i<nodes.Count;i++){var n=(JObject)nodeData[i].DeepClone();n["present"]=nodes[i]!=null;ns.Add(n);}s["nodes"]=ns;
                for(int i=0;i<fields.GetLength(0);i++)s[fields[i,0]]=JToken.FromObject(Field(fields[i,1]).GetValue(null));
                s["time"]=Updater.SimTimePassed.m_rawValue;s["level_time"]=Updater.SimTimeSinceLevelLoaded.m_rawValue;s["time_scale"]=GameTime.PlayerTimeScale.m_rawValue;
                var p=(List<MonoUpdatable>)Field("toBeAddedUpdatables").GetValue(null);var storage=(MonoUpdatable[])typeof(List<MonoUpdatable>).GetField("_items",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(p);
                s["pending"]=new JObject{["items"]=Refs(storage),["length"]=p.Count};s["live"]=Refs(Updater.updatables);s["prefab"]=Refs((List<MonoUpdatable>)Field("toBeAddedPrefab").GetValue(null));s["destroy_buffer"]=Refs((List<MonoUpdatable>)Field("toDestroyBuffer").GetValue(null));
                s["destroyed_objects"]=new JArray(((Queue<GameObject>)Field("destroyedObjects").GetValue(null)).Select(o=>Ref(o,true)));s["destroyed_monos"]=new JArray(((Queue<MonoBehaviour>)Field("destroyedMonos").GetValue(null)).Select(o=>Ref(o)));
                s["inputs"]=new JArray(inputs.Select(pf=>new JObject{["jump"]=pf.jumpButton_IS_DOWN,["pressed"]=pf.jumpButton_DOWN_THIS_FRAME,["previous"]=pf.jumpButton_DOWN_LAST_FRAME}));
                var ps=new JArray();for(int i=0;i<players.Count;i++){var pl=players[i];var value=(JObject)state["players"][i].DeepClone();value["protected"]=pl.isProtectedFromTimeStop;value["invisible"]=pl.isInvisible;value["time_invisible"]=pl.timeSpentInvisible.m_rawValue;value["time_ability"]=pl.timeSinceEnteredAbility.m_rawValue;ps.Add(value);}s["players"]=ps;
                return new JObject{["state"]=s,["events"]=events,["calls"]=JObject.FromObject(calls)};
            }
            var frames=new JArray();foreach(var action in query["actions"])
            {
                events=new JArray();string error=null;
                try{Command(action);}
                catch(Exception e)
                {
                    while(e is TargetInvocationException)e=e.InnerException;
                    if(e is NullReferenceException)error="null_reference";else if(e is MissingReferenceException)error="missing_reference";
                    else if(e is ArgumentOutOfRangeException)error="argument_out_of_range";else if(e is InvalidOperationException)error="compare_failed";
                    else if(e is ArgumentException)error="bad_comparer";else throw;
                }
                if(error!=null)return new JObject{["error"]=error,["action"]=frames.Count,["completed"]=frames,["failed"]=Snapshot()};
                frames.Add(Snapshot());
            }
            return frames;
        }
        finally
        {
            Capture=null;foreach(var go in nodes)if(go!=null)UnityEngine.Object.DestroyImmediate(go);
            foreach(var pair in saved)pair.Key.SetValue(null,pair.Value);
            PlayerHandler.Get().SetPlayerList(oldPlayers);GameTime.PlayerTimeScale=oldScale;Host.recordReplay=oldRecord;Capture=oldCapture;
        }
    }
}
