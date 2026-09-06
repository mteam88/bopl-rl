using System;
using System.Collections.Generic;
using System.Linq;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Observes the actual Revival marker prefab and its fresh DPhysicsCircle registration.
internal sealed class ReferenceRevivalMarker
{
    internal readonly RevivePositionIndicator Indicator;
    internal readonly BoplBody BodyComponent;
    internal readonly DPhysicsCircle CircleComponent;
    internal readonly FixTransform Transform;
    internal readonly ReferencePhysicsScene.Adapter Adapter;
    private readonly JObject template;
    internal static GameObject Prefab(JToken config)
    {
        var source=Resources.FindObjectsOfTypeAll<QuickSpell>().FirstOrDefault(q=>q.name=="Revival"&&q.SpellPrefab!=null)?.SpellPrefab;
        if(source==null||source.GetComponent<RevivePositionIndicator>()==null)throw new InvalidOperationException("Actual Revival marker prefab unavailable");
        var prefab=UnityEngine.Object.Instantiate(source);prefab.name="reference Revival marker prefab";
        var body=prefab.GetComponent<BoplBody>();
        ReferenceBoplBody.Bind(body,new JObject{["config"]=config["body"].DeepClone(),["resizable"]=false,["prev_scale"]=Fix.One.m_rawValue,["frames_until_drop"]=-1,["original_sorting"]=new JObject{["layer"]=JValue.CreateNull(),["order"]=0}});
        var circle=prefab.GetComponent<DPhysicsCircle>();var c=config["circle"];
        Set(circle,"scale",F(c["scale"]));Set(circle,"startRadius",F(c["start_radius"]));circle.initHasBeenCalled=(bool)c["initialized"];circle.manuallyCallInit=(bool)c["manually_call_init"];circle.MinScale=F(c["min_scale"]);circle.MaxScale=F(c["max_scale"]);
        prefab.layer=(int)config["layer"];prefab.GetComponent<FixTransform>().offset=V(config["offset"]);
        if(!(bool)config["particle"])UnityEngine.Object.DestroyImmediate(prefab.GetComponent<ParticleSystem>());
        return prefab;
    }
    internal ReferenceRevivalMarker(GameObject go,List<ReferencePhysicsScene.Adapter> adapters)
    {
        Indicator=go.GetComponent<RevivePositionIndicator>();BodyComponent=go.GetComponent<BoplBody>();CircleComponent=go.GetComponent<DPhysicsCircle>();Transform=go.GetComponent<FixTransform>();
        int id=adapters.Max(a=>a.sourceId)+1;
        template=new JObject{["id"]=id,["hierarchy"]=3,["collider"]=new JObject{["kind"]="circle",["shape"]=new JObject{["center"]=V(Transform.position+Transform.offset),["radius"]=CircleComponent.GetStartRadius().m_rawValue}},["body"]=Body(default(PhysicsBody)),
            ["transform"]=new JObject{["position"]=V(Transform.position),["rotation"]=Transform.rotation.m_rawValue,["offset"]=V(Transform.offset)},["layer"]=go.layer,["active"]=go.activeInHierarchy,["callback"]=true,["callback_destroyed"]=false,["destroyed"]=false,["composite"]=JValue.CreateNull(),["relative_pos"]=V(Vec2.zero),["relative_orientation"]=0};
        Adapter=new ReferencePhysicsScene.Adapter{sourceId=id,shape=Shape.Circle,transform=Transform,parent=CircleComponent.pp,realCollider=CircleComponent,template=template};adapters.Add(Adapter);
    }
    internal JObject Snapshot()
    {
        var c=CircleComponent;
        JToken pending=JValue.CreateNull();if(!c.initHasBeenCalled){var p=(JObject)template.DeepClone();p["transform"]["position"]=V(Transform.position);p["transform"]["offset"]=V(Transform.offset);p["destroyed"]=c.IsDestroyed;pending=p;}
        return new JObject{["owner"]=Indicator.GetPlayerId(),["object"]=Adapter.sourceId,["body"]=ReferenceBoplBody.State(BodyComponent,new Dictionary<int,MagnetGun>()),["circle"]=new JObject{["circle"]=new JObject{["scale"]=c.Scale.m_rawValue,["min_scale"]=c.MinScale.m_rawValue,["max_scale"]=c.MaxScale.m_rawValue,["start_radius"]=c.GetStartRadius().m_rawValue,["initialized"]=c.initHasBeenCalled,["manually_call_init"]=c.manuallyCallInit}},["scale"]=Transform.Scale.m_rawValue,["pending_object"]=pending};
    }
}
