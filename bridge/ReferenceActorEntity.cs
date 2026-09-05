using System;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;
using static ReferenceDynamics;

// Real collider/component ownership for ability activation and handoff tests.
// Awake runs once, before the supplied simulation state is bound.
internal static class ReferenceActorEntity
{
    internal static GameObject Create(JToken state,ReferencePhysicsScene.Adapter adapter)
    {
        var go=adapter.transform.gameObject;go.SetActive(false);
        go.AddComponent<Hitbox>();var body=go.AddComponent<PlayerBody>();var physics=go.AddComponent<PlayerPhysics>();
        var box=go.AddComponent<DPhysicsBox>();var slime=go.AddComponent<SlimeController>();
        slime.AbilityReadyIndicators=Array.Empty<AbilityReadyIndicator>();
        go.SetActive(true);
        Set(physics,"OnAttachedToGround",null);Set(physics,"OnNoLongerGrounded",null);
        box.enabled=adapter.enabled;var parent=adapter.parent;parent.monobehaviourCollider=box;adapter.parent=parent;box.pp=parent;box.initHasBeenCalled=(bool)state["box"]["initialized"];
        Set(box,"scale",F(state["box"]["scale"]));Set(box,"startExtents",V(state["box"]["start_extents"]));
        box.MinScale=F(state["box"]["min_scale"]);box.MaxScale=F(state["box"]["max_scale"]);box.minimumExtents=F(state["box"]["minimum_extents"]);
        body.simulated=(bool)state["body_simulated"];
        go.SetActive((bool)state["active"]);
        return go;
    }
    internal static JObject Snapshot(JToken initial,PlayerBody body)
    {
        var value=(JObject)initial.DeepClone();var box=body.GetComponent<DPhysicsBox>();
        value["active"]=body.gameObject.activeSelf;value["body_simulated"]=body.simulated;
        value["box"]["scale"]=box.Scale.m_rawValue;value["box"]["start_extents"]=V(box.GetStartExtents());
        value["box"]["initialized"]=box.initHasBeenCalled;
        return value;
    }
}
