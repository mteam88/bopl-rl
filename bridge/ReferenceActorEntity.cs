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
    internal static GameObject Create(JToken state,ReferencePhysicsScene.Adapter adapter,JToken ordinary=null,JToken collision=null,JToken bounds=null)
    {
        var go=adapter.transform.gameObject;go.SetActive(false);
        if(ordinary!=null)go.AddComponent<Ability>();else go.AddComponent<Hitbox>();
        var body=go.AddComponent<PlayerBody>();var physics=go.AddComponent<PlayerPhysics>();IPhysicsCollider collider;
        if(state["circle"]!=null)collider=go.AddComponent<DPhysicsCircle>();else collider=go.AddComponent<DPhysicsBox>();
        if(ordinary==null){var slime=go.AddComponent<SlimeController>();slime.AbilityReadyIndicators=Array.Empty<AbilityReadyIndicator>();}
        if(ordinary?["dash"]!=null)ReferenceDashAbility.Create(go);
        if(ordinary?["quick_spell"]!=null)ReferenceQuickSpell.Create(go);
        if(ordinary?["quick_spell"]?["revival"]!=null)go.AddComponent<Revive>();
        if(collision!=null)go.AddComponent<PlayerCollision>();
        if(bounds!=null)go.AddComponent<DestroyIfOutsideSceneBounds>();
        go.SetActive(true);
        if(collision!=null)ReferenceActorCombat.Bind(go.GetComponent<PlayerCollision>(),collision);
        Set(physics,"OnAttachedToGround",null);Set(physics,"OnNoLongerGrounded",null);
        collider.enabled=adapter.enabled;var parent=adapter.parent;parent.monobehaviourCollider=collider;adapter.parent=parent;
        if(collider is DPhysicsBox box)
        {
            box.pp=parent;box.initHasBeenCalled=(bool)state["box"]["initialized"];
            Set(box,"scale",F(state["box"]["scale"]));Set(box,"startExtents",V(state["box"]["start_extents"]));
            box.MinScale=F(state["box"]["min_scale"]);box.MaxScale=F(state["box"]["max_scale"]);box.minimumExtents=F(state["box"]["minimum_extents"]);
        }
        else
        {
            var circle=(DPhysicsCircle)collider;var c=state["circle"];circle.pp=parent;circle.initHasBeenCalled=(bool)c["initialized"];
            Set(circle,"scale",F(c["scale"]));Set(circle,"startRadius",F(c["start_radius"]));
            circle.MinScale=F(c["min_scale"]);circle.MaxScale=F(c["max_scale"]);circle.manuallyCallInit=(bool)c["manually_call_init"];
        }
        body.simulated=(bool)state["body_simulated"];
        go.SetActive((bool)state["active"]);
        return go;
    }
    internal static JObject Snapshot(JToken initial,PlayerBody body)
    {
        var value=(JObject)initial.DeepClone();
        if(initial["present"]!=null)value["present"]=true;
        if(initial["destroyed"]!=null||body.IsDestroyed)value["destroyed"]=body.IsDestroyed;
        value["active"]=body.gameObject.activeSelf;value["body_simulated"]=body.simulated;
        if(initial["circle"]!=null)
        {
            var circle=body.GetComponent<DPhysicsCircle>();value["circle"]["scale"]=circle.Scale.m_rawValue;
            value["circle"]["start_radius"]=circle.GetStartRadius().m_rawValue;value["circle"]["initialized"]=circle.initHasBeenCalled;
        }
        else
        {
            var box=body.GetComponent<DPhysicsBox>();value["box"]["scale"]=box.Scale.m_rawValue;value["box"]["start_extents"]=V(box.GetStartExtents());
            value["box"]["initialized"]=box.initHasBeenCalled;
        }
        return value;
    }
}
