using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceBeamTargets;

internal static class ReferenceActorBounds
{
    internal static void Bind(JToken value,PlayerBody body,PlayerPhysics physics,IPlayerIdHolder holder,ReferenceDeathEffects effects)
    {
        // Real player prefabs use SlimeController as their holder. Remove the
        // harness-only hitbox before testing the original GetComponent lookup.
        var hitbox=body.GetComponent<Hitbox>();if(hitbox!=null)Object.DestroyImmediate(hitbox);
        Set(body,"idHolder",holder);Set(physics,"playerIdHolder",holder);
        var collision=body.GetComponent<PlayerCollision>();if(collision!=null)Set(collision,"playerIdHolder",holder);
        // Create primes Awake before the supplied scheduler queues are bound,
        // including for ability GameObjects that begin the episode inactive.
        var bounds=body.GetComponent<DestroyIfOutsideSceneBounds>();
        Set(bounds,"fixTrans",(bool)value["transform_present"]?body.fixtrans:null);
        bounds.DestroyIfTooFarX=(bool)value["x"];bounds.DestroyIfTooFarY=(bool)value["y"];bounds.DoNotDestroyUp=(bool)value["no_up"];
        GameObject Prefab(string name)=>value[name].Type==JTokenType.Null?null:effects.BoundPrefab(value[name]);
        bounds.XEffect=Prefab("x_effect");bounds.UpEffect=Prefab("up_effect");bounds.DownEffect=Prefab("down_effect");
        bounds.UsePlayerMaterialForXEffect=(bool)value["use_x_material"];bounds.UsePlayerMaterialForUpEffect=(bool)value["use_up_material"];bounds.UsePlayerMaterialForDownEffect=(bool)value["use_down_material"];
    }
}
