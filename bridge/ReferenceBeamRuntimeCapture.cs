using System.Collections.Generic;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Functional beam state at the replay physics boundary. Target membership is
// held across one stage, including bodies removed during that stage.
internal static class ReferenceBeamRuntimeCapture
{
    private static readonly List<FixTransform> targets = new List<FixTransform>();
    private static JArray previousTargets;
    private static int Id(GameObject go) => ReferencePhysicsCapture.Id(go.GetInstanceID());
    private static JObject Beam(DetPhysics.BeamBody b) => new JObject {
        ["id"]=b.id,["owner_id"]=b.ownerId,["position"]=V(b.position),["direction"]=V(b.direction),
        ["time_passed"]=b.timePassed.m_rawValue,["scale"]=b.scale.m_rawValue,["distance"]=b.distance.m_rawValue,
        ["colors"]=ReferenceBeam.ColorsJson(b.colors),["ground"]=b.ground==null?JValue.CreateNull():new JValue(Id(b.ground.gameObject))
    };
    private static string Kind(int layer) => layer==DetPhysics.wallLayer?"wall":layer==DetPhysics.terrainLayer?"terrain":
        layer==DetPhysics.lethalterrainLayer?"lethal":layer==DetPhysics.playerLayer?"player":layer==DetPhysics.affectorLayer?"affector":layer==DetPhysics.orbLayer?"orb":"other";
    private static void AddTargets<T>(PhysicsBodyList<T> list) where T:struct,IShape
    {
        for(int i=0;i<list.Length;i++) {
            var ft=list.physicsParents[i].fixTrans;
            if(ft!=null && Kind(ft.gameObject.layer)!="other" && !targets.Contains(ft)) targets.Add(ft);
        }
    }
    private static JArray Targets(DetPhysics p, bool before)
    {
        if(before) { targets.Clear(); AddTargets(p.circles); AddTargets(p.boxes); AddTargets(p.roundedRects); previousTargets=null; }
        var result=new JArray();
        for(int i=0;i<targets.Count;i++) {
            var ft=targets[i]; var go=ft.gameObject; string kind=Kind(go.layer);
            var collider=go.GetComponent<IPhysicsCollider>(); var body=go.GetComponent<BoplBody>();
            var hole=go.GetComponent<BlackHole>(); var orb=go.GetComponent<Orb>();
            var component=kind=="player"?(object)body:kind=="affector"?hole:kind=="orb"?orb:collider;
            bool destroyed=component is IPhysicsCollider c?c.IsDestroyed:component is MonoUpdatable m&&m.IsDestroyed;
            var force=before?Vec2.zero:V(previousTargets[i]["state"]["force"]);
            if(collider!=null) {
                int id=collider.GetPhysicsParent().instanceId;
                bool present=collider.shape==Shape.Circle?p.circles.ContainsParent(id):collider.shape==Shape.Box?p.boxes.ContainsParent(id):p.roundedRects.ContainsParent(id);
                if(present) force=collider.GetBody().force;
            }
            result.Add(new JObject { ["id"]=Id(go),["state"]=new JObject {
                ["kind"]=kind,["destroyed"]=ft.IsDestroyed,["component"]=component!=null,["component_destroyed"]=destroyed,
                ["force"]=V(force),["velocity"]=V(hole==null?Vec2.zero:Get<Vec2>(hole,"velocity")),["mass"]=hole==null?0:Get<Fix>(hole,"mass").m_rawValue,
                ["orb"]=new JObject { ["kind"]=orb==null?0:(int)Get<Orb.OrbType>(orb,"orbType"),["charge"]=orb==null?0:Get<Fix>(orb,"beamCharge01").m_rawValue,
                    ["rate"]=orb==null?0:orb.beamChargeRate01.m_rawValue,["firing"]=orb!=null&&Get<bool>(orb,"isFiringBeam"),
                    ["colors"]=ReferenceBeam.ColorsJson(orb==null?default(DetPhysics.BeamColors):Get<DetPhysics.BeamColors>(orb,"beamColors")) }
            }});
        }
        previousTargets=result; return result;
    }
    internal static JObject Snapshot(DetPhysics p, bool before)
    {
        var gc=new JObject { ["maximum"]=p.maxBeamDistance.m_rawValue,["water"]=Constants.WATER_HEIGHT.m_rawValue,["space"]=Constants.leveltype==LevelType.space,
            ["mask"]=(uint)p.beamHitMask.value,["cross_offset"]=p.crossBeamOffset.m_rawValue,["environmental_player"]=1000 };
        var protectedPlayers=new JArray();for(int i=1;i<=6;i++) { var player=PlayerHandler.Get().GetPlayer(i); protectedPlayers.Add(player!=null&&player.isProtectedFromTimeStop); }
        var keys=new JArray(); foreach(var k in p.beamWidthAnim.keys) keys.Add(new JObject { ["time"]=k.time.m_rawValue,["value"]=k.value.m_rawValue,["in_tangent"]=k.inTangent.m_rawValue,["out_tangent"]=k.outTangent.m_rawValue });
        var animation=Get<AnimationCurve>(p.beamWidthAnim,"Anim");var prefab=p.beamObjectPrefab;
        var config=new JObject { ["graph"]=gc.DeepClone(),["width"]=p.beamWidth.m_rawValue,["parent_offset"]=p.beamParentOffset.m_rawValue,["layer"]=prefab.gameObject.layer,
            ["hitbox"]=new JObject { ["base_width"]=prefab.baseWidth01.m_rawValue,["extra_length"]=prefab.extraLength.m_rawValue,["minimum_extents"]=prefab.GetComponent<DPhysicsBox>().minimumExtents.m_rawValue },
            ["width_curve"]=new JObject { ["keys"]=keys,["pre_wrap"]=(int)animation.preWrapMode,["post_wrap"]=(int)animation.postWrapMode },
            ["effects"]=new JObject { ["water"]=Constants.WATER_HEIGHT.m_rawValue,["time_scale"]=GameTime.PlayerTimeScale.m_rawValue,["protected"]=protectedPlayers,
                ["platform_push"]=p.beamPlatformPushForce.m_rawValue,["black_hole_push"]=p.beamBlackHolePushForce.m_rawValue,["body_push"]=p.beamRigidBodyPushFOrce.m_rawValue } };
        var bodies=new JArray();var roles=new JArray();var parents=new JArray();var reflected=new JArray();var was=new JArray();var lengths=new JArray();var hits=new JArray();
        var pending=new JArray();int queued=Get<int>(p,"beamsThisFrame");
        for(int i=0;i<p.beamBodies.Length;i++) {
            bodies.Add(Beam(p.beamBodies[i]));if(i<queued)pending.Add(Beam(p.beamBodies[i]));
            roles.Add(p.beamRoles[i].ToString());parents.Add(Get<short[]>(p,"beamParentInd")[i]);reflected.Add(Get<bool[]>(p,"beamIsReflected")[i]);was.Add(Get<bool[]>(p,"beamWasReflected")[i]);lengths.Add(Get<Fix[]>(p,"beamReflectionDistance")[i].m_rawValue);
            var ft=Get<FixTransform[]>(p,"beamRaycastHitRRs")[i];hits.Add(ft==null?JValue.CreateNull():new JValue(Id(ft.gameObject)));
        }
        var previous=new JArray();for(int i=0;i<Get<int>(p,"crossingsLastFrameCount");i++) { var c=Get<DetPhysics.BeamCrossingRecord[]>(p,"crossingsLastFrame")[i];previous.Add(new JObject { ["a"]=c.idA,["b"]=c.idB,["age"]=c.ageTicks }); }
        var mapping=new JArray();var map=Get<HashMap_short>(p,"lineIdtoInd");for(int i=0;i<map.entryCount;i++)mapping.Add(new JArray(map.keys[i],map.values[i]));
        var graph=new JObject { ["config"]=gc,["bodies"]=bodies,["roles"]=roles,["parents"]=parents,["reflected"]=reflected,["was_reflected"]=was,["reflection_distances"]=lengths,["hits"]=hits,
            ["crossing"]=new JArray(),["previous"]=previous,["mapping"]=mapping,["count"]=0,["initial"]=Get<int>(p,"initialBeamCount"),["previous_count"]=Get<int>(p,"totalBeamsLastFrame") };
        var pool=new JArray();var slots=Get<BeamObject[]>(p,"beamObjectPool");int last=slots.Length-1;while(last>=0&&slots[last]==null)last--;
        for(int i=0;i<=last;i++) { var b=slots[i];if(b==null){pool.Add(JValue.CreateNull());continue;}var slot=new JObject { ["id"]=Id(b.gameObject),["hitbox"]=ReferenceBeamHitbox.Snapshot(b) };if(b.IsDestroyed)slot["destroyed"]=true;pool.Add(slot); }
        var targetStates=Targets(p,before);
        return new JObject { ["config"]=config,["graph"]=graph,["pool"]=pool,["targets"]=targetStates,["pending"]=pending,["next_id"]=ReferencePhysicsCapture.NextId };
    }
}
