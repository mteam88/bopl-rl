using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static ReferenceDynamics;
using static ReferenceBeamTargets;

// Advances original components together through Updater.TickSimulationInner.
// Supplied transforms alias the collision scene. The motor is an explicit
// scripted Jump/Move driver or the original SlimeController when contexts exist.
internal static class ReferenceActors
{
    private const BindingFlags Statics=BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic;
    private static void Static(string name,object value)=>typeof(Updater).GetField(name,Statics).SetValue(null,value);
    internal static JToken Evaluate(JObject query,DetPhysics scene,List<ReferencePhysicsScene.Adapter> adapters,Func<JObject> sceneSnapshot,List<ReferenceAnimator.Context> animations=null)
    {
        var saved=typeof(Updater).GetFields(Statics).Where(f=>!f.IsLiteral&&!f.IsInitOnly).ToDictionary(f=>f,f=>f.GetValue(null));
        var oldPlayers=PlayerHandler.Get().PlayerList();var oldScale=GameTime.PlayerTimeScale;var oldSpace=Constants.leveltype;
        var oldTutorial=TutorialGameHandler.isInTutorial;
        var oldDeadzone=Constants.GROUNDALIGNMENTDEADZONE;var oldCapture=ReferenceScheduler.Capture;var oldSound=ReferencePlayerController.SoundCapture;var oldRecord=Host.recordReplay;var oldEntryCapture=ReferenceCapture.ActorAbilityCapture;
        var oldPlayerGravity=scene.playerGravity;var oldMaxStretch=scene.maxStretchLength;var oldRopeStrength=scene.groundedPlayerRopeStr;
        ReferenceActorCombat combat=null;ReferenceDeathEffects deathEffects=null;ReferenceActorClones clones=null;
        var ropes=scene.ropes.ropeBodies.ToList();var meshes=new List<RopeMesh>();
        var objects=new List<GameObject>();var assets=new List<UnityEngine.Object>();var slimes=new List<ReferenceSlimeActors.Context>();
        var achievedField=typeof(AchievementHandler).GetField("isAchieved",Statics);var oldAchieved=achievedField.GetValue(null);
        try
        {
            var runtime=query["runtime"];if(animations!=null)achievedField.SetValue(null,new bool[28]);if((int)runtime["player_layer"]!=LayerMask.NameToLayer("Player"))throw new ArgumentException("Incorrect player layer");var initial=runtime["scheduler"];var actorData=(JArray)runtime["actors"].DeepClone();
            var spawnActors=actorData.Where(a=>a["slime"]?["spawn"]!=null).ToList();
            if(spawnActors.Count>0){bool tutorial=(bool)spawnActors[0]["slime"]["spawn"]["tutorial"];if(spawnActors.Any(a=>(bool)a["slime"]["spawn"]["tutorial"]!=tutorial))throw new ArgumentException("Tutorial state must be shared");TutorialGameHandler.isInTutorial=tutorial;}
            if(runtime["actor_ropes"]!=null)
            {
                var binding=runtime["actor_ropes"];scene.playerGravity=F(binding["player_gravity"]);scene.maxStretchLength=F(binding["max_stretch"]);
                if(binding["metadata"].Count()!=ropes.Count)throw new ArgumentException("Actor rope metadata count");
                for(int i=0;i<ropes.Count;i++)
                {
                    var m=binding["metadata"][i];var r=ropes[i];r.ownerId=(byte)m["owner"];r.timeInAir_start=F(m["time_air_start"]);r.timeInAir_end=F(m["time_air_end"]);
                    r.ropeMesh=new RopeMesh();meshes.Add(r.ropeMesh);assets.Add(r.ropeMesh.mesh);
                }
            }
            if(runtime["combat"]?["hitboxes"]!=null)foreach(var value in runtime["combat"]["hitboxes"]){var adapter=adapters.Find(a=>a.sourceId==(int)value["object"]);var hitbox=adapter.transform.gameObject.AddComponent<Hitbox>();hitbox.SetPlayerId((int)value["owner"]);}
            var bound=ReferenceSurfaceMotion.BindPlatforms(runtime["bindings"],scene,adapters,objects);
            var platforms=runtime["bindings"].Select(b=>bound[(int)b["object"]]).ToList();
            for(int i=0;i<platforms.Count;i++)
            {
                var g=runtime["grounds"][i];var p=platforms[i];p.platformType=(bool)g["ice"]?PlatformType.ice:PlatformType.grass;
                Set(p,"dropPlayersForFrames",(int)g["drop_frames"]);Set(p,"dontDropPlayerId",(int)g["dont_drop"]);Set(p,"dropForce",F(g["drop_force"]));
            }
            var inputs=initial["inputs"].Select(p=>new Player.InputProfile{jumpButton_IS_DOWN=(bool)p["jump"],jumpButton_DOWN_THIS_FRAME=(bool)p["pressed"],jumpButton_DOWN_LAST_FRAME=(bool)p["previous"],joystickAngleByte=(byte?)p["angle"]??0}).ToList();
            var players=initial["players"].Select(p=>{var player=new Player((int)p["id"],0){isProtectedFromTimeStop=(bool)p["protected"],isInvisible=(bool)p["invisible"],timeSpentInvisible=F(p["time_invisible"]),timeSinceEnteredAbility=F(p["time_ability"])};Set(player,"Inputs",inputs[(int)p["input"]]);return player;}).ToList();
            for(int i=0;i<players.Count;i++)if(initial["players"][i]["flags"]!=null){var f=initial["players"][i]["flags"];players[i].InSparkShot=(bool)f["spark"];players[i].InMachoThrow=(bool)f["macho"];players[i].InPlatformTransform=(bool)f["platform"];}
            for(int i=0;i<players.Count;i++)if(initial["players"][i]["scale"]!=null)Set(players[i],"scale",F(initial["players"][i]["scale"]));
            for(int i=0;i<players.Count;i++)if(initial["players"][i]["eaten_count"]!=null)players[i].playersEatenCountThisRound=(int)initial["players"][i]["eaten_count"];
            PlayerHandler.Get().SetPlayerList(players);if(animations!=null)foreach(var player in players)player.IsLocalPlayer=false;Constants.leveltype=(bool)runtime["space"]?LevelType.space:LevelType.grass;
            if(runtime["combat"]?["death_effects"]!=null)deathEffects=new ReferenceDeathEffects(runtime["combat"]["death_effects"],initial);
            if(runtime["combat"]!=null)combat=new ReferenceActorCombat(runtime["combat"],players);
            var entityObjects=actorData.Select(a=>a["entity"]!=null?ReferenceActorEntity.Create(a["entity"],adapters.Find(v=>v.sourceId==(int)a["object"]),a["ordinary"],a["collision"],a["bounds"]):null).ToList();
            for(int i=0;i<entityObjects.Count;i++)if(entityObjects[i]!=null)
            {
                var parent=initial["nodes"][(int)actorData[i]["entity"]["node"]]["parent"];
                if(parent.Type!=JTokenType.Null)
                {
                    int owner=Enumerable.Range(0,actorData.Count).Single(j=>actorData[j]["entity"]!=null&&(int)actorData[j]["entity"]["node"]==(int)parent);
                    entityObjects[i].transform.SetParent(entityObjects[owner].transform,true);
                }
            }
            var bodies=new List<PlayerBody>();var physics=new List<PlayerPhysics>();var transforms=new List<FixTransform>();
            JArray events=null,phases=null,pushes=null,effects=null;bool simulated=false;int currentActor=0;
            void Event(int actor,string kind,JToken value)=>events.Add(new JObject{["actor"]=actor,["event"]=new JObject{["kind"]=kind,["value"]=value}});
            for(int i=0;i<actorData.Count;i++)
            {
                int actor=i;var a=actorData[i];var state=a["controller"];var original=state["physics"];var motion=original["motion"];var bs=a["body"];var config=a["config"];var c=config["physics"];
                var adapter=adapters.Find(v=>v.sourceId==(int)a["object"]);
                var go=a["entity"]!=null?entityObjects[i]:new GameObject("composed player components");if(a["entity"]==null)go.SetActive(false);else go.layer=(int)runtime["player_layer"];objects.Add(go);
                var ft=adapter.transform;transforms.Add(ft);
                ft.position=V(motion["position"]);Set(ft,"rotationInner",F(motion["rotation"]));Set(ft,"upInner",V(motion["up"]));Set(ft,"rightInner",V(motion["right"]));Set(ft,"scale",F(original["scale"]));
                IPlayerIdHolder holder=a["ordinary"]!=null?(IPlayerIdHolder)go.GetComponent<Ability>():(go.GetComponent<Hitbox>()??go.AddComponent<Hitbox>());holder.SetPlayerId((int)bs["holder"]);
                var body=go.GetComponent<PlayerBody>()??go.AddComponent<PlayerBody>();Set(body,"fixTransform",ft);Set(body,"idHolder",holder);bodies.Add(body);
                body.selfImposedVelocity=V(motion["velocity"]);body.externalVelocity=V(motion["external_velocity"]);
                body.prevPosition=V(bs["previous_position"]);Set(body,"prevPlayerVel",V(bs["previous_velocity"]));Set(body,"prevImposedVel",V(bs["previous_imposed"]));Set(body,"prevExternalVel",V(bs["previous_external"]));
                body.extraSwingForce=V(bs["extra_swing_force"]);body.topAttachment=(bool)bs["top"];body.ignoreGravityWhenRoped=(bool)bs["ignore_gravity"];body.RopePlayerMass01=F(bs["rope_mass"]);
                body.ropeBody=bs["rope"].Type==JTokenType.Null?null:ropes[(int)bs["rope"]];
                var p=go.GetComponent<PlayerPhysics>()??go.AddComponent<PlayerPhysics>();physics.Add(p);Set(p,"body",body);Set(p,"playerIdHolder",holder);Set(body,"physics",p);
                Set(p,"isGrounded",(bool)motion["grounded"]);Set(p,"attachedGround",motion["ground"].Type==JTokenType.Null?null:platforms[(int)motion["ground"]]);
                Set(p,"<LocalPlatformPos>k__BackingField",F(motion["local"]));p.radius=F(motion["radius"]);p.groundedSpeed=F(motion["grounded_speed"]);
                Set(p,"prevScal",F(original["previous_scale"]));p.jumpedThisFrame=(bool)original["jumped"];Set(p,"SyncRotationToPlatform",(bool)original["sync_rotation"]);
                p.cannotBeUngroundedByRopes=(bool)original["cannot_unground"];p.skipNextUpdate=(bool)state["skip_update"];Set(p,"ApplyGravity",(bool)state["apply_gravity"]);
                Set(p,"blockableTerrainHitBuffer",(bool)state["blocked"]?new RaycastHit2D[20]:null);
                p.gravity_accel=F(c["gravity"]);p.gravity_maxFallSpeed=F(c["max_fall_speed"]);p.gravity_modifier=F(c["gravity_modifier"]);p.airAccel=F(c["air_accel"]);p.Speed=F(c["max_speed"]);p.accel=F(config["accel"]);
                p.jumpStrength=F(c["jump_strength"]);p.jumpExtraXStrength=F(c["jump_extra_x"]);p.jumpKeptMomentum=F(c["jump_momentum"]);p.jumpNormalScaleFactor=F(c["jump_normal_scale"]);p.extraJumpTeleportMultiplier=F(c["jump_teleport"]);p.inverseMass01=F(c["inverse_mass"]);
                Set(p,"PlatformSlipperyness01",F(config["friction"]));Set(p,"IcePlatformSlipperyness01",F(config["ice_friction"]));
                adapter.parent.hierarchyNumber=(int)a["collision_hierarchy"];
                if(a["entity"]!=null)
                {
                    var collider=go.GetComponent<IPhysicsCollider>();((MonoUpdatable)collider).HierarchyNumber=(int)a["collision_hierarchy"];
                    if(collider is DPhysicsBox box)box.pp=adapter.parent;else ((DPhysicsCircle)collider).pp=adapter.parent;
                }
                if(adapter.shape==Shape.Circle)scene.circles.physicsParents[scene.circles.ColliderIndex(adapter.parent.instanceId)]=adapter.parent;
                else scene.boxes.physicsParents[scene.boxes.ColliderIndex(adapter.parent.instanceId)]=adapter.parent;
                SlimeController slime=null;
                if((animations!=null&&a["slime"]!=null)||a["push_strength"].Type!=JTokenType.Null){slime=go.GetComponent<SlimeController>()??go.AddComponent<SlimeController>();Set(slime,"body",body);Set(slime,"playerPhysics",p);slime.playerNumber=(int)bs["holder"];}
                if(animations!=null)slimes.Add(a["slime"]==null?null:ReferenceSlimeActors.Bind(a["slime"],slime,body,p,a["entity"]!=null?go.GetComponent<IPhysicsCollider>():adapter,animations[i],objects,assets,i,platforms,adapters,entityObjects,ropes));
                if(a["rewind"]!=null)ReferenceActorRewind.Bind(a["rewind"],slime);
                if(a["push_strength"].Type!=JTokenType.Null)
                {
                    // Run the original bite gate and overlap shove together.
                    slime.scaleDiffBeforeNomNom=a["bite_threshold"]==null?Fix.One:F(a["bite_threshold"]);slime.pushOverlappingPlayersStr=F(a["push_strength"]);
                    adapter.Collision=hit=>{
                        currentActor=actor;
                        var before=body.externalVelocity;slime.OnCollide(hit);
                        if(body.externalVelocity!=before)pushes.Add(new JObject{["actor"]=actor,["external_delta"]=V(body.externalVelocity-before)});
                    };
                }
                if((bool?)runtime["combat"]?["collisions"]==true&&a["collision"]!=null)
                {
                    var previous=adapter.Collision;var collision=go.GetComponent<PlayerCollision>();
                    adapter.Collision=hit=>{currentActor=actor;previous?.Invoke(hit);collision.OnCollide(hit);};
                }
                p.OnNoLongerGrounded+=angle=>Event(actor,"detached",angle.m_rawValue);
                p.OnAttachedToGround+=hit=>Event(actor,"attached",new JObject{["object"]=hit.pp.instanceId,["near_position"]=V(hit.nearPos),["far_position"]=V(hit.farPos),["near_distance"]=hit.nearDist.m_rawValue,["far_distance"]=hit.farDist.m_rawValue,["layer"]=hit.layer});
            }
            var ordinary=new Dictionary<int,ReferenceOrdinaryAbilities.Instance>();
            for(int i=0;i<slimes.Count;i++)if(slimes[i]!=null)
            {
                slimes[i].Peers=slimes;var eating=actorData[i]["slime"]["eating"];if(eating==null)continue;
                slimes[i].Slime.scaleIncreaseOnNom=F(eating["scale_increase"]);Set(slimes[i].Slime,"swallowing",(bool)eating["swallowing"]);
                Set(slimes[i].Slime,"playerToEat",eating["target"].Type==JTokenType.Null?null:slimes[(int)eating["target"]].Slime);
            }
            for(int i=0;i<actorData.Count;i++)if(actorData[i]["ordinary"]!=null)ordinary.Add(i,ReferenceOrdinaryAbilities.Bind(actorData[i]["ordinary"],entityObjects[i],slimes,animations?[i],objects,assets));
            if(deathEffects!=null)for(int i=0;i<bodies.Count;i++){var collision=bodies[i].GetComponent<PlayerCollision>();if(collision!=null)Set(collision,"playerIdHolder",slimes[i]!=null?(IPlayerIdHolder)slimes[i].Slime:ordinary[i].Ability);}
            for(int i=0;i<bodies.Count;i++)if(actorData[i]["bounds"]!=null)ReferenceActorBounds.Bind(actorData[i]["bounds"],bodies[i],physics[i],slimes[i]!=null?(IPlayerIdHolder)slimes[i].Slime:ordinary[i].Ability,deathEffects);
            ReferencePlayerPosition.Bind(initial["players"],players,slimes);
            var drivers=new List<ReferenceActorDriver>();
            ReferencePlayerController.SoundCapture=name=>Event(currentActor,name.StartsWith("slimeJump")?"jump_sound":"sound",name);
            if((bool?)query["entry_events"]==true)ReferenceCapture.ActorAbilityCapture=(player,slot,ability)=>Event(currentActor,"ability_entry",slot);
            if((bool?)query["dynamic_actors"]==true)
            {
                clones=new ReferenceActorClones(query,actorData,slimes,bodies,physics,transforms,entityObjects,objects,platforms,adapters,ropes,deathEffects,Event,actor=>currentActor=actor,(actor,delta)=>pushes.Add(new JObject{["actor"]=actor,["external_delta"]=V(delta)}));
                foreach(var player in players){var owner=slimes.First(s=>s!=null&&s.Slime.playerNumber==player.Id);player.Color=owner.Animation.Animator.spriteRen.sharedMaterial;}
            }
            var calls=new Dictionary<string,int>();
            void Command(JToken command)
            {
                effects.Add(command.DeepClone());
                if((string)command["op"]=="bite_actor"){currentActor=(int)command["actor"];slimes[currentActor].Slime.Bite(bodies[(int)command["victim"]].GetComponent<PlayerCollision>());return;}
                if((string)command["op"]=="chew_actor"){currentActor=(int)command["actor"];slimes[currentActor].Slime.Chew();return;}
                if((string)command["op"]=="collision_invulnerable"){bodies[(int)command["actor"]].GetComponent<PlayerCollision>().SetInvulnerable((bool)command["value"],(uint)command["mask"]);return;}
                if((string)command["op"]=="clone_actor")
                {
                    currentActor=(int)command["actor"];var collision=bodies[currentActor].GetComponent<PlayerCollision>();var prefab=collision.reviveEffectPrefab;
                    clones.CreatingPlayer=players[(int)command["player"]].Id;
                    if((bool?)command["null_prefab"]==true)collision.reviveEffectPrefab=null;
                    try{collision.SpawnAbilityLessClone(players[(int)command["player"]],null,V(command["position"]));}finally{collision.reviveEffectPrefab=prefab;}
                    return;
                }
                if((string)command["op"]=="actor_spawn"){currentActor=(int)command["actor"];ReferenceActorSpawn.Apply((string)command["action"],command["position"],slimes[currentActor]);return;}
                if((string)command["op"]=="kill_actor"){var request=command["request"];currentActor=(int)command["actor"];bodies[currentActor].GetComponent<PlayerCollision>().killPlayer((int)request["killer"],(bool)request["spawn_effect"],(bool)request["ignore_invulnerability"],(CauseOfDeath)(int)request["cause"]);return;}
                if((string)command["op"]=="destroy_actor"){Updater.DestroyFix(bodies[(int)command["actor"]].gameObject);return;}
                if((string)command["op"]=="quick_cast"){currentActor=(int)command["actor"];ReferenceQuickSpell.Cast(bodies[currentActor].GetComponent<QuickSpell>());return;}
                if((string)command["op"]=="quick_exit"){currentActor=(int)command["actor"];bodies[currentActor].GetComponent<QuickSpell>().ExitAbility(ReferenceSlimeExit.Info(command["info"],slimes[ordinary[currentActor].Owner]));return;}
                if((string)command["op"]=="ordinary_exit"){currentActor=(int)command["actor"];ordinary[currentActor].Exit(command["info"]);return;}
                if((string)command["op"]=="slime_exit"){currentActor=(int)command["actor"];ReferenceSlimeExit.Apply(command["info"],slimes[currentActor]);return;}
                if(ReferencePlayerPosition.Command(command,players,slimes))return;
                if((string)command["op"]=="actor_active"){bodies[(int)command["actor"]].gameObject.SetActive((bool)command["value"]);return;}
                if((string)command["op"]=="body_simulated"){bodies[(int)command["actor"]].simulated=(bool)command["value"];return;}
                if((string)command["op"]=="player_scale"){players[(int)command["player"]].Scale=F(command["value"]);return;}
                if((string)command["op"]=="box_enabled"){bodies[(int)command["actor"]].GetComponent<IPhysicsCollider>().enabled=(bool)command["value"];return;}
                if((string)command["op"]=="box_scale"){bodies[(int)command["actor"]].GetComponent<IPhysicsCollider>().Scale=F(command["value"]);return;}
                if((string)command["op"]=="circle_radius"){bodies[(int)command["actor"]].GetComponent<DPhysicsCircle>().radius=F(command["value"]);return;}
                if((string)command["op"]=="box_sync"){bodies[(int)command["actor"]].GetComponent<IPhysicsCollider>().UpdatePhysicsPositions();return;}
                if((string)command["op"]!="ragdoll")throw new ArgumentException("Unknown actor command");
                slimes[(int)command["actor"]].Apply(command["request"]);
            }
            void Phase(string kind,int? id)
            {
                phases.Add(new JObject{["kind"]=kind,["id"]=id.HasValue?new JValue(id.Value):JValue.CreateNull()});
                string key=kind+":"+(id.HasValue?id.Value.ToString():"*");int index=calls.TryGetValue(key,out int old)?old:0;calls[key]=index+1;
                var script=query["scripts"]?[key];if(script!=null&&index<script.Count())foreach(var command in script[index])Command(command);
            }
            for(int i=0;i<runtime["components"].Count();i++)
            {
                int id=i;var entry=runtime["components"][i];int actor=(int)entry["actor"];string role=(string)entry["role"];
                var go=entityObjects[actor]??new GameObject("composed dispatch");if(entityObjects[actor]==null)objects.Add(go);var d=go.AddComponent<ReferenceActorDriver>();drivers.Add(d);
                d.Destroyed=()=>Phase("destroy_updatable",id);
                d.HierarchyNumber=(int)initial["components"][i]["hierarchy"];
                if(actorData[actor]["entity"]!=null)d.Activity=()=>bodies[actor].gameObject.activeInHierarchy&&(role!="body"||bodies[actor].simulated)&&(role!="physics"||actorData[actor]["slime"]?["spawn"]==null||physics[actor].enabled);
                d.Initialize=()=>{Phase("init",id);currentActor=actor;if(role=="body")bodies[actor].Init();else if(role=="motor"&&actorData[actor]["slime"]?["spawn"]!=null)slimes[actor].Slime.Init();};
                d.Step=dt=>{
                    Phase("update",id);currentActor=actor;Constants.GROUNDALIGNMENTDEADZONE=F(actorData[actor]["config"]["alignment_deadzone"]);
                    if(role=="collision")bodies[actor].GetComponent<PlayerCollision>().UpdateSim(dt);
                    else if(role=="bounds")bodies[actor].GetComponent<DestroyIfOutsideSceneBounds>().UpdateSim(dt);
                    else if(role=="rewind")bodies[actor].GetComponent<RewindRecorder>().UpdateSim(dt);
                    else if(role=="dash")ordinary[actor].Ability.GetComponent<Dash>().UpdateSim(dt);
                    else if(role=="ability")ordinary[actor].Ability.UpdateSim(dt);
                    else if(role=="physics")physics[actor].UpdateSim(dt);
                    else if(role=="body")bodies[actor].UpdateSim(dt);
                    else if(role=="animation")animations[actor].Animator.UpdateSim(dt);
                    else if(animations!=null&&slimes[actor]!=null)slimes[actor].Slime.UpdateSim(dt);
                    else {if(players[actor].jumpButton_PressedThisFrame()&&physics[actor].IsGrounded())physics[actor].Jump();physics[actor].Move(V(actorData[actor]["input"]),dt);}
                };
                d.Late=dt=>{Phase("late",id);currentActor=actor;scene.groundedPlayerRopeStr=F(actorData[actor]["config"]["physics"]["rope_strength"]);if(role=="physics")physics[actor].LateUpdateSim(dt);else if(role=="body")bodies[actor].LateUpdateSim(dt);};
            }
            if(deathEffects!=null){deathEffects.ActorId=body=>bodies.IndexOf(body);deathEffects.MaterialId=material=>{int id=slimes.FindIndex(s=>s!=null&&s.Animation.Animator.spriteRen!=null&&s.Animation.Animator.spriteRen.sharedMaterial==material);if(id<0)throw new ArgumentException("Unbound death material");return id;};ReferenceDeathEffects.ComponentPhase=(kind,component)=>{var id=deathEffects.Id(component);if(id.HasValue){int actor=deathEffects.Actor(component);currentActor=actor;Phase(kind,id);currentActor=actor;}else if(clones!=null&&(id=clones.Id(component)).HasValue){int actor=clones.Actor(component);currentActor=actor;Phase(kind,id);currentActor=actor;}};}
            Static("addedUpdatables",(int)initial["added"]);Static("prefabsAdded",(int)initial["prefabs_added"]);Static("registeringAPrefab",(bool)initial["registering_prefab"]);Static("hitStopFrames",(int)initial["hitstop"]);
            Static("toBeAddedUpdatables",initial["pending"]["items"].Take((int)initial["pending"]["length"]).Select(id=>(MonoUpdatable)drivers[(int)id]).ToList());
            Updater.updatables=initial["live"].Select(id=>(MonoUpdatable)drivers[(int)id]).ToList();
            Static("toBeAddedPrefab",new List<MonoUpdatable>());Static("toDestroyBuffer",new List<MonoUpdatable>());Static("destroyedObjects",new Queue<GameObject>());Static("destroyedMonos",new Queue<MonoBehaviour>());
            Static("LoadedALevelThisUpdate",(bool)initial["loaded"]);Static("gameHasStopped",(bool)initial["stopped"]);Static("SimulationTicks",(int)initial["ticks"]);
            Updater.SimTimePassed=F(initial["time"]);Updater.SimTimeSinceLevelLoaded=F(initial["level_time"]);Host.recordReplay=false;
            var removed=new Dictionary<int,JObject>();
            JObject ActorSnapshot(int i)
            {
                if(removed.TryGetValue(i,out var cached))
                {
                    var cachedState=(JObject)cached.DeepClone();cachedState["input"]=actorData[i]["input"].DeepClone();return cachedState;
                }
                var state=(JObject)actorData[i].DeepClone();
                if(state["collision"]!=null)state["collision"]=ReferenceActorCombat.Collision(bodies[i].GetComponent<PlayerCollision>());
                var p=physics[i];if(state["entity"]!=null)state["entity"]=ReferenceActorEntity.Snapshot(actorData[i]["entity"],bodies[i]);var c=state["controller"];c["skip_update"]=p.skipNextUpdate;
                c["physics"]["motion"]=ReferencePlayerGround.Motion(p,bodies[i],platforms);
                c["physics"]["scale"]=bodies[i].fixtrans.Scale.m_rawValue;c["physics"]["previous_scale"]=Get<Fix>(p,"prevScal").m_rawValue;c["physics"]["jumped"]=p.jumpedThisFrame;
                if(animations!=null&&slimes[i]!=null)state["slime"]=slimes[i].Snapshot(actorData[i]["slime"]);
                if(ordinary.TryGetValue(i,out var ability))state["ordinary"]=ability.Snapshot(actorData[i]["ordinary"]);
                if(state["rewind"]!=null)state["rewind"]=ReferenceActorRewind.Snapshot(bodies[i].GetComponent<RewindRecorder>());
                state["body"]=ReferencePlayerBody.SnapshotBody(bodies[i],ropes,false)["state"];
                clones?.Snapshot(i,state);return state;
            }
            ReferenceScheduler.Capture=(kind,target)=>{
                int? id=target is int n?(int?)n:null;
                if(kind=="destroy_object")
                {
                    int actor=entityObjects.FindIndex(go=>ReferenceEquals(go,target));
                    if(actor>=0)id=(int)actorData[actor]["entity"]["node"];else if(deathEffects!=null)id=deathEffects.Node((GameObject)target);
                }
                Phase(kind,id);
                if(kind=="physics"){simulated=true;ReferencePhysicsScene.BeginFrame();}
                if(kind=="destroy_object"&&target is GameObject doomed&&doomed!=null)
                {
                    deathEffects?.Removing(doomed);
                    sceneSnapshot();
                    for(int i=0;i<entityObjects.Count;i++)if(entityObjects[i]!=null&&(entityObjects[i]==doomed||entityObjects[i].transform.IsChildOf(doomed.transform)))
                    {
                        var state=ActorSnapshot(i);state["entity"]["present"]=false;removed[i]=state;
                    }
                }
            };
            JObject Snapshot()
            {
                var states=new JArray(Enumerable.Range(0,actorData.Count).Select(ActorSnapshot));
                var world=sceneSnapshot();world["objects"]=new JArray(world["objects"].OrderBy(o=>(int)o["id"]));
                var frame=new JObject{["actors"]=states,["scene"]=world,["report"]=new JObject{["phases"]=phases,["events"]=events,["pushes"]=pushes,["physics"]=simulated?(JToken)ReferencePhysicsScene.Report(adapters):JValue.CreateNull()},
                    ["ticks"]=Updater.SimulationTicks,["time"]=Updater.SimTimePassed.m_rawValue,["level_time"]=Updater.SimTimeSinceLevelLoaded.m_rawValue,
                    ["inputs"]=new JArray(inputs.Select((p,i)=>{var v=new JObject{["jump"]=p.jumpButton_IS_DOWN,["pressed"]=p.jumpButton_DOWN_THIS_FRAME,["previous"]=p.jumpButton_DOWN_LAST_FRAME};if(initial["inputs"][i]["angle"]!=null)v["angle"]=p.joystickAngleByte;return v;}))};
                if((bool?)query["registration_state"]==true)frame["registration"]=deathEffects.Registration(drivers);
                if(combat!=null){frame["combat"]=combat.Snapshot();if(deathEffects!=null)frame["combat"]["death_effects"]=deathEffects.Snapshot();}
                if(runtime["actor_ropes"]!=null){var binding=(JObject)runtime["actor_ropes"].DeepClone();binding["metadata"]=new JArray(ropes.Select(r=>new JObject{["owner"]=r.ownerId,["time_air_start"]=r.timeInAir_start.m_rawValue,["time_air_end"]=r.timeInAir_end.m_rawValue}));frame["actor_ropes"]=binding;}
                if((bool?)query["player_state"]==true)frame["players"]=new JArray(players.Select((player,i)=>{var v=(JObject)initial["players"][i].DeepClone();v["protected"]=player.isProtectedFromTimeStop;v["invisible"]=player.isInvisible;v["time_invisible"]=player.timeSpentInvisible.m_rawValue;v["time_ability"]=player.timeSinceEnteredAbility.m_rawValue;if(v["scale"]!=null)v["scale"]=player.Scale.m_rawValue;if(v["eaten_count"]!=null)v["eaten_count"]=player.playersEatenCountThisRound;if(v["position"]!=null)v["position"]=ReferencePlayerPosition.Snapshot(player,slimes);if(v["flags"]!=null)v["flags"]=new JObject{["spark"]=player.InSparkShot,["macho"]=player.InMachoThrow,["platform"]=player.InPlatformTransform};return v;}));
                if(query["scripts"]!=null)frame["effects"]=effects;
                if((bool?)query["routing_checks"]==true)frame["routing_checks"]=new JArray(slimes.Select(s=>s==null?(JToken)JValue.CreateNull():s.RoutingChecks()));
                return frame;
            }
            var frames=new JArray();int completed=0;bool streaming=ReferenceKernel.ActorFrameSink!=null;foreach(var a in query["actions"])
            {
                events=new JArray();phases=new JArray();pushes=new JArray();effects=new JArray();simulated=false;
                for(int i=0;i<inputs.Count;i++){actorData[i]["input"]=a["inputs"][i]["direction"].DeepClone();inputs[i].jumpButton_IS_DOWN=(bool)a["inputs"][i]["jump"];if(a["inputs"][i]["angle"]!=null)inputs[i].joystickAngleByte=(byte)a["inputs"][i]["angle"];}
                if(clones!=null)for(int i=0;i<actorData.Count;i++){int owner=players.FindIndex(player=>player.Id==Get<IPlayerIdHolder>(bodies[i],"idHolder").GetPlayerId());actorData[i]["input"]=a["inputs"][(int)initial["players"][owner]["input"]]["direction"].DeepClone();}
                if(animations!=null)for(int i=0;i<inputs.Count;i++){var dir=V(a["inputs"][i]["direction"]);inputs[i].UsesMouse=true;inputs[i].w_IS_DOWN=dir.y>Fix.Zero;inputs[i].s_IS_DOWN=dir.y<Fix.Zero;inputs[i].d_IS_DOWN=dir.x>Fix.Zero;inputs[i].a_IS_DOWN=dir.x<Fix.Zero;}
                if(a["routing_inputs"]!=null)for(int i=0;i<a["routing_inputs"].Count();i++)if(slimes[i]!=null)slimes[i].RoutingInputs(a["routing_inputs"][i]);
                GameTime.PlayerTimeScale=F(a["scale"]);for(int i=0;i<drivers.Count;i++)drivers[i].EnabledFlag=(bool)a["enabled"][i];
                string error=null;try{if(a["commands"]!=null)foreach(var command in a["commands"])Command(command);typeof(Updater).GetMethod("TickSimulationInner",Statics).Invoke(null,new object[]{F(a["dt"])});}
                catch(Exception e){while(e is TargetInvocationException)e=e.InnerException;if(e is NullReferenceException)error="null_reference";else if(e is MissingReferenceException)error="missing_reference";else if(e is ArgumentException&&e.Message.StartsWith("The Object you want to instantiate is null"))error="instantiate_null";else if(e is DivideByZeroException)error="division_by_zero";else if(e is IndexOutOfRangeException)error="index_out_of_range";else if(e.Message.StartsWith("animation "))error="animation_missing";else throw;}
                if(error!=null)return new JObject{["error"]=error,["action"]=completed,["completed"]=streaming?(JToken)new JValue(completed):frames,["failed"]=Snapshot()};
                var frame=Snapshot();if(streaming)ReferenceKernel.ActorFrameSink(completed,frame);else frames.Add(frame);completed++;
            }
            return streaming?(JToken)new JObject{["completed"]=completed}:frames;
        }
        finally
        {
            clones?.Dispose();deathEffects?.Dispose();combat?.Dispose();
            foreach(var mesh in meshes)mesh.Dispose();scene.playerGravity=oldPlayerGravity;scene.maxStretchLength=oldMaxStretch;scene.groundedPlayerRopeStr=oldRopeStrength;
            ReferenceScheduler.Capture=null;achievedField.SetValue(null,oldAchieved);foreach(var asset in assets)UnityEngine.Object.DestroyImmediate(asset);foreach(var go in objects)if(go!=null)UnityEngine.Object.DestroyImmediate(go);
            foreach(var pair in saved)pair.Key.SetValue(null,pair.Value);PlayerHandler.Get().SetPlayerList(oldPlayers);GameTime.PlayerTimeScale=oldScale;
            TutorialGameHandler.isInTutorial=oldTutorial;Constants.leveltype=oldSpace;Constants.GROUNDALIGNMENTDEADZONE=oldDeadzone;ReferenceScheduler.Capture=oldCapture;ReferencePlayerController.SoundCapture=oldSound;Host.recordReplay=oldRecord;ReferenceCapture.ActorAbilityCapture=oldEntryCapture;
        }
    }
}
