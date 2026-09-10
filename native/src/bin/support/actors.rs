use bopl_sim::{
    actors::{Report, Runtime},
    fixed::Fix,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
type FrameSink<'a> = &'a mut dyn FnMut(usize, &Value);
pub fn evaluate(q: &Value) -> Value {
    evaluate_with(q, None)
}
pub fn evaluate_streamed(q: &Value, sink: FrameSink<'_>) -> Value {
    evaluate_with(q, Some(sink))
}
fn evaluate_with(q: &Value, mut sink: Option<FrameSink<'_>>) -> Value {
    let mut completed = 0;
    let mut runtime: Runtime = serde_json::from_value(q["runtime"].clone()).unwrap();
    runtime.synchronize();
    let dynamic_actors = q["dynamic_actors"].as_bool() == Some(true);
    let clone_prefab: Option<bopl_sim::actor_clone::Prefab> = q
        .get("clone_prefab")
        .map(|v| serde_json::from_value(v.clone()).unwrap());
    let mut frames = Vec::new();
    let mut calls = BTreeMap::<String, usize>::new();
    let scripted = q.get("scripts").is_some();
    for action in q["actions"].as_array().unwrap() {
        let mut report = Report::default();
        let mut effects = Vec::new();
        for (i, input) in action["inputs"].as_array().unwrap().iter().enumerate() {
            runtime.actors[i].input = serde_json::from_value(input["direction"].clone()).unwrap();
            runtime.scheduler.inputs[i].jump = input["jump"].as_bool().unwrap();
            if let Some(angle) = input.get("angle") {
                runtime.scheduler.inputs[i].angle = Some(angle.as_u64().unwrap() as u8);
            }
        }
        if dynamic_actors {
            for actor in &mut runtime.actors {
                let input = runtime
                    .scheduler
                    .players
                    .iter()
                    .find(|p| Some(p.id) == actor.body.holder)
                    .and_then(|p| p.input)
                    .unwrap();
                actor.input =
                    serde_json::from_value(action["inputs"][input]["direction"].clone()).unwrap();
            }
        }
        if let Some(inputs) = action.get("routing_inputs") {
            for (actor_index, actor) in runtime.actors.iter_mut().enumerate() {
                if !actor.is_present() {
                    continue;
                }
                let input_index = if dynamic_actors {
                    runtime
                        .scheduler
                        .players
                        .iter()
                        .find(|p| Some(p.id) == actor.body.holder)
                        .and_then(|p| p.input)
                        .unwrap()
                } else {
                    actor_index
                };
                let Some(input) = inputs.get(input_index) else {
                    continue;
                };
                let Some(slime) = actor.slime.as_mut() else {
                    continue;
                };
                let state = slime.abilities.as_mut().unwrap();
                state.buttons = serde_json::from_value(input["buttons"].clone()).unwrap();
                state.can_use = input["can_use"].as_bool().unwrap();
            }
        }
        runtime.scheduler.time_scale = serde_json::from_value(action["scale"].clone()).unwrap();
        for (i, enabled) in action["enabled"].as_array().unwrap().iter().enumerate() {
            runtime.scheduler.components[i].enabled = enabled.as_bool().unwrap();
        }
        let mut command = |actors: &mut Vec<bopl_sim::actors::Actor>,
                           context: &mut bopl_sim::actors::Context,
                           report: &mut Report,
                           command: &Value| {
            effects.push(command.clone());
            if command["op"] == "collision_invulnerable" {
                let actor = &mut actors[command["actor"].as_u64().unwrap() as usize];
                if !actor.is_present() {
                    return Err("missing_reference");
                }
                let collision = actor.collision.as_mut().ok_or("null_reference")?;
                let mask = command["mask"].as_u64().unwrap() as u32;
                if command["value"].as_bool().unwrap() {
                    collision.invulnerable |= mask;
                } else {
                    collision.invulnerable &= !mask;
                }
                return Ok(());
            }
            if command["op"] == "bite_actor" {
                return Runtime::bite_actor(
                    actors,
                    context,
                    command["actor"].as_u64().unwrap() as usize,
                    command["victim"].as_u64().unwrap() as usize,
                    report,
                );
            }
            if command["op"] == "chew_actor" {
                return Runtime::chew_actor(
                    actors,
                    context,
                    command["actor"].as_u64().unwrap() as usize,
                    report,
                );
            }
            if command["op"] == "clone_actor" || command["op"] == "full_clone_actor" {
                let caller = actors
                    .get(command["actor"].as_u64().unwrap() as usize)
                    .ok_or("argument_out_of_range")?;
                if !caller.is_present() {
                    return Err("missing_reference");
                }
                caller.collision.as_ref().ok_or("null_reference")?;
                if command["op"] == "full_clone_actor" {
                    Runtime::spawn_clone(
                        actors,
                        context,
                        command["actor"].as_u64().unwrap() as usize,
                        command["player"].as_u64().unwrap() as usize,
                        serde_json::from_value(command["position"].clone()).unwrap(),
                        if command["null_prefab"].as_bool() == Some(true) {
                            None
                        } else {
                            clone_prefab.as_ref()
                        },
                        report,
                    )?;
                    return Ok(());
                }
                Runtime::spawn_abilityless_clone(
                    actors,
                    context,
                    command["player"].as_u64().unwrap() as usize,
                    serde_json::from_value(command["position"].clone()).unwrap(),
                    if command["null_prefab"].as_bool() == Some(true) {
                        None
                    } else {
                        clone_prefab.as_ref()
                    },
                    report,
                )?;
                return Ok(());
            }
            if command["op"] == "actor_spawn" {
                return Runtime::spawn_actor(
                    actors,
                    context,
                    command["actor"].as_u64().unwrap() as usize,
                    serde_json::from_value(command["action"].clone()).unwrap(),
                    serde_json::from_value(command["position"].clone()).unwrap(),
                    report,
                );
            }
            if command["op"] == "kill_actor" {
                Runtime::kill_actor(
                    actors,
                    context,
                    command["actor"].as_u64().unwrap() as usize,
                    &serde_json::from_value(command["request"].clone()).unwrap(),
                    report,
                )?;
                return Ok(());
            }
            if command["op"] == "destroy_actor" {
                return Runtime::destroy_actor(
                    actors,
                    context,
                    command["actor"].as_u64().unwrap() as usize,
                );
            }
            if command["op"] == "revive_effect" {
                let world = context.spells.as_mut().ok_or("spell_world_not_bound")?;
                let index = world.spawn_revive_effect(
                    context.schedule,
                    command["prefab"].as_str().unwrap(),
                    command["actor"].as_u64().unwrap() as usize,
                    serde_json::from_value(command["position"].clone()).unwrap(),
                )?;
                return world.init_revive_effect(
                    index,
                    command["player"]
                        .as_u64()
                        .and_then(|i| context.schedule.players.get(i as usize))
                        .map(|p| p.id),
                    serde_json::from_value(command["targets"].clone()).unwrap(),
                );
            }
            if command["op"] == "revive_particle_init" {
                let world = context.spells.as_mut().ok_or("spell_world_not_bound")?;
                world.sync_positions(context.scene);
                return world.init_particle(
                    context.schedule,
                    command["spell"].as_u64().unwrap() as usize,
                    command["target"].as_u64().map(|i| i as usize),
                    command["slime"].as_u64().map(|i| i as usize),
                    command["clone_id"].as_i64().unwrap() as i32,
                );
            }
            if command["op"] == "revive_particle" {
                let world = context.spells.as_mut().ok_or("spell_world_not_bound")?;
                world.sync_positions(context.scene);
                let index = world.spawn_particle(
                    context.schedule,
                    command["prefab"].as_str().unwrap(),
                    command["actor"].as_u64().unwrap() as usize,
                    serde_json::from_value(command["position"].clone()).unwrap(),
                )?;
                world.init_particle(
                    context.schedule,
                    index,
                    command["target"].as_u64().map(|i| i as usize),
                    command["slime"].as_u64().map(|i| i as usize),
                    command["clone_id"].as_i64().unwrap() as i32,
                )?;
                return Ok(());
            }
            if command["op"] == "spell_command" {
                let bopl_sim::actors::Context {
                    spells,
                    schedule,
                    scene,
                    bindings,
                    ..
                } = context;
                return spells.as_mut().ok_or("spell_world_not_bound")?.command(
                    command["spell"].as_u64().unwrap() as usize,
                    serde_json::from_value(command["command"].clone()).unwrap(),
                    schedule,
                    &mut bopl_sim::shockwave::Targets {
                        actors,
                        scene,
                        bindings,
                    },
                );
            }
            if command["op"] == "quick_cast" {
                return Runtime::cast_quick_spell(
                    actors,
                    context,
                    command["actor"].as_u64().unwrap() as usize,
                    report,
                );
            }
            if command["op"] == "quick_exit" {
                return Runtime::exit_quick_spell(
                    actors,
                    context,
                    command["actor"].as_u64().unwrap() as usize,
                    serde_json::from_value(command["info"].clone()).unwrap(),
                    report,
                );
            }
            if command["op"] == "ordinary_exit" {
                return Runtime::exit_ordinary(
                    actors,
                    context,
                    command["actor"].as_u64().unwrap() as usize,
                    &serde_json::from_value(command["info"].clone()).unwrap(),
                    report,
                );
            }
            if command["op"] == "slime_exit" {
                return Runtime::exit_actor(
                    actors,
                    context,
                    command["actor"].as_u64().unwrap() as usize,
                    &serde_json::from_value(command["info"].clone()).unwrap(),
                    report,
                );
            }
            let bopl_sim::actors::Context {
                schedule, scene, ..
            } = context;
            match command["op"].as_str().unwrap() {
                "actor_active" | "body_simulated" => {
                    let entity = actors[command["actor"].as_u64().unwrap() as usize]
                        .entity
                        .as_mut()
                        .ok_or("missing_actor_entity")?;
                    let value = command["value"].as_bool().unwrap();
                    if command["op"] == "actor_active" {
                        entity.active = value;
                        schedule.nodes[entity.node].active = value;
                        let object = actors[command["actor"].as_u64().unwrap() as usize].object;
                        scene
                            .objects
                            .iter_mut()
                            .find(|o| o.id == object)
                            .or(actors[command["actor"].as_u64().unwrap() as usize]
                                .pending_object
                                .as_mut())
                            .ok_or("missing_actor_object")?
                            .active = value;
                    } else {
                        entity.body_simulated = value;
                    }
                    return Ok(());
                }
                "player_scale" => {
                    let value: Fix = serde_json::from_value(command["value"].clone()).unwrap();
                    schedule.players[command["player"].as_u64().unwrap() as usize].scale =
                        Some(value.max(Fix::from_f64(0.3)).min(Fix::from_int(3)));
                    return Ok(());
                }
                "box_enabled" => {
                    let object = actors[command["actor"].as_u64().unwrap() as usize].object;
                    scene
                        .objects
                        .iter_mut()
                        .find(|o| o.id == object)
                        .or(actors[command["actor"].as_u64().unwrap() as usize]
                            .pending_object
                            .as_mut())
                        .ok_or("missing_actor_object")?
                        .enabled = command["value"].as_bool().unwrap();
                    return Ok(());
                }
                "circle_radius" => {
                    let actor = &mut actors[command["actor"].as_u64().unwrap() as usize];
                    return actor
                        .entity
                        .as_mut()
                        .ok_or("missing_actor_entity")?
                        .collider
                        .set_radius(
                            serde_json::from_value(command["value"].clone()).unwrap(),
                            actor.object,
                            scene,
                        );
                }
                "box_scale" => {
                    return actors[command["actor"].as_u64().unwrap() as usize].set_collider_scale(
                        serde_json::from_value(command["value"].clone()).unwrap(),
                        scene,
                    );
                }
                "box_sync" => {
                    return actors[command["actor"].as_u64().unwrap() as usize]
                        .sync_collider(scene);
                }
                "set_position" => {
                    let priorities = Runtime::position_priorities(actors);
                    let candidate = command["controller"].as_u64().map(|i| i as usize);
                    let value = serde_json::from_value(command["value"].clone()).unwrap();
                    schedule.players[command["player"].as_u64().unwrap() as usize]
                        .position
                        .as_mut()
                        .ok_or("missing_player_position")?
                        .set(value, candidate, &priorities);
                    return Ok(());
                }
                "controller_destroyed" => {
                    actors[command["actor"].as_u64().unwrap() as usize]
                        .slime
                        .as_mut()
                        .ok_or("missing_slime_controller")?
                        .managed_destroyed = Some(command["value"].as_bool().unwrap());
                    return Ok(());
                }
                _ => {}
            }
            if command["op"] != "ragdoll" {
                return Err("unknown_actor_command");
            }
            actors[command["actor"].as_u64().unwrap() as usize].apply_ragdoll(
                &serde_json::from_value(command["request"].clone()).unwrap(),
                scene,
            )
        };
        let result = action
            .get("commands")
            .map_or(Ok(()), |commands| {
                commands.as_array().unwrap().iter().try_for_each(|c| {
                    command(
                        &mut runtime.actors,
                        &mut bopl_sim::actors::Context {
                            schedule: &mut runtime.scheduler,
                            components: &mut runtime.components,
                            scene: &mut runtime.scene,
                            actor_ropes: &mut runtime.actor_ropes,
                            combat: &mut runtime.combat,
                            spells: &mut runtime.spells,
                            object_id_high_water: &mut runtime.object_id_high_water,
                            bindings: &runtime.bindings,
                            grounds: &runtime.grounds,
                            space: runtime.space,
                        },
                        &mut report,
                        c,
                    )
                })
            })
            .and_then(|()| {
                runtime.tick_with(
                    Fix::from_raw(action["dt"].as_i64().unwrap()),
                    &mut report,
                    &mut |actors, context, report, phase| {
                        let key = format!(
                            "{}:{}",
                            phase.kind.as_str(),
                            phase.id.map_or_else(|| "*".into(), |i| i.to_string())
                        );
                        let index = calls.entry(key.clone()).or_default();
                        let invocation = *index;
                        *index += 1;
                        if let Some(commands) = q["scripts"][&key].get(invocation) {
                            for c in commands.as_array().unwrap() {
                                command(actors, context, report, c)?;
                            }
                        }
                        Ok(())
                    },
                )
            });
        if q["entry_events"].as_bool() != Some(true) {
            report.events.retain(|e| {
                !matches!(e.event, bopl_sim::player_controller::Event::AbilityEntry(_))
            });
        }
        let mut scene = runtime.scene.clone();
        scene.objects.sort_by_key(|o| o.id);
        let mut frame = json!({"actors":runtime.actors,"scene":scene,"report":report,"ticks":runtime.scheduler.ticks,
            "time":runtime.scheduler.time,"level_time":runtime.scheduler.level_time,"inputs":runtime.scheduler.inputs});
        if q["registration_state"].as_bool() == Some(true) {
            let s = &runtime.scheduler;
            let mut dynamic:Vec<_>=runtime.combat.as_ref().and_then(|c|c.death_effects.as_ref()).into_iter().flat_map(|e|&e.instances).flat_map(|e|e.components[..2].iter()).map(|&id| {let c=&s.components[id];json!({"id":id,"hierarchy":c.hierarchy,"destroyed":c.destroyed,"present":c.present})}).collect();
            if let Some(spells) = &runtime.spells {
                for instance in &spells.instances {
                    for &id in &instance.components {
                        let c = &s.components[id];
                        if c.kind == bopl_sim::scheduler::Kind::Updatable {
                            dynamic.push(json!({"id":id,"hierarchy":c.hierarchy,"destroyed":c.destroyed,"present":c.present}));
                        }
                    }
                }
                dynamic.sort_by_key(|v| v["id"].as_u64().unwrap());
            }
            if dynamic_actors {
                let initial = q["runtime"]["scheduler"]["components"]
                    .as_array()
                    .unwrap()
                    .len();
                for (id, binding) in runtime.components.iter().enumerate().skip(initial) {
                    if binding.is_some() {
                        let c = &s.components[id];
                        dynamic.push(json!({"id":id,"hierarchy":c.hierarchy,"destroyed":c.destroyed,"present":c.present}));
                    }
                }
                dynamic.sort_by_key(|v| v["id"].as_u64().unwrap());
            }
            for a in runtime
                .actors
                .iter()
                .filter_map(|a| a.slime.as_ref()?.abilities.as_ref())
                .flat_map(|r| &r.pool)
            {
                if let Some(r) = &a.registration {
                    let id = r.component;
                    let c = &s.components[id];
                    dynamic.push(json!({"id":id,"hierarchy":c.hierarchy,"destroyed":c.destroyed,"present":c.present}));
                }
            }
            dynamic.sort_by_key(|v| v["id"].as_u64().unwrap());
            frame["registration"] = json!({"added":s.added,"prefabs_added":s.prefabs_added,"registering_prefab":s.registering_prefab,"pending":s.pending.items[..s.pending.length],"live":s.live,"hitstop":s.hitstop,"dynamic":dynamic});
        }
        if let Some(spells) = &runtime.spells {
            frame["spells"] = json!(spells);
        }
        if let Some(combat) = &runtime.combat {
            frame["combat"] = json!(combat);
        }
        if let Some(binding) = &runtime.actor_ropes {
            frame["actor_ropes"] = json!(binding);
        }
        if q["routing_checks"].as_bool() == Some(true) {
            frame["routing_checks"] = json!(runtime.actors.iter().map(|actor| {
                let Some(slime) = actor.slime.as_ref() else {return Value::Null};
                let state = slime.abilities.as_ref().unwrap();
                let value = |r: Result<bool, &'static str>| match r { Ok(v) => json!(v), Err(e) => json!({"error":e}) };
                Value::Array((0..=state.slots.len()).map(|i| json!({"ready":value(state.ready(i)),"castable":value(state.castable_for_player(i,actor.controller.physics.motion.grounded,runtime.scheduler.players.iter().find(|p|Some(p.id)==actor.body.holder).is_some_and(|p|p.invisible)))})).collect::<Vec<_>>())
            }).collect::<Vec<_>>());
        }
        if q["player_state"].as_bool() == Some(true) {
            frame["players"] = json!(runtime.scheduler.players);
        }
        if scripted {
            frame["effects"] = json!(effects);
        }
        if let Err(error) = result {
            return json!({"error":error,"action":completed,"completed":if sink.is_some(){json!(completed)}else{json!(frames)},"failed":frame});
        }
        if let Some(sink) = &mut sink {
            sink(completed, &frame);
        } else {
            frames.push(frame);
        }
        completed += 1;
    }
    if sink.is_some() {
        json!({"completed":completed})
    } else {
        json!(frames)
    }
}
