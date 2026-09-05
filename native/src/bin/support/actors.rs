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
        if let Some(inputs) = action.get("routing_inputs") {
            for (actor, input) in runtime.actors.iter_mut().zip(inputs.as_array().unwrap()) {
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
        let mut command = |actors: &mut [bopl_sim::actors::Actor],
                           context: &mut bopl_sim::actors::Context,
                           report: &mut Report,
                           command: &Value| {
            effects.push(command.clone());
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
                        .ok_or("missing_actor_object")?
                        .enabled = command["value"].as_bool().unwrap();
                    return Ok(());
                }
                "box_scale" => {
                    return actors[command["actor"].as_u64().unwrap() as usize].set_box_scale(
                        serde_json::from_value(command["value"].clone()).unwrap(),
                        scene,
                    );
                }
                "box_sync" => {
                    return actors[command["actor"].as_u64().unwrap() as usize].sync_box(scene);
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
                            scene: &mut runtime.scene,
                            actor_ropes: &mut runtime.actor_ropes,
                            combat: &mut runtime.combat,
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
                    Fix(action["dt"].as_i64().unwrap()),
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
            let dynamic:Vec<_>=runtime.combat.as_ref().and_then(|c|c.death_effects.as_ref()).into_iter().flat_map(|e|&e.instances).flat_map(|e|e.components[..2].iter()).map(|&id| {let c=&s.components[id];json!({"id":id,"hierarchy":c.hierarchy,"destroyed":c.destroyed,"present":c.present})}).collect();
            frame["registration"] = json!({"added":s.added,"prefabs_added":s.prefabs_added,"registering_prefab":s.registering_prefab,"pending":s.pending.items[..s.pending.length],"live":s.live,"hitstop":s.hitstop,"dynamic":dynamic});
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
