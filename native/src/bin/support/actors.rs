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
        }
        if let Some(inputs) = action.get("routing_inputs") {
            for (actor, input) in runtime.actors.iter_mut().zip(inputs.as_array().unwrap()) {
                let state = actor.slime.as_mut().unwrap().abilities.as_mut().unwrap();
                state.buttons = serde_json::from_value(input["buttons"].clone()).unwrap();
                state.can_use = input["can_use"].as_bool().unwrap();
            }
        }
        runtime.scheduler.time_scale = serde_json::from_value(action["scale"].clone()).unwrap();
        for (i, enabled) in action["enabled"].as_array().unwrap().iter().enumerate() {
            runtime.scheduler.components[i].enabled = enabled.as_bool().unwrap();
        }
        let mut command = |schedule: &mut bopl_sim::scheduler::State,
                           actors: &mut [bopl_sim::actors::Actor],
                           scene: &mut bopl_sim::physics::scene::Scene,
                           command: &Value| {
            effects.push(command.clone());
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
                        &mut runtime.scheduler,
                        &mut runtime.actors,
                        &mut runtime.scene,
                        c,
                    )
                })
            })
            .and_then(|()| {
                runtime.tick_with(
                    Fix(action["dt"].as_i64().unwrap()),
                    &mut report,
                    &mut |schedule, actors, scene, phase| {
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
                                command(schedule, actors, scene, c)?;
                            }
                        }
                        Ok(())
                    },
                )
            });
        let mut scene = runtime.scene.clone();
        scene.objects.sort_by_key(|o| o.id);
        let mut frame = json!({"actors":runtime.actors,"scene":scene,"report":report,"ticks":runtime.scheduler.ticks,
            "time":runtime.scheduler.time,"level_time":runtime.scheduler.level_time,"inputs":runtime.scheduler.inputs});
        if q["routing_checks"].as_bool() == Some(true) {
            frame["routing_checks"] = json!(runtime.actors.iter().map(|actor| {
                let state = actor.slime.as_ref().unwrap().abilities.as_ref().unwrap();
                let value = |r: Result<bool, &'static str>| match r { Ok(v) => json!(v), Err(e) => json!({"error":e}) };
                (0..=state.slots.len()).map(|i| json!({"ready":value(state.ready(i)),"castable":value(state.castable_for_player(i,actor.controller.physics.motion.grounded,runtime.scheduler.players.iter().find(|p|Some(p.id)==actor.body.holder).is_some_and(|p|p.invisible)))})).collect::<Vec<_>>()
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
