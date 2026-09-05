use bopl_sim::{
    actors::{Report, Runtime},
    fixed::Fix,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
pub fn evaluate(q: &Value) -> Value {
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
        runtime.scheduler.time_scale = serde_json::from_value(action["scale"].clone()).unwrap();
        for (i, enabled) in action["enabled"].as_array().unwrap().iter().enumerate() {
            runtime.scheduler.components[i].enabled = enabled.as_bool().unwrap();
        }
        let mut command = |actors: &mut [bopl_sim::actors::Actor],
                           scene: &mut bopl_sim::physics::scene::Scene,
                           command: &Value| {
            effects.push(command.clone());
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
                commands
                    .as_array()
                    .unwrap()
                    .iter()
                    .try_for_each(|c| command(&mut runtime.actors, &mut runtime.scene, c))
            })
            .and_then(|()| {
                runtime.tick_with(
                    Fix(action["dt"].as_i64().unwrap()),
                    &mut report,
                    &mut |_, actors, scene, phase| {
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
                                command(actors, scene, c)?;
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
        if scripted {
            frame["effects"] = json!(effects);
        }
        if let Err(error) = result {
            return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
        }
        frames.push(frame);
    }
    json!(frames)
}
