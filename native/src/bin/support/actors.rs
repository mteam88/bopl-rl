use bopl_sim::{
    actors::{Report, Runtime},
    fixed::Fix,
};
use serde_json::{Value, json};
pub fn evaluate(q: &Value) -> Value {
    let mut runtime: Runtime = serde_json::from_value(q["runtime"].clone()).unwrap();
    runtime.synchronize();
    let mut frames = Vec::new();
    for action in q["actions"].as_array().unwrap() {
        let mut report = Report::default();
        for (i, input) in action["inputs"].as_array().unwrap().iter().enumerate() {
            runtime.actors[i].input = serde_json::from_value(input["direction"].clone()).unwrap();
            runtime.scheduler.inputs[i].jump = input["jump"].as_bool().unwrap();
        }
        runtime.scheduler.time_scale = serde_json::from_value(action["scale"].clone()).unwrap();
        for (i, enabled) in action["enabled"].as_array().unwrap().iter().enumerate() {
            runtime.scheduler.components[i].enabled = enabled.as_bool().unwrap();
        }
        let result = runtime.tick(Fix(action["dt"].as_i64().unwrap()), &mut report);
        let mut scene = runtime.scene.clone();
        scene.objects.sort_by_key(|o| o.id);
        let frame = json!({"actors":runtime.actors,"scene":scene,"report":report,"ticks":runtime.scheduler.ticks,
            "time":runtime.scheduler.time,"level_time":runtime.scheduler.level_time,"inputs":runtime.scheduler.inputs});
        if let Err(error) = result {
            return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
        }
        frames.push(frame);
    }
    json!(frames)
}
