use bopl_sim::{
    fixed::Fix,
    scheduler::{Event, Result, State},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
fn read<T: serde::de::DeserializeOwned>(v: &Value, k: &str) -> T {
    serde_json::from_value(v[k].clone()).unwrap()
}
fn notify(
    state: &mut State,
    event: Event,
    scripts: &Value,
    calls: &mut BTreeMap<String, usize>,
    events: &mut Vec<Event>,
) -> Result<()> {
    let key = format!(
        "{}:{}",
        event.kind.as_str(),
        event.id.map_or("*".into(), |i| i.to_string())
    );
    events.push(event);
    let count = calls.entry(key.clone()).or_default();
    let index = *count;
    *count += 1;
    if let Some(sequence) = scripts[&key].as_array().and_then(|s| s.get(index)) {
        for action in sequence.as_array().unwrap() {
            command(state, action, scripts, calls, events)?;
        }
    }
    Ok(())
}
fn command(
    s: &mut State,
    a: &Value,
    scripts: &Value,
    calls: &mut BTreeMap<String, usize>,
    events: &mut Vec<Event>,
) -> Result<()> {
    match a["op"].as_str().unwrap() {
        "tick" => s.tick(read(a, "dt"), &mut |s, e| {
            notify(s, e, scripts, calls, events)
        })?,
        "register" => s.register(read(a, "id")),
        "begin" => s.begin_prefab(),
        "end" => s.end_prefab()?,
        "reinit" => s.reinit(),
        "pre_level" => s.pre_level(),
        "post_level" => s.loaded = true,
        "destroy_object" => s.destroy_object(read(a, "id"))?,
        "destroy_mono" => s.destroy_mono(read(a, "id")),
        "sort" => s.sort_pending()?,
        "hitstop" => s.hitstop = read(a, "value"),
        "scale" => s.time_scale = read::<Fix>(a, "value"),
        "enable" | "mark" | "hierarchy" => {
            let id = read::<Option<usize>>(a, "id").ok_or("null_reference")?;
            let c = &mut s.components[id];
            match a["op"].as_str().unwrap() {
                "enable" => c.enabled = read(a, "value"),
                "mark" => c.destroyed = read(a, "value"),
                _ => c.hierarchy = read(a, "value"),
            }
        }
        "active" => {
            let id = read::<Option<usize>>(a, "id").ok_or("null_reference")?;
            if !s.nodes[id].present {
                return Err("null_reference");
            }
            s.nodes[id].active = read(a, "value");
        }
        "live_add" => s.live.push(read(a, "id")),
        "live_remove" => {
            let i = read::<usize>(a, "index");
            if i >= s.live.len() {
                return Err("argument_out_of_range");
            }
            s.live.remove(i);
        }
        "live_clear" => s.live.clear(),
        "input" => s.inputs[read::<usize>(a, "id")].jump = read(a, "value"),
        _ => panic!("Unknown scheduler command"),
    }
    Ok(())
}
pub fn evaluate(q: &Value) -> Value {
    let mut state: State = read(q, "state");
    let mut calls = BTreeMap::new();
    let mut frames = Vec::new();
    for a in q["actions"].as_array().unwrap() {
        let mut events = Vec::new();
        let result = command(&mut state, a, &q["scripts"], &mut calls, &mut events);
        let frame = json!({"state":state,"events":events,"calls":calls});
        if let Err(error) = result {
            return json!({"error":error,"action":frames.len(),"completed":frames,"failed":frame});
        }
        frames.push(frame);
    }
    json!(frames)
}
