use bopl_sim::{
    fixed::Fix,
    geometry::RoundedRect,
    physics::{self, contact},
    vector::Vec2,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::io::{self, BufRead, BufWriter, Write};
fn read<T: DeserializeOwned>(v: &Value, k: &str) -> T {
    serde_json::from_value(v[k].clone()).unwrap()
}
fn evaluate(q: Value) -> Value {
    let contact = match q["operation"].as_str().unwrap() {
        "dynamics_player_life" => {
            let mut state: bopl_sim::player_life::Book = read(&q, "state");
            let mut events: Vec<bopl_sim::player_life::Death> = read(&q["state"], "events");
            let mut frames = Vec::new();
            for a in q["actions"].as_array().unwrap() {
                if let Some(event) = state.kill(
                    read(a, "victim"),
                    read(a, "killer"),
                    read(a, "timestamp"),
                    read(a, "cause"),
                ) {
                    events.push(event);
                }
                frames.push(json!({"players": state.players, "registered_kills": state.registered_kills, "events": events}));
            }
            return json!(frames);
        }
        "dynamics_curve" => {
            let curve: bopl_sim::curve::Curve = read(&q, "curve");
            return json!(
                read::<Vec<bopl_sim::fixed::Fix>>(&q, "times")
                    .into_iter()
                    .map(|t| curve.evaluate(t))
                    .collect::<Vec<_>>()
            );
        }
        "dynamics_beam_hitbox" => {
            let config = read(&q, "config");
            let mut hitbox = bopl_sim::beam::hitbox::Hitbox::default();
            let mut states = Vec::new();
            for action in q["actions"].as_array().unwrap() {
                if action["operation"] == "initialize" {
                    hitbox.initialize(config);
                } else {
                    hitbox.set_dimensions(config, &read(action, "dimensions"));
                }
                states.push(hitbox.clone());
            }
            return json!(states);
        }
        "dynamics_beam_effects" => {
            let config = read(&q, "config");
            let mut targets = read::<Vec<bopl_sim::beam::effects::Target>>(&q, "targets");
            let mut states = Vec::new();
            for impacts in read::<Vec<Vec<bopl_sim::beam::effects::Impact>>>(&q, "frames") {
                bopl_sim::beam::effects::apply(&config, &impacts, &mut targets);
                states.push(targets.clone());
            }
            return json!(states);
        }
        "dynamics_beam_capture" => {
            let mut scene: physics::scene::Scene = read(&q, "scene");
            let mut beams: bopl_sim::beam::system::System = read(&q, "beams");
            let step = beams.simulate(&mut scene, read(&q, "dt"), &[]);
            scene.objects.sort_by_key(|o| o.collider.kind());
            return json!({"scene": scene, "beams": beams, "callbacks": step.physics.callbacks});
        }
        "dynamics_beam_scene" => {
            let mut scene: physics::scene::Scene = read(&q, "scene");
            let next_id = scene.objects.iter().map(|o| o.id).max().unwrap_or(0) + 1;
            let mut beams = bopl_sim::beam::system::System::new(
                read(&q, "config"),
                read(&q, "targets"),
                next_id,
            );
            let mut states = Vec::new();
            for frame in q["frames"].as_array().unwrap() {
                let input = read::<Vec<bopl_sim::beam::graph::Beam>>(frame, "beams");
                if let Some(slots) = frame["destroy_slots"].as_array() {
                    for slot in slots {
                        beams.destroy_slot(&mut scene, slot.as_u64().unwrap() as usize);
                    }
                }
                let step = beams.simulate(&mut scene, read(frame, "dt"), &input);
                scene.objects.sort_by_key(|o| o.collider.kind());
                states.push(json!({"scene": scene, "graph": step.graph, "pool": beams.pool, "targets": beams.targets,
                    "callbacks": step.physics.callbacks}));
            }
            return json!(states);
        }
        "dynamics_beam_graph" => {
            let mut graph = bopl_sim::beam::graph::Graph::new(read(&q, "config"));
            let mut scene: physics::scene::Scene = read(&q, "scene");
            let mut output = Vec::new();
            for frame in read::<Vec<Vec<bopl_sim::beam::graph::Beam>>>(&q, "frames") {
                output.push(graph.step(&frame, &mut |p, d, length, mask| {
                    scene.raycast(p, d, length, mask)
                }));
            }
            return json!(output);
        }
        "dynamics_beam_colors" => {
            let a: bopl_sim::beam::color::Colors = read(&q, "a");
            return json!(a.mix(read(&q, "b")));
        }
        "dynamics_beam_lines" => {
            return json!(bopl_sim::beam::intersection::lines(
                read(&q, "a"),
                read(&q, "b")
            ));
        }
        "dynamics_beam_intersection" => {
            return json!(bopl_sim::beam::intersection::beams(
                read(&q, "a"),
                read(&q, "b"),
                read(&q, "minimize_b")
            ));
        }
        "dynamics_rope" => {
            let mut rope: physics::rope::Rope = read(&q, "rope");
            let config = read(&q, "config");
            let rectangles: Vec<RoundedRect> = read(&q, "rectangles");
            let mut output = Vec::new();
            for a in q["actions"].as_array().unwrap() {
                let result = match a["op"].as_str().unwrap() {
                    "step" => {
                        rope.step(config, &rectangles);
                        Value::Null
                    }
                    "bounds" => json!(rope.bounds()),
                    "spawn" => json!(rope.spawn(read(a, "point"))),
                    "reel" => json!(rope.reel(read(a, "minimum"), false)),
                    "reverse_reel" => json!(rope.reel(read(a, "minimum"), true)),
                    "mass" => {
                        rope.set_mass(read(a, "value"), read(a, "top"), read(a, "scale"));
                        Value::Null
                    }
                    "detach" => {
                        rope.detach(read(a, "top"));
                        Value::Null
                    }
                    _ => panic!("unknown rope action"),
                };
                output.push(json!({"rope":rope,"result":result}));
            }
            return json!(output);
        }
        "dynamics_scene" => {
            let mut scene: physics::scene::Scene = read(&q, "scene");
            let mut steps = Vec::new();
            for (tick, dt) in read::<Vec<Fix>>(&q, "steps").into_iter().enumerate() {
                if let Some(commands) = q["commands"][tick].as_array() {
                    for command in commands {
                        let id: u32 = read(command, "id");
                        match command["op"].as_str().unwrap() {
                            "attach" => scene.attach(id, read(command, "root")),
                            "detach" => scene.detach(id),
                            "destroy" => scene.destroy(id),
                            "force" => {
                                scene
                                    .objects
                                    .iter_mut()
                                    .find(|o| o.id == id)
                                    .unwrap()
                                    .body
                                    .force += read::<Vec2>(command, "force")
                            }
                            _ => panic!("unknown scene command"),
                        }
                    }
                }
                let report = scene.step(dt);
                steps.push(json!({"scene":scene,"report":report}));
            }
            return json!(steps);
        }
        "dynamics_circle_circle" => contact::circle_circle(read(&q, "a"), read(&q, "b")),
        "dynamics_circle_box" => contact::circle_box(read(&q, "a"), read(&q, "b")),
        "dynamics_circle_rounded" => contact::circle_rounded(read(&q, "a"), read(&q, "b")),
        "dynamics_box_box" => contact::box_box(read(&q, "a"), read(&q, "b")),
        "dynamics_box_rounded" => contact::rounded_rounded(
            RoundedRect {
                inner: read(&q, "a"),
                radius: Fix::ZERO,
            },
            read(&q, "b"),
        ),
        "dynamics_rounded_rounded" => contact::rounded_rounded(read(&q, "a"), read(&q, "b")),
        "dynamics_integrate" => {
            let mut body: physics::Body = read(&q, "body");
            let translation = body.integrate(read(&q, "dt"), read(&q, "gravity"));
            return json!({"body":body,"translation":translation});
        }
        "dynamics_resolve" => {
            let mut a = read(&q, "a");
            let mut b = read(&q, "b");
            physics::resolve(
                read(&q, "contact"),
                &mut a,
                &mut b,
                read::<Vec2>(&q, "pos_a"),
                read::<Vec2>(&q, "pos_b"),
            );
            return json!({"a":a,"b":b});
        }
        "dynamics_correction" => {
            return json!(physics::positional_correction(
                read(&q, "contact"),
                read(&q, "inverse_a"),
                read(&q, "inverse_b"),
                read(&q, "full")
            ));
        }
        _ => panic!("unknown operation"),
    };
    json!(contact)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::panic::set_hook(Box::new(|_| {}));
    let mut out = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let query = serde_json::from_str(&line?)?;
        let result = std::panic::catch_unwind(|| evaluate(query)).unwrap_or_else(|panic| {
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            if message.contains("division by zero") {
                json!({"error":"division_by_zero"})
            } else if matches!(
                message,
                "negative squared length: Domain"
                    | "normalized vector outside cosine domain: Domain"
                    | "negative black hole mass magnitude: Domain"
            ) {
                json!({"error":"domain"})
            } else {
                json!({"error":"rust_panic", "message":message})
            }
        });
        writeln!(out, "{result}")?;
    }
    Ok(())
}
