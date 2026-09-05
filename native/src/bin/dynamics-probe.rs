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
            } else {
                json!({"error":"rust_panic", "message":message})
            }
        });
        writeln!(out, "{result}")?;
    }
    Ok(())
}
