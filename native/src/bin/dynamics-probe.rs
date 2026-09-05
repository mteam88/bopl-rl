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
