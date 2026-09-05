use bopl_sim::{
    fixed::Fix,
    geometry::{Circle, OrientedBox, RoundedRect},
    surface::Surface,
    vector::Vec2,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{self, BufRead, BufWriter, Write};

#[derive(Deserialize)]
struct Query {
    operation: String,
    center: [i64; 2],
    extent: [i64; 2],
    origin: [i64; 2],
    angle: i64,
    direction: i64,
    radius: i64,
    #[serde(default)]
    player_radius: i64,
    #[serde(default)]
    local: i64,
    #[serde(default)]
    speed: i64,
}
fn vector(raw: [i64; 2]) -> Vec2 {
    Vec2::new(Fix(raw[0]), Fix(raw[1]))
}

fn evaluate(query: Query) -> Result<Value, String> {
    let center = vector(query.center);
    let extent = vector(query.extent);
    let origin = vector(query.origin);
    let basis = Vec2::angle(Fix(query.angle));
    let direction = Vec2::angle(Fix(query.direction));
    if extent.x <= Fix::ZERO || extent.y <= Fix::ZERO || query.radius < 0 {
        return Err("invalid geometry".into());
    }
    let inner = OrientedBox {
        center,
        right: (Vec2::RIGHT * extent.x).complex_mul(basis),
        up: (Vec2::UP * extent.y).complex_mul(basis),
        inverse_extents: Vec2::new(Fix::ONE / extent.x, Fix::ONE / extent.y),
    };
    let circle = Circle {
        center,
        radius: Fix(query.radius),
    };
    let rounded = RoundedRect {
        inner,
        radius: Fix(query.radius),
    };
    let surface = Surface {
        center,
        right: basis,
        up: basis.perp(),
        extents: extent,
        radius: Fix(query.radius),
    };
    let point_result = |result: Result<Vec2, bopl_sim::fixed::MathError>| match result {
        Ok(point) => json!([point.x.0, point.y.0]),
        Err(bopl_sim::fixed::MathError::DivisionByZero) => json!({"error":"division_by_zero"}),
        Err(_) => json!({"error":"domain"}),
    };
    let hit = match query.operation.as_str() {
        "surface_normal" => return Ok(point_result(Ok(surface.normal(origin)))),
        "surface_closest" => return Ok(point_result(surface.closest_point(origin))),
        "surface_position" => {
            return Ok(point_result(Ok(
                surface.position(Fix(query.local), Fix(query.player_radius))
            )));
        }
        "surface_move" => {
            return Ok(json!(
                surface
                    .move_unblocked(
                        Fix(query.local),
                        Fix(query.player_radius),
                        Fix(query.speed),
                        Fix::ONE / Fix::from_int(60)
                    )
                    .0
            ));
        }
        "surface_local" => {
            return Ok(
                match surface.local_position(origin, Fix(query.player_radius)) {
                    Ok(value) => json!(value.0),
                    Err(bopl_sim::fixed::MathError::DivisionByZero) => {
                        json!({"error":"division_by_zero"})
                    }
                    Err(_) => json!({"error":"domain"}),
                },
            );
        }
        "circle_ray" => circle.raycast(origin, direction),
        "box_ray" => inner.raycast(origin, direction),
        "rounded_ray" => rounded.raycast(origin, direction),
        "circle_overlap" => return Ok(json!(circle.overlaps(origin))),
        "box_overlap" => return Ok(json!(inner.overlaps(origin))),
        "rounded_overlap" => return Ok(json!(rounded.overlaps(origin))),
        "closest" => {
            return Ok(match rounded.closest_point(origin) {
                Ok(point) => json!([point.x.0, point.y.0]),
                Err(bopl_sim::fixed::MathError::DivisionByZero) => {
                    json!({"error":"division_by_zero"})
                }
                Err(_) => json!({"error":"domain"}),
            });
        }
        name => return Err(format!("unknown operation: {name}")),
    };
    Ok(match hit {
        Some(hit) => json!({"hit":true,"near":hit.near.0,"far":hit.far.0}),
        None => json!({"hit":false}),
    })
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let result = evaluate(serde_json::from_str(&line?)?)?;
        writeln!(output, "{result}")?;
    }
    Ok(())
}
