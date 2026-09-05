use bopl_sim::{fixed::Fix, random::Random};
use serde::Deserialize;
use std::io::{self, BufRead, BufWriter, Write};
#[derive(Deserialize)]
struct Query {
    seed: u32,
    operations: Vec<Operation>,
}
#[derive(Deserialize)]
struct Operation {
    kind: String,
    min: i64,
    max: i64,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let q: Query = serde_json::from_str(&line?)?;
        let mut rng = Random::new(q.seed);
        let mut results = Vec::new();
        for op in q.operations {
            let value = match op.kind.as_str() {
                "int" => i64::from(rng.int_range(op.min as i32, op.max as i32)),
                "uint" => i64::from(rng.uint_range(op.min as u32, op.max as u32)),
                "fix" => rng.fix_range(Fix(op.min), Fix(op.max)).0,
                _ => return Err("Unknown RNG operation".into()),
            };
            results.push([value, i64::from(rng.state)]);
        }
        writeln!(out, "{}", serde_json::to_string(&results)?)?;
    }
    Ok(())
}
