use bopl_sim::world::{Action, World};
use std::io::{self, BufRead, BufWriter, Write};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = World::level1();
    let mut output = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let raw: Vec<[u8; 2]> = serde_json::from_str(&line?)?;
        let actions: Vec<_> = raw
            .into_iter()
            .map(|[angle, buttons]| Action { angle, buttons })
            .collect();
        world.step(&actions)?;
        writeln!(output, "{}", serde_json::to_string(&world)?)?;
        if world.terminal() {
            break;
        }
    }
    Ok(())
}
