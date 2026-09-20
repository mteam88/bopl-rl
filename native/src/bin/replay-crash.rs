//! Re-step a BOPLREP2 recording tick by tick in the Rust sim. Made for the
//! crash-*.rep files bopl_rl_step writes when a step panics: the recording
//! ends on the panicking tick, so this reproduces it with a backtrace.
//!
//!     RUST_BACKTRACE=1 cargo run --release --bin replay-crash -- <file.rep> [warmup] [limit]
//!
//! Release LTO inlines the whole tick into game_tick; a dev build
//! (`cargo run --bin replay-crash`) keeps the frames and is fast enough.
//! The archive header has no warmup or episode limit; they default to the
//! values in a crash file's name (crash-seedS-mapM-warmupW-limitL-pidP.rep),
//! else to the training reset (240, 7200).
use bopl_sim::rollout::{RolloutWorld, replay};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: replay-crash <file.rep> [warmup] [limit]");
    let raw = std::fs::read(path).expect("read replay");
    let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy();
    let from_name = |key: &str| {
        name.split('-')
            .find_map(|part| part.strip_prefix(key))
            .and_then(|digits| digits.trim_end_matches(".rep").parse::<u32>().ok())
    };
    let warmup = args.get(2).and_then(|s| s.parse().ok()).or_else(|| from_name("warmup")).unwrap_or(240);
    let limit = args.get(3).and_then(|s| s.parse().ok()).or_else(|| from_name("limit")).unwrap_or(7200);
    let replay::Replay { config, ticks } = replay::decode(&raw, warmup, limit).expect("decode replay");
    let players = config.players as usize;
    let loadouts: Vec<String> = config.player[..players]
        .iter()
        .map(|p| p.abilities[..config.slots as usize].iter().map(|a| a.to_string()).collect::<Vec<_>>().join(","))
        .collect();
    println!(
        "seed {} map {} stocks {} warmup {warmup} limit {limit} ticks {} loadouts {}",
        config.seed,
        config.map,
        config.stocks,
        ticks.len(),
        loadouts.join(" vs ")
    );

    let mut world = RolloutWorld::new();
    world.reset(&config).expect("reset");
    // The recording begins with the warmup's neutral frames, which reset
    // already simulated.
    for (tick, actions) in ticks.iter().enumerate().skip(warmup as usize) {
        if tick + 1 == ticks.len() {
            println!("stepping the final recorded tick {tick}");
        }
        world.step(actions, 1).expect("step");
        if world.terminal {
            println!("episode ended at tick {tick}");
            break;
        }
    }
    println!("ok: replayed without a panic");
}
