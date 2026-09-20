//! Re-step a BOPLREP2 recording tick by tick in the Rust sim. Made for the
//! crash-*.rep files bopl_rl_step writes when a step panics: the recording
//! ends on the panicking tick, so this reproduces it with a backtrace.
//!
//!     RUST_BACKTRACE=1 cargo run --release --bin replay-crash -- <file.rep> [warmup] [limit]
//!
//! The archive header has no warmup or episode limit; they default to the
//! values in a crash file's name (crash-seedS-mapM-warmupW-limitL-pidP.rep),
//! else to the training reset (240, 7200).
use bopl_sim::rollout::{CAction, CConfig, CPlayerConfig, RolloutWorld};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: replay-crash <file.rep> [warmup] [limit]");
    let raw = std::fs::read(path).expect("read replay");
    assert!(raw.starts_with(b"BOPLREP2"), "not a BOPLREP2 archive");
    let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy();
    let from_name = |key: &str| {
        name.split('-')
            .find_map(|part| part.strip_prefix(key))
            .and_then(|digits| digits.trim_end_matches(".rep").parse::<u32>().ok())
    };
    let warmup = args.get(2).and_then(|s| s.parse().ok()).or_else(|| from_name("warmup")).unwrap_or(240);
    let limit = args.get(3).and_then(|s| s.parse().ok()).or_else(|| from_name("limit")).unwrap_or(7200);

    let mut at = 136;
    let u8_at = |at: &mut usize| {
        *at += 1;
        raw[*at - 1]
    };
    let seed = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
    at += 4;
    let map = u8_at(&mut at) as u32;
    let mode = u8_at(&mut at) as u32;
    let stocks = i32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as u32;
    at += 4;
    let players = u8_at(&mut at) as usize;
    let mut config = CConfig { seed, map, players: players as u32, stocks, mode, warmup_ticks: warmup, max_episode_ticks: limit, ..CConfig::default() };
    for player in 0..players {
        let _id = u8_at(&mut at);
        let color = u8_at(&mut at) as u32;
        let team = u8_at(&mut at) as u32;
        let keyboard = u8_at(&mut at) as u32;
        let slots = u8_at(&mut at) as usize;
        let mut abilities = [0u32; 3];
        for ability in abilities.iter_mut().take(slots) {
            *ability = u16::from_le_bytes(raw[at..at + 2].try_into().unwrap()) as u32;
            at += 2;
        }
        config.slots = slots as u32;
        config.player[player] = CPlayerConfig { color, team, keyboard, abilities };
    }
    let ticks = i32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as usize;
    at += 4;
    let loadouts: Vec<String> = config.player[..players]
        .iter()
        .map(|p| p.abilities[..config.slots as usize].iter().map(|a| a.to_string()).collect::<Vec<_>>().join(","))
        .collect();
    println!("seed {seed} map {map} stocks {stocks} warmup {warmup} limit {limit} ticks {ticks} loadouts {}", loadouts.join(" vs "));

    let mut world = RolloutWorld::new();
    world.reset(&config).expect("reset");
    // The recording begins with the warmup's neutral frames, which reset
    // already simulated.
    for tick in warmup as usize..ticks {
        let mut actions = vec![CAction::default(); players];
        for (player, action) in actions.iter_mut().enumerate() {
            let record = at + (tick * players + player) * 8;
            action.angle = raw[record + 4];
            action.buttons = u16::from_le_bytes(raw[record + 5..record + 7].try_into().unwrap());
        }
        if tick + 1 == ticks {
            println!("stepping the final recorded tick {tick}");
        }
        world.step(&actions, 1).expect("step");
        if world.terminal {
            println!("episode ended at tick {tick}");
            break;
        }
    }
    println!("ok: replayed without a panic");
}
