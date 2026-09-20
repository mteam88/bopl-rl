//! Random loadouts, random actions, many short episodes: the crash finder for
//! the multi-ability pool. Training only reaches a sim panic after ~10 minutes
//! (any-3, 2026-09-20); this reaches the same states in seconds and prints the
//! seed and loadouts to reproduce. Panics abort, as under bopl_pool_worker, so
//! the last line printed before the backtrace is the offending episode.
//!
//!     cargo run --release --bin loadout-fuzz -- [episodes] [seed] [map-count]
use bopl_sim::rollout::{CAction, CConfig, CPlayerConfig, RolloutWorld};

const ABILITY_COUNT: u32 = 39;
const EXCLUDED: [u32; 2] = [12, 15]; // ABILITY_PORT_GAPS.md partial ports

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let episodes: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let maps: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let pool: Vec<u32> = (1..ABILITY_COUNT).filter(|a| !EXCLUDED.contains(a)).collect();
    let mut rng = Lcg(seed);
    let mut world = RolloutWorld::new();
    for episode in 0..episodes {
        let mut loadouts = [[0u32; 3]; 2];
        for row in loadouts.iter_mut() {
            for slot in row.iter_mut() {
                *slot = pool[rng.below(pool.len() as u32) as usize];
            }
        }
        if rng.below(2) == 0 {
            loadouts[1] = loadouts[0];
        }
        let game_seed = rng.next().max(1);
        let map = rng.below(maps);
        let mut config = CConfig {
            seed: game_seed,
            map,
            players: 2,
            slots: 3,
            stocks: 3,
            warmup_ticks: 60,
            max_episode_ticks: 3600,
            ..CConfig::default()
        };
        for (player, row) in loadouts.iter().enumerate() {
            config.player[player] = CPlayerConfig { team: player as u32, abilities: *row, ..CPlayerConfig::default() };
        }
        println!(
            "episode {episode} seed {game_seed} map {map} loadouts {},{},{} vs {},{},{}",
            loadouts[0][0], loadouts[0][1], loadouts[0][2], loadouts[1][0], loadouts[1][1], loadouts[1][2]
        );
        world.reset(&config).expect("valid config");
        let mut actions = [CAction::default(); 2];
        // Bursty inputs: hold a button set for a few ticks so abilities charge,
        // release, and repeat, with the aim wandering.
        let mut hold = 0u32;
        while !world.terminal {
            if hold == 0 {
                hold = 1 + rng.below(12);
                for action in actions.iter_mut() {
                    action.angle = rng.next() as u8;
                    action.buttons = (rng.next() & 1023) as u16;
                }
            }
            hold -= 1;
            world.step_sampled(&actions, 4, 4).expect("step");
        }
    }
    println!("ok: {episodes} episodes without a panic");
}
