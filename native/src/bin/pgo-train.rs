//! The profile run of the PGO build (tools/pgo_build.sh): every BOPLREP2 game
//! in a directory re-stepped the way the trainer steps it, 4-tick decisions
//! through `step_sampled(actions, 3, 1)` on each decision's first-tick inputs
//! (the recordings hold an input for a whole decision). The games are trained
//! policies' (native/pgo/corpus.tar.xz), so the profile is the state
//! distribution training visits.
//!
//!     pgo-train <dir> [warmup=240] [limit=10800]
use bopl_sim::rollout::{RolloutWorld, replay};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("usage: pgo-train <dir> [warmup] [limit]");
    let warmup = args.get(2).map_or(240, |s| s.parse().expect("warmup"));
    let limit = args.get(3).map_or(10800, |s| s.parse().expect("limit"));
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .expect("read corpus")
        .map(|entry| entry.expect("corpus entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rep"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no .rep games in {dir}");
    let mut world = RolloutWorld::new();
    let (mut games, mut decisions) = (0, 0);
    for path in &paths {
        let raw = std::fs::read(path).expect("read replay");
        let replay::Replay { config, ticks } = replay::decode(&raw, warmup, limit).expect("decode replay");
        world.reset(&config).expect("reset");
        for actions in ticks.iter().skip(warmup as usize).step_by(4) {
            world.step_sampled(actions, 3, 1).expect("step");
            decisions += 1;
            if world.terminal {
                break;
            }
        }
        games += 1;
    }
    println!("pgo-train: {games} games, {decisions} decisions");
}
