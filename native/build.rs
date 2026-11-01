//! Locates the game-derived data the simulator embeds at compile time.
//!
//! The fixed-point lookup tables, the map and prefab export, the reset states
//! and three recorded replays are derived from the game and live in the
//! private `game/` submodule; `native/data/*` and `native/fixtures` are
//! symlinks into it. Point `BOPL_GAME_DATA` at a directory with the same
//! `data/` and `fixtures/` layout to build from somewhere else.
//!
//! Without the submodule the crate still builds: every missing file is
//! embedded as an empty placeholder and `cfg(bopl_game_data)` is unset, so
//! the tests that need the data are skipped and the first use of it panics
//! with a message naming the file (see `src/game_data.rs`).
use std::{env, fs, path::PathBuf};

/// (environment variable read by `include_*!(env!(..))`, path under the data root)
const FILES: &[(&str, &str)] = &[
    ("BOPL_DATA_SINLUT", "data/sinlut.bin"),
    ("BOPL_DATA_TANLUT", "data/tanlut.bin"),
    ("BOPL_DATA_ROLLOUT", "data/rollout.json"),
    ("BOPL_DATA_LEVEL1", "data/level1-movement.json"),
    ("BOPL_DATA_LEVEL1_P1", "data/level1-movement-1.json"),
    ("BOPL_DATA_LEVEL1_P3", "data/level1-movement-3.json"),
    ("BOPL_DATA_LEVEL1_P4", "data/level1-movement-4.json"),
    ("BOPL_DATA_LEVEL1_P5", "data/level1-movement-5.json"),
    ("BOPL_DATA_LEVEL1_P6", "data/level1-movement-6.json"),
    ("BOPL_FIXTURE_REWIND_ARC", "fixtures/rewind-arrival-on-corner-arc.rep"),
    ("BOPL_FIXTURE_MAGNET_DRILL", "fixtures/magnet-held-slime-casts-drill.rep"),
    ("BOPL_FIXTURE_SPIKE_ROCK", "fixtures/spike-shoves-duplicated-rock.rep"),
];

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = env::var_os("BOPL_GAME_DATA").map_or(manifest, PathBuf::from);
    let placeholder = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("missing-game-data");
    fs::write(&placeholder, b"").unwrap();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BOPL_GAME_DATA");
    println!("cargo:rustc-check-cfg=cfg(bopl_game_data)");

    let mut missing = Vec::new();
    for (var, rel) in FILES {
        let path = root.join(rel);
        // A dangling symlink (submodule not checked out) is "missing" too.
        let present = fs::metadata(&path).map(|m| m.is_file()).unwrap_or(false);
        println!("cargo:rerun-if-changed={}", path.display());
        let path = if present { path } else { missing.push(*rel); placeholder.clone() };
        println!("cargo:rustc-env={var}={}", path.display());
    }
    if missing.is_empty() {
        println!("cargo:rustc-cfg=bopl_game_data");
    } else {
        println!(
            "cargo:warning=building without game-derived data ({} of {} files missing, e.g. {}); \
             the simulator panics when it first needs them. Check out the game/ submodule or set BOPL_GAME_DATA.",
            missing.len(),
            FILES.len(),
            missing[0]
        );
    }
}
