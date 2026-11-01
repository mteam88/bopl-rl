//! Game-derived data embedded at compile time (see `build.rs`).
//!
//! It comes from the private `game/` submodule. When it was missing at build
//! time each file is embedded as an empty placeholder, and [`require`] turns
//! the first use into a panic that says what is missing.

/// Whether every game-derived file was present when the crate was built.
pub const AVAILABLE: bool = cfg!(bopl_game_data);

/// Returns `data`, or panics if `name` was embedded as an empty placeholder.
#[track_caller]
pub fn require<'a, T: AsRef<[u8]> + ?Sized>(data: &'a T, name: &str) -> &'a T {
    if data.as_ref().is_empty() {
        missing(name);
    }
    data
}

#[cold]
#[track_caller]
pub fn missing(name: &str) -> ! {
    panic!(
        "{name} is game-derived data from the private game/ submodule and was not present \
         when this library was built; check out the submodule (or set BOPL_GAME_DATA) and rebuild"
    )
}
