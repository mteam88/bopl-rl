//! C exports implementing the bopl_rl.h in-process ABI (the same one
//! BoplSimNative.so exports), so native/rl/bopl_rl.c can dlopen this cdylib
//! as a game backend. Calls for one handle are serialized by the client;
//! separate handles may run on different threads concurrently.
use super::{CAction, CConfig, CState, RolloutWorld};
use std::ffi::c_char;

const OK: i32 = 0;
const ARGUMENT: i32 = 1;
const GAME_ERROR: i32 = 3;
const ENDED: i32 = 4;

pub struct Handle {
    world: RolloutWorld,
    reset_pending: bool,
    reset_result: i32,
}

#[unsafe(no_mangle)]
pub extern "C" fn bopl_rl_abi_version() -> u32 {
    super::ABI_VERSION
}

/// One runtime hosts any number of worlds; the client never copies the .so.
#[unsafe(no_mangle)]
pub extern "C" fn bopl_rl_world_capacity() -> u32 {
    1 << 20
}

/// Mirrors bopl_rl.c's client-side default (seed 37, 3 slots, warmup 240,
/// 7200-tick episodes, Dash/Gust/Invisibility loadout).
///
/// # Safety
/// `config` must point to a BoplRlConfig or be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_default_config(config: *mut CConfig, players: u32) {
    let Some(config) = (unsafe { config.as_mut() }) else {
        return;
    };
    *config = CConfig {
        seed: 37,
        map: 0,
        players,
        slots: 3,
        stocks: 0,
        mode: 0,
        warmup_ticks: 240,
        max_episode_ticks: 7200,
        player: core::array::from_fn(|i| super::CPlayerConfig {
            color: (i % 4) as u32,
            team: i as u32,
            keyboard: 0,
            abilities: [2, 7, 19],
        }),
    };
}

/// # Safety
/// `error` must be valid for `capacity` bytes when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_open(
    _executable: *const c_char,
    _replay: *const c_char,
    _log_path: *const c_char,
    _timeout_ms: u32,
    error: *mut c_char,
    capacity: usize,
) -> *mut Handle {
    if !error.is_null() && capacity > 0 {
        unsafe { *error = 0 };
    }
    Box::into_raw(Box::new(Handle {
        world: RolloutWorld::new(),
        reset_pending: false,
        reset_result: OK,
    }))
}

/// The whole reset runs here (the client calls this on a helper thread and
/// joins in reset_finish); a Rust reset is fast enough that nothing overlaps.
///
/// # Safety
/// `world` must be a live handle from `bopl_rl_open`; `config` must point to a
/// BoplRlConfig.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_reset_begin(world: *mut Handle, config: *const CConfig) -> i32 {
    let Some(handle) = (unsafe { world.as_mut() }) else {
        return ARGUMENT;
    };
    if config.is_null() {
        handle.world.set_error("Null RL configuration");
        return ARGUMENT;
    }
    if handle.reset_pending {
        handle.world.set_error("RL reset already in progress");
        return ARGUMENT;
    }
    let config = unsafe { &*config };
    handle.reset_result = match handle.world.reset(config) {
        Ok(()) => OK,
        Err(message) => {
            handle.world.valid = false;
            handle.world.set_error(&message);
            GAME_ERROR
        }
    };
    handle.reset_pending = true;
    OK
}

/// # Safety
/// `world` must be a live handle from `bopl_rl_open`; `config` must point to a
/// BoplRlConfig.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_reset(world: *mut Handle, config: *const CConfig) -> i32 {
    let result = unsafe { bopl_rl_reset_begin(world, config) };
    if result != OK {
        return result;
    }
    unsafe { bopl_rl_reset_finish(world) }
}

/// # Safety
/// `world` must be a live handle from `bopl_rl_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_reset_finish(world: *mut Handle) -> i32 {
    let Some(handle) = (unsafe { world.as_mut() }) else {
        return ARGUMENT;
    };
    if !handle.reset_pending {
        handle.world.set_error("No RL reset in progress");
        return ARGUMENT;
    }
    handle.reset_pending = false;
    handle.reset_result
}

/// Lookahead copy (bopl_rl.h `bopl_rl_fork`): a handle whose world is a
/// private snapshot of `world`'s — same state, same RNG stream, so the same
/// inputs give the same future. When `reuse` is non-null that handle is
/// refilled (its buffers are reused) and returned; otherwise a new handle is
/// allocated. Null when `world` is null or has no episode. Forks never carry
/// the replay recording and are closed with `bopl_rl_close`.
///
/// # Safety
/// `world` must be a live handle or null; `reuse` a live handle or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_fork(world: *const Handle, reuse: *mut Handle) -> *mut Handle {
    let Some(source) = (unsafe { world.as_ref() }) else {
        return core::ptr::null_mut();
    };
    if !source.world.valid || source.reset_pending || core::ptr::eq(world, reuse) {
        return core::ptr::null_mut();
    }
    let target = match unsafe { reuse.as_mut() } {
        Some(handle) => handle,
        None => Box::leak(Box::new(Handle {
            world: RolloutWorld::new(),
            reset_pending: false,
            reset_result: OK,
        })),
    };
    target.world.fork_from(&source.world);
    target.reset_pending = false;
    target.reset_result = OK;
    target
}

/// # Safety
/// `world` must be a live handle from `bopl_rl_open` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_reset_pending(world: *const Handle) -> i32 {
    unsafe { world.as_ref() }.is_some_and(|handle| handle.reset_pending) as i32
}

/// # Safety
/// `actions` must point to `count` BoplRlAction records.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_step(
    world: *mut Handle,
    actions: *const CAction,
    count: usize,
    repeat: u32,
) -> i32 {
    let Some(handle) = (unsafe { world.as_mut() }) else {
        return ARGUMENT;
    };
    if actions.is_null() || !handle.world.valid || count != handle.world.players.len() {
        return ARGUMENT;
    }
    if !(1..=16).contains(&repeat) {
        return ARGUMENT;
    }
    if handle.world.terminal {
        return ENDED;
    }
    let actions = unsafe { std::slice::from_raw_parts(actions, count) };
    match handle.world.step(actions, repeat) {
        Ok(()) => OK,
        Err(message) => {
            handle.world.set_error(&message);
            if message.contains("Malformed") || message.contains("action") {
                ARGUMENT
            } else {
                GAME_ERROR
            }
        }
    }
}

/// # Safety
/// `actions` must point to `count` BoplRlAction records.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_step_sampled(
    world: *mut Handle,
    actions: *const CAction,
    count: usize,
    first: u32,
    second: u32,
) -> i32 {
    let Some(handle) = (unsafe { world.as_mut() }) else {
        return ARGUMENT;
    };
    if actions.is_null() || !handle.world.valid || count != handle.world.players.len() {
        return ARGUMENT;
    }
    if first < 1 || first + second > 16 {
        return ARGUMENT;
    }
    if handle.world.terminal {
        return ENDED;
    }
    let actions = unsafe { std::slice::from_raw_parts(actions, count) };
    match handle.world.step_sampled(actions, first, second) {
        Ok(()) => OK,
        Err(message) => {
            handle.world.set_error(&message);
            GAME_ERROR
        }
    }
}

/// # Safety
/// `world` must be a live handle from `bopl_rl_open` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_state(world: *const Handle) -> *const CState {
    match unsafe { world.as_ref() } {
        Some(handle) if handle.world.valid => &*handle.world.state,
        _ => core::ptr::null(),
    }
}

/// # Safety
/// `world` must be a live handle from `bopl_rl_open` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_sample(world: *const Handle) -> *const CState {
    match unsafe { world.as_ref() } {
        Some(handle) if handle.world.valid && handle.world.sample_valid => &*handle.world.sample,
        _ => core::ptr::null(),
    }
}

/// Lossless replay export is a .NET-backend feature (BOPLREP2); this backend
/// reports it unsupported rather than emitting an archive the reference
/// implementation would replay differently.
///
/// # Safety
/// `world` must be a live handle from `bopl_rl_open` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_replay(
    world: *mut Handle,
    bytes: *mut *const core::ffi::c_void,
    length: *mut usize,
) -> i32 {
    if bytes.is_null() || length.is_null() {
        return ARGUMENT;
    }
    let Some(handle) = (unsafe { world.as_mut() }) else {
        return ARGUMENT;
    };
    match handle.world.encode_replay() {
        Ok(()) => {
            unsafe {
                *bytes = handle.world.replay_buffer.as_ptr().cast();
                *length = handle.world.replay_buffer.len();
            }
            OK
        }
        Err(message) => {
            handle.world.set_error(&message);
            GAME_ERROR
        }
    }
}

/// # Safety
/// `world` must be a live handle from `bopl_rl_open` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_error(world: *const Handle) -> *const c_char {
    static EMPTY: &[u8] = b"\0";
    match unsafe { world.as_ref() } {
        Some(handle) => handle.world.error.as_ptr().cast(),
        None => EMPTY.as_ptr().cast(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn bopl_rl_source_hash(_world: *const Handle) -> *const c_char {
    static HASH: [u8; 65] = {
        let mut buffer = [0u8; 65];
        let mut i = 0;
        while i < 64 {
            buffer[i] = super::SOURCE_HASH[i];
            i += 1;
        }
        buffer
    };
    HASH.as_ptr().cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn bopl_rl_catalog_hash(_world: *const Handle) -> *const c_char {
    static HASH: [u8; 65] = {
        let mut buffer = [0u8; 65];
        let mut i = 0;
        while i < 64 {
            buffer[i] = super::CATALOG_HASH[i];
            i += 1;
        }
        buffer
    };
    HASH.as_ptr().cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn bopl_rl_map_count(_world: *const Handle) -> u32 {
    super::MAP_COUNT
}

#[unsafe(no_mangle)]
pub extern "C" fn bopl_rl_ability_count(_world: *const Handle) -> u32 {
    super::ABILITY_COUNT
}

/// # Safety
/// `world` must be a live handle from `bopl_rl_open` or null; it is freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_rl_close(world: *mut Handle) {
    if !world.is_null() {
        drop(unsafe { Box::from_raw(world) });
    }
}
