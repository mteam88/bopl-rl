//! The versioned C surface. Pointer validity beyond null checks is the caller's contract.
use crate::world::{Action, World};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    ptr, slice,
};

const OK: i32 = 0;
const INVALID: i32 = 1;
const UNSUPPORTED: i32 = 2;
const ENDED: i32 = 3;
const SHORT: i32 = 4;
const INTERNAL: i32 = 5;
pub struct BoplWorld {
    world: World,
}
#[repr(C)]
pub struct Observation {
    pub x: i64,
    pub y: i64,
    pub vx: i64,
    pub vy: i64,
    pub rotation: i64,
    pub local: i64,
    pub grounded_speed: i64,
    pub cooldown: i64,
    pub id: u32,
    pub ground_id: u32,
    pub alive: u8,
    pub grounded: u8,
    pub dashing: u8,
    pub reserved: u8,
}
fn guarded(action: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(action)).unwrap_or(INTERNAL)
}
#[unsafe(no_mangle)]
pub extern "C" fn bopl_abi_version() -> u32 {
    1
}
#[unsafe(no_mangle)]
pub extern "C" fn bopl_create() -> *mut BoplWorld {
    bopl_create_players(2)
}
#[unsafe(no_mangle)]
pub extern "C" fn bopl_create_players(players: usize) -> *mut BoplWorld {
    catch_unwind(|| {
        World::level1_with_players(players).map_or(ptr::null_mut(), |world| {
            Box::into_raw(Box::new(BoplWorld { world }))
        })
    })
    .unwrap_or(ptr::null_mut())
}
/// # Safety
/// Null or readable live handle; no concurrent mutation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_player_count(handle: *const BoplWorld) -> usize {
    if handle.is_null() {
        0
    } else {
        unsafe { &(*handle).world }.players.len()
    }
}

/// # Safety
/// `handle` is null or a live handle returned by bopl_create, with no concurrent calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_destroy(handle: *mut BoplWorld) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}
/// # Safety
/// `handle` is null or an exclusively owned live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_reset(handle: *mut BoplWorld) -> i32 {
    if handle.is_null() {
        return INVALID;
    }
    guarded(|| {
        let reset = World::level1_with_players(unsafe { &(*handle).world }.players.len())
            .expect("valid player count");
        unsafe {
            (*handle).world = reset;
        };
        OK
    })
}
/// # Safety
/// Handle is exclusively owned. Actions points to count readable, aligned actions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_step(
    handle: *mut BoplWorld,
    actions: *const Action,
    count: usize,
) -> i32 {
    if handle.is_null() || actions.is_null() {
        return INVALID;
    }
    guarded(|| {
        if count != unsafe { &(*handle).world }.players.len() {
            return INVALID;
        }
        let actions = unsafe { slice::from_raw_parts(actions, count) };
        if actions.iter().any(|a| a.buttons & !3 != 0) {
            return UNSUPPORTED;
        }
        let world = unsafe { &mut (*handle).world };
        if world.terminal() {
            return ENDED;
        }
        let mut candidate = world.clone();
        match candidate.step(actions) {
            Ok(()) => {
                *world = candidate;
                OK
            }
            Err(_) => INTERNAL,
        }
    })
}
/// # Safety
/// Live handle, and out points to count writable, aligned observations; no aliasing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_observe(
    handle: *const BoplWorld,
    out: *mut Observation,
    count: usize,
) -> i32 {
    if handle.is_null() || out.is_null() {
        return INVALID;
    }
    guarded(|| {
        if count != unsafe { &(*handle).world }.players.len() {
            return INVALID;
        }
        for (index, p) in unsafe { &(*handle).world }.players.iter().enumerate() {
            let velocity = p.velocity + p.external_velocity;
            let value = Observation {
                x: p.position.x.0,
                y: p.position.y.0,
                vx: velocity.x.0,
                vy: velocity.y.0,
                rotation: p.rotation.0,
                local: p.local.0,
                grounded_speed: p.grounded_speed.0,
                cooldown: p.cooldown.0,
                id: p.id,
                ground_id: p.ground.unwrap_or(0),
                alive: u8::from(p.alive),
                grounded: u8::from(p.grounded),
                dashing: u8::from(p.dash.active),
                reserved: 0,
            };
            unsafe {
                out.add(index).write(value);
            }
        }
        OK
    })
}
/// # Safety
/// Null or readable live handle; no concurrent mutation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_terminal(handle: *const BoplWorld) -> i32 {
    if handle.is_null() {
        -1
    } else {
        i32::from(unsafe { &(*handle).world }.terminal())
    }
}
/// # Safety
/// Readable handle, writable required, and out is null or writable for capacity bytes.
/// The memory regions must not overlap. No concurrent mutation of the handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bopl_state_json(
    handle: *const BoplWorld,
    out: *mut u8,
    capacity: usize,
    required: *mut usize,
) -> i32 {
    if handle.is_null() || required.is_null() || (out.is_null() && capacity != 0) {
        return INVALID;
    }
    guarded(|| {
        let Ok(json) = serde_json::to_vec(unsafe { &(*handle).world }) else {
            return INTERNAL;
        };
        let size = json.len() + 1;
        unsafe {
            required.write(size);
        }
        if capacity < size || out.is_null() {
            return SHORT;
        }
        unsafe {
            ptr::copy_nonoverlapping(json.as_ptr(), out, json.len());
            out.add(json.len()).write(0);
        }
        OK
    })
}
