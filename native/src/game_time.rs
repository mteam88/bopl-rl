//! The fixed simulation clock from GameTime. None means no registered player.
use crate::fixed::Fix;

pub fn fixed_delta_time(dt: Fix, scale: Fix, protected: Option<bool>) -> Fix {
    if protected == Some(false) {
        dt * scale
    } else {
        dt
    }
}
