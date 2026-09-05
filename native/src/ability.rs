//! Ability handoff values and executable instant-ability state. References are
//! canonical actor, asset, platform and rope identities, not native pointers.
use crate::{fixed::Fix, vector::Vec2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlayerInfo {
    pub player: i32,
    pub slot: i32,
    pub position: Vec2,
    pub external_velocity: Vec2,
    pub velocity: Vec2,
    pub material: Option<u32>,
    pub grounded: bool,
    pub grounded_speed: Fix,
    pub rotation: Fix,
    pub forward: Vec2,
    pub up: Vec2,
    pub facing_right: bool,
    pub ground: Option<u32>,
    pub local: Fix,
    pub radius: Fix,
    pub sprite: Option<u32>,
    pub rope: Option<usize>,
    pub top: bool,
    pub use_rope_on_player: bool,
    pub rope_target: Option<u32>,
    pub slime: Option<usize>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlayerFlags {
    pub spark: bool,
    pub macho: bool,
    pub platform: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SlimeEntry {
    pub info: PlayerInfo,
    pub rope_target: Option<u32>,
    pub was_invisible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invisible_on_entry: Option<bool>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExitInfo {
    pub dead: bool,
    pub killer: i32,
    pub position: Vec2,
    pub external_velocity: Vec2,
    pub velocity: Vec2,
    pub facing_right: bool,
    pub grounded: bool,
    pub grounded_speed: Fix,
    pub sprite: Option<u32>,
    pub ground: Option<u32>,
    pub local: Fix,
    pub just_landed: bool,
    pub cause: i32,
    pub radius: Fix,
    pub enter_turn: bool,
    pub enter_jump: bool,
    pub rope: Option<usize>,
    pub top: bool,
}
impl ExitInfo {
    pub fn physics(&self) -> crate::player_physics::Info {
        crate::player_physics::Info {
            grounded: self.grounded,
            ground: self.ground,
            local: self.local,
            radius: self.radius,
            grounded_speed: self.grounded_speed,
            position: self.position,
            velocity: self.velocity,
            external_velocity: self.external_velocity,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Handler {
    Invisibility(Box<Invisibility>),
    Entity { actor: usize },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Invisibility {
    /// Invisibility.cooldown, restored after the effect ends. The initial
    /// InstantAbility cooldown is a separate value (2s vs 6s in the prefab).
    pub restore_cooldown: Fix,
    pub gate_invisible: bool,
    pub component_present: bool,
    pub linked: bool,
    pub transform_present: bool,
    pub position: Vec2,
    pub active: bool,
    pub info: PlayerInfo,
    pub slime: Option<usize>,
    pub audio: Option<String>,
}
