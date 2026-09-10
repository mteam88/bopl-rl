pub mod ability;
pub mod ability_routing;
pub mod actor_bounds;
pub mod actor_clone;
pub mod actor_collider;
pub mod actor_eating;
pub mod actor_entity;
pub mod beam;
pub mod clone_ability;
pub mod curve;
pub mod dash_ability;
#[cfg(not(feature = "q32"))]
pub mod fixed;
#[cfg(feature = "q32")]
#[path = "fixed_q32.rs"]
pub mod fixed;
pub mod geometry;
pub mod ordinary_ability;
pub mod quick_spell;
pub mod shockwave;
mod slime_ability;
mod slime_exit;
pub mod spell_world;
pub mod surface;
pub mod vector;

pub mod world;

pub mod ffi;

pub mod physics;

pub mod random;
pub mod rewind;

pub mod player_life;

pub mod player_collision;

pub mod player_ground;

pub mod bopl_body;
pub mod player_body;
pub mod player_position;
pub mod revival;

pub mod player_physics;

pub mod game_time;

pub mod platform_motion;

pub mod player_controller;

pub mod animator;

pub mod spawn;

mod mono_sort;

pub mod scheduler;

pub mod actors;

pub mod slime;

pub mod actor_combat;

mod actor_contacts;

pub mod death_effect;

pub mod actor_spawn;

pub mod revive_particle;

pub mod revive_effect;
pub mod rollout;
