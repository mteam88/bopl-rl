//! Static match data extracted from the reference export by
//! `tools/extract_rollout_data.py` (native/data/rollout.json). Everything is
//! raw Q32.32 exactly as serialized in Unity; parsed once per process.
use crate::{fixed::Fix, surface::Surface, vector::Vec2};
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct RawVec2 {
    x: i64,
    y: i64,
}
impl RawVec2 {
    fn get(&self) -> Vec2 {
        Vec2::new(Fix(self.x), Fix(self.y))
    }
}
#[derive(Deserialize)]
struct RawCurveKey {
    time: i64,
    value: i64,
}
#[derive(Deserialize)]
struct RawAnimateVelocity {
    #[serde(default)]
    speed: i64,
    #[serde(default)]
    mu: i64,
    #[serde(default)]
    rotation_speed: i64,
    #[serde(default)]
    rotation_mu: i64,
    sudden_death_speed: i64,
    sudden_death_max_delay: i64,
    sudden_death_acceleration: i64,
    sudden_death_y_speed: Vec<RawCurveKey>,
}
#[derive(Deserialize)]
struct RawBody {
    mass: i64,
    bounciness: i64,
    static_friction: i64,
    dynamic_friction: i64,
    moment_of_inertia: i64,
    gravity_scale: i64,
    air_drag: i64,
    rotational_drag: i64,
}
#[derive(Deserialize)]
struct RawOrbit {
    force: i64,
    delay: i64,
    mass: i64,
    air_drag: i64,
    path: Vec<RawVec2>,
}
#[derive(Deserialize)]
struct RawPlatform {
    position: RawVec2,
    rotation: i64,
    scale: i64,
    extents: RawVec2,
    radius: i64,
    animate_velocity: RawAnimateVelocity,
    body: RawBody,
    orbit: Option<RawOrbit>,
}
#[derive(Deserialize)]
struct RawWater {
    position: RawVec2,
    extents: RawVec2,
}
#[derive(Deserialize)]
struct RawSession {
    team_spawns: Vec<RawVec2>,
    time_before_sudden_death: i64,
}
#[derive(Deserialize)]
struct RawMap {
    platforms: Vec<RawPlatform>,
    water: RawWater,
    session: RawSession,
}
#[derive(Deserialize)]
struct RawPhysics {
    max_speed: i64,
    jump_strength: i64,
    accel: i64,
    radius: i64,
    gravity_accel: i64,
    gravity_max_fall_speed: i64,
    #[serde(default)]
    gravity_modifier: Option<i64>,
    #[serde(default)]
    jump_extra_x: Option<i64>,
    #[serde(default)]
    jump_momentum: Option<i64>,
    #[serde(default)]
    jump_normal_scale: Option<i64>,
    #[serde(default)]
    jump_teleport: Option<i64>,
    air_accel: i64,
}
#[derive(Deserialize)]
struct RawPlayer {
    physics: RawPhysics,
    hurtbox_extents: RawVec2,
    spawn_anim_ticks: u32,
}
#[derive(Deserialize)]
struct RawDash {
    cooldown: i64,
    hurtbox_extents: RawVec2,
}
#[derive(Deserialize)]
struct RawBow {
    cooldown: i64,
    hurtbox_extents: RawVec2,
    arrow_speed: i64,
    knockback: i64,
    aireal_control_blend: i64,
    time_before_arrows_hurt_owner: i64,
    max_slow_mo_speed: i64,
    max_arrows: u32,
    physics: RawPhysics,
    anims: RawBowAnims,
}
#[derive(Deserialize)]
struct RawBowAnims {
    transform: Vec<i32>,
    load: Vec<i32>,
    #[serde(rename = "fireAir")]
    fire_air: Vec<i32>,
    #[serde(rename = "fireGround")]
    fire_ground: Vec<i32>,
}
#[derive(Deserialize)]
struct RawArrowBody {
    #[serde(rename = "gravityScale")]
    gravity_scale: i64,
}
#[derive(Deserialize)]
struct RawArrow {
    radius: i64,
    body: RawArrowBody,
}
#[derive(Deserialize)]
struct RawPlatformAbility {
    cooldown: i64,
    hurtbox_radius: i64,
    time_in_exit: i64,
    size_change_speed: i64,
    max_size: i64,
    min_size: i64,
    rotation_speed: i64,
    max_area: i64,
    max_size_time: i64,
    face_height: i64,
    ejection_offset: i64,
    ejection_curve: Vec<RawCurveKey>,
    enter_scale_curve: Vec<RawCurveKey>,
    anims: RawPlatformAnims,
}
#[derive(Deserialize)]
struct RawPlatformAnims {
    #[serde(rename = "Anticipation")]
    anticipation: Vec<i32>,
    #[serde(rename = "Enter")]
    enter: Vec<i32>,
    #[serde(rename = "ToExit")]
    to_exit: Vec<i32>,
    #[serde(rename = "Exit")]
    exit: Vec<i32>,
}
#[derive(Deserialize)]
struct RawResizable {
    extents: RawVec2,
    radius: i64,
    mass_per_area: i64,
    animate_velocity: RawAnimateVelocity,
    body: RawBody,
}
#[derive(Deserialize)]
struct RawPrefabs {
    player: RawPlayer,
    dash: RawDash,
    bow: RawBow,
    arrow: RawArrow,
    platform_ability: RawPlatformAbility,
    resizable_platform: RawResizable,
}
#[derive(Deserialize)]
struct RawRollout {
    maps: Vec<RawMap>,
    prefabs: RawPrefabs,
}

/// A platform's authored geometry plus the behaviours that move it.
pub struct PlatformDef {
    pub surface: Surface,
    /// AnimateVelocity home spring (speed * (home - position)) and damping
    /// (-mu * velocity).
    pub speed: Fix,
    pub mu: Fix,
    pub rotation_speed: Fix,
    pub rotation_mu: Fix,
    pub body: crate::physics::Body,
    /// Sudden death sinking (AnimateVelocity.ActivateSuddenDeath).
    pub sudden_death_speed: Fix,
    pub sudden_death_max_delay: Fix,
    pub sudden_death_acceleration: Fix,
    pub sudden_death_y_speed: Vec<(Fix, Fix)>,
    /// Map 2's AntiLockPlatform orbit (force toward waypoints after a delay).
    pub orbit: Option<OrbitDef>,
}
pub struct OrbitDef {
    pub force: Fix,
    pub delay: Fix,
    pub mass: Fix,
    /// Per-tick velocity multiplier (PhysicsBody.airDrag from airFriction01).
    pub air_drag: Fix,
    pub path: Vec<Vec2>,
}
pub struct MapDef {
    pub platforms: Vec<PlatformDef>,
    pub water_position: Vec2,
    pub water_extents: Vec2,
    pub team_spawns: Vec<Vec2>,
    pub sudden_death_time: Fix,
}
pub struct MovementData {
    pub constants: crate::world::MovementConstants,
    pub radius: Fix,
    pub hurtbox_extents: Vec2,
    pub spawn_anim_ticks: u32,
}
pub struct BowData {
    pub cooldown: Fix,
    pub hurtbox_extents: Vec2,
    pub arrow_speed: Fix,
    pub knockback: Fix,
    pub aireal_control_blend: Fix,
    pub time_before_arrows_hurt_owner: Fix,
    pub max_slow_mo_speed: Fix,
    pub max_arrows: u32,
    pub constants: crate::world::MovementConstants,
    /// Ticks in the pre-aim "transform" animation before loading starts.
    pub transform_ticks: u32,
    /// Frame boundaries of the "load" animation in ticks (loadingFrame 1..4).
    pub load_frame_ticks: Vec<u32>,
    pub fire_air_ticks: u32,
    pub fire_ground_ticks: u32,
}
pub struct ArrowData {
    pub radius: Fix,
    pub gravity_scale: Fix,
}
pub struct PlatformAbilityData {
    pub cooldown: Fix,
    pub hurtbox_radius: Fix,
    pub time_in_exit: Fix,
    pub size_change_speed: Fix,
    pub max_size: Fix,
    pub min_size: Fix,
    pub rotation_speed: Fix,
    pub max_area: Fix,
    pub max_size_time: Fix,
    pub face_height: Fix,
    pub ejection_offset: Fix,
    pub ejection_curve: Vec<(Fix, Fix)>,
    pub enter_scale_curve: Vec<(Fix, Fix)>,
    pub anticipation_ticks: u32,
    pub enter_ticks: u32,
    pub to_exit_ticks: u32,
    pub exit_ticks: u32,
    pub spawn_extents: Vec2,
    pub spawn_radius: Fix,
    pub spawn_mass_per_area: Fix,
    pub spawn_speed: Fix,
    pub spawn_mu: Fix,
    pub spawn_rotation_speed: Fix,
    pub spawn_rotation_mu: Fix,
    pub spawn_body: crate::physics::Body,
    /// AnimateVelocity sudden-death sinking of spawned platforms.
    pub sudden_death_speed: Fix,
    pub sudden_death_max_delay: Fix,
    pub sudden_death_acceleration: Fix,
    pub sudden_death_y_speed: Vec<(Fix, Fix)>,
}
pub struct DashData {
    pub cooldown: Fix,
    pub hurtbox_extents: Vec2,
}
pub struct RolloutData {
    pub maps: Vec<MapDef>,
    pub player: MovementData,
    pub dash: DashData,
    pub bow: BowData,
    pub arrow: ArrowData,
    pub platform: PlatformAbilityData,
}

/// Grounded friction per tick, from the verified Level1 movement goldens
/// (player controller constants; identical on every grass map).
const FRICTION: Fix = Fix(2_147_483_648);
const ICE_FRICTION: Fix = Fix(3_736_621_568);

fn constants(physics: &RawPhysics) -> crate::world::MovementConstants {
    crate::world::MovementConstants {
        max_speed: Fix(physics.max_speed),
        accel: Fix(physics.accel),
        air_accel: Fix(physics.air_accel),
        gravity_accel: Fix(physics.gravity_accel),
        gravity_max_fall_speed: Fix(physics.gravity_max_fall_speed),
        gravity_modifier: Fix(physics.gravity_modifier.unwrap_or(1 << 32)),
        jump_strength: Fix(physics.jump_strength),
        jump_extra_x: Fix(physics.jump_extra_x.unwrap_or(2_576_980_480)),
        jump_momentum: Fix(physics.jump_momentum.unwrap_or(1 << 32)),
        jump_normal_scale: Fix(physics.jump_normal_scale.unwrap_or(1 << 32)),
        jump_teleport: Fix(physics.jump_teleport.unwrap_or(42_949_672)),
        friction: FRICTION,
        ice_friction: ICE_FRICTION,
    }
}

fn ticks(frames: &[i32]) -> u32 {
    frames.iter().map(|d| d.unsigned_abs()).sum()
}

fn curve(keys: &[RawCurveKey]) -> Vec<(Fix, Fix)> {
    keys.iter().map(|k| (Fix(k.time), Fix(k.value))).collect()
}

fn body(raw: &RawBody, orientation: Fix) -> crate::physics::Body {
    crate::physics::Body {
        inverse_mass: if raw.mass == 0 {
            Fix::ZERO
        } else {
            Fix::ONE / Fix(raw.mass)
        },
        inverse_inertia: if raw.moment_of_inertia == 0 {
            Fix::ZERO
        } else {
            Fix::ONE / Fix(raw.moment_of_inertia)
        },
        orientation,
        bounciness: Fix(raw.bounciness),
        static_friction: Fix(raw.static_friction),
        dynamic_friction: Fix(raw.dynamic_friction),
        gravity_scale: Fix(raw.gravity_scale),
        air_drag: Fix(raw.air_drag),
        rotational_drag: Fix(raw.rotational_drag),
        ..crate::physics::Body::default()
    }
}

fn map_def(raw: &RawMap) -> MapDef {
    let platforms = raw
        .platforms
        .iter()
        .map(|p| {
            let rotation = Fix(p.rotation);
            let right = Vec2::new(rotation.cos(), rotation.sin());
            let up = right.perp();
            let scale = Fix(p.scale);
            PlatformDef {
                surface: Surface {
                    center: p.position.get(),
                    right,
                    up,
                    extents: p.extents.get() * scale,
                    radius: Fix(p.radius) * scale,
                },
                speed: Fix(p.animate_velocity.speed),
                mu: Fix(p.animate_velocity.mu),
                rotation_speed: Fix(p.animate_velocity.rotation_speed),
                rotation_mu: Fix(p.animate_velocity.rotation_mu),
                body: body(&p.body, rotation),
                sudden_death_speed: Fix(p.animate_velocity.sudden_death_speed),
                sudden_death_max_delay: Fix(p.animate_velocity.sudden_death_max_delay),
                sudden_death_acceleration: Fix(p.animate_velocity.sudden_death_acceleration),
                sudden_death_y_speed: p
                    .animate_velocity
                    .sudden_death_y_speed
                    .iter()
                    .map(|k| (Fix(k.time), Fix(k.value)))
                    .collect(),
                orbit: p.orbit.as_ref().map(|o| OrbitDef {
                    force: Fix(o.force),
                    delay: Fix(o.delay),
                    mass: Fix(o.mass),
                    air_drag: Fix(o.air_drag),
                    path: o.path.iter().map(RawVec2::get).collect(),
                }),
            }
        })
        .collect();
    MapDef {
        platforms,
        water_position: raw.water.position.get(),
        water_extents: raw.water.extents.get(),
        team_spawns: raw.session.team_spawns.iter().map(RawVec2::get).collect(),
        sudden_death_time: Fix(raw.session.time_before_sudden_death),
    }
}

pub fn get() -> &'static RolloutData {
    static DATA: OnceLock<RolloutData> = OnceLock::new();
    DATA.get_or_init(|| {
        let raw: RawRollout = serde_json::from_str(include_str!("../../data/rollout.json"))
            .expect("validated embedded rollout data");
        let bow = &raw.prefabs.bow;
        let mut load_frame_ticks = Vec::new();
        let mut acc = 0;
        for d in &bow.anims.load {
            acc += d.unsigned_abs();
            load_frame_ticks.push(acc);
        }
        let plat = &raw.prefabs.platform_ability;
        RolloutData {
            maps: raw.maps.iter().map(map_def).collect(),
            player: MovementData {
                constants: constants(&raw.prefabs.player.physics),
                radius: Fix(raw.prefabs.player.physics.radius),
                hurtbox_extents: raw.prefabs.player.hurtbox_extents.get(),
                spawn_anim_ticks: raw.prefabs.player.spawn_anim_ticks,
            },
            dash: DashData {
                cooldown: Fix(raw.prefabs.dash.cooldown),
                hurtbox_extents: raw.prefabs.dash.hurtbox_extents.get(),
            },
            bow: BowData {
                cooldown: Fix(bow.cooldown),
                hurtbox_extents: bow.hurtbox_extents.get(),
                arrow_speed: Fix(bow.arrow_speed),
                knockback: Fix::from_int(bow.knockback),
                aireal_control_blend: Fix(bow.aireal_control_blend),
                time_before_arrows_hurt_owner: Fix(bow.time_before_arrows_hurt_owner),
                max_slow_mo_speed: Fix(bow.max_slow_mo_speed),
                max_arrows: bow.max_arrows,
                constants: constants(&bow.physics),
                transform_ticks: ticks(&bow.anims.transform),
                load_frame_ticks,
                fire_air_ticks: ticks(&bow.anims.fire_air),
                fire_ground_ticks: ticks(&bow.anims.fire_ground),
            },
            arrow: ArrowData {
                radius: Fix(raw.prefabs.arrow.radius),
                gravity_scale: Fix(raw.prefabs.arrow.body.gravity_scale),
            },
            platform: PlatformAbilityData {
                cooldown: Fix(plat.cooldown),
                hurtbox_radius: Fix(plat.hurtbox_radius),
                time_in_exit: Fix(plat.time_in_exit),
                size_change_speed: Fix(plat.size_change_speed),
                max_size: Fix(plat.max_size),
                min_size: Fix(plat.min_size),
                rotation_speed: Fix(plat.rotation_speed),
                max_area: Fix(plat.max_area),
                max_size_time: Fix(plat.max_size_time),
                face_height: Fix(plat.face_height),
                ejection_offset: Fix(plat.ejection_offset),
                ejection_curve: curve(&plat.ejection_curve),
                enter_scale_curve: curve(&plat.enter_scale_curve),
                anticipation_ticks: ticks(&plat.anims.anticipation),
                enter_ticks: ticks(&plat.anims.enter),
                to_exit_ticks: ticks(&plat.anims.to_exit),
                exit_ticks: ticks(&plat.anims.exit),
                spawn_extents: raw.prefabs.resizable_platform.extents.get(),
                spawn_radius: Fix(raw.prefabs.resizable_platform.radius),
                spawn_mass_per_area: Fix(raw.prefabs.resizable_platform.mass_per_area),
                spawn_speed: Fix(raw.prefabs.resizable_platform.animate_velocity.speed),
                spawn_mu: Fix(raw.prefabs.resizable_platform.animate_velocity.mu),
                spawn_rotation_speed: Fix(raw
                    .prefabs
                    .resizable_platform
                    .animate_velocity
                    .rotation_speed),
                spawn_rotation_mu: Fix(raw.prefabs.resizable_platform.animate_velocity.rotation_mu),
                spawn_body: body(&raw.prefabs.resizable_platform.body, Fix::ZERO),
                sudden_death_speed: Fix(raw
                    .prefabs
                    .resizable_platform
                    .animate_velocity
                    .sudden_death_speed),
                sudden_death_max_delay: Fix(raw
                    .prefabs
                    .resizable_platform
                    .animate_velocity
                    .sudden_death_max_delay),
                sudden_death_acceleration: Fix(raw
                    .prefabs
                    .resizable_platform
                    .animate_velocity
                    .sudden_death_acceleration),
                sudden_death_y_speed: curve(
                    &raw.prefabs
                        .resizable_platform
                        .animate_velocity
                        .sudden_death_y_speed,
                ),
            },
        }
    })
}
