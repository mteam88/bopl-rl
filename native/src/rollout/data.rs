//! Static match data extracted from the reference export by
//! `tools/extract_rollout_data.py` (native/data/rollout.json). Everything is
//! raw Q32.32 exactly as serialized in Unity; parsed once per process.
use crate::{curve::Curve, fixed::Fix, surface::Surface, vector::Vec2};
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct RawVec2 {
    x: i64,
    y: i64,
}
impl RawVec2 {
    fn get(&self) -> Vec2 {
        Vec2::new(Fix::from_raw(self.x), Fix::from_raw(self.y))
    }
}
#[derive(Deserialize)]
struct RawAnimateVelocity {
    speed: i64,
    mu: i64,
    rotation_speed: i64,
    rotation_mu: i64,
    sudden_death_speed: i64,
    sudden_death_max_delay: i64,
    sudden_death_acceleration: i64,
    sudden_death_y_speed: Curve,
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
    hierarchy: u32,
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
    teammate_spacing: i64,
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
    fast_spawn_anim_ticks: u32,
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
    mass: i64,
    bounciness: i64,
    friction: i64,
    #[serde(rename = "dynamicFriction")]
    dynamic_friction: i64,
    #[serde(rename = "momentOfInertia")]
    moment_of_inertia: i64,
    #[serde(rename = "gravityScale")]
    gravity_scale: i64,
    #[serde(rename = "airFriction01")]
    air_drag: i64,
    #[serde(rename = "rotationalDrag01")]
    rotational_drag: i64,
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
    ejection_curve: Curve,
    enter_scale_curve: Curve,
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
    remove_curve: Curve,
    animate_velocity: RawAnimateVelocity,
    body: RawBody,
}
#[derive(Deserialize)]
struct RawPickup {
    layer: u32,
    radius: i64,
    mass: i64,
    offset: RawVec2,
}
#[derive(Deserialize)]
struct RawPrefabs {
    player: RawPlayer,
    dash: RawDash,
    bow: RawBow,
    arrow: RawArrow,
    ability_pickup: RawPickup,
    platform_ability: RawPlatformAbility,
    resizable_platform: RawResizable,
    gust: RawGust,
    revival: RawRevival,
    teleport: RawTeleport,
    grenade: RawGrenade,
    tack: RawTack,
}

#[derive(Deserialize)]
struct RawGrenade {
    hurtbox_extents: RawVec2,
    #[serde(rename = "throw")]
    throw_data: RawThrowItem,
    item: RawGrenadeItem,
    explosion: RawGrenadeExplosion,
}
#[derive(Deserialize)]
struct RawThrowItem {
    air_position: RawVec2,
    ground_position: RawVec2,
    ground_dummy: Vec<RawVec2>,
    air_dummy: Vec<RawVec2>,
    hurt_owner_delay: i64,
    max_force: i64,
    min_force: i64,
    force_gain_speed: i64,
    max_slow_mo_speed: i64,
    knockback: i64,
    physics: RawPhysics,
    animations: RawThrowAnimations,
}
#[derive(Deserialize)]
struct RawThrowAnimations {
    #[serde(rename = "gEnter")]
    ground_enter: Vec<i32>,
    #[serde(rename = "gExit")]
    ground_exit: Vec<i32>,
    #[serde(rename = "aEnter")]
    air_enter: Vec<i32>,
    #[serde(rename = "aExit")]
    air_exit: Vec<i32>,
}
#[derive(Deserialize)]
struct RawGrenadeItem {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    radius: i64,
    body: RawBody,
    detonation_time: i64,
    clone_count: u32,
    duplicate_pattern: u32,
    duplicate_spacing: i64,
    duplicate_angle: i64,
}
#[derive(Deserialize)]
struct RawTack {
    hurtbox_extents: RawVec2,
    #[serde(rename = "throw")]
    throw_data: RawThrowItem,
    item: RawTackItem,
}
#[derive(Deserialize)]
struct RawTackItem {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    radius: i64,
    body: RawBody,
    clone_count: u32,
    duplicate_pattern: u32,
    duplicate_spacing: i64,
    duplicate_angle: i64,
    arm_seconds: i64,
    lifetime_seconds: i64,
}
#[derive(Deserialize)]
struct RawGrenadeExplosion {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    extents: RawVec2,
    animation_ticks: u32,
    shockwaves: [RawExplosionShockwave; 2],
}
#[derive(Clone, Copy, Deserialize)]
struct RawExplosionShockwave {
    platform_force: i64,
    default_force: i64,
    grounded_player_multiplier: i64,
    radius: i64,
    collision_mask: u32,
}

#[derive(Deserialize)]
struct RawGust {
    anims: RawGustAnimations,
    shockwave_offset: RawVec2,
    collider_offset: RawVec2,
    hurtbox_radius: i64,
    platform_force: i64,
    default_force: i64,
    grounded_multiplier: i64,
    radius: i64,
    curve: Curve,
}

#[derive(Deserialize)]
struct RawTeleport {
    up_cast_offset: i64,
    player_radius: i64,
    marker: RawTeleportMarker,
    effect_updatable_count: u32,
    effect_duration: i64,
}

#[derive(Deserialize)]
struct RawTeleportMarker {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    offset: RawVec2,
    radius: i64,
    teleport_radius: i64,
    body: RawBody,
}

#[derive(Deserialize)]
struct RawGustAnimations {
    #[serde(rename = "castAir")]
    cast_air: Vec<i32>,
    exit: Vec<i32>,
}

#[derive(Deserialize)]
struct RawRevival {
    anims: RawGustAnimations,
    shockwave_offset: RawVec2,
    collider_offset: RawVec2,
    hurtbox_radius: i64,
    marker: RawRevivalMarker,
    death_effect_updatable_count: u32,
    effect_updatable_count: u32,
    particle_updatable_count: u32,
    effect_wisp_ticks: u32,
    particle_speed: i64,
}

#[derive(Deserialize)]
struct RawRevivalMarker {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    offset: RawVec2,
    radius: i64,
    body: RawBody,
}
#[derive(Deserialize)]
struct RawRollout {
    maps: Vec<RawMap>,
    prefabs: RawPrefabs,
    abilities: Vec<RawAbility>,
}
#[derive(Deserialize)]
struct RawAbility {
    id: u32,
    name: String,
    prefab: String,
    cooldown: i64,
    ground_only: bool,
    stopped_during_time_stop: bool,
    castable: bool,
    instant: bool,
    rope: bool,
    offensive: bool,
    updatable_count: u32,
    effect_cooldown: Option<i64>,
    effect_duration: Option<i64>,
    air_jump_strength: Option<i64>,
    form_jump_speed: Option<i64>,
}

/// A platform's authored geometry plus the behaviours that move it.
pub struct PlatformDef {
    pub surface: Surface,
    /// The collider's authored half-extents (`startExtents * scale`); the
    /// surface carries the perimeter's `One / (One / e)` reading instead.
    pub collider_extents: Vec2,
    pub hierarchy: u32,
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
    pub sudden_death_y_speed: Curve,
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
    pub teammate_spacing: Fix,
    pub respawn_points: [Vec2; 3],
    pub sudden_death_time: Fix,
}
pub struct MovementData {
    pub constants: crate::world::MovementConstants,
    pub radius: Fix,
    pub hurtbox_extents: Vec2,
    pub spawn_anim_ticks: u32,
    pub fast_spawn_anim_ticks: u32,
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
    /// BoplBody.NewPhysicsBody for the Arrow prefab (BoplBody.cs:215-229).
    pub body: crate::physics::Body,
}
/// AbilityPickup_Dynamic prefab: the circle a final death drops
/// (SlimeController.DropAbilities).
pub struct PickupData {
    pub layer: u32,
    pub radius: Fix,
    pub inverse_mass: Fix,
    /// DPhysicsCircle centre offset from the FixTransform (DetPhysics.cs:2513).
    pub offset: Vec2,
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
    pub ejection_curve: Curve,
    pub enter_scale_curve: Curve,
    pub anticipation_ticks: u32,
    pub enter_ticks: u32,
    pub to_exit_ticks: u32,
    pub exit_ticks: u32,
    pub spawn_extents: Vec2,
    pub spawn_radius: Fix,
    pub spawn_mass_per_area: Fix,
    pub remove_curve: Curve,
    pub spawn_speed: Fix,
    pub spawn_mu: Fix,
    pub spawn_rotation_speed: Fix,
    pub spawn_rotation_mu: Fix,
    pub spawn_body: crate::physics::Body,
    /// AnimateVelocity sudden-death sinking of spawned platforms.
    pub sudden_death_speed: Fix,
    pub sudden_death_max_delay: Fix,
    pub sudden_death_acceleration: Fix,
    pub sudden_death_y_speed: Curve,
}
pub struct DashData {
    pub cooldown: Fix,
    pub hurtbox_extents: Vec2,
}
pub struct GustData {
    pub cast_ticks: u32,
    pub exit_ticks: u32,
    pub shockwave_offset: Vec2,
    pub collider_offset: Vec2,
    pub hurtbox_radius: Fix,
    pub platform_force: Fix,
    pub default_force: Fix,
    pub grounded_multiplier: Fix,
    pub radius: Fix,
    pub curve: Curve,
}
pub struct RevivalData {
    pub cast_ticks: u32,
    pub exit_ticks: u32,
    pub shockwave_offset: Vec2,
    pub collider_offset: Vec2,
    pub hurtbox_radius: Fix,
    pub marker_layer: u32,
    pub marker_updatable_count: u32,
    pub marker_collider_hierarchy: u32,
    pub marker_offset: Vec2,
    pub marker_radius: Fix,
    pub marker_body: crate::physics::Body,
    pub death_effect_updatable_count: u32,
    pub effect_updatable_count: u32,
    pub particle_updatable_count: u32,
    pub effect_wisp_ticks: u32,
    pub particle_speed: Fix,
}
pub struct TeleportData {
    pub up_cast_offset: Fix,
    pub player_radius: Fix,
    pub marker_layer: u32,
    pub marker_updatable_count: u32,
    pub marker_collider_hierarchy: u32,
    pub marker_offset: Vec2,
    pub marker_radius: Fix,
    pub marker_teleport_radius: Fix,
    pub marker_body: crate::physics::Body,
    pub effect_updatable_count: u32,
    pub effect_duration: Fix,
}
pub struct GrenadeData {
    pub hurtbox_extents: Vec2,
    pub air_position: Vec2,
    pub ground_position: Vec2,
    pub ground_dummy: Vec<Vec2>,
    pub air_dummy: Vec<Vec2>,
    pub hurt_owner_delay: Fix,
    pub max_force: Fix,
    pub min_force: Fix,
    pub force_gain_speed: Fix,
    pub max_slow_mo_speed: Fix,
    pub knockback: Fix,
    pub constants: crate::world::MovementConstants,
    pub ground_enter_ticks: u32,
    pub ground_exit_ticks: u32,
    pub air_enter_ticks: u32,
    pub air_exit_ticks: u32,
    pub item_layer: u32,
    pub item_updatable_count: u32,
    pub item_collider_hierarchy: u32,
    pub item_radius: Fix,
    pub item_body: crate::physics::Body,
    pub detonation_time: Fix,
    pub clone_count: u32,
    pub duplicate_pattern: u32,
    pub duplicate_spacing: Fix,
    pub duplicate_angle: Fix,
    pub explosion_layer: u32,
    pub explosion_updatable_count: u32,
    pub explosion_collider_hierarchy: u32,
    pub explosion_extents: Vec2,
    pub explosion_animation_ticks: u32,
    pub explosion_shockwaves: [ExplosionShockwaveData; 2],
}
pub struct TackData {
    pub hurtbox_extents: Vec2,
    pub air_position: Vec2,
    pub ground_position: Vec2,
    pub ground_dummy: Vec<Vec2>,
    pub air_dummy: Vec<Vec2>,
    pub hurt_owner_delay: Fix,
    pub max_force: Fix,
    pub min_force: Fix,
    pub force_gain_speed: Fix,
    pub max_slow_mo_speed: Fix,
    pub knockback: Fix,
    pub constants: crate::world::MovementConstants,
    pub ground_enter_ticks: u32,
    pub ground_exit_ticks: u32,
    pub air_enter_ticks: u32,
    pub air_exit_ticks: u32,
    pub item_layer: u32,
    pub item_updatable_count: u32,
    pub item_collider_hierarchy: u32,
    pub item_radius: Fix,
    pub item_body: crate::physics::Body,
    pub clone_count: u32,
    pub duplicate_pattern: u32,
    pub duplicate_spacing: Fix,
    pub duplicate_angle: Fix,
    pub arm_seconds: Fix,
    pub lifetime_seconds: Fix,
}
#[derive(Clone, Copy)]
pub struct ExplosionShockwaveData {
    pub platform_force: Fix,
    pub default_force: Fix,
    pub grounded_player_multiplier: Fix,
    pub radius: Fix,
    pub collision_mask: u32,
}
pub struct AbilityData {
    pub id: u32,
    pub name: String,
    pub prefab: String,
    pub cooldown: Fix,
    pub ground_only: bool,
    pub stopped_during_time_stop: bool,
    pub castable: bool,
    pub instant: bool,
    pub rope: bool,
    pub offensive: bool,
    pub updatable_count: u32,
    pub effect_cooldown: Option<Fix>,
    pub effect_duration: Option<Fix>,
    pub air_jump_strength: Option<Fix>,
    pub form_jump_speed: Option<Fix>,
}
pub struct RolloutData {
    pub maps: Vec<MapDef>,
    pub player: MovementData,
    pub dash: DashData,
    pub bow: BowData,
    pub arrow: ArrowData,
    pub pickup: PickupData,
    pub platform: PlatformAbilityData,
    pub gust: GustData,
    pub revival: RevivalData,
    pub teleport: TeleportData,
    pub grenade: GrenadeData,
    pub tack: TackData,
    pub abilities: Vec<AbilityData>,
}

/// Grounded friction per tick, from the verified Level1 movement goldens
/// (player controller constants; identical on every grass map).
const FRICTION: Fix = Fix::from_raw(2_147_483_648);
const ICE_FRICTION: Fix = Fix::from_raw(3_736_621_568);

fn constants(physics: &RawPhysics) -> crate::world::MovementConstants {
    crate::world::MovementConstants {
        max_speed: Fix::from_raw(physics.max_speed),
        accel: Fix::from_raw(physics.accel),
        air_accel: Fix::from_raw(physics.air_accel),
        gravity_accel: Fix::from_raw(physics.gravity_accel),
        gravity_max_fall_speed: Fix::from_raw(physics.gravity_max_fall_speed),
        gravity_modifier: Fix::from_raw(physics.gravity_modifier.unwrap_or(1 << 32)),
        jump_strength: Fix::from_raw(physics.jump_strength),
        jump_extra_x: Fix::from_raw(physics.jump_extra_x.unwrap_or(2_576_980_480)),
        jump_momentum: Fix::from_raw(physics.jump_momentum.unwrap_or(1 << 32)),
        jump_normal_scale: Fix::from_raw(physics.jump_normal_scale.unwrap_or(1 << 32)),
        jump_teleport: Fix::from_raw(physics.jump_teleport.unwrap_or(42_949_672)),
        friction: FRICTION,
        ice_friction: ICE_FRICTION,
    }
}

fn ticks(frames: &[i32]) -> u32 {
    frames.iter().map(|d| d.unsigned_abs()).sum()
}

fn body(raw: &RawBody, orientation: Fix) -> crate::physics::Body {
    crate::physics::Body {
        inverse_mass: if raw.mass == 0 {
            Fix::ZERO
        } else {
            Fix::ONE / Fix::from_raw(raw.mass)
        },
        inverse_inertia: if raw.moment_of_inertia == 0 {
            Fix::ZERO
        } else {
            Fix::ONE / Fix::from_raw(raw.moment_of_inertia)
        },
        orientation,
        bounciness: Fix::from_raw(raw.bounciness),
        static_friction: Fix::from_raw(raw.static_friction),
        dynamic_friction: Fix::from_raw(raw.dynamic_friction),
        gravity_scale: Fix::from_raw(raw.gravity_scale),
        air_drag: Fix::from_raw(raw.air_drag),
        rotational_drag: Fix::from_raw(raw.rotational_drag),
        ..crate::physics::Body::default()
    }
}

/// The perimeter extents `StickyRoundedRectangle.UpdateBounds` derives from a
/// collider built with these half-extents: `One / (One / extents)` per axis
/// (`DPhysicsRoundedRect.CalcExtents`, `Box.UpdateExtents`).
pub fn surface_extents(extents: Vec2) -> Vec2 {
    Vec2::new(
        Fix::ONE / (Fix::ONE / extents.x),
        Fix::ONE / (Fix::ONE / extents.y),
    )
}

fn map_def(raw: &RawMap) -> MapDef {
    let platforms = raw
        .platforms
        .iter()
        .map(|p| {
            let rotation = Fix::from_raw(p.rotation);
            // FixTransform's `rotation` setter derives the basis with
            // `new Vec2(angle)` (FixTransform.cs:45), i.e. Fix.FastCos /
            // Fix.FastSin (Vec2.cs:31-35): the un-interpolated table, whose
            // truncation offsets every tilted map platform by up to 7.6e-6
            // rad. The interpolated `Sin`/`Cos` gave the exact angle instead.
            let right = Vec2::angle(rotation);
            let up = right.perp();
            let scale = Fix::from_raw(p.scale);
            // StickyRoundedRectangle.UpdateBounds (StickyRoundedRectangle.cs:104,
            // rerun after the level animates in, GameSessionHandler.cs:769)
            // reads its perimeter extents back from the collider as
            // `One / inverseExtents` (DPhysicsRoundedRect.CalcExtents:355-358)
            // where the collider stored `One / startExtents`; the round trip
            // moves non-power-of-two extents by a few raw units in Q32.32.
            let collider_extents = p.extents.get() * scale;
            PlatformDef {
                surface: Surface {
                    center: p.position.get(),
                    right,
                    up,
                    extents: surface_extents(collider_extents),
                    radius: Fix::from_raw(p.radius) * scale,
                },
                collider_extents,
                hierarchy: p.hierarchy,
                speed: Fix::from_raw(p.animate_velocity.speed),
                mu: Fix::from_raw(p.animate_velocity.mu),
                rotation_speed: Fix::from_raw(p.animate_velocity.rotation_speed),
                rotation_mu: Fix::from_raw(p.animate_velocity.rotation_mu),
                body: body(&p.body, rotation),
                sudden_death_speed: Fix::from_raw(p.animate_velocity.sudden_death_speed),
                sudden_death_max_delay: Fix::from_raw(p.animate_velocity.sudden_death_max_delay),
                sudden_death_acceleration: Fix::from_raw(
                    p.animate_velocity.sudden_death_acceleration,
                ),
                sudden_death_y_speed: p.animate_velocity.sudden_death_y_speed.clone(),
                orbit: p.orbit.as_ref().map(|o| OrbitDef {
                    force: Fix::from_raw(o.force),
                    delay: Fix::from_raw(o.delay),
                    mass: Fix::from_raw(o.mass),
                    air_drag: Fix::from_raw(o.air_drag),
                    path: o.path.iter().map(RawVec2::get).collect(),
                }),
            }
        })
        .collect();
    let team_spawns: Vec<Vec2> = raw.session.team_spawns.iter().map(RawVec2::get).collect();
    let (mut min_x, mut max_x, mut max_y) =
        (Fix::from_int(-12), Fix::from_int(12), Fix::from_int(8));
    if let Some(first) = team_spawns.first().copied() {
        min_x = first.x;
        max_x = first.x;
        max_y = first.y;
        for spawn in &team_spawns[1..] {
            min_x = min_x.min(spawn.x);
            max_x = max_x.max(spawn.x);
            max_y = max_y.max(spawn.y);
        }
    }
    let two = Fix::from_int(2);
    let middle = (min_x + max_x) / two;
    let half_span = ((max_x - min_x) / two).max(Fix::from_int(14));
    // SceneBounds.Camera_YMax (40) minus StockRespawn's six-unit margin.
    let respawn_y = (max_y + Fix::from_int(12)).min(Fix::from_int(34));
    let respawn_points = [
        Vec2::new(middle - half_span, respawn_y),
        Vec2::new(middle, respawn_y),
        Vec2::new(middle + half_span, respawn_y),
    ];
    MapDef {
        platforms,
        water_position: raw.water.position.get(),
        water_extents: raw.water.extents.get(),
        team_spawns,
        teammate_spacing: Fix::from_raw(raw.session.teammate_spacing),
        respawn_points,
        sudden_death_time: Fix::from_raw(raw.session.time_before_sudden_death),
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
                radius: Fix::from_raw(raw.prefabs.player.physics.radius),
                hurtbox_extents: raw.prefabs.player.hurtbox_extents.get(),
                spawn_anim_ticks: raw.prefabs.player.spawn_anim_ticks,
                fast_spawn_anim_ticks: raw.prefabs.player.fast_spawn_anim_ticks,
            },
            dash: DashData {
                cooldown: Fix::from_raw(raw.prefabs.dash.cooldown),
                hurtbox_extents: raw.prefabs.dash.hurtbox_extents.get(),
            },
            bow: BowData {
                cooldown: Fix::from_raw(bow.cooldown),
                hurtbox_extents: bow.hurtbox_extents.get(),
                arrow_speed: Fix::from_raw(bow.arrow_speed),
                knockback: Fix::from_int(bow.knockback),
                aireal_control_blend: Fix::from_raw(bow.aireal_control_blend),
                time_before_arrows_hurt_owner: Fix::from_raw(bow.time_before_arrows_hurt_owner),
                max_slow_mo_speed: Fix::from_raw(bow.max_slow_mo_speed),
                max_arrows: bow.max_arrows,
                constants: constants(&bow.physics),
                transform_ticks: ticks(&bow.anims.transform),
                load_frame_ticks,
                fire_air_ticks: ticks(&bow.anims.fire_air),
                fire_ground_ticks: ticks(&bow.anims.fire_ground),
            },
            arrow: ArrowData {
                radius: Fix::from_raw(raw.prefabs.arrow.radius),
                body: {
                    let b = &raw.prefabs.arrow.body;
                    body(
                        &RawBody {
                            mass: b.mass,
                            bounciness: b.bounciness,
                            static_friction: b.friction,
                            dynamic_friction: b.dynamic_friction,
                            moment_of_inertia: b.moment_of_inertia,
                            gravity_scale: b.gravity_scale,
                            air_drag: b.air_drag,
                            rotational_drag: b.rotational_drag,
                        },
                        Fix::ZERO,
                    )
                },
            },
            pickup: {
                let p = &raw.prefabs.ability_pickup;
                PickupData {
                    layer: p.layer,
                    radius: Fix::from_raw(p.radius),
                    inverse_mass: Fix::ONE / Fix::from_raw(p.mass),
                    offset: p.offset.get(),
                }
            },
            platform: PlatformAbilityData {
                cooldown: Fix::from_raw(plat.cooldown),
                hurtbox_radius: Fix::from_raw(plat.hurtbox_radius),
                time_in_exit: Fix::from_raw(plat.time_in_exit),
                size_change_speed: Fix::from_raw(plat.size_change_speed),
                max_size: Fix::from_raw(plat.max_size),
                min_size: Fix::from_raw(plat.min_size),
                rotation_speed: Fix::from_raw(plat.rotation_speed),
                max_area: Fix::from_raw(plat.max_area),
                max_size_time: Fix::from_raw(plat.max_size_time),
                face_height: Fix::from_raw(plat.face_height),
                ejection_offset: Fix::from_raw(plat.ejection_offset),
                ejection_curve: plat.ejection_curve.clone(),
                enter_scale_curve: plat.enter_scale_curve.clone(),
                anticipation_ticks: ticks(&plat.anims.anticipation),
                enter_ticks: ticks(&plat.anims.enter),
                to_exit_ticks: ticks(&plat.anims.to_exit),
                exit_ticks: ticks(&plat.anims.exit),
                spawn_extents: raw.prefabs.resizable_platform.extents.get(),
                spawn_radius: Fix::from_raw(raw.prefabs.resizable_platform.radius),
                spawn_mass_per_area: Fix::from_raw(raw.prefabs.resizable_platform.mass_per_area),
                remove_curve: raw.prefabs.resizable_platform.remove_curve.clone(),
                spawn_speed: Fix::from_raw(raw.prefabs.resizable_platform.animate_velocity.speed),
                spawn_mu: Fix::from_raw(raw.prefabs.resizable_platform.animate_velocity.mu),
                spawn_rotation_speed: Fix::from_raw(
                    raw.prefabs
                        .resizable_platform
                        .animate_velocity
                        .rotation_speed,
                ),
                spawn_rotation_mu: Fix::from_raw(
                    raw.prefabs.resizable_platform.animate_velocity.rotation_mu,
                ),
                spawn_body: body(&raw.prefabs.resizable_platform.body, Fix::ZERO),
                sudden_death_speed: Fix::from_raw(
                    raw.prefabs
                        .resizable_platform
                        .animate_velocity
                        .sudden_death_speed,
                ),
                sudden_death_max_delay: Fix::from_raw(
                    raw.prefabs
                        .resizable_platform
                        .animate_velocity
                        .sudden_death_max_delay,
                ),
                sudden_death_acceleration: Fix::from_raw(
                    raw.prefabs
                        .resizable_platform
                        .animate_velocity
                        .sudden_death_acceleration,
                ),
                sudden_death_y_speed: raw
                    .prefabs
                    .resizable_platform
                    .animate_velocity
                    .sudden_death_y_speed
                    .clone(),
            },
            gust: GustData {
                cast_ticks: ticks(&raw.prefabs.gust.anims.cast_air),
                exit_ticks: ticks(&raw.prefabs.gust.anims.exit),
                shockwave_offset: raw.prefabs.gust.shockwave_offset.get(),
                collider_offset: raw.prefabs.gust.collider_offset.get(),
                hurtbox_radius: Fix::from_raw(raw.prefabs.gust.hurtbox_radius),
                platform_force: Fix::from_raw(raw.prefabs.gust.platform_force),
                default_force: Fix::from_raw(raw.prefabs.gust.default_force),
                grounded_multiplier: Fix::from_raw(raw.prefabs.gust.grounded_multiplier),
                radius: Fix::from_raw(raw.prefabs.gust.radius),
                curve: raw.prefabs.gust.curve.clone(),
            },
            revival: {
                let revival = &raw.prefabs.revival;
                RevivalData {
                    cast_ticks: ticks(&revival.anims.cast_air),
                    exit_ticks: ticks(&revival.anims.exit),
                    shockwave_offset: revival.shockwave_offset.get(),
                    collider_offset: revival.collider_offset.get(),
                    hurtbox_radius: Fix::from_raw(revival.hurtbox_radius),
                    marker_layer: revival.marker.layer,
                    marker_updatable_count: revival.marker.updatable_count,
                    marker_collider_hierarchy: revival.marker.collider_hierarchy,
                    marker_offset: revival.marker.offset.get(),
                    marker_radius: Fix::from_raw(revival.marker.radius),
                    marker_body: body(&revival.marker.body, Fix::ZERO),
                    death_effect_updatable_count: revival.death_effect_updatable_count,
                    effect_updatable_count: revival.effect_updatable_count,
                    particle_updatable_count: revival.particle_updatable_count,
                    effect_wisp_ticks: revival.effect_wisp_ticks,
                    particle_speed: Fix::from_raw(revival.particle_speed),
                }
            },
            teleport: {
                let teleport = &raw.prefabs.teleport;
                TeleportData {
                    up_cast_offset: Fix::from_raw(teleport.up_cast_offset),
                    player_radius: Fix::from_raw(teleport.player_radius),
                    marker_layer: teleport.marker.layer,
                    marker_updatable_count: teleport.marker.updatable_count,
                    marker_collider_hierarchy: teleport.marker.collider_hierarchy,
                    marker_offset: teleport.marker.offset.get(),
                    marker_radius: Fix::from_raw(teleport.marker.radius),
                    marker_teleport_radius: Fix::from_raw(teleport.marker.teleport_radius),
                    marker_body: body(&teleport.marker.body, Fix::ZERO),
                    effect_updatable_count: teleport.effect_updatable_count,
                    effect_duration: Fix::from_raw(teleport.effect_duration),
                }
            },
            grenade: {
                let grenade = &raw.prefabs.grenade;
                let throw = &grenade.throw_data;
                GrenadeData {
                    hurtbox_extents: grenade.hurtbox_extents.get(),
                    air_position: throw.air_position.get(),
                    ground_position: throw.ground_position.get(),
                    ground_dummy: throw.ground_dummy.iter().map(RawVec2::get).collect(),
                    air_dummy: throw.air_dummy.iter().map(RawVec2::get).collect(),
                    hurt_owner_delay: Fix::from_raw(throw.hurt_owner_delay),
                    max_force: Fix::from_raw(throw.max_force),
                    min_force: Fix::from_raw(throw.min_force),
                    force_gain_speed: Fix::from_raw(throw.force_gain_speed),
                    max_slow_mo_speed: Fix::from_raw(throw.max_slow_mo_speed),
                    knockback: Fix::from_raw(throw.knockback),
                    constants: constants(&throw.physics),
                    ground_enter_ticks: ticks(&throw.animations.ground_enter),
                    ground_exit_ticks: ticks(&throw.animations.ground_exit),
                    air_enter_ticks: ticks(&throw.animations.air_enter),
                    air_exit_ticks: ticks(&throw.animations.air_exit),
                    item_layer: grenade.item.layer,
                    item_updatable_count: grenade.item.updatable_count,
                    item_collider_hierarchy: grenade.item.collider_hierarchy,
                    item_radius: Fix::from_raw(grenade.item.radius),
                    item_body: body(&grenade.item.body, Fix::ZERO),
                    detonation_time: Fix::from_raw(grenade.item.detonation_time),
                    clone_count: grenade.item.clone_count,
                    duplicate_pattern: grenade.item.duplicate_pattern,
                    duplicate_spacing: Fix::from_raw(grenade.item.duplicate_spacing),
                    duplicate_angle: Fix::from_raw(grenade.item.duplicate_angle),
                    explosion_layer: grenade.explosion.layer,
                    explosion_updatable_count: grenade.explosion.updatable_count,
                    explosion_collider_hierarchy: grenade.explosion.collider_hierarchy,
                    explosion_extents: grenade.explosion.extents.get(),
                    explosion_animation_ticks: grenade.explosion.animation_ticks,
                    explosion_shockwaves: grenade.explosion.shockwaves.map(|wave| {
                        ExplosionShockwaveData {
                            platform_force: Fix::from_raw(wave.platform_force),
                            default_force: Fix::from_raw(wave.default_force),
                            grounded_player_multiplier: Fix::from_raw(
                                wave.grounded_player_multiplier,
                            ),
                            radius: Fix::from_raw(wave.radius),
                            collision_mask: wave.collision_mask,
                        }
                    }),
                }
            },
            tack: {
                let tack = &raw.prefabs.tack;
                let throw = &tack.throw_data;
                TackData {
                    hurtbox_extents: tack.hurtbox_extents.get(),
                    air_position: throw.air_position.get(),
                    ground_position: throw.ground_position.get(),
                    ground_dummy: throw.ground_dummy.iter().map(RawVec2::get).collect(),
                    air_dummy: throw.air_dummy.iter().map(RawVec2::get).collect(),
                    hurt_owner_delay: Fix::from_raw(throw.hurt_owner_delay),
                    max_force: Fix::from_raw(throw.max_force),
                    min_force: Fix::from_raw(throw.min_force),
                    force_gain_speed: Fix::from_raw(throw.force_gain_speed),
                    max_slow_mo_speed: Fix::from_raw(throw.max_slow_mo_speed),
                    knockback: Fix::from_raw(throw.knockback),
                    constants: constants(&throw.physics),
                    ground_enter_ticks: ticks(&throw.animations.ground_enter),
                    ground_exit_ticks: ticks(&throw.animations.ground_exit),
                    air_enter_ticks: ticks(&throw.animations.air_enter),
                    air_exit_ticks: ticks(&throw.animations.air_exit),
                    item_layer: tack.item.layer,
                    item_updatable_count: tack.item.updatable_count,
                    item_collider_hierarchy: tack.item.collider_hierarchy,
                    item_radius: Fix::from_raw(tack.item.radius),
                    item_body: body(&tack.item.body, Fix::ZERO),
                    clone_count: tack.item.clone_count,
                    duplicate_pattern: tack.item.duplicate_pattern,
                    duplicate_spacing: Fix::from_raw(tack.item.duplicate_spacing),
                    duplicate_angle: Fix::from_raw(tack.item.duplicate_angle),
                    arm_seconds: Fix::from_raw(tack.item.arm_seconds),
                    lifetime_seconds: Fix::from_raw(tack.item.lifetime_seconds),
                }
            },
            abilities: raw
                .abilities
                .into_iter()
                .enumerate()
                .map(|(index, ability)| {
                    assert_eq!(ability.id as usize, index, "ability catalog must be dense");
                    AbilityData {
                        id: ability.id,
                        name: ability.name,
                        prefab: ability.prefab,
                        cooldown: Fix::from_raw(ability.cooldown),
                        ground_only: ability.ground_only,
                        stopped_during_time_stop: ability.stopped_during_time_stop,
                        castable: ability.castable,
                        instant: ability.instant,
                        rope: ability.rope,
                        offensive: ability.offensive,
                        updatable_count: ability.updatable_count,
                        effect_cooldown: ability.effect_cooldown.map(Fix::from_raw),
                        effect_duration: ability.effect_duration.map(Fix::from_raw),
                        air_jump_strength: ability.air_jump_strength.map(Fix::from_raw),
                        form_jump_speed: ability.form_jump_speed.map(Fix::from_raw),
                    }
                })
                .collect(),
        }
    })
}
