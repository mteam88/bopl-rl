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
    interpolate_rotation_home: bool,
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
    start_velocity: RawVec2,
    start_angular_velocity: i64,
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
    intro_order: usize,
    position: RawVec2,
    rotation: i64,
    scale: i64,
    extents: RawVec2,
    radius: i64,
    min_scale: i64,
    platform_type: u32,
    grow_scale: i64,
    hierarchy: u32,
    updatable_count: u32,
    duplicate_collider_hierarchy: u32,
    animate_velocity: RawAnimateVelocity,
    body: RawBody,
    orbit: Option<RawOrbit>,
    vector_field: Option<RawVectorField>,
}
#[derive(Deserialize)]
struct RawVectorField {
    delay: i64,
    orbit_speed: i64,
    expand_speed: i64,
    center: RawVec2,
    normal_speed_friction: i64,
    dead_zone_dist_squared: i64,
    orbit_acceleration_multiplier: i64,
    target_radius: i64,
    ovalness: i64,
}
#[derive(Deserialize)]
struct RawWater {
    position: RawVec2,
    extents: RawVec2,
    hierarchy: u32,
}
#[derive(Deserialize)]
struct RawSession {
    team_spawns: Vec<RawVec2>,
    teammate_spacing: i64,
    time_before_sudden_death: i64,
    level_type: u32,
    level_animation_radius: i64,
    max_random_delay: i64,
    level_animation_time: i64,
    start_animation_progress: i64,
    animate_in: Curve,
}
#[derive(Deserialize)]
struct RawMap {
    platforms: Vec<RawPlatform>,
    water: Option<RawWater>,
    session: RawSession,
    ability_spawners: Vec<RawAbilitySpawner>,
}
#[derive(Deserialize)]
struct RawAbilitySpawner {
    position: RawVec2,
    hierarchy: u32,
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
    boundary_effect_updatable_count: u32,
}
#[derive(Deserialize)]
struct RawDash {
    cooldown: i64,
    hurtbox_extents: RawVec2,
    fields: RawDashFields,
}
#[derive(Deserialize)]
struct RawDashFields {
    #[serde(rename = "leaveSpeed.m_rawValue")]
    leave_speed: i64,
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
    collider_hierarchy: u32,
    updatable_count: u32,
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
    offset: RawVec2,
    collider_hierarchy: u32,
    updatable_count: u32,
    body: RawBody,
    up_force: i64,
    wobble_speed: i64,
    wobble_amount: i64,
    launch_force: i64,
    disabled_seconds: i64,
    swap_period: i64,
    vibration_strength: i64,
    vibration_duration: i64,
    vibrate_timing_offset: i64,
    gravity: i64,
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
    black_hole: RawBlackHole,
    gust: RawGust,
    revival: RawRevival,
    teleport: RawTeleport,
    portal_gun: RawPortalGun,
    beam: RawBeam,
    ropes: RawRopes,
    grenade: RawGrenade,
    missile: RawMissile,
    mine: RawMine,
    macho_throw: RawMachoThrow,
    drill: RawDrill,
    rock: RawRock,
    roll: RawRoll,
    meteor: RawMeteor,
    push: RawPush,
    freeze: RawFreeze,
    blink: RawBlink,
    scale_rays: RawScaleRays,
    duplicator: RawDuplicator,
    magnet: RawMagnet,
    smoke: RawSmoke,
    time_stop: RawTimeStop,
    spike: RawSpike,
    engine: RawEngine,
    tesla: RawTesla,
    tack: RawTack,
    mjolnir: RawMjolnir,
    poke_ball: RawPokeBall,
}

#[derive(Deserialize)]
struct RawRopes {
    pool_capacity: usize,
    max_segments: usize,
    max_contacts: usize,
    segments: usize,
    max_ropes_per_player: usize,
    gravity: i64,
    iterations: usize,
    stretchiness: i64,
    max_stretch: i64,
    grounded_strength: i64,
    segment_radius: i64,
    player_gravity: i64,
    platform_mass: i64,
    platform_heaviness: i64,
    platform_pull: i64,
    boulder_pull: i64,
    platform_break_stretch: i64,
    non_platform_mass: i64,
    web_max_length: i64,
    web_shorten: i64,
    web_cooldown: i64,
    web_miss_cooldown: i64,
    web_release_boost: i64,
    web_grounded_pop: i64,
    grapple_spawn_offset: i64,
    grapple_speed: i64,
    grapple_reel_frames: usize,
    grapple_start_separation: i64,
    grapple_reeled_separation: i64,
    grapple_reel_speed: i64,
    grapple_delete_separation: i64,
    grapple_cooldown: i64,
    grapple_player_cast_radius: i64,
    hook_layer: u32,
    hook_radius: i64,
    hook_collider_hierarchy: u32,
    hook_updatable_count: u32,
    hook_segments_affected: usize,
    hook_speed_falloff: i64,
    hook_fall_time: i64,
    hook_pull: i64,
    rope_reel_out_interval: i64,
    rope_segments_affected: usize,
    rope_segment_wave_length: usize,
    rope_reel_out_segment_speed: i64,
    rope_sprinkler_speed: i64,
    rope_max_travel_time: i64,
}

#[derive(Deserialize)]
struct RawDrill {
    hurtbox_extents: RawVec2,
    collider_offset: RawVec2,
    collider_hierarchy: u32,
    enter_ticks: u32,
    exit_ticks: u32,
    drill_offset: RawVec2,
    drill_hitbox: RawVec2,
    gravity: i64,
    drill_strength: i64,
    rotation_speed_air: i64,
    leave_ground_speed_multiplier: i64,
    enter_ground_speed_multiplier: i64,
    speed_to_rotation: Curve,
    on_enter_offset: RawVec2,
    min_underground_speed: i64,
    keep_strong_gravity_seconds: i64,
    strong_gravity: i64,
    max_early_fall_speed: i64,
    hitboxes: Vec<RawDrillHitbox>,
    hitbox_layer: u32,
    hitbox_updatable_count: u32,
    hitbox_collider_hierarchy: u32,
}

#[derive(Deserialize)]
struct RawDrillHitbox {
    offset: RawVec2,
    extents: RawVec2,
}

#[derive(Deserialize)]
struct RawBlackHole {
    hurtbox_extents: RawVec2,
    collider_hierarchy: u32,
    collider_offset: RawVec2,
    physics: RawPhysics,
    animations: RawBlackHoleAnimations,
    spawn_offset: RawVec2,
    hole: RawBlackHoleBody,
}
#[derive(Deserialize)]
struct RawBlackHoleAnimations {
    enter: Vec<i32>,
    #[serde(rename = "enterAir")]
    enter_air: Vec<i32>,
    exit: Vec<i32>,
    #[serde(rename = "exitAir")]
    exit_air: Vec<i32>,
}
#[derive(Deserialize)]
struct RawBlackHoleBody {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    start_radius: i64,
    starting_mass: i64,
    gravity: i64,
    controlled_platform_multiplier: i64,
    growth: i64,
    base_growth: i64,
    merge_growth: i64,
    size_curve: Curve,
    min_size: i64,
    drag: i64,
    player_force: i64,
    influence_radius_multiplier: i64,
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
struct RawPokeBall {
    hurtbox_extents: RawVec2,
    #[serde(rename = "throw")]
    throw_data: RawThrowItem,
    item: RawPokeBallItem,
}
#[derive(Deserialize)]
struct RawPokeBallItem {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    radius: i64,
    body: RawBody,
    clone_count: u32,
    duplicate_pattern: u32,
    duplicate_spacing: i64,
    duplicate_angle: i64,
    capture_seconds: i64,
    fizzle_seconds: i64,
    pop_up_speed: i64,
}
#[derive(Deserialize)]
struct RawMjolnir {
    hurtbox_extents: RawVec2,
    #[serde(rename = "throw")]
    throw_data: RawThrowItem,
    item: RawMjolnirItem,
}
#[derive(Deserialize)]
struct RawMjolnirItem {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    radius: i64,
    body: RawBody,
    clone_count: u32,
    duplicate_pattern: u32,
    duplicate_spacing: i64,
    duplicate_angle: i64,
    outbound_drag: i64,
    return_speed_threshold: i64,
    max_flight_seconds: i64,
    recall_accel: i64,
    recall_max_speed: i64,
    clang_grace_seconds: i64,
    platform_push_force: i64,
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

#[derive(Deserialize)]
struct RawMissile {
    hurtbox_extents: RawVec2,
    projectile_offset: RawVec2,
    projectile_start_speed: i64,
    rotation_speed: i64,
    physics: RawPhysics,
    animations: RawMissileAnimations,
    spark: RawMissileSpark,
    explosion: RawMissileExplosion,
}
#[derive(Deserialize)]
struct RawMissileAnimations {
    enter: Vec<i32>,
    #[serde(rename = "enterToIdle")]
    enter_to_idle: Vec<i32>,
    idle: Vec<i32>,
}
#[derive(Deserialize)]
struct RawMissileSpark {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    extents: RawVec2,
    body: RawBody,
    minimum_forward_speed: i64,
    accel: i64,
    instability_accel: i64,
    clone_count: u32,
    duplicate_pattern: u32,
    duplicate_spacing: i64,
    duplicate_angle: i64,
}
#[derive(Deserialize)]
struct RawMissileExplosion {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    extents: RawVec2,
    animation_ticks: u32,
    shockwave: RawExplosionShockwave,
}

#[derive(Deserialize)]
struct RawMine {
    hurtbox_extents: RawVec2,
    collider_offset: RawVec2,
    #[serde(rename = "throw")]
    throw_data: RawMineThrow,
    item: RawMineItem,
    explosion: RawMissileExplosion,
}

#[derive(Deserialize)]
struct RawMachoThrow {
    hurtbox_extents: RawVec2,
    collider_offset: RawVec2,
    collider_hierarchy: u32,
    physics: RawPhysics,
    animations: RawMachoAnimations,
    dummy_positions: Vec<RawVec2>,
    prepare_positions: Vec<RawVec2>,
    force_from_boulder: i64,
    smoke_updatable_count: u32,
    player_radius_hack: i64,
    move_after_throw_frames: u32,
    jump_after_throw_frames: u32,
    rotation_velocity: i64,
    ground_angular_multiplier: i64,
    max_attachable_distance: i64,
    shrink_after_frames: u32,
    projectile_position: RawVec2,
    quick_projectile_position: RawVec2,
    throw_speed: i64,
    min_throw_vector: RawVec2,
    extra_strength_scale: i64,
    quick_throw_strength: i64,
    quick_charge_scaling: i64,
    platform_scale_reduction: i64,
    ground_to_hands: i64,
    quick_position_swap_frame: u32,
    boulder: RawMachoBoulder,
}

#[derive(Deserialize)]
struct RawMachoAnimations {
    enter: Vec<i32>,
    idle: Vec<i32>,
    #[serde(rename = "idleToWalk")]
    idle_to_walk: Vec<i32>,
    walk: Vec<i32>,
    #[serde(rename = "beginThrow")]
    begin_throw: Vec<i32>,
    exit: Vec<i32>,
    turn: Vec<i32>,
    pickup: Vec<i32>,
    grab: Vec<i32>,
}

#[derive(Deserialize)]
struct RawMachoBoulder {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    extents: RawVec2,
    radius: i64,
    min_scale: i64,
    max_scale: i64,
    body: RawBody,
    dropped_bounciness: i64,
    down_effect_updatable_count: u32,
}

#[derive(Deserialize)]
struct RawRock {
    radius: i64,
    collider_hierarchy: u32,
    body: RawBody,
    max_duration: i64,
    enter_scale: Curve,
    is_cancellable: bool,
    exit_time: i64,
    exit_scale: Curve,
    player_collision_bounce: i64,
    shockwave_force: i64,
    min_rope_length: i64,
    rope_pull_strength: i64,
    squared_max_rope_separation: i64,
    rope_player_mass: i64,
}

#[derive(Deserialize)]
struct RawPush {
    hurtbox_extents: RawVec2,
    collider_hierarchy: u32,
    collider_offset: RawVec2,
    radius: i64,
    grab_ground_ticks: u32,
    platform_force: i64,
    scene_bound_limit_offset: i64,
    boulder_force_multiplier: i64,
    home_position_lerp_speed: i64,
}
#[derive(Deserialize)]
struct RawFreeze {
    hurtbox_extents: RawVec2,
    collider_hierarchy: u32,
    collider_offset: RawVec2,
    radius: i64,
    physics: RawPhysics,
    air_control_blend: i64,
    firepoint_offset: RawVec2,
    knockback: i64,
    max_slow_mo_speed: i64,
    transform_ticks: u32,
    fire_ticks: u32,
    collision_mask: u32,
    collision_mask_no_water: u32,
    freeze_seconds: i64,
    frozen_updatable_count: u32,
}
#[derive(Deserialize)]
struct RawBlink {
    hurtbox_extents: RawVec2,
    collider_hierarchy: u32,
    collider_offset: RawVec2,
    radius: i64,
    physics: RawPhysics,
    air_control_blend: i64,
    firepoint_offset: RawVec2,
    knockback: i64,
    max_slow_mo_speed: i64,
    transform_ticks: u32,
    fire_ticks: u32,
    collision_mask: u32,
    min_player_duration: i64,
    wall_duration: i64,
    wall_delay: i64,
    wall_shake: i64,
    on_player_hit_cooldown: i64,
    tunnel_release_duration: i64,
    tunnel_updatable_count: u32,
}
#[derive(Deserialize)]
struct RawRoll {
    radius: i64,
    collider_hierarchy: u32,
    physics: RawPhysics,
    max_speed: i64,
    min_speed: i64,
    speed_curve: Curve,
    charge_time: i64,
    dash_time: i64,
    move_collision_density: i64,
    collision_mask: u32,
    platform_turn_speed: i64,
    early_release_jump_strength: i64,
    short_jump_strength: i64,
    trail_hierarchy_count: u32,
}
#[derive(Deserialize)]
struct RawMeteor {
    hurtbox_extents: RawVec2,
    collider_hierarchy: u32,
    collider_offset: RawVec2,
    physics: RawPhysics,
    time_before_drop: i64,
    jump_speed: i64,
    hold_speed: i64,
    max_hold_time: i64,
    min_hold_time: i64,
    speed_curve: Curve,
    vibration_strength: i64,
    fixed_updates_in_dash: u32,
    ground_force: i64,
    player_push_force: i64,
    shake_strength: i64,
    shake_duration: i64,
    spin_curve: Curve,
    raycast_mask: u32,
    ground_enter_ticks: u32,
    air_ticks: u32,
    air_start_hold_ticks: u32,
    shockwave: RawMeteorShockwave,
    smear: RawMeteorSmear,
}
#[derive(Deserialize)]
struct RawMeteorShockwave {
    updatable_count: u32,
    offset: RawVec2,
    platform_force: i64,
    default_force: i64,
    grounded_player_multiplier: i64,
    radius: i64,
    collision_mask: u32,
    curve: Curve,
}
#[derive(Deserialize)]
struct RawMeteorSmear {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    extents: RawVec2,
    offset: RawVec2,
    lifespan_ticks: u32,
}
#[derive(Deserialize)]
struct RawScaleRays {
    growth: RawScaleRay,
    shrink: RawScaleRay,
}
#[derive(Deserialize)]
struct RawScaleRay {
    hurtbox_extents: RawVec2,
    collider_hierarchy: u32,
    collider_offset: RawVec2,
    radius: i64,
    physics: RawPhysics,
    air_control_blend: i64,
    firepoint_offset: RawVec2,
    knockback: i64,
    max_slow_mo_speed: i64,
    transform_ticks: u32,
    fire_ticks: u32,
    collision_mask: u32,
    collision_mask_no_water: u32,
    scale_curve: Curve,
    player_multiplier: i64,
    multiplier: i64,
    small_non_players_multiplier: i64,
    inverse_mass_before_big: i64,
    changer_updatable_count: u32,
}
#[derive(Deserialize)]
struct RawDuplicator {
    hurtbox_extents: RawVec2,
    collider_hierarchy: u32,
    collider_offset: RawVec2,
    radius: i64,
    physics: RawPhysics,
    air_control_blend: i64,
    firepoint_offset: RawVec2,
    knockback: i64,
    max_slow_mo_speed: i64,
    transform_ticks: u32,
    fire_ticks: u32,
    triple_raycast_radius: i64,
    collision_mask: u32,
    collision_mask_no_water: u32,
    platform_shake_duration: i64,
    platform_shake: i64,
    platform_overlap_force: i64,
    platform_effect_updatable_count: u32,
}
#[derive(Deserialize)]
struct RawMagnet {
    hurtbox_extents: RawVec2,
    collider_hierarchy: u32,
    collider_offset: RawVec2,
    radius: i64,
    physics: RawPhysics,
    air_control_blend: i64,
    firepoint_offset: RawVec2,
    knockback: i64,
    player_rock_extra_offset: RawVec2,
    boulder_extra_offset: RawVec2,
    held_lerp: i64,
    fire_force_mines: i64,
    fire_force: i64,
    fire_ragdoll_force: i64,
    fire_black_hole_force: i64,
    fire_rock_force: i64,
    fire_missile_force: i64,
    fire_boulder_force: i64,
    roped_force_multiplier: i64,
    max_slow_mo_speed: i64,
    only_pull: u32,
    only_pickup: u32,
    pickup_radius: i64,
    max_lock_distance_squared: i64,
    close_pull_multiplier: i64,
    fast_pull_radius_squared: i64,
    pull_width: i64,
    pull_strength: i64,
    player_pull_strength: i64,
    projectile_pull_strength: i64,
    wall_pull_strength: i64,
    black_hole_pull_strength: i64,
    macho_pull_strength: i64,
    player_mass: i64,
    locked_angular_velocity: i64,
    small_ragdoll_duration: i64,
    fire_ragdoll_duration: i64,
    wall_mask: u32,
    raycast_offset: RawVec2,
    max_pull_duration: i64,
    parry_push_strength: i64,
    pull_curve: Curve,
    enter_ticks: u32,
    prepull_ticks: u32,
    exit_ticks: u32,
}
#[derive(Deserialize)]
struct RawMineThrow {
    throw_position: RawVec2,
    throw_strength: i64,
    aerial_throw_strength: i64,
    knockback: i64,
    min_throw_strength: i64,
    only_throw_up: bool,
    physics: RawPhysics,
    animations: RawMineAnimations,
}
#[derive(Deserialize)]
struct RawMineAnimations {
    #[serde(rename = "gEnter")]
    ground_enter: Vec<i32>,
    #[serde(rename = "gLoop")]
    ground_loop: Vec<i32>,
    #[serde(rename = "gExit")]
    ground_exit: Vec<i32>,
    #[serde(rename = "aEnter")]
    air_enter: Vec<i32>,
    #[serde(rename = "aExit")]
    air_exit: Vec<i32>,
}
#[derive(Deserialize)]
struct RawMineItem {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    radius: i64,
    body: RawBody,
    clone_count: u32,
    duplicate_pattern: u32,
    duplicate_spacing: i64,
    duplicate_angle: i64,
    hover_strength: i64,
    hover_period: i64,
    time_before_armed: i64,
    chase_time: i64,
    igniters: u32,
    scans_for: u32,
    chase_strength: i64,
    scan_radius: i64,
    shake_force: i64,
    anticipation_duration: i64,
}

#[derive(Deserialize)]
struct RawSmoke {
    hurtbox_extents: RawVec2,
    #[serde(rename = "throw")]
    throw_data: RawThrowItem,
    item: RawSmokeItem,
    cloud: RawSmokeCloud,
    explosion: RawSmokeExplosion,
}
#[derive(Deserialize)]
struct RawSmokeItem {
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
    cloud_offsets: Vec<RawVec2>,
    spread: i64,
    noise: i64,
    burst_updatable_count: u32,
}
#[derive(Deserialize)]
struct RawSmokeCloud {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    base_radius: i64,
    body: RawBody,
    clone_count: u32,
    duplicate_pattern: u32,
    duplicate_spacing: i64,
    duplicate_angle: i64,
    radii: Vec<i64>,
    enter_frames: Vec<i32>,
    min_ignition_delay: i64,
    max_ignition_delay: i64,
}
#[derive(Deserialize)]
struct RawSmokeExplosion {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    extents: RawVec2,
    animation_ticks: u32,
}
#[derive(Deserialize)]
struct RawTimeStop {
    cast_time: i64,
    center_offset: RawVec2,
    exit_offset: RawVec2,
    collider_offset: RawVec2,
    hurtbox_radius: i64,
    collider_hierarchy: u32,
    animations: RawTimeStopAnimations,
    effect_updatable_count: u32,
    duration_f32: f32,
    exit_seconds_f32: f32,
}
#[derive(Deserialize)]
struct RawTimeStopAnimations {
    #[serde(rename = "idleToFall")]
    idle_to_fall: Vec<i32>,
    #[serde(rename = "castSpell")]
    cast_spell: Vec<i32>,
    #[serde(rename = "easeOutSpellCast")]
    ease_out: Vec<i32>,
}
#[derive(Deserialize)]
struct RawSpike {
    hurtbox_extents: RawVec2,
    collider_offset: RawVec2,
    collider_hierarchy: u32,
    animations: RawSpikeAnimations,
    surface_offset: i64,
    cast_force: i64,
    attack: RawSpikeAttack,
}
#[derive(Deserialize)]
struct RawSpikeAnimations {
    cast: Vec<i32>,
    recover: Vec<i32>,
}
#[derive(Deserialize)]
struct RawSpikeAttack {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    extents: RawVec2,
    collider_offset: RawVec2,
    body: RawBody,
    down_thrust: Curve,
    knock_player: i64,
    knock_wall: i64,
    self_push_multiplier: i64,
    push_time: i64,
}
#[derive(Deserialize)]
struct RawEngine {
    hurtbox_extents: RawVec2,
    collider_offset: RawVec2,
    collider_hierarchy: u32,
    animations: RawSpikeAnimations,
    plant_offset: RawVec2,
    cast_force: i64,
    buffer_size: usize,
    rocket: RawRocketEngine,
}
#[derive(Deserialize)]
struct RawRocketEngine {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    extents: RawVec2,
    collider_offset: RawVec2,
    body: RawBody,
    force: Curve,
    start_ticks: u32,
    end_ticks: u32,
    fire_offset: RawVec2,
    fire_radius: i64,
    smoke_mask: u32,
}
#[derive(Deserialize)]
struct RawTesla {
    hurtbox_extents: RawVec2,
    collider_offset: RawVec2,
    collider_hierarchy: u32,
    animations: RawSpikeAnimations,
    plant_offset: RawVec2,
    cast_force: i64,
    buffer_size: usize,
    node: RawTeslaNode,
    lightning: RawTeslaLightning,
}
#[derive(Deserialize)]
struct RawTeslaNode {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    radius: i64,
    collider_offset: RawVec2,
    body: RawBody,
    line_length: i64,
    hitbox_offset_y: i64,
}
#[derive(Deserialize)]
struct RawTeslaLightning {
    layer: u32,
    updatable_count: u32,
    collider_hierarchy: u32,
    extents: RawVec2,
    minimum_extent: i64,
    flicker_period: i64,
    enter_ticks: u32,
    flicker_ticks: u32,
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
    shockwave_updatable_count: u32,
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
    teleportable_mask: u32,
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
struct RawPortalGun {
    shot_speed: i64,
    spawn_offset: i64,
    shot_layer: u32,
    shot_updatable_count: u32,
    shot_collider_hierarchy: u32,
    shot_radius: i64,
    shot_body: RawBody,
    max_flight_seconds: i64,
    portal_updatable_count: u32,
    half_width: i64,
    thickness: i64,
    exit_offset: i64,
    lifetime_seconds: i64,
}

#[derive(Deserialize)]
struct RawBeam {
    hurtbox_extents: RawVec2,
    collider_offset: RawVec2,
    collider_hierarchy: u32,
    physics: RawPhysics,
    enter_ground_ticks: u32,
    enter_air_ticks: u32,
    air_orb_frame8_ticks: u32,
    air_orb_frame9_ticks: u32,
    air_orb_frame10_ticks: u32,
    exit_ground_ticks: u32,
    exit_air_ticks: u32,
    up_alignment: i64,
    min_grounded_angle: i64,
    exit_scale_curve: Curve,
    beam_offset: i64,
    air_extents: RawVec2,
    air_offset: RawVec2,
    angular_aim_speed: i64,
    angular_aim_curve: Curve,
    exit_up_speed: i64,
    max_time: i64,
    max_time_air: i64,
    exit_scale_time: i64,
    shake_duration: i64,
    shake_strength: i64,
    orb_layer: u32,
    orb_updatable_count: u32,
    orb_collider_hierarchy: u32,
    orb_radius: i64,
    orb_exit_ticks: u32,
    maximum_distance: i64,
    width: i64,
    width_curve: Curve,
    parent_offset: i64,
    platform_force: i64,
    black_hole_force: i64,
    rigid_body_force: i64,
    cross_offset: i64,
    hit_mask: u32,
    object_layer: u32,
    object_updatable_count: u32,
    object_collider_hierarchy: u32,
    base_width: i64,
    extra_length: i64,
    minimum_extent: i64,
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
    pub intro_order: usize,
    pub surface: Surface,
    /// The collider's authored half-extents (`startExtents * scale`); the
    /// surface carries the perimeter's `One / (One / e)` reading instead.
    /// DPhysicsRoundedRect's stored inverse extents. GrowOnStart updates this
    /// incrementally (`inverseExtents /= ratio`), which is observably different
    /// from recomputing `One / finalExtents` before the first contact.
    pub collider_inverse_extents: Vec2,
    /// Box.right/up are scaled directly by GrowOnStart's ratio. They therefore
    /// retain a different rounding history from `One / inverseExtents`.
    pub collider_axis_extents: Vec2,
    pub hierarchy: u32,
    pub updatable_count: u32,
    pub duplicate_collider_hierarchy: u32,
    pub min_scale: Fix,
    pub base_scale: Fix,
    pub ice: bool,
    /// AnimateVelocity home spring (speed * (home - position)) and damping
    /// (-mu * velocity).
    pub speed: Fix,
    pub mu: Fix,
    pub rotation_speed: Fix,
    pub rotation_mu: Fix,
    pub interpolate_rotation_home: bool,
    pub body: crate::physics::Body,
    /// Sudden death sinking (AnimateVelocity.ActivateSuddenDeath).
    pub sudden_death_speed: Fix,
    pub sudden_death_max_delay: Fix,
    pub sudden_death_acceleration: Fix,
    pub sudden_death_y_speed: Curve,
    /// Map 2's AntiLockPlatform orbit (force toward waypoints after a delay).
    pub orbit: Option<OrbitDef>,
    pub vector_field: Option<VectorFieldDef>,
}
pub struct OrbitDef {
    pub force: Fix,
    pub delay: Fix,
    pub mass: Fix,
    /// Per-tick velocity multiplier (PhysicsBody.airDrag from airFriction01).
    pub air_drag: Fix,
    pub path: Vec<Vec2>,
}
pub struct VectorFieldDef {
    pub delay: Fix,
    pub orbit_speed: Fix,
    pub expand_speed: Fix,
    pub center: Vec2,
    pub normal_speed_friction: Fix,
    pub dead_zone_dist_squared: Fix,
    pub orbit_acceleration_multiplier: Fix,
    pub target_radius: Fix,
    /// Serialized but unused by the shipped VectorFieldPlatform implementation.
    pub ovalness: Fix,
}
pub struct AbilitySpawnerDef {
    pub position: Vec2,
    pub hierarchy: u32,
}
pub struct MapDef {
    pub platforms: Vec<PlatformDef>,
    pub water: Option<(Vec2, Vec2, u32)>,
    pub level_type: u32,
    pub team_spawns: Vec<Vec2>,
    pub teammate_spacing: Fix,
    pub respawn_points: [Vec2; 3],
    pub sudden_death_time: Fix,
    pub ability_spawners: Vec<AbilitySpawnerDef>,
    pub level_animation_radius: Fix,
    pub max_random_delay: Fix,
    pub level_animation_time: Fix,
    pub start_animation_progress: Fix,
    pub animate_in: Curve,
}
impl MapDef {
    pub fn is_space(&self) -> bool {
        self.level_type == 2
    }

    pub fn water_height(&self) -> Option<Fix> {
        self.water
            .map(|(position, extents, _)| position.y + extents.y)
    }

    /// `SceneBounds.WaterHeight` (sim/Gameplay/SceneBounds.cs:11-33): the
    /// global kill plane the gameplay code reads, -50 on a space level and
    /// -11.3 everywhere else. Distinct from `water_height()`, which is the
    /// map's own authored water collider, and from `lower_boundary()`.
    pub fn scene_water_height(&self) -> Fix {
        if self.is_space() {
            Fix::from_int(-50)
        } else {
            Fix::from_f32(-11.3)
        }
    }

    pub fn lower_boundary(&self) -> Fix {
        if self.is_space() {
            // DestroyIfOutsideSceneBounds uses SceneBounds.Camera_YMin for
            // space objects, not the separate -50 space water height.
            Fix::from_int(-26)
        } else {
            // SceneBounds.WaterHeight, distinct from Constants.WATER_HEIGHT.
            Fix::from_f32(-11.3)
        }
    }
}
pub struct MovementData {
    pub constants: crate::world::MovementConstants,
    pub radius: Fix,
    pub hurtbox_extents: Vec2,
    pub spawn_anim_ticks: u32,
    pub fast_spawn_anim_ticks: u32,
    pub boundary_effect_updatable_count: u32,
}
pub struct BlackHoleData {
    pub hurtbox_extents: Vec2,
    pub collider_hierarchy: u32,
    pub collider_offset: Vec2,
    pub physics_radius: Fix,
    pub constants: crate::world::MovementConstants,
    pub enter_ticks: u32,
    pub enter_air_ticks: u32,
    pub exit_ticks: u32,
    pub exit_air_ticks: u32,
    pub spawn_offset: Vec2,
    pub layer: u32,
    pub updatable_count: u32,
    pub hole_collider_hierarchy: u32,
    pub start_radius: Fix,
    pub starting_mass: Fix,
    pub gravity: Fix,
    pub controlled_platform_multiplier: Fix,
    pub growth: Fix,
    pub base_growth: Fix,
    pub merge_growth: Fix,
    pub size_curve: Curve,
    pub min_size: Fix,
    pub drag: Fix,
    pub player_force: Fix,
    pub influence_radius_multiplier: Fix,
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
    pub collider_hierarchy: u32,
    pub updatable_count: u32,
    /// BoplBody.NewPhysicsBody for the Arrow prefab (BoplBody.cs:215-229).
    pub body: crate::physics::Body,
}
/// AbilityPickup_Dynamic prefab: the circle a final death drops
/// (SlimeController.DropAbilities).
pub struct PickupData {
    pub layer: u32,
    pub radius: Fix,
    /// DPhysicsCircle centre offset from the FixTransform (DetPhysics.cs:2513).
    pub offset: Vec2,
    pub collider_hierarchy: u32,
    pub updatable_count: u32,
    pub body: crate::physics::Body,
    pub up_force: Fix,
    pub wobble_speed: Fix,
    pub wobble_amount: Fix,
    pub launch_force: Fix,
    pub disabled_seconds: Fix,
    pub swap_period: Fix,
    pub vibration_strength: Fix,
    pub vibration_duration: Fix,
    pub vibrate_timing_offset: Fix,
    pub gravity: Fix,
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
    pub leave_speed: Fix,
}
pub struct GustData {
    pub cast_ticks: u32,
    pub exit_ticks: u32,
    pub shockwave_offset: Vec2,
    pub collider_offset: Vec2,
    pub hurtbox_radius: Fix,
    pub shockwave_updatable_count: u32,
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
    pub teleportable_mask: u32,
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
pub struct PortalGunData {
    pub shot_speed: Fix,
    pub spawn_offset: Fix,
    pub shot_layer: u32,
    pub shot_updatable_count: u32,
    pub shot_collider_hierarchy: u32,
    pub shot_radius: Fix,
    pub shot_body: crate::physics::Body,
    pub max_flight_seconds: Fix,
    pub portal_updatable_count: u32,
    pub half_width: Fix,
    pub thickness: Fix,
    pub exit_offset: Fix,
    pub lifetime_seconds: Fix,
}
pub struct BeamData {
    pub hurtbox_extents: Vec2,
    pub collider_offset: Vec2,
    pub collider_hierarchy: u32,
    pub constants: crate::world::MovementConstants,
    pub radius: Fix,
    pub enter_ground_ticks: u32,
    pub enter_air_ticks: u32,
    pub air_orb_frame8_ticks: u32,
    pub air_orb_frame9_ticks: u32,
    pub air_orb_frame10_ticks: u32,
    pub exit_ground_ticks: u32,
    pub exit_air_ticks: u32,
    pub up_alignment: Fix,
    pub min_grounded_angle: Fix,
    pub exit_scale_curve: Curve,
    pub beam_offset: Fix,
    pub air_extents: Vec2,
    pub air_offset: Vec2,
    pub angular_aim_speed: Fix,
    pub angular_aim_curve: Curve,
    pub exit_up_speed: Fix,
    pub max_time: Fix,
    pub max_time_air: Fix,
    pub exit_scale_time: Fix,
    pub shake_duration: Fix,
    pub shake_strength: Fix,
    pub orb_layer: u32,
    pub orb_updatable_count: u32,
    pub orb_collider_hierarchy: u32,
    pub orb_radius: Fix,
    pub orb_exit_ticks: u32,
    pub maximum_distance: Fix,
    pub width: Fix,
    pub width_curve: Curve,
    pub parent_offset: Fix,
    pub platform_force: Fix,
    pub black_hole_force: Fix,
    pub rigid_body_force: Fix,
    pub cross_offset: Fix,
    pub hit_mask: u32,
    pub object_layer: u32,
    pub object_updatable_count: u32,
    pub object_collider_hierarchy: u32,
    pub base_width: Fix,
    pub extra_length: Fix,
    pub minimum_extent: Fix,
}
pub struct RopeData {
    pub pool_capacity: usize,
    pub max_segments: usize,
    pub max_contacts: usize,
    pub segments: usize,
    pub max_ropes_per_player: usize,
    pub gravity: Fix,
    pub iterations: usize,
    pub stretchiness: Fix,
    pub max_stretch: Fix,
    pub grounded_strength: Fix,
    pub segment_radius: Fix,
    pub player_gravity: Fix,
    pub platform_mass: Fix,
    pub platform_heaviness: Fix,
    pub platform_pull: Fix,
    pub boulder_pull: Fix,
    pub platform_break_stretch: Fix,
    pub non_platform_mass: Fix,
    pub web_max_length: Fix,
    pub web_shorten: Fix,
    pub web_cooldown: Fix,
    pub web_miss_cooldown: Fix,
    pub web_release_boost: Fix,
    pub web_grounded_pop: Fix,
    pub grapple_spawn_offset: Fix,
    pub grapple_speed: Fix,
    pub grapple_reel_frames: usize,
    pub grapple_start_separation: Fix,
    pub grapple_reeled_separation: Fix,
    pub grapple_reel_speed: Fix,
    pub grapple_delete_separation: Fix,
    pub grapple_cooldown: Fix,
    pub grapple_player_cast_radius: Fix,
    pub hook_layer: u32,
    pub hook_radius: Fix,
    pub hook_collider_hierarchy: u32,
    pub hook_updatable_count: u32,
    pub hook_segments_affected: usize,
    pub hook_speed_falloff: Fix,
    pub hook_fall_time: Fix,
    pub hook_pull: Fix,
    pub rope_reel_out_interval: Fix,
    pub rope_segments_affected: usize,
    pub rope_segment_wave_length: usize,
    pub rope_reel_out_segment_speed: Fix,
    pub rope_sprinkler_speed: Fix,
    pub rope_max_travel_time: Fix,
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
pub struct MissileData {
    pub hurtbox_extents: Vec2,
    pub physics_radius: Fix,
    pub projectile_offset: Vec2,
    pub projectile_start_speed: Fix,
    pub rotation_speed: Fix,
    pub constants: crate::world::MovementConstants,
    pub enter_ticks: u32,
    pub enter_to_idle_ticks: u32,
    pub idle_ticks: u32,
    pub spark_layer: u32,
    pub spark_updatable_count: u32,
    pub spark_collider_hierarchy: u32,
    pub spark_extents: Vec2,
    pub spark_body: crate::physics::Body,
    pub minimum_forward_speed: Fix,
    pub accel: Fix,
    pub instability_accel: Fix,
    pub clone_count: u32,
    pub duplicate_pattern: u32,
    pub duplicate_spacing: Fix,
    pub duplicate_angle: Fix,
    pub explosion_layer: u32,
    pub explosion_updatable_count: u32,
    pub explosion_collider_hierarchy: u32,
    pub explosion_extents: Vec2,
    pub explosion_animation_ticks: u32,
    pub explosion_shockwave: ExplosionShockwaveData,
}
pub struct MineData {
    pub hurtbox_extents: Vec2,
    pub collider_offset: Vec2,
    pub throw_position: Vec2,
    pub throw_strength: Fix,
    pub aerial_throw_strength: Fix,
    pub knockback: Fix,
    pub min_throw_strength: Fix,
    pub only_throw_up: bool,
    pub constants: crate::world::MovementConstants,
    pub ground_enter_ticks: u32,
    pub ground_loop_ticks: u32,
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
    pub hover_strength: Fix,
    pub hover_period: Fix,
    pub time_before_armed: Fix,
    pub chase_time: Fix,
    pub igniters: u32,
    pub scans_for: u32,
    pub chase_strength: Fix,
    pub scan_radius: Fix,
    pub shake_force: Fix,
    pub anticipation_duration: Fix,
    pub explosion_layer: u32,
    pub explosion_updatable_count: u32,
    pub explosion_collider_hierarchy: u32,
    pub explosion_extents: Vec2,
    pub explosion_animation_ticks: u32,
    pub explosion_shockwave: ExplosionShockwaveData,
}

pub struct MachoThrowData {
    pub hurtbox_extents: Vec2,
    pub collider_offset: Vec2,
    pub collider_hierarchy: u32,
    pub physics_radius: Fix,
    pub constants: crate::world::MovementConstants,
    pub enter_ticks: u32,
    pub idle_ticks: u32,
    pub idle_to_walk_ticks: u32,
    pub walk_ticks: u32,
    pub begin_throw_ticks: u32,
    pub exit_ticks: u32,
    pub turn_ticks: u32,
    pub pickup_ticks: u32,
    pub grab_ticks: u32,
    pub enter_frame_ticks: Vec<u32>,
    pub grab_frame_ticks: Vec<u32>,
    pub dummy_frame_ticks: Vec<u32>,
    pub dummy_positions: Vec<Vec2>,
    pub prepare_frame_ticks: Vec<u32>,
    pub prepare_positions: Vec<Vec2>,
    pub exit_frame_ticks: Vec<u32>,
    pub force_from_boulder: Fix,
    pub smoke_updatable_count: u32,
    pub player_radius_hack: Fix,
    pub move_after_throw_frames: u32,
    pub jump_after_throw_frames: u32,
    pub rotation_velocity: Fix,
    pub ground_angular_multiplier: Fix,
    pub max_attachable_distance: Fix,
    pub shrink_after_frames: u32,
    pub projectile_position: Vec2,
    pub quick_projectile_position: Vec2,
    pub throw_speed: Fix,
    pub min_throw_vector: Vec2,
    pub extra_strength_scale: Fix,
    pub quick_throw_strength: Fix,
    pub quick_charge_scaling: Fix,
    pub platform_scale_reduction: Fix,
    pub ground_to_hands: Fix,
    pub quick_position_swap_frame: u32,
    pub boulder_layer: u32,
    pub boulder_updatable_count: u32,
    pub boulder_collider_hierarchy: u32,
    pub boulder_extents: Vec2,
    pub boulder_radius: Fix,
    pub boulder_min_scale: Fix,
    pub boulder_max_scale: Fix,
    pub boulder_body: crate::physics::Body,
    pub boulder_dropped_bounciness: Fix,
    pub boulder_down_effect_updatable_count: u32,
}

pub struct DrillData {
    pub hurtbox_extents: Vec2,
    pub collider_offset: Vec2,
    pub collider_hierarchy: u32,
    pub enter_ticks: u32,
    pub exit_ticks: u32,
    pub drill_offset: Vec2,
    pub drill_hitbox: Vec2,
    pub gravity: Fix,
    pub drill_strength: Fix,
    pub rotation_speed_air: Fix,
    pub leave_ground_speed_multiplier: Fix,
    pub enter_ground_speed_multiplier: Fix,
    pub speed_to_rotation: Curve,
    pub on_enter_offset: Vec2,
    pub min_underground_speed: Fix,
    pub keep_strong_gravity_seconds: Fix,
    pub strong_gravity: Fix,
    pub max_early_fall_speed: Fix,
    pub hitboxes: Vec<(Vec2, Vec2)>,
    pub hitbox_layer: u32,
    pub hitbox_updatable_count: u32,
    pub hitbox_collider_hierarchy: u32,
}

pub struct RockData {
    pub radius: Fix,
    pub collider_hierarchy: u32,
    pub body: crate::physics::Body,
    pub max_duration: Fix,
    pub enter_scale: Curve,
    pub is_cancellable: bool,
    pub exit_time: Fix,
    pub exit_scale: Curve,
    pub player_collision_bounce: Fix,
    pub shockwave_force: Fix,
    pub min_rope_length: Fix,
    pub rope_pull_strength: Fix,
    pub squared_max_rope_separation: Fix,
    pub rope_player_mass: Fix,
}

pub struct PushData {
    pub hurtbox_extents: Vec2,
    pub collider_hierarchy: u32,
    pub collider_offset: Vec2,
    pub radius: Fix,
    pub grab_ground_ticks: u32,
    pub platform_force: Fix,
    pub scene_bound_limit_offset: Fix,
    pub boulder_force_multiplier: Fix,
    pub home_position_lerp_speed: Fix,
}
pub struct FreezeData {
    pub hurtbox_extents: Vec2,
    pub collider_hierarchy: u32,
    pub collider_offset: Vec2,
    pub radius: Fix,
    pub constants: crate::world::MovementConstants,
    pub air_control_blend: Fix,
    pub firepoint_offset: Vec2,
    pub knockback: Fix,
    pub max_slow_mo_speed: Fix,
    pub transform_ticks: u32,
    pub fire_ticks: u32,
    pub collision_mask: u32,
    pub collision_mask_no_water: u32,
    pub freeze_seconds: Fix,
    pub frozen_updatable_count: u32,
}
pub struct BlinkData {
    pub hurtbox_extents: Vec2,
    pub collider_hierarchy: u32,
    pub collider_offset: Vec2,
    pub radius: Fix,
    pub constants: crate::world::MovementConstants,
    pub air_control_blend: Fix,
    pub firepoint_offset: Vec2,
    pub knockback: Fix,
    pub max_slow_mo_speed: Fix,
    pub transform_ticks: u32,
    pub fire_ticks: u32,
    pub collision_mask: u32,
    pub player_duration: Fix,
    pub wall_duration: Fix,
    pub wall_delay: Fix,
    pub wall_shake: Fix,
    pub on_player_hit_cooldown: Fix,
    pub tunnel_release_duration: Fix,
    pub tunnel_updatable_count: u32,
}
pub struct RollData {
    pub physics_radius: Fix,
    pub radius: Fix,
    pub collider_hierarchy: u32,
    pub constants: crate::world::MovementConstants,
    pub max_speed: Fix,
    pub min_speed: Fix,
    pub speed_curve: Curve,
    pub charge_time: Fix,
    pub dash_time: Fix,
    pub move_collision_density: Fix,
    pub collision_mask: u32,
    pub platform_turn_speed: Fix,
    pub early_release_jump_strength: Fix,
    pub short_jump_strength: Fix,
    pub trail_hierarchy_count: u32,
}
pub struct MeteorData {
    pub hurtbox_extents: Vec2,
    pub collider_hierarchy: u32,
    pub collider_offset: Vec2,
    pub physics_radius: Fix,
    pub constants: crate::world::MovementConstants,
    pub time_before_drop: Fix,
    pub jump_speed: Fix,
    pub hold_speed: Fix,
    pub max_hold_time: Fix,
    pub min_hold_time: Fix,
    pub speed_curve: Curve,
    pub vibration_strength: Fix,
    pub fixed_updates_in_dash: u32,
    pub ground_force: Fix,
    pub player_push_force: Fix,
    pub shake_strength: Fix,
    pub shake_duration: Fix,
    pub spin_curve: Curve,
    pub raycast_mask: u32,
    pub ground_enter_ticks: u32,
    pub air_ticks: u32,
    pub air_start_hold_ticks: u32,
    pub shockwave_updatable_count: u32,
    pub shockwave_offset: Vec2,
    pub shockwave: ExplosionShockwaveData,
    pub shockwave_curve: Curve,
    pub smear_layer: u32,
    pub smear_updatable_count: u32,
    pub smear_collider_hierarchy: u32,
    pub smear_extents: Vec2,
    pub smear_offset: Vec2,
    pub smear_lifespan_ticks: u32,
}
pub struct ScaleRaysData {
    pub growth: ScaleRayData,
    pub shrink: ScaleRayData,
}
pub struct ScaleRayData {
    pub hurtbox_extents: Vec2,
    pub collider_hierarchy: u32,
    pub collider_offset: Vec2,
    pub radius: Fix,
    pub constants: crate::world::MovementConstants,
    pub air_control_blend: Fix,
    pub firepoint_offset: Vec2,
    pub knockback: Fix,
    pub max_slow_mo_speed: Fix,
    pub transform_ticks: u32,
    pub fire_ticks: u32,
    pub collision_mask: u32,
    pub collision_mask_no_water: u32,
    pub scale_curve: Curve,
    pub player_multiplier: Fix,
    pub multiplier: Fix,
    pub small_non_players_multiplier: Fix,
    pub inverse_mass_before_big: Fix,
    pub changer_updatable_count: u32,
}
pub struct DuplicatorData {
    pub hurtbox_extents: Vec2,
    pub collider_hierarchy: u32,
    pub collider_offset: Vec2,
    pub radius: Fix,
    pub constants: crate::world::MovementConstants,
    pub air_control_blend: Fix,
    pub firepoint_offset: Vec2,
    pub knockback: Fix,
    pub max_slow_mo_speed: Fix,
    pub transform_ticks: u32,
    pub fire_ticks: u32,
    pub triple_raycast_radius: Fix,
    pub collision_mask: u32,
    pub collision_mask_no_water: u32,
    pub platform_shake_duration: Fix,
    pub platform_shake: Fix,
    pub platform_overlap_force: Fix,
    pub platform_effect_updatable_count: u32,
}
pub struct MagnetData {
    pub hurtbox_extents: Vec2,
    pub collider_hierarchy: u32,
    pub collider_offset: Vec2,
    pub radius: Fix,
    pub constants: crate::world::MovementConstants,
    pub air_control_blend: Fix,
    pub firepoint_offset: Vec2,
    pub knockback: Fix,
    pub player_rock_extra_offset: Vec2,
    pub boulder_extra_offset: Vec2,
    pub held_lerp: Fix,
    pub fire_force_mines: Fix,
    pub fire_force: Fix,
    pub fire_ragdoll_force: Fix,
    pub fire_black_hole_force: Fix,
    pub fire_rock_force: Fix,
    pub fire_missile_force: Fix,
    pub fire_boulder_force: Fix,
    pub roped_force_multiplier: Fix,
    pub max_slow_mo_speed: Fix,
    pub only_pull: u32,
    pub only_pickup: u32,
    pub pickup_radius: Fix,
    pub max_lock_distance_squared: Fix,
    pub close_pull_multiplier: Fix,
    pub fast_pull_radius_squared: Fix,
    pub pull_width: Fix,
    pub pull_strength: Fix,
    pub player_pull_strength: Fix,
    pub projectile_pull_strength: Fix,
    pub wall_pull_strength: Fix,
    pub black_hole_pull_strength: Fix,
    pub macho_pull_strength: Fix,
    pub player_mass: Fix,
    pub locked_angular_velocity: Fix,
    pub small_ragdoll_duration: Fix,
    pub fire_ragdoll_duration: Fix,
    pub wall_mask: u32,
    pub raycast_offset: Vec2,
    pub max_pull_duration: Fix,
    pub parry_push_strength: Fix,
    pub pull_curve: Curve,
    pub enter_ticks: u32,
    pub prepull_ticks: u32,
    pub exit_ticks: u32,
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
pub struct MjolnirData {
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
    pub outbound_drag: Fix,
    pub return_speed_threshold: Fix,
    pub max_flight_seconds: Fix,
    pub recall_accel: Fix,
    pub recall_max_speed: Fix,
    pub clang_grace_seconds: Fix,
    pub platform_push_force: Fix,
}
pub struct PokeBallData {
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
    pub capture_seconds: Fix,
    pub fizzle_seconds: Fix,
    pub pop_up_speed: Fix,
}
pub struct SmokeData {
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
    pub cloud_offsets: Vec<Vec2>,
    pub spread: Fix,
    pub noise: Fix,
    pub burst_updatable_count: u32,
    pub cloud_layer: u32,
    pub cloud_updatable_count: u32,
    pub cloud_collider_hierarchy: u32,
    pub cloud_base_radius: Fix,
    pub cloud_body: crate::physics::Body,
    pub cloud_clone_count: u32,
    pub cloud_duplicate_pattern: u32,
    pub cloud_duplicate_spacing: Fix,
    pub cloud_duplicate_angle: Fix,
    pub cloud_radii: Vec<Fix>,
    pub cloud_enter_frame_ticks: Vec<u32>,
    pub min_ignition_delay: Fix,
    pub max_ignition_delay: Fix,
    pub explosion_layer: u32,
    pub explosion_updatable_count: u32,
    pub explosion_collider_hierarchy: u32,
    pub explosion_extents: Vec2,
    pub explosion_animation_ticks: u32,
}
pub struct TimeStopData {
    pub cast_time: Fix,
    pub center_offset: Vec2,
    pub exit_offset: Vec2,
    pub collider_offset: Vec2,
    pub hurtbox_radius: Fix,
    pub collider_hierarchy: u32,
    pub idle_to_fall_ticks: u32,
    pub cast_spell_ticks: u32,
    pub ease_out_ticks: u32,
    pub effect_updatable_count: u32,
    pub duration_f32: f32,
    pub exit_seconds_f32: f32,
}
pub struct SpikeData {
    pub hurtbox_extents: Vec2,
    pub collider_offset: Vec2,
    pub collider_hierarchy: u32,
    pub cast_ticks: u32,
    pub recover_ticks: u32,
    pub surface_offset: Fix,
    pub cast_force: Fix,
    pub attack_layer: u32,
    pub attack_updatable_count: u32,
    pub attack_collider_hierarchy: u32,
    pub attack_extents: Vec2,
    pub attack_collider_offset: Vec2,
    pub attack_body: crate::physics::Body,
    pub down_thrust: Curve,
    pub knock_player: Fix,
    pub knock_wall: Fix,
    pub self_push_multiplier: Fix,
    pub push_time: Fix,
}
pub struct EngineData {
    pub hurtbox_extents: Vec2,
    pub collider_offset: Vec2,
    pub collider_hierarchy: u32,
    pub cast_ticks: u32,
    pub recover_ticks: u32,
    pub plant_offset: Vec2,
    pub cast_force: Fix,
    pub buffer_size: usize,
    pub rocket_layer: u32,
    pub rocket_updatable_count: u32,
    pub rocket_collider_hierarchy: u32,
    pub rocket_extents: Vec2,
    pub rocket_collider_offset: Vec2,
    pub rocket_body: crate::physics::Body,
    pub force: Curve,
    pub start_ticks: u32,
    pub end_ticks: u32,
    pub fire_offset: Vec2,
    pub fire_radius: Fix,
    pub smoke_mask: u32,
}
pub struct TeslaData {
    pub hurtbox_extents: Vec2,
    pub collider_offset: Vec2,
    pub collider_hierarchy: u32,
    pub cast_ticks: u32,
    pub recover_ticks: u32,
    pub plant_offset: Vec2,
    pub cast_force: Fix,
    pub buffer_size: usize,
    pub node_layer: u32,
    pub node_updatable_count: u32,
    pub node_collider_hierarchy: u32,
    pub node_radius: Fix,
    pub node_collider_offset: Vec2,
    pub node_body: crate::physics::Body,
    pub line_length: Fix,
    pub hitbox_offset_y: Fix,
    pub lightning_layer: u32,
    pub lightning_updatable_count: u32,
    pub lightning_collider_hierarchy: u32,
    pub lightning_extents: Vec2,
    pub lightning_minimum_extent: Fix,
    pub flicker_period: Fix,
    pub enter_ticks: u32,
    pub flicker_ticks: u32,
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
    pub black_hole: BlackHoleData,
    pub dash: DashData,
    pub bow: BowData,
    pub arrow: ArrowData,
    pub pickup: PickupData,
    pub platform: PlatformAbilityData,
    pub gust: GustData,
    pub revival: RevivalData,
    pub teleport: TeleportData,
    pub portal_gun: PortalGunData,
    pub beam: BeamData,
    pub ropes: RopeData,
    pub grenade: GrenadeData,
    pub missile: MissileData,
    pub mine: MineData,
    pub macho_throw: MachoThrowData,
    pub drill: DrillData,
    pub rock: RockData,
    pub roll: RollData,
    pub meteor: MeteorData,
    pub push: PushData,
    pub freeze: FreezeData,
    pub blink: BlinkData,
    pub scale_rays: ScaleRaysData,
    pub duplicator: DuplicatorData,
    pub magnet: MagnetData,
    pub smoke: SmokeData,
    pub time_stop: TimeStopData,
    pub spike: SpikeData,
    pub engine: EngineData,
    pub tesla: TeslaData,
    pub tack: TackData,
    pub mjolnir: MjolnirData,
    pub poke_ball: PokeBallData,
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

fn scale_ray(raw: &RawScaleRay) -> ScaleRayData {
    ScaleRayData {
        hurtbox_extents: {
            let extents = raw.hurtbox_extents.get();
            Vec2::new(extents.x.abs(), extents.y.abs())
        },
        collider_hierarchy: raw.collider_hierarchy,
        collider_offset: raw.collider_offset.get(),
        radius: Fix::from_raw(raw.radius),
        constants: constants(&raw.physics),
        air_control_blend: Fix::from_raw(raw.air_control_blend),
        firepoint_offset: raw.firepoint_offset.get(),
        knockback: Fix::from_raw(raw.knockback),
        max_slow_mo_speed: Fix::from_raw(raw.max_slow_mo_speed),
        transform_ticks: raw.transform_ticks,
        fire_ticks: raw.fire_ticks,
        collision_mask: raw.collision_mask,
        collision_mask_no_water: raw.collision_mask_no_water,
        scale_curve: raw.scale_curve.clone(),
        player_multiplier: Fix::from_raw(raw.player_multiplier),
        multiplier: Fix::from_raw(raw.multiplier),
        small_non_players_multiplier: Fix::from_raw(raw.small_non_players_multiplier),
        inverse_mass_before_big: Fix::from_raw(raw.inverse_mass_before_big),
        changer_updatable_count: raw.changer_updatable_count,
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
        velocity: raw.start_velocity.get(),
        angular_velocity: Fix::from_raw(raw.start_angular_velocity),
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
            let authored_scale = Fix::from_raw(p.scale);
            let grow_scale = Fix::from_raw(p.grow_scale);
            let base_scale = authored_scale * grow_scale;
            // StickyRoundedRectangle.UpdateBounds (StickyRoundedRectangle.cs:104,
            // rerun after the level animates in, GameSessionHandler.cs:769)
            // reads its perimeter extents back from the collider as
            // `One / inverseExtents` (DPhysicsRoundedRect.CalcExtents:355-358)
            // where the collider stored `One / startExtents`; the round trip
            // moves non-power-of-two extents by a few raw units in Q32.32.
            let authored_extents = p.extents.get() * authored_scale;
            let mut collider_inverse_extents =
                Vec2::new(Fix::ONE / authored_extents.x, Fix::ONE / authored_extents.y);
            if grow_scale != Fix::ONE && grow_scale != Fix::ZERO {
                collider_inverse_extents = collider_inverse_extents / grow_scale;
            }
            let collider_extents = Vec2::new(
                Fix::ONE / collider_inverse_extents.x,
                Fix::ONE / collider_inverse_extents.y,
            );
            // Box.SetRotation (Box.cs:33) builds the collider axes as
            // `Vec2.right / inverseExtents.x`, i.e. the reciprocal of what the
            // collider actually stores - not the authored extent. In Q32.32 that
            // round trip moves non-power-of-two extents by a few raw units, and
            // the axes DetPhysics publishes are the round-tripped ones.
            let collider_axis_extents = collider_extents;
            let mut platform_body = body(&p.body, rotation);
            // GrowOnStart.Init writes the FixTransform scale before the first
            // BoplBody updater pass. Non-resizable bodies then multiply both
            // inverse mass and inverse inertia by prevScale / scaleUp
            // (BoplBody.cs:332-343). The authored platform scale has already
            // been consumed by scene construction; only GrowOnStart is a
            // runtime scale mutation and therefore changes these body terms.
            if grow_scale != Fix::ONE && grow_scale != Fix::ZERO {
                let ratio = Fix::ONE / grow_scale;
                platform_body.inverse_mass *= ratio;
                platform_body.inverse_inertia *= ratio;
            }
            PlatformDef {
                intro_order: p.intro_order,
                surface: Surface {
                    center: p.position.get(),
                    right,
                    up,
                    extents: collider_extents,
                    radius: Fix::from_raw(p.radius) * base_scale,
                },
                collider_inverse_extents,
                collider_axis_extents,
                hierarchy: p.hierarchy,
                updatable_count: p.updatable_count,
                duplicate_collider_hierarchy: p.duplicate_collider_hierarchy,
                min_scale: Fix::from_raw(p.min_scale),
                base_scale,
                ice: p.platform_type == 2,
                speed: Fix::from_raw(p.animate_velocity.speed),
                mu: Fix::from_raw(p.animate_velocity.mu),
                rotation_speed: Fix::from_raw(p.animate_velocity.rotation_speed),
                rotation_mu: Fix::from_raw(p.animate_velocity.rotation_mu),
                interpolate_rotation_home: p.animate_velocity.interpolate_rotation_home,
                body: platform_body,
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
                vector_field: p.vector_field.as_ref().map(|v| VectorFieldDef {
                    delay: Fix::from_raw(v.delay),
                    orbit_speed: Fix::from_raw(v.orbit_speed),
                    expand_speed: Fix::from_raw(v.expand_speed),
                    center: v.center.get(),
                    normal_speed_friction: Fix::from_raw(v.normal_speed_friction),
                    dead_zone_dist_squared: Fix::from_raw(v.dead_zone_dist_squared),
                    orbit_acceleration_multiplier: Fix::from_raw(v.orbit_acceleration_multiplier),
                    target_radius: Fix::from_raw(v.target_radius),
                    ovalness: Fix::from_raw(v.ovalness),
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
        water: raw
            .water
            .as_ref()
            .map(|water| (water.position.get(), water.extents.get(), water.hierarchy)),
        level_type: raw.session.level_type,
        team_spawns,
        teammate_spacing: Fix::from_raw(raw.session.teammate_spacing),
        respawn_points,
        sudden_death_time: Fix::from_raw(raw.session.time_before_sudden_death),
        ability_spawners: raw
            .ability_spawners
            .iter()
            .map(|spawner| AbilitySpawnerDef {
                position: spawner.position.get(),
                hierarchy: spawner.hierarchy,
            })
            .collect(),
        level_animation_radius: Fix::from_raw(raw.session.level_animation_radius),
        max_random_delay: Fix::from_raw(raw.session.max_random_delay),
        level_animation_time: Fix::from_raw(raw.session.level_animation_time),
        start_animation_progress: Fix::from_raw(raw.session.start_animation_progress),
        animate_in: raw.session.animate_in.clone(),
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
                boundary_effect_updatable_count: raw.prefabs.player.boundary_effect_updatable_count,
            },
            black_hole: {
                let black = &raw.prefabs.black_hole;
                let hole = &black.hole;
                BlackHoleData {
                    hurtbox_extents: {
                        let extents = black.hurtbox_extents.get();
                        Vec2::new(extents.x.abs(), extents.y.abs())
                    },
                    collider_hierarchy: black.collider_hierarchy,
                    collider_offset: black.collider_offset.get(),
                    physics_radius: Fix::from_raw(black.physics.radius),
                    constants: constants(&black.physics),
                    enter_ticks: ticks(&black.animations.enter),
                    enter_air_ticks: ticks(&black.animations.enter_air),
                    exit_ticks: ticks(&black.animations.exit),
                    exit_air_ticks: ticks(&black.animations.exit_air),
                    spawn_offset: black.spawn_offset.get(),
                    layer: hole.layer,
                    updatable_count: hole.updatable_count,
                    hole_collider_hierarchy: hole.collider_hierarchy,
                    start_radius: Fix::from_raw(hole.start_radius),
                    starting_mass: Fix::from_raw(hole.starting_mass),
                    gravity: Fix::from_raw(hole.gravity),
                    controlled_platform_multiplier: Fix::from_raw(
                        hole.controlled_platform_multiplier,
                    ),
                    growth: Fix::from_raw(hole.growth),
                    base_growth: Fix::from_raw(hole.base_growth),
                    merge_growth: Fix::from_raw(hole.merge_growth),
                    size_curve: hole.size_curve.clone(),
                    min_size: Fix::from_raw(hole.min_size),
                    drag: Fix::from_raw(hole.drag),
                    player_force: Fix::from_raw(hole.player_force),
                    influence_radius_multiplier: Fix::from_raw(hole.influence_radius_multiplier),
                }
            },
            dash: DashData {
                cooldown: Fix::from_raw(raw.prefabs.dash.cooldown),
                hurtbox_extents: raw.prefabs.dash.hurtbox_extents.get(),
                leave_speed: Fix::from_raw(raw.prefabs.dash.fields.leave_speed),
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
                collider_hierarchy: raw.prefabs.arrow.collider_hierarchy,
                updatable_count: raw.prefabs.arrow.updatable_count,
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
                            start_velocity: RawVec2 { x: 0, y: 0 },
                            start_angular_velocity: 0,
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
                    offset: p.offset.get(),
                    collider_hierarchy: p.collider_hierarchy,
                    updatable_count: p.updatable_count,
                    body: body(&p.body, Fix::ZERO),
                    up_force: Fix::from_raw(p.up_force),
                    wobble_speed: Fix::from_raw(p.wobble_speed),
                    wobble_amount: Fix::from_raw(p.wobble_amount),
                    launch_force: Fix::from_raw(p.launch_force),
                    disabled_seconds: Fix::from_raw(p.disabled_seconds),
                    swap_period: Fix::from_raw(p.swap_period),
                    vibration_strength: Fix::from_raw(p.vibration_strength),
                    vibration_duration: Fix::from_raw(p.vibration_duration),
                    vibrate_timing_offset: Fix::from_raw(p.vibrate_timing_offset),
                    gravity: Fix::from_raw(p.gravity),
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
                shockwave_updatable_count: raw.prefabs.gust.shockwave_updatable_count,
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
                    teleportable_mask: teleport.teleportable_mask,
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
            portal_gun: {
                let portal = &raw.prefabs.portal_gun;
                PortalGunData {
                    shot_speed: Fix::from_raw(portal.shot_speed),
                    spawn_offset: Fix::from_raw(portal.spawn_offset),
                    shot_layer: portal.shot_layer,
                    shot_updatable_count: portal.shot_updatable_count,
                    shot_collider_hierarchy: portal.shot_collider_hierarchy,
                    shot_radius: Fix::from_raw(portal.shot_radius),
                    shot_body: body(&portal.shot_body, Fix::ZERO),
                    max_flight_seconds: Fix::from_raw(portal.max_flight_seconds),
                    portal_updatable_count: portal.portal_updatable_count,
                    half_width: Fix::from_raw(portal.half_width),
                    thickness: Fix::from_raw(portal.thickness),
                    exit_offset: Fix::from_raw(portal.exit_offset),
                    lifetime_seconds: Fix::from_raw(portal.lifetime_seconds),
                }
            },
            beam: {
                let beam = &raw.prefabs.beam;
                BeamData {
                    hurtbox_extents: beam.hurtbox_extents.get(),
                    collider_offset: beam.collider_offset.get(),
                    collider_hierarchy: beam.collider_hierarchy,
                    constants: constants(&beam.physics),
                    radius: Fix::from_raw(beam.physics.radius),
                    enter_ground_ticks: beam.enter_ground_ticks,
                    enter_air_ticks: beam.enter_air_ticks,
                    air_orb_frame8_ticks: beam.air_orb_frame8_ticks,
                    air_orb_frame9_ticks: beam.air_orb_frame9_ticks,
                    air_orb_frame10_ticks: beam.air_orb_frame10_ticks,
                    exit_ground_ticks: beam.exit_ground_ticks,
                    exit_air_ticks: beam.exit_air_ticks,
                    up_alignment: Fix::from_raw(beam.up_alignment),
                    min_grounded_angle: Fix::from_raw(beam.min_grounded_angle),
                    exit_scale_curve: beam.exit_scale_curve.clone(),
                    beam_offset: Fix::from_raw(beam.beam_offset),
                    air_extents: beam.air_extents.get(),
                    air_offset: beam.air_offset.get(),
                    angular_aim_speed: Fix::from_raw(beam.angular_aim_speed),
                    angular_aim_curve: beam.angular_aim_curve.clone(),
                    exit_up_speed: Fix::from_raw(beam.exit_up_speed),
                    max_time: Fix::from_raw(beam.max_time),
                    max_time_air: Fix::from_raw(beam.max_time_air),
                    exit_scale_time: Fix::from_raw(beam.exit_scale_time),
                    shake_duration: Fix::from_raw(beam.shake_duration),
                    shake_strength: Fix::from_raw(beam.shake_strength),
                    orb_layer: beam.orb_layer,
                    orb_updatable_count: beam.orb_updatable_count,
                    orb_collider_hierarchy: beam.orb_collider_hierarchy,
                    orb_radius: Fix::from_raw(beam.orb_radius),
                    orb_exit_ticks: beam.orb_exit_ticks,
                    maximum_distance: Fix::from_raw(beam.maximum_distance),
                    width: Fix::from_raw(beam.width),
                    width_curve: beam.width_curve.clone(),
                    parent_offset: Fix::from_raw(beam.parent_offset),
                    platform_force: Fix::from_raw(beam.platform_force),
                    black_hole_force: Fix::from_raw(beam.black_hole_force),
                    rigid_body_force: Fix::from_raw(beam.rigid_body_force),
                    cross_offset: Fix::from_raw(beam.cross_offset),
                    hit_mask: beam.hit_mask,
                    object_layer: beam.object_layer,
                    object_updatable_count: beam.object_updatable_count,
                    object_collider_hierarchy: beam.object_collider_hierarchy,
                    base_width: Fix::from_raw(beam.base_width),
                    extra_length: Fix::from_raw(beam.extra_length),
                    minimum_extent: Fix::from_raw(beam.minimum_extent),
                }
            },
            ropes: {
                let ropes = &raw.prefabs.ropes;
                RopeData {
                    pool_capacity: ropes.pool_capacity,
                    max_segments: ropes.max_segments,
                    max_contacts: ropes.max_contacts,
                    segments: ropes.segments,
                    max_ropes_per_player: ropes.max_ropes_per_player,
                    gravity: Fix::from_raw(ropes.gravity),
                    iterations: ropes.iterations,
                    stretchiness: Fix::from_raw(ropes.stretchiness),
                    max_stretch: Fix::from_raw(ropes.max_stretch),
                    grounded_strength: Fix::from_raw(ropes.grounded_strength),
                    segment_radius: Fix::from_raw(ropes.segment_radius),
                    player_gravity: Fix::from_raw(ropes.player_gravity),
                    platform_mass: Fix::from_raw(ropes.platform_mass),
                    platform_heaviness: Fix::from_raw(ropes.platform_heaviness),
                    platform_pull: Fix::from_raw(ropes.platform_pull),
                    boulder_pull: Fix::from_raw(ropes.boulder_pull),
                    platform_break_stretch: Fix::from_raw(ropes.platform_break_stretch),
                    non_platform_mass: Fix::from_raw(ropes.non_platform_mass),
                    web_max_length: Fix::from_raw(ropes.web_max_length),
                    web_shorten: Fix::from_raw(ropes.web_shorten),
                    web_cooldown: Fix::from_raw(ropes.web_cooldown),
                    web_miss_cooldown: Fix::from_raw(ropes.web_miss_cooldown),
                    web_release_boost: Fix::from_raw(ropes.web_release_boost),
                    web_grounded_pop: Fix::from_raw(ropes.web_grounded_pop),
                    grapple_spawn_offset: Fix::from_raw(ropes.grapple_spawn_offset),
                    grapple_speed: Fix::from_raw(ropes.grapple_speed),
                    grapple_reel_frames: ropes.grapple_reel_frames,
                    grapple_start_separation: Fix::from_raw(ropes.grapple_start_separation),
                    grapple_reeled_separation: Fix::from_raw(ropes.grapple_reeled_separation),
                    grapple_reel_speed: Fix::from_raw(ropes.grapple_reel_speed),
                    grapple_delete_separation: Fix::from_raw(ropes.grapple_delete_separation),
                    grapple_cooldown: Fix::from_raw(ropes.grapple_cooldown),
                    grapple_player_cast_radius: Fix::from_raw(ropes.grapple_player_cast_radius),
                    hook_layer: ropes.hook_layer,
                    hook_radius: Fix::from_raw(ropes.hook_radius),
                    hook_collider_hierarchy: ropes.hook_collider_hierarchy,
                    hook_updatable_count: ropes.hook_updatable_count,
                    hook_segments_affected: ropes.hook_segments_affected,
                    hook_speed_falloff: Fix::from_raw(ropes.hook_speed_falloff),
                    hook_fall_time: Fix::from_raw(ropes.hook_fall_time),
                    hook_pull: Fix::from_raw(ropes.hook_pull),
                    rope_reel_out_interval: Fix::from_raw(ropes.rope_reel_out_interval),
                    rope_segments_affected: ropes.rope_segments_affected,
                    rope_segment_wave_length: ropes.rope_segment_wave_length,
                    rope_reel_out_segment_speed: Fix::from_raw(ropes.rope_reel_out_segment_speed),
                    rope_sprinkler_speed: Fix::from_raw(ropes.rope_sprinkler_speed),
                    rope_max_travel_time: Fix::from_raw(ropes.rope_max_travel_time),
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
            missile: {
                let missile = &raw.prefabs.missile;
                MissileData {
                    hurtbox_extents: missile.hurtbox_extents.get(),
                    physics_radius: Fix::from_raw(missile.physics.radius),
                    projectile_offset: missile.projectile_offset.get(),
                    projectile_start_speed: Fix::from_raw(missile.projectile_start_speed),
                    rotation_speed: Fix::from_raw(missile.rotation_speed),
                    constants: constants(&missile.physics),
                    enter_ticks: ticks(&missile.animations.enter),
                    enter_to_idle_ticks: ticks(&missile.animations.enter_to_idle),
                    idle_ticks: ticks(&missile.animations.idle),
                    spark_layer: missile.spark.layer,
                    spark_updatable_count: missile.spark.updatable_count,
                    spark_collider_hierarchy: missile.spark.collider_hierarchy,
                    spark_extents: missile.spark.extents.get(),
                    spark_body: body(&missile.spark.body, Fix::ZERO),
                    minimum_forward_speed: Fix::from_raw(missile.spark.minimum_forward_speed),
                    accel: Fix::from_raw(missile.spark.accel),
                    instability_accel: Fix::from_raw(missile.spark.instability_accel),
                    clone_count: missile.spark.clone_count,
                    duplicate_pattern: missile.spark.duplicate_pattern,
                    duplicate_spacing: Fix::from_raw(missile.spark.duplicate_spacing),
                    duplicate_angle: Fix::from_raw(missile.spark.duplicate_angle),
                    explosion_layer: missile.explosion.layer,
                    explosion_updatable_count: missile.explosion.updatable_count,
                    explosion_collider_hierarchy: missile.explosion.collider_hierarchy,
                    explosion_extents: missile.explosion.extents.get(),
                    explosion_animation_ticks: missile.explosion.animation_ticks,
                    explosion_shockwave: ExplosionShockwaveData {
                        platform_force: Fix::from_raw(missile.explosion.shockwave.platform_force),
                        default_force: Fix::from_raw(missile.explosion.shockwave.default_force),
                        grounded_player_multiplier: Fix::from_raw(
                            missile.explosion.shockwave.grounded_player_multiplier,
                        ),
                        radius: Fix::from_raw(missile.explosion.shockwave.radius),
                        collision_mask: missile.explosion.shockwave.collision_mask,
                    },
                }
            },
            mine: {
                let mine = &raw.prefabs.mine;
                let throw = &mine.throw_data;
                MineData {
                    hurtbox_extents: mine.hurtbox_extents.get(),
                    collider_offset: mine.collider_offset.get(),
                    throw_position: throw.throw_position.get(),
                    throw_strength: Fix::from_raw(throw.throw_strength),
                    aerial_throw_strength: Fix::from_raw(throw.aerial_throw_strength),
                    knockback: Fix::from_raw(throw.knockback),
                    min_throw_strength: Fix::from_raw(throw.min_throw_strength),
                    only_throw_up: throw.only_throw_up,
                    constants: constants(&throw.physics),
                    ground_enter_ticks: ticks(&throw.animations.ground_enter),
                    ground_loop_ticks: ticks(&throw.animations.ground_loop),
                    ground_exit_ticks: ticks(&throw.animations.ground_exit),
                    air_enter_ticks: ticks(&throw.animations.air_enter),
                    air_exit_ticks: ticks(&throw.animations.air_exit),
                    item_layer: mine.item.layer,
                    item_updatable_count: mine.item.updatable_count,
                    item_collider_hierarchy: mine.item.collider_hierarchy,
                    item_radius: Fix::from_raw(mine.item.radius),
                    item_body: body(&mine.item.body, Fix::ZERO),
                    clone_count: mine.item.clone_count,
                    duplicate_pattern: mine.item.duplicate_pattern,
                    duplicate_spacing: Fix::from_raw(mine.item.duplicate_spacing),
                    duplicate_angle: Fix::from_raw(mine.item.duplicate_angle),
                    hover_strength: Fix::from_raw(mine.item.hover_strength),
                    hover_period: Fix::from_raw(mine.item.hover_period),
                    time_before_armed: Fix::from_raw(mine.item.time_before_armed),
                    chase_time: Fix::from_raw(mine.item.chase_time),
                    igniters: mine.item.igniters,
                    scans_for: mine.item.scans_for,
                    chase_strength: Fix::from_raw(mine.item.chase_strength),
                    scan_radius: Fix::from_raw(mine.item.scan_radius),
                    shake_force: Fix::from_raw(mine.item.shake_force),
                    anticipation_duration: Fix::from_raw(mine.item.anticipation_duration),
                    explosion_layer: mine.explosion.layer,
                    explosion_updatable_count: mine.explosion.updatable_count,
                    explosion_collider_hierarchy: mine.explosion.collider_hierarchy,
                    explosion_extents: mine.explosion.extents.get(),
                    explosion_animation_ticks: mine.explosion.animation_ticks,
                    explosion_shockwave: ExplosionShockwaveData {
                        platform_force: Fix::from_raw(mine.explosion.shockwave.platform_force),
                        default_force: Fix::from_raw(mine.explosion.shockwave.default_force),
                        grounded_player_multiplier: Fix::from_raw(
                            mine.explosion.shockwave.grounded_player_multiplier,
                        ),
                        radius: Fix::from_raw(mine.explosion.shockwave.radius),
                        collision_mask: mine.explosion.shockwave.collision_mask,
                    },
                }
            },
            macho_throw: {
                let macho = &raw.prefabs.macho_throw;
                let cumulative = |frames: &[i32]| {
                    let mut elapsed = 0u32;
                    frames
                        .iter()
                        .map(|duration| {
                            elapsed += duration.unsigned_abs();
                            elapsed
                        })
                        .collect()
                };
                MachoThrowData {
                    hurtbox_extents: macho.hurtbox_extents.get(),
                    collider_offset: macho.collider_offset.get(),
                    collider_hierarchy: macho.collider_hierarchy,
                    physics_radius: Fix::from_raw(macho.physics.radius),
                    constants: constants(&macho.physics),
                    enter_ticks: ticks(&macho.animations.enter),
                    idle_ticks: ticks(&macho.animations.idle),
                    idle_to_walk_ticks: ticks(&macho.animations.idle_to_walk),
                    walk_ticks: ticks(&macho.animations.walk),
                    begin_throw_ticks: ticks(&macho.animations.begin_throw),
                    exit_ticks: ticks(&macho.animations.exit),
                    turn_ticks: ticks(&macho.animations.turn),
                    pickup_ticks: ticks(&macho.animations.pickup),
                    grab_ticks: ticks(&macho.animations.grab),
                    enter_frame_ticks: cumulative(&macho.animations.enter),
                    grab_frame_ticks: cumulative(&macho.animations.grab),
                    dummy_frame_ticks: cumulative(&macho.animations.pickup),
                    dummy_positions: macho.dummy_positions.iter().map(RawVec2::get).collect(),
                    prepare_frame_ticks: cumulative(&macho.animations.begin_throw),
                    prepare_positions: macho.prepare_positions.iter().map(RawVec2::get).collect(),
                    exit_frame_ticks: cumulative(&macho.animations.exit),
                    force_from_boulder: Fix::from_raw(macho.force_from_boulder),
                    smoke_updatable_count: macho.smoke_updatable_count,
                    player_radius_hack: Fix::from_raw(macho.player_radius_hack),
                    move_after_throw_frames: macho.move_after_throw_frames,
                    jump_after_throw_frames: macho.jump_after_throw_frames,
                    rotation_velocity: Fix::from_raw(macho.rotation_velocity),
                    ground_angular_multiplier: Fix::from_raw(macho.ground_angular_multiplier),
                    max_attachable_distance: Fix::from_raw(macho.max_attachable_distance),
                    shrink_after_frames: macho.shrink_after_frames,
                    projectile_position: macho.projectile_position.get(),
                    quick_projectile_position: macho.quick_projectile_position.get(),
                    throw_speed: Fix::from_raw(macho.throw_speed),
                    min_throw_vector: macho.min_throw_vector.get(),
                    extra_strength_scale: Fix::from_raw(macho.extra_strength_scale),
                    quick_throw_strength: Fix::from_raw(macho.quick_throw_strength),
                    quick_charge_scaling: Fix::from_raw(macho.quick_charge_scaling),
                    platform_scale_reduction: Fix::from_raw(macho.platform_scale_reduction),
                    ground_to_hands: Fix::from_raw(macho.ground_to_hands),
                    quick_position_swap_frame: macho.quick_position_swap_frame,
                    boulder_layer: macho.boulder.layer,
                    boulder_updatable_count: macho.boulder.updatable_count,
                    boulder_collider_hierarchy: macho.boulder.collider_hierarchy,
                    boulder_extents: macho.boulder.extents.get(),
                    boulder_radius: Fix::from_raw(macho.boulder.radius),
                    boulder_min_scale: Fix::from_raw(macho.boulder.min_scale),
                    boulder_max_scale: Fix::from_raw(macho.boulder.max_scale),
                    boulder_body: body(&macho.boulder.body, Fix::ZERO),
                    boulder_dropped_bounciness: Fix::from_raw(macho.boulder.dropped_bounciness),
                    boulder_down_effect_updatable_count: macho.boulder.down_effect_updatable_count,
                }
            },
            drill: {
                let drill = &raw.prefabs.drill;
                DrillData {
                    hurtbox_extents: drill.hurtbox_extents.get(),
                    collider_offset: drill.collider_offset.get(),
                    collider_hierarchy: drill.collider_hierarchy,
                    enter_ticks: drill.enter_ticks,
                    exit_ticks: drill.exit_ticks,
                    drill_offset: drill.drill_offset.get(),
                    drill_hitbox: drill.drill_hitbox.get(),
                    gravity: Fix::from_raw(drill.gravity),
                    drill_strength: Fix::from_raw(drill.drill_strength),
                    rotation_speed_air: Fix::from_raw(drill.rotation_speed_air),
                    leave_ground_speed_multiplier: Fix::from_raw(
                        drill.leave_ground_speed_multiplier,
                    ),
                    enter_ground_speed_multiplier: Fix::from_raw(
                        drill.enter_ground_speed_multiplier,
                    ),
                    speed_to_rotation: drill.speed_to_rotation.clone(),
                    on_enter_offset: drill.on_enter_offset.get(),
                    min_underground_speed: Fix::from_raw(drill.min_underground_speed),
                    keep_strong_gravity_seconds: Fix::from_raw(drill.keep_strong_gravity_seconds),
                    strong_gravity: Fix::from_raw(drill.strong_gravity),
                    max_early_fall_speed: Fix::from_raw(drill.max_early_fall_speed),
                    hitboxes: drill
                        .hitboxes
                        .iter()
                        .map(|hitbox| (hitbox.offset.get(), hitbox.extents.get()))
                        .collect(),
                    hitbox_layer: drill.hitbox_layer,
                    hitbox_updatable_count: drill.hitbox_updatable_count,
                    hitbox_collider_hierarchy: drill.hitbox_collider_hierarchy,
                }
            },
            rock: RockData {
                radius: Fix::from_raw(raw.prefabs.rock.radius),
                collider_hierarchy: raw.prefabs.rock.collider_hierarchy,
                body: body(&raw.prefabs.rock.body, Fix::ZERO),
                max_duration: Fix::from_raw(raw.prefabs.rock.max_duration),
                enter_scale: raw.prefabs.rock.enter_scale,
                is_cancellable: raw.prefabs.rock.is_cancellable,
                exit_time: Fix::from_raw(raw.prefabs.rock.exit_time),
                exit_scale: raw.prefabs.rock.exit_scale,
                player_collision_bounce: Fix::from_raw(raw.prefabs.rock.player_collision_bounce),
                shockwave_force: Fix::from_raw(raw.prefabs.rock.shockwave_force),
                min_rope_length: Fix::from_raw(raw.prefabs.rock.min_rope_length),
                rope_pull_strength: Fix::from_raw(raw.prefabs.rock.rope_pull_strength),
                squared_max_rope_separation: Fix::from_raw(
                    raw.prefabs.rock.squared_max_rope_separation,
                ),
                rope_player_mass: Fix::from_raw(raw.prefabs.rock.rope_player_mass),
            },
            roll: RollData {
                physics_radius: Fix::from_raw(raw.prefabs.roll.physics.radius),
                radius: Fix::from_raw(raw.prefabs.roll.radius),
                collider_hierarchy: raw.prefabs.roll.collider_hierarchy,
                constants: constants(&raw.prefabs.roll.physics),
                max_speed: Fix::from_raw(raw.prefabs.roll.max_speed),
                min_speed: Fix::from_raw(raw.prefabs.roll.min_speed),
                speed_curve: raw.prefabs.roll.speed_curve,
                charge_time: Fix::from_raw(raw.prefabs.roll.charge_time),
                dash_time: Fix::from_raw(raw.prefabs.roll.dash_time),
                move_collision_density: Fix::from_raw(raw.prefabs.roll.move_collision_density),
                collision_mask: raw.prefabs.roll.collision_mask,
                platform_turn_speed: Fix::from_raw(raw.prefabs.roll.platform_turn_speed),
                early_release_jump_strength: Fix::from_raw(
                    raw.prefabs.roll.early_release_jump_strength,
                ),
                short_jump_strength: Fix::from_raw(raw.prefabs.roll.short_jump_strength),
                trail_hierarchy_count: raw.prefabs.roll.trail_hierarchy_count,
            },
            meteor: {
                let meteor = &raw.prefabs.meteor;
                MeteorData {
                    hurtbox_extents: meteor.hurtbox_extents.get(),
                    collider_hierarchy: meteor.collider_hierarchy,
                    collider_offset: meteor.collider_offset.get(),
                    physics_radius: Fix::from_raw(meteor.physics.radius),
                    constants: constants(&meteor.physics),
                    time_before_drop: Fix::from_raw(meteor.time_before_drop),
                    jump_speed: Fix::from_raw(meteor.jump_speed),
                    hold_speed: Fix::from_raw(meteor.hold_speed),
                    max_hold_time: Fix::from_raw(meteor.max_hold_time),
                    min_hold_time: Fix::from_raw(meteor.min_hold_time),
                    speed_curve: meteor.speed_curve.clone(),
                    vibration_strength: Fix::from_raw(meteor.vibration_strength),
                    fixed_updates_in_dash: meteor.fixed_updates_in_dash,
                    ground_force: Fix::from_raw(meteor.ground_force),
                    player_push_force: Fix::from_raw(meteor.player_push_force),
                    shake_strength: Fix::from_raw(meteor.shake_strength),
                    shake_duration: Fix::from_raw(meteor.shake_duration),
                    spin_curve: meteor.spin_curve.clone(),
                    raycast_mask: meteor.raycast_mask,
                    ground_enter_ticks: meteor.ground_enter_ticks,
                    air_ticks: meteor.air_ticks,
                    air_start_hold_ticks: meteor.air_start_hold_ticks,
                    shockwave_updatable_count: meteor.shockwave.updatable_count,
                    shockwave_offset: meteor.shockwave.offset.get(),
                    shockwave: ExplosionShockwaveData {
                        platform_force: Fix::from_raw(meteor.shockwave.platform_force),
                        default_force: Fix::from_raw(meteor.shockwave.default_force),
                        grounded_player_multiplier: Fix::from_raw(
                            meteor.shockwave.grounded_player_multiplier,
                        ),
                        radius: Fix::from_raw(meteor.shockwave.radius),
                        collision_mask: meteor.shockwave.collision_mask,
                    },
                    shockwave_curve: meteor.shockwave.curve.clone(),
                    smear_layer: meteor.smear.layer,
                    smear_updatable_count: meteor.smear.updatable_count,
                    smear_collider_hierarchy: meteor.smear.collider_hierarchy,
                    smear_extents: meteor.smear.extents.get(),
                    smear_offset: meteor.smear.offset.get(),
                    smear_lifespan_ticks: meteor.smear.lifespan_ticks,
                }
            },
            push: PushData {
                hurtbox_extents: raw.prefabs.push.hurtbox_extents.get(),
                collider_hierarchy: raw.prefabs.push.collider_hierarchy,
                collider_offset: raw.prefabs.push.collider_offset.get(),
                radius: Fix::from_raw(raw.prefabs.push.radius),
                grab_ground_ticks: raw.prefabs.push.grab_ground_ticks,
                platform_force: Fix::from_raw(raw.prefabs.push.platform_force),
                scene_bound_limit_offset: Fix::from_raw(raw.prefabs.push.scene_bound_limit_offset),
                boulder_force_multiplier: Fix::from_raw(raw.prefabs.push.boulder_force_multiplier),
                home_position_lerp_speed: Fix::from_raw(raw.prefabs.push.home_position_lerp_speed),
            },
            freeze: FreezeData {
                hurtbox_extents: {
                    let extents = raw.prefabs.freeze.hurtbox_extents.get();
                    Vec2::new(extents.x.abs(), extents.y.abs())
                },
                collider_hierarchy: raw.prefabs.freeze.collider_hierarchy,
                collider_offset: raw.prefabs.freeze.collider_offset.get(),
                radius: Fix::from_raw(raw.prefabs.freeze.radius),
                constants: constants(&raw.prefabs.freeze.physics),
                air_control_blend: Fix::from_raw(raw.prefabs.freeze.air_control_blend),
                firepoint_offset: raw.prefabs.freeze.firepoint_offset.get(),
                knockback: Fix::from_raw(raw.prefabs.freeze.knockback),
                max_slow_mo_speed: Fix::from_raw(raw.prefabs.freeze.max_slow_mo_speed),
                transform_ticks: raw.prefabs.freeze.transform_ticks,
                fire_ticks: raw.prefabs.freeze.fire_ticks,
                collision_mask: raw.prefabs.freeze.collision_mask,
                collision_mask_no_water: raw.prefabs.freeze.collision_mask_no_water,
                freeze_seconds: Fix::from_raw(raw.prefabs.freeze.freeze_seconds),
                frozen_updatable_count: raw.prefabs.freeze.frozen_updatable_count,
            },
            blink: BlinkData {
                hurtbox_extents: {
                    let extents = raw.prefabs.blink.hurtbox_extents.get();
                    Vec2::new(extents.x.abs(), extents.y.abs())
                },
                collider_hierarchy: raw.prefabs.blink.collider_hierarchy,
                collider_offset: raw.prefabs.blink.collider_offset.get(),
                radius: Fix::from_raw(raw.prefabs.blink.radius),
                constants: constants(&raw.prefabs.blink.physics),
                air_control_blend: Fix::from_raw(raw.prefabs.blink.air_control_blend),
                firepoint_offset: raw.prefabs.blink.firepoint_offset.get(),
                knockback: Fix::from_raw(raw.prefabs.blink.knockback),
                max_slow_mo_speed: Fix::from_raw(raw.prefabs.blink.max_slow_mo_speed),
                transform_ticks: raw.prefabs.blink.transform_ticks,
                fire_ticks: raw.prefabs.blink.fire_ticks,
                collision_mask: raw.prefabs.blink.collision_mask,
                player_duration: Fix::from_raw(raw.prefabs.blink.min_player_duration)
                    .max(Fix::from_f64(0.3) * Fix::from_raw(raw.prefabs.blink.wall_duration)),
                wall_duration: Fix::from_raw(raw.prefabs.blink.wall_duration),
                wall_delay: Fix::from_raw(raw.prefabs.blink.wall_delay),
                wall_shake: Fix::from_raw(raw.prefabs.blink.wall_shake),
                on_player_hit_cooldown: Fix::from_raw(raw.prefabs.blink.on_player_hit_cooldown),
                tunnel_release_duration: Fix::from_raw(raw.prefabs.blink.tunnel_release_duration),
                tunnel_updatable_count: raw.prefabs.blink.tunnel_updatable_count,
            },
            scale_rays: ScaleRaysData {
                growth: scale_ray(&raw.prefabs.scale_rays.growth),
                shrink: scale_ray(&raw.prefabs.scale_rays.shrink),
            },
            duplicator: DuplicatorData {
                hurtbox_extents: {
                    let extents = raw.prefabs.duplicator.hurtbox_extents.get();
                    Vec2::new(extents.x.abs(), extents.y.abs())
                },
                collider_hierarchy: raw.prefabs.duplicator.collider_hierarchy,
                collider_offset: raw.prefabs.duplicator.collider_offset.get(),
                radius: Fix::from_raw(raw.prefabs.duplicator.radius),
                constants: constants(&raw.prefabs.duplicator.physics),
                air_control_blend: Fix::from_raw(raw.prefabs.duplicator.air_control_blend),
                firepoint_offset: raw.prefabs.duplicator.firepoint_offset.get(),
                knockback: Fix::from_raw(raw.prefabs.duplicator.knockback),
                max_slow_mo_speed: Fix::from_raw(raw.prefabs.duplicator.max_slow_mo_speed),
                transform_ticks: raw.prefabs.duplicator.transform_ticks,
                fire_ticks: raw.prefabs.duplicator.fire_ticks,
                triple_raycast_radius: Fix::from_raw(raw.prefabs.duplicator.triple_raycast_radius),
                collision_mask: raw.prefabs.duplicator.collision_mask,
                collision_mask_no_water: raw.prefabs.duplicator.collision_mask_no_water,
                platform_shake_duration: Fix::from_raw(
                    raw.prefabs.duplicator.platform_shake_duration,
                ),
                platform_shake: Fix::from_raw(raw.prefabs.duplicator.platform_shake),
                platform_overlap_force: Fix::from_raw(
                    raw.prefabs.duplicator.platform_overlap_force,
                ),
                platform_effect_updatable_count: raw
                    .prefabs
                    .duplicator
                    .platform_effect_updatable_count,
            },
            magnet: {
                let magnet = &raw.prefabs.magnet;
                MagnetData {
                    hurtbox_extents: {
                        let extents = magnet.hurtbox_extents.get();
                        Vec2::new(extents.x.abs(), extents.y.abs())
                    },
                    collider_hierarchy: magnet.collider_hierarchy,
                    collider_offset: magnet.collider_offset.get(),
                    radius: Fix::from_raw(magnet.radius),
                    constants: constants(&magnet.physics),
                    air_control_blend: Fix::from_raw(magnet.air_control_blend),
                    firepoint_offset: magnet.firepoint_offset.get(),
                    knockback: Fix::from_raw(magnet.knockback),
                    player_rock_extra_offset: magnet.player_rock_extra_offset.get(),
                    boulder_extra_offset: magnet.boulder_extra_offset.get(),
                    held_lerp: Fix::from_raw(magnet.held_lerp),
                    fire_force_mines: Fix::from_raw(magnet.fire_force_mines),
                    fire_force: Fix::from_raw(magnet.fire_force),
                    fire_ragdoll_force: Fix::from_raw(magnet.fire_ragdoll_force),
                    fire_black_hole_force: Fix::from_raw(magnet.fire_black_hole_force),
                    fire_rock_force: Fix::from_raw(magnet.fire_rock_force),
                    fire_missile_force: Fix::from_raw(magnet.fire_missile_force),
                    fire_boulder_force: Fix::from_raw(magnet.fire_boulder_force),
                    roped_force_multiplier: Fix::from_raw(magnet.roped_force_multiplier),
                    max_slow_mo_speed: Fix::from_raw(magnet.max_slow_mo_speed),
                    only_pull: magnet.only_pull,
                    only_pickup: magnet.only_pickup,
                    pickup_radius: Fix::from_raw(magnet.pickup_radius),
                    max_lock_distance_squared: Fix::from_raw(magnet.max_lock_distance_squared),
                    close_pull_multiplier: Fix::from_raw(magnet.close_pull_multiplier),
                    fast_pull_radius_squared: Fix::from_raw(magnet.fast_pull_radius_squared),
                    pull_width: Fix::from_raw(magnet.pull_width),
                    pull_strength: Fix::from_raw(magnet.pull_strength),
                    player_pull_strength: Fix::from_raw(magnet.player_pull_strength),
                    projectile_pull_strength: Fix::from_raw(magnet.projectile_pull_strength),
                    wall_pull_strength: Fix::from_raw(magnet.wall_pull_strength),
                    black_hole_pull_strength: Fix::from_raw(magnet.black_hole_pull_strength),
                    macho_pull_strength: Fix::from_raw(magnet.macho_pull_strength),
                    player_mass: Fix::from_raw(magnet.player_mass),
                    locked_angular_velocity: Fix::from_raw(magnet.locked_angular_velocity),
                    small_ragdoll_duration: Fix::from_raw(magnet.small_ragdoll_duration),
                    fire_ragdoll_duration: Fix::from_raw(magnet.fire_ragdoll_duration),
                    wall_mask: magnet.wall_mask,
                    raycast_offset: magnet.raycast_offset.get(),
                    max_pull_duration: Fix::from_raw(magnet.max_pull_duration),
                    parry_push_strength: Fix::from_raw(magnet.parry_push_strength),
                    pull_curve: magnet.pull_curve.clone(),
                    enter_ticks: magnet.enter_ticks,
                    prepull_ticks: magnet.prepull_ticks,
                    exit_ticks: magnet.exit_ticks,
                }
            },
            smoke: {
                let smoke = &raw.prefabs.smoke;
                let throw = &smoke.throw_data;
                let mut cloud_enter_frame_ticks =
                    Vec::with_capacity(smoke.cloud.enter_frames.len());
                let mut elapsed = 0;
                for duration in &smoke.cloud.enter_frames {
                    elapsed += duration.unsigned_abs();
                    cloud_enter_frame_ticks.push(elapsed);
                }
                SmokeData {
                    hurtbox_extents: smoke.hurtbox_extents.get(),
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
                    item_layer: smoke.item.layer,
                    item_updatable_count: smoke.item.updatable_count,
                    item_collider_hierarchy: smoke.item.collider_hierarchy,
                    item_radius: Fix::from_raw(smoke.item.radius),
                    item_body: body(&smoke.item.body, Fix::ZERO),
                    detonation_time: Fix::from_raw(smoke.item.detonation_time),
                    clone_count: smoke.item.clone_count,
                    duplicate_pattern: smoke.item.duplicate_pattern,
                    duplicate_spacing: Fix::from_raw(smoke.item.duplicate_spacing),
                    duplicate_angle: Fix::from_raw(smoke.item.duplicate_angle),
                    cloud_offsets: smoke.item.cloud_offsets.iter().map(RawVec2::get).collect(),
                    spread: Fix::from_raw(smoke.item.spread),
                    noise: Fix::from_raw(smoke.item.noise),
                    burst_updatable_count: smoke.item.burst_updatable_count,
                    cloud_layer: smoke.cloud.layer,
                    cloud_updatable_count: smoke.cloud.updatable_count,
                    cloud_collider_hierarchy: smoke.cloud.collider_hierarchy,
                    cloud_base_radius: Fix::from_raw(smoke.cloud.base_radius),
                    cloud_body: body(&smoke.cloud.body, Fix::ZERO),
                    cloud_clone_count: smoke.cloud.clone_count,
                    cloud_duplicate_pattern: smoke.cloud.duplicate_pattern,
                    cloud_duplicate_spacing: Fix::from_raw(smoke.cloud.duplicate_spacing),
                    cloud_duplicate_angle: Fix::from_raw(smoke.cloud.duplicate_angle),
                    cloud_radii: smoke
                        .cloud
                        .radii
                        .iter()
                        .copied()
                        .map(Fix::from_raw)
                        .collect(),
                    cloud_enter_frame_ticks,
                    min_ignition_delay: Fix::from_raw(smoke.cloud.min_ignition_delay),
                    max_ignition_delay: Fix::from_raw(smoke.cloud.max_ignition_delay),
                    explosion_layer: smoke.explosion.layer,
                    explosion_updatable_count: smoke.explosion.updatable_count,
                    explosion_collider_hierarchy: smoke.explosion.collider_hierarchy,
                    explosion_extents: smoke.explosion.extents.get(),
                    explosion_animation_ticks: smoke.explosion.animation_ticks,
                }
            },
            time_stop: {
                let time_stop = &raw.prefabs.time_stop;
                TimeStopData {
                    cast_time: Fix::from_raw(time_stop.cast_time),
                    center_offset: time_stop.center_offset.get(),
                    exit_offset: time_stop.exit_offset.get(),
                    collider_offset: time_stop.collider_offset.get(),
                    hurtbox_radius: Fix::from_raw(time_stop.hurtbox_radius),
                    collider_hierarchy: time_stop.collider_hierarchy,
                    idle_to_fall_ticks: ticks(&time_stop.animations.idle_to_fall),
                    cast_spell_ticks: ticks(&time_stop.animations.cast_spell),
                    ease_out_ticks: ticks(&time_stop.animations.ease_out),
                    effect_updatable_count: time_stop.effect_updatable_count,
                    duration_f32: time_stop.duration_f32,
                    exit_seconds_f32: time_stop.exit_seconds_f32,
                }
            },
            spike: {
                let spike = &raw.prefabs.spike;
                SpikeData {
                    hurtbox_extents: spike.hurtbox_extents.get(),
                    collider_offset: spike.collider_offset.get(),
                    collider_hierarchy: spike.collider_hierarchy,
                    cast_ticks: ticks(&spike.animations.cast),
                    recover_ticks: ticks(&spike.animations.recover),
                    surface_offset: Fix::from_raw(spike.surface_offset),
                    cast_force: Fix::from_raw(spike.cast_force),
                    attack_layer: spike.attack.layer,
                    attack_updatable_count: spike.attack.updatable_count,
                    attack_collider_hierarchy: spike.attack.collider_hierarchy,
                    attack_extents: spike.attack.extents.get(),
                    attack_collider_offset: spike.attack.collider_offset.get(),
                    attack_body: body(&spike.attack.body, Fix::ZERO),
                    down_thrust: spike.attack.down_thrust.clone(),
                    knock_player: Fix::from_raw(spike.attack.knock_player),
                    knock_wall: Fix::from_raw(spike.attack.knock_wall),
                    self_push_multiplier: Fix::from_raw(spike.attack.self_push_multiplier),
                    push_time: Fix::from_raw(spike.attack.push_time),
                }
            },
            engine: {
                let engine = &raw.prefabs.engine;
                EngineData {
                    hurtbox_extents: engine.hurtbox_extents.get(),
                    collider_offset: engine.collider_offset.get(),
                    collider_hierarchy: engine.collider_hierarchy,
                    cast_ticks: ticks(&engine.animations.cast),
                    recover_ticks: ticks(&engine.animations.recover),
                    plant_offset: engine.plant_offset.get(),
                    cast_force: Fix::from_raw(engine.cast_force),
                    buffer_size: engine.buffer_size,
                    rocket_layer: engine.rocket.layer,
                    rocket_updatable_count: engine.rocket.updatable_count,
                    rocket_collider_hierarchy: engine.rocket.collider_hierarchy,
                    rocket_extents: engine.rocket.extents.get(),
                    rocket_collider_offset: engine.rocket.collider_offset.get(),
                    rocket_body: body(&engine.rocket.body, Fix::ZERO),
                    force: engine.rocket.force.clone(),
                    start_ticks: engine.rocket.start_ticks,
                    end_ticks: engine.rocket.end_ticks,
                    fire_offset: engine.rocket.fire_offset.get(),
                    fire_radius: Fix::from_raw(engine.rocket.fire_radius),
                    smoke_mask: engine.rocket.smoke_mask,
                }
            },
            tesla: {
                let tesla = &raw.prefabs.tesla;
                TeslaData {
                    hurtbox_extents: tesla.hurtbox_extents.get(),
                    collider_offset: tesla.collider_offset.get(),
                    collider_hierarchy: tesla.collider_hierarchy,
                    cast_ticks: ticks(&tesla.animations.cast),
                    recover_ticks: ticks(&tesla.animations.recover),
                    plant_offset: tesla.plant_offset.get(),
                    cast_force: Fix::from_raw(tesla.cast_force),
                    buffer_size: tesla.buffer_size,
                    node_layer: tesla.node.layer,
                    node_updatable_count: tesla.node.updatable_count,
                    node_collider_hierarchy: tesla.node.collider_hierarchy,
                    node_radius: Fix::from_raw(tesla.node.radius),
                    node_collider_offset: tesla.node.collider_offset.get(),
                    node_body: body(&tesla.node.body, Fix::ZERO),
                    line_length: Fix::from_raw(tesla.node.line_length),
                    hitbox_offset_y: Fix::from_raw(tesla.node.hitbox_offset_y),
                    lightning_layer: tesla.lightning.layer,
                    lightning_updatable_count: tesla.lightning.updatable_count,
                    lightning_collider_hierarchy: tesla.lightning.collider_hierarchy,
                    lightning_extents: tesla.lightning.extents.get(),
                    lightning_minimum_extent: Fix::from_raw(tesla.lightning.minimum_extent),
                    flicker_period: Fix::from_raw(tesla.lightning.flicker_period),
                    enter_ticks: tesla.lightning.enter_ticks,
                    flicker_ticks: tesla.lightning.flicker_ticks,
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
            mjolnir: {
                let mjolnir = &raw.prefabs.mjolnir;
                let throw = &mjolnir.throw_data;
                MjolnirData {
                    hurtbox_extents: mjolnir.hurtbox_extents.get(),
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
                    item_layer: mjolnir.item.layer,
                    item_updatable_count: mjolnir.item.updatable_count,
                    item_collider_hierarchy: mjolnir.item.collider_hierarchy,
                    item_radius: Fix::from_raw(mjolnir.item.radius),
                    item_body: body(&mjolnir.item.body, Fix::ZERO),
                    clone_count: mjolnir.item.clone_count,
                    duplicate_pattern: mjolnir.item.duplicate_pattern,
                    duplicate_spacing: Fix::from_raw(mjolnir.item.duplicate_spacing),
                    duplicate_angle: Fix::from_raw(mjolnir.item.duplicate_angle),
                    outbound_drag: Fix::from_raw(mjolnir.item.outbound_drag),
                    return_speed_threshold: Fix::from_raw(mjolnir.item.return_speed_threshold),
                    max_flight_seconds: Fix::from_raw(mjolnir.item.max_flight_seconds),
                    recall_accel: Fix::from_raw(mjolnir.item.recall_accel),
                    recall_max_speed: Fix::from_raw(mjolnir.item.recall_max_speed),
                    clang_grace_seconds: Fix::from_raw(mjolnir.item.clang_grace_seconds),
                    platform_push_force: Fix::from_raw(mjolnir.item.platform_push_force),
                }
            },
            poke_ball: {
                let poke = &raw.prefabs.poke_ball;
                let throw = &poke.throw_data;
                PokeBallData {
                    hurtbox_extents: poke.hurtbox_extents.get(),
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
                    item_layer: poke.item.layer,
                    item_updatable_count: poke.item.updatable_count,
                    item_collider_hierarchy: poke.item.collider_hierarchy,
                    item_radius: Fix::from_raw(poke.item.radius),
                    item_body: body(&poke.item.body, Fix::ZERO),
                    clone_count: poke.item.clone_count,
                    duplicate_pattern: poke.item.duplicate_pattern,
                    duplicate_spacing: Fix::from_raw(poke.item.duplicate_spacing),
                    duplicate_angle: Fix::from_raw(poke.item.duplicate_angle),
                    capture_seconds: Fix::from_raw(poke.item.capture_seconds),
                    fizzle_seconds: Fix::from_raw(poke.item.fizzle_seconds),
                    pop_up_speed: Fix::from_raw(poke.item.pop_up_speed),
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
