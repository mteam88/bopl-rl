//! Independent movement simulation for the exported Level1 reset state.
//! Tick ordering follows PlayerPhysics -> SlimeController -> PlayerBody -> late alignment.
use crate::{
    fixed::Fix,
    geometry::{OrientedBox, RoundedRect},
    orientation::Orientation,
    surface::Surface,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};
use std::cell::{Cell, RefCell};

mod dash;

pub const DT: Fix = Fix::from_raw(71_582_788);
fn default_scale() -> Fix {
    Fix::ONE
}
fn default_true() -> bool {
    true
}
fn default_up() -> Vec2 {
    Vec2::UP
}
fn is_true(value: &bool) -> bool {
    *value
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Platform {
    pub id: u32,
    /// Unity keeps inactive platform colliders registered, but excludes them
    /// from queries and simulation until their GameObject is reactivated.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub active: bool,
    pub surface: Surface,
    pub collider: RoundedRect,
    pub ice: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MovementConstants {
    pub max_speed: Fix,
    pub accel: Fix,
    pub air_accel: Fix,
    pub gravity_accel: Fix,
    pub gravity_max_fall_speed: Fix,
    pub gravity_modifier: Fix,
    pub jump_strength: Fix,
    pub jump_extra_x: Fix,
    pub jump_momentum: Fix,
    pub jump_normal_scale: Fix,
    pub jump_teleport: Fix,
    pub friction: Fix,
    pub ice_friction: Fix,
    /// PlayerPhysics.cannotBeUngroundedByRopes: on for the Throw and Roll
    /// forms', so a grounded rope end's pull never UnGrounds that body
    /// (PlayerPhysics.cs:156).
    #[serde(default)]
    pub cannot_be_ungrounded_by_ropes: bool,
}

fn default_fixed_delta_time() -> Fix {
    DT
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Player {
    pub alive: bool,
    /// Set with `alive = false` when a boundary component destroyed the body.
    /// The death is attributed later in the pass, after an ungrounding exit
    /// may already have moved the body back inside the bounds.
    #[serde(default)]
    pub out_of_bounds: bool,
    pub id: u32,
    pub position: Vec2,
    /// The body FixTransform's rotation and up, only ever set together.
    #[serde(flatten)]
    pub orientation: Orientation,
    pub velocity: Vec2,
    pub external_velocity: Vec2,
    pub radius: Fix,
    /// LevelType.space changes gravity and the lower blast boundary.
    #[serde(default, skip_serializing)]
    pub space: bool,
    /// FixTransform/DPhysics collider scale. Player.Scale lives above the
    /// body and can lead this value by one updater phase.
    #[serde(skip, default = "default_scale")]
    pub scale: Fix,
    /// PlayerPhysics.prevScal: the scale `radius` last followed. UpdateScale
    /// multiplies the radius by `scale / prevScal` (PlayerPhysics.cs:106-123).
    #[serde(skip, default = "default_scale")]
    pub(crate) physics_scale: Fix,
    pub grounded: bool,
    pub ground: Option<u32>,
    pub local: Fix,
    pub grounded_speed: Fix,
    pub previous_jump: bool,
    /// PlayerPhysics already ran on the tick where SlimeController notices
    /// that its attached platform was destroyed, so gravity starts one pass
    /// later for the ordinary slime.
    #[serde(default)]
    pub(crate) skip_next_gravity: bool,
    /// GameTime.FixedDeltaTime(playerIdHolder, simDeltaTime) for the body's
    /// player, taken at its turn: zero while a Time Stop freezes that player.
    /// PlayerPhysics and PlayerBody scale only these steps by it.
    #[serde(default = "default_fixed_delta_time")]
    pub(crate) fixed_delta_time: Fix,
    /// The slime's PlayerPhysics.jumpedThisFrame: set by its Jump, cleared
    /// once the tick's state machine has seen it.
    #[serde(default)]
    pub(crate) jumped_this_frame: bool,
    /// The slime's PlayerBody.simulated: Awake and BeforeSpawnInit clear it,
    /// Spawn sets it, PrepareToRevive clears it again
    /// (SlimeController.cs:499-591, 625). PlayerBody.IsEnabled reads it, so
    /// an unsimulated body takes no UpdateSim or LateUpdateSim turn.
    #[serde(default = "default_true")]
    pub(crate) simulated: bool,
    /// The slime's PlayerPhysics.enabled: BeforeSpawnInit and PrepareToRevive
    /// clear it, Spawn sets it (SlimeController.cs:518, 569, 624), and a
    /// RewindRecorder clears it for its playback and then restores it
    /// (RewindRecorder.cs:149-151, 355).
    #[serde(default = "default_true")]
    pub(crate) physics_enabled: bool,
    /// SlimeController.OldUpdate returns as soon as it has entered an ability
    /// (SlimeController.cs:824-831), so neither the state switch, the jump
    /// block at 867-871 nor the `prevInput` store at 889 runs on a cast pass. Set for the instant
    /// abilities, which keep the ordinary slime body; the form abilities are
    /// already held out of both by `PlayerSlot::active`. Scoped to one pass:
    /// `player_tick` clears it, the cast sets it, and the movement below plus
    /// `RolloutWorld::slime_tail` read it without consuming it.
    #[serde(default)]
    pub(crate) instant_cast_this_pass: bool,
    /// `SlimeController.wasUpsideDown`: `Jump()` records the sign of the
    /// ground normal it is leaving before handing over to PlayerPhysics.Jump
    /// (SlimeController.cs:897-900), and `AlignTransformAir` consumes it on
    /// the same pass (SlimeController.cs:967-972). Only the slime's own jump
    /// records it; the ability jumps call PlayerPhysics.Jump directly.
    #[serde(default)]
    pub(crate) was_upside_down: bool,
    /// The ground frame `OldUpdate` saw. SlimeController.UpdateSim runs
    /// OldUpdate at line 1067 and only calls `PlayerPhysics.Move` at 1084, so
    /// `fix`, `fix2` and both `IsGrounded()` tests (SlimeController.cs:795-798
    /// and 890) read the pose left by PlayerPhysics.UpdateSim's gravity and
    /// landing raycasts, never the one Move goes on to produce.
    #[serde(default = "default_up")]
    pub(crate) pre_move_up: Vec2,
    #[serde(default)]
    pub(crate) pre_move_grounded: bool,
    /// `attachToGround` flips the sprite from the normal at the instant of
    /// attachment (SlimeController.cs:1126-1134). That normal is not the
    /// end-of-tick `up`: on a steep or rotating surface the two differ by a
    /// whole tick of surface rotation. `None` when nothing attached this pass.
    #[serde(default)]
    pub(crate) attach_upside_down: Option<bool>,
    pub constants: MovementConstants,
    pub cooldown: Fix,
    pub dash: DashState,
    /// Player-layer box as DetPhysics.Simulate last synced it (position and
    /// orientation of the tick's UpdateSim phase; a grounded body moves again
    /// in LateUpdateSim, so this lags the live position by one phase).
    pub collider: OrientedBox,
    /// The box as DetPhysics holds it for mid-pass queries: moved only by
    /// its Simulate sync and by explicit syncs (UpdatePhysicsPositions,
    /// SyncPhysicsTo), not by `move_collider`.
    #[serde(default)]
    pub synced: OrientedBox,
    #[serde(default)]
    pub collider_rotation: Fix,
    /// FixTransform.up as `collider` last took it. A form box applies its
    /// offset with FixTransform.right itself, which the angle round trip to
    /// `collider_rotation` can miss by a few Q32 units.
    #[serde(default)]
    pub collider_up: Vec2,
    /// `Vec2::angle(collider_rotation)`, kept so an unchanged rotation does
    /// not pay for the sine table again (zero until the first sync).
    #[serde(default)]
    pub collider_basis: Vec2,
    /// The live PlayerPhysics' SyncRotationToPlatform: its Move and
    /// LateUpdateSim turn a grounded body to the surface only when it is set
    /// (PlayerPhysics.cs:143, 221, 268); AttachToGround and SyncPhysicsTo
    /// align it regardless (PlayerPhysics.cs:508, 569).
    #[serde(default = "default_true")]
    pub sync_rotation_to_platform: bool,
    /// `(up, rotation)` as `align` last derived them: a step along a flat
    /// stretch of surface keeps the same normal, so the arc-cosine is not
    /// paid again for the same input (the value is identical either way).
    #[serde(skip)]
    pub(crate) aligned: Option<(Vec2, Fix)>,
    /// `(input, input.normalized_safe())` from the last grounded step: a
    /// decision holds its stick for several ticks, so the square root and
    /// divisions are usually already known (identical value either way).
    #[serde(skip)]
    pub(crate) input_normalized: Option<(Vec2, Vec2)>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DashState {
    pub active: bool,
    pub time: Fix,
    pub updates: u32,
    pub in_dash: bool,
    pub input: Vec2,
    pub hit_ground: Option<u32>,
    pub hit_local: Fix,
    pub saved_local: Fix,
    pub local: Fix,
    /// The Dash body's rotation when Ability.ExitAbility deactivated it:
    /// the dash direction, or the platform alignment of a landing
    /// (PlayerPhysics.AttachToGround, PlayerPhysics.cs:508). The frozen
    /// FixTransform keeps it for DetPhysics' sync of inactive colliders.
    #[serde(default)]
    pub exit_rotation: Fix,
    /// The Dash body's position when `Ability.ExitAbility` read it
    /// (Ability.cs:151 `SetPosition(potentialCollider.position,
    /// slimeController)`), i.e. before `onExitAbility` hands the slime back
    /// and `SlimeController.EnterFall` runs its velocity rays. That is the
    /// last write to `Player.Position` on the exit pass, so the public
    /// player row publishes the raw dash endpoint for one tick and only
    /// picks up the ray-corrected body on the next.
    #[serde(default)]
    pub exit_position: Vec2,
    /// Dash's own PlayerPhysics radius, which follows the form's scale
    /// history rather than the slime's. Set before every entry.
    #[serde(default)]
    pub radius: Fix,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct Action {
    pub angle: u8,
    pub buttons: u8,
}
impl Action {
    pub fn direction(self) -> Vec2 {
        static DIRECTIONS: std::sync::OnceLock<[Vec2; 256]> = std::sync::OnceLock::new();
        // The 255 stick directions are fixed; evaluating each once with the
        // same expression gives the same value as evaluating it per call.
        DIRECTIONS.get_or_init(|| std::array::from_fn(|angle| Self::direction_of(angle as u8)))
            [usize::from(self.angle)]
    }
    fn direction_of(angle: u8) -> Vec2 {
        if angle == 0 {
            return Vec2::ZERO;
        }
        let angle = Fix::TAU * (Fix::from_int(i64::from(angle) - 1) / Fix::from_int(255));
        Vec2::new(angle.cos(), angle.sin())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct World {
    pub tick: u64,
    pub platforms: Platforms,
    pub players: Vec<Player>,
}

impl World {
    pub fn level1() -> Self {
        serde_json::from_str(include_str!("../data/level1-movement.json"))
            .expect("validated embedded reset")
    }
    pub fn level1_with_players(players: usize) -> Option<Self> {
        let reset = match players {
            1 => include_str!("../data/level1-movement-1.json"),
            2 => return Some(Self::level1()),
            3 => include_str!("../data/level1-movement-3.json"),
            4 => include_str!("../data/level1-movement-4.json"),
            5 => include_str!("../data/level1-movement-5.json"),
            6 => include_str!("../data/level1-movement-6.json"),
            _ => return None,
        };
        Some(serde_json::from_str(reset).expect("validated embedded multiplayer reset"))
    }
    pub fn terminal(&self) -> bool {
        self.players.iter().any(|p| !p.alive)
    }
    pub fn step(&mut self, actions: &[Action]) -> Result<(), String> {
        if self.terminal() {
            return Err("episode ended; reset is required".into());
        }
        if actions.len() != self.players.len() {
            return Err("one action per player is required".into());
        }
        if actions.iter().any(|a| a.buttons & !3 != 0) {
            return Err("this slice supports jump and dash only".into());
        }
        let mut terrain = self.platforms.terrain();
        for (index, action) in actions.iter().enumerate() {
            self.players[index].step(*action, &mut terrain);
            let caster = &self.players[index];
            if caster.dash.active && caster.dash.time == DT {
                let origin = caster.position;
                let direction = caster.dash.input;
                let wall = closest(terrain, origin, direction, Fix::from_int(13), None)
                    .map(|(_, distance, _)| distance);
                let targets: Vec<_> = self
                    .players
                    .iter()
                    .enumerate()
                    .filter(|(i, p)| *i != index && !p.dash.active)
                    .filter_map(|(i, p)| {
                        p.collider
                            .raycast(origin, direction)
                            .filter(|hit| {
                                hit.near <= Fix::from_int(13)
                                    && wall.is_none_or(|distance| hit.near < distance)
                            })
                            .map(|_| i)
                    })
                    .collect();
                for target in targets {
                    self.players[target].push_from(origin, target < index);
                }
            }
        }
        // For the fixed-size default slime boxes, distance < radius implies box
        // overlap (radius .76, each box inscribed radius .81). Only that subset
        // produces SlimeController.OnCollide's push; non-overlap contacts do nothing.
        for a in 0..self.players.len() {
            for b in a + 1..self.players.len() {
                let a_pos = self.players[a].position;
                let b_pos = self.players[b].position;
                if self.players[a].alive && self.players[b].alive && !self.players[a].dash.active {
                    self.players[a].push_from(b_pos, true);
                }
                if self.players[a].alive && self.players[b].alive && !self.players[b].dash.active {
                    self.players[b].push_from(a_pos, false);
                }
            }
        }
        for player in &mut self.players {
            player.sync_collider(player.authored_inverse_extents());
            player.late_step(&self.platforms);
        }
        self.tick += 1;
        Ok(())
    }
}

/// The raycast broad phase: every platform's axis-aligned bounds
/// (`RoundedRect::bounds`, already widened by `Bounds::may_reach`'s 1e-6
/// margin) in lanes of eight, so a query is a few vector compares instead of
/// re-deriving the bounds from each 150-byte `Platform`. Marked stale by
/// `invalidate`, it rebuilds itself from the platforms of the next query.
#[derive(Debug)]
pub struct PlatformBounds {
    chunks: RefCell<Vec<BoundsChunk>>,
    len: Cell<usize>,
    stale: Cell<bool>,
}

impl Default for PlatformBounds {
    fn default() -> Self {
        Self {
            chunks: RefCell::new(Vec::new()),
            len: Cell::new(0),
            stale: Cell::new(true),
        }
    }
}

impl Clone for PlatformBounds {
    fn clone(&self) -> Self {
        Self {
            chunks: RefCell::new(self.chunks.borrow().clone()),
            len: self.len.clone(),
            stale: self.stale.clone(),
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.chunks.get_mut().clone_from(&source.chunks.borrow());
        self.len.set(source.len.get());
        self.stale.set(source.stale.get());
    }
}

const BOUNDS_LANES: usize = 8;

/// Eight platforms' bounds, one axis per array. Unused lanes hold an empty
/// box (`max < min`) that no query meets.
#[derive(Clone, Copy, Debug)]
struct BoundsChunk {
    max_x: [f64; BOUNDS_LANES],
    min_x: [f64; BOUNDS_LANES],
    max_y: [f64; BOUNDS_LANES],
    min_y: [f64; BOUNDS_LANES],
}

impl BoundsChunk {
    const EMPTY: Self = Self {
        max_x: [f64::NEG_INFINITY; BOUNDS_LANES],
        min_x: [f64::INFINITY; BOUNDS_LANES],
        max_y: [f64::NEG_INFINITY; BOUNDS_LANES],
        min_y: [f64::INFINITY; BOUNDS_LANES],
    };
    /// Lane bits of the boxes meeting `[lo, hi]`, as one 8-bit mask: four
    /// ordered vector compares (a NaN lane never hits, like the scalar `<=`).
    #[inline]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    fn candidates(&self, lo_x: f64, hi_x: f64, lo_y: f64, hi_y: f64) -> u64 {
        use std::arch::x86_64::{
            _CMP_GE_OQ, _CMP_LE_OQ, _mm512_cmp_pd_mask, _mm512_loadu_pd, _mm512_set1_pd,
        };
        // SAFETY: avx512f is a compile-time target feature here, and each
        // array is exactly eight f64 (unaligned loads).
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(
                _mm512_set1_pd(lo_x),
                _mm512_loadu_pd(self.max_x.as_ptr()),
            ) & _mm512_cmp_pd_mask::<_CMP_GE_OQ>(
                _mm512_set1_pd(hi_x),
                _mm512_loadu_pd(self.min_x.as_ptr()),
            ) & _mm512_cmp_pd_mask::<_CMP_LE_OQ>(
                _mm512_set1_pd(lo_y),
                _mm512_loadu_pd(self.max_y.as_ptr()),
            ) & _mm512_cmp_pd_mask::<_CMP_GE_OQ>(
                _mm512_set1_pd(hi_y),
                _mm512_loadu_pd(self.min_y.as_ptr()),
            );
            u64::from(mask)
        }
    }
    /// Lane bits of the boxes meeting `[lo, hi]`, as one 8-bit mask.
    #[inline]
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    fn candidates(&self, lo_x: f64, hi_x: f64, lo_y: f64, hi_y: f64) -> u64 {
        let mut mask = 0u64;
        for lane in 0..BOUNDS_LANES {
            let hit = lo_x <= self.max_x[lane]
                && hi_x >= self.min_x[lane]
                && lo_y <= self.max_y[lane]
                && hi_y >= self.min_y[lane];
            mask |= u64::from(hit) << lane;
        }
        mask
    }
}

/// The platforms and their raycast broad phase. DetPhysics' casts read every
/// collider where it stands, so reads go through `Deref` while any mutable
/// access marks the broad phase stale; the next query rebuilds it. No query
/// can see a collider's old bounds.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(from = "Vec<Platform>", into = "Vec<Platform>")]
pub struct Platforms {
    list: Vec<Platform>,
    bounds: PlatformBounds,
}

impl Clone for Platforms {
    fn clone(&self) -> Self {
        Self {
            list: self.list.clone(),
            bounds: self.bounds.clone(),
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.list.clone_from(&source.list);
        self.bounds.clone_from(&source.bounds);
    }
}

impl From<Vec<Platform>> for Platforms {
    fn from(list: Vec<Platform>) -> Self {
        Self {
            list,
            bounds: PlatformBounds::default(),
        }
    }
}

impl From<Platforms> for Vec<Platform> {
    fn from(platforms: Platforms) -> Self {
        platforms.list
    }
}

impl std::ops::Deref for Platforms {
    type Target = Vec<Platform>;
    fn deref(&self) -> &Vec<Platform> {
        &self.list
    }
}

impl<'a> IntoIterator for &'a Platforms {
    type Item = &'a Platform;
    type IntoIter = std::slice::Iter<'a, Platform>;
    fn into_iter(self) -> Self::IntoIter {
        self.list.iter()
    }
}

impl std::ops::DerefMut for Platforms {
    fn deref_mut(&mut self) -> &mut Vec<Platform> {
        self.bounds.invalidate();
        &mut self.list
    }
}

impl Platforms {
    pub fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity).into()
    }
    pub fn terrain(&self) -> Terrain<'_> {
        Terrain {
            platforms: &self.list,
            bounds: &self.bounds,
        }
    }
}

impl PlatformBounds {
    fn invalidate(&mut self) {
        *self.stale.get_mut() = true;
    }
    fn refresh(&self, platforms: &[Platform]) {
        if !self.stale.get() && self.len.get() == platforms.len() {
            return;
        }
        assert!(
            platforms.len() <= 64,
            "platform count exceeds the candidate mask"
        );
        let margin = Fix::from_f64(1e-6);
        self.len.set(platforms.len());
        self.stale.set(false);
        let mut chunks = self.chunks.borrow_mut();
        chunks.clear();
        chunks.resize(platforms.len().div_ceil(BOUNDS_LANES), BoundsChunk::EMPTY);
        for (index, platform) in platforms.iter().enumerate() {
            let bounds = platform.collider.bounds();
            let chunk = &mut chunks[index / BOUNDS_LANES];
            let lane = index % BOUNDS_LANES;
            chunk.max_x[lane] = (bounds.max.x + margin).to_f64();
            chunk.min_x[lane] = (bounds.min.x - margin).to_f64();
            chunk.max_y[lane] = (bounds.max.y + margin).to_f64();
            chunk.min_y[lane] = (bounds.min.y - margin).to_f64();
        }
    }
    /// Bit `i` set when `platforms[i]`'s bounds meet the axis-aligned box
    /// `[lo, hi]` (`Bounds::may_reach` for a segment with those extents).
    fn candidates(&self, platforms: &[Platform], lo: Vec2, hi: Vec2) -> u64 {
        self.refresh(platforms);
        let (lo_x, hi_x, lo_y, hi_y) = (lo.x.to_f64(), hi.x.to_f64(), lo.y.to_f64(), hi.y.to_f64());
        let mut mask = 0u64;
        for (index, chunk) in self.chunks.borrow().iter().enumerate() {
            mask |= chunk.candidates(lo_x, hi_x, lo_y, hi_y) << (index * BOUNDS_LANES);
        }
        mask
    }
}

/// The axis-aligned box covering a set of ray segments.
#[derive(Clone, Copy)]
pub(crate) struct SegmentBox {
    lo: Vec2,
    hi: Vec2,
}
impl SegmentBox {
    pub(crate) fn empty() -> Self {
        Self {
            lo: Vec2::new(Fix::MAX, Fix::MAX),
            hi: Vec2::new(Fix::MIN, Fix::MIN),
        }
    }
    pub(crate) fn add(&mut self, point: Vec2) {
        self.lo = Vec2::new(self.lo.x.min(point.x), self.lo.y.min(point.y));
        self.hi = Vec2::new(self.hi.x.max(point.x), self.hi.y.max(point.y));
    }
    pub(crate) fn add_segment(&mut self, origin: Vec2, end: Vec2) {
        self.add(origin);
        self.add(end);
    }
    pub(crate) fn candidates(self, terrain: Terrain) -> u64 {
        terrain.candidates(self.lo, self.hi)
    }
}

/// The platforms and their broad-phase bounds, as every raycasting step
/// needs them together.
#[derive(Clone, Copy)]
pub struct Terrain<'a> {
    pub platforms: &'a [Platform],
    bounds: &'a PlatformBounds,
}

impl<'a> Terrain<'a> {
    /// Bit `i` set when `platforms[i]`'s bounds meet the axis-aligned box
    /// `[lo, hi]`.
    pub fn candidates(self, lo: Vec2, hi: Vec2) -> u64 {
        self.bounds.candidates(self.platforms, lo, hi)
    }
}

pub(crate) fn closest(
    terrain: Terrain,
    origin: Vec2,
    direction: Vec2,
    distance: Fix,
    exclude: Option<u32>,
) -> Option<(usize, Fix, Vec2)> {
    let end = origin + direction * distance;
    let mut segment = SegmentBox::empty();
    segment.add_segment(origin, end);
    closest_among(
        terrain,
        segment.candidates(terrain),
        origin,
        direction,
        distance,
        exclude,
    )
}

/// `closest` restricted to the platforms in `candidates`, a mask from
/// `PlatformBounds::candidates` over a box that covers this segment (and
/// possibly others: the broad phase is conservative, so testing a platform
/// the segment's own box would have skipped cannot add a hit, and the
/// ascending index order keeps the tie-break on equal distances).
pub(crate) fn closest_among(
    terrain: Terrain,
    mut candidates: u64,
    origin: Vec2,
    direction: Vec2,
    distance: Fix,
    exclude: Option<u32>,
) -> Option<(usize, Fix, Vec2)> {
    let platforms = terrain.platforms;
    let mut nearest = Fix::MAX;
    let mut result = None;
    while candidates != 0 {
        let index = candidates.trailing_zeros() as usize;
        candidates &= candidates - 1;
        let platform = &platforms[index];
        if !platform.active || exclude == Some(platform.id) {
            continue;
        }
        if let Some(hit) = platform.collider.raycast(origin, direction)
            && hit.near < nearest
            && hit.near <= distance
        {
            nearest = hit.near;
            result = Some((index, hit.near, origin + direction * hit.near));
        }
    }
    result
}
/// What player physics moves over and queries: the platforms, and
/// DetPhysics.RaycastToClosestRoundedRect, which PlayerPhysics.checkForWall
/// and StickyRoundedRectangle.maxMovementAllowed call (PlayerPhysics.cs:511-513,
/// StickyRoundedRectangle.cs:441-451).
pub(crate) trait Walls {
    fn platforms(&self) -> &[Platform];
    /// The nearest active platform but `excluded` the ray meets within
    /// `distance`: its index, near distance and hit point.
    fn raycast_to_closest_rounded_rect(
        &mut self,
        origin: Vec2,
        direction: Vec2,
        distance: Fix,
        excluded: Option<usize>,
    ) -> Option<(usize, Fix, Vec2)>;
}

/// Platforms with no destroyed collider to purge: the legacy World's, or a
/// copy of the rollout's.
impl Walls for Terrain<'_> {
    fn platforms(&self) -> &[Platform] {
        self.platforms
    }
    fn raycast_to_closest_rounded_rect(
        &mut self,
        origin: Vec2,
        direction: Vec2,
        distance: Fix,
        excluded: Option<usize>,
    ) -> Option<(usize, Fix, Vec2)> {
        closest(*self, origin, direction, distance, excluded.map(|index| self.platforms[index].id))
    }
}

impl Player {
    pub(crate) fn platform<'a>(&self, platforms: &'a [Platform]) -> Option<&'a Platform> {
        self.platform_indexed(platforms)
            .map(|(_, platform)| platform)
    }
    /// PlayerPhysics.attachedGround, which only a destroyed ground clears: a
    /// deactivated one is still named, and its surface still read.
    pub(crate) fn attached_ground<'a>(&self, platforms: &'a [Platform]) -> Option<&'a Platform> {
        platforms.iter().find(|p| Some(p.id) == self.ground)
    }
    fn platform_indexed<'a>(&self, platforms: &'a [Platform]) -> Option<(usize, &'a Platform)> {
        platforms
            .iter()
            .enumerate()
            .find(|(_, p)| p.active && Some(p.id) == self.ground)
    }
    /// `input.normalized_safe()`, memoized on the exact input vector.
    fn normalized_input(&mut self, input: Vec2) -> Vec2 {
        match self.input_normalized {
            Some((cached, normalized)) if cached == input => normalized,
            _ => {
                let normalized = input.normalized_safe();
                self.input_normalized = Some((input, normalized));
                normalized
            }
        }
    }
    pub fn rotation(&self) -> Fix {
        self.orientation.rotation()
    }
    pub fn up(&self) -> Vec2 {
        self.orientation.up()
    }
    pub fn right(&self) -> Vec2 {
        self.orientation.right()
    }
    /// `body.rotation = value` (FixTransform.cs:42-47).
    pub fn set_rotation(&mut self, rotation: Fix) {
        self.orientation.set_rotation(rotation);
    }
    /// `body.up = value` (FixTransform.cs:55-60).
    pub fn set_up(&mut self, up: Vec2) {
        self.orientation.set_up(up);
    }
    /// `body.right = value` (FixTransform.cs:68-73).
    pub fn set_right(&mut self, right: Vec2) {
        self.orientation.set_right(right);
    }
    pub(crate) fn align(&mut self, platform: &Platform) {
        self.align_to(platform.surface.normal(self.position));
    }
    /// `align` with the surface normal already evaluated.
    pub(crate) fn align_to(&mut self, up: Vec2) {
        let rotation = match self.aligned {
            Some((cached_up, rotation)) if cached_up == up => rotation,
            _ => {
                let rotation = up.perp_cw().normalized_angle();
                self.aligned = Some((up, rotation));
                rotation
            }
        };
        self.orientation.set_up_with_angle(up, rotation);
    }
    fn ground_state(&self) -> crate::player_ground::State {
        crate::player_ground::State {
            position: self.position,
            rotation: self.rotation(),
            up: self.up(),
            right: self.right(),
            velocity: self.velocity,
            external_velocity: self.external_velocity,
            radius: self.radius,
            grounded: self.grounded,
            ground: self.ground,
            local: self.local,
            grounded_speed: self.grounded_speed,
        }
    }
    fn apply_ground_state(&mut self, motion: crate::player_ground::State) {
        self.position = motion.position;
        // The pair player_ground's own setter mirrors built.
        self.orientation.set_up_with_angle(motion.up, motion.rotation);
        self.velocity = motion.velocity;
        self.external_velocity = motion.external_velocity;
        self.grounded = motion.grounded;
        self.ground = motion.ground;
        self.local = motion.local;
        self.grounded_speed = motion.grounded_speed;
    }
    pub(crate) fn attach(&mut self, platform: &Platform, position: Vec2) {
        let mut motion = self.ground_state();
        motion
            .attach(platform.id, &platform.surface, position)
            .expect("valid attachment");
        self.apply_ground_state(motion);
        // SlimeController subscribes attachToGround to OnAttachedToGround
        // (SlimeController.cs:472) and reads `attachedGround.currentNormal(body)`
        // inside that callback (SlimeController.cs:1131). PlayerPhysics has
        // already snapped the body onto the surface by then
        // (player_physics.rs attaches before it notifies), so the normal it
        // sees is the post-attach one - exactly the `up` just stored, and not
        // the normal at the raw raycast point nor the end-of-tick one.
        self.attach_upside_down = Some(self.up().y < Fix::ZERO);
    }
    /// VelocityBasedRaycasts over FixedDeltaTime. Returns whether a ray hit
    /// and attached the body.
    pub(crate) fn collision_rays(&mut self, walls: &mut impl Walls) -> bool {
        self.collision_rays_over(walls, self.fixed_delta_time)
    }
    /// VelocityBasedRaycasts with the caller's time step, which scales the
    /// rays' length. Returns whether a ray hit and attached the body.
    pub(crate) fn collision_rays_over(&mut self, walls: &mut impl Walls, dt: Fix) -> bool {
        let velocity = self.velocity + self.external_velocity;
        // PlayerPhysics.checkForWall, once per ray (PlayerPhysics.cs:410-446).
        if let Some((index, position)) = crate::player_physics::collision_cast(
            self.position,
            velocity,
            self.radius,
            dt,
            &mut |origin, direction, distance| {
                walls
                    .raycast_to_closest_rounded_rect(origin, direction, distance, None)
                    .map(|(index, _, point)| (index, point))
            },
        ) {
            self.attach(&walls.platforms()[index], position);
            return true;
        }
        false
    }

    /// SlimeController.Jump (SlimeController.cs:897-901) on the slime's own
    /// PlayerPhysics, whose jumpedThisFrame the state switch reads
    /// (PlayerPhysics.cs:330-332). A form jumps on its own PlayerPhysics
    /// (Roll.cs:534-544, Machoman.cs:97) and leaves the slime's flag alone.
    pub(crate) fn jump(&mut self, platforms: &[Platform]) {
        self.record_upside_down_jump(platforms);
        self.jump_with_strength(platforms, self.constants.jump_strength);
        self.jumped_this_frame = true;
    }
    /// `SlimeController.Jump` reads the normal of the ground it is leaving and
    /// stores it in `wasUpsideDown` before calling PlayerPhysics.Jump
    /// (SlimeController.cs:897-900). Only the slime's own jump does this: the
    /// ability jumps (Roll, Meteor) reach PlayerPhysics.Jump directly and
    /// leave the flag alone.
    pub(crate) fn record_upside_down_jump(&mut self, platforms: &[Platform]) {
        let normal = self
            .grounded
            .then(|| platforms.iter().find(|p| Some(p.id) == self.ground))
            .flatten()
            .map_or(Vec2::UP, |p| p.surface.normal(self.position));
        self.was_upside_down = normal.y < Fix::ZERO;
    }
    pub(crate) fn jump_with_strength(&mut self, platforms: &[Platform], strength: Fix) {
        // PlayerPhysics.Jump uses world up while airborne and only reads the
        // attached platform when grounded (PlayerPhysics.cs:333). Roll can
        // charge in the air, so an attached platform is not required here.
        // The retained attachedGround is read whether or not it is still
        // active (Blink/Teleport can deactivate it under a grounded player
        // before Move ungrounds them), as in move_grounded_scaled; a ground
        // that no longer exists at all jumps off world up. UnGround still
        // lifts an airborne body off the ground it last left.
        let retained = platforms.iter().find(|p| Some(p.id) == self.ground);
        let platform = retained.filter(|_| self.grounded);
        let normal = platform.map_or(Vec2::UP, |p| p.surface.normal(self.position));
        let amount = ((normal.y + Fix::ONE) / self.constants.jump_normal_scale)
            .min(Fix::ONE)
            .max(Fix::ZERO);
        let up = Vec2::UP * amount + Vec2::new(normal.x, Fix::ZERO) * self.constants.jump_extra_x;
        self.velocity =
            normal.perp_cw() * self.grounded_speed * self.constants.jump_momentum + up * strength;
        self.position += self.velocity * self.constants.jump_teleport;
        self.unground(retained.map(|p| &p.surface), true);
    }
    /// PlayerPhysics.Move's ground input direction: -1, 0 or 1 along the
    /// surface tangent.
    pub(crate) fn ground_input_sign(&mut self, input: Vec2, normal: Vec2) -> Fix {
        let alignment = self.normalized_input(input).dot(normal.perp_cw());
        let into_ground = normal.dot(input);
        if alignment.abs() >= Fix::from_f32(0.2)
            || (into_ground <= Fix::ZERO && alignment.abs() >= Fix::from_f32(0.01))
        {
            Fix::from_int(alignment.sign())
        } else {
            Fix::ZERO
        }
    }
    fn move_grounded(&mut self, input: Vec2, walls: &mut impl Walls) {
        self.move_grounded_scaled(input, Fix::ONE, walls);
    }
    fn move_grounded_scaled(&mut self, input: Vec2, speed_scale: Fix, walls: &mut impl Walls) {
        // PlayerPhysics.Move computes ground input against the retained
        // attachedGround reference before testing activeSelf. Blink therefore
        // leaves one legal pass where the platform is present but inactive.
        let Some((platform_index, platform)) = walls
            .platforms()
            .iter()
            .enumerate()
            .find(|(_, p)| Some(p.id) == self.ground)
        else {
            // SlimeController.UpdateSim handles a destroyed retained ground
            // before calling Move: it zeros rotation, then UnGround(false).
            self.set_rotation(Fix::ZERO);
            self.unground(None, false);
            return;
        };
        let normal = platform.surface.normal(self.position);
        let forward = normal.perp_cw();
        let sign = self.ground_input_sign(input, normal);
        if sign == Fix::ZERO {
            self.grounded_speed *= if platform.ice {
                self.constants.ice_friction
            } else {
                self.constants.friction
            };
        } else {
            self.grounded_speed = crate::player_controller::increment(
                self.grounded_speed,
                sign * self.constants.max_speed,
                self.constants.accel,
                self.fixed_delta_time,
            );
        }
        if !platform.active {
            self.unground(Some(&platform.surface), true);
            return;
        }
        // The position has not moved since `normal` was evaluated above.
        if self.sync_rotation_to_platform {
            self.align_to(normal);
        }
        let speed = (self.grounded_speed + self.external_velocity.dot(forward)) * speed_scale;
        self.move_along_surface(platform_index, forward, speed, walls);
    }
    /// PlayerPhysics.MoveAlongCurrentPlatForm (PlayerPhysics.cs:323-328): the
    /// body slides along its ground by only the external velocity's component
    /// along the surface, and that external velocity is spent.
    pub(crate) fn move_along_current_platform(&mut self, walls: &mut impl Walls) {
        let Some(platform_index) = walls.platforms().iter().position(|p| Some(p.id) == self.ground) else {
            return;
        };
        let forward = walls.platforms()[platform_index].surface.normal(self.position).perp_cw();
        let speed = self.external_velocity.dot(forward);
        self.external_velocity = Vec2::ZERO;
        self.move_along_surface(platform_index, forward, speed, walls);
    }
    /// StickyRoundedRectangle.moveAlongSurface: the surface local advances by
    /// `speed`, cut short by the nearest other platform ahead.
    fn move_along_surface(&mut self, platform_index: usize, forward: Vec2, speed: Fix, walls: &mut impl Walls) {
        let velocity = forward * speed;
        let dt = self.fixed_delta_time;
        // maxMovementAllowed disables the own rounded rect around its three
        // raycasts (StickyRoundedRectangle.cs:441-468).
        let allowed = crate::platform_motion::movement_limit(
            self.position,
            self.radius,
            velocity,
            dt,
            &mut |origin, direction, distance| {
                walls
                    .raycast_to_closest_rounded_rect(origin, direction, distance, Some(platform_index))
                    .map(|(_, near, _)| ((), near))
            },
        )
        .0;
        let allowed = allowed.min(speed.abs()) * Fix::from_int(speed.sign());
        self.local = walls.platforms()[platform_index]
            .surface
            .move_unblocked(self.local, self.radius, allowed, dt);
    }
    /// PlayerPhysics.UpdateSim's gravity and landing raycasts
    /// (PlayerPhysics.cs:116-131). Returns whether a ray attached the body,
    /// which raises OnAttachedToGround.
    pub(crate) fn gravity_step(&mut self, walls: &mut impl Walls) -> bool {
        if std::mem::take(&mut self.skip_next_gravity) || self.grounded {
            return false;
        }
        crate::player_physics::apply_gravity(
            &mut self.velocity,
            self.constants.gravity_accel,
            self.constants.gravity_modifier,
            self.constants.gravity_max_fall_speed,
            self.space,
        )
        .expect("valid movement gravity constants");
        self.collision_rays(walls)
    }
    pub(crate) fn step(&mut self, action: Action, walls: &mut impl Walls) {
        self.step_with_input(action, action.direction(), walls);
    }

    /// Run the ordinary slime movement with `input` while keeping `action`'s
    /// joystick direction available to abilities such as Dash. Gamepad input
    /// uses the same vector for both; keyboard players use W/A/S/D for motion
    /// and the joystick byte independently for ability aim.
    pub(crate) fn step_with_input(&mut self, action: Action, input: Vec2, walls: &mut impl Walls) {
        let dash_inverse_extents = Some(self.authored_inverse_extents_of(true));
        if self.step_unintegrated(action, input, walls, dash_inverse_extents) {
            self.integrate();
        }
    }
    /// `step_with_input` short of PlayerBody's integration. A Dash it enters
    /// takes `dash_inverse_extents` for its box's inverseExtents.
    pub(crate) fn step_unintegrated(
        &mut self,
        action: Action,
        input: Vec2,
        walls: &mut impl Walls,
        dash_inverse_extents: Option<Vec2>,
    ) -> bool {
        self.begin_pass();
        if !self.dash.active {
            self.gravity_step(walls);
            self.dash.radius = self.radius;
        }
        self.capture_pre_move(walls.platforms());
        self.move_unintegrated(action, input, walls, dash_inverse_extents)
    }
    /// Start of this body's updater pass, before PlayerPhysics.UpdateSim.
    /// `attach_upside_down` records whichever of this pass's two raycast
    /// phases fires attachToGround, so it has to be cleared ahead of both.
    pub(crate) fn begin_pass(&mut self) {
        self.attach_upside_down = None;
    }
    /// The boundary between PlayerPhysics.UpdateSim and SlimeController's
    /// OldUpdate. Everything OldUpdate reads about the ground is sampled here,
    /// because Move does not run until SlimeController.cs:1084 - long after
    /// OldUpdate has finished with it.
    ///
    /// InputGroundAlignment and `fix2` take the attached ground's normal at
    /// the body (StickyRoundedRectangle.currentNormal), not the slime's own
    /// `up`, which lags a platform that tilted under it since the last align.
    pub(crate) fn capture_pre_move(&mut self, platforms: &[Platform]) {
        self.pre_move_up = self
            .grounded
            .then(|| platforms.iter().find(|p| Some(p.id) == self.ground))
            .flatten()
            .map_or(self.up(), |p| p.surface.normal(self.position));
        self.pre_move_grounded = self.grounded;
    }
    /// SlimeController.UpdateSim after PlayerPhysics.UpdateSim has applied
    /// gravity: dash entry/continuation, jump, ground or air movement, then
    /// PlayerBody's integration.
    pub(crate) fn move_step(
        &mut self,
        action: Action,
        input: Vec2,
        walls: &mut impl Walls,
        dash_inverse_extents: Option<Vec2>,
    ) {
        if self.move_unintegrated(action, input, walls, dash_inverse_extents) {
            self.integrate();
        }
    }
    /// `move_step` short of PlayerBody's integration. False when Dash took
    /// the pass or the body left the level, and there is nothing to integrate.
    /// Only a caller that lets the dash button through can enter a Dash, and
    /// it passes that box's inverseExtents.
    pub(crate) fn move_unintegrated(
        &mut self,
        action: Action,
        input: Vec2,
        walls: &mut impl Walls,
        dash_inverse_extents: Option<Vec2>,
    ) -> bool {
        let jump = action.buttons & 1 != 0;
        let cast = self.instant_cast_this_pass;
        if self.dash.active {
            self.previous_jump = jump;
            self.dash_step(action, walls);
            return false;
        }
        self.cooldown += DT;
        if action.buttons & 2 != 0 && self.cooldown > Fix::from_int(4) {
            let dash_inverse_extents = dash_inverse_extents.expect("a Dash entry with its box's inverseExtents");
            self.enter_dash(action, input, walls, dash_inverse_extents);
            self.previous_jump = jump;
            self.dash_step(action, walls);
            return false;
        }
        if jump && !self.previous_jump && self.grounded && !cast {
            self.jump(walls.platforms());
        }
        self.previous_jump = jump;
        self.move_only(input, walls);
        if self.outside_bounds() {
            self.alive = false;
            self.out_of_bounds = true;
            return false;
        }
        true
    }
    /// PlayerBody.LateUpdateSim without a rope.
    pub(crate) fn integrate(&mut self) {
        self.position += (self.velocity + self.external_velocity) * self.fixed_delta_time;
        self.external_velocity *= Fix::from_f64(0.95);
    }
    /// PlayerPhysics.Move (PlayerPhysics.cs:230-260) without PlayerBody's
    /// integration. SlimeController.UpdateSim still calls it on the tick
    /// OldUpdate entered an ability (SlimeController.cs:1067-1084), on the
    /// slime body the ability has already copied; that body is deactivated,
    /// so only its observed velocity (and a landing) change.
    pub(crate) fn move_only(&mut self, input: Vec2, walls: &mut impl Walls) {
        if self.grounded {
            self.move_grounded(input, walls);
        } else {
            crate::player_physics::air_movement(
                &mut self.velocity,
                input.x,
                self.constants.air_accel,
                self.constants.max_speed,
            )
            .expect("valid movement air constants");
            self.collision_rays(walls);
        }
    }
    /// PlayerPhysics.MachoMove: normal player acceleration with a reduced
    /// distance around a rounded platform corner.
    pub(crate) fn macho_move(&mut self, input: Vec2, roundedness: Fix, walls: &mut impl Walls) {
        if self.grounded {
            self.move_grounded_scaled(input, roundedness, walls);
        } else {
            crate::player_physics::air_movement(
                &mut self.velocity,
                input.x,
                self.constants.air_accel,
                self.constants.max_speed,
            )
            .expect("valid MachoThrow air constants");
            self.collision_rays(walls);
        }
    }
    pub(crate) fn outside_bounds(&self) -> bool {
        self.position.y
            < if self.space {
                // DestroyIfOutsideSceneBounds uses Camera_YMin in space;
                // SceneBounds.WaterHeight (-50) is only the off-screen water
                // plane used by other component-specific queries.
                Fix::from_int(-26)
            } else {
                Fix::from_f32(-11.3)
            }
            || self.position.y > Fix::from_f64(72.5)
            || self.position.x < Fix::from_f64(-131.25)
            || self.position.x > Fix::from_f64(131.25)
    }
    pub(crate) fn push_from(&mut self, other: Vec2, lower_id: bool) {
        let Some(direction) = crate::player_collision::overlap_push_direction(
            self.position,
            self.up(),
            self.radius,
            other,
            if lower_id { -1 } else { 0 },
        ) else {
            return;
        };
        self.external_velocity += direction * Fix::from_raw(21_474_836) * Fix::from_int(35);
    }
    /// A DetPhysics sync of the hurtbox to the body.
    pub(crate) fn sync_collider(&mut self, inverse_extents: Vec2) {
        self.move_collider(inverse_extents);
        self.synced = self.collider;
    }
    /// The standalone slice's hurtbox inverseExtents: the authored box, the
    /// Dash body's while it dashes, at the body's scale.
    pub(crate) fn authored_inverse_extents(&self) -> Vec2 {
        self.authored_inverse_extents_of(self.dash.active)
    }
    fn authored_inverse_extents_of(&self, dash: bool) -> Vec2 {
        let extents = if dash {
            Vec2::new(Fix::from_raw(5_626_406_912), Fix::from_raw(4_359_391_744))
        } else {
            Vec2::new(Fix::from_raw(5_712_306_688), Fix::from_raw(3_478_923_520))
        } * self.scale;
        Vec2::new(Fix::ONE / extents.x, Fix::ONE / extents.y)
    }
    /// Moves the exported box onto the body without DetPhysics seeing it
    /// until its next sync. `inv` is the inverseExtents of the box the body
    /// currently has: the slime hurtbox's, or the Dash body's own while it
    /// dashes (Dash.cs:70-72).
    pub(crate) fn move_collider(&mut self, inv: Vec2) {
        if self.rotation() != self.collider_rotation || self.collider_basis == Vec2::ZERO {
            self.collider_basis = Vec2::angle(self.rotation());
            self.collider_rotation = self.rotation();
        }
        self.collider_up = self.up();
        let basis = self.collider_basis;
        self.collider = OrientedBox {
            center: self.position,
            right: (Vec2::RIGHT / inv.x).complex_mul(basis),
            up: (Vec2::UP / inv.y).complex_mul(basis),
            inverse_extents: inv,
        };
    }
    /// PlayerPhysics.UnGround's body half (PlayerPhysics.cs:362-376): lift off
    /// `surface`, the retained attachedGround the caller resolved, and keep
    /// `ground`. Its OnNoLongerGrounded subscribers are the caller's; returns
    /// the rotation they are handed.
    pub(crate) fn unground(&mut self, surface: Option<&Surface>, null_rotation: bool) -> Fix {
        let mut motion = self.ground_state();
        let rotation = motion.unground(surface, null_rotation);
        self.apply_ground_state(motion);
        rotation
    }
    pub(crate) fn late_step(&mut self, platforms: &[Platform]) {
        // A grounded body whose attachedGround is null or destroyed stays put;
        // a deactivated one is still followed (PlayerPhysics.LateUpdateSim).
        if self.alive
            && self.grounded
            && !self.dash.active
            && let Some(platform) = self.attached_ground(platforms)
        {
            self.position = platform.surface.position(self.local, self.radius);
            if self.sync_rotation_to_platform {
                self.align(platform);
            }
        }
    }
}
