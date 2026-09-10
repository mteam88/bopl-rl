# Rust rollouts

The Rust simulator (`native/src/rollout`, crate `bopl-sim`) implements the
full `bopl_rl.h` ABI, so the trainer can roll out on Rust instead of the
NativeAOT .NET sim. The goal is transferable fidelity, not byte-exactness: a
policy trained on one backend performs the same when evaluated on the other.
The .NET sim stays the reference for evaluation and replay.

## How to run

```sh
cd native && cargo build --release   # produces target/release/libbopl_sim.so
BOPL_RL_BACKEND=rust native/artifacts/rl/venv/bin/python native/rl/puffer/train.py ...
```

`train.py` resolves the .so itself, sets `BOPL_RL_INPROC_SPARE=0` (Rust
resets are ~1 ms, no spare world needed) and hosts every world in one
runtime. Any tool that takes a `--game`/library path (`evaluate.py`, the
fidelity gate, `benchmark.c`) accepts `native/target/release/libbopl_sim.so`
directly in place of `BoplSimNative.so` — the in-process loader in
`bopl_rl.c` sees the same 17 exported symbols.

## What is implemented

- **ABI**: every `bopl_rl_*` entry point including `step_sampled` (bank
  opponents), `reset_begin`/`reset_finish`, telemetry, and BOPLREP2 replay
  export (`bopl_rl_replay`, byte-compatible with `ReplayArchive`, SHA-256
  trailer), so `evaluate.py`'s per-group replay saves work unchanged.
- **Maps 0-4**: geometry, spawn points, prefab constants and animation
  timings are extracted from the reference export by
  `native/tools/extract_rollout_data.py` into `native/data/rollout.json`
  (embedded via `include_str!`). Every authored and spawned platform has its
  extracted `BoplBody`, an `AnimateVelocity` home spring, and dynamic RR-RR
  collision impulse/positional correction. Map 2's AntiLockPlatform uses the
  same body path and matches the reference waypoint trajectory exactly.
- **Episode lifecycle**: 240-tick warmup simulated for real (players fall
  from spawn and land), deaths (environment/arrow/self), stocks with
  respawn holds, first-death terminal at stocks=0, the 7200-tick timeout,
  and sudden-death platform sinking on the extracted curve. Blast-boundary
  deaths are always routed through match bookkeeping, including while an
  ability body is active.
- **Movement**: the verified movement slice (`world.rs`). Registered input is
  consumed by the live body in the current tick, as in Unity. The public
  `Player.Position` is normally the position published by `SlimeController`
  before the later movement/physics work in that pass; the rollout represents
  that report phase separately from the live body.
- **Dash, Bow, Platform** (catalog ids 2/4/14): ported from `sim/Gameplay`
  with measured timings. Bow: transform 30 ticks, loadingFrame =
  passedSprites (1 at load start, then +8/+7/+7), slow-mo aiming physics,
  fire-on-release (early release fires from transform sprite index 2, tick
  11), knockback recoil, arrow gravity -40 (the DetPhysics override, not
  Unity's -9.82), policy-visible animator charge, delayed hurt-owner, the
  20-arrow owner cap, projectile collision during a dash raycast, and
  post-integration discrete circle contacts in DetPhysics group order.
  Terrain consumes an arrow; PlayerBody-backed hurtboxes kill without
  consuming it, matching `Arrow.OnCollide`.
  Platform: anticipation, aim rotation,
  cursor sizing under the area cap, geometry-dependent mass/inertia,
  the authored translational and rotational home springs, ejection and
  same-pass exit reattachment, one-platform-per-player, curve-driven old-body
  removal, transform following, and squish deaths. Resizing preserves the
  scalar body orientation exactly, as `ResizablePlatform.ResizePlatform` does.
- **Observation**: `bopl_layout.h` layout with the .NET entity id
  assignment order (circles, boxes, rounded rects by first-seen), matching
  kinds/layers/owner/flags, slime ability slots, cooldowns and timers
  (spawn timing -1.5 s at warmup tick 62 reproduced), and the
  SlimeController state machine (fall/land 36t/idle/run/stop 6t/turn 12t,
  tangential ground-alignment deadzones). Rewards and telemetry hook up
  through the same header/flag bits the puffer env reads.

## Deliberate .NET quirks mirrored

- `Player.Position` normally reports the body position from before movement
  in the same updater pass, while the slime/entity rows contain the live body.
  Ability callbacks have specific exceptions: Dash publishes after its
  transform, Platform exit publishes the final ejection, and stock selection
  publishes the selected respawn point in that same pass.
- The .NET `State` enum never takes the `ability` value (nothing assigns
  it): state freezes at its pre-ability value during an ability and
  resumes from `fall` after `ExitAbility` ungrounds. Rust does the same.
- Ability cooldown timers start at -1.5 s when slimes wake at warmup tick
  62, so nothing is ready at the first decision (dash ready at 2.53 s in).

## Known approximations (documented divergence)

- Fixed-loadout RL sessions disable `AbilitySpawner` on both backends. Rust
  intentionally has no spawner path: the extractor retains each map's
  `ability_spawner` position in `rollout.json`, but `rollout/mod.rs` never
  reads it. The .NET Host and NativeAOT RL boundaries disable the authored
  component by default; `BOPL_RL_ABILITY_SPAWNS=1` explicitly opts only the
  oracle back into full-game pickup spawning for diagnostics. This is an RL
  configuration choice, not an open default-backend divergence.
- Only ids 2/4/14 have real behavior. Other catalog ids tick a plausible
  cooldown and never activate. Closing this scope requires the remaining
  catalog abilities and their dynamic bodies/collision pairs.
- Landed arrows are freed from the native entity table instead of remaining
  as attached visual children. The gameplay-relevant projectile collider is
  destroyed in both backends and the 20-arrow live-projectile cap is enforced,
  so the remaining difference is primarily observation/entity history.
- On final elimination, Unity's `SlimeController.DropAbilities` can create a
  pickup in the terminal snapshot. Rust has no corresponding terminal-only
  entity. No later gameplay tick can consume it and reset reseeds the RNG, so
  this remains an observation-only difference rather than the disabled
  periodic-spawner gameplay gap.
- The rollout only contains dynamic bodies needed by ids 2/4/14. Forces from
  unsupported abilities (for example Machoman impulses into a platform) and
  their circle/box/rope collision pairs remain absent.

## Gameplay fidelity audit (2026-09-10)

| Severity | Gap | Result |
|---|---|---|
| exploitable | Spawned/map platforms were immutable surfaces, allowing a transformed player to phase a platform through terrain | fixed: extracted bodies, springs, RR-RR contacts, impulses/correction, riders and transform following |
| exploitable | A movement body crossing a blast boundary was pre-marked dead, then skipped by match death bookkeeping | fixed: the marker is consumed once and crossing on the current tick is caught |
| exploitable | Dash raycasts ignored arrows even though the Unity collision mask includes Projectile | fixed: ordered arrow hits before the first wall now destroy the arrow and kill when owner rules allow |
| behavior-changing | Resizing kept the prefab mass/inertia instead of `10 + MassPerArea * area` and `ApproximateMOI` | fixed, including live observation `inverse_mass` |
| exploitable | Spawned platforms had no rotation spring because the extractor omitted `rotationSpeed`/`rotationMu`; terrain contact torque accumulated into a large persistent tilt | fixed: both fields are required extracted data, the authored 40 / 1/6 spring runs every tick, and resize no longer reconstructs or pins body rotation |
| behavior-changing | Platform aim used scalar-angle interpolation instead of Unity's sine-weighted `Vec2.Slerp`, leaving a fixed home-angle offset | fixed with the source formula, including its dot ceiling and zero-sine fallback |
| behavior-changing | Bow parsed but ignored `maxNumberOfArrows = 20` | fixed with per-owner oldest-projectile eviction |
| behavior-changing | Ability entry/spawn and Dash/Bow/Platform updater phases were delayed or collapsed into the public player position | fixed: current input drives live bodies immediately while Unity's component-specific report phases remain explicit |
| behavior-changing | Bow used the original slime hurtbox during transforms, lost ordinary-slime pushes, raycasted after aerial control, and exposed charge one animator pass early | fixed with active ability colliders and source updater/animator ordering |
| behavior-changing | Respawn point/timing choice was absent and fast-spawn upward motion lagged one pass | fixed: three directional points, early jump confirm, exact selection publication and launch phase |
| behavior-changing | Curve tangents and old-platform shrink removal were absent | fixed: exported Hermite tangents/exact evaluator and collidable `being_removed` bodies |
| behavior-changing | Rust arrow flight was swept while Unity integrates first and dispatches discrete Circle/RR, Circle/Circle, then Circle/Box contacts | fixed; terrain callbacks consume the arrow before player callbacks, while a PlayerBody kill leaves it flying |
| behavior-changing | Cosmetic level-intro ground delays consumed RNG only in .NET, shifting later sudden-death samples | fixed: Rust consumes the samples without rendering the intro; sink activation and trajectories now align |
| behavior-changing | Arrow rotation used post-gravity velocity and its boundary cull ran after physics | fixed: policy-visible rotation and side/water lifetime use the authored pre-physics phases |
| exploitable | Rust deleted a spawned platform when its transformed owner crossed a blast boundary | fixed: the owner dies, but the separate `ResizablePlatform` remains collidable as in Unity |
| behavior-changing | An end-of-tick Rust boundary pass killed ordinary/Bow bodies one tick before Unity's pre-`PlayerBody` boundary component | fixed with component-specific boundary phases |
| configured scope | Ability pickups | disabled on both RL backends by default; explicit .NET-only diagnostic opt-in above |
| exploitable | Abilities other than 2/4/14 are absent | open scope described above |
| cosmetic/observation | Stuck arrow children are not retained | not changed; no live collision remains in Unity either |
| cosmetic/observation | Unity can emit a death-dropped pickup in the terminal snapshot | not changed; the episode cannot advance and reset reseeds RNG |

## Number type: f64 behind the Q32.32 API

Unity's `BoplFixedMath/Fix.cs` is Q32.32 fixed point (i64 raw, i128
products, a 205887-entry sine table, an Euler-series `Atan`) because the game
needs lockstep determinism across machines. The rollout only needs
determinism within one binary (replays re-simulate recorded actions and A/B
traces compare two runs on one box), which IEEE doubles already give. The
profile of the live trainer put ~35% of sim time in `Fix::checked_div`,
`Fix::sin`/`fast_sin`/`atan`/`acos`, so `native/src/fixed.rs` now stores an
`f64` behind the identical `Fix` API; game logic is untouched. The cargo
feature `q32` (`native/src/fixed_q32.rs`) restores the bit-exact reference
arithmetic for the golden verification tooling (`just verify` builds with it
and rebuilds the default afterwards). Raw Q32.32 values only exist at the
boundaries: `Fix::from_raw`/`Fix::raw` convert for the `bopl_rl.h` structs
(`bopl_puffer.h` still divides by 2^32), the embedded `rollout.json`, the
level fixtures and the `bopl.h` state dump. The ABI version and the
observation layout are unchanged.

Decisions, each checked against `Fix.cs`:

- **Constants stay the game's truncated values**: `PI` = 13493037704/2^32,
  `HALF_PI`, `TAU` (= 2·PI + 1 raw), `DT` = 71582788/2^32 (< 1/60). Every
  raw value with < 2^53 magnitude is exact in f64, so every comparison
  against an authored constant lands on the same side in both builds.
- **`from_f32`/`from_f64` snap to the Q32.32 grid** exactly as the
  reference's `(Fix)0.2f` / `(Fix)0.95m` casts do (scale, truncate toward
  zero), so thresholds such as the 0.2/0.01 ground-alignment deadzones, the
  0.4 stick deadzone, -11.3 water level and 0.95 external-velocity decay are
  the same numbers, not their decimal ideals.
- **Timers are exact.** `DT` is a dyadic rational with a 27-bit numerator, so
  `timer += DT` (k·71582788/2^32 has < 2^53 bits for k < 1.2e8) and
  `Fix::from_int(tick) * DT` are exact in f64 and equal the fixed-point sums
  bit for bit: ability cooldown readiness (`timers >= cooldowns`), the -1.5 s
  warmup start, the 4 s dash gate, `dash.time == DT`, `dash.time < 0.01`,
  the 60·DT < 1 s boundary and the sudden-death time all flip on the same
  tick. Unit tests in `fixed.rs` pin this.
- **Saturation/overflow**: the game relies on none of it in the rollout's
  code paths. `slow_*` become plain arithmetic. The only saturation-dependent
  branch, `Atan2`'s `1 + 0.28·r² == MaxValue` fallback to `±HALF_PI`, is kept
  as a `>= MAX` test so near-vertical inputs return the same value.
- **Mul/Div rounding**: Q32.32 `*` floors the 2^-32 grid, `/` rounds to
  nearest; f64 rounds both to the nearest double. This is the source of every
  observed divergence (see below) and is accepted: the reference's own result
  is off the real number by up to one raw unit per operation in the same
  places.
- **Trig**: `sin`/`cos` keep the reference's `ClampSinValue` quadrant
  reduction (modulo the game's `TAU`, fold at `PI` and `HALF_PI`), then call
  libm, so `sin(0)=0`, `sin(HALF_PI)=1`, `sin(PI)=0`, `cos(0)=1`,
  `cos(PI)=-1`, `sin(TAU)=0`, `cos(TAU)=1` exactly; axis-aligned platforms
  and flat-ground rotations (`TAU` from `normalized_angle`) produce exact
  unit bases as in the table lookup. `Cos(HALF_PI)` is -2^-32 (the game's
  `TAU` is one raw unit above 2·PI, and the reference's interpolated `Sin`
  rounds that same value down to 0). `fast_sin`/`fast_cos` (Unity's
  un-interpolated lookup, index = angle >> 15, i.e. the angle truncated to
  7.6e-6 rad) are hardware sin/cos: the game does not depend on the
  truncation error (it only biases `Vec2(angle)` bases by < 7.6e-6 rad; no
  comparison reads those bases against an exact constant, and orientation is
  integrated separately from the vector so nothing drifts).
  `atan`/`acos`/`tan` are libm; `acos` accepts inputs within one raw unit of
  ±1 (a normalized double's component cannot exceed 1, and the Q32.32
  normalization has the same one-unit slack) and rejects the rest as Domain
  like the reference.
- **`Atan2` keeps the reference formula**: `r/(1 + 0.28 r²)` for `|r| < 1`
  and the non-standard `(HALF_PI - r)/(r² + 0.28)` otherwise
  (`Fix.cs:41975`). `Tools.Angle2` feeds it `tan(θ/2)` for the corner arcs
  of `StickyRoundedRectangle`, so the game's perimeter coordinate genuinely
  uses this approximation (max error 0.005 rad); a true atan2 would move
  every corner attachment systematically.
- **`Round`** is half-to-even in the reference and `round_ties_even` here;
  `Floor`/`to_int` floor toward -inf like the arithmetic shift; `%` keeps the
  dividend's sign like C#.

Verification of the representation change (`native/rl/research/rust_ab/repr_ab.c`
drives the Q32 and f64 builds with the fidelity gate's action stream at
action repeat 1 and compares every observation field per tick):

- With warmup 0 and tolerance 0, the first divergence is at tick 1:
  `player0.y` differs by exactly one raw unit (2^-32), the truncation of
  `velocity.y * DT` in `Player::step` versus the correctly rounded product.
  With the 240-tick warmup, the reset state differs on platform and body
  basis vectors by 5-12335 raw units (1e-9 to 2.9e-6 units): the `Vec2::angle`
  table truncation on maps with tilted platforms (3, 4), and one-unit
  products elsewhere. No discrete field (state, flags, ability flags,
  timers, telemetry, entity presence) differs at the first divergence.
- Over 300 episodes (seeds 1000-1299, maps 0-4) the two builds produce the
  same winner in 300/300 and the same episode length in 298/300; the two
  remaining episodes end one tick apart after the position gap grew from
  1e-8 through chaotic platform-riding dynamics (×1.05 per tick during a
  Platform transform) to 2e-2 units before the final water crossing, with
  every discrete field agreeing along the way (`REPR_AB_TRACE=34`).
- The 300-episode gate against .NET is identical to the Q32 build's, column
  for column (67.1 mean decisions, 278/300 winner agreement, same death and
  activation counts, before the platform-physics merges; 62.5 vs 63.0 and
  296/300 after them, still identical to the Q32 build), and every scripted
  A/B trace keeps its tolerance:
  `move`/`state`/`bow`/`aim`/`hit` are textually identical,
  `platform_ab` overlap maxima are unchanged (0.545/0.043/0.183 units),
  `stock_respawn_ab` differs from the Q32 build by at most 5.6e-9 units, and
  the map-2 orbiter (previously identical to 12 places) now drifts from .NET
  by 3e-10 units per tick — one raw unit per tick from the reference's
  truncated `AnimateVelocity` impulse products — reaching 2.6e-7 units at
  tick 1620.
- After the platform-physics merges (final 68c6aa0) the same comparison
  holds: `repr_ab` at tolerance 0 gives 300/300 winners and 300/300 lengths
  between the `q32` and f64 builds of one tree, and the physics tip built as
  committed is bit-identical to the merged `q32` build. Against .NET,
  `sudden_death_ab` (RNG-aligned sink trajectories) has a maximum Y error of
  3.3e-9 units on the Q32 build and 6.1e-9 on f64 — the extra 12 raw units
  over the 90 sink ticks are the per-product rounding difference above.
  `platform_boundary_ab` originally printed the same death tick (461), RR
  counts and phases, with minimum Y 7.6e-5 units apart. The cause was a
  Rust-only algorithmic mistake rather than an inherent f64 difference:
  every size tick reconstructed `body.orientation` by the round trip
  `orientation -> Vec2::angle -> normalized_angle`. Near pi the Q32 sine
  table walks one 2^15-raw cell (7.6e-6 rad) per application, while libm f64
  moved one raw unit. Unity's `PlatformTransform.State.Size` retains the
  transformed `PlayerBody.right`, and `ResizablePlatform.ResizePlatform`
  changes geometry, mass and inertia without writing rotation. Both Rust
  number modes now do the same, so resizing and removal never feed collider
  axes back into scalar body orientation. The remaining small f64/Q32 formed
  orientation offset came from a simplified scalar-angle `slerp`, not an
  accumulating resize drift. The rollout now uses Unity's sine-weighted vector
  formula; in the near-pi paired trace the final f64/Q32 home angles differ by
  9.1e-7 rad and the settled f64/.NET no-contact difference is 3.0e-5 rad.

## Raycast broad phase

`world::closest` (the three collision rays per player per tick from
`collision_cast`/`movement_limit`, the dash and arrow sweeps) tests each
platform's axis-aligned bounds (`OrientedBox::bounds(radius)`: centre +-
(|right| + |up| + radius) per axis) against the query segment before the
rounded-rectangle narrow phase; `raycast_within(origin, direction, distance)`
does the same for the player boxes hit by dashes and arrows. The test is
conservative: every accepted hit has `near >= 0` and puts
`origin + direction * near` on the shape's boundary or at an origin inside
the shape, hence inside the bounds and on the segment; a 1e-6 unit margin
covers evaluation rounding. The single case it does not preserve is a box
ray from outside whose entry is exactly a corner (the reference's strict
inequalities drop that edge and still report `near = 0`), an exact-equality
tie. The edge test itself (`geometry::edge_hits`, Raycast.cs' side loop:
side order, strict inequalities, stop after two) writes into a two-slot
array instead of a `Vec`. Both Rust builds (Q32 and f64) are bit-identical
before and after over 300 gate episodes at tolerance 0 (`repr_ab`).

## Fidelity gate

`native/rl/fidelity_gate.c` drives both backends with identical seeded
sticky action streams (2 players, abilities 2/4/14, maps 0-4, stocks 0)
and compares short-horizon divergence plus outcome distributions.
The A/B loader must keep pools library-specific. Before commit `1cd34b2`, a
second in-process backend could accidentally reuse the first library's runtime
and turn an intended .NET-vs-Rust check into .NET-vs-.NET.

At 300 episodes with that harness fixed (seeds 1000-1299), the base
(`7f6761b`), platform/phase implementation, discrete-arrow implementation,
and the final semantic-audit implementation were:

| | .NET reference | Rust base | Rust platform/phase | Rust discrete arrows | Rust pre-rotation fix | Rust current |
|---|---:|---:|---:|---:|---:|---:|
| mean length (decisions) | 63.0 | 73.4 | 66.5 | 62.4 | 62.5 | 62.3 |
| timeouts | 0 | 0 | 0 | 0 | 0 | 0 |
| environment deaths | 133 | 144 | 153 | 138 | 138 | 135 |
| self deaths | 165 | 153 | 146 | 160 | 160 | 163 |
| opponent deaths | 2 | 3 | 1 | 2 | 2 | 2 |
| dash activations | 221 | 284 | 240 | 217 | 217 | 219 |
| bow activations | 498 | 560 | 510 | 489 | 491 | 492 |
| platform activations | 86 | 129 | 113 | 87 | 88 | 87 |
| arrows fired | 484 | 551 | 497 | 476 | 476 | 479 |
| arrow hits | 2 | 3 | 1 | 2 | 2 | 2 |
| winner agreement | - | 267/300 | 284/300 | 296/300 | 296/300 | 297/300 |

The episode-length gap fell from +10.4 to -0.7 decisions. Mean position
divergence remains under 0.002 through decision 16, then grows chaotically
(0.083/0.199 at decisions 31/46). Restoring the spawned-platform rotation
spring reduced the later decision-106/121 divergences from 1.436/2.690 to
1.238/1.580 units and improved winner agreement by one episode.

Scripted A/B traces (`native/rl/research/rust_ab/*.c`, built against
`bopl_rl.h` the same way as `fidelity_gate.c`) cover movement states, bow
flag/timer and charge launch-speed sequences, active ability hitboxes/pushes,
stock choice, plus platform overlap, recast, landing and map-2 orbiting. Bow
charge has zero flag-mismatch ticks over the measured load sequence. Stock
respawn position/state/flags/timers are identical after the one-tick death
boundary. In `platform_ab`, the orbiter position is identical to 12 decimal
places at every sampled tick with the Q32.32 `Fix` (with the f64 `Fix` it
drifts one raw unit per tick, 2.3e-7 units at tick 1620; see "Number type")
and the landing player settles within 0.0001 units. For the overlap scenario, the maximum map/spawned-platform deltas fell
from 1.385/1.743 to 0.047/0.041 units; their final deltas are 0.046/0.039. The
player's maximum/final delta is 0.033/0.032 after matching Platform's
synchronous exit callback (previous maximum: 0.788). In `ability_hitbox_ab`,
the swept implementation missed an arrow/Bow collision that .NET delivered at
tick 144 and incorrectly removed its arrow (`144/2` versus no death, arrow
counts `1/0`). Discrete contacts match the death exactly at `144/2` and both
backends retain one live arrow (`1/1`).

`platform_rotation_ab` isolates the rotation repair. In an airborne short-size
case with no RR contact, f64 remains exactly at 3.139446572168 rad from tick
329 through 700; Q32 behaves equivalently, while .NET's initial scheduling
kick damps to 5.6e-9 rad/s. In an intentionally tall platform that bears
against terrain, the old f64 build reached 1.398 rad maximum displacement and
1.396 rad/s maximum angular speed over ticks 329-700. The repaired build
reaches 0.0205 rad and 0.0349 rad/s, close to .NET's contact-loaded 0.0110 rad
and 0.0306 rad/s. Its maximum/final orientation error against .NET is
0.0115/0.0092 rad, down from 1.3954/0.8556. The overnight-2 replay's platform
18 changes from a visible 1.5830 -> 1.0453 rad rotation by age 190 to 1.5830
-> 1.5768 rad under the
same continuing terrain load; angular speed falls from -0.3406 to -0.0048
rad/s. The .NET replay lands that platform elsewhere, with no continuing
contact, and settles at 1.5831333 rad and zero angular speed by age 170.

`ability_spawner_ab` observes no layer-13 pickup on either backend under the
default RL configuration and one only in .NET with
`BOPL_RL_ABILITY_SPAWNS=1`. With spawns enabled, `sudden_death_ab` reached
1.427 units maximum Y error. Disabling them alone still left 1.584 units from
the skipped intro RNG draws; after preserving that RNG phase, all three map-0
platforms begin moving on the same ticks (7236/7252/7257), maximum Y error is
3.259e-9, and final error is 4.650e-10. `arrow_bounds_ab` matches the side-exit
lifetime at tick 414 (Rust was one tick early), `platform_boundary_ab` matches
owner death at tick 461 while retaining 7 active rounded bodies, and the stock
water-death trace now matches tick 198 (Rust was one tick early).

## Transfer test

The 1B-step .NET-trained checkpoint (`long-run-1b/007-s73-1b-free-rows`,
policy-705335296) evaluated with `evaluate.py` on both backends, 24 games
per opponent/map plus 8 per held-out map:

| | .NET | Rust |
|---|---|---|
| total wins | 154/160 | 153/160 |
| held-out maps 3/4 | 16/16 | 15/16 |

Every group is within its Wilson interval of the .NET result. Residual
behavioral gap: Rust episodes average ~890 ticks vs ~640 on .NET, and the
policy fires ~3.6 arrows/min vs ~2.3 (it takes longer to close out the
random opponent). Win rates and death-cause mixes agree.

### Rust-trained checkpoint across sim builds (2026-09-10)

`overnight-1/012` final policy (policy-266698752, trained on the Sep 9 Rust
sim before the platform-physics work) evaluated with `evaluate.py` as
`train.py` invokes it (`--opponents anchors/opponents.json`,
`--scripted-opponents mixed-hard,archer-hard,pursuer-hard,evasive-hard,
mixed-medium`, 8 games per opponent and map, 4 per held-out map, 16 per
frozen anchor, 8 workers) on four backends: the sweep's own `.so`, the
platform-physics tip 68c6aa0 as committed (Q32.32 `Fix`), this tree (f64 `Fix`,
raycast broad phase, physics merged) and the matching .NET oracle
(`BoplSimNative.so` sha256 f1e6b7a8..., ability pickups off by default).

| opponent | sweep .so | physics tip (Q32) | this tree (f64) | .NET |
|---|---|---|---|---|
| archer-hard | 20/32 | 16/32 | 15/32 | 16/32 |
| evasive-hard | 22/32 | 6/32 | 8/32 | 5/32 |
| mixed-hard | 27/32 | 17/32 | 17/32 | 19/32 |
| mixed-medium | 22/32 | 17/32 | 18/32 | 19/32 |
| pursuer-hard | 17/32 | 9/32 | 9/32 | 10/32 |
| random | 30/32 | 31/32 | 31/32 | 32/32 |
| initial | 24/24 | 23/24 | 23/24 | 22/24 |
| run007-100m | 38/48 | 33/48 | 34/48 | 27/48 |
| run007-300m | 39/48 | 33/48 | 36/48 | 30/48 |
| run007-final | 23/48 | 18/48 | 16/48 | 12/48 |
| total | 262/360 | 203/360 | 207/360 | 192/360 |
| timeouts | 24 | 0 | 3 | 5 |

Every per-opponent difference between this tree and .NET is inside the
two-sample binomial 95% band (largest: run007-100m 34 vs 27 of 48, band
±9.2; run007-300m 36 vs 30, band ±8.9; run007-final 16 vs 12, band ±8.7;
evasive-hard 8 vs 5 of 32, band ±6.3), and this tree matches the Q32
physics tip group for group (largest difference 3 of 48). Earlier physics
tips gave the same pattern (d71446c: 196/360 Q32 vs 200/360 this tree;
6ce6d27: 192 vs 199; 2f0a93a: 203 vs 206 with the previous oracle at
194/360). Disabling the .NET pickups moved .NET from 194 to 192 with no
per-opponent change larger than 1. The sweep
`.so` is the outlier: the policy trained on it wins 4/4 on the held-out
maps 3/4 against most scripted bots where the other three backends win 0/4,
and 24 of its 360 games time out (mean episode 42 s vs 20-23 s). Those are
the exploits the physics branch closed (a transformed player phasing a
platform through terrain, dash raycasts ignoring arrows), so evaluation
numbers recorded by the overnight-1 sweep on its own sim are inflated
relative to the game; the f64 representation does not move them.

## Throughput

Single-thread sim-only A/B (`native/rl/benchmark.c`, 2 players, repeat 4,
200k ticks, both measured under the same background load):

| | .NET | Rust |
|---|---|---|
| ticks/s stepping | 45.5k | 116.6k |
| ticks/s incl. resets | 16.1k | 79.5k |
| reset cost | ~9 ms | ~1 ms |
| library startup | 1.1 s | 0.3 ms |

Trainer-scale A/B (`bench_scale.sh` with `BOPL_RL_BACKEND=rust`): see the
results appended below when the box is idle (the sim-only ratio is the
upper bound; the GPU learner becomes the limit near 100k learner/s).

For the full change set, an immediately adjacent 2-player/repeat-4/200k-tick
pair, both limited to `CPUQuota=400%`, `CPUWeight=20`, `nice -n 19`, measured
104.9k -> 101.3k stepping ticks/s (-3.5%) and 71.2k -> 68.1k ticks/s including
resets (-4.3%). This was not a quiet box: the live sweep changed available CPU
noticeably even between short runs (a reverse-order adjacent pair measured
98.3k baseline versus 101.7k final), so treat the magnitude as load noise and
only conclude that no large throughput regression was observed. The prior
256-worker learner headline is about 127k learner steps/s; it was deliberately
not rerun against the live sweep.

For the discrete-arrow change alone, two back-to-back, bracketed 400k-tick
runs under the same limits measured old/new stepping at 96.9k/99.3k ticks/s,
then new/old at 100.2k/100.5k. Including resets they measured 62.5k/66.5k and
67.1k/67.3k. The ordering reversal is background-load noise; there is no
measurable throughput cost in this test.

After the final semantic-audit fixes, a new bracketed baseline/final pair
under `CPUQuota=600%`, `nice -n 15` measured stepping at
144.5k/139.5k then 135.8k/139.3k ticks/s (baseline/final then
final/baseline). Averaging each side gives 141.9k baseline versus 137.6k final
(-3.0%); including resets gives 94.5k versus 90.8k (-3.9%). Different reset
counts follow the corrected outcomes, and the ordering reversal again shows
that a few percent is background-load noise rather than a demonstrated hot
path regression.

### Speed work (2026-09-10)

Sim-only (`benchmark.c`, 2 players, repeat 4, 200k ticks, all builds
interleaved in one quiet window) and trainer-scale (`bench_scale.sh 300 --
"128 12 passive so"`, `BOPL_RL_BACKEND=rust`, inside a 600% CPU quota):

| build | stepping ticks/s | incl. resets | learner/s (128 workers) | eval_env ms/step |
|---|---|---|---|---|
| Q32.32 `Fix` (baseline, merge 88a77d5) | 143k | 96k | 100k | 1.42 |
| f64 `Fix` (ca04fea) | 547k | 316k | 172k | 0.60 |
| + raycast broad phase, no-alloc edge test (829009d) | 1.87M | 1.14M | 207k | 0.38 |
| + stick table, libm-free raw() (8797fc3) | 2.19M | 1.28M | – | – |
| + platform-physics 2f0a93a merged (fa7a5ea) | 2.06M | 1.21M | 202k* | 0.42* |
| physics tip 2f0a93a as committed (Q32.32) | 132k | 88k | 95k* | 1.53* |
| + platform-physics 68c6aa0 merged (final) | 2.19-2.29M | 1.26-1.28M | 207k | 0.39 |
| physics tip 68c6aa0 as committed (Q32.32) | 136-139k | 90-91k | 102k | 1.39 |

\* measured back-to-back while the overnight sweep was running (load 36-38);
the 68c6aa0 rows were taken on a quiet box (load 1-9) and are the final
numbers: 16x sim-only stepping throughput over the Q32 tip, 2.0x learner/s
and 3.6x less environment time per step at 128 workers.

Where the time goes now (perf on the trainer process, 128 workers): 60% of
samples are libcuda's spin in `cuStreamSynchronize` (the rollout policy
forward), 15% the Rust sim, 5% libm (sin/acos from the game's angle round
trip), 5% observation `memmove`, 4% the C environment. `eval_gpu` (0.36-0.40
ms/step) now equals `eval_env`; further sim speedups have little headroom at
this shape. Inside the sim (`perf` on `benchmark.c`): `world::closest`
23% (the per-platform bounds test itself; caching bounds on `Platform` would
need every collider write to refresh them), `acos` 11% and `sin` 10%
(`normalized_angle` / `Vec2::angle`, the game's stored-rotation model),
`game_tick` 9%, `refresh_entities` 8% (observation encoding, untouched).
