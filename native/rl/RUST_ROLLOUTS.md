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
- **Episode lifecycle**: 240-tick warmup simulated for real on Unity's
  spawn timeline (`GameSessionHandler.SpawnPlayers` after `timeBeforeTwo`:
  slimes exist from tick 63, play the 105-tick spawn animation with physics
  off, and `SlimeController.Spawn` launches them at `jumpStrength * 0.25`
  before they fall and land — the impact velocity sets the
  `AttachToGround` landing slide, so a free fall from tick 0 settled
  2.5e-3 units off on map 1), deaths (environment/arrow/self), stocks with
  respawn holds, first-death terminal at stocks=0, the 7200-tick timeout,
  and sudden-death platform sinking on the extracted curve. Blast-boundary
  deaths are always routed through match bookkeeping, including while an
  ability body is active.
  Sorted team ids map onto authored spawn slots and teammates fan out with the
  shipped alternating offsets, including wrapped team slots.
- **Movement**: the verified movement slice (`world.rs`). Registered input is
  consumed by the live body in the current tick, as in Unity. The public
  `Player.Position` is normally the position published by `SlimeController`
  before the later movement/physics work in that pass; the rollout represents
  that report phase separately from the live body.
- **NullAbility, Random, Dash, Grenade, Bow, Engine, Blink gun, Gust, Growth ray, Rock, Missile, Spike, Time Stop, Smoke, Platform, Revival, Roll, Shrink ray, Black Hole, Invisibility, Meteor, Throw, Push, Tesla coil, Mine, Teleport, Drill, Grappling hook, Beam, Rewind, Web swing, Poke ball, Gold mushroom, Tack, Freeze ray, Mjolnir, Portal Gun**
  (catalog ids 0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/31/32/33/34/35/36/37/38): NullAbility is the menu placeholder and is rejected as a
  match loadout by the public ABI, matching NativeAOT. Random loadouts are
  resolved during player spawning with the reference RNG sequence, full-catalog
  selection range, and the all-Random offensive fallback. Dash, Bow, and
  Platform are ported from `sim/Gameplay`
  with measured timings. Dash's non-wall ray dispatch now includes the active
  Rock circle: `BounceBall.OnCollide` receives its authored impulse before
  Rock's contact-kill flag kills the dasher at the ray near point. A five-player
  composition matches that same-pass death, owner credit and stock clone for
  300 ticks without allocating ray-target storage. A second composition drives
  two ordinary players inside the target radius and matches the target's
  one-sided overlap push before Dash passes through. A third composition enters
  Dash from a live Web Swing endpoint and matches the Dash-prefab `PlayerBody`
  constraint pass, bottom-endpoint straightening, leave-force handoff and the
  resumed ordinary slime for 300 exact ticks. A close-player Grappling Hook
  composition also matches top-endpoint transfer through Dash exit, with exact
  player/gameplay state and bounded inherited long-strand relaxation. An active
  Bow target now dispatches its own nonlethal `PlayerCollision` callback and
  remains stationary rather than receiving the ordinary slime's overlap push.
  Blink, Growth, Missile, Shrink, Duplicator, Magnet and Freeze forms match the
  same callback boundary in seven exact 240-tick compositions. Held Grenade,
  Smoke, Mine, Poke Ball, Tack and Mjolnir forms match it through six more
  180-tick compositions. Eighteen remaining active forms cover Engine, Gust,
  Spike, Time Stop, Platform, Revival, Black Hole, Invisibility, Meteor, Throw,
  Push, Tesla, Teleport, Drill, Beam, Rewind, Web Swing and Portal Gun through
  the same callback. A live Grappling Hook form and its strand cover the final
  non-Dash form callback through Dash exit with bounded solver geometry. Two
  staggered Dashes cover the remaining active-form callback without a kill or
  push.
  A released lethal Roll uses its live circle for the same ray dispatch, kills
  the dasher at the near point and owns the credit; its cosmetic trail consumes
  no updater hierarchy because neither trail component is a `MonoUpdatable`.
  An unprotected active Bow form is also exact across Time Stop: its charge,
  body and cooldown clocks remain fixed for all 690 stopped passes, then its
  release and Arrow spawn resume on the restored clock.
  Active Rock is exact across the same boundary. Its activation clock freezes,
  while `BounceBall.LateUpdateSim` retains the source's repeated entry-curve
  scale addition at zero delta; rigid-body motion resumes on restoration.
  Charging Roll also freezes for all 690 passes and exactly resumes charge,
  lethal release and ground motion; its later platform-hop branch remains open.
  A slot-two Missile holds its controlling form and guided projectile fixed
  across the same window, including zero-delta stabilization, then resumes
  steering, release and flight on the restored clock.
  Mine's older held form and a separately released autonomous Mine are exact
  across all 690 passes; the form releases after restoration and the live item
  resumes its hover clock and motion.
  Grenade, Smoke, Poke Ball, Tack and Mjolnir also retain their shared
  `ThrowItem2` forms for all 690 passes and release on the restored clock.
  Mjolnir waits for `Fire`'s manual collider registration before beginning its
  outbound updater, matching the source `grenade.hasBeenThrown` guard.
  An attached RocketEngine and its composite platform also freeze for all 690
  passes, then resume the force animation clock and platform motion exactly.
  Connected Tesla nodes retain their composite and lightning rows throughout
  the stop; a paused flicker resumes with the source's `UnFlicker` callback
  before the later lightning clock update, so the next cycle remains exact.
  Other transformed-form callbacks, rope detachment and the public
  horizontal-under-water failure remain open.
  Bow: transform 30 ticks, loadingFrame =
  passedSprites (1 at load start, then +8/+7/+7), slow-mo aiming physics,
  fire-on-release (early release fires from transform sprite index 2, tick
  11), knockback recoil, arrow gravity -40 (the DetPhysics override, not
  Unity's -9.82), policy-visible animator charge, delayed hurt-owner, the
  20-arrow owner cap, projectile collision during a dash raycast, and
  post-integration discrete circle contacts in DetPhysics group order.
  Terrain consumes an arrow; PlayerBody-backed hurtboxes kill without
  consuming it, matching `Arrow.OnCollide`. A released Grenade composition
  resolves both dynamic circles before Arrow's callback removes its projectile
  components; the later Grenade callback therefore leaves the resolved payload
  alive until its ordinary timed explosion. The same Circle/Circle path uses
  each item's authored radius: a released Tack keeps the resolved impulse after
  Arrow destruction, and its later wall callback tests the post-resolution
  velocity rather than pinning a still-upward body. Arrow instantiation now reserves
  all five prefab updaters and publishes the source collider hierarchy rather
  than an id-derived surrogate. A composed Time Stop golden fires an airborne
  arrow immediately before the stop, holds its position for all 690 frozen
  passes, and matches motion on the first restored pass. Magnet also routes a
  live Arrow through the source's stronger layer-9 pull, nearest-circle pickup,
  held spin/pin countdown and doubled projectile launch force. Gust captures a
  live Arrow in its fixed target set and applies its full Shockwave curve with
  orientation retained from the authored between-impulse Arrow update. Released
  Grenade, Smoke, Mine, Poke Ball, Tack and Mjolnir bodies also match the fixed-set item
  branch through both initialization-pass impulses and the complete force curve.
  Mjolnir keeps the source's intervening recall update between those two first-pass
  impulses with a fixed per-item accumulator. Gust
  also queries the live aim-oriented Bow form, the grounded controlling
  Missile form, and the grounded held-Grenade form, applying their one-shot
  PlayerPhysics forces before the later form movement. Missile, Mine and all
  five ThrowItem2 forms retain the source DPhysicsBox's one-pass-lagged basis
  while PlayerBody publishes the newly forced position. Mine's older ThrowItem
  path advances its surface local separately from that body move. Tack's
  released payload also exports the Projectile owner which the other item
  roots do not expose.
  Grenade currently covers the grounded `ThrowItem2` form through its exact
  enter/exit animations, held-object lifecycle, charged release, recoil and
  the thrown item's Q32 body integration. Its layer-10 circle is registered in
  the release pass and published in shape-group order. Player collision honors
  the delayed owner gate and queues the layer-18 explosion for the following
  pass. Both startup shockwaves, the first-frame lethal box, its 90-frame visual
  lifetime, self-kill accounting, stock clone, three choice indicators and the
  complete respawn interval are exact. The base golden also asserts safe owner
  overlap before the one-second gate, clearance and tick-132 contact detonation
  after re-entry. A second exact golden holds the grenade
  through its three-second fuse, including child-local held positioning,
  detonation before manual collider initialization, stable ID allocation and
  the resulting death/respawn path. A 260-tick horizontal throw is exact through
  four rounded-platform bounces, a wall reversal and its timed explosion. The
  immediate non-owner branch is exact through an airborne tick-85 contact,
  deferred blast, owner credit and the victim's complete stock respawn. A
  second 120-tick composition keeps both players in that blast and matches the
  callback-time hierarchy reservation plus global circle-before-box public IDs
  for both stock clones and all six choice markers. Byte-16 aiming leaves only
  a one-raw-Q32-unit form-position rounding difference, with identical contacts
  and lifecycle boundaries. The
  aerial form also uses its own extracted `PlayerPhysics`: 0.03-gravity aiming,
  speed clamp, full-gravity release,
  60%-blended horizontal control, body integration and raycast landing are
  exact. An Arrow-death composition now matches callback-time manual
  registration of the still-held payload, its deferred upward impulse, stock
  clone and later fuse explosion; only NativeAOT's inert destroyed-Arrow spin
  row is excluded. Lost-support/other-form exits and duplication remain open.
  Platform: anticipation, aim rotation,
  cursor sizing under the area cap, geometry-dependent mass/inertia,
  the authored translational and rotational home springs, ejection and
  same-pass exit reattachment, one-platform-per-player, curve-driven old-body
  removal, transform following, and squish deaths. Resizing preserves the
  scalar body orientation exactly, as `ResizablePlatform.ResizePlatform` does.
  Gust uses QuickSpell's measured cast/exit callbacks, its active circle
  hurtbox, deferred Shockwave registration, exact circle/shape target capture,
  one-time grounded-player push, and the shipped 0.25-second force curve. Its
  first time-zero platform impulse runs in prefab initialization before map
  home springs and its second runs later in the same updater pass. The fixed
  capture set now also retains live thrown items; a composed Grenade receives
  both time-zero impulses and the full curve with exact pooled-body motion. A
  composed Arrow receives the same force path while preserving source updater
  order: wave initialization, Arrow orientation, then the wave's normal update.
  A guided Missile exercises the corresponding oriented-box capture and pooled
  body impulse path through its later boundary explosion. Shockwave allocation
  also reserves its source `MonoUpdatable` hierarchy entry, keeping subsequent
  spawned prefab identities exact. A pushable active Rock takes Gust's distinct
  one-shot Player-layer fallback: the unnormalized center force is multiplied
  by 9,000 and queued on its BoplBody, then normal Rock physics carries it to an
  exact ability-body boundary death. An airborne charging Roll takes the other
  pushable-form path through `PlayerPhysics.AddForce`; its later boundary check
  observes the pose before the lower-hierarchy `PlayerBody` movement pass. Bow,
  Missile, Mine, the ThrowItem2 forms and the Blink/Growth/Shrink/Duplicator/
  Freeze/Magnet gun forms use the same force path with their oriented boxes.
  The gun-form slice also preserves the source split where Shockwave initializes
  before a separate grounded `PlayerBody` movement pass. Grenade, Smoke, Tack,
  Mjolnir and Poke Ball are each exact through their measured release boundary.
  Revival executes the measured QuickSpell cast and exit, registers the
  `RevivePositionIndicator` on the following updater pass, clamps it with the
  authored camera/water bounds, and replaces the marker owned by the same
  ability slot on recast. Dynamic colliders are published in DetPhysics shape
  group order, so the new circle precedes the static water box. A death with
  one live marker preserves stocks, replaces the dead slime with a hidden
  clone, waits through the measured wisp and doubly-registered particle
  travel, removes the marker, and plays the ordinary spawn animation before
  restoring input.
  Teleport is an instant two-stage cast. Its first cast creates the extracted
  portal body on the next updater pass; releasing the button keeps the slot at
  exact cooldown so the next press can cross the strict ready boundary. The
  second cast performs both source and marker circle queries, swaps all matched
  players once, ungrounds and zeros their imposed velocity, removes the marker,
  and disables each moved GameObject for the strict 0.6-second visual-effect
  interval. Each effect consumes the same four RNG samples per active tick,
  including its removal pass.
  Rewind records the live body and all three cooldown clocks in a fixed
  188-sample ring. A cast follows the source's arc-length-parametrized
  Catmull-Rom path for 17/25 of the recorded interval, suppresses input and
  PlayerPhysics, pushes intersected platforms, and restores the destination
  velocities, rotation and non-Rewind cooldowns. Arrival depenetration and the
  retained-ground release offset run in their original updater order.
  Invisibility is an instant cast: its five-second player clocks, castability
  gate, 10,000-second live sentinel, six-second restored cooldown, strict
  expiry, duplicate-slot chaining, player flags, and telemetry follow the
  original Player/SlimeController update order. Gold mushroom is also instant:
  it installs a four-second air-jump effect, applies the prefab's 0.85 strength
  multiplier after ordinary movement on each jump edge, and restores its
  authored five-second cooldown after the 10,000-second live sentinel.
  Tack reuses the extracted ThrowItem2 form, then binds its projectile owner
  on the following updater pass, spins and integrates until a top-side terrain
  contact, pins at the resolved landing position, arms after 0.8 seconds, and
  expires after 20 seconds. Its owner must first leave the proximity circle;
  a later owner or opponent overlap kills once and removes the trap.
  Mjolnir uses the same throw form and then applies its authored outbound
  damping, wall-clang/stall return transition, constant recall acceleration,
  speed clamp and owner catch. Rounded-platform callbacks apply its 3000 force
  on every eligible contact while the rigid body continues to resolve normally.
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

- Fixed-loadout RL sessions disable `AbilitySpawner` on both backends by
  default. `BOPL_RL_ABILITY_SPAWNS=1` enables the authored 40-second,
  three-item spawner in NativeAOT and Rust. Spawned and death-dropped pickups
  execute registration, delayed activation, launch, hover physics, reroll RNG,
  collection, third-slot replacement and deferred acquired-ability
  registration. Active ability bodies are rejected because the pickup callback
  requires a `SlimeController`; a Bow composition covers that rejection,
  post-exit collection and the source hierarchy order between its released
  Arrow and acquired ability. The deterministic acquired Revival is also cast
  directly from the replaced slot, covering airborne entry, marker creation
  and exit. A Grenade composition independently covers active-form rejection,
  released-item continuation, post-exit collection and an immediate complete
  cast of the acquired Revival. Dedicated collection and final-death goldens
  cover both origins.
- All catalog ids 0..38 now have executable behavior in the public backend,
  although the partial entries below still lack interaction branches. NullAbility is an
  intentionally rejected placeholder, Random is a spawn-time replacement,
  while Dash/Grenade/Bow/Gust/Time Stop/Platform/Revival/Teleport/Rewind execute casts. Grenade is
  exact through grounded and aerial release, owner-impact or timed-fuse
  detonation, and stock recovery; duplication and forced-exit paths remain absent.
  Tack is exact through throw, settling, arming, owner-clear re-entry, kill and
  expiry; its shared `Item.Duplicate` path remains absent.
  Mjolnir is exact through outbound flight, platform clangs, return and catch;
  its victim-hit and shared `Item.Duplicate` interactions remain open.
  Poke ball uses the shared throw form and bouncy item body, selects the nearest
  eligible opponent with the source tie-break, pins an invisible invulnerable
  ragdoll for the prefab's 1.92-second override, and ejects it with the exact
  horizontal division, upward speed and orientation. Its no-hit five-second
  fizzle is exact. A moving capture scenario has exact ownership, flags,
  capture/release ticks and entity topology; pre-existing moving-platform/player
  contact drift stays below 750,000,000 raw Q32 units (0.175 world units) and
  does not change the selected victim or any lifecycle boundary. Duplication
  and isolated boundary/time-stop interactions remain open.
  Missile is exact through the guided form, steering, stabilization, release into
  instability thrust, rounded-platform collision, and its MineExplosion lifecycle.
  Delayed grounded entry additionally preserves residual external velocity through
  the form's later updater phases, while airborne entry matches the standalone
  gravity query, SparkShot landing query, zero-input Move query, platform alignment,
  continued grounded control and normal Idle exit.
  Direct ordinary-player and active-Bow hits additionally match callback destruction,
  ownerless explosion lethality and telemetry, stock reconstruction, hitbox disable
  and visual expiry. The Bow case uses the registered form for Shockwave selection and
  reproduces its callback-time rotation reset and surface-normal death exit. Other
  active-form contacts, remaining projectile/item types and Portal traversal remain
  open. A crossing Arrow additionally matches dynamic
  Circle/Box resolution, both callback destructions and the complete ownerless
  explosion lifecycle using a fixed stack contact set. Duplicator's Missile branch
  matches three velocity-aligned fan copies with fresh engine state, pooled duplicate
  lifetimes, boundary destruction and four independent explosions. Growth and Shrink
  also match their signed non-player curves on the live box, unchanged inverse mass,
  boundary destruction and explosion expiry. A released Grenade matches the
  complementary callback split: dynamic correction, Missile-only destruction,
  next-pass blast ignition, chained Grenade explosion and independent visual expiry.
  A downward Water-trigger hit also matches callback destruction before the next
  bounds pass, deferred explosion registration, disable and expiry.
  Mine executes the separate original `ThrowItem` form and exact autonomous
  hover, arming, scan/chase, player ignition, anticipation, owner-credited
  MineExplosion kill, and stock recovery path. Chase normalization and the
  resulting shockwave response differ only by bounded sub-Q32-vector rounding;
  the form collider now keeps Unity's direct FixTransform offset basis and no
  longer needs its former 5,000/8,192-raw positional allowance. A 2,524-tick
  pickup composition also matches released-Mine continuation, post-exit slot
  replacement, deferred Revival registration, same-pass held-button activation,
  marker creation and exit. Scaling, boundaries, magnet holding, active-form
  pickup overlap and non-player igniters remain open.
  Rock executes the separate rigid circle body, terrain and Rock/Rock impulses,
  same-tick lethal player callbacks, authored pulse and exit curves, and exact
  stock/Revival collider reconstruction. The live collider retains the source
  DPhysicsCircle scale-ratio residue instead of rebuilding radius from the
  display scale; this makes the PortalShot impact and subsequent rigid motion
  raw-Q32 exact. Gust's active-form BoplBody force, ensuing flight and
  lower-boundary death are exact. Other Shockwaves, scale changes, arrow
  attachment, projectile/item detonation, and mixed transformed-form contacts
  still need dedicated goldens.
  Meteor executes its grounded anticipation and jump, direct airborne entry,
  normalized random vibration, short and maximum-charge downward release,
  charge-scaled platform impact, deferred Shockwave impulses, one-second
  ShakablePlatform displacement, and the two-tick lethal smear. Ray hits on
  players/items, DropAllAttachedPlayers, and isolated water/no-hit exits remain
  without dedicated differential scenarios.
  Roll executes its separate PlayerPhysics form, charged ground release,
  serialized Hermite speed curve, source `n/(n+1)` surface substeps, fixed-size
  swept lethal hurtboxes, jump cancel, airborne gravity/control and exit body
  handoff. Its circle and movement radii remain distinct as authored. A Dash
  composition exactly covers the released Roll's contact-kill callback and
  stock reconstruction, and a Time Stop composition covers frozen charge plus
  restored release and motion. Gust's PlayerPhysics force on an airborne
  charging Roll, subsequent flight and lower-boundary death are also exact. An
  immediate airborne release is exact through its world-up `ShortJump`,
  descending form exit and ordinary landing. Platform hopping, moving-platform turns, wall squish
  and other transformed-form contacts still need dedicated differential scenarios.
  Black Hole executes its grounded and airborne form animations, persistent
  layer-23 circle, Hermite growth, all-wall gravity, controlled-platform
  multiplier, player attraction, center-only wall consumption, hole merging,
  and the source's no-death-effect player callback. The main differential
  matches every lifecycle boundary; attraction while the cast form is active
  differs by at most 70,000,000 raw Q32 units (0.0163 world units) before the
  grounded exit reprojects both backends to the exact same platform pose.
  A map-2 composition also freezes an already released hole for all 690 Time
  Stop passes and resumes its growth afterward. Map 2's serialized low
  hierarchy labels differ between the oracle and rollout inventory, but both
  remain before all runtime prefabs and every dynamic identity, trajectory and
  lifecycle boundary matches.
  A live Bow Arrow also receives the hole's circle-array force before physics,
  integrates to the exact source pose, and is consumed by the trigger callback
  while the hole grows. Other projectile and item prefabs still need composed
  differential scenarios.
  Throw executes its separate MachoThrow2 form, animation callbacks, movement,
  eight ground-shrink passes and both release paths. Quick release creates and
  launches its boulder in one pass; a full pickup holds the zero-velocity body
  at the animated hand position before applying the authored scale-adjusted
  launch. Boulders use the common rounded-platform pool, including gravity,
  torque, collision response, player riding, bounds removal and death/stock
  reconstruction. Transferring Spike, Engine, Tesla and rope attachments from
  the consumed platform to the boulder remains open.
  Drill executes its separate static-box PlayerBody form, 19-update entry and
  13-update exit callbacks, air steering/gravity, first-cast HitboxCombo
  registration, underground wall traversal and owner-credited Explosion-layer
  kills. It preserves Ability/PlayerBody/Drill publication order, the first
  registered overlapping platform, minimum and entry/leave speed multipliers,
  underside forced exit, and the source's PlayerBody-only behavior during Time
  Stop. Four differential scenarios cover ordinary/forced exits, a same-team
  contact kill, bounds death, stock reconstruction, and 690 frozen ticks. Rope
  detachment and direct Shockwave pushing remain to be composed and goldened.
  Beam executes its grounded and airborne PlayerBody forms, charge/aim clips,
  frame-driven retained orb, maximum-time shrink and all three exit routes. Its
  allocation-free live graph preserves crossings and water reflections, pools
  BeamObject hitboxes, exports stable kind-4 trace rows after collider creation,
  applies terrain/BlackHole/Rock endpoint force, and dispatches the previous
  pass's lethal collider before resizing the pool. NativeAOT differential
  scenarios cover charge cancellation, air entry, one-pass kill latency, manual
  and automatic stops, orb destruction/reuse and two-player crossings. A
  BeamObject centre can differ by one raw Q32 unit because Unity associates the
  inverse-extent restoration and rotated half-length products differently; all
  axes, overlaps, lifecycle branches and later fixed-point state remain the
  same. Live Orb.ApplyBeam target charging has no producer in the five shipped
  maps and remains covered by the standalone component golden.
  Web Swing fires its fixed up-forward ray, creates a 12-point pooled rope, attaches both endpoints with the source PlayerBody/RopeAttachment schedule, and exports 11 stable kind-3 segment rows. Its four-map miss path and the complete map-3 latch, dynamic-platform swing, release boost, detached-strand linger and cleanup are exact across 1,310 NativeAOT/Rust stages. Rope arrays, collision scratch, observation storage and fork copies are preallocated; no tick or cast allocates.
  Grappling Hook adds the moving layer-24 hook, falloff/sprinkler flight, one-segment-per-pass growth, strict timeout, wall/player attachment, held reeling to the eight-point floor, dynamic cooldown, and the nearby-teammate transfer path. Seven differential scenarios cover all maps, timeout and teammate attachment over 1,870 stages with identical entity identities and lifecycle ticks. A focused 198-tick differential also makes the grounded third-player top-endpoint contact and its first deferred correction bit-identical; the contact pass no longer injects velocity before Unity's next eligible `PlayerPhysics.LateUpdateSim`. Free flight is exact; moving-platform constraint feedback amplifies a first 334-raw-unit player-position split without changing any contact, reel, deletion or cooldown branch. Unity's destroyed hook ACTIVE bit and cast-pass final-segment mesh sentinel are observation-only destroy/render queue artifacts described in `FULL_COMPATIBILITY.md`.
  Portal Gun is an instant cast with retained PortalShot bodies and portal
  mouths. It matches same-pass ManualInit motion, wall-callback versus timeout
  destruction, dominant-axis capture planes, pair replacement, strict
  inside/outside hysteresis, mirrored momentum, linked expiry, and traversal
  for ordinary players, Rock, arrows, pickups, missiles, thrown items,
  Revival/Teleport markers, and movable rounded bodies. Its primary 900-tick
  differential is exact. A composed grenade enters on tick 383 in both
  backends; its mapped velocity differs by under 0.0024 units/s because the
  existing ThrowItem2 launch pose is already 0.0185 units apart before it
  reaches the portal, without changing capture, exit, or later lifecycle. A
  Bow Arrow composition is exact: the arrow enters on tick 363, maps rightward
  momentum to the exit normal and is destroyed by the exit wall on that same
  pass in both backends. A PortalShot/Rock composition is exact for all 500
  ticks: the dynamic circle pair resolves on tick 244 after Rock's terrain
  contact, including position, linear and angular velocity; a recast collider
  is registered after retained circles and the former Rock traverses on tick
  367.
  An AbilitySpawner pickup composition crosses on tick 2233. Portal and pickup
  updaters merge by spawn hierarchy without allocation, so the older mouth maps
  momentum before the later pickup applies its hover drag and lift, matching
  NativeAOT through continued post-exit motion.
  Time Stop also freezes both endpoint clocks for all 690 stopped passes; an
  ordinary player crosses on tick 1811, after the pair would have reached its
  25-second lifetime on an unfrozen clock.
  Smoke is exact through its grounded throw, five-second fuse, four randomized
  cloud spawns, radius animation, and persistent collider export. Its ignition
  chain, scale-change, ring-eviction, and shared `Item.Duplicate` paths remain
  without dedicated differential goldens.
  Time Stop is exact through grounded channeling, casting, team protection,
  the 10-second hold plus exit-alpha lifetime, restoration, protected-caster
  cooldowns/motion, and global DetPhysics freezing for the currently ported
  projectile, item, smoke, explosion and platform bodies. Arrow entry, active
  Bow charge, active Rock scale/clock behavior, charging Roll, and active
  Blink/Growth/Shrink/Duplicator/Freeze forms are exact across all 690 frozen
  passes and the first restored form/projectile passes. A released persistent
  Black Hole also holds its position and Hermite radius for all 690 passes and
  resumes growth on restoration. Released outbound and returning Mjolnirs hold
  position, velocity and spin for the same interval, and frozen ordinary-slime
  animation state no longer advances until restoration. Released Grenade,
  Smoke, Poke Ball and Tack bodies likewise freeze all motion and resume their
  updater and physics paths together. Grenade's fuse also stays paused for all
  690 passes, then matches restored detonation, first-frame hitbox disable and
  complete explosion removal; Tack pauses 43 ticks into its 48-tick arming
  delay and later completes its owner re-entry kill. A Mine triggered
  into anticipation retains its jitter pose, body state and detonation clock
  across the same 690 passes, then registers `MineExplosion` five active ticks
  after restoration. All four spawned Smoke clouds also hold
  their enter-animation radii and resume the frame clock after restoration.
  A live Poke Ball capture pauses its ball, pinned victim, cooldowns and
  116-pass capture clock for exactly 690 passes before restored ejection.
  Frozen DetPhysics boxes
  retain their stop-entry sync through the restoration pass, matching the
  source updater delta captured at the pass start. Cross-ability interactions
  with catalog entries that remain absent inherit those entries' open scope.
  Spike is exact through its grounded cast and recovery, opposite-surface
  rounded-rectangle raycast, persistent down-thrust attack, platform-composite
  mass and inertia, cast impulse, and box/wall collision response. Recast
  replacement and player-lethality paths are implemented but still need
  dedicated differential scenarios.
  Engine is exact through grounded placement and recovery, scaled attached-box
  registration, composite mass and inertia, cast impulse, startup delay, fixed
  thrust curve, parent spring re-homing, and the resulting platform collisions.
  Its flame ignites the first overlapping Smoke collider without allocation.
  Smoke chaining, ring eviction, scale changes, platform destruction, time stop,
  and multiple simultaneous engines still need dedicated differential scenarios.
  Tesla coil is exact within 16 raw Q32 position units through grounded
  placement, two-node connection, staggered orb activation, lightning flicker,
  and third-cast ring eviction. It preserves DetPhysics' temporary four-body
  cast impulse before rebuilding the surviving three-body composite, and the
  evicted lightning remains visible for one pass as a destroyed trigger.
  Noncoincident moving-platform placement currently has bounded root/collider
  reconstruction differences below 0.0001 world units. Its owner-contact kill
  and uncredited `Explosion.PlayerOwnerId` telemetry match on the same tick;
  scale changes, platform destruction, time stop, smoke/mine ignition, and
  mixed attachments still need dedicated differential scenarios.
  Push is exact through grounded entry, animation/channel timing, direct
  control of an `AnimateVelocity` platform, home-spring retargeting, rigid-body
  force and torque, collider publication, and button-release restoration.
  Platform removal, sudden death, and overlapping-player cases remain to be
  exercised by dedicated differential scenarios.
  Freeze ray executes the shared `GunTransform` form, its authored slow-motion
  aiming and recoil physics, hitscan, one-pass water reflection, and
  `FrozenSolid` pin/ragdoll/thaw lifecycle. The frozen effect is stored in a
  reusable vector and performs no per-tick allocation. Grounded and rising
  ordinary-player hits match NativeAOT through freeze, thaw, resumed fall and
  landing; an airborne diagonal miss also matches through cast and recovery.
  A downward Water-reflected self-hit is exact through direct form cancellation,
  the exit-pass aimed collider basis, `FrozenSolid`, cooldown reset, thaw flight
  and airborne timed-ragdoll exit. A grounded active-Bow-form hit is also exact
  through the retained inactive-form basis, restored-slime air reattachment,
  next-pass grounded movement, FrozenSolid pin, grounded thaw and timed-ragdoll
  exit.
  Growth and Shrink rays share that `GunTransform` implementation and spawn
  the signed `ScaleChanger` on a player hit. Rust preserves Player.Scale,
  collider scale and PlayerPhysics radius as separate updater-phase values.
  It also retains DPhysicsBox inverse extents because assigning an unchanged
  non-unit scale continues to round them each tick in Q32. Grounded ordinary-
  player hits match through the complete resize and later movement. Growth and
  Shrink each also match an ordinary player hit while rising, including the
  complete signed resize, changed-body fall and platform contacts. Both rays now
  resize a live guided-Missile box through their signed non-player curves,
  unchanged zero-inertia mass and exact boundary/explosion lifecycle. Growth's
  downward Water reflection now also matches the rounded wall it resizes,
  including incremental collider ratios, next-pass body mass changes and the
  attached rider's live-radius motion. Shrink's corresponding Water-reflected
  wall case is exact through its authored minimum-scale clamp and complete
  signed curve. Remaining non-player, active-form, BlackHole and orb targets
  remain open.
  Blink gun uses the same exact GunTransform form and hitscan ordering. A
  player hit immediately disables the victim, changes the cast cooldown to
  five seconds, and advances a collider-free QuantumTunnel using player time.
  Grounded release uses the reference RNG to select an eligible rounded
  platform and sample its perimeter without allocating in the tick path.
  Relocation, inactive collider synchronization, strict 1.2-second lifetime,
  reactivation and the one-pass Player.Position publication lag all match the
  oracle. A second 500-tick differential catches an opponent immediately after
  jumping, then matches inactive airborne relocation, reactivation, delayed
  position publication, fall and landing. A downward shot now reflects from
  Water into a rounded wall and matches all 61 strict-delta samples of the
  serialized one-second shake. The wall then becomes inactive while retaining
  its enabled observation row, its attached rider detaches on the following
  player pass, and the body remains frozen for four seconds before restoring
  with its queued spring force. Rigid/item/orb targets, active forms, rope and
  magnet detachment remain open.
  Duplicator uses its exact smaller-radius GunTransform and implements the
  shipped delayed rounded-platform copy path, including overlap forces,
  next-pass spring suppression, fresh-body initialization and locally
  renumbered collider hierarchy. Its last-overlap/triple-ray query and
  allocation-free circular-item fan copies are also live, including deferred
  collider registration, copied body state, 1,800-pass expiry, grenade reset
  fields, ordinary rigid contacts, explosion shockwaves and water-boundary
  cleanup. The Grenade fixture asserts every source/copy removal plus the
  original and copied explosion registrations; held payloads are unreachable
  because their circles are not registered until release. A Smoke-copy
  differential additionally covers its cloned-Grenade
  reset, wall/Water contact detonation and same-pass removal of clouds born
  below WaterHeight. Tack copies now match fresh prefab runtime state, delayed
  behavior registration, arming and two trap consumptions over 450 ticks.
  Mjolnir copies match the same clone initialization, outbound damping, recall
  and original-hammer removal, then all three owner-return removals on tick 451
  over 500 exact ticks.
  Guided Missile copies match the item-layer three-copy fan, fresh engine
  state, owner propagation, velocity-derived orientation, boundary removal and all
  four deferred explosion lifetimes over 450 exact ticks.
  Poke Ball copies match registration, launch initialization, DetPhysics sweep
  ordering, the first shared dynamic-platform contact, later bounces and all
  four boundary/fizzle removals over 900 ticks. A second exact differential
  covers duplicated callback capture, a second queued copy callback against
  the already-held victim and 12 pinned passes; multi-captor expiry/ejection
  remains open. Mine copies now match the source's low-strength button-release
  throw, three-copy fan, deferred single `Mine.Init` RNG sample, arming and
  hover trajectories, reverse-list final shake and simultaneous 1,800-pass
  expiry for 2,200 ticks. The pickup diagnostic also replaces an
  equipped Duplicator with Revival and matches deferred registration, acquired
  cast, marker creation and exit.
  Player/projectile/weapon/Black Hole, attached-terrain and orb targets remain
  open, as does duplicated Poke Ball multi-captor expiry/ejection.
  Magnet uses its authored smaller-radius player form, animation clocks,
  grounded/airborne release and recoil, and strict landing/animation exits.
  Pulling is live on the cast pass as in the modified source. Its rounded-wall
  query, off-centre platform impulses, source-buggy close-distance test and
  ordinary/Macho player pulls are executable without per-tick allocation.
  Circular thrown items now receive pull force, transfer momentum on nearest
  pickup, pause their fuse while held, track the clamped firepoint and launch
  with the authored force; a 450-tick Grenade composition now matches complete
  public state within 64 raw Q32 units after source updater-phase separation.
  A second same-player Grenade composition matches the 214/243/245/250
  cast/release/catch/fire schedule with no positional allowance.
  Smoke, Poke Ball and Tack repeat that strict released-item schedule, including
  Tack's distinct Projectile ownership.
  The post-player static parry sweep also uses
  fixed player-slot storage and matches the source's ordered overlap, pause and
  symmetric `AddForce` pushes in a two-Magnet golden. A composed Time Stop
  golden preserves the retained pre-stop aim and locked item for 690 frozen
  passes, then fires on the first restored pass; the item pool carries the
  authored Hold countdown. Guided Missiles also use the source's item-layer
  pull strength, nearest pickup and locked motion. Their independent Hold
  countdown suppresses the 1,000-force instability engine on the 75-unit
  launch pass, matching the composed 275-tick differential exactly. A second
  500-tick differential keeps the Missile held while the caster lands. Magnet's
  source updater phases, its self-overlapping pull, item
  `MoveObjectOutOfWalls`, residual platform contact and suppressed held-item
  callback all match within 64 raw Q32 units. A live Bow arrow also matches the
  distinct layer-9 projectile pull, pickup, held countdown and `FireForce * 2`
  launch path. Other projectiles, Boulder/Black Hole bodies, ragdolls,
  weapon/spell holding and lock loss remain open.
  Revival's public death
  path currently supports one live marker; multiple simultaneous markers
  require the general multi-clone rollout representation. Teleport's player
  swaps are complete, but its projectile/item/boulder targets await the dynamic
  entity types supplied by those ability ports. Every slot uses its exact authored
  cooldown and common ground/castability metadata from the exported prefab.
  Closing this scope requires the remaining partial interaction branches and
  their dynamic-body collision pairs.
- Landed arrows are freed from the native entity table instead of remaining
  as attached visual children. The gameplay-relevant projectile collider is
  destroyed in both backends and the 20-arrow live-projectile cap is enforced,
  so the remaining difference is primarily observation/entity history.
- The rollout only contains dynamic bodies needed by ids 2/3/4/7/9/10/11/12/13/14/18/22/24/26/28/29/35/37/38. Forces from
  unsupported abilities (for example Machoman impulses into a platform) and
  their circle/box/rope collision pairs remain absent.
- f64 build only (2026-09-12): `Fix::sin` evaluates the quadrant-reduced
  argument with fdlibm's `__kernel_sin`/`__kernel_cos` polynomials instead
  of libm's `sin`. The result is within one ulp (~1e-16) of libm; the
  `fast_sin` table floor to the 2^-32 grid absorbs the difference except
  when the exact value lies within 1e-16 of a grid boundary. This is the
  only change of the speed work that alters trajectories (300 gate episodes
  bit-identical; over a 217-game `evaluate.py` suite 216 games are identical
  in winner and length, one diverges in length with the same winner; the
  trainer's fixed-seed bench moves EV 0.5985 -> 0.5926, entropy 5.7903 ->
  5.7571, the same values every other build of the branch reproduces). The
  q32 build keeps its table arithmetic.

## Gameplay fidelity audit (2026-09-10)

| Severity | Gap | Result |
|---|---|---|
| exploitable | Spawned/map platforms were immutable surfaces, allowing a transformed player to phase a platform through terrain | fixed: extracted bodies, springs, RR-RR contacts, impulses/correction, riders and transform following |
| exploitable | A movement body crossing a blast boundary was pre-marked dead, then skipped by match death bookkeeping | fixed: the marker is consumed once and crossing on the current tick is caught |
| exploitable | Dash raycasts ignored arrows even though the Unity collision mask includes Projectile | fixed: ordered arrow hits before the first wall kill when owner rules allow; the PlayerBody callback leaves the arrow live, as in Unity |
| behavior-changing | Resizing kept the prefab mass/inertia instead of `10 + MassPerArea * area` and `ApproximateMOI` | fixed, including live observation `inverse_mass` |
| exploitable | Spawned platforms had no rotation spring because the extractor omitted `rotationSpeed`/`rotationMu`; terrain contact torque accumulated into a large persistent tilt | fixed: both fields are required extracted data, the authored 40 / 1/6 spring runs every tick, and resize no longer reconstructs or pins body rotation |
| behavior-changing | Platform aim used scalar-angle interpolation instead of Unity's sine-weighted `Vec2.Slerp`, leaving a fixed home-angle offset | fixed with the source formula, including its dot ceiling and zero-sine fallback |
| behavior-changing | Bow parsed but ignored `maxNumberOfArrows = 20` | fixed with per-owner oldest-projectile eviction |
| behavior-changing | Ability entry/spawn and Dash/Bow/Platform updater phases were delayed or collapsed into the public player position | fixed: current input drives live bodies immediately while Unity's component-specific report phases remain explicit |
| behavior-changing | Bow used the original slime hurtbox during transforms, lost ordinary-slime pushes, raycasted after aerial control, and exposed charge one animator pass early | fixed with active ability colliders and source updater/animator ordering |
| behavior-changing | Respawn point/timing choice was absent and fast-spawn upward motion lagged one pass | fixed: three directional points, early jump confirm, exact selection publication and launch phase |
| behavior-changing | Curve tangents and old-platform shrink removal were absent | fixed: exported Hermite tangents/exact evaluator and collidable `being_removed` bodies |
| behavior-changing | Destroying a recast platform detached riders immediately instead of on their next updater pass | fixed: preserve the destroyed ground through the removal observation, then apply Unity's 0.05-unit normal separation and ordinary-slime gravity phase |
| behavior-changing | Rust arrow flight was swept while Unity integrates first and dispatches discrete Circle/RR, Circle/Circle, then Circle/Box contacts | fixed; terrain callbacks consume the arrow before player callbacks, while a PlayerBody kill leaves it flying |
| behavior-changing | Cosmetic level-intro ground delays consumed RNG only in .NET, shifting later sudden-death samples | fixed: Rust consumes the samples without rendering the intro; sink activation and trajectories now align |
| behavior-changing | Arrow rotation used post-gravity velocity and its boundary cull ran after physics | fixed: policy-visible rotation and side/water lifetime use the authored pre-physics phases |
| exploitable | Rust deleted a spawned platform when its transformed owner crossed a blast boundary | fixed: the owner dies, but the separate `ResizablePlatform` remains collidable as in Unity |
| behavior-changing | An end-of-tick Rust boundary pass killed ordinary/Bow bodies one tick before Unity's pre-`PlayerBody` boundary component | fixed with component-specific boundary phases |
| behavior-changing | Match termination consumed deaths after physics in the same tick, while Unity's earlier `GameSessionHandler` sees them next tick | fixed: result consumption now runs at the beginning of the following updater pass |
| behavior-changing | Keyboard-configured players discarded the high S/D bits and used joystick aim for movement | fixed: W/A/S/D drive motion while the angle byte remains independent ability aim |
| configured scope | Ability pickups | disabled on both RL backends by default; the shared diagnostic opt-in executes the authored spawner, while death drops remain live in ordinary matches; both paths include flight, rerolls, collection and slot replacement |
| exploitable | Abilities other than 2/4/14 are absent | open scope described above |
| cosmetic/observation | Stuck arrow children are not retained | not changed; no live collision remains in Unity either |
| cosmetic/observation | Unity can emit a death-dropped pickup in the terminal snapshot | fixed: the pickup remains a live simulated entity when the match continues |

## Warmup, phase and physics audit (2026-09-11)

Trigger: v1 `evasive-hard` (kind 3, level 2) lost to an idle opponent on
map 1, side 0, seed 140001 on the Rust sim (t=3246) while winning on .NET.
The first state difference was at t=0: player x differed by 2.46e-3 units
after the warmup, and platform bases by ~4e-6 rad. The bot's decisions were
identical through d065; d066 flipped `bot_landing_steer(-1)` after the
positions had drifted 0.2 units through a jump/landing loop.

| Gap | Unity source | Fix |
|---|---|---|
| Warmup let slimes free-fall from tick 0; Unity instantiates them at tick 63, plays the 105-tick spawn animation with physics off and launches at `jumpStrength * 0.25`; the landing speed (-19.92 vs -19.0 u/s) sets `AttachToGround`'s `groundedSpeed` slide on a tilted platform | `GameSessionHandler.StartSpawnPlayersRoutine`/`SpawnPlayers`, `SlimeController.BeforeSpawnInit`/`Spawn` (SlimeController.cs:567), `PlayerPhysics.AttachToGround` (PlayerPhysics.cs:495-509) | reset runs that timeline; a warmup ending mid-animation hands the rest to `respawn_spawn_ticks` |
| Map platform bases used exact sin/cos; `FixTransform.rotation` uses `new Vec2(angle)` = `FastCos`/`FastSin` | FixTransform.cs:36-47, Vec2.cs:31-35, Fix.cs `FastSin` | `Vec2::angle` for map bases; the f64 `fast_sin` reproduces the LUT exactly (see "Number type") |
| Sticky perimeter extents were `e`; Unity reads `One / (One / e)` back from the collider | `StickyRoundedRectangle.UpdateBounds` (:104), `DPhysicsRoundedRect.CalcExtents` (:351-359), refreshed after animate-in at GameSessionHandler.cs:769 | `data::surface_extents` on map and spawned platforms |
| Collider axes were `basis * e`; Unity builds `ComplexMul(Vec2.right / inverseExtents, basis)` (one raw unit apart on negated products and non-power-of-two extents) | `Box.SetRotation` (Box.cs:31-35), `RoundedRect.SetRotation` (RoundedRect.cs:22-26) | `OrientedBox::from_basis`; platform and hurtbox rows expose those axes |
| `ability_time` counted every ability tick; Unity accumulates `timeSinceEnteredAbility` only while invisible | Player.cs:337-353 | stays zero |
| Grounded hurtbox phase: riders were re-positioned before the collider sync and the body row read the live position; Unity syncs the Player-layer box in `DetPhysics.Simulate` and moves a grounded body afterwards in `LateUpdateSim`, so the observed box of a walking slime lags one phase | Updater.cs:303-323, PlayerPhysics.cs:138-146, `RlObservation.Collider` | `sync_collider` before the late phase, which runs after arrow contacts; the body row reports the synced box |
| Dash exit tick: the ejected slime kept the leave velocity until the next tick; `Dash.ExitAbility` hands over an ungrounded body and `onExitAbility` enters Fall, whose `VelocityBasedRaycasts` can land on the ejection tick | Dash.cs:224-241, SlimeController.cs:249 and 263-272, PlayerPhysics.cs:495-509 | `collision_rays` on the exit tick after the ejected position is published; the `dash_exit_followup` flag and a double gravity step on airborne dash entry are gone |
| Landing-tick publication: the pre-tick position was published; `PlayerPhysics.UpdateSim` applies gravity and `AttachToGround` before `SlimeController.OldUpdate` publishes `body.position` | PlayerPhysics.cs:116-131, SlimeController.cs:712 | publish after the gravity phase |
| Warmup shorter than 63 ticks had live slimes from tick 0; `SpawnPlayers` instantiates them on tick 63 (no slime rows and `Player.Position` zero before, then the spawn animation) | `GameSessionHandler.SpawnPlayers`, SlimeController.cs:516-533 | `unspawned` countdown per player; timers keep -1.5 s + (tick-62)*DT for every warmup (`warm_ab` 0/60/63/100/240 match) |
| The deactivated slime skipped the entry pass's `Move`; `SlimeController.UpdateSim` still calls it once, and that frozen body is what the hurtbox row shows during the ability | SlimeController.cs:1067-1084, PlayerPhysics.cs:230-260 | `ability_origin` from a `move_only` clone |
| Platform transform ran in physics order; Unity's `PlatformTransform.UpdateSim` copies the platform FixTransform before DetPhysics moves it, `AnimateVelocity.Init` homes on the current position, and `Box.UpdateExtents` rescales the existing axes | PlatformTransform.cs:122, 281-300, AnimateVelocity.cs:74-79 and 126-129, Box.cs:37-44, Updater.cs:265-273 | component/physics phase order; sizing keeps stable axes |
| Bow aim started at the raw ground normal and hurtbox axes were `basis * e` | SlimeController.cs:666 (`Normalized(upVector)`), BowTransform.cs:246, Box.cs:30-35, FixTransform.cs:56-59 | re-normalised normal; `Box.SetRotation` axes |
| Dash/Bow hurtbox rows read the live body; DetPhysics syncs them before the movement phase, the Dash rotation being `Tools.GetRotationFromDirection` | Dash.cs:123-124, Box.cs:30-35 | rows report the synced box |
| Bow exits: the landing exit re-derives the surface local through `SyncPhysicsTo(AbilityExitInfo)`; the animation-end exit ungrounds, publishes, then raycasts | BowTransform.cs:116-130 and 225-239, PlayerPhysics.cs:548, 556-570, 577, SlimeController.cs:251-259, Ability.cs:151 | `bow_exit` |
| Grounded Bow entry kept the slime velocity; `SyncPhysicsTo(PlayerInfo)` re-derives the body from its local with zero velocity and `OnEnterAbility` clears the external one | PlayerPhysics.cs:526-535, BowTransform.cs:91 | zero velocity, surface position |
| Bow aiming clamped before the landing raycast; `Vec2.Lerp`/speed clamps evaluated in a different operand order | BowTransform.cs:222, 240-243, 249, 262, Fix.cs:41608 | Unity's order |
| Arrow facing used a vector division and unwrapped angles; DetPhysics divides per component and wraps into [0, 2pi) | DetPhysics.cs:1645 | per-component division, wrap |
| Ability exits landed one tick early: the slime's `SpriteAnimator` (hierarchy 7) fires the land callback at F+N+1; the jump edge follows `Player`'s input profile | SpriteAnimator.cs:190-231, SlimeController.cs:249, 796 | 37-tick land after exits |
| A landing arrow left its platform untouched; DetPhysics resolves the Circle/RR contact (impulse and correction) before the landing callback | DetPhysics.cs:3012-3048, 3195-3199, BoplBody.cs:215-229 | `arrows_contacts` applies the impulse |
| Dead players' rows stayed; `PlayerCollision` destroys slime and ability objects, DetPhysics drops updatable-phase casualties before this pass's contacts and callback casualties one pass later | PlayerCollision.cs:201-210, Arrow.cs:124-128, DetPhysics.cs:2513 | updatable-phase deaths vanish, callback deaths are observed once as destroyed rows |
| Platform aim reset rotation without stick input and Slerp operands were reordered; a landing in the entry pass's `PlayerPhysics.UpdateSim` sets `State.land` before `EnterAbility` freezes the state machine | PlatformTransform.cs:276-285, SlimeController.cs:1126-1128 | rotation only overridden while aiming; land observed on the entry tick |
| RR pair roles: `Fix` multiplication floors, so `IdPair(active, current)` order changes impulses and corrections by an LSB | DetPhysics.cs:2169-2236 | pairs follow the sweep |
| Sweep axis used active platforms only; DetPhysics means over active colliders but spreads over every registered one, including frozen inactive hurtboxes | DetPhysics.cs:2827-2894, 2498-2530 | every row feeds the variance; inactive rows keep their frozen FixTransform |
| Ability rows froze at the slime's post-exit position; `Ability.ExitAbility` deactivates the ability body where it ended, `AttachToGround` aligns a landing Dash, Bow and Platform zero their rotation first | Ability.cs:151-152, PlayerPhysics.cs:508, BowTransform.cs:119 and 225, PlatformTransform.cs:462 | published exit pose, `dash.exit_rotation` (this flipped the sweep axis in a three-platform chain: ep 98) |
| Final death dropped no pickup; `DropAbilities` spawns a `DynamicAbilityPickup` circle at body + (0, 0.04) | SlimeController.cs:274-310, DynamicAbilityPickup.cs:147-155 | live layer-13 pickup with exact hidden boundary-effect hierarchy reservation, sampled launch, flight, rerolls and collection |
| `PlatformArea` factor grouping, `ResizePlatform`'s swapped `(rHeight, rWidth)` mass arguments and `State.Size`'s operand order each round differently | DPhysicsRoundedRect.cs:621-627, ResizablePlatform.cs:53, PlatformTransform.cs:335-336 and 354 | source expressions |
| A jump that re-attaches in the same pass showed `fall`; `Jump` ungrounds (`State.fall`) and `Move`'s air raycasts land it again | SlimeController.cs:868, 956-958, 1126-1128, PlayerPhysics.cs:236-238 | `jumped_this_frame` |
| Boundary deaths inside an ability used `Accident` and the slime's position; the ability body exits with a default `AbilityExitInfo` (`Other`), the Bow after `UnGround` and before `PlayerBody` integrates, and the slime takes that position | DestroyIfOutsideSceneBounds.cs:135-148, SlimeController.cs:215-219, BowTransform.cs:120 and 124 | cause `Other`, pre-integration Bow position |
| A dasher killed by an arrow died at the slime's origin; Dash moves its body to `nearPos` before `KillPlayerOnCollision` and keeps it on a kill | Dash.cs:177-183 | dies at the ray hit |
| Removal sizes used a symmetric lerp and resized before the spring; `Fix.Lerp` clamps only `a`'s weight (the curve's overshoot past 1 scales the original size) and `ResizablePlatform` follows `AnimateVelocity` in its prefab | Fix.cs:41606-41609, ResizablePlatform.cs:123-134, ResizablePlatform.prefab component order | one-sided clamp; resize after the spring impulse |
| A death inside an ability dropped the pickup at the frozen slime; `killPlayer` on the ability object moves the slime to the ability body first | PlayerCollision.cs:124-131, SlimeController.cs:215-219 | drop at the ability body |

Evidence (300 gate episodes, seeds 1000-1299, `repr_ab <NET> <Q32> 300 1000
0`, every observation field at tolerance 0): bit-identical ticks 32.7% at
the branch base -> 52.8% after the phase fixes -> 70.7% (arrow impulse) ->
90.0% (removal rows) -> 98.4% (sweep axis, frozen rows) -> 99.9% (exit
poses) -> 100.0% (74809/74809 ticks) with 300/300 equal lengths and
winners. The last classes were closed in order: RR pair roles, frozen
inactive rows feeding the sweep axis, ability boundary deaths, the dash-arrow
death position, the removal lerp and the pickup drop position. The later
three-player death-drop flight, reroll and collection path is covered by
`ability_pickup_death_drop_ab.c`. `entity_count` differs by design: Rust keeps a fixed
table and marks rows with `E_REMOVED`, consumers filter on the active bit.
`TICK_STEP_OK` and the scripted-bot idle gate is 9/9. Sim-only cost
(`benchmark.c`, 2 players, repeat 4, 200k ticks, `perf stat`): retired
instructions +8.0% over the branch base (1728M -> 1867M), cycles +6-11%;
wall-clock on the loaded box is within its noise. The per-pass Player-layer
box sync and the hurtbox rows are the added work Unity also does; the sine
table is consulted only when the rotation changed (`collider_basis`) and the
active body row copies the synced box.

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
  un-interpolated lookup, index = angle >> 15) reproduce the table exactly:
  entry `i` of `data/sinlut.bin` is `floor(sin(i * (pi/2) / 205886) * 2^32)`
  for all 205887 entries, so the f64 build evaluates that formula at the
  truncated, mirrored index. The game does depend on it: `Vec2(angle)`
  builds every FixTransform basis (FixTransform.cs:45), so tilted map
  platforms sit at the table's angle (up to 7.6e-6 rad from the exact one)
  and a slime resting on one settles where that basis puts it. Both number
  modes now use the table for map platform bases as well.
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

Verification of the representation change (`repr_ab` drives two builds
with the fidelity gate's action stream at action repeat 1 and compares every
observation field per tick): the Q32 and f64 builds of one tree give 300/300
winners and lengths at tolerance 0, differing only by one-raw-unit products
(`velocity.y * DT` truncation versus the correctly rounded double) that grow
chaotically through platform riding; no discrete field differs at a first
divergence. Every scripted A/B trace keeps its tolerance on both builds. One
Rust-only mistake surfaced here rather than any inherent f64 difference: size
ticks reconstructed `body.orientation` through `Vec2::angle -> normalized_angle`,
which walks one sine-table cell (7.6e-6 rad) per application near pi in Q32;
`PlatformTransform.State.Size` retains `PlayerBody.right` and
`ResizablePlatform.ResizePlatform` never writes rotation, so neither build does
now.

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
| mean length (decisions) | 63.0 | 73.4 | 66.5 | 62.4 | 62.5 | 62.4 |
| timeouts | 0 | 0 | 0 | 0 | 0 | 0 |
| environment deaths | 133 | 144 | 153 | 138 | 138 | 134 |
| self deaths | 165 | 153 | 146 | 160 | 160 | 164 |
| opponent deaths | 2 | 3 | 1 | 2 | 2 | 2 |
| dash activations | 221 | 284 | 240 | 217 | 217 | 218 |
| bow activations | 498 | 560 | 510 | 489 | 491 | 494 |
| platform activations | 86 | 129 | 113 | 87 | 88 | 86 |
| arrows fired | 484 | 551 | 497 | 476 | 476 | 480 |
| arrow hits | 2 | 3 | 1 | 2 | 2 | 2 |
| winner agreement | - | 267/300 | 284/300 | 296/300 | 296/300 | 297/300 |

The episode-length gap fell from +10.4 to -0.6 decisions. Mean position
divergence remains under 0.002 through decision 16, then grows chaotically
(0.082/0.198 at decisions 31/46). Restoring the spawned-platform rotation
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
`dash_arrow_ab` covers the separate Dash raycast callback path. Unity calls
`Arrow.OnCollide` with Dash's Player-layer `PlayerBody` hitbox, so it returns
before destroying the arrow; `PlayerCollision` can still kill the dasher.
Rust previously killed both. At activation ticks 284/285 the repaired trace
matches the .NET victim and death tick and retains one live arrow (`1/1`,
previously `1/0`).

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
default RL configuration. With `BOPL_RL_ABILITY_SPAWNS=1`, both create the
first row on level tick 2401 and enable it on 2408; over the measured 2700
episode ticks metadata is exact and the maximum position/velocity gaps are
3,180/13,737 raw Q32 units (7.5e-7/3.2e-6 world units). The difference starts
when the hover ray enters its strict half-unit damping boundary and comes from
the already documented separately rounded rounded-rectangle ray geometry; it
does not alter a boundary, reroll, or collision in the golden. Before this
port, with spawns enabled, `sudden_death_ab` reached
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

Bracketed before/after pairs for each change set (2 players, repeat 4,
200k-400k ticks, same CPU quota and nice) stayed within the run-to-run load
noise of the box (-4% to +3%); none showed a measurable throughput cost.

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

### Speed work (2026-09-12): per-decision CPU

Sim-only (`benchmark.c`, 2 players, repeat 4, 400k ticks; `perf stat`
user instructions and cycles per tick are load-independent, the rates were
taken in one interleaved window with one other job on the box). Every row
except the sine kernel is bit-identical to the previous one over the
300-episode gate (`repr_ab`: 74807/74809 ticks within tolerance, 300/300
equal length and winner) and byte-identical in observations, masks and
rewards over 200k environment decisions (delayed, undelayed and
no-autoreset paths).

| build (commit) | ins/tick | cycles/tick | stepping ticks/s |
|---|---|---|---|
| main 59be9a2 | 9314 | 2750 | 1.91M |
| release profile lto=fat, codegen-units=1 | 9130 | 2701 | 1.95M |
| + `target-cpu=native` (fab5039) | 7867 | 2540 | 2.05M |
| + fdlibm sine kernel, branch period reduction, align/basis caches (9f43e11) | 7235 | 2604 | 1.99M |
| + cached platform bounds, one broad phase per 3-ray group (f8e9474) | 5802 | 1863 | 2.75M |
| + libm-free `Fix::raw`, platform epoch skips (b0c3a57) | 4678 | 1572 | 3.32M |
| + slice-copy publish, AVX-512 candidate mask (377ff7f) | 4328 | 1518 | 3.44M |
| + no rays without candidates, one sqrt in movement_limit, input memo (f2c1b86) | 4056 | 1390 | 3.71M |
| + platform stage reuses its buffers (39ef3f0), stack winner check (c816dcb) | 3958 | 1372 | 3.77M |

2.35x fewer instructions and 2.0x fewer cycles per tick; the reset cost
went from 14.7 to 6.2 us. The sine kernel row is slower in cycles on its
own (the libm call it replaces was well predicted) and pays off once the
callers around it shrink; it is kept for the instruction count.

Where the sim's time goes now (`perf` on `benchmark.c`, final build):
`game_tick` 19% (flat: the per-player ability/movement dispatch),
`move_grounded` 12%, `refresh_actor_entities` 11% (the entity rows the
DetPhysics sweep axis reads every tick, mostly `Fix::raw` conversions),
`normal_and_zone` 8%, `collision_rays` 5%, `fast_sin` 4%, `sin` 4%
(`Surface::position` corners, dash), `acos` 4% (`align`), `step` 4%,
`Surface::position` 3%, `edge_hits` 3%, `sync_collider` 2%, `fill_slimes`
2%. No per-tick heap allocation remains on the common path (the dash
target collect and arrow contact lists allocate only when they have
entries). Deferring `refresh_actor_entities` to the observation was
measured as a 6-12% cycle bound and rejected: `platform_pairs` reads
every row's position for the sweep axis whenever two or more platforms
exist, so the per-tick pass is part of the physics, not the export.

Environment side (`bopl_puffer.h`, no bots, `env_bench`, 200k decisions,
9-tick delay): 3.2 us per decision of which the four ticks are ~1.6 us,
`bopl_encode` 0.5 us (two rows), the two state exports (`observe_sample`
at phase 3, `observe_main` at the end: entity rows, slime rows, header)
~0.3 us and the phase-3 state snapshot ~0.05 us. The delay path used to
copy the 13 KB observation four times per decision (scratch -> sample ->
history -> publish) and memset it twice; it now encodes each observation
exactly once into its routed slot from a ~3 KB state snapshot and zeroes
only the entity/slime/player slots the previous encode into that row left
non-zero. Trainer bench (`bench_shape.sh`, w128 b4 p4, 300 updates, the
branch's own trainer, i.e. without the later trainer-side work; the
`eval_env` column is the environment time per step):

| build | sps | learner sps | rollout ms/step | eval_gpu | eval_env | EV | entropy |
|---|---|---|---|---|---|---|---|
| main 59be9a2 | 222866 | 167149 | 0.7179 | 0.2914 | 0.3977 | 0.5985 | 5.7903 |
| new header, main .so | 231104 | 173328 | 0.6778 | 0.3211 | 0.3263 | 0.5985 | 5.7903 |
| + sim .so, bots | 237279 | 177959 | 0.6494 | 0.3452 | 0.2742 | 0.5926 | 5.7571 |
| + tracked zero tails (823e3d0) | 245542 | 184156 | 0.6115 | 0.2927 | 0.2979 | 0.5926 | 5.7571 |
| final (c816dcb .so) | 242728 | 182046 | 0.6237 | 0.2919 | 0.3057 | 0.5926 | 5.7571 |

`perf` shares of the trainer process (10 s mid-run): before, libcuda 35.8%,
libbopl_sim 27.7 (game_tick 4.9, rounded_rounded 2.8, closest 2.8,
refresh_entities 2.3, refresh_actor_entities 1.8), libc 13.0 (memmove 8.6,
memset 2.0, malloc 0.9), `_C` 12.8 (compute_observations_from 4.4,
bot_shot_clear 1.6, bot_predict_foe 1.2, bot_surface_y 1.0), libm 5.8,
vdso 4.0. After (final), libcuda 49.9%, libbopl_sim 28.8 (game_tick 9.3,
rounded_rounded 3.5, refresh_actor_entities 1.9, fast_sin 1.6), `_C` 10.4
(bopl_encode 4.7, c_step 1.7, bot_sdf_avx512 1.5), vdso 5.4, libc 3.3
(memmove 0.8; malloc no longer in the list), libm 1.0. The absolute environment work per
step fell by ~40% at this shape while the GPU round trip did not move, so
the spin in `cuStreamSynchronize` takes the freed share.

Scripted bots (`bopl_bots.h`, v1 code frozen by SCRIPTED_BOTS.md): every
edit is a bit-identical transformation checked by fixed-seed fingerprints
of all eight `kind-level` opponents (observation, mask, reward, terminal and
applied-action hashes over 200k decisions, both the AVX-512 and the
`-DBOT_SDF_AVX512=0` scalar path). mixed-hard 24.1 -> 7.3 us per decision,
archer-v2-hard 30.7 -> 8.1 (including the ~3 us of ticks and encode).

Intentional deviations of this branch, all documented above or in the
code: the fdlibm sine kernel (the only trajectory change, quantified in
"Known approximations"); `target-cpu=native` in `.cargo/config.toml` (the
trainer .so runs on the box that built it; not portable); the observation
overflow telemetry is counted when a state is sampled rather than when it
is published (same totals, two decisions earlier); the golden movement
projection drops the Rust-only bookkeeping fields (`golden.py`).
