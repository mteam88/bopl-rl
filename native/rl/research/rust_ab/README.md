# Scripted Rust vs .NET A/B tools

Small C drivers that open both backends through `bopl_rl.c` and print
per-decision traces side by side. Each takes `<BoplSimNative.so>
<libbopl_sim.so>` and expects the fidelity gate's bootstrap replay at
`/tmp/gate_bootstrap.rep`. Build like the fidelity gate:

```sh
cd native/rl && cc -O2 -I. -o /tmp/move_ab research/rust_ab/move_ab.c bopl_rl.c -ldl -lm
```

- `move_ab.c`: fixed-action angle sweep, positions per decision.
- `keyboard_ab.c`: keyboard-configured W/A/S/D movement with an independent
  joystick aim byte, including the high `S/D` action bits.
- `state_ab.c`: idle/run/turn/stop, SlimeController state and timers.
- `entity_velocity_ab.c`: distinguishes the one-tick-stale public player
  position and zero-velocity trigger PhysicsBody from the current-tick moving
  PlayerBody state in slime records.
- `dash_phase_ab.c`: tick-level dash activation, anticipation, collider, and
  displacement phases.
- `dash_arrow_ab.c`: dashing through an incoming arrow; verifies the dash
  victim dies under the same owner rules while the PlayerBody contact leaves
  the projectile live.
- `dash_rock_ab.c`: a Dash ray crosses an active Rock circle, dispatches the
  Rock's BounceBall impulse first, kills the dasher at the exact ray near point,
  credits the Rock owner and reconstructs the dasher's stock clone.
- `dash_roll_ab.c`: a Dash ray crosses a released lethal Roll circle, kills the
  dasher at the ray near point, credits the roller and reconstructs the stock
  clone with the source's zero-entry cosmetic-trail hierarchy behavior.
- `roll_broad_ab.sh`: one broad gate over five optimized-Q32 Roll scenarios:
  grounded charge/release and stock reconstruction, immediate airborne
  ShortJump exit, lethal Dash contact, Gust force and a complete charging-form
  Time Stop freeze/restoration; all 3,000 differential ticks pass.
- `dash_player_ab.c`: an ordinary target overlaps the Dash ray origin, receives
  its one-sided `SlimeController.OnCollide` push, and survives the pass-through
  before the Dash completes its normal endpoint and exit.
- `dash_rope_ab.c`: a live map-3 Web Swing bottom endpoint is carried into
  Dash, constrained by Dash's own `PlayerBody`, straightened at exit and handed
  back to the ordinary slime with exact rope-solver history through tick 300.
- `dash_grapple_rope_ab.c`: a close-player Grappling Hook attaches a grounded
  victim at its top endpoint before that victim Dashes; gameplay state and all
  rope lifecycle fields match through exit, with bounded long-strand geometry.
- `dash_grapple_form_ab.c`: a Dash ray crosses another player's active
  Grappling Hook form while its strand is live, covering the form's nonlethal
  no-push callback through Dash exit with bounded rope geometry.
- `dash_bow_ab.c`: a Dash ray crosses an overlapping active Bow hurtbox and
  verifies the form's nonlethal `PlayerCollision` callback does not receive the
  ordinary slime's one-sided push.
- `dash_dash_ab.c`: staggered overlapping Dashes cover the final active-form
  callback, tolerating only the inactive duplicate prefab's inert ENABLED bit.
- `dash_gun_forms_ab.c`: the same overlapping ray callback across live Blink,
  Growth, Missile, Shrink, Duplicator, Magnet and Freeze forms, covering seven
  more nonlethal `PlayerCollision` owners without an ordinary-slime push.
- `dash_throw_forms_ab.c`: the same callback across held Grenade, Smoke, Mine,
  Poke Ball, Tack and Mjolnir forms through each Dash exit.
- `dash_remaining_forms_ab.c`: eighteen more active forms (Engine, Gust,
  Spike, Time Stop, Platform, Revival, Black Hole, Invisibility, Meteor, Throw,
  Push, Tesla, Teleport, Drill, Beam, Rewind, Web Swing and Portal Gun) through
  the same callback.
- `grapple_player_attach_ab.c`: a close-player Grappling Hook flight attaches
  its top endpoint to the grounded third player after physics, deferring the
  grounded correction until the next pass exactly as `PlayerPhysics` does.
- `bow_ab.c`: hold arrow N decisions then release, bow slot flags/timers.
- `bow_phase_ab.c`: tick-level airborne Bow entry, scheduler-staggered body
  position, charge/release flags, and deferred arrow registration/flight.
- `bow_ground_exit_ab.c`: grounded fire/landing callback order and the
  following original-slime movement.
- `bow_broad_ab.c`: 300 ticks through grounded Bow loading, aim, fire and
  exit, Arrow rigid-body flight, active-Rock Circle/Circle resolution,
  BoplBody-sensitive Arrow destruction and Rock's projectile callback.
- `ability_hitbox_ab.c`: an incoming arrow grazing Bow's larger active
  ability hurtbox after the original slime hurtbox is disabled; also verifies
  post-integration discrete contact and that a PlayerBody kill leaves the
  arrow flying.
- `arrow_bounds_ab.c`: `Arrow.prefab` side-boundary destruction phase and
  removal from the live projectile observation.
- `grenade_bounce_ab.c`: a horizontal Grenade release through four exact
  rounded-platform bounces, a wall reversal and the later timed explosion over
  260 ticks.
- `grenade_non_owner_ab.c`: an upward Grenade throw hits an airborne opponent
  before the owner gate opens, then verifies deferred blast credit and the
  victim's complete stock respawn over 700 ticks.
- `grenade_multi_kill_ab.c`: the same early contact leaves both players in the
  blast and verifies callback-time hierarchy reservation plus shape-grouped IDs
  for two stock clones registered together.
- `grenade_forced_exit_ab.c`: an Arrow kills the active held form and verifies
  callback-time payload registration, next-pass upward impulse, stock clone and
  the later fuse explosion over 400 ticks.
- `time_stop_arrow_ab.c`: a live Bow projectile across Time Stop entry, all
  690 frozen physics passes, and its first restored gravity/flight pass; also
  verifies source-derived Arrow prefab hierarchy allocation.
- `time_stop_bow_ab.c`: an unprotected Bow form charging across all 690 Time
  Stop passes, followed by restored release, form exit and Arrow registration.
- `time_stop_gun_forms_ab.c`: active Blink, Growth, Shrink, Duplicator and
  Freeze forms across all 690 stopped passes, first restored sync and exit.
- `time_stop_rock_ab.c`: an unprotected active Rock across all 690 stopped
  passes, including Unity's repeated zero-delta entry-scale update and the
  first restored rigid-body motion.
- `time_stop_roll_ab.c`: an unprotected charging Roll across all 690 stopped
  passes, then restored release and ground motion up to the separately open
  platform-hop boundary.
- `time_stop_broad_ab.c`: the complete cast, team protection and restoration
  lifecycle overlapping an unprotected airborne Meteor; verifies 690 raw
  vibration/RNG updater passes with a frozen ability clock and zero-delta
  physics, retained input, protected cooldown progress and restored release.
- `time_stop_missile_ab.c`: a slot-two Missile controller and its guided box
  projectile across all 690 stopped passes, zero-delta stabilization, restored
  steering, release and subsequent flight.
- `time_stop_mine_form_ab.c`: the held original `ThrowItem` Mine form across
  all 690 stopped passes, restored release and autonomous item registration.
- `time_stop_mine_ab.c`: a previously released autonomous Mine held motionless
  for all 690 stopped passes and resuming its hover clock and motion afterward.
- `time_stop_mine_anticipation_ab.c`: a collision-triggered Mine paused during
  anticipation for all 690 stopped passes, then registering its disabled-first
  `MineExplosion` exactly five active ticks after restoration.
- `time_stop_throw_forms_ab.c`: slot-two Grenade, Smoke, Poke Ball, Tack and
  Mjolnir held forms across all 690 stopped passes, followed by exact restored
  release and item registration.
- `time_stop_mjolnir_ab.c`: released Mjolnirs in both outbound and recall
  flight across 690 stopped passes each, including frozen position, velocity,
  spin, owner cooldowns and land-animation state, then resumed motion after
  restoration.
- `time_stop_released_items_ab.c`: released Grenade, Smoke, Poke Ball and Tack
  bodies across all 690 stopped passes, restored motion, and Tack's later exact
  owner-clear/re-entry self-kill and stock transition; Tack enters the stop 43
  ticks into its 48-tick arming delay.
- `time_stop_smoke_cloud_ab.c`: all four spawned Smoke clouds across 690
  stopped passes, preserving identities, positions and enter-animation radii,
  then resuming radius growth on restoration.
- `time_stop_poke_capture_ab.c`: a live Poke Ball capture beginning shortly
  before Time Stop, 690 paused capture passes, exact restored expiry/ejection,
  and the expected one-pass pinned-player publication handoff.
- `time_stop_engine_ab.c`: an attached live RocketEngine and its composite
  platform across all 690 stopped passes, then restored force-clock and body
  motion.
- `time_stop_tesla_ab.c`: two connected Tesla nodes and their lightning across
  all 690 stopped passes, including paused flicker animation, restored
  `UnFlicker` callback order and the next complete flicker cycle.
- `time_stop_black_hole_ab.c`: an already released persistent Black Hole across
  all 690 stopped passes, with exact frozen position/radius/cooldowns and
  resumed Hermite growth after restoration.
- `ability_spawner_ab.c`: ability-pickup-layer bodies after the authored 40-second
  pickup boundary. The default RL configuration is 0/0; with the shared
  `BOPL_RL_ABILITY_SPAWNS=1` opt-in it compares the first spawn, seven-pass
  collider delay, launch and hover flight in NativeAOT and Rust.
- `ability_pickup_collect_ab.c`: keyboard pursuit of that first live pickup,
  exact trigger callback at level tick 2506, third-slot replacement, the
  one-pass destroyed row, and the newly instantiated ability collider that
  joins DetPhysics on tick 2507.
- `ability_pickup_active_form_ab.c`: holds Bow across the same pursuit to
  verify that an active ability body cannot collect a pickup, then releases,
  collects it as the ordinary slime and checks same-pass Arrow/acquired-ability
  hierarchy and next-pass observation order.
- `ability_pickup_cast_ab.c`: collects the deterministic Revival pickup while
  airborne, activates the newly instantiated third slot on the following pass,
  and follows its QuickSpell form, marker creation and exit exactly.
- `grenade_pickup_ab.c`: overlaps the first pickup with an active Grenade form,
  verifies rejection and released-item continuation, then collects after exit
  and casts the acquired Revival through marker creation and exit.
- `grenade_broad_ab.sh`: one broad gate over 16 optimized-Q32 scenarios for the
  complete Grenade source surface: grounded/airborne casts, fuse, bounces,
  owner gates, multi-kill, forced release, pickup replacement, Arrow, Gust,
  Magnet, Time Stop, Duplicator and Dash.
- `black_hole_ability_ab.c`: BlackHoleClap form entry/exit, persistent hole
  spawn and growth, whole-map rounded-platform pull, center-only platform
  destruction, no-effect player kill, and stock-clone reconstruction over 500
  NativeAOT/Rust ticks.
- `blink_ability_ab.c`: grounded Blink entry, player hitscan and the complete
  QuantumTunnel hide, seeded relocation, reactivation and delayed publication
  lifecycle over 500 ticks.
- `blink_airborne_ab.c`: a second 500-tick Blink trace that catches the target
  immediately after a jump and compares its inactive airborne relocation,
  reactivation, fall and landing.
- `blink_water_wall_ab.c`: a downward Blink shot through Water reflection,
  upward rounded-wall selection, all 61 strict-delta shake samples, delayed
  deactivation, rider detachment, four-second frozen lifetime, restoration and
  resumed spring motion over 600 exact ticks.
- `blink_broad_ab.c`: a 500-tick generic-target composition in which Blink
  selects a released Grenade from the complete collider registry, immediately
  hides it, freezes its body and fuse for the strict four-second tunnel, then
  restores the unchanged object on tick 493.
- `growth_airborne_ab.c`: a 500-tick Growth trace that hits an ordinary player
  while rising, then follows the complete ScaleChanger curve, enlarged fall
  and platform contacts.
- `growth_broad_ab.c`: a 520-tick combined Growth/Black Hole composition that
  covers the full RigidBodyAffector ray branch, serialized 0.0199999998 inverse-
  mass absorption and the resulting Hermite radius growth.
- `growth_water_wall_ab.c`: a 500-tick downward Growth cast through Water
  reflection into a rounded wall, including the complete non-player scale
  curve, incremental collider/body-mass changes and attached-rider motion.
- `shrink_water_wall_ab.c`: the corresponding 500-tick Shrink reflection,
  including the wall's authored minimum-scale clamp, complete signed curve,
  inverse-mass updates and attached-rider motion.
- `shrink_broad_ab.c`: a 520-tick combined Shrink/Black Hole composition
  covering negative serialized inverse-mass absorption, white-hole conversion
  and the absolute-mass Hermite radius path.
- `shrink_airborne_ab.c`: the corresponding 500-tick Shrink trace through its
  negative ScaleChanger, reduced-body fall and platform contacts.
- `freeze_airborne_hit_ab.c`: a rising ordinary player caught by Freeze through
  `FrozenSolid` takeover, thaw, resumed fall and landing over 500 exact ticks.
- `freeze_broad_ab.c`: a 620-tick duplicate-slot composition covering the
  correct per-prefab collider, active-Bow cancellation, a second hit while the
  victim is pinned, overlapping FrozenSolids and both shatter boundaries.
- `duplicator_smoke_ab.c`: Smoke's three-copy fan through cloned-Grenade reset,
  rigid and Water contact detonation, randomized clouds and same-pass removal
  of a cloud born below WaterHeight over 450 ticks.
- `smoke_broad_ab.c`: one 500-tick Smoke composition covering an airborne
  charged release, a post-owner-gate contact with Bow's non-igniting Arrow,
  later ordinary-contact detonation, all four randomized cloud registrations
  and persistent cloud animation.
- `duplicator_tack_ab.c`: Tack's three-copy fan through fresh prefab runtime
  state, deferred collider/behavior registration, ballistic contact, arming
  and two exact trap consumptions over 450 ticks.
- `duplicator_mjolnir_ab.c`: Mjolnir's three-copy fan through fresh prefab
  physics/runtime state, deferred launch initialization, outbound damping,
  recall, original-hammer removal and simultaneous owner-return removal of all
  three copies over 500 exact ticks.
- `duplicator_poke_ab.c`: Poke Ball's three-copy fan through fresh prefab
  physics/runtime state, split collider/behavior registration, global
  Circle/Circle sweep ordering, shared dynamic-platform contacts, later
  bounces and all boundary/fizzle removals over 900 exact ticks.
- `duplicator_poke_capture_ab.c`: 248 exact ticks through the fan, an airborne
  victim captured by one copy, a second queued copy callback capturing the
  already-held victim on the next pass, and the original ball's simultaneous
  callback capture immediately after a different player exits Bow form.
- `duplicator_mine_ab.c`: a released low-strength Mine followed by its
  three-copy fan, single deferred `Mine.Init` RNG sample per clone, arming and
  hover trajectories, reverse-order final shake and simultaneous 1,800-pass
  expiry over 2,200 exact ticks.
- `mine_pickup_ab.c`: the first authored AbilitySpawner pickup across Mine's
  form and autonomous-item continuation, post-exit collection, slot-three
  replacement, deferred Revival registration, same-pass held-button cast,
  marker creation and QuickSpell exit over 2,524 ticks.
- `duplicator_pickup_ab.c`: deterministic AbilitySpawner collection replaces
  equipped Duplicator in slot three, then the acquired Revival registers,
  casts, creates its marker and exits exactly.
- `duplicator_broad_ab.sh`: one broad gate over 13 optimized-Q32 Duplicator
  scenarios spanning grounded/airborne form use, platform and item families,
  guided Missile, Poke Ball capture, pickup-driven loadout replacement, Gust,
  Time Stop and Dash.
- `ability_push_ab.c`: one-sided overlap push on a normal slime touching an
  active Bow ability hurtbox.
- `aim_map.c`: full-charge arrow velocity per angle byte.
- `hit_ab.c`: charge and fire at the opponent, hit outcome and death cause.
- `bow_test.c`: Rust-only bow scenario used while porting.
- `platform_ab.c`: grounded platform transform overlap, jump-and-land player
  carry, and the map-2 AntiLockPlatform trajectory.
- `platform_broad_ab.sh`: one optimized-Q32 gate over Platform entry,
  anticipation, rotation, sizing, ejection, overlap-loaded motion, landing,
  boundary death, retained spawned-wall collision and map-2 AntiLockPlatform
  motion; six scenarios and 4,140 differential ticks pass together.
- `gust_ability_ab.c`: complete QuickSpell cast and exit, active Gust hurtbox,
  deferred Shockwave registration, exact captured targets, both time-zero
  impulses around the map spring updater, force-curve platform trajectories,
  and recovery. It compares every public player, slime, telemetry, and entity
  field for 360 ticks; static authored geometry allows one Q32.32 least bit.
- `gust_broad_ab.c`: one 430-tick four-player composition with a released Black
  Hole, live PortalShot, ordinary player/platform targets and a spawned Macho
  boulder. The boulder takes Shockwave's tag-selected default-force branch;
  the full crowded state stays within the existing Black-Hole/form spatial
  bound without any lifecycle or interaction difference.
- `gust_grenade_ab.c`: a released Grenade captured by Gust's fixed target set,
  both time-zero impulses, the complete force curve and subsequent item flight.
- `gust_smoke_ab.c`: the same fixed-set released-item path for Smoke, including
  same-pass Gust entry/capture, both initialization impulses and pooled flight.
- `gust_poke_ball_ab.c` and `gust_tack_ab.c`: released Poke Ball and
  Projectile-backed Tack capture, ownership, full force curves and pooled flight.
- `gust_mjolnir_ab.c`: released Mjolnir capture through the initialization
  impulse, intervening recall update, ordinary update impulse, complete force
  curve and resumed flight over 340 exact ticks.
- `gust_mine_ab.c`: a released autonomous Mine through fixed-set capture, both
  first-pass impulses, the full force curve, subsequent flight, hover and
  arming over 340 exact ticks.
- `gust_grenade_form_ab.c`: Gust's one-shot PlayerPhysics force against the
  grounded Grenade form while its manually initialized payload is still held.
- `gust_throw_forms_ab.c`: the same exact active-form branch for Smoke, Tack,
  Mjolnir and Poke Ball, including release and Tack's public projectile owner.
- `gust_mine_form_ab.c`: Gust's one-shot PlayerPhysics force against Mine's
  older ThrowItem form and its split surface-local/PlayerBody movement phase.
- `gust_gun_forms_ab.c`: exact Gust pushes on active Blink, Growth, Shrink,
  Duplicator and Freeze forms, including their separate grounded PlayerBody
  movement after Shockwave initialization.
- `gust_magnet_form_ab.c`: the corresponding exact active-Magnet push with its
  live pull field aimed through already-covered player and platform targets.
- `gust_arrow_ab.c`: a live Bow Arrow captured by Gust, including the split
  init/Arrow-update/wave-update ordering, force curve and projectile flight.
- `gust_missile_ab.c`: a live guided Missile box captured by Gust, including
  the full force curve, later right-boundary explosion and prefab hierarchy.
- `gust_missile_form_ab.c`: Gust's one-shot PlayerPhysics force against the
  grounded controlling Missile form, including its lagged collider basis.
- `missile_player_ab.c`: a guided Missile directly hits an ordinary player, then
  matches ownerless MineExplosion lethality, telemetry, stock reconstruction,
  first-frame disable and visual expiry over 500 bit-exact ticks.
- `missile_bow_ab.c`: the same guided impact against an active Bow form, covering
  registered-form Shockwave selection, callback-time Bow death exit and the same
  ownerless explosion/stock lifecycle over 500 bit-exact ticks.
- `missile_arrow_ab.c`: a crossing Arrow and guided Missile resolve as two dynamic
  bodies, then both callbacks destroy their colliders before the deferred ownerless
  explosion runs to expiry over 500 bit-exact ticks.
- `missile_grenade_ab.c`: a released Grenade and guided Missile resolve as
  dynamic bodies; only Missile dies in the first callback, its next-pass explosion
  ignites Grenade, and both chained explosion lifetimes match over 500 ticks.
- `missile_water_ab.c`: a downward guided Missile crosses the Water trigger,
  dies in the same callback pass and matches deferred explosion registration,
  first-frame disable and visual expiry over 430 bit-exact ticks.
- `missile_broad_ab.sh`: one broad gate over 16 optimized-Q32 Missile
  scenarios spanning entry variants, steering/instability, players/forms,
  Arrow, Grenade, Water, Gust, Time Stop, duplication, scale, Magnet and Dash.
- `mine_broad_ab.sh`: one broad gate over nine optimized-Q32 Mine scenarios
  spanning ordinary chase/detonation, owner scan immunity and owner ignition,
  held and released Gust, held/released/anticipation Time Stop, Duplicator fan
  copies and Dash against all shared ThrowItem forms.
- `mjolnir_broad_ab.sh`: one broad gate over six optimized-Q32 Mjolnir
  scenarios spanning aerial flight/recall, grounded wall clang and platform
  push, released Gust, both Time Stop phases, duplication and Dash against all
  shared ThrowItem forms.
- `rock_broad_ab.sh`: one broad gate over six optimized-Q32 Rock scenarios
  spanning rigid cast/terrain motion, lethal player contact, Gust, Time Stop,
  PortalShot plus later portal transit, and Dash.
- `duplicator_missile_ab.c`: a live guided Missile receives three
  velocity-aligned fan copies with fresh engine state, then all four projectiles
  match boundary destruction, deferred explosions and visual expiry over 450 ticks.
- `scale_missile_ab.c`: Growth and Shrink each resize a live item-layer
  Missile box through the signed non-player curve, unchanged zero-inertia mass,
  boundary destruction and explosion expiry over 900 differential ticks.
- `gust_rock_ab.c`: Gust's one-shot push of a pushable active Rock through the
  BoplBody fallback, subsequent rigid flight and lower-boundary death.
- `gust_roll_ab.c`: Gust's one-shot PlayerPhysics force on an airborne charging
  Roll, subsequent form motion and correctly phased lower-boundary death.
- `gust_bow_ab.c`: Gust's one-shot PlayerPhysics force against a live
  aim-oriented Bow form and its subsequent charging-form motion.
- `magnet_broad_ab.sh`: the broad Magnet gate; compiles and runs 12 exact
  differentials spanning grounded/airborne form use, walls and players, parry,
  all ordinary ThrowItem2 bodies, Arrow, guided Missile/landing, Time Stop and
  Gust with one command.
- `magnet_ability_ab.c`: Magnet's first-pass pull, rounded-platform impulses,
  player pull, grounded and airborne form motion, release recoil, landing and
  animation exits over 1,000 optimized-Q32 ticks. The active form centre allows
  2,048 raw Q32 units for the independent FixTransform offset reconstruction;
  axes, rotations, gameplay bodies and every interaction boundary remain exact.
- `magnet_item_ab.c`: a live Grenade pulled, picked up, held and launched by
  Magnet over 450 ticks, including fuse suppression and holder cleanup.
- `magnet_grenade_ab.c`: a strict same-player released-Grenade composition
  through item pull, nearest pickup, locked pinning and 10,000-force launch.
- `magnet_smoke_ab.c`, `magnet_poke_ball_ab.c` and `magnet_tack_ab.c`: the same
  strict released-item Magnet path for the remaining ordinary ThrowItem2 bodies.
- `magnet_parry_ab.c`: ordered empty-pickup circles from two Magnet casters,
  the post-update static overlap sweep, pause and symmetric push over 240 ticks.
- `magnet_time_stop_ab.c`: a held Grenade across 690 frozen passes, retained
  aim and hold refresh, followed by first-restored-pass launch and form exit.
- `magnet_missile_ab.c`: an airborne Magnet pulls a guided Missile through the
  item-layer branch, catches and pins it with engine thrust suppressed, then
  launches it at the authored Missile speed while `BoplBody.Hold` expires. The
  complete 275-tick state is exact within 64 raw Q32 units.
- `magnet_missile_landing_ab.c`: the same guided Missile remains held while
  the airborne Magnet lands. It covers MagnetGun/Ability/PlayerBody updater
  phase publication, the source's self-overlapping pull box,
  `MoveObjectOutOfWalls`, residual DetPhysics contact and the held Missile's
  suppressed collision callback over 500 exact ticks.
- `magnet_arrow_ab.c`: a live layer-9 Bow projectile follows Magnet's stronger
  pull branch, nearest-circle pickup, held spin/pin countdown and doubled
  launch force. Held position inherits the documented Magnet form bound;
  velocities and interaction boundaries match.
- `teleport_ability_ab.c`: first and second instant casts, deferred portal
  marker registration, release-driven readiness, caster transfer, inactive
  effect interval, RNG advancement, and exact reactivation.
- `teleport_multi_ab.c`: a three-player two-way portal swap with a teammate at
  each query circle. It also verifies sorted-team spawn assignment, authored
  teammate offsets, two simultaneous effects, and entity order.
- `teleport_cast_state_ab.c`: marker-placing and swapping casts under four aim
  directions. `SlimeController.OldUpdate` returns straight after
  `EnterAbility`, so the run/stop/turn switch never runs on a cast tick, and an
  instant ability leaves no form behind to keep it frozen afterwards; aim and
  movement share one stick, so a horizontal aim used to drive the port into
  run -> stop -> idle while the reference stayed idle.
- `tesla_exit_state_ab.c`: the Tesla plant-exit branch.
  `PlantObjectOnPlatform.ExitAbility` reaches `enterTurn` (SlimeController
  `EnterStop`) when the held ground direction opposes the facing, because
  `OnEnterAbility` set `localScale.x` to -1 while facing right, and the state
  its recover callback installs survives the exit tick. Both orientations are
  required, so neither branch can regress unnoticed.
- `mine_owner_contact_ab.c`: `Mine.ScanForPlayers` skips the owner, so an armed
  mine hovering inside its own six-unit scan radius never chases or
  proximity-detonates on the player that threw it; `Mine.OnCollide` has no such
  exemption, so the owner's own hurtbox still ignites it and the resulting
  `Explosion` (`PlayerOwnerId = item.OwnerId`) kills the owner with
  `CauseOfDeath.Accident` while `Player.Kill` still credits it a kill. Entities
  are matched by hierarchy number because the fatal tick's explosion and
  death-drop pickup receive swapped observation ids under the documented
  deferred-prefab drain-order gap.
- `portal_arrow_ab.c`: a live Bow Arrow crosses a linked Portal Gun pair,
  mirrors its momentum and collides at the exit on the same pass.
- `portal_rock_ab.c`: a flying PortalShot resolves an ordinary circle/circle
  impact against an active Rock exactly, then the former Rock player crosses
  the linked pair after leaving the form.
- `portal_pickup_ab.c`: an authored AbilitySpawner pickup crosses an older
  linked portal pair, covering hierarchy-ordered hover physics and mapped exit
  momentum.
- `time_stop_portal_ab.c`: Time Stop pauses both linked-mouth lifetime clocks
  for 690 passes, then an ordinary player traverses after their unfrozen expiry.
- `repr_ab.c`: number-representation A/B between two builds of the Rust
  backend (Q32.32 `--features q32` vs the default f64 `Fix`), or .NET vs Rust:
  gate action stream at repeat 1, every observation field compared per tick,
  first divergence and outcome agreement reported (`REPR_AB_TRACE=<episode>`
  prints the gap growth).
- `platform_phase_ab.c`: tick-level platform button/anticipation/spawn/resize
  phases and transformed-player following.
- `platform_boundary_ab.c`: platform-form death below water while the separate
  spawned `ResizablePlatform` remains collidable.
- `platform_rotation_ab.c`: spawned-platform orientation and angular velocity
  across one or more ABI-compatible backends; pair it with the Rust-only
  `platform_rotation_trace` binary to inspect the private spring home rotation.
- `recast_ab.c`: two platform casts, including the evicted platform's live
  shrink/removal interval.
- `stock_respawn_ab.c`: stock consumption, directional point selection, early
  jump confirmation, and the accelerated spawn interval.
- `sudden_death_ab.c`: authored-platform sink activation and randomized-delay
  trajectories around the 120-second transition.

- `scale_boundary_ab.c`: a grounded player pressing a collider-rescaling
  ability slot (Dash) while sudden death has carried its platform under
  SceneBounds.WaterHeight. `player_tick`'s entry-pass boundary check runs
  `move_only` on a copy of the body, and `move_only` does not call
  `sync_collider`, so the copy tests with the hurtbox basis the previous pass
  left behind. Airborne, clone and real Move agree; grounded they need not.
  Twelve cases on map 0, sweeping the press offset around the plane crossing
  and a one-pass versus six-pass hold. Excludes one separately-owned known
  difference, the `sudden_death` header flag publication lag, and says so on
  stdout when it does.

  **An unreached case is not a passing case.** `run_case` returns a distinct
  code when no tick ever published a living player below the plane, the probe
  prints a `SCALE_BOUNDARY_MAP` line per map, and it exits non-zero if any
  case went unreached -- `SCALE_BOUNDARY_VACUOUS` when a whole map did.
  This matters: the probe ran for a while over maps 0 and 2 and reported
  `cases=24`, but all twelve map-2 cases were unreached. Sudden death drops
  map 2's platforms fast enough that the player drowns on the same pass it
  first dips below the plane (-11.95 there, not -11.3 -- the kill plane is
  per-map and comes from the water trigger), so no published tick ever shows
  it alive underneath. Twelve vacuous cases read from outside exactly like
  twelve passing ones, which is the same failure as a trace-only probe
  exiting 0. Map 2 was therefore removed rather than left inflating the
  count; each subject now carries its own plane, and a map only belongs in
  the list once it demonstrably reaches.

- `sudden_death_chunk_ab.c`: the endgame. Maps 0-4 played out with **no input
  at all**, every entity, slime and player field compared exactly from reset
  through the tick that resolves the round (level 7302-8218 depending on the
  map). Sudden death begins at level tick 7200 and `map-sweep-3000` stops
  4,200 ticks short of it, so before this probe nothing in the collapsing
  endgame -- or in round resolution itself -- was covered by any gate.
  `make -C native/rl sudden-death-chunk-ab`.

  What it caught: map 2's `AntiLockPlatform` orbiter (entity 18, hierarchy
  162, the one platform in the set with an `OrbitPath`). `WinGame` opens with
  `ActivateSuddenDeath(activate: false)` (GameSessionHandler.cs:655), and
  `GameSessionHandler.UpdateSim` runs ahead of the authored level updaters,
  so on the pass that decides the round `SuddenDeathInProgress` is already
  down and AntiLockPlatform.cs:35 fires once more -- one 600-unit kick toward
  waypoint `(0, 24.674)`, `600/18/60 * 0.98 = 0.5445` of velocity, which is
  exactly the `(-0.464, +0.285)` the reference had and the port did not. The
  port held a single flag for both the session state and
  `AnimateVelocity.inSuddenDeath`, which is a *per-component* latch nobody
  ever lowers; those are now separate (`sudden_death` vs
  `sudden_death_latched`, plus `game_in_progress` for `gameInProgress`).

  `SD_CHUNK_DUMP=<first>[:<last>]` prints entity rows for a tick range,
  `SD_CHUNK_NEAR=<id>` restricts that to one body's neighbourhood and
  `SD_CHUNK_SLACK` widens it. Note when reading a neighbourhood dump that
  these platforms are rounded rectangles with 0.01 x 0.01 extents and a 6.03
  corner radius -- effectively discs -- so a filter written against the
  extents finds nothing.

  One announced exclusion, as narrow as a single field on a single class of
  row: the port leaves `reserved[0]` set on entity rows it has already
  flagged `DESTROYED`, where the reference clears it. That is publication
  metadata on a body that no longer simulates, and an open item against the
  observation writer. It prints a `NOTE` when it applies.

## Trace-only probes: exit status is not a result

Most probes here compare internally and `return 1` when the backends disagree.
**Seventeen do not.** They print a per-backend trace and end with `return 0`,
so their exit status is clean whether or not the two backends agree. A suite
that checks only exit status reports them as passing for ever -- including on
a build with a known bug, which is how a 30/30 sweep was once obtained on an
unfixed port.

For those probes the comparison lives in `trace_compare.py`, and it is not
optional:

```sh
cc -O2 -I. -o /tmp/platform_rotation_ab \
  research/rust_ab/platform_rotation_ab.c bopl_rl.c -ldl -lm
( cd ../.. && /tmp/platform_rotation_ab 4 200 400 900 \
    net=<BoplSimNative.so> rust=<libbopl_sim.so> ) > /tmp/out.csv
python3 research/rust_ab/trace_compare.py /tmp/out.csv   # <- the real verdict
```

## The endgame was outside every gate

`map-sweep-3000` compares 124 maps for 3,000 ticks and is worth what it says
it is -- 417,009 snapshots, every authored map row exact. It is also
structurally blind to everything after level tick 3000, and sudden death does
not start until 7200. Two of the three divergences found in one session lived
past that line.

`make -C native/rl map-sweep-endgame` is the same probe run to 8,200 ticks, so
every map reaches sudden death and resolves its round. It costs about three
times the runtime, which is why both targets exist: keep 3000 as the quick
gate and run the endgame sweep before claiming a build is clean.

The first thing it found, on its first run: map 53, probe tick 6964 (level
7204), authored platform hierarchy 117 (`Big_Ice (3)`). The reference platform
was still completely at rest at level 7210 while the port had been drifting
*upward* since 7204 (`v=(0, +0.0000999)`, growing to `+0.0323`). Upward is the
tell. The first segment of `suddenDeathYSpeed` has value 0 with an out-tangent
of -0.4655, so a curve sampled just past t = 0 returns a small *negative*
speed and `homePosition += Vec2.down * speed * ...` lifts the home instead of
lowering it. The port was therefore past that platform's
`randomSuddenDeathDelay` several passes before the reference: the two
disagreed about the delay, not the curve.

The delays are drawn once when sudden death activates, one per platform. The
port drew them in `map.platforms` order, which is sorted by serialized
hierarchy number. The reference draws each one inside
`AnimateVelocity.ActivateSuddenDeath` as the components notice the flag, in
Updater registration order -- the scene's sibling order, which is what
`intro_order` records. On nearly every map the two coincide, which is why
this survived. On map 53 `LongSnowCapsule` (hierarchy 104) is first by
hierarchy and last by sibling order, so drawing in list order rotated every
delay on the level. Fixed by sorting on `intro_order` before drawing.

`MAP_AB_EXCUSE_SUDDEN_DEATH_FLAG=1` (set by the `map-sweep-endgame` recipe
only) excuses one field: the port raises the header's `sudden_death` flag one
observation before the reference. `GameSessionHandler.UpdateSim` activates it
at its own updater slot; the port does it from post-physics session work on
the preceding pass, with nothing but the observation write in between, so the
physics either side is identical and the rest of the sweep proves it. Delete
the variable the moment the observation writer is fixed. `map-sweep-3000`
never sets it and never reaches the flag.

Run the whole suite, comparator included, with

```sh
research/rust_ab/run_suite.sh          # or: make -C native/rl rust-ab-suite
```

which exits non-zero if anything fails. `run_suite.sh` carries the list of
trace-only probes; add to it whenever you add a probe that only prints.

Two rules `trace_compare.py` enforces, both learned the hard way:

* **Exact equality, never a tolerance.** Under `--features q32` both backends
  execute identical Q32.32 integer arithmetic, so any difference is a real
  divergence. One raw unit is 1/2^32 ~= 2.33e-10. A comparator written with
  `TOL = 1e-6` would pass every one-raw-unit bug this suite has ever caught,
  including the `Box.UpdateExtents` reciprocal drift (2 raw units) and the
  sudden-death negation order (1 raw unit).
* **A missing row is a divergence.** If one backend emits a sample for a tick
  and the other does not, that is an entity that exists on one side only --
  not a pair to skip. An earlier comparator skipped unmatched keys and so
  missed `platform_rotation_ab` map 0 / angle 200 entirely, reporting 29/30
  where the truth was 28/30.

If a trace-only probe would be better off asserting internally, convert it --
`magnet_ability_ab.c` is the model, comparing headers, players, slimes and
entities by `memcmp` and printing the first differing field. The external
comparator is a reasonable alternative for probes whose value is the readable
trace (`move_ab`, `state_ab`, `platform_rotation_ab`), but it must be wired,
not remembered.

The seventeen, as of this writing: `ability_push_ab`, `arrow_bounds_ab`,
`bow_ab`, `bow_ground_exit_ab`, `bow_phase_ab`, `dash_phase_ab`,
`entity_velocity_ab`, `keyboard_ab`, `move_ab`, `platform_ab`,
`platform_boundary_ab`, `platform_phase_ab`, `platform_rotation_ab`,
`recast_ab`, `state_ab`, `stock_respawn_ab`, `sudden_death_ab`.

See RUST_ROLLOUTS.md "Fidelity gate" for the validated scopes and measured
numeric tolerances. Do not infer fidelity from an all-identical dual-backend
trace unless `bopl_rl.c` includes the library-identity pool fix (`1cd34b2`).
