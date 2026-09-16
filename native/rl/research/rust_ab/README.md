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
- `growth_airborne_ab.c`: a 500-tick Growth trace that hits an ordinary player
  while rising, then follows the complete ScaleChanger curve, enlarged fall
  and platform contacts.
- `shrink_airborne_ab.c`: the corresponding 500-tick Shrink trace through its
  negative ScaleChanger, reduced-body fall and platform contacts.
- `freeze_airborne_hit_ab.c`: a rising ordinary player caught by Freeze through
  `FrozenSolid` takeover, thaw, resumed fall and landing over 500 exact ticks.
- `duplicator_smoke_ab.c`: Smoke's three-copy fan through cloned-Grenade reset,
  rigid and Water contact detonation, randomized clouds and same-pass removal
  of a cloud born below WaterHeight over 450 ticks.
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
- `duplicator_pickup_ab.c`: deterministic AbilitySpawner collection replaces
  equipped Duplicator in slot three, then the acquired Revival registers,
  casts, creates its marker and exits exactly.
- `ability_push_ab.c`: one-sided overlap push on a normal slime touching an
  active Bow ability hurtbox.
- `aim_map.c`: full-charge arrow velocity per angle byte.
- `hit_ab.c`: charge and fire at the opponent, hit outcome and death cause.
- `bow_test.c`: Rust-only bow scenario used while porting.
- `platform_ab.c`: grounded platform transform overlap, jump-and-land player
  carry, and the map-2 AntiLockPlatform trajectory.
- `gust_ability_ab.c`: complete QuickSpell cast and exit, active Gust hurtbox,
  deferred Shockwave registration, exact captured targets, both time-zero
  impulses around the map spring updater, force-curve platform trajectories,
  and recovery. It compares every public player, slime, telemetry, and entity
  field for 360 ticks; static authored geometry allows one Q32.32 least bit.
- `gust_grenade_ab.c`: a released Grenade captured by Gust's fixed target set,
  both time-zero impulses, the complete force curve and subsequent item flight.
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
- `gust_rock_ab.c`: Gust's one-shot push of a pushable active Rock through the
  BoplBody fallback, subsequent rigid flight and lower-boundary death.
- `gust_roll_ab.c`: Gust's one-shot PlayerPhysics force on an airborne charging
  Roll, subsequent form motion and correctly phased lower-boundary death.
- `gust_bow_ab.c`: Gust's one-shot PlayerPhysics force against a live
  aim-oriented Bow form and its subsequent charging-form motion.
- `magnet_ability_ab.c`: Magnet's first-pass pull, rounded-platform impulses,
  player pull, grounded and airborne form motion, release recoil, landing and
  animation exits over 1,000 optimized-Q32 ticks. The active form centre allows
  2,048 raw Q32 units for the independent FixTransform offset reconstruction;
  axes, rotations, gameplay bodies and every interaction boundary remain exact.
- `magnet_item_ab.c`: a live Grenade pulled, picked up, held and launched by
  Magnet over 450 ticks, including fuse suppression and holder cleanup.
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

See RUST_ROLLOUTS.md "Fidelity gate" for the validated scopes and measured
numeric tolerances. Do not infer fidelity from an all-identical dual-backend
trace unless `bopl_rl.c` includes the library-identity pool fix (`1cd34b2`).
