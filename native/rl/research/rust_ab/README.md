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
- `dash_player_ab.c`: an ordinary target overlaps the Dash ray origin, receives
  its one-sided `SlimeController.OnCollide` push, and survives the pass-through
  before the Dash completes its normal endpoint and exit.
- `dash_rope_ab.c`: a live map-3 Web Swing bottom endpoint is carried into
  Dash, constrained by Dash's own `PlayerBody`, straightened at exit and handed
  back to the ordinary slime with exact rope-solver history through tick 300.
- `dash_grapple_rope_ab.c`: a close-player Grappling Hook attaches a grounded
  victim at its top endpoint before that victim Dashes; gameplay state and all
  rope lifecycle fields match through exit, with bounded long-strand geometry.
- `dash_bow_ab.c`: a Dash ray crosses an overlapping active Bow hurtbox and
  verifies the form's nonlethal `PlayerCollision` callback does not receive the
  ordinary slime's one-sided push.
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
- `time_stop_arrow_ab.c`: a live Bow projectile across Time Stop entry, all
  690 frozen physics passes, and its first restored gravity/flight pass; also
  verifies source-derived Arrow prefab hierarchy allocation.
- `time_stop_bow_ab.c`: an unprotected Bow form charging across all 690 Time
  Stop passes, followed by restored release, form exit and Arrow registration.
- `ability_spawner_ab.c`: ability-pickup-layer bodies after the authored 40-second
  pickup boundary. The default RL configuration is 0/0; with the shared
  `BOPL_RL_ABILITY_SPAWNS=1` opt-in it compares the first spawn, seven-pass
  collider delay, launch and hover flight in NativeAOT and Rust.
- `ability_pickup_collect_ab.c`: keyboard pursuit of that first live pickup,
  exact trigger callback at level tick 2506, third-slot replacement, the
  one-pass destroyed row, and the newly instantiated ability collider that
  joins DetPhysics on tick 2507.
- `black_hole_ability_ab.c`: BlackHoleClap form entry/exit, persistent hole
  spawn and growth, whole-map rounded-platform pull, center-only platform
  destruction, no-effect player kill, and stock-clone reconstruction over 500
  NativeAOT/Rust ticks.
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
  launches it at the authored Missile speed while `BoplBody.Hold` expires.
- `teleport_ability_ab.c`: first and second instant casts, deferred portal
  marker registration, release-driven readiness, caster transfer, inactive
  effect interval, RNG advancement, and exact reactivation.
- `teleport_multi_ab.c`: a three-player two-way portal swap with a teammate at
  each query circle. It also verifies sorted-team spawn assignment, authored
  teammate offsets, two simultaneous effects, and entity order.
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
