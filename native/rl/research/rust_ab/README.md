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
- `ability_spawner_ab.c`: ability-pickup-layer bodies after the authored 40-second
  pickup boundary; default RL configuration is 0/0, while the explicit .NET
  `BOPL_RL_ABILITY_SPAWNS=1` opt-in proves the original spawner remains live.
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
