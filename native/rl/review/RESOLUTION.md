# Ability-port semantic review resolution

Each entry records an independent reread of the C# oracle and Rust rollout
implementation. Commit hashes are intentionally left to Git history; every
fixed entry is committed separately with its finding id in the subject.

## Guns and spells

- **#1 — fixed.** C# keys permanent duplicated platforms by the casting
  Duplicator hierarchy. Each new platform from that gun now schedules the
  previous copy for the same 80-pass delayed removal (`60` shake passes plus
  the authored `20`-pass grace), while the new copy receives the permanent
  `i32::MAX` lifetime. The expiry path also mirrors duplicated-object shake
  displacement and removes the old platform from physics.
- **#37 — fixed.** Death now consumes an owned Revival marker reference even
  when the marker object was already destroyed by a Black Hole. C# leaves the
  destroyed component in `Player.RespawnPositions`, and `killPlayer` tests the
  list count before calling the stale indicator's retained `GetPos()`.
- **#7 — fixed.** Grounded Time Stop now checks button release during the
  initial `idleToFall` animation, matching `CastSpell.UpdateSim`'s release
  check on every pass rather than forcing the first twenty animation ticks.
- **#22 — fixed.** Gust shockwave curves now advance with raw `DT`, including
  during Time Stop. `Shockwave.UpdateSim` receives `simDeltaTime` directly and
  never calls `GameTime.FixedDeltaTime`; only the affected physics integration
  is frozen.
- **#23 — fixed for the confirmed forms.** Frozen airborne Gust and Time Stop
  forms now return before gravity, aerial steering, and velocity damping, as
  `QuickSpell` and `CastSpell` do. Other active forms retain their authored
  per-component frozen behavior rather than applying this exception globally.
- **#41 — fixed.** Rocket Engine start and end SpriteAnimator counters now
  stop when global time is stopped. The firing phase was already gated; all
  three phases now match the environment-owned C# components.
- **#38 — fixed.** Time Stop's exit offset now multiplies by the caster's live
  player scale, matching `CastSpell.ExitAbility`.
- **#6 — fixed.** Roll now scales its physics radius, lethal hurtbox radius,
  swept-contact radius, platform-hop geometry, and exit radius/offset by the
  player's live scale.
- **#4 — fixed.** Grounded Roll now calls the ordinary zero-input
  `PlayerPhysics.Move` path on every charging pass. This decays the inherited
  grounded speed, updates its retained surface-local coordinate, and performs
  the same live-platform/drop checks as C# before release.
- **#5 — fixed.** Grounded Roll exit now replaces the ordinary slime's
  grounded speed with the magnitude of the form's total `PlayerBody.Velocity`
  (self-imposed plus external), matching the `AbilityExitInfo` populated by
  C# instead of retaining pre-entry motion.
- **#18 — fixed.** Duplicated Engine and Tesla attachments now scale their
  authored surface offsets by the cloned object's scale before rebuilding
  composite-relative geometry.
- **#28 — fixed.** Teleport marker collider radius now remains at its authored
  value when the owner is scaled. The separate teleport-area radius continues
  to use the marker's logical particle scale, matching `TeleportIndicator`.
- **#29 — fixed.** Freeze animation exits and shared gun landing exits now
  restore the ordinary slime's scaled radius/body geometry, matching the
  already-correct scale-ray animation exit.
- **#32 — fixed.** Tesla nodes now retain separate raw FixTransform and
  clamped DPhysicsCircle scales. Surface/hitbox offsets use the raw caster
  scale while collider radius honors the authored `0.6..5` clamp.
- **#35 — fixed.** On the pass after a scale ray changes a spawned
  `ResizablePlatform` collider, its updater now recomputes inverse mass and
  moment of inertia from the live extents and bevel radius. This follows
  `ResizablePlatform.UpdateSim` calling `ResizePlatform` every pass rather
  than applying `BoplBody`'s ordinary linear scale adjustment.

## Projectiles

- **#1 — fixed.** `ThrowItem2.Fire` now applies caster scale to held-item
  dummy poses, release offsets, launch impulse, and the owner-immunity delay
  (with the source's `Max(scale, 1)` rule).
- **#2 — fixed.** Mine's instantiated payload offset now scales both authored
  local axes by the caster's live scale, matching `ThrowItem.Fire`.
- **#3 — fixed.** Guided missiles now inherit the caster's scale for both the
  spawn offset and live projectile collider/visual scale.
- **#4 — fixed.** Meteor's lethal smear half-width now multiplies by the
  caster's live scale; its travel-derived length remains unchanged.

## Cross-cutting

- **C1 — not a divergence.** The `TimeStop` metadata branch in
  `RlObservation.Entity` is unreachable in the C# oracle. `Write` emits only
  DetPhysics circles, boxes, rounded rectangles, rope segments, and beams;
  `Assets/GameObject/TimeStopSphere.prefab` contains only `Transform`,
  `SpriteRenderer`, `TimeStop`, and `FixTransform`, with no DetPhysics
  collider. NativeAOT therefore has no Time Stop entity row either, and its
  `obs[63]` / `obs[64]` remain zero exactly as Rust's do. Adding a synthetic
  Rust-only row would create the distribution split the report intended to
  prevent.

- **C2 — not a divergence.** The report conflated the generic `Orb` used by
  `UseOrb` with Beam's visual orb. Beam instantiates `orbAnim.prefab`, whose
  components include `DPhysicsCircle`, `Explosion`, animation, and
  `DisableIfNotHeld_deter`, but no `Projectile`. A fresh NativeAOT build
  confirms the live layer-18 Beam circle lacks `E_PROJECTILE`, matching Rust.

- **G3 — fixed.** `Player.Kill` returns immediately when `WonThisRound` is set
  (`sim/Gameplay/Player.cs`), before rumble, death accounting, telemetry, or
  killer credit. `RolloutWorld::kill` now performs the same guard before any
  exit/drop behavior. `native/src/player_life.rs` remains the independently
  verified accounting oracle; merging the two implementations would be a
  larger refactor and is not needed for this localized correction.
