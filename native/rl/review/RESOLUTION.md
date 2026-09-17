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
- **#2 — fixed (report heading reversed the two values).** Growth/Shrink and
  Blink now self-hit when the initial muzzle is at or below C#
  `SceneBounds.WaterHeight` (`-11.3f`). Rust previously used the exported
  water collider top (`-11.95`), so muzzles in the intervening band fired an
  ordinary ray. The report's described gameplay consequence and cited C#
  lines were correct; only its opening value attribution was backwards.
- **#3 — fixed.** An initial Duplicator shot at or below
  `SceneBounds.WaterHeight` now replaces the ray result with the casting gun's
  own Player body and spawns the self-clone upward. Reflected water hits keep
  the existing recursive no-water-mask path.
- **#13 — fixed.** Growth/Shrink, Blink, and Freeze muzzle point checks now
  keep the last overlapping rounded rectangle in DetPhysics registration
  order. Their prior iterator `find` selected the first; Duplicator already
  implemented the source's overwrite-on-each-overlap behavior correctly.
- **#16 — fixed.** Generic-object ScaleChangers now retain scale by entity
  identity and synchronize concurrent effects after every write. Sequential
  hits therefore start from the victim's live `FixTransform.Scale`, while
  simultaneous Growth/Shrink effects compose additively in updater order as
  C# does instead of advancing independent accumulators from one.
- **#17 — fixed.** Generic circles/boxes and Smoke clouds now reproduce
  `DPhysicsCircle.Scale`'s Q32 grouping exactly: compute
  `(Fix.One / oldScale) * newScale` first, then multiply each collider
  dimension. The algebraically equivalent `new / old` and
  `(radius / old) * new` forms round differently in fixed point.
- **#19 — fixed.** Magnet's three held-item wall rays now exclude the held
  Macho boulder's rounded-platform collider. This mirrors the source
  temporarily deactivating an item GameObject around both `RaycastForWalls`
  and `UpdateHeldPosition`, preventing the boulder from clamping itself.
- **#20 — fixed (the report mislabeled this as Scale-ray/Blink).**
  Duplicator's platform spawn-position helper now replaces a degenerate
  rounded rectangle's rotated axes with world right/up when both extents are
  below `1/20`, exactly as `ShootDuplicator.PlatformSpawnPos` does.
- **#21 — fixed.** Blink's form entity is now included in all three death
  cleanup paths (Revival replacement, stock replacement, and elimination).
  It therefore receives the same removed/destroyed transition as every other
  ability body instead of remaining an active ray-blocking ghost.
- **#27 — fixed.** Scale rays now dispatch Teleport markers through their
  `ISpell.AddScale` semantics: only the private teleport-area/particle scale
  changes, clamped to the authored `0.1..6`; collider radius and inverse mass
  remain untouched. Revival's `AddScale` intentionally delegates to its
  BoplBody scale and retains the ordinary `0.1..5` collider path.
- **#14 — fixed.** When Magnet steals a player from an active ability form,
  the forced exit packet now relocates the restored slime to the stealing
  gun's current firepoint before ragdoll/hold state is applied. Ordinary
  slime victims retain their existing position, matching the source branch.
- **#8 — fixed.** Duplicated Rocket Engines now advance seven world units
  around the parent platform perimeter before attachment. The prior `0.3`
  argument was the separate surface offset, not Duplicator's placement
  distance.
- **#9 — fixed.** Duplicated Black Holes now enter DetPhysics' duplicate
  expiry list with the explicit source lifespan of 480 passes (eight seconds),
  rather than inheriting the generic 1800-pass object lifetime.
- **#10 — fixed.** A duplicated Black Hole now starts with
  `previous_target_size = 100`, matching the `BlackHole.Init` call made by
  Unity instantiation before Duplicator assigns its copied mass. Its first
  growth interpolation therefore has the source radius and pull/eat reach.
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
- **#6 — fixed.** Push entry now preserves the captured ordinary slime
  `groundedSpeed` through `PlayerPhysics.SyncPhysicsTo`; only its self-imposed
  and external velocity channels are cleared by `ControlPlatform.OnEnterAbility`.
- **#5 — fixed.** Every active Push pass now calls zero-input
  `PlayerPhysics.Move` immediately after the authored `0.8` external-velocity
  damping. This restores grounded friction, surface-local movement from
  inherited force, destroyed-ground detachment, and platform drop checks.

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
