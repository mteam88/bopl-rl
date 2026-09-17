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
