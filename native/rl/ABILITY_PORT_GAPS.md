# Ability differential gaps

This queue started from the 29 `partial` and two `slice-only` entries in
`native/data/ability-coverage.json` at commit `2f3cf18`; an entry is removed when
the ability's remaining source behavior is ported wholesale and its catalog
status becomes `ported`.
Rank reflects how often the
missing branch should arise in unconstrained RL play: ordinary casts, airborne
casts and pickup-driven loadout changes come first; cross-ability contacts come
next; rare destruction, time-stop and pool-eviction boundaries come last. Each
entry names source behavior that is still missing or has not yet been audited
against an end-to-end NativeAOT/Rust differential through the public rollout
ABI. Per-branch differential coverage can trail a completed source port; that
distinction stays explicit in the catalog's verification text.

The semantic review found three structural source gaps that remain after the
localized 39-ability corrections:

- **Time Stop / transformed forms:** the scheduler still derives a shared
  `updater_dt` and has not yet represented each prefab component's independent
  `GameTime.FixedDeltaTime` opt-in. The confirmed Gust clock, Engine animator,
  grounded cancel, exit offset/velocity, and Gust/Time Stop no-gravity symptoms
  are fixed, but the remaining active forms need a per-updatable clock audit.
- **Revival multi-marker actors:** one player can own several full live
  `SlimeController` actors in C#. `PlayerSlot` can currently represent only one
  full actor, so simultaneous retained markers collapse to one revival body.
- **Clone cap with Revival:** C# counts all full actors plus abilityless clones
  against `MaxClones + 1`; the native count cannot match until the multi-actor
  player model exists.

One cross-ability scheduler gap also remains: deferred prefab initialization is
drained in Rust's type-category order rather than the single global hierarchy
order used by `Updater`. These items are marked `partial` in the affected
catalog entries; `native/rl/review/RESOLUTION.md` contains the implementation
plans and the disposition of every review finding.
