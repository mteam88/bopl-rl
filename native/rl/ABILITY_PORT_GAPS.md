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

1. **ID 15 — Revival (`partial`):** An active-form nonlethal Dash no-push callback is exact. Pickup replacement followed by immediate airborne activation of the acquired Revival, marker creation and exit is exact. The public pool still lacks an exact differential for simultaneous markers belonging to multiple live clones, copied equipped-ability state beyond the currently represented clone graph, marker interaction with Portal/Teleport/Black Hole/scale effects, and deaths during Time Stop or another active form.
