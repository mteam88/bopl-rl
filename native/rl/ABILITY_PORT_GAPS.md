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

No ability-specific source gaps remain. All 39 catalog entries are `ported`.
Composition branches described as unmeasured in the catalog are translated but
have not each received a dedicated public NativeAOT/Rust differential.
