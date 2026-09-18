## Verdict

The port is a large, real regression on the RL rollout path. Under the **default loadout (Dash/Gust/Invisibility)** the NEW library runs **4.3–4.4x slower at 6 players (−77%)** and **5.2–5.4x slower at 2 players (−81%)** than the pre-port main build; both libraries are default-feature (f64 `Fix`) builds, so this is apples-to-apples (the `--features q32` claim in the brief is wrong — the 04:51 `.so` fingerprint and its `src/fixed.rs` string show `features: ["default"]`, same as OLD). Under **ability-heavy loadouts** the NEW library loses a further **0–8x on top of that**, worst case Grappling Hook at 25,191 ticks/s (12% of NEW's own default loadout, i.e. ~35x below OLD). The regression is dominated by *fixed per-tick cost*, not per-entity cost: NEW's per-tick intercept is ~1.9 µs vs OLD's ~0.25 µs (~8x), because `game_tick` now calls ~100 per-subsystem `*_tick`/`register_pending_*`/`*_integrate` routines unconditionally every tick regardless of equipped abilities, and `.text` grew 3.2x (634 KB → 2.01 MB) into one giant LTO-inlined `game_tick`. Separately, **six abilities crash the NEW library outright** and the 6-player default loadout aborts after ~215 episode resets. The plain (non-rollout) `bopl_step` World path is unaffected.

## 1. Default loadout, OLD vs NEW (100,000 ticks, 5 reps, interleaved, `taskset -c 11`)

Two independent interleaved sweeps; load avg 5.8–7.4 (bopl-sweep-v4 running), second sweep ran under a decaying load spike (15.4).

| players | run | OLD median t/s | NEW median t/s | slowdown | best-of-5 OLD | best-of-5 NEW | best-of-5 ratio |
|---|---|---|---|---|---|---|---|
| 6 | A | 953,877 (883k–967k) | 223,035 (222k–229k) | **4.28x (−76.6%)** | 967,032 | 228,573 | 4.23x |
| 6 | B | 816,807 (639k–881k) | 184,165 (179k–209k) | **4.44x (−77.5%)** | 881,034 | 209,485 | 4.21x |
| 2 | A | 1,935,489 (1.65M–2.16M) | 359,220 (346k–406k) | **5.39x (−81.4%)** | 2,163,894 | 405,687 | 5.33x |
| 2 | B | 1,621,291 (1.44M–1.78M) | 308,710 (262k–346k) | **5.25x (−81.0%)** | 1,777,943 | 346,007 | 5.14x |

NEW spread is much tighter than OLD (±3% vs ±15%), i.e. NEW is compute-bound, not noise-bound. C# NativeAOT reference ceiling for context: **17,888 ticks/s** at 6 players — NEW is still 11–13x faster than it (OLD was 47–54x).

**Crash:** NEW aborts at 6 players after ~215 episode resets (works at 100k ticks, dies at 120k) — `index out of bounds: the len is 25 but the index is 30` at `native/src/rollout/mod.rs:28354` (`self.entities[entity].flags |= E_PROJECTILE` in the `meteor_smears` loop: stale entity index after the entity table shrinks). OLD never crashes.

## 2. NEW with ability-heavy loadouts (all 3 slots = one ability, 6 players, 20,000 ticks, frequent casts)

Baseline (default loadout, same driver): **203,068 ticks/s**.

| slowest | ticks/s | % of default | max entities |
|---|---|---|---|
| 27 GrapplingHook | **25,191** | **12%** | **1020** |
| 32 WebSwing | 102,298 | 50% | 156 |
| 13 Smoke | 109,777 | 54% | 274 |
| 38 PortalGun | 123,617 | 61% | 29 |
| 21 Throw | 136,152 | 67% | 41 |
| 33 PokeBall | 141,662 | 70% | 70 |
| 24 Mine / 35 Tack / 23 Tesla / 26 Drill | 153k–158k | 76–78% | 49–84 |
| 14 Platform / 5 Engine / 3 Grenade / 10 Missile | 167k–172k | 82–85% | 43–49 |

Fastest: GoldMushroom 288k (142%), Dash 254k, Invisibility 253k, Rock 237k. Full list in `ability_results.txt`.

**Crashes (6 of 37 abilities abort):** 11 Spike (`fixed division by zero`, `src/fixed.rs:450`), 16 Roll (`Option::unwrap()` on `None`, `src/world.rs:572` — `jump_with_strength` assumes a platform), 29 Duplicator (`assertion left == right failed: stale platform bounds`, `src/world.rs:423`), and 20 Meteor / 22 Push / 36 Freeze (all the same `rollout/mod.rs:28354` OOB).

## 3. perf profile (self time, `perf record -F 4999`)

Ability-heavy mix (15 abilities, 26,987 samples), top 15:

| % self | symbol |
|---|---|
| 25.56 | `RolloutWorld::game_tick` |
| 25.45 | `physics::rope::Rope::try_step` |
| 9.73 | `__memmove_avx512_unaligned_erms` |
| 5.17 | `RolloutWorld::publish_entities` |
| 4.73 | `round` (libm) |
| 4.66 | `RolloutWorld::refresh_actor_entities` |
| 2.41 | `RolloutWorld::refresh_entities` |
| 2.01 | `Fix::fast_sin` |
| 1.43 | `RolloutWorld::fill_slimes` |
| 1.40 | `geometry::edge_hits` |
| 1.34 | `RoundedRect::raycast` |
| 1.17 | `RolloutWorld::player_tick` |
| 0.88 | `RoundedRect::closest_point` |
| 0.83 | `RolloutWorld::write_state` / 0.83 `floor` |
| 0.76 | `contact::rounded_rounded` |

Default loadout, absolute attribution (best-of-N: OLD 1034 ns/tick, NEW 4375 ns/tick):

| symbol | OLD % → ns | NEW % → ns | Δ ns/tick | share of the +3341 ns |
|---|---|---|---|---|
| `game_tick` | 16.36 → 169 | 30.63 → 1340 | **+1171** | 35% |
| `__memmove_avx512` | 3.37 → 35 | 15.06 → 659 | **+624** | 19% |
| `refresh_actor_entities` | 11.80 → 122 | 8.73 → 382 | +260 | 8% |
| `publish_entities` | <8 | 4.46 → 195 | +187 | 6% |
| `refresh_entities` | <8 | 3.65 → 160 | +152 | 5% |
| `Fix::fast_sin` + `round`/`floor`/`fmod` | 1.16+10.48 → 120 | 3.19+5.07+0.93+1.29 → 457 | +337 | 10% |
| `RoundedRect::raycast`, `edge_hits` | 2.53+4.72 → 75 | 2.99+2.60 → 245 | +170 | 5% |
| `fill_slimes`, `write_state`, `candidates` | 17.00 → 176 | 5.58 → 244 | +68 | 2% |

Player-count scaling (fit over 1–6 players): NEW ≈ 2.8 µs fixed + 0.75 µs/player; OLD ≈ 0.3 µs fixed + 0.22 µs/player. **The fixed term grew ~9x.**

## Ranked optimisation opportunities

1. **Gate the ~100 unconditional subsystem calls in `game_tick`** — `native/src/rollout/mod.rs:4161` (lines ~4172–4576: `register_pending_*`, `duplicated_objects_tick`, `gust_waves_init`, `meteor_waves_init`, `black_holes_tick`, `portal_shots_tick`, `rewind_recorders_tick`, `tacks_tick`, `poke_balls_tick`, `meteor_smears_tick`, `*_integrate`, …). Each is a loop over a Vec that is empty when the ability is not equipped, but the LTO-inlined blob is 2 MB of `.text` and pays I-cache + branch cost every tick. A per-world `active_ability_mask` computed at reset, or a "non-empty collections" bitset, would skip them. **~35% of the regression (≈1.2 µs/tick).**
2. **`publish_entities` + `observe_main` copy churn** — `rollout/mod.rs:28231` and `rollout/mod.rs:28174`. Four full `Vec::extend` passes over `self.entities` per tick copying 136-byte `CEntity` rows, plus `mem::take`/restore swaps. This is the `__memmove` 15%. Publish in one pass with a stable ordering index, or keep the three kind-groups in separate pre-sorted vectors. **~19% (≈0.62 µs/tick).**
3. **`refresh_actor_entities` reads 26 per-ability entity handles × 6 players every tick** — `rollout/mod.rs:28586`. Almost all are `None` for a 3-slot loadout. Iterate only the slots actually equipped. **~8%.**
4. **`refresh_entities` runs 6 sub-refresh sweeps unconditionally** — `rollout/mod.rs:28263` (`refresh_spike_entities`, `refresh_engine_entities`, `refresh_tesla_entities`, `refresh_platform_entities`, `refresh_entity_metadata`). Same gating fix as (1); `refresh_entity_metadata` re-walks the whole entity table. **~5%.**
5. **Grappling Hook rope simulation** — `native/src/physics/rope.rs:98` (`Rope::try_step`, 25% of ability-heavy self time) driven by `rollout/rope_abilities.rs:487` `grapple_ropes_update` / `:795` `ropes_update` / `:1204` `append_rope_entities`. 1,020 published entities for 6 grapples means every rope segment becomes a `CEntity` in the observation each tick. Cap or decimate segment publication; the per-segment `checked_div` at `rope.rs:136` is a known-unused division executed every segment, every tick. **Dominates the ability-heavy case.**
6. **libm calls in the fixed-point layer** — `round`/`floor`/`fmod` are 8% of NEW's default tick (vs 10.5% of a 4x smaller OLD tick, so ~2x absolute). `Fix::fast_sin` tripled. Worth `#[inline]`/intrinsic replacements.
7. **Correctness-first blockers** (these make any perf number provisional): `rollout/mod.rs:28354` stale entity index (4 abilities + default 6p), `world.rs:572` unwrap, `world.rs:423` stale platform bounds assert, `fixed.rs:450` div-by-zero.

## 4. Binary / state sizes

| | bytes | `.text` |
|---|---|---|
| OLD `libbopl_sim.so` | 1,174,904 | 633,662 |
| NEW `libbopl_sim.so` | 2,946,216 (**2.51x**) | 2,008,046 (**3.17x**) |
| C# `BoplSimNative.so` | 31,768,712 | — |

Tick-state size is **unchanged**: `sizeof(BoplRlEntity) = 136`, `sizeof(BoplRlSlime) = 136`, `sizeof(BoplRlState) = 728` (ABI v3 header shared by both; `CEntity` at `rollout/mod.rs:146` still 8×u32 + 13×i64). Entity counts at default loadout are comparable (25–32 both).

## 5. `native/tools/benchmark.py` (continuity with the 2.2M figure)

| library | steps/s (200k steps, 2 players, pinned) |
|---|---|
| OLD | 606,089 |
| NEW | **625,770** |

**No regression on this path** — it drives `bopl_step` on the plain `World`, not `RolloutWorld`, and is ctypes-overhead-bound (~1.6 µs/call). The historical 2.2M ticks/s does not reproduce for either library on this loaded box, so it is not a usable baseline today; the regression is confined to the RL/rollout ABI the trainer actually uses.

## Scratch files (left in place)

- `/home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject/native/target/old-libbopl_sim.so` — copy of the pre-port library
- `.../native/target/perf/` — `ability_bench.c`, `ability_bench`, `benchmark`, `libbopl_rl.so`, `run_ab.sh`, `ab_results.txt`, `ab_results_final.txt`, `ability_results.txt`, `new-libbopl_sim.so`, `dbg-build.log`, `perf-abilities.data`, `perf-abilities-flat.data`, `perf-default-new.data`, `perf-default-old.data`, `perf-default-new-cg.data`, `perf-dbg-default.data`
- `.../native/target-dbg/` — 247 MB `RUSTFLAGS=-g` build (line info was stripped anyway; safe to delete). It is the only untracked path `git status` reports; nothing was committed and `native/target/release/libbopl_sim.so` is untouched (still 04:51).

**Memory candidates for the parent:** (a) the 2026-09-17 ability-port merge 24e40abd regresses the Rust RL rollout path 4.3x at 6p / 5.2x at 2p vs pre-port main, root cause ~100 unconditional per-subsystem calls in `RolloutWorld::game_tick` plus 4-pass `publish_entities` CEntity copying; (b) `native/src/rollout/mod.rs:28354` (`meteor_smears` stale entity index) aborts the sim for abilities 20/22/36 and for the default 6p loadout after ~215 resets; (c) the `.so` at `native/target/release` is a default-feature build, not q32 — check `.fingerprint/*/lib-bopl_sim.json` or `strings | grep src/fixed`.