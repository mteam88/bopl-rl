All four investigators are done. Here is the review.

---

## Summary

The merge itself is clean: `native/rl/bopl_rl.h` is byte-identical to main, `git diff 6d9bdb9 24e40abd -- native/src/rollout/mod.rs` contains **only** ABI v3 additions and nothing else, and no function or constant that existed in main's `mod.rs` is absent from the merged file (verified by set-diff of all `fn` and `const` names). Every v3 field is present and populated. The resolution is actually an improvement over main's hunks in three places (real pickup swap timer instead of a hardcoded 600, `pickup.ability` instead of a "last non-zero slot" heuristic, and a single generic `refresh_entity_metadata` pass instead of scattered per-site flag literals). I found **one confirmed ABI defect introduced by the resolution's design** (the time-stop observation feature is structurally dead under the Rust backend) and **one confirmed missing flag** (beam orb). On the cross-cutting game logic, the four parallel comparisons produced ten confirmed C#/Rust divergences, the highest-impact being an incorrect kill-credit fallback, a missing round-winner guard in `Rollout::kill`, a per-cast vs per-player smoke-cloud ring, and category-ordered rather than hierarchy-ordered same-tick object registration (which is exactly the tie-break ordering the task asked about).

---

## CONFIRMED — ABI / merge (Part 2)

**C1. `obs[63]` / `obs[64]` (time-stop remaining) are permanently zero under the Rust sim.**
`native/rl/puffer/bopl_puffer.h:460-463` derives the time-stop countdown by scanning for the one entity satisfying `header.player_time_scale == 0 && (flags & TIMER_KNOWN) && clock_rate > 0`. In the C# oracle exactly one branch produces that combination: `sim/Gameplay/RlObservation.cs:263-270`, the `TimeStop`-component row, which sets `clockRate = Fix.One` unconditionally. Every other C# timer branch uses `GameTime.PlayerTimeScale` (zero during a stop) and the pickup branch explicitly forces `Fix.Zero` when stopped (`RlObservation.cs:255`). The merged Rust has no time-stop branch at all in `refresh_entity_metadata` (`native/src/rollout/mod.rs:28276-28435`): pickups get `pickup_clock = 0` when stopped (`:28314-28318`), grenade/smoke fuses and smoke clouds get `clock_rate = self.player_time_scale.raw()` = 0 (`:28375`, `:28385`). No row can ever have `clock_rate > 0` while `player_time_scale == 0`. So `time_stop_known` never fires and both observation features read 0 for the entire episode under the Rust backend while the Unity/NativeAOT backend supplies a real countdown. This is a train/eval distribution split on two of the 66 prefix features, active on every tick of every time stop. Not listed in `FULL_COMPATIBILITY.md` (no hit for `clock_rate` or `obs[63]`).

**C2. Beam orb is missing `E_PROJECTILE`.**
`sim/Gameplay/Orb.cs:119,261` — the Orb prefab carries a `Projectile` component (`projectile = GetComponent<Projectile>()`), so `RlObservation.cs:239` sets `1u << 8`. The merged Rust sets only `E_EXPLOSIVE` on beam orbs (`native/src/rollout/mod.rs:28405-28409`). Entity feature `row[26]` (`bopl_puffer.h:565`) is therefore 0 for every beam orb. Affects every Beam cast.

Two things the task asked me to flag that turned out **correct**, worth recording so they are not "fixed" into bugs:
- `item.ability != ABILITY_MINE` gating `E_EXPLOSIVE` and `reserved[0]` (`mod.rs:28360-28371`) is right: `Mine` is `MonoUpdatable, ICollisionCallback` only (`sim/Gameplay/Mine.cs:5`), with no `Grenade` component and not an `IGrenadeDetonation`, so C#'s `explosive` test at `RlObservation.cs:237` is false.
- Missiles correctly lack `E_PROJECTILE`: `sim/Gameplay/Missile.cs:4` has no `Projectile` component. Tack/Mjolnir/Poke Ball correctly have it (`BananaPeel.cs:22`, `Mjolnir.cs:31`, `PokeBall.cs:27`).
- `E_HITBOX` on drill hitboxes only is exact: the `Hitbox` component exists solely under `HitboxCombo` (`sim/Gameplay/HitboxCombo.cs:70`), used only by `Drill.cs:53`.
- Engine rockets and spike attacks carry `reserved[0] = 0` because `refresh_engine_entities`/`refresh_spike_entities` overwrite the whole row with `..CEntity::default()` (`mod.rs:28523`, `:28562`) and the metadata pass never restores it — but this is **correct**, since `RocketEngine` and `SpikeAttack` are `IPlatformAttachable` with no `Ability`/`AbilityMonoBehaviour` ancestor.

---

## Entity-emitting sites (effective values after `refresh_entity_metadata`)

| `mod.rs` | object | `reserved[0]` | flags set | verdict |
|---|---|---|---|---|
| 2771, 3784, 3919 | ability prefab circle/box collider | `ability` | TRIGGER + CDK (+CD if Rock) | ok |
| 2978 | slime body / ability collider | body 0; ability `ability` | ±ACTIVE/ENABLED/TRIGGER, CDK on ability rows | ok |
| 3107 | map terrain platform | 0 | ACTIVE\|ENABLED | suspected (S4) |
| 3488, 20455 | ability pickup / duplicated pickup | `pickup.ability` | PICKUP\|TIMER_KNOWN, clock_rate 1/0 | ok |
| 4760 | portal shot | owner's Bow slot else 0 | PROJECTILE | ok |
| 6427 | rock body | `ABILITY_ROCK` | CDK\|CD | ok |
| 8328 | meteor smear | 0 | PROJECTILE | suspected (S2) |
| 9052, 19964, 28445 | tesla node | Tesla or Missile slot | SPARK_NODE\|EXPLOSIVE | ok |
| 9259/19879, 9449/19801 | engine rocket, spike attack | 0 | COMPOSITE (+EXPLOSIVE for spike) | ok |
| 9833, 28477 | tesla lightning line | 0 | TRIGGER | suspected (S3) |
| 10371, 19508 | black hole | 0 | TRIGGER | suspected (S3) |
| 11424 | drill hitbox | `ABILITY_DRILL` | HITBOX\|CDK | ok |
| 11813 | beam orb | 0 | EXPLOSIVE | **C2: no PROJECTILE** |
| 12448 | beam segment | 0 | TRIGGER\|EXPLOSIVE | suspected (S3) |
| 13246 | macho boulder | 0 | ACTIVE\|ENABLED | ok |
| 13733, 13761 | revival / teleport marker | 0 | ACTIVE\|ENABLED | ok |
| 17166, 20516 | released/duplicated throw item | `item.ability` (not Mine) | EXPLOSIVE (not Mine); PROJECTILE for Tack/Mjolnir/Poke; TIMER_KNOWN+fuse for Grenade/Smoke | ok |
| 19562, 19630, 19706 | duplicated player target, lethal rock | 0 | — | ok |
| 21997 | duplicate platform | inherits `..source_row` | ACTIVE\|ENABLED | ok |
| 22562 | `spawn_ability_platform` | `ABILITY_PLATFORM` | ACTIVE\|ENABLED | ok (main's e5a78eb credit preserved; `refresh_platform_entities` at `:29728` mutates in place and does not clobber it) |
| 22958, 23010 | arrow | `ABILITY_BOW` | PROJECTILE | ok (matches e5a78eb) |
| 23057, 23103 | mine | 0 | — | ok |
| smoke clouds | 0 | FLAMMABLE_SMOKE (+TIMER_KNOWN while igniting) | ok |

`E_DETACHED` (1<<5) is declared at `mod.rs:237` and never set anywhere; the C# oracle also never writes `1u << 5`. Dead on both sides — harmless.

`native_check.c` is correct: `sim/Native/NativeApi.cs:135-136` exports `bopl_rl_world_capacity` returning 4096, and the new test logic (`native/rl/native_check.c:91-102`) asserts "capacity > 1 ⟹ second open must succeed, capacity ≤ 1 ⟹ must fail", which matches `NativeApi.cs:166-168`. One nit: `native_check.c` hard-fails on a missing `bopl_rl_world_capacity` symbol via `RESOLVE`, while `native/rl/bopl_rl.c:361-363` tolerates absence and defaults to 1. Not a bug for this target.

**Merge verdict: accept.** Nothing from either parent was lost. C1 and C2 are pre-existing/design gaps surfaced by the resolution, not regressions from it.

---

## CONFIRMED — cross-cutting game logic (Part 1)

**G1. `projectile_credit` drops owner credit on Explosion/LethalTerrain-layer rows.** `mod.rs:8182` falls through to `(ENVIRONMENT, true, CAUSE_OTHER)`; C# `PlayerCollision.cs:360-367` reads `GetComponent<Hitbox>().GetPlayerId()` for layers 16/18 — the owning player. The Dash path does this right (`mod.rs:16264-16265`); `projectile_credit`, used by Meteor (`:8086`) and Roll (`:27241`) swept contacts, does not. A Meteor or lethal Roll hitting a dropped Rock body (pushed as a layer-16 row carrying `owner`, `:19565`, `:19709`) credits ENVIRONMENT: victim gets `deaths_environment` instead of `deaths_opponent` and the Rock owner's `kills` is never incremented (`:27809`).

**G2. Uncredited missile explosion maps to `ENVIRONMENT` at one of three sites.** `mod.rs:8144-8152` uses ENVIRONMENT where `:16288-16292` and `:26130-26134` correctly use `0`. C#: `Missile.cs:185` never assigns `PlayerOwnerId`, so `PlayerCollision.cs:371` yields 0 and `RlTelemetry.cs:42` records **deaths_other**. `:8145` yields deaths_environment instead. Fires on every guided-missile detonation that kills a Meteor or Roll form.

**G3. `Rollout::kill` is missing the round-winner guard.** `Player.cs:443-447` returns immediately for `WonThisRound` — no death, no telemetry, no kill credit. The verified sibling port honours it (`native/src/player_life.rs:50-52`); `mod.rs:27722-27811` never reads `PlayerSlot::won`. Since `session_result_tick` sets `won = true` before the rest of the pass, a winner killed later in the terminal tick is recorded dead — corrupting the final observation, which is the reward tick.

**G4. Smoke cloud ring is per-cast in Rust, per-player in C#.** `ThrowItem2.cs:43,112` — `effectBuffer` is a field of the player's Smoke ability instance, capacity 40, and `SmokeGrenadeExplode2.cs:44` adds every cloud from every cast. `mod.rs:24277-24297` counts/evicts only clouds with matching `source_group` (per thrown item) against a hardcoded 40, so with 4 clouds per cast the cap is unreachable and clouds are **never** evicted. `FULL_COMPATIBILITY.md:2366` asserts the wrong contract ("each cast owns its own 40-cloud ring"). ~11 casts to diverge.

**G5. Same-tick registration order: hierarchy-sorted in C#, category-sorted in Rust.** `Updater.cs:270` sorts all pending updatables by `HierarchyNumber` then inits/appends in that one global order. `mod.rs:4172-4188` registers by a fixed category sequence (arrows → duplicated pickups → picked abilities → mines → missiles → …), each pool by slot index; hierarchy is allocated at spawn but never used to order registration. A mine planted before an arrow is fired in the same tick registers Mine→Arrow in C# and Arrow→Mine in Rust, changing collider registration order, DetPhysics pair discovery, the callback queue at `DetPhysics.cs:3197-3200`, and therefore **which player dies first on a tie**. The port fixes only the Arrow-vs-acquired-ability pair explicitly (`mod.rs:4174-4178`). This is the direct answer to task item (e): contact dispatch order matches (deferred to end of `Simulate`, pair-discovery order, A before B), but spawn order does not.

**G6. Rope-icon override suppresses the death drop in C#, never in Rust.** `SlimeController.DropAbilities` resolves the dropped ability by indicator sprite and spawns nothing on index −1 (`SlimeController.cs:296-309`); rope abilities override the slot icon with `ReelinRopeIcon` / `CloseToPlayerRopeIcon` (`:754-771`), neither of which is in `abilityIconsFull`. Rust drops unconditionally (`mod.rs:28022-28025`). Diverges by an entity, three hierarchy slots and an RNG draw on most rope-ability eliminations. (Rests on the two UI sprites not being catalog members — the `!= -1` guard exists precisely for that case, but I could not open the Unity asset.)

**G7. Simultaneous pickup collection.** `DetPhysics.cs:3196-3200` dispatches every queued callback with no destroyed re-check and `DynamicAbilityPickup.OnCollide` destroys only at the end, so two overlapping slimes both receive the ability. `mod.rs:3691-3694` takes the first player in index order and `break`s.

**G8. Time-stop teardown ignores `HasGameEnded()`.** `TimeStop.cs:72-86` makes `End()` a no-op after the game ends, leaving scale at 0 and protection bits set; `mod.rs:9589-9595` unconditionally restores. Also no analogue of `GameSessionHandler.cs:632-638`. Matters whenever a round is won during a stop.

**G9. Overlapping stops resolve in vector-slot order.** `mod.rs:9582-9602` iterates `time_stop_effects` by slot, and spawning reuses freed slots (`:8776-8783`), so a newer stop can occupy a lower index than a live one — inverting C#'s hierarchy order and protecting the wrong team.

**G10. Solo-session termination.** `GameSessionHandler.cs:519-523` wins the game for a 1-player session on death; `mod.rs:28131` guards with `players.len() > 1`, so a 1-player episode runs to `self.limit`. Debug-only.

**Verified matching** (do not re-investigate): pickup cooldown-on-entry (`timers[2] = 100000` + cooldown wipe), slot-2 replacement rule, mid-cast replacement being unreachable and leaving the orphaned collider alive, the 7 s swap period and RNG draw order, death-drop position/direction/RNG ordering; the full death-classification tree and `cause_of_death` encoding, teammate kills counting as `deaths_opponent`, suicide-by-own-ability counting as `deaths_self` **and** crediting the victim's own `kills`; **no team-protection or spawn-grace window exists in the C# source at all** (`PlayerCollision.Mask_OnSpawnBuff` is declared and never set — there are no time constants to compare); the global-vs-per-owner time-scale split, unconditional tick/`simulation_time_since_level_loaded` advance (`Updater.cs:310-312` advances both time fields in lockstep, so Rust setting both to `tick * DT` is correct), `clock_rate` for grenade/smoke fuses and pickups, cooldown freezing and `stopped_during_time_stop`; Engine (4) / Tesla (2) / Bow (20) / Platform (1) pool capacities and FIFO eviction; deferred-destroy semantics.

---

## SUSPECTED

- **S1.** No per-body `wasKilledThisFrame` latch in Rust (C# has one at `PlayerCollision.cs:28,76,453,470`). All call sites I traced are guarded by `playable()`, so I could not construct a double-count, but the structural divergence is real. A targeted test (two lethal contacts in one tick on a body with a revival marker) would settle it.
- **S2.** Meteor smear carries `E_PROJECTILE` but `reserved[0] = 0` (`mod.rs:8328`, `:28352-28356`); if the smear is parented to the Meteor ability, C# `SourceAbility` would credit it.
- **S3.** Tesla lightning lines, beam segments and black holes are lethal triggers with neither `E_CONTACT_DAMAGE_KNOWN` nor `reserved[0]`. C# sets `1u << 14` for anything with an `Ability` on itself **or any ancestor** (`RlObservation.cs:244-249`); the Rust rule is narrower (`LAYER_PLAYER && reserved[0] != 0`, plus drill hitboxes). Main's e5a78eb validated agreement only on the abilities exercised on maps 0-4, which predates ~35 ported abilities.
- **S4.** Map terrain platforms get `reserved[0] = 0`; C# `PlatformAbility` falls back to "any platform slot" for any `ResizablePlatform`. Safe iff map terrain is not a `ResizablePlatform` — e5a78eb's whole-episode agreement suggests it is not.
- **S5.** Portal clocks double-apply the time scale (`mod.rs:4806`, `:5727` multiply `updater_dt`, already `DT * player_time_scale`, by the scale again). Harmless while the scale is binary; diverges under the fractional scales the reference harness drives.
- **S6.** `DetPhysics.cs:2422` tests `p1.collisionCallback.IsDestroyed` when queueing **p2**'s callback — a source bug whose Rust counterpart I did not find.
- **S7.** `session_result_tick` runs before the per-player loop (`mod.rs:4207` vs `:4216-4222`); C# runs `Player.UpdatePlayer` for all players before any updatable, so deaths caused by `UpdatePlayer` are seen in the same tick in C# and one tick later in Rust.

---

Part 1(f) has no answer on the Rust side by construction: the rollout has no round/level transition — `session_result_tick` sets `terminal` and the episode ends, with no analogue of `Updater.PreLevelLoad` or `LoadNextLevelScene`. That is scope, not a defect, for single-round episodes. The C# reset contract is documented at `Updater.cs:65-82` and `GameSessionHandler.cs:401-429` should it ever be needed; notably nothing there resets ability cooldowns (cooldown state dies with the destroyed prefabs).

**Candidate memories for you to register:** (1) `native/src/player_life.rs` is the reference-verified `Player.Kill` port while `rollout/mod.rs::kill` is a second independent implementation of the same accounting — they disagree, and likewise `projectile_credit` (`mod.rs:8124`) is the older credit dispatch versus the verified Dash block at `:16264`. (2) `SlimeController.DropAbilities` resolves the dropped ability *by indicator sprite*, and rope icon overrides make that lookup fail — a non-obvious behaviour likely to bite anything else reading the last-slot ability. (3) The authoritative Bopl tick order is `sim/Gameplay/Updater.cs:234-354`; all collision callbacks are queued during `DetPhysics.Simulate` and dispatched together at `DetPhysics.cs:3197-3200`; `RingBuffer.Add` (`RingBuffer.cs:41-64`) evicts FIFO only when every slot is live.