# Observation layout v3 plan (2026-09-14)

Outcome of a three-round design review between two bb threads reading this
checkout: `thr_yu8jkaqfic` (Codex gpt-6-astra high, rift child `design-astra`)
and `thr_9d9vr3upy6` (Cursor grok-4.6 high, rift child `design-cursor`).
Every code claim below was checked against the C# and the C environment.
Goal: an input layer that stays correct for any of the 39 abilities, crowds
(duplicated grenades, 16 clones, rope abilities), mid-episode pickups, and
per-body inventories, without a kernel-backward redesign.

## Measured entity histogram (fills THROUGHPUT.md §E TODO)

`native/artifacts/rl/coverage <game> native/fixtures/movement.rep <log>`,
480 ticks per case, random input, all 124 maps, 235 cases, 0 failures.

| Slice | rows per tick |
|---|---|
| 196 mixed loadouts, 1–6 players | median 27, p90 34, max 89; 10 cases > 48, 0 > 96 |
| 1 player, static map only | 8–26 |
| 6 players, all slots one ability | Grappling hook 363, Smoke 128, Web swing 90, Tesla 75, Drill 70, Mine 49; all others < 48 |

Rope abilities dominate because every rope segment is a row. Slimes never
exceeded 6 here (no duplicator crowd in the sweep); `Constants.MaxClones=16`.

## What is wrong with v2 for the general game

- `bopl_encode` takes the first slime with the player's owner id for slot
  state. Revival clones carry copied ability graphs (`PlayerCollision.SpawnClone`),
  duplicator clones are ability-less (`SpawnAbilityLessClone`), and pickups
  mutate one specific slime (`AddAdditionalAbility` replaces index 2 when the
  slime already has 3). First-match can show a dying body or the wrong clone.
- 39-hot ability ids per slot (3 × 46 per player) do not transfer and cost 828
  floats of the 3,298.
- 96 entity rows × 21 with center-distance sort; no counts survive truncation,
  so 20 grenades and 100 grenades look alike once capped.
- No timers on world objects. Age from spawn tick is the wrong feature:
  grenade fuse runs on `PlayerTimeScale`, time stop is per owner
  (`GameTime.cs:23`), pickups pause in time stop, duplicated grenades get
  staggered fuses (`ShootDuplicator.cs:301`), smoke's fuse starts at ignition.
- Invisible opponents lose all owned entities (`bopl_hidden`), including
  grenades and smoke a human would see.
- Slime cap 12 < `MaxClones` 16 **per player** (`PlayerCollision.cs:226`), so a
  6-player duplicator game can hold 102 bodies; eval aborts on overflow
  (`environment.py:124`).

## Agreed design

### Ability identity
Shared embedding table `E[40, 64]` (0 = none/terrain) gathered inside the
fused kernel and added to the 64-d projection before ReLU, for slot rows and
entity rows. Ids travel as exact integer-valued floats in the row; never as an
ordinal scalar into a projection. Both reviewers agreed after round 1.

### Wire (bump `BoplRlHeader.version`, both writers)
- `BoplRlEntity.reserved[0]` = source/granted ability id (0 none).
- `reserved[1]` = remaining ticks of the object's next event if known.
- Entity flags add: `timer_known` (64), component-presence bits read at write
  time: pickup, projectile, spark node, flammable smoke, explosive
  (`Grenade`/`Explosion`/`IGrenadeDetonation`), hitbox; `contact_damage` and
  `contact_damage_known` from the collider's `Ability.KillPlayersOnContact`
  (the field `PlayerCollision.cs:418` tests; Roll toggles it at runtime).
- Entity clock rate: 1 float, the rate at which `remaining` decreases in ticks
  per tick (0 in time stop unless the owner is protected; `PlayerTimeScale`
  for grenades). Replaces a single pause bit (Astra's objection, verified).
- Slime flags add `abilityless` (32).
- Rope geometry stays raw on the wire; the encoder collapses (Astra).
- No hand-authored trait schema, no prefab defaults, no generated JSON.

### Rows (capacity × width)

| Block | Cap × width | Contents |
|---|---:|---|
| Prefix | 1 × 66 | v2 prefix 10; round clock: seconds to sudden death / 120, sudden-death active, seconds since sudden death / 30, ticks to episode cap / cap; pre-truncation counts per source id 1..39 (skip id 0, fixed scale); n_nonstatic, n_pickup; 8 angular-sector counts of nonstatic objects; time-stop remaining + known; controlling-slime pickup-target slot |
| Self (flat) | 1 × 20 | v2's 16 scalars + spatial-known, invisibility elapsed, ability elapsed, body count |
| Players (set) | 6 × 20 | same 20, shared 20→64, max |
| Slots (set) | 24 × 14 | valid, self, ally, slot index, timer/cd, cd/60, ready, over, active, charge/4, charge-known, owning body dx/64, dy/64, ability id → E[id] |
| Entities (set) | 64 × 34 | valid, kind 5, self/ally/enemy, rel pos/vel/axes 8, radius, enabled/trigger/composite/layer, remaining/T, timer-known, clock rate, 6 presence bits, contact-damage + known, source id → E[id] |
| Slimes (set) | 32 × 30 | v2's 28 + abilityless + spatial-known |
| Previous | 1 × 12 | unchanged |

Exactly 3,690 floats at these caps (24 slot rows), the same order as v2's 3,298,
with 828 floats of one-hot ids replaced by rows that carry information.
Fusion input becomes H + 4×64 (player, slot, entity, slime max-pools).

Slot rows: one per (slime, slot) for every slime with `abilities.Count > 0`,
including suppressed-input and ragdoll bodies (Astra: inventory is not gone
because input is suppressed). Body association is the body's relative
position on the row (Cursor); Astra notes overlapping clones then alias,
which I accept as a known limitation rather than adding a hierarchical
encoder. The player block's "controlling slime" is the nearest slime of that
owner with abilities and without `suppressInput`, else any with abilities;
never first-match.

Entity ordering: surface distance (center distance minus radius; segment
distance for ropes and beams), pickups force-inserted, each rope collapsed to
its endpoints plus at most two interior spans (three rows). Overflow counted,
never fatal.

Visibility: an invisible opponent loses coordinates and slime rows and the
entities that are its body/hurtbox hierarchy; independent owned objects
(grenades, spark nodes, smoke, pickups) stay. Disabled-but-alive colliders
stay (ignited smoke disables its collider, `FlammableSmoke.cs:66`).

Pooling: max only. Learned max+sum is experiment arm B after v3 parity; both
reviewers agreed the sum backward is not free on the current winner-only
`bs_fused_backward` (needs a ReLU sign mask or a recompute).

Actions: unchanged heads (aim 256, buttons 16, hold 5); buttons are
jump/slot1/slot2/slot3 already on the wire. Mask: bow-only forced release
keyed by the controlling slime's slot id, not a per-ability charges table.
Roll, ThrowItem2, MagnetGun, Beam, Drill, Katana all read held input and a
forced release there would be illegal relative to the game.

### Leave out of v3
Attention, learned sum, threat-weighted sort, prefab trait schema, spawn age,
owner one-hot, H=256 / 2 layers, universal charge mask, loadout selection,
dropping disabled colliders, hiding all owned entities of an invisible player.

## Migration order and gates

1. Wire v3: header version, `reserved[]` semantics, new flags, clock rate;
   C# `RlObservation.cs` and the Rust writer both emit it. Gate: C# vs Rust
   observation diff on the existing goldens is unchanged for old fields.
2. `bopl_layout.h` v3 + `research/layout.py` + CPU reference encoder
   (`bopl_set_encoder.h`) + `bopl_encode` rewrite (controlling slime, slot
   rows, rope collapse, surface distance, pickup insert, counts, visibility).
   Gate: synthetic fixtures encode as specified: 20 duplicated grenades,
   16 clones, 3 hooks, invisible opponent with live smoke, pickup mid-episode.
3. CUDA `bopl_set_encoder.cu`: N=64 tiles, KSTEPS 5, new slot pool, E gather
   pre-ReLU for slots and entities, static_asserts updated. Gate: forward and
   gradient parity with the CPU encoder on the fixtures.
4. Eval overflow non-fatal (`environment.py:124`), matching train.
5. Parity study on Dash/Bow/Platform, H=128, 1 layer: v3 vs v2, multiple
   seeds, Elo on pool-v5; obs change and mask change measured separately.
   Gate: no regression beyond seed noise; throughput within 10% of v2.
6. All-ability curriculum when the Rust port (`thr_snnrmv2u4c`, branch
   `ability-port`) lands: random 3-slot loadouts from the implemented pool,
   rare ids oversampled, pickups on; search bot as eval anchor only.

## Matthew's review (2026-09-14)

- Slime cap: `MaxClones` is per player, not global. Cap raised to 32 above.
- Aiming: bow, grenade, gun, spark and more aim. The 2026-09-13 aim study
  found autoregressive buttons-then-aim hurt every metric and halved learner
  throughput, while Fourier K=4 aim helped (Elo 575 vs 377). v3 keeps the
  independent Fourier aim head; the decision about *which* slot to press
  conditions on the same hidden state. Revisit AR only as an arm after v3
  parity, with the slot buttons conditioning aim rather than the reverse.
- Round clock: v2 has only episode fraction and sim time / 120 in the prefix.
  Sudden death starts at `SimTimeSinceLevelLoaded > 120 s`
  (`GameSessionHandler.cs:501`) and platforms sink with acceleration steps
  (`AnimateVelocity.cs:133`). Prefix gains seconds-to-sudden-death, a
  sudden-death flag, seconds since it started, and ticks to the episode cap.
  The wire must carry `SimTimeSinceLevelLoaded` and `SuddenDeathInProgress`.
- Invisibility: an invisible player's sprite alpha follows
  `Invisibility.SetAlphaBasedOnPlayerInvisibiltyTime`, which fades with
  `timeSinceEnteredAbility` (reset on every ability enter/exit,
  `SlimeController.cs:188/200`) and fully in the last second. So an invisible
  opponent is visible while aiming or using any ability, and the writer must
  expose that: hide coordinates only while `isInvisible` and the effective
  alpha is below a threshold; keep `timeSpentInvisible` and
  `timeSinceEnteredAbility` on the player row (both already on the wire).
- Grappling hook: `Rope.nrOfSegments` is 5 per rope, so one hook is 4 rows;
  the 363 in the sweep is 18 hooks across 6 players × 3 slots. Encoder
  collapses each rope to its endpoints plus up to 2 interior spans (3 rows).

## Where the reviewers still differ
- Slot/body binding: Cursor accepts position binding; Astra wants an explicit
  body reference. Chosen: position binding in v3, revisit if crowded fixtures
  show inventory confusion.
- Timers: Astra says multi-phase components (smoke before/after ignition,
  MagnetGun holds) need more than remaining+rate. Chosen: remaining is the
  next event; phase is visible through presence bits and `enabled`; extend
  per component when a fixture shows a gap.

## Implementation notes

Implemented on `layout-v3` in migration-order commits beginning at `3721f8a`.

- The exact row arithmetic is 3,690 floats: prefix 66, self 20, players
  6x20, slots 24x14, entities 64x34, slimes 32x30, previous action 12. The
  plan's earlier "about 3,300" estimate was corrected above.
- Counts use a fixed scale of 64. General object timers use 600 ticks; the
  prefix time-stop timer uses 300 ticks. Player feature 5 retains v2's
  saturated `alive_bodies / 6`, while feature 19 carries the actual visible
  slime-row count divided by 32.
- The wire had no pickup-target field. The writer records the target slot in
  otherwise unused slime flag bits 6-7, with a dedicated known bit. Bow
  charge-known follows the known Bow ability id even while inactive; charge
  and fired state remain zero unless its runtime object is active.
- The final three-row rope rule in Matthew's review overrides the earlier
  eight-span sentence. The fixture feeds three four-span hooks and verifies
  that exactly nine rope rows survive.
- The crowded fixture starts at the shared `BoplRlState` ABI boundary. It has
  20 duplicated grenades, 16 mixed inventory/ability-less clones, 27 slot
  rows, three hooks, an invisible opponent in and out of ability use, live
  disabled smoke, and a far pickup which must survive truncation. It verifies
  controller selection, pre-truncation counts, visibility, timer fields,
  pickup insertion and overflow telemetry. Existing C#/NativeAOT-versus-Rust
  observation comparison remained exact over 1,304 compared ticks on five
  maps for the fields those shared gameplay fixtures exercise.
- CUDA qualification uses both those exact fixture rows and random rows.
  Largest fixture CPU/CUDA forward difference was 1.304e-8 and largest weight
  gradient difference was 8.941e-8. Packed/unpacked rows, permutation,
  gathered training rows, CUDA graph replay and finite differences over all
  seven weight blocks also passed. A full CUDA build succeeded.
- Evaluation reports observation overflow as a cumulative result field and
  continues; game errors remain fatal. The standalone contract and crowded
  fixture both verify truncation and counting.
- The matched smoke used Rust rollouts, seed 314159, maps 0/1/2, 128 games,
  four buffers, H=128, one layer, independent Fourier K=4 aim, and 1,000,000
  requested learner decisions. v2 at `39ca452` completed 1,007,616 decisions
  in 3.115 measured seconds (323,479 learner/s overall; 316,506 median full
  update), 6,199 matches, pool Elo +10.40 and 50.49% pool win rate. Final v3
  completed 1,007,616 in 3.065 seconds (328,777 learner/s overall; 323,964
  median full update), 6,509 matches, pool Elo -1.88 and 49.94% pool win rate.
  Throughput was +1.6% overall and +2.4% by median versus v2. An initial v3
  rollout put four unequal pools in one block and measured only 220,196
  learner/s; splitting the long entity pool into its own 64-thread block
  removed the idle-warp bottleneck. The -12.28 internal-Elo difference and
  -0.55 percentage-point win-rate difference are a short-run observation,
  not evidence of equivalent or regressed learning strength.
