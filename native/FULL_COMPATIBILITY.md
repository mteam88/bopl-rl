# Full compatibility work

The target is the developed Unity game, including all one-through-six-player
configurations and all built-in/custom abilities, pinned at source commit
`c5b73bb0e17f2a87a7c19d3de24d78403a910e77`.
`data/full-reference.json` fingerprints 5,199 tracked source, asset, package and
project-setting files. The new recording and probe instrumentation is additional
to that baseline; every capture also records its actual game assembly hash.

**This target is not achieved yet.** The public Rust environment remains the
Level1 movement/dash slice, now with one through six players. A reference replay successfully
running in Unity does not mean its mechanics are implemented in Rust.

The ability inventory is `data/ability-coverage.json`: 39 catalog entries,
including NullAbility, Random, and eight custom entries from Rewind through
Portal gun. No entries are omitted because they are custom or difficult.

## Required completion gates

- Lossless setup and tick inputs for one through six players, including keyboard
  input, all ability buttons, teams, stocks, duplicate loadouts and custom prefabs.
- Explicit source/assets and ability-catalog identity, legacy format validation,
  actual local recording/playback comparison and malformed-input checks.
- Independent Rust initialization and full round lifecycle, including deaths,
  stocks, revival, clones, ability drops, sudden death, scoring and round changes.
- Moving/scaled/rotating platforms, all collider pairs and collision ordering,
  rigid and composite bodies, ropes, beams, duplication and all time effects.
- Every map and ability, followed by targeted interaction coverage, seeded
  stochastic play, mutated real inputs, long episodes and first-divergence reports.
- Rust/C/Python interfaces whose supported configuration matches verified scope.

Finite testing cannot prove every reachable interaction. The completion claim
must distinguish ported source coverage from exercised differential coverage,
and list any remaining source/runtime nondeterminism or state omitted from comparison.

## Replay v2

`BOPLREP2` archives retain `.rep` filenames. Existing complete four-player
recordings still load through the old codec. Incomplete legacy five/six-player
files are rejected because their missing input streams cannot be recovered.

Local recording now resolves abilities by prefab identity, fixing the old
prefab-name/display-name mismatch (e.g. Gold mushroom / Super star and Tack /
Banana peel). The new setup includes source/catalog hashes, map, seed, stocks,
game mode and an ordered list of IDs, colors, teams, keyboard modes and ability
indices. Every frame contains one complete input packet per listed player.

The codec is independent of Unity and used directly by both the player and
`tools/replay-codec`. Integers are little-endian. Layout:

- Eight ASCII magic bytes `BOPLREP2`.
- 64 ASCII bytes each for source hash and ability-catalog hash.
- Seed u32, map u8, game mode u8, stocks i32, player count u8.
- Per player: ID u8, color u8, team u8, keyboard u8, ability count u8, indices u16.
- Frame count i32; each frame has player-count packets in header order.
- Per packet: sequence u32, angle u8, buttons u16, target delay u8.
- SHA256 of all preceding bytes (32 binary bytes).

Button bits: jump=1, slot1=2, slot2=4, slot3=8, start=16, select=32,
W=64, A=128, S=256, D=512. Reserved bits, duplicate player IDs, missing streams,
checksum failures and partial or extra frames are errors. The network packet
format remains separate from this local recording format.

The current archive is a **round input recording**, not a complete arbitrary
mid-match save-state. Initial carried session/player state still needs auditing
against actual local recording, especially later rounds and random loadouts.
That audit is a required completion gate, not an implicit guarantee of the format.

```sh
dotnet run --project native/tools/replay-codec -c Release -- selftest
python3 native/tools/generate_full_scenarios.py native/artifacts/full-scenarios
python3 native/tools/verify_full_replays.py native/artifacts/full-scenarios
python3 native/tools/verify_dynamics.py
```

## Verified during the expansion

- Shared replay codec: 21,504 packet roundtrips across 1..6 players, all 1,024
  button combinations; malformed, incomplete and corrupted archives rejected.
- Actual Unity: 900 ticks per player count at render batches 1 and 16; 6/6
  playback comparisons passed with the documented presentation projection.
- Rust: 951 additional independent ticks across those six movement episodes,
  ending at first death; the goldens run through the real C API. Earlier
  movement fixtures remain regression checks.
- RNG: 65,000 mixed draws across 65 seeds, including zero, empty ranges and
  wrapping bounds. Outputs and internal state match the actual game assembly
  (`Updater.RandomFix` is invoked directly).
- Rigid-body primitives: 27,000 exact Unity comparisons across all six collider
  pair types, integration (including zero timestep), impulses/friction and
  positional correction. This does not yet exercise a complete Rust rigid-body
  scene or integrate those primitives into every ability.
- Broad ability replays include all catalog entries 1..38 and six active players.
  All three 900-tick Unity playback comparisons pass at batches 1 and 16.
  Composite membership uses canonical identities (trace schema v3). Cosmetic
  exclusions are source-audited in `data/presentation-fields.json`; raw traces
  are retained. Ability-entry events now distinguish loadout selection from casts.
- Ordered rigid-body scenes: 1,000 scenes, 12,000 consecutive steps and 164,124
  callback records match actual `DetPhysics.Simulate`, with no excluded fields
  and no failed scenes. 560 scenes contain composite bodies. Coverage includes
  all collider groups, layer masks, inactive bodies, triggers, offsets, zero
  timestep, attachments, detachment, destroyed objects and composite-slot reuse.
  A 60-scene/720-step subset is checked in for standalone regression.
  The oracle runs the original solver with synthetic collider parents that record
  callbacks. Ropes, beams and ability callback effects are not in these scenes;
  integrating this stage into full native gameplay remains required.

Pending: actual local recording versus replay, full state-capture audit, full
physics integration with gameplay, all abilities and map/round lifecycle in Rust, broad
interaction verification and final API/adapter integration. The goal stays active.

## Physics stages from actual gameplay

The recorder can capture a complete rigid/composite scene immediately before
`DetPhysics.Simulate` and its result before gameplay callbacks run:

```sh
python3 native/tools/reference.py capture game.rep native/artifacts/stages.jsonl.gz --physics --ticks 900 --batch 16 --timeout 600
python3 native/tools/verify_physics_capture.py native/artifacts/stages.jsonl.gz
```

Each comparison supplies the recorded pre-physics state to Rust. This tests the
physics stage on states produced by real abilities; it is separate from the
independent native movement episodes and does not certify native ability logic.
Beam ticks and older rope captures without rope state are counted explicitly
and remain outside the stage comparison. New captures include the full rope
physics state.
The capture preserves collision queues, composite membership and the cached
`FixTransform.right` vector, whose raw components can differ from recomputing its
angle. Gameplay found this missing state after synthetic scenes passed; the
original failing traces are retained in `artifacts/physics-gameplay/`.

The three six-player ability-loadout captures now pass all 2,700 physics stages
and 16,930 queued callbacks, with no state exclusions. No active rope or beam
was present at these captured physics stages; their implementations are still
required. This result is deliberately narrower than ability compatibility.

The rope kernel also matches 2,000 actual Unity cases: 15,808 successful
actions and 24 matching source arithmetic errors. It includes relaxation,
platform contact, normal/space gravity, zero timestep, detached endpoints,
40-segment capacity, spawning and both reeling directions. 197 cases are retained
as standalone regressions. Rope broad-phase integration is now verified below. The grappling ability
lifecycle remains separate work.

The combined rigid/composite/rope solver matches another 1,000 Unity scenes
and 12,000 steps, including 45,168 active rope updates. Nine scenes exercised
the original 32-platform rope contact capacity. 69 scenes and 828 steps remain
in `fixtures/rope-scenes.jsonl.gz`.

A dedicated six-player grappling-hook replay matches all 600 physics stages and
3,834 callbacks. Active ropes occurred in 268 stages, with six simultaneous ropes
at the peak. The recorder includes all 128 rope slots, active segment counts,
current/previous segments, endpoint forces/masses, contact buffers, gravity,
relaxation settings and the physics subsystem's cached space-level flag.
The native grappling ability still needs its own independent lifecycle port;
these comparisons initialize the physics stage from Unity's recorded state.
