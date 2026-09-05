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


## Beam geometry and persistent crossing graph

`src/beam/` ports line/beam intersections, color mixing, water reflection,
ordered scene raycasts and the persistent `DetPhysics.UpdateBeams` graph.
The graph retains crossing ages, stable IDs, parent chains, accumulated strength,
reflected segments and the source's branch/tie behavior across frames.

Actual Unity comparisons pass:

- 20,100 primitive queries: 15,000 intersection queries and 5,100 color mixes,
  including grayscale, primary/secondary colors, zero alpha and HDR values.
- 999 complete graph scenarios / 11,988 frames, plus one matching source
  math-domain error. These include one through six initial beams, all six used
  graph roles, 2,610 reflected segments and 2,961 crossing records.
- 71 longer scenarios / 8,520 frames, plus one matching source math-domain error.
  Each scenario runs 120 frames with three beams per player (up to 18 initial
  beams across six owners), including 3,752 reflections and 10,385 crossings.

The fixed-point source error is retained as an expected outcome, not hidden or
counted as a complete frame sequence. The color comparison also found that the
pinned Mono runtime uses double intermediates inside float expressions and
rounds at float storage boundaries; the port matches those boundaries explicitly.

995 primitive cases, 62 short graph scenarios and 51 longer scenarios are kept
as standalone fixtures in `fixtures/beam*.jsonl.gz`; source errors and every
observed graph role are retained. `just verify` runs all three corpora. Regenerate
against the instrumented Unity build with `just verify-beams`.

This is a **graph-stage** comparison. The probe captures immediately before
`renderBeams`, which also creates gameplay hitboxes and applies damage/forces.
The graph result alone does not verify those effects. The component ports below
cover hitbox geometry and target force/charge rules; beam ability startup/shutdown,
pooled object lifecycle and independent native beam gameplay remain required.
Graph scenarios currently use null ground references and synthetic obstacles;
they do not certify every live-game beam state or the entire Beam ability.
Evidence and assembly/input/output hashes are in `artifacts/beam/`,
`artifacts/beam-graph/` and `artifacts/beam-graph-stress/`.


## Beam hitboxes and target effects

The next component layer is implemented in `src/curve.rs`, `src/beam/hitbox.rs`
and `src/beam/effects.rs`:

- `AnimationCurveFixed` interpolation, pre/post wrapping, boundary comparisons,
  empty/single-key behavior and arithmetic errors match 2,000 Unity scenarios:
  92,935 returned values and 57 matching source errors.
- The actual beam prefab's `BeamObject.SetDimensions` / `Initialize` path matches
  1,000 sequences and 14,000 states. This includes cached direction versus angle,
  pre-initialization extents, subsequent box updates, animated widths, exact-zero
  extent replacement, negative extents and zero direction.
- The actual `DetPhysics.renderBeams` target path matches 980 complete scenarios
  and 7,840 frames, plus 20 matching errors from zero-mass black holes. The probe
  uses real DPhysicsBox, BoplBody, BlackHole and Orb components and invokes their
  unchanged methods. It verifies platform/player forces, black-hole velocity,
  orb type/charge/firing/colors, owner protection from time stop, environmental
  ownership, role/reflection exclusions and the strict water-height threshold.
  Every affected target kind changed in the corpus: wall 574, terrain 577, lethal
  terrain 601, player 1,234, black hole 1,051 and orb 963 frame/target changes.

The component tests retain 250 curve scenarios, 196 hitbox sequences and 214
force/charge scenarios as standalone fixtures, including every source error.
All run under `just verify`; regenerate with `just verify-beam-components`.
Hashes and raw evidence live in `artifacts/curves/`, `artifacts/beam-hitbox/` and
`artifacts/beam-effects/`.

These component tests initialize the components directly. The combined-stage
evidence below adds graph/pool/physics update order and composite target routing.
Explosion collision callbacks, deaths, ability startup/shutdown and a native beam
episode from player inputs remain required. The public C environment is still the movement/dash slice;
Beam, Orb and BlackHole are not complete native abilities.


## Combined beam and physics stage

`beam::system::System::simulate` now connects the beam graph, hitbox pool and
target effects to the rigid/composite/rope solver. It preserves the source order:
integrate bodies, find potential collision pairs, update ropes, update beams and
their hitboxes/forces, then test and resolve the previously selected pairs.
A new beam hitbox enters the pair search on the following step; an existing
hitbox can change geometry after its pairs have already been selected.

The system retains its graph and pool across steps, accumulates queued beams
when an empty physics scene returns early, routes forces to composite bodies,
and handles DestroyFix flags and replacement of destroyed beam slots. The
source's deactivation loop stops at a destroyed or inactive slot; later slots
can remain active. The port preserves this behavior.

The Unity oracle runs the actual `DetPhysics.Simulate`, real DPhysics components,
beam prefabs and target components. It stops immediately before collision
callbacks execute and compares their complete ordered queue along with scene,
rope, graph, pool and target state.

- Normal corpus: 994 complete scenarios / 11,928 steps and 457,090 callbacks,
  plus six matching source arithmetic errors. It includes 44,736 rope updates,
  3,312 steps with composite bodies and 990 destroyed-pool-slot observations.
- Stress corpus: 118 complete scenarios / 4,248 steps and 174,062 callbacks,
  plus two matching source errors. These use three beams per player (up to 18
  initial beams across six owners), 36 steps and repeated pool destruction.
- Standalone fixtures retain 52 normal scenarios (552 steps and six errors)
  and 21 stress scenarios (684 steps and two errors). `just verify` runs both.

Regenerate with `just verify-beam-scenes`. Raw queries, Unity/native states and
hashes are in `artifacts/beam-scene/` and `artifacts/beam-scene-stress/`.

This is independent multi-step physics-stage evidence from synthetic initial
scenes, with supplied beam bodies. Replay-stage import is now covered below.
Ability callbacks, explosion damage, full player/ability lifecycle, map loading
and the full C environment remain open.


## Beam state from actual gameplay

Physics traces now include the functional beam state before and after each stage:
all graph arrays, crossing history, queued bodies, pooled hitboxes (including
null gaps), target force/velocity/orb charge, and source configuration. Rust
imports that state and executes the same combined physics stage. Target membership
is held across the stage so removal does not hide a target's final state.

Three complete six-player captures match **1,780/1,780 stages and 15,514 queued
collision callbacks**. Two 540-tick captures cover charge/cancellation without
emission. The third has four Beam users and two idle Dash users: **700/700 stages,
141 stages with emitted beams, and 6,898 callbacks**. Emission uses owners 1–4,
with up to four queued beams. No replay reflections or crossings occur in this
corpus; those remain covered by the synthetic graph and combined-stage corpora.

The all-six-Beam stationary replay exposes a source defect at input index 561.
`AudioManager.PlayInterruptable` returns index -1 for absent `beamCharge5` or
`beamCharge6`; `Beam.ExitAbility` passes it to the unchecked `TimesSoundsPlayed`
array lookup. Death during charging/firing can therefore interrupt the tick.
The short-press and moving held-input cases also expose this path. These are
rejected full traces, not successful matches or native ability validation.

Opt-in replay capture now preserves the first exception in a separate
`.failure.json` snapshot and stops the capture, while rethrowing the original
exception. The normal game path keeps its original exception behavior. The
snapshot includes the input and the completed physics stage, if physics reached
its pre-callback boundary. Its provenance is separate from successful traces.
Rust exactly matches the stationary replay's final physics stage: four emitted
beams owned by **3, 4, 5 and 6**, plus ten queued callbacks. This verifies every
owner ID in actual gameplay across the two emission cases; it does not prove a
completed six-simultaneous-beam match or reproduce the subsequent native death.

`fixtures/beam-gameplay.jsonl.gz` retains 284 completed-stage regressions, including
all 141 emission stages. `fixtures/beam-failure.jsonl.gz` retains the final stage
before the source exception. Both run under `just verify`. Raw captures, failure
logs, reports and hashes are in `artifacts/beam-gameplay/`.

Generate the input setups with `python3 native/tools/beam_gameplay_cases.py DIR`
(after the standard asset export), then encode them with `tools/replay-codec`.
Use `reference.py capture --physics` and `verify_physics_capture.py` for completed
captures. Use `verify_physics_failure.py CAPTURE.failure.json` for a recorded
exception boundary; it never upgrades the failed tick to a successful rollout.

Beam's player controller, animation-driven lifecycle and collision damage remain
unported. The public C environment remains the movement/dash slice.
