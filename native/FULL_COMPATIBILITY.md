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

Current training research uses the actual Unity game through the
[C/PufferLib environment](rl/README.md). Its first experiment fixes the loadout to
dash, arrow and platform. Successful training or playback in that environment
does not complete the independent Rust port or the broader interaction audit.

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
as standalone regressions. Rope broad-phase integration and both shipped rope
abilities are now verified below.

The combined rigid/composite/rope solver matches another 1,000 Unity scenes
and 12,000 steps, including 45,168 active rope updates. Nine scenes exercised
the original 32-platform rope contact capacity. 69 scenes and 828 steps remain
in `fixtures/rope-scenes.jsonl.gz`.

A dedicated six-player grappling-hook replay matches all 600 physics stages and
3,834 callbacks. Active ropes occurred in 268 stages, with six simultaneous ropes
at the peak. The recorder includes all 128 rope slots, active segment counts,
current/previous segments, endpoint forces/masses, contact buffers, gravity,
relaxation settings and the physics subsystem's cached space-level flag.
The independent Grappling Hook lifecycle golden described below complements
this supplied-state corpus with actual input, spawning, contacts and reeling.


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
cover hitbox geometry and target force/charge rules; the executable rollout
integration is covered in "Executable Beam" below.
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
evidence below adds graph/pool/physics update order and composite target routing;
the executable rollout section adds player input, ability lifecycle and deaths.


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

The captures in this section isolate the imported physics stage. End-to-end
player control, animation lifecycle and collision damage are covered by the
executable rollout differential below.


## Player death accounting

`player_life::Book::kill` ports the shared `Player.Kill` accounting stage for
one through six registered players: body/clone counts, kills and deaths, rumble,
final-death time and cause, per-frame achievement kill counts, and the pending
elimination notification. The caller receives that notification after the state
updates and can dispatch it immediately; the native simulation does not retain
the test listener's event history as gameplay state.

The implementation preserves source behavior: a round winner ignores the call;
self-kills credit both a death and a kill; teams do not change kill credit; only
an initial body count of exactly one updates final-death metadata and emits the
elimination event. Repeated calls can drive body counts below zero, and integer
counters wrap. The native method does not silently clamp or repair those inputs.

The real Unity method matches **1,200 scenarios / 38,400 calls**, including 2,100
final-body deaths/events, 3,906 winner no-ops, 11,223 self-kills and 16,841 calls
with no registered killer. Some direct cases deliberately exercise counter limits
and inconsistent initial state; they are method-level evidence, not legal matches.

Two six-player, 700-tick gameplay captures also match all **six actual accounting
calls**. With five stocks, Unity first creates a replacement body; the three
deaths therefore leave another body alive and emit no elimination event. With
stocks disabled, the three deaths are final and emit elimination notifications.
These capture points stop immediately before `playerDied` listeners execute.
Both traces' 1,400 physics stages match too (13,830 queued collision callbacks).
The stocks-enabled replay repeats the previous Beam scenario; its 700 stages are
revalidation, not a new independent input scenario.

`fixtures/player-life.jsonl.gz` retains 196 direct scenarios / 6,272 calls;
`player-life-gameplay.jsonl.gz` retains all six actual calls. The no-stocks trace
adds 182 physics regressions in `elimination-physics.jsonl.gz`, including all 141
emission stages. All run under `just verify`. Regenerate direct cases with
`just verify-player-life`; use `verify_player_life.py --traces TRACE...` for
completed gameplay captures. `beam_gameplay_cases.py --stocks 0` reproduces the
no-stocks setup. Raw evidence is in `artifacts/player-life/` and
`artifacts/player-life-gameplay/`.

Ability exits, wiring rope detachment into death, body and ability destruction, clone
creation, stock choice and revival remain unported as
native lifecycle stages. This checkpoint supplies shared accounting for those
handlers; it does not turn the public movement/dash environment into full combat.


## Player collision decisions

`player_collision::Receiver::collide` ports `KillPlayerOnCollision` and the
`OnCollide` duplicate guard. It preserves invulnerability masks, ignored hierarchy
and beam IDs, platform squish/pickup selection, projectile owner immunity, orb and
contact-ability rules, achievement counters, and death-cause priority. Effects are
issued in source order: attachment includes its target and exact position, then
lightning sound or a kill request where applicable. The kill handler's return
value remains the direct-call result, including false after the duplicate guard
has already been set.

The actual Unity method matches **600 generated scenarios**, with **12,551
completed actions** and **177 failing actions**. Failure checks retain the first
error's index, completed prefix and partially mutated state: 111 null-reference
errors and 66 integer-absolute-value overflows. The corpus exercises all six
ignored-collider slots, rotated squish points, identity aliases, mixed recorded
and current layers, self-hits, parent abilities, counter limits and replay versus
live achievement behavior. These are component tests, including deliberately
inconsistent inputs; they are not 600 complete games.

The normal six-player replay matches all **4,432 OnCollide decisions**, including
three lethal beam contacts. The source's known six-Beam shutdown failure supplies
another **four decisions**, including the kill request before its downstream
handler throws. Its failed tick is not counted as a successful rollout. The normal
trace's 700 physics stages and the failed tick's last completed physics stage also
match. This reuses existing input scenarios with additional instrumentation.

Both collision fixtures run under `just verify`. Regenerate with
`just verify-player-collision`; use `verify_player_collision.py --traces TRACE...
--failure SNAPSHOT` for captured decisions. Raw evidence and provenance are in
`artifacts/player-collision/` and `artifacts/player-collision-gameplay/`.

The component receives resolved metadata snapshots. Persistent entity references
(including copied ignored colliders) still need resolution in the native world.
Synthetic checks replace downstream kill/attach handlers with recording sinks;
gameplay capture runs the real handlers and projects their changes out of this
comparison. The collision module delegates ground attachment (ported below), body
destruction and ability exits; it does not provide a full combat episode through
the C API.


## Ground attachment and detachment

`player_ground::State` executes `PlayerPhysics.AttachToGround` and `UnGround`.
Attachment stores platform identity before computing speed, perimeter position,
body position and orientation, then clears both velocities. Detachment retains
that identity and the old perimeter position. It clears grounded speed, nudges
the body along the retained platform's current normal even on repeated airborne
calls, optionally resets rotation and its cached axes, and returns the old angle
for immediate listener dispatch. A platform marked `IsDestroyed` still supplies
the normal if its Unity object reference is non-null.

The existing native movement environment now uses these shared methods for
attachment and jump detachment. All independent C-ABI movement episodes remain
exact; this is executable integration rather than a probe-only implementation.

Actual Unity component calls match **1,000 scenarios / 10,741 completed actions**:
3,497 attachments and 7,244 detachments/events. Another 126 actions match the
source's first division-by-zero failure, including the completed prefix and
partial state. Some exact corners and degenerate surfaces expose this source
behavior; the native component preserves it. Inputs vary requested position
separately from current position, platform identity, cached axes, player radius,
platform rotation/extent/radius, velocities and repeated detachment flags.

The fixture runs under `just verify`; regenerate with `just verify-player-ground`.
Raw reference inputs, outputs and hashes are in `artifacts/player-ground/`.
Rendering and optional jump sounds are outside the comparison. The return value
supplies a pending notification; arbitrary listener effects, moving-platform
updates, rope-driven ungrounding, and full death/revival handlers still require
integration and validation in the full native world.


## Player body and rope transitions

`player_body::State` ports `PlayerBody.Init`, `AddForce`, `AttachRope`,
`EnterAbilityClearRope`, `UpdateSim` and `LateUpdateSim`. Its live `BodyRope`
objects reuse the physics solver and endpoint operations, adding owner identity
and endpoint airtime. The methods preserve velocity history, time-scaled motion,
airborne/grounded rope coupling, overstretch detachment, jump-release momentum,
extra swing force and velocity decay.

Replacing a rope modifies the old rope using the incoming endpoint flag before
storing the new reference. Entering an ability clears that reference without
detaching the rope. Movement uses player-scaled delta time; rope airtime uses the
raw simulation delta; release momentum divides by the fixed base step. External
velocity still decays on zero-time updates. These are source behaviors, including
their unusual interaction with time manipulation.

Actual Unity methods match **600 generated scenarios / 8,666 completed actions**,
including 1,731 body updates, 2,309 late updates, 1,709 coupled rope-solver steps,
1,318 attachment changes, 399 ability-entry clears and 93 release-sound requests.
The 391 complete sequences and 209 first-error boundaries retain state throughout:
187 missing-player/holder null references and 22 rope division-by-zero errors.
The component oracle calls each original rope solver serially to preserve a
deterministic first-error boundary; it does not test parallel exception scheduling.
`Rope::try_step` now preserves mutations before an arithmetic error for callers
that need a recoverable result, including the first endpoint movement before
the source fails on a one-segment rope.

A real six-player grappling replay matches **4,405 body calls** (2,198 updates and
2,207 late updates). Of these, 773 start with a rope reference, 769 with an enabled
rope and 439 with an arrived hook. The same replay's **600 physics stages / 3,834
callbacks** match, with 268 active-rope stages and up to six active ropes. This
reuses the existing grappling input with new capture instrumentation; it is not a
new independent input scenario. There are no jump-release sounds in this replay;
that branch is covered by the generated cases.

Both body fixtures run under `just verify`. Regenerate with
`just verify-player-body`; use `verify_player_body.py --traces TRACE...` for
actual update captures. All captured calls with an initial rope reference are
retained in the gameplay fixture, along with a sample of unroped calls. Raw
artifacts and hashes are under `artifacts/player-body/` and
`artifacts/player-body-gameplay/`.

This verifies the body/rope methods and their interaction. The gameplay check
imports Unity state before each method. Full-world component identity, scheduler
enablement, physics/body update ordering across an independent episode, ability
controllers, and death/revival orchestration still need integration. The public
C environment remains the movement/dash slice.


## Ability physics handoff and grounded ropes

`player_physics::State` ports the fixed-point state changes in `UpdateScale`,
`AddGravityFactor`, `AirealMovement`, `AddForce`, `Jump`, `AlignToGround`,
`ApplyOffset`, both `SyncPhysicsTo` overloads and `LateUpdateSim`. It shares ground
attachment state and the existing body/rope objects. The C movement environment
now uses the same gravity and air-movement functions; its independent episodes
remain exact.

The implementation preserves the differences between entering and exiting an
ability. Entry uses the receiving body's radius, while exit projects the old
perimeter position using the outgoing radius before computing the receiving
body's local coordinate. An invalid grounded exit clears the platform reference
but leaves the incoming grounded flag set. Scaling and partial failures retain
source order. Perimeter lookup also preserves the case where no segment is
selected and the source returns zero without reading a missing shape component.

Grounded rope updates align the body first, add endpoint force at the original
strict threshold, then detach when the outward velocity exceeds the source limit.
The detachment listener runs before the final rope endpoint assignment. The
native callback can mutate the body at that point; generated Unity cases include
listeners that move it to verify this ordering.

The actual component methods match **1,000 scenarios / 15,026 completed calls**,
including 2,432 ability handoffs, 3,414 late updates and 2,286 jumps. There are
2,718 matching detachment notifications. The 701 complete sequences and 299
first-error boundaries cover missing platforms/shapes, invalid rope indices,
zero scale denominators and invalid movement constants: 191 null references,
102 division errors and six index errors. Inputs include all six owner IDs,
space gravity, different radii, destroyed platforms and boundary rope forces.
These are supplied component states, not 1,000 independent gameplay episodes.

`fixtures/player-physics.jsonl.gz` retains 405 scenarios, including every source
error; it runs under `just verify`. Regenerate with `just verify-player-physics`.
Raw evidence is in `artifacts/player-physics/`. The raycast, traversal and player
controller sections below cover movement built on these handoff methods. Arbitrary
ability callback effects and full world scheduling still require integration. Rendering and jump audio are outside this fixed-point state
comparison. Full independent ability episodes remain required.


## Player raycasts and attachment

`Scene::raycast_rounded` ports `DetPhysics.RaycastToClosestRoundedRect`, including
destroyed-collider cleanup, current registration order, complete near/far hit
information, inclusive distance limits and first-registered wins at equal
distance. All rounded-rectangle layers participate. Circle and box objects are
excluded from hits but still participate in the source cleanup pass.

`player_physics::State::collision_raycasts` ports `CollisionRaycasts` and the
velocity selection used by `VelocityBasedRaycasts`. It tries the front ray, then
the two side rays in source order, correcting the attachment position for the
selected side. Attachment updates run before the listener; listener movement is
retained. Missing platform/shape components preserve partial state before the
source error. The shared clock helper preserves missing-holder/player behavior
and time-stop protection. The C movement environment uses the same three-ray
selection, with its existing independent episodes as regressions.

Actual Unity component calls match **1,000 scenarios / 22,711 completed actions**:
10,000 direct queries, 4,958 velocity-based calls and 7,753 explicit-velocity
calls. There are 6,452 direct hits across all 32 layers, 2,738 player-ray hits
and 1,203 attachment notifications. The 958 complete sequences and 42 matching
null-reference boundaries include the complete scene, collider orders, player
state and listener records. Inputs cover one through six registered players,
protected/unprotected/missing players, zero and negative time, duplicate hit
distances, both side rays, inactive/destroyed objects and composites.

`fixtures/player-rays.jsonl.gz` keeps 224 cases, including every source error.
It runs under `just verify`; regenerate with `just verify-player-rays`. Raw
evidence is in `artifacts/player-rays/`. This is a component oracle with supplied
scene and platform caches. The traversal and controller work below integrates
these functions. Independent full-world ability episodes remain required.


## Blocked surface traversal and hop queries

`platform_motion::Terrain` ports the blocked and unblocked `moveAlongSurface`
methods, the multiple-body variant, `maxMovementAllowed` and `PlatformHopMove`.
It uses the shared scene raycasts and platform perimeter coordinates. The C
movement environment now shares its three-ray limit selection and checked
perimeter calculation. Movement
keeps the source's distance-versus-speed arithmetic and always re-enables the
platform collider after a successful query, even if it started disabled. The
multiple-body method runs both limit checks but uses the second result as the
final limit. These behaviors are preserved from the developed game.

Direct hopping uses a separate scan: it does not clean up destroyed colliders,
ignores collider enablement, and checks transform destruction/object activity
inside the disallowed-ID loop. An empty exclusion array therefore skips those
validity checks. The scan retains registration ties and returns the source's
signed remaining movement for the player controller to consume.

Collider enablement is now separate from object activity in native scenes and
new gameplay captures. Ordinary integration, broad-phase, ropes and raycasts use
both flags; hop queries retain their source-specific behavior. Older captures
with combined activity remain valid for their original physics-stage checks.

The actual Unity methods match 1,000 supplied-state scenarios, including shifted,
rotated and differently sized cached surfaces, missing shapes, duplicate hits,
zero/negative time, disabled/inactive/destroyed objects and empty exclusion arrays.
All 56 first-error boundaries retain completed prefixes and partial scene state.
The fixture keeps 205 scenarios, including all source errors. A separate 200-scene
rigid/composite/rope check verifies disabled colliders over 2,400 consecutive
steps, 11,052 callbacks and 9,024 enabled-rope updates; 110 scenes contain composites.

Both fixtures run under `just verify`; regenerate with `just verify-surface-motion`.
Evidence is in `artifacts/surface-motion/` and `artifacts/disabled-colliders/`.
This covers traversal primitives with supplied platform caches and non-null
collider transforms. The controller work below adds recursive platform changes,
Move/MachoMove, UpdateSim and drop-callback boundaries. Scheduler integration and
independent complete ability episodes remain required. The public C environment is still the verified
movement/dash slice.


A fresh six-player grappling replay verifies the updated capture representation:
all 600 physics stages / 3,834 callbacks match, including 287 stages containing
disabled colliders on active objects (1,255 collider states). There are 268 active
rope stages and up to six active ropes. This reuses the established six-hook input;
it is not a new independent gameplay scenario. Unity still supplies each stage's
pre-state. All stages with disabled colliders are retained in
`fixtures/disabled-gameplay.jsonl.gz`, along with sampled remaining stages (316 stages total).


## Player movement controller

`player_controller::State` composes the verified physics state, ordered raycasts
and platform traversal for `UpdateSim`, `Move`, `MachoMove`, `DropPlatformTest`,
`MoveAlongCurrentPlatForm` and recursive `MoveAlongCurrentPlatFormSwapIfBlocked`.
It also ports input alignment and the increment/acceleration helpers. The public
C movement environment now shares the grounded increment helper; its independent
trajectory fixtures remain exact.

The implementation preserves double time scaling in airborne Move/MachoMove,
per-call friction, speed mutation before invalid-platform detachment, skipped
updates before scaling, and external-velocity clearing before a failing holder
lookup. Drop callbacks run after detachment listeners have changed the body;
the exit record captures the resulting position/velocity with the source's
other default fields. Native listeners can modify controller state at these
boundaries. The oracle runs the actual component methods, records sound requests,
and checks those exit defaults rather than invoking complete ability controllers.

Recursive hopping retains the source's repeated delta-time multiplication and
signed remaining movement, without adding an attachment event that the source
does not emit. The component oracle uses real GameObject IDs for registration
and hop exclusions, then canonicalizes its results. All 1,000 scenarios
match: 817 complete sequences, 32,976 completed
calls, 169 platform transitions and up to four transitions
in one hop call. Completed calls emit 601 attachment,
1,233 detachment, 414 drop-exit and
196 sound notifications. All 132 null-reference and 51
division-by-zero boundaries retain exact partial state and completed prefixes.

Inputs include one through six registered players, protected/missing players,
zero/negative time, ice friction, dropped/destroyed/inactive platforms, scale
changes, missing shapes, rotated caches and listeners that change velocity.
There are 334 cases with Roll's five-slot hop buffer and 666 with 64 slots. The
five-slot cases contain at most five platforms to keep recursion finite;
exhausted buffers with additional cyclic/overlapping platforms remain unverified.
These are supplied-state component sequences, not independent full-game episodes.

`fixtures/player-controller.jsonl.gz` retains 315 scenarios, including
all source errors, and runs under `just verify`. Regenerate with
`just verify-player-controller`; raw evidence is in `artifacts/player-controller/`.
World identity/lifecycle, scheduler enablement, full callback effects, ability
controllers and independent full-round initialization still need integration.
The C API remains the movement/dash slice until those episodes pass.


## Animation clocks and callbacks

`animator.rs` ports the simulation portion of `SpriteAnimator`: looped animation,
Y/XY velocity maps, rotation selection, both 2D entry methods, frame skips,
time-stop filtering, all four callback entry methods and reentrant callbacks.
The unused speedMapX mode keeps its source no-op behavior. Sprite identities,
frame arrays, counters, flips, speed-container references and callback state are
compared; there is no presentation exclusion in these component checks.
`SlimeController.SpeedUpAnimation` supplies the fast respawn frame conversion.

All 1,000 randomized scenarios match, including 115,372 completed calls and
2,043 callbacks. Expected errors retain their completed prefix and partial state.
An additional 1,237 cases exercise all 194 named animations from the 35 exported
AnimationData assets, 1,042 sprite frames and the player's velocity map. These
match 60,625 calls and 946 callbacks. Four empty/deprecated animations fail just
as the source does. The exported fields were also checked against the current
asset YAML. Long frame holds get boundary snapshots rather than millions of
idle ticks. The player spawn frames invoke their completion callback at update
106 normally, or 21 after the game's fast respawn conversion.

Both checked fixtures run without Unity or the ignored asset export:

```sh
python3 native/tools/verify_animator.py --golden
python3 native/tools/verify_animator.py --assets --golden
```

Fresh asset checks require `just export-assets`; `just verify-animator` runs both
live corpora. This verifies the clock and callback boundaries, not complete
SlimeController spawning, arbitrary ability callbacks, object registration or
world scheduling. Those remain integration work.


## Spawn and revival transitions

`spawn.rs` composes SlimeController's Init, BeforeSpawnInit, Spawn, ReSpawn,
ReSpawnFast, PrepareToRevive and PopIntoAir with the animator and ground-loss
handler. It preserves body/collider enablement, input suppression, invulnerability
bits, player revival/ability flags, shared speed references, cooldown resets and
indicator identities. The actual OnNoLongerGrounded callback runs before external
listeners and sound; it can replace the spawn animation with a velocity map.
Reentrant spawn callbacks run synchronously, including after interrupted revival.

All 600 actual Unity component scenarios match: 84,312 completed calls, 1,377
animation-triggered spawn callbacks and 3,374 detachment/sound pairs. There are
100 setups at each player count from one to six. Only the selected slime is
advanced; other player state is retained and compared. The corpus includes the
real spawn/fall animation data, random signed frame durations, tutorial mode,
time-stop protection, repeated references and more than three abilities. Another
150 setups begin with an in-flight spawn callback already registered.
All 129 errors include the completed prefix and partial failing state, including
13 failed ReSpawnFast calls whose fast-animation flag remains set.

The 250-case fixture runs in `just verify`; `just verify-spawn` regenerates it
against Unity. The C movement API still resets from its tick-240 snapshot.
Full prefab creation, stock choice, revive travel effects, death selection,
ability drops and the world scheduler remain outside this component stage.


## Registration and deferred destruction

`scheduler.rs` ports Updater's registration and prefab priority offsets, pending
initialization, update/late-update dispatch, destruction queues, level-reset
methods and tick counters. Callbacks may register or remove entries while a
loop is running. Mono's exact introsort is used for pending initialization,
including equal-priority ordering, wrapping comparisons and backing-array
capacity in partial error states. Physical presence is separate from the
IsDestroyed marker; deleted managed references can still receive
OnDestroyUpdatable through the source's CLR null check.

All 1,002 lifecycle scenarios match: 29,538 completed calls, 8,394 ticks and
224,574 component callbacks. One-through-six-player setups include shared input
profiles, jump-edge updates, invisibility clocks and time-stop protection.
The additional 24 stress cases all complete, with up to 4,096 components and
26,199 initialization/update/late/destruction callbacks in each category.
The largest exported map has 171 updatables before players and spawned objects.

The tests invoke the actual Updater with callback components and an empty
physics scene. Typed phase events mark duplication, magnet, physics and rope
boundaries. The rope-mesh boundary is retained because the source also updates
its cached level type there. The fixtures retain 392 lifecycle and all 24 stress
cases and run under `just verify`.

This does not yet connect the gameplay kernels to that dispatcher. Player
position-setter rollover, gameplay-specific Unity OnDestroy handlers, real
prefab construction, outer frame/hit-stop pacing and full map loading remain
outside these tests. The C movement environment still uses its existing loop.
A fresh Unity replay with the new optional recording hooks matches 473
independent C steps through first death; that is an instrumentation regression,
not evidence that the new scheduler already drives the C environment.


## Composed player simulation

`actors.rs` now connects the scheduler, PlayerPhysics controller, PlayerBody,
and the ordered collision scene. It runs independent native ticks from supplied
initial component state. Movement/jump inputs, time scales and component-enable
flags are the only subsequent inputs; Unity state is not imported between ticks.
Controller/body projections and collider transforms share updates after every
component call, including partial failures. Collision callbacks run after the
physics stage synchronizes transforms, before late updates.

The scripted motor invokes Jump/Move directly. The Unity oracle invokes the
original PlayerPhysics and PlayerBody methods through callback components in
`Updater.TickSimulationInner`, and the original SlimeController.OnCollide for
equal-scale shoves. It deliberately does not run SlimeController.UpdateSim.
All actor colliders are player-layer boxes, and the terrain is the supplied
Level1 geometry with static platform bindings. Actor collision priorities are
separate from component update priorities.

All 120 scenarios match: 20 for each player count, 115 complete 120-tick
sequences and five matching division-by-zero failures from zero previous scale.
There are 13,801 completed ticks, 13,778 physics stages, 29,440 collision
callbacks and 682 applied shoves. The comparison includes controller/body state,
scene state, pair/callback order, jump edges, simulation clocks and phase order.

Cases permute component registration and priorities, including equal-priority
sorts, disable components independently, skip physics on loaded-level ticks,
exercise normal/fractional/zero/negative/doubled player time scales and the
physics time-stop threshold, and include zero-dt and coincident-player ticks.
Coincident shoves use wrapping collision-priority subtraction. The same shove
direction helper now serves the existing C movement environment.

The checked fixture retains 36 cases. This runner is not exposed as a full-game
C environment. Full SlimeController input/animation/ability transitions, biting,
ropes, spawning, deaths, stocks, clone identity, moving platform bindings and
map construction remain integration requirements. Attached ropes and unequal
actor scales with shove handlers are explicitly rejected by this runner until
their full gameplay handlers are connected. Existing rope/scaling component
checks do not establish their integration here.


## Slime locomotion in the composed runner

`slime.rs` adds the locomotion part of SlimeController to `actors.rs`. The
reference now calls the original SlimeController.UpdateSim with empty ability
lists, and schedules its real SpriteAnimator alongside PlayerPhysics and
PlayerBody. The native side implements idle/run/stop/turn/land transitions,
input suppression, upside-down/air facing, jump's immediate animation advance,
ground-loss and landing callbacks, and the final movement/speed-container update.
The empty-ability path also preserves the nothing/biting movement gates.

Animations use the exported MCAnimData idle, run, stop, runTurnAround, land and
fallLeftLoop frames plus the fallLeft velocity map. Turn callbacks retain their
captured input only while installed; cancellation and reentry are compared.
Physics callbacks now return errors, so a failing animation listener stops the
controller immediately and retains exactly the partial state Unity reached.

All 120 scenarios match: 20 per player count, 115 complete 120-tick sequences,
13,801 completed ticks and five matching zero-previous-scale failures. There
are 13,778 physics stages, 49,746 collision callbacks, 620 applied shoves,
179,286 component updates and 178,993 late updates. A further three fault
scenarios match failures inside a landing listener, a jump's ground-loss
listener and a stop animation's completion callback. Golden selection includes
the first occurrence of each observed state transition, including rare completed
turn callbacks, in addition to fixed/random samples and all failures.

This is a supplied-state locomotion comparison, not full SlimeController parity.
Ability lists are empty; swallowing and invisibility start false, scales remain
equal, and there are no attached ropes or ragdolls. Achievement flags are false
and reference players are nonlocal, allowing the moonwalk timer to be compared
without awards. Player position-setter state and collider Scale bookkeeping are
not compared here. Ragdoll recovery is covered in the extension below.
The extensions below cover cooldown routing, Invisibility, position arbitration
and collider scale. Swallowing, other abilities, prefab initialization and round
lifecycle remain required. The public C environment still uses its existing movement slice.


## Ragdoll in the composed runner

The actor runtime now accepts ragdoll impulses and runs SlimeController's timed
recovery and invalid-ground repair. The impulse preserves Unity's order: begin
the optional animation before changing mode, divide horizontal imposed speed by
1.5, derive orientation from the original vector, and retain ground attachment
and external velocity. The supplied magnet argument is unused in the pinned
source. Existing holder identity remains until recovery clears it.

Recovery uses the player's scaled/protected clock and a strict duration boundary.
It transfers imposed velocity into external velocity, reads the current sprite,
aligns facing, begins the fall animation, then casts using the standard fixed dt.
Missing/destroyed ground invokes the normal ground-loss listener and sound path.
Partial failures preserve the state reached before the exception. A scheduler
hook allows commands between components, including commands affecting other
actors; the tests do not assume all of one player's work occurs together.

All 120 scenarios match Unity, with 20 cases at each player count 1..6:
13,801 completed ticks, 3,812 impulses, 15,308 ragdoll actor snapshots,
7,313 grounded ragdoll snapshots and 842 with retained animation callbacks.
Four targeted failures match missing fast/slow/fall animations and a missing
renderer during recovery. Both fixtures run in the standalone verification suite.

This extends the supplied-state, empty-ability runner. It does not yet connect
ragdoll to ability casts, MagnetSteal, swallowing, ropes, deaths or full prefab
initialization. The full-game C environment and remaining integration are still
required; this checkpoint does not certify complete ability behavior.


## Ordinary ability cooldowns and readiness

`ability_routing.rs` now supplies cooldown updates and cast gates to the scheduled
SlimeController. Slots alias ability objects, including null references and
managed-destroyed or physically absent Unity objects. Updates retain the source
order: compare readiness, increment the timer if permitted, reset an indicator
on a readiness crossing, then apply the zero-cooldown adjustment only when an
indicator slot exists. Ragdoll/biting still update cooldowns; the nothing state
and controllers already inside an ability skip them.

The indicator threshold is `>=`, while casting requires `>`. The selection loop
checks cooldown-ignore before buttons, then readiness, ground restrictions,
player permission, ability castability and suppression. Unlike the cooldown
loop, selection does not skip null or destroyed slots. Released cooldown-ignore
slots reset to exactly the cooldown. Indices beyond the first three never have
a pressed ability button in the pinned Player implementation.

The current metadata models ordinary Ability behavior and the custom
IgnoreNextCooldown property. Rope-specific spatial queries and dynamic cooldowns
are not represented. Indicators are explicitly headless: reset clocks and null
reference failures are compared, while renderer/material/sprite presentation is
absent. Ability entry is still a required integration step: if selection returns
a slot, the runner returns `ability_entry_not_implemented` before executing it.
That is a capability boundary, not a claimed Unity failure or a successful cast.

All 120 reference scenarios match, including 100 complete sequences, 12,002
completed ticks and 20 matching partial source failures. Casts are blocked using
released buttons or CanUseAbilities; separate calls to Unity's actual readiness
and castability methods exercise true and false predicates. These checks do not
yet prove successful selection/entry or any complete ability implementation.

### Rollout catalog routing and Random (2026-09-14)

The public rollout backend now embeds common metadata for all 39 catalog entries,
generated from each shipped prefab by `extract_rollout_data.py`. Cooldown,
ground-only, time-stop cooldown, castability, instant/rope and offensive fields
therefore retain their authored Q32.32 values rather than using the previous
four-second placeholder for unimplemented abilities.

Catalog id 1 (Random) is resolved at the same point as
`GameSessionHandler.SpawnPlayers`: after level-intro RNG draws and before ability
objects are constructed. Each Random slot draws uniformly from ids 2 through 38.
When all equipped slots are Random and no draw through the final slot was
offensive, the final slot is redrawn once, matching the source's deliberately
non-guaranteed fallback. The replay header retains the configured Random ids while
the live slime observation exposes the resolved abilities.

`random_ability_ab.c` compares the NativeAOT oracle and optimized Q32 Rust backend
over six seeds, all five maps, mixed and all-Random configurations: 72 resolved
slot ids and cooldowns match exactly. This certifies the complete behavior of the
Random catalog entry; behavior of the concrete ability selected by it remains
covered only when that selected ability itself is listed as ported.

### Executable Gold mushroom (2026-09-14)

Catalog id 34 now executes the shipped `SuperStar` and `SuperStarEffect`
components in the public rollout backend. The prefab starts on a seven-second
cooldown. Casting installs one effect for the selected slot, changes that slot
to the common 10,000-second live sentinel and begins updating it on the following
updater pass. The exported effect lasts exactly four seconds; its stale source
comment is not used as behavioral evidence. On each airborne jump edge it
overwrites the player body's self-imposed vertical velocity with the normal
38.25 jump strength multiplied by the prefab's exact 0.85 factor. Expiry restores
the separate five-second `SuperStar.cooldown` value.

`gold_mushroom_ab.c` compares NativeAOT and optimized Q32 Rust for 636 consecutive
ticks from loadout initialization through the initial cooldown, cast, next-pass
effect initialization, an air jump, exact expiry/restoration and a second cast.
Player position and velocity, slot cooldown and timer, observation flags and
activation telemetry match exactly. The state uses a fixed three-slot array and
does no per-tick allocation. Form-specific jump speed and the rope exclusion are
retained in extracted metadata for the later form and rope integrations; neither
path is reachable in this isolated verified loadout yet.

### Executable Teleport (2026-09-14)

Catalog id 25 now executes the two-stage `Teleport`/`TeleportIndicator` path in
the public rollout backend. The first instant cast allocates the indicator with
its extracted zero-gravity body and 1.32-radius layer-27 circle; registration is
deferred to the next updater pass and consumes the original three hierarchy
numbers. While the marker exists, every released tick resets the slot timer to
the exact three-second cooldown. A held second press first increments past that
strict threshold, then queues the indicator update later in the same pass.

The indicator performs the original two circle queries, de-duplicates the
marker-side results against the caster-side list, ungrounds matched players,
zeros their imposed velocity and applies opposite offsets. Each moved object is
inactive for the strict 0.6-second `TeleportedObjectEffect` interval. Its visual
has no policy row, but `RandomInsideUnitCircle` consumes four RNG state advances
on every update, including the removal pass; Rust preserves those advances and
the stale `Player.Position` publication while the player object is inactive.
The implementation uses fixed six-player query arrays and allocation-reusing
marker/entity pools, with no hot-path allocation.

`teleport_ability_ab.c` matches all NativeAOT public fields for 300 ticks through
the stationary caster transfer. `teleport_multi_ab.c` uses three players and
moves the caster away from a teammate left at the marker; both directions of
the swap, both inactive effects and reactivation match exactly for 450 ticks.
That test also exposed the source's sorted-team spawn mapping and alternating
teammate offsets, now reproduced at reset. Projectile, item and tagged-boulder
targets are part of Teleport's mask but cannot be certified until their dynamic
entity types are present in the rollout backend; the player mechanic is exact.

### Executable Grenade throw and flight (2026-09-14)

Catalog id 3 now enters the shipped `ThrowItem2` form and creates the pinned
Grenade item. Extraction retains both grounded and aerial animation lengths and
throw offsets, charge-force bounds, recoil, owner-hurt delay, item hierarchy,
circle radius, mass, bounce/friction, gravity and drag, and the downstream
explosion metadata. The held object consumes its prefab hierarchy at the
original enter callback but has no DetPhysics row until `Fire` calls the
manually initialized circle in the release pass.

The grounded path preserves the source's split bodies: release advances the
active ability hurtbox once by the ten-unit recoil while the deactivated slime
body remains at its platform pose; exit snaps the slime back and starts the
`justlanded` animation at the measured callback phase. The thrown circle then
uses the extracted one-third inverse mass, gravity -40 and 0.999 air drag with
the same force-to-impulse conversion. Storage is a reusable item pool, so the
per-tick ballistic pass allocates nothing.

`grenade_ability_ab.c` compares every public header, player, slime, telemetry and
entity field between NativeAOT and Q32 Rust for 700 ticks. It covers readiness,
15-tick grounded entry, a seven-frame hold, charged release, 12-tick exit, the
free trajectory, delayed owner contact and callback destruction. The explosion
registers on the next updater pass with its two extracted startup shockwaves,
one-frame lethal box and exact 90-frame collider lifetime. The self-kill also
exercises the shared death path: kill credit, old collider removal, new player
and copied-ability hierarchy, DetPhysics circle-before-box IDs, three stock
choice indicators, their 360-tick timeout, fast spawn and subsequent landing
all match.

`grenade_fuse_ab.c` adds a separate 700-tick exact comparison which never
releases the cast button. It covers the authored three-second fuse while held,
the prefab child's local-x-only facing mirror, detonation before the manually
initialized DetPhysics circle can consume an entity ID, and the same lethal
explosion/stock recovery path. Held grenades remain payload state rather than
registered colliders, and their fuse pass reuses the existing item pool.

`grenade_air_ab.c` jumps four ticks before readiness and compares 100 ticks
through the airborne form, held charge, release, free flight and raycast
landing. Grenade's ability prefab has a distinct serialized `PlayerPhysics`
from the ordinary slime: gravity acceleration 1.60000002, maximum fall speed
27, air acceleration 10 and maximum speed 19. Those values are now extracted
rather than borrowed from the player. The implementation preserves the 0.03
pre-fire modifier, magnitude-nine clamp, full gravity after release, exact
`Fix.Lerp` evaluation with its serialized 0.6 float, and `PlayerBody`'s later
position integration/external-velocity damping. Moving ability-body deaths
also publish the source death position before parking a stock replacement.

The two shockwaves use the authored masks, radii and forces, apply their impulses
before map home springs, and reuse existing fixed arrays/entity pools. Grenade
terrain bounce, non-owner hits, grenade duplication and grounded-form forced
exits still require separate goldens before id 3 is complete.

### Executable Tack throw and trap (2026-09-14)

Catalog id 35 now shares the same allocation-reusing `ThrowItem2` form while
loading every form, body and trap constant from the Tack and BananaPeel prefabs.
The item receives its Projectile owner on the first updater pass after release,
sets angular velocity to four, and follows the normal circle-body integration.
A wall callback settles it only on a top-side contact after the 0.1-second launch
grace and with non-upward velocity. Each later updater resets the body to the
captured contact position before physics, matching the source's stable pin.

After the strict 0.8-second arm threshold, the trap scans live vulnerable
players in source order with its scaled-radius test. The owner remains immune
until leaving that radius once. A subsequent overlap creates the ordinary death
effect, kills with the Tack owner as attacker, and removes the trap before that
pass's DetPhysics contacts. The strict 20-second lifetime uses the player time
scale clock and removes an unused trap without a pop sound.

`tack_ability_ab.c` compares 1,450 exact NativeAOT/Q32 ticks covering airborne
entry, charged release, deferred owner binding, ballistic flight, settling,
stable pinning and lifetime expiry. `tack_kill_ab.c` compares 400 exact ticks and
drives the thrower away and back across the armed trap, covering owner-clear
immunity, the one-use kill, updater-phase removal and the stock-choice prefix.
The shared `Item.Duplicate` behavior remains for the Duplicator integration.

### Executable Mjolnir flight and recall (2026-09-14)

Catalog id 37 is the third consumer of the extracted `ThrowItem2` form. Its
MjolnirItem body has zero gravity and receives the source's signed angular
velocity and Projectile owner on the first updater pass after release. The
outbound pass applies the prefab's per-tick exponential-style damping and uses
the strict grace/velocity thresholds to enter recall. Recall adds a constant
acceleration toward 1.1 units above the owner's current live form, clamps speed,
coasts when no owner body is active, and removes the hammer inside the strict
two-unit catch radius or after its maximum lifetime.

Rounded-platform contacts retain ordinary rigid-body resolution. Once the clang
grace has elapsed, every queued callback adds the authored 3000 force along the
hammer's post-resolution velocity. The first outbound clang switches to recall;
later clangs keep applying force without resetting the phase. Per-hammer victim
history uses a six-bit fixed mask rather than the source `HashSet<int>`, avoiding
allocation while retaining its once-per-player semantics.

`mjolnir_ability_ab.c` compares 1,450 exact NativeAOT/Q32 ticks through an aerial
throw, outbound flight, stall-triggered recall and owner catch.
`mjolnir_wall_ab.c` compares another 1,450 exact ticks through a grounded throw,
terrain collision, platform motion, clang-triggered recall and catch. Victim-hit
and shared `Item.Duplicate` interactions still need dedicated goldens before the
catalog entry is complete.

### Executable Gust (2026-09-14)

Catalog id 7 now executes the shipped Gust `Ability`, `QuickSpell`, and
`AirWave_Shockwave` lifecycle in the public rollout backend. The extractor
retains the prefab's active circle and offsets, cast/exit animation durations,
forces, grounded-player multiplier, radius, and Hermite force curve as Q32.32.
QuickSpell copies the incoming grounded body, performs its first 0.05-unit
`UnGround` separation, clears both velocity channels, publishes the ability
collider through `Player.SetPosition`, casts after the measured animation
callback, and exits after the measured recovery callback. Because the original
`PlayerPhysics.UnGround` leaves `attachedGround` populated, exit deliberately
performs a second separation along the now-moving platform's current normal.

The spawned shockwave uses fixed storage and preserves Unity's unusual updater
phases. `Init` runs on the pass after the cast, captures at most 50 colliders
with exact circle/shape narrow phases, and applies the first time-zero impulse
before any existing updater. Map `AnimateVelocity` then sees and damps that
first impulse. The wave's later `UpdateSim` applies a second time-zero impulse
and advances the authored 0.25-second curve. Its captured platform-id array
rejects destroyed targets and computes closest surface points, current normals,
distance scaling, impulse arms, and body torque each pass. A center-radius player
approximation was rejected during measurement: the map-0 opponent passes that
broad phase but misses the actual oriented box.

`gust_ability_ab.c` compares NativeAOT with optimized Q32 Rust for 360 complete
ticks, including initial loadout rows, cast and public position arbitration, the
active Gust circle, deferred wave registration, all six map-platform trajectories,
the complete force curve, exit and later recovery. Headers, players, slimes,
telemetry, dynamic positions/velocities/rotations, hierarchy numbers and entity
lifecycle match. Static authored collider vectors allow one Q32.32 least bit;
the observed case is Dash's inactive half-extent imported through a Unity float
path and has no semantic effect. Cross-ability targets beyond the currently
executable public forms remain part of those abilities' later integration.


## Executable Invisibility and streamed actor verification

The composed runner now executes the original Invisibility behavior from player
input through InstantAbility entry, the effect, expiration and cooldown recovery.
`ability.rs` supplies the complete PlayerInfo handoff shape and shared player
flags. `slime_ability.rs` writes the entry package in source order, preserves the
instant ability's original-body rope reference, updates its own transform and
shared player clocks, requests audio, deactivates the instant object and resets
the selected timer. Missing renderer/transform/effect links retain the exact
partial state reached before the exception. Unbound handlers still fail explicitly.

The asset's initial cooldown is 2 seconds; its restored cooldown is 6 seconds.
A cast sets the live cooldown to 10,000 seconds and resets both invisibility
clocks. At expiration SlimeController restores each sentinel cooldown, then resets
the first minimum-age Invisibility slot's timer. Aliased slots, strict ties and
managed-destroyed components are preserved. Multiple distinct Invisibility slots
can chain casts before the slime observes a visible frame, delaying restoration.
The reference uses the original InstantAbility/Invisibility methods and UnityEvent
callback, with values checked against the exported prefab.

The public rollout backend now uses this behavior directly. `Player.UpdatePlayer`
runs for every player before any slime update, expiry uses the strict `> 5`
comparison, and the following SlimeController pass performs cooldown restoration.
The optimized Q32 cross-backend `invisibility_ab.c` scenario matches NativeAOT for
637 consecutive ticks, including an initial cast, a second-slot cast on the exact
expiry frame, a later visible frame with restoration, and a recast after the
restored six-second cooldown. Every player flag, invisibility clock, ability clock,
slot cooldown, timer, id, and activation counter in that projection matches.

All 24 long scenarios match: four per player count 1..6, 23 complete 1,800-tick
sequences, 41,400 completed ticks, 339 Invisibility audio requests and 203 observed
expirations. The remaining case matches a source division-by-zero partial failure.
Seven targeted entry/expiry cases also match, including four first-tick failures.

Full frame state can now be written and compared incrementally. The streamed
protocol retains every state field while bounding memory to the current frame;
it requires ordered case/action IDs and a complete success or partial-failure end
record. It rejects missing, duplicate and excess records. A fresh 13-case check
reproduces the earlier buffered Invisibility and failure outputs exactly. The
long buffered attempt was incomplete and is not evidence of parity.

This is the first executable ability in the composed runner, with supplied initial
actor state. It does not certify full matches: death/revival/stocks, actor rope
integration, map/prefab initialization, other ability interactions and presentation
still require work. Position arbitration and collider scale are covered below.
Indicator renderers are absent in this corpus. Physically destroyed instant-object
binding is explicitly unsupported in this reference setup. The public C environment
still uses the earlier Level1 movement/dash slice.

### Shared player position arbitration

`player_position::State` now runs from the scheduled Player update and original
Slime motor order. Controller identity survives across frames. Valid controllers
with more ability slots take precedence; equal-priority calls only replace the
current value when their controller was the previous frame's winner. Destroyed
controllers cannot claim the position. Empty frames retain the position value
while advancing the setter references. Optional serialized state preserves the
older fixture format.

The Unity runner binds and snapshots actual private setter references and invokes
original `Player.SetPosition`, including calls before player updates and between
component/physics phases. Shared controller ownership and managed destruction are
covered. This does not establish clone spawning, physical destruction of bound
controllers, or ability object activation/exit parity.

### Actual actor object activity and DPhysicsBox

Actors can now own a real Unity component graph on the collider's GameObject in
the reference runner. Original Awake methods run once before supplied state is
bound. GameObject activation, PlayerBody.simulated, collider enablement and scale
are connected to scheduled updates and the shared physics scene. This retains
supplied-state initialization; it does not invoke prefab Init/spawn or enable all
components registered by Awake.

The native entity tracks collider scale separately from the fixed transform and
Player.Scale. SlimeController writes the player's scale after OldUpdate and
invisibility cleanup and before Move, even in nothing/biting/ragdoll states.
DPhysicsBox scaling mutates its cached scale, transform offset/scale and geometry
in source order, retaining each successful prefix on division errors. Explicit
physics synchronization rotates the collider and updates its center/body angle.

The original callback gates are also implemented: detach always changes the
slime's fall state, sprite flip and speed cache, but only changes animation when
body simulation is enabled. Attach callbacks require both body simulation and
the hurtbox to be enabled. Actor activity controls later component calls in the
same update. The reference driver continues to call original components in the
scheduler order; callback recording still uses the scene adapter.

48 independent 300-tick episodes and eight scale/synchronization edge cases are
retained as evidence. The following section extends this to ordinary ability
entry/exit. Actual clone/prefab spawning, body death/rope interaction and the full
environment remain unimplemented here.


### Ordinary Ability actors and SlimeController return

The composed runtime now supports separate ordinary ability entities, linked to
primary slimes by their ability pools. Original Ability entry receives the full
PlayerInfo, activates its object, binds the slime and callback, updates collider
scale and arbitrates shared player position. Its scheduled UpdateSim applies
scale changes and writes position. Exit deactivates the ability and synchronously
calls the primary slime's original onExitAbility. Entity changes affect subsequent
components within the same scheduler tick.

Slime returns restore invisibility cooldowns when needed, scale the hurtbox,
clear player flags, synchronize physics, select ground/air animation, run landing
or jump behavior, synchronize the collider and reactivate readiness indicators.
The native port preserves mutations before null-reference, index and division
failures. The dead return branch currently reproduces the failure from the absent
PlayerCollision component; successful death/respawn is not covered here.

The source runner installs real Ability components with an empty IAbilityComponent
array. Entry events come from the existing passive hook at the actual cast boundary.
No substitute concrete ability callback is installed. Return calls and stress
activation commands are supplied by the corpus; concrete abilities must still
supply their own timing, effects and exit payloads. Rope transfers are explicitly
unsupported, and complete prefab Init, clone ownership and full matches remain
outstanding. The public C environment does not yet use this composed runtime.


### Concrete Dash in the actor scheduler

An ordinary ability actor can now install an original Dash component. Ability
entry invokes its OnEnterAbility and the scheduler subsequently invokes UpdateSim.
The native path keeps the same cached PlayerInfo/player, pause clock, aim, render
flags, update counter, ray hit list, wall reference and local landing position.
Returns occur inside Dash's own update, through Ability and the original slime
callback; the main corpus supplies no exit or activity commands.

Aim uses the player's quantized angle byte independently of keyboard movement.
The existing exact input-vector implementation supplies native aim. RaycastAll
appends rounded rectangles, circles and boxes, excludes the caster, applies the
layer mask/activity gates and sorts with the game's Mono algorithm. Hits retain
PhysicsParent hierarchy by value, including the source's wrapped subtraction tie
breaker. Scene objects used by this query carry explicit hierarchy identities.

Dash entry synchronizes its own physics/collider, detaches from ground and clears
velocity. After windup it selects a ray result; after the configured number of
updates it relocates to the wall or air endpoint and automatically returns. Exit
uses space-level speed scaling and the original water clipping calculation.
The source can divide by zero when a horizontal dash exits below water; the port
matches that failure and its partial state.

The long corpus uses a wall-only mask. Non-wall edge cases compare the actual
failure prefix from the absent PlayerCollision component. They do not establish
combat damage or successful non-wall traversal. Rope transfers, death/respawn,
complete prefab spawning and full environment integration remain outstanding.
Smear renderer geometry/material colors are presentation-only and are not in this
projection. The existing C environment still uses its earlier restricted Dash
implementation; this new component belongs to the separate composed runtime.

## Shared actor ropes and ability handoffs

Actors now reference the same rope segment arrays that the scene solver updates.
Player-owned rope metadata records owner ID and both airborne clocks separately,
without copying solver state. Body update/late update, grounded physics pulls,
jump release, ordinary ability entry/exit, Invisibility packages and both Dash
endpoint paths use those references. Slime detachment callbacks run before the
late-update endpoint write, including callbacks that fail.

The source comparison covers 60 complete movement episodes / 14,400 ticks,
plus 12 Invisibility/Dash episodes / 2,031 completed ticks (11 complete episodes
and one original division-by-zero failure), and four late-pull boundary cases.
This supplies existing rope objects; Hookshot creation, flight, reeling, rope
pool reuse, disabled-this-frame lifecycle and full combat remain unfinished.
The public C environment still exposes the documented movement slice.

## Composed actor destruction and death

Actor entities now retain final snapshots after Unity object deletion. DestroyFix
marks the actor and descendant components immediately; queries may prune their
colliders before the current component call returns. The updater then removes
marked components and deletes objects in its original order. Surviving actors
continue stepping. Player position references and ability links reflect removal.

PlayerCollision.killPlayer now shares the scheduled body, rope, player death
ledger and ability graph. Direct deaths detach ropes before the invulnerability
and activity gates, dispatch Dash exits through the original handoff order,
apply camera trauma and death accounting, and destroy represented ability actors
before their owner. Collision callbacks can call this same path and preserve the
callback prefix on failure. These are composed-runtime features; the public C
world retains its documented movement scope.

Fresh source comparisons cover 162 cases and 3,216 successful ticks: 54 removal
cases, 66 direct-death cases, 18 rope/Dash handoffs and 24 contact cases. Of these,
138 episodes complete and 24 reproduce original failed states. The contact set
accounts for 42 body deaths and 42 final-death events, including duplicate-contact
suppression. See VALIDATION.md for the specific conditions and evidence.

This is not full combat or respawning. The death-effect prefab is absent from
this binding; successful direct and Dash deaths request no visual effect, while
successful contact deaths use the original already-reviving branch that skips
that spawn. Missing-prefab failures are explicit. Effect animations/hitstop,
revive indicators, stocks, permanent revival, clone creation, drops and pooled
non-actor ability destruction remain unimplemented. Contact binding currently
covers Hitbox hazards and actor colliders with default collision settings;
projectile/orb/explosion metadata, ignored references, pickup platforms and
Dash's non-wall damage ray traversal still need integration.

## Spawned death effects in the actor runtime

`death_effect.rs` connects the actual `deathEffect` prefab to composed actor
kills. Two newly registered components follow the scheduler's pending/init/live
queues and hierarchy offsets. The original anticipation/start/end animation
callbacks drive vibration state, all 15 rounded-rectangle rays, splash positions,
and deferred GameObject deletion. Normal NoRevive contact deaths can now finish.
Dash death handoffs also create the effect before returning to and killing the
owning slime. Material IDs preserve the source player's material lineage across
Unity's renderer-local material clone; scale retains the original float cast.

The source adapter instantiates the real loaded player prefab and observes its
original DeathEffect2 and SpriteAnimator methods. It compares full animator state,
raw fixed-point surface positions, registered component identities/hierarchies,
pending/live queues, hitstop, actor state, death bookkeeping, and subsequent
physics. Source sprite aliases and animation durations are checked against the
pinned exported asset. The collision adapter binds the actual SlimeController or
Ability holder for this path, avoiding its auxiliary Hitbox's null material.

The retained corpus has 48 cases (eight each for one through six players), 42
complete episodes and 5,160 completed ticks. Six missing-camera cases retain the
source exception after prefab birth, before effect initialization or death
accounting. Across the corpus there are 72 effect births, 66 effect removals, 593
surface splash hits, 60 accounted body deaths and 60 final-death events. The six
effects left alive belong to the aborted source cases. Normal collision deaths
account for 21 of the body deaths. Tests include time stop, simultaneous deaths,
Dash handoffs and blocked owner deaths; all raw callback and registration order
is compared. Earlier lifecycle/death/Dash/rope fixtures still pass.

Run `python3 native/tools/verify_death_effects.py --golden` for all 48 retained
cases. Fresh capture defaults to `artifacts/death-effects`; select a new
`--directory` when that path already contains evidence. The successful capture
and checksums are in `artifacts/death-effects-full/report.json`.

Unity render-frame vibration displacement, randomly oriented/scaled visual
splashes, and audio sample selection are outside this gameplay comparison. The
prefab's pinned hitstop is zero; nonzero hitstop in the outer host loop is not
certified here. Stock/respawn/drop/clone creation and other ability handlers are
still separate outstanding work. This does not expand the public C environment
beyond its documented movement/dash slice.

## Scheduled first spawn and revival

`actor_spawn.rs` binds the existing spawn state machine to live actors, shared
player revival/ability flags, colliders, indicator clocks, and scheduler gates.
SlimeController.Init now runs for actors carrying spawn metadata. Its animation
callback enables physics and body simulation at the source's exact position in
the update order. PrepareToRevive, ReSpawn, ReSpawnFast and the permanent-revival
kill branch use the same body and animator. Partial writes survive callback or
reference failure. SpriteAnimator's spawn callback has its own identity, separate
from locomotion and death-effect callbacks.

The reference calls the original SlimeController methods, including Init through
Updater and Spawn through the actual SpriteAnimator callback. Source and native
traces compare subsequent movement, collider participation, shared player flags,
indicator activation/reset clocks and death-effect scheduling. Tests include
calls before ticks, before physics, and inside a body's late-update phase;
protected/unprotected actors during time stop; repeated fast respawn; tutorial
hurtboxes; and null/deleted ability, indicator, renderer and animation failures.

The `actor-spawn` corpus has 78 cases: 48 complete 210-tick episodes (10,080 ticks)
and 30 exact source failures (six missing-animation and 24 null-reference cases).
`actor-spawn-initial` adds six complete episodes with every player's initial
HasSpawned flag false: 1,260 ticks and 21 observed first-spawn completions.
Together these are 84 cases, 54 complete episodes and 11,340 completed ticks,
with every one-through-six-player roster represented in both sets. The main
corpus observes 24 false-to-true HasSpawned transitions, 99 physics re-enables and
48 indicator activations. These are frame-state transitions, not an exhaustive
count of calls within a frame. The permanent-revival branch keeps its body alive
and emits no Player.Kill death event, matching the source.

All cases are retained in `fixtures/actor-spawn*.jsonl.gz`. Run
`python3 native/tools/verify_actor_spawn.py --golden` and the same command with
`--initial`; raw inputs/outputs and hashes are under the matching artifact names.
Existing standalone spawn, locomotion, death and death-effect fixtures pass.
This integration uses explicitly bound actors and indicator lifecycle state.
Clone allocation, Revival ability placement/travel, stock choice and full match
initialization remain outstanding; the public C environment's scope is unchanged.

## Rewind history recording

`rewind.rs` connects the original RewindRecorder history-write behavior to the
actor scheduler. It stores all 188 samples, including position, both velocities,
rotation and three cooldown timers. Recording continues during time stop and
while body simulation is disabled, and pauses when the owner is inactive or its
controller is marked destroyed. Whole-object deletion retains the final history.
The standalone actor runner still limits this fixture to history recording;
the public rollout integration below executes playback against the full map.

The original Unity RewindRecorder.UpdateSim matches in all 48 comparisons, eight
cases for each one-through-six-player roster: 36 complete 410-tick episodes
(14,760 ticks) and 12 source index errors from short timer/history arrays. Every
history slot is compared after every tick, along with the complete actor, scene
and scheduler trace. Coverage includes 48,852 writes, 234 buffer wraparounds,
15,090 writes while body simulation is disabled, 4,410 at zero time scale, and
2,850 frames preserving the entire recorder state. Respawn, repeated recording
pauses, controller destruction flags and deferred object removal are included.

All cases are retained in `fixtures/actor-rewind.jsonl.xz`; XZ's larger compression
window stores the repeated full history buffers efficiently without dropping any
samples. The verifier uses Python's standard-library lzma support. Run
`python3 native/tools/verify_actor_rewind.py --golden`; fresh captures use the same
command without `--golden`. Raw evidence and exact source/input/output hashes are
in `artifacts/actor-rewind/report.json`. Existing spawn and death regressions,
Rust tests/lints, Python checks and the C smoke test pass. Clone allocation is
covered below.

### Executable Rewind playback (2026-09-14)

Catalog id 31 now connects that fixed 188-sample history to the public rollout.
Its fixed-capacity path contains at most 15 ten-tick Catmull-Rom segments and a
121-entry arc-length table, so recording, path construction and playback do not
allocate. A successful cast selects at most 150 prior ticks, plays them over the
source's integer-truncated 17/25 duration, and evaluates the same smootherstep,
spline and arc reparameterization arithmetic. `PlayerBody.AddForce` uses the
original inverse-mass cancellation rather than assigning velocity directly.

Playback disables PlayerPhysics and input while preserving PlayerBody integration
and ordinary cooldown advancement. Each path segment raycasts against rounded
platforms and applies the authored 3000 force; waking a quiet map also invalidates
the platform fast path. Completion occurs on the following updater pass. It runs
the six-pass rounded-rectangle depenetration, restores both velocity channels,
rotation and the two other cooldowns, preserves the live Rewind cooldown, and
repeats `UnGround(false)` against the platform reference deliberately retained by
the source. The other cooldowns then advance once later in the same pass.

`rewind_ability_ab.c` compares every public header, player, slime, telemetry and
entity field for 300 NativeAOT/Q32 ticks. The scenario records grounded and aerial
movement, reverses direction, casts from a moving platform, observes the full
102-tick suppressed playback, pushes that platform, and completes the retained
ground release with no death. All fields match exactly.

## Scheduled map-boundary deaths

`actor_bounds.rs` runs the player's DestroyIfOutsideSceneBounds behavior inside
the actor scheduler: strict grass/space limits, axis priority and enable flags,
water and ordinary death effects, player material selection, and the distinct
dash/PlayerCollision/no-collision-component exit paths. The existing effect
registry now also runs the actual water prefab's PlayAnimation lifecycle. Water
effects scale with the body; side/upper effects keep their prefab scale. Effects
remain observable when a later renderer or camera access fails.

The three retained corpora contain 198 exact source comparisons, with every
one-through-six-player roster represented: 174 complete episodes, 33,300 ticks,
and 24 null-reference failure prefixes. The basic set has 150 cases; renderer/
camera failures add 24; dash and permanent-revival interactions add 24. All 24
interaction cases actually create their intended boundary effect. Across the
corpora, 126 effects are created: 108 are deleted and 18 remain only in aborted
source frames. The traces compare 96 accounted body deaths and 90 final-player
death events; the difference includes bodies that still have a surviving clone
in the player ledger. This verifies ledger behavior, not clone allocation.

Two binding details are checked against the real source. The loaded water
prefab's nonserialized FixTransform.position is zero even though its saved
position is -1.47. Components on inactive ability objects must run Awake before
the supplied scheduler state is installed; otherwise activation registers an
unmapped second boundary checker. Registration capture now rejects unknown
components. Dash cases start airborne so SyncPhysicsTo does not move them back
to a platform before the boundary check.

Run `python3 native/tools/verify_actor_bounds.py --golden`, adding `--faults` or
`--interactions` for the other sets. Full traces are in
`fixtures/actor-bounds*.jsonl.xz`; raw evidence and assembly/input/output hashes
are under `artifacts/actor-bounds`, `actor-bounds-failures`, and
`actor-bounds-interactions-exits`. Earlier diagnostic/incomplete interaction runs
are kept separately and are not the retained regression corpus. Existing death
effect and spawn fixtures, Rust checks, Python tests and the C smoke test pass.
Render-frame-only water rings/audio are outside these Updater traces. Projectile/
item boundary paths and the full match environment remain
outstanding; the public C environment retains its documented movement/dash scope.


## Scheduled abilityless clone allocation

`actor_clone.rs` now creates the pinned empty-player prefab in the composed actor
runtime. Clones receive distinct actor, collider, hierarchy and scheduler IDs,
share their owner's input and player ledger, and keep the prefab's real Awake,
Init, preparation and animation behavior. Their initially null cooldown array
remains null until SlimeController initialization. A clone deleted before Init
never gains a collider. The source probe instantiates the original prefab and
observes its components rather than constructing replacement controllers.

The basic corpus covers creation before a tick and during Init, Update, physics
and LateUpdate, two births in one callback, the body-count limit, a null prefab,
and deletion before initialization. It contains 54 cases across one through six
players: 48 complete episodes, 7,200 ticks and six matching null-reference
prefixes. The source increments the body ledger before accessing a null prefab;
the failed allocation therefore reserves a body without creating an actor.

Six saturation cases create 336 clones across the six rosters and remove all of
them. The six-player case reaches 102 simultaneous player bodies: each player
has the original plus 16 clones. The 21 further calls return null without
incrementing the ledger. This follows the original strict count guard.

The 60 interaction cases complete 14,400 ticks, create and remove 90 clones,
account for 96 body deaths, and create and finish 90 death effects. They
exercise clone death and replacement, creation from a
death-effect Init callback, cloning from a surviving clone after the original
dies, water deaths, shared movement/buttons, time stop and protection, scale
changes before Init, and inherited player scale. Water cases explicitly select
the grass boundary; spawn invulnerability does not prevent boundary death.
Coverage checks require the intended births, body deaths, spawn completions,
removals and completed death effects before retaining the corpus. Full actor,
physics, scheduler and shared-player state is compared after each tick.

Six additional cases change shared buttons after the original body dies and
create a replacement from a surviving clone. All 1,440 ticks match, including
12 clone births/removals and 18 body deaths. New clones inherit current buttons
from a surviving body; deleted originals retain their final observed snapshots.

Run `python3 native/tools/verify_actor_clones.py --golden`, adding `--saturation`,
`--interactions` or `--orphan-inputs` for the other sets. Full traces are retained in
`fixtures/actor-clones*.jsonl.xz`; raw evidence and assembly/input/output hashes
are under `artifacts/actor-clones-verified`, `actor-clones-saturation`, and
`actor-clones-interactions-verified`, and `actor-clones-orphan-inputs`. The earlier interaction pilot matched
state but missed two intended water deaths and is not the canonical corpus.

This is the shared abilityless-clone factory, not the complete Duplicate or
Revival ability. Their casting/placement, stock selection and the
full match environment remain outstanding. Eating is covered below.
The public C environment retains its documented movement/dash scope.


## Eating and shared player growth

`actor_eating.rs` connects SlimeController.OnCollide, Bite and Chew to the live
actor runtime. Contact entry checks grounded state, size difference, teams,
time stop and the victim's active collision component. Bite keeps its target
reference; Chew resolves that target's active body again when the animation
finishes. A victim can enter or leave Dash between those calls. Chew uses the
existing death/rope/ability cleanup path with cause Eaten and no death effect.
Swallow and FinishEating run through the original animation callback order.
Growth updates shared Player.Scale and its 0.3-to-3 clamp, then updates the
actor's box, including other surviving clones on their scheduled turns.

Six retained corpora match the original Unity methods in 225 cases: 201 complete
episodes, 24 matching missing-animation failure prefixes, 24,504 completed ticks
and 18,562 collision callbacks. The sets contain 187 bite audio requests, 166
swallow requests, 99 accounted body deaths, 152 eaten-counter increments, 2,183
frames of increasing scale and five frames of decreasing scale. There are 36
real clone births/removals in the eating interactions. Contact and Dash tests
cover two through six players; the direct-method, clone, clock and failure tests
also include one player. Direct self-bite cases exercise the source method;
they do not imply that normal one-player contact can target itself.

The corpora distinguish initial invulnerability from becoming invulnerable
before Chew, target deletion/inactivity, repeated Bite, an active Dash versus its
inactive owner, ability exit/deletion, pending and spawned clones, replacement
clones, zero dt, slow time, time-stop protection, signed counter overflow, scale
limits, and missing bite/chew/swallow/idle animations. Every intended contact,
death, counter increment and clone lifecycle has a coverage check. The controlled
Dash set keeps the eater stationary so unrelated landing animations cannot cancel
the callback being tested; the earlier moving pilot remains separate evidence.

Two unusual source behaviors are retained. Chew increments the eaten counter
when it finds a live collision component even if killPlayer refuses an
invulnerable victim. Growth adds a fixed amount per update, including zero-dt
updates when time is not stopped. Ragdoll can replace the swallow animation and
its completion callback while leaving the swallowing flag set; six cases then
grow to the size clamp and retain that flag. No repair is applied to these rules.

This work also fixes the observed timing of ability references: a surviving
owner sees a deleted ability become absent in the same frame. Deleted owners
keep their cached final state. Existing lifecycle, ordinary/Dash, ragdoll, death,
clone, Rust/Python and C regression checks pass.

Run `python3 native/tools/verify_actor_eating.py --golden`, adding `--faults`,
`--contacts`, `--handoffs`, `--clones` or `--timing` for the other sets. Traces are
in `fixtures/actor-eating*.jsonl.xz`. Assembly/input/output hashes and full raw
captures are in `artifacts/actor-eating-verified`, `actor-eating-failures`,
`actor-eating-contacts-verified`, `actor-eating-handoffs-controlled`,
`actor-eating-clones-verified`, and `actor-eating-timing`. Steam achievements and
render-only presentation are outside these headless traces. Remaining abilities,
stock selection and the complete match environment remain outstanding; the
public C environment retains its documented movement/dash scope.

### Circular collider components in composed actors

The composed runtime now binds either the original `DPhysicsBox` or
`DPhysicsCircle` component. Circle scale changes preserve the source's operation
order: divide by the old component scale, clamp, update the FixTransform scale
and offset, then scale the radius. A zero new scale can succeed; a subsequent
scale write divides by zero. Radius writes address the initial radius before
initialization and the registered geometry afterward. Circle synchronization
updates center and body orientation, including the existing composite path.

Four complete retained Unity corpora cover 198 cases across one through six
players, 16,890 completed ticks and 36,696 collision callbacks:

- 48 movement/activity cases: 11,520 ticks, mixed boxes and circles, radius
  writes, player scale, time scale, collider enable state and active GameObjects.
- 72 boundary cases: 54 completed ticks and 18 matching division-by-zero
  prefixes, including zero/negative radius, reversed clamps, uninitialized
  property writes and uninitialized synchronization.
- 24 plain Ability handoff cases: 4,320 ticks and 449 actual entries between
  independently scheduled SlimeController and circular Ability bodies.
- 54 destruction cases: 996 ticks, 69 removed actor bodies, parent-child
  cleanup and six matching null-reference prefixes for already-removed actors.

Every frame and exception prefix is retained in `fixtures/actor-circles*.jsonl.xz`.
Assembly/input/output hashes and raw captures are in
`artifacts/actor-circles-verified`, `actor-circles-faults-verified`,
`actor-circles-handoffs`, and `actor-circles-lifecycle`. The handoff suite uses
plain Ability components without a concrete spell effect. Fresh circle
instantiation, manual initialization and BoplBody initialization are not covered
by these corpora. This extends composed collider behavior; it does not complete
the remaining spells or the public C environment's full-game integration.

### QuickSpell animation and return path

The composed runtime now invokes the original QuickSpell entry, cast and exit
sequence: synchronize player physics, synchronize the circular collider, apply
the entry offset, unground, clear velocity, play cast frames, create the cast
particle, play exit frames and return through Ability to SlimeController.

A retained 48-case corpus covers 4,800 ticks and 168 actual entries, first cast
particles and automatic returns across one through six players. It includes
Gust/Revival animation and offset settings, nonzero custom offsets, time stop,
protected players, zero delta time, changing player scales and disabled
animation components. A further 54 cases retain 144 completed ticks and exact
failure prefixes for missing/null/empty animations, a missing renderer, a null
particle prefab and casting without a bound player. All 102 cases and 29,726
collision callbacks match Unity. The verifier requires an entry, cast and return
for every player in every normal case, and the intended exception in every
failure case.

These corpora deliberately leave QuickSpell's optional spell reference empty;
they do not validate Gust's shockwave or Revival's extra-life indicator. They
retain the actual AirWave asset frames shared by those abilities. The particle's
presence is observed, while its rendering and nondeterministic visual simulation
are excluded. Full frames are in `fixtures/quick-spell*.jsonl.xz`, with raw
captures and assembly/input/output hashes in `artifacts/quick-spell-verified`
and `artifacts/quick-spell-faults`.

### Gust shockwaves in the composed world

QuickSpell now instantiates a scheduled Shockwave from the pinned Gust prefab.
The new spell registry shares scheduler node and component identities with
actors, clones and death effects. The wave records the original 50-slot target
buffer, queries circles before boxes before rounded rectangles, pushes eligible
player bodies on activation, applies timed impulses to captured platform bodies
and removes itself through Updater. The boolean circle query preserves the
source's strict radius boundary, which differs from its collision-manifold test.

Five retained corpora cover 176 cases across one through six players, 15,194
completed ticks and 124,454 collision callbacks:

- 30 ordinary/time/scale cases create 105 waves; the 21 waves under zero delta
  time remain present, and the other 84 expire.
- 45 contact cases exercise 30 actual target pushes and 15 inactive, disabled or
  unpushable exclusions, including grounded force reduction and time stop.
- 18 clone cases create and remove 420 original empty-player prefab instances,
  interleave death effects with recasts, and fill the 50-target buffer in all
  four capacity cases with three through six players.
- 48 custom-configuration cases cover single activation, forced early removal,
  the unused ActivateOnStart flag, shifted/single-key/empty curves and errors
  after spell allocation. Eighteen exact failure prefixes retain 33 new waves.
- 35 interaction cases exercise ignored players/platforms, requerying inactive
  or destroyed targets, zero/negative scale and explicit destruction. Ignoring
  a player does not prevent its immediate push on repeated activation; the
  source only consults that set for continuous impulses.

The complete records are in `fixtures/gust*.jsonl.xz`. Raw captures and exact
assembly/input/output hashes are in `artifacts/gust-verified`,
`gust-contacts-verified`, `gust-clones-verified`, `gust-custom`, and
`gust-interactions`. Coverage guards require actual pushes, expected exclusions,
clone allocation/removal, full target buffers, wave expiry and intended errors.

This validates Gust against the composed player, clone, plain Ability and
bound-platform graphs. Concrete lethal-terrain BoplBody targets, BlackHole,
boulder tags and grenade achievement state are not yet covered. The remaining
spell kinds and full public C match integration remain open.

### BoplBody component methods

`bopl_body` preserves the original mass/inertia initialization, pre-initialization
setters, effective composite-body access, forces, scale correction and magnet
hold/drop state. The actual Unity BoplBody runs on real circle, box and rounded
rectangle components in the oracle; no collider force implementation is replaced.

The full retained corpus has 432 cases and 13,568 completed actions: 346 complete
cases and 86 intentional division errors with exact partial state. It includes
196 scale-driven mass changes, zero/inverted scale, live/destroyed composites,
zero mass/inertia, and magnet duration zero/negative/overflow behavior. All 432
records are in `fixtures/bopl-body.jsonl.gz`; hashes and raw captures are in
`artifacts/bopl-body-verified`. Run `verify_bopl_body.py --golden` to replay them.

These are component-method checks against seeded registered physics state.
The public rollout integration described below now covers fresh collider
initialization, dynamic marker registration, and the one-marker death path.

### Revival marker allocation and lifetime

QuickSpell can now create the pinned Revival marker with its original
BoplBody, circle and fixed transform. The world preserves the original Awake
registration order, fresh circle initialization, body scaling, camera/water
clamping, and component removal. Revive.SetReviveFlag shares the player's
extra-life status and marker references. Replacing a live marker ends it first;
ending it separately and then casting again retains the old destroyed reference,
as the source does. Particle-only visual tails are outside the fixed gameplay
state after their gameplay components are removed.

The retained standard corpus covers 72 cases, 7,200 ticks and 109,510 collision
callbacks across one through six players. It creates 273 markers and ends 42.
The custom corpus adds 72 cases, 5,508 completed ticks and 50,592 callbacks,
including 18 exact failure prefixes. It covers velocity, gravity, offsets,
pre-initialization scaling/removal, manual initialization, missing particles or
spells, missing exit animation and recasting after a marker was ended.

A further 30 interaction cases cover 3,000 ticks, 38,762 collision callbacks
and 84 real empty-player clones. They mix markers with clone births/removal,
repeated End calls and a custom QuickSpell that creates the marker without a
Revive component. Clone and marker physics IDs share a monotonic allocator.
Finishing a clone spawn clears ExtraLife while retaining the marker list; a
later recast sets the flag again.
Complete records and hashes are in `fixtures/revival-marker*.jsonl.xz` and
`artifacts/revival-marker-verified`, `revival-marker-custom`, and
`revival-marker-interactions`. Coverage requires marker/clone creation and
removal, the expected extra-life flag/reference counts, and the exact errors.

Marker creation now connects to the death and respawn path described below.
Copying all equipped ability types remains open; this is not a complete Revival ability
or full public C match implementation.

The public Q32 rollout now executes Revival's cast-side path as well. It uses
the exported QuickSpell animations and offsets, active 2.3-radius ability
circle, one-pass-deferred marker registration, exact layer/owner/body fields,
and global spawned-prefab hierarchy numbering. `revival_ability_ab.c` compares
650 complete NativeAOT/Rust ticks, including the cast, active body, marker row,
exit, landing, and stationary marker lifetime. The comparison is exact. It
also exposed the source observation order for dynamic shapes: DetPhysics
enumerates all circles before boxes and rounded rectangles, independent of
when a circle was registered. The Rust observation writer now performs those
three allocation-free linear passes while preserving registration order within
each group.

`revival_death_ab.c` then drives that same player out of the map and compares
900 complete ticks. At the death tick the old slime and its ability colliders
are removed while the marker and player life remain. On the following pass,
the copied Player/Revival/Dash/Platform prefabs receive their source hierarchy
numbers and collider identities, and the hidden clone appears after the
surviving original slime. The clone remains suppressed through the 60-tick
wisp and distance-derived particle flight. The port accounts for
ReviveParticle's duplicate updater registration, removes the marker on the
same pass as `ReSpawn`, advances fresh cooldowns through the 105-frame spawn,
and launches at exactly 11.25 units/second. All player, slime, telemetry and
entity fields match NativeAOT exactly across the scenario. Multiple marker
positions still need the general multi-clone rollout representation.

### Revival death, effects and particles

The composed runtime now executes the original marker-driven death sequence
for clones with no equipped abilities: allocate replacement bodies, clear the
old marker references, remove the dead body, animate ReviveEffect, send its
ReviveParticles to their markers and start the replacement spawn. Particle
initialization registers its update twice, matching the source, and arrival
ends the marker before starting the spawn animation. Time-stop ownership and
reads through destroyed managed transform references retain source behavior.

Six complete retained corpora cover one through six players:

- Particle: 126 cases, 24,072 ticks, including 6 exact null-reference prefixes.
  Covers repeated initialization, shared/moving/destroyed targets, zero and
  negative speed/time, explicit removal and arrival callbacks.
- Effect: 90 cases, 22,536 ticks, including 18 exact failure prefixes. Covers
  empty/null target lists, multiple targets, removal and missing references;
  all 189 emitted particles arrive and are removed.
- Death: 48 cases, 15,570 ticks, including 6 null-prefab prefixes. Covers
  simultaneous deaths, the body limit, multiple markers and time stop. Two
  space-map cases are negative water-boundary controls.
- Water: 6 additional cases, 2,160 ticks, with water deaths enabled for every
  player count. All 21 revival effects and particles finish and are removed.
- Mature clones: 6 cases, 1,800 ticks; 21 fully spawned clones die, allocate
  normal death effects and produce 21 replacement bodies and particles.
- Edge cases: 30 cases, 3,672 ticks; 18 mixed-owner time-stop cases finish
  63 particles. Twelve invalid holder casts preserve allocated objects and
  the source's deliberate SetPlayerId exceptions.

The exact records are in `fixtures/revive-particle.jsonl.xz`,
`revive-effect.jsonl.xz`, `actor-revival.jsonl.xz`,
`actor-revival-water.jsonl.xz`, `actor-revival-mature.jsonl.xz` and
`revival-edges.jsonl.xz`. Raw captures, hashes and semantic coverage reports
are in `artifacts/revive-particle-complete`, `revive-effect-complete`,
`actor-revival-complete`, `actor-revival-water`, `actor-revival-mature` and
`revival-edges`. Some particle/effect cases cause additional late deaths;
those captures retain the resulting pending effects and do not establish
that every newly allocated effect finishes within their time windows.

The equipped-clone extension below now covers Invisibility. Copying other
concrete ability graphs and integrating the full public C match remain open.

### Equipped clones and empty instant callbacks

SpawnClone selects fresh abilities from the old indicator sprites. It does not
copy the old ability objects or their mutable cooldowns. The runtime now models
this for Invisibility, including the separate pooled object registration,
three-slot indicator array, spawn completion,
casting and removal before the owning body. It preserves allocations made
before a later indicator or fixed-transform access fails.

The main corpus contains 66 cases and 14,400 completed ticks for one through
six players: 48 cases complete and 18 preserve exact failure prefixes. It
records 126 casts, 231 first-generation abilities, 21 second-generation clones
and 42 clone removals. Null and aliased old ability slots still produce fresh
objects when the indicator sprites select them. Short/null/oversized indicator
arrays retain the source's partial allocation and list-assignment order.

Twelve longer cases add 10,800 ticks, 105 casts and 21 visible expiries. They
cover cooldown restoration and repeated casts after the fresh prefab's initial
two-second cooldown. Six marker-revival cases add 2,160 ticks: 21 equipped
clones and their abilities are removed, 21 equipped replacements finish their
spawn, and all 21 revival particles arrive and are removed. A further six
Dash death cases cover 144 ticks with aliased ability slots and require the
same destruction sequence as Unity.
A six-case continuation adds 3,360 ticks and 18 casts by revived replacements.
Three replacements die at scene boundaries before casting, and one dies after
casting; all four removals match the source. Initial casters are deactivated at
action 200 so the subsequent input targets the surviving replacement bodies.

The NullAbility prefab is a placeholder with an InstantAbility component and
no FixTransform. Fixed instantiation therefore throws after allocating and
registering the object. Eighteen exact failure cases retain 24 allocated
prefabs, including an Invisibility allocation before a subsequent NullAbility
failure. These checks preserve the shipped failure; they do not make the
placeholder usable. Separately, 24 supplied component-graph cases cover the
empty-callback InstantAbility with a bound transform, audio requests and a
missing transform link: 2,700 ticks, 18 complete cases, 6 exact failures and 378 casts.
The empty callback never makes a player invisible.

The public match ABI also rejects catalog id 0 before scene construction. The
`null_ability_ab.c` cross-backend check confirms both NativeAOT and Rust return
the invalid-argument status while reporting the same 39-entry catalog. Together
with the retained instantiation failure prefixes above, this covers every
reachable behavior of the shipped placeholder.

Complete fixtures are `clone-abilities.jsonl.xz`,
`clone-abilities-long.jsonl.xz`, `clone-revival.jsonl.xz`,
`clone-revival-casts.jsonl.xz`,
`actor-death-aliases.jsonl.gz`, `null-ability-clones.jsonl.xz` and
`instant-ability.jsonl.xz` under `fixtures/`. Raw captures, hashes and coverage
reports are under `artifacts/clone-abilities-expanded`, `clone-abilities-long`,
`clone-revival`, `clone-revival-casts-complete`, `actor-death-aliases`,
`null-ability-clones` and
`instant-ability`. The 231 first-generation count excludes second-generation
and failed-prefix allocations; it is not the total number of pooled objects.

Physical destruction of arbitrary initially supplied instant graphs, cloning
other concrete abilities, pickups/drops, complete match lifecycle and the
full C environment remain open.

### Executable Smoke throw and persistent cloud field (2026-09-14)

Catalog id 13 now shares the extracted `ThrowItem2` form and carries its own
SmokeGrenade body, five-second timed detonation, and four authored cloud offsets.
`SpawnCloud` preserves the source RNG order: each offset consumes the inside-unit
angle and radius before its independent cloud rotation. Clouds register on the
following updater pass in hierarchy order, draw their ignition delay there, and
export their layer-22 circle with the seven authored enter radii. The hot path
uses retained vectors and fixed arrays; it does not allocate per tick.

The implemented interaction path also disables a cloud collider when an
ignite-capable grenade or smoke explosion overlaps it, waits the cloud's sampled
strict delay, and spawns the owner-attributed first-frame lethal explosion.
Per-form cloud retention is capped at the source's 40-effect ring size.
`smoke_ability_ab.c` compares 800 exact NativeAOT/Q32 ticks through a grounded
charged throw, ballistic item flight, timed removal, exact randomized placement,
radius animation, and persistence of all four cloud entities. Ignition chaining,
scale changes, ring eviction, and `Item.Duplicate` remain to be exercised by
dedicated goldens, so the catalog entry remains partial.

### Executable Time Stop and global simulation clock (2026-09-14)

Catalog id 12 now executes the shipped `CastSpell` form and `TimeStop` effect.
The grounded entry preserves the source's updater phases: `Ability.UpdateSim`
publishes the pre-callback root while `SpriteAnimator` can unground and resync
the circle later in the same pass. The collider retains the exact Slerp basis
rather than reconstructing it from its derived angle. Channel time uses the
caster-protected clock, release before the strict nine-second boundary cancels,
and a completed cast runs the authored cast and ease-out animations before
returning the slime at its 1.8-unit exit offset.

The spawned effect registers on the following pass, protects all members of the
caster's team, sets the global player time scale to zero, and restores it after
the exact float32 10-second duration plus the final 1.501036167-second alpha key.
Unprotected players retain their last accepted input and continue the source's
unscaled gravity, air damping, and external-velocity damping while their
position, raycasts, input changes, and cooldown clocks receive a zero delta.
Protected players keep their normal player clock, except abilities whose prefab
explicitly opts into `cooldownStoppedDuringTimeStop`. DetPhysics integration now
uses the post-effect zero delta for arrows, thrown items, and dynamic platforms;
their earlier updatables use the scale visible at the start of the pass. Timed
projectile, smoke, and explosion animation clocks follow the same split without
adding per-tick allocation.

`time_stop_ability_ab.c` compares 1,600 NativeAOT/Q32 ticks exactly. It covers a
grounded cast, effect registration and removal, team protection bits, 690 frozen
ticks, an airborne opponent with nonzero velocity, frozen opponent position and
cooldowns, protected-caster movement and cooldown exceptions, and final global
clock restoration. Interactions with ability entity types that have not yet
been ported remain part of those abilities' open scope, so coverage is marked
partial until the catalog is complete.

### Executable Spike and platform composites (2026-09-14)

Catalog id 11 now executes the shipped grounded `Spike` form. The cast callback
uses the attached rounded rectangle's live normal and exact far raycast distance,
replaces the attack retained by that ability slot, instantiates the manual
layer-16 box on the opposite surface, and applies the authored force at the
caster's contact point. Cast and recovery animation callback boundaries, the
root collider's pre-physics publication, grounded movement, and grounded exit
pose match the updater hierarchy.

`SpikeAttack` is represented as a child of the platform's DetPhysics composite.
The implementation preserves the source's summed mass and moment, separate raw
child bodies for observations, relative child position and orientation, and the
otherwise surprising split between the rounded-root position and
`CompositeBody.position` during box/rounded positional correction. The physics
pass re-anchors that split from the root `FixTransform` on the next tick, just as
`SynCompositeToFixTransAndRelativeValues` does. The attack evaluates the exact
fixed down-thrust curve without allocating, remains lethal after the curve ends,
and follows platform translation and rotation. Box/wall contacts use the source
manifold order, effective composite body for impulses, raw child inverse masses
for correction, and the callback's one-self-push-per-frame rule. Player-layer
contacts invoke the persistent Explosion's default owner-zero lethal callback.

`spike_ability_ab.c` compares 500 NativeAOT/Q32 ticks exactly through a grounded
cast, attack registration, all curve frames, recovery exit, persistent collider,
composite platform flags, cast-force motion, collision with a neighboring
rounded platform, positional correction, bilateral push, and later repeated
rigid-body contacts. The attack collider's vertical half-extent uses the same
`One / (One / extent)` round trip as `DPhysicsBox`; this accounts for the three
raw Q32 units between the serialized extent and observed axis and is not a
semantic difference. Dedicated recast-replacement and player-lethality goldens
remain open, so the catalog entry is marked partial.

### Executable Missile and guided box projectile (2026-09-14)

Catalog id 10 now executes `SparkShot` with its extracted form physics and exact
animation callbacks. The Spark prefab registers on the following updater pass,
starts along the form's copied up vector, and is then steered by rotating its
live `FixTransform` basis while preserving speed. The implementation mirrors the
otherwise subtle scheduler order: the older form stabilizes the projectile
before the appended `Missile` updater increments its instability counter, and
DetPhysics rebuilds the transform basis from body orientation after each pass.
Releasing the form leaves the projectile alive; two unstabilized updates later
its authored 1000-unit thrust begins.

The layer-10 box resolves and position-corrects against rounded platforms before
its callback queues the ownerless MineExplosion and marks the Spark destroyed
for one observation. The next pass removes that collider, registers the scaled
layer-18 trigger, applies its constant one-shot Shockwave, and disables the
lethal box after its first physics frame. Collision projections now fold over
fixed slices directly, removing their former short-lived heap allocation from
this projectile's per-platform hot path.

`missile_ability_ab.c` compares 500 NativeAOT/Q32 ticks exactly. It covers the
grounded entry, delayed spawn, two-direction guided turn, release, instability,
box/rounded manifold and platform response, callback destruction, deferred
explosion, first-frame hitbox flags, Shockwave, and the full 82-tick visual
lifetime. The observed explosion collider axes differ from the serialized
extents only by the source's `One / (One / extent)` fixed-point scale round trip.
Projectile/item/form contacts, time-stop and scale interactions, duplication,
and a dedicated explosion victim remain open, so coverage is marked partial.

### Executable Mine, original ThrowItem, and autonomous chase (2026-09-14)

Catalog id 24 now uses the older `ThrowItem` controller rather than the shared
`ThrowItem2` approximation. Its grounded enter/loop/exit callbacks, forced
surface-normal throw, aerial throw, deferred prefab registration, ownerless item
observation, and grounded exit restoration follow the authored component order.
The pooled item stores the Mine state machine directly, so hover, arming, target
selection, chase force, anticipation jitter, and detonation add no per-tick heap
work. Mine and Missile share their actual `MineExplosion` prefab, while Mine
alone supplies `Explosion.PlayerOwnerId`; the co-located Shockwave observation
owner remains zero as in Unity.

`mine_ability_ab.c` compares a five-player, 334-tick NativeAOT/Q32 scenario. A
fifth opposing team wraps onto the owner's map spawn, jumps into scan range, and
exercises the two-second arm, chase, collision ignition, random anticipation,
deferred destroyed-item row, credited lethal explosion, hitbox disable, and
stock recovery. IDs, ordering, layers, flags, radii, axes, inverse masses,
players, slime state, kill credit, and telemetry match. Dynamic entity vectors
are within 64 raw Q32 units (1.49e-8 units) after chase-direction normalization
and the resulting shockwave; the inactive/active Mine form center is within
5,000 raw units (1.17e-6 units) because `PlayerPhysics.SyncPhysicsTo` repeats a
grounded surface projection before adding the collider offset. These bounds are
far below collision or control thresholds and do not change state transitions,
hits, credit, or timing. `MineExplosion`'s remaining visual lifetime is already
covered exactly by the Missile golden because both abilities reference the same
prefab. Duplication, scaling, boundary effects, magnet holding, and dedicated
Projectile/Explosion igniter scenarios remain open, so coverage is partial.

### Executable Rock rigid form and bounce contacts (2026-09-14)

Catalog id 9 now executes `BounceBall` as its own dynamic circle rather than
moving the hidden slime body. Entry copies the grounded surface position into
the form, installs the extracted mass, inertia, gravity, damping, friction and
bounciness, and registers the circle synchronously on the first cast. The
physics pass resolves Rock/rounded-platform, Rock/Rock, and Rock/player groups
in stable source order using a reusable contact buffer. A player contact kills
through the normal callback bookkeeping and queues the source's bounce force
for the following integration pass; contacts with another Rock are nonlethal.
Arrows and explosion hitboxes no longer treat the hidden slime box as a Rock
hurtbox.

The late updater reproduces the authored additive pulse curve and the strict
`time > maxDuration` transition. Exit accumulates the inverse pulse deltas,
multiplies them by the exit curve, and restores the slime at the rigid body's
last position and velocity. The spawned collider row remains as a frozen,
disabled ability object after exit and is reconstructed with its original
hierarchy when stock or Revival creates a replacement slime.

`rock_ability_ab.c` compares 500 NativeAOT/Rust Q32 ticks through entry, falling,
terrain bounce, pulse, timed exit, and restored slime movement.
`rock_contact_ab.c` compares a three-player 360-tick scenario whose teammate
shares the spawn point; the first Rock frame kills that target on tick 181 and
matches owner credit, telemetry, destroyed rows, and stock recovery. IDs,
ordering, flags, layers, inverse masses and every discrete state transition
match. Position, velocity, rotation and pulse-scale values differ by at most 64
raw Q32 units (1.49e-8 game units) because the two runtimes round successive
curve interpolation and normalized contact operations at slightly different
points. That bound is far below every overlap and transition threshold in both
goldens and does not alter contacts, deaths, timing, or control state.

Dedicated Shockwave, scale-change, arrow-attachment, projectile/item-detonation,
and mixed transformed-form contact scenarios remain open, so coverage is marked
partial.


### Executable Push platform control (2026-09-14)

Catalog id 22 now executes the grounded `ControlPlatform` form. Entry copies the
source `PlayerInfo` pose, clears both velocity channels, disables the attached
platform's authored ground animation, and advances the exact ten-tick
`grabGround` callback boundary before channeling. While held, the form applies
the authored deadzone and 10-unit force at the player's contact point, disables
the platform's normal ground animation, and retargets `AnimateVelocity.HomePosition`
with the source's `Lerp(body.position, oldHome, t)` argument order and scene
bounds. Release restores the ordinary slime on the same surface.

The ability collider keeps Unity's two representations of rotation: its box axes
come from the scalar-angle lookup, while its local offset uses the exact stored
ground-normal basis. The rollout records that basis at DetPhysics sync, so the
later grounded-rider update cannot change the already-published collider. This
is retained scalar state and adds no allocation to the tick path. The time-stop
early return also preserves ControlPlatform's preceding 0.8 external-velocity
damping.

`push_ability_ab.c` compares 420 NativeAOT/Rust Q32 ticks exactly through
grounded entry, collider activation, 120 controlled frames, platform linear and
angular motion, spring-home movement, channel timing, button release, inactive
collider state, and restored slime movement. Platform destruction, sudden death,
and overlapping-player scenarios remain to be separately goldened, so coverage
is marked partial.

### Executable Freeze ray and FrozenSolid (2026-09-14)

Catalog id 36 now enters the shipped `GunTransform` form with the prefab's own
movement constants. The form copies the exact ground basis, uses 0.03 gravity
and a one-tenth airborne entry velocity while aiming, fires on button release,
and switches to full gravity, 50-unit recoil, half-strength aerial steering and
the 20-tick fire-animation exit. Its DPhysics box retains the same split as the
reference: offset translation uses the stored `FixTransform` basis while the
box axes use the scalar-angle lookup.

`FreezeRayGun` raycasts the registered Player and wall shapes and reflects once
from the map's Water box with the no-water mask. A player hit spawns the
collider-free `FrozenSolid` updater. `Begin` immediately applies zero-speed
ragdoll and the standard 0.05-unit ground detach; subsequent updates pin the
victim, refresh the one-second ragdoll and suppress the next PlayerPhysics pass.
After the serialized 1.6 seconds, thaw applies the `(0, 4)` fast ragdoll impulse
and resets all three cooldown clocks. The rollout stores these small records in
a capacity-reused vector, so the steady-state update path allocates nothing.

`freeze_ability_ab.c` compares the grounded player-hit scenario through the
complete freeze, thaw, ragdoll landing and cooldown restart, plus an airborne
diagonal cast through recoil and animation exit. The two scenarios cover 1,000
NativeAOT/Rust Q32 ticks exactly. Water reflection is source-ported but does
not yet have a differential scenario. A measured hit against an active Bow
form exposed an unresolved updater-order mismatch while restoring and freezing
the original slime: ordinary-slime hits are exact, but transformed-player hits
are not yet compatible. Coverage therefore remains partial.

### Executable Growth and Shrink rays on players (2026-09-15)

Catalog ids 8 and 17 now execute the same serialized `GunTransform` used by
Freeze: grounded and airborne entry, aim tracking, release-to-fire, 50-unit
recoil, gravity, half-strength aerial steering, landing exit and the 20-tick
fire animation. `ShootScaleChange` raycasts Player and wall rows and retains
the source's one-pass Water reflection path.

An ordinary player hit queues the collider-free `ScaleChanger` prefab for the
next updater pass. It evaluates the exported five-key fixed Hermite curve at
the source's pre-increment time and applies the signed 0.8 Player multiplier,
including the strict duration boundary, time-stop rule and `[0.3, 3]` clamp.
The rollout stores changers in a capacity-reused vector.

Scaling has three separately visible clocks in Unity. `Player.Scale` changes
after player components, the slime hurtbox copies it on the following player
pass, and `PlayerPhysics.radius` consumes that FixTransform scale one pass
later. Rust represents all three. It also retains the hurtbox's inverse
extents: `SlimeController` assigns the same scale every tick, and
`DPhysicsBox.Scale` repeatedly rounds `(One / oldScale) * newScale`, so its Q32
axes continue drifting by raw units after the curve finishes. Reconstructing
axes from `authoredExtents * scale` was visibly different; updating stored
inverse extents matches the reference and preserves the actual collision
shape.

`scale_rays_ab.c` compares 500 Growth and 500 Shrink ticks exactly through
form entry and exit, ScaleChanger registration, the complete curve, collider
and radius phase lags, and subsequent scaled movement. Water reflection,
non-player targets, active transformed-player targets, BlackHole growth and
orb conversion still require their concrete entity implementations and
differential scenarios, so both abilities remain partial.

### Executable Roll form (2026-09-15)

Catalog id 16 now runs the source `Roll` form rather than advancing an inert
cooldown. Entry copies the ordinary body's grounded attachment or airborne
channels into the form, uses the authored 1.14-unit `PlayerPhysics` radius, and
keeps the independent 1.20-unit lethal circle. A short release while airborne
uses the serialized ten-unit jump; a full one-second charge waits for landing,
then enables contact kills and drives the ground speed with the exported fixed
Hermite curve for two seconds. Jumping during the run transfers 75 percent of
the current surface speed into the ordinary body before the synchronous exit.

Ground motion preserves `Roll.Move`'s unusual collision discretization. It
computes `n = floor(abs(speed) * dt * density / hurtboxRadius) + 1`, advances
exactly `n` times by `speed/(n+1)`, and retains all intermediate circle centres
in a fixed 30-entry state array for `CollideExtraHurtBoxes`. This has no tick
allocation. The trail's first charged release also consumes the two measured
runtime hierarchy entries, so a later stock clone has the same entity ordering
as NativeAOT.

`roll_ability_ab.c` compares 500 NativeAOT/Rust Q32 ticks from charge through
release, repeated curved-platform circuits, jump cancel, high-speed airborne
motion, boundary death, and stock replacement. All state and entity metadata
match. Repeated surface local-to-world transforms accumulate at most 2,048 raw
Q32 units (less than 0.0000005 world units) in the comparator; velocity is exact
through the measured jump and fall. The difference comes from reconstructing
the curved surface position in Rust instead of retaining Unity's separately
rounded transform fields and does not cross a zone, contact, or state boundary
in the golden. Cross-platform hopping, an isolated early airborne release,
moving-platform turns, wall squish and mixed transformed-form contacts remain
open, so coverage is partial.

### Executable Blink gun and player QuantumTunnel (2026-09-15)

Catalog id 6 now executes its serialized `GunTransform`: grounded and airborne
entry, aim tracking, release-to-fire, the authored firepoint, 50-unit recoil,
slow aiming gravity, half-strength aerial control, landing exit and the 20-tick
fire animation. `ShootQuantum` uses the registered Player, wall and Water shape
order. A player hit replaces the normal 2.5-second cooldown with the prefab's
five-second cooldown and immediately initializes a collider-free
`QuantumTunnel` on the victim.

The tunnel disables the victim in the shot pass and advances by player-scaled
fixed delta time starting on its deferred registration pass. At
`LifeSpan - last(ScaleAnim.keys).time`, it reproduces the source's platform
candidate order, seeded `RandomInt` choice and `RandomFix(0, 1)` perimeter
sample. It checks the sampled player center against every other rounded
rectangle, attaches through the same surface-local conversion, and updates the
inactive FixTransform. The slime and DetPhysics collider therefore expose the
new location immediately while `Player.Position` remains at its last published
location until the GameObject is active and `Player.UpdatePlayer` runs again.
The strict `age > LifeSpan` comparison is retained. Candidate collection uses
a fixed 64-entry stack buffer, so neither the tunnel clock nor placement path
allocates per tick.

`blink_ability_ab.c` compares 500 NativeAOT/Rust Q32 ticks across the complete
measured sequence. The cast starts on tick 181, release hits player two and the
grounded recoil exits the form on tick 241, the seeded tunnel move occurs on
tick 295 at raw position `(125472406545, -43290835442)`, the victim remains
inactive through tick 314, reactivates on tick 315, and publishes the moved
`Player.Position` on tick 316. Header, player, slime, telemetry and every entity
row match, with the comparator allowing 64 raw Q32 units for dynamic vectors
but observing the exact measured landing. Wall, rigid-body, item and orb
tunnels, water reflection, transformed victims, rope/magnet detachment and a
dedicated airborne-release differential remain open, so coverage is partial.

### Executable Engine and attached thrust body (2026-09-15)

Catalog id 5 now executes the grounded `PlantObjectOnPlatform` form and its
`PlaceRocketEngine` payload. The five-frame cast projects the authored lateral
plant offset onto the attached rounded rectangle, preserves the live surface
normal, scales the plant height and collider with `Player.Scale`, adds the
10-unit off-centre cast impulse, and returns through the three-frame recovery.
Each equipped ability instance retains the source's four-entry ring buffer; a
stock or Revival clone receives a fresh buffer generation while already planted
engines remain in the world.

The spawned layer-21 box is a persistent child of the platform composite. Its
one-unit body mass and ten-unit moment contribute to the combined body, its
startup animation delays ignition for ten ticks, and its fixed Hermite force
curve drives the platform at the live surface point for 5.5181274 seconds. The
engine updates movable-platform homes after thrust, stops below the water line,
and tests its scaled flame circle against Smoke before the time-stop early
return. Engine and Smoke records reuse existing vectors and do not allocate in
the tick path.

This port also records two general DetPhysics composite rules exposed by the
Engine trajectory. `ResolveCollision` redirects impulses through a composite,
but rounded-rectangle positional correction still receives each serialized
child inverse mass. After integration, a positional correction moves the
`CompositeBody` while child collider centers retain their already-synchronized
value until the next physics pass. Preserving both rules removed a same-pass
0.00024-unit platform displacement and 0.0134-unit attached-box observation
difference without changing the previously matched pre-contact trajectory.

`engine_ability_ab.c` compares 500 NativeAOT/Rust Q32 ticks exactly through the
cast, recovery, persistent engine export, startup, sustained thrust, platform
spring response and the first platform collision. Dedicated smoke-chain,
four-engine eviction, scale-change, platform-destruction, time-stop and
multi-engine goldens remain open, so coverage is marked partial.

### Executable Tesla coil and connected lightning (2026-09-15)

Catalog id 23 now executes the grounded `PlantObjectOnPlatform` form and its
`PlaceSparkNode` payload. The cast uses the exported lateral offset, surface
projection and normal, player-scaled plant height, five-frame callback, 10-unit
off-centre platform impulse and three-frame recovery. Each ability instance has
the source's two-entry `RingBuffer`; planted nodes survive form exit and stock
or Revival cloning starts a fresh ring-buffer generation.

Each `SimpleSparkNode` registers its one-unit circle as a child of the rounded
platform's DetPhysics composite. A pair connects on the updater pass after the
new node is planted. Its lightning box registers one pass later while inactive,
the parent orb and line become lethal after the 54-tick enter animation, and the
child orb follows one tick later. The active line uses the exported thickness
and minimum extent, disables its collider after the strict five-second clock,
and reenables it after the serialized 82 animation ticks plus the callback pass.
Node and line contacts use their live circle and oriented-box geometry and keep
the source behavior that permits owner damage.

The third placement exposes two ordering details that are easy to lose in a
cleaner implementation. `RingBuffer.Add` marks the oldest node before the new
circle is attached, but DetPhysics removes that circle only at the start of its
physics pass. The cast impulse therefore uses a temporary root-plus-three-node
body; removal then rebuilds the root-plus-two-node body while copying that
temporary body's velocity and angular velocity. The old lightning row remains
for the destruction pass with owner zero and `TRIGGER|DESTROYED`, disappears on
the following pass, and the replacement line registers one pass after that.
Rust models those phases explicitly with capacity-reused node and destruction
buffers, with no per-tick allocation.

`tesla_ability_ab.c` compares 900 NativeAOT/Rust Q32 ticks across three casts,
including connection, staggered activation, a complete flicker and reenable,
ring eviction, destroyed-line lifetime, reconnection, platform composite motion
and persistent objects after form exit. Header, player, slime, telemetry and
entity metadata remain exact; entity axes differ by at most one raw Q32 unit and
positions by at most 16 raw units. Those position bits come from reconstructing
the node's child collider centre from the platform root instead of retaining
Unity's separately rounded `FixTransform` and DetPhysics circle values. A
separate noncoincident moving-platform probe found the same operation-order
effect below 0.0001 world units and no state or collision-boundary change in
the measured interval. `tesla_contact_ab.c` then separates the nodes, compares
death and telemetry state for 700 ticks, and observes the same owner collision
on tick 381. The kill is classified as `Other` with owner zero: both Tesla
prefabs publish their `IPlayerIdHolder`, but neither assigns the adjacent
`Explosion.PlayerOwnerId`, so its default is the value PlayerCollision uses for
attribution. Scaling, platform destruction, time stop, smoke/mine ignition and
mixed attachment types still need complete differential scenarios. Coverage
therefore remains partial.

### Executable Meteor Smash form (2026-09-15)

Catalog id 20 now executes the source `MeteorSmash` form. Grounded entry holds
for the five authored anticipation ticks, ungrounds with the two FixTransform
teleports used by the source, and follows the 21-tick air animation with its
serialized Hermite velocity curve. Airborne entry skips that jump directly
into vibration and enforces the authored 0.05-second minimum hold. Vibration
consumes both `RandomInsideUnitCircle` draws but normalizes the sampled vector,
matching the source's constant-radius offset and its PlayerBody damping.

Release raycasts down to the nearest rounded platform or Water. Platform
landing applies the charge-scaled 250-unit impact, attaches the player at the
surface local, and defers the spawned Shockwave to the next updater pass. The
wave captures a fixed 50-platform target set, applies both initialization and
per-frame curve impulses, and ignores the struck platform. The platform's
one-second `ShakablePlatform` spin curve replaces its previous random offset
each tick without moving its spring home. The lethal MeteorSmear exports its
rescaled box immediately and removes it after exactly two registered updates.
All hot-path target storage is fixed-size and cast updates allocate nothing.

`meteor_ability_ab.c` compares 1,500 NativeAOT/Rust Q32 ticks across short
grounded release, a held maximum-charge cast, and direct airborne entry. It
checks full public player, slime, entity, telemetry, platform-shake and
shockwave outcomes. Separately rounded FixTransform positions and normalized
vectors differ by at most 50,000 raw Q32 units (less than 0.000012 world units);
platform motion is exact in the measured casts and no state, collision, entity
lifetime or animation boundary differs. Downward player/item ray callbacks,
`DropAllAttachedPlayers`, and isolated Water/no-hit exits still need dedicated
interaction goldens, so coverage remains partial.

### Executable Poke Ball capture (2026-09-15)

Catalog id 33 now uses the shared `ThrowItem2` form and the serialized
`PokeBall` item body. The pooled item binds its owner on the first post-release
updater pass, computes the source's velocity-dependent spin, resolves ordinary
circle/platform contacts, and bursts only after the strict five-second fizzle
boundary. Its proximity query scans the fixed player slots without allocation,
keeps the closest eligible opponent with the lower-id tie-break, and applies
the collider-plus-scaled-player capture radius exactly.

Capture applies PlayerCollision mask 16, invisibility and a refreshed pinned
ragdoll. The shipped prefab overrides the source field default: its serialized
capture duration is 1.92 seconds, not the class initializer's 2.4 seconds.
Release preserves the previous invisibility clocks for the callback frame,
then ejects at the ball position plus 0.4 units with horizontal velocity divided
by 4.5, vertical velocity 8.5, and orientation derived from the pre-ragdoll
ejection vector. The existing entity pool owns all state; update and contact
paths allocate nothing per tick.

`poke_ball_ability_ab.c` compares the complete 500-tick no-hit lifecycle exactly,
including form, item ownership, ballistic contacts and fizzle removal.
`poke_ball_capture_ab.c` drives an opponent across a moving-platform contact,
then observes capture on tick 116 and release on tick 232 in both NativeAOT and
Rust. Flags, invisibility clocks, ragdoll state, ownership, timers and entity
topology are exact. The scenario inherits the rollout's known player/platform
contact drift after traversal; spatial values remain within 750,000,000 raw
Q32 units (0.175 world units), while the selected victim and every capture
boundary remain unchanged. Shared `Item.Duplicate` and isolated boundary/time
stop interactions still need dedicated differential scenarios, so coverage
remains partial.

### Live AbilitySpawner pickup flight (2026-09-15)

The existing `BOPL_RL_ABILITY_SPAWNS=1` diagnostic now enables the map's
`AbilitySpawner` in Rust as well as NativeAOT. Each map retains the authored
spawn position and hierarchy. The strict 40-second clock creates at most three
`DynamicAbilityPickup` instances, consumes `RandomUnitVector` and the
catalog draw in source order, and reserves the exact three updater hierarchy
entries per prefab. Although the serialized sprite is Bow, the spawner calls
`InitPickup(null, null, ...)`, clearing that sprite before the excluding draw;
Rust preserves the resulting no-exclusion selection.

The live pickup keeps its collider disabled for the serialized 0.1 seconds.
During that phase its direct gravity changes velocity while the launch impulse
and hover force remain queued; the seventh updater pass enables the collider
and integrates the accumulated force. Subsequent updates reproduce the
platform raycast hover force, air drag, 7-second ability reroll, and late-cycle
two-draw vibration. Layer 13 does not collide with layer 11 in the shipped
Physics2D matrix, so terrain affects the item through the raycast hover force
rather than a rigid contact. State lives in a capacity-retained vector bounded
by the authored three spawns, with no per-tick allocation.

`ability_spawner_ab.c` verifies the default disabled 0/0 configuration and the
enabled NativeAOT/Rust path for 2,700 episode ticks. Both create collider id 16
at level tick 2401, keep it disabled until 2408, and match all metadata and
phase boundaries. Maximum spatial and velocity differences are 3,180 and
13,737 raw Q32 units (7.5e-7 and 3.2e-6 world units). They begin at the strict
half-unit damping boundary because Rust reconstructs the rounded platform ray
from its separately rounded exported basis; no measured behavior boundary
changes.

`ability_pickup_collect_ab.c` steers a keyboard player through the same first
pickup. NativeAOT and Rust are observation-identical through collection at
level tick 2506: collider id 16 becomes a destroyed row for one pass, the
pickup's selected Revival id 15 replaces slot three, and its timer becomes
exactly 100,000 seconds. On tick 2507 both remove the pickup row and register
the acquired Revival collider as id 17, hierarchy 100000071, inactive and
unowned. The registration range is reserved in the collision callback and the
bounded player state carries only a pending hierarchy scalar; no hot-path
allocation or global lookup was added.

`ability_pickup_death_drop_ab.c` keeps a three-player match alive after a
final elimination and compares the dropped pickup through launch, hover, the
first strict seven-second reroll, pursuit and collection. Both backends drop
Bow as collider id 20 at level tick 318 and hierarchy 100000095, then player
two collects its rerolled Rock ability at tick 950. The hierarchy includes the
two updater entries consumed by `WaterDeathEffect 1` before
`killPlayer(..., spawnEffect: false)`; this distinct boundary effect is now
extracted from the Player prefab rather than inferred from the ordinary death
effect. Pickup position and velocity differ by at most 2,103 raw Q32 units
(4.9e-7 world units), originating in `Vec2.NormalizedSafe` of the sampled
launch direction. Entity identities, hierarchy, lifecycle ticks, the reroll,
collector and acquired ability remain exact. The pickup vector is stored in a
capacity-retained collection and adds no per-tick allocation.

### Executable Black Hole form and persistent gravity body (2026-09-15)

Catalog id 18 now executes `BlackHoleClap` with the prefab's separate
PlayerPhysics body. Grounded entry reprojects to the form's 0.76-unit radius,
preserves external velocity, and runs the authored 15-tick enter and exit
animations. The callback rotates the `(6, 3)` local spawn offset by the live
surface basis and facing, applies the source's player-scale clamp, permanently
marks that equipped ability instance uncastable, and returns to the ordinary
slime through the same grounded body handoff. Air entry retains both velocity
channels at five percent and uses the form's extracted gravity constants.

The spawned layer-23 circle keeps the source's 15-unit starting mass, fixed
Hermite size animation, 0.99 velocity drag and persistent lifetime. Its updater
pulls every rounded platform because the shipped boulder exception bypasses
the nominal influence cutoff for ordinary holes. Player and ability boxes use
the separate `PlayerPhysics.AddForce` path, including the infinite-mass trigger
body sign. Wall callbacks consume only when the retained narrow-phase rounded
rectangle contains the hole centre, publish the wall as destroyed for one
pass, grow and reset the size curve, then remove it on the following physics
pass. Positive holes merge by mass with the later sorting order as the equal
mass tiebreak. Player collisions grow first and call `killPlayer` with
`spawnEffect: false`; the rollout death path now represents callback timing and
effect creation independently so stock-clone hierarchy stays exact.

`black_hole_ability_ab.c` compares 500 NativeAOT/Rust Q32 ticks. The persistent
hole appears on differential tick 199, platform id 13 is destroyed on tick 231
and removed on 232, the attracted owner dies on 241, and its stock clone
registers on 242. IDs, hierarchy, flags, entity counts, platform trajectories,
growth resets, death state and reconstruction boundaries match. The hole's
curve radius differs by at most about 100 raw Q32 units. During the active cast
form, separately synchronized PlayerPhysics and DetPhysics positions differ by
at most 70,000,000 raw units (0.0163 world units); grounded exit reprojects to
the exact same platform pose and no collision or lifecycle decision differs.
Standalone projectile/item force and consumption paths remain to be exercised
as their catalog abilities are composed with Black Hole, so coverage is marked
partial. Hole and contact state reuse capacity-retained vectors and fixed player
arrays; the tick path adds no allocation.

### Executable Throw form and Macho boulders (2026-09-15)

Catalog id 21 now executes `MachoThrow2` with its separate one-unit-radius
PlayerPhysics form. Grounded entry recomputes the surface-local pose for the
larger form, preserves the source velocity channels and facing rule, and runs
the enter, grab, pickup, idle/walk/turn, begin-throw and exit animator callback
boundaries. The grab animation applies the source's eight scale reductions to
the supporting platform. Collider geometry changes immediately while
`StickyRoundedRectangle` perimeter data refreshes on its later updater pass;
the non-resizable BoplBody mass and inertia update on that same later pass.

Quick release creates a boulder from the low hand position and launches it in
the pickup callback pass. Full pickup creates the rounded body at the animated
dummy position, holds it with zero linear and angular velocity, feeds its prior
velocity back through `PlayerPhysics.AddForce`, and launches after the authored
anticipation. Scale clamps, inverse mass, gravity, bounciness, angular velocity
and strength all come from the pinned prefabs. Boulders share the platform
arrays and entity pool, so ordinary rounded-body contacts, riders, water
removal, death callbacks and stock reconstruction see the same body without a
second collision path or per-tick allocation.

`macho_throw_ability_ab.c` compares two 500-tick NativeAOT/Rust Q32 scenarios.
Quick release shrinks the ground starting on tick 218 and creates and launches
the boulder on tick 226. The held path creates the boulder on tick 256,
launches it on tick 279, kills the owner on tick 461 and registers the stock
clone on tick 462. Positions, velocities, identities, metadata, contact
response and lifecycle boundaries match. Repeated reciprocal/multiply updates
to the shrinking collider axes differ by at most four raw Q32 units
(9.4e-10 world units); this is below one serialized source coefficient step
and does not change a contact or branch. The form collider preserves the source
split where its centre uses the stored FixTransform right vector while its axes
are rebuilt from scalar rotation.

The source can detach an existing Spike, Engine, Tesla node or rope anchor from
the consumed platform and reattach it to the held boulder. Those composed
attachment transfers still require implementation and differential scenarios,
so coverage remains partial.

### Executable Drill form and combo hitboxes (2026-09-15)

Catalog id 26 now executes the shipped `Drill` form. Entry copies both velocity
channels and the scalar rotation into its separate `PlayerBody`, moves the root
by the authored 2.01-unit local offset, and allocates its three static
Explosion-layer boxes on the first cast of each equipped prefab. The inactive
boxes retain their registered identity and hierarchy; the 19th animator update
starts the loop and activates all three, while the 13th exit-animation update
returns to the ordinary slime. `Ability.UpdateSim`, `PlayerBody`, and `Drill`
retain their distinct publication phases, so `Player.Position` exposes the
pre-movement form while DetPhysics entities expose the synchronized post-update
colliders.

The oriented drill query scans wall platforms in registered order. Air motion
uses the strong 102-unit gravity for the strict initial 0.4-second interval and
45 thereafter, with the source's normalized-vector steering. Entering a valid
surface clamps nonzero speed to 10, applies the 0.66 multiplier, clears external
velocity, then constrains and accelerates motion along the drill axis. Leaving
applies the 1.5 multiplier; a held release is suppressed underground and exits
without the positional offset on the emergence pass. An underside normal exits
before later gravity, platform motion, release, or hitbox synchronization, so
the combo boxes correctly retain their preceding pose. The root bounds handler
also retains its earlier component phase and kills before another body update.

`drill_ability_ab.c` compares 2,700 NativeAOT/Rust Q32 ticks across a downward
burrow and bounds death, an airborne release through the full exit animation, a
three-player same-team contact that verifies the first lethal callback and
stock clone, and an opposing Time Stop. The hitboxes spawn on tick 181 and
activate on 199; the bounds death and replacement clone occur on 291/292. The
contact kills on 199 and the invalid-surface exit plus replacement registration
occur on 200. The frozen case covers all 690 zero-scale ticks and verifies the
Drill-specific rule that `PlayerBody` damps external velocity while Drill only
updates combo transforms. Metadata, velocities, ownership, hierarchy, entity
counts, contacts, deaths and callback ticks match. Entry, exit and collider
positions differ by at most two raw Q32 units (4.7e-10 world units) because the
Rust path groups the scaled local-vector multiplication differently; no query,
contact or branch changes. State and hitbox storage are inline or retained and
the tick path allocates nothing. Rope detachment and direct Shockwave pushing
remain to be exercised as composed interactions, so catalog coverage remains
partial.

### Executable Portal Gun, linked planes and rigid-body transit (2026-09-15)

Catalog id 38 now executes the shipped instant `PortalGun`. A neutral aim uses
the slime's facing direction; a directed aim is normalized before the shot is
instantiated 1.7 units from the live body with speed 22. `ManualInit` registers
the 0.3-radius layer-10 collider immediately, so DetPhysics moves it on the cast
pass even though `PortalShot.UpdateSim` does not start its clock until the next
updater snapshot. A strict 1.2-second timeout plants a free-floating mouth
facing the shooter and removes the collider in that updater pass. A wall
contact first applies ordinary BoplBody resolution and positional correction,
then plants using the negated contact normal and retains the destroyed row for
the callback pass.

The first landed shot replaces any old pair with blue; the second replaces
orange and links both mouths. Each endpoint sits 0.55 units outside the hit
position and snaps its normalized normal to the dominant axis. Capture uses
the source's strict 1.6-by-0.6 plane bounds. Per-player flags and retained
per-body bitsets reproduce link-time marking, exit-mouth marking, and the rule
that a body must leave before it can traverse again. Player transit combines
self-imposed and external velocity, ungrounds without resetting rotation,
zeros external velocity, and skips the next PlayerPhysics pass. Portalable
arrows, pickups, thrown items, guided missiles, Rock bodies, Revival/Teleport
markers and movable rounded bodies use the same mirrored momentum map. Pool
indices and entity generations keep that hysteresis allocation-free during
ticks and safe when projectile rows are reused. A strict 25-second endpoint
clock closes both linked mouths.

`portal_gun_ability_ab.c` compares 1,290 NativeAOT/Rust Q32 ticks. The primary
900-tick scenario is exact across cast readiness, collider identity and
hierarchy, every projectile position, the tick-254 timeout, the tick-270
wall-callback row, the tick-275 player transit, pair replacement, and the
second timeout. A composed Grenade crosses on tick 383 in both backends and
lands at the same exit. Its mapped velocity differs by less than 0.0024
units/second because the existing ThrowItem2 port creates that grenade 0.0185
units away from NativeAOT before portal capture; the portal resets its position
and preserves the already-different incoming components, so no capture,
branch, or lifecycle decision differs. Interactions with the four catalog
abilities that remain unimplemented have not yet been exercised, so coverage
is marked partial.

### Executable Beam forms, pooled graph and damage (2026-09-15)

Catalog id 28 now connects the previously verified Beam kernels to the public
rollout. The separate grounded and airborne `PlayerBody` forms preserve their
authored radii, offsets, movement constants and animation schedules. Ground
entry chains `enterFrame1` into `enter` or `enterUp`, requests the exact
ShakablePlatform trauma each pass, and keeps its clamped relative staff vector
as the platform rotates. Air entry rotates the hurtbox with the approached aim
and reproduces the three `getNumberOfPassedSprites` orb poses at ticks 32, 48
and 64. Release during entry, manual release during the loop, jump cancellation,
landing and the strict maximum-time condition all follow their distinct exit
paths. The grounded exit reproduces `SyncPhysicsTo(AbilityExitInfo)`'s two
positions: `Player.Position` sees the ability collider's published world pose,
while the ordinary slime box is reconstructed from the newly advanced platform
local using Beam's radius and then converted to the ordinary radius.

The lazily instantiated layer-18 orb preserves its disabled charge state,
enables on the firing transition, completes `whiteExit` when that animation is
shorter, and remains as the same inactive object when an air form exits first.
Recasting that equipped prefab reuses its id, hierarchy and previous cast-tick
pose exactly. Destroying the holder instead removes the orb on its following
updater pass. Beam inputs feed the retained crossing graph without allocating;
the graph snapshot, reflected-id scratch, BeamObject pool, previous-pass contact
boxes and kind-4 observation rows all reuse capacity. Collider objects are
created before trace ids, matching `renderBeams` followed by `RlObservation`.
Collision callbacks consume the box DetPhysics simulated before the current
graph resized it, so a newly created beam is visible one pass before it can kill.
Terrain, Black Hole and Rock endpoint forces use the exported scale clamp and
time-stop protection. Water reflection and every graph role remain supplied by
the same verified graph implementation.

`beam_ability_ab.c` compares 3,810 optimized Q32 NativeAOT/Rust ticks across a
ground lifecycle with stock death, charge cancellation on a shaken moving
platform, complete airborne entry/release, retained-orb recast, automatic
maximum-duration shrink/stop, and two simultaneous upward beams that produce
the persistent crossing child. All headers, player/slime records, telemetry,
entity order, identities, ownership and lifecycle boundaries match. The entity
comparison permits one raw Q32 unit for positions, axes and radii: the observed
case is a BeamObject centre where Unity restores inverse extents and associates
the rotated half-length product one operation earlier. That is 2.33e-10 world
units, while both axes, every overlap decision and all later state match.

The retained standalone corpora continue to cover 995 primitive rays, 62 short
and 51 stress graph scenarios, 250 curves, 196 hitbox sequences, 214 endpoint
effect scenarios, and 73 combined physics scenes with no mismatches. Live
`Orb.ApplyBeam` has no producer in the five shipped maps or the 39-ability
catalog; its type/charge/firing/color behavior remains verified in the endpoint
component corpus.

### Duplicator GunTransform, platform and thrown-item copies (2026-09-15)

Catalog id 29 now executes its independently authored GunTransform lifecycle.
The form uses Duplicator's 0.70 PlayerPhysics radius, 3.78-unit firepoint,
36-tick transform clip, 17-tick fire clip and 50-unit recoil. The smaller form
radius makes grounded entry and landing observably different from Freeze and
the scale rays: Player.Position moves to the new-radius surface pose while the
disabled slime row retains its pre-cast pose, and a recoil landing converts the
form local back through the ordinary slime radius before LateUpdate.

Wall hits reproduce the shipped `useOldPlatformBehaviour` branch. The fixed
effect pool holds the 1.3-second delay without tick allocations, applies the
50-unit overlap force to every intersecting rounded platform, carries
`ActAsFloatingPlatformForAFrame` into the next home-spring pass, and creates a
geometry-identical platform with the source prefab's locally renumbered
collider hierarchy. A duplicate receives a fresh zero-velocity PhysicsBody and
homes at its spawn position, matching Unity initialization rather than copying
the source collider's live accumulator. `duplicator_ability_ab.c` compares
1,500 optimized-Q32 NativeAOT/Rust ticks across grounded and airborne miss
forms and a downward platform hit; all public state, source-platform impulses,
clone identity, hierarchy, geometry and later motion match exactly.

The live ray now follows `ShootDuplicator`'s full query order: last overlapping
rounded rectangle first, then three parallel closest rays, each scanning
rounded rectangles, circles and boxes with strict-distance tie handling. Water
recurses once with the reflected direction. Circular item targets use their
authored three-copy fan, spacing and angle without allocating. Copies inherit
the live body and item state, receive locally renumbered updater hierarchies,
register one updater pass later, and expire after the source's 1,800-pass
duplicated-object lifetime. Grenade copies additionally reproduce the explicit
thrower, owner-hit and staggered-fuse reset.

`duplicator_item_ab.c` drives a live grenade into that moving triple ray and
compares 450 optimized-Q32 NativeAOT/Rust ticks. All copy positions and
velocities match within one raw unit through overlapping Circle/Circle
manifolds, the source grenade's two startup shockwaves, a water-boundary
removal, a stock clone, and a later copied-grenade explosion. Fresh explosion
colliders append in DetPhysics box order even when their fixed native state
slot is reused. The only accepted item difference is 32,722 raw Q32 rotation
units (7.62e-6 radians): Unity copies the source `BoplBody.rotation`
accumulator, while Rust retains the previously published circular collider
orientation. Rotation has no geometric or collision meaning for these circle
items; their positions, velocities, mass and every contact decision match.
Player-layer form axes and rotation retain the existing dynamic ThrowItem2
allowance of 30,000,000 raw units (0.00699 radians); player/body state and all
translational results remain exact.

Duplicator remains partial while its player, projectile, weapon, Black Hole,
attached-terrain and orb branches are integrated. Smoke, Tack, Mjolnir, Poke
Ball and Mine use the same allocation-free circular-item clone path but still
need their own composed goldens. Magnet gun is the only catalog entry with no
executable Rust form.

### Executable Web Swing and observed rope rows (2026-09-15)

Catalog id 32 now executes the shipped instant Web Swing. The cast uses the
source's fixed up-forward direction and DetPhysics rounded-rectangle ray. A hit
allocates one body from the fixed 128-rope pool, initializes all 12 authored
points with the original division order, shortens their separation by the
prefab's exact serialized factor, and attaches the platform and ordinary
PlayerBody endpoints. A grounded cast performs the source 0.05-unit unground,
zeros rotation, and adds the 3.5 upward impulse before PlayerBody integration.
The rope and attachment enter the Updater on the following pass, while the
already-authored PlayerBody pins its endpoint on the cast pass. This ordering is
needed for exact first-frame segment state and subsequent moving-platform force.

The existing verified `physics::rope` solver now runs in rollout DetPhysics with
preallocated 40-point and 32-contact arrays per body. Platform attachment,
player endpoint mass interpolation, air correction, grounded pull, stretch
breaks, jump release and held-button cooldown routing use the source phase
order. Releasing after the six-tick guard copies endpoint Verlet velocity with
the exact 1.12 serialized multiplier, detaches the player, starts the 1.25-second
cooldown, and removes the dangling strand after the 0.9-second cleanup clock.
Observations insert `segmentCount - 1` kind-3 rows between rounded rectangles and
beams with stable ids, owner, hierarchy, radius, direction, and detached flag.
Rope bodies, broad-phase rectangles, rows, and fork state reuse capacity; there
is no per-tick or per-cast allocation.

`web_swing_ability_ab.c` compares 1,310 optimized-Q32 NativeAOT/Rust stages. Four
maps exercise the miss and exact 0.45-second retry cooldown. Map 3 exercises the
hit at tick 181, all 11 rope rows, sustained dynamic-platform swing, tick-360
release velocity, detached observation lifetime, landing, and tick-413 cleanup.
Every compared header, player record, slime record, telemetry value, rope row,
and lifecycle boundary matches exactly. The source-prefab extraction now records
all rope, attachment, DetPhysics, and Web Swing constants in `rollout.json`; the
hot loop uses compile-time values and asserts them against that generated data
when the fixed pool is created. Duplicator/Magnet interactions with ropes remain
outside this golden.

### Executable Grappling Hook (2026-09-15)

Catalog id 27 now executes `HookshotInstant`, `Rope`, `RopeHook`,
`RopeAttachment`, and the PlayerBody rope path. A cast publishes the layer-24
hook before its 11 initial segment rows, advances it at the authored 100-unit
speed, applies the eight-segment velocity falloff, grows the rope one segment
per updater pass to the 40-point limit, and times out at the strict 0.8-second
boundary. Wall callbacks require the hook centre to overlap the rounded
rectangle before they attach. Player callbacks exclude the rope owner and
original endpoint. Holding the slot keeps cooldown zero while the owned rope is
valid, shortens separation by `0.4 / segmentCount`, removes one segment per pass
below 0.3, stops at eight points, then restores the three-second cooldown.

The SlimeController proximity branch is also preserved. A three-player team
golden places a teammate inside `CircleCastClosestPlayers`; the rope starts at
that teammate, attaches its bottom endpoint, leaves the caster unattached and
immediately retains the full cooldown. The resulting strand persists after the
hook lands because the caster has no reel ownership. Rope, hook and contact
state stay in the existing fixed pools and fixed-capacity scratch arrays.

`grappling_hook_ability_ab.c` compares 1,870 optimized-Q32 NativeAOT/Rust stages:
horizontal casts on all five maps, one upward timeout, and the close-player team
cast. All hook/segment ids, hierarchy, counts, cooldown and reel/delete
boundaries match. Free-flight segment positions are exact. On map 1 the attached
rope and player remain exact through impact; the first player-position split is
334 raw Q32 units (7.78e-8 world units) four ticks later. Constraint relaxation
then amplifies that rounding difference on moving platforms, while impact,
segment deletion and cooldown ticks remain identical on every map. Unity also
varies the destroyed hook collider's otherwise-irrelevant ACTIVE bit by map
(retained on two maps, cleared on three); both backends mark it DESTROYED and
remove it on the next pass, so the golden masks only that bit on that one frame.
