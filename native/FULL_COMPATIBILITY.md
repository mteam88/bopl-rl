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
