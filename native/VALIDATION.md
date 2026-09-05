# Validation record

Measured on 2026-09-04 on the PowerSpec Linux host against the reconstructed Unity reference
`67484d097087562123f860e1ed209bcb56e99d2e` with the instrumentation in this change.
These results establish the scope below, not complete-game equivalence.

| Check | Result | Scope |
|---|---:|---|
| Original C# versus Rust math | 303,016 / 303,016 | 24 operations, boundary and seeded random inputs; expected errors compared |
| Actual Unity versus Rust geometry | 24,000 / 24,000 | 12 operations × 2,000 cases: raycasts, overlaps, closest points, perimeter coordinates/normals/movement |
| Independent native episodes through C ABI | 5,870 / 5,870 ticks | Nine two-player episodes; exact represented state, no state injection after reset |
| Platform surface coverage | 8 / 8 regions | Four sides and four corners encountered in episode goldens |
| Synthetic Unity self-comparison | 1,200 / 1,200 ticks | Raw managed state, one versus sixteen ticks per update |
| Selected real/mutated Unity self-comparison | 3 / 3 captures pairs, 1,200 ticks each | Gameplay view with explicitly audited presentation exclusions |
| C ABI and Python contracts | Passed | Real compiled C client, invalid input, reset, termination, independent handles |
| Rust checks | Passed | Formatting, Clippy with warnings denied, interleaved/threaded worlds |
| Gymnasium environment checker | Passed | Reset/step/seed contract; not training performance or policy quality |

The episode corpus contains:

| Fixture | Verified ticks | First death reached |
|---|---:|---|
| movement | 473 | Yes |
| random-1 | 88 | Yes |
| random-2 | 151 | Yes |
| random-3 | 182 | Yes |
| random-4 | 262 | Yes |
| held-jump | 1,200 | No |
| perimeter reversals | 1,200 | No |
| simultaneous-dash | 1,114 | Yes |
| surface-circuit | 1,200 | No |

Counts stop at the native episode boundary: the first death or the end of the
input sequence. They do not count unverified post-death gameplay as passed.
Goldens preserve source trace/replay/assembly hashes. The circuit's actions were
chosen with a test driver, then replayed independently in Unity; Unity supplied
the expected states. It is not a feedback-controlled comparison.

The C ABI benchmark is approximately **190,000 two-player world steps/second**
on one CPU thread through ctypes, with periodic resets. The exact final result,
workload, and elapsed time are in `artifacts/benchmark-report.json`. This excludes
policy inference, Python observation encoding, and complete-game physics. It is
not a PufferLib training benchmark.

## Mac archive and stress cases

329 replay files were copied from
`/Users/mte/Library/Application Support/Johan Gronvall/BoplBattle/replays`.
All 329 decode with the original codec. Their headers span 121 maps and 32
resolved ability IDs, plus unresolved ID 255.

- 19 files declare five players but have only four encoded input streams.
- 137 files contain unresolved ability ID 255. The current Unity loader rejects
  such a loadout. No substitute ability was invented to make it play.
- 183 files pass both metadata gates. This is a preflight count, not 183 verified
  matches; the exclusion counts overlap.
- Selected replays `297.rep` and `259.rep`, plus a seed-67484 mutation of `297.rep`,
  each pass a 1,200-tick Unity gameplay comparison across batch sizes 1 and 16.
- The raw comparisons correctly report differences in rumble, visual shaking,
  particle lifetime, or Poké Ball sprite wiggle. Four source-audited fields are
  excluded by the optional gameplay view; all other recorded state remains
  compared. See `data/presentation-fields.json` for the individual reasons.

The originals, inventory, mutation, full traces, binary/replay hashes, and both
raw and gameplay comparison reports remain in ignored `artifacts/`. Personal
replays are not committed. Recordings do not include their original game build
ID, so replaying them in this reference cannot prove historical Mac behavior.

## Evidence files

- `data/reference.json`: source/table pin.
- `data/assets-reference.json`: Unity asset-export hashes.
- `data/corpus.json`: reset and checked-in golden hashes.
- `fixtures/reference/*.jsonl.gz`: compact source-backed episode regressions.
- `artifacts/math-report.json`, `artifacts/geometry/report.json`.
- `artifacts/golden-report.json`, `artifacts/standalone-validation.log`.
- `artifacts/mac-replay-inventory.jsonl`, `artifacts/mac-replay-summary.json`.
- `artifacts/reference-suite/`: raw synthetic self-comparison.
- `artifacts/reference-stress/final-report.json`: real/mutated comparisons and
  preflight rejections, with explicit scope and exclusions.

## Limits that matter next

The native API supports one fixed map, one spawn seed, equal-size slimes, and
dash. It stops at the first death. The separately verified rigid/composite/rope solver and beam graph are not yet
integrated into this API. Moving/scaled platforms, damaging abilities, clones,
time manipulation, match scoring, stocks and respawn still require native gameplay
integration. Float conversions outside normal finite
game values are not part of the verified arithmetic contract; exported assets
use raw integers. The Unity recorder has the serialization limits documented in
README.md. No trained policy or PufferLib backend has been validated.

Expand these gates alongside each new mechanic. Preserve failures and report
which fields, episodes, and interactions were actually compared.

## Full-compatibility expansion checkpoint

The expanded goal remains incomplete; see [FULL_COMPATIBILITY.md](FULL_COMPATIBILITY.md).
One-through-six-player movement/dash is now exposed through the C/ctypes APIs.
Six additional Unity episodes add 951 exact independent C-ABI ticks (6,821 total
across 15 episodes). Unity's own playback matches for all six player counts at
batches 1 and 16, 900 ticks per episode.

The new rigid-body primitives match 27,000 actual Unity queries across six
collider pairs, integration, impulse/friction resolution and positional correction.
The suite includes source arithmetic failures; two initial one-raw-unit negation
ordering bugs were fixed. A 1,754-case stratified regression is checked in.

RNG output and state match 65,000 mixed operations across 65 seeds against the
actual managed game assembly, including calls to Updater.RandomFix; 5,000 draws
are kept as a standalone regression. The BOPLREP2 codec passed 21,504 complete
packet roundtrips across 1..6 players and all 1,024 input-bit combinations, with
corruption, partial streams, reserved bits and invalid IDs rejected.

These primitive checks do not certify an integrated Rust physics world or the
unported abilities. The new reference setup/capture code has been compiled and
run in Unity; actual local recording parity, later-round carried state, full
physics/global state capture and the complete Rust gameplay port remain required.

## Ordered physics scenes

The Rust rigid/composite scene solver matches the actual Unity
`DetPhysics.Simulate` function on 1,000 generated scenes and 12,000 consecutive
steps, with 164,124 ordered callback records and 137,486 broad-phase pairs.
All scenes passed; none ended in an arithmetic exception. 560 scenes contain
composites, including attachment, detachment, child/platform destruction and
reuse of destroyed composite slots. Layer masks, triggers, inactive objects,
transform offsets, angular motion, drag and zero timestep are included.

`artifacts/physics-scene/report.json` contains reference assembly and corpus
hashes. `fixtures/physics-scene.jsonl.gz` retains 60 scenes and 720 steps;
`just verify` runs them through the standalone Rust probe. These compare every
represented scene field without presentation exclusions. The Unity adapter
records callbacks without executing abilities; ropes and beams are absent.
This verifies the shared physics stage, not full combat or native match parity.

The same solver also matches 2,700 individually initialized physics stages from
three actual six-player ability-loadout replays, including 16,930 callbacks.
These exposed a missing cached transform direction: converting its angle back
to a vector changed offset collider contacts. Capturing and using the actual
`FixTransform.right` fixed the divergence. Original failures remain in
`artifacts/physics-gameplay/`; corrected traces/reports are in
`artifacts/physics-gameplay-v2/`. There were no active ropes or beams in these
stages. This is stage-level evidence with Unity state supplied before each step,
not an independent native combat rollout.

Rope primitives match 2,000 reference cases, with 15,808 successful actions and
24 matching division-by-zero outcomes. The probe invokes the actual
`SimulateRopes_parallel` and `RopeBody` methods. A 197-case fixture covers
integration, collisions with platforms, endpoint mass/attachment state,
segment spawning and reeling; full grappling gameplay is not yet implemented.

Combined rigid/composite/rope scenes pass 1,000/1,000 cases and 12,000 steps,
including 45,168 enabled-rope updates and nine scenes that filled the 32-platform
rope contact buffer. A 69-scene, 828-step fixture is checked in. The six-player
hook replay passes 600/600 physics stages, with 268 stages containing active
ropes (peak six), and 3,834 exact callbacks. Full evidence is in
`artifacts/rope-scenes/` and `artifacts/rope-gameplay/`. The same pre-stage
initialization limitation applies; grappling's ability state machine is unported.


## Beam graph checkpoint

The actual Unity intersection and color methods match 20,100 primitive queries.
The actual `UpdateBeams` graph, captured before `renderBeams`, matches 1,070
complete scenarios and 20,508 frames across two corpora, plus two matching source
math-domain errors. Short scenarios cover one through six initial beams; longer
120-frame scenarios use up to 18 beams belonging to six players. The graph state
comparison includes every represented field, including raw float color bits.

The three standalone fixtures retain 995 primitive cases, 62 short scenarios and
51 longer scenarios. All six graph roles observed in Unity and both source errors
are retained. Regenerate with `just --justfile native/justfile verify-beams`.
The final reference assembly is
`18ee71f003e42ac9ce3c22b05c94e1d440462549168843e3d57988dec9a76d39`;
full input/output hashes are in each beam artifact directory's report.

This oracle intentionally stops before beam hitbox creation and target effects.
The separate component evidence below covers hitbox geometry and target effects.
Combining the stages and implementing the beam ability lifecycle remain required;
graph parity alone is not full combat parity. [Full scope](FULL_COMPATIBILITY.md) remains open.


## Beam hitbox and target component checkpoint

Against actual Unity methods, fixed curves match 92,935 returned values across
1,943 successful scenarios and 57 source-error scenarios. Hitbox initialization
and resizing match 1,000 sequences / 14,000 states. Beam target effects match
980 sequences / 7,840 frames, plus 20 matching zero-mass arithmetic failures.
Force/velocity/charge changes occurred for every intended target kind; counts
are recorded in `artifacts/beam-effects/report.json`.

These probes execute the actual beam prefab and `renderBeams`, with real target
components. They check the represented gameplay state exactly. They do not run
collision-driven deaths, composite targets, or the full ability lifecycle, and
are not independent native combat episodes. Color and renderer animation output
from `renderBeams` are not compared by the target-effect probe; beam color mixing
has its own bit-exact fixture.

The reference assembly for all three corpora is
`c079c5b54e61534ca1228339651e0179233ff706fb5338c0ebf780ff1c219f60`.
Standalone fixtures retain 250 curve scenarios (9,119 values and 57 errors),
196 hitbox sequences (2,744 states), and 214 target scenarios (1,552 frames and
20 errors). They are included in `just verify`.


## Combined beam / rigid / composite / rope scenes

The combined stage matches 1,112 complete scenarios / 16,176 steps and 631,152
ordered collision callbacks across two corpora, plus eight matching source
arithmetic-error scenarios. The ordinary corpus uses one through six initial
beams; the stress corpus uses up to 18 beams across six owners. Both retain state
between steps, destroy/recreate pool objects and exercise the source's deactivation
loop behavior. Scene, rope, graph, pool, force and orb state are compared exactly.

The probe invokes actual `DetPhysics.Simulate` with real components and suppresses
only the final dispatch of gameplay collision callbacks. Those queued callbacks
are compared; their damage/death effects are not ported by this checkpoint.
The native entry point is `beam::system::System::simulate`, not a replacement
calculation implemented inside the test harness.

Reference assembly:
`8c397c6a720302d54261628cf28c07a04c3e7488c7704dd54858f3fadd3d318c`.
`artifacts/beam-scene/report.json` and `artifacts/beam-scene-stress/report.json`
retain input/output hashes and counts. Standalone fixtures retain 73 scenarios,
1,236 successful steps and all eight source errors. An independent native
player-input episode with beam abilities remains required; replay-stage import
is covered next.


## Beam replay stage import and exception evidence

Three completed six-player traces match 1,780 stages and 15,514 callbacks. Of
these, the four-Beam/two-Dash replay covers 700 stages, 141 with actual emission,
and 6,898 callbacks. Beam owners 1–4 emit; at most four beams are queued. The two
540-stage all-Beam prefixes contain no emitted beams and are counted separately
in their reports. None of these replay cases exercises beam crossing/reflection.

Completed-capture assembly:
`5bf8b058428d859c0c3cd8b73ed8e9a8496fb9298bdcfd608d6f94b161cd72e1`.
The 700-stage replay SHA is
`d2c7f8d27932ff6d6655b63c8c879c4e7f30370f9d8b76e88b5849c9b769ddd0`;
its trace SHA is
`e9316afa59f7eff4bf42e849ede2c844edb2303b74076d74cda16ac1f315640b`.

The all-six-Beam stationary replay throws in `Beam.ExitAbility` at input index
561 because the missing player-5/6 charge sound produces index -1. Capture now
records the first exception separately and rejects the incomplete full trace.
The completed physics stage preceding that exception exactly matches Rust,
including four queued beams owned by 3–6 and ten callbacks. Subsequent damage,
death, and the exception itself are not native gameplay implementation evidence.

Exception-capture assembly:
`0e2136f5c6a64388415f87060d6451f401ba00bf08e112cdf326f2ac05184f25`.
Failure snapshot SHA:
`55ca48de830519c6f78e9349585fca7885ba4ff94edbc5864b406f51a33718a9`.
Both before/after beam state and scene state are compared, including retained
history, target state, pool contents and canonical IDs. Fixtures retain 284 completed
stages and one exception-boundary stage. All evidence and source-failure logs
are retained under `artifacts/beam-gameplay/`.


## Player.Kill accounting

Direct calls to the actual Unity method match 1,200 scenarios / 38,400 actions.
This includes 2,100 final-body deaths and notifications, 3,906 winner no-ops,
11,223 self-kills and 16,841 calls without a registered killer. Inputs span all
six player IDs, clone counts, all six cause enum values, and integer limits.
The fixture retains 196 scenarios / 6,272 actions.

Direct-oracle assembly:
`ea65cc7c504a668f19fa1f0ec43cc39bdd9a343a02db60d8e0b9b2e77cce6f69`.
Input SHA:
`5c5fb6bf286b442596c4d53fe9f11ddfa9a819acbfe4896ce2be6f5b29ae1bec`.
Output SHA:
`4d6d3bd0a5795858e2f34fbd5c74d67ce17306399d9e79e988352edf6c98df00`.

Actual six-player gameplay supplies six accounting calls across two 700-tick
traces: three stock-replacement deaths and three final eliminations with stocks
disabled. All six calls match, including the notification about to be dispatched.
The capture boundary precedes `playerDied` listeners, so their effects are not
part of this accounting check. Both physics traces also match: 700 stages / 6,898
callbacks with stocks, and 700 stages / 6,932 callbacks without stocks. The former
repeats the previous input scenario with the new accounting capture enabled.

Gameplay assembly:
`a6374cbb2a0ad82e33d9ac866752258855a320dc54090dee082f81d1ee8f65c6`.
The no-stocks replay SHA is
`08b44c2fbe3cf20febf4ea8cfdeaddbb1c29d9681fee7950e04ee053159ba49a`;
its trace SHA is
`02b7fcdf643ebca209d8f8353e8ee8c34ebd2b242e7e33238e5f48f4ae6d7d9e`.
Reports under `artifacts/player-life-gameplay/` retain the other trace's hashes
and the exact counts. All six gameplay accounting calls and 182 no-stocks physics
stages are checked-in regressions. Body/ability lifecycle and collision damage
remain separate, incomplete native work.


## PlayerCollision selection

All 600 component scenarios match: 12,551 completed actions and 177 first-error
snapshots, including their partial mutations and completed prefixes. There are
303 attachment requests, 2,926 kill requests and 171 lightning requests. Causes
are Other 1,653, Squished 428, Accident 531 and Beamed 314. Errors are 111 null
references and 66 integer absolute-value overflows. Tests use the actual Unity
collision method with recording sinks for downstream kill/attachment handlers.

Assembly:
`60273bb8f3c14dd2b3fc8ffd95bd3f7a59368b789f36e40ac824915e467eebb2`.
Input SHA:
`9275c8a33ac7fce1460c5bd30f34df3271be634fff5d4d5ba00fe55e5815e77e`.
Output SHA:
`21e9be4c0f847cf1b9d3a50530dfee826eb37e5c73e085a0516be125247919f1`.

The 700-tick six-player gameplay capture matches all 4,432 decisions and three
Beamed kill requests. Four further decisions, including one Beamed request, match
at the known source exception on input 561. Real downstream handlers execute in
these captures; their changes are excluded from the decision projection. The
700 physics stages / 6,898 callbacks and the failure's final physics stage / ten
callbacks also match. These repeat existing input scenarios.

Gameplay assembly:
`a3b02610093e2e665c29aedc1ec43a9c36153bd46418e8e6f6f7332073a431e3`.
Completed trace SHA:
`93a17edda9f12866ded7bde4405f0cb71f82589c2481b5c061cdf4c004df17f4`.
Failure snapshot SHA:
`42ffafecdfaaa1afcb92953b26c451903de813a4a7ae62cc86ea9906f7bfe427`.
The later assembly adds the position argument to attachment records; the gameplay
corpus has no attachment requests and retains its original provenance. Full
reports are under `artifacts/player-collision/` and
`artifacts/player-collision-gameplay/`. Both fixture sets run in `just verify`.
Persistent component references and downstream lifecycle execution remain outside
this component's coverage and are not exposed as full combat through the C API.


## PlayerPhysics attachment and detachment

The actual Unity methods match 1,000 generated scenarios: 874 complete sequences,
10,741 completed actions (3,497 attachments / 7,244 detachments and listener
notifications), and 126 first-error snapshots. All errors are division by zero;
completed prefixes and partial state are compared. The source can fail at exact
corners and degenerate surfaces. Rendering and the optional jump sound are
excluded, with a recording listener attached for detachment notifications.

Assembly:
`669b9c2b6b2cccfeecaf58900b644da0a205f754fb2299b255c8df20824e2bcd`.
Input SHA:
`b49a04a95225738b0c77f2e7b418b500322bc4dc8260ef0ff8adae9e4ff16a70`.
Output SHA:
`252bf39edb0e66355cf16b8f819436b8b0eb1af07ac645a4977aaa7513865d0e`.

The movement world uses the shared code for attachment and jump detachment;
existing independent C-ABI episodes remain exact. The fixture is included in
`just verify`. Full component-reference lifecycle, arbitrary event listeners,
rope-driven detachment and death/revival execution remain outside this check.


## PlayerBody motion and rope lifecycle

All 600 generated scenarios match, with 391 complete sequences, 8,666 completed
actions and 209 first-error snapshots (187 null references / 22 division errors).
The actions include 1,731 body updates, 2,309 late updates, 1,709 original rope
solver calls, 1,318 attachment transitions and 399 ability-entry clears; 93
release-sound requests match. There are 1,890 body calls with enabled ropes and
1,285 with arrived hooks. The oracle executes original component and solver
methods, with serial rope dispatch for deterministic error snapshots.

Assembly:
`881b7707d4df94faea974a1884b81c5f43e413695535aa250a1ef5287e1f49c8`.
Input SHA:
`81715ecb202c48c605c814b8376248d0667cce96549de7715bceda496ff87397`.
Output SHA:
`1112cfef2e9580eaaa371b78a91315a343a5afbfa674c457281eb0cfc949d1d1`.

Actual six-player gameplay matches 4,405 body calls: 2,198 updates and 2,207 late
updates. Of these, 773 begin with a rope reference, 769 with an enabled rope and
439 with an arrived hook. No release sounds occur in this trace. All 600 physics
stages / 3,834 callbacks also match, including 268 stages with active ropes and
up to six active ropes. Unity supplies each method's pre-state; this is not an
independent native grappling episode. The input repeats the existing six-hook
scenario with body capture enabled.

Replay SHA:
`545be94ee333638e63be2756d9fa21b817dc42d40444c65709a439ff9075ac67`.
Trace SHA:
`95a80354c79754306e1fe20727aef02f6ffc5c4bbbf2f15f4f5a5c71d3316c89`.
The gameplay assembly is the same as the direct oracle. Reports and raw evidence
are in `artifacts/player-body/` and `artifacts/player-body-gameplay/`. Both body
fixtures run under `just verify`. Global scheduler/component identity and full
ability/death/revival orchestration remain outside this check.


## PlayerPhysics handoff, forces and grounded rope updates

All 1,000 actual-component scenarios match: 701 complete sequences and 15,026
completed actions. There are 2,432 ability entry/exit handoffs, 3,414 late updates,
2,286 jumps and 2,718 detachment notifications. Error snapshots compare the first
failing action, its completed prefix and partial state: 191 null references, 102
division-by-zero errors and six invalid indices. Detachment listeners can move
the body, and the resulting rope endpoint is checked after the callback.

The corpus exercises scale ratios, different outgoing/receiving radii, missing
and destroyed platforms, missing rounded-rectangle components, space gravity,
zero movement denominators and force thresholds. The fixture retains 405 cases,
including all 299 errors. Gravity and air movement are shared with the public C
movement environment; its existing independent trajectories remain exact.

Assembly:
`89a8b02c61b3756feab35f99a5d7e60fa1fe3d3c5ac5ae4b2a4e1de3f0c18dc7`.
Input SHA:
`32aaa8141e671407f385b31c2451ac69dd5aa3d4a93a742a64c804808de13061`.
Output SHA:
`1286d658b35b2c8c988a8e31897d862449d107772c3870eb39ba05891b6f6705`.

Raw evidence is in `artifacts/player-physics/`; the fixture runs under
`just verify`. This tests the fixed-point component state and listener ordering,
not rendering, jump audio, whole-world scheduling, blocked surface traversal or
an independent native ability episode.


## Player raycast ordering and callbacks

All 1,000 scenarios match the actual Unity raycast and player methods. There are
958 complete sequences, 22,711 completed actions, 1,203 attachment callbacks and
42 matching null-reference failures with partial state and completed prefixes.
The comparison includes full scene state, all three collider registration orders,
near/far hit records, player state and callback effects. Direct queries hit all
32 layers. Scenarios use one through six registered players, varied time-stop
protection and missing holder/player references, with zero/negative time,
registration ties, side-ray hits, cleanup and missing platform components.

The fixture retains 224 cases (917,904 compressed bytes). It is a supplied-state
component check, not 1,000 independently simulated full-game episodes. Shared
three-ray selection and time scaling remain covered by the existing independent
C movement and player-body fixtures.

Assembly:
`1ab7a0ba5cd41e50982a389389bff9bf76216f200e992c0952693e72c51cdd27`.
Input SHA:
`fa4fac7517944c68cb33b073d42b744705b0eb612b5b9f9151011dd880320856`.
Output SHA:
`e98f938e40a51c3d601c5173ff533c5a973267ee0745b989d8466430ce9b16ba`.

Raw evidence is in `artifacts/player-rays/`; regenerate with
`just verify-player-rays`. The fixture runs under `just verify`.

After extracting the shared scene snapshot helper, a fresh Unity rigid/composite
regression also matches 100 scenes / 1,200 steps / 14,898 callbacks, including 55
composite scenes. Evidence is in `artifacts/player-rays-scene-regression/`.
The full standalone verification suite passes with the new raycast fixture.


## Blocked platform movement and direct hopping

All 1,000 Unity scenarios match: 944 complete sequences,
34,016 completed actions, 10,416 surface moves,
14,160 hop calls, 4,720 explicit limit queries
and 4,720 multiple-body moves. There are 7,245
hop results with a platform and 7,556 nonzero remainders.
The 24 division-by-zero and 32 null-reference failures include partial state and
completed prefixes. Scene state, collider order, enablement and return values
are compared without tolerances. The 205-case fixture is 859,986 bytes.

This includes rotated/translated surface caches, different platform/player radii,
registration ties, independent collider enablement and object activity, destroyed
colliders/transforms, missing shape components and empty/nonempty hop exclusions.
It does not exercise null collider transforms or recursive PlayerPhysics hopping.

Assembly:
`c6acf8a0be043269b9fea641362e8b97a6f11445466cbe2884c94d583e35ff25`.

Input SHA:
`0036e15e9cbdadaed17f39b0a6cc41c5185b47386d634513bf87407fab98541e`.

Output SHA:
`d5ae901667822429a7bec1af4589f5c4aa579c396763ce2383ad83b02e5a9dec`.

Disabled-collider integration also matches 200 generated rigid/composite/rope
scenes, 2,400 steps, 11,052 callbacks and 9,024 enabled-rope updates, with 110
composite scenes. Its fixture retains 56 scenes / 672 steps. Both fixtures run
under `just verify`; raw evidence and provenance are under
`artifacts/surface-motion/` and `artifacts/disabled-colliders/`.

The independent C movement environment also shares the new limit-selection
helper and perimeter calculation. Its full trajectory fixtures remain the
regression gate for this integration.


The updated gameplay capture preserves object activity and collider enablement
separately. A repeated six-player grappling input matches all 600 physics stages
and 3,834 callbacks, with 268 active-rope stages and up to six ropes. Of those
stages, 287 contain disabled colliders on active objects: 1,255 such collider
states are compared. This is supplied-pre-state physics verification and reuses
the existing six-hook input. It does not prove an independent native ability
rollout. The 316-stage gameplay fixture keeps every stage with a disabled collider.

Gameplay assembly:
`0ca9c7fc57d1a09818b334f3af717b64f6d6080423438cdc85a5cd3d2fdeae48`.

Replay SHA:
`545be94ee333638e63be2756d9fa21b817dc42d40444c65709a439ff9075ac67`.

Trace SHA:
`ff297da7d0e5330f8c2dcd009818681a64b2aae607efe3f7f18b4dac61aa94a1`.

Raw evidence is in `artifacts/surface-motion-gameplay/`.


## Player controller composition

All 1,000 actual Unity component scenarios match: 817 complete
sequences and 32,976 completed calls. Coverage includes
4,223 UpdateSim calls, 4,149 Move calls,
4,124 MachoMove calls, 4,105 external-force
surface moves, 4,299 drop checks and 942 recursive
hop calls. There are 169 platform transitions, with at
most four in one call. Scalar increment/acceleration and alignment checks are
also included. All callback payloads, controller/scene state, collider order,
hop-buffer contents and partial failures compare exactly.

Completed calls emit 601 attachment, 1233
detachment, 414 drop-exit and 196 sound
notifications. The 132 null references and 51 division errors compare their
completed prefixes and partial failing state. Listeners modify position and
velocity; other AbilityExitInfo fields are checked against their source defaults.

The fixture retains 315 scenarios (1,327,990 compressed bytes).
The five-slot buffer used by Roll is exercised in 334 cases with at most five
platforms; 666 cases use 64 slots. Cyclic arrangements after exclusion-buffer
exhaustion remain unverified. This does not certify independent ability episodes,
full world scheduling or arbitrary ability callback side effects.

Assembly:
`270b7590040bcdd9db5acf071ed399e94eacb8e4bc7efdc8c340926583f7f5f1`.

Input SHA:
`7d20de68759fbe4cdf096b3b94d6e8c70c2360e2ce9038ec6b7d93ee5bebad0e`.

Output SHA:
`49752b24a621922a932defdc5a0475b4ad9d23676c0e2b1c780ee701786c8cae`.

Raw evidence is in `artifacts/player-controller/`; the fixture runs under
`just verify`. The shared platform setup/snapshot refactor also passes a fresh
100-case / 3,387-call Unity traversal regression under
`artifacts/controller-surface-regression/`. Grounded acceleration is shared with
the C movement environment, whose independent trajectory fixtures remain exact.


## Sprite animator simulation

The actual Unity component matches all 1,000 randomized cases: 368 complete
sequences, 115,372 completed calls and 2,043 callbacks. The remaining cases match
632 source errors (66 null references, 475 invalid indices, 71 integer overflows
and 20 divisions by zero), including the failing action's partial state.
Coverage includes all six enum modes and an unknown no-op mode, every animation
entry method, positive/negative/zero frame durations, counter wrapping, nullable
arrays/renderers, mutable velocity sources, time-stop protection for one through
six players, persistent callbacks and callbacks that start another animation.
The fixture retains 351 cases (695,159 compressed bytes).

Assembly: `f0e68ff8d5a35338c53dd3eaaa934f1d4d29179b80ad47a210640acf07d93a01`.

Random input SHA: `6d0a04a23c1574bf7900241b1103a88ae35154d22c0cc0d03807a8ea618921e1`.

Random output SHA: `85b8d80f9ca8bfebe550556b5f48a526fc02bd6430e24da8f53951e0b69825f5`.

The asset corpus matches 1,237 cases: 1,233 complete sequences, 60,625 calls and
946 callbacks. All 35 exported AnimationData assets are included: 194 named
animations, 1,042 frame records, one velocity map and 733 distinct sprite IDs.
Every frame has a three-update duration-boundary case, including long holds.
The four failures are empty animations: MCAnimData fall/jump and Revive
deprecated/deprecated2. These are not failed playable episodes. The player spawn
callback is reached at updates 106 and 21 for normal/fast frames respectively.
The asset fixture retains every case (502,489 compressed bytes).

Asset input SHA: `cdd0396e3b9e228675295f3fe1b73b2e08be037cca6e0f7c948044b708cd166d`.

Asset output SHA: `f153fa7daea9902f191788e6af677dea4ce09409f78c14915c5b0a0c822528cb`.

Export SHA: `dab5953ba916ef0e0ba78f07d5134f40ebb374d047521160050d9688a0a00f9f`.

The exported animation names, sprites, durations and velocity-map values were
cross-checked against all 35 current YAML assets. Raw evidence is under
`artifacts/animator/` and `artifacts/animator-assets/`. Both fixtures run under
`just verify`. Opaque sprite identities are compared exactly; this component
check does not certify texture rendering, arbitrary callback side effects,
complete spawning, ability lifecycles or independent world scheduling.


## Coupled spawning and revival

All 600 actual SlimeController scenarios match: 471 complete sequences,
84,312 completed calls, 1,086 transitions from unspawned to spawned and 1,377
animation-triggered Spawn callbacks. The body can already be marked spawned
when another completion callback runs, so these counts measure different things.
There are 3,374 completed detachment/sound pairs. The actual
SlimeController.OnNoLongerGrounded listener changes animation state before an
additional listener changes position and velocity; those effects are compared.

There are 100 setups at each player count from one through six, 130 cases with
repeated ability references, 125 with repeated indicator references and 29 with
more than three abilities. Only one slime advances per scenario. Other players'
revival/ability flags are compared throughout. Real spawn/fall animation data,
mutable shared speed containers, arbitrary initial masks, tutorial mode,
time-stop protection, abilityless initialization and interrupted revival are
included. Another 150 cases start with an in-flight spawn callback, rebound to
the actual slime instance. The pinned Player prefab uses the same renderer for
SlimeController and SpriteAnimator; this identity is preserved in the test.
Registration/destruction and complete multiplayer episodes are not.

The 129 errors match their completed prefixes and partial state: 82 null
references, 24 invalid indices, 12 missing animations, six missing velocity maps
and five overflows. All 13 failing fast-respawn calls preserve the source's
uncleared fast-animation flag. The fixture retains 250 cases
(788,282 compressed bytes).

Assembly: `a161745418fb9a70e55069855884c374b7743bd6788c2eee1d712d12714c0b81`.

Input SHA: `3be8f47d478546324552ebc3e7df1c5adc8850770825213ee1329fde01f592de`.

Output SHA: `0110dc2cf7fe08c9ff502013dd38c3bd6e324aa7b4165c7f6abcc501539e839d`.

Animation fixture SHA: `212a2f2510ee283430e7142d3ad7442b4089eb419ef18b313066d96809e7f38a`.

Raw evidence is in `artifacts/spawn/`. `just verify` runs the checked fixture
without Unity. Stock-choice logic, clone creation, death handlers, revive travel
effects and the full world scheduler remain separate completion requirements.

The reusable animator harness also passes a fresh 100-case / 11,219-call Unity
regression (215 callbacks), retained under `artifacts/spawn-animator-regression/`.


## Registration, scheduling and deferred destruction

The actual Updater matches all 1,002 scenarios: 554 complete sequences,
29,538 completed calls and 8,394 ticks. Component events comprise 33,455 Init,
103,261 UpdateSim, 73,025 LateUpdateSim and 14,833 OnDestroyUpdatable calls.
Both player input and player update phases execute 30,422 times. The cases use
one through six players with shared input-profile references, jump-edge state,
invisibility expiry and time-stop protection. Position-setter fields remain
null in these tests; their arbitration/rollover is not covered.

There are 1,000 randomized scenarios plus two targeted activation cases. The
latter cover active/inactive transitions, null MonoBehaviour deletion and the
Linux player's null-reference error when activating an already deleted object.

Coverage includes nested inactive hierarchies, separate physical presence and
IsDestroyed markers, deleted managed references, duplicate queue entries,
nulls, prefab priority wrapping, registrations during Init and later callbacks,
loaded-level early return, mutations during backwards removal, and queued
objects retained across reset methods. Pending-sort backing capacity and all
partial failures are compared. The 448 errors are 72 comparator failures,
321 null references and 55 invalid list indices. The checked fixture retains
392 cases (1,543,021 compressed bytes).

Assembly: `a800a3f8516a4d30767881e985197847f4d1dc2b832e5d18f211e511e7f15d67`.

Input SHA: `5324158e565690bc054f939b120420505cd2836b989b897df0be4262faedea9b`.

Output SHA: `8318fc0fe7d0c3c4701451cea9275d99629911ac4c6007953c13a2278792d1f9`.

All 24 larger cases complete, covering queue lengths 171, 256, 512, 1,024 and
4,096 with equal, ascending, descending, organ-pipe, repeated and extreme
integer priorities. Across 72 ticks, each of Init, UpdateSim, LateUpdateSim and
OnDestroyUpdatable executes 26,199 times. All 24 cases are retained in the
2,079,514-byte stress fixture. Export inspection found a maximum of 171 static
updatables on map 58; this count excludes players and later spawns.

Stress input SHA: `1449687e7fa301917da332db6a2327353cd941832bf7825e0968d0b6cc01c63f`.

Stress output SHA: `d1072b2f2ebc83a2f515c643325ac40371f08f907c9297777b59e4a3da376947`.

Raw evidence is under `artifacts/scheduler/` and `artifacts/scheduler-stress/`.
The Mono sort implementation was checked against the bundled mscorlib's
ArraySortHelper, not replaced with the Rust library's unstable sort. The oracle
uses the actual tick implementation, callback components and an empty physics
scene. Gameplay callback implementations, Unity OnDestroy handlers, non-null
player position setters, outer frame/hit-stop pacing and full map construction
remain integration requirements.

A fresh 1,200-tick movement replay verifies the optional Updater recording hooks
in normal gameplay. Its projection matches 473 independent C steps through
first death. It reuses the existing movement input and is not an additional
independent scenario or proof of scheduler integration into the C environment.

Replay SHA: `813fda56cec79199cb3930c12c081352c7f80d8f15483983d9ff8f1e7595e7eb`.

Trace SHA: `53be761dc33bf979c0292caadbd5984a53c0989d32498adbe2d84e68c22e9b65`.

This evidence is retained under `artifacts/scheduler-gameplay/`. The complete
standalone regression suite passes with both scheduler fixtures included.


## Composed scheduler, players and collisions

`verify_actors.py --cases 120 --steps 120` matches all 120 supplied-state
scenarios: 115 complete sequences, 13,801 completed ticks, 13,778 physics
stages and five matching division-by-zero partial failures. Each of the player
counts 1 through 6 has 20 scenarios. Across completed ticks there are 132,527
component updates, 132,302 late updates, 1,221 initializations, 29,440 collision
callbacks, 682 applied shoves, 1,177 attachments, 1,417 detachments and 1,270
jump-sound requests. Player input/update phases each run 48,602 times.

The reference calls the actual Updater with dispatch adapters for original
PlayerPhysics/PlayerBody methods and a scripted Jump/Move motor. Original
SlimeController.OnCollide supplies the equal-scale shove effects. This is a
composed component simulation, not a full SlimeController or gameplay episode.
The raw inputs supply initial Level1 component state, static platform geometry
and actor collider aliases. No per-tick Unity state is injected into Rust.
Ropes, bite, abilities, spawn/map initialization and death lifecycle are excluded.

The corpus covers shuffled/equal priorities, component disablement, loaded-level
early return, fractional and stopped player clocks, zero dt, jumps and coincident
players with extreme collision priorities. Native state and ordered events match
at each completed tick and at the first error. The fixture retains 36 cases
in 1,596,918 compressed bytes.

Assembly: `eddff5f1176ed0d60c99a7a9147a82383c1db21303c4e357d544c0c9bb9f527a`.

Input SHA: `4d3b68a2cbcadee52cf560ab90f16b40d57f250c68718705ce8ee321767c5911`.

Output SHA: `627fc421cb531110925c2bf17753838e657914b1bb261d19b4d55bac19001566`.

Evidence: `artifacts/actors/` and `artifacts/actors-verification.log`.
A fresh shared-oracle regression also matches 100 rigid scenes, 1,200 steps,
55 composite scenes and 14,898 callbacks after extracting its reporting helper;
evidence is in `artifacts/actors-physics-regression/`. The public C movement
world only shares the extracted shove direction; it does not yet use this runner.

The complete `just verify` suite passes with the composed fixture and shared
shove helper, including the existing independent C movement episodes.


## Scheduled SlimeController locomotion

All 120 supplied-state scenarios match the original SlimeController.UpdateSim
with empty abilities, including its SpriteAnimator, PlayerPhysics and PlayerBody
updates. Each player count 1..6 has 20 cases. The 115 complete sequences contain
13,800 ticks; a further tick completes before one of the five zero-previous-scale
failures, giving 13,801 completed ticks overall. There are 13,778 physics stages,
49,746 collision callbacks, 620 applied shoves, 898 attachments, 1,048 detachments,
910 jump-sound requests and 942 walk/land sound requests. Component update/late
counts are 179,286 and 178,993; both player phases run 48,602 times.

The comparison includes the full existing actor/scene projection and ordered
phase/collision events, plus locomotion state, facing, cached input, moonwalk
time, animation arrays/clocks/callback identity and the installed turn callback's
captured argument. Initial states cover modes 0..9. The six used MCAnimData frame
arrays and fallLeft velocity map come from the same asset export used by the
standalone animator tests. This corpus does not execute abilities, ragdoll,
swallowing, invisibility changes, spawn or death; player-position arbitration
and collider Scale writes are not included in its comparison.

Assembly: `792d5b0116b01308d07bbb10c7a754a727d83a4ca3da6bc3908fc0d124d75b23`.

Input SHA: `26d0f270484182a728b45b572cdb1482efbed0a67dfdf22ac02a9421e8fb04b1`.

Output SHA: `ee1eddc8fb46867161676a596bb94985be4d7c998fd09a6245ef90c8770b71f9`.

The retained fixture has 38 cases and 4,409,840 compressed bytes.
Raw evidence is under `artifacts/slime-actors/`.

All three targeted fault cases also match, including partial state at the failed
call: an empty land animation fails inside PlayerPhysics's attachment listener;
a missing fallLeftLoop fails inside the jump/ground-loss listener; a missing idle
animation fails in the stop-animation completion callback. The last scheduled
component is asserted to be physics, motor and animation respectively. The
third case advances 21 ticks before failing; the other two fail on their first
tick. All three are retained in the 18,457-byte failure fixture.

Failure input SHA: `d558cfc84bbd5ec75b1beaeea24baa32d1ec5d7f1fe0068fb69cefccf828b3b0`.

Failure output SHA: `1cc980ddb4229aec064cff0aa963e71818895dc02a10ae1af0c646496bc8b057`.

Evidence: `artifacts/slime-actors-failures/`. A fresh 60-case spawn regression
also matches after extending animator callback registration: 9,138 completed
calls and 154 spawn callbacks, including 15 initially installed callbacks. Its
evidence is in `artifacts/slime-spawn-regression/`. The complete `just verify`
suite passes with both new fixtures and all existing C movement checks.


## Scheduled ragdoll impulses and recovery

`verify_ragdoll.py --cases 120 --steps 120` matches all 120 supplied-state
scenarios, 20 at each player count 1..6. There are 115 complete sequences,
13,801 completed ticks and five matching division-by-zero partial failures.
The comparison includes 13,778 physics stages, 40,012 collision callbacks,
428 shoves, 179,654 component updates and 179,361 late updates.

There are 3,812 impulses across completed ticks: 1,922 reset animations and
1,890 retain them; 1,897 select fast and 1,915 slow; 647 have zero speed and
2,849 supply a magnet argument. The 15,308 ragdoll actor snapshots include
7,313 still grounded and 842 with an installed callback. Recovery clears an
existing magnet holder 308 times. These counts exclude effects in failed ticks.

Cases vary strict duration boundaries, positive/zero/negative dt, fractional and
stopped player clocks, protection from time stop, component order/disablement,
missing or destroyed ground and commands at multiple scheduler phases. The
reference invokes original ApplyRagDoll and UpdateSim; Rust retains its own
state throughout each sequence. The fast/slow arrays use exported MCAnimData.

Assembly: `dfa96c2c226fdb47b5c058ac14d820c65ff37d77db875b0c45b2de83f37967ef`.

Input SHA: `5713d1e64128d807673333e58e91da7bc47e382f851fd14b7cfd28316af2714d`.

Output SHA: `91ba92119d2875befb485a099f2bc8a0709b1da126e4f12853120386f25ca86e`.

The golden fixture retains 38 cases in 3,769,441 compressed bytes. Evidence is
under `artifacts/ragdoll/` and `artifacts/ragdoll-verification.log`.

All four targeted faults also match the partial first tick. Missing ragdollFast
or ragdollSlow fails before the impulse changes mode/body. Missing fallLeftLoop
fails after recovery transfers velocity and clears the holder. A missing renderer
fails earlier in recovery, while evaluating the sprite argument before entering
fall. Each case asserts the failing scheduled component and one applied command.
All four are retained in the 17,555-byte failure fixture.

Failure input SHA: `d398899abe0c771b6b0f4b07800ddc9da2c17c2128f3c1d0ea5b6efb5c991314`.

Failure output SHA: `4f3262eb04ba316b75c79f26110cba167ee9b3553b0be9ee3835ad933f88956f`.

Evidence: `artifacts/ragdoll-failures/`. The complete `just verify` suite passes,
including both ragdoll fixtures and the public C movement episodes. The pinned
5,199-file source/assets manifest remains unchanged and verifies successfully.
Ability-driven impulses, MagnetSteal, ropes, lifecycle and the full C integration
are outside this corpus and remain required.

A fresh capture of all 41 previously retained locomotion/failure cases through
the updated Unity harness reproduces their earlier expected outputs exactly and
matches Rust. Evidence is in `artifacts/ragdoll-slime-regression/`; this checks
the shared oracle as well as native regression behavior.


## Ordinary ability routing in scheduled actors

`verify_ability_routing.py --cases 120 --steps 120` matches all 120 scenarios,
20 at each player count 1..6. There are 100 complete sequences, 12,002 completed
ticks, 11,982 physics stages, 34,510 collision callbacks, 252 shoves and 55,053
timer changes. Partial source failures match in ten null-reference cases, five
short-timer-array cases and five division-by-zero cases. Failed ticks retain
and compare the full partial actor/scene projection.

Across completed ticks, direct calls to the original private isAbilityReady and
isAbilityCastable return 73,163 true/51,040 false readiness results and 14,630
true/109,573 false castability results. Each predicate also produces 41,403
expected out-of-range errors for the extra queried slot. The controller samples
include 6,602 snapshots already inside an ability and 13,912 ragdoll snapshots.

The original SlimeController.UpdateSim runs with real Ability cooldown and
castability methods. A reference subclass supplies the custom IgnoreNextCooldown
property. Slots include aliases, nulls, destroyed flags and physically destroyed
objects. Timer/indicator array lengths vary independently, including missing
indicators and readiness crossings. Ground/time-stop/protection, player permission
and suppression vary within the existing actor and ragdoll scenarios.

The reference deliberately prevents casts with released buttons or player
permission; this verifies the no-cast selection path and predicate boundaries.
Native successful selection currently reports `ability_entry_not_implemented`.
That path is not compared to an invented source exception. Rope cooldown updates,
ability entry/effects and indicator sprite/material presentation remain outside
this corpus. Indicator renderers are absent on both sides; reset clocks and
null-indicator exceptions are included.

Assembly: `29617c7346a48c9eabe63847c65bbabe7161170244488c864bc19d068e874551`.

Input SHA: `14f74805e68fa1b9ff47a30e32eeabf586be054c2224e7cd9d9a611ef49a060e`.

Output SHA: `756521a394fbc6a58c8f1fcb619a6afc333596366a923ca52fecfe657cde18a4`.

The retained fixture contains 48 cases in 4,898,777 compressed bytes, including
all partial failures. Evidence is in `artifacts/ability-routing/` and
`artifacts/ability-routing-verification.log`.

The full `just verify` suite passes with this fixture. A fresh capture of all 42
retained ragdoll/failure cases through the extended harness exactly reproduces
their previous source outputs and matches Rust; evidence is under
`artifacts/ability-routing-ragdoll-regression/`. The pinned 5,199-file manifest
also verifies unchanged.


## Executable Invisibility episodes

The 24 streamed scenarios match all 41,424 records: 41,400 successful frames and
24 end records. Each player count 1..6 has four scenarios; 23 complete all 1,800
ticks and one matches a zero-previous-scale division-by-zero failure before its
first completed tick. There are 41,395 physics stages, 56,462 collision callbacks,
748 shoves, 536,585 component updates and 536,510 late updates. Both player phases
run 142,200 times.

The original prefab's Invisibility audio is requested 339 times. There are 243
observed transitions to invisible and 203 to visible, with 115,299 invisible and
26,901 visible player snapshots. These transition counts differ from casts because
a different Invisibility slot can recast within the same frame that the previous
effect expires. Pool snapshots include 161,700 sentinel cooldown values and
62,752 restored cooldown values. Counts exclude events in the failed tick.

Every frame compares actors, scene, ordered events, all PlayerInfo fields, instant
object transform/active state, ability cooldown state, the slime's cached visibility,
and the shared player clock/flag projection. Material, slime, sprite and platform
references use canonical identities. The source setup primes Awake before supplying
state, then runs the real SlimeController, InstantAbility and Invisibility methods.
No Unity state is injected into Rust after initialization. This corpus has static
terrain and equal-scale shoves; full death/lifecycle/rope interactions and position
arbitration are not claimed by these tests.

Assembly: `e58e6c7675a8f4a0afe2971149e74ae2c0a6995e1a64706d9a52adceb5bf9173`.

Input SHA: `9ac9624dce2c2fe6fd8b23db1fe5d4871d4b1a8bdf38a84ade42efeeb8fed920`.

Output SHA: `27b876df67c33957ef99712ff7174fbb788cb19408996a3af911972913bfa0bd`.

Evidence: `artifacts/invisibility-stream/`. The standalone fixture retains complete
selected scenarios, including the source failure. The older six-case buffered
pilot remains a regression: 2,520 ticks, 35 cast audio requests and nine observed
expirations, with its own recorded assembly/input/output hashes in the fixture.

Seven targeted cases match: missing renderer, missing InstantAbility transform,
missing Invisibility link, missing effect component, a null slot during expiry,
managed-destroyed expiry and aliased minimum-age ties. The renderer/transform/link/
null-slot cases fail on the first tick in scheduled motor component 1, preserving
all partial state. The other three complete their tick. Both buffered and streamed
failure fixtures are retained.

Streaming was checked against all 13 buffered pilot/failure cases; every successful
and failed source frame reproduces the previous expected values and matches Rust.
The new protocol also has rejection tests for truncation, ordering, duplicate/excess
records, incorrect counts and missing partial state.

Streaming regression input SHA: `62d33c5549938915e1cf3ea3a48640ba970b6e576a8cbfefa342ef12dbd77e68`.

Streaming regression output SHA: `8ac66cc08bfd943c71f2d2ab04fcf3dcad69fdef5bfe82ad4e7fdfa3aef7e403`.

Evidence: `artifacts/invisibility-stream-regression/`. The earlier long buffered
attempt under `artifacts/invisibility/` did not finish and is excluded from the
results. Streaming reduced observed Unity memory from roughly 11 GB to 1.3 GB
while allowing the full 1,800-tick scenarios to complete.

### Shared player position and clone priority

48 independent cases (eight each with one through six actor identities) match
13,800 completed ticks and two original division-by-zero failures at action zero.
46 cases complete all 300 ticks. 32 cases assign multiple controllers to the same
registered Player; the actor-count breakdown is not a distinct-owner breakdown.
The comparison includes all actor, physics, scheduler, input and shared-player
fields on every frame and the complete failed frame.

The source produces 47,700 player snapshots, 25,583 with a current setter, 17,987
with the previous setter retained, and 12,082 winner changes. There are 56,261
explicit position/lifetime commands interleaved with real SlimeController updates.
One third of cases omit explicit position writes and exercise natural motor
arbitration. Managed destruction, null candidates, unequal slot counts, ties,
update disablement and multiple time scales are included. The 14-case retained
fixture includes both source failures. The full capture also records 24,284
collision callbacks and 158 shoves.

- Unity assembly SHA-256: `42aa4a7b0d3ae8f63dda59a0b985f2366ad5955dc8505c34200e3129e222894a`
- Input SHA-256: `31fc1d81051f6251c30e488b5fbff36b0a9ff055153638a512ef2a20b4ab7be0`
- Output SHA-256: `032cc0df055a32518d7232b82a6d8d3de98a8bf3c1d464272bc8249ca1ffd92e`

Run `just --justfile native/justfile verify-player-position` with a fresh artifact
directory for a live capture, or `python3 native/tools/verify_player_position.py
--golden` to check the retained independent trajectories. Physical controller
destruction, clone creation and ordinary ability entry/exit remain outside this
capture.

### Actor activity, body gates and real collider scale

All 48 independent 300-tick actor episodes match Unity (14,400 ticks, eight cases
each with one through six actors). The full state comparison covers actual
GameObject activation, body simulation enablement, collider enablement, player
scale clamping, DPhysicsBox scaling, transform offsets and explicit physics
synchronization. The source runs 14,390 physics stages and 16,974 callbacks.
Variable-size player bite/shove behavior is excluded.

50,400 actor snapshots include 1,686 inactive objects, 24,486 bodies with simulation
disabled and 33,862 non-unit collider scales. There are 417 detach events while
body simulation is disabled. The source uses 15,744 activation commands, 2,064
body simulation changes, 2,880 box scale writes, 864 player scale writes, 768
collider enablement changes and 14,400 explicit box synchronizations.

Eight targeted edge cases match: three complete a tick and five stop on original
division errors. These cover zero old collider scale, zero old transform scale,
zero resulting scale, initialized/uninitialized colliders, zero inverse extents
on either axis, and nonzero transform offsets. Failed frames retain the exact
partial scale, offset, basis and state mutations. No failed capture is promoted.

- Unity assembly SHA-256: `19181b8cd02bb7fa2717027b137c6c30357d6f347742c1bab557a0f666f72a28`
- Main input SHA-256: `0b5c8886087a861d2dc86fcf51aabb304d6ccb0b214ae3ddd610264ff889a2a5`
- Main output SHA-256: `370fb7c8005c5de56a482c9bea60d3072f32fb75037b4a2523f22f3c2eeeda0e`
- Edge input SHA-256: `bc1ba84d1f28814f10f96d1f929429be737572ad96701f34a8c13dd6bb3d8e4d`
- Edge output SHA-256: `44218f4a90dd1b10406e2a640b5b58e6db6c109112c32dcfb376529efbde9067`

A fresh 13-case Unity run also reproduces all 2,536 records from the prior
Invisibility source corpus exactly (output SHA-256
`8ac66cc08bfd943c71f2d2ab04fcf3dcad69fdef5bfe82ad4e7fdfa3aef7e403`).
The retained actor-entity fixtures contain 12 complete 300-tick episodes and all
eight edge cases. Original Awake runs, but prefab Init, the complete registered
component graph, ordinary ability exit and full round lifecycle are not covered.


## Ordinary ability ownership and return callbacks

Two independent source corpora each match all 48 complete 300-tick episodes
(14,400 ticks each), with eight cases for every player count from one through six.
The first invokes original SlimeController.onExitAbility and continues scheduled
motion; it includes 18,200 physics callbacks. Nine targeted cases include two
successful returns and seven original failures (six null references, one index
error), with exact failed-state comparison.

- Slime-return assembly SHA-256: `ef0b392936f6425756b3c4ed2237118caef2c904308f32b17470f35dda6e7e69`
- Main input SHA-256: `38d64604b85edb9914dabc7cf1e00d28cfff383f983e3931eab31acfc6a85f7b`
- Main output SHA-256: `f324691865907586a007db63210e4e9e882c5bdb48501fa01859d77ada85e73b`
- Edge input SHA-256: `11cfc8e750bb00b009bcbef48acc26fbdf7946675934b47b0cf7bd5bb8c5a5f9`
- Edge output SHA-256: `3c7fe063c88ba54c14e4ab965c3c66215ac50e70509609845fae2c11f4d0bad7`

The second corpus binds one separate ordinary Ability body per primary slime
(two through twelve actors). Held player inputs produce 1,499 actual ability-entry
events. There are 1,680 supplied exit calls, 14,390 physics stages and 74,216
collision callbacks. The source compares shared player flags/position, all actor
and collider state, complete PlayerInfo and callback ownership, ordered scheduler
phases and events. Five targeted cases cover missing collider, destroyed target,
ready entry, absent exit callback and zero previous scale: two complete and three
match original failures (two null references, one division error).

Stress commands deliberately change primary object activity independently of the
ability lifecycle. Among 50,400 primary/ability pair snapshots, 23,688 have only
the slime active, 5,046 only the ability, 20,897 both, and 769 neither. These are
supplied-state stress episodes, not ordinary full-match legality tests. Between
frames there are 1,499 ability activations and 1,498 deactivations; actual entry
events, rather than frame transitions, are the cast-count authority.

- Ordinary assembly SHA-256: `42cdaf099b16498200c19fff9d73f57903a4f13b5574586b18a1fb706d90c046`
- Main input SHA-256: `f58994f0200a47d03bfd44404329d404b12063b749f351b94f59ad527a87debf`
- Main output SHA-256: `14b78ba361fe22bb5e1cec27f3111350694365098d8377751958f933d654af49`
- Edge input SHA-256: `164045930a9843fd84a608c5c34cd56a9084d19fdcc69447a35f1b872f03842b`
- Edge output SHA-256: `73ea7b0ddc69a97b2f0847e47d09c4ea85cf1bde7f0a37bf827f7d8ac7cf185a`

Fresh Unity regression with this assembly also reproduces all 2,536 records from
the prior 13-case Invisibility corpus byte for byte (output SHA-256
`8ac66cc08bfd943c71f2d2ab04fcf3dcad69fdef5bfe82ad4e7fdfa3aef7e403`).
Each new main fixture retains twelve complete episodes; all edge cases are retained.
Plain Ability contains no concrete effect component. Concrete ability lifecycle,
rope transfer, successful death returns and complete matches are not established.


## Concrete Dash with automatic returns

All 14,345 source records match across 48 scenarios, eight at each player count
from one through six. Forty-seven reach 300 ticks; case 3 stops at action 197 on
an original division-by-zero error. The comparison includes 14,297 completed ticks
and the complete failed state. That failure occurs when Dash actor 6, aiming
exactly right, attempts its air return while below the grass water height.

The completed frames contain 2,003 real entry events and 1,910 automatic returns.
Because this corpus supplies no activity or exit commands, returns are counted
from entry events and conserved ability activity, including possible same-frame
return/reentry (none occurs here). There are 15,938 active windup snapshots and
7,850 active dash snapshots. First raycast transitions include 1,515 air casts,
405 wall casts and 57 casts with multiple hits. Exactly 105 automatic returns end
at the relevant water height. The source executes 14,287 physics stages and
20,576 collision callbacks. Later cases also change player scale.

Ten boundary cases match: five complete twelve ticks each and five retain exact
failure state. They cover zero aim, the exact pause threshold, water clipping,
horizontal movement below water, a non-wall contact with missing PlayerCollision,
a wall-layer object missing StickyRoundedRectangle, twenty tied-distance hits,
twenty fully equal sort keys, stopped time, and a negative update-count threshold.
The errors are one division-by-zero and four null references. Both twenty-hit
cases compare the full sorted list before the expected first-contact failure.

- Unity assembly SHA-256: `e382cb1d2434e5cf524d1ea64c96d62f0afc878cf1e991dabc68a946e1fd5880`
- Main input SHA-256: `e3149af32ab90fdec0a5e91b40d9fcc676d8f2da7b927efd1ebcae9d08c71c92`
- Main output SHA-256: `ba45e5853aae9ae5dd646497105ab520fce9f813e95085a479935107c6348a90`
- Edge input SHA-256: `d1bf2a59fe8e3e2ee1144fbae99e8176ee845b730cf40908c69814c9c0954d61`
- Edge output SHA-256: `cf4b3fd5ee642049a0a0b594c083018d80ef694c8b5ca208b99cd41671b928fa`

A fresh combined 19-case source regression exactly matches the 3,142 prior
Invisibility and plain Ability records after canonical case reindexing. Its output
SHA-256 is `d4f12599f02b521d7c55a0b40bddede82f90afae4703306fb2abb40812481e34`.
The retained main fixture contains eleven complete 300-tick episodes plus the
197 completed ticks and failed state of case 3. All ten boundary cases are retained.
Successful non-wall damage, rope transfer and full-match lifecycle are not covered.
