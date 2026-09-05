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
