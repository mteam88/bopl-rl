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
