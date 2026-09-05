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
dash. It stops at the first death. The complete rigid-body solver, moving/scaled
platforms, damaging abilities, ropes, clones, time manipulation, match scoring,
stocks, and respawn flow are not ported. Float conversions outside normal finite
game values are not part of the verified arithmetic contract; exported assets
use raw integers. The Unity recorder has the serialization limits documented in
README.md. No trained policy or PufferLib backend has been validated.

Expand these gates alongside each new mechanic. Preserve failures and report
which fields, episodes, and interactions were actually compared.
