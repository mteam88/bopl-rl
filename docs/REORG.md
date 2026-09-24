# The reorganization, and the decisions behind it

The repo grew by accretion: one directory per experiment, dead Python left
behind whenever an approach was abandoned, and measurement records filed in
three different places. This is the approved plan to reorganize it around
AlphaStar's workflow, and the standing decisions that shape it. It exists so
that the plan survives a session ending.

Approved 2026-09-23. Little backwards compatibility is required: the work is in
git, and retraining the best checkpoints takes a few hours.

## Target layout

| directory | holds |
|---|---|
| `sim/` | the simulator: the game's C# (the oracle) and the Rust port |
| `train/` | the trainer, the policy, PFSP opponent selection |
| `league/` | the league registry and its members - a permanent artifact in git |
| `eval/` | evaluation, the gates, the divergence instrument |
| `board/` | the dashboard |
| `docs/` | everything written down |

## Decisions

**One generalist, not a population of mains.** A single main agent, on the
expectation that it ends up with a better grasp of dynamic strategy than
several narrower mains would.

**Exploiters start from random init, not from a checkpoint.** An exploiter
seeded from the generalist inherits the generalist's habits, and we want
exploitive ability combinations that do *not* transfer the way we would guess.
Random init is the point, not a cost.

**The league is a permanent artifact in git.** It grows: exploiters get added,
and its strength grows with them. It is not rebuilt per run. Ratings are only
comparable within one (sim hash, reward semantics, loadout) condition, which is
why the registry keys payoff counts by condition.

**Promotion stays implicit.** No hardcoded promotion rule. What is needed is the
framework for organizing what sits in which pool, plus the recognition that
pools saturate - which has already happened here.

**No held-out abilities as the generalization split.** The abilities are very
different from each other and hard to learn; holding out nine of 38 tests
whether a policy can learn nine new abilities, not whether it learned the game.
Maps are the generalization axis.

**Surgery stays available but stays simple.** It is a worse fit for this problem
than it was for AlphaStar, so it should be as simple as possible.

**q32 is a measurement instrument, not a goal** (2026-09-23). Bit-exact
agreement with the oracle is how we *measure* fidelity; it is not a target worth
spending fidelity work on for its own sake. The evidence: q32 and f64 produce
identical discrete medians at training horizons, so a residual f64/q32
divergence that does not move a discrete outcome does not change what a policy
learns.

- A divergence that is f64-only - both Rust builds agree with each other but not
  the oracle - is a logic defect and is worth chasing. That is the useful
  signal; `entity[].right_x` aim latching (FIXTURES #1) is the standing example.
- A divergence that is q32-only, i.e. arithmetic, is a measurement artifact.
  Record it; do not chase it.
- Keep the q32 build working, because the exactness gates depend on it:
  `map-sweep-3000`, `map-sweep-endgame`, and `leakprobe-q32`/`selfcheck-q32`,
  which detect cross-episode contamination with no oracle at all.
- f64 stays the build that trains. `native/src/fixed.rs` is f64 behind the
  Q32.32 `Fix` API; `native/src/fixed_q32.rs` behind the `q32` feature is the
  bit-exact one. Unit tests in `fixed.rs` pin the constants and the timer
  exactness that both builds rely on, so that knowledge lives in code, not here.
- After editing anything under `native/src/`, rebuild **both**:
  `cargo build --release --manifest-path native/Cargo.toml` (f64, which the
  divergence harness loads as backend 2) and `make rust-q32`.
  Rebuilding one leaves the other stale and a working fix measures as broken.

## Deletion inventory

Deleting is most of the work. Verified dead, to be removed as the reorg lands:

- ~3,200 lines of dead Python. PFSP landed 2026-09-24 and took the old opponent
  selection with it: `--swap-winrate`, `--elo-init`/`--elo-k` and the trainer's
  online Elo are gone (the banks stayed — they are a native construct)
- `review/` - removed 2026-09-23; 9 files of prose from a completed code review
- `reports/` - removed 2026-09-23; it was a single stale `.pyc`
- `evidence/` - removed 2026-09-23; 38 dated measurement records, ~2.0M
- **43 stale documents, 21,348 lines** - removed 2026-09-23. They described
  approaches that were abandoned, carried version numbers in their names, or
  recorded one-off spikes and reviews that had already landed. Six documents
  survive, and each is either a live contract or actively used:
  `docs/REORG.md` (this file), `docs/README.md`, `docs/LEAGUE.md`,
  `docs/METRICS_CONTRACT.md` (the observation/reward field contract - prose
  only, nothing asserts it in code),
  `eval/divergence/FIXTURES.md` (open defects and their repro recipes),
  `eval/rust_ab/README.md` (the probe suite behind `rust-ab-suite`).

All of it is recoverable: everything was committed and pushed to the private
origin before deletion. `git log --diff-filter=D --name-only` finds it.

Two things were lifted out before the docs went, because they were the only
record of something still open: the q32 decision above, and three open defects
now recorded at the end of `eval/divergence/FIXTURES.md` (the oracle's
`ShootDuplicator` NullReferenceException, the 1-in-17 determinism flake, and the
AOT `rd.xml` residual risk).

**Still to do: 139 comment references across 87 files point at deleted docs** -
things like "see POLICY_V4_PLAN.md section 6". A pointer to an explanation that
no longer exists is worse than no pointer. Strip the reference and keep the
substantive text: `# SIM_OVERHAUL.md section 4: one GC heap under every world
costs 25-45%` becomes `# One GC heap under every world costs 25-45%`. This is
deliberately folded into the reorg rather than done separately, because the reorg
moves nearly all of those files anyway. `train/sweep/configs/*.toml` is the
bulk of it.

## The mapping, concretely

Decided 2026-09-24. Stage 1 moves the RL tree, which is where the accretion is.
It deliberately does **not** move the Cargo root, the Rust port, or the artifacts
tree - see "What does not move" below for why.

| from | to |
|---|---|
| `eval/` (minus the dashboard parts) | `eval/` |
| `board/web/`, `board/serve.py`, `eval/start-dashboard.example.sh` | `board/` |
| `train/` | `train/` |
| `train/sweep/` | `train/sweep/` |
| `league/` | `league/` |
| the gate and benchmark C programs at `native/rl/*.c` - `fidelity_gate`, `backend_check`, `native_check`, `coverage`, `dump_state`, `sampled_check`, `reset_check`, `reset_profile`, `frame_profile`, `record`, `benchmark`, `writer_bench`, plus `verify_replay.py` | `eval/` |
| `sim/abi/bopl_rl.c`, `sim/abi/bopl_rl.h` | `sim/abi/` - the C ABI both backends sit behind, so it belongs with the simulator rather than with either consumer |
| `Makefile` | `Makefile` at the repo root, so the command is `make ability-sweep`, not `make -C native/rl ability-sweep` |
| the six surviving docs | `docs/` |
| `native/rl/autoresearch/` | deleted - accretion era. `check_routing.py` and `smoke.py` were repointed and the `.gitignore` entry dropped |

`native/rl/` should not exist when this is done.

## What does not move, and why

- **`native/artifacts/` stays exactly where it is.** It is gitignored live data,
  and absolute paths into it are baked into archived run metadata - 293
  references across 179 files. Never rename a live `sweeps/<run>` directory.
- **`native/src/` (the Rust port), `native/Cargo.toml`, `native/target/`** stay
  put in stage 1. Moving the Cargo root relocates the `.so` that every gate
  loads and orphans a warm `target/` that costs a full rebuild of a 30k-line
  file. Folding the port into `sim/` is stage 2, once stage 1 is green.
- `native/data/`, `native/fixtures/`, `native/reference-export/`,
  `native/tools/`, `native/justfile` stay in stage 1.
- `Assets/`, `ProjectSettings/`, `Packages/`, `Tools/`, `UserSettings/` are the
  decompiled Unity project, not our code. Never touch them.

## Order of work

1. ~~The reorg itself: move to the layout above, execute the deletion
   inventory.~~ **Done 2026-09-24**, merged `c2293d22`, pushed `654dd50e`.
2. PFSP into the trainer, replacing the old opponent selection. *In progress.*
3. Minimal surgery. *In progress.*

Then the first generalist run over randomised loadouts, the first random-init
exploiter, and time-to-exploit as the headline metric. The board already reports
time-to-exploit (`683d1fc9`), so that metric needs a run, not code.

### Stage 1, as verified

All seven gates green on the merged commit with **every gate binary deleted and
rebuilt from source first**, which is not ceremony: the gates live in gitignored
`native/artifacts/rl/`, `make` only compares mtimes, and `bounds-ability-ab` had
been running a binary older than its committed source - printing 1 known
divergence where the rebuilt gate prints 6, under a byte-identical summary line.
A gate result nobody forced a rebuild for is not evidence.

| gate | result |
| --- | --- |
| `check-reward` | pass |
| `bounds-ability-ab` | `cases=38 water=-11.3 survivors=4`, 6 known divergences |
| `sudden-death-publication-ab` | `maps_mismatched=0` |
| `ability-sweep` | 24,800 episodes, 4588/4588 ability-map cells, 0 crashes, 0 divergent |
| `map-sweep-3000` | 417,009 snapshots, 0 nonexact |
| `map-sweep-endgame` | 801,501 snapshots, 0 nonexact |
| `rust-ab-suite` | `pass=211 fail=0 skipped=4` |

`Tools/` was relinked afterwards (`ea57067d`): it had eight dead `native/rl/...`
references and was skipped during the move because the "do not touch" rule was
read too broadly. That rule is about `Assets/` and the 5,301 decompiled Unity
files, not about the four scripts we wrote ourselves.

## Naming

Simple and easy to understand. No version numbers, no `_v2` suffixes, no
backwards-compatibility shims. If a name needs a number to disambiguate it from
its predecessor, the predecessor should have been deleted.
