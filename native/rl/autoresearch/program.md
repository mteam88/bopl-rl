# Bopl throughput research

You are Fable. Make Bopl RL training substantially faster so we can afford
strong players across abilities, maps, combinations, and adaptive opponents.
Own the research: investigate, experiment, measure, keep useful improvements,
and continue until Matthew stops you. A failed experiment is useful evidence.

You may change any relevant implementation, architecture, runtime, or tooling.
The benchmark and checks are starting equipment, not an edit allowlist. Use
your judgment about promising directions and how much validation each needs.

Start by reading `native/rl/README.md` and
`native/rl/evidence/throughput/summary.json`. Preserve the starting commit and
measure a fresh baseline before changing performance code. Work in this Rift;
commit experiments and keep a small `results.tsv` and `notes.md` here with the
hypothesis, revision, workload, speed, memory, validation, and keep/discard
decision. Keep raw runs under `native/artifacts/rl/autoresearch/`.

Matthew wants a live progress graph. Write results to
`native/rl/autoresearch/results.tsv` with these tab-separated columns:
`timestamp_utc`, `commit`, `experiment`, `workload`, `learner_dps_wall`, `status`,
`description`. Use ISO UTC timestamps and status `baseline`, `keep`, `discard`,
or `invalid`; the score is the measured median. Give changed workloads distinct
names so the graph compares like with like. Extra columns are welcome. A BB
monitor checks hourly; keep researching between checks without waiting for it.

## Starting measurement

From the repository root, use a new output directory for every invocation:

```sh
native/artifacts/rl/venv/bin/python native/rl/research/bench_throughput.py \
  --output native/artifacts/rl/autoresearch/baseline \
  --workers 16 --repeats 3 --decisions 393216
```

Primary score: `table.base-w16.median_learner_dps_wall` in `summary.json`.
This counts learner samples consumed by completed PPO updates per process wall
second, including startup, resets, snapshots, and finalization. Also inspect
`median_learner_dps_active` to understand sustained throughput. Opponent-only
actions are not learner samples. Evaluation/video are excluded in this benchmark.

Calibrate noise, alternate incumbent/candidate runs, and remeasure promising
changes over longer, representative workloads. Run performance measurements
one at a time; this workstation is shared. Improve the benchmark when useful,
keeping the original measurement reproducible and explaining changed scores.

## What makes a gain real

Faster execution should still train a useful agent in the real game. Preserve
game behavior and learning semantics, or explicitly measure the tradeoff if
you change them. Bit-identical floating-point results are not required. Do not
count omitted work, missing observations, or a weaker workload as a speedup.
Use `native/rl/research/check_pipeline.py`, game replay checks, and longer
training comparisons as appropriate. Validate changes beyond the tuning cases.

The existing loop spends most time collecting gameplay; IPC, synchronization,
small inference batches, and copying are leads, not a prescribed plan. Crowded
games have previously hit the observation capacity: keep that failure visible.
Do not narrow the project to Bow or quietly substitute a simpler game.

Keep updates to Matthew brief: best verified gain, what you learned, next idea.
Commit useful work in this workspace; preserve results for handoff.

## Claude Code

Run `native/rl/autoresearch/run.sh` from a dedicated Rift checkout. It loads the
locally available official Ralph Loop plugin for this session and selects
`claude-fable-5-1`. No global plugin installation is needed. Pass `--bg` to run
in the background; use `claude attach <id>` / `claude stop <id>` to manage it.
The prepared Rift has the existing proxy credentials in a private, gitignored
`native/rl/autoresearch/.env`, so launching from a fresh SSH shell works too.
In an interactive session, `/ralph-loop:cancel-ralph` cancels the repeat hook.
Ralph repeats when Claude finishes a turn; its iterations are not experiments.

## Priority change (Matthew, 2026-09-07)

Matthew: "we should definitely use puffer's native trainer, we never should
have made our own in the first place." The research continues in this order,
recorded here so the Ralph loop keeps pursuing it:

1. **IL2CPP player.** Install the Linux IL2CPP support module for the
   installed editor (2022.3.60f1, revision 5f63fdee6d95, at
   `/home/mte/Documents/Coding/bopl-re/unity-editor`; module
   `https://download.unity3d.com/download_unity/5f63fdee6d95/LinuxEditorTargetInstaller/UnitySetup-Linux-IL2CPP-Support-for-Editor-2022.3.60f1.tar.xz`),
   build the headless game with the IL2CPP backend, and benchmark it with the
   current trainer: game stepping first, then full training on the sustained
   workload. Tool installation on this dev box is authorized.
2. **PufferLib native CUDA trainer.** Move the main training path onto
   PufferLib's native trainer (`native/rl/puffer/`), preserving the real game,
   delayed observations, held inputs, match-long historical-opponent ownership,
   learner-only updates and the reset semantics. Known audit findings to
   qualify or fix during integration: the native prioritised sampling adds
   1e-6 to every row including frozen rows and alpha=0 gives zero-advantage
   rows weight 1, so `zero_frozen_advantages` alone does not guarantee
   learner-only updates; `count_aligned`-style boundary tracking means every
   tagged env has hit *some* boundary, not that all are between matches, and
   Bopl needs match-long ownership rather than bank-weight swaps mid-match;
   recurrent resets need checking. Further source-inspection findings from
   Matthew: `net_callback_wrapper` copies terminals but calls `policy_forward`
   without clearing learner/frozen RNN state at episode boundaries; `TrainGraph`
   stores `mb_state` but no segment-start rollout state or reset mask, and
   `train_impl` only zeroes `mb_state` when `hypers.reset_state` is set while
   `mingru_forward_train` scans across internal episode boundaries (both differ
   from the current GRU contract); `train_impl` hard-clamps rewards to [-1,1]
   while Bopl's round+kill reward can exceed 1; native throughput counters
   include frozen rows and possibly a final unlearned transition, so report
   learner samples actually used by completed updates separately. Account for
   these explicitly with focused upstream adaptations, not another replacement
   trainer. Then qualify throughput **and** strength per wall hour against the
   existing trainer. Do not stop at a plan or a smoke run.
3. **Standalone simulator.** Assess that investment from the results of 1 and
   2; no full rewrite is requested now.

The measurement discipline above still applies (one perf run at a time,
distinct workload names, results.tsv + notes.md, raw runs under
`native/artifacts/rl/autoresearch/`). The in-flight exp13 (parallel branches
inside the turn graph) is finished or checkpointed first, not abandoned
mid-measurement.

### Division of work (Matthew, 2026-09-07 evening)

Matthew owns, in a separate root Rift: native checkpoint evaluation support using the upstream CPU
PufferNet (`src/puffernet.h`), so native MinGRU policies can enter the existing real-game evaluator
and crossplay against GRU checkpoints. Their files, which this Rift must not create or edit:
`native/rl/puffer/policy_eval.c`, `native/rl/research/native_policy.py`, and their focused parity
checks. They will return the local commit and API. This Rift keeps IL2CPP and the native
trainer/self-play integration. Do not run GPU validation runs alongside Matthew's benchmark-free
work; one heavy job at a time still holds.

Native detail from Matthew: `bindings.cu:140` resets learner/frozen RNN state at whole rollout
boundaries when `reset_state=true`, which is distinct from episode resets. The Bopl contract needs
per-episode resets, so the integration has to add that rather than rely on `reset_state`.

Build attribution: `train.py` used to hash `Builds/Linux/.../Assembly-CSharp.dll` regardless of
`BOPL_RL_GAME`, and `verify_replay.py` assumed the Mono layout. Both now record the selected player
and its runtime payload (Mono: Assembly-CSharp.dll; IL2CPP: GameAssembly.so + global-metadata.dat)
through `environment.game_identity()`; IL2CPP results are only build-attributable with that in place.
