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

### Cleanup requirement (Matthew, 2026-09-07 evening, after the native decision)

Matthew: "I want you to strip out the old optimizer and any old stuff as if it had never existed.
its in git as a backup, you don't need to save it anywhere". This supersedes the earlier suggestion
to retain the old trainer for reproduction. Finish the approved IL2CPP comparison using its current
baseline, then make PufferLib native the ONLY shipped training path. Delete the custom PyTorch PPO
optimizer/trainer, old CUDA-graph helpers and obsolete scripts/config/docs/compatibility paths that
the native integration replaces; do not create legacy/archive copies or keep a backend switch to the
old implementation. Update callers and docs around the final native design. Git is the fallback. Use
existing git revisions only as needed for qualification, then remove temporary comparison
implementations. Preserve necessary real-game environment, native evaluation and replay
functionality; avoid sweeping unrelated gameplay/source files. Apply this before calling the
migration complete.

Dependency audit from Matthew: many retained evaluators/video helpers import `ROOT`/`atomic_json`
from the old `research/train.py`; move those utilities to a small shared module rather than
retaining the old trainer to satisfy imports. Old `train_bow.py`, `imitate.py`,
imitation/league/extension launchers and old `check_*` scripts directly depend on the custom
Policy/PPO; remove obsolete ones or migrate genuinely needed callers. `puffer/train.py` has a
`--backend torch` fallback and `puffer/smoke.py` imports `torch_pufferl` too: the final entrypoints
exercise native only. Matthew will audit dangling imports and stale docs after the cleanup.

## Matthew's additional decision, relayed by his Cursor thread, 2026-09-07

PufferLib native is the only trainer. Add a native Bopl set encoder and completely
remove the old implementation. This extends the priority above and is required
before the finite migration task is complete. Keep one heavy job at a time.

### Required order

1. Finish the native migration with the upstream default linear encoder first:
   routing, historical opponent ownership, per-episode resets and learner-only
   updates. Qualify throughput and strength per wall hour against the old trainer
   on the sustained-w16 workload. Use Git revisions for temporary comparisons.
2. Then land the Bopl set encoder specified below and qualify its strength per
   wall hour against that linear native baseline. Record the two native workloads
   distinctly in results.tsv and notes.md.
3. Assess the standalone simulator investment using the resulting evidence, as
   previously requested; a full simulator rewrite is not requested now.

Comparison clarification from the parent: the old trainer uses a standard GRU,
whereas the native trainer uses upstream MinGRU. The first comparison retains a
linear encoder, but it is not an identical-model or pure execution-engine
comparison. Identify the architecture/algorithm differences in its report.
The set-versus-linear native comparison keeps the qualified native recurrent
core, decoder, optimizer and environment unchanged.

### Native Bopl set encoder

Implement a custom CUDA encoder through `create_custom_encoder(env_name,
&encoder)` in the pinned PufferLib `src/ocean.cu`: add the Bopl branch through
the build.py overlay mechanism. Follow the nmmo3 encoder's Encoder vtable:
forward, backward, init_weights, reg_params, reg_train, reg_rollout,
create_weights, free_weights, free_activations and activation_size. Keep the
upstream MinGRU and decoder architectures unchanged.

Observation layout from bopl_puffer.h (3257 floats): 10 header, 6 x 148 players,
96 x 21 entities, 12 x 28 slimes, 7 previous-action values. Presence is column
zero in each entity/slime row. The initial architecture is:

- Flat block: header + players + previous action = 905 floats, Linear then
  ReLU. Self remains player slot zero, so the player block is positional.
- Entities: reshape to (B*96) x 21; shared Linear(21 -> E) then ReLU using one
  puf_mm; masked max pool over 96 rows to E. Invalid rows cannot win the max;
  an empty set pools to zero.
- Slimes: analogous shared Linear(28 -> S), ReLU, masked max pool over 12
  rows, with the same mask/empty-set behavior.
- Concatenate flat features, entity pool and slime pool; Linear then ReLU
  to hidden_size=128 for the upstream MinGRU. Start with E=S=64.

There is no native autograd. Implement every backward: ReLU, masked max pooling
(save argmax in activations and route gradient only to the winning valid row),
concat splitting, and matmul gradients using puf_mm_tn/puf_mm_nn. Register
rollout B and train B_TT activations and gradient scratch following nmmo3.
Prove the implementation with finite-difference gradients on a small batch and
comparison to a throwaway Torch implementation of the identical encoder. That
reference is only a test oracle, never a shipped fallback beside the trainer.

Implement the matching CPU encoder forward for policy_eval.c/native_policy.py
and upstream puffernet.h evaluation, and prove CPU versus CUDA forward parity
on saved inputs. This CPU support is now explicitly part of the parent's work;
coordinate the implementation boundary before editing those evaluator files.
Trained set-encoder checkpoints must be evaluable and crossplayable.

### Complete removal of the old implementation

Matthew: "completely rip out any old stuff." No archive copies, backend switches,
compatibility shims or legacy directories. Git is the backup. On completion,
remove all of the following from the shipped tree:

- Custom research/train.py, ppo.py, inference.py/Stepper, selfplay.py, turn.py,
  policy.py's Torch Policy/distribution/input_mode/action_mode variants,
  continuation.py, scripted_opponents.py and the scripted-opponent path. Remove
  the Python Vector pipeline-group layer if native training does not use it.
- Trainer-side flags BOPL_TURN_GRAPH*, BOPL_COLLECT_GRAPH, BOPL_FAST_MULTINOMIAL,
  BOPL_OPPONENT_STREAMS, BOPL_ROLLOUT_ENTROPY, --pipeline-groups,
  --pipeline-serial and --cuda-graphs.
- --backend torch, torch_pufferl imports, and imports of the old Torch Policy.
- Old comparison oracles/checks after qualification: check_pipeline.py,
  check_selfplay.py, check_padded_sampling.py, BOPL_SPLIT_STEP's two-roundtrip
  path, step_mode fallback plumbing for old players, and sampled_check if it
  exists only to test that fallback.
- Game-side incumbent toggles BOPL_SPARE_GAME, BOPL_RL_HEADLESS,
  BOPL_RL_RAYCULL, BOPL_RL_COALESCE, BOPL_RL_EAGER_IO, BOPL_PARALLEL_OPEN,
  BOPL_RL_TICKS_PER_FRAME and the exact fast Fix div/sqrt fallbacks. Make kept
  behavior unconditional; remove off branches and tests solely comparing on
  versus off. Retain correctness checks for replay parity, telemetry
  conservation and reset semantics. Arithmetic must remain exact over its full
  supported domain; replace required numerical handling equivalently rather
  than silently narrowing that domain.
- Old .pt GRU checkpoint loading. Use a Git revision for the old-trainer
  qualification, then remove temporary comparison code and .pt dispatch.
- Docs, README sections, dashboard serve.py/web code and metrics.jsonl readers
  for the old outputs: update them to native or delete them. Move shared ROOT
  and atomic_json out of the deleted trainer into a small common module.

Keep: the real game and bridge (RlBridge.cs, bopl_rl.c/h), the Bopl native
environment/bindings, puffer/train.py, native checkpoint evaluation/crossplay,
replay export and video, series.py and the evaluation protocol retargeted to
native checkpoints, the autoresearch harness, and correctness checks for kept
environment behavior. The parent's deletion of old series.py in 6a6dcaf removes
its old implementation; provide its native replacement as required here.

Final audit: rg for torch_pufferl, `from policy import`, Stepper, TurnGraph,
BOPL_SPLIT_STEP, --backend and every deleted flag name. Resolve all shipped-tree
hits, including stale docs and harness instructions. Once these migration
instructions are fulfilled, rewrite this program around the final native design
so completed removal checklists do not preserve obsolete workflows or tokens.
Matthew will independently audit dangling imports and stale docs after completion.


## Latest Matthew instructions: preserve Learning Lab and queue the long native run

These instructions supersede the earlier request to delete the dashboard.
Full native-only cutover remains mandatory after the sustained-w16 comparison
against the old trainer. The old trainer, its flags, and checkpoint loading go.

Learning Lab is required: retain research/serve.py, research/web/,
study_data.py, replay_library.py, the dashboard checks, and port 8876. Preserve
learning-health curves, fixed-opponent win rates, throughput, entropy, button
rates, match video and physics playback, the existing replay browser, and the
study view. Native training must provide the metrics.jsonl and behavior data
these panels require, either directly or through an explicit native-output
adapter. Native checkpoints must remain playable through evaluation and
replay/video export. Thread thr_iiychwrq9p owns the dashboard UI in its dashboard Rift. Root restored
the UI files and made no UI adaptation edits. Trainer/evaluator/metric output
remain owned by this migration team. Do not apply UI-deletion commit 5960253. The live
base dashboard has not been retired. New panels will be specified separately.

After the native-versus-old sanity comparison, full cutover, and qualification
of the SET encoder (correctness, throughput, short strength comparison with
linear), hand off to thr_wq22hezkbf. That thread owns launch, configuration,
monitoring, and reporting of the long run; DO NOT launch it from this migration
session. Its requested configuration is:

- 400,000,000 LEARNER decisions, seed 73, maps 0, 1, 2, sixteen workers.
- Checkpoint every 1,000,000 learner decisions, recording the actual step.
- Use the qualified native SET encoder.
- Default episode cap is 7200 ticks (two minutes), configurable. Preserve
  truncation counts and truncations per completed episode in telemetry.
- Sparse terminal outcome reward only: win +1, loss -1, timeout -0.5 for BOTH
  players. No kill/death shaping. Keep kill/death event telemetry as behavior
  statistics, but its reward contribution must be zero.
- Entropy coefficient starts at 0.01 and anneals to a floor near 0.0005.
  Use the pinned native anneal_ent_coef / min_ent_coef_ratio support with ratio
  0.05, and record the effective coefficient in the training metrics.

Commit 361c7bb in the null-hypothesis Rift provides the timeout-reward and
reward_outcome telemetry change as a reference. Its environment hunk still
contains kill/death shaping, so it is not sufficient for the sparse-only
experiment; its train.py hunk belongs to the old trainer and must not be used.
Keep the earlier reward unchanged during the native-versus-old comparison so
that this new reward and entropy treatment do not contaminate that comparison.
Matthew explicitly corrected the intermediate linear-run proposal: the long
run uses the NEW set encoder and starts only after handoff. Give thr_wq22hezkbf
the exact launch command/config, Rift path, commit, validation evidence, and
any remaining unverified behavior.

The previous warning that bopl-null-hypothesis.service owns the machine was
retracted by Matthew. It was inactive when checked. This migration and its
qualification retain the machine until handoff; the long experiment is now
requested ON the native trainer. Continue to allow only one heavy job at a time.


### Native metrics and evaluation ownership

The dashboard team will provide the exact metrics.jsonl/config/evaluation
contract. Write it natively or through a data-format adapter. Requested new
statistics are entropy per aim/button head, advantage standard deviation BEFORE
normalization, and per-decision evaluation value/action-probability records.
Root owns evaluator dump wiring and the Python metric-format helper; this Fable
session owns the native kernel/C counters and trainer integration. Coordinate
with thr_iiychwrq9p if a requested field is impractical; do not silently invent
missing measurements.

Do not place learner behavior counters only in PufferLib's episode Log: pinned
static_vec_log excludes unfinished-episode contributions and clears all env logs
when any episode log has n>0. Keep separate cumulative learner-only counters,
including unfinished episodes, excluding frozen players and warmup, and export
interval differences. Preserve reward/telemetry conservation and timeout counts.
