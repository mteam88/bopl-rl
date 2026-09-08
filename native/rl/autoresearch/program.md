# Bopl throughput research

You are Fable. Make Bopl RL training substantially faster so we can afford
strong players across abilities, maps, combinations, and adaptive opponents.
Own the research: investigate, experiment, measure, keep useful improvements,
and continue until Matthew stops you. A failed experiment is useful evidence.

You may change any relevant implementation, architecture, runtime, or tooling.
The benchmark and checks are starting equipment, not an edit allowlist. Use
your judgment about promising directions and how much validation each needs.

This file describes the training system as it ships now (native only) and the
open research items. The migration that produced it is finished; its history
is in `notes.md`, `results.tsv` and git. Do not reintroduce the removed
implementation or its toggles: git is the backup, and the shipped tree must
stay free of the old trainer, its flags and its checkpoint format.

## The shipped design

Read `native/rl/README.md` first; it is the reference for the contract below.

- **Trainer.** `native/rl/puffer/train.py` is the only training entrypoint. It
  drives the pinned PufferLib native CUDA trainer
  (`native/artifacts/rl/dependencies/PufferLib-42f70d69…`, rebuilt through
  `native/rl/puffer/build.py --device cuda`, which regenerates the Bopl overlay
  into the pinned checkout). Rollouts, the MinGRU recurrent core, the loss,
  backward pass and Muon optimizer run in native code; Python owns
  configuration, counters, checkpoints, metrics and evaluation calls.
- **Environment.** `native/rl/puffer/bopl_puffer.h` + `binding.c` wrap the
  persistent IL2CPP game processes (`native/rl/bopl_rl.c`). Delayed
  observations (9 ticks), held inputs, four-tick decisions, match-long
  historical-opponent ownership, per-episode recurrent resets and learner-only
  updates are implemented in the binding, not in Python. The spare game
  process, threaded open and eager transport are unconditional.
- **Encoders.** `--encoder linear|set` (`BOPL_NATIVE_ENCODER`). `set` is the
  Bopl set encoder `bopl-set-mingru-v1` in `native/rl/puffer/bopl_set_encoder.cu`:
  flat block 905 floats (header + six positional players + previous action)
  through Linear+ReLU; shared Linear(21→64)+ReLU and masked max over the 96
  entity rows; shared Linear(28→64)+ReLU and masked max over the 12 slime
  rows; concat → Linear+ReLU to hidden 128 for the upstream MinGRU. Bias-free,
  hand-written backward, argmax routing, empty set pools to zero. The CPU
  forward for evaluation lives in `native/rl/puffer/policy_eval.c` and
  `native/rl/research/native_policy.py` (root-owned).
- **Rewards and schedule.** Sparse terminal outcome only: win +1, loss −1,
  timeout −0.5 for both players, zero kill/death shaping (telemetry keeps the
  events). Episode cap default 7200 ticks. Entropy coefficient decays
  exponentially in learner decisions with `--entropy-half-life` (default 30M)
  from `--entropy` .01 to the `--entropy-floor` .0005, independent of the run
  budget; the learning rate .0003 is held constant (`--anneal-lr 0`, the
  default, recorded as native `anneal_lr=0`; `--anneal-lr 1 --min-lr-ratio`
  selects the native cosine decay over the run budget). Constant matches the
  old trainer and keeps the late sparse-reward signal from being frozen. Both
  effective values are written per update. With `--learner-passes 1` each
  update is a single gradient step on the whole learner rollout, so `kl` and
  `clip_fraction` are exactly zero by construction (ratio 1); they only carry
  information at two or more passes.
- **Budgets.** `--steps`, `--checkpoint-every` and `--eval-every` count learner
  decisions; `step` and checkpoint filenames count physical decisions. With 16
  workers one update holds 3072 learner rows (24 learner rows × horizon 128).
- **Checkpoints and evaluation.** Immutable `.bin` + `.bin.json` sidecar with
  SHA256 digests; `evaluate.py` runs the real game on CPU with the same weight
  layout; replay/video/mind dumps feed Learning Lab (`serve.py`, port 8876;
  contract in `native/rl/research/METRICS_CONTRACT.md`, dashboard-owned).

## Measurement discipline

Score: `table.<arm>-w16.median_learner_dps_wall` from `bench_throughput.py`
(learner samples consumed by completed updates per process wall second,
startup included) with `median_learner_dps_active` alongside. The harness
takes `--arms name=ENV…` and pass-through trainer flags after `--`; each
invocation needs a new output directory under `native/artifacts/rl/autoresearch/`.

Current reference workload `native-w16-786432-fresh-sparse-7200-final`:

```sh
native/artifacts/rl/venv/bin/python native/rl/research/bench_throughput.py \
  --output native/artifacts/rl/autoresearch/<name> --workers 16 --repeats 3 \
  --decisions 786432 --seed 73 --arms "linear=,set=" \
  -- --game native/artifacts/rl/autoresearch/builds/game-il2cpp-final/BoplBattle.x86_64 \
     --banks 4 --horizon 128 --training-maps 0,1,2 --episode-ticks 7200 \
     --checkpoint-every 262144 --cudagraphs 1
```

(`bench_throughput.py` passes `--no-eval`; arms select `--encoder`.) Record
every measurement with `native/rl/autoresearch/record.py <summary.json>
<experiment> <workload> <status> "<desc>" [--table <arm>-w16]`, which appends
to `results.tsv` (columns `timestamp_utc, commit, experiment, workload,
learner_dps_wall, status, description, learner_dps_active, wall_seconds`;
status `baseline|keep|discard|invalid`). Give changed workloads distinct names.
Run perf measurements one at a time, alternate incumbent and candidate, and
remeasure promising changes over longer workloads. Keep `notes.md` current with
hypothesis, revision, workload, speed, memory, validation and decision.

A gain is real only if the agent still learns in the real game with the same
observation, delay, reward and reset semantics. Do not count omitted work,
missing observations or a weaker workload as a speedup. Observation-capacity
overflow (96 entities, 12 slimes) must stay visible and fatal.

Correctness checks to rerun after touching the environment, binding or
encoder (evidence layout: `native/artifacts/rl/autoresearch/final-core-checks/`):

- `make -C native/rl all` (binaries land in `native/artifacts/rl/`) then
  `native/artifacts/rl/sampled_check` and `native/artifacts/rl/reset_check`
  (sampled-step parity, reset isolation);
- `native/rl/puffer/smoke.py` linear / set / set16 and `check_routing.py`
  (bank routing, historical ownership, per-episode resets);
- `check_set_encoder.py` (CUDA forward/backward vs an independent oracle,
  masking, permutation, empty sets, graph replay, finite differences);
- `check_recurrent.py`, `test_native_pool.py`, and the research unit tests
  (`python -m unittest discover -s native/rl/research`).
  `libbopl_research.so` needs `make -B -C native/rl research` to pick up
  header changes. Unity player launches need
  `OPENSSL_CONF=/home/mte/Documents/Coding/bopl-re/openssl-unity-sha1.cnf`.
- Watching a run without touching it: `python native/rl/autoresearch/watch_run.py
  <run> --every 5000000` prints milestone rows (entropy against the uniform
  8.318 nats, schedule coefficient, explained variance, gradient norm, kl/clip)
  and pooled evaluation win rates, reading only the run directory.

## Where things stand (2026-09-08)

1. **IL2CPP player: done.** +21% learner throughput over Mono on the old
   sustained workload, exact gameplay parity. The qualified build is
   `Builds/LinuxIL2CPP/BoplBattle.x86_64` (player sha256
   0bfe51325720c950b4ad75ce7c3b65595fd7e939908520ec33600919919327a8), copied to
   `native/artifacts/rl/autoresearch/builds/game-il2cpp-final/`.
2. **Native trainer: done and qualified.** Linear and SET encoders validated
   on the final build (all checks above pass). Final 16-worker throughput,
   medians of three interleaved runs: linear 14957 active / 14004 wall, SET
   14616 / 13683 learner decisions per second (SET costs 2.3%). Learning at
   786k random-init decisions is nil for both arms, as expected for that budget.
3. **Handoff: written.** `native/rl/NATIVE_HANDOFF.md` (root-owned) carries the
   400M launch configuration for thread thr_wq22hezkbf, the qualification
   evidence and the limits. This session does NOT launch that run.
4. **Standalone simulator: assessed, not started** (notes.md, 2026-09-08). Collection is ~98 % of logged training-loop time. A training-loadout simulator (dash/bow/platform, two players) in `bopl-sim` has a speculative 5–10× throughput upside, without demonstrated mechanic parity. Defer it until the 400M run helps establish whether collection rate limits learning.

Ownership while root's Rift is active: root owns `train.py`, `native_pool.py`,
`native_vector.py`, `policy_eval.c`, `native_policy.py`, `evaluate.py`,
`native_metrics.py`, `environment.py`, dashboard files, `series.py` and
`NATIVE_HANDOFF.md`. This Rift owns `bopl_rl.c`, `bopl_puffer.h`, `binding.c`,
`eval_binding.c`, `bopl_set_encoder.cu`, `smoke.py`, the C checks, the
autoresearch harness and this program. Send corrections to root through
`bb thread tell thr_6bmdfjijez --mode queue`; do not overwrite root files.
One heavy job (GPU training, benchmark, Unity build) at a time on this box;
the slot is negotiated explicitly between the Rifts.

## Open research items

- **SET vs linear strength (optional post-launch A/B).** Matthew chose SET
  without a matched strength gate. `native/rl/autoresearch/strength_check.py`
  is the fail-closed protocol (fresh matched runs per seed, final evaluation,
  side-swapped round robin with a random anchor, Bradley-Terry Elo, pooled
  Wilson score; verdict rule fixed before the run). It passes the entropy
  half-life explicitly and rejects cached runs trained under another
  schedule. Run it only when the GPU is free of the 400M run, or on a second
  machine.
- **Learner passes (post-launch A/B, top lead).** The 400M run does one
  gradient step per 3072-decision rollout (`--learner-passes 1`); the old
  trainer did 12 (3 epochs × 4 minibatches). Train time is ~4.4 ms per pass
  against ~170 ms of rollout per update, so `--learner-passes 3` costs under
  5% of wall and turns PPO's clip and `kl` back into live quantities. If the
  400M run shows entropy stuck near uniform, this is the first matched
  comparison to run (same seed, same budget, `strength_check.py`-style
  fail-closed protocol with `--learner-passes` in the plan).
- **Throughput.** Collection is ~98% of wall time at 16 workers (rollout 44 s
  of 56 s; train ~1 s). Leads: game stepping cost inside the IL2CPP player,
  worker count vs GPU utilization (34–38%), observation packing, and the
  synchronous evaluation cadence (root: 5M cadence, 57 games per suite).
- **Standalone simulator.** Assessed in notes.md (2026-09-08): worth it only
  once the 400M run shows more decisions per hour would change the outcome, and
  then scoped to the training loadout, mechanic by mechanic with Unity goldens,
  arrows first. No rewrite now.
- **Native scripted evaluation opponents** (archer/pursuer/evasive) as fixed
  anchors after launch; the ladder grows a scripted-anchor option then.
- **Visibility review** before any experiment with invisibility or smoke.

## Claude Code

Run `native/rl/autoresearch/run.sh` from a dedicated Rift checkout. It loads the
locally available official Ralph Loop plugin for this session and selects
`claude-fable-5-1`. Pass `--bg` to run in the background; use `claude attach
<id>` / `claude stop <id>` to manage it. The prepared Rift has the proxy
credentials in a private, gitignored `native/rl/autoresearch/.env`. In an
interactive session, `/ralph-loop:cancel-ralph` cancels the repeat hook. Ralph
repeats when Claude finishes a turn; its iterations are not experiments. Keep
updates to Matthew brief: best verified gain, what you learned, next idea.
