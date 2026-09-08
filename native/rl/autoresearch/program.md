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
  update is one optimizer step on prioritized sampled rollout segments. Its
  pre-step `kl` is near zero (subject to roundoff) and `clip_fraction` is zero;
  neither measures the policy change caused by that optimizer step. Additional
  passes can reveal changes from preceding steps.
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
- `check_recurrent.py`, `puffer/test_native_selfplay.py`, and the research unit tests
  (`python -m unittest discover -s native/rl/research`).
  `libbopl_research.so` needs `make -B -C native/rl research` to pick up
  header changes. Unity player launches need
  `OPENSSL_CONF=/home/mte/Documents/Coding/bopl-re/openssl-unity-sha1.cnf`.
- Watching a run without touching it: `python native/rl/autoresearch/watch_run.py
  <run> --every 5000000` prints milestone rows (entropy against the uniform
  8.318 nats, schedule coefficient, explained variance, gradient norm, kl/clip)
  and pooled evaluation win rates, reading only the run directory. Add
  `--drift` to also measure, from the saved checkpoints (CPU numpy), whether
  each weight slice shows persistent motion or motion consistent with a random
  walk (notes.md 2026-09-07 21:50: at 19M the critic row moved consistently,
  the policy heads showed little interval-to-interval persistence; gradient
  sources are not identified by this diagnostic).

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

Ownership while root's Rift is active: root owns `train.py`, `native_selfplay.py`,
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
  against ~170 ms of rollout per update. Measured on the 786k bench workload
  (results.tsv `native-linear-passes3`, medians of 3): 13,510 vs 13,756
  learner decisions/s wall, train 2.6 vs 0.9 s of 58 s, about 2% slower. At
  that budget the three-pass runs' `kl` stayed ~1e-7 and `clip_fraction` 0,
  so the extra passes made no visible policy change there; whether they do
  at 10M+ is the question. This is the first matched comparison to run
  once the GPU is free: `strength_check.py --compare learner-passes`
  (candidate `passes3`, reference `passes1`, both arms SET, same seeds and
  budget, fail-closed reuse checks `learner_passes` per arm, report carries
  per-run drift). Specific
  hypothesis (notes.md 2026-09-07 22:08): the sampler draws 24 segments with
  replacement. With uniform learner priorities, expected unique coverage is
  ~64% for one pass and ~95% for three fixed-probability passes; actual
  coverage depends on priorities. Judge the A/B with `watch_run.py --drift`
  on both arms as well as evaluations. Post-run alternative: sample
  without replacement over eligible learner rows for full coverage at one
  pass, with revised importance weighting while preserving frozen-row exclusion.
  This changes the sampling objective and needs correctness and timing checks.
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
- **Evaluation side asymmetry** (notes.md 2026-09-07 22:26). Across the examined
  evaluations, pooled side-0 / left-spawn wins are ~.54 vs ~.36 for side 1.
  Side 0 has more wins in each evaluated checkpoint, including the first,
  with larger pooled gaps on maps 1, 2 and 4. Two follow-ups when
  game jobs are allowed: (a) evaluate the initial checkpoint against itself on
  both sides (no GPU; shows whether the effect survives identical weights),
  (b) done in `watch_run.py` (3413a3d: pooled and per-map left/right win
  rate under every evaluation line, self-play excluded); the dashboard
  summary is root's. On the stocks-0 run the gap is map 1 only.
- **Opponent pool age and diversity** (notes.md 2026-09-07 23:41). The four
  banks only ever hold the last four checkpoints (0.1 to 5M physical behind
  the learner, a bank retires as soon as a newer snapshot is pending), and
  half of matches are self-play (`historical_every=2`). The learner's
  training-time win rate against the pool stayed at .50 to .51 for the whole
  46.5M stopped run while evaluations against the initial checkpoint reached
  .58 to .67, so training outcomes carry no strength signal and the sparse
  reward has mean ~0 by construction. Root owns `train.py` and
  `native_pool.py`; the lead is a matched A/B in the `strength_check.py`
  protocol: (a) one bank pinned to the initial checkpoint (a fixed
  reference visible in `env.historical_wins`), (b) log-spaced pool ages
  instead of four consecutive checkpoints. Parent decision (2026-09-07
  23:47 EDT): no separate A/B on the old pool; the self-play thread
  (thr_rfgz498pxv) is replacing `SnapshotPool` with the PufferLib Elo pool,
  and the parent forwarded three requirements to it: retain old opponents
  rather than a rolling window, pin the initial checkpoint as a permanent
  anchor, and log member ages, Elo and win rates to `pool.jsonl`.
  `watch_run.py`'s per-milestone `pool win` column reads the env log's
  `historical_episodes`/`historical_wins`; the self-play Rift's working
  tree (read 23:50 EDT) keeps both fields with the same meaning, so the
  column should survive the swap. Re-checked on the self-play smoke run
  (notes.md 2026-09-08 00:27): the column survived, and `watch_run.py` now
  also prints learner Elo, pool listed/distinct snapshots, mean opponent lag
  and swaps per milestone from the `pool` record. Open: `pufferlib.selfplay`
  appends one snapshot per bank that swaps in the same update (same path
  listed 2 to 4 times, each rated separately, duplicates evicted silently);
  on the 500M defaults the synchronized early swaps will hit this. Fixed by
  root on main (adb6f50: `native_selfplay.step` keeps the first occurrence of
  each path), merged here 2026-09-08 00:40 (notes.md); the listed/distinct
  column should read equal on any run made after it.
- **Per-match reset cost on the critical path** (notes.md 2026-09-08 00:08).
  Rollout time on 001-s73 rises ~2 ms per player-episode ended in the update
  (~4 ms per match, ~35 % of rollout at stocks 0; stocks-3 runs 17 % faster
  for this reason alone). It is CPU, not latency: per match the game spends
  ~51 ms of main-thread time playing ~412 ticks and ~32 ms producing the
  next round's initial state (240 warmup ticks ~18 ms, scene load 14 to
  19 ms), i.e. 38 % of game CPU at stocks 0, 14 % at stocks 3. The spare
  process hides the wait but not the CPU on a saturated machine (Gate 0:
  flat 16 → 40 workers). Warmup trimming is small (game is physically ready
  by tick ~162; 240 → 170 saves ~6 % and changes the initial state). The
  lever is an in-place round reset without scene reload, plus restoring the
  post-warmup state from a snapshot instead of re-simulating it; the
  parent forwarded this to the many-worlds design (2026-09-08 00:1x EDT)
  and asked thr_nzrpteey2z to time reset ms/match separately from tick µs.
  `reset_check.c` is the acceptance test (reset after dirty play must
  reproduce the fresh seeded trajectory). Re-fitted on all six Gate 0 points
  (notes.md 2026-09-08 00:40): 3 ms per match at 16 workers, 4 to 5 ms at
  32, reset share of rollout .32 → .49 from 16 to 32 workers. Post-raycull
  per-tick budget from `exp08-profiles-raycull`: sim buckets 67 µs per
  simulated tick and flat (largest item 12.5 µs), against ~111 µs of scene
  load and the 240 warmup ticks per reset; no single bit-exact sim trim
  left is worth more than ~5 %, so "cheaper ticks" means cheaper resets and
  per-process overhead (many-worlds), not physics. One such ≤3 % trim is
  queued unbuilt on branch `autoresearch/raycull-early` (3cadbe0: bounds
  cull before `IsActive`; notes.md 00:52) for a parity check + benchmark when
  the machine is free.
- **Visibility review** before any experiment with invisibility or smoke.
  Snapshot-restore read (notes.md 2026-09-08 01:10): the post-warmup state is
  seed-independent per map and loadout; the only seed-carried state is the
  `Updater.random` stream position (two draws per ground in `GameSessionHandler.Init`),
  rebuilt by `InitSeed(seed)` plus one `RandomFix(0,1)` per ground. One snapshot
  per map at tick 240 suffices; the `Random` ability (id 1) is the exception.
  Check binary queued unbuilt on branch `autoresearch/warmup-check` (edceb65,
  `native/rl/warmup_check.c`).

## Claude Code

Run `native/rl/autoresearch/run.sh` from a dedicated Rift checkout. It loads the
locally available official Ralph Loop plugin for this session and selects
`claude-fable-5-1`. Pass `--bg` to run in the background; use `claude attach
<id>` / `claude stop <id>` to manage it. The prepared Rift has the proxy
credentials in a private, gitignored `native/rl/autoresearch/.env`. In an
interactive session, `/ralph-loop:cancel-ralph` cancels the repeat hook. Ralph
repeats when Claude finishes a turn; its iterations are not experiments. Keep
updates to Matthew brief: best verified gain, what you learned, next idea.
