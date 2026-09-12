# Trainer throughput: where the time goes and how to get 2-3x

Written 2026-09-11 against the live `overnight-4` sweep (trial 000, workers
128, buffers 4, banks 4, hidden 128, horizon 128, learner_passes 4, Rust sim
d0a170e5, PufferLib 42f70d6 + overlay). Nothing here has been implemented;
this is the measured bottleneck model and the ranked options. Fidelity
constraint per Matthew: a close approximation is fine (bit-exactness with the
.NET oracle is not required for training speed work).

## 1. Measured state

Per update (448 agent rows x 128 steps = 57,344 rows; 256 physical players
= 32,768 physical decisions; 192 learner rows = 24,576 learner samples):

| phase | wall | share | notes |
|---|---|---|---|
| rollout | 0.086 s | 61% | 4 buffer threads in lockstep, 128 steps each |
| train | 0.055 s | 39% | main thread, strictly after rollout, GPU only |
| total | 0.140 s | | 232k physical sps, 174k learner sps |

Per buffer-step (32 envs, 112 rows per buffer, 4 buffers concurrent):

| piece | wall | what it is |
|---|---|---|
| eval_gpu | 0.27 ms | cudagraph (learner + 4 bank forwards, ~85 kernels) + D2H actions + `cudaStreamSynchronize` + previous H2D obs (1.48 MB) |
| eval_env | 0.47 ms | `#pragma omp parallel for num_threads(32)` over 32 `c_step` (4 sim ticks x 2 players + obs encode) |
| wall | 0.77 ms | the two are serial within a buffer |

CPU: the process uses ~410% of 1200% (65% of the box idle). 144 threads: 4
buffer threads at ~30%, main at ~29%, ~124 OMP workers at ~5% each. 17% of
CPU time is system time (futex/wake). `perf` user cycles, 10 s:

| where | share | detail |
|---|---|---|
| libcuda | 35% | spin loop inside `cuStreamSynchronize` (4 buffer threads) |
| libbopl_sim | 32% | game_tick 6.5, rounded_rounded 3.8, refresh_entities 2.8, closest 2.5, refresh_actor_entities 1.9, Vec2::angle 1.4, Vec alloc/retain/malloc ~3 |
| libc | 13% | memmove 8.2 (obs scratch -> sample -> history -> publish), memset 2.0, malloc 1.2 |
| `_C` env | 8% | compute_observations_from 5.2 |
| libm | 6.5% | sin 2.7, acos 0.9, fmod |
| vdso | 3.9% | clock_gettime (native_perf timers) |

Sim cost check: 65,536 ticks per update; sim-only bench is 2.2M ticks/s =
30 ms CPU. Measured sim CPU in the trainer is ~110 ms per update (32% of 4.1
cores x 0.086 s... including train phase, roughly 3-4x the bench). The gap
is cold caches and SMT contention from 128 OMP threads on 12 hardware
threads, not sim work. Even so, sim is ~1 core. The trainer is not
compute-bound anywhere; it is latency/synchronisation-bound:

1. Lockstep round trip: each buffer waits GPU forward -> D2H -> sync -> CPU
   step -> H2D, 128 times per update. GPU forward of a tiny model over 112
   rows is launch-latency bound (~85 kernels, ~2 us each) and does not get
   cheaper with fewer rows or more expensive with more rows.
2. OMP oversubscription: vecenv.h computes `num_workers = num_threads /
   buffers` = 128/4 = 32 and puts it in the `num_threads(32)` clause, which
   overrides `BOPL_OMP_THREADS=12`. 4 buffers x 32 = 128 OMP threads on 12
   hardware threads; every step is a fork/join of 32 threads for 32 envs of
   ~40 us each. `eval_env` = 0.47 ms for ~1.3 ms of real CPU work that 3
   threads would finish in ~0.45 ms; with 12 threads and no fork/join tax it
   is ~0.15 ms.
3. Spin sync: `cuStreamSynchronize` spins (default `cudaDeviceScheduleAuto`
   with 4 threads > cores picks spin). 4 buffer threads x 35% of samples.
   This steals the CPU the OMP workers want.
4. Train never overlaps rollout (39% of wall). The GPU idles during
   `eval_env` and the CPU idles during train.
5. Bandwidth in train: set-encoder entity activations are fp32
   `B x 96 x 64` = 604 MB per 24,576-row minibatch, forward + backward ~2.4
   GB traffic; ~7 minibatches per update (4 passes x 1.71 replay) = ~17 GB
   = ~25 ms of the 55 ms at the 5070's ~670 GB/s. `select_copy` gathers 324
   MB of obs per minibatch (another ~3 ms). `puff_advantage` recomputes GAE
   over the whole batch per minibatch (small, but serial).
6. 43% of inference rows are idle bank slots (3 of 4 bank rows per
   historical env are unused each episode) — 4 extra bank forwards run
   every step for rows whose actions are discarded.

Memory is not a constraint: trainer RSS 240 MB with 128 Rust worlds; GPU
5.8 of 12 GB.

## 2. How other people get millions of steps/s

- PufferLib Ocean (1-10M sps on one core, 20M claimed for 4.0): observations
  are tens of bytes (uint8), thousands of agents per buffer, C envs with no
  allocation, tiny MLPs. The lockstep design is identical to ours; it wins
  because each round trip amortises over thousands of decisions, not 64,
  and the H2D is kilobytes, not 1.5 MB. Our per-step round trip (~0.3 ms) is
  in the same range as theirs; we just carry 20-50x fewer decisions per
  trip and 100x more bytes.
- PufferLib 5.0 (upstream default is still 4.0 = our pin; 5.0 is active,
  last commit 2026-09-11, pure C, no Python). Three things relevant to us:
  `async` (Cleanba two-slot: collect epoch k+1 while training epoch k, with
  a snapshot of actor weights; one-epoch policy lag, V-trace available),
  `PUF_GPU` env backend (env written in CUDA, whole horizon net+env captured
  as one cudagraph, bf16 obs, no host at all), and `Agent.policy` per-agent
  policy routing for self-play (still layout-contiguous per buffer, like
  our bank rows). Its CPU worker loop is byte-for-byte the same lockstep as
  4.0 (forward -> D2H -> sync -> `omp parallel for num_threads` -> H2D), so
  5.0 alone does not fix items 1-3 above. Porting means re-porting the
  Python driver, selfplay pool, sweep runner (5.0 has Protein in C).
- GPU-resident sims (Madrona, Jax2D/Kinetix, MJWarp, Isaac, WarpDrive):
  millions of sps on one GPU because there is no host round trip; step and
  policy live on the device. Kinetix/Jax2D is the closest analogue (2D
  Box2D-style rigid bodies, heterogeneous scenes via vmap) but is O(n^2) in
  entities and, more to the point, a GPU port of Bopl's game logic (Unity
  physics, abilities, level scripts, ~74k-tick oracle parity) is a rewrite
  of the sim, not a port. Estimate: weeks to months, and every future sim
  change has to be made twice unless the CPU sim is abandoned.
- Async actor/learner (IMPALA, Sample Factory, Cleanba, EnvPool async):
  overlap collection with training and let envs run without waiting for
  the slowest one. Sample Factory gets 100k+ sps on VizDoom on a desktop by
  double-buffered sampling: two env groups per worker so the CPU steps one
  while the GPU infers the other. That is the pattern that fits our shape.

So: "millions" is a GPU-resident sim or a 64-core box, not this box with a
13 KB observation. 2-3x is reachable inside the current architecture.

## 3. Options, ranked by gain per effort

Gains are per-lever estimates against the live numbers; they compound
roughly multiplicatively until the next bottleneck binds.

### A. Overlap train with rollout (1.5-1.6x, medium effort, tiny fidelity cost)

Wall becomes max(rollout, train) = 0.086 s instead of 0.140 s: ~380k
physical / 285k learner sps at today's per-step costs. Two ways:

- Backport 5.0's `async`: two rollout slots (2x rollout memory, ~1.5 GB
  fine), snapshot of learner weights for the actor, `train_stream`
  non-blocking, buffer threads start slot k+1 while main trains slot k.
  Policy lag = one epoch; `vtrace` is already in our trainer (off in
  overnight-4) and corrects it. Cleanba reports no learning loss at lag 1.
- Or cheaper: keep sync semantics but start the next rollout the moment
  `train_impl` has launched its graphs (they are async on the GPU anyway);
  needs the actor to read a weight copy, which is the same snapshot work.

GPU contention: train kernels compete with the ~85 tiny inference kernels;
inference has priority via stream priorities. Expect rollout to slow
5-10%, still a big net win.

**Done 2026-09-12 (branch `a-async`; pufferlib.cu/bindings.cu/vecenv.h overlay
in `pufferlib-bopl.patch`, train.py, build.py).** `bench_shape.sh` at the
canonical shape on top of the merged D+B trees (seed 73, 300 updates, the box
shared with CPU sim work, so +-3% between runs):

| build | workers | sps | learner sps | rollout s | train s | wall/update s | eval_gpu ms | eval_env ms |
|---|---|---|---|---|---|---|---|---|
| merged D+B, serial (`BOPL_ASYNC=0`, bit-identical) | 128 | 544-547k | 408-410k | 0.0441 | 0.0156 | 0.0603 | 0.172 | 0.149 |
| async, shared GPU (`BOPL_GPU_SPLIT=0`) | 128 | 657-667k | 493-500k | 0.0486-0.0494 | 0.0276 | 0.0491-0.0499 | 0.222 | 0.138-0.146 |
| async, 32/16 SM split (default `auto`) | 128 | 692-698k | 519-523k | 0.0464-0.0469 | 0.0385 | 0.0470-0.0474 | 0.198-0.210 | 0.135-0.137 |
| async, 40/8 split | 128 | 478k | 359k | 0.0443 | 0.0679 | 0.0685 | 0.185 | 0.143 |
| async, 8/40 and 16/32 splits | 128 | 330k / 530k | | 0.099 / 0.061 | 0.022 / 0.024 | | 0.62 / 0.34 | |
| serial, w256 | 256 | 610-622k | 458-467k | 0.0765 | 0.0300 | 0.1074 | 0.217 | 0.363 |
| async, shared GPU, w256 | 256 | 761k | 571k | 0.0852 | 0.0529 | 0.0861 | 0.306 | 0.343 |
| async, 32/16 split, w256 | 256 | 845k | 634k | 0.0769 | 0.0744 | 0.0776 | 0.277 | 0.315 |
| async, 24/24 split, w256 | 256 | 797k | 598k | 0.0816 | 0.0584 | 0.0822 | 0.343 | 0.283 |

wall/update is the new `wall_seconds` in metrics.jsonl (what sps divides by;
with overlap rollout and train no longer sum to it). w128: 0.0603 -> 0.0471
(+27%), of which the ideal is 0.0441 (rollout alone on an idle GPU, 735k);
w256: 0.1074 -> 0.0776 (+36%), now bound by rollout and train equally.

What is in it (the 5.0 Cleanba design, per PufferLib's pufferl.cu):
two rollout slots (`rollouts`, `rollout_start_states`, per-(slot,t,buffer)
inference graphs, one train graph per slot because the set encoder bakes the
slot's observation pointer into the graph); a second `PolicyWeights` in its
own `Allocator` that the actor plays, refreshed by one flat D2D copy of the
parameter buffer when a rollout starts (same registration order, so the
copy is a snapshot); a non-blocking low-priority `train_stream` and
high-priority per-buffer and bank streams; `rollouts_start`/`rollouts_finish`
split at the C level (`_C.rollouts` = both) with the env pool's
`static_vec_omp_start/wait` split to match; the trainer thread parks on a
`cudaEventBlockingSync` event instead of spinning (`BOPL_TRAIN_WAIT=block`,
spin is within noise). `train` advances `global_step` (data trained), and
train.py keeps `counts` = trained-through and `ready` = the slot collected
for the next update, so every update still accounts for exactly one batch
(the per-update learner_steps check holds), a batch collected but not
trained at SIGTERM is dropped, `--resume` restarts from trained data, and
env logs/counters/bank loads/`load_weights` only run between rollouts (the
bindings throw if a rollout is in flight). Async is deterministic per seed:
three runs gave identical losses.

Learning: medians over updates 200-300, w128: serial ev 0.568 / entropy
5.780 / grad-norm 0.279, async 0.577 / 5.800 / 0.334 (identical for every
split); w256: 0.541 / 5.774 / 0.395 vs 0.538 / 5.751 / 0.456. One epoch of
policy lag; the stored logprobs are the actor snapshot's, so PPO ratios are
exact and the 4.0 advantage kernel's V-trace correction (rho/c clip 1.0 from
default.ini, always on) sees the lag as part of the ratio. Checked: serial
mode is bit-identical to the merged tree over all 300 updates,
check_routing, smoke (`--encoder set`, async and serial), a 40-update run
with eval + checkpoints every 250k then `--resume` to 60 and a SIGTERM run
resumed to completion (accounting exact in both), test_native_selfplay.

GPU contention is the whole story of the remaining gap. Shared GPU: the
inference graphs (~85 kernels of a few blocks each per step) queue behind
the resident training blocks even at highest stream priority (priority only
orders pending blocks), eval_gpu 0.172 -> 0.222 ms/step and train 0.0156 ->
0.0276 s. It is not the CPU: `train_s` is unchanged with 6 instead of 8 env
threads (0.0274 vs 0.0279 shared, 0.0385 vs 0.0385 split; the box has 12
threads, the pool uses 8 and the trainer is one `cudaGraphLaunch` + an event
wait per update). The fix that worked is spatial: CUDA green contexts
(`cuDevSmResourceSplitByCount`, driver API, `-l:libcuda.so.1`) put the
rollout streams in a partition of n SMs and the train stream in the rest;
`cudaGraphLaunch`, cuBLAS and events all work on the green streams and the
memory is shared. The 5070's split granularity is 8 SMs, so the choices are
32/16 (best: inference +20% latency over idle, training 2.5x slower but
hidden under the rollout up to w256) and 40/8 (inference at idle speed,
training 4.4x slower, becomes the binding phase). Inference wants all 48
SMs: 8/16/24 SMs give 0.62/0.34/0.25 ms eval_gpu.

Did not help: stream priorities alone (that is the "shared" row); fewer env
threads to free a core for the trainer (only slows eval_env); spin instead
of blocking wait for the train stream.

Headroom: at w128 the floor for this design is the idle rollout, 0.0441 s
(735k), 6% away, and the 6% is inference on 32 instead of 48 SMs. Beyond
that the update is rollout-bound and the levers are C/E/G again (a cheaper
inference step, or more agents per round trip); at w256 training on 16 SMs
is already as long as the rollout, so 3.B-style train work (or a finer SM
split on a bigger GPU) is needed before larger shapes pay off.

### D. Fix the CPU side of the round trip (eval_env 0.47 -> ~0.15 ms, ~1.4x on rollout, low effort)

- Stop the `num_threads(32)` override: size per-buffer workers to
  `hw_threads / buffers` (3) or run one shared pool. Trivial patch in
  vecenv.h overlay.
- Replace fork/join per step with persistent per-buffer workers spinning
  briefly on an atomic then futex (the 5.0 buffer threads already do
  atomic handshakes; extend to env workers). Kills the 17% system time.
- Make `cuStreamSynchronize` not spin: `cudaSetDeviceFlags(
  cudaDeviceScheduleBlockingSync)` or `Yield`, or poll a `cudaEvent` with
  a short spin then `sched_yield`. Frees ~1.4 cores for the sim.
- Pin the buffer threads and workers; disable the `clock_gettime` per-step
  timers in production (3.9%).

**Done 2026-09-12 (branch `d-omp`, vecenv.h overlay in `pufferlib-bopl.patch`).**
`bench_shape.sh` at the canonical shape (w128 b4 p4, 300 updates, quiet box),
same seed, every loss/metric bit-identical to the baseline row (the env RNG is
per env, so scheduling cannot change results):

| build | sps | learner sps | rollout s | train s | eval_gpu ms | eval_env ms |
|---|---|---|---|---|---|---|
| baseline (OMP 32 threads x 4 buffers, spin sync) | 227k | 170k | 0.0893 | 0.0545 | 0.302 | 0.364 |
| (a) only: OMP team = 12 or 24 threads total, passive | 250-257k | 188-193k | 0.072-0.076 | 0.0545 | 0.29-0.33 | 0.240 |
| env pool, `BOPL_ENV_THREADS=8`, LPT, steal, poll | 292-297k | 219-223k | 0.0555 | 0.0545 | 0.29 | 0.12 |

What is in the pool (all knobs documented in train.py): persistent workers
claiming envs off a per-buffer atomic counter (dynamic, no static partition);
buffer threads step envs too, and step other buffers' envs while their own
forward runs; idle workers spin 100 us then park on a futex; the trainer <->
buffer handshake is a futex (the 20/50 us nanosleep polls are gone);
`cudaStreamSynchronize` is replaced by a `cudaStreamQuery` poll the buffer
thread fills with env steps; each step's claim order is longest-last-step
first (`BOPL_ENV_LPT`), which cut the per-step straggler wait from ~65 us to
~13 us (the scripted-bot envs run lookahead rollouts and are consistently the
slow ones). Per c_step: 17 us at 8 stepping threads, 21 us at 12 (SMT
contention), so the box is close to CPU-bound on the sim: 0.55 ms of
thread-time per buffer-step x 4 buffers per 0.43 ms step = ~5 hardware
threads in `c_step` continuously.

Measured and not adopted: `cudaDeviceScheduleBlockingSync` (eval_env 0.14 but
eval_gpu +45 us wake latency, net -2%); `Yield` (no change); pinning (within
noise); one or two "driver" threads polling several streams each (serialises
launches, +10-20 us on eval_gpu, net -2..-5%); more threads (12-16: SMT
inflates c_step, futex wakes rise); longer idle spins (400-1000 us: worse at
12 threads, neutral at 10); buffer threads stealing the fastest envs from the
tail of the LPT order (eval_gpu -5 us, stragglers +5-20 us, neutral); 2
buffers (275k: E doubles, G unchanged); 8 buffers (rollout 0.0546 = same as
b4, eval_env 0.09, but `train` goes 0.054 -> 0.30 s at 8 buffers, a
train-side scaling bug independent of this work; also 8 buffers x 4 streams
overflowed the 32 cuBLAS handles per thread, raised to 64).

Where the floor is now: per buffer-step = G 0.29 ms (graph launch is 9-10 us
of CPU; the rest is GPU execution + D2H + PCIe, flat across b2/b4/b8, so the
GPU is latency-bound with spare capacity) + E 0.12 ms (32 x 17 us over ~5
effective threads + 13 us straggler). Double-buffered halves per buffer (3.D
step e) is equivalent to b8 without the extra pollers and would buy <= 5%;
not done. The next CPU-side lever is `c_step` itself (memmove 9%, obs encode
4%, bots 5%, libm 6% of cycles), i.e. E/F, and G is C.

### C. Shrink the GPU round trip (eval_gpu 0.27 -> ~0.15 ms, ~1.2x on rollout, medium effort)

- Fuse the 5 forwards (learner + 4 banks) into one batched forward: same
  arch, different weights, so a grouped/batched GEMM per layer with a
  per-row weight index. ~85 kernels -> ~20. The minGRU step and set encoder
  are already fused kernels; this is the layer loop x 5.
- Skip idle bank rows: only 1 of 4 bank rows per historical env is live.
  Today the row permutation pins banks to fixed slots; either gather live
  rows before the forward or accept the 43% wasted rows (they cost little
  in a latency-bound regime). Low priority unless we move to bigger batches.
- Send obs as bf16 (or int8 for the mostly-categorical parts) over PCIe:
  halves the 1.48 MB H2D per buffer-step (~60-70 us of the 270). 5.0 does
  exactly this (`obs_t = bf16` for GPU envs; `cast` on the way into the
  fp32 rollout tensor).

### E. Compact observation (1.2-1.5x across the board, medium effort, small fidelity cost)

3298 floats/agent: 96 entity slots x 21 = 2016 of them. Cutting to 32-48
nearest entities (already nearest-first in `compute_observations_from`)
halves obs bytes, memmove, H2D, `select_copy`, and the set-encoder
activations (the 25 ms in train). This is the single lever that helps
every phase. It changes what the policy sees only when >32-48 entities
are in range, which is rare on the three maps (worth measuring: dump the
entity-count histogram from a few thousand steps). Also drop the double
full-state write per decision (phase-3 sample + final) to one memcpy.

### B. Train-side fixes (0.055 -> ~0.03 s, needed once A makes train the binding phase)

- bf16 entity activations in `bopl_set_encoder.cu` (store and reload as
  bf16, accumulate fp32): halves the ~17 GB of activation traffic per
  update. Combined with E it is a quarter.
- Compute GAE once per update, not once per minibatch (`puff_advantage` is
  full-batch inside the minibatch loop).
- minGRU scan: one thread per (b,h) looping T=128 is fine for 24576 x 128
  = 3.1M threads; not the bottleneck. Leave.
- Fewer passes: 4 learner passes x 1.71 replay = ~7 gradient steps per
  update. Puffer's defaults are 1-2. Halving passes halves train at a
  sample-efficiency cost that the sweep should decide, not us.

### F. Sim micro-work (~10% of a phase that is not binding, low priority)

No per-tick `Vec` allocation (`from_iter`/`retain`/malloc ~3%), cache
platform bounds for `closest` (23% of sim-only), sin/acos table (libm
6.5%). Worth ~0.1 core. Do it after A/D/E, when the sim is actually hot.

### G. More agents per round trip (the Puffer way; 1.5-2x on top, once D lands)

The round trip is ~0.4 ms regardless of rows. After D frees the CPU, run
256-512 envs (2-4x more rows per buffer-step) at nearly the same per-step
wall: RSS is ~1 MB/world, rollout obs grows to 1.5-3 GB on GPU (bf16 obs or
E halves that). Batch per update grows 2-4x, so gradient steps per sample
fall unless minibatch/passes are retuned; the sweep must own that. Prior
`bench_scale` showed a flat curve versus workers, but that was measured
with the OMP oversubscription in place, which is exactly what G needs D
to remove first.

### H. Tear-outs considered and not recommended now

- Move to PufferLib 5.0: buys `async` (A) and `Agent.policy`, costs a
  re-port of the Python side with no fix to items 1-3. Backport A instead;
  revisit 5.0 when it is the default branch.
- GPU-resident sim (Madrona/JAX/CUDA port of the Rust sim): the only path
  to millions; a rewrite. Not for a 2-3x goal.
- CPU policy inference (ONNX/int8 on the 6 cores): removes the GPU round
  trip but the set encoder over 96 x 64 for 112 rows x 5 policies is ~40
  MFLOP per buffer-step; at ~200 GFLOP/s achievable on 3 cores that is
  ~0.2 ms, no better than the GPU path, and it steals the cores the sim
  needs. No.

## 4. Suggested order and expected curve

| step | rollout/update | train/update | wall | physical sps | learner sps |
|---|---|---|---|---|---|
| today | 0.086 | 0.055 | 0.140 | 232k | 174k |
| D (OMP + spin fix) | ~0.055 | 0.055 | 0.110 | ~300k | ~225k |
| + A (overlap) | ~0.058 | 0.055 | ~0.060 | ~550k | ~410k |
| + B/E (bf16 acts, 48 slots) | ~0.045 | ~0.030 | ~0.047 | ~700k | ~520k |
| + G (256 envs) | ~0.070 for 2x rows | ~0.060 for 2x rows | ~0.072 | ~900k | ~680k |

The first two rows are engineering with no model change and no fidelity
change; the third changes obs width and activation precision; the fourth
changes batch size. 2-3x is D + A, possibly with the bf16 obs H2D from C.

Measurement to do before E: entity-count histogram per decision on the
three maps (cheap: count non-zero entity slots in the obs scratch for a
few thousand steps in `bench_throughput.py`, outside the sweep's cgroup).
