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
| libc | 13% | memmove 8.2 (obs scratch -> sample -> history -> publish; gone since 2026-09-12: the delay keeps two ~3 KB state snapshots and encodes once into the routed slot, libc is now 3.7%), memset 2.0, malloc 1.2 |
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

**Re-tune on the merged build (2026-09-12, after B's fused encoder, `d-omp`
follow-up).** `bench_shape.sh`, 150 updates, box idle (load 1-3), b4 unless
noted, `BOPL_ENV_PROF=0`:

| shape | threads | sps | learner | rollout s | eval_gpu | eval_env |
|---|---|---|---|---|---|---|
| w128 | 8 | 556-558k | 417-418k | 0.0427-0.0430 | 0.174 | 0.141 |
| w128 | 9 | 557-560k | 418-420k | 0.0425-0.0428 | 0.178-0.180 | 0.133 |
| w128 | 10 | 542-555k | 407-417k | 0.0430-0.0444 | 0.180-0.183 | 0.129-0.140 |
| w128 b8 | 8 / 10 | 531k / 539k | 398k / 404k | 0.0417 / 0.0408 | 0.179 / 0.185 | 0.121 / 0.105 (train 0.0195) |
| w256 | 8 | 611-625k | 459-469k | 0.0743-0.0766 | 0.216-0.223 | 0.338-0.361 |
| w256 | 9 | 626-634k | 469-475k | 0.0729-0.0741 | 0.223-0.227 | 0.319-0.331 |
| w256 | 10 | 627-647k | 471-485k | 0.0707-0.0739 | 0.234-0.237 | 0.292-0.316 |
| w256 | 12 | 596k (pinned 644k) | 447k | 0.0793 | 0.252 | 0.317 |
| w256 | 14 | 518k | 389k | 0.0958 | 0.265 | 0.437 |
| w256 b8 | 10 / 12 | 599k / 596k | 449k / 447k | 0.0708 / 0.0714 | 0.204 / 0.225 | 0.320 / 0.285 (train 0.038) |

Default `BOPL_ENV_THREADS` moved to CPUs - 3 (9 here); 10 for a w256 run.
More than ~10 stepping threads loses twice: c_step inflates with SMT
sharing (self-play env 12.2 us at 5 threads, 15.1 at 8, 17.9 at 10, 19.7 at
12, 21.0 at 14) and, once every hardware thread is occupied, a worker holding
a claimed env gets preempted and the buffer waits for it (straggler wait 14 us
at t8, 41 at t10, 104 at t12, 204 at t14). Pinning removes the t12 chaos
(596k -> 644k) but does not beat unpinned t10. b8 shortens the rollout 4-5%
and costs more in train (0.0155 -> 0.0195 at w128, 0.030 -> 0.038 at w256):
b4 stays. `BOPL_ENV_SPIN_US` 0/10/100, `BOPL_GPU_POLL_PAUSE` 40/200/1000,
`BOPL_GPU_WAIT=sync`, `BOPL_ENV_PIN` are all within noise or worse
(sync: eval_gpu 0.152 but eval_env 0.195, no owner steals). New knobs, both
measured and left off: `BOPL_GPU_STEAL_MARGIN_US` (stop stealing that long
before the predicted stream completion: moves 10-20 us per step from eval_gpu
to eval_env one-for-one, net -1..-2%) and `BOPL_ENV_AFFINITY=1` (stable env ->
thread slices instead of the LPT order: c_step 15.7 vs 15.1 us, -3% overall,
because an env's working set is ~130 KB and each thread touches 16-32 envs per
round, so nothing stays in a 1 MB L2 whatever the mapping; the L3 holds it all).
A dead knob was fixed (`BOPL_GPU_POLL_PAUSE` was a literal 40).

Where the CPU time goes (`BOPL_ENV_PROF=2` timers between c_step's phases,
w256 b4, us per env-step; `perf record` of the same run for the split inside
"sim"):

| phase | self-play (50% of envs) | historical (38%) | scripted evasive-v2 (12%) | at 5 threads (no SMT) |
|---|---|---|---|---|
| begin: actions, bot decide | 0.2 | 0.2 | 16.9 | 0.1 / 13.4 |
| sim: `bopl_rl_step_sampled` (4 ticks + 2 state exports) | 8.9 | 8.4 | 9.0 | 7.0 |
| sample: `compute_observations_from` + 26 KB copy | 3.2 | 3.1 | 3.4 | 2.6 |
| finish: telemetry, 2 x 26 KB history copies, masks, publish 2 x 13 KB | 2.9 | 3.1 | 3.1 | 2.5 |
| total | 15.1 | 14.8 | 32.5 | 12.2 / 25.7 |

Mean over the mix 17.2 us; the v2 play bots do no sim ticks (analytic
raycasts/prediction: `bot_shot_clear`, `bot_predict_foe`, `bot_raycast`,
`bot_surface_y`, `fmin/fmax` ~8.5% of cycles). Inside the 8.9 us "sim"
(perf shares of all cycles, 96% of which are c_step): game tick + physics
~22% (`game_tick` 6.7, `rounded_rounded` 4.2, `closest` 3.4, `Vec2::angle`
1.6, ...) = ~3.8 us; state export ~15% (`refresh_entities` 3.2,
`refresh_actor_entities` 2.6, `observe_main` 1.5, `fill_slimes` 1.1,
`step_sampled` 1.1, `write_state` 0.6, a sort 1.0, `Vec::retain` +
`from_iter` 2.5, malloc/free 1.6) = ~2.6 us; libm 8.7% (`sin` 3.1, `acos`
1.3, `fmod`/`floor` 1.9; `fmin`/`fmax` 2.9 are mostly the bots) = ~1.5 us.
`memmove` is 16.6% of all cycles and `memset` 4.4%: the observation path
copies 132 KB per env-step (memset scratch 26 KB, scratch -> sample 26 KB,
history -> scratch and sample -> history 26 KB each, publish 2 x 13 KB into
the pinned slot) plus the sim's state export. That 21% is the single largest
item and is a pointer rotation, not a copy (track F / E: encode straight into
the pinned slot or rotate scratch/sample/history pointers; skip the memset by
writing every field). `perf stat` on the trainer: IPC 1.6, L1d miss rate
9.8%, ~820 cache-misses and ~70 dTLB misses per env-step (of the ~2000 cache
lines the copy chain touches), 0 migrations.

Why eval_env cannot reach (64 x c_step) / 12 at w256: the rollout is at the
CPU roof. 32768 env-steps x 17.2 us = 0.56 CPU-s per update on ~8 effective
hardware threads = 0.070 s; measured 0.071-0.074. The 8-10 stepping threads
are >95% busy in c_step, the 4 buffers overlap their env phases (~2.4 in the
env phase at any moment), so each buffer sees ~3.3 threads: 64 x 17.2 / 3.3
= 0.33 ms, as measured. Scheduling has nothing left to give at this shape;
the levers are c_step (the 21% copy chain, the state export, libm) and the
GPU phase (eval_gpu 0.22 ms is 30% of the step and the owner threads' launch
+ poll is ~5% of their time).

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

**Done 2026-09-12 (branch `c-infer`; `native/rl/puffer/bopl_rollout.cu`,
vecenv.h/pufferlib.cu overlay in `pufferlib-bopl.patch`, native_vector.py,
binding.c).** eval_gpu per buffer step on the 32-SM partition (BOPL_ASYNC=1,
BOPL_GPU_SPLIT=auto, 300 updates, seed 73, load1 3-6):

| shape    | before (per-bank chain) | after (batched) | sps before -> after |
|----------|-------------------------|-----------------|---------------------|
| w128 b4  | 0.2105 ms (0.229 in §1) | **0.0951 ms**   | 836k -> 935k        |
| w256 b4  | 0.3177 ms (0.35 in §1)  | **0.1258 ms**   | 880k -> 982k        |
| w256 b2  | 0.3246 ms               | 0.1915 ms       | 948k -> 980k        |

The update is now training-bound at every shape (`train_s` 0.035 vs
`rollout_s` 0.018 at w128), so the sps gain is the rollout no longer
peeking out from under the trainer, not the full 2.2x of the round trip.

What a buffer step looks like now (CUPTI, w128 b4, one buffer stream,
median window 78.6 us, p10 70 / p90 91, 9 records instead of ~23 per chain
x 5 chains): one H2D memcpy of the live rows plus the packed rewards |
terminals | mask (862 KB, 19.6 us) -> `br_encode` 9.4 -> `br_gemm` (flat)
6.3 -> `br_gemm` (fusion) 4.8 -> `br_gemm` (gru) 5.7 -> `br_gate` 1.4 ->
`br_gemm` (decoder) 5.6 -> `br_sample` 6.3 -> D2H of the actions 0.3 us.
Gaps between kernels are 0.1 us (one graph); the D2H waits ~15 us behind
the other buffers' uploads on the copy engine. Stream events
(BOPL_GPU_PROF=1): h2d 1 + forward 86 + d2h 2 us of eval_gpu 98 us (the
copies are graph nodes now, so they count as "forward").

How:

- One chain per buffer over the live rows only. `native_vector.py` orders
  each buffer as [learner rows | bank 0 live rows | ... | idle rows]
  (`primary_rows` vec kwarg fixes `bank_layout[1]`, `frozen_bank_pct` the
  live rows per bank); the idle tail (48 of 112 rows at w128 b4) is never
  uploaded, computed or sampled. check_routing.py / smoke.py cover it.
- `bopl_rollout.cu`: grouped kernels take a per-row-group weight table
  (group 0 = the actor snapshot, 1.. = the frozen banks). `br_encode` (one
  row per block) stages the row in shared memory once, writes the rollout
  copy and flat input, and does the set-encoder projections/max-pools;
  it reads only the populated slots (the header carries the player /
  entity / slime counts and bopl_encode packs them from slot 0), which is
  exact and 3-4x fewer bytes. `br_gemm` is a small-M GEMM on
  `mma.sync.m16n8k8` tf32 with the weights pre-stored in B-fragment order
  (transposed once per rollout by `br_prepare`), 8 warps split K in
  64-chunks with a fixed-order shared-memory reduction (deterministic),
  spare warps take extra column subtiles; the fp32 path (BOPL_TF32=0) is
  CUDA-core FMAs, within 3e-8 of cuBLAS fp32. `br_gate` fuses the MinGRU
  gate with the terminal reset; `br_sample` is sample_logits_warp with the
  same Philox streams reading a byte mask. Falls back to the per-bank chain
  when the shape is not supported (non-set encoder, continuous actions,
  non-mingru, >4 layers, >9 groups, banks of different shape).
- Uploads: rewards | terminals | mask of the live rows are packed by the
  buffer thread into the idle tail of the buffer's own pinned observation
  rows, so one memcpy carries everything; that memcpy and the actions D2H
  are captured into the buffer's CUDA graph (one API call per step). Each
  `cudaMemcpyAsync` was ~10 us of copy-engine time under four buffers
  (h2d 45 -> 27 -> 20 us for 4 -> 2 -> 1 copies).
- Reset semantics preserved: the pack mirrors what the legacy device
  copies would hold (`static_vec_reset` zeros, `bopl_vec_reset` raises every
  terminal), so sync fp32 runs of the legacy and batched paths agree to
  ~1e-6 over 6 updates (`BOPL_ASYNC=0 BOPL_TF32=0 BOPL_BATCHED_FWD=0|1`), and
  the batched path is bit-identical across runs (fp32 and tf32).

Metric medians, updates 200-300, seed 73, w128 b4 (explained_variance /
entropy / gradient_norm): sync before 0.563 / 5.758 / 0.286, sync after
0.575 / 5.793 / 0.270; async before 0.576 / 5.805 / 0.339, async after
0.574 / 5.766 / 0.385. Within run-to-run noise.

Knobs (all read at trainer start): `BOPL_BATCHED_FWD` (1; 0 = per-bank
chain), `BOPL_TF32` (1; also selects the tf32 GEMM path here),
`BOPL_ZERO_COPY` (0 = memcpys; 1 = obs memcpy, rest read from mapped host
memory; 2 = kernels read every env buffer over PCIe: measured 112 / 127 /
135 us eval_gpu - kernels reading host memory stall the shared SMs),
`BOPL_ZERO_COPY_ACTIONS` (0; 1 writes actions straight into host memory:
saves the D2H but the sampler kernel then waits ~10 us for the PCIe write),
`BOPL_GRAPH_COPIES` (1), `BOPL_OBS_SPARSE` (1; 0 stages the whole row),
`BOPL_PACK_TAIL` (1), `BOPL_GPU_PROF=1` (stream-event split), and the
CUPTI trace (`cupti_profile.cpp`, `BOPL_CUPTI_TRACE=<file>` under
`CUDA_INJECTION64_PATH`, summarised by `cupti_timeline.py`).

Tried and dropped: zero-copy reads of the observations (27 us isolated but
45-130 us under contention: 32-byte PCIe read requests from SMs run at a
third of the DMA rate), cudaMemcpy2D of only the populated columns (no
faster than the full linear copy), clwb of the env-written lines before the
DMA (no effect), wmma with B fragments from global (4-byte lane loads),
software-pipelined K chunks in the GEMM (250 registers -> one block per
SM -> the other buffers' chains queued for 50 us).

Left for later:

- The 20 us H2D is now the biggest single item and is bandwidth: 862 KB
  per buffer step, four buffers, on one copy engine. bf16 observations
  halve it (the sim would write bf16 into the pinned rows, `br_encode`
  converts while staging, the rollout copy stays fp32 so training is
  untouched; `OBS_TENSOR_T` and the Python obs views need the dtype). A
  CPU-side compaction into the packed layout `br_encode` already decodes
  (~270 KB) would cut it 3x but costs ~15 us of buffer-thread time.
- `br_gemm` single-block cost is ~3.6 us with 8 active warps vs 1.3 us with
  2 (the empty launch is 1.4 us); ablating A loads, B loads or the mma
  changes it by <1 us each, so it is issue/sync overhead in the chunk
  loop. Worth ~8 us per step if found.
- The actions D2H waits behind the other buffers' uploads (~15 us): a
  second stream for the D2H, or a tiny "actions to host" kernel after the
  sampler instead of the copy.

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

**Done 2026-09-12, part 1 (branch `e-train`, commit `cbb0b69`).** Set
encoder forward projects only the valid slots (ballot-compacted, packed-row
fast path reads 16 of 96 slots), backward gathers only the winning slots,
flat GEMM in TF32 with the flat block padded to 948 columns, grouped Muon
Newton-Schulz (one launch per step instead of per matrix), PPO reads the
action mask in place and writes the decoder gradient directly. w128 b4
train_s 0.0388 -> 0.0283.

**Done 2026-09-12, part 2 (branch `e-train`, commits `4603c6a`..`00b9a61`;
`bopl_set_encoder.cu`, `check_set_encoder.{py,cu}`, kernels.cu / muon.cu /
pufferlib.cu overlay in `pufferlib-bopl.patch`, build.py, train.py).**
Training path only; the rollout step code is untouched. `bench_shape.sh`,
seed 73, 300 updates, medians of updates 20-300, 32/16 SM split (the box
was shared with CPU sim work: load1 2-8, so +-3% between runs):

| shape | build | sps | learner sps | rollout s | train s | wall s | load1 |
|---|---|---|---|---|---|---|---|
| w128 b4 | before (`4aed572`) | 803-819k | 603-615k | 0.0393-0.0402 | 0.0388 | 0.0400-0.0408 | 8.3 |
| w128 b4 | after | 860k | 645k | 0.0376 | 0.0217 | 0.0381 | 2.4 |
| w256 b4 | before | 875k | 656k | 0.0638 | 0.0742 | 0.0749 | 7.8 |
| w256 b4 | after | 1168k | 876k | 0.0555 | 0.0413 | 0.0561 | 3.8 |
| w256 b2 | before | 941k | 706k | 0.0662 | 0.0689 | 0.0697 | 6.9 |
| w256 b2 | after | 1126k | 844k | 0.0576 | 0.0406 | 0.0582 | 5.6 |
| w128 b4, `BOPL_TF32=0` | before | 690k | 518k | 0.0427 | 0.0468 | 0.0475 | 6.3 |
| w128 b4, `BOPL_TF32=0` | after | 796k | 597k | 0.0406 | 0.0328 | 0.0412 | 6.4 |

w128 is rollout-bound either way (train 0.0217 < rollout 0.0376, wall =
rollout); w256 was train-bound and is now rollout-bound too, +33% sps.
The goal was <= 0.020 / <= 0.035; what is left is listed under "headroom".

Learning behaviour, medians of updates 200-300 (seed 73):

| run | explained var | entropy | grad norm | policy loss | value loss |
|---|---|---|---|---|---|
| w128 b4 before | 0.586 | 5.782 | 0.347 | -0.0075 | 0.0211 |
| w128 b4 after | 0.582 | 5.804 | 0.332 | -0.0054 | 0.0213 |
| w128 b4 before, `BOPL_TF32=0` | 0.562 | 5.749 | 0.310 | -0.0080 | 0.0212 |
| w128 b4 after, `BOPL_TF32=0` | 0.564 | 5.770 | 0.379 | -0.0078 | 0.0212 |
| w256 b4 before / after | 0.537 / 0.550 | 5.751 / 5.769 | 0.448 / 0.430 | -0.0063 / -0.0058 | 0.0198 / 0.0207 |
| w256 b2 before / after | 0.531 / 0.527 | 5.742 / 5.710 | 0.445 / 0.453 | -0.0064 / -0.0052 | 0.0196 / 0.0199 |

Exactness: with every numerics knob at its old value (`BOPL_TF32=0
BOPL_GEMM_TUNE=0 BOPL_MUON_SPLITK=0 BOPL_MUON_FUSED=0`) the trainer's
metrics are bit-identical to `4aed572` over 40 updates in `BOPL_ASYNC=0`
(and async) mode, so the packed forward, the padded GEMM shapes, the PPO
shared-memory staging, the fused-gradient / mask-gather paths and the
harness changes are exact. `BOPL_MUON_FUSED=1` (part 1) is the only
default that reorders fp32 sums in exact mode.

CUPTI (`cupti_profile.cpp`, w128 b4, 50 updates, train stream, per
update; injected runs are ~10% slower than the bench):

| | launches | sum us | top kernels (us per update) |
|---|---|---|---|
| before | 943 | 40024 | bs_fused_forward 8126, flat fwd GEMM (magma fp32 SIMT) 4687, wgrad align1 3371, wgrads align4 2555, bs_fused_backward 2542, flat wgrad align1 1732, mingru_scan_backward 1649, ppo_loss 1564, muon nn GEMMs (130 launches) 1448, fusion/decoder tn 1276 + 1259, assemble_decoder_grad 939 |
| after | 325 | 23153 | bs_fused_forward_mma 3134, wgrad nt 1755, dgrad nn 1674, mingru_scan_backward 1663, flat fwd bf16 1275, mingru-in tn 1245, muon_grouped_gemm (63) 1234, flat wgrad bf16 1225, ppo_loss (smem) 1209, decoder tn 1143, dgrad nn 825, bs_fused_backward 792 |

Per 24576-row minibatch on the 16-SM partition, before -> after:
set-encoder forward 1935 -> 746 us, flat forward GEMM 1116 -> 298, flat
weight gradient 414 -> 286, set-encoder backward 605 -> 189, PPO loss 372
-> 288, Muon per optimizer step 480 -> 340, ReLU / assemble kernels
gone (-2 launches, -300 us).

What is in it:

- Tensor-core set-encoder forward (`bs_fused_forward_mma`, `BOPL_SET_MMA`,
  default follows `BOPL_TF32`): entity and slime projections as
  `mma.sync m16n8k8` TF32 with the weights as the A operand (16 channels x
  21/28 features per tile) and 16 slots as two n-tiles, max-pool over the
  slots as a 4-slot local pool plus two shuffles. Head-chunked slot staging
  (`Xs[4][16 x 21]`, chunk loop for crowded rows), slime weights
  transposed in shared memory, flat block staged with `cp.async` so no
  prefetched value can be spilled, per-row 3-warp round robin over 8
  channel tiles. On 16 SMs the forward's compute is fully exposed and
  additive to its memory time (same SM-us on 48 SMs), so this is what
  moved it: fp32 kernel 1173 -> 746 us with 10 slimes, zero-row floor 532.
- Measured GEMM kernel choice (`bopl_gemm` in kernels.cu, `BOPL_GEMM_TUNE`,
  default 1; `BOPL_GEMM_TUNE_LOG=1` prints the table): the first call of
  each (op, shape, type, epilogue) on the registered training stream times
  every cublasLt heuristic candidate (workspace split-K reductions allowed
  - fixed order, deterministic; atomics excluded; `BOPL_GEMM_SPLITK=0`
  forbids reductions) and caches the fastest; the captured graph replays
  it. Rollout streams are never touched. Worth 1.9 ms/update together
  with the fused epilogues. cuBLAS's own heuristic assumes 48 SMs: the
  k=24576 weight-gradient shapes are 10 tiles, so without split-K they ran
  2x slower than the best candidate. The training GEMMs are TF32
  compute-bound on the partition: 10.5 TF/s achieved of ~11.9 peak, which
  is why the tuner alone gains little and bf16 gains a lot.
- bf16 flat block (`BOPL_SET_FLAT_BF16`, default follows `BOPL_TF32`): the
  forward stores the 946 flat obs columns as bf16 (padded to 952 for
  16-byte rows, in the fp32-sized allocation), the flat weights are padded
  to bf16 per forward, `grad_flat` is written as bf16 by `bs_flat_grad`,
  and both 24576 x 952 x 128 GEMMs run bf16 x bf16 -> fp32 at twice the
  TF32 rate: 549 + 574 -> 298 + 286 us. Worth 2.4 ms/update.
  `check_set_encoder` has `flat_bf16` cases: float64 oracle fed the same
  bf16-rounded rows / weights / incoming gradient (forward 5e-7, gradient
  7e-5 at 1030 rows), packed == every-slot bit for bit, graph replay, both
  forward kernels, and the fp32 oracle must be further away.
- ReLU fused into the tuned GEMM epilogues (flat projection and fusion
  GEMM, `CUBLASLT_EPILOGUE_RELU`); the kernels still run when the plain
  path is taken (harness, `BOPL_GEMM_TUNE=0`).
- PPO loss: the row's masked logits are read into shared memory once
  instead of once per pass (with 64 resident warps the 2.2 KB row did not
  stay in L1, so the four passes were L2 traffic): 372 -> 288 us, bit
  identical. Rows wider than `PPO_SMEM_LOGITS` = 288 fall back.
- Muon gram products split-K (`BOPL_MUON_SPLITK`, default auto = K/64 in
  [1, 8]): the flat gram (M=128, K=946) was 4 blocks over 60 k steps;
  slices write partial tiles, `muon_grouped_reduce` sums them in slice
  order. 1922 -> 1362 us/update. First version reduced unsplit problems
  too (overwrote the entity / slime grams with stale partials): every
  kernel check passed and only the update-200-300 metric medians caught
  it (explained variance 0.28 vs 0.58). Bisect by knob before trusting a
  numerics change.
- `train.py --minibatch-div N` (default 1, unchanged): N optimizer steps
  per learner pass on 1/N of the rows. Measured at w128 b4: div 2 train_s
  0.0239, div 4 0.030 vs 0.0217 - the partition is latency-bound per
  launch, smaller minibatches lose.
- Harness: `BOPL_CHECK_SMS=<n>` runs `check_set_encoder --time` on a green
  context of n SMs (the number that matters), `BOPL_GPU_SPLIT=<n>`
  partitions even with one slot (`BOPL_ASYNC=0`) for train-only timing.

What did not help: a "lane = channel pair, weights in 42 registers" fp32
tiling (spilled the prefetched row, +275 us even on empty rows); pooling
across mma fragments with shuffles per slot (+433 us, the pool dominated
the mma itself); smaller minibatches (above); the tuner without split-K
(candidates within 5% of cuBLAS's choice); generic bf16 conversion of the
128-wide activations (conversion traffic >= the TF32 -> bf16 GEMM saving
unless the producer writes bf16). Profiling lessons: warm-up launches read
one segment (prioritized sampling on zero priorities -> duplicate indices
-> L2 hits), so first-launch timings are meaningless; the 16-SM per-block
critical path is LDS / SHFL latency chains (~2000 clocks per row
iteration even with zero valid slots), and any spill of a prefetched value
serialises the load latency.

Headroom (per 24576-row minibatch on 16 SMs, from the after trace):
set-encoder forward 746 us (floor ~530 memory, compute exposed);
`mingru_scan_backward` 396 + forward 177 (latency-bound over T=128 -
option 5); fusion / mingru-in / decoder TF32 GEMMs 200-300 each (bf16
activations from their producers would halve them, a models.cu change);
PPO 288 (probably ~150 with several rows per warp); elementwise
`bs_relu_grad` 112, `add_kernel` 93, `puff_advantage` / `ppo_var_mean` per
minibatch (GAE once per update, option 6, was not trivial with the
per-slot graphs and is worth ~0.15 ms/update); Muon 340 per step. Reaching
0.020 at w128 needs roughly three of these.

### F. Sim micro-work (~10% of a phase that is not binding, low priority)

No per-tick `Vec` allocation (`from_iter`/`retain`/malloc ~3%), cache
platform bounds for `closest` (23% of sim-only), sin/acos table (libm
6.5%). Worth ~0.1 core. Do it after A/D/E, when the sim is actually hot.

Done 2026-09-12 (RUST_ROLLOUTS.md "Speed work (2026-09-12)"): 2.35x fewer
instructions per tick, sim-only 1.9M -> 3.8M ticks/s, the observation copy
chain and per-tick allocations removed, scripted bots 3x cheaper; at w128
the environment time per step went 0.40 -> 0.30 ms and sps 223k -> 246k
on the old trainer, with the GPU round trip now the binding phase.

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
