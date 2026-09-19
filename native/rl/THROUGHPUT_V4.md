# Throughput with policy v4: where the time goes, and the levers

Companion to `THROUGHPUT.md` (the v3 campaign, 227k -> 1.5M physical sps at
w128 b4) for the trainer running policy v4 (`POLICY_V4_PLAN.md`; encoder v4,
observation layout 4, `sweep/configs/v4-aim.toml`). Written 2026-09-18 from
code inspection and the CPU-only microbenchmarks below, before any GPU
measurement of the v4 arm: sections 2 and 3 are hypotheses with the
measurement that confirms or kills each, not results. The GPU was busy for
the day this was written; `puffer/bench_shape.sh` with
`BOPL_BENCH_ENCODER=v4` is the first thing to run when it is free.

Branch: `throughput-v4` = main `4d66b2f8` (124-map / 39-ability Rust port)
+ policy-v4-work `maps-round2` `6c19394` (v4 + per-player writer bits +
64-map trainer), the same base as the sibling `plumbing-124` branch.

## 1. What is known

### 1.1 The regression

`v4-aim/000` (the production recipe, 2026-09-16): 1.73B learner steps in
10,386 s = 167k learner sps, 0.147 s per update (24,576 learner rows). The
v3 recipe it replaces (`overnight-9/024`, same w128 b4 shape) ran at
~1.14M learner sps, 0.0216 s per update (`THROUGHPUT.md` 4, `eec504b`):
**6.8x slower per update**. `train.py`'s note on `BOPL_GPU_SPLIT` gives the
one per-phase measurement that exists, for v4-lite at w128 b4: train
0.090-0.107 s, rollout 0.114-0.120 s per update, shared GPU 5% better than
the 24/24 green-context split. Both phases are 5-6x v3 (rollout 0.0184,
train 0.0210); with `BOPL_ASYNC` they overlap, so the wall is the larger
one: the rollout, with train close behind. **Any rollout-only campaign is
capped at ~0.09 s per update = ~270k learner sps (1.6x)** until the train
pass moves too.

### 1.2 Observation writer and sim, measured (CPU, this branch)

`writer_bench` (`make -C native/rl writer-bench`; `native/rl/writer_bench.c`),
core 11 pinned, nice 19, Q32 sim, variants interleaved per repeat, 4,096
recorded states x 2 players per variant, 5 repeats, two runs per loadout
(seeds 38, 39). Load average 5.6-7.4 throughout (a training job owned the
box): the spread between runs is that load, and absolute numbers are
10-20% pessimistic against an idle box.

| per row, us (median of 5, both runs) | default loadout 2/4/14 | random loadouts |
|---|---|---|
| v3 writer (layout 3) | 0.85 / 1.02 | 1.13 / 1.02 |
| v4-aim (layout 4, writer bits 8: production) | **10.03 / 10.49** | **10.88 / 10.70** |
| v4 aim-only (bits 10: aim/LOS casts, no fan, no terrain) | 2.55 / 2.92 | 2.48 / 2.71 |
| v4 no-rays (bits 2: no geometry channel) | 2.00 / 2.05 | 2.51 / 1.84 |
| privileged block (critic, once more per step) | 0.09 / 0.10 | 0.10 / 0.07 |
| = geometry (fan + terrain block) | 7.5 / 7.6 | 8.4 / 8.0 |
| = aim + line-of-sight casts | 0.5 / 0.9 | 0.0 / 0.9 |
| = v4 rows/relational columns over v3 | 1.1 / 1.0 | 1.4 / 0.8 |

`bopl4_relation` alone: 45-62 ns per call plain, 160-220 ns ballistic (30
arc samples); ~14 calls per row = ~1 us, the whole "v4 rows over v3" line.

**The v4 row costs 10-12x the v3 row, and 72-78% of it is the geometry
channel**: 36 sphere-marched rays plus the terrain block's ~40 marched
probes (drop, ceiling, 16 edge probes, 16 dash landings, SDF and gradient),
all against the wall SDF (`bopl_obs_v4.h`, `bopl4_terrain` / `bopl4_rays`).
This code is already vectorised (AVX-512 SDF over 8 walls, 6-ray grouped
marches with per-ray cut-offs); what is left is the algorithm's step count
x wall count.

Sim, same runs (`bopl_rl_step` alone, 4-tick decisions, 2 players): default
loadout 386-580k ticks/s (9.8-10.3 us per decision under load; 500-580k in
a quieter interleaved rerun), random loadouts 290-320k (12.4-13.7 us; up to
124 entities). `benchmark.c` continuity row (loadout 2/7/19, 1 tick per
step): 610-620k ticks/s. Episode resets cost 10-13% on top (401-422 resets
per 130k ticks at the 7,200-tick cap with random inputs).

**Per env decision (2 rows), CPU:** v3 = 10 us sim + 2 us writer; v4 = 10 us
sim + 21 us writer. The v4 env step is writer-bound, 2.6x the v3 env step.
Per horizon step of the w128 b4 shape (128 games, 256 rows): ~4.0 ms of env
CPU for v4 vs ~1.5 ms for v3; over the pool's 8 stepping threads 0.5 ms vs
0.19 ms of wall. Against the measured v4-lite rollout of 0.89-0.94 ms per
horizon step (0.114-0.120 s / 128), **the writer is roughly half the v4
rollout wall on its own**.

Valid rows per set (player 0's production row, 4,096 states), against the
trainer's packed budget `DEFAULT_BUDGET` 6/12/20/8 (`bopl_v4.cu`):

| set | default: mean / p90 / max | random: mean / p90 / max | budget | capacity |
|---|---|---|---|---|
| players | 2.0 / 2 / 2 | 2.0 / 2 / 2 | 6 | 6 |
| slots | 5.95 / 6 / 6 | 5.95 / 6 / 6 | 12 | 24 |
| entities | 9.7 / 11 / 13 | 11.0 / 15 / 31 | 20 | 48 |
| slimes | 1.98 / 2 / 2 | 1.95 / 2 / 3 | 8 | 32 |

The budget packs 46 rows per sample where 19.6-21 are valid: the set GEMMs,
the embedding gather and its gradient run over 2.2-2.3x the rows they need
(the capacity is rounded to 64 rows and fixed for graph capture; rows past
it are dropped and counted, `set_overflow`).

### 1.3 Shapes that changed under v3 too

`BOPL_OBS_STRIDE` is the larger layout's size for every row, so since the
v4 merge **every observation row is 4,812 floats (19.2 KB), including v3's
(3,690 before)**: H2D bytes, the rollout obs tensors (4 buffers x 32 games x
2 rows x 128 steps x 19.2 KB = 630 MB) and the encode kernels' row reads all
grew 1.3x for v3. `THROUGHPUT.md`'s 1.14M is from before this; the set arm
of `bench_shape.sh` must be re-measured on this branch before any v4 ratio
is quoted against it.

The trainer's v4 path is not the naive one: the rollout already runs a
fused chain (`bopl_rollout_v4.cu`: ~18 launches per buffer step for every
policy, `br4_sets` one block per (row, set), TF32 grouped GEMMs), and the
train pass uses cuBLASLt bf16 GEMMs with fused ReLU epilogues over the
packed rows. The obvious "many tiny kernels" story is already spent.

## 2. Profile hypotheses (ranked by the share of the wall they predict)

Confirm with `bench_shape.sh v4-base` (`BOPL_BENCH_ENCODER=v4`, PROF on: the
JSON row carries `env_pool.env_cpu_ms_per_buffer_step`, `c_step.*.sim_us` /
`finish_us` (the writer runs in finish), `gpu_step.{h2d,forward,d2h}_us`
and `set_overflow_max`), then `BOPL_CUPTI_TRACE` + `cupti_timeline.py` for
the kernel-level split of the train pass.

H1. **Rollout, env CPU: the v4 writer's geometry channel.** ~2.7 ms of the
    ~4 ms of env CPU per horizon step; the env pool's 8 threads make it
    ~0.35 ms of wall of the ~0.9. Prediction: `c_step` `finish_us` >=
    `sim_us` x 2 for self-play envs (v3: finish << sim).
H2. **Rollout, GPU chain: `br4_sets`.** One block per (row, set), 1,024
    blocks per 256-row buffer step, each staging W2 (128x128 bf16, 32 KB)
    into shared memory and converting it from fp32 before touching its
    1-15 valid rows: ~32 MB of weight reads and 16M conversions per step
    for ~5k useful row-MLPs. Plus the same block copies the 19.2 KB row
    into the rollout tensor. Prediction: `gpu_step.forward_us` at v4 is
    >= 2x v3's and `br4_sets` is the longest kernel of the chain.
H3. **Rollout, H2D.** 256 rows x 19.2 KB = 4.9 MB per horizon step for
    either layout now (3.8 MB for v3 before the merge): 0.2-0.4 ms at PCIe
    rates, overlapped across the 4 buffers but not free. Prediction:
    `gpu_step.h2d_us` ~ 100-200 us per buffer step, the same for the set
    and v4 arms.
H4. **Train pass: the set path over the 46-row budget.** cap = 24,576 x 20
    = 491k packed entity rows per pass: `embed_rows` writes 126 MB of bf16
    addend per id set, the two GEMMs run over all 491k rows, backward
    twice more, `embedding_partials` over 960 chunks; x4 passes. The FLOPs
    are small (~0.3 TFLOP per update); the bytes are ~2-3 GB per pass.
    Prediction: the packing/embedding kernels, not the GEMMs, top the
    train timeline; `--v4-row-budget 3,8,16,4` cuts train_s measurably.
H5. **Train pass: hidden 256 + FFN 512 + critic 512 x T=128 recurrence.**
    The MinGRU scan is sequential over 128 steps per pass (forward and
    backward), 4x the FLOPs of v3's width-128 at the same launch count;
    the decoder adds FFN in/out and three critic GEMMs per step. Prediction:
    a long tail of small GEMMs in the timeline, latency- not FLOP-bound;
    `--minibatch-div` changes nothing, `--layers`/`--hidden` change a lot.
H6. **Memory pressure.** v4 takes ~6 GB device memory vs 2.6 (BOPL_V4_MEMLOG);
    with a co-tenant this becomes allocation stalls or OOM, not slowness.
    Prediction: `cuda_mb` in the bench row; no lever, a constraint.

## 3. Levers, ranked

Gain = the phase it moves, at the w128 b4 shape, if the hypothesis holds.
Gate: **exact** = bit-identical rows / logits (`puffer/check_dump.py` for
the CPU writer against the trainer's own rows, `puffer/check_v4.py` for
the GPU chain against `research/policy_v4.py`) and a `bench_shape.sh` A/B
(`BOPL_BENCH_AB=...`, 3 interleaved repeats, every candidate repeat above
every incumbent one); **learning** = also `puffer/sanity_v4.sh` (160M
learner steps of the v4-aim recipe against `sweeps/v4-aim/000`, band rating
of the 160M checkpoint within 75 Elo).

| # | lever | phase | expected gain | effort | gate |
|---|---|---|---|---|---|
| L1 | **Wall culling per row** in `bopl4_terrain_fill`: keep only walls whose SDF at the body is under R = `BOPL4_RAY_LENGTH` + 13 + 0.05 (the probes' furthest origin plus the march's hit threshold), and always the nearest wall plus any within 0.2 of it. For a sample point p at distance t along a march from an origin within 13 of the body, a culled wall is at least R - 13 - t = L - t + 0.05 away: it can never be the hit (d < 0.05) and, when it is the minimum, t + d >= max ends the march with the same `max` as before, so every march returns the same `t`; the lethal distance (clipped at 16), the SDF at the body (clipped at 8) and its gradient (the kept nearest walls) are unchanged. Halves the AVX-512 lanes on the 3-map set (wall counts ~20-40 -> the 8-16 in reach). | rollout env CPU | geometry 7.5 -> ~4 us per row; env CPU -3.5 ms/8 threads = -0.2 ms of the ~0.9 ms rollout step (-20%) | medium (one function, one proof) | exact (`check_dump.py` on maps 0-2, the bench's recorded states through both writers) |
| L2 | **`--v4-row-budget 3,8,16,4`** (or the p90 of the bench's row counts + 1): 31 packed rows per sample instead of 46; the set GEMMs, `embed_rows`, `zero_tail`, `embedding_partials` shrink 1.5x. | train | if H4: train -10-20% | none (flag) | exact when `set_overflow_max` stays 0 over the bench run (it is not exact the moment a minibatch overflows: rows are dropped) |
| L3 | **`BOPL_ENV_THREADS` 8 -> 10** (and `BOPL_OMP_THREADS` down to match): v3 tied 8-10; v4 is CPU-bound so the tie should break. | rollout env CPU | up to -20% env wall | none (env var) | exact |
| L4 | **`br4_sets` without the per-block W2 staging**: convert W2 (and W1, E) to bf16 once per update in `br4_prepare` and read them from global (L1/L2 resident: 4 sets x groups x 32 KB), or one block per row over all 4 sets with W2 staged once. | rollout GPU | if H2: forward_us -30-50% | medium (one kernel; the parity harness exists) | exact (same bf16 rounding of the same operands) |
| L5 | **Skip rows nobody reads**: scripted-bot players get a full v4 row written (`bopl_encode` writes every player); bots read `BotView`, not the row. With 3 scripted banks at share 0.11-1.0 that is 5-35% of the rows. Needs the row to also drop out of the live-row H2D and the frozen-bank chain, or it is only the writer that is saved. | rollout env CPU (+GPU) | -5-35% of writer time, proportional to the scripted share (large early in a run, small late) | low (writer) to medium (live-row layout) | exact for the learner's rows; the bot's row is never consumed. Verify the trainer's `bank_layout` already excludes scripted rows before counting the GPU half |
| L6 | **Compact obs rows for H2D and the rollout tensors**: transfer flat + privileged + the valid set rows with per-set counts (the writer already knows them: `env->written[]`), expand on the device in `br4_sets`/`pack_flat`. 19.2 KB -> ~5.5 KB per row; the rollout obs tensors shrink 3.5x (630 -> 180 MB), the train pass's `pack_flat`/`scatter_rows` read 3.5x fewer bytes. | rollout H2D + GPU, train bytes | if H3+H4: -0.1-0.2 ms per rollout step, train -5-10% | high (writer output format, `vecenv.h` upload, both chains, `check_dump`) | exact |
| L7 | **Fewer geometry probes, same channel**: the terrain block's 16 dash landings and 16 edge probes at 1.5-unit steps are the bulk of ~40 marches; 8 landings and 3-unit edge steps halve them. | rollout env CPU | geometry -2-3 us per row | low | **learning** (changes the row) |
| L8 | **`rays16`** (writer bit 4, POLICY_V4_PLAN's v4-rays16 ablation: 16 around + 4 down): halves the fan. | rollout env CPU | -2-3 us per row | none (flag) | **learning** (the v4-rays16 sweep entry is the gate; it has not run) |
| L9 | **Float SDF / fewer march iterations** (32 -> 16 steps with a proportional step floor). | rollout env CPU | geometry -30-50% | low | **learning** (different rays) |
| L10 | **Hoist the train forward's bf16 weight conversions** (9 `to_bf16` launches per forward; weights change once per optimizer step) and the `count_rows` strided reads into one row-major pass. | train | <2% | low | exact |
| L11 | **Critic only on learner rows in the rollout chain**: the frozen banks' 64 of 256 rows run the 512-wide privileged critic for a value nobody uses. | rollout GPU | -25% of the critic's ~3 GEMMs + 2 LNs per step (~5% of the chain) | low | exact |
| L12 | **Re-tune `BOPL_GPU_SPLIT`** after L1-L5 (v4 default 0 = shared; `auto` = 24/24 was -5% for v4-lite because both sides were SM-bound). Scheduling only. | wall | 0-5% | none | exact |
| L13 | **w256 b4** (THROUGHPUT.md G): more rows per launch. v4 is CPU-bound at w128 (~4 ms env CPU per horizon step on 8 threads); doubling rows doubles that. Only after L1/L3/L5 bring env CPU under the GPU chain. | wall | 0 now; 1.2x later | none | exact |

Not levers: the sim (10 us per decision is a third of the v4 env step, and
`THROUGHPUT.md` F already took its 2x); `bopl4_relation` (1 us per row,
14 calls; the ballistic arc is 30 iterations and could be 10, but it is
under 3% of the row); the privileged block (0.1 us).

## 4. Suggested order

1. Measure: `BOPL_BENCH_ENCODER=set bench_shape.sh v3-base` and
   `BOPL_BENCH_ENCODER=v4 bench_shape.sh v4-base` (300 updates each, PROF
   on), then a CUPTI timeline of one v4 update. Decide H1-H5 from the
   `c_step`, `gpu_step`, `env_pool` fields. ~15 GPU-minutes.
2. Free levers first, as A/Bs: L3 (`BOPL_BENCH_AB="BOPL_ENV_THREADS=10"`),
   L2 (`BOPL_BENCH_AB_ARGS="--v4-row-budget 3,8,16,4"`, keep only with
   `set_overflow_max` all zero), L12 last.
3. L1 (writer culling): exact, the largest single CPU lever; `check_dump.py`
   before the A/B.
4. L4 and L11 on the rollout chain if H2 holds; L5 once the row accounting
   is verified.
5. Only if the rollout is then under train: L6 (the big refactor) or the
   train-side findings from the timeline (H4/H5).
6. Learning-gated levers (L7-L9) last, each through `sanity_v4.sh`; they
   change the observation and so the policy family, and the campaign's
   product is the *same* policy faster.

## 5. Notes

- `writer_bench` found a sim bug on the way: `beam_contacts` used
  `i32::MIN` as the "no own beam" sentinel and `.abs()`'d it, aborting the
  process on the first Beam (ability 28) contact against a beamless player
  under `overflow-checks = true` (27 of 40 random-loadout seeds within 130k
  ticks). Fixed on this branch (`de6a2a34`); training never hit it because
  the fixed loadout 2/4/14 has no Beam.
- Load matters more than usual for these microbenchmarks: the pinned core
  is still preempted by a co-tenant's threads at load 6-7 on 12 CPUs;
  the sim swung 386k-580k ticks/s between runs minutes apart. Every JSON
  row carries `load1`; compare rows only at similar load, or interleave.

## 6. Campaign 3 wrap (2026-09-18 20:00 -> 2026-09-19, throughput-v4 Rift) — DRAFT

Metric: `learner_sps` on `bench_shape.sh` (`BOPL_BENCH_ENCODER=v4`, 300
updates, 3x3 A/Bs, load1 on every row); learning gate `sanity_v4.sh`
against `sweeps/v4-aim/000`, equal wall-clock (band rating of the
checkpoint nearest 955 s, 240 games) for recipe changes. Full record:
`autoresearch/notes-v4.md` and `results.tsv`.

### What landed

    stage                                            learner_sps   step
    v4-aim baseline, post-merge, 12-thread box           166.7k
    exact kernels, wave 1 (br4_sets bf16 registers,
      one-hot embedding in the set GEMM, active-row
      graphs, Lt epilogues, L11 critic on learner rows,
      Muon D!=C + side streams, LN float4, pool4,
      br_gemm float4 A, relu_grad x4)                    248.4k   +49%
    row cache (packed learner rows once per update)      250.7k
    shape: v4-lite-aim (set inner 64, FFN off, critic
      384; Matthew)                                       316.7k   +26%  gate ok
    writer: march refill lanes + axis-aligned walls
      (exact), then layout bits 10 = aim ray + LOS only
      (learning-gated: 475.7 vs 523.5 band at 160M, ok)   338.3k   +7%
    recipe: learner passes 4 -> 2 (equal-wall winner)    448.3k   +33%  gate ok
    packed obs transport (6 KB slots, unpack into the
      rollout tensor), PDL on the chain, br_gemm B
      prefetch                                            471.7k   +5%
    recipe: w256 b4, learner_passes 1 x minibatch_div 2
      (replay 1.0), LR x sqrt2 (b4; seeds 73/74 band
      612.0 / 579.2 at 955 s vs p2 572.8, ref 523.5)      626k     +33%  gate ok x2
    env pool: 11 stepping threads (CPUs - 1)             645.2k   +3%
                                                        ------
                                                        x3.9 over the baseline; the same policy family, lite shape

Discarded with a number (all reverted, none left behind a toggle): bf16
GEMM operands (+3.7%, -27 Elo: fail), fast transcendentals + row budget
default (0), fused dh1 (-0.5%), Muon grouped prefetch (-1.6%), br_sample
fused into the decoder (-2.1%), br4_sets header counts (-0.3%) /
occupancy 2 (-0.3%) / staged smem (+0.3%, forward unchanged), fused
set-MLP forward+backward (-4.1% once its launch failure was caught; -1.5%
staged walk; -0.8% forward alone), flat-layout 948 -> 648 (sized +2%,
not built: Matthew), set D 128 -> 64 (Matthew: keep), buffers 2/8 at
w128 and 8 at w256 (-9..+0.5%), w512 (OOM at 12 GB), thread placement
(6 pinned / 6 / trainer core / trainer core + home mask: -4..-22%),
SM split 32/16 and 36/12 (-14% / -23%), writer to GPU (sized: +0..5%
for 8-10 turns, not built).

### The measured ceiling at this geometry: ~700k

Two floors, both measured, agree:

- Writer-free bound: with the observation writer skipped entirely
  (throwaway build) c_step 20.6 -> 13.7 us and learner_sps 645k -> 688k
  only; the buffer step becomes GPU-chain-bound (eval_gpu 0.336 vs
  eval_env 0.206 ms per step; forward 272 -> 308 us under the denser
  train overlap).
- Contention-free bound: with the rollout streams on 36 SMs the forward
  drops to 239 us and the rollout to 71.3 ms per update (from 75.3):
  ~4 ms is all the train stream costs the rollout. The train pass itself
  is 24.8 ms alone, 57-60 under the rollout: it fits three times over.

Per buffer step (128 rows): env CPU 0.26 ms + chain 0.29 ms, serial
within the buffer, four buffers co-running imperfectly: ~0.55 ms floor
x 128 steps = 70 ms per update = ~700k learner_sps at 49152 rows per
update. The chain's 0.29 is 170 us of isolated forward (19 launches,
latency-bound, PDL already; 90 us at 64 rows) plus launch tail and
HtoD; the env's 0.26 is 64 envs x 21 us of c_step over the pool's
share of 11 threads, of which the sim is 10 us (4 ticks at 1 us
isolated, x2.3 SMT) and the aim-only writer 8 (4.1 isolated).

### What 1M would take (this reading)

Per update at 1M: 49 ms for 49152 rows, i.e. a 0.38 ms buffer step at
128 rows, or the same 0.55 ms step at 180 rows. Neither half of the
step gets there alone:

- Hardware: the box is 6 cores / 12 threads; the env CPU is 626 ms of
  c_step per update over 11 threads. A 16-core part at the same per-core
  speed takes the env half to ~0.13 ms per buffer step and the wall to
  the chain's 0.29 (+ the env's remainder): ~800-850k. More SMs do not
  help the chain (latency-bound) but a newer GPU with lower launch
  latency and faster L2 shortens the 170 us; 16 cores + that is the
  hardware road to 1M with the present architecture.
- Architecture, same box: (i) the chain per row: rows per chain up
  (w512 needs the training buffers to shrink: bf16 rollout storage or
  fewer frozen banks resident; 12 GB is the cap) or the 19-launch chain
  collapsed into 3-4 persistent-kernel stages (a megakernel: sets +
  flat GEMM + GRU + heads in one launch per step, ~40-60 us instead of
  170; a rewrite of bopl_rollout_v4.cu's chain, not exact-gated by
  construction but bit-identical in principle); (ii) the CPU per
  decision: the writer to the GPU (+0..5% alone, worth it only after
  (i)), the sim's per-tick cost (platforms 22%, entity export 17%:
  exact housekeeping worth ~3%); (iii) both together: with (i) at 60 us
  the step is env-bound again at ~0.26 -> 128 x 0.3 = 38 ms -> ~1.2M
  before the writer even moves. So the architecture road is the
  megakernel chain first (the only lever sized above 20%), then the
  writer.
- Not the road: recipe (replay 1.0 is the PufferLib default already;
  further passes/batch changes trade Elo), SM partition, thread
  placement, more buffers, kernel-level work on the chain's individual
  kernels (each read ~0 against the contention and launch floor).
