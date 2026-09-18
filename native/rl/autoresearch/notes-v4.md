# Throughput campaign 3: policy v4 (notes)

Rift `throughput-v4`, branch `throughput-v4`, starting commit `05020541`
(program: `program-v4.md`; background `../THROUGHPUT.md`, `../THROUGHPUT_V4.md`).
Results table: `results.tsv`, rows appended by `record.py --shape <label>` from
`native/artifacts/rl/bench-shape/results.jsonl` (gitignored; the run dirs and
logs sit next to it). Column mapping for campaign-3 rows: `learner_dps_wall`
= `learner_sps` (the score), `learner_dps_active` = physical `sps`,
`wall_seconds` = `wall_s` per update.

## Workload

`shape-v4-w128-b4-p4-u300`: `BOPL_BENCH_ENCODER=v4 bench_shape.sh <label>`
(train.py, encoder v4, hidden 256, writer bits 8, aim Fourier 4, 128 workers x
4 buffers, 4 banks, 4 learner passes, horizon 128, seed 73, maps 0-2, 300
updates; medians of updates 21-300). `BOPL_BENCH_PROF=1` (default) adds the
env-pool / c_step / gpu-step split to the row. The v3 arm is
`shape-set-w128-b4-p4-u300` (`BOPL_BENCH_ENCODER=set`, hidden 128).
Candidates: `BOPL_BENCH_AB=... [BOPL_BENCH_AB_ARGS=...] BOPL_BENCH_REPEATS=3`,
interleaved incumbent / candidate; keep when every candidate repeat beats every
incumbent repeat (or the gain is clearly outside the repeats' spread) at
similar `load1`.

Gates: exact = `puffer/check_dump.py --only v4` + `puffer/check_v4.py`,
`set_overflow_max` all zero; learning = `puffer/sanity_v4.sh run/report`
(needs Matthew's word first).

## GPU ownership

The campaign owns the GPU (Matthew, 2026-09-18 15:29). Before every GPU job:
`systemctl --user list-units 'bopl-*'` must show no `bopl-sweep-*` unit
(`bopl-gpu-queue.service` and `bopl-dashboard.service` stay). Wave 3's last
run `bopl-sweep-v4-no-rays-maps-s74-000.scope` was at 1.612B of 1.735B learner
steps at 15:32 EDT (167k learner sps, ~12 min left).

## Baselines

Both taken at commit `c2376d48` while the wave-3 sweep driver was still
scoring at ~560% CPU (load 12-14; the box has other agents' CPU jobs at nice
19 too), so they are noisy on the CPU side. The v4 arm is train-bound, so its
score is trustworthy to a few percent; the v3 arm is CPU-bound and was hit
hard (campaign 2 measured 1.146M learner_sps on this shape at load 5.5).

| label | learner_sps | sps | rollout_s | train_s | wall_s | per buffer-step ms rollout / eval_gpu / eval_env | load1 |
|---|---|---|---|---|---|---|---|
| v3-base-load (set, hidden 128) | 258071 | 344094 | 0.0941 | 0.0320 | 0.0952 | 0.735 / 0.144 / 0.464 | 12.6 |
| v4-base-load (v4, hidden 256) | 166657 | 222209 | 0.1272 | 0.1461 | 0.1475 | 0.993 / 0.479 / 0.473 | 13.9 |

Quieter re-measurements of the v4 incumbent (l2-budget A/B, load 8.3):
168.9-169.1k learner_sps, rollout_s 0.1026-0.1036, train_s 0.1441-0.1444,
wall_s 0.1453-0.1455, per buffer-step 0.80 / 0.48 / 0.31 ms.

### Wall split (v4, 128 workers x 4 buffers, 4 passes, rollout/train overlap on)

- `wall_s` = 0.145 per update = max(train 0.144, rollout 0.103): **train-bound**,
  with the rollout only 30% behind. Both must shrink for a multiple.
- Train (CUPTI, stream 13, 62.5 updates): 8674 ms busy in a 9849 ms span
  (88%), 139 ms of kernels per update, 4 passes x 24576 rows. Of it, per update:
  GEMMs ~72 ms (fp32 TF32 puf_mm for flat / fusion / FFN / actor ~5 x 0.45 ms
  per pass, the bf16 set MLP GEMMs over the packed capacity ~35 ms), and
  **~51 ms of memory-bound elementwise kernels** (ln_backward 7.4 ms,
  embed_rows 6.1, to_bf16 5.8, ln_forward 5.3, relu_grad_from 4.7,
  unpool_rows 4.5, pool_rows 3.4, add_into 3.0, embedding_partials 2.0,
  relu_into 1.8, count_rows 1.4, pack_flat 1.2, scatter_rows 2.2, zero_tail
  1.0, copy_cols 0.6, concat_priv 0.6), mingru scans 3.2 ms, PPO loss 2.2 ms,
  memcpy DtoD 2.7 ms. The elementwise kernels ran at 65-190 GB/s (one thread
  per element, 2-byte stores, one 256-thread block per 256-wide row with two
  block reductions): the softer target.
- Rollout (streams 14-17, one per buffer): per buffer step the GPU window is
  413 us (p10 283, p90 645) of which the stream is busy 300 us; 19 records:
  H2D 1.25 MB 23 us, `br4_sets` 96-118 us (the one big kernel: the four set
  MLPs for 64 rows), 9 x `br_gemm` 10-28 us (125 us), `br_sample` 22-25 us
  (v3's was 6), 4 x `br4_rows` 4 us, `br_gate`, D2H. Host gap between steps
  381 us (the sim: `eval_env` 0.31-0.47 ms per step, `c_step` 37-44 us per
  env with 75% of the writer in geometry per THROUGHPUT_V4.md). The gaps
  inside the window (49 us before the second gemm, 66 us before br_sample)
  are the other streams' kernels: the four buffers and the trainer contend
  for the SMs, so any GPU work removed on one side speeds up the other.
- Physical rows: sps = 4/3 learner_sps (the scripted / frozen opponents'
  rows are simulated and inferred but not learned from).

## Experiment log

Each entry: hypothesis, change, commit, numbers, gate, decision.

### prep (CPU only, before the GPU was free)

- `record.py --shape` (this commit): results.tsv rows from bench_shape rows.
- Read program-v4.md, THROUGHPUT.md, THROUGHPUT_V4.md; the CUDA extension in
  this Rift was built 14:47 from the 14:41 sources (nothing newer), libbopl_sim.so 15:25.
- writer_bench now cycles maps 0-2 and has a `march-check` (vector march vs
  scalar march, bit for bit). Wall counts on the benchmark maps are 6.4 per
  row on average, not the 20-40 THROUGHPUT_V4.md assumed, so L1 (wall
  culling) had nothing to cull (6.4 -> 5.8 walls/row): discarded before any
  GPU run. The lane-transposed AVX-512 march (rays in the lanes, walls in
  the loop) was exact but no faster; a refilling variant (a lane whose ray
  ends takes the next ray) measured -12% us/row on the v4-aim writer in
  writer_bench (7.4 vs 8.4) but the sim side is not the binding constraint
  at this shape (see the split), so it is parked, unbuilt, for when the GPU
  side is fixed and the sim becomes binding again.

### L2: row budget 3,8,16,4 (`--v4-row-budget`), commit c2376d48, A/B 3x3

Hypothesis (THROUGHPUT_V4.md L2): the packed set capacities are sized by the
default budget {6,12,20,8} rows per sample; the set GEMMs, zero_tail,
embed/pool/unpool run over the whole capacity, so a tighter budget shrinks
them proportionally. Result: learner_sps 178.4k [177.9k, 178.6k] vs 169.0k
[168.9k, 169.1k], **x1.0555**, train_s 0.1367 vs 0.1442, rollout unchanged,
every candidate repeat above every incumbent; `set_overflow` 0 in all 900
candidate updates. To know the headroom I made the trainer log the packed
row totals (`set_rows` per update = packed rows / batch rows per set;
bench_shape reports `set_rows_max`): on this workload the batches need at
most **2.0 / 6.0 / 11.7 / 2.0** rows per sample (players, slots, entities,
slimes); the 1.7B-step wave-3 run `v4-no-rays-maps-s74` never overflowed the
default in 70595 updates, but nothing recorded its actual need. Decision:
kept as a measured flag; changing `DEFAULT_BUDGET` is policy-affecting the
moment a batch overflows (rows dropped, deterministically) so it needs
Matthew's word. Proposal: {3, 8, 16, 4} (+5.6%), or {2, 6, 12, 2}-ish after
one long run logs `set_rows` on all maps.

### Train pass: fewer memory passes (commit 70e5450f), keep

Hypothesis (from the split): 51 of the train pass's 139 ms are elementwise
kernels running far below bandwidth, several of them pure format changes.
Change, all in `bopl_v4.cu`, same math:
- LayerNorm forward/backward: one warp per row (8 rows per block, the row in
  registers, moments by warp shuffle) instead of a 256-thread block per row
  with two `__syncthreads` reductions and three reads of the row. The forward
  reads `[y | priv]` itself (concat_priv gone, `zin` written by LN as the
  backward's input) and writes the bf16 GEMM operand directly (`lnin_b`,
  `ln2_b`: four `to_bf16` passes and the fp32 `lnin`, `ln2` buffers gone);
  the backward writes an optional bf16 copy (`g_z2_b`, `g_z_b`) and
  accumulates the LN_in gradient's first H columns straight into `g_y`
  (copy_cols + add_into + `g_zin` gone; `g_z2` fp32 + its memcpy gone).
- `bblk` written bf16 by the GEMM's ReLU epilogue; its backward mask reads
  the bf16 (`relu_grad_bf16` writes `g_b_b` directly: relu_grad_from +
  to_bf16 gone).
- `embed_rows` (769 us x 2 per pass at 65 GB/s: one thread per bf16 with a
  64-bit division) gone: `scatter_rows` copies E[id] into `erows` as it
  packs each row (it is latency bound on its per-sample walk with the memory
  system idle); `zero_tail` zeroes the `erows` tail together with `xpack`'s.
- fp32 sums in LN are now reassociated (warp order vs block order): not
  bit-identical to the old kernel, same semantics; check_v4.py (torch
  reference, rel 2e-2) pass_all.
Result: fuse-1 179188, fuse-2 179436 learner_sps (load 4.0-4.2), train_s
0.1361 / 0.1359 vs the incumbent's 0.1442 (**train -5.7%, score +6.1%**
against the 168.9-169.1k incumbent repeats taken 15 minutes earlier at load
8.3; rollout_s 0.103 in both, so the load difference did not contribute).
With L2 on top it should stack (different kernels).

### Rollout: br4_sets without the per-block W2 staging (commit 9b3777ec), keep

Hypothesis (THROUGHPUT_V4.md L4, and the split: `br4_sets` was the rollout's
one big kernel at 96-118 us of the ~300 us GPU-busy per buffer step, min 21
us alone): each of the 256 (row, set) blocks per buffer step re-read the
group's fp32 W2 (64 KB), converted it to bf16 pairs into 33 KB of shared
(2 blocks per SM), and thread 0 walked the set's 48 rows serially to find the
valid ones (a chain of dependent loads). Change (`bopl_rollout_v4.cu`):
`br4_prepare` packs each group's W1 (transposed to (features, I), so thread
i of the h1 loop reads column i coalesced) and W2 to bf16 once per rollout
(`br4_pack_set_weights`, the same bf16 roundings the kernel applied on the
fly); each h2 thread holds its W2 row (I bf16) in registers; valid rows by
warp ballot. Shared per block 45.7 -> 12.2 KB. The fma order per element is
unchanged. Exactness: BOPL_BR_CHECK=1 turned out vacuous on this trainer (it
only runs in eager steps, and those see all-zero rows: every line says
`max|logit| 0`, on the old build too), so I compiled the old kernel in
temporarily and compared the pooled `joined` columns bit for bit on every
eager buffer step of a `--cudagraphs -1` run: **0 of 838,860,800 values
differ** (12800 buffer steps, 598M of them nonzero). The check code was
removed before the commit. Result: sets-1 194293, sets-2 194447 learner_sps
(load 3.8-4.7) vs 179.2-179.4k: **+8.4%**; rollout_s 0.089 vs 0.103,
eval_gpu 0.407 vs 0.497 ms per buffer step, and train_s 0.1254 vs 0.136
although the train pass did not change: the SMs the set blocks no longer
occupy go to the trainer's stream (the contention the split predicted).

Running total: 169.0k -> 194.4k (+15.0%), train-bound still (0.125 vs 0.089).
