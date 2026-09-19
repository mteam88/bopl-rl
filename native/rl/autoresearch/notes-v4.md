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

### L2 upper bound: row budget 2,6,12,2, A/B against 9b3777ec, flag only

The measured need on this workload (`set_rows` mean 2.0 / 6.0 / 10.9 / 2.0,
max 2.0 / 6.0 / 11.5 / 2.0 per sample) as the budget: learner_sps 214084,
214458 vs 194494 (**+10.2%**), train_s 0.1136 vs 0.1253, rollout unchanged,
`set_overflow` 0 in 600 candidate updates. Incumbent repeats 2-3 and
candidate 3 refused to start (stale `pufferlib._C`) because I edited
`bopl_v4.cu` while the A/B ran: do not touch tracked sources during a bench.
Recorded as the size of the prize for Matthew; the default stays.

### Train pass: embedding as a one-hot block of the GEMM (commit a606f1ca), keep

Hypothesis: the entity set's embedding add cost two extra passes over the
packed rows per forward (scatter_rows writing `erows`, the cuBLASLt C-addend
reading it: ~126 MB each at 65k rows x 64 x 2 B... times 4 passes x 4
buffers) plus `embedding_partials` in the backward (a 64-slot atomics kernel
over every row). Change (`bopl_v4.cu`): the packed row is
`[features | one-hot(id) (16)]` (kpad: entities 48 -> 96, slots 32 -> 64;
players/slimes have no id and keep 32) and the set's GEMM operand is
`w1b = [W1 | E^T]` (`pack_w1`), so the GEMM computes `x W1^T + E[id]` with no
addend; the embedding gradient is the one-hot columns of dW1 transposed
(`embed_grad_from_w1`, replaces `embedding_partials`; `erows`, `partials`
and the E `to_bf16` gone). Same bf16 roundings of x, W1 and E as before; the
fp32 GEMM accumulation order differs (the E term is now one product inside
the dot). check_v4.py pass_all (embedding grad rel 5e-4, same as before the
change; the reference's fp32 sum vs bf16 GEMM). Result: onehot-1 198635,
onehot-2 198766 vs sets-1/2 194.3-194.4k: **+2.2%**, train_s 0.1226 vs
0.1253, cuda_mb 5868 vs 6018. Less than the 4% I hoped: the wider kpad
doubles the GEMM's A-operand and dW1 traffic for the two id sets, eating
half of what the removed passes gave back.

Running total: 169.0k -> 198.8k (+17.6%, exact-class only); with the
flagged budget change on top it would be ~+30%. Train-bound (0.123 vs 0.089).

Profile of this build (CUPTI, 60 updates, train stream 7384 ms = 119 ms per
update, 96% of train_s): GEMMs ~73 ms per update (61%), v4 elementwise ~33 ms
(ln_backward 6.6, unpool 4.3, pool 3.8, relu_grad_from 2.8, ln_forward 2.6,
add_into 2.4, relu_into 2.0, relu_grad_bf16 1.6, count_rows 1.4, scatter
2.7, pack_flat 1.3, zero_tail 1.3), memcpy DtoD 2.0, mingru 3.1, PPO 1.9.
The set GEMMs (the relu bf16 128x256 forward 13.4 ms, drelu 9.2, the nt
wgrads ~7) run over the capacity while the batch fills 33 / 50 / 56 / 25% of
it (set_rows 2 / 6 / 11.2 / 2 vs budget 6 / 12 / 20 / 8): the tight-budget
experiment's +10% is that tail.

### Train pass: set GEMMs over the batch's rows, one graph per bucket (commit 431a53b4), keep

Hypothesis: the row budget's gain comes from GEMM rows, not from the buffer
size, so it can be had without changing the budget: run the pass over the
minibatch's packed rows. The M is baked into the captured train graph, so
the trainer now runs a prepass before each pass's launch (`enc_prepass`:
count_rows + scan_rows, the totals published into mapped pinned memory by a
one-warp kernel, a blocking-event wait), rounds each set's total up to a
granule (cap / 16, TILE multiple) -> `act[s]`, and keys the train graph by
the bucket tuple: one captured graph per (ready slot, key), a key's first
pass eager (the cuBLASLt plan tuning runs there), captured on its second.
Inside the pass every `cap` in the set MLP path (zero_tail, the four GEMMs'
M / K, mask_rows, the RELU_AUX threshold) became `act`; scatter's drop cap
and pool/unpool (valid rows only) are untouched, rows [act, cap) are never
read. Two things bit on the way: (1) the eager passes now happen after
warmup, while the buffer threads capture their rollout graphs in
cudaStreamCaptureModeGlobal, so the eager cuBLASLt calls failed ("an
internal operation failed"); captures are ThreadLocal now (rollout and
train). (2) Reading the totals back with cudaMemcpyAsync made the rollout
40% slower (eval_gpu 0.53 vs 0.40 ms per step with the GPU window itself
shorter): the D2H sat on the copy engine behind the previous pass's ~25 ms
graph and the buffer streams' action copies queued behind it. Publishing
through a kernel into mapped host memory fixed it (rollout 0.089 again).
Numerics: GEMM shapes follow the bucket (tile order, split-K), so fp32 sums
reassociate; check_v4.py pass_all with the harness running the prepass and
recapturing on a key change (batch 257 differs from the fixed-cap run in the
6th digit, so the bucketed path is exercised). 8 captures in 300 updates
(keys 5221 then 31931 / 31948 / 31965: the entity bucket 9 -> 10 -> 11 as
episodes fill in). Result: actrows-1 218166, actrows-2 218293 vs 198.6-198.8k:
**+9.8%**, train_s 0.1115 vs 0.1226, rollout_s 0.089 unchanged, cuda_mb
+12 (graph execs). The tight-budget flag measured +10.2%: this is that gain
without the budget or drop semantics changing, so the DEFAULT_BUDGET
question is now only about memory (the packed buffers) and worst-case drops.

Running total: 169.0k -> 218.3k (**+29.2%**, exact-class only). Train-bound
(0.1115 vs 0.089).

### Muon: grouped Newton-Schulz for v4's 19 matrices, discard

The profile of the new build (train stream 6590 ms / 61.5 updates = 107 ms
per update) showed the per-matrix cuBLAS Muon path running: 22390 tiny
`s1688gemm_64x64` launches, 50844 DtoD memcpys, 37238 splitKreduce: ~530
launches per pass, ~12 ms per update (11%). v3's grouped fp32 path
(`BOPL_MUON_FUSED`, default on) has a `MUON_MAX_MATS 16` cap and v4 has 19
matrices, so it fell back silently. At a cap of 32: muon32-1 216895,
muon32-2 217507 vs 218.2-218.3k: no gain; CUPTI says grouped 11.2 ms per
update (15 `muon_grouped_gemm` per pass at 140 us: the 64 x 64 fp32 SIMT
tiles run at ~7 TFLOPS under the rollout's contention, and v4's grams are
big: fusion 256 x 256 over K = 1280, critic 512 x 512 over 512) vs 12.0.
Reverted to 16 (the wave-3 runs were trained with the TF32 cuBLAS path, so
this also keeps the optimizer's numerics); the fallback prints a note now.
A tensor-core grouped NS would take Muon from ~11 to ~3-4 ms but changes the
optimizer's rounding: Matthew's call, not this campaign's.

### Train pass: elementwise ops into the cublasLt epilogues (commit 61c317ff), keep

Hypothesis: the train profile still had ~6 ms per update of standalone
elementwise kernels around GEMMs (add_into 2.4, relu_into 2.0, relu_grad_bf16
1.6, relu_grad_from 3.2 ms per 61 updates x 4 passes), each a full read +
write of a (B, 256..512) fp32 activation; cublasLt can do them in the GEMM
epilogue. `lt_gemm` got an fp32-operand twin (`lt_gemm32`, both over a
shared `lt_matmul`; `LtKey.ab_f32`). Fused: the encoder's flat GEMM writes
ReLU(x W^T) straight into `joined` at stride LDJ (flat_pre gone); the FFN
input GEMM has the RELU epilogue; the FFN output and the critic block's
second GEMM take the residual as the beta = 1 C operand (add_into gone); the
critic block (bf16 GEMMs) runs RELU_AUX forward and DRELU backward with a
B x C/32 bitmask, so relu_grad_bf16 and the fp32 g_b buffer are gone there.
Tried and dropped inside the experiment: RELU_AUX / DRELU on the fp32 FFN.
It passes check_v4 but the CUPTI profile showed cublasLt running them as a
separate `cublasLt::epilogue::impl::globalKernel` (0.38 ms forward, 0.45 ms
backward per call), slower than the relu_into / relu_grad_from kernels they
replaced: first build measured 219.1-219.7k, i.e. flat. The fp32 plain RELU
epilogue is fused (relu_into vanished, no new kernel). Result, all at load
5-7: epi2-1..3 219758 / 221615 / 223224 vs epi2-inc-1 217073 (39d7f1a9
rebuilt in place between runs): **+1.2..+2.8%**, train_s 0.1091-0.1105 vs
0.1121, rollout unchanged. check_v4 pass_all; check_v4 now also runs a
2048-row case so the decoder's B >= LT_AUX_MIN_ROWS mask path is covered
(the 1024-row case only reaches it in the encoder's set GEMMs).

Running total: 169.0k -> ~221.6k (median of 3) (**+31%**, exact-class
only). Train-bound (0.110 vs 0.091).

### L12 and stream priority: where the contention lands, discard

L12 (green-context partition, `BOPL_GPU_SPLIT`), A/B 2x2 at load 6.3-6.7:
rollout 16 SMs / train 32: 155.3-155.9k (rollout 0.157 s, train 0.0874);
24 / 24: 205.8-206.0k (0.118 / 0.106); shared (default): 223.1-223.3k.
Then the priority itself, shared GPU (`BOPL_TRAIN_PRIO`, new, kept as a
diagnostic knob in the patch; the default is unchanged): equal 202.0k
(train 0.0825, rollout 0.1207), train ahead 199.2k (train 0.0711, rollout
0.1225). A mix (first k of the 4 passes on a second, high-priority train
stream with event-ordered hops; the two train streams never run
concurrently, the GEMM tune list covered both) k=1: 224.2k (+0.6%: train
0.1086, rollout 0.101); k=2: 221.6k. Reverted.

What these say, with the serial run (`BOPL_ASYNC=0`, 61c317ff): rollout
alone 71.6 ms per update (eval_gpu 0.243 ms per buffer step against 0.418
shared, env 0.309), train alone 63.0 ms, serial wall 136 ms (180.8k);
overlapped wall 110 ms (223k). The overlap recovers 26 of the 64 ms it
could: the GPU is saturated, the wall is about the sum of the two sides'
GPU work, and priority / partition only choose which side the contention
shows up in. The rollout's chain (~19 kernels per 64-row buffer step, each
tiny and latency-bound, 512 steps per update) is ~40% of the GPU's time,
the train pass ~60%. From here only less GPU work counts, on either side.

## Merge of main 46fb0382 (commit 556e7a7b)

Merged (not rebased: the branch carries the maps-round2 merge). Conflicts:
the two comment blocks (`Tools/play-vs-agent-mac.sh`,
`native/tools/extract_rollout_data.py`) took main's side; `native/rl/Makefile`
kept the campaign's `writer-bench` target next to main's rules;
`research/elo_ladder.py`'s `on_result` callback (dropped by the maps-round2
merge e52b83bf; tournament.py and tournament_cross.py pass it) restored in
9f2b2753. Rebuilt the sim (`cargo build --release`, nice
19), the CUDA extension, writer_bench; `check_v4.py --build` pass_all,
`check_dump.py --only v4` ok. New incumbent `v4-base-m46fb-1/2`: 223918 /
224117 (train_s 0.1086 / 0.1085, rollout_s 0.0858 / 0.0841; the merged
sim's env step is cheaper, eval_env 0.24 vs 0.28 ms per buffer step, which
the train-bound wall does not see). Candidates from here compare against
~224k.

## Rollout chain: br_gemm column rounds per block (`BOPL_BR_UNITS`), parked

The chain's GEMMs at M = 64 (7 row tiles: 3 of the learner's 48 rows, then
one 4-of-16 tile per frozen bank) launch one block per (row tile, column
subtile group): 9 launches per buffer step of 7 x 4..64 blocks. `units`
rounds per block (`BrGemmArgs.units`, `BOPL_BR_UNITS`, default 1) makes a
block cover `units` column groups in sequence with the same chunk -> warp
assignment and partial-sum order, so the sums are bit-identical for any
`units` (checked on the device word for word over a 300-update run with a
temporary compare against `units = 1`, since removed: 0 mismatching words).
A/B 2x2 at load 6.7: `units = 4` 226.5k vs 224.1k (+1.1%: train 0.1074 vs
0.1085, rollout 0.102 vs 0.084); `units = 2` 228.0k vs 225.2k (+1.2%:
train 0.1068 vs 0.1080, rollout 0.091 vs 0.084). Real but small, and it is
the same trade as the priority experiments (fewer, longer rollout blocks
leave SMs to the train pass while the rollout takes longer). Knob kept,
default 1. Re-run on top of L11: `units = 2` 238.2k vs 236.1k (+0.9%,
repeats 237.4 / 239.0 vs 235.7 / 236.5; rollout 0.090 vs 0.085 s). Parked
at default 1: the rollout side (0.085) is closing on the train side
(0.103), and this lever lengthens it.

### L11: the rollout chain's critic on learner rows only (commit 8cb488c6), keep

The v4 chain ran its critic (3 GEMMs at N = 512: cin over K = W + 12,
b1, b2 over K = 512, plus 2 row kernels) over all 64 rows of a buffer:
the learner's 48 rows in 3 row tiles and one 4-of-16 tile per frozen bank
(4 banks), 7 tiles, of which the banks' 4 are 57% of the blocks for 6% of
the useful rows. Nobody reads a frozen row's value: its advantages are
zeroed (`zero_frozen_advantages_cuda`), the minibatch sampler and the
advantage / value statistics take learner rows only. Now the three GEMMs
launch with the plan's group count set to 1 (grid.x = the learner's row
tiles; `br_gemm` does not read gridDim, so the learner's tiles run the
same blocks with the same arguments: bit-identical logits and values) and
the two row kernels over `g.end[1]` rows. The banks' value column keeps
the zero A.dec was allocated with; `BOPL_BR_CHECK` skips it for v4.
A/B 2x2, twice, at load 6.0-6.2 (cand = the old all-rows critic behind a
temporary toggle, since removed): 236.0-236.1k vs 223.4-224.3k and
236.5-236.7k vs 223.8-225.1k: **+5.4%**. Rollout unchanged (0.0837 s), train
0.103 vs 0.1085: the saved rollout GPU work goes to the train pass.
`check_v4 --build` pass_all, `check_dump --only v4` ok.

### Muon per-matrix path: out-of-place cublasLt (D != C) for the Newton-Schulz addends, keep

v4's 19 matrices run upstream's per-matrix cuBLAS Newton-Schulz (the
grouped path is capped at 16 and measured no faster, above). Each of the 5
NS steps was gram GEMM, `puf_copy(gram_buf <- gram)`, in-place GEMM
`gram = c2 gram_buf gram_buf + c1 gram`, `puf_copy(dst <- src)`, in-place
GEMM `dst = src gram_buf + c0 dst`: the copies exist because cuBLAS's C is
also its D, and the beta != 0 GEMMs were excluded from the kernel tuning
(timing an in-place accumulate corrupts C), so they ran the heuristic
default with its split-K reduce kernels (the profile's 830 DtoD memcpys and
600 splitKreduce per update). `bopl_gemm` now takes an optional D:
`D = alpha op(A) op(B) + beta C` through cublasLt with C read in place,
tuned like the rest (idempotent, so the timing loop is safe); the fallback
without a tuned kernel is the old copy + in-place GEMM. The two NS GEMMs
write out of place: `gram_buf = c2 gram gram + c1 gram`, `dst = src
gram_buf + c0 src`. Same products, same operands, same TF32 compute type;
the algorithm is the tuned choice instead of the default, as for every
other train GEMM. A/B 2x2 twice (cand = the old copies behind a temporary
toggle, removed): 237.7k vs 236.1k (+0.7%) at load 7.9, 238.3k vs 235.0k
(+1.4%) at load 6.8; train 0.1022 vs 0.1033. Tune log: the D!=C shapes get
2-8 candidates at 4.3-6.4 us.

### Muon per-matrix path: the matrices' NS chains on concurrent side streams (lanes), keep

With the copies gone each matrix is still ~30 launches of 4-10 us in a
row on the train stream, latency-bound, 19 matrices in turn: the profile
had Muon at ~12 ms of the ~100 ms update. The chains are independent, so
`muon_step` now forks `BOPL_MUON_STREAMS` lanes (default 8) off the train
stream by events and joins them at the end; under capture they become
parallel graph branches. The matrices go to lanes longest-first onto the
least loaded (NS flops), each lane has its own gram / gram_buf / x_buf /
norm scratch, and the lanes' cuBLAS handles and tune workspaces are made
on the first (eager) step, since a handle cannot be created under capture
(`bopl_gemm_tune_stream_add`: the side streams tune like the train stream
and share its kernel choices). A matrix's kernels and their inputs are the
same as on one stream, so the result is the same; the streams get the
train stream's priority. `BOPL_MUON_STREAMS=1` is the old serial order.
A/B 2x2 (cand = 1 lane): 242.4k vs 238.5k (+1.6%), 242.8k vs 237.6k
(+2.2%), train 0.1003 vs 0.1022; lanes 8 vs 4: 244.0k vs 241.9k (+0.9%,
train 0.0997 vs 0.1006); 2 vs 4: 241.9k vs 242.6k. Default 8; each lane
past the first costs ~45 MB of GPU memory (cuBLAS handle + tune workspace,
6.37 GB total at 8 vs 6.06 at 4 on the 12 GB card), the knob trims it.
`check_v4 --build` pass_all, `check_dump --only v4` ok (the forward is
untouched); the optimizer change is covered by the learning gate
(`sanity_v4.sh`), scheduled below.

Running total: 169.0k -> ~244k (**+44%**, exact-class only). Train 0.0997,
rollout 0.085.

### Profile of this build (CUPTI, 280 updates, train stream 27.48 s busy of a 31.4 s span = 98 ms per update)

Per update, train stream: TF32 fp32 GEMMs ~25 ms (128x256 tn 10.3, 128x128
nt 4.4, 256x128 nn 3.6, 128x256 nn 3.5, others), bf16 GEMMs ~19 ms (relu
128x256 6.9, drelu 64x64 4.8, 64x128 3.7, 64x64 tn/nn 3.1 + 2.5, drelu
128x128 1.8, ...), `ln_backward` 7.9 (5.4 per pass, avg 367 us),
`unpool_rows` 5.4, `relu_grad_from` 3.6, `pool_rows` 3.5, `ln_forward` 2.9,
mingru scan 2.4 + 1.2, `count_rows` 1.7 + `scatter_rows` 2.5 + `pack_flat`
1.2 (the prepass reading the observations), `ppo_loss` 1.8, splitKreduce
1.4, DtoD memcpy 1.3, `br_transpose_frag` 1.0. Muon's lanes sit on their
own streams (~0.3 s busy each over the run, ~1 ms per update on the
critical path). Rollout, per buffer stream (4): `br_gemm` 4.2 s (9 per
step, avg 12 us, min 3.7), `br4_sets` 2.15 s (avg 56, min 5.8), HtoD 1.04
s (27 us per step: the observation upload, copy engine), `br_sample` 0.9 s
(avg 23, min 9), `br4_rows` 0.8 s, `br_gate` 0.15 s. The averages are
contention (min = isolated); as GPU work the rollout is ~27 ms of SM time
per update against the train pass's ~63. B = 24576 rows per pass (4
buffers x 48 learner rows x 128 steps), one minibatch per pass, so every
train kernel is a pass over 24576 x (256 | 512) fp32: the LN, pool /
unpool, relu_grad and prepass kernels are memory-bound at ~50-60% of the
bandwidth. Exact-class levers left on this side are each 1-3%: LN loads as
float4 (next), pool / unpool traffic (int32 winner -> narrower, unpool's
h2 re-read), relu_grad_from. The GEMMs (44 ms) move only with bf16
operands (gated).

### Train pass: LayerNorm lanes as float4, keep (marginal)

`ln_forward` / `ln_backward` (10.8 ms per update, memory-bound) loaded one
float per lane per step. Now templated on the vector width: float4 loads
and stores (bf16 outputs as 8-byte pairs) when the widths and leading
dimensions are multiples of 4 (all the shipped shapes: H 256, C 512, the
critic input 256 + 12), the scalar form otherwise. The lanes' partial sums
group the columns differently, so the moments round differently in the
last bit; `check_v4 --build` pass_all (tolerance gate against PyTorch),
`check_dump --only v4` ok. A/B 2x2 twice (cand = scalar behind a temporary
toggle, removed): 244.8k vs 244.0k (+0.3%), 245.1k vs 243.6k (+0.65%),
train 0.0993 vs 0.0997. Every cand below every inc in both, but ~0.4 ms of
the 10.8: the kernels were already close to the bandwidth. Kept as the
simpler code path forward; not a lever.

### Train pass: pool / unpool with 4 columns per thread, keep (marginal)

`pool_rows` / `unpool_rows` (8.9 ms per update) ran one thread per column
with 2-byte bf16 loads (64 bytes per warp instruction). `pool_rows4` /
`unpool_rows4` give each thread 4 columns (8-byte loads, float4 / int4
stores for the joined row and the winner), a warp per sample at D = 128, 4
samples per 128-thread block; the per-column scan over the rows is the
same sequence, so max / sum / winner are bit-identical (the scalar
kernels stay for D not a multiple of 4). `check_v4 --build` pass_all,
`check_dump --only v4` ok. A/B 2x2 twice (cand = scalar behind a
temporary toggle, removed): 246.1k vs 245.1k (+0.4%), 245.9k vs 244.8k
(+0.44%), train 0.0989 vs 0.0993. Like the LN: these passes were already
near the bandwidth; the remaining cost is the traffic itself (the winner
int32 per (sample, column), unpool's re-read of h2 for the ReLU mask).

### Train pass: count_rows as a warp per sample, discard

`count_rows` (the prepass, 1.7 ms per update: one thread per (set,
sample) walking up to 48 slots at a 19 KB stride, latency-bound) as one
warp per sample with a ballot per 32 slots. Counts identical (check_v4's
overflow numbers unchanged, pass_all). A/B 2x2 twice (cand = the old
kernel): 247.2k vs 246.7k (+0.2%, repeats overlapping), 247.0k vs 246.5k
(+0.2%). Inside the noise; reverted to the old kernel.

### Learning gate at ae63d96d: pass

`sanity_v4.sh run/report exact-ae63d96d` (the v4-aim recipe, seed 73, 160M
learner steps, 12.7 min of training at 243.6k learner_sps vs the
reference's 169.3k: x1.44 on the production recipe). Readout
(`sanity-exact-ae63d96d.json`): refit Elo 516 at 80M (floor 304), 574 at
160M (floor 571; reference 646, one seed each, spread 75); band rating of
the 160M checkpoint 475.7 [434, 515] vs the reference's 523.5 [479, 568],
floor 448.5: ok. explained_variance ratio 1.00, entropy 0.96, kl 1.66
(0.0057 vs 0.0034; flagged as drift by the readout; the in-run kl at the
213M step was 0.0045 vs 0.0043). passed: true.

Because the Muon changes (D != C, lanes) are not bit-identical, a
temporary self-check (BOPL_MUON_CHECK, removed) recomputed each eager step
from copies of the weights and the post-Nesterov gradients with the
pre-campaign sequence (in-place cuBLAS default GEMMs with the copies, one
stream) and compared the weights: worst max|w - w_ref| / max|w_ref - w0| =
0.6e-3 .. 3.1e-3 with TF32, 1.5e-4 .. 4.3e-4 with BOPL_TF32=0. That is
the amplification of the GEMMs' summation order through the five
Newton-Schulz steps (slope 4.08 at zero: ~1e3 on the smallest singular
directions), the same class as the tuner's algorithm choice the branch
already had; a race would show as O(1). No systematic difference.

### MinGRU scans: operands loaded in groups, discard (and what the scans are bound by)

`mingru_scan_forward` / `_backward` (0.25 + 0.46 ms per pass, min = avg:
not stretched by contention) read like latency-bound loops: one thread
per (sequence, unit), 128 dependent steps, one load per operand per step.
Rewritten with the operands loaded SCAN_P = 4 observations at a time
(forward: the next group issued before this group's recurrence; backward:
the chunk's 27 loads up front), the arithmetic sequence untouched:
bit-identical (a temporary device compare against the old kernels over 8
eager passes each: 0 of 25.4M / 25.2M words differ). Kernel time (CUPTI,
same build, toggle): forward 243 vs 248 us min, backward 432 vs 483. A/B
2x2 twice: 246.9k vs 246.4k (+0.2%), 246.6k vs 246.3k (+0.1%): noise.
Discarded (the experiment's three commits reset away; tip stays
bd540952 = ae63d96d's source).

Why: the scans are instruction-bound, not latency-bound. Per step the
log-space recurrence is ~5 accurate transcendentals (expf, log1pf, logf,
the softplus pair, logaddexp) plus a sigmoid: ~400 dependent instructions;
1536 warps x 128 steps x 400 = 79M warp-instructions over 48 SMs x 4
schedulers = ~0.2 ms, which is the forward's time. Cheaper math (__logf /
__expf intrinsics, or the fast-math variants) is not exact-class; the
grouped loads bought the 5-10% that was latency. Nothing left here
without a numerics decision.

### Rollout chain: br_gemm's A tile as float4 loads (commit 6028e9ca), keep

A microbenchmark of `br_gemm` alone (the chain's shapes at M = 64, /tmp,
not kept) put the learner's three 16-row tiles at ~3x the cost of a
bank's 4-row tile per launch: the A loads, not the weights. The A chunk
(16 rows x 64 k) was loaded as float2 with the lanes along the rows: 16
instructions per chunk, each touching 16 lines for 8 bytes. Now float4,
8 instructions per chunk, lanes still along the rows (the fragment
layout's row stride is 16 floats, so lanes along k would turn the smem
stores 8-way bank-conflicted; this keeps them 4-way as before). Same
values into the same smem slots: bit-identical (the microbench compared
the two paths word for word over 8 shapes x 4 row layouts, 32 cases, 0
differences; `check_dump --only v4` ok). Isolated per launch: fusion
15.1 -> 11.4 us, gru 7.65 -> 6.25, actor 4.5 -> 3.9, cin 6.8 -> 6.4,
b1/b2 8.7 -> 8.2 (the launch floor is ~4 us). A/B 2x2 twice (cand = the
float2 form behind a temporary template flag, removed): 249.2k vs 246.5k
(+1.1%), 250.0k vs 246.3k (+1.5%); train 0.0974 vs 0.0988, rollout
0.086 vs 0.088. `br_gemm_launch` now aborts on a row stride or base not
16-byte aligned (every chain input is a cudaMalloc'd buffer or the
4812-float observation row).

Running total: 169.0k -> ~249.5k (**+48%**, exact-class only).

### Rollout chain: br_gemm with the learner's row tiles sharing the weight loads, discard

After the float4 A loads the microbench's cost per byte moved was the same
for bank tiles and learner tiles, so I read `br_gemm` as traffic-bound and
built the traffic lever: one block per group covering all its row tiles
(RT <= 3), the weight chunk loaded once per block instead of once per row
tile (the learner's 48 rows re-read every matrix 3x: 3.9 of the 5.9 MB a
fusion launch moves), the A fragments gathered straight from global memory
(no smem staging), the k order / chunk -> warp assignment / reduction
order kept: bit-identical (32 of 32 microbench cases, 0 words differ).
Isolated it is slower, learner-only 7.95 -> 16.0 us (128 registers with
spills; the learner block's mma chain is 3x longer): at M = 64 the kernel
is latency-bound per block, and the traffic is not the wall. A/B 2x2 once
(cand = the old kernel behind a temporary toggle): 250.1k vs 250.0k;
rollout 0.097 vs 0.087 (+12%, now equal to the train pass), train 0.0967
vs 0.0973 (-0.6%: the freed L2 traffic is worth that much to the train
pass). No gain and the rollout's headroom gone: discarded, code reverted.

The transposed-mma bank tiles (halving the banks' tensor work) is the same
story: the bank tiles' cost is their weight bytes and the per-block
latency, neither of which that formulation changes. Not built.

### Train pass: relu_grad_from / relu_grad_bf16 four elements per thread, keep (marginal)

The last scalar elementwise kernels in the train pass (two ReLU backward
launches on B x 256 / B x 512 fp32, one B x 512 into the bf16 critic
operand) now do four columns per thread with float4 / uint2 accesses when
the widths and pointers allow (they always do for the shipped shapes; the
scalar kernels stay as the fallback). Elementwise, so bit-identical;
check_v4 5/5. A/B 2x2 twice against the scalar kernels behind a temporary
toggle: 250.7k vs 249.4k, 250.1k vs 248.4k (+0.5%, +0.7%); train 0.0970
vs 0.0975. Kept. Incumbent ~250.5k.

Exact-class list exhausted at this point. Everything left with real
weight is gated: bf16 operands on the TF32 GEMMs (44 of the train pass's
~63 ms of kernel time), the row budget (~10%), fast transcendentals in the
MinGRU scans (~3 ms per update). Waiting on Matthew's word for those.

### Learning gate at 5fb70ec9 (the whole exact-class stack): pass

`sanity_v4.sh run/report exact-5fb70ec9`: refit Elo 443 at 80M (floor
304), 574 at 160M (floor 571; reference 646); band rating of the 160M
checkpoint 555.6 [511, 599] vs the reference's 523.5 [479, 568] (floor
448.5). explained_variance ratio 1.28, entropy 1.02, kl 1.47 (0.0051 vs
0.0034; flagged as drift by the readout, as at ae63d96d). Production
recipe at 246.4k learner_sps vs the reference's 169.3k (x1.455).

### Campaign re-plan (Matthew, 2026-09-18 21:00)

Target 1M+ learner_sps at the production shape: 4x from here, which has
to come from less GPU work per learner step, not scheduling. No runtime
toggles: a kept lever is the code, a discarded one is reverted; A/B by
building the candidate against the incumbent commit's extension
(`BOPL_BENCH_AB_SO`, added to bench_shape.sh). The acceptance criterion
is the learning gate (sanity_v4 within band), not bit-exactness: numerics
(bf16 operands, fast transcendentals), the row budget default and reduced
precision through the chain are all allowed if they pass. Shape changes
(hidden, FFN, critic width, T, passes) stay Matthew's call: bench their
upper bound and report. Land on throughput-v4, no merge to main.

### Train pass: bf16 operands on the flat / fusion / FFN / actor GEMMs (b2418637), A/B +3.7%, gate pending

The trainer's remaining fp32 (TF32) GEMMs in bopl_v4.cu now run on bf16
operands: each fp32 operand is rounded to bf16 into a per-source scratch
once per pass (first use converts, later uses share it), fp32
accumulation and output, cublasLt epilogues as before. No fp32 path left
(lt_gemm32 removed). The MinGRU layer's three GEMMs (models.cu, puf_mm)
and the critic head gemv are still TF32. Build A/B against the 5fb70ec9
extension (`BOPL_BENCH_AB_SO`, bench_shape.sh): 259.0k vs 249.7k
(+3.7%), train 0.0938 vs 0.0973, rollout 0.0835 vs 0.0881.

CUPTI, train stream per update, 5fb70ec9 -> b2418637: TF32 GEMMs 30.0 ->
9.0 ms, bf16 GEMMs 26.7 -> 38.4, to_bf16x4 conversions 0.1 -> 5.4 (the
joined operand is 24576 x 1280 fp32 per pass), train busy 89.4 -> 86.6.
So the bf16 GEMMs run at ~1.8x the TF32 ones and the conversions eat a
third of the gain. Next, numerics-neutral: produce bf16 at the source
(d_bf16 epilogues for the flat / ffn_in outputs, LN's bf16 copy,
unpool_rows writing bf16) ~5 ms; MinGRU GEMMs to bf16 (9 ms TF32).

check_v4 vs the fp32 oracle: encoder 3.3e-3, logits 2.7e-3, value 4.4e-3
relative; grads q99 5-8e-3 (tolerances 2e-2 / 3e-2 hold; one per-tensor
max over 0.25: bf16 on the smallest gradient entries). Learning gate
`sanity_v4.sh run bf16-b2418637` decides.

Also in b2418637: the campaign's env toggles are gone (BOPL_BR_UNITS ->
1, BOPL_MUON_STREAMS -> 8 lanes, BOPL_TRAIN_PRIO -> rollout ahead), and a
v3 regression fixed: the br_gemm float4 A-load keep (86938d24) aborted on
v3's flat block (row stride 98, not 16-byte aligned); v4 benches and gates
never ran v3, check_dump did. br_gemm now falls back to scalar A loads
for unaligned rows (same values); check_dump passes v3 and v4.

### Learning gate bf16-b2418637: FAIL (160M refit Elo 544 vs floor 571)

`sanity_v4.sh run/report bf16-b2418637`: refit Elo 421 at 80M (floor
304, ok), 543.9 at 160M (floor 571, reference 646: fail); band rating of
the 160M checkpoint 495.0 [451, 537] vs reference 523.5, floor 448.5 (ok).
explained_variance 1.33, entropy 0.974, kl 1.78 (0.0061 vs 0.0034). 254.4k
learner_sps on the production recipe (x1.50). The exact-class stack passed
the same check at 574 twice and the refit's interval is ~+-150 Elo on 48
games, so one run 27 below the floor is within the gate's noise; kl 1.78
vs the exact stack's 1.47 is the only hint of an effect. Repeating the
gate with SEED=74 before discarding (bf16-b2418637-s74); if it fails too,
bf16 as built is reverted and a narrower variant (forward + dgrad bf16,
the K = 24576 weight-gradient GEMMs kept TF32) is the fallback candidate.

### Learning gate bf16-b2418637-s74: FAIL; backward-only bf16 +0.8%; bf16 discarded

Seed 74 repeat: refit Elo 443 at 80M (ok), 490 [365, 692] at 160M (floor
571: fail); band 468.3 [428, 509] vs floor 448.5 (ok); explained_variance
1.65, entropy 0.916, kl 1.19; 255.0k learner_sps. Two fails: bf16 on the
forward GEMMs is out. The signature (entropy falling in both runs) points
at the forward / PPO-ratio path: the train-side logprobs disagree with
the TF32 rollout chain's by ~3e-3 relative, a systematic ratio bias.

Backward-only variant (9c294026: weight- and data-gradient GEMMs on bf16
operands, every forward GEMM TF32 as before; check_v4 5/5): build A/B vs
5fb70ec9 251.8k vs 249.9k (+0.8%), train 0.0966 vs 0.0973. The operand
conversions (~5 ms per update) eat most of what the backward GEMMs save.
Not worth a gate at +0.8% for a numerics change: discarded, bopl_v4.cu
reverted to 5fb70ec9's (the toggle removals and the v3 br_gemm fix stay).

What would make bf16 pay later: emit the bf16 copies inside passes that
already touch the data (ln_forward already writes a bf16 output for the
critic path; relu_grad_from reads joined; pool / unpool write joined)
instead of separate to_bf16 kernels, backward-only: ~+4% for one gate.
Ranked behind the structural levers in the re-profile below.

### Row budget default 2,6,14,2 + MinGRU scans on hardware transcendentals (fc30bb2a): discard, reverted

Built together for one gate as asked. Build A/B vs e8f13657: 248.7k vs
248.9k (-0.1%), train 0.0978 vs 0.0977; cuda_mb 5.9 vs 6.4 GB; no drops
(set_rows_max 2 / 6 / 11.8 / 2 under a 14 entity budget). The budget's
compute gain was taken by active rows (431a53b4); what is left is memory.
The __expf / __logf / __fdividef scans are inside the noise (the scans are
3.5 ms of an 87 ms train pass). No throughput value in either, so no gate
for a numerics / drop-semantics change: reverted. `--v4-row-budget` stays
the way to trade 470 MB for a tighter drop threshold, Matthew's call.

### Re-profile at the incumbent (5fb70ec9-equivalent build) and the path to 1M

Wall per update 98 ms (24576 learner steps -> ~250k learner_sps; 1M needs
24.6 ms). CUPTI, 120 updates, w128 b4 p4:

Train stream: busy 89.4 ms per update (4 passes x 22 ms), span 105.
TF32 GEMMs 30.0 (wgrad 128x256 tn 9.9 over 20 launches, fusion/FFN dgrad
nn 7.0, fwd nt 4.5, ...), bf16 GEMMs 26.7 (set MLPs, critic block),
LayerNorm 8.8, prepass 6.2 (count / scatter / pack_flat: re-read of the
observations every pass, 4x per update), pool / unpool 6.0, MinGRU scans
3.3, ppo 2.3, other 4.4, memcpy 1.1. Muon on its lanes: ~1 ms critical.

Rollout, per buffer stream: 30 ms busy per update under contention
(br_gemm 13.0 at avg 11.2 us vs 3.5 isolated, br4_sets 7.1 at 55 vs 5.8,
HtoD 3.6, br_sample 3.2, br4_rows 2.8). Per buffer step 0.69 ms wall:
eval_gpu 0.41 (forward 381 us contended; ~80-100 isolated) + eval_env
0.27. CPU: c_step 33 us per env-step = sim 10 + finish 21.5 (the v4-aim
observation writer: rays + terrain probes), 2 decisions per env-step;
1.06 ms CPU per buffer-step; at 333k sps the sim uses ~5.5 of the 12
hardware threads (6 cores).

Upper bounds (shape, Matthew's call): --learner-passes 2: 315.4k (+27%),
rollout 0.077 now the bound (GPU forward 304 us still contended by a 54
ms train), train 0.054. --learner-passes 1 hits a pre-existing train.py
bug (KeyError 'entropy_coefficient': the entropy schedule with one pass).

Ceilings on this box at the current per-step costs: CPU 12 threads / 33
us x 2 decisions ~ 730k sps = 545k learner_sps with the GPU free; the GPU
side needs train <= 25 ms per update (3.6x) and an uncontended rollout
forward. 1M learner_sps needs all three: (1) passes <= 2 (recipe), (2)
train GPU work per pass down ~3x, of which the exact / fusion levers left
are worth ~1.3x (prepass once per update -4.6 ms, fused bf16 emission
backward-only -4 ms gated, LN / pool / relu_grad into GEMM epilogues -3
ms; the rest is the fusion GEMM's K = 1280 joined width and the set MLPs:
shape), (3) the writer's geometry off the CPU per-step cost (21.5 of 33
us: per-map precomputed terrain / ray fields, or the observation writer
on the GPU), the largest single rollout lever and a native/ (Rust) change.
Kernel-level work alone tops out around 280-300k.

### Row cache: the observation-side packing once per update (2c51f2dd): KEEP, +1.1%

Premise correction: the passes cannot share a prepass, every pass resamples
its minibatch (prio_replay draws segments with replacement). What is fixed
for the update is the slot's set of learner rows, so enc_precompute packs
those once per update with the existing kernels (count_rows, scatter_rows
into a slab of rows[s] slots per row, pack_flat) and the passes gather from
the cache: cache_counts (4 ints per row), cache_copy_rows (int4 copies of
the packed bf16 rows, the pass's scan / cap deciding the drops exactly as
before), cache_flat (float4 rows). Bit-identical: check_v4's gather case
(assert_array_equal on encoder, decoder, gradients) runs through the cache.
Trainer hook before the minibatch loop (pufferlib.cu), harness hook in
check_v4.cu. Build A/B vs e8f13657: 250.7k vs 247.9k (+1.1%), train 0.0970
vs 0.0982 (-1.2 ms of the estimated -2.5: the per-pass gathers cost ~0.5 ms
against the 1.5 ms they replace), rollout unchanged. Memory +470 MB (the
slabs hold the worst case, 48 entity / 32 slime / 24 slot / 6 player rows per
learner row, so no drop can happen before the pass's cap). New incumbent
build: inc-2c51f2dd.so.

### Epilogue fusions (flat / FFN DRELU, Lt-tuned weight gradients, LN rows): discard

Measured in the check_v4 timing harness (B = 24576, one encoder + decoder
pass isolated, 16.63 ms at the incumbent):
- ReLU bitmask from the flat and FFN-in GEMMs (fp32 RELU_AUX) and the
  ReLU' applied by the DRELU epilogue of the fusion dgrad (split into the
  flat and set column ranges) and the g_f dgrad, dropping relu_grad_from4:
  check_v4 5/5, but the fp32 AUX epilogue kernels are slower than RELU
  (enc.flat_gemm 0.35 -> 0.46, dec.ffn 0.80 -> 1.14) while DRELU saves
  ~0.1 ms of relu_grad: pass 17.04 ms, net worse.
- The long-K weight gradients (fusion, flat, FFN in / out; K = B) through
  lt_gemm32 with lt_plan's candidate timing instead of cublasGemmEx:
  bwd.enc.fusion_flat 1.87 -> 2.96, bwd.dec.ffn 1.56 -> 2.35. cublasGemmEx's
  choice is the better one here.
- LN_ROWS 8 -> 16: identical (ln3_bwd 0.303, ln2_bwd 0.360 ms: ~580 GB/s,
  within 1.5x of bandwidth; the LN headroom is ~0.1 ms per pass).
Nothing kept; bopl_v4.cu unchanged. Unpool cannot fold into the set dgrad
GEMM (the max-pool half is a per-column winner scatter, row-inseparable).

### Direction change (Matthew 22:51): v4-lite-aim is the production shape; recipe passes are the wall

v4-lite shape (set inner 64, FFN off, critic 384) on the v4.1 aim layout
(writer bit 8): sweep/configs/v4-lite-aim.toml. PufferLib 5.0's recipe is
async=1, replay_ratio=1.0 (each sample trained on about once); ours is 4
learner passes x 1.71 replay (THROUGHPUT.md line 584). The train-side wall
is the pass count, not the kernels. Plan: (1) lite-aim bench + gate, (2)
passes 4 -> 2 -> 1 on the lite shape, gated at equal wall-clock (Elo at the
GPU minutes the reference took for 160M), (3) re-measure the split, then
the writer geometry precompute, then w256.

Also landed first: 3d4bd503 runs the minibatch whose work was just captured
into a new train graph (it was skipped: one pass short per capture update;
with one learner pass the update trained nothing and the empty loss dict
raised train.py's KeyError entropy_coefficient). --learner-passes 1 works:
full shape 349.2k, rollout-bound at 0.0696 s per update (train 0.028).

Bench, lite-aim p4 (3d4bd503 build): 316.7k learner_sps (+26% over the full
shape's 250.7k), train 0.072 (from 0.097), rollout 0.0767 -> rollout-bound
already; GPU forward 297 us per step (contended), env 0.26 ms; 5.9 GB.
Gate lite-aim-3d4bd503 launched 02:58 UTC (reference v4-aim/000, same
layout; readout also against v4-lite/000).

### Gate lite-aim-3d4bd503 (v4-lite-aim, 4 passes): PASS

Run 160M in 579 s of trainer time (299k learner_sps in the sanity recipe
with evaluations; the bench says 317k). Against v4-aim/000 (same layout):
80M refit 490 [390, 588] vs 379; 160M 574 [443, 770] vs 646, floor 571
(pass, at the floor); band 496.5 [455, 542] vs 523.5, floor 448.5 (pass).
Metrics at 160M: EV 0.371 vs 0.299, entropy 4.37 vs 3.97, kl 0.0010 vs
0.0034 (all flagged as drift by the 10% rule: the smaller model moves less
per update). Against v4-lite/000 (same shape, old layout): 80M 490 vs 421,
160M 574 vs 574 (identical), band 496.5 vs 456.4 [412, 499]. So lite-aim
sits on the lite reference's curve and within the spread of the full
shape's. Reference for the lite shape from here: the shape comparison is
v4-lite/000; the wall-clock and layout comparison is v4-aim/000 (the
production trainer's recipe, 160M in 955 s), which is what the passes
table reads against. Both are the same recipe otherwise (config diff:
v4_shape only).

### Recipe: learner passes on lite-aim (bench, 3d4bd503 build)

  passes  learner_sps   sps     rollout_s  train_s  GPU fwd us  env ms/step
  4       316.7k        422k    0.0767     0.0721   297         0.260
  2       355.6k        474k    0.0683     0.0372   216         0.276
  1       376.0k        501k    0.0646     0.0197   177         0.287

Rollout-bound at every pass count: the buffer step is 0.50-0.60 ms of
which the GPU forward is 0.18-0.30 (isolated ~0.09: still contended by
the train stream and the other buffers) and the env CPU 0.26-0.29. Passes
1 -> 4 costs only 16% of throughput now; the rollout is the wall.

Equal-wall-clock gate (WALL_SECONDS=955, v4-aim/000's trainer time to
160M): each candidate runs STEPS past 955 s at its own rate, the
evaluation nearest 955 s is read out (refit Elo + band rating) next to the
reference's 160M numbers (646 refit / 523.5 band), and the 160M readings
give the sample-efficiency cost at equal steps. Launched:
lite-aim-p2-3d4bd503 (STEPS=360M).

#### lite-aim-p2-3d4bd503 (passes 2, seed 73, 360M learner steps in 1247 s)

In-run rate 310-319k learner_sps (bench 355.6k: the evaluation hook and
the box's other load take the rest); the p4 lite run did 299k against its
317k bench, so in the sanity recipe passes 2 buys only +4%.

  reading                         refit Elo        band rating (240 games)
  reference v4-aim/000 @160M      646              523.5 [478, 568]
  lite-aim p4 @160M (579 s)       574  (floor 571) 496.5 [455, 542]  pass
  lite-aim p2 @160M (547 s)       608  (floor 571) 429.5 [389, 467]  FAIL band (floor 448.5)
  lite-aim p2 @955 s (280M, 970s) 574 [400, 824]   547.8 [510, 586]

Equal steps: passes 2 costs ~70 band points against p4 at 160M (429.5 vs
496.5, intervals disjoint), the sample-efficiency cost Matthew asked for.
Equal wall-clock (the reference's 955 s to 160M): p2 reaches 280M learner
steps and its band rating 547.8 [510, 586] is at/above the reference's
523.5 [478, 568] (intervals overlap; refit Elo 574 with a wide interval
vs 646). Drift flags on the equal-steps report (EV 0.43 vs 0.30, entropy
1.15x, KL 0.4x) are the halved gradient-steps-per-sample showing in the
statistics, not a fault. Verdict: p2 loses at equal samples, ties or wins
at equal wall. Matthew's call; the p1 and p4 (extended to 955 s) rows
follow. Launched lite-aim-p1-3d4bd503 (STEPS=400M).

#### lite-aim-p1-3d4bd503 (passes 1, seed 73, 400M learner steps in 1335 s)

In-run 303k learner_sps (bench 376k). About 30 s of writer_bench runs on
core 11 landed inside this run (the rate dipped to 247k for those
updates), so its wall reading is a few percent pessimistic: not enough
to change the verdict.

  reading                         refit Elo        band rating (240 games)
  reference v4-aim/000 @160M      646              523.5 [478, 568]
  lite-aim p4 @160M (579 s)       574  (floor 571) 496.5 [455, 542]  pass
  lite-aim p2 @160M (547 s)       608               429.5 [389, 467]  FAIL band
  lite-aim p1 @160M (~530 s)      574               396.0 [353, 439]  FAIL band
  lite-aim p2 @955 s (280M, 970s) 574 [400, 824]    547.8 [510, 586]
  lite-aim p1 @955 s (280M, 921s) 747 [620, 950]    437.0 [395, 477]

Passes 1 loses at equal samples (-100 band vs p4) and at equal wall
(437 vs the reference's 523.5, intervals disjoint; its refit Elo of 747
is the bots-only fit, the band is the headline). At the same 280M steps
p2 rates 548 and p1 437, both seed 73: one gradient step per sample costs
more than the samples it buys at this shape. In the sanity recipe the
three pass counts run 299k / 313k / 303k learner_sps (p4 / p2 / p1): with
the evaluation hook and the box's load the rollout is the wall and the
pass count barely moves the rate. Launched lite-aim-p4w-3d4bd503
(STEPS=320M) for the p4 row at 955 s.

#### lite-aim-p4w-3d4bd503 (passes 4 again, seed 73, 320M steps in 1210 s) and the closed table

  reading                         refit Elo        band rating (240 games)
  reference v4-aim/000 @160M      646              523.5 [478, 568]
  lite-aim p4 @160M (run 1)       574              496.5 [455, 542]
  lite-aim p4 @160M (run 2)       544              472.7 [428, 514]
  lite-aim p2 @160M               608              429.5 [389, 467]
  lite-aim p1 @160M               574              396.0 [353, 439]
  lite-aim p4 @955 s (240M, 904s) 608 [517, 770]   526.5 [489, 566]
  lite-aim p2 @955 s (280M, 970s) 574 [400, 824]   547.8 [510, 586]
  lite-aim p1 @955 s (280M, 921s) 747 [620, 950]   437.0 [395, 477]

The two p4 runs share the seed and differ by 24 band points at 160M (the
env pool's timing is not deterministic): that is the run-to-run noise
floor for one-seed readings. At equal wall p4 ties the reference (526.5
vs 523.5) and p2's 547.8 is within that noise of it; p1 is out (437).
At equal samples p2 costs ~50-70 band and p1 ~90-100. Verdict: passes 4
stays; passes 2 buys nothing the noise can see at equal wall and costs
sample efficiency; the rollout is the wall at every pass count anyway
(299k / 313k / 303k in-run). Table closed; Matthew has the numbers.

### Writer geometry, exact form: refilling lanes + axis-aligned walls

Where the row's time goes (writer_bench, default loadout, maps 0-2, the
box under the p1 gate so absolute numbers run 10-20% high): v4-aim 7.9
us/row, of which geometry 5.8, aim/LOS casts + v4 rows 2.1, v3 0.6. A
debug build counted the march: per row 4.7 march calls, 63.5 rays, 80.6
vector steps, 380 lane-steps (6 steps per ray, lanes 59% occupied), 505
vector wall-SDF evaluations over 6.3 walls (2.4 axis-aligned, 3.9
rotated). So the geometry is ~500 x (20 zmm ops + a sqrt) plus the
per-step control: the algorithm's steps x walls, as THROUGHPUT_V4.md said.

What an exact form can still do:
1. Refilling lanes (the parked variant from prep, `BOPL4_MARCH` 40): a
   lane whose ray ended takes the next ray, so the fan's 36 rays march in
   one call with full lanes instead of 5 groups draining separately.
2. Axis-aligned walls skip the frame rotation: for right = (+-1, 0), up =
   (0, +-1) (or the 90-degree turn with the extents swapped) the IEEE
   value of rpx*(+-1) + rpy*0 is +-rpx, and the abs that follows kills the
   sign, so |lx| = |rpx| exactly: 6 of the 20 ops per wall gone on 38% of
   the walls. `Bopl4Terrain` carries the two wall lists; the min over the
   walls is order-independent (the same +0/-0 argument as the lane march).
Both bit-identical: writer_bench's march-check (vector vs scalar march,
8192 rows) 0 mismatches, and its new `rows_hash` (FNV-1a over every
production row of the recorded states) equal between the 3d4bd503 build
and the candidate on both loadouts (059772146b5e1a80 / 7e1f0b96aa258525).
Timing, interleaved on core 11 under the gate's load: geometry 5.15-5.47
(refill) and 5.10-5.17 (refill + axis) vs 5.84-6.51 base, v4-aim row 7.2
vs 7.9: -12-15% geometry, -8-10% per row, ~-1.3 us of the 32 us c_step
(the writer runs twice per step). Rollout A/B after the p1 gate.

Build A/B (`BOPL_BENCH_AB_SO`, the 3d4bd503 extension vs 2333ea99, lite-aim
p4, 3x3): learner_sps 326.8k vs 318.9k (+2.5%, repeats 326.2-327.4k vs
318.0-319.5k), c_step 30.6 vs 32.5 us, finish 19.5 vs 21.3, env
0.235 vs 0.255 ms per buffer step, GPU forward unchanged (~300 us). Kept:
exact and the predicted -1.9 us of c_step. New incumbent 326.8k (lite-aim).

What it cannot do: the remaining cost is steps x walls x 20 ops, and an
exact march cannot take fewer steps or skip a wall that might be the
minimum (L1 found nothing to cull at 6.3 walls). Merging the fan with the
drop/ceiling and aim casts into one 40-ray call (per-lane max) would lift
lane occupancy from 59% for maybe another -5%; not done. Beyond that only
the learning-gated forms move the geometry: L7 (8 dash landings, 3-unit
edge steps), L8 rays16, L9 16 march steps.

### Geometry, learning-gated form: aim-only layout (writer bits 10) on lite-aim

Evidence first (readout of the existing ablation sweeps, IQM-style last
three band ratings at 1735M learner steps, full shape, lr 3.43e-4):

  v4-aim (bits 8)      s73 675/712/705   s74 669/650/649   maps s73 709/706/718, r2 702/699/702, s74 678/689/689
  v4-aim-only (bits 10) s73 772/767/756   s74 685/676/682   s75 726/721/713
  v4-no-rays (bits 2)  s73 697/695/693   maps s73 689/656/663
  v4-rays16 (bits 12)  never ran

The v4-aim seed spread is ~650-718; aim-only sits inside or above it on
all three seeds (the geometry channel, fan + terrain block, was not
earning its 5 us per row at 1.7B on the full shape). No-rays is inside
the spread too but drops the aim/LOS casts (0.7 us) that aim-only keeps
and scores below it. rays16 has no evidence. So the largest cut with
evidence is aim-only: the whole geometry channel off, the two casts on.

Bench (b4caf432 writer, lite-aim p4, `--v4-ablate 10`): writer row 2.0-2.1
us vs 7.2 (writer_bench v4-aim-only vs v4-aim); c_step 17.3 vs 30.6 us,
finish 7.1 vs 19.6, env 0.157 vs 0.235 ms per buffer step; learner_sps
338.3k vs 326.8k (+3.5%) only, because with the env CPU cut the update is
train-bound at p4 (train 0.0717 s vs rollout 0.0615 s: wall = train). The
rollout side is now 0.48 ms per buffer step of which the GPU forward is
0.32 (contended by the train stream; isolated ~0.09). At passes 2 the cut
would show in full; the pass count is Matthew's. Gate launched:
lite-aim-only-b4caf432 (seed 73, 160M), read against v4-aim/000 and the
two lite-aim p4 runs (496.5 / 472.7 band at 160M).

#### lite-aim-only-b4caf432: pass on the band; bits 10 is the layout

160M (489 s): band 475.7 [436, 516] vs floor 448.5 (reference 523.5; the
two lite-aim p4 runs 496.5 and 472.7, so aim-only sits inside their
run-to-run pair). Refit Elo 466 [353, 617] misses the 571 refit floor,
as the p4 rerun did at 544 (the bots-only fit; the band is the headline,
program-v4.md). 80M refit 574 vs 379. Drift flags EV 0.43 vs 0.30 and KL
0.33x: the same statistics every lite run shows. In-run 336.5k
learner_sps against 299k for lite-aim p4 in the same recipe. Kept: writer
bits 10 is the production layout (v4-lite-aim.toml v4_ablate = 10,
sanity_v4.sh and bench_shape.sh defaults). Per row 2.0 us (from 7.2),
c_step 17.3 us (from 30.6).

### Passes reopened on aim-only: bench p2, equal-wall gate launched

With the geometry channel off the env CPU is 0.14 ms per buffer step and
the pass count moves the rate again: lite-aim-only p4 338.3k (train-bound,
train 0.0717 s vs rollout 0.0615), p2 448.3k (+33%, rollout-bound:
rollout 0.0541 vs train 0.0414; buffer step 0.42 ms = GPU forward 0.25
under train contention + env 0.14). The earlier equal-wall table was
taken at 313k vs 299k (p2 vs p4 in the sanity recipe), so p2 tied on Elo
with ~5% more samples; here it should have ~1.4x. Launched
lite-aim-only-p2-a2c9aa7e (seed 73, STEPS=440M, WALL_SECONDS=955 readout
against the aim-only p4 run, which needs its own 955 s extension: the p4
aim-only gate stopped at 160M / 489 s). Matthew picks.

### Train pass re-profile on the lite shape (CUPTI, a2c9aa7e, w128 b4 p4, 83 updates; prof-lite-aim-only-a2c9aa7e.txt)

Train stream busy 69.7 ms per update (bench train_s 0.0729), B = 24576
rows per pass, 4 passes. Per update:

  fp32 TF32 GEMMs (cutlass s1688)             16.9 ms  24%   128x256 tn 6.3 (12/upd, avg 517 us, min 199: the fusion GEMM family, K = 1280 joined), 128x128 nt 4.1, 256x128 nn 3.4, 128x256 nn 2.1, 256x64 nt 1.1
  bf16 set-MLP GEMMs (s16816, 8 per pass)     14.4 ms  21%   relu 128x256 3.0, 64x128 nt 2.3, 64x64 tn 1.9, relu 64x64 nn 1.8, 64x64 nn 1.7, relu 64x64 tn 1.5, 64x64 nt 1.2, drelu 128x128 1.0
  LayerNorm fwd + bwd                          5.7 ms   8%   ln_backward 4.1 (16/upd, avg 255 us), ln_forward 1.7
  Muon (grouped gemm + reduce)                 5.7 ms   8%   60 grouped GEMM launches of avg 80 us + 20 reduces, on the train stream
  pool / unpool                                5.6 ms   8%   unpool_rows 3.5, pool_rows 2.1
  MinGRU scans                                 3.3 ms   5%   backward 2.2, forward 1.1
  mask_rows (ReLU' for the set MLPs)           2.3 ms   3%   16/upd: set inner 64 is not a multiple of 128 bits, so the Lt DRELU bitmask is off and dh1 is masked by a pass over act x I
  cache_flat + cache_copy_rows                 1.8 ms   3%   the once-per-update prepass + per-pass gathers
  ppo_loss                                     1.7 ms   2%
  memcpy                                       1.2 ms   2%
  the rest (zero_tail, relu_grad, reduce_partials, embed, ...)  ~11 ms 16%

Rollout, per buffer stream (4): 23 ms per update each: br_gemm 9.1 (874
launches = 7 per step now that the FFN is off; avg 10.4 us, min 3.3),
br4_sets 4.8 (avg 38, min 5.7), H2D memcpy 3.7 (29 us per step: the
observation upload), br_sample 3.0, br4_rows 2.1. The avg/min ratios of
3-7x are the train stream's occupancy: the GPU is saturated (70 ms train
+ 4 x 23 ms rollout of stream time in a 73 ms update), so at p4 nothing
on the rollout side moves the wall.

Exact-class levers left on the train pass, ranked:
1. Muon 5.7 ms on the critical stream: the grouped path runs 60 launches
   per update at 80 us; with the lite matrices more of them are small.
   Fusing the Newton-Schulz iteration's GEMMs per group, or pushing the
   per-matrix Lt path to the side lanes: maybe -2-3 ms (3-4%).
2. mask_rows 2.3 ms: the Lt DRELU aux needs 128-bit rows (I = 64 is 64
   bits). Options: a custom bf16 K=128, N=64 GEMM with the ReLU' fused
   (br_gemm-shaped), or a bitmask written in the forward and a masking
   dW1 read. -2 ms (3%), medium effort.
3. Memory-bound passes (LN 5.7, pool/unpool 5.6, cache 1.8, zero_tail /
   relu_grad in the rest): ~16 ms at 50-60% of bandwidth. Narrower pool
   winners (int32 -> int8), unpool's h2 re-read, fusing zero_tail into
   unpool: each 0.3-1 ms.
4. The fp32 GEMMs (16.9 ms) have no exact lever left: TF32 already, bf16
   failed the gate twice. The K = 1280 joined width (flat 948 + pooled
   4 x 2 x D) is a shape question (flat_dim / a flat pre-projection):
   Matthew's.
5. The bf16 set GEMMs (14.4 ms) run 8 launches per pass with per-set M
   (players 49k rows, entities 295k); their tiles are 64x64 at I = 64.
   One grouped launch over the four sets would need a custom kernel.
Sum of 1-3 if all land: ~-6 ms of 70 (-9%). The pass count (p2: train
41 ms) moves more than all of them, which is why (a) is the gate running.
Also noticed: `env_int("BOPL4_DRELU", 1)` in bopl_v4.cu is a leftover
toggle (removal rule); to drop with the next bopl_v4.cu edit.

#### lite-aim-only-p2-a2c9aa7e (passes 2 on aim-only, seed 73, 440M in 1103 s)

In-run 399k learner_sps (p4 aim-only 336.5k: 1.19x in the sanity recipe,
1.33x in the bench). 160M (394 s): band 415.9 [374, 459], refit 516: the
equal-steps cost again (p4 aim-only 475.7). At 955 s the nearest
evaluation is 400M learner steps (1003 s): band 572.8 [529, 617], refit
747 [633, 950]; the reference's 160M band is 523.5 [478, 568] and lite
p4's 955 s reading was 526.5. The p4 aim-only run stopped at 160M, so its
955 s row is being taken now (lite-aim-only-p4w-04322d89, STEPS=360M).

### Toggles: BOPL4_DRELU, BOPL4_LT_TUNE, BOPL4_ACTIVE_ROWS removed (04322d89)

Fixed on; the env_int reader stays for the shape variables train.py
exports (BOPL4_SET_DIM, ...) and BOPL_V4_MEMLOG's allocation log (both
pre-campaign). bopl_rollout.cu's campaign-2 knobs (BOPL_BATCHED_FWD,
BOPL_ZERO_COPY*, BOPL_GRAPH_COPIES, BOPL_TF32, BOPL_OBS_SPARSE,
BOPL_PACK_TAIL) are not mine and were left; say if they should go too.

#### lite-aim-only-p4w-04322d89 and the aim-only passes table (seed 73, 240-game bands)

  reading                              refit Elo        band rating
  reference v4-aim/000 @160M           646              523.5 [478, 568]
  aim-only p4 @160M (run 1 / run 2)    466 / 516        475.7 [436, 516] / 472.7 [428, 521]
  aim-only p2 @160M                    516              415.9 [374, 459]
  aim-only p4 @955 s (320M, 983 s)     691 [502, 972]   549.4 [507, 594]
  aim-only p2 @955 s (400M, 1003 s)    747 [633, 950]   572.8 [529, 617]

In-run rates 336.5k / 325k (p4, the second under ~35 s of my nvcc runs)
vs 399k (p2): p2 reaches 400M learner steps by 955 s against p4's 320M.
At equal wall p2 is +23 band over p4 (both above the reference); the two
p4 runs at 160M differ by 3 and the lite p4 pair by 24, so +23 is at the
one-seed noise floor: p2 does not lose at equal wall, and does not
clearly win. At equal samples p2 costs ~57 band. Matthew picks; if the
wall is what matters p2 is at worst equal at 1.2x the rate (bench 1.33x).

### Candidates written while the gate ran (A/B pending, GPU-free next)

- 7e91f1b5 `dh1_relu_grad_gemm`: the set MLP's dh1 = (dh2 W2) * ReLU'(h1)
  as one bf16 mma.sync kernel (M x 64 of K = 128, 64-row blocks, W2 and
  the dh2 tile in shared memory, fp32 accumulators staged back for the
  masked bf16 store). Replaces the Lt GEMM + mask_rows (2.3 ms per update
  of re-reading dh1 and h1). Not bit-identical to Lt (summation order);
  check_v4 tolerance + the learning gate decide. Dispatch: I = 64 and
  D = 128 without an Lt bitmask; other shapes keep their paths.
- 7da18ef8 muon_grouped_gemm: 32-deep k slabs with the next slab's loads
  issued into registers during the current slab's FMAs (the loop was
  latency-bound: load, sync, compute, sync at 16 k per step). Exact: the
  per-thread k order is sequential as before, split-K bounds still round
  to 16, lanes past k_end add fmaf(0, 0, acc).

### w256 b4 at p2 on aim-only (bench, 04322d89 build)

  shape            learner_sps  sps    rollout_s  train_s  wall_s  buffer step ms (gpu / env)  cuda GB
  w128 b4 p2       448.3k       598k   0.0541     0.0414   0.0548  0.42 (0.25 / 0.14)          5.9
  w256 b4 p2       524.2k       699k   0.0926     0.0923   0.0938  0.72 (0.44 / 0.27)          10.8

+17% from twice the rows per round trip: the GPU forward costs 3.2 us per
row at 128 rows against 3.9 at 64, the env side scales linearly (c_step
17.4 us both; workers 56%, owner 21%, stolen 22%, 0.8 futex sleeps and
271 polls per buffer step), set_overflow 0 at rows_max 2 / 6 / 11.2 / 2.
Train and rollout are now equal (92 ms per update of 2x the samples), so
the next gain needs both sides. Memory 10.8 of 12.2 GB: w256 leaves ~1.4
GB, no room for w512. Doubling the batch is a recipe change (B = 49152
per pass, same passes): benched, not defaulted; Matthew's call together
with the pass count.

### dh1 fused bf16 GEMM (7e91f1b5): discarded, 0.995x

A/B 3x3 at lite-aim-only w128 p4: learner_sps 337.2k [336.2, 337.5] vs
339.0k [338.0, 339.4]; train_s 0.0720 vs 0.0716. check_v4 at the lite
shape passed (set-MLP grads 1.3e-3 rel, the incumbent's bf16 level). The
hand kernel (64-row blocks, 4 warps of mma.sync m16n16k16, W2 + dh2 tile
through shared memory, fp32 staged back for the masked store) is slower
than cuBLASLt's bf16 GEMM plus the separate mask_rows pass, so the 2.3 ms
of mask_rows is not the cost it looked like: the Lt kernel is far enough
from the memory roofline that a first-cut mma.sync kernel loses more than
the pass saves. Reverted (41785792). Not profiled per kernel; a CUPTI
look at the two would say whether a 128-row / cp.async version could win,
but the ceiling is ~2 ms per update (3%), so it goes to the back.

### Muon grouped GEMM prefetch (7da18ef8): discarded, 0.984x

A/B 3x3 at lite-aim-only w128 p4: 332.3k [332.2, 332.5] vs 337.8k
[337.7, 338.2]; train_s 0.0730 vs 0.0719 (+1.1 ms per update, i.e. the
Muon share grew from 5.7 to ~6.8 ms). The 32-deep slab with the next
slab's 16 loads held in registers (plus the 32 precomputed indices) costs
more in registers and per-slab smem writes than the hidden latency
returns at these grids; the sync count halving did not show. Exact by
construction but slower, so reverted; the incumbent .so restored from the
saved copy (no rebuild). What is left for Muon in the exact class is a
different design (a warp-level mma.sync tf32 path is not exact; a
persistent cooperative kernel over the 15 launches is the launch-gap
lever, worth ~0.3 ms per step at most) - moved behind the balance work.

### Code-only look while the w256 p2 gate runs: memory-bound passes and the H2D

Memory-bound train passes (per pass B = 24576, D = 128, packed rows per
set ~49k / 147k / 295k / 49k = 540k rows x 256 B = 138 MB of h2):
- pool_rows4: reads h2 138 MB, writes joined 4 x 25 MB (fp32, 2D per
  set) + winners 4 x 12.6 MB (int32): ~290 MB per pass = 0.48 ms at
  600 GB/s against 2.1 / 4 = 0.53 ms measured: ~90% of bandwidth. int8
  winners take 37 MB off (-0.25 ms per update).
- unpool_rows4: reads h2 138 + winners 50 + grad_joined 100, writes dh2
  138 MB: 426 MB = 0.71 ms ideal vs 0.88 measured (80%). int8 winners
  -37 MB (-0.3 ms per update).
- zero_tail: bytes are the active-rows granule (<= 3072 rows x 256 B),
  launch cost only, 8 launches per pass: fusing into unpool / the pack
  saves ~0.2 ms per update.
- ln_backward 255 us per launch, 4 per pass: at width 1280 fp32 the
  x + dy read and dx write are 378 MB = 0.63 ms ideal per pass vs 1.0
  measured (63%); a two-pass-in-one restructure is worth <= 1.5 ms per
  update, the largest of the four and the most work.
Sum if everything lands: ~2.2 ms of 70 (3%). The fp32 joined activation
(B x 1280 x 4 B = 126 MB written by pool + the flat GEMM, read by LN and
the fusion GEMM) is the traffic; halving it means bf16 operands into the
fusion GEMM, which is the change that failed the gate. Ranked behind the
recipe decision.

H2D: the obs row is still BOPL_OBS_STRIDE = 4812 fp32 dense, 19.2 KB,
uploaded whole per live row (one graph memcpy per buffer step, 64 rows
= 1.23 MB, 29 us measured at w128 = 3.7 ms per update per rollout
stream; w256: 128 rows, ~58 us of the 412 us forward). The geometry the
aim-only layout zeroes is 300 floats (terrain 12 + rays 36 x 8) = 6% of
the row: a compact row for bits 10 alone is worth ~2 us per step. The
row's bulk is empty set slots: players 6 x 32, slots 24 x 18, entities
48 x 48, slimes 32 x 42 = 4272 floats of capacity against at most
2 / 6 / 11 / 2 present rows (set_rows_max) = 784 floats, ~3 KB per row
typical. A packed transport (writer emits counts + present rows into
the pinned buffer, one memcpy of the packed bytes, a device unpack into
the same dense row the chain and the rollout buffer already read) is
exact by construction (1:1, check_dump on the unpacked rows) and cuts
the upload 4-6x: 29 -> ~6 us per step at w128, 58 -> ~12 at w256, i.e.
-3 ms of the 23 ms rollout-stream time per update and -45 us of the
0.72 ms w256 buffer step (6%) if the copy sits on the step's critical
path (it does: memcpy then kernels in the same graph). The writer
already zeroes only the slots it wrote last time (bopl_written), so
there is no memset to save on the env side. Effort: writer output
format (C), transport in bopl_rollout.cu, an unpack kernel; medium.
Worth it once the rollout side is the wall again (w256 p2 is balanced).

#### lite-aim-only-w256-p2-04322d89: the recipe table (aim-only layout, seed 73, 240-game bands)

In-run 443k learner/s average over 1175 s (489k between evaluations);
520M learner steps. The equal-steps 160M eval falls at run step
213385216 (w256's batch granularity) against the reference's 213352448,
so the readout's automatic band was skipped; the 160M-learner checkpoint
was rated by hand (band/run160). 80M refit 300.8 vs floor 304.1.

  recipe (aim-only)   rate      @160M band          @955 s: learner steps   band              refit
  reference v4-aim    169k      523.5 [478, 568]    160M (955 s)            523.5             646
  w128 p4             336k      475.7 / 472.7       320M (983 s)            549.4 [507, 594]  691
  w128 p2             399k      415.9 [374, 459]    400M (1003 s)           572.8 [529, 617]  747
  w256 p2             443k      389.8 [346, 433]    440M (992 s)            532.6 [495, 576]  574

Reading: at equal samples each halving of optimizer steps per sample
costs ~60 band (p4 -> p2: 475 -> 416; p2 w128 -> w256: 416 -> 390);
at equal wall the extra samples pay it back for w128 p2 (572.8) but not
for w256 p2 (532.6, 10% more samples than w128 p2 and 40 lower: about
1.5x the one-seed noise of ~24). Every row is above the reference at
955 s. If wall time is the objective the order is w128 p2 > w128 p4 ~
w256 p2 > reference; the batch doubling buys throughput (524k bench)
that the recipe does not convert into Elo at this horizon. Matthew's
call; the sample-efficiency cost of w256 is the honest caveat.

### Recipe decision (parent, 03:22): w128 p2 is the campaign recipe

The equal-wall winner on the aim-only layout (572.8 at 955 s vs p4's
549.4 and the reference's 523.5; ~60 band cost at equal samples).
learner_passes = 2 in sweep/configs/v4-lite-aim.toml, bench_shape.sh
(BOPL_BENCH_PASSES default 2) and sanity_v4.sh (--learner-passes 2);
both scripts now also carry the lite shape flags by default (set inner
64, FFN 0, critic 384), so the bench and the gate run the production
config without -- args. w256 stays a bench row (524k), not a default.
Matthew reviews before merge. The incumbent bench row from here is w128
p2 on aim-only: 448.3k learner_sps (rollout-bound: rollout 0.0541 s,
train 0.0414 s per update), so the next lever is the rollout side: the
packed obs transport (exact, check_dump-gated), then the split.

### Packed observation transport (2bcd3980): kept, +0.5% (marginal)

Design: bopl_obs_pack.h. The v4 row goes up as a 1536-float slot: header
(mode, four prefix lengths), flat 492 + privileged 48, then each set's
present rows (the writer fills sets as prefixes and zeroes behind them,
bopl_written), 6 KB against the 19.2 KB dense row. bopl_step_finish packs
the env's live rows (both c_step and the profiled step path); the buffer
step uploads the buffer's slots with one memcpy (the reward / terminal /
mask pack rides in the slots' idle tail) and bopl4_unpack_rows rebuilds
the dense device row the chain and the rollout copy already read. A row
that does not fit or has no writer record goes mode 1: the kernel reads
the mapped host row (rare; the init state and every reset mark all slots
dense). Exactness: dump mode (--dump-rows) runs bopl4_transport_check on
every buffer step; check_dump.py asserts 0 differing floats. 0 at the
tiny shape and at the production shape (w128 b4, 4 x 128 x 64 uploads).

A/B 3x3 at w128 p2 (the recipe) vs 04322d89: 450.9k [449.0, 451.4] vs
448.6k [448.6, 448.6] learner_sps, all repeats above; forward_us 248.5 ->
244.7 (-4 us of the 29 us memcpy), c_step 17.2 -> 17.7 us (the pack),
train_s 0.0415 -> 0.0409. Kept on the rule (candidate above the incumbent
on every repeat) but the gain is a tenth of the sizing: the memcpy ran on
the copy engine while the SMs were the wall, so removing 23 us of copy
per step shortened the step's latency chain by only 4 us, and the unpack
kernel costs SM time. The traffic cut (PCIe 7.7 -> 2.6 GB/s sustained,
1.3 GB less host-side row copying per update) matters more at w256 and if
the copy engine is ever shared. Incumbent: 450.9k at w128 p2 (2bcd3980).

The split at w128 p2 (2bcd3980): rollout 53.8 ms per update = 128 steps
x 0.420 ms; buffer step = eval_gpu 0.269 (GPU forward 245 us, isolated
90) + env 0.144 (env CPU 0.56 ms per step over 8 workers; c_step 17.7
us, workers 56%, owner 21%, stolen 23%). Train 40.9 ms runs under it.
Stream time per update: 4 rollout streams x ~19 ms of kernels + train 41
= ~120 ms in a 54 ms wall: the GPU is saturated and the forward's 155 us
over its isolated time is queueing behind the train stream and the other
three buffers. Levers that cut GPU stream time: the rollout kernels at
M = 64 (br_gemm 7 launches per step, latency-bound, avg 10 us), or more
rows per launch without changing the recipe: --buffers 2 at w128 (128
rows per launch, same samples per update, less env/GPU overlap). Bench
next.

Buffers at w128 p2 (2bcd3980), same samples per update:

  buffers  rows/launch  learner_sps  buffer step ms (gpu fwd us / env)
  2        128          396.2k       0.478 (266 / 0.188)
  4        64           450.9k       0.420 (245 / 0.144)
  8        32           408.0k       0.464 (222 / 0.089)

b4 is the optimum: at b2 the env half (1.13 ms of CPU per step across 8
workers) stops hiding under the GPU, at b8 the forward is launch-bound
(222 us for 32 rows). Not kept; the default stays. With the recipe fixed
at w128 p2 b4 and the GPU saturated (~120 ms of stream time in a 54 ms
wall), what is left in the exact class is stream time: the rollout
forward's kernels at M = 64 (7 br_gemm launches per step, avg 10 us,
latency-bound; br4_sets 38 us; sample 23 us) and the train pass's
memory-bound passes (~1.1 ms per update at p2). Anything larger is shape.

### Rollout-stream ranking at w128 p2 (CUPTI, 2bcd3980, 81 updates; /tmp/prof-p2.txt) and the first cut

Per rollout stream, per 64-row buffer step (avg / min us, the min is the
kernel nearly alone on the SMs, the avg is what it costs queued behind the
train stream and the other three buffers):

  br_gemm x7      66.9 (9.6 / 3.1 each)
  br4_sets        33.2 (5.9 min)
  br_sample       20.4 (8.8 min)
  memcpy HtoD     15.8 (8.3 min; the 6 KB packed slots)
  br4_rows x4     14.5 (3.6 / 1.3 each)
  bopl4_unpack     9.8 (2.7 min)
  br_gate          3.3 (1.1 min)
  memcpy DtoH      0.7
  sum             165 us of stream time per 420 us step; train stream 41 ms/update

Every rollout kernel runs 3-6x its isolated time: the rollout partition is
oversubscribed, so what pays is SM time removed, not launch count. Top by
that measure after br_gemm: br4_sets, whose set-0 blocks copied the dense
row gpu_observations -> rollout_obs (19 KB per row, 1.2 MB read + 1.2 MB
written per step) on top of the set MLPs. With the packed transport the
dense row is rebuilt by bopl4_unpack_rows anyway, so it now unpacks
straight into the rollout tensor's rows for the step and the chain (flat
GEMM, sets, privileged) reads them there; the copy in br4_sets is skipped
when obs == rollout_obs (the dense paths still copy). Exact by
construction: check_dump tiny v4 ok, production shape (w128 b4 lite-aim-
only p2) transport mismatches 0.

A/B 3x3 at w128 p2 vs 2bcd3980: 458.7k [457.7, 459.2] vs 451.1k [450.7,
451.5] learner_sps (+1.8%, all repeats above), rollout_s 0.0537 ->
0.0528, buffer step 0.420 -> 0.412 ms (eval_gpu 0.268 -> 0.265, env 0.144
-> 0.141), train_s unchanged. Kept: 3f921308, incumbent 458.7k.

Left on the hints: (a) br4_sets sized to present rows from the packed
header (the ballot scan itself is two warp iterations; the launch is 256
blocks x 256 threads per step regardless of n, and blocks with n = 0 exit
after the scan, so the win there is block count, not the scan), (b)
br_sample 8.8 us isolated for 64 rows: a dependent chain (max, exp, sum,
prefix, draw) per row, one launch already; folding it into the decoder
GEMM's consumer needs the whole logit row in one block.
