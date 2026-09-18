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
