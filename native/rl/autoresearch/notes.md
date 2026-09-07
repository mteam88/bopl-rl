# Throughput autoresearch notes

Rift `throughput-autoresearch`, branch `autoresearch/throughput`, starting commit 5931f7c.
Results table: `results.tsv` (rows appended by `record.py` from each benchmark's `summary.json`).
Raw runs: `native/artifacts/rl/autoresearch/<dir>` (gitignored).

## Workloads

### w16-393216-r3 (original, startup-heavy)
`bench_throughput.py --workers 16 --repeats 3 --decisions 393216`: map 0, fresh policy, pool grows
from one version, 96 updates. Wall is ~118 s of which ~41 s is serial game spawning
(16 games x ~2.2 s handshake each). Kept for comparability with the first row.

### sustained-w16-786432-r3 (primary from 2026-09-07)
Resume `dap-multimap-001-s73` (10M-step multimap run, 16 workers, maps 0,1,2) at all-player
step 5017600 with its four most recent historical snapshots
(`sustained-w16-continuation.json`, schema-1 continuation context), maps 0,1,2,
evaluation maps 0,1,2, held-out 3,4, no eval/video, 786432 decisions = 192 updates.
Why: it is the shape of real training. Full 4-version pool -> 8 historical steppers
from the first update; the mid-training policy plays longer matches (episode length
~440-490 ticks vs ~390 from scratch) so reset/spawn costs occur at the real rate; the 100k
snapshot cadence recurs (~1 checkpoint + frozen copy + graph capture per 100k
decisions, ~2 per run); three maps exercise the different entity/slime observation
loads. Startup (~42 s) is ~20% of wall instead of ~35%.
`bench_throughput.py --start-step` makes decision counts relative to the loaded step
and checks the first metrics record actually starts there.
Primary score: median `learner_dps_active` (learner-consumed decisions per second of
train.py's own clock). `learner_dps_wall` (includes startup) is recorded alongside.

## Noise
w16-393216-r3 baseline, 3 runs: wall 2466.52-2516.03 (2.0% spread), active 3923.69-3994.89 (1.8%).
sustained-w16-786432-r3 baseline, 3 runs: active 3578.20-3868.54 (7.8% spread, median 3717.7),
wall 2821.05-3000.65 (median 2901.2). The sustained workload is noisier than the
short one; run 2 was slowest and overlapped a 10-15 s /proc sampling script of mine
(light, but noted). Treat single-run differences under ~8% as noise on this workload;
confirm candidates with alternating incumbent/candidate runs.
Sustained active is *lower* than from-scratch (3718 vs 3965): the full pool runs
8 historical steppers from update 1 and matches are longer, so it is the harder,
more honest target.

## Baseline profile (5931f7c, 16 workers, 2 pipeline groups, CUDA graphs)
Per update: rollout ~0.73-0.78 s, PPO ~0.04 s; collection ~94% of loop time.
Each group launch (one 4-tick decision for 8 games) ~3.1-3.2 ms env time, i.e. the
games themselves dominate; GPU util ~15%, ~6.3 busy cores of 12.
Startup 41 s = `my_vec_init` opening games serially.

## Experiment log
(see results.tsv for numbers; each entry: hypothesis, change, workload, outcome)

### exp01 — slim opponent sampling (commit 82a3e43) — INVALID (interrupted)
- Hypothesis: py-spy showed ~55% of main-thread time inside torch `Categorical` construction/sampling for the historical-opponent steppers. Replace `distribution(logits).sample()` with `policy.sample(logits)` (logsumexp-normalise, softmax, `torch.multinomial`), which replicates Categorical's kernels bit-identically, so opponent draws are unchanged for the same RNG state.
- Workload: `sustained-w16-786432-r3`, arms interleaved base (`BOPL_OPPONENT_SAMPLING=distribution`) / slim.
- Outcome: 2 of 6 runs completed (base-w16-1: 3698 active / 2866 wall; slim-w16-1: 4047 / 3079). base-w16-2 was killed after 150/192 updates when systemd-oomd killed peace.service (host cgroup of the agent shell + benchmark) at 2026-09-07T04:53:35Z (Avg10 pressure 89.5, 12.2G usage, MemoryHigh 12G). Single pair suggests +9.5% active / +7.4% wall, inside the ~8% noise band and possibly distorted by memory pressure inside the cgroup. Recorded as `invalid`; rerun as exp01b from a separate systemd unit (`bopl-throughput-fable.service`).

### exp01b — slim opponent sampling, rerun in the new unit — KEEP
- Same code (82a3e43 + record-keeping commit 56406e4), same arms, interleaved base/slim × 3 in `bopl-throughput-fable.service` (no oomd events).
- base (distribution): active 3802/3848/3850, wall 2962/3004/2979 → median 3848 / 2979. Recorded as `exp01b-control-distribution` (status baseline) because it is the incumbent measured under the new conditions: +3.5% active over the 04:20Z sustained baseline that ran inside the pressured peace.service cgroup, so the two are not directly comparable and the base arm is the control for this and following experiments.
- slim: active 4040/4047/4048, wall 3115/3128/3125 → median 4047 / 3125. **+5.2% active, +4.9% wall**, every slim run above every base run; the slim arm's run-to-run spread is 0.2%, the base arm's 1.3%.
- Lesson: the earlier ~8% noise estimate came from runs in the pressured cgroup; in this unit interleaved triples resolve ~2% differences. Sampling on the critical path is worth removing wholesale: exp02 moves it into the historical steppers' CUDA graphs.
- Record caveat: base-w16-3's `source/` snapshot (copied ~05:29:57Z) contains the exp02 edits to inference.py/selfplay.py/train.py, which I made in the working tree while the last runs were still queued and stashed about a minute later. The run itself behaved exactly like base-w16-1/2 (192 updates, 3802 active / 2979 wall): its train.py had been imported before the edit (the edited train.py at that moment would have raised NameError on `OPPONENT_SAMPLING`), and under `BOPL_OPPONENT_SAMPLING=distribution` the selfplay edits are behaviourally identical to 82a3e43. slim-w16-3's snapshot matches HEAD. Lesson: never edit the tree while a benchmark still has runs to launch; edit in a second rift instead.

### exp02 — in-graph opponent sampling (commit 9feb7c1) — KEEP
- Change: historical steppers are captured with `Stepper(sample='actions')`, so the forward pass *and* `policy.sample` (logsumexp normalise, softmax, one-sample multinomial) replay as one CUDA graph on the padded batch; `Opponents.act` slices the real rows. `BOPL_OPPONENT_SAMPLING=graph|slim|distribution` selects the path (default graph).
- Why it is exact: torch's one-sample multinomial draws element i from Philox subsequence i and advances the generator by a fixed 4 counters regardless of batch size, so the padded rows change neither the real rows' draws nor the generator offset. The padded *forward* is not bitwise equal to an unpadded forward (cuBLAS batch dependence), but train.py already ran the padded forward before this change, so the equivalence claim is "padded forward graph + eager sample of the real rows". `check_padded_sampling.py`: 50 cases (capacities 4/8/16/64, various k, 6 steps) identical actions, state and generator state; `check_pipeline.py` phase F: g2-graphs-slim and g2-graphs-distribution dump-identical to g2-graphs (`native/artifacts/rl/autoresearch/exp02-check-pipeline/result.json`).
- Benchmark (`exp02-graph-sampling/`, interleaved slim/graph × 3, same unit): slim active 4074/4104/4094, wall 3142/3168/3151 → median 4094 / 3151 (reproduces exp01b's slim arm within 1%); graph active 4457/4523/4527, wall 3362/3402/3405 → median 4523 / 3402. **+10.5% active, +8.0% wall**; every graph run above every slim run. Cumulative over the new-unit control: 4523/3848 = +17.5% active, 3402/2979 = +14.2% wall. Busy cores rose 6.6 → 7.3 (the games are fed faster); GPU util unchanged at ~17%.
- Per-update rollout time fell from ~0.72 s to ~0.65 s at unchanged PPO time (~0.04 s); startup is still 41 s of every run (24% of wall at this workload).
- What I learned: the main thread's Python per (t,group) is the lever, and each ~100 µs removed per group step is worth ~4–5% at 16 workers. The remaining Python in the loop is bookkeeping (game_decisions counting, script_workers scan, per-step tensor writes, `collect`), then the two `wait_group` stalls.

