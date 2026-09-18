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

(to be filled: v3-base, v4-base rows, CUPTI timeline, wall split)

## Experiment log

Each entry: hypothesis, change, commit, numbers, gate, decision.

### prep (CPU only, before the GPU was free)

- `record.py --shape` (this commit): results.tsv rows from bench_shape rows.
- Read program-v4.md, THROUGHPUT.md, THROUGHPUT_V4.md; the CUDA extension in
  this Rift was built 14:47 from the 14:41 sources (nothing newer), libbopl_sim.so 15:25.
