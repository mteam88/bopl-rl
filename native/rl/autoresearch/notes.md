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
Sustained workload: see the rows below once measured.

## Baseline profile (5931f7c, 16 workers, 2 pipeline groups, CUDA graphs)
Per update: rollout ~0.73-0.78 s, PPO ~0.04 s; collection ~94% of loop time.
Each group launch (one 4-tick decision for 8 games) ~3.1-3.2 ms env time, i.e. the
games themselves dominate; GPU util ~15%, ~6.3 busy cores of 12.
Startup 41 s = `my_vec_init` opening games serially.

## Experiment log
(see results.tsv for numbers; each entry: hypothesis, change, workload, outcome)

