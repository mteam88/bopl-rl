# Bopl throughput research, campaign 3: policy v4

Make training with the v4 policy much faster. v3 reached 1.36M learner sps
at w256; v4-aim, the production shape, started at ~165k. Target: **1M+
learner sps at the production shape** (Matthew, 2026-09-18). 250k is a
waypoint. From the serial split (rollout ~72 ms + train ~63 ms of GPU work
per update) 4x has to come from much less GPU work per learner step, not
from scheduling. Measure, hypothesise, change, gate, keep or discard,
record, repeat until Matthew stops you.

Read first: `native/rl/THROUGHPUT.md` (how campaign 2 worked, v3 numbers),
`native/rl/THROUGHPUT_V4.md` (what is known about v4, hypotheses H1-H6,
levers L1-L13, suggested order), `native/rl/autoresearch/notes.md` (how the
log is kept).

## Workspace

Branch `throughput-v4` in the Rift child of the same name
(`ripped/.rifts/ExportedProject/throughput-v4`). Commit per experiment.
Other Rifts belong to other threads. The GPU is yours; still check
`systemctl --user list-units 'bopl-*'` before a GPU job and report if
something else is training.

## Method

Score: `learner_sps` from `native/rl/puffer/bench_shape.sh` with
`BOPL_BENCH_ENCODER=v4` (w128 b4 p4, 300 updates). The row's per-phase
fields (`env_pool`, `c_step.*`, `gpu_step.*`, `rollout_s`, `train_s`,
`set_overflow_max`, `load1`) tell you where the time went.

1. Baselines: `v3-base` (`BOPL_BENCH_ENCODER=set`) and `v4-base`, then one
   CUPTI timeline of a v4 update. Re-take v3 here: `BOPL_OBS_STRIDE` is 4812
   for every row since the v4 merge, so old numbers do not transfer. Write
   the wall split down before changing code.
2. Follow THROUGHPUT_V4.md section 4, re-ranking as the split says.

A/B by build, not by runtime switch: build the incumbent commit's extension,
copy it aside, and run `BOPL_BENCH_AB_SO=<that .so>` with repeats (the inc
arm runs the incumbent extension, the cand arm the checkout's build).
`BOPL_BENCH_AB` (env / flags) is only for train.py flags that already
exist. Keep a change only when the gain is clear over the repeats' noise.
Compare rows at similar load.

Every experiment: an entry in `native/rl/autoresearch/notes-v4.md`
(hypothesis, change, commit, numbers, gate, decision) and a `results.tsv`
line via `record.py`.

Gates. The acceptance criterion is the learning gate: `puffer/sanity_v4.sh
run <label>` then `report`, against `sweeps/v4-aim/000`, within band.
Anything that passes it is allowed: numerics (bf16 operands, fast
transcendentals, reduced precision through the chain), the row budget
default, dropped or reordered work. The exact gates (`puffer/check_dump.py`,
`puffer/check_v4.py`, `set_overflow_max` 0) stay as the fast pre-check where
they apply (kernel / writer / transport / scheduling changes that should be
bit-identical); a lever that is exact needs only those plus the A/B.

Shape changes (hidden, FFN, critic width, horizon T, learner passes) are
Matthew's call: bench their upper bound, report the numbers, do not make
them default.

Not a speedup: a smaller shape, a measurement taken while another job held
the box.

## Rules

- No runtime toggles or experiment flags. A kept lever is the code, a
  discarded one is reverted; fixed defaults in code. Diagnostic env knobs
  that only switch scheduling or precision paths go too.
- Land on `throughput-v4`; do not merge to main.
- Keep `notes-v4.md` current. Updates to Matthew: best verified gain, what
  the split says, next lever. Brief.
- Anything outside the Rift: write down the command and ask.
- You are a subagent: no `memo`. Report candidate memories in your final
  message.

## Plan update 2026-09-19 (Matthew, via the parent)

- Shape / layout levers (set D, hidden, flat layout, critic width) are
  closed: c1 (set D 128 -> 64) declined; keep the per-entity width. Shape
  is not on the plan any more.
- The road is (b) recipe levers gated at equal wall-clock (955 s, band
  rating inside the p2 aim-only run's run-to-run spread, v4-aim/000 as
  the floor), decided by the campaign and reported per gate, and (a)
  exact-class CPU / kernel work (check_dump / check_v4 / lockstep +
  throughput A/B).
- Two tracks: a CPU-track step runs inside the same turn as a GPU gate;
  a turn that launches a gate ends by saying when to poke.
