# Bopl throughput research, campaign 3: policy v4

Make training with the v4 policy much faster without changing the policy.
v3 reached 1.36M learner sps at w256; v4-aim, the production shape, runs at
~165k. Same v4-aim policy, a multiple of that rate. Measure, hypothesise,
change, gate, keep or discard, record, repeat until Matthew stops you.

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

Use `BOPL_BENCH_AB` with repeats for candidates; keep a change only when the
gain is clear over the repeats' noise. Compare rows at similar load.

Every experiment: an entry in `native/rl/autoresearch/notes-v4.md`
(hypothesis, change, commit, numbers, gate, decision) and a `results.tsv`
line via `record.py`.

Gates:

- exact, for kernel/writer/transport/scheduling changes: `puffer/check_dump.py`
  and `puffer/check_v4.py`; `set_overflow_max` stays 0.
- learning, for anything that changes a row, a sample, the loss or the
  schedule: `puffer/sanity_v4.sh run <label>` then `report`, against
  `sweeps/v4-aim/000`. Say so before starting one.

Not a speedup: omitted rows or work, a smaller shape, a measurement taken
while another job held the box.

## Rules

- Same policy, same semantics. Learning-gated changes need Matthew's word.
- A kept lever becomes the default; its A/B toggle goes away.
- Keep `notes-v4.md` current. Updates to Matthew: best verified gain, what
  the split says, next lever. Brief.
- Anything outside the Rift: write down the command and ask.
- You are a subagent: no `memo`. Report candidate memories in your final
  message.
