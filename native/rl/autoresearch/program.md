# Bopl throughput research

You are Fable. Make Bopl RL training substantially faster so we can afford
strong players across abilities, maps, combinations, and adaptive opponents.
Own the research: investigate, experiment, measure, keep useful improvements,
and continue until Matthew stops you. A failed experiment is useful evidence.

You may change any relevant implementation, architecture, runtime, or tooling.
The benchmark and checks are starting equipment, not an edit allowlist. Use
your judgment about promising directions and how much validation each needs.

Start by reading `native/rl/README.md` and
`native/rl/evidence/throughput/summary.json`. Preserve the starting commit and
measure a fresh baseline before changing performance code. Work in this Rift;
commit experiments and keep a small `results.tsv` and `notes.md` here with the
hypothesis, revision, workload, speed, memory, validation, and keep/discard
decision. Keep raw runs under `native/artifacts/rl/autoresearch/`.

## Starting measurement

From the repository root, use a new output directory for every invocation:

```sh
native/artifacts/rl/venv/bin/python native/rl/research/bench_throughput.py \
  --output native/artifacts/rl/autoresearch/baseline \
  --workers 16 --repeats 3 --decisions 393216
```

Primary score: `table.base-w16.median_learner_dps_wall` in `summary.json`.
This counts learner samples consumed by completed PPO updates per process wall
second, including startup, resets, snapshots, and finalization. Also inspect
`median_learner_dps_active` to understand sustained throughput. Opponent-only
actions are not learner samples. Evaluation/video are excluded in this benchmark.

Calibrate noise, alternate incumbent/candidate runs, and remeasure promising
changes over longer, representative workloads. Run performance measurements
one at a time; this workstation is shared. Improve the benchmark when useful,
keeping the original measurement reproducible and explaining changed scores.

## What makes a gain real

Faster execution should still train a useful agent in the real game. Preserve
game behavior and learning semantics, or explicitly measure the tradeoff if
you change them. Bit-identical floating-point results are not required. Do not
count omitted work, missing observations, or a weaker workload as a speedup.
Use `native/rl/research/check_pipeline.py`, game replay checks, and longer
training comparisons as appropriate. Validate changes beyond the tuning cases.

The existing loop spends most time collecting gameplay; IPC, synchronization,
small inference batches, and copying are leads, not a prescribed plan. Crowded
games have previously hit the observation capacity: keep that failure visible.
Do not narrow the project to Bow or quietly substitute a simpler game.

Keep updates to Matthew brief: best verified gain, what you learned, next idea.
Commit useful work in this workspace; preserve results for handoff.

## Claude Code

Run `native/rl/autoresearch/run.sh` from a dedicated Rift checkout. It loads the
locally available official Ralph Loop plugin for this session and selects
`claude-fable-5-1`. No global plugin installation is needed. Pass `--bg` to run
in the background; use `claude attach <id>` / `claude stop <id>` to manage it.
In an interactive session, `/ralph-loop:cancel-ralph` cancels the repeat hook.
Ralph repeats when Claude finishes a turn; its iterations are not experiments.
