# Bopl throughput research, campaign 3: policy v4 (draft 2026-09-18)

Make training with the v4 policy substantially faster without changing the
policy. v3 reached 1.36M learner sps at w256 (THROUGHPUT.md section 4); v4-aim,
the production shape since 2026-09-18, runs at ~165k. The product of this
campaign is the *same* v4-aim policy family at a multiple of that rate, so that
the multi-ability / all-map runs and their Protein sweeps become affordable.
Own the research: measure, hypothesise, change, gate, keep or discard, record,
continue until Matthew stops you. A discarded experiment with a number is
useful evidence; an unmeasured claim is not.

Read first: `native/rl/README.md`, `native/rl/THROUGHPUT.md` (campaign 2: the
method and the v3 numbers), `native/rl/THROUGHPUT_V4.md` (what is known about
v4, six profile hypotheses H1-H6, thirteen ranked levers L1-L13 with gates),
`native/rl/autoresearch/notes.md` (campaign 1: how the log is kept),
`POLICY_V4_PLAN.md` sections 1-3, 6 and 8.

## Workspace and ownership

Branch `throughput-v4` in the Rift child of the same name (main `4d66b2f8`
plus policy v4 `6c19394`, plus the Beam fix `de6a2a34` and the campaign
tooling). Commit per experiment on this branch. Sibling Rifts (`policy-v4*`,
`plumbing-124`) are other threads' and are not edited. One GPU; nothing else
trains while this campaign runs — the campaign owns the GPU until Matthew
says otherwise. Check `systemctl --user list-units 'bopl-*'` before every
GPU job anyway; if a training unit is active, stop and report.

## Measurement discipline

Score: `learner_sps` from `native/rl/puffer/bench_shape.sh` at the
production shape (`BOPL_BENCH_ENCODER=v4`, w128 b4 p4, 300 updates, first 20
dropped), with the per-phase fields in the same JSON row (`env_pool`,
`c_step.*`, `gpu_step.*`, `rollout_s`, `train_s`, `wall_s`, `cuda_mb`,
`set_overflow_max`, `load1`). Every row goes to
`native/artifacts/rl/bench-shape/results.jsonl`; every experiment gets an
entry in `native/rl/autoresearch/notes-v4.md` (hypothesis, change, commit,
workload, numbers, gate result, decision) and a line in `results.tsv`
(`record.py`, status `baseline|keep|discard|invalid`).

Order of work:

1. Baselines before anything else: `v3-base` (`BOPL_BENCH_ENCODER=set`) and
   `v4-base`, three interleaved repeats each at w128, then one pair at w256.
   The v3 number must be re-taken on this branch: `BOPL_OBS_STRIDE` is 4812
   floats for every row since the v4 merge, so THROUGHPUT.md's 1.14M/1.36M
   do not transfer. Then one CUPTI timeline of a v4 update
   (`BOPL_CUPTI_TRACE`, `cupti_timeline.py`). Decide H1-H5 from the fields
   and write the split down before touching code.
2. Free levers as A/Bs (`BOPL_BENCH_AB`, `BOPL_BENCH_REPEATS=3`): L3, L2, L12.
3. L1 (writer wall culling), then L4/L11/L5, then L6 only if rollout is
   still the binding phase — the order in THROUGHPUT_V4.md section 4.
4. Learning-gated levers (L7-L9) last and only with Matthew's word: they
   change the observation and therefore the policy family.

Noise rule: a change is kept only when, over three interleaved repeats,
every candidate repeat beats every incumbent repeat and the median gain is
above 3%; below that, record and discard. Compare rows only at similar
`load1`. Never quote a single run.

Gates, one command each:

- **exact** (default for every kernel, writer, transport and scheduling
  change): `puffer/check_dump.py` (CPU writer rows against the trainer's
  own rows, maps 0-2 and the bench's recorded states) and `puffer/check_v4.py`
  (GPU chain against `research/policy_v4.py`), both bit-identical or the same
  bf16 rounding of the same operands; `BOPL_TF32=0`, `BOPL_BR_CHECK=1` where
  they apply. `set_overflow_max` must stay 0 across the bench run for any
  row-budget change.
- **learning** (anything that changes a row, a sample, the loss or the
  schedule): `puffer/sanity_v4.sh run <label>` then `report` — 160M learner
  steps of the v4-aim recipe, refit Elo within 75 of `sweeps/v4-aim/000` at
  80M and 160M, band rating within 75, EV/entropy/KL within 10%. ~16 GPU
  minutes. Do not start a learning-gated experiment without saying so.

Do not count as a speedup: omitted rows or work, a weaker workload, fewer
opponents, a smaller shape, or a run measured while another job held the
box. Observation overflow stays visible in the bench row.

## Rules

- Same policy, same semantics. No observation, reward, delay, cadence, mask
  or reset change unless explicitly learning-gated and approved.
- No new env-var toggles in shipped code beyond what an experiment needs to
  A/B itself; a kept lever becomes the default and its toggle goes away in
  the same or the next commit. No "gates" or defensive fallbacks.
- No new tests unless they replace a manual check you actually ran.
- Keep `notes-v4.md` current as you go, not at the end. Updates to Matthew:
  best verified gain so far, what the split says, next lever. Brief.
- If a step needs something outside the Rift (a base lib rebuild, another
  thread's branch), write down the exact command and ask; do not do it.
- Memory: you are a subagent; do not run `memo`. Report candidate memories
  to the parent thread in your final message.
