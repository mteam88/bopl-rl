# Trainer bugs, 2026-10 audit

Written 2026-10-19 against main at 077646a0 (train/train.py, the PufferLib overlay
`train/pufferlib-bopl.patch` as built in `native/artifacts/rl/puffer-overlay`, train/bopl_v4.cu,
train/native_selfplay.py, league/pfsp.py, train/bopl_puffer.h rewards). The question is which bugs
make seeds differ for the wrong reasons or make a resumed run differ from an uninterrupted one.
E0 (research-e0, running from tp-spec) uses none of the fix flags, never resumes, and is unaffected
by this change.

Tests: `train/test_trainer_fixes.py` (CPU, 20 tests; run it as
`CUDA_VISIBLE_DEVICES= PYTHONPATH=<PufferLib> venv/bin/python train/test_trainer_fixes.py`) and
`train/research/check_trainer_fixes_gpu.sh` (GPU, about 2 min, **not yet run**: the GPU belongs to E0).

## 1. Known bugs

| # | Bug | Status | Commit | Test |
|---|---|---|---|---|
| 1 | The prioritized-replay beta doesn't anneal across `--resume`. Native `beta0 + (1-beta0)*alpha*epoch/total_epochs`: the epoch restarts in every process, and the `alpha` factor stops it at 0.84 instead of 1. The l2 main ran at about 0.21. | Flag-gated: `--prio-beta-fix 1` (beta0 -> 1 over learner steps) or `--prio-beta-const B`. Default is native. | 077646a0 | CPU: `PrioBeta` (Python schedule; native formula modelled). GPU: script part 2. None before this audit. |
| 2 | Latent row-drop leak. In bopl_v4.cu, `scan_rows`/`scatter_rows` cap each set's packed rows at `ceil(B*budget/64)*64` for the minibatch and drop the rows past the cap in batch order. A sample's encoding then depends on the samples before it, and training sees fewer rows than the rollout acted on (the rollout and the CPU evaluator drop none). | Flag-gated: `--v4-full-rows 1` or `P,S,E,L` caps. Default still drops. **Has not fired in E0** (below). | 077646a0 | GPU: `check_v4.py` `overflow`, `full_rows`, `full_rows_caps` (other half of the batch bit-identical). CPU: flag validation only; the kernel needs the GPU. |
| 3 | all_combos / combo_set credited A's side-1 games to B's kit (`a_kit=y` on side 1). | Fixed. | d5eb7065 | CPU: `AllCombosKits` (new; asserts `a_kit == kits[3*side:3*side+3]` on both tools). |
| 4 | The cosine LR restarts at every `--resume`. It then spans the whole `--steps` from the resume point, so it ends above its floor. | Flag-gated: `--anneal-lr-fix 1`. Default is native. | 077646a0 | CPU: `LearningRate`. GPU: script part 2. None before this audit. |

What remained open in the fixes for 1 and 4 (closed here):

- **Both fixes rewind when a resume grows `--steps`.** They follow `learner_steps / --steps`, and
  league_loop.py passes a larger `--steps` on every turn. With the fix on, a 10B -> 20B resume halves
  the progress at once: the LR climbs back up the cosine and beta drops. This change adds
  `--schedule-steps N`, which pins the horizon. It is recorded in `config.json` `fixes` only when set,
  so older runs resume without a spurious "fixes changed". A resume that would rewind prints a
  warning.
- **The fixes cannot span `--init-weights` stages.** Counters start at 0 in a warm start, so arm.py's
  anneal stage (and every ff1 arm) restarts beta at beta0 even with `--prio-beta-fix`. Only
  `--prio-beta-const` gives a schedule with no restart.

Bug 2 evidence from E0's metrics (read-only), per stage, up to 2026-10-19 18:56 EDT:

| run | rows logged | `set_overflow` nonzero | max mean rows per sample (P, S, E, L) |
|---|---|---|---|
| l2-s301/302/303 constant | 601 / 601 / 569 | 0 | entities 16.84 / 17.24 / 16.99 (budget 20) |
| obsfix-s301/302 constant | 601 / 601 | 0 | entities 16.70 / 16.46 |
| 4 anneal stages | 101 each | 0 | entities <= 16.59 |
| `prio_beta` (every stage) | | | first 0.200, last 0.840: bug 1's formula, restarted in each anneal stage |

The coverage is thin. `set_overflow` is the packing of the last training forward of one update, and
E0 logs one update every 10M steps. That is about 600 of 244k forwards (0.25%). The entity margin is
about 14%. This change adds `set_overflow_total` / `set_overflow_updates` / `set_cut_total` to each
metrics row: running sums over every update of the process, logging only. Resumes restart them. The
first minibatch of an update is still unread at `--minibatch-div 2`; covering it needs a cumulative
counter in bopl_v4.cu.

## 2. New findings

Severity counts impact on a seed-replicated comparison times the likelihood that it is live.

**N1. Medium, confirmed, fixed (records only).** A resume silently accepts changes to hypers that
`RESUME_HYPERS` doesn't name: `--gamma`, `--gae-lambda`, `--prio-alpha`, `--min-lr-ratio`,
`--minibatch-div`, `--mirror-share`, `--kl-coef`, the `--loadout-*` env knobs and so on. The run
trains with the new values. But `config.json` is `previous_config`, "the record of the run", so it
keeps the old `native` values and the record is wrong. Fix: `native_changes()` compares the recorded
native config with the new one, JSON-normalised. The budget and the hypers recorded elsewhere
(entropy, EMA) are left out. Differences go to `resumed[-1].changed.native`, `config.native` is
updated, and a line is printed. Numerics are unchanged. Test: `NativeChanges`.

**N2. Medium, confirmed, fixed (fails loudly).** Bank swaps and pool snapshots can fail silently.
`pufferl_load_frozen_bank` returns void. A missing, short or wrong-size file only prints to stderr and
leaves the bank on its previous weights, but `native_selfplay.step` still records `cur_opp_path = new`
and credits that bank's games and PFSP score to the new member. `save_weights` ignores `fwrite`'s
result, so a full disk leaves a truncated `pool/<step>.bin`. That snapshot then fails to load on a
later swap, and the same mis-crediting follows. The PowerSpec has run out of space on btrfs before
(2026-10-05). Fix: `native_selfplay.CheckedBackend` wraps the pool's backend. It checks the native
load's own condition (file size == the run's weight-file size) before every bank load and after every
snapshot write, and raises. It is **on by default**, which departs from the opt-in rule on purpose:
the check is the same condition the native code tests, so it can only fire where training was already
corrupt, and a healthy run's numerics are unchanged. Test: `CheckedBackend`.

**N3. Medium, confirmed by code, not fixed (needs a GPU A/B).** The PER importance weight scales only
the PG term. In the ppo kernel `wa = -w * adv_normalized`, while the value loss and the entropy bonus
are unweighted. `w = (N*P)^-beta` is not normalised. Over the sampling distribution, `E_P[w] =
sum P^(1-beta) N^-beta` is 1 at beta 0 and 1 and below 1 in between (log-convex in beta). So the beta
ramp is a hidden schedule on the PG-vs-entropy/value balance. Two consequences:

- (a) the native ramp 0.2 -> 0.84 changes the effective entropy coefficient during every stage and
  resets it in the anneal stage. This compounds the "anneal +0.05 confounded with beta" note in
  PLAN-2026-10.md.
- (b) in ff1, factor F (`prio-alpha 0`, so `w = 1`) also changes the effective entropy and value
  weight relative to the alpha 0.8 arms. F is therefore partly aliased with C (entropy).

Proposed fix: normalise `w` to mean 1 per minibatch (the standard PER max- or mean-normalisation),
opt-in, and log the mean of `w`. It would be a CUDA change in pufferlib-bopl.patch, so it is left
undone until it can be checked on the GPU.

**N4. Low, confirmed.** A resume replays the random streams from the start of the run. The
action-sampling Philox states are `rng_init(seed+i)` at create, the prio multinomial offset
(`rng_offset_puf`) is not saved, and the env seed is reused. All envs also reset together, so about
21 updates (10800 ticks / 4 / 128) see phase-locked episodes. This makes a resumed run differ from an
uninterrupted one beyond the documented recurrent-state reset. It doesn't bias seeds against each
other. Not fixed (an opt-in reseed per resume would be cheap).

**N5. Low, confirmed.** The first update after a resume (and after a fresh start) uses the
unnormalised button/hold entropy head scale, because `free` is None until a rollout's env log
arrives. That is one update.

**N6. Low, confirmed.** With overlap (`BOPL_ASYNC=1`), the checkpoint after update k already holds
the PFSP credit and bank swaps from rollout k+1. A resume discards that rollout untrained, and the env
counters exclude it, so pool statistics run one rollout ahead of `counts`. Negligible.

**N7. Low, latent, upstream.** `puff_advantage_row_scalar` computes `delta = rho*r + gamma*V' - V`,
while the vector kernel computes the V-trace `rho*(r + gamma*V' - V)`. The scalar kernel runs when
`horizon % (16/sizeof(precision)) != 0`, and the two differ wherever rho != 1 (minibatches after the
first). It is unused at horizon 128. Don't run a horizon that isn't a multiple of 8 without fixing it.

**Info (upstream design, checked, not bugs):**

- Advantage at the last index of each horizon segment is 0, and its value target is its old value.
  An episode ending on a segment's last action never trains its terminal reward. That is 1/128 of
  terminals, uniform in phase.
- MinGRU reset semantics agree between rollout (`br_gate`: reset before consuming obs t when
  `terminals[t]`) and training (`mb_terminals` -> `reset_ptr`). `check_recurrent.py` covers training.
- Training sequences start from the rollout-start state (`bopl_gather_start_states`). Under overlap
  that state comes from actor weights one update stale, which is standard truncated BPTT.
- EMA: reset after the warmup and at `load_weights`. `--resume` restores the double average after
  `load_weights` (correct order). arm.py's per-arm decay accounts for `--minibatch-div`
  (0.999996593 = 5B half-life at 24576 decisions per step, matching E0).
- Entropy schedule: a pure function of restored learner steps, correct across resumes.
- PFSP: draws use the restored `numpy` Generator over `sorted` names, so they are deterministic per
  pool seed. Scores are lifetime ledger records, never decayed, so `f_hard` lags the learner's
  current strength. A mid-match swap keeps the frozen bank's recurrent state from the previous
  weights (dropped from the statistics but trained against). Both are design choices to revisit, not
  silent bugs.
- Rewards (`bopl_puffer.h` `bopl_step_finish_with`): kill/death shaping, the outcome, the timeout
  charge to both players, the self-death charge and team-kill refiling check out. The telemetry
  reward tails match the paid reward.

## 3. Defaults

- **ff1: no schedule-fix flags.** arm.py never resumes (a resume taints the stage), so bugs 1 and 4
  don't fire. `--prio-beta-fix` would only move beta's endpoint from 0.84 to 1, and it would break
  the design's link that the all-low cell ff-1 replicates E0. `--prio-beta-const` would too.
- **ff1: `--v4-full-rows 6,12,48,8` only if the E+ arms still fit.** It is bit-identical unless a
  minibatch overflows (check_v4), so it can't confound the factorial. But it costs about 0.58 GB at
  minibatch 24576 and roughly twice that at E+ (`--minibatch-div 1`, 49152), against about 10.2 GB of
  the 12 GB card. Measure the E+ peak first. If it doesn't fit, keep it off: the new
  `set_overflow_total` will show whether the leak fires.
- **After ff1, as defaults** for league loops and anything that resumes:
  - `--anneal-lr-fix 1`
  - `--prio-beta-fix 1` with `--schedule-steps` pinned to the run's planned length, or
    `--prio-beta-const` (with N3 open, a constant beta is the only setting that doesn't move the
    PG/entropy balance)
  - `--v4-full-rows 6,12,48,8`

  Change defaults only after `check_trainer_fixes_gpu.sh` passes. The fixes leave fresh-run behaviour
  nearly unchanged, so no learning A/B is needed for them. A/B N3's weight normalisation before it
  becomes a default.
- **Already on by default (numerics-neutral):** `CheckedBackend` (N2), the resume record of native
  changes (N1), the rewind warning, and the cumulative overflow metrics.

## 4. For the parent, after E0

1. Run `bash train/research/check_trainer_fixes_gpu.sh` from a checkout with a current binding:
   - `check_v4` with `pass_all` and the full_rows cases;
   - the `native` arm's LR/beta must differ after the split (bugs 1 and 4 reproduced);
   - the `fixed` arm (`--anneal-lr-fix 1 --prio-beta-fix 1 --schedule-steps`) must match the
     uninterrupted run exactly.
2. Measure the ff1 E+ peak memory with `--v4-full-rows 6,12,48,8` before adding it to ff1.toml.
