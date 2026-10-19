# Red-team review of PLAN-2026-10.md

Written 2026-10-18 by an independent review subagent (read-only; no runs launched). S = T/native/artifacts/rl/sweeps,
L = T/native/artifacts/rl/league-loops/l2. Below, sigma = sd of one run's endpoint score:
sigma^2 = s_seed^2 + eval variance / n.

## Verdict

The measurement thinking is far above the norm: endpoint pinned by the anneal, MDE tables, honest re-grading. Four
problems still decide whether the 4 weeks produce real knowledge:
(1) the primary opponent f5 is a training opponent;
(2) the screen design (single-seed arms vs one shared control seed, about 12 arms, no false-discovery control)
will promote 1-2 false winners and has too little power to confirm +0.03;
(3) E0 measures seed variance at 2B+1B, but the screens run 6B+1B;
(4) the stage-1 skeleton fixes, and the dominant failure mode Stage 1 found (passivity and sudden-death deaths),
are not scheduled before E0, and no experiment targets that failure mode.
The plan also uses only 225 of 672 GPU-h. Spend the slack on seeds, not on more arms.

## 1. Statistical validity

- **f5 as the primary is a biased metric.** `final5-18666m`, `anneal20-26666m` (A20), and the exploit panel
  (`x2-drill/missile/blinkplat`) are all in L/state.json `current.opponents`. PFSP with p 3 upweights the
  opponents the main loses to, so "score vs f5" is partly a training target. Any arm that changes the opponent
  mix (all of E3, and indirectly E5/E4) changes how much it trains on f5. That is a direct confound with the
  metric. f5 and A20 are also one lineage (A20 -> final-3/4/5), so the panel holds one opponent, not two.
- **Fake power in the screens.** Take s_seed 0.02 and 4300 games: one arm minus one control has sd about 0.030.
  The +0.03 promotion bar is then 1 sd, so each null arm passes with p ≈ 0.16. Across about 12 screen arms
  (E1 2, E3 3, E4 3, E5 2, E6 2), expect about 2 false promotions. Because every arm is compared to the same
  control seed, an unlucky control promotes everything at once (correlated errors). Confirmation at 3+3 seeds
  has MDE ≈ 0.049 at s_seed 0.02, so it cannot confirm the +0.03 effects the screens target. Screen winners are
  also biased upward (winner's curse) if the screen seed is reused in confirmation.
- **E0 is mis-sized.** With 6 seeds, the 95% CI for s_seed spans a factor of 4 (×0.62 to ×2.45), so it cannot
  tell 0.015 from 0.03, and that is exactly the decision E0 has to make. Divergence between seeds grows with
  steps (2a showed nondeterministic divergence after one update, and the smoke run showed sqrt(k) drift), so
  s_seed at 2B+1B underestimates it at 7B.
- **Eval variance is not binomial.** The gate residual sd is 0.019 against the 0.025 binomial value (I recomputed
  this: 50 gates, OLS slope +0.0049 ± 0.0009 per 10B, lag-1 autocorrelation 0.22). Side-swapped pairs evidently
  cancel most of the setup variance, which fits the r ≈ 0 cross-net pairing result: side-swapping already
  removed what pairing would remove. The binomial n values (1555/4318/9715) are therefore about 1.7x too
  conservative. Use the empirical per-pair variance.
- **Matched anneal.** `--min-lr-ratio 0.05` from each arm's own peak LR does not give a matched anneal for E1:
  the 6e-5 arm has less noise left to remove. Anneal every arm to the same absolute floor LR. Also state the
  bias this design accepts: interventions that reduce noise (E1, E2, E6) have their raw-level gain erased by
  design, so for them the question is rate (level after anneal at 2 budgets), not level.
- **The cross-base "never compare" rule rests on one result at 2.4 se.** Under the model in section 7
  (setup×net interaction averages out over 2000 games), a 0.046 shift should not happen. Either bases differ in
  their (map, kit) composition or this was a fluke. Test it once on CPU (below) rather than encoding it as law.
- **Section 1 CIs ignore seed variance.** Every "refuted, >= +0.02 excluded" verdict is eval-noise-only from one
  seed. Re-grade with s_seed added after E0. Most "refuted" rows become "underpowered".

## 2. Confounds

- Lineage: all arms start from l2-main with Muon momentum cold (`--init-weights`). This is fine if shared, but
  results apply only to "warm, 100B constant-LR self-play" and not to from-scratch training. E8 distillation
  changes the starting strength, so score its init and compare gains from the init, plus equal-wall-clock.
- Pool drift: the frozen opponent list is good. Log the realized per-opponent game share per arm (PFSP adapts),
  so a data-mix effect can be told apart from an algorithm effect.
- Time and order drift: binary rebuilds (the obs fix, EMA), eval load and thermals all vary over the 4 weeks.
  Pin one binary sha per block, run cells in randomized order, and never run all of one arm's seeds back to back.
- CPU contention: harmless if rollout is synchronous; if the env workers run ahead of the learner, contention
  changes policy lag. Log sps and lag per run, and keep eval load uniform across the arms of a comparison.
- Eval seeds: check that base 211001 does not overlap the training env seeds 301-306 or any later seeds.

## 3. Priorities (672 GPU-h, one GPU, serial)

1. **Fix the skeleton before E0.** Section 7 lists the defects: owner-0 missiles and platforms, smoke crowding
   out entity rows, Spike not lethal in rays, Missile/Spike kills uncredited. Any E0 run before the fix must be
   redone. Order: fix -> obs-fix A/B -> E0 on the fixed checkpoint.
2. **Attack the measured failure mode directly.** 55% of games reach sudden death, 22% of losses are
   sudden-death sinks, 51% of loser deaths are unforced environment deaths, and the main idles in 9.7% of long
   games. A strong researcher would add a passivity/offense axis before tuning gamma: a sudden-death
   start-state curriculum, and behaviour metrics (sudden-death win rate, unforced environment deaths per game,
   ability presses per minute, time to first hit).
3. **Replace one-at-a-time screens with a fractional factorial.** Karpathy's one-change-at-a-time rule is for
   debugging (stages 1-2). For estimating effects under seed noise, a 2^(5-1) design (16 runs) estimates every
   main effect with se = sigma/2, against sqrt(2)·sigma for an arm vs a control: about 3x the variance
   efficiency at equal compute. All two-factor interactions also come out clean (resolution V), which E9
   currently pays for separately.
4. **Add an equal-compute framing and curve extrapolation.** The loop is concave (bins 0.483 / 0.502 / 0.508), so
   fit score = a + b·ln(steps) rather than the linear +0.07 per 150B in E10. Report every arm at equal GPU-h,
   which matters for E8 and E6.
5. **Add held-out and human-relevant metrics.** Measure exploitability as the score of a fresh 1B best-response
   exploiter against the release; keep the scripted-style panel; at release, run a 20-game human playtest and
   check reaction delay and fairness.

## 4. Concrete edits to the plan

- **Section 2, primary metric.** Remove `final5-18666m` and `anneal20-26666m` from the `--opponents` list of every
  program arm; control included. Primary = equal-weight mean score vs a held-out panel {f5, A20, team5b-6250m,
  g111-20b-26027m}, all four dropped from training, with 500 pairs each per endpoint. Report f5 alone as
  secondary. Confirmation and release use a fresh base, 221001, never seen in screening.
- **Section 2, endpoint.** Anneal every arm 1B to an absolute floor LR of 1.2e-5. For E1/E6, also score at 3B+1B
  (rate).
- **New step, CPU only, about 2 h:** score l2-main-133336m on 4 seed bases × 2000 games. If the sd across bases is
  within 1.5× the binomial value, drop the "never compare bases" rule; otherwise stratify scores by (map, kit
  pair).
- **New step, about 17 GPU-h: obs/reward fix A/B.** 4 fixed seeds against 4 control seeds, each 4B + 1B anneal.
  Adopt the fix if the held-out mean is >= control − 0.01 and missile/spike-kill credit works; the fix is a
  correctness fix, so non-inferiority is enough.
- **E0, 18 GPU-h.** 6 control seeds at the screen length (6B + 1B) on the fixed skeleton, and save a checkpoint
  at 2B to measure how s_seed grows. Size everything later from the 80% upper confidence bound of s_seed, not
  the point estimate. These 6 runs double as the control cell's replicates in the factorial.
- **E1+E3+E4+E5+E6 become one 2^(5-1) factorial, 48 GPU-h.** Use defining relation E = ABCD and randomized run
  order. The factors:
  - A: LR 2.43e-4 vs 1.21e-4.
  - B: gamma 0.995 vs 0.998.
  - C: discrete-head entropy at the current value vs 0.005.
  - D: exploiter cap, control banks vs `league,league,all,forgotten`.
  - E: minibatch-div at the current value vs 4.

  Every run carries EMA, which supplies E2 as a within-run contrast. Each run is scored on 2000 annealed games
  (as many runs as possible beats more games per run while s_seed >= eval se). Decision rule: adopt a factor if
  the Benjamini-Hochberg procedure at q 0.10 passes across the 5 main effects plus the EMA contrast, the effect
  is >= +0.02, and the exploit panel loses <= 0.05. Report interactions; adopt a pair jointly only if its
  interaction is >= +0.02 at |z| >= 2. Then confirm the adopted configuration vs control at 4 vs 4 fresh seeds on
  base 221001 (one-sided alpha 0.05, Holm if more than one candidate).
- **New E-SD, 12 GPU-h.** Sudden-death curriculum: reset 25% of training episodes to tick 7200 with randomized
  positions. Run 4 seeds and share the E0 controls. Predict >= +0.03 on the held-out mean and a >= 25% relative
  drop in sudden-death-sink losses.
- **Early stopping.** Never kill factorial cells (that breaks the balance), except on divergence: score vs random
  < 0.90, explained_variance < 0.5, or NaN. For one-off arms (E-SD, E8): futility at 3B if the raw held-out
  score < control-mean@3B − 0.05 with 1000 games.
- **E10.** The bar is the log-fit counterfactual of the loop plus its own end anneal, on base 221001 with 10k
  games. Add the fresh-exploiter exploitability and the behaviour metrics to the release card.
- **Budget.** obs A/B 17 + E0 18 + factorial 48 + E-SD 12 + confirmation 24 + E8 40 + E10 66 ≈ 225 GPU-h. Put
  the remaining ~440 GPU-h into 4 more seeds per confirmation and into resuming the loop. CPU eval (about 40 runs
  × 1 h) is the real wall-clock constraint, so queue evals continuously.
