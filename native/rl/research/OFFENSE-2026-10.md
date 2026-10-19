# Offense: why the main is passive, and a powered experiment to fix it (2026-10-18)

Written by the offense-design subagent. Design and analysis only; no GPU used. Paths: T = tp-spec,
S = T/native/artifacts/rl/sweeps, L = T/native/artifacts/rl/league-loops/l2, D = S/game-diag/diag.
Tools: T/native/rl/research/offense/ (offense_metrics.py, hazards.py, offense_diag.py, run.sh, probe.sh).
All intervals are 95%: Wilson for proportions, a seed-pair bootstrap for scores, medians and ratios.
Main = l2-main-133336m in D, l2-main-136002m in the new probe (section 5).

## 1. Diagnosis

**The game the main trains on is decided by the environment, not by combat.** Main vs f5, all kits
(D/mf5-*, n 1000 games, 986 decided):

| | before sudden death | sudden death (tick >= 7200) |
|---|---|---|
| both-alive exposure | 1333 min | 139 min |
| decisive-death hazard | 0.34/min [0.31, 0.37] | 3.9/min [3.5, 4.2] |
| of which credited kills | 0.17/min | 0.65/min |
| of which unforced env deaths | 0.11/min | 2.56/min |
| main's score | 0.503 (452 games) | 0.521 (548 games) |
| main's wins without a credited kill | 117 of 226 (52%) | 235 of 280 (84%) |

- The main's own credited-kill rate before sudden death (KH_PRE, section 2) is **0.083/min [0.066, 0.100]**:
  one kill per 12 min of both-alive play. Main vs main: 0.055 [0.031, 0.084] (n 300). Main vs rusher-hard:
  0.72 [0.61, 0.86] (n 1000).
- Of the 505 "unforced" losing deaths, 411 are ring-outs/falls, 55 squishes, and 38 (7.5%) are actually
  uncredited Spike (31) or Missile (7) kills. 354 of 505 (70%) happen in sudden death.

**Why passivity is optimal, or at least stable:**

1. **Reward: a free win pays 78% of a kill.** A win by credited kill pays 1 + 0.288 = 1.29. A win by the
   opponent's own fall pays 1.0. Every death costs −1.29, own falls included. Of the main's win reward vs
   f5, the kill bonus is **8%** (0.198 of 0.550 per game comes from credited wins, of which 0.288/1.288 is the
   bonus). Waiting for the opponent to fall pays almost as much as killing it and carries none of the
   approach risk.
2. **Self-play makes combat zero-sum and movement negative-sum.** Against a copy of itself, attacking
   raises both players' kill hazards by about the same amount, so it nets to zero. Approaching means
   leaving safe ground, though, and unforced falls are the largest loss cause before sudden death
   (0.11/min, about equal to each side's credited-kill hazard). The best response is to stand still and
   let the opponent fall. Two pieces of evidence:
   - PPO from the aggressive human-BC init drifted to f5-style passive play unless the KL-to-BC beta was
     >= 0.03 (2026-10-15, docs/HUMAN_BC.md).
   - The aggressive September specialist v4-full-2313m scores only 0.165 [0.115, 0.220] vs f5 on its own
     Dash/Bow/Platform mirror (S/aggression, n 400). f5 waits farther away (36 vs 27 units) while v4-full
     dies on its own.
3. **Gamma 0.995 is blind to the clock.** At 15 decisions/s the half-life is 138 decisions = 9.2 s. A
   terminal reward at the median 121 s has weight 1.1e-4 at kickoff. Sudden death has weight 0.011 at
   60 s and 0.22 at 100 s. The 180 s timeout (−1, 0.7% of games) has 0.011 at the start of sudden death.
   - The timeout and the sudden-death "deadline" therefore exert no pressure before about 100 s, so
     neither is a lever.
   - With per-decision hazards h_w and h_l, V ≈ (h_w − h_l)·R / (1 − γ + h_w + h_l). Here h ≈ 4e-4 per
     decision, much less than 1 − γ = 5e-3. In that regime, raising both hazards rewards tempo only when
     the agent is ahead. In the mirror, h_w ≈ h_l, so tempo is worth nothing. A higher gamma makes play
     *more* tempo-neutral; it is not an offense lever (section 3, I5).
4. **Sudden death rewards survival, and survival is the skill that has no value against humans.**
   - 55% [51, 58] of sim games reach sudden death, and in those games the loser falls unforced 76% of the
     time.
   - In 393 human 1v1 seats (S/aggression-human, 17 sessions), the median round is 10.8 s, **0 of 393 reach
     120 s**, and humans win 0.69 (mostly September bots).
   - Training spends over half its games on an endgame humans never reach, and is weakest exactly where
     humans play: fast rounds with a decisive-event hazard of ~5.5/min, against 0.34/min in league play.
5. **The opponent mix rewards camping.**
   - Kits with no weapons (Dash/Dash/Platform, Dash/Drill/Platform...) score **0.66, 0.71, 0.70, 0.71**
     when *the main itself* plays them against the main (camper exploit-eval control column, L/turns.jsonl
     turns 1/31/57/85, 400 games each).
   - The camper exploiter still adds +0.08 [+0.03, +0.13] at turn 85, against +0.15 at turn 1. In 100B
     the main never learned to kill a pure survivor.
   - Exploiters fill ~30% of the main's games, at learner score 0.13-0.22 (PLAN section 0). The rusher
     is 2.3% of games, so it barely registers.
6. **Missing kill credit is real but small overall.** Uncredited Missile and Spike kills are 3.9% of
   decisive deaths vs f5 (38/986). Crediting them is worth ≈ +0.011 reward per game: needed for
   correctness, and it matters for Missile/Spike kits (vs x2-missile: 0 credited of 247 decisive
   deaths), but it cannot move aggregate offense on its own.
7. **The button heads are collapsed, which makes "never press" a fixed point** (S/model-checks,
   2026-10-18). Buttons and hold entropy sit at 1-1.5% of their maximum (median 0). The effective
   coefficients are 1.4e-5 (aim), 7.7e-4 (buttons) and 5.1e-4 (hold), which is negligible against
   normalised advantages. A deterministic no-press policy gets no exploration pushback, so the idle games
   (7.8% [5.6, 10.4] of long games vs f5, tied to the setup) are expected to persist. Offense that needs a
   new button sequence would be found only by chance.
8. **The net barely reads projectiles.** Ablating the dynamic entity rows still leaves 0.43 vs the
   unablated main; for comparison, removing opponent state leaves 0.29 and map inputs 0.05. A policy that
   ignores projectiles can neither dodge nor aim well at close range. That makes engagement riskier
   than it needs to be, and v4.2 (owners on missiles and cast platforms) is the precondition for fixing
   it.
9. **Early self-deaths on fixed setups are a different failure.** They are reproducible on some setups,
   even against random (seed 211020, side 1, map 53: deaths at 151-240 ticks). In main vs f5 they are
   29 of 1000 games: an unforced or self death before 1000 episode ticks, which is 6% of the main's 480
   losses. That is a spawn and kit blunder, not passivity. It is counted in "unforced deaths" but needs
   its own fix (a fixed-setup regression list in the eval), not an offense arm.
10. **Observation delay (9 ticks = 150 ms): no evidence it causes passivity.** It is verified exact
   (S/watch-games), symmetric in the mirror, and close to human reaction time. It is not a target.

**Prior aggression work, graded by sample size:**

| Work | n | Result | Grade |
|---|---|---|---|
| eval/aggression.py, 2,4,14 mirror (S/aggression) | 400/cell | v4full fastest and most aggressive (20 s vs mixed-v2-hard, 12.6 arrows/min, 4.7 dash/min) but loses to f5 0.165; f5/team5b/A20 0.67-0.71 vs the bot in 26 s | Solid. Caveat: the `kills_pm` and `first_kill` columns equal `arrow_hits_pm` and `seconds` in every row (likely a copy bug in the summary) |
| Scripted rusher/pursuer (S/aggro-scripted) | 400/net | rusher-hard vs nets 0.24-0.39 in 20-30 s; nets set distance ~19-23u | Solid. The rusher is a weak human proxy |
| Anticamp punisher-1/2 (S/anticamp; pressure + punish_spent_mobility + arrow_hit styles, timeout −1, vs frozen f5) | 300-400 | On its kits 0.616 [0.57, 0.66] vs f5's own 0.53, games 70 s vs 83 s, kills/min 1.9 vs 1.2; off-kit 0.37-0.41 vs f5's 0.475 | Solid, but an exploiter: shaping makes a net more aggressive and better vs f5 where it trained, worse elsewhere. Not a main-agent test |
| l2 loop campers/punishers as main exploiters | 4 rotations × 400 | Main never closes the camper gap (+0.15 → +0.08); punisher +0.12-0.24 | Solid: opponent-mix alone does not teach offense in 100B |
| selfdeath-1 (`--reward-self-death`) | 1000 | pen vs ctl 0.5035 [0.470, 0.538] | Underpowered null; it targets safety anyway |
| Human BC → PPO | n not logged | Drifts to passive play unless beta >= 0.03 | Qualitative but consistent |

## 2. Metrics, fixed in advance

These are computed by offense_metrics.py on game_diag/offense_diag rows. Fields marked * also come from
`all_combos.py --games-out` rows (telemetry death classes), which agree within 1% on D/mf5:
KH_PRE 0.0825 vs 0.0818.

| Metric | Definition | Main vs f5 now (n 1000) |
|---|---|---|
| **KH_PRE*** (primary offense) | A's credited kills / A's both-alive minutes before tick 7200 (ratio, bootstrap) | 0.083 [0.066, 0.100] |
| sd* | share of games decided at or after sudden death | 0.548 [0.511, 0.584] |
| len_s* | median game seconds; secondary only, since the distribution is bimodal (§5) | 121 [119, 123] |
| unforced deaths | A lost by an uncredited death with nothing of B's within 3u | 0.236 [0.206, 0.265] |
| free_win | A won by B's unforced death | 0.269 [0.238, 0.298] |
| idle | A pressed abilities on < 0.5% of decisions, games >= 750 decisions | 0.078 [0.056, 0.104] (n 676) |
| first_hit_s | median s to A's first threat (own entity < 3u of the living opponent; censored at game end) | section 5 |
| engage_s, close_frac | s to bodies < 12u; share of both-alive time < 12u | section 5 |
| ckpm*, ckpm_b* | credited kills by / of A per game minute; ckpm_b checks reckless trading | 0.107 / 0.110 |

- **Strength stays primary**, following the plan revisions. It is the mean score vs the held-out panel
  {final5-18666m, anneal20-26666m, team5b-6250m, g111-20b-26027m}, all-kit, fresh setups, K = 1, seed base
  211001 for screens and 221001 for confirmation. All four nets are dropped from every arm's
  `--opponents`, the control included. KH_PRE and the * metrics come free from the same games.
- **Human-proxy panel (report only):**
  - rusher-hard, `--kits bot`;
  - anticamp-punisher-4000m and l2-me29-punisher-2000m on their pinned kits, `--kits league`;
  - v4-full-2313m on the 2,4,14 mirror, `offense_diag --kits fixed`.
  - 400 games each. Report score and len_s.
- **KH_PRE power:** at 2000 games per seed, se ≈ 0.006 per seed (relative 7%). Over 4 seeds per arm, a
  +20% relative change (0.017) has se_diff ≈ 0.004 before behavioural seed variance. E0 measures that
  variance for free.

## 3. Interventions, ranked

Every arm runs on observation v4.2 (commit 03c66236, `--v4-ablate 16`: Missile/Spike kill credit,
caster owners on missiles, cast platforms and smoke, lethal spikes in rays, smoke out of rays) and keeps
the l2 config otherwise. That includes E0's controls, so KH_PRE counts Missile/Spike kills on both
sides of every comparison. Frozen bank members keep their own writer bits. Style files are in T/styles and compile cleanly with train/style_compile.py.

**I1. Early-kill premium (styles/offense_kill.toml). Rank 1.**
- **Hypothesis:** passivity is reward-optimal (§1.1-1.2). Paying the killer only, never mirrored as a death
  penalty, 0.3·max(0, 1 − now/6960) per credited kill makes trading worth it before sudden death. A
  credited win at kickoff then pays 1.59; a free win stays 1.0.
- **Prediction:** KH_PRE ≥ 1.3× control, sd −0.08, len_s −15 s, panel score ≥ −0.02.
- **Falsified if** KH_PRE < 1.15×. Then offense is skill/exploration-limited, not reward-limited: see I2.
- **Change:** `--style styles/offense_kill.toml --style-fade-steps 1000000000000000000` (permanent) in both
  the 6B phase and the anneal.
- **Hacking risk:** low. One life means no farming. Reckless trades show up as ckpm_b ↑ and in the
  strength guardrail. In self-play a losing seat is indifferent to how it dies, so it has no pressure to
  gift kills either way. The frozen-panel eval is immune.

**I3. Per-decision time cost (styles/offense_time.toml; LITERATURE 3b, Blade & Soul). Rank 3.**
- **Hypothesis:** a cost of 8e-5 per both-alive decision (−2e-5·dt, ≈0.14 per median game) makes long
  games costly for both sides and shortens them.
- **Prediction:** len_s −20 s and sd −0.10. KH_PRE rises less than in I1 (≤ 1.2×), because ending the game
  by any route, the opponent's fall included, pays.
- **Falsified if** len_s changes by < 10 s.
- **Change:** `--style styles/offense_time.toml --style-fade-steps 1e18`. No C code is needed: the style VM
  pays it, contrary to LITERATURE 3b.
- **Hacking risk:** medium. When a loss is likely, dying sooner saves cost, but at 0.14 per game against
  1.29 for a death the effect is small. Watch unforced deaths and ckpm_b. Judge on panel win rate only.

**I2. Faded threat shaping (styles/offense_threat.toml). Rank 4: the diagnostic arm.**
- **Hypothesis:** offense is better under the true reward but undiscovered (an exploration problem).
- **Change:** 6e-5 per tick while an own projectile or explosive is within 4u of the opponent (≤ 0.43 per
  120 s). The term fades 1 → 0 over the first 4.2B (`--style-fade-steps 6000000000`). The last 1.8B and
  the anneal run without it.
- **Prediction (exploration):** KH_PRE stays ≥ 1.2× at the annealed endpoint with panel ≥ −0.02.
- **Prediction (reward-limited, which §1.2 favours):** KH_PRE returns to within 10% of control by the
  endpoint.
- **Hacking risk:**
  - high while the term is on, through spray near the opponent;
  - removed by the fade;
  - body proximity is deliberately not paid, because two self-play seats can share it by hugging;
  - misses guided missiles until the owner fix lands.

**I4. Button/hold entropy floor (`--entropy-head-scale 0.0288,15,10`). Rank 2 (no reward change).**
- **Hypothesis:** idling and low offense are partly an exploration collapse (§1.7). Raising the buttons and
  hold coefficients 10× (to ≈7.7e-3 and 5.1e-3) restores exploration of button sequences.
- **Prediction:**
  - buttons entropy ≥ 5% of max at 6B;
  - idle rate halved (to ≤ 4%);
  - presses_pm up ≥ 20%;
  - KH_PRE ≥ 1.1×;
  - panel ≥ −0.02. The matched anneal keeps the floor, because the scale applies per head.
- **Falsified if** button entropy rises but idle and KH_PRE stay within 10% of control. Then collapse is a
  symptom, not a cause.
- **Change:** only `--entropy-head-scale`; aim is unchanged. Target-entropy control (adapt the coefficient
  so button entropy tracks e.g. 5% of max) is the better version but needs ~40 lines in train.py. Use the
  fixed 10× first.
- **Hacking risk:** none (no reward change). The risk is noise: random presses waste cooldowns and lower
  strength. The guardrail catches it.
- **Overlap:** this is the factorial's entropy factor C restricted to the discrete heads. Run it as factor
  C, not as a separate arm, and read the offense metrics off the C cells.

**E-SD. Sudden-death start-state curriculum (red-team review). Rank 5 for offense.**
- **Change:** 25% of training episodes start at episode tick 6960 with randomized positions. This needs C
  reset code; train.py has no flag for it.
- **Hypothesis (review):** panel ≥ +0.03, sudden-death-sink losses −25% relative.
- **Offense prediction:** KH_PRE unchanged (±10%), sd unchanged or ↑. It trains the skill humans never
  test (§1.4).
- Run it as a strength arm with the offense readout. If it wins on strength and loses offense, that
  confirms §1.4.

**Not run as arms:**
- **Kill-credit fix alone:** it is in the base of every arm (§1.6, predicted < 5% behavioural effect).
- **Gamma 0.998:** the factorial already has it. Read offense metrics off those cells for free. The
  prediction is len_s ≥ control (§1.3).
- **Shorter episode cap:** the 9.2 s horizon makes a 150 s cap invisible before ~130 s, and 3-minute
  games were Matthew's choice on 2026-09-24 ("so sudden death is played out").

## 4. Protocol

- **Init:** l2-main-136002m (L/main/checkpoints/policy-136002994176.bin). 6B constant at the l2 config:
  - LR 2.43e-4;
  - the frozen opponent list minus the 4 panel nets;
  - `--bank-groups all,all,all,forgotten --pfsp-p 3`, rusher bank;
  - EMA on.
- **Anneal:** 1B cosine to an absolute LR floor of 1.2e-5, with `--init-weights <arm 6B> --seed <same>`.
- **Seeds:** 4 per arm vs 4 of E0's 6 controls (use all 6 if E0 has run); no arm-specific control.
- **Futility at 3B:** one seed, 1000 panel games. Stop the arm if KH_PRE < 1.1× the E0 control mean at 3B
  (I1, I3), or if the panel score is below control − 0.05.
- **Eval per seed at the annealed endpoint:**
  - 2000 panel games (500 per net) with `--games-out`;
  - 1000 offense_diag games vs f5 (first_hit_s, idle, engage);
  - the human-proxy panel, 1600 games, report only;
  - total 4600 games ≈ 1.2 h at 6 workers.
- **Cost per arm:** 4 × 3.0 GPU-h = 12 GPU-h plus ~5 CPU-h of eval. I1 + I3 + I2 = 36 GPU-h (I4 rides on the factorial at 0 extra); E-SD 12 more
  once its code is built.
- **Decision rule (pre-registered):** adopt an arm if both hold:
  1. KH_PRE ratio arm/control ≥ 1.20, with the seed-level bootstrap 95% CI excluding 1.0 (for I3: len_s
     down ≥ 10 s with the CI excluding 0);
  2. non-inferiority on strength: the one-sided 95% lower bound of (panel_arm − panel_ctl) > −0.02.

  Holm across the arms tested. Hacking vetoes:
  - ckpm_b up > 50%;
  - unforced deaths up > 0.03;
  - idle up.
- **Power of the non-inferiority test** (2000 panel games per seed). Rows are the training-seed sd from
  E0; cells give the power at a true difference of 0, and in brackets at +0.01:

  | s_seed | 4 vs 4 | 6 vs 6 |
  |---|---|---|
  | 0.010 | 0.60 (0.88) | 0.75 (0.97) |
  | 0.015 | 0.45 (0.73) | 0.58 (0.87) |
  | 0.020 | 0.34 (0.58) | 0.45 (0.73) |

  4 vs 4 at the −0.02 margin is underpowered for an arm with exactly zero strength effect. If E0 finds
  s_seed ≥ 0.015, either use 6 vs 6 or widen the margin to −0.03 for screening and confirm at 6 vs 6 on
  base 221001. KH_PRE itself is well powered at 4 vs 4 for effects ≥ 1.2×.
- **Matthew sign-off needed:** on 2026-10-15 he said "don't change the main bot's curriculum". I1 and I3
  change the main's reward permanently, which is not the curriculum but is close. Confirm before
  launching.

Command template (arm I1, seed k), on top of the l2-anneal-2 command line (bopl-l2-anneal-2.service) without
its anneal and with the 4 panel nets removed from `--opponents`:

```
train.py --output S/offense-i1/s$k/run --init-weights L/main/checkpoints/policy-136002994176.bin --seed $k \
  --steps 6000000000 --anneal-lr 0 <l2 flags, --v4-ablate 16> --style styles/offense_kill.toml --style-fade-steps 1000000000000000000
train.py --output S/offense-i1/s$k/anneal --init-weights <6B checkpoint> --seed $k --steps 1000000000 \
  --anneal-lr 1 <absolute 1.2e-5 floor> <l2 flags> --style styles/offense_kill.toml --style-fade-steps 1000000000000000000
```
I2 uses `--style styles/offense_threat.toml --style-fade-steps 6000000000` in the 6B phase only; I3 swaps in
offense_time.toml. Before launch, check payouts with `python league style-rates <l2-main-136002m> --styles
styles/offense_<x>.toml --games 480` (expect ≤ 0.15/game for I1/I3, ≤ 0.3 for I2).

## 5. First measurement (CPU, 2026-10-18)

Run 2026-10-18 23:20 to 2026-10-19 00:23 EDT: 3 capped units (`offense-*`, CPUWeight 20, 4 workers, machine
load 90-100), seed base 231001. The base is unused elsewhere, so its numbers are not comparable with the
211001 panel. Command: native/rl/research/offense/probe.sh. Rows are in S/offense/*.jsonl and summaries in
S/offense/summary-*.json. Opponent f5 throughout (pre-revision; descriptive only).

| | l2-main-136002m, all kits | l2-main-136002m, 2,4,14 mirror | v4-full-2313m, 2,4,14 mirror |
|---|---|---|---|
| games | 240 | 120 | 120 |
| score vs f5 | 0.537 [0.460, 0.613] | 0.529 [0.442, 0.621] | **0.108 [0.050, 0.175]** |
| KH_PRE (kills/min) | 0.095 [0.059, 0.140] | 0.083 [0.046, 0.125] | 0.052 [0.021, 0.088] |
| sd rate | 0.450 [0.371, 0.529] | 0.692 [0.608, 0.775] | 0.700 [0.608, 0.783] |
| median length s | 70 [43, 118] | 145 [131, 161] | 129 [123, 133] |
| A lost by unforced death | 0.233 [0.179, 0.292] | 0.242 [0.158, 0.317] | **0.750 [0.675, 0.825]** |
| free wins | 0.304 [0.246, 0.367] | 0.200 [0.133, 0.267] | 0.017 [0.000, 0.042] |
| ability-held decisions/min | 321 [244, 408] | 118 [110, 126] | **254 [238, 270]** |
| presses (rising edges)/min | 40.7 [32.5, 49.7] | 47.8 [44.8, 50.9] | 54.5 [51.2, 57.8] |
| idle (long games) | 0.062 [0.024, 0.106] (n 130) | 0 (n 91) | 0 (n 86) |
| engage_s (bodies < 12u) | 4.6 [3.9, 5.8] | 3.8 [2.7, 5.2] | 6.0 [4.9, 7.8] |
| close_frac | 0.235 [0.195, 0.280] | 0.139 [0.116, 0.167] | 0.096 [0.076, 0.119] |
| first_hit_s (first threat) | 16.3 [11.7, 26.3] | 37.7 [27.9, 52.4] | 60.3 [18.5, 119.4] |

What it shows:
- **136002m is no more offensive than 133336m.** KH_PRE is 0.095 [0.059, 0.140] vs 0.083 [0.066, 0.100],
  and the unforced-loss rate is 0.233 vs 0.236.
- **The main is not a camper.** It meets the opponent within ~5 s, spends 24% of both-alive time within
  12u, and makes its first threat at a median of 16 s. Passivity here means it does not convert: one
  credited kill per ~10 min. Bodies are not avoiding each other. This matches the 2026-10-13 camping
  check, where bots stayed as close as humans.
- **v4-full's "aggression" is ability spam that does not convert.**
  - It holds abilities 2.2× as often as the main on the same kit, yet credits fewer kills (0.052 vs
    0.083) and loses by its own fall in 75% of games.
  - What Matthew felt as aggression is pressure, not lethality. Pressure works against humans, who
    fall to it, and fails against f5, which waits.
  - The human-proxy panel (section 2) should therefore score pressure, i.e. presses and threats per
    minute, next to kills.
- **Use sd rate and mean length, not median length, as the length metric.** The all-kit length
  distribution is bimodal (17% of games end before 1000 ticks, the rest near sudden death), so the median
  jumps between modes: 70 s here vs 121 s on D/mf5, with a CI of [43, 118]. KH_PRE stays primary.

## 6. Memory worth storing (for the parent)

- 2026-10-18: the Bopl main's passivity is non-conversion, not camping. It closes to < 12u within ~5 s
  but credits ~0.08-0.10 kills per minute before sudden death, and 52% of its pre-SD wins (84% in SD)
  come from the opponent's own fall. A free win pays 1.0 against 1.29 for a kill, and the kill bonus is
  8% of the main's win reward.
- 2026-10-18: v4-full-2313m's "aggression" is ability spam that loses in sim: 2.2× the main's
  ability-held rate on 2,4,14, fewer credited kills, 75% unforced losses vs f5, score 0.108
  [0.050, 0.175] (n 120).
- 2026-10-18: human 1v1 rounds have a median of 10.8 s, and 0 of 393 seats reached sudden death (120 s).
  55% of sim games do.
