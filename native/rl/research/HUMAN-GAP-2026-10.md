# Human gap and offense-metric audit (2026-10-19)

Written by the humangap research subagent. CPU only, capped units. T = tp-spec,
O = native/artifacts/rl/sweeps/humangap (regenerate with offense/humangap/run.sh; summaries are committed
in offense/humangap/). "Bot" means the E0 l2-s301 annealed endpoint
(T/native/artifacts/rl/research/e0/l2-s301/anneal/checkpoints/policy-1333395456.bin, held-out panel
0.615) unless named otherwise. Intervals are 95%: session-cluster bootstrap for human rows, seed-pair
bootstrap for sim rows, Wilson for proportions.

## A. Offense metrics

### A1. aggression.py's kills_pm == arrow_hits_pm: two identities, not a copy bug

Every one of the 3000 rows in S/aggression has `kills_pm == arrow_hits_pm`. 2646 rows have
`first_kill == seconds`, and the remaining rows differ from it by a few ticks.

- `arrow_hits` is `RlTelemetry.ArrowHit`. The sim counts it only on a `killPlayer` by an arrow
  (`count_arrow_hit`, PlayerCollision.cs:455), so it counts lethal arrows only. In the 2,4,14
  (Dash/Bow/Platform) kit the only weapon is the Bow, so every credited kill is a lethal arrow and
  the two columns are equal by construction.
- Rounds have one life, so the subject's first kill ends the game. The old fallback ("no kill
  counts the game length") therefore made `first_kill` equal to the game length in every game.
- A third bug, found in the audit: the three `*_activations` counters are per kit slot
  (`ability_activations[3]` in the ABI). eval/telemetry.py names them in the 2,4,14 order, so any
  other kit order mislabels them. In human replays, kit [4,14,2] read Bow presses as "Dash".

Fix (eval/aggression.py):
- kills now come from step events with v4.2 credit (eval/kill_credit.py);
- `kills_pm` is pooled credited kills per both-alive minute, and `kill_win` is added;
- `first_kill` is None without a kill, and the summary reports a Kaplan-Meier median with no-kill
  games censored;
- `arrow_hits` is renamed `arrow_kills`;
- the activation columns are named by the kit.

Regression test: eval/test_offense_metrics.py, 8 tests. The live test plays 12 scripted games and
checks game by game that event kills equal lethal arrows on 2,4,14 (a dropped event would show
there), that `first_kill` is None if and only if there was no kill, and the slot naming.

### A2. Audit of every metric the rule could use

| Metric | Where | Verdict |
|---|---|---|
| Credited kills, all eval rows | all_combos `--games-out` `deaths.*`, offense_diag `actor_is` | **Undercounts.** Telemetry and the game's `actor` credit nobody for Missile blasts and cast Spikes, which v4.2 training pays the caster for. They land in `deaths_other`. Vs x2-missile, 22 of 80 games (27%) had such a kill. Fixed: games-out rows now carry `credited` [a, b] (elo_ladder, from events), and offense_diag uses the same credit, keeping the old value as `game_actor_is`. The docs claim that "KH_PRE counts Missile/Spike kills on both sides" was false for every eval tool. |
| kills/min, KH_PRE (offense_metrics.py) | ratio estimator | Correct. The tick alignment is verified: the death event tick is episode ticks + 239 (WARMUP 240), so tick 7200 is episode tick 6960, the I1 horizon. |
| ckpm, ckpm_b | per game minute | Correct. Same Missile/Spike undercount as above. |
| time to first kill | aggression `first_kill` | Was the game length (A1). Now KM. offense_metrics has no TTFK; `first_hit_s` is time to first *threat*, and it misses owner-0 entities (missiles, cast platforms), as documented. |
| approach distance / closing | aggression, offense_diag `mean_dist`, `engage_s`, `close_frac` | Correct and consistent. Both alive, decision cadence. |
| ability use | aggression `*_activations_pm`; offense_metrics `apm`, `presses_pm`, `idle` | Activations were mislabeled off the 2,4,14 order (fixed). `apm` counts button-down decisions, not presses. `idle` exists only in offense_diag rows. |

### A3. How "kill rate" would have been computed for the I1/I3 rule: it would not

- train/research/arm.py computed panel score only. research_report.py compared groups on score only,
  and only within one program. Nothing computed kills/min, so the pre-registered rule had no
  implementation.
- The control was ambiguous. I1/I3 run on v4.2 (`v4-ablate 16`), but E0's `l2` control runs
  `--v4-ablate 0`. Only `obsfix` (seeds 301-304) is a v4.2 control.
- The sign-off says "kills/min", while OFFENSE §4 says KH_PRE.

**Definition, now implemented in eval/kill_rate.py. Thresholds unchanged.**
- Kill rate = KH_PRE of the annealed endpoint, pooled over its held-out panel games (4 nets, equal games
  each; 4300 per seed).
- Per game, seat A = the arm. kill = B's death credited to A by the game (`deaths[1].opponent`).
  It counts only if the round ends before episode tick 6960. Exposure = min(ticks, 6960)/3600.
  KH_PRE = Σ kill / Σ exposure per seed. The group value is the mean over its seeds.
- Why game credit and not v4.2 credit:
  - I1's premium (styles/offense_kill.toml, `ev.actor == self`) pays game credit only. The style VM
    has no `creator`, so a Missile/Spike kill gets the base kill reward but no premium.
  - E0's obsfix evals, produced by the frozen T, have no `credited` field.
  - `--credit cast` gives the v4.2 version once both sides carry it.
- Rule:
  - ratio ≥ 1.20, with the seed-bootstrap 95% CI above 1 (Holm across the arms tested, one-sided 0.025);
  - panel non-inferiority: one-sided 95% lower bound (Welch over seed means) > −0.02.
- Vetoes:
  - ckpm_b ×1.5;
  - unforced deaths +0.03, via the proxy `env_loss` (share of games A lost to an environment death;
    the 3u "near" split needs positions that panel rows lack);
  - idle, reported unmeasured.
- Control: e0/obsfix. research_report now accepts `compare = "e0/obsfix"`.
- The 19:16 replan moved I1/I3 into the offense 2^(5-1) factorial, "judged on its main effect".
  `kill_rate.py --factorial <program dir>` finds the I1 and I3 factors by their style flags and
  applies the same rule to high vs low runs.
- arm.py results now carry per-opponent `offense` stats.
- Reference: the l2-s301 annealed endpoint scores KH_PRE 0.131/min on its panel, ckpm 0.154, sd 0.49,
  median 114 s.

## B. The human gap

Data:
- **Mac archive** (re-synced from the Mac today, 629 games, newest 2026-10-17): all 2-player rounds
  re-stepped in the q32 sim by eval/human_gap.py into offense_diag rows, so humans and bots share
  one set of definitions (offense_metrics.py). 380 faithful human seats in 1v1 vs bots, 11 sessions,
  mostly Matthew on a gamepad, 76% Dash/Bow/Platform.
- **Older human-human corpus:** 284 faithful seats, 9 sessions.
- **Bot reference:** l2-s301 vs f5, 300 any-kit games and 200 games on the 2,4,14 mirror
  (offense_diag), plus eval/aggression.py on the 2,4,14 mirror (200 games per subject vs f5).

### B1. Bot results against humans

- **1v1, recent bot:** only g111-20b-26027m (2026-10-10, 2-stock rounds, Dash/Blink/Grenade).
  **The bot beat Matthew 17/25 (0.68 [0.48, 0.83]).** The bot of this report beats g111 0.808 in
  sim, so no data says the current bot loses 1v1 to Matthew.
- **1v1, September nets:** 254 faithful rounds, humans won 0.77.
- **Free-for-all** (bot round-win vs a fair share of 1/n):

  | Bot | Round-win | Fair share |
  |---|---|---|
  | g111, 3p | 0.34 | 0.33 |
  | allkit-1, 3p | 0.24 | 0.33 |
  | f5, 4p | 0.24 | 0.25 |
  | f5, 6p with 2 bots | 0.43 | 0.33 |
  | team5b, 3p | 0.58 (n 12) | 0.33 |

  None of these bots trained on FFA. "The bot loses to humans" is not supported. "Plays nothing like
  humans" is, by a wide margin.

### B2. Where the bot differs (same definitions on both sides)

| | humans vs bots | human vs human | bot vs f5, any kit | bot vs f5, 2,4,14 |
|---|---|---|---|---|
| median round (s) | 10.5 | 27.3 | 120 | 142 |
| rounds over in < 30 s | 86% | 58% | 25% | 20% |
| reach sudden death | 0.5% | 0% | 55% | 69% |
| **KH_PRE** (credited kills / both-alive min, pre-SD) | 1.50 [1.10, 1.87] | 1.13 [0.83, 1.52] | 0.096 [0.062, 0.133] | 0.104 [0.070, 0.140] |
| KM median time to first credited kill (s) | 24 | 41 | never (17% kill) | — |
| engage_s (bodies < 12u) | 4.3 | 5.3 | 4.3 | 3.5 |
| mean distance both alive (u) | 27.2 | 24.9 | 32.9 | 29.8 |
| first threat (own entity < 3u of opponent), median s | 8.9 | 12.1 | 17.6 | 23.9 |
| loser died credited / unforced | 57% / 33% | 60% / 24% | 34% / 45% | 50% / 32% |
| loser cause: ring-out | 10% | 17% | 58% | 32% |

On 2,4,14 only:

| | humans vs bots | human vs human | bot vs f5 | f5 vs f5 |
|---|---|---|---|---|
| Bow activations / min | 8.6 | 9.3 | 4.2 | 2.5 |
| lethal arrows / min | 1.40 | 0.77 | 0.18 | 0.08 |
| arrows per kill | 6 | 12 | 23 | 32 |
| share of ability-button decisions on Bow / Dash / Platform | 62 / 4 / 34% | 56 / 11 / 34% | 20 / 47 / 32% | — |

(Human-vs-bot rows are inflated by weak September opponents; human-vs-human is the cleaner reference.)

Where the bot differs, concretely:
1. **Conversion, not approach.** The bot engages as fast as humans (≈4 s to 12u) and sits only
   3-8 units farther away. Its kill rate before sudden death is **11× lower** than human-vs-human
   (0.10 vs 1.13 per both-alive minute), and it takes twice as long to put a threat near the
   opponent (18-24 s vs 9-12 s).
2. **It does not shoot.** On 2,4,14 the bot fires half as often as humans (4.2 vs 9.3 Bow/min) and
   spends its ability use on Dash (47% vs 4-11%), that is, mobility and evasion rather than the
   weapon. f5 fires 2.5/min. Each arrow is half as likely to kill (1 in 23 vs 1 in 12; the
   opponents differ, so this is partly f5's dodging).
3. **Its games are decided by the environment.** Bot-vs-bot losers die unforced 45% of the time
   (any kit; ring-out 58%), against 24-33% in human rounds. 55-69% of its games reach sudden death,
   a regime humans never reach.
4. **Its tempo is a different game.** 75-80% of its games last over 30 s, against 14% of rounds vs
   bots and 42% of human-human rounds. The bot is trained and scored almost entirely in a regime
   humans never play.

The bots do get kills when humans force the pace: g111 killed Matthew 1.83/min, against 0.10 vs f5.
The passivity is a property of bot-vs-bot equilibrium play. It is not an inability to fight. That is
§1.2 of OFFENSE (mirror tempo is worth nothing) observed from the human side.

### B3. Do I1 and I3 target this gap?

- **I1: yes.** The gap is kill conversion before sudden death, and I1 pays exactly game-credited kills,
  most in the first seconds. Its premium is still 0.22 at 30 s, where human kills happen.
  KH_PRE is the right adoption metric.
- **I3: weakly.**
  - Both seats pay the same time cost, so in the mirror it is close to a constant. It shortens games
    by any route, the opponent's fall included, and falls are already the bot's dominant decider.
  - Human rounds are short because of kills (57-60% credited), not because they end somehow.
  - Expect I3 to fail the KH ratio, or to show up in the `env_loss` veto.
- **Missed by both:** the reward still pays a sudden-death free win as much as a pre-SD free win
  (1.0). That is the payoff that makes waiting optimal (84% of the main's SD wins have no credited
  kill).

### B4. Proposed arm I7: sudden-death win discount (styles/offense_sd_discount.toml)

- **Change:** −0.5 to the winner when the opponent's last body dies at or after episode tick 6960
  (`--style styles/offense_sd_discount.toml --style-fade-steps 1e18`). Losses are unchanged, so no
  seat gains by dying, and sudden death becomes negative-sum for the pair.
- **Check:** compiles (144 bytes). In 20 l2-s301-vs-f5 games it fired exactly once on each of the 10
  SD-decided games, on the winner's seat, and never before SD.
- **Hypothesis:** waiting for a fall in sudden death is what the main's equilibrium is built on.
  Halving that payoff makes pre-SD conversion worth more for both seats in the mirror, where I1's
  kill premium is only a relative incentive.
- **Pre-set rule:** identical to I1/I3. KH_PRE ratio ≥ 1.20 with the seed-bootstrap CI above 1
  (Holm with the other offense factors), panel (scored normally, SD wins = 1) non-inferior at −0.02,
  and the same vetoes.
- **Falsified if** the sd share drops by less than 0.05 absolute.
- **Prediction:** sd −0.10 or more, KH_PRE ≥ 1.2×.
- **Placement:** run it as a factor in place of I3 in the 2^(5-1), or as a sixth factor if budget
  allows.
- **Risk:** within a few seconds of SD, a seat ahead on position may rush an attack. That is the
  intended pressure. No farming is possible (one life).

### B5. What should change before I1/I3 run

1. Judge them with eval/kill_rate.py (or its `--factorial` mode), with control **e0/obsfix**, not
   e0/l2.
2. Decide game credit vs cast credit before results exist. This report pre-registers game credit
   for the reasons in A3. Optionally expose `creator` to the style VM so I1 pays Missile/Spike
   kills too.
3. Get one human baseline: 20+ 1v1 rounds by Matthew vs the l2-s301 anneal at stocks 0 (training
   rules), on 2,4,14 and random kits. The replan defers this. It is the only direct measure of the
   stated goal, and all 1v1 evidence on recent bots is 25 rounds vs g111.
4. Consider I7 in place of I3.
