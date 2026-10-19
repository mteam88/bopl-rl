# Literature review for the 2026-10 program: techniques ranked for Bopl

Written 2026-10-18 by the research subagent, to go with PLAN-2026-10.md (cited as "plan"; E-numbers
and stages are the plan's). Costs use the plan's 0.43 GPU-h per 1B learner steps; a standard screen
(6B constant + 1B matched anneal) costs 3.0 GPU-h. Effect sizes are the ones the papers report. Most
of them are not win rates and do not convert to our scale. T = tp-spec, S = T/native/artifacts/rl/sweeps,
L = T/native/artifacts/rl/league-loops/l2.

## 0. A new measurement first: the main's weights are random-walking

This is a CPU-only check on the 204 l2 main checkpoints in L/main/checkpoints. The files are flat
float32 with 1.11M parameters. File step = physical decisions, about 4/3 of the learner steps.

| file step | ‖W‖ | ‖W − W₀‖ |
|---|---|---|
| 0 | 274.9 | 0 |
| 8B | 279.2 | 33.7 |
| 48B | 301.1 | 92.8 |
| 88B | 322.5 | 133.4 |
| 136B | 347.9 | 173.9 |

- Consecutive 0.89B displacements have norm 11.3 each and pairwise cosine −0.017 to +0.006, so they
  are orthogonal. Displacements over 13B have a lag-1 cosine of only 0.08–0.10.
- Distance from start grows as about t^0.58, close to a pure random walk (t^0.5).
- ‖W‖ grows linearly, +27% over the run. muon.cu applies `weight_decay = 0.0` (pufferlib.cu:2109).
- The trainer agrees: per-update KL is 3e-4, clip_frac is 0.002 and gradient_norm is 0.52, below the
  1.5 clip (S/opt-audit/metrics-sub1000.jsonl).

Reading: each update is tiny, and almost all of the motion is noise. The directed component only
shows over windows of about 10B. This is the strongest single piece of evidence for the plan's
noise-floor hypothesis (E1/E2). It also explains why a one-off anneal is worth about 2× the whole loop.

## 1. Optimisation at the noise floor

**1a. Weight averaging (EMA/SWA/LAWA).**
- What: release an average of recent weights instead of the last iterate.
- Evidence:
  - Hägele et al. 2024 (arXiv 2405.18392, NeurIPS): constant LR plus a short cooldown matches cosine.
    SWA "improves performance along the training trajectory … while it does not match the cooldown,
    it reduces the gap" (their Takeaway 5).
  - Nikishin et al. 2018 (SWA for RL, UAI workshop): SWA reduces forgetting and raises final reward on
    Atari and MuJoCo with A2C/DDPG.
  - Hilton, Cobbe & Schulman 2022 (arXiv 2110.00641, NeurIPS): using an EWMA of the policy as PPO's
    proximal policy gave a gain that is "small but remarkably consistent", beating PPO on all 8
    held-out Procgen games.
- Mapping:
  - `--ema-decay` landed in 85f1a974. The decays planned in E2 (0.99997, 0.99999) give half-lives of
    0.57B and 1.7B.
  - Section 0 says signal accumulates over about 10B, so add a long arm at 0.9999966 (half-life 5B
    learner steps).
  - At that decay, write the update as `ema += (1-d)*(w-ema)` so the fp32 increment does not round
    away.
  - The offline uniform averages in S/opt-audit/avg (avg3/8/15 = windows of about 2/5/10B of l2) are
    built but unscored. Score them on seed base 211001 with 4300 games each, not the 1000 in probe.sh.
- Cost: 0 GPU; about 1 CPU-h per arm.
- Plan: supports E2 and E7 (gate the EMA file). It contradicts nothing.
- Caveat: weight averaging assumes the iterates share a basin. A wide uniform average losing to a
  narrow one would show that they don't.

**1b. Batch size and the critical batch.**
- What: the gradient noise scale B_simple = tr Σ / |G|² predicts the largest useful batch.
- Evidence:
  - McCandlish et al. 2018 (arXiv 1812.06162): Dota's noise scale is in the millions and rises during
    training. Optimal batches for Pong and Dota 5v5 differ by more than 10,000×.
  - OpenAI Five (Berner et al. 2019, arXiv 1912.06680, Fig. 5a): Rerun's 983k batch gave about a
    2.5× speedup over 123k at TrueSkill 175 (8× the data per step), "appears to increase with higher
    TrueSkill". LR was not retuned.
  - Our per-step batch is about 24.5k decisions (49k learner rows × 128 horizon split by
    `--minibatch-div 2`, 40.7k optimizer steps per B). That is 5× below Five's *small* baseline.
- Mapping:
  1. Measure B_simple for free. E[|g_B|²] = |G|² + tr Σ / B, and `gradient_norm` is already logged
     before clipping. Fit it across E6's arms (div 1/2/4 give B = 49k/24.5k/12k).
  2. Replace E6's `--minibatch-div 4` arm, which halves the batch and goes the wrong way per the
     literature, with an 8× arm: `--horizon 512 --minibatch-div 1` (196k per step). Check GPU memory
     first. Horizon also moves BPTT truncation, so this is not a pure batch test. Gradient
     accumulation over 4 rollouts would be pure but needs C.
  3. Under Muon the step length is fixed by LR, so a bigger batch buys a cleaner direction at the
     same step. That is not the same knob as E1's lower LR, which shrinks signal and noise together.
- Cost: 3.0 GPU-h per arm.
- Plan: modifies E6 and complements E1.

**1c. Sample reuse.**
- Evidence: Five Fig. 5c: reusing data 2–3× "can cause a factor of two slowdown", and 8× can prevent
  learning. Five targeted reuse of about 1.
- Plan: agrees with the refuted `--learner-passes 3` (−0.048). Don't retest.

**1d. LR schedule.**
- Evidence:
  - Five's Rerun cut LR 10× (5e-5 → 5e-6) over 1–2 days mid-run and planned 1e-6 (Fig. 7). Its
    entropy fell 10× and its horizon doubled on the same schedule.
  - Andrychowicz et al. 2021 (arXiv 2006.05990): LR is the most sensitive optimizer choice; linear
    decay helps a little.
  - Hägele 2024: WSD (constant LR, then a cooldown from any checkpoint) is as good as cosine and lets
    you branch releases.
- Mapping: this is E7's "anneal-and-continue" and polish-2's half-LR restart (+0.038, 1 seed).
- Plan: supports E1 and E7. The literature expects the anneal gain to reappear after every new
  constant-LR stretch, not to stack.

**1e. Muon done the way the large-scale papers do it.**
- Evidence:
  - Liu et al. 2025 (Moonlight, arXiv 2502.16982): without weight decay, "weights and layer output RMS
    keep growing" and Muon's gain over AdamW fades in long runs. Decoupled WD restored it. Matching
    update RMS to AdamW (0.2·√max(A,B)) let them reuse AdamW's LR and WD. Overall about 2× compute
    efficiency.
  - "When does Muon help agentic RL?" (arXiv 2607.16169, 2026): Muon's RL gain came from a 3.5×
    larger effective update. "A full-budget RMS-matched control removes the late-success gain."
  - PufferLib 3.0's blog: Muon gave "a step-change" only after full sweeps. Its default Muon LR is
    0.015; ours is 2.43e-4, 60× lower.
- Our implementation:
  - Uses Keller's √max(1,R/C) scale with WD 0, so a 256×256 update has per-element RMS ≈ lr/16.
  - Applies orthogonalised updates to the output heads.
  - Gives 1-D tensors plain clipped Nesterov momentum at the Muon LR, which means they barely move.
    Moonlight and Keller route heads, embeddings and 1-D tensors to AdamW.
- Mapping (C change):
  - Pass `hypers.weight_decay` into `muon_init` in place of the literal 0.0, and add `--weight-decay`.
    Choose λ so that the norm settles near today's: the equilibrium is ‖W‖² ≈ lr·‖U‖²/(2λ) per step.
  - Optional: an AdamW group for heads and 1-D tensors.
- Cost: 0.5 day of code plus 3.0 GPU-h per arm.
- Plan: refines the stage-3 "Adam is safe" arm. The literature predicts that Adam vs Muon at matched
  update RMS is a near tie, so the informative contrast is update *size*, which is E1.

**1f. Value-loss details.**
- Evidence: Andrychowicz 2021 and Engstrom et al. 2020 ("Implementation matters", arXiv 2005.12729)
  find PPO value clipping neutral to harmful.
- Our setup: we clip at `vf_clip_coef 0.2` (PufferLib default.ini) with vf_coef 2.0. Explained
  variance is already 0.95, so the critic is not the bottleneck at γ 0.995. That may change at
  γ 0.998+ (E4).
- Mapping: an ini override, with no train.py flag yet.
- Cost: 3.0 GPU-h. Low priority. Run it only as a fallback if E4 shows EV collapse.
- PopArt (Hessel et al. 2019) addresses reward scales that vary across tasks. Ours are fixed (±1,
  ±0.29), so skip it.

## 2. League mechanics

**2a. AlphaStar's actual proportions** (Vinyals et al. 2019, Nature 575; Methods and the published
pseudocode).
- Main agents:
  - 35% self-play, 50% PFSP over all past players, and 15% "verification" games.
  - Verification draws main-exploiter snapshots the main scores under 0.3 against, weighted
    (1−x)², and forgotten past mains it scores under 0.7 against, weighted x(1−x).
  - When the self-play opponent is too strong, the main falls back to an x(1−x) curriculum over
    that opponent's history.
  - Snapshot every 2e9 steps. Mains never reset.
- Main exploiters: play the current mains, falling back to an x(1−x) curriculum over the mains'
  snapshots when they win less than 10–20% (20% in Methods, 0.1 in the pseudocode). Added at more than 70% wins or after a timeout, then reset.
- League exploiters: (1−x) capped at 0.5. Reset to the supervised init with probability 0.25.
- Shape: 12 agents (3 main, 3 main-exploiter, 6 league-exploiter). Three quarters of agents were
  exploiters, but they fed the main only through the 15% branch and through PFSP.
- Ablations (Fig. 3c,d, ED Fig. 5): PFSP-based training beat FSP on population strength,
  exploitability and final-agent strength. The league-composition ablation reports the exploiters'
  value mainly as lower exploitability. About 3,000,000 RPS cycles in
  the final league involved exploiters versus about 200 among mains only (ED Fig. 8). In other
  words, exploiters stay exploitable by each other while the mains turn transitive.
- How mains absorb exploiter lessons: only by playing them. There is no distillation and no reward
  term.
- Mapping:
  - Our `forgotten` bank reproduces the 15% (12.5%), but the three `all` banks at `--pfsp-p 3` also
    draw exploiters. Exploiters reach 61–64% of PFSP games (about 30% of all games) at learner
    scores of 0.13–0.22.
  - Under f_hard p=3, a 0.15 opponent gets 4.9× the weight of a coin-flip opponent. AlphaStar's p=2
    would give 2.9×.
  - E3(a) (`--bank-groups league,league,all,forgotten`) is the closest to AlphaStar.
  - E3(b) (`--pfsp-p 1`) is AlphaStar's league-exploiter weighting.
  - Add an arm that uses `pfsp.f_var` for the `all` banks. It is a one-line change in
    `native_selfplay.intent`, which already switches to f_var when everything is hopeless.
- Plan: supports E3 and E7 (dropping league exploiters). AlphaStar's league exploiters started from
  human imitation and trained 2–4e9 steps each. Ours start random and never landed (0/16).

**2b. Other selection schemes.**
- OpenAI Five: 80% latest self, 20% past versions, with sampling weights from a learned quality
  score (App. N).
- Blade & Soul (Oh et al. 2021, IEEE ToG, arXiv 1904.03821):
  - Setup: three style-shaped learners shared one opponent pool. The newest k models of each style
    took p = 0.8 → 0.1 of the games, annealed over training.
  - Shared pool vs. separate pools: 73.6% vs. 64.4% average win rate against unseen-style opponents
    (7,000 games).
  - The style agents beat a no-style self-play baseline 62.2% (1,000 games each), and scored 62% vs
    pros.
- FTW / Capture the Flag (Jaderberg et al. 2019, Science, arXiv 1807.01281):
  - Population of 30. Matchmaking favours opponents of similar Elo. An agent losing under 70% copies
    the winner's weights, internal-reward weights and hyperparameters, then mutates them (PBT).
  - With win/loss reward only there was "no learning". Evolved internal rewards beat hand-shaped
    rewards on Elo (Fig. 2b).
- Mapping: the B&S recipe matches our style×kit exploiter recipes, but there all three styles were
  full learners. PBT needs a population; on one GPU, 4 members × 2B costs 3.4 GPU-h per generation.
  That is affordable only as a small reward-weight search after E4/E6 settle.

**2c. Absorbing a found exploit faster.**
- Kickstarting / policy distillation (Schmitt et al. 2018, arXiv 1803.03835; Rusu et al. 2016).
- Honor of Kings CSPL (Ye et al. 2020, NeurIPS, arXiv 2011.12692): train specialists on fixed
  lineups, distill them into one student driven by the student's own play, then continue RL.
  - 40 heroes: 336 h vs. more than 480 h with no convergence for random lineups (Table 1).
  - 20 heroes: 144 h vs. 192 h.
  - The student lands "marginally below" each teacher within about 2 days.
- Minimax Exploiter (Bairamian et al. 2024, AAMAS, arXiv 2311.17190): shape the exploiter's reward
  with the *main's own value function*. A big data-efficiency gain for exploiters in For Honor.
- ROA-Star (Huang et al. 2023, NeurIPS): goal-conditioned exploiters plus opponent modelling gave a
  less exploitable main at far below AlphaStar compute.
- Mapping:
  1. Plan 2d (fine-tune the main on the drill-grenade exploiter's matchup, ≤1B, with a mirror-kit
     control) is step one: is the gap fixable or kit-intrinsic?
  2. If it is fixable, distill that specialist back into the main only on that matchup's games, while
     RL continues elsewhere. `--kl-reference/--kl-coef` exists but applies one reference to every
     learner game, so it needs a per-bank or per-kit mask in C. `train/distill.py` is the offline
     alternative.
  3. Missile is different. The findings log shows guided missiles carry owner 0 and missile kills are
     uncredited, so that exploit is observation- and reward-limited, and no league method fixes it.
     Five's answer to such defects was surgery (§3.3: about 20 surgeries, one every two weeks).
  4. Minimax-style exploiter reward makes exploiter turns cheaper, but our exploiters already land
     +0.2–0.3. The bottleneck is absorption, not discovery.
- Cost: step 1 is 0.43 GPU-h; distillation is about 1 day of code plus about 3 GPU-h.
- Plan: new item, attached to 2d.

## 3. Passivity: 55% of games reach sudden death, and the main idles

**3a. Horizon.**
- Five defines the horizon as H = T/(1−γ). Resuming a skilled agent with a longer horizon "makes it
  perform better, up to the longest horizons we explored (6–12 minutes)" (Fig. 6), in 45-minute
  games. Their Rerun stepped 180 s → 360 s, with 840 s planned.
- Ours: γ 0.995 at about 15 decisions/s gives H ≈ 13 s against a 121 s median game (11%).
  - 0.998 → 33 s; 0.999 → 67 s.
  - With a 13 s horizon, a sudden-death win is invisible at the start of the game. Early play then
    answers only to ±0.29 kill/death terms and to avoiding unforced environment deaths (51% of
    losses). That is a recipe for caution.
- Mapping: E4 as written. Andrychowicz 2021 also ranks γ among the most important choices.
- Cost: 9 GPU-h.
- Plan: strongly supports E4. Add sudden-death share and median length as E4 outcome metrics.

**3b. Explicit aggression shaping.**
- Blade & Soul shaped style along three axes (Table 2):
  - Per-step time penalty: aggressive 0.008, balanced 0.004, defensive 0.
  - Own-to-opponent HP weight: 5:5 or 6:4.
  - Distance penalty: 0.002 / 0.0002 / 0.
- B&S results: the style agents beat the unshaped baseline (above). Average game length tracked the
  style. The shared pool helped the aggressive agent least, "because one side should fight back if the
  other side approaches".
- Five scales every non-terminal reward by 0.6^(T/10 min) and makes all rewards zero-sum by
  subtracting the enemy's mean (App. G).
- GT Sophy (Wurman et al. 2022, Nature 602): progress reward plus penalties for leaving the track,
  walls, tyre slip and at-fault collisions, and a passing bonus. It also uses mixed-scenario training,
  oversampling hard tactical situations. The reward weights were retuned between the July and October 2021
  events, balancing aggression against sportsmanship.
- Mapping:
  - A per-decision cost c to both players, or equivalently a win reward that decays with time
    (1 − κ·t/T_sd), makes long games costly for both sides. That changes the objective, so judge it
    only on win rate vs. f5 at T=1, never on shaped return.
  - Size: c·(decisions in a median game, about 1800) ≈ 0.1–0.2, so c ≈ 6e-5 to 1e-4, about the same
    ratio to the win reward as B&S's balanced agent.
  - Credit missile and spike kills (findings log) before shaping. A kill term that misses whole
    abilities biases which abilities look worth pressing.
  - Potential-based shaping (Ng et al. 1999) is the policy-invariant alternative. Bopl is one-hit-kill
    with no HP, so the only obvious potential is distance or position, which B&S used without the
    invariance guarantee.
  - There is no train.py flag: `reward_timeout` only fires at the 10,800-tick cap. This needs a C
    reward term.
- Cost: 0.5 day of code plus 3.0 GPU-h per arm.
- Plan: new. It coordinates with the parallel passivity thread.

**3c. Passive no-ops.** B&S dropped forced "passive no-op" steps (no skill available) from training.
They reached 80% vs the built-in AI in 70k steps instead of 170k. Holding a move decision for 10 ticks
beat 1, 3 or 5. We already have hold lengths {1, 2, 4, 8, 16}. If idle decisions while all abilities
are on cooldown are a large share of the data, masking them from the loss is cheap. Measure first with
the 1b per-game rows. This is diagnostic only.

## 4. Generalisation across 7,140 kits × 124 maps

- Procgen (Cobbe et al. 2020, arXiv 1912.01588): more training levels and bigger nets close the
  generalisation gap.
- PLR (Jiang et al. 2021, ICML, arXiv 2010.03934): replay levels in proportion to the rank of mean
  |GAE|, mixed with a staleness term. It gave "statistically significant gains in mean normalized test
  and train returns". Score alone or staleness alone was worse than uniform.
- HoK CSPL (above) exists because random lineups caused "learning collapse" at 40 heroes.
- XLand (OEL Team 2021, arXiv 2107.12808): dynamic task generation plus generational distillation.
- Mapping:
  - Our per-kit deltas are pure noise (plan §0) and held-out maps 3 and 4 are not yet scored. There
    is no sign of collapse, so CSPL is not indicated.
  - PLR's cheap form is setup prioritisation: keep a running mean |GAE| per (learner kit ability,
    opponent kit ability, map) cell, and sample setups by rank instead of uniformly.
  - Kit weighting (kitw, −0.031, underpowered) was a fixed weighting, not an adaptive one.
- Cost: about 1 day of code (setup sampler in `bopl_puffer.h`/`native_selfplay.py`) plus 3.0 GPU-h.
- Plan: belongs in Stage 4, and only if the held-out-map check shows a gap or the exploit panel stays
  kit-concentrated.

## 5. Already tried here, so don't re-propose

- Decision-time search `8,1,8` adds about +0.16 (0.660 [0.582, 0.733] vs. raw, docs/EI.md).
- Distilling it into the net (expert iteration) did not keep the gain (docs/EI.md).
- Sharper sampling, T = 0.75 on all heads: 0.492 [0.438, 0.544]. The trained entropy is about right.
- Use search at play time for the release, as AlphaZero and Pluribus do. That is a release decision,
  not a training experiment.

## 6. Where the literature disagrees with the plan

1. **E6 direction.** The literature (Five, McCandlish) says our batch is small. Halving it
   (`--minibatch-div 4`) is the low-value arm. Swap it for an 8× arm (1b).
2. **E2 half-lives.** Section 0's drift only appears over about 10B, so 0.5B and 1.7B windows may be
   too short. Add 5B (0.9999966) and score avg8/avg15.
3. **E1 vs E6.** Under SGD these are one "noise temperature" knob, but under Muon they are not (1b).
   The B_simple fit tells which arm should win before the evals finish.
4. **Stage-3 Adam arm.** The 2026 Muon-RL evidence says a raw Adam-vs-Muon comparison mostly measures
   update size. Report the update RMS of both arms, or match it.
5. **Long runs without weight decay** (E10, 150B) are exactly where Moonlight saw Muon's gain fade.
   Decide on WD before E10, not after.

## 7. Top 8 to try, ranked

Rank = (prior × size of effect) / cost, given 672 GPU-h and a screening MDE of about 0.03 at 4300
games. "+x" is the annealed (or EMA) score vs. f5 on seed base 211001 unless stated.

| # | Technique | Hypothesis (falsifier) | Flags / code | GPU-h | Plan |
|---|---|---|---|---|---|
| 1 | **EMA / uniform weight averaging with long windows** | EMA at half-life ~2–5B ≥ raw + 0.03 and ≥ annealed − 0.02 (false if EMA − raw < +0.015, paired within run). Offline avg8/avg15 of l2 already ≥ last + 0.02 | `--ema-decay 0.99999` and `0.9999966`; score S/opt-audit/avg/* at 4300 games | 0 (CPU ~1 h/arm) | E2, E7 |
| 2 | **Bigger per-step batch + measured noise scale** | B_simple from E6 arms ≥ 100k; then 8× batch at the same LR raises the *raw* level ≥ 0.03 and halves the raw-to-anneal gap (false if B_simple < 50k or raw within ±0.02) | E6 arms' `gradient_norm` fit; `--horizon 512 --minibatch-div 1` | 3 | modifies E6 |
| 3 | **AlphaStar-proportioned PFSP** (exploiters ≤ 15% of main games; f_var or p ≤ 1 elsewhere) | +0.03 general at ≤ 0.05 exploit-panel cost (false: within ±0.02, or panel drops > 0.10) | E3(a) `--bank-groups league,league,all,forgotten`; E3(b) `--pfsp-p 1`; new f_var arm (1 line in `native_selfplay.intent`) | 9 | E3 |
| 4 | **Longer horizon** γ 0.998 / 0.999 (Five) | +0.03 annealed and sudden-death share down ≥ 10 points (false: ≤ +0.01, or EV < 0.8 with value loss doubled) | `--gamma 0.998`, `--gamma 0.999`, `--gamma 0.998 --gae-lambda 0.95` | 9 | E4 |
| 5 | **Specialist then distill** to absorb an exploit (CSPL / kickstarting) | Step 1: ≤1B fine-tune closes ≥ half the drill-grenade gap (+0.13) and the mirror does not, so it is fixable. Step 2: distilling the specialist into the main on those games closes ≥ 0.10 of it at ≥ −0.01 general (false: step 1 fails, which means kit-intrinsic, so stop) | plan 2d; per-bank `--kl-reference` mask (C) or `train/distill.py` | 0.43 + ~3 | 2d (new follow-up) |
| 6 | **Anti-passivity cost** (B&S time penalty / time-decayed win), after crediting missile and spike kills | c ≈ 6e-5–1e-4 per decision cuts sudden-death share from 55% to < 40% at win rate ≥ −0.01. If win rate *rises* ≥ 0.03, passivity was a local optimum, not the equilibrium (false: share unchanged or win rate < −0.02) | new C reward term `--reward-step`; kill credit fix | 0.5 day + 6 (2 arms) | new; pairs with E4 |
| 7 | **Muon hygiene: decoupled WD (+ AdamW for heads and 1-D tensors)** | ‖W‖ stays flat and the 7B screen is ≥ control − 0.01. The real test is E10's slope: WD arm ≥ +0.02 per 20B over no-WD (false: screen < −0.02) | pass `weight_decay` through `muon_init` (pufferlib.cu:2109); `--weight-decay` | 0.5 day + 3 | stage 3, before E10 |
| 8 | **PLR-style setup prioritisation** (rank of mean \|GAE\| per kit-ability × map cell, plus staleness) | +0.02 general and the exploit panel improves ≥ 0.05 on kit-concentrated exploiters (false: within ±0.02 on both) | setup sampler in `bopl_puffer.h` / `native_selfplay.py` | 1 day + 3 | Stage 4, conditional |

Total for screens 1–8: about 36 GPU-h, which fits inside the plan's week 1–2 allocation (83 GPU-h).
Items 1, 3 and 4 are already plan items; the literature raises their priors. Items 2, 5, 6, 7 and 8 are
additions or modifications. Next, ranked lower: PPO-EWMA proximal policy (1a; small consistent gains,
+30% learner compute in their setup), turning value clipping off (1f), a small PBT over reward weights
(2b; 3.4 GPU-h per generation), and plasticity diagnostics (dormant units, effective rank per layer;
Sokar et al. 2023, Dohare et al. 2024) on the 204 checkpoints at CPU cost.
