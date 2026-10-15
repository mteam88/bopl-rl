# Human behaviour cloning: Matthew's replays into f5's action space

Goal: a net that plays like Matthew, from his recorded games, in f5's exact
observation and action space, as the start of a human-style exploiter of f5
(`final5-18666m`). Work of 2026-10-15 (rift `human-exploiter`).

Code:

- `train/bopl_agent.{c,h}`: `bopl_agent_force(agent, angle, buttons, hold)` makes
  the shim's next decision a given action instead of a sample. The policy still runs,
  so the shim's observation, logits, recurrent state, latch and own-action tail are
  what the net would have had if it had picked that action. The action is
  legalised against the shim's own latch and mask, and `bopl_agent_forced_fixed`
  reports what changed (bit flags). The non-forced path is unchanged:
  `check_agent_fidelity sim` gives byte-identical output with the old and new library.
- `eval/human_dataset.py`: replays to training rows (below). It uses its own build of the
  shim (`native/artifacts/rl/human-bc/libbopl_agent.so`, `make OUT=...`), so the
  shared library is untouched until the parent rebuilds it.
- `train/human_bc.py`: BC trainer (distill.py's loader and loss helpers, single-seat sequences).
- `eval/human_aggression.py`: aggression.py's movement metrics for the human seats of the
  converted replays (the human reference for the BC and RL nets).
- `eval/aggression.py --temperature`: sample the subjects at a temperature.

## 1. Dataset

Sources:
- `native/artifacts/rl/human-mac` (559 .rep, 17 sessions, all gamepad, human vs bot).
- `native/artifacts/human-replays` (974 rows; human-human and human-vs-agent). Sessions
  are cut at file-mtime gaps of more than 45 minutes. Seats played on a keyboard
  (48 games) are skipped.

Rows are deduplicated by replay sha256, Mac first. Kits are read from the
state's slimes at frame 240, so a Random pick (ability id 1) is resolved by the
game's RNG. Only 2-player games for now: f5 is a 1v1 net, and the 3p games (419)
would need a 3p observation f5 never trained on. They are left for an ablation.

Per replay:
- Re-step the recording in the bit-exact q32 Rust sim (`ReplayWorld`, warmup 0, so frame f is episode tick f).
- Feed each human seat's state to an agent shim loaded with f5, from frame 240 (episode tick f-240).
- At every decision (every 4 ticks), force the converted human action and record
  f5's observation (v4, 4812 floats, f16), mask, the action, whether the buttons
  were latched, the legaliser flags, f5's per-head log-prob of the action, and f5's value.

Conversion of the 4 recorded ticks of a decision window to one (aim, buttons, hold):

- **aim** = the most frequent angle byte in the window (ties go to the earliest).
  Byte 0 is neutral. The bins are the game's own angle bytes, so there is no spatial quantisation.
- **buttons** = each of jump and the three ability slots is pressed if it is down on at least 2 of the 4 ticks.
- **hold** = 0 except to charge Bow. A bow slot down for n consecutive decisions
  becomes one latch of the nearest length in {1,2,4,8,16} decisions (ties to the shorter),
  because the mask bans a bow slot that was down at the last free decision. A
  charged shot must therefore be a single latch. Bow is stripped for the rest of a
  human run that outlasts its latch.

Of 9 rules tried (aim mode/first/last × buttons or/majority/first, 150 replays, fork test below),
mode-majority had the lowest 32-tick trajectory error. Majority buttons drop taps
shorter than 2 ticks. "Or" keeps taps but drifts more (0.329 vs 0.284).

Lossy parts, all faithful rows:

| | |
|---|---|
| ticks whose aim differs from its window's applied aim | 28.9% [24.4, 32.8] |
| ticks whose buttons differ (taps < 2 ticks, release mid-window) | 4.3% [3.4, 5.4] |
| bow charge length error \|L - n\| | 3.2 decisions mean, 1,765 bow runs |
| legaliser fixes | 0 |

Fidelity is tested by forking the sim every 8 decisions and playing the converted
packets for the human seat (the other seat replays its recording). The reference is the
human's exact recorded inputs one tick late. Session-cluster bootstrap, 95% CIs:

| | converted | 1-tick-lag reference |
|---|---|---|
| position error after 4 ticks (median, units) | 0.000 | 0.000 |
| after 32 ticks (median) | 0.205 [0.172, 0.246] | 0.172 [0.155, 0.197] |
| after 32 ticks (p90) | 1.23 [1.02, 1.41] | 0.58 [0.55, 0.62] |
| whole round open loop: recorded winner reproduced | 54.1% | 53.3% |

The paired winner difference is +0.009 [-0.029, +0.049]. A round is chaotic: one
tick of lag changes the winner almost half the time. The converted actions are as
good as a one-tick lag, and no conversion can do better.

Self-check: bot seats were converted with the bot's own checkpoint (21 faithful games).
- The conversion is exactly lossless: tick mismatches 0, fork error 0, winner 21/21.
- The bot's own NLL is 4.47 nats/decision (aim 3.27).
- "Faithful" means the sim ends within 4 ticks of the recorded end with the recorded winners.
  Unfaithful replays score 6-19 nats on the bot's own actions, so they are dropped.

Result: `native/artifacts/rl/human-bc/data-2p/`.
- 607 replays, 774 human seats, 689 faithful (Mac 380, old 309).
- 239,505 decisions, 20 sessions.
- Validation: 3 whole sessions (`old-20260916-165454`, `mac-20261010-003004`,
  `mac-20260917-000139`), 102 seats and 43,286 decisions (18%).
- Kits: Dash/Bow/Platform (2,4,14) is 64% of seats and Dash/Grenade/Drill (2,3,26) 10%.

How f5 rates the human's actions (free decisions; all faithful rows, session bootstrap):

| | NLL nats/decision | argmax agreement |
|---|---|---|
| aim | 44.9 [42.0, 47.7] | 0.7% (median angle error 67°) |
| buttons | 11.3 [10.5, 11.9] | 61.9% |
| hold | 0.46 [0.25, 0.70] | 98.8% |

f5's Fourier aim head is extremely sharp: it puts almost no mass where Matthew aims,
and almost none on neutral (p 0.0006 vs the human's 16%).

## 2. Behaviour cloning

`train/human_bc.py`: v4 mingru, f5's architecture (hidden 256, aim_fourier 4, critic 384). Sequences
are single seats, 32 sequences of 32 decisions (BPTT 32) per step, for 12 epochs at lr 3e-4. The loss
is the NLL of the human action on free decisions (latched decisions carry no choice). A value
term (coefficient 1) regresses the critic onto f5's value of the same state, recorded with the row,
so RL starts from a fitted critic. Checkpoints are .bin plus .json, loadable by train.py
`--init-weights`, the evaluators and the shim.

Validation: the 3 held-out sessions, best epoch, session bootstrap 95% CIs.

| init | best epoch | total NLL | aim NLL | aim acc | aim error (median) | buttons NLL | buttons acc |
|---|---|---|---|---|---|---|---|
| random, cosine LR (`bc-randcos`) | 10 | **3.88** [3.46, 4.12] | 3.30 | 0.21 | 4.2° | 0.51 | 0.88 |
| f5, actor re-initialised (`bc-f5re`) | 11 | 4.17 [3.80, 4.38] | 3.57 | 0.18 | 7.1° | 0.52 | 0.88 |
| f5, cosine LR (`bc-f5cos`) | 12 | 4.32 [3.91, 4.55] | 3.71 | 0.18 | 8.5° | 0.56 | 0.87 |
| f5 (`bc-f5`) | 10 | 4.32 [3.88, 4.59] | 3.72 | 0.18 | 8.5° | 0.55 | 0.88 |
| f5 trunk frozen, heads trained (`bc-f5head`) | 11 | 5.11 [4.76, 5.33] | 4.42 | 0.16 | 16.9° | 0.62 | 0.86 |
| f5 itself (reference) | - | 49.2 | 38.5 | 0.005 | 67° | 11.1 | 0.62 |

f5's features do not help: the random init fits Matthew best, and freezing f5's trunk costs
1.2 nats. All the BC nets match the human's marginals: P(neutral aim) is 0.16-0.21 (human 0.157)
and P(any button) is 0.22-0.26 (human 0.244). f5 itself puts 0.002 on neutral and 0.96 on pressing.

Play (eval/aggression.py, 200 games per cell from both seats; distance in units, closing/retreating
are the fractions of decisions the gap shrinks/grows). Human column: eval/human_aggression.py on
the replays.

| Dash/Bow/Platform | human | bc-randcos | bc-f5 | f5 |
|---|---|---|---|---|
| win vs uniform-hard bot | - | 0.63 [0.56, 0.69] | 0.66 [0.59, 0.72] | 0.96 [0.93, 0.99] |
| win vs f5 | - | 0.005 [0, 0.015] | 0.04 [0.015, 0.07] | 0.51 |
| distance | 26.6 [25.0, 28.0] | 27.1 vs f5, 31.5 vs bot | 25.8 / 31.6 | 31.2 / 31.9 |
| closing | 0.45 [0.40, 0.49] | 0.50 / 0.55 | 0.54 / 0.60 | 0.32 / 0.58 |
| game length vs f5 (s) | - | 16.7 | 15.9 | 147 |

On Dash/Grenade/Drill both BC nets win 0.04-0.05 vs f5 and 0.76-0.79 vs the bot. The BC nets
press in, as the human does, and f5 punishes it: they lose in about 16 s, while f5 vs f5 lasts 2.5 minutes.

## 3. KL-to-reference (train.py `--kl-reference`, `--kl-coef`)

A frozen copy of a reference net (the BC net) scores every learner action during the rollout.
Each update, the learner's reward for decision t is reduced by beta * (log pi(a_t) - log rho(a_t)),
paid at reward t+1, which GAE credits to action t (the last step of the horizon is uncredited).
That is the sampled KL(pi || rho) as a reward penalty, as in RLHF and AlphaStar's
human-statistics anchor. `kl_reference` in metrics.jsonl is its mean over learner decisions,
logged even at `--kl-coef 0`.

- pufferlib-bopl.patch: `bopl_ref_logprob_warp` uses the sampler's own game logits, mask and head
  gating, so log pi and log rho compare like with like. Also `bopl_kl_penalty`, a ref bank with
  its own recurrent state (reset at episode boundaries like the learner's), and
  `_C.load_reference_bank`.
- The reference runs in the per-bank rollout chain. A KL run therefore disables the batched
  chain, and runs at 120-145k learner steps/s instead of about 700k.
- Off (no `--kl-reference`) nothing runs. A serial run (BOPL_ASYNC=0, TF32) of the new binary
  reproduced a checkpoint sha256 of the old binary (39085e...). Serial runs are not bit-deterministic
  run to run on either binary (3 distinct hashes in 5 runs), so a match is the strongest check available.
- Sanity: KL is exactly 0 at the first update (pi = rho). A BC learner against f5 as reference reads
  57 nats/decision, consistent with f5's 49-nat NLL on human actions.

### Bug fixed on the way: cuBLASLt workspace race in the per-bank chain

`train/bopl_v4.cu` ran every cuBLASLt GEMM on one global 32 MB workspace. The per-bank rollout chain
runs v4 forwards concurrently on the 4 buffer streams and the bank side streams (and, async, beside
the train stream), so split-K partial sums of concurrent GEMMs overwrote each other.
- Evidence: the same BC net, seed and config. The per-bank chain played 135 decisions/episode with
  6.6 deaths per 1k decisions; the batched chain played 357 and 2.6.
- Fix: one workspace per stream, the rule cuBLAS handles already followed (kernels.cu), sized to what
  the stream's algorithms need. Algorithm choice is unchanged. LtPlan storage is now a deque, so
  references survive other threads' push_back.
- After the fix the per-bank chain gives 376 decisions/episode and 2.5 deaths per 1k decisions,
  and the same learning dynamics as the batched chain.
- Production was not affected: the batched rollout chain never calls cuBLASLt, and no training
  log in tp-spec or exploiters-2 shows the batched chain disabled.
- Regression check: `train/check_rollout_paths.py` plays one checkpoint at lr 0 through both chains
  and fails if deaths, kills, episodes or win share differ by more than 4 SE.
