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
- Regression check: `train/check_rollout_paths.py` plays human-bc.bin through both chains at
  lr 1e-20 (train.py rejects 0), 3M learner steps each, about 2 minutes on the GPU. It fails if
  deaths, kills, episodes or win share per decision differ by more than 4 SE.
  - Fixed binary: ok, max |z| 1.1.
  - Pre-fix binary: FAIL, win share 0.036 vs 0.154 (z +35), deaths z -9.8. Here the race corrupted
    f5's forwards as well as the learner's.

## 4. RL from the BC net against f5

Setup: `native/artifacts/rl/sweeps/human-exploiter/train.sh`.
- f5's training config (anticamp hyperparameters), learner-first kits from lineups-human-exploiter.txt.
- 256 workers × 4 buffers × 4 banks, horizon 128.
- Init bc-randcos, which is `human-bc.bin`.

Win share and score are against f5 unless the text says otherwise.

### Probes, 20M learner steps

The last window is 18-20M learner steps. "outcome" is the mean of win 1, loss -1, so score = (1 + outcome) / 2.

| probe | entropy | dec/ep | dash/min | kills / 1k dec | outcome | KL to BC |
|---|---|---|---|---|---|---|
| no KL | 0.10 | 1243 | 0.0 | 0.00 | -0.983 | - |
| no KL, per-bank chain (Lt-fixed) | 0.45 | 1190 | 0.0 | 0.00 | -0.989 | - |
| pressure-hx style, timeout reward off | 0.18 | 1228 | 0.2 | 0.00 | -0.977 | - |
| pressure ×10 | 0.16 | 1213 | 0.0 | 0.00 | -0.988 | - |
| entropy 0.01, floor 0.003, head scale 1 | 4.89 | 1109 | 0.0 | 0.00 | -0.978 | - |
| γ 0.9997, timeout reward off | 1.68 | 874 | 0.0 | 0.00 | -0.989 | - |
| init bc-f5re | 1.07 | 1392 | 0.0 | 0.00 | -0.989 | - |
| init bc-f5head (f5 trunk) | 3.69 | 1747 | 0.6 | 0.00 | -0.902 | - |
| KL β 0.01 | 0.12 | 1355 | 0.0 | 0.00 | -0.984 | 0.045 |
| KL β 0.01 + pressure | 0.11 | 1318 | 0.0 | 0.00 | -0.983 | 0.045 |
| **KL β 0.03, γ 0.999** | 3.08 | 250 | 6.7 | 0.05 | -0.955 | 0.005 |
| KL β 0.03, γ 0.999 + pressure | 3.03 | 267 | 6.4 | 0.05 | -0.960 | 0.006 |
| KL β 0.1 | 2.99 | 266 | 6.7 | 0.04 | -0.964 | 0.003 |
| KL β 0.1, γ 0.999 | 3.26 | 216 | 7.6 | 0.05 | -0.963 | 0.001 |
| KL β 0.3 | 3.25 | 213 | 7.6 | 0.05 | -0.962 | 0.0005 |
| KL β 1.0 | 3.30 | 200 | 7.8 | 0.05 | -0.966 | 0.0002 |

The BC net starts at about 2% wins against f5. With β < 0.03 and no KL, every variant collapses
within 10M steps to the same passive policy, and so does every non-KL fix tried: entropy floor,
γ, a pressure style at ×10, an f5-trunk init and a value warmup (the critic is fitted to f5's
value; EV 0.86-0.97).
- The passive policy stops pressing buttons (button entropy below 0.01) and plays about 1200
  decisions an episode.
- It never kills. It dies more slowly, so it loses later.

With β ≥ 0.03 the policy stays BC-like: 7 dashes a minute, about 250 decisions an episode,
KL 0.0002-0.006 nats a decision. In 20M steps it does not get better either.

Why β must be at least 0.03: KL(π‖ρ) is mode-seeking, so collapsing onto one of the BC net's
low-entropy modes costs little KL. A penalty of β = 0.01 (0.045 nats × 0.01 per decision) is small
next to the advantage the passive mode buys, which is dying later.

### Long runs (β 0.03, γ 0.999, pressure-hx, timeout reward off)

| run | opponents (bank groups, pool) | learner steps | in-training result | end state |
|---|---|---|---|---|
| rl-kl03 | f5 only (league ×4, pool 2) | 142M | win vs f5 0.022 → 0.027, flat from 40M | dec/ep 273 → 474, drifting passive |
| rl-cur | f5 + v4-lite-1920m + l1-main-2666m + own snapshots (all,all,all,league) | 80M | stuck on its first snapshot for 50M, then the hard members drove passivity | win 0.24 vs the pool, about 800 dec/ep, button entropy 0.09 |
| **rl-self** | f5 + own snapshots (all ×4, pool 8, snapshot 10M, opponent timeout 10M) | 200M (checkpoint policy-400031744, global steps) | win vs pool 0.50-0.62, f5 score 0.045 [0.038, 0.051] at 0-50M → 0.088 [0.055, 0.122] at 200M | stable: kills/ep 0.19 → 0.26, dec/ep 265 → 330, entropy 2.4 → 2.2, KL 0.06 |

- Only the self-snapshot pool stayed stable.
- The f5-only pool and the pool with weak and strong league members both drifted passive.
- rl-cur's pitfall: the PFSP band gate only retires an opponent whose score CI lies outside
  15-85%. The opponent timeout is the only escape, so set `--opponent-timeout`.

### Paired evaluation against f5

eval/exploit_eval.py, 200 seed pairs × both seats on the human-exploiter kits, the same seeds for
every subject. Score is win 1 and draw 0.5, with a 95% bootstrap CI over seed pairs. The control is f5 as the subject.

| subject | score vs f5 | diff vs humanbc | diff vs control | game s | closing | uses/min |
|---|---|---|---|---|---|---|
| control (f5) | 0.341 [0.290, 0.394] | - | - | 123 | 0.30 | 8.8 |
| humanbc (BC net) | 0.031 [0.016, 0.048] | - | -0.310 [-0.364, -0.258] | 13.4 | 0.47 | 13.8 |
| rl-kl03 at 142M | **0.065** [0.044, 0.089] | **+0.034** [+0.009, +0.060] | -0.276 [-0.331, -0.223] | 33.5 | 0.39 | 11.1 |
| rl-cur at 80M | 0.035 [0.019, 0.052] | +0.004 | - | 51.8 | - | - |
| rl-self at 100M | 0.046 [0.029, 0.065] | +0.015 [-0.009, +0.039] | - | 37.8 | 0.35 | 11.3 |
| rl-self at 200M | 0.046 [0.028, 0.068] | +0.015 [-0.010, +0.041] | -0.295 [-0.350, -0.242] | 47.4 | 0.33 | 10.7 |

- rl-self at 200M minus rl-self at 100M is +0.000 [-0.025, +0.026]. rl-self at 200M minus rl-kl03 is -0.019 [-0.049, +0.011].
- The in-training f5 rise does not show up in the paired eval.
- No net comes close to the control, so nothing was registered.

Style drift: KL β 0.03 keeps the action distribution near the BC net (0.06 nats a decision), but
play moves toward f5's.
- Games last 13 s → 47 s.
- Closing falls 0.47 → 0.33 (f5 0.30).
- Item uses fall 13.8 → 10.7 a minute (f5 8.8).
- These differences are significant, all CIs exclude 0.

A higher β (0.1) holds the style more tightly but learned nothing in 20M steps.

## 5. Artifact and the l2 loop

`native/artifacts/rl/human-bc/human-bc.bin` (+ `.json`), sha256
8c66d73c877814d43bf8fd3020597889c2adbfebe8bbce852da23c4d7c32f8fd. This is bc-randcos epoch 10, the f5
architecture, loadable with `--init-weights`.

l2.toml reset point: league-exploiter resets (league_exploiter_reset = 0.25) should restart from
human-bc.bin with KL to it, instead of the newest main or a random init.

What the loop needs for human-style league-exploiter turns:
- **Code (tp-spec):**
  - Merge the `human-exploiter` commits: train.py `--kl-reference`/`--kl-coef`/`--gamma`,
    pufferlib-bopl.patch, train/bopl_rollout.cu, and the bopl_v4.cu Lt fix. Then rebuild `_C`
    with `train/build.py --device cuda` and run `train/check_rollout_paths.py`.
  - Add `kl_reference`, `kl_coef` and `gamma` to `sweep_config.TRAIN_FLAGS`. Without them the
    loop rejects the keys.
  - league_loop.py: a `[loop] league_exploiter_init = "native/artifacts/rl/human-bc/human-bc.bin"`
    path. `start_flags` uses it for a fresh league-exploiter run in place of the newest main, and
    `slot['init']` (the parent recorded by `add_member`) becomes `human-bc`.
  - Set `league_exploiter_random_init = false`.
- **[league-exploiter] flags:**
  - `kl_reference = "native/artifacts/rl/human-bc/human-bc.bin"`, `kl_coef = 0.03`, `gamma = 0.999`.
  - `kl_coef` below 0.03 collapses. 0.1 or more stays human-like but learns slowly.
  - Kits: `loadout_set = "native/data/lineups-human-exploiter.txt"`, `loadout_learner_first = true`.
    These are Matthew's kits; the BC data covers only them.
  - Optional: `style = "styles/pressure-hx.toml"` with `reward_timeout = -1`. It had no
    measurable effect in the probes.
  - Pool: the stable configuration was the run's own snapshots plus the target, with
    `bank_groups = "all,all,all,all"`, `pool_size = 8`, `snapshot_interval = 10000000` and
    `opponent_timeout = 10000000`.
  - l2's current league-exploiter pool (`league` ×4, pool 2, PFSP hard) is the configuration that
    drifted passive here.
- **Speed:** KL forces the per-bank rollout chain, 80-145k learner steps/s against about 700k
  batched.
  - One exploiter_turn of 1B learner steps takes about 2-3.5 hours.
  - The 4B league_exploiter_timeout takes about 8-14 hours.
  - The per-bank run uses about 11.5 of 12.2 GB of GPU memory.
- **Expectation:** at 0.05 vs f5 after 200M steps, a human-style exploiter will time out rather
  than beat the league. A timed-out turn is added as a member, so these turns add weak,
  human-style members.
