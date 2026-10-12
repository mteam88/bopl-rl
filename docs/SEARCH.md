# Decision-time search, phase 1 (search-v1)

A policy+value guided lookahead on forks of the live game, wrapped around an
existing checkpoint and measured against the same checkpoint played raw. Built
2026-10-11/12 in the `search-v1` rift; results below are the g111 checkpoint
`native/artifacts/rl/sweeps/g111-league-1/000/checkpoints/policy-12800032768.bin`
(hidden 256, 1 MinGRU layer, aim_fourier 4). Rows live in
`native/artifacts/rl/sweeps/search-v1/all_combos.jsonl` and `h2h.jsonl`
(artifacts are not tracked; this file is the record).

## Code

- `train/bopl_search.h`: the engine. A `BoplSearch` owns a copy of the
  checkpoint, a batch-2 CPU evaluator (`train/policy_eval.c`), and `k*n`
  shadow environments: struct copies of the live `BoplPuffer` with their own
  rows, masks, delay slots and a forked world (`bopl_rl_fork`, 17 us), stepped
  with the training step functions, so rollout observations carry the same
  nine-tick delay and latch bookkeeping as training.
- `train/eval_binding.c`: `research_search_*` exports in `libbopl_research.so`.
- `eval/search_policy.py`: ctypes `Handle`, the `Search` participant
  (`elo_ladder.Model` contract: one handle per worker thread, counters per
  game, `summary()`), `parse_budget`, and `--bench`.
- `league/all_combos.py --search <budget>`: A is the search-wrapped first
  checkpoint, B the raw second one; the JSON row carries `search=` (budget,
  games, decisions, ms per decision, rollouts per searched decision).
- `native/artifacts/rl/sweeps/search-v1/h2h.py --search <budget>`: the same
  on the 16 trained lineups (copy of `capacity-widen/h2h.py`).
- The research library compiles `policy_eval.c` with
  `-O3 -march=native -ffast-math`: 800 -> 88 us per policy row at hidden 256
  (logits and values agree to five decimals), 9x. Batching does not help (the
  evaluator's loops are per row), so the library is machine-specific.

Commits in the rift: `8b751410` (engine, exports, participant, `--search`),
`ac40fa7d` (common random numbers, prior-mixed selection, budget parser),
`78817f89` (one root forward per live decision, per-candidate return
variance), plus this document.

## What a decision costs (PowerSpec, one thread, box under a training load of 14-19)

| piece | cost |
| --- | --- |
| `bopl_rl_fork` | 17 us |
| one 4-tick decision step of a shadow (both players' actions applied, rows encoded) | 45 us |
| policy forward, one row, plain `-O3` | 800 us (batch 1..64 all the same per row) |
| policy forward, one row, `-march=native -ffast-math` | 88 us |
| full shadow decision (reset + fork + step + 2-row forward) | 0.22 ms |

A searched decision with `k` candidates, `n` rollouts each and depth `d` costs
about `2*k*n*d` policy rows (both players at every rollout step) plus `k*n*d`
sim steps plus `k*n` forks: `k=8,n=1,d=8` is 130 rows and 64 steps, 16.6 ms
measured (11.4 ms of it policy). The raw policy is one row, 0.1 ms. The
policy forward is the whole budget; the simulator is a tenth of it.

## The search

Per decision of player `me` (every 4 ticks):

1. Root forward of both players' real rows (advances the real recurrent
   states once per live tick; a second `act()` on the same tick, the other
   player's, reuses it). The root value of both players is recorded.
2. Candidates: Gumbel-top-k over the joint masked prior, aim x buttons x hold
   (`k` distinct joint actions; the latch mask zeroes held buttons and the
   hold head when `hold_left>0`, aim stays free).
3. For each candidate, `n` rollouts: fork the world, play the candidate at
   the root while the opponent samples its root policy (the opponent model is
   the same checkpoint: self-model), then both players follow the policy for
   `depth-1` more decisions, accumulating `gamma^t` x training reward (kill
   +0.288, death -0.288, win +1, loss -1, timeout -1) and bootstrapping the
   leaf with the value head (critic with the privileged block, encoded from
   the fork's undelayed state exactly as in training). Rollout `i` of every
   candidate shares the random stream of rollout `i` of the others (common
   random numbers), so the candidates' returns differ by the action, not the
   dice.
4. Optional sequential halving (`h`): after the first round the top half by
   mean return gets another `n` rollouts, and so on.
5. Pick: `score = value_scale * mean_return + prior_mix * (prior_logp +
   Gumbel)`. The defaults (`vs=1, mix=0`) are plain argmax of the mean
   return; `mix=1, vs=V` is the Gumbel-top-k "improved policy" pick that
   keeps the prior in the loop.
6. The chosen action is committed; the environment's latch keeps the buttons
   for the hold length, and the search runs again next decision (aim is free
   while latched; `latched=0` samples instead while latched).

Budget string (`eval/search_policy.py: parse_budget`):
`k,n,depth[,h][,mix=M][,vs=V][,latched=0]`

| field | meaning | default |
| --- | --- | --- |
| `k` | candidates drawn Gumbel-top-k from the prior (<= 64) | required |
| `n` | rollouts per candidate per round | required |
| `depth` | rollout length in decisions (4 ticks each; 8 = 0.53 s of game) | required |
| `h` | sequential halving: successive rounds keep the top half | off |
| `mix=M` | weight of `prior_logp + Gumbel` in the pick | 0 |
| `vs=V` | weight of the mean rollout return in the pick | 1 |
| `latched=0` | sample the prior while latched instead of searching aim | search |
| `gamma` | discount, the training 0.995 (kwarg only) | 0.995 |

Per-candidate stats for a later expert-iteration target are exported
(`Handle.stats()`): aim, buttons, hold, visits, mean return, prior logp,
chosen, return variance; `Handle.counters()` gives decisions, searched,
rollouts, steps, forward rows, seconds and both root values.

## Results: search-wrapped g111 vs raw g111, all-combo league (side-swapped seed pairs, 4 workers)

| budget | score | 95% bootstrap CI | games | ms / decision | rollouts / decision | policy rows / decision |
| --- | --- | --- | --- | --- | --- | --- |
| raw vs raw (baseline, l1-main2 vs g111 reference) | 0.506 | | 1000 | 0.1 | 0 | 1 |
| `4,1,4` | 0.575 | [0.526, 0.623] | 400 | 4.9 | 4 | 34 |
| `8,1,8,mix=1,vs=20` | 0.603 | [0.546, 0.658] | 400 | 16.2 | 8 | 130 |
| `8,1,8` | **0.672** | [0.621, 0.723] | 400 | 16.6 | 8 | 130 |
| `16,1,8,h` | 0.677 | [0.626, 0.726] | 400 | 59.8 | 30 | 480 |
| `8,8,8,h` | 0.690 | [0.590, 0.790] | 100 | 219.8 | 112 | 1788 |
| `8,8,8,h,mix=1,vs=20` | 0.690 | [0.590, 0.790] | 100 | 231.2 | 112 | 1788 |

Ablations at 100 pairs (200 games), same setup:

| budget | score | 95% CI | ms / decision | rollouts | rows / decision | note |
| --- | --- | --- | --- | --- | --- | --- |
| `8,1,1,mix=1,vs=0` | 0.530 | [0.455, 0.605] | 7.9 | 8 | 18 | null control: 8 candidates, prior-only pick |
| `8,1,1` | 0.598 | [0.518, 0.680] | 8.2 | 8 | 18 | one-step rollouts: the critic one decision ahead |
| `8,1,2` | 0.560 | [0.478, 0.645] | 12.1 | 8 | 34 | two-step rollouts |
| `2,1,8` | pending | | | | | |
| `8,1,8,latched=0` | pending | | | | | no search while latched |
| `8,1,16` | pending | | | | | |

The null control sits at 0.5: picking among the Gumbel-top-k candidates by the
prior alone is just sampling the policy, so the gain is the returns, not the
candidate machinery. One-step rollouts (`8,1,1`: the critic's value of the
state one decision ahead, 8 ms) already give 0.60 of the 0.67, so most of the
signal is the critic ranking the candidates' immediate consequences.

On g111's own 16 trained lineups (`h2h.py`, the league's rating condition)
the gain is smaller and not resolved: `8,1,8` scores 0.292 [0.167, 0.438] over
48 games (seed-base 91001) and 0.573 [0.492, 0.651] over 192 games (seed-base
92001), 0.517 pooled over 240. The search's gain is where the net is unsure
(random any-3 kits), not where it is sharp; `sweeps/search-v1/h2h.jsonl`.

Reading it:

- The success bar (>60-65%) is met at an affordable budget: `8,1,8` scores
  0.672 (+17 points, the lower CI bound 0.62) at 16.6 ms per decision, which
  is real-time (a decision is 66.7 ms of game) on one core.
- Strength saturates fast. Four times the budget (`16,1,8,h`, 60 ms) buys
  nothing measurable (0.677), and 13x (`8,8,8,h`, 220 ms) gives 0.69 on a wide
  CI. The knee is at `8,1,8`.
- Keeping the prior in the pick (`mix=1,vs=20`) hurts (0.603): the prior is
  what the raw opponent already plays, and the value signal, scaled by 20, is
  small against a Gumbel of unit scale. Pure argmax of the mean return wins.
  At the big budget the mix is neutral (`8,8,8,h,mix=1,vs=20` 0.690, the same
  as `8,8,8,h`): with 112 rollouts the mean returns separate enough that the
  scaled value dominates the Gumbel anyway.
- The same search on the stronger all-kit net gives the same gain:
  `8,1,8` around allkit-1@14667m scores 0.660 [0.582, 0.733] against raw
  allkit-1@14667m (200 games, 23.0 ms per decision on the loaded box;
  `native/artifacts/rl/sweeps/ei-1/all_combos.jsonl`). The search's gain is
  not a g111 artefact, which is what expert iteration (docs/EI.md) needs.
- Per ability (pooled 8,1,8 and 16,1,8,h, 800 games): the gain is broad,
  lowest on Black hole (0.51), Grappling hook (0.58), Platform (0.59),
  Grenade (0.60), highest on Gold mushroom (0.83), Roll (0.77), Dash (0.77),
  Gust (0.75), Tesla coil (0.73). Abilities whose payoff is delayed or
  indirect (Black hole, Platform, Grenade) gain the least from an 8-decision
  (0.5 s) horizon.

## Why it works despite a noisy value head (diagnosis)

Critic check (24 raw self-play games on random any-3 kits, root values of
both players every decision): `V0-V1` correlates with the final outcome at
-0.01 in the first tenth of the game, +0.12 by mid-game, +0.45 at 75-90% and
+0.66 in the last tenth; mean `|V|` is 0.05 early and 0.22 at the end. The
value head is near-useless as a game-outcome predictor until the end, and
its decision-to-decision jitter (std of dV 0.016) is of the order of its
level. It is not a long-horizon oracle.

Search-signal check (`k=8, n=8`, so every candidate has eight returns):
between-candidate variance of mean returns over within-candidate rollout
variance (ANOVA F) has median 0.73 at depth 8 and 0.40 at depth 32; only 31%
(18%) of searched decisions at depth 8 have F > 2 (> 5), 7% (3%) at depth 32.
The candidate means spread by 0.003 at the median, 0.017 at the 90th
percentile, against rollout std 0.003 (depth 8) and 0.008 (depth 32). The
best candidate is the prior's top choice only 10-12% of the time.

So the search's edge comes from the minority of decisions where the signal is
large: imminent kills, deaths and off-map falls inside the 0.5 s horizon,
where the shaped reward (0.288) and the terminal reward dominate the leaf
value and the candidates differ by far more than the noise. Most decisions
are a coin flip among reasonable prior samples, which is harmless (the pick is
still a prior sample). Deeper rollouts do not help because the rollout
variance grows faster than the signal (F falls from 0.73 to 0.40 from depth 8
to 32) and the leaf value is as noisy as the return it replaces; `8,1,16` in
the ablation batch tests this on games.

## Recommended budget for phase 2 (expert iteration)

`8,1,8` (argmax, no halving) for the acting policy and for the search
targets. Reasons: it holds the whole measured gain (0.67) at 16.6 ms, 130
policy rows per decision; `16,1,8,h` costs 3.6x for nothing, and
`8,8,8,h` 13x for a gain inside its CI. For targets that need more than one
return per candidate (a variance estimate, or a completed-Q policy
improvement that trusts the means), `8,2,8` doubles the cost to about 33 ms
and is the most that is worth testing before the targets are known to help.

Data-generation arithmetic at `8,1,8`: 16.6 ms per searched decision per
core, so 4 cores give ~240 decisions/s, ~0.2 games/s (games average ~1,200
decisions), ~750 games or ~0.9 M searched decisions per hour. Training used
13 G environment steps; expert iteration will see three to four orders of
magnitude fewer searched decisions, so phase 2 should be a distillation of
search picks into the policy on a fixed buffer, not a replacement for PPO.
The policy forward is 70% of the cost; batching the `2*k*n` rollout rows per
step across the shadows into one GEMM (the CUDA trainer path, or a BLAS-backed
evaluator) is the lever if more data is needed, and the shadows are already
laid out for it (`roll_state`, one row per rollout per player).

What the target should be: the exported stats per decision (aim, buttons,
hold, visits, mean return, prior logp, chosen, variance). With `n=1` the
visit counts are flat, so the useful target is the improved policy of the
Gumbel MuZero paper: `softmax(prior_logp + sigma(completed_q))` over the `k`
candidates with the chosen action's advantage, or simply the chosen action as
a hard target weighted by its F-like signal (spread over rollout std) so the
coin-flip decisions do not pull the policy anywhere. The value target should
be the mean rollout return at the root (it is better than the head's own
estimate where the two differ, see the diagnosis).

## Pitfalls

- `systemd-run` units do not inherit the shell's cwd or `PATH`: give the
  script an absolute path and `--working-directory`. A relative path fails
  with status 127 and no useful log line.
- Open an `Environment` only after `league/games.configure(workspace)`
  (sets `BOPL_RL_BACKEND/GAME/BOOTSTRAP/LOG_DIR`); otherwise the process
  aborts with "Set BOPL_RL_GAME...".
- `make research` compiles `policy_eval.c` against `$(PUFFERLIB)` under
  `$(OUT)`; building with another `OUT` needs `PUFFERLIB=` pointed at the
  pinned PufferLib source. `BOPL_RESEARCH_LIBRARY=<path>` makes
  `eval/environment.py` load a library built elsewhere.
- The Makefile rule only rebuilds when a listed source changes; after editing
  only flags, `rm` the `.so`.
- `-ffast-math` is applied to the evaluator only (the ABI and binding stay
  `-O3 -std=c11`); the outputs were checked against the plain build on real
  rows (five decimals) before trusting it.
- A harness that drives both sides through one `Handle` must not advance the
  recurrent state twice: fixed in `78817f89` by keying the root forward on the
  live tick. `Search` participants use one handle per worker thread and act
  for one side, so they were not affected.
- `by_ability` and `search` in the all_combos JSON row make a run
  self-describing; the per-game list is `games` and the row is appended, so
  re-running a budget appends a second row rather than replacing it.
