# Expert iteration: distilling the decision-time search into the all-kit net

Phase 2 of the search work (docs/SEARCH.md is phase 1). Goal: the raw net keeps
most of the +16 points the `8,1,8` search adds, on `league/all_combos.py`.
Base: `allkit-1@14667m` (tp-spec `sweeps/allkit-1/run/checkpoints/policy-14666694656.bin`,
v4, hidden 256, `--loadout-pool all`), which has plateaued at 0.65-0.67 against
g111-l1-12800m.

Code:

- `eval/search_record.py`: plays a checkpoint under the search on random any-3
  kits (`all_combos.POOL`) and random maps (`registry.MAPS`, 124), both seats
  searched, and writes one npz per game (fp16 obs rows, masks, the k candidates'
  search stats, action, root value, training reward, outcome; docstring has the
  layout). Resumable: existing `g<seed>.npz` are skipped, `games.jsonl` lists
  what was recorded.
- `train/distill.py`: fine-tunes the net in torch (`eval/policy_v4.PolicyV4`,
  which matches the C evaluator to 1e-4) on those games, teacher-forced from
  the zero state with truncated BPTT, and writes native checkpoints.
- `train/distill_diag.py`: replays held-out games through base and student and
  bins the search's candidate-return spread: did the student learn to *rank* the
  candidates the way the search did, where the ranking mattered?
- `native/artifacts/rl/sweeps/ei-1/{measure,train,eval}.sh`: the unit scripts.

## 1. Does the search still add on allkit-1?

Yes, the same as on g111. `8,1,8` around allkit-1@14667m vs raw allkit-1@14667m,
100 side-swapped pairs: **0.660 [0.582, 0.733]** (200 games, 23.0 ms per searched
decision on the loaded box, 8 rollouts, 130 policy rows per decision;
`sweeps/ei-1/all_combos.jsonl`). Phase 1 had 0.672 [0.621, 0.723] on g111.

On g111's own 16 trained lineups the gain is smaller and not resolved (0.29
over 48 games, 0.57 over 192; docs/SEARCH.md): the search helps where the net
is unsure, which is the all-kit condition.

## 2. Data

Datagen = the search playing itself, both seats searched (`search_record.py`),
`8,1,8`, random kits from the 36-ability pool, random map, 10800 ticks:

- 48.5 s of one thread per game (both seats; 1,240 decisions per game on average,
  24 ms per seat-decision), 2.2 KB per decision compressed, ~2.6 MB per game.
- Two units of 3 workers (`bopl-ei1-gen-a/-b`, nice 10, MemoryHigh 12G) on the
  shared box: ~4.5 games/min, ~5,600 decisions/min, ~1.5 M training rows (two
  seats) per 300 games per hour-and-a-bit.
- Iteration 0: 1,200 games requested (seeds 1000-1599, 2000-2599); 3 timeouts and
  5 draws in the first 239, so almost every game ends in a kill.

What the search actually does, from the recorded candidates (12 games, 30 k
decisions): the k=8 candidates' mean returns differ by 0.003 at the median,
0.022 at the 90th percentile, more than 0.05 on 3% of decisions and more than
0.2 (a kill) on 1%. The search picks a candidate other than the highest-prior
one on 81% of decisions, mostly on differences of 1e-4. So per decision the
search is close to "a random one of 8 Gumbel-top-k samples", with a slight,
consistent edge, and the +16 points come from that edge compounded over 1,200
decisions, not from a few large corrections.

## 3. Loss

Why not an aux term inside PPO: the trainer's loss lives in a CUDA kernel
(pufferlib-bopl.patch); adding a per-row KL target means new buffers, a new
kernel path and a replay of searched sequences through the native rollout
buffer. Behaviour cloning in torch over `PolicyV4` needed none of that, trains
in minutes, and the KL anchor keeps it from leaving the base's basin. PPO can
resume from the distilled checkpoint afterwards (same architecture, same
checkpoint format).

`train/distill.py`, per recorded decision (both seats), teacher-forced replay so
the recurrent state is the student's own:

    policy   sum over heads CE(target_h, student_h)
    anchor   kl * sum over heads KL(base_h || student_h)        (kl = 0.3)
    value    value * (v_student - v_root_base)^2                (value = 1)

with the recorded masks applied to base, target and student. Two targets:

- `--target onehot` (used): the one-hot of the action the search chose, per
  head. The search is a deterministic argmax over k Gumbel-top-k prior samples
  under common random numbers, so over the data this is exactly the search
  policy's action distribution, including the part of it that is sampling
  without replacement (it plays a flatter aim than the raw net: 2.1 vs 1.6 nats
  on held-out rows after distillation). Held-out log-prob of the search's
  choice goes from -4.44 (base) to -4.04 after two epochs on 146 games, so the
  choices are predictable from the observation.
- `--target soft`: the Gumbel-MuZero completed-Q improved policy over the joint
  action space, pi'(a) = pi_base(a) exp(beta (q(a) - q_bar)) for the candidates
  (q their mean rollout return, q_bar the prior-weighted mean) and pi_base
  elsewhere, marginalised exactly onto the three heads (the M-projection onto
  factorised policies). With the Gumbel-MuZero scale (beta = 50, the paper's
  c_visit) the target moves 0.012 nats per decision from the base and the
  student does not learn it from 300 games (held-out policy term rises from
  0.0121 to 0.0155): the per-decision edge is below what the data can resolve
  at that scale, which is the same fact as above. Kept as an option.

The anchor weight has a clean reading: CE(target) + kl * KL(base || student) is
minimised by the mixture (target + kl * base) / (1 + kl), so kl = 0.3 asks for
77% search policy, 23% base.

Training cost: 1,700 rows/s on the RTX 5070 shared with `bopl-allkit-1c`
(32 games = 64 sequences per bucket, BPTT 32), so an epoch over 300 games
(770 k rows) is 7.5 minutes. `bopl-allkit-1c` was not stopped.

## 4. Results (all_combos, 500 pairs, seed-base 131001, tp-spec results.jsonl)

| label | games | data | target | vs allkit-1@14667m | vs g111-l1-12800m |
| --- | --- | --- | --- | --- | --- |
| allkit-1@13.3B (reference) | | | | | 0.661 [0.629, 0.696] |
| `ei1@14667445776` (run 0a) | 146 | iter0 | onehot, kl 0.3, 2 epochs | 0.507 [0.473, 0.539] | (not run) |
| `ei1@14668744656` (run 0c) | 386 | iter0 | onehot, kl 0.3, 3 epochs | 0.521 [0.488, 0.556] | (not run) |
| `ei1@14670992528` (run 0d) | 536 | iter0 | onehot gated by candidate spread (`--weight-spread 0.02`), kl 0.3, 3 epochs | 0.486 [0.451, 0.520] | (not run) |

## 5. What the student learned (run 0c, 20 held-out games, 50 k decisions)

`train/distill_diag.py`, per bin of the candidates' return spread (max - min of
the k=8 mean returns). `pref` is the net's probability of the search's choice
given it plays one of the 8 candidates (1/8 = no preference); `rank` the
Spearman correlation between the net's candidate log-probs and the search's
returns; `chosen` the log-prob of the search's action.

| spread | share of decisions | agree base / student | chosen base / student | pref base / student | rank base / student |
| --- | --- | --- | --- | --- | --- |
| < 0.005 | 63% | 0.055 / 0.052 | -4.11 / -3.68 | 0.105 / 0.112 | 0.02 / 0.05 |
| 0.005-0.02 | 28% | 0.117 / 0.106 | -5.04 / -4.14 | 0.158 / 0.153 | 0.06 / 0.05 |
| 0.02-0.05 | 7% | 0.118 / 0.096 | -6.56 / -4.79 | 0.155 / 0.140 | 0.08 / 0.07 |
| 0.05-0.2 | 1.8% | 0.130 / 0.115 | -7.01 / -5.00 | 0.165 / 0.153 | 0.11 / 0.12 |
| > 0.2 | 0.2% | 0.112 / 0.103 | -7.19 / -5.14 | 0.155 / 0.151 | 0.09 / 0.11 |

The student raised the log-prob of the search's action everywhere (by 2 nats
where the base was confident), but its preference among the candidates did not
move, on any bin, and its argmax agrees with the search slightly less than the
base's does. It learned the search's sampling pattern (a flatter policy, the
Gumbel-top-k without-replacement part) and none of its selection. The base
already prefers the search's pick a little (0.16 vs 0.125 on the bins that
matter), and the data did not sharpen that.

Why the selection is hard to clone: with one rollout per candidate the chosen
candidate is the argmax of eight return estimates whose spread is 0.003 at the
median, so the label is mostly rollout noise around a small systematic edge,
and the systematic part comes from the simulator state (no 9-tick observation
delay, the opponent's rollout actions under common random numbers), which the
observation does not contain. A per-decision one-hot label carries very little
of the +16 points; it has to be averaged over far more decisions per situation
than 1,200 games give, if it is recoverable from the observation at all.

Units `bopl-ei1-eval-0c` and `-0d` played only the allkit-1 leg (`OPPS` set to
it); the g111 leg was skipped because neither student differs from its base.
`bopl-ei1-gen-a`/`-b` finished iter0: 1,200 games (seeds 1000-1599,
2000-2599), ~1.6 M decisions, both seats.

The same diagnostic with the selection gain in return units (`qg`, x1000: the
student's expected candidate return minus the base's, and the search's own pick
minus the base's), on the 20 newest iter0 games (recorded after both students
were trained; `sweeps/ei-1/diag-newest.txt`):

| spread | share | qg 0c | qg 0d | qg search |
| --- | --- | --- | --- | --- |
| < 0.005 | 63% | -0.00 | -0.00 | 0.49 |
| 0.005-0.02 | 27% | -0.13 | -0.08 | 4.7 |
| 0.02-0.05 | 7-8% | -0.39 | -0.27 | 12.3 |
| 0.05-0.2 | 2% | -0.75 | -0.92 | 27 |
| > 0.2 | 0.2% | 1.9 | 1.05 | 230 |
| all | | -0.07 | -0.06 | 3.5 |

Both onehot students select *worse* among the candidates than the base,
everywhere except the 0.2% kill/death decisions. Spread gating (0d) does not
change that.

## 6. A listwise target on the same data (runs 1a, 1b)

The onehot target asks the net to raise the search's action against the whole
joint action space, and that is mostly the search's sampling pattern. The part
that carries the gain is the *selection* among the k candidates, so
`--target listwise` (`train/distill.py`) trains only that:

    policy   KL(t || softmax_c log pi_student(c))   over the k candidates c
    t(c)  ∝  pi_base(c) exp(beta q(c))              (completed-Q improved policy on the searched set)

Where the returns are equal, `t` is the base's own preference and the term has
no gradient, so the 63% coin-flip decisions pull nowhere and the high-spread
ones carry the update; everything outside the candidate set is left to the
anchor. A held-out number in return units goes with it: `q_gain`, the
expected candidate return under the student's candidate distribution minus the
base's (rollout noise on held-out games is independent of the net, so this is
an unbiased measure of how much of the search's selection the net took over),
next to `q_gain_search`, the search's own pick minus the base expectation.

On iter0 (841 training games, 2.2 M rows, 44 held out), kl 0.3, lr 1e-4:

| run | beta | held-out q_gain after 1 epoch | q_gain_search | pref student / base | held-out policy term epoch 0 -> 1 |
| --- | --- | --- | --- | --- | --- |
| 1a | 50 | -0.000007 | 0.0038 | 0.1317 / 0.1305 | 0.019 -> 0.029 |
| 1b | 200 | -0.000006 | 0.0038 | 0.1313 / 0.1305 | 0.123 -> 0.132 |

Nothing: the student's selection gain is zero to six decimals against the
search's 0.0038 per decision, and the per-bucket numbers (each bucket is
measured before the step that trains on it) were zero all through the epoch.
The held-out policy term *rose*: the net fit the training labels' noise. Both
runs were stopped after epoch 1 and not played (an all_combos eval is 4
CPU-hours for an expected 0.50).

How noisy is an n=1 label where it matters? One short game recorded with the
screen below (`8,1,8` on every decision, `8,4,8,h` again on the decisions whose
`8,1,8` spread is >= 0.02; the second search draws its own candidates, 3.7 of
8 shared on average): over the shared candidates, the within-decision return
deviations of the n=1 search and the n=4-plus-halving search correlate at
0.51 (596 pairs, 166 decisions), and the n=1 spread regresses from 0.046 to
0.028. So even on the high-spread decisions the n=1 label is about half
noise, and on the 63% below 0.005 it is almost all noise.

## 7. Iteration 1: label choice

Options, scored by expected gain per CPU hour:

- (b) Search returns as value targets, PPO turns them into policy. Rejected.
  The raw net never reads its value head at play time, so any gain has to come
  through a new PPO run on the GPU with a value-target path in the CUDA loss
  (pufferlib-bopl.patch), and PPO's critic already fits the real returns of
  13 G steps. The search's root return is an 8-decision lookahead
  bootstrapped on that same critic. A small, indirect lever on a plateaued run,
  at the most engineering.
- (c) High-spread decisions only. Already inside the listwise target: equal
  returns give no gradient, and beta weights decisions by their spread. 1a/1b
  show that this alone doesn't help on n=1 labels.
- (a) Several rollouts per candidate with halving. This attacks the measured
  problem, label noise (reliability ~0.5 at n=1 even where the spread is
  large), but `8,4,8,h` is 56 rollouts per decision against 8, so labelling
  every decision costs 7x.
- Chosen: **(a)+(c) screened**. Play under `8,1,8` as in iter0 and use it as a
  screen. Where its candidate spread is >= 0.02 (about 1 decision in 10;
  more early in games), search again at `8,4,8,h` on the same tick, play that
  pick, and keep only those decisions as labels (`search_record.py
  --label-budget 8,4,8,h --screen-spread 0.02`). Train `--target listwise` on
  them. Each game costs about 2x an iter0 game, and every label is a
  high-spread decision with 4-12 rollouts per candidate, where the search's
  gain lives. The screen's own n=1 candidates are kept (`screen`) for the
  noise measurement above.

This is also the clean test of what limits distillation. If cleaner labels on
the decisions that matter give held-out `q_gain > 0`, the limit was noise and
more data pays. If they don't, the search's selection is not recoverable from
the observation: it reads the simulator's exact, undelayed state through the
privileged critic, which the delayed observation does not contain (`8,1,1`,
the critic one decision ahead, already gives 0.60 of the 0.66). In that case
EI by distillation is done at this scale, and the search should be used at
play time.

Side test, zero training: sample the raw net sharper. The search's pick at a
critical decision is in part "don't take the rare bad sample", and a lower
softmax temperature does some of that for free. A temperature can be baked
into the checkpoint by scaling the actor rows. `all_combos.py --temperature-a`,
`sweeps/ei-1/temp.sh`, 200 pairs against T=1.

Temperature result: T=0.75 on all three heads scores **0.492 [0.438, 0.544]**
against T=1 (200 pairs, `sweeps/ei-1/all_combos.jsonl`). Sharpening buys
nothing, so T=0.5 was stopped unplayed to give the CPU to iter1 generation. The
trained entropy is about right for this net.
