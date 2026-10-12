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
- `native/artifacts/rl/sweeps/ei-1/{measure,train,eval}.sh`: the unit scripts.

## 1. Does the search still add on allkit-1?

Yes, the same as on g111. `8,1,8` around allkit-1@14667m vs raw allkit-1@14667m,
100 side-swapped pairs: **0.660 [0.582, 0.733]** (200 games, 23.0 ms per searched
decision on the loaded box, 8 rollouts, 130 policy rows per decision;
`sweeps/ei-1/all_combos.jsonl`). Phase 1 had 0.672 [0.621, 0.723] on g111.

Where the gain is not: on g111's own 16 trained lineups (`h2h.py`, the league's
rating condition) the same search scores 0.29 [0.17, 0.44] over 48 games, i.e.
it *loses* (a 96-pair repeat is in `sweeps/search-v1/h2h.jsonl`). The search
helps where the net is unsure (random kits) and hurts where it is sharp. See
"what the search actually does" below; it is also why the distillation target
needs care.

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
| `ei1@14667445776` (run 0a) | 146 | iter0 | onehot, kl 0.3, 2 epochs | pending | pending |

## 5. Next

(filled in with the results)
