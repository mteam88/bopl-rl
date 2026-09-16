# Policy v4: observation and network, from scratch

Status: design (2026-09-16). Sections marked *measured* are filled in as the
code lands; everything else is the plan the code is built to.

## 1. Why a new policy, not a wider v3

Overnight-9's capacity study (v3 at hidden 128 / 256 / 512, one and two
layers) probes width. Width is not where v3 is weak. The v3 encoder
(`bopl_set_encoder.h`) is one linear projection per row followed by ReLU and a
max over the rows, one 128-float pooled vector per set, one linear fusion and
a single-layer MinGRU. Its failure modes, from the layout-v3 review
(`LAYOUT_V3_PLAN.md`) and from the loss-taxonomy thread's first findings:

- A linear row projection cannot compute products of features. "Is this arrow
  going to hit me" is `relative velocity . relative position` divided by a
  norm: three multiplications the row encoder cannot express, so the policy has
  to learn it downstream from a max-pooled summary that already discarded which
  row was which. Every strong entity-set policy hand-computes these relations
  (OpenAI Five: per-unit distance and relative position to the controlled hero,
  Berner et al. 2019 appendix; Necto/Nexto for Rocket League: relative position
  and relative velocity per entity; collision-avoidance RL uses CPA/TCPA,
  closest point of approach and time to it, as inputs).
- Max pooling alone reports the most extreme row per channel and loses counts
  and averages ("how many arrows are in the air" needs a sum or mean; Deep Sets,
  Zaheer et al. 2017, is about sum/mean decompositions). v3 patches this with 39
  hand counts in the prefix.
- The MinGRU highway gate is the only residual path; nothing is normalised. A
  256-wide MinGRU on unnormalised fusion outputs is exactly the regime where
  pre-LN residual blocks were introduced (Xiong et al. 2020 for transformers;
  GTrXL, Parisotto et al. 2020, for RL memory; SimBa, Lee et al. 2024, for RL
  MLPs).
- The value comes from one extra row of the 128 x 278 decoder matrix. On-policy
  RL is usually limited by value estimation quality (Andrychowicz et al. 2020
  found the separate, larger value network one of the most reliable wins; SimBa
  and BRO, Nauman et al. 2024, scale the critic 4x wider than the actor).
- The critic sees only what the actor sees: a 9-tick-delayed, visibility-
  filtered view. The baseline in AlphaStar (Vinyals et al. 2019) and the
  asymmetric actor-critic of Pinto et al. 2017 use the full state; training-only
  privileged inputs cost nothing at play time.

v4 rebuilds all four: relational per-row features, a per-row MLP with max and
mean pooling, a normalised residual core, a wide separate critic with an
optional privileged block. The wire ABI (`bopl_rl.h`, header/entity version 3)
is unchanged: every v4 feature is arithmetic on state already in it.

## 2. Observation layout v4 (`bopl_layout_v4.h`, `layout.py` mirror)

One row per player, self-relative coordinates (positions minus the player's
position, /64 as in v3), velocities /100 (walk 19, jump 58, arrow 75 u/s, so
arrows no longer saturate a /64 scale and are not squashed by v3's /256).
Times in seconds /2 unless noted. `clip` = clamp to [-1, 1]. Fixed scales,
no running statistics: every scale is a game constant and the checkpoint is
then a pure function of the weights (nothing to export, nothing to drift).

| Block | Rows x width | Contents |
|---|---|---|
| global | 66 | v3 prefix with columns 8-9 (absolute self x/64, y/64) zeroed: clocks, time scale, counts, 39 per-ability entity counts, sector counts, time stop. No absolute position anywhere in the row (see "Map invariance" below) |
| self | 44 | v3 self 20 (id/team/alive/stocks/kills/deaths/pos/scale/flags/timers) + controlling slime: vx, vy, speed, rotation sin/cos, state one-hot 11, in-ability, input-suppressed, spawned, abilityless, scale, grounded (vy==0 and vx==0 is the sim's grounded encoding), pickup slot |
| opponent | 56 | the primary enemy (nearest visible enemy player's controlling slime): the same 44 as self, plus relative: dx, dy, distance, bearing sin/cos, dvx, dvy, closing speed, tangential speed, time to closest approach, miss distance, position-known flag. Zero with known=0 while invisible. A fixed slot for the one opponent that matters in 1v1, so the policy does not depend on pooling to find it |
| previous | 12 | v3 own-action tail unchanged |
| terrain | 12 | egocentric, from the static-collider SDF the scripted bots use (`bot_wall_sdf`): drop to the surface below /32, ceiling above /32, distance to the edge left and right (floor falls away by more than 4 units within one dash, 13 units) /13 with a wall-blocks flag per side, height above the water /32, distance to lethal terrain /16, ground within one dash (some of 16 dash endpoints has a surface within 4 units below), SDF at the body /8 and its gradient |
| rays | 16 x 6 | the primary geometry channel: 16 rays from the body at 22.5 degree steps, length 32: distance /32 (1 = nothing), hit class one-hot wall / lethal wall / water / dynamic entity, closing speed of an entity hit along the ray. Walls by sphere-marching the SDF, water as the surface plane, entities as circles of their radius |
| players | 6 x 32 | v3 20 + the 12 relational columns (below) |
| slots | 24 x 18 | v3 13 + remaining cooldown /10 s, position-known, owner-is-primary-enemy, owner distance /64, ability id (last column) |
| entities | 48 x 48 | v3 33 + the 12 relational columns (surface distance; ballistic pair for projectiles: nearest approach of the gravity arc, 40 u/s^2 measured on the sim, and its time), distance and miss distance of the same body seen from the primary enemy (my arrow versus the opponent), ability id (last column). Static colliders keep their rows (exact extents and orientation) but the network is meant to read terrain from the rays and terrain block; the rows are the secondary, exact description |
| slimes | 32 x 42 | v3 30 (its spare columns now: owner stocks, owner-is-primary-enemy, is-a-controlling-body, abilityless) + the 12 relational columns |
| privileged | 48 | critic only, never read by the actor: the primary enemy's undelayed true state (relative position, velocity, state, rotation, flags, three slots' remaining cooldown/ready/active/charge, invisibility alpha, latched buttons, hold left, last aim) the staleness of the actor's view (true relative enemy position minus the delayed one) and the observer's own undelayed velocity. Written by the environment from the current state, next to the 9-tick-delayed actor blocks |

Total 4606 floats (v3: 3690, +25%; H2D upload is 6% of the rollout step,
THROUGHPUT.md 1.E). *Measured*: the v4 writer costs 6.3 us per row against
v3's 3.3 (research_pack_layout on a map-0 state with 17 entities, one core);
at 24576 rows over 16 environment threads that is about 4.6 ms more per
rollout step (20-36 ms today). The terrain block's edge walk and landing
search are the heavy part and the first thing to trim if the smoke shows the
rollout environment-bound.

Map invariance (parent's finding, 2026-09-16: 024 wins 90% on training maps
0-2 against the scripted bots and 78% on the held-out maps 3-4; 299 of
Matthew's 330 real games were on maps outside the sim's five). With three
training maps an absolute position is a map-and-location lookup key that
lets the policy skip the geometry rows (Cobbe et al. 2018 / 2020 on level
memorisation), so v4 has none: the v3 prefix's self x/y are zero, every other
position is relative to the observer, and the map is seen through the
egocentric terrain block and the 16 rays (over static colliders, the water
surface and dynamic entities). The static-collider entity rows remain as the
exact description but are secondary. Whether the relative wall rows still
leak location is a question for the probes thread; if they do, the lever is
to cap the number of static-collider rows (they are a truncation-by-distance
set already). Entity rows drop from 64 to 48: overnight-8 mixed-loadout
p90 was 34 rows and maps 0-2 with two players never exceed 20; truncation is
by distance and non-fatal, as before.

Row ordering keeps v3's rules (nearest first, pickups first, the controlling
slime's slots first, rope spans thinned) with one change: the opponent's
controlling slime's slots are forced in, position-hidden, when the opponent is
invisible, so cooldown information does not vanish with the body.

Relational definitions, per row with relative position r and relative
velocity v (entity minus self): distance = |r| - radius (surface); bearing =
r/|r|; closing speed = -(r.v)/|r|; tangential = (r x v)/|r|; TCA = clamp(-(r.v)
/(v.v), 0, 2 s); miss = |r + v TCA|. The ballistic pair samples the arc at
1/15 s steps to 2 s. All are O(1) per row in the writer; the whole v4 encode is
measured against v3's in section 6.

Every v3 writer rule about visibility (`bopl_spatial_hidden`), delay and
truncation applies unchanged. The layout is selected per player
(`BoplPuffer.layout[player]`); the row stride is the larger layout so both
fit in one environment buffer.

## 3. Network

```
flat  = ReLU(W_f [global, self, opponent, previous, terrain, rays])       286 -> 256
row_s = ReLU(W2_s ReLU(W1_s x + E[id]))  per valid row of set s          F_s -> 128 -> 128
pool_s = [max_rows row_s , mean_rows row_s]                               256 per set
u     = LN(W_u [flat, pool_player, pool_slot, pool_entity, pool_slime])   1280 -> 256
h     = MinGRU(u)   (upstream cell, highway gate: h = g*cell + (1-g)*u)   256, 1 layer
y     = h + W_o ReLU(W_i LN(h))                                           256 -> 512 -> 256
logits = W_a y                                                            256 -> 30 (aim Fourier K=4: 9, buttons 16, hold 5)
value  = W_v LN(z + W_b2 ReLU(W_b1 LN(z))),  z = W_z LN([y, privileged])  304 -> 512 -> 512 -> 512 -> 1
```

- Per-row MLP, two layers, 128 wide, ability embedding added before the first
  ReLU as in v3 (PointNet, Qi et al. 2017; Deep Sets). Max and mean pooling
  concatenated: max keeps "the nearest / most threatening", mean keeps counts
  and averages (Deep Sets; the OpenAI Five encoder max-pools, the mean is the
  cheap fix for what its prefix counts had to carry).
- Attention over entities: evaluated, not built (section 7). The relational
  features give each row its relation to the observer and to the primary enemy
  directly, which is the pairwise information a 1v1 policy needs; a full
  variable-length attention forward and backward in the fused CUDA path costs
  more than the rest of v4 and its benefit is what the probes thread is
  measuring offline. The encoder leaves a hook: one extra per-set stage between
  the row MLP and the pool.
- Core: pre-LN into the MinGRU (unchanged upstream cell and scan, so the
  horizon-128 parallel scan and its parity tests stay), then a pre-LN residual
  FFN block after it. Width 256. LayerNorm has no affine parameters: the trainer
  is Muon for matrices and momentum SGD at the same learning rate for
  everything else, under which a gain would barely move from 1 (Xu et al. 2019
  find the affine terms are not what makes LN work).
- Critic: SimBa block (pre-LN residual MLP, post-LN, Lee et al. 2024), 512 wide,
  one block, reading the trunk output and, when `--critic-privileged 1`, the
  privileged block. Training-only: the checkpoint carries the weights (so the
  CPU evaluator can report values in decision dumps), the Unity shim never runs
  it. `--critic-detach 1` stops its gradient into the trunk (ablation).
- Heads: unchanged Fourier aim head (K=4, `bopl_aim.h`), buttons, hold; the
  entropy head-scale plumbing is untouched because the decoder row format
  (logits then value) is exactly v3's.
- Parameters: actor path 0.93M (sets 88k, embedding 5k, flat 46k, fusion 328k,
  MinGRU 197k, FFN 262k, heads 8k); critic 0.68M; total 1.61M. v3 at hidden
  128: 0.15M.

Learning rate. Hilton et al. 2023 keep the Adam learning rate proportional to
the initialisation scale, 1/sqrt(width) under Kaiming init. This trainer is
Muon (`muon.cu`): the update of every matrix is orthogonalised and scaled by
sqrt(max(1, rows/cols)), which makes its RMS per element 1/sqrt(fan_in), the
same scaling as the initialisation, at every width. Muon's step is therefore
already width-invariant in Hilton's sense and the default is to keep 024's
3.431e-4. `--lr-width-exponent p` multiplies the learning rate by
(hidden/128)^p for anyone wanting the Adam rule (p = -0.5); it is recorded in
the sidecar config.

## 4. Compatibility

- Wire ABI unchanged (`bopl_rl.h` version 3).
- Observation layout per policy, on the eval/scoring path only: the v3 writer
  stays as `bopl_encode_v3`, v4 is `bopl_encode_v4`, `BoplPuffer.layout[p]`
  selects per player, the buffer row stride is `BOPL_OBS_STRIDE = max`. The
  research binding gets `research_set_layout(handle, player, version)`;
  `native_policy.load` reads the architecture string; `reference_pool.py` and
  `elo_ladder.py` set each side's layout from its checkpoint. The Unity shim
  picks the writer from the sidecar. Training builds always write v4 for every
  row (opponents are frozen learners or scripted bots).
- Architecture string `bopl-v4-mingru`; sidecar gains `layout_version: 4`,
  `set_dim`, `set_inner`, `ffn_dim`, `critic_dim`, `critic_privileged`,
  `lr_width_exponent`. `obs_size` is the v4 row width. Weight order is
  documented in `layout.py` (`policy_v4_weights`) and every tensor starts on an
  8-float boundary, as v3.
- v3 bit-identity: `research/check_policy_golden.py` verifies 409 recorded
  decisions of pool v7's v3b-007-final (observations, raw decoder rows,
  recurrent state, sampled actions) after every change on this branch.

## 5. Implementation

CUDA training path (`bopl_v4_encoder.cu`, `bopl_v4_decoder.cu`, own files,
selected by `BOPL_NATIVE_ENCODER=v4`):

- Rows of every set are compacted across the minibatch into one packed
  (rows, F) bf16 matrix per set by a deterministic per-sample prefix sum. The
  row MLP runs as bf16 tensor-core GEMMs with the row count read on the device,
  so the launch shape is fixed and CUDA-graph capture keeps working while the
  work is proportional to the valid rows (about 24 per sample on maps 0-2
  against a capacity of 110). Backward: segmented max/mean unpool (winner
  indices saved in the forward), the two GEMMs per layer with a fixed-chunk
  split-K reduction over the variable row count, embedding gradient by
  segmented sum. All deterministic.
- Flat, fusion, FFN, heads and critic are fixed-shape GEMMs through the tuned
  cuBLAS path (`bopl_gemm`), bf16 operands from producers this code owns.
- LayerNorm forward/backward kernels (no affine).
- The decoder is a custom `Decoder` whose activations embed upstream's
  `DecoderActivations` first, so the PPO loss, the fused gradient write and the
  sampler keep reading the row `[logits..., value]` at the same stride.
- Rollout: the per-bank chain works unchanged through the module interfaces;
  the batched grouped chain (`bopl_rollout.cu`) gets a v4 branch if the smoke
  shows the rollout step is inference-bound.

CPU evaluator (`policy_eval.c`): a v4 forward next to the v3 one, dispatched
on the architecture string; `bopl_agent.c` inherits it.

Reference: `research/policy_v4.py`, a torch module over the flat weight
buffer, is the oracle for the CUDA parity harness (`check_v4_encoder.py`:
forward, backward, finite differences, gathered/permuted rows, graph replay,
the crowded rows from `check_observations.c`) and for the CPU evaluator.

## 6. Throughput (*measured* sections filled by the smoke)

Budget: overnight-9's recipe at w128 spends 21 ms per update in training and
20-36 ms in rollout (THROUGHPUT.md). The task allows 2x for a clear Elo gain,
not 5x.

GEMM micro-benchmark (torch, bf16, 24576-row minibatch, fwd+bwd, on the
shared GPU while a sweep held it at 98%, so absolute numbers are roughly 2x
pessimistic; ratios hold): fusion 1280->256 2.3 ms, MinGRU projection 1.6,
FFN 1.6, critic 512 4.1 (384: 2.6, 256: 1.5), row MLP over 600k packed rows
11.5 (300k: 5.6); the v3 dense layers 1.5. Projection: v4-full ~15 ms of GEMM
per minibatch against v3's 1.5, times 7 minibatches per update, i.e. an update
near 3x v3's if nothing else moved. Levers, in the order they will be pulled
if the measured smoke is above 2x: critic width 384, set inner width 64
(PointNet's narrow first layer), FFN off. All are flags; the configs in
section 8 name the shapes.

Rollout inference: the v3 batched chain is 0.095 ms per step; the per-bank
fallback 0.21 ms. v4 adds LN, FFN and critic launches. Measured in the smoke.

## 7. Ablation plan (sweep configs, `sweep/configs/v4-*.toml`)

All at overnight-9 024's recipe and steps, scored on pool v7 (v3 opponents
via the per-player layout), against `capacity-h256` as the width-only control:

1. `v4-full`: everything above, privileged critic on.
2. `v4-sym-critic`: privileged block off (is the asymmetric critic worth it).
3. `v4-no-relational`: relational columns zeroed in the writer (same shapes;
   isolates the feature work from the encoder work).
4. `v4-maxpool`: mean pool off (max only, v3 style).
5. `v4-lite`: critic 384, set inner 64, FFN off: the within-2x shape.
6. `v4-critic-shared`: critic reads the trunk through the actor's small
   decoder row as v3 does (is the wide separate critic the gain).
7. `v4-no-rays`: terrain block and rays zeroed in the writer (same shapes):
   is the egocentric geometry channel what carries held-out-map play.
8. Held-out maps: every v4 config is also scored on maps 3-4 against the
   scripted bots (the in-run evaluation already does it); the map-invariance
   claim is that the v4 gap between maps 0-2 and 3-4 is smaller than 024's
   12 points.

Queue order: `v4-full`, `v4-lite`, `v4-sym-critic`, `v4-no-relational`,
`v4-critic-shared`, `v4-maxpool`, `v4-no-rays`; if the smoke measures
`v4-full` above ~2.5x overall (environment plus update, sps, not the GEMM
projection of section 6) `v4-lite` goes first. Both smokes report sps.

Success criterion: not a fixed Elo. The 024 recipe is being re-run at seeds
74/75/76 ahead of v4 in the queue; the spread of those three on pool v7 is
the seed floor. `v4-full` is a win when its pool-v7 Elo exceeds the best of
the three seeds by more than their spread (and by more than the width-only
control `capacity-h256` manages), and its held-out-map win rate against the
scripted bots is within half of 024's 12-point training/held-out gap. A
result inside the seed spread is a null result, whatever its sign.

Attention: if `v4-full` beats v3 by a margin the probes attribute to
relations the fixed features do not cover, a one-layer attention stage over
the entity set is the next experiment, costed first.

## 8. Config and flags

`train.py --policy v4` (default `v3` so existing configs are unchanged) with
`--set-dim 128 --set-inner 128 --ffn-dim 512 --critic-dim 512
--critic-privileged 1 --critic-detach 0 --mean-pool 1 --lr-width-exponent 0`.
`--hidden 256 --layers 1` as before. All are recorded in the checkpoint
sidecar config and honoured by the evaluator.
