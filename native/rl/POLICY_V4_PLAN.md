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
| opponent | 56 | the primary enemy (nearest visible enemy player's controlling slime): the same 44 as self, plus relative: dx, dy, distance, bearing sin/cos, dvx, dvy, closing speed, tangential speed, time to closest approach, miss distance, position-known flag. Zero with known=0 while invisible. A fixed slot for the one opponent that matters in 1v1, read by the flat linear straight into fusion, never through a pooled set: the probes measured the v3 set path losing exactly this (foe dash cooldown R2 0.87 in the raw row vs 0.65 at the GRU, platform-ready AUC 0.997 vs 0.945, foe velocity 0.49 vs 0.32) |
| previous | 26 | v3 own-action tail (the decision as applied, latch left, hold runs) + the two decisions applied before it (no-aim flag, aim cos/sin, four buttons each). The observation is 9 ticks old and a decision is 4 ticks, so the last three applied decisions are exactly what the agent knows about its own inputs inside the delay window; the probes measured own velocity R2 0.72 delayed vs 1.00 undelayed (PROBES.md), and own inputs let the network dead-reckon its own state exactly and the foe's approximately |
| terrain | 12 | egocentric, from the static-collider SDF the scripted bots use (`bot_wall_sdf`): drop to the surface below /32, ceiling above /32, distance to the edge left and right (floor falls away by more than 4 units within one dash, 13 units) /13 with a wall-blocks flag per side, height above the water /32, distance to lethal terrain /16, ground within one dash (some of 16 dash endpoints has a surface within 4 units below), SDF at the body /8 and its gradient |
| rays | 36 x 8 | the primary geometry channel: 32 rays from the body at 11.25 degree steps plus 4 in a 120 degree fan about straight down (-135, -105, -75, -45 degrees), length 40: distance /40 (1 = nothing), hit class one-hot wall / lethal wall / water or blast plane / hostile projectile or object / opponent body / own object, closing speed of a body or object hit along the ray. Walls by sphere-marching the SDF, water and the blast planes as planes, projectiles and enemy bodies as circles of their radius. The probes' ray study (PROBES.md section 5) fixed the count: 32+4 rays alone beat the whole 1458-float v3 observation on falls and geometry (fall_30 AP 0.72 vs 0.16, drop-to-surface R2 0.94 vs 0.56, lands_dry AUC 0.98 vs 0.93) and 16+4 rays did not (hit_inflight_30 AP 0.18 vs 0.28); 0.1 us per ray. 16 rays is the `v4-rays16` ablation |
| players | 6 x 32 | v3 20 + the 12 relational columns (below) |
| slots | 24 x 18 | v3 13 + remaining cooldown /10 s, position-known, owner-is-primary-enemy, owner distance /64, ability id (last column) |
| entities | 48 x 48 | v3 33 + the 12 relational columns (surface distance; ballistic pair for projectiles: nearest approach of the gravity arc, 40 u/s^2 measured on the sim, and its time), distance and miss distance of the same body seen from the primary enemy (my arrow versus the opponent), ability id (last column). Static colliders keep their rows (exact extents and orientation) but the network is meant to read terrain from the rays and terrain block; the rows are the secondary, exact description |
| slimes | 32 x 42 | v3 30 (its spare columns now: owner stocks, owner-is-primary-enemy, is-a-controlling-body, abilityless) + the 12 relational columns |
| privileged | 48 | critic only, never read by the actor: the primary enemy's undelayed true state (relative position, velocity, state, rotation, flags, three slots' remaining cooldown/ready/active/charge, invisibility alpha, latched buttons, hold left, last aim) the staleness of the actor's view (true relative enemy position minus the delayed one) and the observer's own undelayed velocity. Written by the environment from the current state, next to the 9-tick-delayed actor blocks |

Total 4812 floats (v3: 3690, +30%; H2D upload is 6% of the rollout step,
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
egocentric terrain block and the 36 rays (over static colliders, the water
surface and blast planes, projectiles and enemy bodies). The probes confirmed
the direction: the v3 GRU state holds absolute position (R2 0.93) and a 0.1
shift of self x/y moves the policy 0.2 nats, while map identity is barely
held (0.70 vs 0.35 majority). The static-collider entity rows remain as the
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
flat  = ReLU(W_f [global, self, opponent, previous, terrain, rays])       492 -> 256
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
- Parameters: actor path 1.01M (sets 88k, embedding 5k, flat 126k, fusion 328k,
  MinGRU 197k, FFN 262k, heads 8k); critic 0.68M; total 1.69M. v3 at hidden
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

## 3b. The critic: at its ceiling, not a lever

Probes (2026-09-16, 024-final, gamma 0.995, gae_lambda 0.9, horizon 128,
vtrace clip 1.0): the value head's explained variance against the realised
gamma-discounted return over 906k of its own decisions is 0.11 (value std
0.16 against return std 0.54); against the full-episode TD(lambda=0.9) target
it is 0.57 (train.py's `explained_variance` 0.63 measures that bootstrap
target, ~90% the critic's own output) and against TD(0) 0.88.

That looked like an under-fit critic until the probes thread measured the
achievable ceiling on 024's own decisions (test split by game): an MLP from
the raw observation reaches EV 0.129, from the GRU state 0.134, with
privileged labels (idle-fork hit/fall/death within 60 ticks, the undelayed
opponent state, geometry) 0.149; only the opponent's identity, which is never
observable, lifts it to 0.21-0.23. The critic's 0.110 sits at that ceiling:
under this opponent mix the round outcome is mostly irreducible at decision
time, and no amount of critic scale, target choice or loss weighting can
recover variance that is not in the input.

Consequences for v4:

- Keep the critic as built (section 3: wide SimBa head, the privileged block
  behind `BOPL4_CRITIC_PRIVILEGED`, `--value-lambda` separate from
  `--gae-lambda` so the regression target's horizon is a flag), since the
  cost is already paid, and keep `v4-critic-shared` as one ablation.
- Log realised-return EV in-run (`value_ev_realised`, from the evaluation
  games' per-decision values against the realised discounted return) next
  to the bootstrapped EV, so the number that matters is visible without the
  offline tool.
- Not pursued, with the numbers above as the reason: PopArt / return
  normalisation, `vf_coef` 1.0 and 2.0, a separate critic learning rate, and
  the value-lambda sweep (0.9 / 0.95 / 0.98). Each would at best close
  0.110 -> 0.149 while the encoder, the relational and ray features and
  throughput decide the run.
- For later, not built: about half of the recoverable variance is the
  opponent's identity. A recurrent critic could infer it from history but the
  current GRU state does not (0.134 vs 0.21-0.23); an explicit opponent-style
  inference head (predict the opponent's bank / bot kind from the episode so
  far, feed the posterior to the critic) is a future experiment.

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
- Dump round trip: `puffer/check_dump.py` trains one rollout at a tiny shape
  for v3 (set) and v4 (`train.py --dump-rows`, `_C.bopl_policy_rows`) and
  asserts the CPU evaluator and the torch reference reproduce the trainer's
  own decoder rows from the checkpoint it wrote (v3 4e-5, v4 2e-3 relative;
  TF32/bf16 against fp32). Added after v4-full's first seven in-run
  evaluations read the MinGRU before the decoder (the trainer registers
  encoder, decoder, recurrent core): the trial's Elo read 250 at 560M learner
  steps with the swapped weights and 827 with the fixed reader. The swapped
  order is off by 1.4 relative here; nothing in the trainer consumed the bad
  evaluations (frozen banks are trainer-internal snapshots, the pool Elo is
  the trainer's own games, the driver scores after the run).

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
fallback 0.21 ms. v4's per-bank path is a chain of ~60 launches per bank per
buffer step, so with the learner and three frozen banks it was launch-bound
(5x v3's eval_gpu at 96 rows per buffer). The batched multi-bank v4 forward
(`bopl_rollout_v4.cu`, hooked into `bopl_rollout.cu`'s plan) runs every bank
in one chain of ~18 launches per buffer step: a block-per-(sample, set) kernel
for the per-row MLPs with bf16-rounded operands like the trainer's WMMA path,
the grouped GEMM for flat / fusion / MinGRU / FFN / actor / critic, and
row-wise LayerNorm / residual / privileged-concat glue; `br_gate` and
`br_sample` are v3's. `BOPL_BR_CHECK=1` compares its decoder rows with the
per-bank chain on eager steps (max |diff| ~7e-4 on logits of order 1, the
bf16 / TF32 rounding order); `BOPL_BATCHED_FWD=0` keeps the per-bank path as
the reference. At a quarter of 024's shape (32 workers, 1 buffer) the rollout
went from 0.224 s to 0.183 s per update; the full-shape number comes from the
smoke.

## 7. Ablation plan (sweep configs, `sweep/configs/v4-*.toml`)

All at overnight-9 024's recipe and steps, scored on pool v7 (v3 opponents
via the per-player layout), against `capacity-h256` as the width-only control:

Configs are `sweep/configs/v4-*.toml` (capacity-h256's form: one Protein
trial, 024's recipe verbatim, `encoder = "v4"`, the shape flags below; flags
not listed are `layout.V4_DEFAULTS`):

1. `v4-full`: everything above, privileged critic on.
2. `v4-sym-critic`: privileged block off (is the asymmetric critic worth it).
   `v4_critic_privileged = 0`.
3. `v4-no-relational`: relational columns zeroed in the writer (same shapes;
   isolates the feature work from the encoder work). `v4_ablate = 1`
   (`BOPL4_ABLATE` bit 1, bopl_obs_v4.h `bopl4_ablation`; recorded in the
   checkpoint's `v4_shape` so the evaluator writes the same rows). The bits
   are per player, like the layout: `env->ablate[player]` (bopl_puffer.h,
   0 = the process default `BOPL4_ABLATE`, which is all training does),
   `research_set_ablation` / `Environment.set_layout(side, 4, ablate=...)`
   from the sidecar's `v4_shape.ablate` (`native_policy.writer_ablation`,
   `elo_ladder.Model.attach`, evaluate.py), and the in-game shim's
   `BoplAgent.ablate`; so checkpoints of the four families 0 / 2 / 8 / 10
   share a game and a process (tournaments, two agents on the Mac).
4. `v4-maxpool`: mean pool off (max only, v3 style). `v4_mean_pool = 0`.
5. `v4-lite`: critic 384, set inner 64, FFN off: the within-2x shape.
   `v4_set_inner = 64`, `v4_ffn_dim = 0`, `v4_critic_dim = 384`.
6. `v4-critic-shared`: the v3-style value, one extra row of the actor linear
   from the trunk, no separate critic and no privileged block (is the wide
   separate critic the gain). `v4_critic_dim = 0`, `v4_critic_privileged = 0`;
   critic_dim 0 means exactly this in the trainer, the batched rollout, the
   CPU evaluator and the torch reference.
7. `v4-no-rays`: terrain block and rays zeroed in the writer (same shapes):
   is the egocentric geometry channel what carries held-out-map play.
   `v4_ablate = 2`.
8. `v4-rays16`: 16 of the 32 rays (every other one) and none of the down
   fan, same shapes: the probes' 16+4 vs 32+4 gap, in play. `v4_ablate = 4`.
9. Held-out maps: every v4 config is also scored on maps 3-4 against the
   scripted bots (the in-run evaluation already does it; evaluate.py and
   elo_ladder.py write each side's row in its checkpoint's layout, so v4
   candidates meet pool v7's v3 members and v3 frozen opponents); the
   map-invariance claim is that the v4 gap between maps 0-2 and 3-4 is
   smaller than 024's 12 points.

Queue order (set 2026-09-16 after `v4-full`, gpu-queue jobs 11-18): `v4-lite`,
`v4-no-rays`, `v4-full-s74` (10: `v4-full` verbatim at seed 74, v4's own seed
noise, which reads the ablations better than more v3 seeds), `v4-sym-critic`,
`v4-no-relational`, `v4-rays16`, `v4-critic-shared`, `v4-maxpool`; the v3 seed
job stays behind them and the v3 capacity study is parked. All eight run at
nice 0 (the queue runner too), `max_seconds` 21600, 024's 1734936776 steps,
scored on pool v7. The in-run eval hook rates every evaluation checkpoint on
pool v7 as it goes (160 games, `source` in-run, train.py) so the curve is on
the score's scale from the first checkpoint.

### 7b. Layout v4.1: the aim ray and line of sight (next increment, ablation 19)

Matthew played ~25 rounds against `v4-full` policy-1813348352 (maps 0-4,
2026-09-16): clearly better than v3 and fun, but "pretty bad at aiming and
situational awareness, it often tries to shoot through platforms". The 36-ray
fan (32 around, 4 down) has no ray along the aim direction: the policy has no
direct line-of-sight signal for its own shot and has to infer the platform
between it and the target from the terrain block and the fan. Next layout
increment, behind a `v4_ablate` bit so it runs as ablation 19 on the same
encoder build (rows stay `BOPL4_OBS_SIZE`; the bit selects what the writer
puts in the reserved/zeroed slots or replaces the least useful down rays):

- (a) aim ray: cast from the bow origin along the current aim angle (the
  previous decision's aim, which is what the next arrow would follow), the
  fan's 8 fields plus one bit "the opponent is the first hit".
- (b) line of sight to the opponent: blocked/unblocked, distance to the
  blocker, blocker class (terrain, platform, entity), computed as one cast
  from own centre to the opponent's centre.
- (c) candidate: the same for the opponent's aim toward us (their arrow
  path): first hit is us, distance, so the policy can read incoming shots.

Built 2026-09-16 as writer bit 8 (`v4_ablate = 8`, `BOPL4_ABLATE_AIM`,
bopl_obs_v4.h `bopl4_aim_rays`; config `v4-aim.toml` = `v4-full` plus the
bit, gpu-queue job 19). Rows stay 4812 and the encoder build is untouched:
two of the four down rays duplicate around rays (-135 and -45 degrees are
around rays 20 and 28, 11.25-degree multiples), so their slots (ray indices
32 and 35) carry the new casts, each in the fan's 8 fields (distance /40,
hit class one-hot, closing speed):
- slot 32, the aim ray: from the body along the aim of the decision applied
  this step (`BoplPuffer.applied`, set in `bopl_step_finish_from` before the
  rows are encoded; the shim and the evaluator share that path), zero when
  that decision had no aim. The opponent-body class is "the shot is on
  target", a wall class is "shooting into terrain" at that distance.
- slot 35, the line of sight: the cast toward the opponent's centre. The
  opponent-body class is a clear line; a wall, water or object class names
  the blocker at its distance; all zero when the opponent is unseen.
- (c) is not buildable: the wire state carries no opponent aim (only arrows
  in flight, which the rays and entity rows already show).
With bit 2 as well (`v4_ablate = 10`, config `v4-aim-only.toml`) the two
casts are the only thing on the channel: `v4-no-rays` plus the aim ray and
line of sight, no fan and no terrain block, the pair that attributes a lift
to the casts alone (`v4-no-rays` scored IQM 693 against `v4-full`'s 703 on
2026-09-17: the fan and terrain block are worth at most a few points at
1.73B, inside the seed band, while costing ~20% of learner sps).
Over 10k bot-game rows: the aim ray is present in 98% (56% meets nothing
within 40, 36% a wall, 4% the opponent); the line of sight is clear in 40%,
wall-blocked in 46%, beyond 40 units in 11%. The in-game shim applies the
sidecar's `v4_shape.ablate` (it ignored it before; the ablations were never
shipped), so a `v4-aim` checkpoint plays with its own rows
(check_agent_fidelity.py passes with the bit on).

Evidence check before building: the human replays are on the Mac at
`~/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl/human/.excluded/20260916-1717*.rep`
(excluded from the ladder at Matthew's request); the taxonomy tooling can
count arrows fired with a platform on the own-to-opponent segment, per game,
against the same count for pool-v7 members.

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

Own-platform probe (2026-09-17, after `v4-aim-only` scored 759 with 4x the
platform use of the fan shapes: 0.23-0.29 activations per game against
0.06): does the fan / terrain block flip when the caster's own platform
appears, so that casting reads as a state discontinuity? No. On the CPU
evaluator with the full geometry channel (`BOPL4_ABLATE=8`), the platform
entity appears 6 decisions (24 ticks) after the activation and the caster
stands still for the cast, so at the appearance decision the terrain block
does not move at all (0 of 103 own spawns, random self and the `v4-aim`
policy, changed any of the 12 features by more than 0.25; mean |delta| at or
below 0.004 against a 0.02-0.16 baseline) and 0.7-0.9 rays move against
2.4-3.3 at a baseline decision: the new wall enters one or two rays, which is
the geometry. The spawned platform is a wall (kind 2, layer 11) like the
level's, not an own-object circle (its entity owner is 0), so the hit class
is consistent. The opponent's spawns do move the learner's block (sdf flip
24% vs 3%, edge 15% vs 7%) because the bots drop platforms at the learner:
real events. So there is no writer bug to fix for a v4.2; the platform gap
is behavioural. `v4-aim-only` casts while airborne (all 14 casts in 60
games at |vy| > 0, 29% falling fast, 21% within 12 units of the water,
against 21% / 9% of all decisions), a rescue and mobility use that the fan
shapes' landing / edge features may make unnecessary.

Second seed (2026-09-17, `v4-aim-s74`: bands 669/650/649, IQM 648, squish
14.7% of kills on the scorer's checkpoints): the "fan suppresses platform
play" reading above does not survive it. Platform activations per game over
the last five evals (1085 games each, 217 per eval), with the squish share
of kills: `v4-aim` s73 0.06 (0.03-0.10 per eval; 0.27/min), squish 6.6%;
`v4-aim-s74` 0.19 (0.14-0.24; 0.70/min), squish 10.4% (6.9-16.0%, the
16.0% is the last eval); `v4-full` 0.20 / 9.3%; `v4-full-s74` 0.14 / 7.3%;
`v4-lite` 0.13 / 5.1%; `v4-no-rays` 0.29 / 10.9%; `v4-aim-only` 0.23
(0.18-0.30; 1.17/min) / 14.4%. The same shape spans 0.06-0.19 activations
and 6.6-10.4% squish across two seeds, which covers every shape's mean but
`v4-aim-only`'s 14.4%: `v4-aim` s73 was the low draw, not the fan. What
remained shape-attributable was small: the no-fan shapes (`no-rays`,
`aim-only`) at the top of the platform range and `aim-only` alone above
the two-seed squish band, one seed each. `v4-aim-only-s74` (2026-09-17
22:20) closed that too: 0.09 platforms per game (0.40/min), squish 5.3% of
kills (52/972) over its last five evals, against s73's 0.23 / 14.4%; so the
same shape spans 0.09-0.23 activations and 5-14% squish, the whole
between-shape range. Platform use and squish share are seed draws, not
shape. `v4-aim-only-s75` (00:20): 0.25 platforms per game (1.24/min) with
squish 6.8% (67/988), so the two are not even one behaviour: s75 casts as
often as s73 and squishes like s74. Aim-only over three seeds: 0.23 / 0.09
/ 0.25 platforms, 14.4 / 5.3 / 6.8% squish.

### 7c. Queue after the aim runs (2026-09-17)

Production shape = `v4-aim` (full + aim/LOS, rays on) unless `v4-aim-only`
clearly beats it (IQM above `v4-aim`'s 697 + ~30, outside the v4 seed
spread of ~27 measured by `v4-full` 703 / `v4-full-s74` 676). Revised
2026-09-17 with `v4-aim-s74` at 648: the same-shape spread is ~50 (aim
697/648, full 703/676), so single-seed differences below ~50 are noise, and
`v4-aim-only`'s 759 was +62..+111 over the two aim seeds, the only shape
effect that cleared the band. Revised again 2026-09-17 22:20 with
`v4-aim-only-s74` at 683 (bands 685/667/695; held-out gap 5.4, EV 0.08, 19
losses / 217, kills 68.7% arrow / 6.0% squish): 759 was a high draw, the
same-shape spread is up to ~75 (aim-only 759/683), and every v4 shape sits
in one 650-760 cluster (lite 676; no-rays 693; full 703/676; aim 697/648;
aim-only 759/683). The aim-only production-shape call of 13:22 is
withdrawn. Single-seed pool-v7 runs cannot rank v4 shapes; the only
resolved result is v4 over v3 (+40..+130 over 628/636). The production
shape is decided on the three-seed means, aim-only (s73/s74/s75) against
aim (s73/s74 plus a third if the means are within the spread), together
with the tournament's cross-family head-to-head (1267 pairs, running
overnight 2026-09-17/18 on the per-player ablation build); it defaults to
`v4-aim` on a tie because the more-maps runs and the Mac shim are built on
it. 2026-09-18 00:20, `v4-aim-only-s75`: bands 726/715/696, IQM 714
(held-out gap 14.9, EV 0.09, 20 losses / 217, kills 69.6% arrow / 6.9%
squish). Aim-only three seeds 759/683/714, mean 719 (s.e. ~22); aim two
seeds 697/648, mean 672; difference +47 with aim's third seed missing, and
aim-only ahead on every seed pair. Decided 2026-09-18 03:40 when the
tournament reported (68,400 games, 2,850 pairs, all four families
cross-played on the per-player writer-ablation stack;
`sweeps/anchors/tournament-20260917/`): aim-only vs aim head-to-head
1107-1053-480 = 51.0%, a tie; three-seed means +47 for aim-only with s.e.
~31 on the difference. Under the rule (means + head-to-head, `v4-aim` wins
ties) the production shape is `v4-aim`: rays + aim/LOS, writer bit 8. Two
more tournament facts: the joint ladder's top 20 are all v4 checkpoints in
736-789 with CI widths down from ~130 to ~31, and `v4-lite`, the smallest
net, holds #1 and #4, so capacity is not the lever (the opponent
distribution is, 7c'); and tournament-only ratings fall outside the prior
pool-v7-anchored intervals for 71 of 76 members, because the v4s beat the
v7 members ~90% but are even among themselves and one Bradley-Terry scale
cannot hold both: pool v7's saturation, the reason for pool v8. Shape ablations stop here: `25 v4-no-relational`
and `27 v4-critic-shared` are single-seed input / head ablations that
cannot resolve below ~75; proposed to park them with `sym-critic` /
`rays16` / `maxpool`; parked by the parent at 22:30 together with `28
seeds-rest` (the v3 seed floor is no longer the question). Queue at 22:30:
`19c v4-aim-only-s75` (running) -> `19d v4-aim-maps` -> `19e
v4-no-rays-maps` -> `19f v4-aim-maps-r2` -> `19g v4-aim-maps-s74` -> `19h
v4-no-rays-maps-s74` (second seeds of 19d/19e, seed line only; 19f waits
for its first result, it is the newest sim build). Every run from here is
two seeds from the start. Rays cost no
sps (`v4-no-rays`, bit 2, runs the same 163k), the evidence on them is flat
not negative, and the more-maps run below is the test where the egocentric
geometry channel is the point. Queue (gpu-queue/, lexical order): `13zz
v4-aim-only` -> `19a v4-aim-s74`, `19b v4-aim-s75`, `19c v4-no-rays-s74`
(seed replicates of the production shape and its no-rays control) -> `25
v4-no-relational`, `27 v4-critic-shared` (renumbered from 15/17 so the seeds
sort first) -> `28 seeds-rest` (v3, was 20). Parked, not deleted:
`sym-critic`, `rays16`, `maxpool` (single-seed ablations can only resolve
effects above ~30 points). The more-maps runs (7d) went in behind 19c once
their gates passed: `19d v4-aim-maps`, `19e v4-no-rays-maps` (claim 3 is the
point of keeping rays, so its control gets the slot).

### 7c'. Pool v8 (CPU-only, after the tournament reports)

Pool v7 saturates: v4 candidates win ~0.9 of their games against its bots,
so 217-game evals give +-25-point bands and the same-shape seed spread
(~50-75) dominates every comparison. Pool v8 puts the candidates
mid-scale: members = the six finished v4 checkpoints and the v4 seed
replicates (the tournament's cross-family games, played on the per-player
ablation build), bridges = the pool v7 bots and the v3b anchors so the
scale stays pinned to random = 0 the same way. Written as
`sweeps/anchors/pool-v8/pool.json` in this rift; every finished v4 run's
last three checkpoints are rated on it offline, a v8 column next to v7 for
the whole wave. Pool v7 stays the pre-registered primary for 19d-19h and
the in-run hook is not changed mid-wave; v8 is reported alongside and
becomes primary for the next wave (opponent distribution: stronger and more
varied members, a self-play fraction; two seeds each; design after v8
exists, nothing on the GPU until 19h is queued and the parent says go).

Built 2026-09-18 03:35 (`reference_pool.py build-from-games`, commit
03a8a1f): 34 members, pool sha `151fedb0`. The three pool-v7 band
checkpoints (2133M/2240M/2313M) of `v4-full`, `v4-no-rays`, `v4-aim`,
`v4-aim-only`, `v4-full-s74` and of `v4-lite` (its 2313M was not in the
tournament roster; 1920M stands in), plus random, the seven pool-v7 bots
and the eight v3b anchors; 13,464 of the tournament's 68,400 games (24 per
pair, side-swapped fixed seeds on maps 0-2) fitted by the same
Bradley-Terry with random = 0. Scale: v4 members 567-665, search bots
522-528, v3b finals 429-520, the other bots 244-341. Every finished v4
run's last three checkpoints rated on it 03:36-05:40 (240 games each, the
worktree's per-player evaluator, sim sha differs from the tournament's
build so `game_mismatch` is set; maps 0-2 are bit-identical across the two
builds):

| run | v7 bands | v7 IQM | v8 bands | v8 IQM | v8-v7 |
|---|---|---|---|---|---|
| v4-aim-only | 772/763/732 | 756 [722,792] | 611/620/616 | 616 [581,653] | -140 |
| v4-aim-only-s75 | 726/715/696 | 712 [681,747] | 590/596/607 | 598 [563,635] | -115 |
| v4-full | 680/707/712 | 699 [669,734] | 636/630/598 | 621 [591,656] | -78 |
| v4-no-rays | 697/694/689 | 693 [661,729] | 584/581/644 | 603 [568,640] | -90 |
| v4-aim | 675/748/690 | 704 [671,740] | 588/613/638 | 613 [576,653] | -92 |
| v4-aim-only-s74 | 685/667/695 | 682 [650,716] | 549/582/601 | 577 [542,613] | -105 |
| v4-lite | 681/680/656 | 672 [640,706] | 607/612/632 | 617 [581,655] | -55 |
| v4-full-s74 | 674/674/687 | 678 [647,712] | 601/576/617 | 598 [562,635] | -80 |
| v4-aim-s74 | 669/630/648 | 649 [617,684] | 585/570/573 | 576 [538,613] | -73 |

Reading: the nine runs span 106 points on v7 and 45 on v8 (576-621, all
inside one another's intervals; Spearman 0.33 between the columns). The
same-shape seed pairs are 37 / 23 / 38 / 18 points apart on v8 (55 / 21 /
73 / 43 on v7); aim-only's three-seed mean is 597 against aim's two-seed
594: the +40 v7 gap is +2 on v8, the tournament's tie again. So v8 does
not rank the v4 shapes either; it confirms they are one cluster, and it
pins the scale the next wave must move on: a real improvement shows as a
v8 gain beyond ~40 (the widest same-shape pair) with the v7 IQM flat. The
240-game bands give +-37; the next wave's v8 bands go to 480 games. Same
principle as v7's replacement of v5: the pool is rebuilt when the
candidates saturate it, and rebuilt from games already played when a
tournament has them.

### 7d. More maps: the 31-level catalogue (`native/data/map-set.json`)

The five-map runs test map invariance on two held-out levels that share the
training levels' scale. The game has 124 level scenes and full dumps of all
of them (`native/reference-export/map-{0..123}.json`): 64 grass (levelType
0, the constants the sim was validated on), 38 snow, 22 space. Of the grass
levels, 57 use only components the Rust backend already ports (static
`StickyRoundedRectangle` platforms, animate-in, `AntiLockPlatform` orbits);
the other 7 carry `GrowOnStart` / `VectorFieldPlatform` and are skipped,
as are snow and space (they switch gameplay constants: sim work, not data).

Catalogue (2026-09-17, `extract_rollout_data.py LEVELS`): RL map index i is
Unity level `LEVELS[i]`; 0-4 are the original five, 5-30 are 26 new grass
levels chosen per platform-count stratum (4-6, 7-8, 9-10, 11-15), alternately
over each stratum's width span so the set covers the spread. The cap is 31
because `training_map_mask` and the per-map counters are 31 wide
(`BOPL_COUNTER_MAPS`); widening them touches the trainer the live queue
loads, so the second half of the 57 waits for a later round. Round 2, before
any further level is appended: widen `training_map_mask` (u32 -> u64, the
`--training-maps` parser and the `bopl_world_ready` / `bopl_draw_config`
loops), `BOPL_COUNTER_MAPS` / `map_fraction[31]` / `map_keys[31]` in
binding.c and the counters' Python readers (native_metrics.py,
metrics_store.py), and `evaluation_protocol.map_ids`'s 0-30 bound; then the
remaining 26 portable grass levels join the catalogue. That build goes to a
separately staged library path that new configs point at, not over the live
queue's `.so` (the 2026-09-17 swap was accepted on identical 0-4 digests,
not as precedent). Append only:
RL indices are stable once a run has trained on them. The table with scene
paths and roles is `native/data/map-set.json` (`map_set.py` for Python and
shells); `rollout.json` for maps 0-4 is byte-identical to before.

Split (fixed in the config, never in `training_map_mask`): training = 0-2
plus 20 new (RL 5, 6, 7, 9, 11-16, 18, 20-22, 24-28, 30); held out = 3-4
(continuity with the 12/10-point gaps of the five-map runs) plus six new,
one or two per stratum at the stratum's median width so held-out is not a
distribution shift: RL 8, 10, 17, 19, 23, 29 = levels 10, 14, 49, 68, 73, 90.

Gates, all passed before the job file exists (`v4-aim-maps.toml`):
- per-map smoke (`check_map_catalogue.py`, 8 seeds per map): spawns dry, no
  platform moving in the first two idle seconds beyond the authored orbits,
  random-input and bot-vs-bot episodes to the terminal with finite v3/v4 rows
  and ray distances in [0, 1]: 31/31 ok; the 0-4 row digests equal the
  previous build's, and `repr_ab` old vs new .so on 0-4 is identical at
  tolerance 0 (2597/2597 ticks). The pre-existing `check_maps.py` (reset
  idempotence, capacity flags, tick accounting) passes on 5-30 too.
- Unity parity (`repr_ab` against the .NET oracle `BoplSimNative.so`, the
  game code compiled headless, per-backend map pins: the oracle takes the
  level, Rust the RL index; 12 episodes per level, tolerance 2^-19): every one
  of the 26 new levels agrees on the winner 12/12 and on the episode length
  311/312 (one 709/710 case from a 2.4e-6 velocity rounding). Ticks within
  tolerance 75-100% per level against 93.6-99.4% on maps 0-4; the excess on
  levels 6, 13, 19, 90, 93 is the known classes (the constant basis rounding
  of the water box's `right_x`, the arrow's spawn-row rotation 3pi/2 vs 0 one
  tick before flight, terminal-tick death positions), no new class. Level 80
  (four orbit platforms already moving, seed-dependently, when the warmup
  ends: 16 fields differ after reset, 13/20 lengths) failed and was replaced
  by level 69.
- zero-shot baseline: `v4-aim` policy-986677248 (trained on 0-2) evaluated on
  the new split by `evaluate.py`: 81.0% on 0-2 vs 74.0% on the eight held-out
  maps (71/96) and 75.7% over all 17 unseen maps (222 games).

Metric: pool v7 stays primary, scored exactly as every other v4 run (its
members trained on 0-2; scoring on new maps would re-scale the anchor and
penalise v3 for our distribution, not theirs). Second band: scripted-bot win
rate on the held-out maps, reported as the trained/held-out gap with CI
(readout.py section 3) and as `flags.py heldout_win_rate` / `heldout_gap`
with the config flag `heldout-gap = "heldout_gap > 8"`. In-run evaluation
keeps its budget: 12 training maps x 2 games (was 3 x 8) and 8 held-out maps
x 2 games (was 2 x 4).

Pre-registered claims (success criteria, written before the run):
1. `v4-aim-maps` holds pool v7 within seed noise of `v4-aim`'s 697 (IQM3 at
   or above 697 - 50 = 647): training on 23 maps costs nothing on the
   anchor. Threshold history: pre-registered as 697 - 27 = 670 on the
   `v4-full` 703 / `v4-full-s74` 676 pair; widened to 647 on 2026-09-17
   18:25, before `19d` started, on the basis of the two independent
   same-shape seed pairs (aim 697/648, full 703/676) and on no more-maps
   result. The readout reports against both thresholds (670 and 647).
2. The held-out gap halves against the five-map runs: `v4-aim-maps`'s
   trained-minus-held-out scripted-bot gap on the eight held-out maps is at
   most half of `v4-aim`'s 10.4 points (and of `v4-full`'s 16.1) at 1.73B.
3. `v4-no-rays-maps` (bits 2|8: fan and terrain block off, aim ray and line
   of sight kept, so only the geometry channel differs; same split, the
   follow-up job) loses more than `v4-aim-maps` on more maps: its pool v7 IQM
   and its held-out win rate are both below `v4-aim-maps`'s by more than the
   v4 seed spread, the test of the geometry channel that five maps could not
   give.
A result inside the seed spread on (1) with (2) met is the intended outcome;
(2) missed at equal pool v7 says the gap is about the opponents, not the
geometry; (3) unresolved keeps rays on by default.

Human tests on the Mac (`Tools/play-vs-agent-mac.sh --maps training |
held-out | <rl list>`, translated to the player's `BOPL_LEVEL_SET` levels by
`map_set.py`): training maps for a fair test, held-out for generalisation.
`games.jsonl` rows carry `rl_map` and `map_role`, and `reference_pool.py
human --map-role training|held-out|unported` rates them apart.

### 7e. Next wave: the opponent distribution (design 2026-09-18; go 05:55; queued 06:15 as 20a-20f; PARKED 14:40)

What the tournament says. The v4 checkpoints beat every pool-v7 member
about 90% of the time and are even among themselves (aim-only vs aim
1107-1053-480); the smallest net (`v4-lite`) holds #1 and #4 of the joint
ladder. So the learner has exhausted what its opponents teach: the training
distribution is three scripted banks whose bot share decays to 11% once the
roster is beaten, and one self-play bank drawn from the run's own snapshots
(`pool_size` 16, `sample_opponent` weights index^0.5 over all but the five
newest, no win-rate term, eviction of the oldest half). Late in a run
almost every game is against a slightly older copy of itself or a bot it
already beats; nothing in the mix plays the way the other v4 seeds play,
which is the only kind of opponent the tournament found it does not beat.
Capacity is not the lever (lite); the critic is at its ceiling (3b); the
gap to Matthew is aiming and awareness against a moving opponent (7b), an
opponent-distribution problem.

Runs (all `v4-aim`, the production shape, 024's recipe, 1.73B steps, two
seeds each, 73 and 74; scored on pool v8 primary (480-game band, in-run 160
games) with pool v7 alongside (in-run 96 games after the v8 rating in the
same nice-19 chain; the 480-game v7 band offline right after the run, so
every run has both columns), plus a fixed head-to-head band. Contamination
rule (parent, 05:55): the band holds no checkpoint the league trained
against. Band = the finals of `v4-aim-only` s73 / s74 / s75, `v4-full` and
`v4-lite` (cross-family, through the per-player writer bits), 16 games x 3
evaluation maps = 48 each, every evaluation
(`anchors/wave3-opponents/head-to-head.json`). The learner's win rate
against the pinned members is reported from the pool ledger
(metrics `pool.pinned`: games, wins, draws, draw share, game share) as a
training-curve diagnostic, not a claim. Pool v8 has 3 of its 34 members
among the league's training opponents (the v4-aim s73 / s74 2313M
checkpoints, and v4-aim s73 2133M / 2240M are near-copies); stated,
accepted: the band is the clean measure, v8 the common scale.

1. `v4-aim-league` (the main bet): a fourth kind of opponent, other runs.
   The self-play pool starts pinned with the finals of `v4-aim` s73/s74
   and `v4-aim-maps` (same net, same writer bit 8: they load into a bank as
   any snapshot does), never evicted, sampled by a prioritised rule
   (weight (1 - winrate)^2 per entry from the bank ledger, PFSP) instead of
   index^0.5, with the own-snapshot pool behind them at `pool_size` 32.
   Scripted banks 2, share min 0.11 as now. Trainer changes: pinned pool
   entries and the sampler (our `native_selfplay.step` wraps
   `pufferlib.selfplay.step`; the sampler is one function in the overlay),
   no C change. Cross-family members (aim-only, bits 2|8) would need the
   per-bank writer mask in the trainer's vec (env->ablate[player] exists
   since 2026-09-17; the bank-to-player routing does not carry it yet):
   second step, not this wave. Implemented (wave3 8b5083b):
   `native_selfplay.add_pinned` / `install_sampler` route
   pufferlib.selfplay's module-level `sample_opponent` / `evict` through the
   league rules (pinned members at the front of the pool, never evicted,
   `--pool-size` bounds the run's own snapshots; the draw is over pinned +
   own minus the five newest, weights `(1 - p)^2` with `p = (wins+1)/(games+2)`
   from the ledger, floor 0.01, so an unplayed member is drawn like an even
   one and a beaten one still comes back now and then); banks start on the
   bootstrap as before. Flags `--pinned-opponents name=path,...` (sidecar
   shape and writer bits checked against the run), `--opponent-sampler
   index|pfsp`; both recorded in config.selfplay and restored by --resume.
   Without either flag the upstream path runs untouched (unit tests:
   puffer/test_native_selfplay.py, 11 pass).
2. `v4-aim-pfsp`: the sampler change alone on the own-snapshot pool
   (`pool_size` 32, weight (1 - winrate)^2, no pinned externals). Separates
   "who" from "how sampled": if 1 beats 2, the externals carry the effect.
3. `v4-aim-search`: `search-medium` (pool v7 589, the strongest bot, absent
   from the training roster because `bopl_rl_fork` costs ~4 ms a decision)
   added to the scripted roster with its share of the scripted banks' draws
   capped at 0.25 (`--scripted-draw-cap search-medium=0.25`; every
   scripted bank draws from one roster, so the cap is on draws, not on a
   bank). Gated on a 2-minute sps check at full width
   (`sweep/sps_gate.py`, run by the job itself before the driver): below
   120k learner sps the job exits without training, the run is dropped
   from the wave, not slowed.

Pre-registered claims. (1) `v4-aim-league` beats the `v4-aim` two-seed mean
on pool v8 (594; 613 / 576) by more than 40, the widest same-shape seed
pair on v8 (7c'), and wins more than 55% of the head-to-head band; (2) `v4-aim-pfsp` does not: the externals, not the
sampler, are the effect; (3) the trained-minus-held-out scripted-bot gap
does not widen (no overfit to the pinned opponents). A league win with the
pool-v7 IQM unchanged is the expected shape of the result: v7 cannot see
it, which is why v8 exists. Cost: 6 runs x ~3 h = 18 h GPU, behind 19h;
`v4-aim-search` only if its sps gate passes.

Parked 2026-09-18 14:40 (Matthew, via the parent): no GPU hours on wave 3
yet. 20a-20f moved to gpu-queue/parked/ intact, the `policy-v4-wave3`
worktree and its build left as they are, so the wave can be unparked by
moving the six files back; nothing else is needed. Context: the Rust sim's
all-124-maps + 39-abilities port landed on main at 4d66b2f8 the same day,
and the next wave is likely multi-ability / all-maps on that foundation,
planned with Matthew first; the 7e trainer changes (pinned members, PFSP
draw, the sps gate, the secondary in-run pool) carry over to whatever
recipe that wave uses.

Queue as prepared (parent 05:55, both league seeds before anything else so
claim 1 resolves first): 20a `v4-aim-league` (s73), 20b `v4-aim-pfsp` (s73), 20c
`v4-aim-league-s74`, 20d `v4-aim-pfsp-s74`, 20e `v4-aim-search` (gated),
20f `v4-aim-search-s74` (gated); job files in gpu-queue/, same shape as 19g,
nice 0, max_seconds 21600, steps 1734936776, scoring pool v8. Where they
run: the policy-v4 rift cannot take the cross-family band (its evaluator
predates the per-player writer bits, and landing 6c19394 there touches
bopl_puffer.h / bopl_obs_v4.h, which the trainer's staleness check would
turn into a failed start for 19g / 19h), and policy-v4-work carries the
64-map trainer and keeps moving for round 2. So a third worktree,
`policy-v4-wave3` (branch `wave3` = policy-v4 a2a9b82 + 6c19394 + the 7e
trainer commit), with its own `_C` / research / policy_eval build and the
rift's venv, sim (`native/target` symlink, the same .so 19d-19h load) and
`sweeps/anchors` pools by symlink; frozen once the jobs are queued. The
pinned and band checkpoints are copies under
`anchors/wave3-opponents/` (README.json maps them to their runs).

Smoke (06:03, `sps_gate.py` on the league config, `--workers 32`, 150 s
then SIGTERM, beside 19e at 128 workers): 1109 updates, 6.8M learner
steps, 58k learner sps mid-run (47.6k median over the last 40 once the
evaluation and the rating chain took their cores); `min_games 16` and
`opponent_timeout 1M` to force swaps: 31 draws, 26 of them pinned (draw
share 0.84, game share 0.85), all four banks on pinned members at the end,
8 own snapshots behind them; the learner at 6.8M steps wins 8-11% against
the three finals. Evaluation ran the five-member band (97 games with
2 per opponent, 37 s), the rating chain wrote v8 then v7 records, train.py
stopped clean on SIGTERM. In a full run the own pool grows to 32 (total
weight 3-8 against the pinned ~2.4-3), so the pinned share should settle
near 0.3-0.5 rather than the smoke's 0.85; the metrics say.

## 8. Config and flags

`train.py --policy v4` (default `v3` so existing configs are unchanged) with
`--set-dim 128 --set-inner 128 --ffn-dim 512 --critic-dim 512
--critic-privileged 1 --critic-detach 0 --mean-pool 1 --lr-width-exponent 0`.
`--hidden 256 --layers 1` as before. All are recorded in the checkpoint
sidecar config and honoured by the evaluator.
