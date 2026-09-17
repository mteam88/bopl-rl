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
   checkpoint's `v4_shape` so the evaluator writes the same rows).
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

### 7c. Queue after the aim runs (2026-09-17)

Production shape = `v4-aim` (full + aim/LOS, rays on) unless `v4-aim-only`
clearly beats it (IQM above `v4-aim`'s 697 + ~30, outside the v4 seed
spread of ~27 measured by `v4-full` 703 / `v4-full-s74` 676). Rays cost no
sps (`v4-no-rays`, bit 2, runs the same 163k), the evidence on them is flat
not negative, and the more-maps run below is the test where the egocentric
geometry channel is the point. Queue (gpu-queue/, lexical order): `13zz
v4-aim-only` -> `19a v4-aim-s74`, `19b v4-aim-s75`, `19c v4-no-rays-s74`
(seed replicates of the production shape and its no-rays control) -> `25
v4-no-relational`, `27 v4-critic-shared` (renumbered from 15/17 so the seeds
sort first) -> `28 seeds-rest` (v3, was 20). Parked, not deleted:
`sym-critic`, `rays16`, `maxpool` (single-seed ablations can only resolve
effects above ~30 points). The more-maps run (7d) goes behind 19c once its
gates pass.

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
loads, so the second half of the 57 waits for a later round. Append only:
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
   or above 697 - 27): training on 23 maps costs nothing on the anchor.
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

## 8. Config and flags

`train.py --policy v4` (default `v3` so existing configs are unchanged) with
`--set-dim 128 --set-inner 128 --ffn-dim 512 --critic-dim 512
--critic-privileged 1 --critic-detach 0 --mean-pool 1 --lr-width-exponent 0`.
`--hidden 256 --layers 1` as before. All are recorded in the checkpoint
sidecar config and honoured by the evaluator.
