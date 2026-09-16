# Representation probes for the v3 policy, and a raycast feature study

Branch `probes`, code in `native/rl/research/probes/` (README there documents
the dataset schema and pipeline), data and plots in
`native/artifacts/rl/probes/dataset/` (gitignored; `results/` holds every
number quoted here as JSON). Checkpoint under study: overnight-9 trial 024
final, `policy-2313256960.bin` (the brief's "1734942720" is its learner-step
count; 634 Elo on pool v7). Contrast checkpoint: the same trial at 320M
steps, `policy-320012288.bin` (409 Elo). Trial 000 has no comparable final
checkpoint.

## 1. Conclusion (one page)

**What the input lacks.** Three things are not decodable from the raw
observation even by an MLP on 1.4M rows, or only weakly:

1. *Egocentric terrain geometry.* Height above the surface below (R² 0.56),
   signed distance to the platform edge (0.58), signed support offset (0.66)
   and the nearest terrain distance along 16 rays (0.81) are all poorly
   recoverable from the raw observation, and are the labels that fall
   furthest through the encoder (edge distance 0.58 → 0.26 fused → 0.24 GRU).
   Terrain is present only as static entity rows (relative position, extents,
   radius) and has to be reassembled from up to 32 rows; the fall labels ride
   on it (`fall_30` AP 0.16 from raw).
2. *The present.* The 9-tick observation delay is the single largest
   information gap. Everything kinematic is much better decodable from the
   undelayed encode of the same state: own velocity R² 0.72 → 1.00, opponent
   velocity 0.49 → 1.00, opponent cooldown fraction 0.85 → 1.00, and the
   projectile-hit labels' average precision roughly doubles (`hit_15` AP
   0.10 → 0.22, `hit_inflight_30` 0.15 → 0.25). A hostile arrow moves ~4
   world units in 9 ticks (median 27 u/s) and the median arrow is 21 u away.
   The recurrent state compensates partially (GRU out beats the raw-obs probe
   on every `hit_*` label: `hit_15` AP 0.15 vs 0.10) but does not close the gap.
3. *Who the opponent is.* The round outcome is mostly irreducible at decision
   time: the best probe reaches EV 0.13 of the discounted return from the raw
   observation and 0.15 with the privileged labels added; adding the (never
   observable) opponent identity one-hot lifts it to 0.21–0.23. The critic's
   own EV is 0.11 (section 4).

**What the encoder destroys.** Less than the hypothesis predicted for
projectiles, more for geometry and opponent state:

- The per-row hypothesis is confirmed at the row level: a *linear* function of
  one hostile arrow's 34 features cannot tell whether that arrow kills the
  observer (AUC 0.64, AP 0.04 at 3.5% base rate; with relational features
  AUC 0.83 / AP 0.12; MLP on the row 0.83 / 0.13, MLP with relational 0.89 /
  0.21). But at the state level the pooled 64-channel entity vector, the fused
  128 and the GRU state carry the hit labels as well as or better than an MLP
  on the raw observation (`hit_inflight_30` AUC 0.967 raw MLP vs 0.977 pooled
  vs 0.977 GRU out; AP 0.145 vs 0.196 vs 0.206). The 64 max-pooled channels
  act as a bank of thresholds over (dx, dy, vx, vy) and the fusion layer
  reconstructs "an arrow is on a collision line" from them; the encoder is not
  the bottleneck for dodging as far as *detection* goes.
- What it does lose: opponent cooldowns (`foe_dash_cd` R² 0.87 raw → 0.72
  fused → 0.65 GRU; `foe_platform_ready` AUC 0.997 → 0.958 → 0.945),
  opponent velocity (0.49 → 0.35 → 0.32), the opponent's idle drift over 30
  ticks (`idle_opp_dx30` 0.64 → 0.32 → 0.31), own velocity (0.72 → 0.63 →
  0.64), and terrain geometry as above. Map identity is 0.999 from the raw
  observation (the static entity rows are the map) but only 0.74 from the
  fused vector and 0.70 from the GRU state, while absolute position survives
  at R² 0.96/0.94: the representation keeps *where I am*, not *which map*.

**Do relational features or rays recover it?** (section 5, table 5.1)
%FAMILIES%

**Absolute position and map memorisation** (section 6). The policy is
sensitive to obs[8:10]: shifting the absolute position by ±0.1 (6.4 world
units) with everything else fixed changes the action distribution by 0.18–0.20
nats mean KL and the value by 0.017; ±0.3 gives 1.3–2.0 nats. The same shift of
a never-active input gives 0.0004 nats and of the episode clock 0.0009, so the
sensitivity is real, but the opponent's relative position at the same shift
gives 1.0 nats — the policy leans on relative geometry five times harder than
on absolute position. Sensitivity is the same on held-out maps 3–4 (0.22–0.27
nats at ±0.1), so it is not a per-map lookup that fails off-distribution; it is
a smooth dependence that generalises as well or badly as anything else. Zeroing
obs[8:10] costs the raw-observation probes nothing (every label within 0.002),
because position is redundantly encoded in the terrain rows.
%NOPOS%

**Critic.** Explained variance of the 024 value head against the realised
γ=0.995 return is 0.11 (corr 0.33; value std 0.16 vs return std 0.54), against
the round outcome sign 0.06, against 128-decision bootstrapped returns 0.12,
against its TD(λ=0.9) training target 0.57 (which is what train.py's 0.63
measures) and against r + γV(s') 0.88. The critic is self-consistent but not
predictive — and it is not under-capacity: the ceiling probes above sit at
0.13–0.15 on the same input. The critic is not the lever.

**Recommendation for policy-v4.**
%RECOMMENDATION%

## 2. Method

**Dataset.** 2,520 games in the Rust sim (`BOPL_RL_BACKEND=rust`, forkable),
1,774,112 rows (one per player per decision; median 181 decisions per player
per game). Schedule (`generate.schedule`): 024-final vs each pool-v7
checkpoint (96 games each), vs archer/pursuer/evasive/mixed-hard and
mixed-medium (96 each), search-medium (48), random (24), itself (192) and its
320M ancestor (192); the 320M checkpoint vs pool and scripted (24 each); maps
0–2 round-robin, side-swapped seed pairs, seeds 500000+. A held-out supplement
of 504 games on maps 3–4 (never trained on) at a quarter of the volume. Rows
by controller: 51% 024-final, 9% 320M, 31% pool checkpoints, 9% scripted bots.
Train/test split 80/20 by game (358,632 test rows).

Per row: the delayed v3 observation the controller saw (3,690 floats, stored
compact as 1,458 — rows beyond 2 players / 8 slots / 32 entities / 4 slimes
are always empty in these games — as f16), the undelayed encode of the same
state, the action, and 64 labels from the full state:

- *Idle fork* (`probe_idle_future`): `bopl_rl_fork` the live world, hold no
  input for both players for 60 ticks, record death tick and cause per player
  (environment/fall, squish, opponent's projectile, own projectile), and for
  projectile deaths the killing arrow, matched back to its entity row in the
  observation (`killer_row`). Labels `hit_15/30/60`, `hit_inflight_*` (the
  killer already visible), `fall_30/60`, `squish_60`, `own_arrow_in_flight`,
  `my_shot_hits_60`, positions at t+10 and t+30 for both players.
- *Geometry* (`probe_geometry`, from the collider set): grounded, height
  above the surface below (`drop`), signed distance to the supporting
  platform's nearest edge (`support`), `edge_distance`, whether the idle fall
  lands dry, terrain within one dash (`terrain_within_dash`), nearest terrain
  along 16 rays, own/opponent velocity, opponent grounded, opponent dash/bow/
  platform readiness and cooldown fractions.
- *Realised future* along the rollout: positions at t+12/t+32, death within 60,
  win/loss/timeout, ticks to end, the per-decision env reward (outcome ±1,
  kill/death ±0.288) and the γ=0.995 discounted return, and the controller's
  own value at that step.

Base rates: `hit_15` 0.37%, `hit_30` 0.71%, `hit_60` 1.1%, `hit_inflight_30`
0.48%, `fall_30` 0.17%, `fall_60` 0.53%, `squish_60` 0.6%, `any_death_60`
2.5%, `own_arrow_in_flight` 12.9%; a hostile projectile is visible in 11.2%
of states. 65% of `hit_60` deaths have the killer already in the observation.

**Encoder activations** (`activations.py`): the exported weights are re-read
with puffernet's 8-float alignment and the CPU reference forward
(`bopl_set_encoder.h`: bias-free 64-channel projection per row, +embedding by
ability id, ReLU-max over valid rows with memset-zero semantics for empty sets;
`Linear(98,128)` flat; fusion `Linear(384,128)`; MinGRU 128×1; decoder) is
re-implemented in torch and scanned over every (game, player) sequence. It
matches the native policy to 1e-5 in state and 2e-4 in value (the f16 storage
of observations and activations bounds the rest).

**Probes** (`probe_fit.py`). For each feature set, one multi-output linear
layer (each output an independent linear probe) and a grouped MLP with a
separate 256-unit ReLU hidden layer per label, trained jointly with
BCE/MSE on 700k train rows for 6 epochs (Adam, cosine schedule), features
standardised on train statistics, regression targets standardised. Reported on
the test games: AUC, average precision and accuracy vs majority for binary
labels, R² for regression, argmax accuracy for map identity. Linear probes
follow Alain & Bengio (2016); the linear-vs-MLP pair follows Hewitt & Liang
(2019)'s point that probe capacity must be reported with the result, and
Belinkov (2022) on reading probes as "is it linearly/nonlinearly present", not
"is it used". Feature sets: raw delayed observation (1,458), undelayed
(`raw_now`), raw with obs[8:10] zeroed (`raw_nopos`), the flat block after
ReLU (128), each pooled set vector (64), all four (`pooled`, 256), fused 128,
GRU state, GRU output, and the candidate families below appended to raw.

**Candidate features.** (a) *Relational per-row* (`features.py`, 299 floats):
for each entity row and for the opponent, distance and 1/distance, bearing
cos/sin, relative velocity (entity minus own slime velocity), radial closing
speed and tangential speed, linear time to closest approach (clipped to 60
ticks) and the miss distance at that time; plus own velocity — the pairwise
quantities relation networks compute inside the model (Santoro et al. 2017;
Zambaldi et al. 2019), here supplied as inputs. (b) *Rays* (`probe_rays` in
`probe_binding.c`): from the player's centre, N uniform rays plus 4 spread
around straight down, range 40 u; per ray the distance to the first hit over
range (1 if none), a 6-way one-hot of the hit (terrain/wall, water or kill
zone, hostile projectile, opponent body, own slime/object, nothing) and the
hit's velocity along the ray — the Unity ML-Agents `RayPerceptionSensor`
layout (one-hot tag + miss flag + normalised distance per ray), against the
same colliders `bopl_surface_distance` uses and the entity circles, with the
first 1 or 3 distinct hits per ray. Configurations: 16+4, 32+4, 32+4 × 3 hits.

## 3. Input vs encoder

Full tables in the appendix (A.1–A.2); heat maps
[`heat-encoder-binary-mlp.png`](../../artifacts/rl/probes/dataset/results/heat-encoder-binary-mlp.png),
[`heat-encoder-regression-mlp.png`](../../artifacts/rl/probes/dataset/results/heat-encoder-regression-mlp.png).
MLP probe, test set; AUC for binary, R² for regression; `raw_now` is the
undelayed observation (what the input *could* contain without the delay).

| label | raw | raw_now | players | slots | entities | slimes | fused | GRU out | note |
|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.945 | 0.983 | 0.856 | 0.892 | 0.897 | 0.823 | 0.952 | 0.969 | AP 0.10 / 0.22 / … / 0.12 / 0.15 |
| hit_inflight_30 | 0.967 | 0.985 | 0.817 | 0.941 | 0.953 | 0.826 | 0.971 | 0.977 | AP 0.15 / 0.25 / … / 0.16 / 0.21 |
| my_shot_hits_60 | 0.967 | 0.978 | 0.797 | 0.941 | 0.944 | 0.845 | 0.961 | 0.963 | |
| fall_30 | 0.954 | 0.962 | 0.722 | 0.753 | 0.833 | 0.925 | 0.964 | 0.958 | AP 0.16 / 0.30 / … / 0.10 / 0.11 |
| squish_60 | 0.934 | 0.943 | 0.894 | 0.895 | 0.912 | 0.885 | 0.919 | 0.926 | |
| lands_dry | 0.928 | 0.967 | 0.673 | 0.751 | 0.838 | 0.836 | 0.901 | 0.900 | |
| terrain_within_dash | 0.975 | 0.992 | 0.758 | 0.916 | 0.900 | 0.882 | 0.950 | 0.949 | |
| grounded | 0.980 | 1.000 | 0.719 | 0.815 | 0.823 | 0.925 | 0.962 | 0.958 | |
| foe_platform_ready | 0.997 | 1.000 | 0.721 | 0.997 | 0.801 | 0.734 | 0.958 | 0.939 | in the slot rows, lost in fusion |
| map (acc, maj 0.35) | 0.999 | 0.999 | 0.411 | 0.462 | 0.990 | 0.430 | 0.737 | 0.688 | |
| drop (R²) | 0.562 | 0.604 | 0.082 | 0.176 | 0.296 | 0.369 | 0.508 | 0.489 | height above surface |
| edge_distance | 0.579 | 0.660 | 0.030 | 0.063 | 0.235 | 0.204 | 0.261 | 0.243 | |
| support | 0.656 | 0.779 | 0.064 | 0.127 | 0.166 | 0.422 | 0.504 | 0.475 | |
| terrain_min_ray | 0.813 | 0.885 | 0.124 | 0.408 | 0.367 | 0.438 | 0.647 | 0.630 | |
| vx (own) | 0.720 | 0.997 | 0.049 | 0.173 | 0.149 | 0.450 | 0.628 | 0.625 | delay |
| foe_vx | 0.486 | 0.996 | 0.049 | 0.167 | 0.142 | 0.448 | 0.354 | 0.322 | delay, then fusion |
| foe_dash_cd | 0.869 | 0.998 | 0.063 | 0.860 | 0.183 | 0.203 | 0.715 | 0.643 | |
| idle_dx30 | 0.806 | 0.916 | 0.108 | 0.403 | 0.420 | 0.244 | 0.468 | 0.464 | own drift, 30 ticks |
| idle_opp_dx30 | 0.642 | 0.757 | 0.114 | 0.092 | 0.395 | 0.116 | 0.317 | 0.294 | opponent drift |
| real_opp_dx12 | 0.281 | 0.401 | 0.044 | 0.171 | 0.143 | 0.196 | 0.247 | 0.249 | opponent's actual move |
| return_disc | 0.113 | 0.119 | 0.017 | 0.067 | 0.031 | 0.024 | 0.108 | 0.130 | |
| abs_x | 0.986 | 0.988 | 0.436 | 0.473 | 0.977 | 0.509 | 0.958 | 0.932 | |

Reading: columns `players`…`slimes` are each pooled set vector alone; the
slot rows carry the opponent's cooldowns perfectly (0.997) and the fused
vector drops them (0.958 → GRU 0.939); the entity rows alone carry the map
(0.99) and absolute position (0.98), the fused vector keeps position and drops
the map. The linear-probe versions of the same tables (appendix) show the
encoder *linearises* the hit labels: linear AUC on `hit_inflight_30` is 0.903
from raw and 0.929 from GRU out.

## 4. Critic

`results/critic.json`,
[`critic-024-final.png`](../../artifacts/rl/probes/dataset/results/critic-024-final.png).
906,075 decisions by 024-final; γ = 0.995, GAE λ = 0.9, horizon 128 (from the
checkpoint's train block; `bopl.ini` agrees).

| target | EV | corr | target std |
|---|---|---|---|
| realised discounted return | 0.110 | 0.333 | 0.538 |
| 128-decision return, bootstrapped with own V | 0.122 | 0.349 | 0.492 |
| 512-decision return, bootstrapped | 0.110 | 0.333 | 0.538 |
| TD(λ = 0.9) return (the GAE target) | 0.567 | 0.754 | 0.207 |
| r + γ V(s') | 0.877 | 0.937 | 0.172 |
| round outcome sign | 0.062 | 0.272 | 1.000 |

Value std 0.162, return std 0.538; sign accuracy 0.60 (majority 0.51).
Calibration by value decile is good and slightly under-confident at the
extremes (decile means: value −0.204 realises −0.222, 0.007 → 0.028, 0.222 →
0.242, 0.368 → 0.426): the critic is honest about knowing little. By time to
the end of the round the EV is 0.22 within the last 60 ticks (corr 0.55,
return std 1.12), 0.17 at 60–120, 0.13 at 120–240, 0.09 at 240–480, 0.05 at
480–960 ticks: even one second before a kill or death the critic explains a
fifth of the outcome. The 320M checkpoint: 0.013 / 0.017 / 0.359 / 0.788 /
−0.002 on the same targets.

Ceiling probes (`critic-ceiling.json`; MLP, 024-final rows, test EV):

| input | return EV | outcome-sign EV | win acc (maj 0.52) |
|---|---|---|---|
| raw delayed obs | 0.129 | 0.111 | 0.625 |
| undelayed obs | 0.132 | 0.114 | 0.626 |
| raw + privileged labels | 0.149 | 0.116 | 0.626 |
| GRU state | 0.134 | — | 0.613 |
| raw + GRU state | 0.146 | — | 0.627 |
| opponent identity one-hot alone | 0.159 | — | 0.693 |
| raw + opponent identity | 0.214 | ~0.20 | 0.653 |
| raw + privileged + opponent identity | 0.230 | — | 0.651 |

## 5. Feature families

%FAMILY_TABLE%

CPU cost (`bench.py`, single thread, unoptimised C, per player observation,
mean over 36 mid-game states on maps 0–2; the env step is one decision of the
Rust sim including hold ticks and the encode, timed from Python):

| configuration | µs / observation |
|---|---|
| 8 rays | 0.8 |
| 8 + 4 down | 1.6 |
| 16 rays | 1.7 |
| 16 + 4 down | 2.4 |
| 32 rays | 3.3 |
| 32 + 4 down | 3.5 |
| 64 + 4 down | 7.0 |
| 16 + 4, 2 hits | 2.2 |
| 32 + 4, 3 hits | 3.9 |
| v3 encode, both players (`probe_pack`) | 2.4 |
| env step (one decision, both players) | 89 (median 60) |

Rays cost ~0.1 µs each; 16+4 rays for both players (4.8 µs) doubles the
observation-writer cost and adds ~5% to a decision step; extra hits per ray
are nearly free (the shapes along a ray are already sorted).

**Literature.** Unity ML-Agents' `RayPerceptionSensor(Component2D/3D)`
(Juliani et al. 2020; docs "Learning-Environment-Design-Agents") emits, per
ray, a one-hot over detectable tags, a miss flag and the normalised hit
distance, with `RaysPerDirection`, `MaxRayDegrees`, optional sphere-cast
radius, and stacked observations; the docs recommend "as few rays and tags as
necessary" for learning stability, and their WallJump example stacks two ray
sensors at different heights so the agent can tell whether the wall is
jumpable — the analogue of our extra downward rays. Alonso et al. (IJCAI 2021,
Ubisoft, "Deep RL for navigation in AAA video games") use 2D depth maps from
ray casts plus a 3D occupancy box-cast and find in ablation that removing the
ray casts significantly slows learning while *removing absolute position does
not hurt and slightly helps* (p < 0.05 on their big map) — directly relevant
to the memorisation question. Sriram (Northeastern MS thesis, 2019, "Automated
playtesting of platformer games using RL", ML-Agents) drives a 2D platformer agent with
radial fixed-length rays tagging platforms, enemies, spikes, lava and coins,
and transfers to an unseen level. DeMaio & Guida (2023, project report on GitHub, not peer reviewed) compare camera vs ray
sensors for obstacle avoidance and find ray agents generalise better to more
complex environments with smaller models. Common thread: rays give an
egocentric, map-invariant local geometry that generalises across levels,
which is exactly the channel v3 lacks.

## 6. Absolute position and map identity

`results/perturb-*.json`, [`perturb.png`](../../artifacts/rl/probes/dataset/results/perturb.png).
Native CPU policy replayed along 160 recorded 024-final sequences (~30k
decisions) per map set, its recurrent state reproduced; at every step the
input is shifted by δ with everything else fixed; KL(base‖shifted) summed over
the aim/buttons/hold heads (unmasked), |ΔV|.

%PERTURB%

Map identity is decodable from the raw observation at 0.999 (and 0.999 with
obs[8:10] zeroed — the static entity rows *are* the map), from the pooled
entity vector at 0.99, but only 0.74 from the fused 128 and 0.70 from the GRU
state (majority 0.35). Absolute position itself survives to the GRU state at
R² 0.93. So (a) the representation is not map-keyed in the sense of holding a
map id; it holds a position, and the perturbation shows the policy reads it.
%NOPOS_ENCODER%

## 7. Corrections to the brief

- The 024 final checkpoint file is `policy-2313256960.bin`; trial 000 has no
  final checkpoint comparable to it, so the contrast is 024's own 320M step.
- The brief's "entity rows carry ABSOLUTE velocity, self velocity not in the
  row" is right; own velocity does exist in the observation, in the own slime
  row (`slimes[..., 9:11]`), and the self block does not carry it.
- The brief's "must be reconstructed after pooling" is right and the
  reconstruction *works* for detection (section 3); the losses are elsewhere
  (delay, geometry, opponent state).
- PufferLib's logged explained variance (0.63) is against the TD(λ) target,
  not the return; see section 4.

## Appendix: full tables

%APPENDIX%

## References

- Alain, G. & Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes. arXiv:1610.01644.
- Hewitt, J. & Liang, P. (2019). Designing and interpreting probes with control tasks. EMNLP.
- Belinkov, Y. (2022). Probing classifiers: promises, shortcomings, and advances. Computational Linguistics 48(1).
- Santoro, A. et al. (2017). A simple neural network module for relational reasoning. NeurIPS.
- Zambaldi, V. et al. (2019). Deep reinforcement learning with relational inductive biases. ICLR.
- Juliani, A. et al. (2020). Unity: a general platform for intelligent agents. arXiv:1809.02627; ML-Agents docs, "Agents: raycast observations" and `RayPerceptionSensor` API.
- Alonso, E., Peter, M., Goumard, D. & Romoff, J. (2021). Deep reinforcement learning for navigation in AAA video games. IJCAI. doi:10.24963/ijcai.2021/294.
- Sriram, V. (2019). Automated playtesting of platformer games using reinforcement learning. MS thesis, Northeastern University.
- DeMaio, L. & Guida, C. (2023). Adaptability of reinforcement learning agents from simple to complex environments (ray vs camera sensors). Pace University, project report.
