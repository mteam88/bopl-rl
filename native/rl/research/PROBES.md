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
Rays, decisively; relational features marginally and only on top of raw.
Appending 32+4 rays (288 floats) to the raw observation lifts the projectile
labels' average precision 2–3× (`hit_15` 0.10 → 0.28, `hit_inflight_30` 0.15
→ 0.35, `hit_60` 0.16 → 0.31), the fall labels 4× (`fall_30` 0.16 → 0.71,
`fall_60` 0.19 → 0.64), `any_death_60` 0.21 → 0.39, and the geometry
regressions to where the raw observation could not go (height above surface
R² 0.56 → 0.94, edge distance 0.58 → 0.78, support 0.66 → 0.85, own velocity
0.72 → 0.97, own 30-tick vertical drift 0.78 → 0.92). The rays *alone* (no
observation) already beat the whole observation on every terrain and fall
label (`fall_30` AP 0.72, drop 0.94), and they make the hit labels more
*linearly* decodable (linear AUC on `hit_15` 0.81 → 0.90, AP 0.014 → 0.096),
which is what a bias-free-linear-then-max encoder needs. 16+4 rays get about
two thirds of the gain (`hit_15` 0.23, `fall_30` 0.62); the third hit per ray
adds another step (`hit_inflight_30` AP 0.28 → 0.32 alone; `fall_30` 0.72 →
0.79; `squish_60` 0.04 → 0.26, because the second hit sees the platform
behind the opponent's body or arrow; edge distance R² 0.64 → 0.79; on top of
raw: `squish_60` 0.25 → 0.33, `any_death_60` 0.39 → 0.42, edge 0.78 → 0.83)
at no extra CPU cost. Relational per-row features on top of raw add `hit_15` 0.10 → 0.13,
`fall_30` 0.16 → 0.30 and nothing on the linear probes (the type of the row
and its geometry still have to be ANDed); on top of raw+rays they add a
further `hit_15` 0.28 → 0.33 and `squish_60` 0.25 → 0.28. Neither family
recovers opponent state (velocity, cooldowns, drift) — that is the delay, not
the features.

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
Re-running the encoder itself with obs[8:10] zeroed (`acts-024-nopos`) moves
the probes from its fused/GRU features by ≤ 0.02 on every label except
absolute position (fused R² 0.96 → 0.91, GRU 0.94 → 0.87 — it is rebuilt from
the terrain rows), and slightly *improves* the hit labels (fused `hit_15`
0.952 → 0.965). The 320M checkpoint's encoder holds *more* map identity
(fused 0.81 vs 0.74) and more absolute position (0.98 vs 0.96) than the
final one, and more opponent cooldown (foe_platform_ready 0.991 vs 0.958)
but less arrow information (GRU `hit_15` 0.92 vs 0.97): training moved the
representation away from map/position and towards projectiles, at the
expense of opponent state. The behavioural test (`ablate.py`, 200 games per
cell vs three hard scripted bots and two pool checkpoints, side-swapped, same
seeds): zeroing obs[8:10] at inference drops 024-final's score from 0.80 to
0.58 on maps 0–2 and from 0.79 to 0.59 on maps 3–4. The policy *relies* on
absolute position heavily, but the cost is identical on trained and held-out
maps, so it is not map memorisation; position is the network's cheapest
readout of "where am I relative to the water and the blast walls", which is
exactly the egocentric geometry rays supply directly. (Zeroing is
out-of-distribution for a network that never saw zeros there, so 0.22 is an
upper bound on the dependence; the KL numbers are the in-distribution
version.) Note also that in this 1000-game sample 024 does not lose ground on
maps 3–4 at all (0.80 vs 0.79).

**Critic.** Explained variance of the 024 value head against the realised
γ=0.995 return is 0.11 (corr 0.33; value std 0.16 vs return std 0.54), against
the round outcome sign 0.06, against 128-decision bootstrapped returns 0.12,
against its TD(λ=0.9) training target 0.57 (which is what train.py's 0.63
measures) and against r + γV(s') 0.88. The critic is self-consistent but not
predictive — and it is not under-capacity: the ceiling probes above sit at
0.13–0.15 on the same input. The critic is not the lever.

**Recommendation for policy-v4.**
1. *Rays as a first-class input block.* 32 uniform rays + 4 downward from the
   slime centre, range 40 u, per ray: distance/40 (1 if none), one-hot over
   {terrain, water/blast, hostile projectile, opponent body, own object,
   none}, and the hit's velocity relative to the slime along the ray /64
   (negative = closing; for terrain this is −v_self·ray, so the block also
   carries own velocity). 288 floats, 3.5 µs per observation single-threaded
   (~5% of a decision step); a second hit per ray is free and worth
   keeping for arrows behind terrain. Feed the block through its own small
   MLP or a 1-D conv over angle (the ray index is a circular axis) before
   fusion, not through the set encoder. This is the map-invariant geometry
   channel: `raw_nopos+rays32` equals `raw+rays32` on every label.
2. *Relational per-row features in the entity rows*, cheap and specific:
   distance/64, bearing cos/sin, relative velocity (entity − own slime)/256,
   closing speed/256, time-to-closest-approach/1 s (clipped at 1 s), miss
   distance/32 (clipped). Small but real gains on hits and squish, and they
   are the quantities the encoder currently has to synthesise from
   thresholds; with them a linear row projection can represent "this arrow's
   miss distance is below my radius". Add own velocity to the self block.
3. *Attack the delay.* Nothing else in the observation is as far from the
   present as the kinematics are (own velocity R² 0.72, opponent 0.49,
   opponent cooldowns 0.85, all 1.00 without delay; hit AP doubles). Options,
   in order of cheapness: dead-reckon entity positions by 9 ticks in the
   writer (positions + 9/60·velocity, gravity for arrows), expose the
   delayed opponent cooldown *timers* as time-since-fired so the network can
   extrapolate, or revisit whether the 9-tick delay is a faithful model of
   the human latency the policy must survive.
4. *Absolute position: replace, do not just remove.* The trained policy
   leans on obs[8:10] (−0.22 score when zeroed at inference) but equally on
   trained and held-out maps, so it is not the cause of any map gap; what it
   provides is distance to water and to the blast walls, which the ray
   block's kill-type hits give egocentrically and map-invariantly. A v4 with
   rays can drop obs[8:10] (Alonso et al. found removing absolute position
   harmless with rays present); without rays, keep it.
5. *Do not spend effort on the critic.* EV 0.11 is the ceiling of this
   input (0.13–0.15 with privileged labels); the missing outcome information
   is who the opponent is, which only a longer-memory critic conditioned on
   the opponent's behaviour could recover.

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

Heat maps:
[`heat-families-binary-mlp.png`](../../artifacts/rl/probes/dataset/results/heat-families-binary-mlp.png),
[`heat-families-regression-mlp.png`](../../artifacts/rl/probes/dataset/results/heat-families-regression-mlp.png).
Feature sets are appended to the raw delayed observation (`raw+…`) or used
alone; `rel` = relational per-row (299), `rays16` = 16+4 rays × 8 (160),
`rays32` = 32+4 (288), `rays32x3` = 32+4 × 3 hits (864).

Table 5.1 — rare labels, MLP probe, test average precision (a random scorer
scores the positive rate; AUCs are all 0.8–0.99 and hide the differences):

**Feature families: rare labels, average precision** (mlp probe; test average precision; random = the positive rate)

| label | positive rate | raw | rel | raw+rel | rays16 | rays32 | rays32x3 | raw+rays16 | raw+rays32 | raw+rays32x3 | raw+rel+rays32 | raw_nopos+rays32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.0033 | 0.098 | 0.047 | 0.130 | 0.122 | 0.197 | 0.218 | 0.228 | 0.284 | 0.299 | 0.334 | 0.282 |
| hit_30 | 0.0067 | 0.138 | 0.055 | 0.161 | 0.132 | 0.198 | 0.224 | 0.257 | 0.307 | 0.304 | 0.325 | 0.308 |
| hit_60 | 0.0109 | 0.164 | 0.058 | 0.182 | 0.123 | 0.176 | 0.204 | 0.268 | 0.308 | 0.311 | 0.325 | 0.309 |
| hit_inflight_30 | 0.0045 | 0.145 | 0.049 | 0.164 | 0.182 | 0.280 | 0.323 | 0.291 | 0.350 | 0.349 | 0.356 | 0.350 |
| hit_inflight_60 | 0.0071 | 0.184 | 0.046 | 0.198 | 0.157 | 0.240 | 0.283 | 0.311 | 0.346 | 0.342 | 0.345 | 0.344 |
| my_shot_hits_60 | 0.0071 | 0.209 | 0.064 | 0.212 | 0.100 | 0.130 | 0.183 | 0.238 | 0.248 | 0.266 | 0.249 | 0.249 |
| fall_30 | 0.0016 | 0.163 | 0.251 | 0.299 | 0.642 | 0.724 | 0.786 | 0.615 | 0.711 | 0.753 | 0.706 | 0.713 |
| fall_60 | 0.0051 | 0.185 | 0.199 | 0.296 | 0.525 | 0.591 | 0.697 | 0.575 | 0.640 | 0.693 | 0.638 | 0.641 |
| squish_60 | 0.0049 | 0.222 | 0.175 | 0.256 | 0.037 | 0.041 | 0.256 | 0.245 | 0.252 | 0.334 | 0.276 | 0.252 |
| any_death_60 | 0.0234 | 0.213 | 0.119 | 0.251 | 0.212 | 0.258 | 0.334 | 0.357 | 0.386 | 0.417 | 0.405 | 0.386 |


Table 5.2 — regression labels, MLP probe, test R²:

**Feature families: regression labels** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | rel | raw+rel | rays16 | rays32 | rays32x3 | raw+rays16 | raw+rays32 | raw+rays32x3 | raw+rel+rays32 |
|---|---|---|---|---|---|---|---|---|---|---|
| drop | 0.562 | 0.485 | 0.605 | 0.938 | 0.940 | 0.951 | 0.941 | 0.942 | 0.951 | 0.943 |
| edge_distance | 0.579 | 0.349 | 0.607 | 0.588 | 0.640 | 0.786 | 0.766 | 0.783 | 0.826 | 0.792 |
| support | 0.656 | 0.455 | 0.683 | 0.780 | 0.805 | 0.851 | 0.843 | 0.854 | 0.869 | 0.857 |
| terrain_min_ray | 0.813 | 0.713 | 0.853 | 0.891 | 0.904 | 0.928 | 0.935 | 0.941 | 0.939 | 0.948 |
| vx | 0.720 | 0.493 | 0.740 | 0.930 | 0.940 | 0.935 | 0.961 | 0.966 | 0.961 | 0.967 |
| vy | 0.660 | 0.534 | 0.700 | 0.983 | 0.985 | 0.986 | 0.989 | 0.990 | 0.991 | 0.992 |
| foe_vx | 0.486 | 0.445 | 0.491 | 0.083 | 0.134 | 0.165 | 0.504 | 0.513 | 0.519 | 0.518 |
| foe_vy | 0.513 | 0.432 | 0.518 | 0.075 | 0.124 | 0.177 | 0.530 | 0.541 | 0.552 | 0.549 |
| foe_dash_cd | 0.869 | 0.110 | 0.869 | 0.026 | 0.034 | 0.047 | 0.871 | 0.873 | 0.871 | 0.873 |
| foe_bow_cd | 0.847 | 0.166 | 0.848 | 0.071 | 0.108 | 0.150 | 0.854 | 0.859 | 0.861 | 0.862 |
| foe_platform_cd | 0.922 | 0.178 | 0.921 | 0.047 | 0.059 | 0.107 | 0.923 | 0.923 | 0.923 | 0.923 |
| idle_dx10 | 0.460 | 0.219 | 0.485 | 0.267 | 0.272 | 0.283 | 0.527 | 0.527 | 0.509 | 0.543 |
| idle_dy10 | 0.634 | 0.418 | 0.677 | 0.592 | 0.597 | 0.607 | 0.816 | 0.822 | 0.805 | 0.830 |
| idle_dx30 | 0.806 | 0.311 | 0.820 | 0.112 | 0.114 | 0.289 | 0.845 | 0.848 | 0.839 | 0.851 |
| idle_dy30 | 0.779 | 0.674 | 0.815 | 0.741 | 0.748 | 0.775 | 0.913 | 0.917 | 0.908 | 0.919 |
| idle_opp_dx10 | 0.222 | 0.157 | 0.226 | 0.028 | 0.044 | 0.053 | 0.228 | 0.229 | 0.223 | 0.232 |
| idle_opp_dy10 | 0.383 | 0.277 | 0.387 | 0.044 | 0.077 | 0.101 | 0.393 | 0.400 | 0.404 | 0.406 |
| idle_opp_dx30 | 0.642 | 0.111 | 0.636 | 0.012 | 0.018 | 0.063 | 0.649 | 0.649 | 0.622 | 0.642 |
| idle_opp_dy30 | 0.670 | 0.386 | 0.678 | 0.050 | 0.088 | 0.122 | 0.677 | 0.678 | 0.672 | 0.688 |
| real_dx12 | 0.489 | 0.231 | 0.500 | 0.269 | 0.284 | 0.301 | 0.531 | 0.534 | 0.524 | 0.539 |
| real_dy12 | 0.438 | 0.295 | 0.470 | 0.387 | 0.393 | 0.402 | 0.569 | 0.570 | 0.561 | 0.573 |
| real_opp_dx12 | 0.281 | 0.183 | 0.292 | 0.044 | 0.064 | 0.082 | 0.291 | 0.291 | 0.286 | 0.300 |
| real_opp_dy12 | 0.299 | 0.192 | 0.305 | 0.031 | 0.050 | 0.063 | 0.304 | 0.304 | 0.301 | 0.311 |
| real_opp_dx32 | 0.302 | 0.172 | 0.315 | 0.048 | 0.067 | 0.089 | 0.314 | 0.316 | 0.308 | 0.326 |
| real_opp_dy32 | 0.374 | 0.228 | 0.383 | 0.032 | 0.056 | 0.064 | 0.380 | 0.380 | 0.375 | 0.387 |
| return_disc | 0.113 | 0.029 | 0.115 | 0.016 | 0.023 | 0.025 | 0.117 | 0.118 | 0.112 | 0.117 |
| outcome_sign | 0.060 | 0.003 | 0.062 | 0.002 | 0.001 | 0.000 | 0.058 | 0.059 | 0.055 | 0.059 |
| log_ticks_to_end | 0.112 | 0.060 | 0.119 | 0.045 | 0.053 | 0.061 | 0.121 | 0.124 | 0.123 | 0.127 |
| abs_x | 0.986 | 0.969 | 0.987 | 0.451 | 0.490 | 0.692 | 0.992 | 0.991 | 0.990 | 0.992 |
| abs_y | 0.986 | 0.948 | 0.986 | 0.814 | 0.832 | 0.997 | 0.994 | 0.994 | 0.998 | 0.994 |


Linear-probe versions in the appendix: rays also make the hit labels
linearly decodable (`hit_15` linear AUC 0.81 raw → 0.90 raw+rays32, AP 0.014
→ 0.096; `fall_30` AP 0.007 → 0.165), relational features do not (`hit_15`
0.814 → 0.817).

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

| input shifted | maps | KL @-0.3 | KL @-0.2 | KL @-0.1 | KL @+0.1 | KL @+0.2 | KL @+0.3 | dV @+-0.1 | dV @+-0.3 |
|---|---|---|---|---|---|---|---|---|---|
| position | maps012 | 1.3245 | 0.6470 | 0.1766 | 0.2011 | 0.8442 | 1.9676 | 0.0168 | 0.0494 |
| opp_relpos | maps012 | 5.5870 | 3.2253 | 1.0481 | 0.9927 | 2.7127 | 4.2404 | 0.0319 | 0.0860 |
| time_left | maps012 | 0.0081 | 0.0036 | 0.0009 | 0.0007 | 0.0023 | 0.0042 | 0.0064 | 0.0178 |
| null_count | maps012 | 0.0034 | 0.0015 | 0.0004 | 0.0004 | 0.0015 | 0.0034 | 0.0013 | 0.0039 |
| position | maps34 | 1.6383 | 0.8101 | 0.2229 | 0.2713 | 1.1308 | 2.5448 | 0.0187 | 0.0533 |
| opp_relpos | maps34 | 6.6503 | 3.8484 | 1.2853 | 1.1415 | 2.9822 | 4.4696 | 0.0330 | 0.0863 |
| time_left | maps34 | 0.0087 | 0.0038 | 0.0009 | 0.0007 | 0.0024 | 0.0043 | 0.0065 | 0.0180 |
| null_count | maps34 | 0.0036 | 0.0016 | 0.0004 | 0.0004 | 0.0015 | 0.0034 | 0.0013 | 0.0040 |

The null input is the source-count column of an ability that never occurs in
the data (its weights never received a gradient); the opponent's relative
position is shifted in both player-row coordinates.

Map identity is decodable from the raw observation at 0.999 (and 0.999 with
obs[8:10] zeroed — the static entity rows *are* the map), from the pooled
entity vector at 0.99, but only 0.74 from the fused 128 and 0.70 from the GRU
state (majority 0.35). Absolute position itself survives to the GRU state at
R² 0.93. So (a) the representation is not map-keyed in the sense of holding a
map id; it holds a position, and the perturbation shows the policy reads it.
(b) Behaviourally (`ablate.py`, `results/ablate-position.json`; 40
side-swapped games per opponent per condition, identical seeds; score = win
rate with timeouts at 0.5):

| condition | mean | archer-hard | pursuer-hard | mixed-hard | v3b-007-500m | v3b-059-final |
|---|---|---|---|---|---|---|
| maps012/normal | 0.800 | 0.89 | 0.88 | 0.95 | 0.90 | 0.39 |
| maps012/zero_pos | 0.575 | 0.62 | 0.85 | 0.72 | 0.47 | 0.20 |
| maps34/normal | 0.788 | 0.88 | 0.88 | 0.97 | 0.80 | 0.41 |
| maps34/zero_pos | 0.590 | 0.78 | 0.80 | 0.78 | 0.47 | 0.12 |

Zeroing obs[8:10] costs 0.22 on both map sets. The dependence is large and
map-independent: not a lookup that only works on maps 0–2, but a genuine use
of "how far am I from the water / the blast walls", which the terrain rows
only give implicitly (the water line and blast planes are not entities). Any
gap on maps 3–4 (none is visible in this sample: 0.80 vs 0.79) would have to
come from the terrain rows themselves or the learned arrow prior. Caveat:
zeros are out-of-distribution for these two inputs, so 0.22 is an upper bound;
the in-distribution measure is the KL table above. (c) Zeroing obs[8:10]
costs the raw-observation probes nothing (appendix, "Absolute position
ablation": every label within 0.005 of `raw`; `raw_nopos+rays32` =
`raw+rays32`), and the encoder's own features lose only the position readout
when fed zeros (section 1): the information is redundant in the input, but
the trained network does not use the redundant copy.

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

Generated by `report.py` from `results/probes-024.json`, `probes-024-nopos.json`, `probes-early.json`, `perturb-*.json`.

**Input vs encoder: rare labels, average precision** (linear probe; test average precision; random = the positive rate)

| label | positive rate | raw | raw_now | flat | players | slots | entities | slimes | pooled | fused | gru_state | gru_out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.0033 | 0.014 | 0.064 | 0.004 | 0.012 | 0.010 | 0.011 | 0.006 | 0.024 | 0.029 | 0.053 | 0.111 |
| hit_30 | 0.0067 | 0.048 | 0.117 | 0.009 | 0.027 | 0.045 | 0.038 | 0.015 | 0.052 | 0.088 | 0.120 | 0.153 |
| hit_60 | 0.0109 | 0.080 | 0.158 | 0.015 | 0.037 | 0.070 | 0.084 | 0.038 | 0.089 | 0.109 | 0.117 | 0.155 |
| hit_inflight_30 | 0.0045 | 0.041 | 0.120 | 0.006 | 0.011 | 0.039 | 0.042 | 0.008 | 0.062 | 0.073 | 0.075 | 0.120 |
| hit_inflight_60 | 0.0071 | 0.069 | 0.148 | 0.009 | 0.019 | 0.057 | 0.065 | 0.019 | 0.090 | 0.100 | 0.110 | 0.147 |
| my_shot_hits_60 | 0.0071 | 0.075 | 0.130 | 0.011 | 0.020 | 0.056 | 0.064 | 0.015 | 0.081 | 0.092 | 0.106 | 0.117 |
| fall_30 | 0.0016 | 0.007 | 0.024 | 0.004 | 0.002 | 0.002 | 0.003 | 0.003 | 0.006 | 0.006 | 0.003 | 0.007 |
| fall_60 | 0.0051 | 0.035 | 0.068 | 0.019 | 0.006 | 0.008 | 0.008 | 0.017 | 0.024 | 0.054 | 0.027 | 0.034 |
| squish_60 | 0.0049 | 0.076 | 0.112 | 0.062 | 0.052 | 0.028 | 0.037 | 0.024 | 0.041 | 0.050 | 0.051 | 0.083 |
| any_death_60 | 0.0234 | 0.106 | 0.166 | 0.068 | 0.043 | 0.069 | 0.073 | 0.056 | 0.098 | 0.101 | 0.121 | 0.132 |

**Feature families: rare labels, average precision** (linear probe; test average precision; random = the positive rate)

| label | positive rate | raw | rel | raw+rel | rays16 | rays32 | rays32x3 | raw+rays16 | raw+rays32 | raw+rays32x3 | raw+rel+rays32 | raw_nopos+rays32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.0033 | 0.014 | 0.013 | 0.017 | 0.075 | 0.118 | 0.086 | 0.056 | 0.096 | 0.089 | 0.110 | 0.096 |
| hit_30 | 0.0067 | 0.048 | 0.027 | 0.057 | 0.087 | 0.121 | 0.108 | 0.093 | 0.133 | 0.128 | 0.137 | 0.133 |
| hit_60 | 0.0109 | 0.080 | 0.033 | 0.085 | 0.084 | 0.114 | 0.107 | 0.115 | 0.141 | 0.140 | 0.147 | 0.141 |
| hit_inflight_30 | 0.0045 | 0.041 | 0.013 | 0.039 | 0.113 | 0.168 | 0.146 | 0.102 | 0.155 | 0.152 | 0.159 | 0.155 |
| hit_inflight_60 | 0.0071 | 0.069 | 0.016 | 0.068 | 0.110 | 0.154 | 0.141 | 0.120 | 0.164 | 0.160 | 0.163 | 0.164 |
| my_shot_hits_60 | 0.0071 | 0.075 | 0.031 | 0.069 | 0.062 | 0.064 | 0.070 | 0.093 | 0.108 | 0.106 | 0.106 | 0.108 |
| fall_30 | 0.0016 | 0.007 | 0.019 | 0.011 | 0.112 | 0.071 | 0.236 | 0.106 | 0.165 | 0.229 | 0.132 | 0.165 |
| fall_60 | 0.0051 | 0.035 | 0.051 | 0.047 | 0.232 | 0.226 | 0.286 | 0.214 | 0.272 | 0.311 | 0.291 | 0.272 |
| squish_60 | 0.0049 | 0.076 | 0.054 | 0.089 | 0.024 | 0.016 | 0.130 | 0.082 | 0.102 | 0.150 | 0.134 | 0.102 |
| any_death_60 | 0.0234 | 0.106 | 0.063 | 0.118 | 0.102 | 0.116 | 0.156 | 0.146 | 0.163 | 0.198 | 0.177 | 0.163 |

**Input vs encoder: binary labels** (linear probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | raw_now | flat | players | slots | entities | slimes | pooled | fused | gru_state | gru_out | positive rate / n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.814 | 0.945 | 0.572 | 0.790 | 0.732 | 0.754 | 0.686 | 0.827 | 0.807 | 0.909 | 0.935 | 0.003 / n=358632 |
| hit_30 | 0.870 | 0.953 | 0.579 | 0.789 | 0.797 | 0.834 | 0.721 | 0.865 | 0.913 | 0.924 | 0.925 | 0.007 / n=358632 |
| hit_60 | 0.876 | 0.952 | 0.581 | 0.780 | 0.828 | 0.856 | 0.765 | 0.885 | 0.903 | 0.906 | 0.915 | 0.011 / n=358632 |
| hit_inflight_30 | 0.903 | 0.961 | 0.558 | 0.762 | 0.868 | 0.901 | 0.670 | 0.907 | 0.922 | 0.917 | 0.929 | 0.004 / n=358632 |
| hit_inflight_60 | 0.915 | 0.960 | 0.566 | 0.747 | 0.877 | 0.911 | 0.716 | 0.930 | 0.936 | 0.921 | 0.922 | 0.007 / n=358632 |
| my_shot_hits_60 | 0.898 | 0.940 | 0.615 | 0.735 | 0.879 | 0.899 | 0.676 | 0.910 | 0.903 | 0.891 | 0.895 | 0.007 / n=358632 |
| fall_30 | 0.779 | 0.878 | 0.691 | 0.502 | 0.551 | 0.597 | 0.685 | 0.721 | 0.664 | 0.623 | 0.685 | 0.002 / n=358632 |
| fall_60 | 0.853 | 0.904 | 0.740 | 0.536 | 0.607 | 0.579 | 0.760 | 0.796 | 0.865 | 0.798 | 0.810 | 0.005 / n=358632 |
| squish_60 | 0.844 | 0.862 | 0.848 | 0.872 | 0.827 | 0.859 | 0.759 | 0.863 | 0.867 | 0.822 | 0.847 | 0.005 / n=358632 |
| any_death_60 | 0.801 | 0.859 | 0.656 | 0.668 | 0.725 | 0.738 | 0.723 | 0.795 | 0.801 | 0.814 | 0.816 | 0.023 / n=358632 |
| lands_dry | 0.863 | 0.906 | 0.822 | 0.670 | 0.702 | 0.742 | 0.802 | 0.835 | 0.846 | 0.832 | 0.832 | 0.927 / n=358632 |
| terrain_within_dash | 0.919 | 0.954 | 0.825 | 0.751 | 0.893 | 0.841 | 0.852 | 0.928 | 0.921 | 0.919 | 0.917 | 0.943 / n=358632 |
| grounded | 0.950 | 1.000 | 0.847 | 0.707 | 0.787 | 0.776 | 0.910 | 0.922 | 0.941 | 0.933 | 0.933 | 0.459 / n=358632 |
| foe_dash_ready | 0.991 | 1.000 | 0.665 | 0.659 | 0.989 | 0.698 | 0.720 | 0.990 | 0.941 | 0.922 | 0.909 | 0.295 / n=358632 |
| foe_bow_ready | 0.986 | 1.000 | 0.620 | 0.618 | 0.982 | 0.747 | 0.664 | 0.985 | 0.962 | 0.954 | 0.951 | 0.425 / n=358632 |
| foe_platform_ready | 0.994 | 1.000 | 0.673 | 0.715 | 0.987 | 0.769 | 0.704 | 0.989 | 0.915 | 0.887 | 0.884 | 0.199 / n=358632 |
| real_death_60 | 0.766 | 0.804 | 0.627 | 0.711 | 0.732 | 0.656 | 0.665 | 0.784 | 0.807 | 0.816 | 0.807 | 0.021 / n=358632 |
| win | 0.627 | 0.629 | 0.608 | 0.547 | 0.598 | 0.545 | 0.544 | 0.611 | 0.618 | 0.627 | 0.626 | 0.409 / n=358632 |
| own_arrow_in_flight | 0.986 | 1.000 | 0.735 | 0.649 | 0.917 | 0.978 | 0.719 | 0.987 | 0.978 | 0.968 | 0.962 | 0.133 / n=358632 |
| opp_alive | 0.513 | 1.000 | 0.536 | 0.744 | 0.504 | 0.633 | 0.541 | 0.569 | 0.721 | 0.730 | 0.693 | 1.000 / n=358632 |
| map | 0.974 | 0.976 | 0.609 | 0.408 | 0.445 | 0.924 | 0.419 | 0.925 | 0.685 | 0.638 | 0.628 | maj 0.353 |

**Input vs encoder: regression labels** (linear probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | raw_now | flat | players | slots | entities | slimes | pooled | fused | gru_state | gru_out | positive rate / n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drop | 0.405 | 0.438 | 0.266 | 0.071 | 0.102 | 0.159 | 0.299 | 0.384 | 0.387 | 0.357 | 0.346 | std 17.83 / n=358632 |
| edge_distance | 0.294 | 0.355 | 0.112 | 0.023 | 0.029 | 0.090 | 0.153 | 0.202 | 0.175 | 0.162 | 0.157 | std 2.52 / n=358632 |
| support | 0.423 | 0.559 | 0.201 | 0.059 | 0.084 | 0.091 | 0.363 | 0.414 | 0.376 | 0.355 | 0.335 | std 0.46 / n=358632 |
| terrain_min_ray | 0.574 | 0.662 | 0.232 | 0.117 | 0.283 | 0.211 | 0.363 | 0.535 | 0.493 | 0.488 | 0.455 | std 5.06 / n=358632 |
| vx | 0.477 | 0.764 | 0.373 | 0.032 | 0.059 | 0.046 | 0.300 | 0.337 | 0.371 | 0.369 | 0.378 | std 7.22 / n=358632 |
| vy | 0.462 | 0.760 | 0.101 | 0.017 | 0.051 | 0.055 | 0.424 | 0.461 | 0.377 | 0.314 | 0.318 | std 12.96 / n=358632 |
| foe_vx | 0.314 | 0.723 | 0.018 | 0.036 | 0.051 | 0.048 | 0.270 | 0.341 | 0.137 | 0.129 | 0.132 | std 7.22 / n=358632 |
| foe_vy | 0.349 | 0.688 | 0.005 | 0.019 | 0.038 | 0.048 | 0.389 | 0.440 | 0.277 | 0.212 | 0.199 | std 12.95 / n=358632 |
| foe_dash_cd | 0.854 | 0.985 | 0.032 | 0.055 | 0.828 | 0.120 | 0.155 | 0.836 | 0.556 | 0.455 | 0.427 | std 1.33 / n=358632 |
| foe_bow_cd | 0.822 | 0.980 | 0.044 | 0.033 | 0.806 | 0.452 | 0.175 | 0.821 | 0.742 | 0.711 | 0.681 | std 0.93 / n=358632 |
| foe_platform_cd | 0.913 | 0.989 | 0.093 | 0.410 | 0.858 | 0.422 | 0.278 | 0.881 | 0.610 | 0.556 | 0.519 | std 2.23 / n=358632 |
| idle_dx10 | 0.277 | 0.315 | 0.235 | 0.015 | 0.039 | 0.033 | 0.102 | 0.143 | 0.238 | 0.228 | 0.230 | std 1.24 / n=358632 |
| idle_dy10 | 0.356 | 0.681 | 0.139 | 0.053 | 0.089 | 0.086 | 0.281 | 0.334 | 0.308 | 0.280 | 0.279 | std 2.45 / n=358632 |
| idle_dx30 | 0.689 | 0.727 | 0.317 | 0.043 | 0.109 | 0.149 | 0.086 | 0.286 | 0.298 | 0.289 | 0.276 | std 2.57 / n=358632 |
| idle_dy30 | 0.596 | 0.728 | 0.344 | 0.248 | 0.305 | 0.317 | 0.439 | 0.576 | 0.543 | 0.532 | 0.513 | std 6.04 / n=358632 |
| idle_opp_dx10 | 0.111 | 0.232 | 0.010 | 0.016 | 0.033 | 0.028 | 0.097 | 0.144 | 0.063 | 0.064 | 0.068 | std 1.24 / n=358632 |
| idle_opp_dy10 | 0.263 | 0.624 | 0.007 | 0.055 | 0.082 | 0.084 | 0.259 | 0.321 | 0.220 | 0.186 | 0.179 | std 2.45 / n=358632 |
| idle_opp_dx30 | 0.162 | 0.225 | 0.032 | 0.057 | 0.035 | 0.108 | 0.070 | 0.313 | 0.129 | 0.119 | 0.118 | std 2.57 / n=358632 |
| idle_opp_dy30 | 0.506 | 0.634 | 0.030 | 0.247 | 0.293 | 0.307 | 0.418 | 0.561 | 0.432 | 0.390 | 0.379 | std 6.04 / n=358632 |
| real_dx12 | 0.354 | 0.386 | 0.338 | 0.031 | 0.072 | 0.062 | 0.109 | 0.182 | 0.367 | 0.353 | 0.359 | std 2.40 / n=355602 |
| real_dy12 | 0.257 | 0.476 | 0.125 | 0.045 | 0.086 | 0.074 | 0.179 | 0.243 | 0.239 | 0.228 | 0.234 | std 3.43 / n=355602 |
| real_opp_dx12 | 0.121 | 0.219 | 0.021 | 0.031 | 0.067 | 0.058 | 0.107 | 0.181 | 0.115 | 0.118 | 0.126 | std 2.40 / n=355602 |
| real_opp_dy12 | 0.189 | 0.385 | 0.010 | 0.046 | 0.078 | 0.073 | 0.166 | 0.235 | 0.162 | 0.146 | 0.140 | std 3.43 / n=355602 |
| real_opp_dx32 | 0.109 | 0.178 | 0.030 | 0.044 | 0.089 | 0.085 | 0.090 | 0.189 | 0.146 | 0.150 | 0.159 | std 5.12 / n=350552 |
| real_opp_dy32 | 0.246 | 0.314 | 0.014 | 0.084 | 0.145 | 0.138 | 0.179 | 0.294 | 0.220 | 0.201 | 0.198 | std 6.98 / n=350552 |
| return_disc | 0.063 | 0.068 | 0.042 | 0.014 | 0.042 | 0.007 | 0.012 | 0.050 | 0.073 | 0.090 | 0.104 | std 0.57 / n=358632 |
| outcome_sign | 0.041 | 0.043 | 0.029 | 0.000 | 0.024 | -0.005 | -0.004 | 0.025 | 0.041 | 0.046 | 0.046 | std 0.98 / n=358632 |
| log_ticks_to_end | 0.092 | 0.112 | 0.069 | 0.040 | 0.066 | 0.047 | 0.050 | 0.089 | 0.105 | 0.109 | 0.104 | std 1.35 / n=358632 |
| abs_x | 0.976 | 0.980 | 0.985 | 0.412 | 0.417 | 0.956 | 0.455 | 0.965 | 0.919 | 0.882 | 0.866 | std 22.80 / n=358632 |
| abs_y | 0.977 | 0.989 | 0.972 | 0.406 | 0.481 | 0.934 | 0.448 | 0.953 | 0.930 | 0.894 | 0.887 | std 16.02 / n=358632 |

**Feature families: binary labels** (linear probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | rel | raw+rel | rays16 | rays32 | rays32x3 | raw+rays16 | raw+rays32 | raw+rays32x3 | raw+rel+rays32 |
|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.814 | 0.745 | 0.817 | 0.781 | 0.821 | 0.823 | 0.890 | 0.899 | 0.893 | 0.919 |
| hit_30 | 0.870 | 0.768 | 0.876 | 0.769 | 0.807 | 0.818 | 0.908 | 0.912 | 0.909 | 0.920 |
| hit_60 | 0.876 | 0.750 | 0.876 | 0.731 | 0.762 | 0.778 | 0.900 | 0.900 | 0.894 | 0.906 |
| hit_inflight_30 | 0.903 | 0.714 | 0.889 | 0.809 | 0.884 | 0.896 | 0.931 | 0.934 | 0.926 | 0.934 |
| hit_inflight_60 | 0.915 | 0.685 | 0.908 | 0.771 | 0.825 | 0.840 | 0.933 | 0.930 | 0.919 | 0.931 |
| my_shot_hits_60 | 0.898 | 0.794 | 0.896 | 0.756 | 0.763 | 0.779 | 0.915 | 0.910 | 0.897 | 0.907 |
| fall_30 | 0.779 | 0.817 | 0.807 | 0.923 | 0.888 | 0.971 | 0.943 | 0.950 | 0.964 | 0.943 |
| fall_60 | 0.853 | 0.818 | 0.849 | 0.882 | 0.885 | 0.924 | 0.932 | 0.935 | 0.939 | 0.936 |
| squish_60 | 0.844 | 0.841 | 0.830 | 0.869 | 0.824 | 0.877 | 0.863 | 0.861 | 0.840 | 0.860 |
| any_death_60 | 0.801 | 0.725 | 0.810 | 0.688 | 0.709 | 0.771 | 0.834 | 0.837 | 0.844 | 0.848 |
| lands_dry | 0.863 | 0.844 | 0.884 | 0.952 | 0.953 | 0.956 | 0.957 | 0.957 | 0.958 | 0.958 |
| terrain_within_dash | 0.919 | 0.907 | 0.960 | 0.965 | 0.965 | 0.971 | 0.978 | 0.979 | 0.980 | 0.987 |
| grounded | 0.950 | 0.865 | 0.961 | 0.818 | 0.826 | 0.845 | 0.968 | 0.969 | 0.971 | 0.977 |
| foe_dash_ready | 0.991 | 0.624 | 0.991 | 0.599 | 0.608 | 0.639 | 0.991 | 0.991 | 0.992 | 0.991 |
| foe_bow_ready | 0.986 | 0.637 | 0.986 | 0.604 | 0.614 | 0.649 | 0.988 | 0.988 | 0.988 | 0.988 |
| foe_platform_ready | 0.994 | 0.663 | 0.994 | 0.586 | 0.593 | 0.618 | 0.993 | 0.993 | 0.993 | 0.993 |
| real_death_60 | 0.766 | 0.747 | 0.783 | 0.653 | 0.658 | 0.703 | 0.796 | 0.800 | 0.803 | 0.814 |
| win | 0.627 | 0.551 | 0.630 | 0.544 | 0.551 | 0.557 | 0.630 | 0.631 | 0.630 | 0.633 |
| own_arrow_in_flight | 0.986 | 0.842 | 0.986 | 0.769 | 0.816 | 0.843 | 0.995 | 0.997 | 0.996 | 0.997 |
| opp_alive | 0.513 | 0.617 | 0.542 | 0.639 | 0.649 | 0.626 | 0.602 | 0.606 | 0.589 | 0.608 |

**Feature families: regression labels** (linear probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | rel | raw+rel | rays16 | rays32 | rays32x3 | raw+rays16 | raw+rays32 | raw+rays32x3 | raw+rel+rays32 |
|---|---|---|---|---|---|---|---|---|---|---|
| drop | 0.405 | 0.329 | 0.462 | 0.918 | 0.920 | 0.934 | 0.924 | 0.925 | 0.936 | 0.925 |
| edge_distance | 0.294 | 0.149 | 0.321 | 0.323 | 0.380 | 0.452 | 0.419 | 0.451 | 0.508 | 0.468 |
| support | 0.423 | 0.273 | 0.454 | 0.462 | 0.487 | 0.565 | 0.600 | 0.614 | 0.649 | 0.630 |
| terrain_min_ray | 0.574 | 0.539 | 0.703 | 0.564 | 0.574 | 0.601 | 0.744 | 0.749 | 0.763 | 0.798 |
| vx | 0.477 | 0.429 | 0.528 | 0.667 | 0.683 | 0.693 | 0.758 | 0.768 | 0.774 | 0.801 |
| vy | 0.462 | 0.452 | 0.529 | 0.934 | 0.937 | 0.939 | 0.941 | 0.944 | 0.948 | 0.950 |
| foe_vx | 0.314 | 0.413 | 0.418 | 0.016 | 0.019 | 0.023 | 0.317 | 0.318 | 0.319 | 0.429 |
| foe_vy | 0.349 | 0.342 | 0.440 | 0.006 | 0.007 | 0.013 | 0.351 | 0.351 | 0.353 | 0.445 |
| foe_dash_cd | 0.854 | 0.044 | 0.856 | 0.007 | 0.009 | 0.018 | 0.863 | 0.863 | 0.863 | 0.863 |
| foe_bow_cd | 0.822 | 0.070 | 0.824 | 0.043 | 0.055 | 0.091 | 0.836 | 0.836 | 0.837 | 0.837 |
| foe_platform_cd | 0.913 | 0.081 | 0.912 | 0.024 | 0.027 | 0.051 | 0.920 | 0.920 | 0.920 | 0.920 |
| idle_dx10 | 0.277 | 0.166 | 0.286 | 0.177 | 0.182 | 0.189 | 0.317 | 0.319 | 0.321 | 0.325 |
| idle_dy10 | 0.356 | 0.325 | 0.408 | 0.528 | 0.531 | 0.535 | 0.573 | 0.576 | 0.582 | 0.594 |
| idle_dx30 | 0.689 | 0.140 | 0.696 | 0.072 | 0.074 | 0.138 | 0.713 | 0.714 | 0.716 | 0.719 |
| idle_dy30 | 0.596 | 0.531 | 0.667 | 0.604 | 0.610 | 0.632 | 0.729 | 0.732 | 0.744 | 0.755 |
| idle_opp_dx10 | 0.111 | 0.142 | 0.155 | 0.007 | 0.008 | 0.010 | 0.114 | 0.115 | 0.115 | 0.159 |
| idle_opp_dy10 | 0.263 | 0.174 | 0.304 | 0.007 | 0.008 | 0.012 | 0.264 | 0.264 | 0.265 | 0.306 |
| idle_opp_dx30 | 0.162 | 0.064 | 0.178 | 0.005 | 0.005 | 0.019 | 0.190 | 0.188 | 0.191 | 0.199 |
| idle_opp_dy30 | 0.506 | 0.143 | 0.520 | 0.017 | 0.021 | 0.029 | 0.511 | 0.511 | 0.510 | 0.526 |
| real_dx12 | 0.354 | 0.162 | 0.360 | 0.183 | 0.192 | 0.202 | 0.386 | 0.389 | 0.393 | 0.393 |
| real_dy12 | 0.257 | 0.214 | 0.293 | 0.331 | 0.334 | 0.339 | 0.391 | 0.392 | 0.400 | 0.405 |
| real_opp_dx12 | 0.121 | 0.137 | 0.160 | 0.014 | 0.017 | 0.024 | 0.126 | 0.128 | 0.131 | 0.166 |
| real_opp_dy12 | 0.189 | 0.107 | 0.210 | 0.007 | 0.010 | 0.012 | 0.189 | 0.189 | 0.190 | 0.210 |
| real_opp_dx32 | 0.109 | 0.103 | 0.140 | 0.017 | 0.020 | 0.030 | 0.117 | 0.119 | 0.124 | 0.148 |
| real_opp_dy32 | 0.246 | 0.091 | 0.252 | 0.012 | 0.017 | 0.020 | 0.246 | 0.247 | 0.247 | 0.252 |
| return_disc | 0.063 | 0.014 | 0.068 | 0.009 | 0.013 | 0.013 | 0.066 | 0.068 | 0.065 | 0.072 |
| outcome_sign | 0.041 | -0.006 | 0.043 | 0.000 | 0.002 | -0.006 | 0.036 | 0.036 | 0.032 | 0.039 |
| log_ticks_to_end | 0.092 | 0.049 | 0.102 | 0.028 | 0.031 | 0.038 | 0.099 | 0.100 | 0.101 | 0.108 |
| abs_x | 0.976 | 0.888 | 0.977 | 0.345 | 0.375 | 0.562 | 0.988 | 0.988 | 0.989 | 0.989 |
| abs_y | 0.977 | 0.861 | 0.978 | 0.753 | 0.765 | 0.992 | 0.990 | 0.991 | 0.995 | 0.991 |

**Absolute position ablation: binary labels** (linear probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | raw_nopos | pos | raw_nopos+rays32 | raw_nopos+rel | fused | gru_state |
|---|---|---|---|---|---|---|---|
| hit_15 | 0.814 | 0.814 | 0.468 | 0.899 | 0.903 | 0.807 | 0.909 |
| hit_30 | 0.870 | 0.870 | 0.545 | 0.912 | 0.915 | 0.913 | 0.924 |
| hit_60 | 0.876 | 0.877 | 0.534 | 0.901 | 0.901 | 0.903 | 0.906 |
| hit_inflight_30 | 0.903 | 0.903 | 0.455 | 0.934 | 0.932 | 0.922 | 0.917 |
| hit_inflight_60 | 0.915 | 0.916 | 0.520 | 0.930 | 0.931 | 0.936 | 0.921 |
| my_shot_hits_60 | 0.898 | 0.899 | 0.597 | 0.910 | 0.924 | 0.903 | 0.891 |
| fall_30 | 0.779 | 0.777 | 0.789 | 0.950 | 0.834 | 0.664 | 0.623 |
| fall_60 | 0.853 | 0.853 | 0.697 | 0.935 | 0.871 | 0.865 | 0.798 |
| squish_60 | 0.844 | 0.845 | 0.677 | 0.861 | 0.863 | 0.867 | 0.822 |
| any_death_60 | 0.801 | 0.801 | 0.631 | 0.837 | 0.819 | 0.801 | 0.814 |
| lands_dry | 0.863 | 0.863 | 0.566 | 0.957 | 0.889 | 0.846 | 0.832 |
| terrain_within_dash | 0.919 | 0.919 | 0.754 | 0.979 | 0.963 | 0.921 | 0.919 |
| grounded | 0.950 | 0.950 | 0.584 | 0.969 | 0.963 | 0.941 | 0.933 |
| foe_dash_ready | 0.991 | 0.991 | 0.565 | 0.991 | 0.993 | 0.941 | 0.922 |
| foe_bow_ready | 0.986 | 0.986 | 0.559 | 0.988 | 0.988 | 0.962 | 0.954 |
| foe_platform_ready | 0.994 | 0.994 | 0.489 | 0.993 | 0.995 | 0.915 | 0.887 |
| real_death_60 | 0.766 | 0.767 | 0.577 | 0.800 | 0.803 | 0.807 | 0.816 |
| win | 0.627 | 0.627 | 0.497 | 0.631 | 0.628 | 0.618 | 0.627 |
| own_arrow_in_flight | 0.986 | 0.986 | 0.575 | 0.997 | 0.988 | 0.978 | 0.968 |
| opp_alive | 0.513 | 0.512 | 0.525 | 0.607 | 0.537 | 0.721 | 0.730 |
| map | 0.974 | 0.974 | 0.343 | 0.983 | 0.981 | 0.685 | 0.638 |

**Absolute position ablation: regression labels** (linear probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | raw_nopos | pos | raw_nopos+rays32 | raw_nopos+rel | fused | gru_state |
|---|---|---|---|---|---|---|---|
| drop | 0.405 | 0.405 | 0.022 | 0.925 | 0.462 | 0.387 | 0.357 |
| edge_distance | 0.294 | 0.294 | 0.030 | 0.450 | 0.327 | 0.175 | 0.162 |
| support | 0.423 | 0.423 | 0.013 | 0.613 | 0.453 | 0.376 | 0.355 |
| terrain_min_ray | 0.574 | 0.574 | 0.080 | 0.748 | 0.705 | 0.493 | 0.488 |
| vx | 0.477 | 0.477 | 0.008 | 0.768 | 0.527 | 0.371 | 0.369 |
| vy | 0.462 | 0.462 | 0.016 | 0.944 | 0.529 | 0.377 | 0.314 |
| foe_vx | 0.314 | 0.314 | 0.002 | 0.318 | 0.427 | 0.137 | 0.129 |
| foe_vy | 0.349 | 0.349 | 0.000 | 0.351 | 0.445 | 0.277 | 0.212 |
| foe_dash_cd | 0.854 | 0.854 | 0.002 | 0.863 | 0.863 | 0.556 | 0.455 |
| foe_bow_cd | 0.822 | 0.822 | 0.004 | 0.836 | 0.835 | 0.742 | 0.711 |
| foe_platform_cd | 0.913 | 0.913 | 0.001 | 0.920 | 0.920 | 0.610 | 0.556 |
| idle_dx10 | 0.277 | 0.277 | 0.001 | 0.319 | 0.285 | 0.238 | 0.228 |
| idle_dy10 | 0.356 | 0.356 | 0.025 | 0.576 | 0.408 | 0.308 | 0.280 |
| idle_dx30 | 0.689 | 0.689 | 0.009 | 0.714 | 0.701 | 0.298 | 0.289 |
| idle_dy30 | 0.596 | 0.596 | 0.061 | 0.732 | 0.668 | 0.543 | 0.532 |
| idle_opp_dx10 | 0.111 | 0.111 | 0.002 | 0.115 | 0.157 | 0.063 | 0.064 |
| idle_opp_dy10 | 0.263 | 0.263 | -0.000 | 0.264 | 0.304 | 0.220 | 0.186 |
| idle_opp_dx30 | 0.162 | 0.162 | 0.006 | 0.188 | 0.199 | 0.129 | 0.119 |
| idle_opp_dy30 | 0.506 | 0.506 | 0.001 | 0.511 | 0.525 | 0.432 | 0.390 |
| real_dx12 | 0.354 | 0.354 | 0.009 | 0.389 | 0.360 | 0.367 | 0.353 |
| real_dy12 | 0.257 | 0.257 | 0.032 | 0.392 | 0.293 | 0.239 | 0.228 |
| real_opp_dx12 | 0.121 | 0.120 | 0.002 | 0.128 | 0.160 | 0.115 | 0.118 |
| real_opp_dy12 | 0.189 | 0.189 | 0.000 | 0.189 | 0.209 | 0.162 | 0.146 |
| real_opp_dx32 | 0.109 | 0.109 | 0.002 | 0.118 | 0.140 | 0.146 | 0.150 |
| real_opp_dy32 | 0.246 | 0.245 | 0.000 | 0.246 | 0.251 | 0.220 | 0.201 |
| return_disc | 0.063 | 0.063 | 0.001 | 0.067 | 0.067 | 0.073 | 0.090 |
| outcome_sign | 0.041 | 0.042 | -0.009 | 0.036 | 0.040 | 0.041 | 0.046 |
| log_ticks_to_end | 0.092 | 0.092 | 0.002 | 0.099 | 0.103 | 0.105 | 0.109 |
| abs_x | 0.976 | 0.962 | 0.985 | 0.968 | 0.966 | 0.919 | 0.882 |
| abs_y | 0.977 | 0.958 | 0.972 | 0.969 | 0.966 | 0.930 | 0.894 |

**Input vs encoder: rare labels, average precision** (mlp probe; test average precision; random = the positive rate)

| label | positive rate | raw | raw_now | flat | players | slots | entities | slimes | pooled | fused | gru_state | gru_out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.0033 | 0.098 | 0.217 | 0.014 | 0.033 | 0.062 | 0.059 | 0.025 | 0.122 | 0.123 | 0.146 | 0.153 |
| hit_30 | 0.0067 | 0.138 | 0.258 | 0.028 | 0.040 | 0.102 | 0.123 | 0.058 | 0.184 | 0.178 | 0.206 | 0.211 |
| hit_60 | 0.0109 | 0.164 | 0.280 | 0.037 | 0.048 | 0.128 | 0.135 | 0.074 | 0.199 | 0.193 | 0.209 | 0.217 |
| hit_inflight_30 | 0.0045 | 0.145 | 0.245 | 0.023 | 0.018 | 0.098 | 0.126 | 0.039 | 0.196 | 0.160 | 0.188 | 0.206 |
| hit_inflight_60 | 0.0071 | 0.184 | 0.277 | 0.028 | 0.024 | 0.136 | 0.147 | 0.059 | 0.213 | 0.180 | 0.205 | 0.217 |
| my_shot_hits_60 | 0.0071 | 0.209 | 0.268 | 0.035 | 0.023 | 0.133 | 0.136 | 0.059 | 0.196 | 0.172 | 0.194 | 0.196 |
| fall_30 | 0.0016 | 0.163 | 0.299 | 0.020 | 0.004 | 0.006 | 0.012 | 0.034 | 0.118 | 0.095 | 0.070 | 0.105 |
| fall_60 | 0.0051 | 0.185 | 0.282 | 0.051 | 0.011 | 0.016 | 0.026 | 0.038 | 0.112 | 0.131 | 0.109 | 0.128 |
| squish_60 | 0.0049 | 0.222 | 0.267 | 0.111 | 0.050 | 0.043 | 0.100 | 0.059 | 0.116 | 0.102 | 0.115 | 0.154 |
| any_death_60 | 0.0234 | 0.213 | 0.303 | 0.133 | 0.050 | 0.104 | 0.126 | 0.081 | 0.175 | 0.174 | 0.184 | 0.202 |

**Feature families: rare labels, average precision** (mlp probe; test average precision; random = the positive rate)

| label | positive rate | raw | rel | raw+rel | rays16 | rays32 | rays32x3 | raw+rays16 | raw+rays32 | raw+rays32x3 | raw+rel+rays32 | raw_nopos+rays32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.0033 | 0.098 | 0.047 | 0.130 | 0.122 | 0.197 | 0.218 | 0.228 | 0.284 | 0.299 | 0.334 | 0.282 |
| hit_30 | 0.0067 | 0.138 | 0.055 | 0.161 | 0.132 | 0.198 | 0.224 | 0.257 | 0.307 | 0.304 | 0.325 | 0.308 |
| hit_60 | 0.0109 | 0.164 | 0.058 | 0.182 | 0.123 | 0.176 | 0.204 | 0.268 | 0.308 | 0.311 | 0.325 | 0.309 |
| hit_inflight_30 | 0.0045 | 0.145 | 0.049 | 0.164 | 0.182 | 0.280 | 0.323 | 0.291 | 0.350 | 0.349 | 0.356 | 0.350 |
| hit_inflight_60 | 0.0071 | 0.184 | 0.046 | 0.198 | 0.157 | 0.240 | 0.283 | 0.311 | 0.346 | 0.342 | 0.345 | 0.344 |
| my_shot_hits_60 | 0.0071 | 0.209 | 0.064 | 0.212 | 0.100 | 0.130 | 0.183 | 0.238 | 0.248 | 0.266 | 0.249 | 0.249 |
| fall_30 | 0.0016 | 0.163 | 0.251 | 0.299 | 0.642 | 0.724 | 0.786 | 0.615 | 0.711 | 0.753 | 0.706 | 0.713 |
| fall_60 | 0.0051 | 0.185 | 0.199 | 0.296 | 0.525 | 0.591 | 0.697 | 0.575 | 0.640 | 0.693 | 0.638 | 0.641 |
| squish_60 | 0.0049 | 0.222 | 0.175 | 0.256 | 0.037 | 0.041 | 0.256 | 0.245 | 0.252 | 0.334 | 0.276 | 0.252 |
| any_death_60 | 0.0234 | 0.213 | 0.119 | 0.251 | 0.212 | 0.258 | 0.334 | 0.357 | 0.386 | 0.417 | 0.405 | 0.386 |

**Input vs encoder: binary labels** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | raw_now | flat | players | slots | entities | slimes | pooled | fused | gru_state | gru_out | positive rate / n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.945 | 0.983 | 0.768 | 0.856 | 0.892 | 0.897 | 0.823 | 0.951 | 0.952 | 0.968 | 0.969 | 0.003 / n=358632 |
| hit_30 | 0.943 | 0.978 | 0.763 | 0.840 | 0.907 | 0.915 | 0.839 | 0.958 | 0.957 | 0.966 | 0.967 | 0.007 / n=358632 |
| hit_60 | 0.932 | 0.972 | 0.735 | 0.820 | 0.903 | 0.908 | 0.833 | 0.950 | 0.948 | 0.955 | 0.954 | 0.011 / n=358632 |
| hit_inflight_30 | 0.967 | 0.985 | 0.771 | 0.817 | 0.941 | 0.953 | 0.826 | 0.977 | 0.971 | 0.975 | 0.977 | 0.004 / n=358632 |
| hit_inflight_60 | 0.965 | 0.981 | 0.743 | 0.800 | 0.946 | 0.955 | 0.842 | 0.974 | 0.969 | 0.972 | 0.973 | 0.007 / n=358632 |
| my_shot_hits_60 | 0.967 | 0.978 | 0.796 | 0.797 | 0.941 | 0.944 | 0.845 | 0.969 | 0.961 | 0.964 | 0.963 | 0.007 / n=358632 |
| fall_30 | 0.954 | 0.962 | 0.900 | 0.722 | 0.753 | 0.833 | 0.925 | 0.953 | 0.964 | 0.951 | 0.958 | 0.002 / n=358632 |
| fall_60 | 0.951 | 0.967 | 0.847 | 0.706 | 0.745 | 0.819 | 0.878 | 0.936 | 0.942 | 0.933 | 0.940 | 0.005 / n=358632 |
| squish_60 | 0.934 | 0.943 | 0.904 | 0.894 | 0.895 | 0.912 | 0.885 | 0.923 | 0.919 | 0.922 | 0.926 | 0.005 / n=358632 |
| any_death_60 | 0.868 | 0.918 | 0.727 | 0.702 | 0.776 | 0.796 | 0.775 | 0.862 | 0.875 | 0.875 | 0.881 | 0.023 / n=358632 |
| lands_dry | 0.928 | 0.967 | 0.870 | 0.673 | 0.751 | 0.838 | 0.836 | 0.900 | 0.901 | 0.901 | 0.900 | 0.927 / n=358632 |
| terrain_within_dash | 0.975 | 0.992 | 0.858 | 0.758 | 0.916 | 0.900 | 0.882 | 0.965 | 0.950 | 0.950 | 0.949 | 0.943 / n=358632 |
| grounded | 0.980 | 1.000 | 0.872 | 0.719 | 0.815 | 0.823 | 0.925 | 0.941 | 0.962 | 0.958 | 0.958 | 0.459 / n=358632 |
| foe_dash_ready | 0.996 | 1.000 | 0.681 | 0.670 | 0.995 | 0.737 | 0.745 | 0.997 | 0.982 | 0.975 | 0.969 | 0.295 / n=358632 |
| foe_bow_ready | 0.992 | 1.000 | 0.637 | 0.642 | 0.992 | 0.787 | 0.712 | 0.993 | 0.984 | 0.982 | 0.979 | 0.425 / n=358632 |
| foe_platform_ready | 0.997 | 1.000 | 0.708 | 0.721 | 0.997 | 0.801 | 0.734 | 0.997 | 0.958 | 0.945 | 0.939 | 0.199 / n=358632 |
| real_death_60 | 0.834 | 0.854 | 0.699 | 0.750 | 0.793 | 0.771 | 0.761 | 0.849 | 0.860 | 0.864 | 0.864 | 0.021 / n=358632 |
| win | 0.641 | 0.644 | 0.616 | 0.554 | 0.607 | 0.567 | 0.556 | 0.621 | 0.629 | 0.637 | 0.632 | 0.409 / n=358632 |
| own_arrow_in_flight | 0.994 | 1.000 | 0.789 | 0.662 | 0.935 | 0.984 | 0.774 | 0.991 | 0.990 | 0.988 | 0.986 | 0.133 / n=358632 |
| opp_alive | 0.881 | 1.000 | 0.589 | 0.739 | 0.819 | 0.811 | 0.699 | 0.873 | 0.867 | 0.854 | 0.850 | 1.000 / n=358632 |
| map | 0.999 | 0.999 | 0.711 | 0.411 | 0.462 | 0.990 | 0.430 | 0.981 | 0.737 | 0.702 | 0.688 | maj 0.353 |

**Input vs encoder: regression labels** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | raw_now | flat | players | slots | entities | slimes | pooled | fused | gru_state | gru_out | positive rate / n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drop | 0.562 | 0.604 | 0.327 | 0.082 | 0.176 | 0.296 | 0.369 | 0.514 | 0.508 | 0.491 | 0.489 | std 17.83 / n=358632 |
| edge_distance | 0.579 | 0.660 | 0.166 | 0.030 | 0.063 | 0.235 | 0.204 | 0.332 | 0.261 | 0.245 | 0.243 | std 2.52 / n=358632 |
| support | 0.656 | 0.779 | 0.260 | 0.064 | 0.127 | 0.166 | 0.422 | 0.518 | 0.504 | 0.478 | 0.475 | std 0.46 / n=358632 |
| terrain_min_ray | 0.813 | 0.885 | 0.282 | 0.124 | 0.408 | 0.367 | 0.438 | 0.688 | 0.647 | 0.638 | 0.630 | std 5.06 / n=358632 |
| vx | 0.720 | 0.997 | 0.438 | 0.049 | 0.173 | 0.149 | 0.450 | 0.507 | 0.628 | 0.641 | 0.625 | std 7.22 / n=358632 |
| vy | 0.660 | 0.997 | 0.163 | 0.023 | 0.124 | 0.143 | 0.465 | 0.525 | 0.547 | 0.503 | 0.513 | std 12.96 / n=358632 |
| foe_vx | 0.486 | 0.996 | 0.033 | 0.049 | 0.167 | 0.142 | 0.448 | 0.496 | 0.354 | 0.319 | 0.322 | std 7.22 / n=358632 |
| foe_vy | 0.513 | 0.996 | 0.009 | 0.023 | 0.088 | 0.091 | 0.463 | 0.515 | 0.434 | 0.387 | 0.392 | std 12.95 / n=358632 |
| foe_dash_cd | 0.869 | 0.998 | 0.040 | 0.063 | 0.860 | 0.183 | 0.203 | 0.865 | 0.715 | 0.650 | 0.643 | std 1.33 / n=358632 |
| foe_bow_cd | 0.847 | 0.998 | 0.056 | 0.045 | 0.841 | 0.490 | 0.227 | 0.846 | 0.819 | 0.803 | 0.800 | std 0.93 / n=358632 |
| foe_platform_cd | 0.922 | 0.998 | 0.114 | 0.418 | 0.912 | 0.494 | 0.332 | 0.916 | 0.760 | 0.713 | 0.701 | std 2.23 / n=358632 |
| idle_dx10 | 0.460 | 0.831 | 0.273 | 0.023 | 0.109 | 0.094 | 0.172 | 0.236 | 0.333 | 0.331 | 0.329 | std 1.24 / n=358632 |
| idle_dy10 | 0.634 | 0.907 | 0.200 | 0.059 | 0.141 | 0.148 | 0.320 | 0.401 | 0.436 | 0.415 | 0.419 | std 2.45 / n=358632 |
| idle_dx30 | 0.806 | 0.916 | 0.401 | 0.108 | 0.403 | 0.420 | 0.244 | 0.527 | 0.468 | 0.474 | 0.464 | std 2.57 / n=358632 |
| idle_dy30 | 0.779 | 0.896 | 0.394 | 0.256 | 0.389 | 0.409 | 0.547 | 0.670 | 0.652 | 0.649 | 0.644 | std 6.04 / n=358632 |
| idle_opp_dx10 | 0.222 | 0.453 | 0.018 | 0.022 | 0.090 | 0.089 | 0.163 | 0.230 | 0.155 | 0.152 | 0.150 | std 1.24 / n=358632 |
| idle_opp_dy10 | 0.383 | 0.812 | 0.012 | 0.060 | 0.118 | 0.121 | 0.304 | 0.391 | 0.331 | 0.308 | 0.308 | std 2.45 / n=358632 |
| idle_opp_dx30 | 0.642 | 0.757 | 0.049 | 0.114 | 0.092 | 0.395 | 0.116 | 0.524 | 0.317 | 0.305 | 0.294 | std 2.57 / n=358632 |
| idle_opp_dy30 | 0.670 | 0.835 | 0.043 | 0.258 | 0.339 | 0.380 | 0.464 | 0.659 | 0.585 | 0.553 | 0.546 | std 6.04 / n=358632 |
| real_dx12 | 0.489 | 0.618 | 0.366 | 0.044 | 0.180 | 0.148 | 0.200 | 0.317 | 0.452 | 0.455 | 0.458 | std 2.40 / n=355602 |
| real_dy12 | 0.438 | 0.615 | 0.166 | 0.050 | 0.135 | 0.121 | 0.215 | 0.330 | 0.362 | 0.354 | 0.357 | std 3.43 / n=355602 |
| real_opp_dx12 | 0.281 | 0.401 | 0.038 | 0.044 | 0.171 | 0.143 | 0.196 | 0.309 | 0.247 | 0.244 | 0.249 | std 2.40 / n=355602 |
| real_opp_dy12 | 0.299 | 0.522 | 0.015 | 0.051 | 0.121 | 0.105 | 0.206 | 0.321 | 0.265 | 0.249 | 0.250 | std 3.43 / n=355602 |
| real_opp_dx32 | 0.302 | 0.394 | 0.051 | 0.064 | 0.203 | 0.183 | 0.176 | 0.334 | 0.287 | 0.287 | 0.295 | std 5.12 / n=350552 |
| real_opp_dy32 | 0.374 | 0.463 | 0.020 | 0.089 | 0.203 | 0.184 | 0.224 | 0.404 | 0.342 | 0.323 | 0.320 | std 6.98 / n=350552 |
| return_disc | 0.113 | 0.119 | 0.068 | 0.017 | 0.067 | 0.031 | 0.024 | 0.088 | 0.108 | 0.125 | 0.130 | std 0.57 / n=358632 |
| outcome_sign | 0.060 | 0.060 | 0.041 | 0.001 | 0.033 | 0.009 | 0.002 | 0.047 | 0.055 | 0.060 | 0.060 | std 0.98 / n=358632 |
| log_ticks_to_end | 0.112 | 0.131 | 0.084 | 0.046 | 0.079 | 0.069 | 0.060 | 0.102 | 0.128 | 0.134 | 0.137 | std 1.35 / n=358632 |
| abs_x | 0.986 | 0.988 | 0.988 | 0.436 | 0.473 | 0.977 | 0.509 | 0.978 | 0.958 | 0.938 | 0.932 | std 22.80 / n=358632 |
| abs_y | 0.986 | 0.994 | 0.975 | 0.415 | 0.534 | 0.962 | 0.476 | 0.972 | 0.960 | 0.946 | 0.942 | std 16.02 / n=358632 |

**Feature families: binary labels** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | rel | raw+rel | rays16 | rays32 | rays32x3 | raw+rays16 | raw+rays32 | raw+rays32x3 | raw+rel+rays32 |
|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.945 | 0.902 | 0.961 | 0.813 | 0.854 | 0.882 | 0.957 | 0.961 | 0.961 | 0.972 |
| hit_30 | 0.943 | 0.866 | 0.952 | 0.803 | 0.847 | 0.873 | 0.957 | 0.961 | 0.958 | 0.966 |
| hit_60 | 0.932 | 0.833 | 0.938 | 0.771 | 0.811 | 0.841 | 0.949 | 0.954 | 0.954 | 0.958 |
| hit_inflight_30 | 0.967 | 0.871 | 0.971 | 0.843 | 0.909 | 0.925 | 0.977 | 0.981 | 0.976 | 0.980 |
| hit_inflight_60 | 0.965 | 0.839 | 0.967 | 0.803 | 0.858 | 0.883 | 0.974 | 0.977 | 0.974 | 0.977 |
| my_shot_hits_60 | 0.967 | 0.881 | 0.968 | 0.848 | 0.887 | 0.898 | 0.972 | 0.972 | 0.969 | 0.973 |
| fall_30 | 0.954 | 0.945 | 0.964 | 0.982 | 0.989 | 0.987 | 0.986 | 0.987 | 0.988 | 0.989 |
| fall_60 | 0.951 | 0.930 | 0.961 | 0.974 | 0.981 | 0.989 | 0.984 | 0.988 | 0.990 | 0.988 |
| squish_60 | 0.934 | 0.913 | 0.945 | 0.904 | 0.906 | 0.934 | 0.955 | 0.955 | 0.959 | 0.959 |
| any_death_60 | 0.868 | 0.794 | 0.884 | 0.765 | 0.799 | 0.836 | 0.913 | 0.919 | 0.924 | 0.926 |
| lands_dry | 0.928 | 0.888 | 0.941 | 0.979 | 0.982 | 0.986 | 0.987 | 0.988 | 0.989 | 0.989 |
| terrain_within_dash | 0.975 | 0.945 | 0.983 | 0.991 | 0.993 | 0.996 | 0.995 | 0.996 | 0.997 | 0.997 |
| grounded | 0.980 | 0.910 | 0.983 | 0.957 | 0.960 | 0.963 | 0.998 | 0.998 | 0.997 | 0.998 |
| foe_dash_ready | 0.996 | 0.697 | 0.996 | 0.632 | 0.637 | 0.663 | 0.996 | 0.996 | 0.996 | 0.996 |
| foe_bow_ready | 0.992 | 0.709 | 0.992 | 0.626 | 0.643 | 0.684 | 0.994 | 0.994 | 0.995 | 0.994 |
| foe_platform_ready | 0.997 | 0.720 | 0.997 | 0.616 | 0.623 | 0.658 | 0.997 | 0.997 | 0.997 | 0.997 |
| real_death_60 | 0.834 | 0.788 | 0.850 | 0.716 | 0.746 | 0.778 | 0.863 | 0.867 | 0.868 | 0.872 |
| win | 0.641 | 0.563 | 0.645 | 0.549 | 0.555 | 0.564 | 0.645 | 0.644 | 0.645 | 0.647 |
| own_arrow_in_flight | 0.994 | 0.927 | 0.994 | 0.790 | 0.839 | 0.863 | 0.997 | 0.998 | 0.998 | 0.998 |
| opp_alive | 0.881 | 0.751 | 0.884 | 0.745 | 0.767 | 0.829 | 0.889 | 0.893 | 0.897 | 0.904 |

**Feature families: regression labels** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | rel | raw+rel | rays16 | rays32 | rays32x3 | raw+rays16 | raw+rays32 | raw+rays32x3 | raw+rel+rays32 |
|---|---|---|---|---|---|---|---|---|---|---|
| drop | 0.562 | 0.485 | 0.605 | 0.938 | 0.940 | 0.951 | 0.941 | 0.942 | 0.951 | 0.943 |
| edge_distance | 0.579 | 0.349 | 0.607 | 0.588 | 0.640 | 0.786 | 0.766 | 0.783 | 0.826 | 0.792 |
| support | 0.656 | 0.455 | 0.683 | 0.780 | 0.805 | 0.851 | 0.843 | 0.854 | 0.869 | 0.857 |
| terrain_min_ray | 0.813 | 0.713 | 0.853 | 0.891 | 0.904 | 0.928 | 0.935 | 0.941 | 0.939 | 0.948 |
| vx | 0.720 | 0.493 | 0.740 | 0.930 | 0.940 | 0.935 | 0.961 | 0.966 | 0.961 | 0.967 |
| vy | 0.660 | 0.534 | 0.700 | 0.983 | 0.985 | 0.986 | 0.989 | 0.990 | 0.991 | 0.992 |
| foe_vx | 0.486 | 0.445 | 0.491 | 0.083 | 0.134 | 0.165 | 0.504 | 0.513 | 0.519 | 0.518 |
| foe_vy | 0.513 | 0.432 | 0.518 | 0.075 | 0.124 | 0.177 | 0.530 | 0.541 | 0.552 | 0.549 |
| foe_dash_cd | 0.869 | 0.110 | 0.869 | 0.026 | 0.034 | 0.047 | 0.871 | 0.873 | 0.871 | 0.873 |
| foe_bow_cd | 0.847 | 0.166 | 0.848 | 0.071 | 0.108 | 0.150 | 0.854 | 0.859 | 0.861 | 0.862 |
| foe_platform_cd | 0.922 | 0.178 | 0.921 | 0.047 | 0.059 | 0.107 | 0.923 | 0.923 | 0.923 | 0.923 |
| idle_dx10 | 0.460 | 0.219 | 0.485 | 0.267 | 0.272 | 0.283 | 0.527 | 0.527 | 0.509 | 0.543 |
| idle_dy10 | 0.634 | 0.418 | 0.677 | 0.592 | 0.597 | 0.607 | 0.816 | 0.822 | 0.805 | 0.830 |
| idle_dx30 | 0.806 | 0.311 | 0.820 | 0.112 | 0.114 | 0.289 | 0.845 | 0.848 | 0.839 | 0.851 |
| idle_dy30 | 0.779 | 0.674 | 0.815 | 0.741 | 0.748 | 0.775 | 0.913 | 0.917 | 0.908 | 0.919 |
| idle_opp_dx10 | 0.222 | 0.157 | 0.226 | 0.028 | 0.044 | 0.053 | 0.228 | 0.229 | 0.223 | 0.232 |
| idle_opp_dy10 | 0.383 | 0.277 | 0.387 | 0.044 | 0.077 | 0.101 | 0.393 | 0.400 | 0.404 | 0.406 |
| idle_opp_dx30 | 0.642 | 0.111 | 0.636 | 0.012 | 0.018 | 0.063 | 0.649 | 0.649 | 0.622 | 0.642 |
| idle_opp_dy30 | 0.670 | 0.386 | 0.678 | 0.050 | 0.088 | 0.122 | 0.677 | 0.678 | 0.672 | 0.688 |
| real_dx12 | 0.489 | 0.231 | 0.500 | 0.269 | 0.284 | 0.301 | 0.531 | 0.534 | 0.524 | 0.539 |
| real_dy12 | 0.438 | 0.295 | 0.470 | 0.387 | 0.393 | 0.402 | 0.569 | 0.570 | 0.561 | 0.573 |
| real_opp_dx12 | 0.281 | 0.183 | 0.292 | 0.044 | 0.064 | 0.082 | 0.291 | 0.291 | 0.286 | 0.300 |
| real_opp_dy12 | 0.299 | 0.192 | 0.305 | 0.031 | 0.050 | 0.063 | 0.304 | 0.304 | 0.301 | 0.311 |
| real_opp_dx32 | 0.302 | 0.172 | 0.315 | 0.048 | 0.067 | 0.089 | 0.314 | 0.316 | 0.308 | 0.326 |
| real_opp_dy32 | 0.374 | 0.228 | 0.383 | 0.032 | 0.056 | 0.064 | 0.380 | 0.380 | 0.375 | 0.387 |
| return_disc | 0.113 | 0.029 | 0.115 | 0.016 | 0.023 | 0.025 | 0.117 | 0.118 | 0.112 | 0.117 |
| outcome_sign | 0.060 | 0.003 | 0.062 | 0.002 | 0.001 | 0.000 | 0.058 | 0.059 | 0.055 | 0.059 |
| log_ticks_to_end | 0.112 | 0.060 | 0.119 | 0.045 | 0.053 | 0.061 | 0.121 | 0.124 | 0.123 | 0.127 |
| abs_x | 0.986 | 0.969 | 0.987 | 0.451 | 0.490 | 0.692 | 0.992 | 0.991 | 0.990 | 0.992 |
| abs_y | 0.986 | 0.948 | 0.986 | 0.814 | 0.832 | 0.997 | 0.994 | 0.994 | 0.998 | 0.994 |

**Absolute position ablation: binary labels** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | raw_nopos | pos | raw_nopos+rays32 | raw_nopos+rel | fused | gru_state |
|---|---|---|---|---|---|---|---|
| hit_15 | 0.945 | 0.945 | 0.623 | 0.961 | 0.963 | 0.952 | 0.968 |
| hit_30 | 0.943 | 0.943 | 0.624 | 0.962 | 0.955 | 0.957 | 0.966 |
| hit_60 | 0.932 | 0.932 | 0.607 | 0.955 | 0.942 | 0.948 | 0.955 |
| hit_inflight_30 | 0.967 | 0.967 | 0.605 | 0.981 | 0.973 | 0.971 | 0.975 |
| hit_inflight_60 | 0.965 | 0.965 | 0.584 | 0.977 | 0.969 | 0.969 | 0.972 |
| my_shot_hits_60 | 0.967 | 0.967 | 0.665 | 0.972 | 0.970 | 0.961 | 0.964 |
| fall_30 | 0.954 | 0.955 | 0.548 | 0.987 | 0.973 | 0.964 | 0.951 |
| fall_60 | 0.951 | 0.951 | 0.572 | 0.988 | 0.964 | 0.942 | 0.933 |
| squish_60 | 0.934 | 0.934 | 0.579 | 0.955 | 0.948 | 0.919 | 0.922 |
| any_death_60 | 0.868 | 0.867 | 0.634 | 0.919 | 0.889 | 0.875 | 0.875 |
| lands_dry | 0.928 | 0.928 | 0.751 | 0.988 | 0.946 | 0.901 | 0.901 |
| terrain_within_dash | 0.975 | 0.975 | 0.768 | 0.996 | 0.986 | 0.950 | 0.950 |
| grounded | 0.980 | 0.980 | 0.630 | 0.998 | 0.986 | 0.962 | 0.958 |
| foe_dash_ready | 0.996 | 0.996 | 0.572 | 0.996 | 0.996 | 0.982 | 0.975 |
| foe_bow_ready | 0.992 | 0.992 | 0.559 | 0.994 | 0.993 | 0.984 | 0.982 |
| foe_platform_ready | 0.997 | 0.997 | 0.504 | 0.997 | 0.997 | 0.958 | 0.945 |
| real_death_60 | 0.834 | 0.834 | 0.584 | 0.867 | 0.853 | 0.860 | 0.864 |
| win | 0.641 | 0.642 | 0.519 | 0.645 | 0.644 | 0.629 | 0.637 |
| own_arrow_in_flight | 0.994 | 0.994 | 0.578 | 0.998 | 0.995 | 0.990 | 0.988 |
| opp_alive | 0.881 | 0.882 | 0.594 | 0.892 | 0.896 | 0.867 | 0.854 |
| map | 0.999 | 0.999 | 0.418 | 0.999 | 0.999 | 0.737 | 0.702 |

**Absolute position ablation: regression labels** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | raw | raw_nopos | pos | raw_nopos+rays32 | raw_nopos+rel | fused | gru_state |
|---|---|---|---|---|---|---|---|
| drop | 0.562 | 0.562 | 0.114 | 0.942 | 0.614 | 0.508 | 0.491 |
| edge_distance | 0.579 | 0.578 | 0.077 | 0.783 | 0.645 | 0.261 | 0.245 |
| support | 0.656 | 0.655 | 0.026 | 0.854 | 0.695 | 0.504 | 0.478 |
| terrain_min_ray | 0.813 | 0.814 | 0.091 | 0.941 | 0.865 | 0.647 | 0.638 |
| vx | 0.720 | 0.719 | 0.016 | 0.965 | 0.746 | 0.628 | 0.641 |
| vy | 0.660 | 0.656 | 0.031 | 0.990 | 0.718 | 0.547 | 0.503 |
| foe_vx | 0.486 | 0.485 | 0.011 | 0.513 | 0.496 | 0.354 | 0.319 |
| foe_vy | 0.513 | 0.512 | 0.001 | 0.542 | 0.521 | 0.434 | 0.387 |
| foe_dash_cd | 0.869 | 0.869 | 0.008 | 0.873 | 0.870 | 0.715 | 0.650 |
| foe_bow_cd | 0.847 | 0.846 | 0.003 | 0.859 | 0.849 | 0.819 | 0.803 |
| foe_platform_cd | 0.922 | 0.922 | 0.006 | 0.923 | 0.923 | 0.760 | 0.713 |
| idle_dx10 | 0.460 | 0.460 | 0.004 | 0.529 | 0.495 | 0.333 | 0.331 |
| idle_dy10 | 0.634 | 0.633 | 0.033 | 0.821 | 0.696 | 0.436 | 0.415 |
| idle_dx30 | 0.806 | 0.806 | 0.013 | 0.848 | 0.828 | 0.468 | 0.474 |
| idle_dy30 | 0.779 | 0.778 | 0.067 | 0.917 | 0.827 | 0.652 | 0.649 |
| idle_opp_dx10 | 0.222 | 0.222 | 0.005 | 0.229 | 0.229 | 0.155 | 0.152 |
| idle_opp_dy10 | 0.383 | 0.382 | 0.000 | 0.400 | 0.388 | 0.331 | 0.308 |
| idle_opp_dx30 | 0.642 | 0.643 | 0.007 | 0.649 | 0.652 | 0.317 | 0.305 |
| idle_opp_dy30 | 0.670 | 0.669 | 0.001 | 0.678 | 0.682 | 0.585 | 0.553 |
| real_dx12 | 0.489 | 0.490 | 0.017 | 0.534 | 0.508 | 0.452 | 0.455 |
| real_dy12 | 0.438 | 0.439 | 0.038 | 0.569 | 0.494 | 0.362 | 0.354 |
| real_opp_dx12 | 0.281 | 0.282 | 0.010 | 0.291 | 0.299 | 0.247 | 0.244 |
| real_opp_dy12 | 0.299 | 0.298 | 0.001 | 0.304 | 0.309 | 0.265 | 0.249 |
| real_opp_dx32 | 0.302 | 0.303 | 0.015 | 0.316 | 0.323 | 0.287 | 0.287 |
| real_opp_dy32 | 0.374 | 0.373 | 0.002 | 0.379 | 0.386 | 0.342 | 0.323 |
| return_disc | 0.113 | 0.114 | 0.003 | 0.117 | 0.113 | 0.108 | 0.125 |
| outcome_sign | 0.060 | 0.060 | -0.007 | 0.059 | 0.057 | 0.055 | 0.060 |
| log_ticks_to_end | 0.112 | 0.111 | 0.011 | 0.124 | 0.119 | 0.128 | 0.134 |
| abs_x | 0.986 | 0.982 | 0.985 | 0.986 | 0.986 | 0.958 | 0.938 |
| abs_y | 0.986 | 0.982 | 0.972 | 0.992 | 0.985 | 0.960 | 0.946 |

| input shifted | maps | KL @-0.3 | KL @-0.2 | KL @-0.1 | KL @+0.1 | KL @+0.2 | KL @+0.3 | dV @+-0.1 | dV @+-0.3 |
|---|---|---|---|---|---|---|---|---|---|
| position | maps012 | 1.3245 | 0.6470 | 0.1766 | 0.2011 | 0.8442 | 1.9676 | 0.0168 | 0.0494 |
| opp_relpos | maps012 | 5.5870 | 3.2253 | 1.0481 | 0.9927 | 2.7127 | 4.2404 | 0.0319 | 0.0860 |
| time_left | maps012 | 0.0081 | 0.0036 | 0.0009 | 0.0007 | 0.0023 | 0.0042 | 0.0064 | 0.0178 |
| null_count | maps012 | 0.0034 | 0.0015 | 0.0004 | 0.0004 | 0.0015 | 0.0034 | 0.0013 | 0.0039 |
| position | maps34 | 1.6383 | 0.8101 | 0.2229 | 0.2713 | 1.1308 | 2.5448 | 0.0187 | 0.0533 |
| opp_relpos | maps34 | 6.6503 | 3.8484 | 1.2853 | 1.1415 | 2.9822 | 4.4696 | 0.0330 | 0.0863 |
| time_left | maps34 | 0.0087 | 0.0038 | 0.0009 | 0.0007 | 0.0024 | 0.0043 | 0.0065 | 0.0180 |
| null_count | maps34 | 0.0036 | 0.0016 | 0.0004 | 0.0004 | 0.0015 | 0.0034 | 0.0013 | 0.0040 |

**Encoder features: 024 vs 024-nopos (binary)** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | flat@024 | flat@024-nopos | fused@024 | fused@024-nopos | gru_out@024 | gru_out@024-nopos | gru_state@024 | gru_state@024-nopos | pooled@024 | pooled@024-nopos |
|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.768 | 0.766 | 0.952 | 0.965 | 0.969 | 0.971 | 0.968 | 0.972 | 0.951 | 0.968 |
| hit_30 | 0.763 | 0.764 | 0.957 | 0.965 | 0.967 | 0.969 | 0.966 | 0.970 | 0.958 | 0.966 |
| hit_60 | 0.735 | 0.734 | 0.948 | 0.954 | 0.954 | 0.957 | 0.955 | 0.959 | 0.950 | 0.956 |
| hit_inflight_30 | 0.771 | 0.778 | 0.971 | 0.977 | 0.977 | 0.979 | 0.975 | 0.979 | 0.977 | 0.981 |
| hit_inflight_60 | 0.743 | 0.753 | 0.969 | 0.974 | 0.973 | 0.976 | 0.972 | 0.976 | 0.974 | 0.977 |
| my_shot_hits_60 | 0.796 | 0.799 | 0.961 | 0.966 | 0.963 | 0.967 | 0.964 | 0.969 | 0.969 | 0.972 |
| fall_30 | 0.900 | 0.897 | 0.964 | 0.954 | 0.958 | 0.941 | 0.951 | 0.939 | 0.953 | 0.956 |
| fall_60 | 0.847 | 0.848 | 0.942 | 0.937 | 0.940 | 0.931 | 0.933 | 0.929 | 0.936 | 0.945 |
| squish_60 | 0.904 | 0.901 | 0.919 | 0.922 | 0.926 | 0.929 | 0.922 | 0.926 | 0.923 | 0.931 |
| any_death_60 | 0.727 | 0.726 | 0.875 | 0.876 | 0.881 | 0.879 | 0.875 | 0.876 | 0.862 | 0.871 |
| lands_dry | 0.870 | 0.838 | 0.901 | 0.907 | 0.900 | 0.905 | 0.901 | 0.903 | 0.900 | 0.907 |
| terrain_within_dash | 0.858 | 0.827 | 0.950 | 0.955 | 0.949 | 0.953 | 0.950 | 0.955 | 0.965 | 0.968 |
| grounded | 0.872 | 0.862 | 0.962 | 0.965 | 0.958 | 0.963 | 0.958 | 0.964 | 0.941 | 0.943 |
| foe_dash_ready | 0.681 | 0.681 | 0.982 | 0.985 | 0.969 | 0.973 | 0.975 | 0.978 | 0.997 | 0.997 |
| foe_bow_ready | 0.637 | 0.636 | 0.984 | 0.987 | 0.979 | 0.983 | 0.982 | 0.985 | 0.993 | 0.994 |
| foe_platform_ready | 0.708 | 0.725 | 0.958 | 0.968 | 0.939 | 0.948 | 0.945 | 0.952 | 0.997 | 0.998 |
| real_death_60 | 0.699 | 0.706 | 0.860 | 0.860 | 0.864 | 0.863 | 0.864 | 0.865 | 0.849 | 0.853 |
| win | 0.616 | 0.619 | 0.629 | 0.630 | 0.632 | 0.637 | 0.637 | 0.636 | 0.621 | 0.619 |
| own_arrow_in_flight | 0.789 | 0.852 | 0.990 | 0.991 | 0.986 | 0.989 | 0.988 | 0.990 | 0.991 | 0.992 |
| opp_alive | 0.589 | 0.670 | 0.867 | 0.878 | 0.850 | 0.880 | 0.854 | 0.876 | 0.873 | 0.900 |
| map | 0.711 | 0.742 | 0.737 | 0.727 | 0.688 | 0.695 | 0.702 | 0.706 | 0.981 | 0.993 |

**Encoder features: 024 vs 024-nopos (regression)** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | flat@024 | flat@024-nopos | fused@024 | fused@024-nopos | gru_out@024 | gru_out@024-nopos | gru_state@024 | gru_state@024-nopos | pooled@024 | pooled@024-nopos |
|---|---|---|---|---|---|---|---|---|---|---|
| drop | 0.327 | 0.282 | 0.508 | 0.500 | 0.489 | 0.486 | 0.491 | 0.482 | 0.514 | 0.523 |
| edge_distance | 0.166 | 0.144 | 0.261 | 0.276 | 0.243 | 0.253 | 0.245 | 0.264 | 0.332 | 0.356 |
| support | 0.260 | 0.231 | 0.504 | 0.518 | 0.475 | 0.492 | 0.478 | 0.492 | 0.518 | 0.530 |
| terrain_min_ray | 0.282 | 0.258 | 0.647 | 0.655 | 0.630 | 0.647 | 0.638 | 0.652 | 0.688 | 0.706 |
| vx | 0.438 | 0.424 | 0.628 | 0.649 | 0.625 | 0.649 | 0.641 | 0.667 | 0.507 | 0.514 |
| vy | 0.163 | 0.133 | 0.547 | 0.573 | 0.513 | 0.533 | 0.503 | 0.534 | 0.525 | 0.529 |
| foe_vx | 0.033 | 0.024 | 0.354 | 0.384 | 0.322 | 0.345 | 0.319 | 0.338 | 0.496 | 0.503 |
| foe_vy | 0.009 | 0.007 | 0.434 | 0.445 | 0.392 | 0.406 | 0.387 | 0.402 | 0.515 | 0.521 |
| foe_dash_cd | 0.040 | 0.039 | 0.715 | 0.731 | 0.643 | 0.660 | 0.650 | 0.666 | 0.865 | 0.868 |
| foe_bow_cd | 0.056 | 0.074 | 0.819 | 0.826 | 0.800 | 0.811 | 0.803 | 0.813 | 0.846 | 0.849 |
| foe_platform_cd | 0.114 | 0.192 | 0.760 | 0.778 | 0.701 | 0.721 | 0.713 | 0.727 | 0.916 | 0.919 |
| idle_dx10 | 0.273 | 0.267 | 0.333 | 0.344 | 0.329 | 0.345 | 0.331 | 0.347 | 0.236 | 0.240 |
| idle_dy10 | 0.200 | 0.184 | 0.436 | 0.453 | 0.419 | 0.436 | 0.415 | 0.433 | 0.401 | 0.406 |
| idle_dx30 | 0.401 | 0.368 | 0.468 | 0.486 | 0.464 | 0.482 | 0.474 | 0.493 | 0.527 | 0.538 |
| idle_dy30 | 0.394 | 0.373 | 0.652 | 0.658 | 0.644 | 0.648 | 0.649 | 0.654 | 0.670 | 0.675 |
| idle_opp_dx10 | 0.018 | 0.014 | 0.155 | 0.168 | 0.150 | 0.158 | 0.152 | 0.160 | 0.230 | 0.234 |
| idle_opp_dy10 | 0.012 | 0.014 | 0.331 | 0.337 | 0.308 | 0.315 | 0.308 | 0.317 | 0.391 | 0.397 |
| idle_opp_dx30 | 0.049 | 0.036 | 0.317 | 0.335 | 0.294 | 0.303 | 0.305 | 0.321 | 0.524 | 0.535 |
| idle_opp_dy30 | 0.043 | 0.065 | 0.585 | 0.595 | 0.546 | 0.556 | 0.553 | 0.564 | 0.659 | 0.664 |
| real_dx12 | 0.366 | 0.361 | 0.452 | 0.460 | 0.458 | 0.468 | 0.455 | 0.467 | 0.317 | 0.327 |
| real_dy12 | 0.166 | 0.153 | 0.362 | 0.370 | 0.357 | 0.363 | 0.354 | 0.361 | 0.330 | 0.338 |
| real_opp_dx12 | 0.038 | 0.028 | 0.247 | 0.259 | 0.249 | 0.258 | 0.244 | 0.255 | 0.309 | 0.318 |
| real_opp_dy12 | 0.015 | 0.015 | 0.265 | 0.270 | 0.250 | 0.254 | 0.249 | 0.254 | 0.321 | 0.327 |
| real_opp_dx32 | 0.051 | 0.038 | 0.287 | 0.295 | 0.295 | 0.301 | 0.287 | 0.297 | 0.334 | 0.344 |
| real_opp_dy32 | 0.020 | 0.021 | 0.342 | 0.344 | 0.320 | 0.321 | 0.323 | 0.324 | 0.404 | 0.410 |
| return_disc | 0.068 | 0.065 | 0.108 | 0.107 | 0.130 | 0.131 | 0.125 | 0.126 | 0.088 | 0.087 |
| outcome_sign | 0.041 | 0.039 | 0.055 | 0.051 | 0.060 | 0.060 | 0.060 | 0.056 | 0.047 | 0.040 |
| log_ticks_to_end | 0.084 | 0.079 | 0.128 | 0.130 | 0.137 | 0.139 | 0.134 | 0.136 | 0.102 | 0.100 |
| abs_x | 0.988 | 0.861 | 0.958 | 0.905 | 0.932 | 0.865 | 0.938 | 0.880 | 0.978 | 0.982 |
| abs_y | 0.975 | 0.774 | 0.960 | 0.895 | 0.942 | 0.862 | 0.946 | 0.871 | 0.972 | 0.975 |

**Encoder features: 024 vs early (binary)** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | flat@024 | flat@early | fused@024 | fused@early | gru_out@024 | gru_out@early | gru_state@024 | gru_state@early | pooled@024 | pooled@early |
|---|---|---|---|---|---|---|---|---|---|---|
| hit_15 | 0.768 | 0.776 | 0.952 | 0.921 | 0.969 | 0.927 | 0.968 | 0.921 | 0.951 | 0.955 |
| hit_30 | 0.763 | 0.769 | 0.957 | 0.931 | 0.967 | 0.932 | 0.966 | 0.929 | 0.958 | 0.958 |
| hit_60 | 0.735 | 0.739 | 0.948 | 0.924 | 0.954 | 0.925 | 0.955 | 0.921 | 0.950 | 0.948 |
| hit_inflight_30 | 0.771 | 0.771 | 0.971 | 0.949 | 0.977 | 0.946 | 0.975 | 0.939 | 0.977 | 0.977 |
| hit_inflight_60 | 0.743 | 0.745 | 0.969 | 0.955 | 0.973 | 0.952 | 0.972 | 0.946 | 0.974 | 0.974 |
| my_shot_hits_60 | 0.796 | 0.803 | 0.961 | 0.966 | 0.963 | 0.964 | 0.964 | 0.965 | 0.969 | 0.970 |
| fall_30 | 0.900 | 0.900 | 0.964 | 0.955 | 0.958 | 0.960 | 0.951 | 0.955 | 0.953 | 0.958 |
| fall_60 | 0.847 | 0.852 | 0.942 | 0.934 | 0.940 | 0.938 | 0.933 | 0.937 | 0.936 | 0.943 |
| squish_60 | 0.904 | 0.907 | 0.919 | 0.918 | 0.926 | 0.920 | 0.922 | 0.917 | 0.923 | 0.926 |
| any_death_60 | 0.727 | 0.726 | 0.875 | 0.854 | 0.881 | 0.857 | 0.875 | 0.852 | 0.862 | 0.858 |
| lands_dry | 0.870 | 0.866 | 0.901 | 0.901 | 0.900 | 0.900 | 0.901 | 0.897 | 0.900 | 0.900 |
| terrain_within_dash | 0.858 | 0.864 | 0.950 | 0.957 | 0.949 | 0.959 | 0.950 | 0.957 | 0.965 | 0.968 |
| grounded | 0.872 | 0.873 | 0.962 | 0.960 | 0.958 | 0.958 | 0.958 | 0.954 | 0.941 | 0.942 |
| foe_dash_ready | 0.681 | 0.683 | 0.982 | 0.984 | 0.969 | 0.975 | 0.975 | 0.970 | 0.997 | 0.997 |
| foe_bow_ready | 0.637 | 0.635 | 0.984 | 0.975 | 0.979 | 0.957 | 0.982 | 0.960 | 0.993 | 0.994 |
| foe_platform_ready | 0.708 | 0.719 | 0.958 | 0.991 | 0.939 | 0.988 | 0.945 | 0.988 | 0.997 | 0.998 |
| real_death_60 | 0.699 | 0.701 | 0.860 | 0.836 | 0.864 | 0.840 | 0.864 | 0.835 | 0.849 | 0.842 |
| win | 0.616 | 0.617 | 0.629 | 0.616 | 0.632 | 0.608 | 0.637 | 0.611 | 0.621 | 0.616 |
| own_arrow_in_flight | 0.789 | 0.803 | 0.990 | 0.991 | 0.986 | 0.991 | 0.988 | 0.991 | 0.991 | 0.991 |
| opp_alive | 0.589 | 0.641 | 0.867 | 0.875 | 0.850 | 0.876 | 0.854 | 0.858 | 0.873 | 0.891 |
| map | 0.711 | 0.717 | 0.737 | 0.810 | 0.688 | 0.730 | 0.702 | 0.732 | 0.981 | 0.972 |

**Encoder features: 024 vs early (regression)** (mlp probe; binary = test AUC, regression = test R^2, map = accuracy)

| label | flat@024 | flat@early | fused@024 | fused@early | gru_out@024 | gru_out@early | gru_state@024 | gru_state@early | pooled@024 | pooled@early |
|---|---|---|---|---|---|---|---|---|---|---|
| drop | 0.327 | 0.320 | 0.508 | 0.496 | 0.489 | 0.482 | 0.491 | 0.471 | 0.514 | 0.522 |
| edge_distance | 0.166 | 0.154 | 0.261 | 0.270 | 0.243 | 0.245 | 0.245 | 0.236 | 0.332 | 0.340 |
| support | 0.260 | 0.247 | 0.504 | 0.479 | 0.475 | 0.466 | 0.478 | 0.444 | 0.518 | 0.507 |
| terrain_min_ray | 0.282 | 0.288 | 0.647 | 0.667 | 0.630 | 0.666 | 0.638 | 0.662 | 0.688 | 0.705 |
| vx | 0.438 | 0.440 | 0.628 | 0.608 | 0.625 | 0.663 | 0.641 | 0.658 | 0.507 | 0.493 |
| vy | 0.163 | 0.167 | 0.547 | 0.520 | 0.513 | 0.518 | 0.503 | 0.503 | 0.525 | 0.524 |
| foe_vx | 0.033 | 0.033 | 0.354 | 0.315 | 0.322 | 0.301 | 0.319 | 0.267 | 0.496 | 0.492 |
| foe_vy | 0.009 | 0.007 | 0.434 | 0.425 | 0.392 | 0.408 | 0.387 | 0.397 | 0.515 | 0.517 |
| foe_dash_cd | 0.040 | 0.040 | 0.715 | 0.787 | 0.643 | 0.730 | 0.650 | 0.699 | 0.865 | 0.870 |
| foe_bow_cd | 0.056 | 0.057 | 0.819 | 0.753 | 0.800 | 0.706 | 0.803 | 0.695 | 0.846 | 0.845 |
| foe_platform_cd | 0.114 | 0.123 | 0.760 | 0.847 | 0.701 | 0.823 | 0.713 | 0.812 | 0.916 | 0.921 |
| idle_dx10 | 0.273 | 0.272 | 0.333 | 0.333 | 0.329 | 0.341 | 0.331 | 0.334 | 0.236 | 0.229 |
| idle_dy10 | 0.200 | 0.205 | 0.436 | 0.430 | 0.419 | 0.435 | 0.415 | 0.411 | 0.401 | 0.398 |
| idle_dx30 | 0.401 | 0.403 | 0.468 | 0.492 | 0.464 | 0.489 | 0.474 | 0.482 | 0.527 | 0.543 |
| idle_dy30 | 0.394 | 0.395 | 0.652 | 0.645 | 0.644 | 0.650 | 0.649 | 0.645 | 0.670 | 0.665 |
| idle_opp_dx10 | 0.018 | 0.018 | 0.155 | 0.143 | 0.150 | 0.140 | 0.152 | 0.129 | 0.230 | 0.227 |
| idle_opp_dy10 | 0.012 | 0.010 | 0.331 | 0.331 | 0.308 | 0.319 | 0.308 | 0.315 | 0.391 | 0.392 |
| idle_opp_dx30 | 0.049 | 0.047 | 0.317 | 0.322 | 0.294 | 0.289 | 0.305 | 0.271 | 0.524 | 0.532 |
| idle_opp_dy30 | 0.043 | 0.040 | 0.585 | 0.603 | 0.546 | 0.584 | 0.553 | 0.573 | 0.659 | 0.662 |
| real_dx12 | 0.366 | 0.364 | 0.452 | 0.442 | 0.458 | 0.445 | 0.455 | 0.440 | 0.317 | 0.309 |
| real_dy12 | 0.166 | 0.167 | 0.362 | 0.349 | 0.357 | 0.349 | 0.354 | 0.334 | 0.330 | 0.327 |
| real_opp_dx12 | 0.038 | 0.038 | 0.247 | 0.233 | 0.249 | 0.226 | 0.244 | 0.216 | 0.309 | 0.306 |
| real_opp_dy12 | 0.015 | 0.013 | 0.265 | 0.267 | 0.250 | 0.254 | 0.249 | 0.247 | 0.321 | 0.321 |
| real_opp_dx32 | 0.051 | 0.052 | 0.287 | 0.274 | 0.295 | 0.265 | 0.287 | 0.256 | 0.334 | 0.331 |
| real_opp_dy32 | 0.020 | 0.018 | 0.342 | 0.361 | 0.320 | 0.344 | 0.323 | 0.331 | 0.404 | 0.404 |
| return_disc | 0.068 | 0.059 | 0.108 | 0.088 | 0.130 | 0.084 | 0.125 | 0.081 | 0.088 | 0.084 |
| outcome_sign | 0.041 | 0.036 | 0.055 | 0.039 | 0.060 | 0.028 | 0.060 | 0.030 | 0.047 | 0.037 |
| log_ticks_to_end | 0.084 | 0.080 | 0.128 | 0.107 | 0.137 | 0.104 | 0.134 | 0.101 | 0.102 | 0.098 |
| abs_x | 0.988 | 0.988 | 0.958 | 0.975 | 0.932 | 0.968 | 0.938 | 0.966 | 0.978 | 0.983 |
| abs_y | 0.975 | 0.975 | 0.960 | 0.982 | 0.942 | 0.979 | 0.946 | 0.979 | 0.972 | 0.975 |



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
