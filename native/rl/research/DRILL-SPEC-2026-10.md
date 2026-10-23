# Drill showcase specialist: mechanics, data, observations, reward, plan (2026-10-23)

Written 2026-10-23 by the drill-spec subagent. CPU only. Goal (Matthew): a Drill + Dash + Dash bot that
is "ridiculously good with drill", rewarded for longer drills so it learns crazy movement. This doc is
the evidence and a pre-registered plan. Nothing here has trained.

Updated 2026-10-23 with Matthew's decisions: v4.3 is built and verified (Section 4.1), the style is
final and stays on for the whole run (Section 5), all three arms are queued with P1 as the primary
metric (Section 6), and Section 7 records the answers.

Artifacts: `RL = native/artifacts/rl`, `DS = RL/research/drill-spec/` (look/<condition>/ games.jsonl,
decisions.jsonl, rows.npz, frames-g*.json.gz; renders/*.png; probe/probe.{txt,json}; analysis.json).
Scripts: `native/rl/research/drill-spec/` (copies of `DS/scripts/`).

## 0. Summary

- **Legal.** Drill/Dash/Dash is a legal game kit: no select screen dedups slots (Section 1.2).
- **No current net burrows.** Every net "drills" by jumping, casting at the apex, and diving
  0.3-0.6 s through the air onto the opponent. me31 (the drill exploiter) does this with 84-100% of
  its kills. Underground time is incidental. Drill episodes with 1 s or more of continuous
  underground travel are 0.1-3.6% of episodes, and kills after 0.5 s or more underground are
  1.5-13% of drill kills (Section 2).
- **The observation is nearly blind underground** (Section 3). The controlling slime record
  freezes during a form, so the self body block's velocity, speed and rotation are stale (rotation
  is exactly 0 in 97% of me31's underground rows). Once the delayed position is inside the wall,
  every ray is distance 0, class wall. There is no speed, no "underground" bit and no ground-ahead
  signal. The heading is only in the own form entity row (an up vector scaled 1/64).
- Held-out probes on 10.5k underground rows recover the opponent offset (R^2 0.93) and part of the
  heading (0.70 cos, 0.47 sin). They do not recover the ground left along the heading: 0.07 from
  the row, 0.11 even given the true heading. Memory of the last above-ground row adds nothing.
  Under v4.3 (as built, Section 4.1) the ray along the heading reads the ground left within 1 u on
  80-87% of underground rows at the observation delay (v4.2: 4-5%).
- **Recommended obs change: writer revision bit 32, "layout v4.3 drill"**. Same width, same field
  meanings, the v4.2 pattern, no surgery.
  - The sim exports the form body's velocity and an UNDERGROUND flag on the form row.
  - The body/relation blocks read the form body during a form, for self and for the opponent.
  - Rays cast from inside a wall report the distance to where the ray leaves the ground.
- **Reward: style `drill_burrow`**, on for the whole run (no fade; Matthew 2026-10-23).
  - Underground net approach to the opponent, paid as a difference so oscillation pays nothing.
  - A drill-kill premium scaled by underground time in that dive.
  - A penalty for unforced drill deaths.
  - No per-tick underground pay (it is farmable: circling in a thick platform is possible below
    ~15 u/s). Zero in sudden death.
- **Plan (as decided 2026-10-23):** arms C (no style), S (style, v4.2) and SO (style, v4.3), 5 runs
  of 3.5B (~6-10 GPU-h), ahead of research-recipe's remaining steps. Primary metric P1: the
  specialist on DDD vs l2-main-136002m on random kits.

## 1. Mechanics (decompiled game: Drill.cs, Drill.prefab; Rust sim native/src/rollout/mod.rs drill_tick)

### 1.1 How Drill works

| Property | Value | Source |
|---|---|---|
| Cooldown | 4.0 s, effectively from exit: me31's exit-to-next-cast minimum is 4.07 s. The slime is inactive while in the form. | Drill.prefab; drill_look |
| Enter animation | 6 frames x 3 ticks (0.3 s) before `isDrilling` and the hitboxes go on | Drill.cs |
| Air | gravity 45, strong gravity 102 for the first 0.4 s (max early fall 45 u/s); air turn 7 rad/s (~400 deg/s) | Drill.prefab |
| Entering ground | needs the platform normal . body.up > 0 (nose into the surface), else the ability ends at once. Speed is floored at 10, then x0.66 (min 6.6 u/s). External velocity is zeroed. | Drill.cs |
| Underground | v = heading*\|v\| + heading*6*dt: 6 u/s^2 along the heading, **no cap**. Moves with the platform's own velocity. | Drill.cs, drill_tick |
| Underground turning | 7 rad/s x SpeedToRotationSpeed01(speed): 1.83 rad/s at 5 u/s, 2.64 at 10, 3.67 at 20, 4.2 at 40. The turn radius v/omega grows with speed: 3.8 u at 10 u/s, 5.4 u at 20, 9.5 u at 40. | Drill.prefab curve |
| Leaving ground | speed x1.5 | Drill.prefab |
| Hop chain | enter x0.66, exit x1.5 = 0.99 per hop, plus 6 u/s^2 underground: speed builds by chaining hops through platforms | derived |
| Stopping | exit only when the button is released **and** the drill is not underground. Underground, the button does nothing, so it cannot stop until it surfaces. No duration limit (the game's achievement is at 10 s). Ropes detach underground. | Drill.cs |
| What it drills | layer 11 "wall" only: platforms and StickyRoundedRectangle. Not lethal terrain, water, or ability objects. | Drill.prefab drillables |
| Kills | 3 HitboxCombo boxes on layer 18 Explosion with a Hitbox. Any non-owner player hurtbox touching them dies, including enemy forms, so drill-vs-drill trades. Box (0.27, 1.29), offset (0, -2). | HitboxCombo.cs, Hitbox.cs, PlayerCollision.cs 330-420 |
| Dies to | anything lethal touching its Player-layer hurtbox (projectiles, explosions, beams, contact, lethal terrain), and DestroyIfOutsideSceneBounds (ring-out and water: `form_bounds_kill`) | PlayerCollision.cs, drill_tick |
| Dash interaction | the cast takes PlayerInfo self and external velocity, so dash momentum carries into the dive. The exit hands the velocity back to the slime, so a dash right after surfacing chains. | Drill.cs OnEnterAbility/ExitAbility |
| Sudden death | platforms sink (AnimateVelocity suddenDeathYSpeed, after a random delay). A driller underground rides the sinking platform. | AnimateVelocity.cs |

The speed curve means a slow driller can circle inside a thick platform. The common rounded
platforms are radius-5.88 capsules, ~11.8 u thick, and a 10 u/s drill turns in a 3.8 u radius.
Underground acceleration (6 u/s^2) widens the circle until it must surface, after roughly 1.5-3 s.
"Time underground" is therefore farmable on big platforms. The reward must not pay it directly
(Section 4).

### 1.2 Legality of Drill + Dash + Dash

No duplicate check exists in CharacterSelectHandler, SelectAbility.Select, AbilitySelect* or
MidGameAbilitySelect; each slot cycles independently. `evaluation_protocol.loadout_ids` documents
"Duplicate abilities are game-legal". The sim's per-slot duplicate bugs were fixed 2026-09-28. The
kit is legal as asked. Two Dashes means two 4 s cooldowns, so one dash is nearly always up for
entry speed and escape.

## 2. How the current nets drill

Protocol (`scripts/drill_look.py`): seat A on the given kit vs l2-main-136002m on iid random kits
(3 draws from pool `all`, i.e. 2..38 minus 12 and 15). 400 games = 200 side-swapped seed pairs, seed
base 4100001, training-111 maps. Every A drill decision is logged with the undelayed SDF ("truly
underground" = SDF < 0 at A's position), heading, ground left along the heading, the opponent, and
the row. All six conditions share seeds, maps and B kits, so differences are paired by setup.

Checkpoints:

- me31 = l2-me31-drill-grenade-2000m
- x2 = x2-drill-1500m (drill_snipe style)
- main = l2-main-136002m
- ann = off1/ff-1 anneal `policy-1333395456.bin` (v4.2)

Kits: DDG = [2,26,3], DDD = [26,2,2].

| | me31 DDG | x2 DDG | main DDG | me31 DDD | main DDD | ann DDD |
|---|---|---|---|---|---|---|
| Score vs main136 (95% CI) | 0.695 [.645,.745] | 0.596 [.545,.647] | 0.496 [.442,.549] | 0.627 [.574,.682] | 0.472 [.419,.527] | 0.463 [.405,.520] |
| Games reaching SD | 14% | 12% | 45% | 13% | 52% | 56% |
| Drill casts/min | 4.19 | 4.74 | 1.55 | 3.91 | 2.03 | 2.05 |
| Episode length, median | 0.60 s | 0.67 s | 0.60 s | 0.67 s | 0.67 s | 0.67 s |
| Episodes touching ground underground | 19% | 45% | 31% | 32% | 41% | 46% |
| Longest underground run, median (episodes that go under) | 0.13 s | 0.13 s | 0.20 s | 0.20 s | 0.27 s | 0.33 s |
| Episodes with >= 1 s continuous underground | 0.1% [0,0.5] | 0.9% [.5,1.6] | 1.1% [.6,2.1] | 0.4% [.2,1.1] | 1.5% [.9,2.3] | 3.6% [2.7,4.8] |
| Longest underground run, max | 2.1 s | 1.5 s | 1.8 s | 1.4 s | 3.1 s | 3.1 s |
| Episodes ending in a drill kill | 19.2% | 14.5% | 6.4% | 21.0% | 7.8% | 6.1% |
| Drill share of A's kills | 0.84 [.78,.88] | 0.85 [.80,.90] | 0.45 [.36,.54] | 1.00 | 1.00 | 0.97 |
| Drill kills after >= 0.5 s underground in that dive | 1.5% [0.5,4.3] (3/203) | 4.4% [2.2,8.4] | 7.7% [3.0,18] | 6.3% [3.7,10.5] | 4.4% [1.7,10.8] | 13% [7.4,23] (10/75) |
| A kills/min | 0.96 | 0.81 | 0.22 | 0.82 | 0.16 | 0.13 |
| Deaths while in drill / A deaths | 29/130 | 45/182 | 46/226 | 17/157 | 74/233 | 83/247 |
| Unforced (ring-out/self) deaths, % of episodes | 2.1% | 1.2% | 4.1% | 1.0% | 5.4% | 4.9% |
| Surfacing within 6 u of the opponent | 23% | 11% | 8% | 19% | 9% | 8% |
| Median closing per episode | 7.5 u | 7.4 u | 3.7 u | 7.5 u | 2.6 u | 1.7 u |

Paired differences, by seed pair (bootstrap):

- me31 DDD - main DDD: +0.155 [+0.089, +0.225]
- me31 DDG - main DDG: +0.199 [+0.131, +0.264]
- x2 DDG - main DDG: +0.100 [+0.031, +0.168]
- ann DDD - main DDD: -0.010 [-0.072, +0.051]
- main DDD - main DDG: -0.024 [-0.090, +0.040]
- me31 DDG - me31 DDD: +0.068 [+0.005, +0.129]

What the renders show (DS/renders):

- **me31: the dive-bomb.** It jumps (often off a dash), casts at the apex, and dives diagonally 6-10 u
  onto an opponent who is coming over a platform edge, about 0.3-0.4 s in the air. Kills happen in
  the air or a fraction into a thin platform (me31-ddg-kill.png, 9 of 9 kills airborne). It never
  plans a route underground.
- **main and ann: incidental burrows.** Their "long" underground episodes (main-ddd-ug.png) are
  dives that clip the edge of a rounded platform and travel 0.7-1.1 s along a shallow chord just
  under the surface. They surface away from the opponent: 8-9% of surfacings are within 6 u, and
  the median closing is 1.7-2.6 u. Most of their drill deaths are unforced ring-outs after
  surfacing (main DDD 57 of 74).
- **Skilled play.** No human drill data was measured; the 2026-09 Mac replays are Matthew vs bots,
  and parsing them for drill is future work. The mechanics suggest what skilled play would look
  like: hop chains (x0.99 per hop, +6 u/s^2 underground) to build speed, routes under the
  opponent's platform, and surfacing under them. None of the nets does this.

Kit and init:

- me31 is the best drill prior: +0.155 over main on DDD, with 1% unforced drill deaths.
- ann and main are indistinguishable on DDD. Both waste drills: 5% of episodes end in an unforced
  death.
- The second Dash costs me31 0.068 vs Grenade.

## 3. What the policy sees while drilling (obs v4, `train/bopl_rows_v4.h`, `bopl_rays_v4.h`)

Checked on decoded rows (`scripts/probe_fields.py`, `DS/probe/probe2.txt`) and on every drill row of
the six conditions. The writer serves a 9-tick-delayed state (registry.DELAY = 9, 0.15 s, 1-3 u of
underground travel at 7-20 u/s).

| Block | Underground content | Status |
|---|---|---|
| Self player row | player x/y (moves with the form) | ok |
| Self body (24) | `bopl4_body(controlling slime)`: vx, vy, speed, sin/cos rotation from the **slime record, frozen at cast**. Rotation is 0 in 97% of me31's underground rows (78% main DDG, 47% main DDD, 33% ann). in_ability = 1. | **stale** |
| Relations (self and opponent) | relative velocity = enemy slime v - self slime v, both stale when either is in a form | **stale** |
| Opponent block | relative position through terrain (no occlusion), 9 ticks old | ok |
| Terrain (12) | drop/ceiling 0 inside a wall. SDF (/8) is negative with a gradient toward the nearest surface, which is not the exit along the heading. | partial |
| Rays (36 x 8) | origin inside a wall, so the march returns t = 0: **all 36 rays are distance 0, class wall** in 45-69% of underground rows. The rest are rows whose delayed position is still outside. The opponent is never seen by a ray underground. | **blind** |
| Entity set | own form row (kind 1, layer 8): up vector /64 (~0.015) = heading. vx = vy = 0 (not exported). Drill hitboxes are layer 18 rows. | weak |
| Slot rows | Drill ACTIVE flag, cooldowns | ok |

Probe (`scripts/probe.py`, `DS/probe/probe.txt`):

- Data: 10,514 truly-underground rows (2,118 episodes, five conditions). Ridge and 1-layer MLP,
  GroupKFold(5) by game, held-out R^2.
- Targets: heading, ground left along the heading (clip 20 u), decisions to surfacing, opponent
  offset.
- Feature sets:
  - row: the 492-d flat block
  - row+mem: plus the flat block of the episode's last above-ground decision, an upper bound on
    what recurrence could add
  - row+hd: plus the true heading
  - hd+sdf: heading, SDF and gradient only

Held-out R^2, the better of ridge and MLP (fold range in probe.txt; "given" = the target is an input):

| Target (sd) | row | row+mem | row+hd | hd+sdf |
|---|---|---|---|---|
| heading cos (0.75) | 0.70 | 0.69 | given | given |
| heading sin (0.58) | 0.47 | 0.37 | given | given |
| ground left along heading (4.4 u) | **0.07** | -0.22 | **0.11** | 0.19 |
| decisions to surfacing (12.7, clip 30) | 0.24 | 0.04 | 0.25 | 0.04 |
| opponent dx (11.4 u) | 0.93 | 0.91 | 0.93 | 0.35 |
| opponent dy (10.0 u) | 0.93 | 0.91 | 0.94 | 0.12 |

The memory set overfits (984 inputs) and is no better than the row.

Reading: the row carries the opponent, carries the heading only partly, and carries the ground
ahead and the time to surface barely or not at all. Memory of the pre-dive view does not recover them. A planner of underground routes
needs the exit distance along candidate headings: how much ground is left, and where it comes out.
Rays from inside the wall give exactly that.

## 4. Observation change: writer revision bit 32, "layout v4.3 drill"

Constraint: keep `V4_FLAT` and every set's width. `train/surgery.py` moves weights between
architectures, not observation layouts. A width change would need a new input-column surgery plus
`widen_league.py` copies of every league member. A writer *revision* (docs/LEAGUE.md, today bit 16
= v4.2) changes what fields hold without changing their shape or meaning. It is masked out of the
member shape check (`registry.WRITER_REVISIONS`), so a v4.3 run inits from v4.2/v4.0 checkpoints and
still loads every league member. Each frozen bank writes its member's own rows.

**Sim (export only; replay and divergence gates must stay byte-identical):**

1. Drill form row: vx/vy = the drill body velocity (DrillState plus the platform velocity it rides),
   not 0. Do the same for other forms whose sim state has a velocity. `sync_form_row` /
   form-row refresh in mod.rs.
2. New entity flag `BOPL_ENTITY_UNDERGROUND` on the drill form row while `DrillState.underground`.
   Also usable by styles (one new state field; docs/STYLES.md design rule).

**Writer, under bit 32 (CPU and GPU paths share `bopl_rows_v4.h`/`bopl_rays_v4.h`; check `.cu` mirrors):**

1. Body block: when the controlling slime is IN_ABILITY and the player has an active own form row,
   vx, vy and speed come from the form row, and sin/cos from its up vector (heading = -up for
   Drill). `body[22]` (still) follows. The same applies to the opponent's body block and to every
   relation velocity. Main then sees an enemy driller's real velocity, which also serves the
   main-bot program: drill was the one real unforced-death cost in the 2026-10-22 early-kill look.
2. Rays from inside: when the delayed origin's SDF is < 0, each ray marches to the first point
   where it **leaves** the ground. dist = that exit distance /40, class wall. The class beyond the
   exit (lethal, water, opponent within the ray's sector) is set if the continued march hits it
   within the remaining length. Inside mode is visible to the net through the same row's terrain
   SDF < 0, so no new column is needed.
3. Nothing else. Time in form is in the slot row, and the opponent offset is already through-terrain.

**Cost:**

- Code: about 1 agent-day. Sim export (~30 lines, plus the .NET backend marked unsupported for the
  flag), writer bit, rays inside mode, `registry.WRITER_REVISIONS`, and the style compiler name
  `UNDERGROUND`.
- Checks:
  - encode_check: CPU == GPU rows with bit 32 on and off, with a negative control.
  - env_bench hash and telemetry_hash unchanged with the bit off.
  - check_v4 --build.
  - check_style_binding.
  - drill_look on a v4.3 build: underground rows must show rays != 0 and body rotation != 0.
- No GPU beyond the runs.

**Test that it is needed before building it (cheap, sequential):**

1. The probe above (done): ground ahead is not decodable today.
2. Arm S in Section 5 runs first on v4.2 obs. If it already burrows with intent, v4.3 is not built.
   Intent means passing P2 and surfacing within 6 u of the opponent in >= 20% of surfacings.
3. A frame ablation on today's nets (zero the rays and terrain on drill frames) would be
   uninformative: no current net burrows, so there is nothing for the ablation to break. Run it on
   arm S's checkpoint instead. If zeroing the underground rays and terrain changes S's underground
   actions by under 0.01 nats KL, S is not using them, which is consistent with blindness.

### 4.1 As built (2026-10-23; Matthew: build it now)

**Sim** (`native/src/rollout/mod.rs` `stamp_forms` / `unstamp_forms`, around `publish_observed`):
with `BOPL_RL_FORM_BODY=1` at reset (`form_body_export`), each live Drill form's published row gets
`BOPL_ENTITY_FORM_BODY` (1<<25), `BOPL_ENTITY_UNDERGROUND` (1<<24) while `DrillState.underground`,
and the drill body's velocity (body + external velocity + the point velocity of the platform it is
in) in its **rotation and angular_velocity** fields, Q32.32, for the length of a publish only.
Deviations from the draft above:

- The velocity goes in rotation/angular_velocity, not vx/vy. The v4.2 writer, the bots and the
  styles read a form row's vx/vy, so this keeps every default-export consumer unchanged.
- The flag is the sim's own `DrillState.underground` (current-platform overlap), not the centre SDF.
  On 9471 form rows (main136, 200 games) flag vs SDF < 0 at the player: 00 8627, 01 222, 10 328,
  11 294. The disagreements sit at the surface (|SDF| median 0.76 u, p90 2.54).
- Drill only; no other form exports a body.
- The export is a process switch, off by default: the reference backend, replays and the divergence
  gates never see it. train.py sets it for `--v4-ablate` with bit 32, for any style naming
  `underground`, and for v4.3 league members; eval/native_policy.py sets it when it loads a v4.3
  policy.

**Writer, bit 32 (`BOPL4_ABLATE_DRILL`, layout v4.3; CPU `bopl_obs_v4.h`, GPU pack mode 4 via
`bopl_puffer.h` and `bopl_rays_v4.cu`):**

1. A slime in an ability whose player has a FORM_BODY row is read through a patched view: vx, vy
   from the form row, rotation = atan2 of its right axis. This covers the own and opponent body
   blocks, relation velocities, ray closing speeds, and slime rows. Positions and the critic's
   block are unchanged.
2. Rays from inside (`BOPL4F_INSIDE`, the fan's and the aim's rays): an origin with SDF < 0 marches
   by steps of max(-SDF, 0.05) to the first point with SDF >= 0 and reports that exit distance.
   There is no class beyond the exit (a deviation: the exit reads as a wall).
3. Under bit 32, a wall or water entity of the scene is not also a circle. Found by the probe
   below: a floating platform (a wall with mass) was also a class-"object" circle of its corner
   radius (3 u) about its centre. A ray from inside it hit that circle at distance ~0 before its
   exit, which was half of all underground heading rays.

`registry.WRITER_REVISIONS` = 16 | 32, so v4.3 runs init from and play against every v4.0/v4.2
member. The binding masks accept bits up to 63.

**Verification:**

- **Bit off is exact.** env_bench (tp-spec corpus, 120 games): state hash 9288e9a187eb638c,
  train_hash 85bf3e6781b01535, telemetry e9663ffa57413a92, equal for the base build, this build,
  and T's live PGO library. With the export on, state hashes move (the published rows carry the
  stamps) and telemetry is equal. `make ability-sweep` passes with the export off and on (24800
  episodes, 4588/4588 cells, 0 crashes, 0 divergent).
- **Decode** (`drill-spec/decode_v43.py`: three lockstep environments at delay 0, l2-main-136002m,
  200 games, 334,735 decisions):
  - Rows with the export on and the bit off equal rows with the export off on every decision
    (0 mismatches), and seat 1's bit-off rows are unchanged.
  - The body block equals the form export on 4625/4625 own rows and 4846/4846 opponent rows.
    Rotation is nonzero on 4619/4625 rows (v4.2: 324).
  - Underground rows with every ray at 0: v4.2 156/159, v4.3 0/159. Ray distance vs the exact
    exit along the ray, all 32 rays: |error| median 0.029 u, p95 0.074, 96.5% within 0.1 u
    (4991 rays). Only 97 rays (1.9%) were skipped for a body or projectile nearer than the exit;
    before the circle fix, 24% were.
  - The exported velocity vs the finite difference of position: median error 1.08 u/s at a median
    speed of 9.8 u/s.
- **Play is unchanged.** The 5 Section 2 conditions, re-run with seat A's rows logged at bits 48
  in a lockstep shadow environment (`drill_look.py --rows-ablate 48`, DS/look43/), replay all
  2000 games identically.
- **Ground ahead from the row** (`drill-spec/ray_heading.py`; the around ray nearest the true
  heading vs the true ground left along it, compared at the 9-tick observation delay):

  | Rows | within 1 u (ann DDD / main DDD / me31 DDD / me31 DDG / x2 DDG) | corr (ann / main) |
  |---|---|---|
  | v4.2 | 0.04 / 0.05 / - / - / - | -0.03 / -0.05 |
  | v4.3 before the circle fix | 0.48 / 0.46 / - / - / - | +0.22 / +0.32 |
  | v4.3 | 0.87 / 0.83 / 0.80 / 0.83 / 0.81 | +0.73 / +0.65 |

  Most of the remaining misses are rows at the surface (SDF at the truth -0.4 to -0.5), where the
  delayed origin is still above the ground.
- **Probes** (`probe.py` on DS/look43/, against undelayed truth; DS/probe43/, v4.2 in DS/probe/):
  heading cos from the row, ridge R^2 0.94 (v4.2: 0.70). Ground ahead from the row, ridge 0.12
  (v4.2: 0.07), MLP 0.11 (v4.2: -0.11), row + true heading 0.15 / 0.16. **The jump the draft
  predicted for this probe did not come, and the probe cannot show it.** Its target is the
  undelayed truth, but the row is 9 ticks (2 decisions, ~1.5 u of drill travel plus turning) old.
  Picking the ray along the heading is also a product that a ridge probe cannot form. With the
  target aligned to the delay, ridge reaches 0.22 (v4.2: degenerate). The direct readout above is
  the evidence that the information is in the row. Whether the policy uses it is for arm SO vs S.
- encode_check CPU == GPU and the training smoke run: Section 6's rift GPU gate.

## 5. Reward: style `drill_burrow` (final, 2026-10-23: `styles/drill_burrow.toml`)

Design rules:

1. Pay outcomes that need skill: net approach while underground, and kills.
2. Never pay duration. Underground time is farmable by circling in thick platforms.
3. Zero in sudden death and when no opponent is alive.
4. **On for the whole run, no fade** (Matthew 2026-10-23: small, as flair). The arms pass
   `--style-fade-steps 1e18`. Its strength cost is reported against arm C (report only, Section 6);
   the "insanely good" bar is unchanged.

```toml
name = "drill_burrow"
kits = [[26, 2, 2], [2, 26, 2], [2, 2, 26]]

[let]
ug = "any(entities where owner == self and underground == 1)"
drilling = "form_ability(self) == ability.Drill"
closer = "prev(min(dist(self, opp), 30)) - min(dist(self, opp), 30)"   # approach since the last decision, u

[[term]]   # net approach underground at both ends: a gain counts at most 2 u a decision, a loss in full
name = "burrow_approach"
when = "header.sudden_death == 0 and alive(opp) and ug and prev(ug)"
value = "0.006 * min(closer, 2)"   # <= 0.18 for a 30 u approach

[[term]]   # retreat off the underground stretches, in a Drill form that has been underground, costs the same rate
name = "drill_retreat"
when = "header.sudden_death == 0 and alive(opp) and drilling and prev(drilling) and not (ug and prev(ug)) and since(ug) < since_form_entry(self)"
value = "0.006 * min(closer, 0)"

[[term]]   # burrow-kill premium: drill kill after >= 0.5 s underground in the last 1.5 s, full at 1.5 s
name = "burrow_kill"
on = "death"
when = "ev.actor == self and ev.subject != self and ev.source_ability == ability.Drill and header.sudden_death == 0 and count_over(90, ug) >= 8"
value = "0.12 * min(1, count_over(90, ug) / 22)"

[[term]]   # unforced drill death (ring-out, self) costs extra
name = "drill_unforced"
on = "death"
when = "ev.subject == self and form_ability(self) == ability.Drill and (ev.cause == death.RING_OUT or ev.cause == death.SELF)"
value = "-0.08"
```

`underground` is a new entity field (style VM field 22, the form row's BOPL_ENTITY_UNDERGROUND bit;
`flags` itself exposes bits 0-23 only). train.py turns on the sim's form export
(`BOPL_RL_FORM_BODY=1`) for any style that names it, so arm S (v4.2 rows) gets the flag too.

Changes from the draft, all found by the scripted tests in
`native/rl/research/drill-spec/test_drill_burrow.py` (23 scenarios; the C VM and the Python
interpreter must agree bit for bit; `DRILL_BURROW TESTS PASS`). The draft fails 4 of them:

- **Hop-chain dip farm.** The draft paid approach only on underground-to-underground decisions.
  Approach underground, then back off above ground within the same form (a hop chain through thin
  platforms), and the retreat is never charged: 0.48 per 200 decisions in `dip_cycle`, 0.72 in
  `dip_fast_retreat`. Fix: `drill_retreat` charges any retreat on the off-underground decisions of a
  Drill form that has gone under (`since(ug) < since_form_entry(self)`). A form that never went
  under is not charged, so ordinary apex dives are untouched.
- **Slow-in, fast-out farm.** The symmetric clamp(-2, 2) let a body approach 1 u a decision and
  retreat 5 u in one decision with the loss cut to 2: 0.55 per 180 decisions. Fix: gains are capped
  at 2 u a decision, losses are charged in full (`min(closer, 2)`).
- With both, any closed loop within one form pays <= 0 (`oscillate_fast` -1.81, `dip_cycle` and
  `dip_fast_retreat` ~0 to 1e-4, `slow_in_fast_out` +0.03 over 30 cycles, the first decision's
  residual). A farm must end the form and recast (4 s cooldown).
- Fade removed (decision above).
- Found while testing: the style VM's entity field limit was 22 (`bs_field_limit`), so a program
  naming field 22 failed with "field out of range" at load. Raised to 23.

Compiled: 1124 bytes. `make style-check` (40 games) passes with the new style.

Scale and sanity:

- reward-kill is 0.288 per kill. The approach term's ceiling per stint (~0.18 for 30 u) is below
  one kill. The burrow-kill premium (<= 0.12) is the same size as drill_snipe's 0.08, which x2-drill
  learned from: its fire rate rose from 9e-5 to 1e-3 per decision over 1.5B steps
  (sweeps/exploiters-2/drill/run).
- Measured payout at init (the draft term, EXPLOSIVE stand-in), replaying it on the Section 2 logs: the approach term fires on
  0.4-0.55% of decisions (consecutive underground decisions, pre-SD). Its mean is **negative**:
  -0.0016 (main DDD), -0.0019 (ann DDD) and -0.0001 (me31 DDD) per firing, i.e. ~-1e-5 per
  decision overall. Today's burrows move away from the opponent (~0.3 u per decision for
  main/ann), so the gradient starts pointing at "burrow toward them".
- The start is too small to teach "never go underground". No entry credit is added (Matthew's runs
  start as pre-registered). Do not add a positive-only approach term: it is farmable by
  oscillating.

Hack audit:

| Exploit | Why it fails |
|---|---|
| Circle in a platform | approach telescopes, so circling ~0 |
| Approach/retreat oscillation | the approach term charges losses in full and caps gains, so it nets <= 0 |
| Approach underground, retreat above ground (hop chain) | drill_retreat charges the retreat within the form |
| Slow approach, fast retreat | gains capped at 2 u a decision, losses uncapped |
| Dip through a thin platform before a kill | burrow_kill needs >= 8 underground decisions (0.53 s) in the last 1.5 s |
| Drill toward the opponent in SD for safety | zero in SD |
| Burrow toward a dead opponent | alive(opp) gate |
| Spawn-adjacent farming | the approach term is bounded by 30 u per stint and needs a 4 s cooldown between stints (~0.18 per 4 s at most). Watch it in telemetry (`style/burrow_approach` fire rate) together with kills/min. |

## 6. Training plan (pre-registered here, before any run)

**Base.** Use off1's ff-1 command (v4.2, `--v4-ablate 16`, held-out panel excluded, l2 turn-100
flags). Run it through `train/research/arm.py` like `x2d-drill.toml`.

**Init.** off1/ff-1 anneal (`off1/ff-1/anneal/checkpoints/policy-1333395456.bin`), the current
research base (Matthew 2026-10-23).

**Kit.**

- `--loadout-learner-first --loadout-set native/data/lineups-drill-ddd.txt`.
- Rows are [learner | opponent]. The learner's Drill rotates over slots 0/1/2 with Dash in the
  other two. The opponent is 3 iid draws from pool `all` (duplicates legal).
- Build the file like lineups-2d-drill-iid-eval.txt: 3000 rows, seed 20261023.

**Opponents.**

- `--opponents league` minus the held-out panel (final5-18666m, anneal20-26666m, team5b-6250m,
  g111-20b-26027m), PFSP hard p = 1, own snapshots.
- Plus `--pinned-opponents l2-me31-drill-grenade-2000m` (it plays its own DDG kits), so drill-vs-drill
  trades are in the curriculum.

**Steps.** 3B at constant LR plus a 0.5B anneal to 1.2e-5. Checkpoints every 500M.

**Arms** (Matthew 2026-10-23: all three run; SO is no longer conditional on S).
`train/research/arms/drill.toml`, trained interleaved C, S-s7111, SO-s7121, S-s7112, SO-s7122.

| Arm | Obs | Style | Seeds |
|---|---|---|---|
| C | v4.2 (`--v4-ablate 16`) | none | 7101 |
| S | v4.2 | drill_burrow, no fade | 7111, 7112 |
| SO | v4.3 (`--v4-ablate 48` = 16 + 32) | drill_burrow, no fade | 7121, 7122 |

C, the kit-pinned no-style control, separates "drill kit specialisation" from the style and prices
the permanent style (report only).

As built: the lineup file `native/data/lineups-drill-ddd.txt` has 255 rows, not 3000
(`BOPL_LOADOUT_SET_MAX` is 256), each [learner | opponent], Drill in slot (row % 3) with Dash in the
other two, opponent 3 iid draws from pool `all`, seed 20261023. `arm.py --dry-run` gives ff-1's
command except the intended overrides (init, steps, seed, loadout and opponent flags, style).

**Evaluation** (`train/research/arms/drill.sh eval <arm>`, the drill_look protocol at the annealed
endpoint; the specialist plays Drill/Dash/Dash with Drill in slot pair % 3, its training rotation):

- **Primary, P1:** vs l2-main-136002m on iid random kits, 600 seed pairs (1200 games), seed base
  4100001 (pairs with Section 2's baselines). Per-pair score sd 0.39, so se ~0.016 per endpoint.
- Secondary:
  - P2 and P4 from the same games; P3 vs me31 on its kits (DDG and GDD alternating by pair),
    600 seed pairs, seed base 4300001
  - the held-out panel (arm.py's all_combos, 2000 games, report only)
  - drill telemetry as in Section 2
- CPU: two single-threaded units per arm (CPUWeight 30), started as each arm's anneal completes.

**"Insanely good", pre-registered on the showcase run's endpoint.** The showcase is the styled run
(S or SO) with the best P1 point, chosen after the fact among the four; one run is a showcase, not
a claim; arm claims use the seed mean.

| Criterion | Target | Today |
|---|---|---|
| P1 (primary) | Score vs main136 (random kits) >= 0.75, 95% LB >= 0.70 | me31 DDD 0.627, me31 DDG 0.695 |
| P2 | Burrows: >= 10% of drill episodes have >= 1 s continuous underground, **and** >= 25% of drill kills come after >= 0.5 s underground in that dive | 0.4-3.6% and 1.0-13% |
| P3 | Score vs me31 on me31's kits >= 0.50 | l2-main-125336m on iid kits vs me31: 0.25-0.26 |
| P4 | Guard: unforced drill deaths <= 3% of drill episodes, and SD share <= 30% | main DDD 5.4% and 52% |

P2's kill half is computed per kill as the longest underground run of that drill episode before
the kill (>= 8 decisions = 0.53 s; drill-spec/analyze.py `kill_ug05`): ann DDD 13.3% (10/75),
me31 DDG 1.0% (the draft reported 1.5% for me31 DDG; this rule is the one applied).

**Arm contrasts** (seed means paired by seed pair; eval-only 95% intervals, seed variance not in
them; `train/research/analyze.py drill`):

1. S - C and SO - C on P1: the permanent style's strength cost. Report only.
2. SO - S: adopt v4.3 for the specialist if SO - S >= +0.05 on P1, or SO passes P2 + intent where
   S does not. Intent: >= 20% of surfacings within 6 u of the opponent.

Power: two seeds per arm with seed sd ~0.03 (exploiter-scale guess) and eval se 0.016 gives a
difference se of ~0.034, so the MDE at 80% power is ~0.095. These arms detect only large effects.
That is acceptable for a showcase.

**Total.**

- GPU: C + 2S + 2SO = 5 runs x 3.5B. At off1's measured 1.8-3.1B learner steps/h that is
  1.1-2.0 h per run, ~6-10 GPU-h in all.
- CPU: ~2 x 1200 games per arm plus arm.py's panel, ~3 CPU-h.

**Schedule** (Matthew 2026-10-23: the drill runs go ahead of the queued recipe steps).
User unit `drill-queue` (RL/research/drill-queue.sh, log RL/research/drill-queue.log, ntfy topic
bopl): pause research-recipe while it is still in its wait loop; wait for research-offense; require
the rift GPU gate's pass; pull T and rebuild (pgo_build.sh, train/build.py --device cuda,
native_policy.py --build); gate on env_bench hashes equal to the pre-change build and on
check_trainer_fixes_gpu.sh; run the five arms with each eval started as its anneal completes;
relaunch research-recipe as soon as the last run ends (also on any failure); then analyze.py drill.

## 7. Matthew's answers (2026-10-23)

1. **Showcase vs strength.** Keep a small permanent burrow bonus, as flair: no fade. Its strength
   cost is reported (S - C, SO - C), report only. The "insanely good" bar stays.
2. **Kit.** DDD (Drill rotating over the slots).
3. **Init.** off1/ff-1 anneal.
4. **Who it shows off against.** Main136 on random kits (P1, primary); the other checks stay as
   secondary.
5. **Queue priority.** The drill runs go ahead of the queued recipe steps, without idling the GPU.
6. **v4.3 scope.** Build it now (sim export behind `BOPL_RL_FORM_BODY=1`, writer bit 32), and run
   SO alongside S rather than conditionally.
