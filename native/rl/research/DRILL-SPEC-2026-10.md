# Drill showcase specialist: mechanics, data, observations, reward, plan (2026-10-23)

Written 2026-10-23 by the drill-spec subagent. CPU only. Goal (Matthew): a Drill + Dash + Dash bot that
is "ridiculously good with drill", rewarded for longer drills so it learns crazy movement. This doc is
the evidence and a pre-registered plan. Nothing here has trained.

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
- **Recommended obs change: writer revision bit 32, "layout v4.3 drill"**. Same width, same field
  meanings, the v4.2 pattern, no surgery.
  - The sim exports the form body's velocity and an UNDERGROUND flag on the form row.
  - The body/relation blocks read the form body during a form, for self and for the opponent.
  - Rays cast from inside a wall report the distance to where the ray leaves the ground.
- **Reward: style `drill_burrow`**, fading to 0 at 60% of the run.
  - Underground net approach to the opponent, paid as a difference so oscillation pays nothing.
  - A drill-kill premium scaled by underground time in that dive.
  - A penalty for unforced drill deaths.
  - No per-tick underground pay (it is farmable: circling in a thick platform is possible below
    ~15 u/s). Zero in sudden death.
- **Plan:** arm S (style, v4.2 obs) runs first. Arm SO (style + v4.3) is built and run only if S
  burrows blind. There is also a kit-pinned no-style control. Each run is 3.5B steps (~1.2-1.5
  GPU-h), 5 runs in all, ~7.5 GPU-h plus ~1 day of agent code for v4.3. It fits after
  research-recipe drains (x2d + optimizer screen ~13-37 GPU-h from 2026-10-25), well before
  2026-11-18.

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

## 5. Reward: style `drill_burrow` (draft; needs the UNDERGROUND name from Section 4)

Design rules:

1. Pay outcomes that need skill: net approach while underground, and kills.
2. Never pay duration. Underground time is farmable by circling in thick platforms.
3. Zero in sudden death and when no opponent is alive.
4. Fade to 0 so the final policy is graded on winning alone. The burrow habit survives only if it
   wins.

```toml
name = "drill_burrow"
kits = [[26, 2, 2], [2, 26, 2], [2, 2, 26]]
fade = { start = 0.0, end = 0.6 }

# ug = own drill form row underground (the new flag), pre-SD, opponent alive
[[term]]   # net approach while underground: c * (d_prev - d_now) only when both decisions were underground.
           # Over a stint it telescopes to c * (entry distance - exit distance): oscillating nets 0, circling ~0.
name = "burrow_approach"
when = "header.sudden_death == 0 and alive(opp) and any(entities where owner == self and flags & UNDERGROUND) and prev(any(entities where owner == self and flags & UNDERGROUND))"
value = "0.006 * clamp(prev(min(dist(self, opp), 30)) - min(dist(self, opp), 30), -2, 2)"   # <= 0.18 for a 30 u approach

[[term]]   # burrow-kill premium: drill kill after >= 0.5 s underground in the last 1.5 s, full at 1.5 s
name = "burrow_kill"
on = "death"
when = "ev.actor == self and ev.subject != self and ev.source_ability == ability.Drill and header.sudden_death == 0 and count_over(90, any(entities where owner == self and flags & UNDERGROUND)) >= 8"
value = "0.12 * min(1, count_over(90, any(entities where owner == self and flags & UNDERGROUND)) / 22)"

[[term]]   # unforced drill death (ring-out, self) costs extra: burrowing raises exposure to bounds
name = "drill_unforced"
on = "death"
when = "ev.subject == self and form_ability(self) == ability.Drill and (ev.cause == death.RING_OUT or ev.cause == death.SELF)"
value = "-0.08"
```

With `EXPLOSIVE` standing in for `UNDERGROUND`, it compiles today: 135 words, 58 history floats per
player, `count_over(90)` = 24 decision samples.

Scale and sanity:

- reward-kill is 0.288 per kill. The approach term's ceiling per stint (~0.18 for 30 u) is below
  one kill. The burrow-kill premium (<= 0.12) is the same size as drill_snipe's 0.08, which x2-drill
  learned from: its fire rate rose from 9e-5 to 1e-3 per decision over 1.5B steps
  (sweeps/exploiters-2/drill/run).
- Measured payout at init, replaying the term on the Section 2 logs: the approach term fires on
  0.4-0.55% of decisions (consecutive underground decisions, pre-SD). Its mean is **negative**:
  -0.0016 (main DDD), -0.0019 (ann DDD) and -0.0001 (me31 DDD) per firing, i.e. ~-1e-5 per
  decision overall. Today's burrows move away from the opponent (~0.3 u per decision for
  main/ann), so the gradient starts pointing at "burrow toward them".
- The start is too small to teach "never go underground". If arm S shows the underground share
  falling in its first 250M steps, add a small entry credit paid only on kills
  (burrow_kill already is one). Do not add a positive-only approach term: it is farmable by
  oscillating.

Hack audit:

| Exploit | Why it fails |
|---|---|
| Circle in a platform | approach telescopes, so circling ~0 |
| Approach/retreat oscillation | the approach term pays both signs, so it nets 0 |
| Dip through a thin platform before a kill | burrow_kill needs >= 8 underground decisions (0.53 s) in the last 1.5 s |
| Drill toward the opponent in SD for safety | zero in SD |
| Burrow toward a dead opponent | alive(opp) gate |
| Spawn-adjacent farming | the approach term is bounded by 30 u per stint and needs a 4 s cooldown between stints (~0.18 per 4 s at most). Watch it in telemetry (`style/burrow_approach` fire rate) together with kills/min. |

## 6. Training plan (pre-registered here, before any run)

**Base.** Use off1's ff-1 command (v4.2, `--v4-ablate 16`, held-out panel excluded, l2 turn-100
flags). Run it through `train/research/arm.py` like `x2d-drill.toml`.

**Init.** off1/ff-1 anneal (`off1/ff-1/anneal/checkpoints/policy-1333395456.bin`), the current
research base. me31 is the stronger drill prior (Section 2), but it is a 2B exploiter of one target
and starts on v4.0. Arm S-me31 is optional (open question 3).

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

**Steps.** 3B at constant LR plus a 0.5B anneal to 1.2e-5. Checkpoints every 500M. 3.5B x 0.33-0.43 =
1.2-1.5 GPU-h per run.

**Arms.**

| Arm | Obs | Style | Seeds | When |
|---|---|---|---|---|
| C | v4.2 | none | 1 (seed 7101) | with S |
| S | v4.2 | drill_burrow | 2 (7111, 7112) | first |
| SO | v4.3 (bit 48 = 16 + 32) | drill_burrow | 2 (7121, 7122) | only if S fails "intent" (Section 4) |

C, the kit-pinned no-style control, separates "drill kit specialisation" from the style.

**Evaluation.**

- Primary: the drill_look protocol at the annealed endpoint, against l2-main-136002m on iid random
  kits. 600 seed pairs, seed base 4100001 (the same as Section 2, so it pairs with the Section 2
  baselines). Per-pair score sd 0.39, so se ~0.016 per endpoint.
- Secondary:
  - the mirror vs me31 on its kits (600 pairs)
  - the held-out panel (2000 games, report only)
  - drill telemetry as in Section 2
- CPU: ~0.25-0.6 s per game per worker, about 4,000 games per run (~0.5 CPU-h), at <= 4 workers,
  CPUWeight 30.

**"Insanely good", pre-registered on the best seed's endpoint.** One seed is a showcase, not a
claim; arm claims use the seed mean.

| Criterion | Target | Today |
|---|---|---|
| P1 | Score vs main136 (random kits) >= 0.75, 95% LB >= 0.70 | me31 DDD 0.627, me31 DDG 0.695 |
| P2 | Burrows: >= 10% of drill episodes have >= 1 s continuous underground, **and** >= 25% of drill kills come after >= 0.5 s underground | 0.4-3.6% and 1.5-13% |
| P3 | Score vs me31 on me31's kits (mirror-ish) >= 0.50 | l2-main-125336m on iid kits vs me31: 0.25-0.26 (x2d check, EXPLOITER-GAPS 1d) |
| P4 | Guard: unforced drill deaths <= 3% of episodes, and SD share <= 30% | main DDD 5.4% and 52% |

**Arm decisions.** Seed means, Section 2 protocol, paired by seed pair:

1. S - C >= +0.05 on P1 with S passing P2: the style works. Otherwise the style is not doing the
   work, so look before tuning the magnitude.
2. Go to SO if S fails intent: P2 passes but < 20% of surfacings are within 6 u, or P2 fails with
   underground share up and unforced deaths up.
3. Adopt v4.3 if SO - S >= +0.05 on P1, or SO passes P2/intent where S failed.

Power: two seeds per arm with seed sd ~0.03 (exploiter-scale guess) and eval se 0.016 gives a
difference se of ~0.034, so the MDE at 80% power is ~0.095. These arms detect only large effects.
That is acceptable for a showcase; a third seed brings the MDE to ~0.08.

**Total.**

- GPU: C + 2S + 2SO = 5 runs x 1.2-1.5 = 6-7.5 GPU-h, or 3.6-4.5 GPU-h if SO is not needed.
- Agent code: lineups file and style (0.5 h), v4.3 (~1 day, conditional).
- CPU: ~3 CPU-h of evaluation.

**Schedule.**

- The GPU is busy until ~2026-10-25 (research-offense, then research-recipe).
- research-recipe then runs x2d (0.9 GPU-h) and the optimizer screen (12 GPU-h) plus follow-ups
  (up to 24 GPU-h).
- S and C fit in any 3-4.5 h gap after that queue. Put them at the end of the recipe queue, or
  between its stages. That needs Matthew's priority call, because the queue pauses at decision
  points.
- Start the v4.3 code during S so SO is ready when S is read.
- The whole program needs < 8 GPU-h of the ~400 GPU-h of slack PLAN-2026-10 leaves. It finishes
  before 2026-11-01 if started by 2026-10-28, well inside the 2026-11-18 deadline.

## 7. Open questions only Matthew can answer

1. **Showcase vs strength.** The style fades to 0, so the final bot burrows only if burrowing wins.
   If you want visible burrows even when they don't pay, keep a small permanent burrow bonus. That
   trades strength for flair, and P1 may then fail.
2. **Kit.** DDD as asked, or DDG? Grenade is +0.068 [+0.005, +0.129] for me31. Is DDG an
   acceptable fallback if DDD misses P1?
3. **Init.** off1 anneal (research base, weaker drill) or me31 (stronger drill prior, exploiter of
   one target)? Running both doubles S.
4. **Who it shows off against.** Main136 on random kits (P1), humans (you), or the league? A human
   target needs your games: the 2026-09 replays are vs bots.
5. **Queue priority.** Can S and C (3.6-4.5 GPU-h) cut into research-recipe, or must they wait
   for it to finish?
6. **v4.3 scope.** OK to change the sim's exported form-row velocity and add an entity flag? It is
   export only and gated, but the body-block fix changes what main would see if main adopts bit 32.
