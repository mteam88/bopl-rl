# How Matthew beats the agents: a loss taxonomy of the human-vs-agent rounds

330 rounds Matthew played on the Mac against layout-v2 checkpoints and scripted bots
(`native/artifacts/rl/human/mac/games.jsonl`, 2026-09-11 .. 09-14), every one replayed
tick-for-tick in the Unity Linux build from its `.rep`, and every death classified by
mechanics. The same taxonomy is then run on 1152 rounds the same checkpoints played in the
same Unity build against the scripted bots and against pool members, so "does this happen in
training conditions too?" is answered with the same code on the same engine.

Pipeline (all under `native/rl/research/`):

| step | tool | output |
|---|---|---|
| replay the 330 rounds, one frame per level tick | `human_replays.py extract` | `native/artifacts/rl/human/mac/extracted/` (325/330 reproduce the logged end tick, outcome and causes exactly; the 5 that do not are excluded from every table below) |
| per-round mechanical facts + pattern classification | `human_taxonomy.py facts` / `classify()` | `extracted/facts.json` |
| same checkpoints vs bots and pool members in Unity | `agent_games.py play` (shim on both seats) | `native/artifacts/rl/human/mac/unity_agent_games/` |
| Rust-sim cross-check on maps 0-4 | inline (see §6) | — |
| search-bot check (Rust sim only) | `loss_taxonomy.py play/classify` | `native/artifacts/rl/human/mac/rust_search_games/` |
| all tables | `human_taxonomy.py tables --facts … --sim-facts …` | appendix |
| clips | `render_video.py` | `native/artifacts/rl/human/mac/clips/*.mp4` |

Player 0 is always "the human seat" (Matthew, or a bot / pool member in the Unity
comparison), player 1 the agent under study.

## 1. Strata

The brief assumed training maps and the standard Dash/Bow/Platform loadout; the data does
not. Every table is split:

| stratum | definition | rounds | human W-L | vs strong ckpt | maps | replay-verified |
|---|---|---|---|---|---|---|
| **A** | training maps 0-2, standard loadout both sides | 22 | 18-4 (82%) | 3-2 | 3 | 22/22 |
| **B** | maps outside 0-4 (unseen in training), standard loadout | 241 | 188-53 (78%) | 47-22 | 105 | 237/241 (4 excluded) |
| **C** | non-standard human loadout (grenade/drill/magnet/mine/poke ball/blink/beam) | 62 | 43-19 (69%) | 43-19 | 55 | 61/62 (1 excluded) |
| **D** | held-out maps 3-4, standard loadout | 5 | 3-2 | 1-1 | 2 | 5/5 |

Agent identity: the main opponent is `o7-016-final` (sha 7a1cf7f7; the
`policy-1333362688.bin` copies in overnight-7/016 are the same weights), 111 rounds. Tiers
in the tables: strong = o7-041/o7-016/policy-1333362688; mid = o7-000/007/033/035,
o3-032/017/019/012; weak = o3-001/002/009, run007.

**What A can and cannot say.** 22 rounds, 18 agent deaths, 5 of them against a strong
checkpoint. A supports *existence* claims ("the platform ram works on training maps
against o7-016": 4 of 18 A deaths, one against o7-016 itself) and rough family shares
(arrow 9, squish 6, self 2, fall 1), not rates per pattern; anything below ~4 counts in A is
noise. B carries the statistics (185 agent deaths), and the Unity comparison (§5) is what
separates "unseen map" from "human opponent" in B, because it plays the same agents on maps
0-2 *and* on 7 of the B maps against non-human opponents.

## 2. Agent deaths (human wins), the headline table

| family | all | A | B | C | D |
|---|---|---|---|---|---|
| arrow | 112 (45%) | 9 | 100 | 0 | 3 |
| squish | 85 (34%) | 6 | 72 | 7 | 0 |
| novel weapon (C only) | 23 (9%) | 0 | 0 | 23 | 0 |
| fall | 22 (9%) | 1 | 9 | 12 | 0 |
| self (own arrow) | 5 (2%) | 2 | 3 | 0 | 0 |
| environment | 1 | 0 | 1 | 0 | 0 |
| **all** | 248 | 18 | 185 | 42 | 3 |

Top patterns (full table in the appendix):

| pattern | all | A | B | C | D | what it is |
|---|---|---|---|---|---|---|
| arrow/hit-while-drawing-bow | 40 (16%) | 4 | 36 | 0 | 0 | agent dies inside its own bow morph |
| arrow/dash-in-shot | 27 (11%) | 0 | 25 | 0 | 2 | human dashes, releases ≤60 ticks later, range ≥5 |
| squish/foe-platform-rammed-victim | 23 (9%) | 4 | 19 | 0 | 0 | human in platform morph moves into the agent |
| arrow/short-range | 13 (5%) | 2 | 11 | 0 | 0 | |
| arrow/point-blank-after-dash | 13 (5%) | 1 | 11 | 0 | 1 | dash then release inside 5 units |
| squish/own-platform-exit-pinch | 12 (5%) | 0 | 9 | 3 | 0 | agent leaves its own platform morph into an obstruction |
| squish/moving-map-platform | 11 (4%) | 0 | 10 | 1 | 0 | |
| fall/jumped-off-edge | 11 (4%) | 1 | 4 | 6 | 0 | |
| squish/own-platform-exit-pinch-contested | 9 (4%) | 0 | 9 | 0 | 0 | as above with the human also in platform morph |
| novel-weapon/Grenade+Drill | 9 (4%) | 0 | 0 | 9 | 0 | |
| squish/own-platform-drifted-into-map | 9 (4%) | 0 | 7 | 2 | 0 | agent standing on its own platform, crushed against level geometry |
| squish/foe-platform-dropped-on-victim | 8 (3%) | 2 | 6 | 0 | 0 | human platform spawns onto the agent (≤31 ticks) |
| novel-weapon/Magnet gun+Mine | 8 (3%) | 0 | 0 | 8 | 0 | |

Tags on the 112 arrow deaths: **dash-in 61%**, **shooter airborne 62%**, victim dash on
cooldown 49%, shot from above 49%, victim jumped into the arrow 29%. Only 7% of the
human's kill shots are "unreactable" (dash → release ≤13 ticks, inside the 9-tick observation
delay + 4-tick cadence); the rest are visible for at least one decision.

Pre-death window (120 ticks before the death, medians): the agent that is about to die is
**static** (airborne fraction 0.0, speed ≈0) at 19.4 units mean distance, has fired 0 arrows
while the human fired 1, and last shot 178 ticks ago; the agent that is about to win is
moving (airborne 0.2, speed 6.5), has fired 1, last shot 33 ticks ago. The human closes to
8.3 units minimum distance when the agent wins vs 11.4 when the agent loses: the agent wins
the brawls it is allowed to have and loses the ones where it stands still. (Speed for a
player in a morph reads 0 from the slime record; the airborne and shot statistics carry the
point.)

Human deaths (77) for contrast: hit-while-drawing-bow 13, long-flight arrow not dodged 12,
hit in platform morph 7, point-blank 7; squish of all kinds only 13 (17%). The agent kills
the human mostly with arrows at range; it almost never squishes him.

## 3. Priority 1 — squish (85 = 34% of agent deaths)

### 3.1 Breakdown

| who crushed the agent | human data | of which A | B | C | Unity: same agents vs bots+pool (498 deaths) |
|---|---|---|---|---|---|
| **human's platform** (rammed 23, dropped 8, walked into 5) | 36 (14%) | 6 | 30 | 0 | **13 (3%)** — rammed 1, dropped 10, walked into 2 |
| **agent's own platform** (exit pinch 12+9, drift 9, pushed 5) | 35 (14%) | 0 | 29 | 6 | **105 (21%)** |
| map platform (moving 11, other 3) | 14 (6%) | 0 | 13 | 1 | 43 (9%) |

### 3.2 The ram is a human-only weapon, and it is a pinch against the agent's own platform

20 of the 23 rams (and 7 of 8 drops) find the agent **standing on its own spawned
platform** (`standing-on-own-platform` tag); in 15 of 23 the agent is *itself still in
platform morph* when the human starts his. The human morphs 5.7-39 units away (median
~20), 69-146 ticks before the crush, and carries the platform body into the agent, who
moves <2 units in 5 of the 23 windows and never leaves the platform in the others. The
human's platform activations are rare (0.8/round vs the agent's 1.2-3.6) and only 4% are
within 8 units of the agent; the weapon is the *travel*, not the plant.

Nothing in the training population does this. In 1152 Unity rounds the four v2-hard bots
and four pool members produced **one** ram (o7-041) and 10 drops; the seat-0 platform
activations within 8 units of the agent are 1-15% for bots and 1-2% for pool members, and
they do not travel. The search bots (Rust sim, `loss_taxonomy.py`, search-hard vs
v3b-016/v3b-041 on maps 0-4, 60 games) win 11 of their 60 games by squish: 4 with their own
platform growing/settling onto the policy (the drop, not the ram), 3 are the policy's own
platform, 4 the map.

→ **curriculum hole, not fidelity.** The Unity comparison is the real engine, so the 3%
vs 14% gap cannot be physics; the population simply never rams. Also the ram has a
69-146 tick window in which the agent stays put: it has never had to react to an
approaching platform body.

### 3.3 Own-platform pinch: same rate without the human, and it is a map problem

The self-inflicted squishes happen at **21% of agent deaths against bots and pool
members**, higher than the 14% against the human. Split by map:

| Unity, all three agents | rounds | agent deaths | own-platform squish | falls | agent loss rate |
|---|---|---|---|---|---|
| training maps 0-2 | 288 | 102 | **2 (2%)** | 15 (15%) | 35% |
| held-out 3-4 | 192 | 86 | **19 (22%)** | 17 (20%) | 45% |
| 7 unseen maps the human played (71,12,31,30,61,13,6) | 672 | 310 | **84 (27%)** | 62 (20%) | 46% |

Human data agrees: A has 0 own-platform squishes in 18 agent deaths, B has 29 in 185 (16%).

→ **curriculum/policy, specifically map generalisation**: the agent has memorised where
a platform can be planted and exited on maps 0-2; on any other geometry it exits its morph
into a wall or lets the platform it stands on drift into a ledge. The Rust sim does
implement the crush (§6: all 6 squish deaths in the 31 human rounds on maps 0-4 reproduce,
two with a 1-3 tick timing offset), so the sim is not hiding the mechanic; it is only ever
trained on three maps.

Answer to the question asked: o7-016's own-platform-pinch share is 14% vs the human, 5%
on maps 0-2 / 18% on 3-4 / 24% on unseen maps vs bots+pool in the same engine. Rates match
or exceed the human's off the training maps and vanish on them. Fix maps, not physics.

## 4. Priority 2 — bow-draw deaths (40 agent, 13 human)

### 4.1 Commitment window

| drawer | draws | median release | release histogram (ticks) | died drawing | opp. distance at commit (median) | opp. drawing at commit | opp. dashed inside window |
|---|---|---|---|---|---|---|---|
| agent (all) | 902 | 32 | 10-19: 310 · 20s: 49 · 30s: 181 · 40s: 17 · 50s: 26 · 60+: 282 | 38 (4.2%) | 12.9 | **26 (68%)** | 3 (8%) |
| human (all) | 586 | 51 | 10s: 48 · 20s: 40 · 30s: 64 · **40s: 114 · 50s: 119** · 60s: 58 · 70-90s: 60 · ≥100: 69 | 11 (1.9%) | 7.8 | 8 (73%) | 0 |

(Stratum split in the appendix; A and B agree on every column.)

The agent's release time is **quantised by the hold head**: it releases at ~10, ~32 or
~64 ticks (hold lengths 2/8/16 decisions × 4 ticks) and essentially never holds 40-59
ticks (5% of releases) — the band where the human releases 41% of his arrows. 12% of the
human's draws are held ≥100 ticks: a fully drawn bow parked, waiting.

### 4.2 It is a draw duel the agent enters late, not a dash-in

Of the 38 agent deaths inside a bow morph:

- **28 are duels**: the human was already drawing when the agent committed, and in 25 of
  them he started first (median 27 ticks earlier, range -7..149). He released a median 17
  ticks after the agent's commit; the agent needed 32-64.
- The human **dashed inside the window in only 3**; he was airborne at commit in 27 (jumping
  while fully drawn, which also explains "shooter airborne" on 61% of kill shots).
- Distance at commit: 7 deaths under 6 units, 8 at 6-10, 6 at 10-15, 17 beyond 15. Death
  rate of a draw by distance: 13% (<6), 10% (6-10), 4% (10-15), 3% (>15) for the agent vs
  10 / 3 / 3 / 0.5% for the human. Opponent dash ready at commit changes the agent's rate
  only in the <6 bucket (25% vs 6%).

The v2 observation carries the opponent's per-ability `active` bit and bow charge
(`bopl_puffer.h` v2: `ability[44]`, `ability[45]`), the opponent's dash cooldown ratio and
ready flag, position and velocity. With the 9-tick delay the agent still saw the drawn bow
for a median ~18 ticks (≥4 decisions) before committing.

→ **policy failure, with a small perception component**: the information was in the
observation; the agent draws into a drawn bow at 13 units. The perception component is
the release-time quantisation of the hold head, which denies the agent the 40-60 tick
draw and the parked full draw, so it cannot mirror the human's duel timing even if it
wanted to. Against bots the same agents die drawing at 10% of deaths (50/498), against
the human at 16%, so the pattern exists in training conditions and the human merely
exploits it harder.

The 13 human deaths inside a bow morph are the mirror image: 5 of 11 with a concurrent
agent draw, the agent never dashed in, median window 21 ticks, 9 of 11 at <13 units.

## 5. Priority 3 — the dash-in / airborne shot is absent from the training population

| shooter (rounds) | releases | dash ≤60 ticks before release | airborne at release | kill shots | of kills: dash ≤60 | dash ≤13 | airborne | point-blank |
|---|---|---|---|---|---|---|---|---|
| **human** (A+B+D) | 572 | **68%** | **70%** | 112 | **61%** | 7% | 62% | 23% |
| agent vs human | 865 | 10% | 32% | 46 | 13% | 0% | 41% | 35% |
| archer-v2-hard (Unity seat 0) | 407 | 8% | 47% | 31 | 6% | 0% | 42% | 6% |
| pursuer-v2-hard | 339 | 8% | 49% | 30 | 7% | 0% | 63% | 27% |
| mixed-v2-hard | 498 | 9% | 47% | 37 | 3% | 0% | 49% | 22% |
| evasive-v2-hard | 691 | 1% | 28% | 38 | 0% | 0% | 21% | 0% |
| o7-000 (pool) | 771 | 10% | 43% | 36 | 19% | 0% | 47% | 61% |
| o7-041 (pool) | 484 | 7% | 15% | 24 | 8% | 0% | 25% | 50% |
| o3-032 (pool) | 495 | 1% | 4% | 7 | 14% | 0% | 29% | 57% |
| o7-007 (pool) | 1625 | 1% | 3% | 4 | 0% | 0% | 0% | 25% |
| o7-016 as the agent (seat 1) | 2709 | 15% | 38% | 65 | 32% | 3% | 51% | 18% |

The human dashes before two thirds of his releases; the best any training opponent
manages is 10% (o7-000), and the dash-in kill pattern (`arrow/dash-in-shot`) accounts for
2 of 498 agent deaths in Unity vs 27 of 248 against the human. o7-016 itself is the
closest thing to a dash-shooter in the pool (15% / 32% of kills), which is presumably why
it is the strongest of them against the human, and it still dies to the pattern.

→ **curriculum hole** for the exploiter/league design: no scripted bot has a dash→release
combo and self-play has not discovered it. The reactable part (dash → release 14-60 ticks,
54% of the human's kills) is learnable; the 7% unreactable part is a perception-latency
floor.

## 6. Cross-checks on the simulators

**Unity Linux build, v2 protocol.** All 330 rounds replay; 325 match the logged end tick,
outcome and causes exactly. 5 (4 in B, 1 in C) diverge — two never end, three end with a
different winner — presumably a desync of the original session (the Mac build recorded the
inputs, the round ran online); they are excluded from every table. HEAD's
`Builds/Linux` and Mac builds still speak RL protocol **v2** (`RlBridge.cs`), while
`native/rl/bopl_rl.c` at HEAD is v3 (header 48→64, entity 128→136 bytes, commit
`3721f8a`); `human_replays.py` compiles the pre-`3721f8a` `bopl_rl.c` into
`libbopl_rl_v2.so` to drive them. The Sep-14 `libbopl_agent.so` is also v2 (obs 3298) and is
what `agent_games.py` uses on both seats; the search bots cannot run there (they fork the
in-process sim).

**Rust sim (`libbopl_sim.so`, HEAD).** Feeding the 31 human rounds on maps 0-4 through the
Rust sim (source-hash check bypassed): **22/31 reproduce the Unity trajectory to the tick**
including 6 squish deaths. Of the 9 that diverge: 4 are stratum C (grenade/mine — not in
the sim, expected); 2 diverge during a platform-vs-platform contact (r0-602 map 0, both
players in platform morph, tick 693; r12-837 map 4, tick 1448); 2 squish deaths land 1-3
ticks late/early (r0-778 894 vs 897, r35-817 1791 vs 1792) with identical positions
otherwise; 1 arrow kill on map 3 (r40-898, tick 1387) is not registered in the sim although
positions match to <0.01. So: squish physics reproduce, platform-body-vs-platform-body
contact and one arrow hit test do not, at a rate of ~3 rounds in 27 standard-loadout
rounds. Worth a fidelity ticket, but it is not what hides the ram (that is the absence of
rammers, §3.2).

## 7. Stratum B: map generalisation vs. human-specific patterns

Patterns that appear in B but not in A, with the Unity control showing they are the map,
not the human:

| pattern | A (18 deaths) | B (185) | Unity maps 0-2 (102) | Unity maps 3-4 + unseen (396) |
|---|---|---|---|---|
| own-platform squish (all four kinds) | 0 | 29 (16%) | 2 (2%) | 103 (26%) |
| moving map platform / map platform other | 0 | 13 (7%) | see appendix | |
| fall/* | 1 | 9 (5%) | 15 (15%) | 79 (20%) |

Patterns present in A **and** B at similar shares, i.e. the human, not the map:
hit-while-drawing-bow (4/18 vs 36/185), foe-platform ram/drop (6/18 vs 25/185), dash-in and
point-blank-after-dash (1/18 vs 36/185 — under-represented in A, but A is 22 rounds and 13
of them on map 0, where the geometry gives fewer approach lines).

Agent loss rate vs the *same* opponents rises from 35% (maps 0-2) to 45-46% (everything
else) in Unity; vs the human the win rate is 82% (A) vs 78% (B) — the human's edge is
nearly map-independent, the self-inflicted squishes are what the unseen maps add.

## 8. Stratum C: abilities the agent has never observed

62 rounds, 42 agent deaths. 23 (55%) are kills by an ability the agent never saw in
training (grenade+drill 9, magnet gun+mine 8, poke ball+drill 5, beam 1) — these are
`other`/`beamed` deaths with no arrow of the human's in flight, so the novel weapon is the
killer by elimination. The other 19 are ordinary: falls 12 (jumped-off-edge 6,
walked-or-knocked-off 4, ground-reached-water 2 — the drill and magnet displace the agent,
hence the high fall share), own-platform squish 6, moving map platform 1. No arrow deaths,
because the human had no bow. The agent's own kills in C (19) are unchanged in kind (arrow 6,
falls 5, squish 4, self 3, level hazard 1).

## 9. Examples (clips in `native/artifacts/rl/human/mac/clips/`, tick = level tick, ≈tick/60 s from round start)

| clip | round | stratum / map / agent | pattern | ticks |
|---|---|---|---|---|
| `rammed-o7-016-map0-r0-778.mp4` | 20260914-110241-r0-778 | A / 0 / o7-016 | squish/foe-platform-rammed-victim | human morphs 807, agent (on its own platform, 7.7 u away) crushed 896; platform moved 5.2 u in the last 30 ticks |
| `rammed-o3-032-map1-r14-664.mp4` | 20260911-163657-r14-664 | A / 1 / o3-032 | squish/foe-platform-rammed-victim | morph 761 → crush 830, agent in platform morph on its own platform |
| `bow-duel-o7-016-map0-r0-667.mp4` | 20260913-163449-r0-667 | A / 0 / o7-016 | arrow/hit-while-drawing-bow | agent commits at 21.5 u while the human is drawn; arrow released, 17-tick flight, agent jumps into it, dies 853 |
| `bow-draw-dash-in-mixed-v2-map0-r0-604.mp4` | 20260911-154247-r0-604 | A / 0 / mixed-v2-hard | arrow/hit-while-drawing-bow + dash-in | dash → release 27 ticks, airborne, 4.6 u, dies 786 |
| `dash-in-shot-o7-000-map80-r13-852.mp4` | 20260914-220357-r13-852 | B / 80 / o7-000 | arrow/dash-in-shot | dash → release 48 ticks, airborne, from above, 10.3 u, agent dash on cooldown, dies 377 |
| `own-platform-exit-pinch-archer-map118-r41-899.mp4` | 20260914-220357-r41-899 | B / 118 / archer-hard | squish/own-platform-exit-pinch | agent's own platform (72 ticks old, stationary) crushes it on exit, human 24 u away |
| `own-platform-drift-o7-016-map42-r11-758.mp4` | 20260914-000342-r11-758 | B / 42 / o7-016 | squish/own-platform-drifted-into-map | agent standing on its own platform, crushed against map geometry at 847, human 11.6 u away |

## 10. Conclusions: the dominant human winning patterns

1. **Dash-in / airborne shot** (arrow deaths with dash-in tag: 68 = 27% of all agent deaths;
   as primary pattern 40). *Curriculum.* No bot and no pool member dashes before shooting
   more than 10% of the time; the human does 68%. The 9-tick delay makes 7% of these
   unreactable, the rest are visible for ≥1 decision.
2. **Draw duel** (40 = 16%). *Policy*, with a *perception/action* component. The agent
   commits a 32-64 tick draw at ~13 units into an opponent whose drawn bow is in its
   observation; the hold head's {10, 32, 64}-tick release grid gives it no 40-60 tick draw and
   no parked full draw to duel with.
3. **Platform ram** (23 rams + 8 drops = 12%). *Curriculum.* A human-only weapon; the
   victim is on its own platform (often still morphed) and has 70-150 ticks to move and does
   not, because no training opponent has ever moved a platform body at it.
4. **Own-platform pinch on unfamiliar geometry** (35 = 14% vs the human; 21% vs bots, 2% on
   maps 0-2 vs 26% elsewhere). *Curriculum (maps)*, confirmed not to be a physics-fidelity
   gap: same engine, same agents, the rate is set by the map.
5. **Standing still** (pre-death window: static, no shots for ~3 s). *Policy.* Common to
   1-4; the human's edge is nearly the same on training and unseen maps (82% vs 78%).

Recommendations:

- **policy-v4**: (a) widen or reshape the hold head so draws of 40-60 ticks and a held full
  charge are expressible (e.g. hold lengths {1,2,3,4,6,8,12,16} or a "hold until release"
  latch), and mask/penalise a bow commit when the opponent's `active bow` bit is set within
  ~15 units; (b) a moving-platform-body threat feature is already in the entity rows
  (kind/velocity); check the value head actually reads platform velocity, since a ram
  is 70-150 ticks of an approaching entity with nonzero vx.
- **curriculum**: (a) train on all 124 levels, or at least a rotating dozen, with the
  0-2 share capped — the own-platform pinch alone is 20+ points of loss rate off the training
  maps; (b) add a *rammer* scripted bot (morph at 10-30 units, steer the platform body into a
  stationary/morphed opponent) and a *dash-shooter* bot (dash toward the opponent, release
  14-40 ticks later while airborne, aim from above); both are tens of lines in
  `bopl_bots_v2.h`; (c) exploiter league: seed the exploiter with these two behaviours
  rather than hoping self-play finds them.
- **reward**: (a) small shaping penalty for being squished by one's own platform (the
  agent's own morph is identifiable, `platform_owners` in the taxonomy shows how); (b) a
  penalty for a bow morph that ends in death without a release (38 of 902 draws); (c) do
  not reward stillness — the current policies camp fully drawn far less than the human and
  yet die static.

## 11. Caveats

- **Unity builds at HEAD are stale (RL protocol v2).** `Builds/Linux/BoplBattle.x86_64`,
  the Mac build Matthew played, and `libbopl_agent.so` (Sep 14 15:39) all predate the v3 ABI
  in `bopl_rl.c`/`bopl_layout.h`; `replay_data.py`/`check_agent_fidelity.py` at HEAD refuse
  them. `human_replays.py build_v2()` compiles the `3721f8a^` runtime; schedule a rebuild
  of the players and the shim, then delete the shim in `human_replays.py`.
- The Unity comparison runs the same v2 checkpoints Matthew faced, so it is on-distribution
  for weights but the opponents are bots and four pool-v6 members, not the pool-v7 the
  parent named (those are layout v3 and cannot load in the v2 shim). Search bots were run in
  the Rust sim against v3b-016/v3b-041 instead (60 games, maps 0-4).
- Stratum A is 22 rounds; treat every A count below 4 as anecdotal (§1).
- Rust sim reproduces 22/31 standard rounds on maps 0-4 to the tick; platform-vs-platform
  contact and one arrow hit test diverge (§6).
- `classify()` names are mechanical, not intentional: "rammed" means the human's platform
  body moved >1 unit in the 30 ticks before the crush; "drifted" means the map crushed the
  agent while it stood on its own platform. Owner attribution of a spawned platform is the
  nearest player in platform morph within 6 units at spawn.
- Speed in the pre-death table is 0 for a morphed player (slime record); airborne fraction
  and shot counts are reliable.
- 116 of 1152 Unity rounds hit the 7200-tick truncation (draws), mostly o7-041 vs pool
  members on the unseen maps; they are excluded from death tables.

## Appendix: all tables (`human_taxonomy.py tables`)

_5 of 330 rounds do not reproduce the logged outcome and are excluded: 20260914-000342-r5-752, 20260914-000342-r8-741, 20260914-220357-r10-835, 20260914-220357-r54-912, 20260914-220357-r9-834_

## Rounds and outcomes by stratum

| stratum | rounds | human wins | agent wins | human win % | vs strong | human wins vs strong | maps | replay-verified |
|---|---|---|---|---|---|---|---|---|
| A | 22 | 18 | 4 | 82% | 5 | 3 | 3 | 22 |
| B | 237 | 185 | 52 | 78% | 68 | 47 | 104 | 237 |
| C | 61 | 42 | 19 | 69% | 61 | 42 | 54 | 61 |
| D | 5 | 3 | 2 | 60% | 2 | 1 | 2 | 5 |


## Per-tier outcomes by stratum

| tier | rounds | human wins | A (W/n) | B (W/n) | C (W/n) | D (W/n) |
|---|---|---|---|---|---|---|
| strong-checkpoint | 136 | 93 (68%) | 3/5 | 47/68 | 42/61 | 1/2 |
| mid-checkpoint | 103 | 88 (85%) | 12/12 | 74/88 | 0/0 | 2/3 |
| weak-checkpoint | 17 | 16 (94%) | 1/1 | 15/16 | 0/0 | 0/0 |
| bot | 69 | 51 (74%) | 2/4 | 49/65 | 0/0 | 0/0 |


## Agent deaths (human wins) by pattern and stratum

| pattern | all | A | B | C | D |
|---|---|---|---|---|---|
| arrow/hit-while-drawing-bow | 40 (16%) | 4 | 36 | 0 | 0 |
| arrow/dash-in-shot | 27 (11%) | 0 | 25 | 0 | 2 |
| squish/foe-platform-rammed-victim | 23 (9%) | 4 | 19 | 0 | 0 |
| arrow/short-range | 13 (5%) | 2 | 11 | 0 | 0 |
| arrow/point-blank-after-dash | 13 (5%) | 1 | 11 | 0 | 1 |
| squish/own-platform-exit-pinch | 12 (5%) | 0 | 9 | 3 | 0 |
| squish/moving-map-platform | 11 (4%) | 0 | 10 | 1 | 0 |
| fall/jumped-off-edge | 11 (4%) | 1 | 4 | 6 | 0 |
| squish/own-platform-exit-pinch-contested | 9 (4%) | 0 | 9 | 0 | 0 |
| novel-weapon/Grenade+Drill | 9 (4%) | 0 | 0 | 9 | 0 |
| squish/own-platform-drifted-into-map | 9 (4%) | 0 | 7 | 2 | 0 |
| squish/foe-platform-dropped-on-victim | 8 (3%) | 2 | 6 | 0 | 0 |
| novel-weapon/Magnet gun+Mine | 8 (3%) | 0 | 0 | 8 | 0 |
| arrow/point-blank | 7 (3%) | 0 | 7 | 0 | 0 |
| fall/ground-reached-water | 6 (2%) | 0 | 4 | 2 | 0 |
| squish/own-platform-pushed-into-victim | 5 (2%) | 0 | 4 | 1 | 0 |
| squish/victim-moved-into-foe-platform | 5 (2%) | 0 | 5 | 0 | 0 |
| self/own-arrow-mortar-returned | 5 (2%) | 2 | 3 | 0 | 0 |
| fall/walked-or-knocked-off | 5 (2%) | 0 | 1 | 4 | 0 |
| novel-weapon/Poke ball+Drill | 5 (2%) | 0 | 0 | 5 | 0 |
| arrow/hit-mid-dash | 4 (2%) | 1 | 3 | 0 | 0 |
| arrow/hit-while-platform-morph | 4 (2%) | 1 | 3 | 0 | 0 |
| squish/map-platform-other | 3 (1%) | 0 | 3 | 0 | 0 |
| arrow/long-flight-not-dodged | 2 (1%) | 0 | 2 | 0 | 0 |
| arrow/long-flight-dodge-failed | 2 (1%) | 0 | 2 | 0 | 0 |
| novel-weapon/Beam | 1 (0%) | 0 | 0 | 1 | 0 |
| environment/level-hazard | 1 (0%) | 0 | 1 | 0 | 0 |
| **all** | 248 | 18 | 185 | 42 | 3 |


## Agent deaths by family and stratum

| pattern | all | A | B | C | D |
|---|---|---|---|---|---|
| arrow | 112 (45%) | 9 | 100 | 0 | 3 |
| squish | 85 (34%) | 6 | 72 | 7 | 0 |
| novel-weapon | 23 (9%) | 0 | 0 | 23 | 0 |
| fall | 22 (9%) | 1 | 9 | 12 | 0 |
| self | 5 (2%) | 2 | 3 | 0 | 0 |
| environment | 1 (0%) | 0 | 1 | 0 | 0 |
| **all** | 248 | 18 | 185 | 42 | 3 |


## Human deaths (agent wins) by pattern and stratum

| pattern | all | A | B | C | D |
|---|---|---|---|---|---|
| arrow/hit-while-drawing-bow | 13 (17%) | 0 | 13 | 0 | 0 |
| arrow/long-flight-dodge-failed | 12 (16%) | 0 | 9 | 3 | 0 |
| arrow/hit-while-platform-morph | 7 (9%) | 1 | 6 | 0 | 0 |
| arrow/point-blank | 7 (9%) | 0 | 6 | 1 | 0 |
| fall/jumped-off-then-platform-failed | 5 (6%) | 0 | 2 | 3 | 0 |
| squish/map-platform-other | 3 (4%) | 1 | 1 | 1 | 0 |
| squish/foe-platform-dropped-on-victim | 3 (4%) | 1 | 1 | 1 | 0 |
| squish/moving-map-platform | 3 (4%) | 0 | 1 | 2 | 0 |
| squish/foe-platform-rammed-victim | 3 (4%) | 0 | 2 | 1 | 0 |
| self/own-arrow | 3 (4%) | 0 | 3 | 0 | 0 |
| self/own-weapon | 3 (4%) | 0 | 0 | 3 | 0 |
| arrow/short-range | 3 (4%) | 0 | 3 | 0 | 0 |
| squish/own-platform-exit-pinch | 3 (4%) | 0 | 3 | 0 | 0 |
| arrow/point-blank-after-dash | 2 (3%) | 1 | 0 | 1 | 0 |
| fall/jumped-off-edge | 2 (3%) | 0 | 1 | 1 | 0 |
| squish/own-platform-exit-pinch-contested | 1 (1%) | 0 | 1 | 0 | 0 |
| arrow/hit-mid-dash | 1 (1%) | 0 | 0 | 0 | 1 |
| fall/ground-reached-water | 1 (1%) | 0 | 0 | 1 | 0 |
| environment/level-hazard | 1 (1%) | 0 | 0 | 1 | 0 |
| arrow/long-flight-not-dodged | 1 (1%) | 0 | 0 | 0 | 1 |
| **all** | 77 | 4 | 52 | 19 | 2 |


## Agent deaths by pattern and agent tier

| pattern | all | strong-checkpoint | mid-checkpoint | weak-checkpoint | other-checkpoint | bot |
|---|---|---|---|---|---|---|
| arrow/hit-while-drawing-bow | 40 (16%) | 6 | 13 | 0 | 0 | 21 |
| arrow/dash-in-shot | 27 (11%) | 4 | 12 | 4 | 0 | 7 |
| squish/foe-platform-rammed-victim | 23 (9%) | 9 | 6 | 4 | 0 | 4 |
| arrow/short-range | 13 (5%) | 3 | 7 | 1 | 0 | 2 |
| arrow/point-blank-after-dash | 13 (5%) | 3 | 6 | 0 | 0 | 4 |
| squish/own-platform-exit-pinch | 12 (5%) | 7 | 2 | 1 | 0 | 2 |
| squish/moving-map-platform | 11 (4%) | 5 | 5 | 0 | 0 | 1 |
| fall/jumped-off-edge | 11 (4%) | 6 | 4 | 0 | 0 | 1 |
| squish/own-platform-exit-pinch-contested | 9 (4%) | 5 | 4 | 0 | 0 | 0 |
| novel-weapon/Grenade+Drill | 9 (4%) | 9 | 0 | 0 | 0 | 0 |
| squish/own-platform-drifted-into-map | 9 (4%) | 6 | 3 | 0 | 0 | 0 |
| squish/foe-platform-dropped-on-victim | 8 (3%) | 2 | 5 | 1 | 0 | 0 |
| novel-weapon/Magnet gun+Mine | 8 (3%) | 8 | 0 | 0 | 0 | 0 |
| arrow/point-blank | 7 (3%) | 1 | 4 | 1 | 0 | 1 |
| fall/ground-reached-water | 6 (2%) | 2 | 2 | 1 | 0 | 1 |
| squish/own-platform-pushed-into-victim | 5 (2%) | 1 | 1 | 1 | 0 | 2 |
| squish/victim-moved-into-foe-platform | 5 (2%) | 1 | 2 | 1 | 0 | 1 |
| self/own-arrow-mortar-returned | 5 (2%) | 2 | 2 | 0 | 0 | 1 |
| fall/walked-or-knocked-off | 5 (2%) | 4 | 1 | 0 | 0 | 0 |
| novel-weapon/Poke ball+Drill | 5 (2%) | 5 | 0 | 0 | 0 | 0 |
| arrow/hit-mid-dash | 4 (2%) | 0 | 2 | 0 | 0 | 2 |
| arrow/hit-while-platform-morph | 4 (2%) | 1 | 2 | 0 | 0 | 1 |
| squish/map-platform-other | 3 (1%) | 1 | 2 | 0 | 0 | 0 |
| arrow/long-flight-not-dodged | 2 (1%) | 0 | 2 | 0 | 0 | 0 |
| arrow/long-flight-dodge-failed | 2 (1%) | 1 | 0 | 1 | 0 | 0 |
| novel-weapon/Beam | 1 (0%) | 1 | 0 | 0 | 0 | 0 |
| environment/level-hazard | 1 (0%) | 0 | 1 | 0 | 0 | 0 |
| **all** | 248 | 93 | 88 | 16 | 0 | 51 |


## Agent deaths per map (stratum B, maps with >= 3 agent deaths)

| map | rounds | agent deaths | families |
|---|---|---|---|
| 61 | 4 | 4 | arrow 3, squish 1 |
| 14 | 3 | 3 | squish 2, arrow 1 |
| 12 | 4 | 3 | squish 2, arrow 1 |
| 31 | 4 | 3 | fall 1, arrow 1, squish 1 |
| 30 | 4 | 3 | arrow 3 |
| 53 | 3 | 3 | arrow 2, self 1 |
| 9 | 3 | 3 | arrow 3 |
| 89 | 3 | 3 | squish 3 |
| 71 | 5 | 3 | arrow 2, squish 1 |
| 7 | 3 | 3 | arrow 2, squish 1 |
| 93 | 4 | 3 | arrow 3 |
| 118 | 3 | 3 | arrow 2, squish 1 |
| 11 | 3 | 3 | arrow 1, squish 1, fall 1 |
| 42 | 4 | 3 | squish 2, arrow 1 |
| 59 | 3 | 3 | arrow 3 |
| 58 | 3 | 3 | arrow 3 |
| 6 | 4 | 3 | squish 1, self 1, arrow 1 |
| 34 | 4 | 3 | squish 2, fall 1 |
| 65 | 3 | 3 | squish 2, arrow 1 |
| 96 | 3 | 3 | squish 3 |
| 119 | 4 | 3 | squish 2, arrow 1 |
| 122 | 3 | 3 | arrow 2, squish 1 |
| 39 | 3 | 3 | self 1, arrow 1, environment 1 |
| 100 | 4 | 3 | squish 2, arrow 1 |


## Tags on agent arrow deaths by stratum

| tag | all arrow deaths | A | B | C | D |
|---|---|---|---|---|---|
| shooter-airborne | 69 (62%) | 6 | 62 | 0 | 1 |
| dash-in | 68 (61%) | 5 | 60 | 0 | 3 |
| victim-dash-on-cooldown | 55 (49%) | 4 | 51 | 0 | 0 |
| from-above | 55 (49%) | 5 | 50 | 0 | 0 |
| victim-jumped-into-it | 32 (29%) | 3 | 29 | 0 | 0 |
| victim-airborne | 27 (24%) | 4 | 23 | 0 | 0 |
| victim-just-dashed | 23 (21%) | 1 | 22 | 0 | 0 |
| from-below | 18 (16%) | 1 | 15 | 0 | 2 |


## Pre-death window (120 ticks) of the AGENT: rounds the agent lost vs won

| stratum | outcome | n | distance mean | distance min | airborne fraction | dash ready end | platform ready end | bow ready end | arrows fired window | opponent arrows window | ticks since last shot | ticks since last dash | height advantage end | speed mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | agent lost | 248 | 19.4 | 11.4 | 0.0 | 41% | 13% | 29% | 0.0 | 1.0 | 177.5 | 289.0 | -5.7 | 0.0 |
| all | agent won | 77 | 21.5 | 8.3 | 0.2 | 29% | 14% | 8% | 1.0 | 0.0 | 32.5 | 207.5 | 1.1 | 6.5 |
| A | agent lost | 18 | 18.5 | 6.6 | 0.2 | 39% | 6% | 22% | 0.0 | 1.0 | 148.0 | 339.0 | -7.8 | 6.1 |
| A | agent won | 4 | 12.2 | 3.9 | 0.2 | 25% | 50% | 0% | 1.0 | 0.0 | 35.5 | 340.5 | -1.7 | 3.3 |
| B | agent lost | 185 | 18.4 | 10.7 | 0.0 | 45% | 14% | 27% | 0.0 | 1.0 | 179.5 | 294.5 | -5.5 | 0.0 |
| B | agent won | 52 | 20.7 | 7.9 | 0.3 | 37% | 15% | 8% | 1.0 | 0.0 | 23.5 | 215.5 | -0.0 | 6.5 |
| C | agent lost | 42 | 24.1 | 19.8 | 0.0 | 21% | 12% | 38% | 0.0 | 0.0 | 173.5 | 186.0 | -8.3 | 1.9 |
| C | agent won | 19 | 24.4 | 17.0 | 0.2 | 11% | 0% | 11% | 1.0 | 0.0 | 108.5 | 173.0 | 7.4 | 6.7 |
| D | agent lost | 3 | 22.3 | 15.0 | 0.0 | 100% | 33% | 33% | 1.0 | 1.0 | 26.0 | 629.0 | 14.6 | 0.0 |
| D | agent won | 2 | 10.3 | 5.0 | 0.3 | 0% | 50% | 0% | 1.0 | 0.0 | 20.0 | 160.5 | 6.2 | 6.7 |


## Shot style: every arrow release and every kill shot

| shooter | shots | dash<=60 before release | airborne | range<10 | kill shots | dash<=60 | dash<=13 (unreactable) | airborne | point-blank |
|---|---|---|---|---|---|---|---|---|---|
| human (all) | 572 | 391 (68%) | 398 (70%) | 111 (19%) | 112 | 68 (61%) | 8 (7%) | 69 (62%) | 26 (23%) |
| agent (all) | 865 | 88 (10%) | 279 (32%) | 128 (15%) | 46 | 6 (13%) | 0 (0%) | 19 (41%) | 16 (35%) |
| human (A) | 53 | 37 (70%) | 41 (77%) | 11 (21%) | 9 | 5 (56%) | 1 (11%) | 6 (67%) | 2 (22%) |
| agent (A) | 65 | 8 (12%) | 17 (26%) | 10 (15%) | 2 | 1 (50%) | 0 (0%) | 1 (50%) | 1 (50%) |
| human (B) | 503 | 342 (68%) | 346 (69%) | 98 (19%) | 100 | 60 (60%) | 7 (7%) | 62 (62%) | 23 (23%) |
| agent (B) | 637 | 56 (9%) | 195 (31%) | 92 (14%) | 37 | 4 (11%) | 0 (0%) | 15 (41%) | 13 (35%) |
| human (C) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| agent (C) | 141 | 21 (15%) | 63 (45%) | 20 (14%) | 5 | 1 (20%) | 0 (0%) | 3 (60%) | 2 (40%) |
| human (D) | 16 | 12 (75%) | 11 (69%) | 2 (12%) | 3 | 3 (100%) | 0 (0%) | 1 (33%) | 1 (33%) |
| agent (D) | 22 | 3 (14%) | 4 (18%) | 6 (27%) | 2 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| unity: archer-v2-hard (seat 0) | 407 | 32 (8%) | 193 (47%) | 26 (6%) | 31 | 2 (6%) | 0 (0%) | 13 (42%) | 2 (6%) |
| unity: evasive-v2-hard (seat 0) | 691 | 8 (1%) | 195 (28%) | 4 (1%) | 38 | 0 (0%) | 0 (0%) | 8 (21%) | 0 (0%) |
| unity: mixed-v2-hard (seat 0) | 498 | 47 (9%) | 235 (47%) | 58 (12%) | 37 | 1 (3%) | 0 (0%) | 18 (49%) | 8 (22%) |
| unity: o3-032 (seat 0) | 495 | 7 (1%) | 18 (4%) | 14 (3%) | 7 | 1 (14%) | 0 (0%) | 2 (29%) | 4 (57%) |
| unity: o7-000 (seat 0) | 771 | 76 (10%) | 333 (43%) | 77 (10%) | 36 | 7 (19%) | 0 (0%) | 17 (47%) | 22 (61%) |
| unity: o7-007 (seat 0) | 1625 | 16 (1%) | 53 (3%) | 34 (2%) | 4 | 0 (0%) | 0 (0%) | 0 (0%) | 1 (25%) |
| unity: o7-041 (seat 0) | 484 | 35 (7%) | 75 (15%) | 79 (16%) | 24 | 2 (8%) | 0 (0%) | 6 (25%) | 12 (50%) |
| unity: pursuer-v2-hard (seat 0) | 339 | 27 (8%) | 166 (49%) | 32 (9%) | 30 | 2 (7%) | 0 (0%) | 19 (63%) | 8 (27%) |
| unity: o7-000 (seat 1) | 1865 | 166 (9%) | 621 (33%) | 269 (14%) | 81 | 9 (11%) | 1 (1%) | 35 (43%) | 37 (46%) |
| unity: o7-016 (seat 1) | 2709 | 403 (15%) | 1034 (38%) | 185 (7%) | 65 | 21 (32%) | 2 (3%) | 33 (51%) | 12 (18%) |
| unity: o7-041 (seat 1) | 1075 | 49 (5%) | 118 (11%) | 204 (19%) | 146 | 9 (6%) | 3 (2%) | 28 (19%) | 36 (25%) |


## Bow-draw commitment window

| drawer | draws | released | median release tick | release histogram (10-tick bins) | died drawing | median window | opp. distance at commit | opp. dash ready at commit | opp. dashed in window | opp. airborne at commit | opp. drawing at commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agent (all) | 902 | 865 | 32 | 10:310 20:49 30:181 40:17 50:26 60:282 | 38 | 33 | 12.9 | 18 (47%) | 3 (8%) | 27 (71%) | 26 (68%) |
| human (all) | 586 | 572 | 51 | 10:48 20:40 30:64 40:114 50:119 60:58 70:31 80:15 90:14 100:69 | 11 | 21 | 7.8 | 4 (36%) | 0 (0%) | 6 (55%) | 8 (73%) |
| agent (A) | 68 | 65 | 32 | 10:29 20:1 30:12 40:3 50:2 60:18 | 3 | 37 | 8.6 | 1 (33%) | 1 (33%) | 3 (100%) | 2 (67%) |
| human (A) | 53 | 53 | 50 | 10:4 20:7 30:1 40:13 50:12 60:4 70:5 90:1 100:6 | 0 | - | - | 0 | 0 | 0 | 0 |
| agent (B) | 667 | 637 | 32 | 10:221 20:33 30:142 40:12 50:15 60:214 | 33 | 30 | 12.8 | 16 (48%) | 2 (6%) | 22 (67%) | 24 (73%) |
| human (B) | 517 | 503 | 51 | 10:42 20:32 30:63 40:96 50:103 60:53 70:25 80:15 90:13 100:61 | 11 | 21 | 7.8 | 4 (36%) | 0 (0%) | 6 (55%) | 8 (73%) |
| agent (C) | 145 | 141 | 32 | 10:47 20:13 30:23 40:2 50:9 60:47 | 2 | 46 | 31.6 | 1 (50%) | 0 (0%) | 2 (100%) | 0 (0%) |
| human (C) | 0 | 0 | - |  | 0 | - | - | 0 | 0 | 0 | 0 |
| agent (D) | 22 | 22 | 13 | 10:13 20:2 30:4 60:3 | 0 | - | - | 0 | 0 | 0 | 0 |
| human (D) | 16 | 16 | 49 | 10:2 20:1 40:5 50:4 60:1 70:1 100:2 | 0 | - | - | 0 | 0 | 0 | 0 |
| unity: agents vs archer-v2-hard | 421 | 418 | 10 | 10:293 20:13 30:34 40:9 50:11 60:58 | 2 | 42 | 11.3 | 0 (0%) | 0 (0%) | 1 (50%) | 2 (100%) |
| unity: agents vs evasive-v2-hard | 824 | 812 | 60 | 10:244 20:30 30:59 40:34 50:36 60:409 | 10 | 40 | 67.6 | 8 (80%) | 0 (0%) | 2 (20%) | 1 (10%) |
| unity: agents vs mixed-v2-hard | 511 | 507 | 10 | 10:356 20:16 30:48 40:9 50:11 60:67 | 4 | 36 | 10.3 | 0 (0%) | 0 (0%) | 2 (50%) | 3 (75%) |
| unity: agents vs o3-032 | 998 | 992 | 46 | 10:319 20:43 30:95 40:57 50:49 60:429 | 1 | 24 | 72.5 | 1 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| unity: agents vs o7-000 | 769 | 759 | 46 | 10:287 20:15 30:60 40:23 50:17 60:357 | 8 | 9 | 3.4 | 0 (0%) | 0 (0%) | 6 (75%) | 6 (75%) |
| unity: agents vs o7-007 | 897 | 894 | 53 | 10:293 20:38 30:80 40:27 50:31 60:425 | 1 | 30 | 3.9 | 1 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| unity: agents vs o7-041 | 918 | 909 | 46 | 10:295 20:41 30:92 40:32 50:34 60:415 | 8 | 8 | 12.8 | 4 (50%) | 0 (0%) | 2 (25%) | 3 (38%) |
| unity: agents vs pursuer-v2-hard | 361 | 358 | 10 | 10:250 20:6 30:30 40:8 50:8 60:56 | 2 | 12 | 3.8 | 1 (50%) | 0 (0%) | 0 (0%) | 2 (100%) |


## Death rate of a bow draw by opponent distance at commit

| drawer | opp. distance | draws | died | died | opp. dash ready | died | opp. dash not ready |
|---|---|---|---|---|---|
| agent | 0-6 | 53 | 7 (13%) | 5 (25%) | 2 (6%) |
| agent | 6-10 | 81 | 8 (10%) | 4 (10%) | 4 (10%) |
| agent | 10-15 | 146 | 6 (4%) | 1 (1%) | 5 (7%) |
| agent | 15-25 | 228 | 7 (3%) | 2 (2%) | 5 (5%) |
| agent | 25-99 | 394 | 10 (3%) | 6 (3%) | 4 (2%) |
| human | 0-6 | 42 | 4 (10%) | 2 (12%) | 2 (8%) |
| human | 6-10 | 63 | 2 (3%) | 1 (5%) | 1 (2%) |
| human | 10-15 | 99 | 3 (3%) | 1 (2%) | 2 (4%) |
| human | 15-25 | 149 | 2 (1%) | 0 (0%) | 2 (2%) |
| human | 25-99 | 233 | 0 (0%) | 0 (0%) | 0 (0%) |


## Unity agent-vs-agent rounds (same build, same taxonomy): agent (seat 1) deaths by pattern and opponent

| pattern | all | archer-v2-hard | evasive-v2-hard | mixed-v2-hard | o3-032 | o7-000 | o7-007 | o7-041 | pursuer-v2-hard |
|---|---|---|---|---|---|---|---|---|---|
| fall/jumped-off-edge | 59 (12%) | 0 | 7 | 2 | 12 | 11 | 5 | 11 | 11 |
| arrow/long-flight-dodge-failed | 58 (12%) | 10 | 16 | 11 | 2 | 5 | 3 | 4 | 7 |
| arrow/hit-while-drawing-bow | 50 (10%) | 6 | 5 | 10 | 1 | 11 | 1 | 9 | 7 |
| squish/own-platform-exit-pinch | 45 (9%) | 3 | 8 | 7 | 6 | 5 | 10 | 3 | 3 |
| squish/own-platform-drifted-into-map | 36 (7%) | 3 | 6 | 3 | 2 | 4 | 2 | 8 | 8 |
| squish/moving-map-platform | 35 (7%) | 2 | 10 | 1 | 6 | 3 | 2 | 7 | 4 |
| arrow/long-flight-not-dodged | 26 (5%) | 4 | 12 | 5 | 0 | 0 | 0 | 1 | 4 |
| fall/ground-reached-water | 26 (5%) | 3 | 3 | 2 | 3 | 2 | 5 | 5 | 3 |
| arrow/point-blank | 26 (5%) | 1 | 0 | 5 | 2 | 10 | 0 | 7 | 1 |
| self/own-arrow | 23 (5%) | 3 | 3 | 2 | 5 | 4 | 2 | 2 | 2 |
| arrow/short-range | 17 (3%) | 5 | 0 | 4 | 0 | 2 | 0 | 2 | 4 |
| squish/own-platform-pushed-into-victim | 15 (3%) | 1 | 0 | 1 | 2 | 2 | 5 | 3 | 1 |
| arrow/hit-while-platform-morph | 14 (3%) | 3 | 2 | 1 | 1 | 2 | 0 | 0 | 5 |
| squish/foe-platform-dropped-on-victim | 10 (2%) | 2 | 0 | 6 | 0 | 1 | 0 | 0 | 1 |
| squish/own-platform-exit-pinch-contested | 9 (2%) | 1 | 2 | 1 | 3 | 0 | 0 | 2 | 0 |
| self/own-arrow-mortar-returned | 9 (2%) | 1 | 1 | 2 | 0 | 1 | 0 | 2 | 2 |
| arrow/hit-mid-dash | 9 (2%) | 0 | 3 | 0 | 1 | 3 | 0 | 1 | 1 |
| fall/walked-or-knocked-off | 8 (2%) | 0 | 2 | 1 | 1 | 1 | 2 | 1 | 0 |
| squish/map-platform-other | 8 (2%) | 2 | 1 | 1 | 1 | 0 | 2 | 0 | 1 |
| arrow/point-blank-after-dash | 5 (1%) | 1 | 0 | 1 | 0 | 2 | 0 | 0 | 1 |
| environment/level-hazard | 4 (1%) | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| arrow/dash-in-shot | 2 (0%) | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| squish/victim-moved-into-foe-platform | 2 (0%) | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| squish/foe-platform-rammed-victim | 1 (0%) | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| fall/shoved-off | 1 (0%) | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **all** | 498 | 53 | 82 | 67 | 48 | 70 | 40 | 70 | 68 |


### seat-0 opponent deaths by pattern (how the bots / pool members die to the agent)

| pattern | all | archer-v2-hard | evasive-v2-hard | mixed-v2-hard | o3-032 | o7-000 | o7-007 | o7-041 | pursuer-v2-hard |
|---|---|---|---|---|---|---|---|---|---|
| arrow/hit-while-drawing-bow | 132 (25%) | 26 | 28 | 30 | 10 | 6 | 5 | 6 | 21 |
| arrow/point-blank | 62 (12%) | 16 | 3 | 10 | 7 | 3 | 3 | 2 | 18 |
| fall/ground-reached-water | 43 (8%) | 12 | 2 | 11 | 0 | 3 | 0 | 4 | 11 |
| squish/own-platform-exit-pinch | 40 (7%) | 0 | 0 | 0 | 12 | 6 | 19 | 3 | 0 |
| fall/jumped-off-edge | 38 (7%) | 4 | 2 | 1 | 0 | 17 | 5 | 7 | 2 |
| arrow/long-flight-dodge-failed | 36 (7%) | 7 | 1 | 1 | 8 | 7 | 5 | 3 | 4 |
| squish/moving-map-platform | 26 (5%) | 3 | 2 | 1 | 6 | 3 | 9 | 2 | 0 |
| squish/own-platform-drifted-into-map | 19 (4%) | 0 | 0 | 0 | 7 | 2 | 8 | 2 | 0 |
| arrow/short-range | 15 (3%) | 3 | 1 | 3 | 3 | 0 | 0 | 4 | 1 |
| squish/own-platform-exit-pinch-contested | 15 (3%) | 0 | 0 | 0 | 7 | 1 | 5 | 1 | 1 |
| arrow/dash-in-shot | 14 (3%) | 1 | 0 | 3 | 3 | 0 | 1 | 4 | 2 |
| arrow/hit-while-platform-morph | 13 (2%) | 2 | 1 | 4 | 0 | 0 | 2 | 1 | 3 |
| squish/own-platform-pushed-into-victim | 12 (2%) | 2 | 1 | 2 | 0 | 3 | 2 | 0 | 2 |
| squish/victim-moved-into-foe-platform | 12 (2%) | 3 | 0 | 1 | 0 | 2 | 2 | 1 | 3 |
| squish/map-platform-other | 10 (2%) | 3 | 1 | 4 | 0 | 2 | 0 | 0 | 0 |
| arrow/long-flight-not-dodged | 9 (2%) | 0 | 2 | 1 | 2 | 0 | 3 | 1 | 0 |
| self/own-arrow | 9 (2%) | 2 | 0 | 1 | 0 | 1 | 3 | 1 | 1 |
| environment/level-hazard | 7 (1%) | 1 | 2 | 1 | 0 | 2 | 0 | 1 | 0 |
| self/own-arrow-mortar-returned | 7 (1%) | 0 | 0 | 2 | 2 | 1 | 1 | 0 | 1 |
| arrow/hit-mid-dash | 6 (1%) | 3 | 0 | 0 | 2 | 0 | 0 | 0 | 1 |
| arrow/point-blank-after-dash | 5 (1%) | 0 | 1 | 1 | 0 | 0 | 1 | 2 | 0 |
| squish/foe-platform-rammed-victim | 3 (1%) | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| squish/foe-platform-dropped-on-victim | 3 (1%) | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| squish/carried-into-map-on-foe-platform | 1 (0%) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| fall/walked-or-knocked-off | 1 (0%) | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| **all** | 538 | 90 | 47 | 77 | 69 | 60 | 75 | 45 | 75 |


### agent deaths by family: training maps 0-2 vs unseen maps, per agent

| agent | maps | rounds | agent lost | draws | agent deaths | arrow | squish | fall | self | own-platform squish |
|---|---|---|---|---|---|---|---|---|---|---|
| o7-000 | maps 0-2 | 96 | 40 (42%) | 7 | 40 | 28 (70%) | 5 (12%) | 5 (12%) | 2 (5%) | 0 (0%) |
| o7-000 | maps 3-4 | 64 | 36 (56%) | 2 | 36 | 14 (39%) | 15 (42%) | 7 (19%) | 0 (0%) | 11 (31%) |
| o7-000 | unseen | 224 | 126 (56%) | 9 | 126 | 53 (42%) | 43 (34%) | 23 (18%) | 6 (5%) | 36 (29%) |
| o7-016 | maps 0-2 | 96 | 43 (45%) | 9 | 43 | 20 (47%) | 10 (23%) | 5 (12%) | 8 (19%) | 2 (5%) |
| o7-016 | maps 3-4 | 64 | 34 (53%) | 2 | 34 | 14 (41%) | 10 (29%) | 8 (24%) | 2 (6%) | 6 (18%) |
| o7-016 | unseen | 224 | 125 (56%) | 24 | 125 | 42 (34%) | 50 (40%) | 18 (14%) | 13 (10%) | 30 (24%) |
| o7-041 | maps 0-2 | 96 | 19 (20%) | 13 | 19 | 12 (63%) | 2 (11%) | 5 (26%) | 0 (0%) | 0 (0%) |
| o7-041 | maps 3-4 | 64 | 16 (25%) | 3 | 16 | 12 (75%) | 2 (12%) | 2 (12%) | 0 (0%) | 2 (12%) |
| o7-041 | unseen | 224 | 59 (26%) | 47 | 59 | 12 (20%) | 24 (41%) | 21 (36%) | 1 (2%) | 18 (31%) |
