# Probe dataset and probing code

Code for the representation probes of the Bopl v3 policy (`PROBES.md` in the
parent directory has the results). The dataset lives in
`native/artifacts/rl/probes/dataset/` (gitignored); this file is also copied
there as its schema README.

## Pipeline

| step | script | what |
|---|---|---|
| 0 | `build.py` | compiles `probe_binding.c` (+ `bopl_rl.c`) into `libbopl_probe.so`: idle-fork futures, geometry, raycasts, undelayed encode |
| 1 | `generate.py [--heldout]` | plays the schedule in the Rust sim (4 workers), one `.npz` per game under `games/` |
| 2 | `merge.py` | concatenates the games into memmapped `.npy` arrays + `index.json`, assigns the 80/20 train/test split by game |
| 3 | `activations.py --tag T [--checkpoint C] [--zero-pos]` | runs the checkpoint's encoder + MinGRU offline over every (game, player) sequence, writes `acts-T.npy` |
| 4 | `probe_fit.py` | linear + grouped-MLP probes from feature sets to labels, `results/probes-T.json` |
| 5 | `critic.py`, `row_probe.py`, `perturb.py`, `bench.py`, `report.py` | critic EV/calibration, per-arrow-row probe, absolute-position sensitivity, raycast CPU cost, tables/plots |

Everything is deterministic given the schedule (seeds `500000+`, held-out
`900000+`; game ids `0..2015` on maps 0-2, `10000..10503` on maps 3-4).

## Rows

One row = one player's decision (the env steps once per decision; hold lengths
1-16 ticks). Both players are recorded, whoever controls them. Rows of one
(game, player) are contiguous and in time order; games are in id order.

| file | dtype | width | contents |
|---|---|---|---|
| `obs.npy` | f16 | 1458 | the delayed v3 observation the policy saw, compact (`schema.compact`; `schema.expand` rebuilds the exact 3,690 floats) |
| `obs_now.npy` | f16 | 1458 | the same rows encoded from the *live* (undelayed) state, previous-action block copied from `obs` |
| `meta.npy` | i32 | 20 | `schema.META`: game, decision, player, controller, opponent_controller, map, seed, tick, aim, buttons, hold, latched, killer_row, killer_row_now, entity_rows, entity_rows_now, dropped, dropped_now, terminal, split (0 train / 1 test) |
| `labels.npy` | f32 | 64 | `schema.LABELS`, below |
| `rays16.npy` | f16 | 160 | 16 uniform + 4 downward rays x 8 fields, 1 hit per ray |
| `rays32.npy` | f16 | 288 | 32 + 4 rays x 8 fields |
| `rays32x3.npy` | f16 | 864 | 32 + 4 rays x 3 hits x 8 fields |
| `acts-<tag>.npy` | f16 | 769 | encoder intermediates: flat 128, pooled players/slots/entities/slimes 4x64, fused 128, GRU state 128, GRU out 128, value |

Compact observation: blocks prefix(66), self(20), players(2x20), slots(8x14),
entities(32x34), slimes(4x30), previous(12); rows beyond the kept count are
always zero in 1v1 Dash/Bow/Platform games (`dropped` counts the exceptions).

Controllers (`generate.CONTROLLERS`): 0 random, 1 `024-final`
(policy-2313256960), 2 `024-early-320m` (policy-320012288), 3-9 pool-v7
checkpoints, 10-15 scripted bots (archer/pursuer/evasive/mixed-hard,
mixed-medium, search-medium).

## Labels

From the idle fork (`probe_idle_future`: fork the live world, both players hold
no input for 60 ticks):

- `idle_death_tick` (-1 if none), `idle_cause` (0 alive, 1 environment/fall, 2 squished, 3 opponent's projectile, 4 own projectile, 5 other)
- `hit_15/30/60`: killed by a hostile projectile within N ticks; `hit_inflight_30/60`: same and the killer arrow already exists in the observed rows (`killer_row` = its entity-row index in `obs`, `killer_row_now` in `obs_now`); `killer_known`
- `fall_30/60`, `squish_60`, `self_hit_60`, `any_death_30/60`
- `own_arrow_in_flight`, `my_shot_hits_60` (own arrow in flight and the opponent dies to a projectile in the fork)
- `idle_x10/y10/x30/y30`, `idle_opp_*`: positions at t+10/t+30 under no input (NaN after death)

Geometry of the live state (`probe_geometry`, world units): `grounded`,
`drop` (height above the surface below), `support` (signed distance to the
supporting platform's nearest edge), `edge_distance`, `lands_dry`,
`lands_dry_any_steer`, `terrain_within_dash`, `terrain_min_ray`, own and
opponent velocity, `foe_grounded`, opponent cooldown readiness/fractions.

Live kinematics: `x`, `y`, `opp_x`, `opp_y`, `opp_alive`.

Realised future along the actual rollout: `real_x12/y12`, `real_opp_x12/y12`,
`real_opp_x32/y32` (12/32 ticks ahead, NaN past the end), `real_death_60`,
`real_death_tick`, `real_cause`, `win`, `loss`, `timeout`, `ticks_to_end`,
`reward` (this decision's env reward: outcome + kill/death shaping),
`return_disc` (gamma = 0.995 per decision, the training gamma), `value_online`
(the controller's own value head, NaN for bots).

## Rays (`probe_rays`)

Per ray, 8 fields: distance/range (1 if nothing), one-hot hit type over 6
classes [terrain/wall, water/kill zone, hostile projectile, opponent body, own
slime or own object, nothing within range], the hit's velocity *relative to the
observer's slime* projected on the ray (/64, clipped; negative = closing). For
static terrain this is minus the observer's own velocity along the ray, so the
ray set also encodes own velocity. Rays start at the player's centre;
the N uniform rays start at angle 0 and the extra downward rays are spread
around -90 degrees; range 40 world units. `--hits 3` records the first three
distinct hits along each ray.
