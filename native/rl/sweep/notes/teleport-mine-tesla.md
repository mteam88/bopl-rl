# Teleport / Mine / Tesla (25,24,23)

## 2026-09-21 — why 4x lost to 1x

Head-to-head (rater): 4x vs 1x 27-42-51. Ratings said "regression"; the study says why.

Study: `python rl/research/study.py --out /tmp/study-teleport --a long=<4x> --b short=<1x> --loadout 25,24,23 --pairs 3`
(18 games, all saved: .rep, physics .json, decision dumps for both sides).

Result 5 long / 7 short / 5 both-dead / 1 timeout. Cause of death across the 18 games:

| policy | alive | killed by opponent | accident (own entity or fell) |
|---|---|---|---|
| 4x (long)  | 6 | 3 | **9** |
| 1x (short) | 8 | 7 | 3 |

The 4x policy is three times more active than 1x (movement 0.30 vs 0.10 units/frame,
button presses 3x, mostly Mine) and half of its games end with it killing itself:
standing on its own mine (stationary, dies with the opponent 0.6-6 units away), or
falling off the map after a burst of teleports at ticks ~7050-7090, right before the
7200 limit. 1x mostly stands still and lets the 4x come to it.

Also seen: ~40% of games run past tick 6000 with the players 20+ units apart and no
approach. Sudden death never fires in training (map sudden_death_time = 120 s = exactly
the 7200-tick episode limit), so there is no pressure to close.

Things to try (one at a time, short run, check the accident rate):
1. Death-cause-aware reward: an own-entity/fall death costs more than being killed.
2. Shorter episode or sudden death inside the episode (e.g. 5400 ticks) so stalling loses.
3. Check Mine self-arming in the sim vs Mine.cs (does the owner get an arming grace period?).
