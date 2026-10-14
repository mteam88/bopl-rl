# Team play: 2v2 training and evaluation

Asked 2026-10-14 (Matthew): "also do a bit of training on friendly/team play".
The best net, f5 (league `final5-18666m`, all-combos Elo 669), was trained
only in 1v1. The in-game picker can now put nets on teams, so a net has to
act sensibly next to a teammate: it should not kill its ally, and its team
should win. This document is the plan, what changed, and what was measured.

## What already worked

**The observation is already team-aware, so nothing changes in the layout.**
Layout v4 (`train/bopl_rows_v4.h`) marks allies everywhere it marks players:

- each player row has `out[2] = same team`;
- each slime row has `bopl_ally`;
- each entity row carries `ally`.

The primary enemy is the nearest player on another team, and the critic's
privileged enemy is the first player on another team. Rays class only enemy
bodies as the opponent body. A teammate's projectile counts as hostile, like
every projectile not owned by the agent. f5's weights load unchanged.

f5 has seen the ally bits only on its own row. In 1v1 every other player is an
enemy, so the bits were constant there, and how f5 reads them is untrained.

**The in-game shim needs no change.** `train/bopl_agent.h` (used by
`RlAgent.cs`) runs the same row writer on the real game's state, and that
state already carries the game's teams.

**The Rust sim plays teams.** It has `config.player[i].team` and team spawns.
`Player.Kill` credits a teamkill like any kill: the killer's `kills` goes up
and so does the victim's `deaths_opponent`. The win flag (`players[i].flags &
2`) is set on every player of the winning team, including dead ones.

## What changed

**Reward** (`train/bopl_puffer.h`, `BoplPuffer.team_play`):

- A kill of a teammate is not paid as a kill. It is charged like a death:
  shaping is `reward_kill * (kills - deaths - team_kills)`, and `kills`
  excludes team kills.
- `bopl_team_kills` reads team kills from the game's death events
  (`BOPL_EVENT_DEATH`, actor and subject on one team, self-kills excluded).
  The counters cannot tell which victim a kill was.
- The outcome stays the game's ±1 through the team win flag, so both teammates
  share it.
- A backend without events fails the env instead of scoring a team game wrong.
- With `team_play = 0` every line of this is a no-op, because `team_kills` is
  all zero.

**Environments** (`train/binding.c`, `train/native_vector.py`,
`train/train.py --team-every k`):

- With `historical_every 2`, a group is one self-play environment plus one
  environment against a frozen opponent.
- With `--team-every k`, the self-play environment of every k-th group is a
  2v2 instead: four learner players, seats 0,1 against seats 2,3.
- Games against the frozen banks stay 1v1.
- At `k = 2`, each buffer has 64 learner rows in team games out of 128. The
  other 64 play 1v1: 32 in self-play and 32 against the frozen banks.
- `NativeVector` packs the extra learner rows into the primary slice.
- The trainer's per-rollout accounting counts the extra players
  (`physical_per_step`).
- The team metrics `team_episodes`, `team_wins` and `team_kills` are logged
  when non-zero.

**Scripted bots** (`train/bopl_bots.h` `bot_view`): the foe is now the first
living player on another team. It used to be the first other living player,
so a bot could pick its teammate. In 1v1 each player is its own team, so the
change does not alter any 1v1 game.

**Evaluator:**

- `train/eval_binding.c` adds `research_open_teams` (N players with given
  teams) and `research_set_kits` (per-player kits).
- `eval/team_games.py` plays A+A vs B+B, or A+P vs B+P with `--partner`:
  - Each seed pair is played twice with the teams swapped.
  - There are 12 random kits drawn from the all-combos pool.
  - It uses the league maps, ticks and delay.
  - It reports A's team score with a seed-pair bootstrap 95% CI, team kills
    per game per model, and the share of games with a team kill on each side.

## Proofs

- **1v1 golden.**
  - What it plays: f5 vs A20 through `research_open` on 9 fixed games. In 3 of
    them player 1 is a scripted bot.
  - What it hashes: every decision's observation rows, mask, telemetry
    (including the reward columns) and game state.
  - Libraries compared: the base library (95e4a2f0) and the team-play library.
    The digests are identical (`eval/team_golden.py`; 13,148 decisions, run in `sweeps/team-play/golden/`).
- **`make check-routing`.** It now includes 2v2 trainer cases:
  - the slots follow the permutation;
  - the learner rows fill the primary slice;
  - each team row sees exactly its teammate as an ally, mirrored;
  - the counters match.
- **`make check-reward`** passes.
- **`make ability-sweep`:**
  - 24,800 episodes;
  - 4588/4588 cells;
  - 0 crashes;
  - 0 divergent.

## Experiment

The candidate is team-1 (`sweeps/team-1/run.sh`):

- It starts from f5 and uses the final-5/6 recipe: a half-LR cosine anneal,
  the same four frozen opponents, PFSP p 3, `--team-every 2`, and 5B learner
  steps.
- It is compared with f5 on the same seeds:
  - **team strength:** T+T vs A20+A20, against f5+f5 vs A20+A20;
  - **head to head:** T+T vs f5+f5;
  - **as an ally:** T+A20 vs f5+A20;
  - **teamkill rates** in all of the above;
  - **1v1 regression:** all-combos T vs f5 over 250+ pairs, and the probe vs
    random.

Results are below.
