# Playing against a trained policy in the real game

The Linux Unity build (`Builds/Linux/BoplBattle.x86_64`, Mono) can add one AI
player to a local match. The AI is either a training checkpoint
(`policy-*.bin` + its `.bin.json` sidecar) or a scripted bot (`bopl_bots.h`),
and it is driven by the **training decision loop itself**, compiled into a small
shared library that the game feeds with its own state once per simulation tick.
Nothing about the policy, its observation, its action cadence or its sampler is
re-implemented on the Unity side.

```
Tools/play-vs-agent.sh <checkpoint.bin | scripted-name> [--seed N]
Tools/play-vs-agent.sh                       # default agent (native/artifacts/rl/human/default-agent, else mixed-v2-hard)
Tools/play-vs-agent.sh --install-desktop     # ~/.local/share/applications/bopl-battle-vs-agent.desktop
```

e.g.

```
Tools/play-vs-agent.sh /path/to/sweeps/anchors/run007-final.bin --seed 7
Tools/play-vs-agent.sh mixed-v2-hard
```

The launcher starts the game the Sunshine/Moonlight-safe way (as `play.sh` /
`launch-desktop.sh`: `-force-glcore -screen-fullscreen 0 -screen-width 2560
-screen-height 1440`, then `wmctrl` maximize + `xdotool windowactivate`, on
`DISPLAY=:0` with the Xwayland `XAUTHORITY`), with

```
BOPL_AGENT=<spec>  BOPL_AGENT_SEED=<N>  BOPL_AGENT_RESULTS=<session file>
```

Join with the gamepad in the character select as usual, pick anything, ready,
start. The AI player needs no device and no select box; it is appended after
the human players when the match starts (`CharacterSelectHandler.TryStartGame_inner`
→ `RlAgent.AddPlayers`). Starting with a single human is allowed by the stock
readiness rule (`CharacterSelectHandler.Update`, `num == 1`). Between rounds
the human presses start/jump as usual; the AI never presses anything in menus.
Unset `BOPL_AGENT` and the game is stock (every hook is a no-op).

### macOS

Same thing natively on the Mac (arm64, Metal, Mono — `BuildScript.BuildMac`
forces Mono because the Mac editor has no IL2CPP module):

```
just sync-mac && just build-mac                       # from the PowerSpec; rsync + batchmode build over ssh
Tools/play-vs-agent-mac.sh [<checkpoint.bin | scripted-name>] [--seed N]   # on the Mac
```

The launcher builds `native/artifacts/rl/libbopl_agent.dylib` on demand
(`make -C native/rl agent-mac`), copies it into
`BoplBattle.app/Contents/Plugins/` (ad-hoc codesigned) and execs the binary
directly (`open` only focuses a running copy). `sync-mac` excludes
`native/artifacts/`, so checkpoints and `human/default-agent` (with the Mac
path) are copied separately. Results and replays land in the Mac's
`native/artifacts/rl/human/`; Player.log is
`~/Library/Logs/Johan Gronvall/BoplBattle/Player.log`.

## What the AI joins as

`Assets/Scripts/Assembly-CSharp/RlAgent.cs`:

- `Player` with `IsAgent = true`, `IsLocalPlayer = false` (so
  `GameSessionHandler.SpawnPlayers` creates no `InputUpdater` for it and
  achievements ignore it), `inputDevice = null` (every menu that casts the device
  goes through the null-safe `Gamepad.IsUsable_()` extension), the first team and
  colour no human took, id after the humans.
- Loadout: the checkpoint's training loadout from the sidecar
  (`config.native.env.ability1..3`), or the training default for bots —
  catalog indices **2, 4, 14 = Dash, Bow, Platform** (`binding.c` / `bopl.ini` /
  `eval_binding.c research_open`). Always three slots regardless of the
  Settings "number of abilities". Ability slots and icons come from
  `SteamManager.instance.abilityIcons.sprites`, the same catalog `RlBridge.Reset`
  resolves RL ability indices with.
- Ability pickups (`AbilitySpawner`) are disabled for the round, as the training
  host does (`sim/Host/RlMode.cs`), unless `BOPL_RL_ABILITY_SPAWNS=1`.

## Timing (matches training exactly)

Per level scene the game has its own `SimUpdater` (`Updater.Update`, autoUpdate);
`RlAgent.BeforeTick/AfterTick` wrap every `Updater.TickSimulation` call. Ticks
are counted per `TickSimulation` call since the level's first one, which is the
simulation timeline the training sims, `RlBridge` and the replay share
(`Updater.Update` skips the call on hitstop frames without simulating anything;
`TickSimulationInner` records one replay frame per call).

- Level ticks `0..239` are the **training warmup** (`warmup_ticks = 240`,
  `RlBridge.PumpInner` runs them with all-zero inputs): every player, the human
  included, gets a zero `InputPacket` through `Player.OverrideInputWithNetworkInput`,
  and `Host.CurrentInputAsPacket` reports those zeros so the replay records
  what was applied. At level tick 240 the humans get their device profile back
  (`Player.RestoreLocalInputs`).
- From level tick 240 on, the state before the tick (`RlObservation.Write`, the
  record `RlBridge.Observe` sends, ABI v2) with `episode_ticks = level tick − 240`,
  `simulation_time = level tick × FixedTimeStep` (what `SimTimePassed` is under
  `RlBridge`, where no hitstop frames exist), flags STARTED once
  `gameInProgress` has been seen and TERMINAL when the round has ended, goes to
  the shim, which answers with the packet the agent applies to that tick. This is
  the point where `RlBridge.Tick` applies training actions
  (`OverrideInputWithNetworkInput` right before `TickSimulation`).
- Decision `k` is made at episode tick `4k` from the published observation and
  held for ticks `4k+1..4k+4`; the observation for decision `k+1` is sampled from
  the state after tick `4k+3` and published nine ticks late (two-slot history),
  i.e. `obs_k = f(S_{4k−9})` for `k ≥ 3`, `obs_0..2 = f(S_0)`. All of this is
  `bopl_puffer.h` code (see below), not a re-statement of it.
- The round end (`GameSessionHandler.HasGameEnded()` after `gameInProgress`) is
  handed to the shim as the terminal state so it closes the decision like
  `research_step_finish`. Training truncated at `max_episode_ticks = 7200`; the
  game keeps playing (the episode-progress feature `obs[2]` clips at 1), and the
  stock 120 s sudden death applies as in the game.

## The shim: `native/rl/puffer/bopl_agent.h` → `libbopl_agent.so`

Build: `make -C native/rl agent` (output `native/artifacts/rl/libbopl_agent.so`;
links `puffer/bopl_agent.c`, `bopl_rl.c` and `puffer/policy_eval.c` against the
pinned PufferLib `src`). The launcher copies it to
`Builds/Linux/BoplBattle_Data/Plugins/x86_64/libbopl_agent.so`, where Mono
resolves `DllImport("bopl_agent")`, and also puts that directory on
`LD_LIBRARY_PATH`. The library is a build artifact, so it is not checked into
`Assets/Plugins` (a rebuilt `.so` would otherwise require a Unity rebuild).

C ABI (`bopl_agent_abi_version() == 1`):

```c
BoplAgent *bopl_agent_open(const char *spec, int players, int player, char *error, size_t capacity);
void       bopl_agent_reset(BoplAgent *, uint32_t seed);                         // per round
int        bopl_agent_tick(BoplAgent *, const BoplRlState *, BoplRlAction *out); // 0 ok, 1 episode ended, -1 misuse
int        bopl_agent_tick_bytes(BoplAgent *, const void *bytes, size_t length, BoplRlAction *out); // RlObservation wire format; -2 malformed
void       bopl_agent_close(BoplAgent *);
int  bopl_agent_ability(const BoplAgent *, int slot);  int bopl_agent_warmup_ticks(const BoplAgent *);
int  bopl_agent_repeat(const BoplAgent *);  int bopl_agent_observation_delay(const BoplAgent *);
int  bopl_agent_max_episode_ticks(const BoplAgent *);  const char *bopl_agent_id(const BoplAgent *); // "policy:<sha256>" | "bot:<name>"
int  bopl_agent_is_policy(const BoplAgent *);
const float *bopl_agent_observation(const BoplAgent *);  const unsigned char *bopl_agent_mask(const BoplAgent *);
const float *bopl_agent_logits(const BoplAgent *);  void bopl_agent_decision(const BoplAgent *, int out[3]);
int  bopl_agent_decisions(const BoplAgent *);  int bopl_agent_obs_size(void);  int bopl_agent_logits_size(void);
```

`spec` is a checkpoint path ending in `.bin` (sidecar `<path>.json` required:
architecture `bopl-set-mingru-v2` or `puffernet-mingru`, `obs_size`, `hidden`,
`num_layers`, `action_sizes`, `precision float32`, `weights_sha256` — verified
against the file — plus `observation_delay_ticks`, `action_repeat`,
`warmup_ticks`, `max_episode_ticks`/`episode_ticks`, `ability1..3`) or a bot
name `<archer|pursuer|evasive|mixed>[-v2]-<easy|medium|hard>`. `search-*` is
refused with a message: it plans by forking the in-process simulator every
decision and cannot follow a game it does not step.

What the shim reuses (none of it re-implemented):

- `BoplPuffer` state, `compute_observations_from` (observation encoder),
  `bopl_previous_action` (own-action tail), `bopl_latch` / `BOPL_HOLD_LENGTHS`
  (hold-length latch), `bopl_write_mask` (action mask), the nine-tick delayed
  observation history and `bopl_step_sample` — via `bopl_reset_observations`,
  `bopl_step_begin_from`, `bopl_step_finish_from`. Those three are the
  state-parameterised halves the training functions were split into
  (`c_reset`, `bopl_step_begin`, `bopl_step_finish` are now thin wrappers that
  read the state from `env->world`; training behaviour is unchanged). The tick
  split (`begin` at episode tick `4k`, `sample` at `4k+3` or the terminal state,
  `finish` at `4k+4` or the terminal state) is `eval_binding.c`'s
  `research_step_begin/tick/finish` with the world replaced by the caller's state.
- Scripted bots: `bopl_bot_reset(kind, level, bopl_bot_seed(env, player))` and
  `bopl_bot_decide` inside `bopl_step_begin_from`, exactly as `research_bot` pins
  them; `config.seed` is the round seed passed to `bopl_agent_reset`.
- Policy: `policy_eval.c` (`bopl_policy_eval_create[_set]`, `_step`, `_close`,
  batch 1, recurrent state zeroed per round as `evaluate.py` / `NativePolicy.initial`).
- Sampler: per head, softmax over the masked logits (disallowed → −∞) and one
  categorical draw — the distribution `evaluate.py`'s `torch.multinomial` samples.
  The draw uses the shim's own splitmix64 stream seeded from the round seed, so
  a match is reproducible from `(spec, seed, states)` but does **not** reproduce
  torch's random stream for the same seed (different RNG, same distribution).

Python view: `native/rl/research/agent_shim.py` (`Agent(spec, players, player)`,
`.reset(seed)`, `.tick(state_or_bytes) -> (status, (angle, buttons))`,
`.observation`, `.mask`, `.logits`, `.decision`).

## Fidelity proof

`native/rl/research/check_agent_fidelity.py` (needs `make -C native/rl research agent`):

**1. `sim` — shim vs research environment, tick for tick.** A research
`Environment` (the training environment, `eval_binding.c`) is driven decision by
decision with a random legal "human" on player 0 and the shim's decisions on
player 1; after every tick the environment's state goes to the shim exactly as
the game would send it. At every decision boundary the shim's observation row
and action mask must be bit-identical to `env.obs[1]` / `env.mask[1]`, for
policies the shim's decoder output must be bit-identical to `NativePolicy.step`
on `env.obs`, and each decision's packet must be constant over its four ticks.
For bots the environment runs its own copy of the bot (`env.bot(1, kind, level)`)
from the same seed; a diverging decision would show up as a diverging own-action
tail. Rust sim backend (`BOPL_RL_GAME=native/target/release/libbopl_sim.so`),
maps 0, 1, 3, seeds 1001..1003, all rounds played to the end, zero mismatches:

| agent | ticks (3 rounds) | decisions | logits compared |
|---|---|---|---|
| `anchors/run007-final.bin` | 1504 | 377 | 377 |
| `overnight-3/012/checkpoints/policy-266698752.bin` | 2508 | 628 | 628 |
| `mixed-v2-hard` | 1367 | 343 | — |
| `archer-hard` | 1946 | 487 | — |
| `evasive-easy` | 2312 | 579 | — |
| `pursuer-v2-medium` | 613 | 154 | — |
| `mixed-easy` | 1497 | 375 | — |
| `archer-v2-medium` | 2803 | 702 | — |

```
BOPL_RL_GAME=$PWD/native/target/release/libbopl_sim.so native/artifacts/rl/venv/bin/python \
  native/rl/research/check_agent_fidelity.py sim <checkpoint.bin | bot-name> --maps 0,1,3 --seed 1001
```

(The research environment draws the episode seed from its own stream —
`init()` scrambles `env->rng` once, `bopl_draw_config` takes one xorshift32
step — the script replicates that draw to seed the shim identically.)

**2. `replay` — a real in-game `.rep` vs the research environment on the Unity
build.** The game auto-saves each round (`Host.SaveReplay`,
`~/.config/unity3d/Johan Gronvall/BoplBattle/replays/N.rep`, BOPLREP2). The
check decodes it (`replay_data.decode`), opens the same Unity player through the
RL socket (`ReplayWorld`, `config.warmup_ticks = 240` so the world runs the
neutral warmup inside its reset like training), replays the **human's** recorded
inputs, hands every tick's state to a fresh shim seeded with the round seed from
the result row, and compares the shim's packet with the agent packet the game
recorded, tick for tick, while feeding the recorded packets to the world (so the
world follows the recorded game whatever the shim says).

Idle-human runs on the real build under Xvfb (human joined with keyboard and
never moved; replays and rows are in `native/artifacts/rl/human/`):

| agent | round | level | seed | outcome | episode ticks | frames | packets compared | matched |
|---|---|---|---|---|---|---|---|---|
| `mixed-v2-hard` (4242) | 0 | 0 | 4242 | agent won (human killed at tick 273) | 273 | 513 | 273 | **273** |
| `mixed-v2-hard` | 1 | 25 | 4243 | agent won | 301 | 541 | 301 | **301** |
| `anchors/run007-final.bin` (777, via `Tools/play-vs-agent.sh --headless`) | 0 | 0 | 777 | agent won (sudden death took the idle human at episode tick 7365) | 7365 | 7605 | 7365 | **7365** |

The bot's round 1 is a local "next level" round (its first `TickSimulation`
is the `LoadedALevelThisUpdate` tick), so both level-load paths are covered;
the policy round runs past `max_episode_ticks` (7200) and past the 40 s ability
pickup time. All three worlds ended at the same tick the game did
(`ended: true`); the game ran at 60 sim ticks/s throughout (`[PERF] simTicks=300`
per 5 s in `Player.log`) with the policy evaluated in-process.

Found by this check: the Unity RL host (`RlBridge`) let `AbilitySpawner`
spawn pickups after 40 s while the .NET host (`sim/Host/RlMode.cs`) and the
Rust sim have none — the replay check diverged exactly at the first decision
that saw the pickup (episode tick 2172, level tick 2412 = 2400 + 3 + 9).
`RlBridge` now disables the spawners on scene load like the .NET host
(`BOPL_RL_ABILITY_SPAWNS=1` opts back in), which is also what `RlAgent` does
in the real game.

```
BOPL_RL_GAME=$PWD/Builds/Linux/BoplBattle.x86_64 native/artifacts/rl/venv/bin/python \
  native/rl/research/check_agent_fidelity.py replay <match.rep> <spec> --seed <row.seed> --agent-index <row.agent_index>
```

The replay check needs a build whose `SimulationReference` hash matches the
recording (the strict identity check in `ReplayWorld`); the Rust sim of this
checkout carries the base hash and refuses replays from this build.

## Result log

Each round, `RlAgent.WriteResult` appends a JSON line to `BOPL_AGENT_RESULTS`;
when the game exits the launcher runs `Tools/play_vs_agent_results.py`, which
copies the round's replay to `native/artifacts/rl/human/<session>-r<round>-<N>.rep`,
the `Player.log` to `<session>-Player.log`, and appends one row per round to
`native/artifacts/rl/human/games.jsonl`, shaped like `elo_ladder.play_games`
records so `elo_ladder.tally` rates them:

```
a="human"  b="policy:<weights sha256>" | "bot:<kind>-<level>"  map=<level>  seed=<round seed>
side=<human index, 0>  ticks=<episode ticks>  outcome="a" | "b" | "draw"
agent (spec) agent_path agent_sha256 agent_index agent_player
human_abilities agent_abilities human_keyboard players (full roster: id, team, abilities, kills, deaths, won_round, cause_of_death)
replay replay_sha256 replay_source  game game_sha256 assembly_sha256 library library_sha256
source_hash catalog_hash  warmup_ticks level_ticks decisions stocks game_mode failed round session time
```

`seed` + `agent_index` + `replay` are what the `replay` fidelity check needs.

## Rating sessions (rosters)

A spec ending in `.roster` is a text file of specs (one per line, optional
weight, `#` comments). The shim opens every member at once and draws one per
round from the round seed, so one game session spreads ~100 rounds over a
pool and every round is a Bradley-Terry observation of the human against a
rated opponent. Members must share loadout, warmup and episode length (the
game reads those once at character select). Each draw is appended to
`$BOPL_AGENT_ROSTER_LOG` (the launchers set `.session-<id>.roster.jsonl`),
and `play_vs_agent_results.py --roster-log` attributes the rounds: `games.jsonl`
rows carry the member as `agent`/`agent_sha256`/`b` plus `roster=<path>`, so
`reference_pool.rate_human` and the ladder see plain human-vs-member games.

`Tools/rating_roster.py` writes one from a pool: the scripted members (minus
`search-*`, which the shim cannot run) and the checkpoints, weighted by a
Gaussian in Elo distance from the human's current `rate_human` estimate
(floored so every member keeps appearing), with paths rewritten for the Mac
(`--mac-root`). Sync the roster and the anchors it names, then
`Tools/play-vs-agent-mac.sh native/artifacts/rl/human/rosters/<name>.roster`.

## Limitations / notes

- Training opponents carried the same fixed Dash/Bow/Platform loadout; a human
  with another loadout (or several humans) is out of the training distribution
  by construction. The agent still observes them through the same encoder.
- Hitstop frames (`Updater.Update`) freeze the picture without simulating; the
  agent is not consulted on them and its clock does not advance, which is the
  training timeline. `SimTimePassed` does advance on them in the stock game,
  which is why the observation's `simulation_time` is computed from the tick
  count instead.
- The stock local game never resets `Updater.SimulationTicks` between matches;
  the replay recorder starts from `LoadedALevelThisUpdate || SimulationTicks == 0`,
  so a second match started from the character select in the same process has
  not been exercised here (rounds within a match are fine: the round boundary is
  detected from the level scene's new `SimUpdater` instance).
- The sampler's RNG stream differs from torch's (same distribution); a shim
  match is reproducible against the shim, not against `evaluate.py` for the same
  seed.
- Online play and replay playback are untouched (`RlAgent.Active` refuses both).
- Headless driving under Xvfb needs `xdotool windowfocus <wid>` before any key
  or click reaches the game (no window manager → no focus).
