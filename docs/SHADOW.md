# In-game shadow sim (milestone 1 of in-game search)

Decision-time search (docs/SEARCH.md) forks a sim world in the game's state.
The `bopl_rl.h` ABI has no "set state" call, so inside the real game the only
way to get such a world is to keep one in lockstep: reset it from the round's
own setup and step it with every player's applied packets, tick for tick. This
milestone builds that shadow and checks it against the game every tick. It
does not search yet.

## What it is

- `train/bopl_shadow.{h,c}`, linked into `libbopl_agent.so`, has
  `open / begin / check_bytes / step / end / report`.
  - The sim is the q32 Rust build (`libbopl_sim_q32.so`, or `.dylib` on the
    Mac) loaded in-process. `bopl_rl_in_process` now accepts `.dylib` too.
  - The library is found from `BOPL_AGENT_SHADOW_LIB`, otherwise next to the
    shim.
  - The comparator is the replay differential's comparator, moved to
    `sim/abi/bopl_rl_compare.h`. `replay_ab.c` uses it unchanged: output is
    byte-identical on 150 games.
  - The usual options are available: tolerance, visible-only, strict, ignore
    list, flags mask, telemetry (`BOPL_AGENT_SHADOW_*`, see the header).
- `RlAgent.cs` runs it with `BOPL_AGENT_SHADOW=1`.
  - It opens the shadow with the agent.
  - After level tick 1 it resets the shadow from `ReplaySession.Recording`
    (the BOPLREP2 setup: map, seed, kits, stocks, mode, source hash). The reset
    runs on a thread, well before tick 240.
  - Every tick it checks the game's observation (the wire record the agent
    already gets), then steps the shadow with the packets that tick applied.
    The packets come from the recording's frame, in PlayerList order.
  - The terminal state is checked before `end`.
  - Log lines are `BOPL_AGENT_SHADOW: round N ...`. A desync is a
    `Debug.LogError` naming the tick, field, both values and every differing
    field. Shadowing then stops for that round.
  - The results JSONL gains `shadow`, `shadow_compared`, `shadow_report` and
    `hitstop_clock`.
- Source check: the archive's `SourceHash` must equal the library's
  `bopl_rl_source_hash`.
  - The Rust port reports the placeholder `c0debab5`×8. For that placeholder
    the game must carry the pin the port was verified against,
    `d0562ac89d03d46f24f122a08f4f5366c675cc94a015a4ccaf7adc124af089b7`.
    Override with `BOPL_AGENT_SHADOW_SOURCE=<hash>|any`.
  - On a mismatch the round is refused with
    `shadow refused: game source X, shadow LIB reports Y ...`.
- `Tools/play-vs-agent.sh` with `BOPL_AGENT_SHADOW=1` builds and copies the q32
  lib (`make agent-shadow-lib`).

## Differences the wrapper has to absorb (found live, fixed)

| symptom | cause | fix |
| --- | --- | --- |
| "recording is not this round's" on the round-end tick | `Host.SaveReplay` calls `ReplaySession.ClearRecording()` on the tick the round ends, before `RlAgent.AfterTick` | cache the recording object at round begin |
| `player[].deaths` 3 vs 0 at tick 0 of round 2+ | `Player.Kills/Deaths` count the whole match; the sims start each round at 0 | compare relative to the round's first check |
| `entity[0].hierarchy` 100000245 vs 100000061 at tick 0 of round 2+ | `Updater.PreLevelLoad` never resets `prefabsAdded` (only `ReInit` does, at match start), so prefab HierarchyNumbers carry earlier rounds' prefabs | constant offset learned at the first check, subtracted from prefab (>= 1e8) hierarchies. Hierarchy numbers are only tested for equality and sorted, so the offset doesn't change behaviour |
| `header.flags` STARTED differs | RlAgent sets STARTED after the warmup; the sims set it during the warmup | compare only TERMINAL by default (`BOPL_AGENT_SHADOW_FLAGS`) |
| telemetry differs | RlAgent never fills telemetry counters | telemetry not compared by default |

Hitstop: stock `Updater.Update` advanced `SimTimePassed`,
`SimTimeSinceLevelLoaded` and `SimulationTicks` on hitstop frames. Every sim
skips those frames entirely (.NET Native/Host/WorldsCheck call TickSimulation
directly, and Rust only stores the hitstop). While the agent is active the
clock now stays put (`RlAgent.FreezesHitstopClock`); `BOPL_AGENT_HITSTOP_TIME=1`
restores the stock clock. In this game version the fix is inert:
`DeathEffect2` is the only `Updater.HitStop` caller, and its one prefab
(`game/Assets/GameObject/deathEffect.prefab`) has `hitStopFrames: 0`. The live A/B
confirms this: with the stock clock, a round with 3 deaths stayed in sync. The
guard stays in case a hitstop is ever non-zero.

## Proof

### Offline (`make shadow-replay GAMES=...`, `eval/shadow_replay.c`)

The .NET reference sim stands in for the game. Each tick its state is
serialized to RlAgent's wire record:
- TERMINAL/STARTED flags as RlAgent sets them;
- clock = level ticks × FixedTimeStep;
- telemetry zeroed;
- warmup inputs zeroed, as RlAgent forces.

The shim's own `bopl_shadow_*` path is fed exactly as `RlAgent.cs` feeds it.
Three self tests pass on every shard:
- a wrong source hash is refused;
- packets perturbed from tick 150 on, fed to the shadow only, are caught at
  tick 150;
- prefab hierarchies shifted by +184 (as in a later live round) give the
  same result as unshifted. With the baseline disabled, this test reproduces
  the live failure exactly.

Corpus: `native/artifacts/human-replays` plus `native/artifacts/rl/human/mac`
(2026-10-12, 4 shards).

| games | in sync | desync | stand-in .NET threw | unreadable / < warmup | ticks shadowed |
| --- | --- | --- | --- | --- | --- |
| 1445 | 1339 | 14 | 1 (379.rep: NullReferenceException at episode tick 415; shadow in sync to there) | 91 | 1,754,542 |

Coverage:
- players: 2p 975, 3p 375, 4p 3, 1p 1;
- at least 114 distinct maps;
- 33 ability ids, including 1 (Random), resolved by the seeded RNG on both
  sides;
- 1323 games with deaths (up to 11 in one round);
- 12 games reaching sudden death.

All 14 desyncs land on the recording's last tick (the round-end tick). They
are .NET-vs-Rust sim divergences, not wrapper problems. 769.rep, 545.rep and
1131.rep reproduce exactly in `replay_ab` (.NET vs Rust, no shim).

| game | tick | field (game vs shadow) |
| --- | --- | --- |
| 330 | 2803 | entity[1].radius, 1 raw unit |
| 367, 370 | 1357, 1906 | slime[2].timers[1] -1.4833 vs -1.5 |
| 462 (sudden death), 475 | 7208, 5586 | player[0].alive_bodies 16 vs 9 |
| 456, 1179 | 1173, 4317 | player[0].x |
| 769 (×2), 772 (×2) | 818, 1458 | entity[0].x |
| 1131 | 1630 | entity[0].flags (CONTACT_DAMAGE bit) |
| 1184 | 459 | entity[22].right_x, 384 raw |
| 545 (1p) | 811 | slime[0].state 7 vs 1 |

### Live (real Linux build, headless)

`eval/shadow_live/`:
- `session.sh` runs Xvfb on `:97`, uinput Xbox 360 pads (`vpad.c`) and the
  game, each in its own capped user unit, with an isolated config dir;
- `human.py` is a random human on the pads.

Menu: `session.sh play` (hover PLAY + Return; mouse clicks don't register),
then `session.sh join 1 [2]`. Opponent: `BOPL_AGENT=mixed-v2-hard`,
`BOPL_STOCKS=3`. Logs are in `/tmp/shadow-live/run{1,2,3}`.

| run | rounds | result |
| --- | --- | --- |
| 1 (before the hierarchy fix) | 2p: 1085 ticks in sync; rounds 2-3 desynced at tick 0 on `entity[0].hierarchy` | found the prefab-hierarchy carry-over |
| 2 | round 0: **3p**, 1805 ticks, 7 deaths; rounds 1-5: 2p, maps 0/0/0/83/6, 569-1254 ticks each, carried match deaths up to 6 | **all 6 rounds in sync to the terminal state** |
| 3 (`BOPL_AGENT_HITSTOP_TIME=1`) | 2p, 533 ticks, 3 deaths | in sync (hitstop is 0 in this build) |

Live coverage:
- shown: 2p and 3p; multi-round matches with carried kills/deaths and
  hierarchy; deaths and round ends; terminal-state agreement; source check
  passes; in-process load of the q32 `.so`; human kits drawn by the game
  between rounds.
- not shown live: sudden death (rounds were too short), 4p+, a keyboard
  player, the Mac `.dylib` load (code path only), the ability-pickup mode,
  online play. These are covered offline where the sims can represent them:
  sudden death, 4p, all kits.

## Next step: done (milestone 2)

The shim now searches every decision on forks of this shadow world with
`BOPL_AGENT_SEARCH=8,1,1`: docs/INGAME_SEARCH.md. The five points of the plan:
the env points at the shadow world for the decision only (forks, never steps
it); `8,1,1` runs synchronously (~2 ms 2p, ~3.5 ms 3p live); every opponent,
humans included, is modelled by the agent's own policy; rollouts stop at the
shim's limit (`bopl_search_set_limit`), not the shadow's 2160000; and a desync,
the cap or a search error falls back to the raw policy for the round.

Found live while doing it: a jump held at the end of a round and pressed again
in the next warmup desynced the shadow at episode tick 1 (the game kept the
device's stale jump-edge memory across the freeze; the replay and the shadow
see a fresh edge). `Player.RestoreLocalInputs` now carries the frozen
override's edge memory over; see INGAME_SEARCH.md.
