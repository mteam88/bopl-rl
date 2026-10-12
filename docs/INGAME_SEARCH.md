# In-game decision-time search (milestone 2)

The agent shim searches each of its decisions on forks of the shadow world
(docs/SHADOW.md), so a human plays the search-wrapped bot in the real game.
It is the research library's search (`train/bopl_search.h`, docs/SEARCH.md),
compiled into `libbopl_agent`, and it runs synchronously in the shim tick on
Unity's main thread.

## Use

    BOPL_AGENT_SEARCH=8,1,1 Tools/play-vs-agent.sh <policy-*.bin>
    BOPL_AGENT_SEARCH=8,1,1 Tools/play-vs-agent-mac.sh <policy-*.bin>

- The value is a budget `k,n,depth[,h][,mix=M][,vs=V][,latched=0]`, as in
  `eval/search_policy.py`. `1` means `8,1,1`. Setting it implies
  `BOPL_AGENT_SHADOW=1`.
- The checkpoint must be policy v4 (a `.roster` searches with each v4
  member). Scripted bots play unsearched.
- Player.log lines:
  - `BOPL_AGENT_SEARCH: budget ... on the shadow world` at open;
  - `BOPL_AGENT_SEARCH: round N level L: <reason>` when a round falls back;
  - `BOPL_AGENT_TIMING: round N ...` at every round end: the shim tick on
    decision ticks (search included) and on the others, frame times while the
    agent played, and the search counters.
- The result JSONL (`BOPL_AGENT_RESULTS`) gains `timing` and `search` objects.

### Mac

The q32 shadow sim has to be built on the Mac. `ssh mac` has cargo 1.96.1
(`~/.cargo/bin`). `just sync-mac` excludes `native/`, so copy the crate once
(and after sim changes), from the PowerSpec:

    just sync-mac && just build-mac
    cd ~/Documents/Coding/bopl-re/ripped/ExportedProject/native && \
      rsync -az Cargo.toml Cargo.lock src data fixtures include \
      mac:Documents/Coding/bopl-re/ripped/ExportedProject/native/

Then on the Mac, `Tools/play-vs-agent-mac.sh` builds everything itself:
`make agent-mac` (policy_eval with `-mcpu=native -ffast-math`) and, with
`BOPL_AGENT_SEARCH` or `BOPL_AGENT_SHADOW` set and `native/Cargo.toml`
present, `make agent-shadow-lib-mac` (`cargo build --release --features q32`,
then a copy to `libbopl_sim_q32.dylib`). Both dylibs go into
`Contents/Plugins`, and both variables are passed through `open --env`. The
checkpoint (`.bin` and `.bin.json`) must be on the Mac.

The Mac path was proven offline with the same harness on the M4 Pro, below.
The Unity Mac player itself was not launched with search.

## How it works

At each decision tick e (every 4 episode ticks):

1. `RlAgent.BeforeTick` writes S_e. The shadow checks it, then
   `bopl_agent_tick_bytes` runs the raw policy forward and samples, as before.
2. If the search is live, `bopl_agent_search_decide` checks the shadow. It must
   be TRACKING, at episode tick e, with the right player count, and
   e < the shim's limit. If so, it points the shim's env at the shadow world
   for this call, rebases kills and deaths on the shadow state, and calls
   `bopl_search_act_at`. The search forks that world and never steps it. The
   searched action replaces the sampled one.
3. Opponents: every other player, humans included, is modelled by the agent's
   own policy on its own v4 row. Their rows and masks are built from the
   shadow, with the humans' real packets mapped to aim, buttons and hold:
   - angle and `buttons & 15`;
   - hold 0;
   - `hold_run` continued across decisions.
4. Rollouts end at the shim's round limit (`bopl_search_set_limit`, the
   sidecar's `max_episode_ticks`, 10800 for allkit-1) and pay the timeout
   reward there, as in training. The shadow's own limit (2160000) never ends
   a rollout.
5. Fallback: on a desync, a refused or missing shadow, the episode cap
   (the game does not truncate rounds, the agent's limit is only a horizon),
   or any search error, the agent plays the raw policy for the rest of the
   round. It logs the reason once. A sim error inside a rollout used to
   `abort()`. Now it longjmps back to the shim (`BOPL_PUFFER_RECOVER`) and
   falls back, so it cannot kill the game.

N players: the search, previously 2-player, takes 2 to 6 players. All rows
are v4, and the root forward and rollout forwards are batch-P. For 2 players
it makes the same choices from the same RNG stream as before; see the
strength check below, where it is game-for-game identical to the 1ac0376f
library.

## Proof

### Offline equivalence (`eval/shim_search_check.c`, `make shim-search-check`)

This plays a P-player research env R on the q32 lib as "the game". R's agent
seat decides by the research library's `bopl_search_act`. The humans are the
same policy sampled raw. The real shim is fed R's wire record every tick, and
its shadow is stepped with R's packets. At every decision the harness checks:

- the shim's choice against R's search choice, from the same forked state and
  seed;
- the shim's packet against R's latched action;
- every player's row (bitwise) and every mask;
- a search-attached shim whose shadow never tracks against the plain raw shim
  (identical row, logits and choice: making every row v4 leaves the raw path
  alone).

| run | decisions | result |
| --- | --- | --- |
| 2p `8,1,1`, 6 episodes, maps 0,3,36(space),12,50,80 | 8265 | IDENTICAL |
| 3p `8,1,1`, same maps | 7525 | IDENTICAL |
| 2p `8,1,1`, limit 400 on every other episode (3 truncated) | 5392 | IDENTICAL |
| 3p `8,1,1`, the same | 3121 | IDENTICAL |
| 2p `4,2,2`, 3 episodes | 6149 | IDENTICAL |
| 2p `4,2,8`, limit 403 (rollouts cut by the cap) | 807 | IDENTICAL |
| 2p `8,1,8`, 2 episodes | 3789 | IDENTICAL |
| Mac M4 Pro: 2p, 3p (maps 36, 0, 110, limit 400), 2p `8,1,8` | 10312 | IDENTICAL |

Zero choice, row, mask or raw-path differences anywhere. The search shim's
privileged (critic) block differs from the raw shim's at most decisions. That
is expected: it now holds the humans' real latched inputs.

### Live (Linux build, Xvfb, `eval/shadow_live`, allkit-1@14667m)

| run | rounds | shim decision tick (ms) | frames (ms) |
| --- | --- | --- | --- |
| 2p `8,1,1`, random human | 0-2 searched throughout, in sync | mean 2.0-2.1, p95 2.3-2.5, max 3.8 | p50 16.6-16.9, p95 17.2-17.8, max 24.5, 0 over 33 |
| same, round 3 (fell back at tick 4) | raw for the round | mean 0.12, max 2.1 | p50 16.5, p95 17.1, max 23.9 |
| 3p `8,1,1`, 2 random humans | 0-5 searched throughout, in sync | mean 3.3-3.7, p95 4.6-5.2, max 11.0 | p50 16.9-17.7, p95 17.9-19.4, max 32.8, 0 over 33 |
| 2p `8,1,1`, jump-edge repro after the fix | 0-3 in sync | mean 2.4-2.5, p95 3.3-3.5, max 4.6 | p50 17.0-17.5, p95 18.1-19.2, 1 frame over 33 of 8000 |
| 2p `8,1,8`, random human | 0-4 searched throughout, in sync | mean 10.3-10.7, p95 11.0-11.9, max 42.7 (one stall) | p50 15.8-16.7, p95 16.4-17.4, max 77 (one), 2 over 33 of 8000 |
| 2p `8,1,8`, after the kills rebase (fef970e6), one match | 1-7 searched throughout, in sync | mean 10.3-11.7, p95 11.1-12.5, max 14.6 | p50 15.8-17.0, p95 16.6-17.8, max 29.4, 0 over 33 |
| 3p `8,1,8`, 2 random humans, same build | 0-2 searched throughout, in sync | mean 15.6-17.8, p95 16.8-19.1, max 23.8 | p50 15.9-16.9, p95 17.6-19.7, max 25.4, 0 over 33 |

Fallback was exercised live twice: run 1 round 3 and run 2 round 6, both by
the desync below. Each logged its reason and played the round out raw.

At `8,1,1` the search costs about 2 ms (2p) or 3.5 ms (3p) on the decision
frame. Even `8,1,8` (10.5 ms) leaves the PowerSpec's frames at 60 Hz. Frame
times with search match the raw round (16.5-17 ms p50). The frame budget under
Xvfb at 1280x720 is generous; a real 1440p display leaves less.

Offline per-decision cost (shim tick at a decision, `8,1,1`):

| machine | 2p mean / p95 / max | 3p mean / p95 / max | 2p `8,1,8` mean / max |
| --- | --- | --- | --- |
| PowerSpec 7600X3D | 1.9 / 2.3 / 3.9 | 2.7 / 3.4 / 5.0 | 10.5 / 14.8 |
| M4 Pro (battery) | 2.8-3.1 / 3.5 / 21 (one spike) | 4.2 / 4.6 / 7.2 | 17.0 / 20.7 |

M4 Pro, 2026-10-12 (battery, `eval/shim_search_check.c`, maps 0,3,36, 3 full
episodes each, all IDENTICAL), shim tick at a decision, mean / p95 / max ms:

| budget | 2p | 3p |
| --- | --- | --- |
| `8,1,8` | 16.9 / 17.5 / 24.9 | 25.1 / 25.9 / 43.1 |
| `4,1,8` | 8.9 / 9.3 / 29.1 | 12.8 / 13.9 / 15.8 |
| `4,1,6` | | 10.0 / 10.6 / 29.6 |
| `8,1,4` | | 13.3 / 13.9 / 47.5 |
| `4,1,4` | | 6.9 / 7.4 / 7.9 |

### Strength (`league/all_combos.py --search 8,1,1`, 200 pairs)

allkit-1@14667m searched at `8,1,1` against raw allkit-1@14667m scores
**0.486 [0.433, 0.539]** (400 games, 1.8 ms per decision). That is no gain.
The 0.60 for `8,1,1` in SEARCH.md is g111's, over 100 pairs ([0.518, 0.680]).
The library at 1ac0376f gives the identical 0.48625 with the identical 458819
decisions, so this is the budget, not the change. On allkit-1 the gain needs
depth: `8,1,8` scored 0.660 there (SEARCH.md). `8,1,8` costs 10.5 ms per
decision on the PowerSpec, which holds 60 Hz live (above), so
`BOPL_AGENT_SEARCH=8,1,8` is the budget to play on the PowerSpec. On the M4
Pro it costs 17 ms, which run synchronously stretches every fourth frame to
about 33 ms. Making `8,1,8` smooth on the Mac means taking the search off the
frame. One way is a worker thread on a fork of the shadow taken at e. It
would either delay the decision by a tick or two (training never saw that
delay) or search from e-4 with the opponents' packets predicted. That is the
next step.

Cheaper budgets for the Mac (same benchmark, allkit-1@14667m searched vs raw,
200 pairs, 400 games each, 2026-10-12):

| budget | score | 95% CI | ms / decision (PowerSpec, 2p) |
| --- | --- | --- | --- |
| `8,1,1` | 0.486 | [0.433, 0.539] | 1.8 |
| `4,1,4` | 0.564 | [0.515, 0.615] | 4.4 |
| `4,1,6` | 0.609 | [0.554, 0.665] | 5.8 |
| `4,1,8` | 0.584 | [0.529, 0.636] | 8.2 |
| `8,1,4` | 0.588 | [0.530, 0.644] | 8.2 |
| `8,1,8` | 0.660 | [0.582, 0.733] | 10.5 (SEARCH.md) |

On the M4 Pro, `8,1,8` does not fit a 16.7 ms frame even 2p (16.9 ms), and 3p
is 25 ms. `4,1,6` keeps most of the gain at about 40% of the cost (3p 10.0 ms
mean on the Mac), so it is the Mac budget:
`BOPL_AGENT_SEARCH=4,1,6`.

## Found on the way

- **Round-start jump desync, fixed** (`Player.RestoreLocalInputs`). A jump
  held at the end of one round and pressed again during the next warmup
  desynced the shadow at episode tick 1: the human jumped in the shadow (and
  in the replay) but not in the game. The cause: RlAgent freezes humans with
  zero override packets for the warmup, and on restore the device profile's
  `jumpButton_DOWN_LAST_FRAME` was still the stale `true` from the last round's
  winner screen, so the game saw no edge. Restore now carries the override's
  edge memory over.
  - Before: random humans hit it in 2 of 11 rounds. A scripted repro hit it
    2/2: press A on the winner screen to go on, press it again about 1.5 s
    into the warmup, hold it 6 s.
  - After: the same repro stayed in sync in 3/3 rounds (the human's packet at
    tick 240 had jump=1 in two of them).
  - The saved replays of those rounds were also unfaithful to the game. Stock
    rounds without RlAgent never freeze inputs, so this path is RlAgent-only.
- **The shim never set the map.** `env->config.map` stayed 0, so space maps
  were encoded as grass in the v4 rows of raw play too. RlAgent now calls
  `bopl_agent_set_map(level)` each round.
- **Kills and deaths are match-cumulative in the game.** The v4 row encodes
  them (clip /10, `bopl_rows_v4.h`), and every training episode is one round
  from zero, so rounds after the first were out of distribution for the raw
  forward (the search's own rows come from the shadow, which starts at zero).
  Fixed in fef970e6: `RlAgent.BeginRound` calls `RlObservation.RebaseRound`,
  and `Write` subtracts each player's round-start counts. Player.log shows
  `BOPL_AGENT: round N first row kills/deaths p1 0/0 (match 2/15), ...` and
  the same for the terminal row. Live check (Linux, `8,1,8`, allkit-1@14667m):
  a 2p match of 7 rounds (match totals up to 3/21) and a 3p match of 3 rounds
  started every round at 0/0, stayed in sync and searched every decision.
- **macOS secondary threads get 512 KB of stack.** The in-process Rust
  world's constructor overflows that. The research env's world-open threads
  (`bopl_puffer.h`) and the in-process reset thread (`sim/abi/bopl_rl.c`) now
  ask for 16 MB and 8 MB on Apple. Linux is unchanged. The game's shadow
  reset runs on that reset thread.
- **Fast-math ends bitwise fidelity to torch.** With the shim's policy_eval
  on `-ffast-math`, `eval/check_agent_fidelity.py sim` fails on logits at the
  first decision. That check compares bitwise, and fast-math reorders float
  sums. The search equivalence is unaffected: the harness builds both sides
  with the same flags. For the fidelity check, build exactly with
  `make agent AGENT_EVAL_CPU= AGENT_EVAL_FAST=`. That build passed with 0
  mismatches over 506 decisions on maps 0, 1 and 3, the same as 1ac0376f.

## Files

- `train/bopl_search.h`: N players, `root_rows`, `bopl_search_set_limit`,
  `bopl_search_root` / `bopl_search_act_at`.
- `train/bopl_agent.{h,c}`: `bopl_agent_search`, `_search_active`,
  `_search_note`, `_search_stats`, `_set_map`, the decision hook, human
  mapping, fallback and timing.
- `train/bopl_puffer.h`: `BOPL_PUFFER_RECOVER`, the Apple thread stack.
- `train/bopl_shadow.{h,c}`: `bopl_shadow_packets` (the last applied packets).
- `Assets/Scripts/Assembly-CSharp/RlAgent.cs`: env, attach, map, timing,
  JSONL. `Player.cs`: the jump-edge fix.
- `eval/shim_search_check.c`: the equivalence harness. The Makefile has
  `-march=native` / `-mcpu=native` and `-ffast-math` (`AGENT_EVAL_FAST`) for the shim's
  policy_eval, plus `agent-shadow-lib-mac`.
- `Tools/play-vs-agent{,-mac}.sh`: search and shadow env, q32 lib build and
  copy.
