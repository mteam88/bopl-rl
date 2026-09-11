# PLAY.md — a human plays a checkpoint or a scripted bot in the browser

`research/play.py` serves a page where one person plays one real-time match
on the Rust sim against a native checkpoint (`policy-*.bin`) or a scripted bot
(`mixed-hard`, …). Its purpose is calibration: ~20 games per opponent give a
human row for the same Bradley–Terry fit `elo_ladder.py` uses, so "beats
mixed-hard 70 %" can be placed on a human scale.

## Run

```sh
cd <checkout>
BOPL_RL_BACKEND=rust native/artifacts/rl/venv/bin/python native/rl/research/play.py \
  --port 8891 \
  --checkpoints 'native/artifacts/rl/sweeps/anchors/*.bin' \
  --checkpoints '<sweep-root>/overnight-3/0*/checkpoints/policy-*.bin' --latest-per-dir \
  --out native/artifacts/rl/human
# -> http://0.0.0.0:8891/  (tailnet: http://100.89.48.52:8891/)
```

`--checkpoints` is a repeatable glob; `--latest-per-dir` keeps only the newest
`policy-<step>.bin` per directory (named anchors are always kept). The globs are
rescanned on every lobby load, so a running sweep's new checkpoints appear.
`--game` overrides the sim library (default `native/target/release/libbopl_sim.so`,
`cargo build --release` in `native/`). `libbopl_policy_eval.so` must match the
checkpoint architecture: `python native_policy.py --build` rebuilds it.

As a user service (a new unit; never touch other `bopl-*` units):

```sh
systemd-run --user --unit bopl-play -p CPUWeight=1000 -p MemoryMax=4G -p Restart=on-failure \
  --working-directory "$PWD" -E BOPL_RL_BACKEND=rust -E PYTHONUNBUFFERED=1 \
  -- native/artifacts/rl/venv/bin/python native/rl/research/play.py --port 8891 \
     --checkpoints '<sweep-root>/anchors/*.bin' \
     --checkpoints '<sweep-root>/overnight-3/0*/checkpoints/policy-*.bin' --latest-per-dir \
     --out native/artifacts/rl/human
journalctl --user -u bopl-play -f       # one JSON line per finished game
systemctl --user stop bopl-play          # (re)start: stop, then systemd-run again
```

The server does ~0.6 ms of work per 66.7 ms decision (Rust step + CPU MinGRU
policy + frame JSON), so it needs no GPU and well under one core.

## What the human plays

- **Tick and cadence.** The sim runs Unity's fixed step, `Fixed Timestep:
  0.01666667` s (`ProjectSettings/TimeManager.asset`), i.e. 60 ticks/s; the
  Rust `DT` is the same value in Q32.32 (`native/src/world.rs`). The RL env
  applies each action for 4 ticks (`env->repeat=4`, `puffer/eval_binding.c`
  `research_open`), so the match advances in 66.7 ms decisions and the page
  receives one frame per decision (15/s). The human's key state is sampled once
  per decision, exactly like a policy's action. The state between decisions is
  not observable through the binding (the phase-3 sample is internal to the
  9-tick delay), so the page optionally extrapolates each body along its
  velocity for the time since the last frame (checkbox; off shows raw frames).
- **Opponent.** A checkpoint runs through `native_policy.load` and
  `evaluate.masked`/`evaluate.sample` — the same masked multinomial sampling,
  9-tick observation delay (`observation_delay_ticks` from the checkpoint
  config), episode limit (`episode_ticks`, 7200) and latch/mask rules as
  `evaluate.py`; the recurrent state starts from `policy.initial(2)` and the
  policy sees both rows as in evaluation. A scripted bot is pinned with
  `Environment.bot` and decides inside the C env from the true state
  (`SCRIPTED_BOTS.md` "Contract"). Reward telemetry is left at the evaluator
  defaults; it does not affect behaviour.
- **Sides.** Human = P1 (left spawn, yellow) or P2 (right, purple);
  "alternate" (default) flips with each finished game.
- **Seed.** Random 31-bit per game unless given; recorded. The env's own
  seed (`env_seed`, drives the scripted bot RNG stream) is recorded too.

## Key bindings

Bopl's default keyboard-and-mouse scheme (`Assets/MonoBehaviour/SlimeControls.asset`,
group `KeyboardAndMouse`): W/A/S/D and the arrow keys are the `W`/`A`/`S`/`D`
movement actions, Space is `Jump`, mouse left/right/middle are `Ability0/1/2`,
`AimPosition` is the mouse position; Enter/Escape are Start/Select (not part of
the RL action space). `InputUpdater.cs` (`Update`, keyboard branch) turns the
mouse into `joystickAngleByte` as the byte of `mouseWorld − player.Position`
(zero if its squared length is `< 0.2`), using
`Player.InputProfile.JoyStickAngleByteFromVector` (`Player.cs:141`):
`int(angle/2π·255) % 255 + 1`, 0 = neutral.

| Action | Page | Unity default |
| --- | --- | --- |
| Move | W A S D / arrow keys | W/A/S/D, arrows |
| Jump (button bit 0) | Space | Space |
| Dash, ability 0 (bit 1) | left mouse or J | left mouse |
| Bow, ability 1 (bit 2) | right mouse or K | right mouse |
| Platform, ability 2 (bit 3) | middle mouse or L | middle mouse |
| Aim | mouse (mouse mode) / movement keys (keys mode) | mouse |
| Bow charge | 1–5 or wheel: hold 1/2/4/8/16 decisions | hold, release |
| Abort match (not recorded) | Esc | — |

## Keyboard state → policy action, per decision

The policy action is `(aim byte 0..255, buttons 0..15, hold index 0..4)`
(`bopl_layout.h`; `c_step` aborts on anything else, so the keyboard movement bits
W/A/S/D = 64/128/256/512 of `BoplRlAction.buttons` in `bopl_rl.h` cannot be sent
through the trainer's action space). The human player is therefore a
**gamepad-configured** player (`keyboard=0`, the `bopl_rl_default_config`
default the policies train with): its one aim vector drives movement and
ability aim alike (`world.rs` `step_with_input` uses `action.direction()` for
both; only `keyboard=1` players split W/A/S/D motion from the aim byte).

Each decision (`HumanInput.decide`, `play.py`), given the env's action mask for
the human row (`bopl_write_mask`):

1. `keys = down | pressed-since-last-decision` — a tap shorter than one decision
   still counts once.
2. **Buttons** = jump/dash/platform key state; **bow** is edge-triggered: a
   press since the last decision sets the arrow bit once; holding the key does
   not re-press.
3. **Latched** (hold head reduced to index 0 in the mask): the only allowed
   button state is sent, aim stays live — identical to what `bopl_latch` would
   do to a policy's ignored button head.
4. **Free**: if the mask forbids the arrow bit (the forced-release decision
   after an arrow latch, `bopl_write_mask`) it is dropped. If the arrow bit is
   set, `hold` = the selected charge (default index 4 = 16 decisions); every
   other free decision sends hold index 0 (length 1), so jump/dash/platform are
   re-read every decision like a policy that never latches.
5. **Aim**: keys mode — the 8-way vector `(D−A, W−S)` through
   `JoyStickAngleByteFromVector` (deadzone `Constants.DEADZONE = 0.4`,
   `Constants.cs:17`; unit or zero vectors here). Mouse mode — the mouse byte
   (computed in the page from the arena camera, the Unity formula above)
   whenever an ability button is in this decision, a latch is in force, or an
   ability is active (`ability_flags & 8`: bow charge, platform morph, dash);
   otherwise the movement keys. The action is asserted against the mask before
   it is sent.

**Arrow charge.** A policy charges by latching the arrow bit for
`HOLD_LENGTHS[hold]` decisions; the latch expiry forces the release and the bow
fires on release (`RUST_ROLLOUTS.md`: "fire-on-release", loadingFrame 1 then
+8/+7/+7 ticks; charge 0 for holds 1/2/4, 1 for hold 8, 4 = full for hold 16;
`SCRIPTED_BOTS.md` game facts). A human cannot release early through this
interface, so the charge is chosen **at press time** (1–5 / wheel, shown in the
HUD) instead of by release timing. While latched all four buttons are pinned
(no jump during a charge), as for a policy.

**Divergences from Unity's keyboard scheme (why):**
- Mouse aim moves the slime toward the mouse while an ability is pressed/active,
  because the sim player is gamepad-configured (one vector). A Unity
  keyboard player moves with W/A/S/D independently of the mouse. Keys mode has
  no such coupling but aims in 8 directions only.
- Charge is committed at press; Unity releases to fire.
- Decisions are 4 ticks; Unity polls input every frame (InputSystem polling
  120 Hz, `InputUpdater.Awake`).
- No Start/Select; W/A/S/D bits are folded into the aim byte.

**Information delay.** A policy acts on a 9-tick-old observation (150 ms). The
human sees the state at the end of each decision (tick 4k) and acts at the
next boundary at the earliest: 4 ticks (67 ms) plus network, render (≤ 16.7 ms
at 60 Hz) and reaction time. No artificial delay is added; the human's
reaction time dominates.

## Latency and bandwidth (measured 2026-09-10, headless Firefox on the box, sweep running)

`check_play.mjs` (below), 663-frame game vs mixed-hard, keys mode:

| Quantity | Value |
| --- | --- |
| frame interval p50 / p95 / max | 67 / 72 / 99 ms (one decision = 66.7 ms; 0 re-anchors) |
| input → frame acknowledging it, p50 / p95 | 36 / 67 ms (uniform wait to the next decision boundary + ~1 ms; add ≤ 16.7 ms to the next paint and the network RTT) |
| server work per decision, mean / max | 0.63 / 18.2 ms (load average 21 on 12 CPUs from the sweep) |
| frame message | ~780 B mean (2.2 KB peak with many arrows); 11–12 KiB/s at 15 frames/s |
| input message | ~60 B, sent on key/aim change (mouse aim throttled to ≥ 15 ms) |
| realtime ratio | 0.9995–1.0000 |

Over Tailscale from the Mac add the RTT (~5–20 ms); the HUD shows live p50/p95
input→frame, frame interval, RTT and KiB/s, and each record stores the server
timing (`timing`).

## Records

Every finished (not aborted) game appends one line to `<out>/games.jsonl`:

- Bradley–Terry row in `elo_ladder.py`'s game shape: `a="human"`, `b=<opponent
  name>`, `map`, `seed`, `side`, `outcome` ∈ `a`/`b`/`draw` (timeout = draw,
  as the ladder scores it). Feed it by accumulating
  `records[(a,b)] = [wins_a, wins_b, draws]` alongside the ladder's pool and
  calling `bradley_terry(names, records, 'random')`; a checkpoint's `b` is the
  first 12 hex of its weights SHA-256 (full `opponent.sha256` and
  `metadata_sha256` are in the row, so it joins to ladder participants by hash).
- `timestamp`, `result` (`win`/`loss`/`timeout`), `ticks`, `seconds`.
- `human`: side, kills/deaths, `arrows_fired`, `arrow_hits`, ability
  activations, input mode, decisions, button rates, hold choices, aim source
  counts. `opponent`: kind (`checkpoint` with path, step, sha256s, or
  `scripted`), kills/deaths, arrows fired/hit, activations.
- `replay` (+ `replay_sha256`): the BOPLREP2 `.rep` written by
  `Environment.replay` (the `research_replay` writer `evaluate.py` uses);
  `visual`: the physics-view JSON in `evaluate.py`'s shape (`frames` with
  `actions`, `tick_rate` 60, `decision_rate` 15) — the page's "watch" button
  plays it, and `prepare_replay.py` can rebuild it from the `.rep`.
- `game` (sim path + sha256), `env_seed`, `timing`.

## Check

```sh
firefox --headless --remote-debugging-port 9225 --profile "$(mktemp -d)" --no-remote &
node native/rl/research/check_play.mjs http://127.0.0.1:8891/ /tmp/play-shots
```

Loads the lobby, plays a keys-mode game vs mixed-hard as P1 with scripted key
presses and a mouse-mode game as P2 with synthetic mouse aim and right-button
bow presses, and checks the frame rate, that `games.jsonl` gained the record,
the `.rep` and visual JSON exist, the viewer renders, mouse aim reached the
action, and the console has no errors. Prints `PLAY_OK` with the numbers above.
