# Oracle: Unity reference traces and how to diff against them

The oracle is the game's own `ReferenceCapture` (`Assets/Scripts/Assembly-CSharp/ReferenceCapture.cs`), which compiles into the port unchanged. A port run is correct when its trace matches the Unity player's trace for the same `episode.rep`.

## Recipe

```sh
# Unity side (records a controlled episode, replays it, cross-checks the two):
python native/rl/verify_replay.py --players 2 --ticks 240 --level 0 --output /tmp/oracle/l0-p2

# Port side:
BOPL_TRACE=/tmp/oracle/l0-p2/dotnet.jsonl.gz dotnet run --project sim/Host -- replay /tmp/oracle/l0-p2/episode.rep --ticks 240

# Diff:
python sim/Oracle/diff.py /tmp/oracle/l0-p2/replayed.jsonl.gz /tmp/oracle/l0-p2/dotnet.jsonl.gz --summary
```

`diff.py` exits 0 on full match, 1 on divergence (prints matched tick count, first differing JSON path, both values; `--summary` lists which top-level sections differ), 2 on header/setup mismatch. It always ignores `native/data/process-identity-fields.json` (raw `GetInstanceID` values, see below); `--ignore-presentation` additionally applies the audited cosmetic exclusions in `native/data/presentation-fields.json` — the same projection `verify_replay.py` uses.

Level selection: `record.c` takes an optional trailing `[map]` argument (`verify_replay.py --level N`); the level is stored in the `.rep` archive (`ReplayArchive.level`), so replay playback and the port follow it automatically. `GameSession.CurrentLevel()` also honours `BOPL_FORCE_LEVEL`, but that hook is not needed on the replay path.

## Checkpoint fixtures (2 players, ability group 0 = Dash/Gust/Invisibility, seed 785921, `--ticks 240`)

Recorded 2026-09-08 on the pinned player (`Builds/Linux/BoplBattle.x86_64`, Mono; the replay side of `verify_replay` uses the IL2CPP build, so the match is also cross-backend). `--ticks` is the post-warmup budget; every episode runs 240 warmup ticks first, and the trace covers warmup too, so a full run is 480 trace ticks. Episodes can end earlier when a player dies (terminal), which is why l1/l4 are shorter.

`episode.rep` + `report.json` for each level are committed under `sim/Oracle/fixtures/l{N}-p2/`. The traces are 11–14 MB gzipped each, above the 5 MB fixture limit, so they live in `/tmp/oracle/` (regenerate with the commands above if lost):

| run | match | trace ticks | controlled.jsonl.gz sha256 | replayed.jsonl.gz sha256 |
| --- | --- | --- | --- | --- |
| /tmp/oracle/l0-p2 | true | 480 | ccdfa66dbd0addaff66c4e97f2789b6b3c1b9be3c4f28765e9c2c7e922d9396b | 66b47891ecc8d4b7c58daf5bd612de4f926c3c699320ca5d5be0507e53d820fc |
| /tmp/oracle/l1-p2 | true | 444 | 45439210dadc1bcb86c86a387c98d326ff97f61affe4606742d1abc6dbad977b | e5b29c00b0a03dab52efe40ebfb09a37110c1cdda21fe7322d938e6ee1044a7a |
| /tmp/oracle/l2-p2 | true | 480 | 10db1086bba29cb60a1e25969fab910fb218ac8d09f931a0ce1fdc6657bbe6ef | 4772f3418b01ec738c5b1ec953d6f9d6d7f8d2dbddff246695476e4e8fb061b8 |
| /tmp/oracle/l3-p2 | true | 480 | ee3e15469cc6f416816ebab190d471a196c9e96f81a9e8253eec3826860fb078 | 0f408036183c87ca6512b28af25b1f7633708c9205e8284c467161261a6444c5 |
| /tmp/oracle/l4-p2 | true | 418 | 0f15eb03bec31f46965961317032f8e7c97c282b72fff183364e83fea0240abb | b71a7e78c874f597ad4150f6bbb7b1482b062a09b999370526fb84a5099f430d |

`episode.rep` sha256s are in each `report.json` (`replay_sha256`).

## Ability-coverage fixtures (level 0, `--ticks 480`, seed 785921+G)

`record.c` ability groups (`record <game> ... <players> [ability_group] [ticks] [map]`):
group 0 keeps `bopl_rl_default_config` loadouts (all players Dash/Gust/Invisibility,
catalog indices 2/7/19) with the button-mash input script; groups 1–3 assign catalog
index `1+18*(G-1)+3*player+slot` per slot (indices ≥39 clamp to Dash) with a scripted
ability-press input pattern and 5 stocks; group 4 is the fixed training loadout.
**With 2 players only 6 slots exist, so groups 1–3 cover indices 1–6, 19–24, 37–38 —
13 distinct abilities; indices 7–18 and 25–36 need 6-player fixtures** (the formula
was sized for 6 players; 18 per group).

| fixture | loadouts (catalog index = name) | exercises |
| --- | --- | --- |
| l0-p2-g0 | both: 2 Dash, 7 Gust, 19 Invisibility | default RL smoke loadout |
| l0-p2-g1 | p0: 1 Random, 2 Dash, 3 Grenade; p1: 4 Bow, 5 Engine, 6 Blink gun | projectiles, Random re-roll |
| l0-p2-g2 | p0: 19 Invisibility, 20 Meteor, 21 Throw; p1: 22 Push, 23 Tesla coil, 24 Mine | area effects, placeables |
| l0-p2-g3 | p0: 37 Mjolnir, 38 Portal gun; rest clamp to 2 Dash | end-of-catalog abilities |
| l0-p2-g4 | both: 2 Dash, 4 Bow, 14 Platform | **the live training loadout** (`puffer/bopl.ini` ability1/2/3 = 2/4/14, `eval_binding.c`; `random_loadouts=0`) |
| l0-p3-g0 | 3 players, default loadout | 3-player physics/team handling |

All recorded 2026-09-08, `match: true` controlled-vs-replayed on the first pass — no
new process-identity exclusions were needed beyond the existing
`native/data/process-identity-fields.json` set. `episode.rep` + `report.json`
committed under `sim/Oracle/fixtures/<name>/`; traces stay in `/tmp/oracle/`
(11–33 MB gzipped). Trace ticks = 240 warmup + up to 480 episode ticks; g0 runs
end early on a terminal death (509/557 ticks).

| run | match | trace ticks | controlled.jsonl.gz sha256 | replayed.jsonl.gz sha256 |
| --- | --- | --- | --- | --- |
| /tmp/oracle/l0-p2-g0 | true | 509 | 1681221a07afb3b7b5de6946f671adf8d41b4eecdb225af4c9eac2933929b227 | 5297a93a383abc8df5e8a159e27e716bc2f6e55e20b4c47ee323cfcabd60d4ac |
| /tmp/oracle/l0-p2-g1 | true | 720 | 2b708bf389e7359de500d9a47776b3e99f40241e449cc8089d7d8b02ecc6d50b | d1c69ab0e4fcafd69a4255889971e2007e2db7d72bbdb66ae6a42e88adefb590 |
| /tmp/oracle/l0-p2-g2 | true | 720 | e2b716ed79f5b6d945341b3ee93d3092c3fb9407048dcd121ee2fea3d0c01313 | 6b7ed9dfb6f4658876b41095eaac78558484fdfc614f8f1e33d93a7d7cd694db |
| /tmp/oracle/l0-p2-g3 | true | 720 | 3b900e19a9b820244c93ea108517e2e9091f49a0499d637260019d464c3e6b08 | 820c2c5f0d5596b25bc4d747896974b03a2a01b0fca3756be8b446874b5dfed3 |
| /tmp/oracle/l0-p2-g4 | true | 720 | 0fdb9267869cb201f012e0970058fdd4a65b7864cdba6b4cad7cccc2774ac987 | d1f9b0d5e832b55e3e523c2634aa245e2379562f080684e2a5ca3ebb84034a68 |
| /tmp/oracle/l0-p3-g0 | true | 557 | d29b06a7488a5ad4ab4abefb8599b37d2262b86105805db32915d386e822f8c7 | d66ac9d8fed6594137327c3b72d005eeb5865509c88b85079834b7a2594b2348 |

Full ability catalog (39 entries, index = the ushort in `.rep` player loadouts, from
`Assets/MonoBehaviour/AbilityIcons.asset` `sprites` order): 0 NullAbility, 1 Random,
2 Dash, 3 Grenade, 4 Bow, 5 Engine, 6 Blink gun, 7 Gust, 8 Growth ray, 9 Rock,
10 Missile, 11 Spike, 12 Time stop, 13 Smoke, 14 Platform, 15 Revival, 16 Roll,
17 Shrink ray, 18 Black hole, 19 Invisibility, 20 Meteor, 21 Throw, 22 Push,
23 Tesla coil, 24 Mine, 25 Teleport, 26 Drill, 27 Grappling hook, 28 Beam,
29 Duplicator, 30 Magnet gun, 31 Rewind, 32 Web swing, 33 Poke ball,
34 Gold mushroom (Super star.prefab), 35 Tack (Banana peel.prefab), 36 Freeze ray,
37 Mjolnir, 38 Portal gun.

Rerun any of these: `python3 native/rl/verify_replay.py --players P --ability-group G
--level 0 --ticks 480 --output /tmp/oracle/l0-pP-gG`. Include them in the scorecard
with `sim/Oracle/checkpoint.sh --full`.

### Unity ticks/s (for the port comparison)

- Traced (`BOPL_TRACE` writing full state JSON): ~40–50 ticks/s. Controlled Mono player `[PERF] simTickAvg=20.0–25.0ms`; replayed IL2CPP player did 480 ticks in 11.6 s including boot. Trace encoding dominates.
- Untraced Mono player: a full 480-tick episode (record step, level 0, 2 players) completes in **3.24 s wall including ~3.1 s boot/shutdown**; a 241-tick run takes 3.19 s, so the marginal tick cost is ~0.1–0.2 ms → roughly **5k ticks/s** steady-state single-process. `native/rl/autoresearch/notes.md` reports ~2,000 ticks/s per worker under full pool load (47.5k aggregate over 16 workers).

## Trace format (`bopl-managed-state-v3`)

Gzipped JSONL (`BOPL_TRACE=path.jsonl.gz`). Row kinds, in order:

1. `header` — `schema`, `reference` (pinned source commit `67484d0…`), `unity` version, `batch`, `limit`, `replayFormat` (`BOPLREP2` when driven by a replay), `setup` (from `ReplaySession.Playing`: `sourceHash`, `catalogHash`, `seed`, `level`, `gameMode`, `stocks`, `players[]` with id/color/team/keyboard/abilities), `scope`. `diff.py` compares `schema` + `setup` only; `limit`/`batch` legitimately differ between capture setups.
2. `input` rows — `index`, `value` = the tick's `InputPacketQuad` (or `InputPacket[]`), field-encoded like any struct.
3. `state` rows — `index`, `value` with these sections (what `--summary` reports):
   - `tick` (`Updater.SimulationTicks`), `scene` (active scene name)
   - `physicsStage` — only when `BOPL_TRACE_PHYSICS=1` (per-stage DetPhysics capture from `ReferencePhysicsCapture`)
   - `playerBodyStages`, `playerCollisionStages`, `playerLifeStages` — intra-tick stage snapshots pushed by instrumented gameplay
   - `events` — `ability_entry` events (`player`, `slot`, `prefab`, `instance` identity)
   - `updater` — selected `Updater` statics (`random`, `SimTimePassed`, destroy queues, …)
   - `time` — all `GameTime` statics
   - `players` — `PlayerHandler.Get().PlayerList()`
   - `updatables` — every `Updater.updatables` entry: `id` (identity), `active`, `enabled`, `state` = all declared instance fields via reflection
   - `transforms` — every `FixTransform` (incl. inactive), sorted by identity
   - `facing` — every `FlippingTransform`, sorted by identity
   - `circles` / `boxes` / `roundedRects` — `DetPhysics` body lists: per body `collider`, `body`, `parent` identity
   - `session` — selected `GameSessionHandler` statics; `sessionInstance` — its instance fields
   Rows strictly alternate `input`/`state` with matching `index` (`read_trace` enforces this).
4. `end` — `ticks`, `reason` (`tick-limit` | `simulation-exception` | replay end; transport-specific, ignored by the differs), `abilityEntries` (name → count).
5. On a thrown tick: a sidecar `<trace>.failure.json` (`bopl-replay-failure-v1`) with the exception, the last input, and the completed physics stage; the trace itself ends with `end.reason = "simulation-exception"`.

### Value encoding (`ReferenceCapture.Encoder`)

- `Fix` → `{"$fix": m_rawValue}`; `float` → `{"$f32": IEEE bits}`; `double` → `{"$f64": bits}`; enums → `{"$enum": type, "value": n}`; strings/bools/ints raw.
- `UnityEngine.Object` references → `{"$unity": "<identity>"}` (scheme below); destroyed → null.
- `AnimationCurve` → `{"$curve": keys, preWrap, postWrap}`; delegates → invocation list (method name + target).
- Reference-type cycles → `{"$ref": "<path first seen>"}` — the path where a shared object is FIRST encountered is part of the format, so the port must produce identical enumeration order.
- Dictionaries → entry list sorted by encoded key; other `IEnumerable` → array in iteration order.
- Objects from the game assembly / `Unity.Mathematics` / value types → `ObjectFields`: all declared instance fields, walking up to (excluding) `MonoBehaviour`/`ScriptableObject`/`object`, sorted by `DeclaringType.FullName + "." + Name`, keyed `DeclaringType.Name.fieldName`.
- Everything else → `{"$unsupported": type}` (diagnostic only).

### Identity and GetInstanceID

Object references in traces never use raw instance ids. `Encoder.Identity`:

- `MonoUpdatable` → `TypeName:HierarchyNumber`. **`HierarchyNumber` is the game's own monotonic counter, deterministic across processes — the port MUST reproduce it.**
- other `Component` → `<transform path>:TypeName`; `GameObject` → transform path. Transform paths anchor at the nearest `MonoUpdatable` (`entity:N/...`), at a `FollowTransform` leader (`follow:...`), or at the scene root with sibling ordinals.

Raw `GetInstanceID()` integers do leak into three captured gameplay fields, listed in `native/data/process-identity-fields.json` and excluded from every comparison (they are process-global in Unity, a monotonic shim counter in the port — incomparable by construction):

- `Shockwave.gameObjectIdsHit`, `Shockwave.extraObjectsToIgnore` (HashSet<int> of DetPhysics `hits[i].instanceId`)
- `Roll.platformHopBuffer` (int[5] of attached-ground instance ids; unused by group 0, matters for Roll loadouts)

The level-0 2-player oracle diverged at tick 463 on exactly `Shockwave.gameObjectIdsHit` before this exclusion existed. Also relevant: the encoder itself skips fields named `instanceId` / `instanceIdToColliderIndex`, and translates `CompositeBody.{circleIds,boxIds,roundedRectIds}` (instance ids) into stable identities via the DetPhysics registration tables. `Mjolnir.alreadyKilled` looks similar but stores *player ids* — deterministic, compared. `DetPhysics.activeIds`/`idReflectedLastFrame` hold instance ids but `DetPhysics` is not in `Updater.updatables`, so they never reach a trace.

### Env vars

| var | default | meaning |
| --- | --- | --- |
| `BOPL_TRACE` | unset | output path; `.gz` suffix enables gzip. Tracing is on iff set |
| `BOPL_TRACE_TICKS` | 600 | tick limit; capture finishes with `end.reason=tick-limit` |
| `BOPL_TRACE_BATCH` | 1 | simulation ticks per player-loop frame while tracing (`verify_replay` uses 16) |
| `BOPL_TRACE_PHYSICS` | off | `1` adds the per-stage `physicsStage` section (requires `BOPL_TRACE`) |

## Resources.Load audit

Gameplay (`Assets/Scripts/Assembly-CSharp`, instrumentation `Reference*`/`Rl*` files aside) contains exactly **one** `Resources.Load` call:

| call site | path | in `assets.json`? |
| --- | --- | --- |
| `ReplaySession.SourceHash` (ReplaySession.cs:13) | `SimulationReference` (TextAsset, `Assets/Resources/SimulationReference.txt`) | **No** — the export covers prefabs and ScriptableObjects only. The file is one line: `d0562ac89d03d46f24f122a08f4f5366c675cc94a015a4ccaf7adc124af089b7` (the pinned source hash, also in every trace header `setup.sourceHash` and `.rep` archive). The loader should special-case it, e.g. read the file directly or embed the constant. |

`Resources.FindObjectsOfTypeAll` appears only in `Reference*` instrumentation files, never on the simulation path. All other prefab/asset access in gameplay goes through serialized references, which the export resolves by GlobalObjectId.
