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
