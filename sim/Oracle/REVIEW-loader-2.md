# Loader review, round 2 — Unity serialization rules

Reviewed: the `dotnet-sim` rift at `93bc718` — `sim/Export/ExportDatabase.cs`,
`GraphBuilder.cs`, `PropertyApplier.cs`, `execution-order.json` (commit
`242f258`), plus the shim surfaces the loader contracts with (`Events.cs`,
`Curves.cs`, `World.cs`, `ScriptExecutionOrder.cs`, `UI.cs`,
`Host/Program.cs`; review-1 fixes `76b437d`). Read-only, checked against the
exporter (`Assets/Editor/ReferenceExport.cs`) and the actual
`native/reference-export` JSON (bopl-assets-v2), not against assumptions.

Verdict up front: **the loader is sound on every [EXACT] serialization rule I
could test against the export data.** Three real findings (one likely
divergence source, two hygiene), then the checklist with evidence.

---

## Findings

### L1. Infinity curve tangents decode to −2^31 — float Evaluate loses
constant-mode segments

- The exporter's curve summary applies `(Fix)k.inTangent` (ReferenceExport.cs:
  95-100). Unity constant-mode keyframes carry ±Infinity tangents, and C#'s
  unchecked `(long)(float.PositiveInfinity * 4.2949673E9f)` collapses +∞, −∞
  and NaN to `long.MinValue`. The export contains 4 such keys (found by scan:
  `assets.json` — `AbilitySelectController.OpenAnimation01`,
  `busyDotsAnimation`, and `preGlowAnim` on asset `551e7eb6…`).
- `PropertyApplier.DecodeCurve` (PropertyApplier.cs:355-373) recovers floats by
  Fix-raw preimage search; for `long.MinValue` the preimage found is
  `-2147483648f`, not Infinity. Shim `AnimationCurve.EvaluateSegment`
  special-cases `float.IsInfinity` for constant segments (Curves.cs:128) — the
  −2^31 tangent bypasses it and the Hermite blows up instead of holding
  `k1.value`.
- **Why this is only [PRESENTATION] today**: gameplay's deterministic path is
  `AnimationCurveFixed.Init`, which performs the *same* lossy `(Fix)float` cast
  at runtime in Unity — so Fix raw values round-trip bit-exactly even for the
  infinity keys (the loader's own `FixRawFromFloat(candidate) == raw` check
  proves each preimage). Only float `Evaluate` (presentation code) misbehaves,
  and the 4 affected curves are menu/glow visuals.
- Fix (pick one): decode keys from the flattened `m_Curve.Array.data[N].time/
  value/inSlope/outSlope` **f32 leaves** instead of the Fix-raw summary — the
  exact bits, including ±∞, are already in the JSON next to the summary; or
  keep the summary and map `long.MinValue → float.PositiveInfinity` (sign is
  unrecoverable, but Unity constant-mode uses +∞, and `IsInfinity` doesn't
  care). The first option also removes the preimage search entirely.
- Also verified: no exported curve has nonzero `weightedMode`/weights, so the
  shim ignoring `inWeight/outWeight/weightedMode` is safe.

### L2. Flattened curve/LayerMask children spam the warning channel

- Every AnimationCurve is exported twice: the summary object AND flattened
  `<field>.m_Curve.Array.*` leaves (the exporter does not set
  `enterChildren = false` for curves). The walker resolves `m_Curve` against
  shim `AnimationCurve` (fields: `keys`) → "no member AnimationCurve.m_Curve"
  warnings on every scene/prefab with a curve. Same pattern for LayerMask:
  `<field>` gets the int (correct), then `<field>.m_Bits` fails against shim
  `LayerMask.m_Mask` (Math.cs:854-861). Harmless today, but this noise will
  bury real "no member" misses in `DumpWarnings` output.
- Fix: skip paths containing `.m_Curve.` (and `.m_Bits`) in `ApplyOne`, or —
  if L1 is fixed by reading the leaves — invert: apply leaves, skip the
  summary. Either way the warning stream should stay clean enough that a
  nonzero count is actionable.

### L3. Stale PORTING.md TODOs say settings.json/tags are "blocked on oracle"

- `sim/PORTING.md` TODO still lists "settings.json reader — blocked on oracle
  thread" and "GameObject tag — blocked on oracle thread re-export". Both
  shipped: `sim/Export/settings.json` and the bopl-assets-v2 re-export with
  `tag` are merged into the shim rift (`4b3db3f`), and `ExportDatabase`
  already consumes both (LoadSettings, GraphBuilder tag). The "Untagged"
  fallback (GraphBuilder.cs:18) can only trigger on a v1 export. Drop the
  TODOs — a reader could conclude tags don't work.

---

## Checklist (coordinator's list, with evidence)

1. **Constructor/field initialisers, then serialized overwrite** — correct.
   `GraphBuilder.BuildGameObject` runs pass 1 (ctors via `AddComponentSilent`,
   no lifecycle), registers every exported id, then pass 2 applies properties
   (`PendingProperties`), then activation (`ActivateLoadedObjects`). Field
   initialisers run in the ctor; serialized values overwrite; fields the
   export lacks keep their C# defaults — matching Unity, because the exporter
   walks the full SerializedObject (nothing is missing for serialized fields).
   Two-pass also makes intra-scene forward references resolve.

2. **Only serializable fields written** — by construction: the loader applies
   exactly what the exporter emitted, and the exporter walks
   `SerializedObject.GetIterator()` (serialized fields only). No risk of the
   loader inventing writes to `[NonSerialized]`/static/property state. The
   camelCase `m_X → x` property fallback (PropertyApplier.cs:249-256) only
   fires for shim builtin types whose setters are plain auto-properties;
   `ApplyTransform` intercepts the one type with side-effectful setters.

3. **[FormerlySerializedAs]** — a non-feature, correctly absent: zero
   occurrences in the 391 gameplay files (grep), and the export carries
   *current* field names anyway (SerializedObject on loaded objects reflects
   post-migration names), so the loader never needs the attribute.

4. **Enums / LayerMask** — `Enum.ToObject(target, (long)value)` and
   `(LayerMask)(int)` (PropertyApplier.cs:330-333); shim LayerMask has
   implicit int conversions. Correct. (See L2 for the `.m_Bits` noise.)

5. **Nested structs, arrays, List<T>** — the boxed write-back chain in `Walk`
   (PropertyApplier.cs:125-203) is correct: value-type containers are re-boxed
   and written back through both `NameSegment` (`SetMember`) and
   `IndexSegment` (`list[i] = element`), so `offset.x.m_rawValue` mutates the
   real field. `AllocateSized` preserves prefixes on resize and default-fills
   list slots (Unity does the same). `Array.size` before `data[i]` ordering is
   guaranteed by the exporter's iterator and Newtonsoft's insertion-order
   JObject. String char-array shadows (`m_Name.Array.*`, `m_MethodName.Array.*`)
   are explicitly skipped (:140, :178).

6. **Fix via m_rawValue** — exact. Exported as `SerializedPropertyType.Integer
   → longValue`, applied to the public `long m_rawValue` field through the
   struct chain. Floats ride as raw bits (`{"f32": <int bits>}`) and are
   reconstituted with `Int32BitsToSingle` — bit-exact by construction.

7. **AnimationCurve keys** — the answer to "Fix raw → what type?": the
   summary's Fix raws are *inverted back to floats* whose `(Fix)` cast
   reproduces the exact raw (`FloatFromFixRaw`, self-checking, warns on
   failure), so `AnimationCurveFixed.Init` in gameplay re-derives bit-identical
   Fix values. Verified sound for all finite tangents; see **L1** for ±∞.
   `preWrap/postWrap` map to the same WrapMode numeric values. ✓

8. **Object reference resolution** — correct. Same-file refs resolve via the
   pass-1 id registry; cross-file prefab refs parse the guid out of the
   GlobalObjectId and recursively `BuildAsset` (cycle-safe: the root registers
   in `BuiltAssets` *before* its properties apply, ExportDatabase.cs:165);
   null stays null; GameObject↔Component coercion handles field-type
   mismatches. Unexported assets (sprites/materials/audio) become one stable
   placeholder **per GlobalObjectId** — exactly what
   `abilityIcons.IndexOf(sprite)` needs (SEMANTICS.md §4 sprite identity).
   The UnityEngine/TMPro namespace gate makes a missing *gameplay* asset a
   loud null instead of a silent placeholder. Edge worth remembering: the
   placeholder's concrete type is chosen by the FIRST referencing field; a
   later field of an incompatible type gets null + warning.

9. **Prefab instance modifications in scenes** — already flattened, verified
   in the JSON: the exporter serializes the *loaded* scene, so
   `m_PrefabInstance`/`m_CorrespondingSourceObject` are null everywhere and
   instance overrides are baked into the values. The loader correctly has no
   prefab-patch logic. (Level scenes also contain zero live PrefabInstances.)

10. **m_Enabled / active / layer / tag** — `m_Enabled` writes `enabledSelf`
    directly with lifecycle deferred to the activation pass (matches Unity
    deserialization); `active`/`layer`/`tag` come from the hierarchy node
    (GraphBuilder.cs:17-19). ✓

11. **Transform local TRS** — all ten leaf paths intercepted and written
    through local setters; `m_Father`/`m_Children`/`m_RootOrder` correctly
    ignored (hierarchy and sibling order come from JSON nesting, and roots are
    created in file order = RootOrder). ✓

12. **RectTransform** — `UseRectTransform()` keeps the component in its
    serialized slot; `m_AnchorMin.x` etc. resolve through the camelCase
    property fallback onto UI.cs auto-properties, with correct boxed
    write-back of the Vector2. Instantiate clones preserve the type (review-1
    F9 fix). ✓

13. **UnityEvent m_PersistentCalls** — implemented and correct: the persistent
    tree (`PersistentCallGroup/PersistentCall/ArgumentCache`, Events.cs) is
    plain serialized fields the normal walker fills — `m_Target` resolves via
    GlobalObjectId to the sibling component, `m_MethodName` string applies
    (char-array shadow skipped), `m_Mode`/`m_CallState` ints match Unity's
    enum values. `Invoke()` runs persistent calls before runtime listeners,
    in list order, skipping `Off` — Unity's order. Two notes: (a) the
    arity-only overload fallback in `FindMethod` (Events.cs:70-76) could pick
    a wrong overload — no current consumer, but consider matching parameter
    assignability; (b) reflection dispatch is the AOT hazard from AOT.md §1 —
    fine on CoreCLR now, swap to the closed-set dispatch table before the
    NativeAOT build.

14. **Awake/OnEnable timing of a loaded scene** — scene load, Instantiate and
    SetActive(true) all use one two-pass batch (all Awakes, then all
    OnEnables) stable-sorted by script execution order. **I independently
    re-extracted `executionOrder` from all `Assets/Scripts` `.cs.meta` files:
    162 nonzero entries, `execution-order.json` matches all 162 exactly.**
    Keying by `type.Name` without inheritance propagation matches Unity.
    Remaining modeling choice (documented in PORTING.md, oracle decides):
    Unity's scene load interleaves Awake→OnEnable per execution-order batch
    rather than globally two-phase; every sim-class OnEnable body is
    presentation (SEMANTICS.md §2.6), so a trace divergence from this is
    unlikely — if one appears at tick 0-1, look here first.
    ScriptableObjects get OnEnable after deserialization (BuildAsset), like
    Unity. `ISerializationCallbackReceiver`: only `FixTransform`, empty body,
    correctly skipped.

## Also checked

- **Host CLI == checkpoint.sh contract** (`reset --level N --players 2
  --ticks N`; `replay <rep> --ticks N`; `BOPL_TRACE` env) — no adaptation
  needed. Host's default 16 ticks/frame matches the oracle traces
  (verify_replay.py sets `BOPL_TRACE_BATCH=16`), so frame-cadence-observable
  behavior (raw `Destroy`, `Start` flushes) lines up. ✓
- Menu scene loads are warned no-ops and `nameToBuildIndex` returns −1 for
  unexported names — acceptable for the RL path; round-end
  `LoadScene("CharacterSelect")` never fires inside an episode.
- `root.scene` is set on roots only; children keep the default — safe, no
  gameplay code reads `gameObject.scene` (grep).
- `UnloadCurrentScene` spares DontDestroyOnLoad roots (F5) and fires
  OnDestroy via DestroyImmediate — Unity does the same on Single loads.

## For the coordinator

PORTING.md asks the oracle thread for two exports: `WinnerTriangle.unity`
(round-end UI scene, needed for multi-round parity) and the boot-scene
AudioManager sound-name list (SoundCapture parity). Both are exporter-side
work in this rift — queue them for round 3.
