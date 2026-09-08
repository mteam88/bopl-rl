# NativeAOT feasibility spike (day 3)

Verdict: **feasible, working today.** After adding reflection roots (`sim/Host/rd.xml`,
~15 lines), the NativeAOT binary runs the full sim, the tracer works, and the traced
output is **bit-identical to CoreCLR over all 480 ticks of the l0-p2 fixture with zero
exclusions**. Steady-state throughput is ~80% of warmed JIT, but cold throughput is
~6x JIT cold and process startup is near-instant, which is the shape RL rollouts want.

Environment: linux-x64, .NET 8, clang 22.1.8, `dotnet publish sim/Host -c Release
-r linux-x64 -p:PublishAot=true -p:StripSymbols=false`, HEAD = merge of shim rift
(`6bd955c`).

## ILC compile

| metric | no rd.xml | with rd.xml |
|---|---|---|
| publish wall time | 27.6 s | 18.9-21 s |
| peak RSS (publish tree) | 250 MB | 196 MB |
| binary size (unstripped) | 52.7 MB | 69 MB |

- The `Fix.cs` SinLut/TanLut 2x205k `long[]` initializers were **not** a problem —
  no pathological ILC time or memory. No `.bin` swap needed.
- The +16 MB with rd.xml is the price of `Dynamic="Required All"` on Gameplay +
  UnityShim (reflection metadata + reflection-invokable stubs for everything).
  `StripSymbols=true` would shrink both numbers; not measured in this spike.

## AOT warnings

44 warnings with rd.xml (57 without): IL2075 x11, IL2070 x8, IL2026 x7, IL2067 x6,
IL2072 x4, IL3053 x4, IL3050 x2, IL2104 x1, IL2062 x1. All are *analysis* warnings
("reflection may break if the target got trimmed"), clustered exactly where
`sim/Oracle/AOT.md` predicted:

- `sim/Export/PropertyApplier.cs` — `ResolveMember`/`FindField`/`AllocateSized`/`CreateInstance`
- `sim/Export/ExportDatabase.cs` — `ResolveType`/`BuildAsset`/`CreatePlaceholder`
- `sim/UnityShim/CloneRules.cs` — `CloneValue`/`CloneScriptableObject`
- `sim/UnityShim/Object.cs`/`Component.cs` — `MonoBehaviour.MessagesFor`, `AddComponent`
- `sim/UnityShim/Events.cs` — `PersistentCall.FindMethod`
- `sim/UnityShim/ScriptableObject` — `DispatchOnEnable`
- `sim/Host/Program.cs` — `SetPrivateField`
- Newtonsoft.Json / System.Linq.Expressions / Microsoft.CSharp / System.Private.Xml
  (IL3053/IL2104 assembly-level) — expected and acceptable; tracing runs fine anyway
  (see below).

With the rd.xml roots in place these warnings are satisfied at runtime; they cannot be
silenced without annotating the loader (tracked in AOT.md).

## Runtime breakage found (and fixed): 2 call-site groups

Without any rd.xml the binary compiles but **fails at runtime**, and the failure mode
is nasty — see "hazard" below.

1. **`ExportDatabase.BuildAsset` → `Activator.CreateInstance(type, true)`**
   (`sim/Export/ExportDatabase.cs:216`, also `:276` CreatePlaceholder, and
   `CloneRules.cs:134,142`): `MissingMethodException: No parameterless constructor
   defined for type 'Settings'`. Types reach these calls via
   `ExportDatabase.ResolveType(string)` (`ExportDatabase.cs:314`), so ILC cannot see
   which ctors to keep. **Fix:** `sim/Host/rd.xml` roots the `Gameplay` and
   `UnityShim` assemblies `Dynamic="Required All"`. This also covers every
   `GetField`/`GetMethod`-by-name site (PropertyApplier, MessagesFor,
   PersistentCall.FindMethod, SetPrivateField).

2. **`PropertyApplier.AllocateSized`/`CreateInstance` →
   `Activator.CreateInstance(typeof(List<NamedSprite>))`**
   (`sim/Export/PropertyApplier.cs:217,235`): assembly-level rd.xml roots do NOT
   cover constructed CoreLib generics, and `List<NamedSprite>` is a value-type
   instantiation (NamedSprite is a struct) so there is no `__Canon` sharing to fall
   back on. Every element apply under `sprites.Array.*` failed with "no parameterless
   ctor for List`1". **Fix:** explicit rd.xml entry
   `System.Collections.Generic.List`1[[NamedSprite,Gameplay]]`.
   Why only this one: gameplay classes initialize their serialized `List<T>` fields
   inline (`= new List<T>()`), so `AllocateSized` finds `current != null` and never
   calls Activator. `NamedSpriteList.sprites` has no initializer. **Residual risk:**
   any other serialized List/array field without a field initializer, on assets not
   exercised by l0-p2 (other levels, WinnerTriangle scene). The closed set is
   enumerable by scanning gameplay serialized fields; if more trip, generate the
   rd.xml entries from that scan instead of whack-a-mole.

### Hazard: reflection failures are swallowed into never-dumped warnings

`PropertyApplier.ApplyOne`'s catch turns the Activator failure into `db.Warn(...)`,
and warnings only print at the next scene load — bootstrap asset loads happen before
any. So breakage #2 surfaced as a `NullReferenceException`-shaped crash 30 frames
downstream (`ReplaySession.get_CatalogHash` LINQ over a null `sprites`). This spike
adds `db.DumpWarnings("bootstrap")` at the end of Host bootstrap (and makes
`DumpWarnings` public) so loader failures are visible immediately. Recommend the shim
thread keep that or add a strict mode that throws on `no parameterless ctor` /
`asset type missing` warnings.

## Runtime results (all with rd.xml)

Untraced `reset --level 0 --players 2 --ticks 2400`:

| metric | NativeAOT | CoreCLR JIT |
|---|---|---|
| cold ticks/s (first episode) | **74,060** | ~12,000 |
| steady ticks/s (in-process episodes 2-5, 4800 ticks each) | 70,400-72,000 | 85,000-89,000 |
| whole-process wall (2400 ticks incl. bootstrap) | **0.96 s** | ~2.6 s (`dotnet run`) |
| `ResetRl` per episode | **3.3 ms** | ~15 ms |
| GC / alloc per 4800-tick episode | ~0 collections, 5.2 MB | same |

- AOT steady state is ~80% of the warmed JIT plateau (no dynamic PGO / tiered
  recompilation), but there is **no warmup cliff at all**: the first 2400 ticks run at
  74k vs JIT's 12k. For short-lived rollout workers AOT wins outright; for a
  long-lived vectorized host JIT is ~20% faster after ~5k ticks.
- Reset being 4.5x faster under AOT also matters for RL (reset-heavy workloads).

Traced (`BOPL_TRACE=...` replay of `sim/Oracle/fixtures/l0-p2/episode.rep`, 480 ticks):

- **Works.** Newtonsoft-based ReferenceCapture runs under AOT against the rooted
  gameplay types: 86 ticks/s (CoreCLR traced ~120-130, Unity traced 40-50).
- `python3 sim/Oracle/diff.py jit-trace aot-trace` → **`match: 480 ticks`, exit 0,
  no exclusions**. The AOT binary is simulation-equivalent to CoreCLR bit-for-bit.
- Against the Unity oracle the AOT trace has exactly the same standing as CoreCLR:
  identical except the known export-gap fields (`Player.playerDied` subscriber from
  the unexported menu flow).

Caution for future measurement: an early comparison showed AOT "diverging" from
CoreCLR on ability identities — it was a stale `--no-build` CoreCLR binary from
before the shim-rift merge. `publish` always rebuilds; `dotnet run --no-build` does
not. Rebuild before comparing.

## What the spike changed (no Gameplay edits)

- `sim/Host/rd.xml` (new): assembly roots + the `List<NamedSprite>` entry.
- `sim/Host/Host.csproj`: `<RdXmlFile>` item, only when `PublishAot=true`.
- `sim/Host/Program.cs`: `db.DumpWarnings("bootstrap")` after bootstrap loads.
- `sim/Export/ExportDatabase.cs`: `DumpWarnings` internal → public (one word).

## Recommendations

1. Adopt rd.xml now; it is small and CoreCLR builds are unaffected.
2. Generate the constructed-generic entries from a scan of gameplay serialized
   fields lacking initializers (oracle can produce this from the export), or have
   the loader fail fast on `no parameterless ctor` so gaps surface in CI.
3. Longer term (AOT.md): replace `ResolveType`/Activator with a generated
   type-factory dispatch table to drop rd.xml and the 16 MB metadata tax.
4. Keep JIT for the long-lived vectorized training host (steady +20%); use AOT for
   short-lived workers, evaluation farms, and anywhere startup dominates.
