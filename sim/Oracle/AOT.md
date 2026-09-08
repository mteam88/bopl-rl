# NativeAOT hazards on the simulation path

Audit of `Assets/Scripts/Assembly-CSharp` at commit `8a66de5`, excluding the
`Reference*`/`Rl*` instrumentation. Verdict up front: **the gameplay code
itself is nearly AOT-clean** — no `dynamic`, no expression trees, no
`Activator`/`MakeGenericType`, no `Type.GetType(string)`, no
`Enum.Parse/GetValues`, no P/Invoke, no `unsafe`/`stackalloc`/`fixed`
(grep-verified empty). The hazards live in three places: serialized
UnityEvent dispatch, the shim types the port must supply, and anything
outside the sim compile set that gets dragged in.

## 1. UnityEvent persistent calls — the one real reflection dependency

- `InstantAbility.cs:10` `public UnityEvent funcOnEnter`, invoked at :80 on
  ability cast. `QuickSpell.cs:14` `private UnityEvent spellFunc`, invoked
  at :131. **These are on the sim path** and ability prefabs serialize
  persistent calls into them: e.g. assets.json wires
  `m_Target = <sibling component>`, `m_MethodName = "UseAbility"`.
- Unity resolves persistent calls by reflection (method name string →
  `MethodInfo`) at first invoke. Under NativeAOT the target methods would be
  trimmed or the lookup would fail.
- Suggested handling: the shim's `UnityEvent` should NOT use reflection.
  The full method-name set across assets.json + all maps is 26 names, and
  the ones reachable in a match are `UseAbility`, `CastAbility`,
  `CastInvisibility` (ability prefabs; the `On*` names are `PlayerInput`
  UI/input bindings bypassed by `OverrideInputWithNetworkInput`, the rest are
  menu buttons). Implement a static dispatch table
  `(declaring type, method name) → Action<Component>` populated for that
  closed set, and make the loader fail loudly on an unknown name so new
  wiring is caught at load time, not silently skipped.
  Alternative: `[DynamicallyAccessedMembers]`/rd.xml roots on the three
  ability component types — worse, because it hides the dependency.

## 2. Shim-supplied types that must be AOT-friendly

- `Unity.Mathematics.Random` — Updater.cs:5 (using), :45 (field seeded 37u),
  drawn via `RandomInt/RandomFix/...` (Updater.cs:184-214). Pure managed
  math; vendor the reference source of `Unity.Mathematics` (it is MIT
  licensed, plain C#) or reimplement `NextUInt/NextInt` exactly
  (xorshift + bias-free range reduction). Already proven bit-exact by the
  Rust port's 65k-draw fixture. No AOT issue if vendored as source.
- `Unity.Collections.NativeArray<T>` — RopeMesh.cs:7-9 allocates two
  `NativeArray` (80 elems, `Allocator.Persistent`) per rope mesh;
  `DetPhysics.OnDestroy` (DetPhysics.cs:304-307) disposes via
  `ropes.Dispose()`. No Burst, no `IJob`, no `.Schedule(` anywhere
  (grep-verified). Shim it as a struct wrapping `T[]` with a no-op/flag
  `Dispose`; do NOT bring in the real Unity.Collections package (its safety
  system uses statics + finalizers that fight trimming).
- `UnityEngine.Object` fake-null — implemented with operator overloads, no
  reflection needed; just noting the shim must not rely on
  `object.ReferenceEquals` semantics leaking through generics (e.g.
  `List<MonoUpdatable>.Contains` uses `EqualityComparer<T>.Default` →
  `Object.Equals` override must match Unity: destroyed == destroyed is
  reference equality, destroyed == null is true).

## 3. Sites in gameplay sources that are NOT on the sim path but sit in the
same assembly (decide: exclude from compile, or handle)

- `SettingSelector.cs:43` —
  `settings.GetType().GetProperty(setting).GetValue(settings)` with a
  serialized property-name string. Menu UI only. Preferred: keep menu files
  out of the sim compile set. If compiled for convenience, root `Settings`'s
  properties (`[DynamicallyAccessedMembers(PublicProperties)]` on the field
  or an rd.xml entry) — it is the only string-reflection site in 391 files.
- `SteamSocket.cs:46` — `Marshal.Copy` (plus the Steamworks.NET dependency
  with real P/Invoke into steam_api). Networking; must be excluded from the
  sim compile anyway (no Steam in the port). The only `Marshal` use in the
  gameplay sources.
- `Client.cs` / `SteamManager.cs` / `SteamConnection.cs` — `SendMessage`
  matches are `Steamworks.Connection.SendMessage` (network), not Unity's
  reflective `GameObject.SendMessage`. Unity's string-dispatch `SendMessage`
  / `BroadcastMessage` / string `Invoke` / `InvokeRepeating` are used
  NOWHERE — the shim can omit them entirely.
- `MainMenu.cs:92`, `CharacterStatsList.cs:490-618` — the only
  `StartCoroutine` sites; menus. Excluding menus from the compile also
  removes the need for a coroutine scheduler in the shim.

## 4. Data-size and codegen notes (not correctness, but AOT-relevant)

- `BoplFixedMath/Fix.cs` is 42,156 lines because `SinLut`/`TanLut`
  (Fix.cs:68, :20661) are two 205,887-element `long[]` written as C#
  initializers. Roslyn/ILC handle it (Unity already compiles it), but it
  dominates compile time and produces ~3.3 MB of RVA data. If ILC chokes or
  compile times hurt, replace with `ReadOnlySpan<byte>` over an embedded
  resource (`.bin` files already exist at `native/data/sinlut.bin`,
  `tanlut.bin`) — bit-identical loads, sub-second compile. Verify equality
  against the arrays once.
- `DetPhysics.cs:316-325` — the former `Parallel.ForEach` is already
  serialized (comment in source); no `System.Threading.Tasks` parallelism
  remains on the tick path. Keep it that way for determinism, not AOT.
- Generic instantiations are all closed over concrete types
  (`List<MonoUpdatable>`, `PhysicsBodyList<T>` with known T's) — no
  `MakeGenericType`, so ILC sees everything statically. Grep-verified: no
  `Activator.`, no `Assembly.`, no `GCHandle`, no `RuntimeHelpers`.

## 5. Out of scope but adjacent: the tracer

`ReferenceCapture`/`Rl*` are excluded from this audit by instruction, but one
consequence matters for planning: the tracer serializes every gameplay type
via Newtonsoft.Json + `FieldInfo` reflection. That will not survive NativeAOT
trimming without rooting essentially the whole gameplay assembly. Run
oracle-diff builds on CoreCLR (`dotnet run`), where reflection is unrestricted
and speed does not matter, and ship the NativeAOT binary without the tracer
(or behind a build flag). Do not spend effort making the tracer AOT-safe.
