# Source reuse and translation audit

The RL target permits any implementation language that provides fast stepping
and a C interface for PufferLib. The complete gameplay, one-through-six-player,
map, ability, lifecycle and replay requirements remain. The handwritten Rust
runtime is still partial. Do not keep expanding it mechanically before comparing
its development cost with executing the existing game code.

Run from the Unity project root with .NET 8 and the reference Linux build present:

```sh
dotnet run --project native/tools/source-audit/source-audit.csproj -c Release -- \
  . native/artifacts/source-audit/report.json
```

The tool uses the Roslyn compiler shipped with the SDK, without a NuGet package
dependency. It reads source and serialized assets directly from the commit in
`native/data/full-reference.json`, checks their pinned hashes, compiles the C#
semantic model against the built game's managed dependencies, and refuses to
report success with compiler errors. The JSON records dependency assembly hashes,
type references, explicit external members, operator/conversion bindings and
source examples. It excludes later reference instrumentation from the source set.

At source `c5b73bb0e17f2a87a7c19d3de24d78403a910e77`, the audit found:

- 393 C# files, 439 declared types, and zero compiler errors.
- 1,504 pinned `.prefab`, `.unity` and `.asset` files; script GUID references
  select 234 source types, conservatively including sibling and nested types.
- Adding all `MonoUpdatable` subclasses and seven central runtime classes gives
  260 roots and 386 types reached through static type references.
- That graph names 148 Unity-related types and 649 distinct named symbols.
  Components, transforms and game objects have the most references.
- `DetPhysics` directly names only `Physics2D.GetIgnoreLayerCollision` from the
  Unity 2D physics API. The main solver is game-owned C#.

These are dependency counts, not an estimate of required shim implementations.
The graph includes presentation, menus and networking. Conversely, dynamic
dispatch, reflection and persistent callbacks need separate auditing. Implicit
Unity object comparisons and conversions are recorded separately: a replacement
runtime must preserve their behavior, including destroyed-object checks.

The immediate route is to expose actual headless Unity stepping through a C
interface and measure complete episodes including observations and resets. This
preserves existing gameplay and provides an end-to-end throughput baseline.
If process/engine overhead is too expensive, reuse C# with a headless Unity
compatibility layer and evaluate NativeAOT behind the same interface. Neither
standalone compatibility nor acceptable training throughput has been established
by this source audit. Existing replay and differential artifacts remain useful
for validating either route.
