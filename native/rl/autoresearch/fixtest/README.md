# Fix div/sqrt exactness harness (exp08)

`Ref` is the game's original bit-serial `Fix.operator/` and digit-by-digit `Fix.Sqrt`
copied verbatim (raw `long` in, raw `long` out). `Cand` is the fast replacement now in
`Assets/Scripts/Assembly-CSharp/BoplFixedMath/Fix.cs`. `test N` compares them on special
values, saturation boundaries, 2^60/2^61 edges, near-perfect squares and N random pairs
drawn from five magnitude distributions; `bench` times both.

```
dotnet build -c Release
dotnet bin/Release/net8.0/fixtest.dll test 100000000   # 2026-09-07: 224223723 cases, 0 mismatches
dotnet bin/Release/net8.0/fixtest.dll bench            # div 21 -> 6.5 ns, sqrt 106 -> 10 ns (7600X3D, .NET 8)
```
