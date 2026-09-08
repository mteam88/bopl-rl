# Fixed-point arithmetic correctness

The project compiles the actual `Assets/Scripts/Assembly-CSharp/BoplFixedMath/Fix.cs`.
`Cand` is a thin call wrapper; it contains no arithmetic implementation. `Ref` is
an independent test oracle containing the original bit-serial division and
digit-by-digit square root. The game itself uses one implementation for all inputs.

`test N E` checks exceptions, special values, saturation boundaries, large-operand
edges, square-root rounding boundaries constructed with wide integers, E batches
of boundary cases, and N random pairs from five magnitude distributions. E defaults
to 2,000,000. Any mismatch makes the process fail. The standalone profiling stub
only disables counters; it does not replace any arithmetic.

For large square-root inputs, the original result is the rounded integer square
root of `(raw << 32) - 2^30` when `raw - floor(sqrt(raw))^2 > UInt32.MaxValue`.
Otherwise it is the rounded integer square root of `raw << 32`. The implementation
preserves that exact high-range rule with a corrected floating-point estimate and
integer product comparisons. Division uses exact 128-bit product comparisons when
a signed residual cannot safely represent the correction.

```
dotnet build -c Release
dotnet bin/Release/net8.0/fixtest.dll test 100000000
dotnet bin/Release/net8.0/fixtest.dll test 100000 10000
dotnet bin/Release/net8.0/fixtest.dll bench
```
