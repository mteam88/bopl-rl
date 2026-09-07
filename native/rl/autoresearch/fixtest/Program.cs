using System.Diagnostics;
using System.Runtime.CompilerServices;

static class Ref
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int CountLeadingZeroes(ulong x)
    {
        int num = 0;
        while ((x & 0xF000000000000000uL) == 0L) { num += 4; x <<= 4; }
        while ((x & 0x8000000000000000uL) == 0L) { num++; x <<= 1; }
        return num;
    }
    // verbatim Fix.operator/ on raw values
    public static long Div(long rawValue, long rawValue2)
    {
        if (rawValue2 == 0L) throw new DivideByZeroException();
        ulong num = (ulong)((rawValue >= 0) ? rawValue : (-rawValue));
        ulong num2 = (ulong)((rawValue2 >= 0) ? rawValue2 : (-rawValue2));
        ulong num3 = 0uL;
        int num4 = 33;
        while ((num2 & 0xF) == 0L && num4 >= 4) { num2 >>= 4; num4 -= 4; }
        while (num != 0L && num4 >= 0)
        {
            int num5 = CountLeadingZeroes(num);
            if (num5 > num4) num5 = num4;
            num <<= num5;
            num4 -= num5;
            ulong num6 = num / num2;
            num %= num2;
            num3 += num6 << num4;
            if ((num6 & ~(ulong.MaxValue >> num4)) != 0L)
            {
                if (((rawValue ^ rawValue2) & long.MinValue) != 0L) return long.MinValue;
                return long.MaxValue;
            }
            num <<= 1;
            num4--;
        }
        num3++;
        long num7 = (long)(num3 >> 1);
        if (((rawValue ^ rawValue2) & long.MinValue) != 0L) num7 = -num7;
        return num7;
    }
    // verbatim Fix.Sqrt on raw values
    public static long Sqrt(long rawValue)
    {
        if (rawValue < 0) throw new ArgumentOutOfRangeException();
        ulong num = (ulong)rawValue;
        ulong num2 = 0uL;
        ulong num3;
        for (num3 = 4611686018427387904uL; num3 > num; num3 >>= 2) { }
        for (int i = 0; i < 2; i++)
        {
            while (num3 != 0L)
            {
                if (num >= num2 + num3) { num -= num2 + num3; num2 = (num2 >> 1) + num3; }
                else { num2 >>= 1; }
                num3 >>= 2;
            }
            if (i == 0)
            {
                if (num > uint.MaxValue) { num -= num2; num = (num << 32) - 2147483648u; num2 = (num2 << 32) + 2147483648u; }
                else { num <<= 32; num2 <<= 32; }
                num3 = 1073741824uL;
            }
        }
        if (num > num2) num2++;
        return (long)num2;
    }
}

static class Cand
{
    const ulong DivFallback = 1UL << 61;

    // Exact replacement for Fix.operator/: q = floor(|x| * 2^33 / |y|), result = (q+1)>>1 with sign,
    // saturating when q >= 2^64. One hardware division for the integer part; the 33 fraction bits come
    // from a second hardware division when it cannot overflow, else from a double estimate corrected
    // exactly with a signed residual (|residual| < 2^62 because d < 2^60 and the estimate is within 1).
    public static long Div(long rawValue, long rawValue2)
    {
        if (rawValue2 == 0L) throw new DivideByZeroException();
        ulong n = (ulong)((rawValue >= 0) ? rawValue : (-rawValue));
        ulong d = (ulong)((rawValue2 >= 0) ? rawValue2 : (-rawValue2));
        bool negative = ((rawValue ^ rawValue2) & long.MinValue) != 0L;
        if (d >= (1UL << 60) || n >= (1UL << 60)) return Ref.Div(rawValue, rawValue2);
        ulong hi = n / d;
        if (hi >= (1UL << 31)) return negative ? long.MinValue : long.MaxValue;
        ulong r = n - hi * d;
        ulong lo;
        if (d < (1UL << 31))
        {
            lo = (r << 33) / d;
        }
        else
        {
            lo = (ulong)((double)(long)r * 8589934592.0 / (double)(long)d);
            long t = (long)((r << 33) - lo * d);
            while (t < 0) { lo--; t += (long)d; }
            while (t >= (long)d) { lo++; t -= (long)d; }
        }
        ulong q = (hi << 33) + lo;
        q++;
        long result = (long)(q >> 1);
        return negative ? -result : result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong Mul128(ulong a, ulong b, out ulong lo)
    {
        ulong a0 = (uint)a, a1 = a >> 32, b0 = (uint)b, b1 = b >> 32;
        ulong p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
        ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
        lo = (mid << 32) | (uint)p00;
        return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    }

    // Exact replacement for Fix.Sqrt: r = floor(sqrt(N * 2^32)), +1 if remainder > r.
    public static long Sqrt(long rawValue)
    {
        if (rawValue < 0) throw new ArgumentOutOfRangeException();
        // Above 2^62 the original's second pass takes its overflow branch, whose final rounding
        // compare is not the ideal one; keep the original there (never reached by gameplay values).
        if (rawValue >= (1L << 61)) return Ref.Sqrt(rawValue);
        ulong n = (ulong)rawValue;
        ulong r = (ulong)Math.Sqrt((double)rawValue * 4294967296.0);
        ulong xhi = n >> 32, xlo = n << 32;
        // bring r to floor(sqrt(X)) with exact 128-bit compares
        while (true) {
            ulong rhi = Mul128(r, r, out ulong rlo);
            if (rhi > xhi || (rhi == xhi && rlo > xlo)) { r--; continue; }
            break;
        }
        while (true) {
            ulong s = r + 1;
            ulong shi = Mul128(s, s, out ulong slo);
            if (shi < xhi || (shi == xhi && slo <= xlo)) { r = s; continue; }
            break;
        }
        ulong r2hi = Mul128(r, r, out ulong r2lo);
        ulong rem = xlo - r2lo;   // remainder < 2r+1 < 2^49, fits
        if (rem > r) r++;
        return (long)r;
    }
}

static class Program
{
    static ulong NextU(Random rng) { Span<byte> b = stackalloc byte[8]; rng.NextBytes(b); return BitConverter.ToUInt64(b); }
    static long Sample(Random rng, int mode)
    {
        ulong u = NextU(rng);
        switch (mode) {
            case 0: return (long)u;                                  // full range
            case 1: { int bits = rng.Next(1, 64); return (long)(u >> (64 - bits)) * (rng.Next(2) == 0 ? 1 : -1); } // log-uniform magnitude
            case 2: { long v = (long)(u >> (64 - rng.Next(24, 48))); return rng.Next(2) == 0 ? v : -v; }         // gameplay-ish
            case 3: return rng.Next(2)==0 ? (long)(u >> (64 - rng.Next(1, 20))) : (long)(u >> 1) | (1L << 61);      // tiny, or huge (>= 2^61)
            default: { long v = (long)(u >> rng.Next(0, 64)); return v << rng.Next(0, 8); } // trailing zeros
        }
    }
    static void Main(string[] args)
    {
        string what = args.Length > 0 ? args[0] : "test";
        long iters = args.Length > 1 ? long.Parse(args[1]) : 20_000_000;
        var rng = new Random(12345);
        if (what == "test") {
            long[] specials = { 0, 1, -1, 2, -2, 3, 4294967296L, -4294967296L, 4294967295L, 4294967297L, long.MaxValue, long.MinValue, long.MaxValue - 1, long.MinValue + 1,
                (1L << 62), -(1L << 62), (1L << 61), (1L << 61) - 1, (1L << 61) + 1, (1L << 31), (1L << 31) - 1, (1L << 31) + 1, (1L << 32) + 1, 6148914691236517205L, 6148914691236517206L, 0x7FFFFFFF00000000L, 0x00000000FFFFFFFFL, 0x100000000L * 100, 42949672960L, 13493037704L };
            long bad = 0, n = 0;
            foreach (var a in specials) foreach (var b in specials) {
                if (b == 0) continue; n++;
                long r = Ref.Div(a, b), c = Cand.Div(a, b);
                if (r != c) { bad++; if (bad < 20) Console.WriteLine($"DIV MISMATCH a={a} b={b} ref={r} cand={c}"); }
            }
            foreach (var a in specials) { if (a < 0) continue; n++; long r = Ref.Sqrt(a), c = Cand.Sqrt(a); if (r != c) { bad++; Console.WriteLine($"SQRT MISMATCH a={a} ref={r} cand={c}"); } }
            // saturation borderline: n*2^33/d ~ 2^64  <=> n ~ d*2^31
            for (int k = 0; k < 2000000; k++) {
                ulong d = (NextU(rng) >> (64 - rng.Next(1, 31))) | 1UL; if (d >= (1UL<<30)) d >>= 1; if (d==0) d=1;
                long nb = (long)(d << 31) + rng.Next(-4, 5);
                foreach (var (a, b) in new[]{ (nb, (long)d), (-nb, (long)d), (nb, -(long)d), (-nb, -(long)d) }) {
                    if (b == 0) continue;
                    long r = Ref.Div(a, b), c = Cand.Div(a, b);
                    if (r != c) { bad++; if (bad < 20) Console.WriteLine($"DIV MISMATCH(sat) a={a} b={b} ref={r} cand={c}"); }
                    n++;
                }
                // near 2^60 boundaries for either operand, and d in [2^31, 2^60) with r close to d
                long big = (1L << 60) + rng.Next(-3, 4);
                long e = (long)(NextU(rng) >> (64 - rng.Next(31, 60)));
                if (e != 0) { foreach (var (a, b) in new[]{ (big, e), (e, big), (e - 1, e), (e + 1, e), ((e << 1) - 1, e) }) {
                    if (b == 0) continue; long r = Ref.Div(a, b), c = Cand.Div(a, b);
                    if (r != c) { bad++; if (bad < 20) Console.WriteLine($"DIV MISMATCH(edge) a={a} b={b} ref={r} cand={c}"); } n++; } }
                // sqrt near perfect squares
                ulong sq = NextU(rng) >> (64 - rng.Next(1, 48)); long v = (long)(sq * sq >> 32); if (v >= 0) { foreach (long dd in new long[]{-2,-1,0,1,2}) { long a = v + dd; if (a < 0) continue; long rs = Ref.Sqrt(a), cs = Cand.Sqrt(a); if (rs != cs) { bad++; if (bad < 20) Console.WriteLine($"SQRT MISMATCH(sq) a={a} ref={rs} cand={cs}"); } n++; } }
            }
            for (long i = 0; i < iters; i++) {
                int mode = (int)(i % 5);
                long a = Sample(rng, mode), b = Sample(rng, (int)((i / 5) % 5));
                if (b == 0) continue;
                long r = Ref.Div(a, b), c = Cand.Div(a, b);
                if (r != c) { bad++; if (bad < 20) Console.WriteLine($"DIV MISMATCH a={a} b={b} ref={r} cand={c}"); }
                long s = a < 0 ? -a : a; if (s < 0) s = long.MaxValue;
                long rs = Ref.Sqrt(s), cs = Cand.Sqrt(s);
                if (rs != cs) { bad++; if (bad < 20) Console.WriteLine($"SQRT MISMATCH a={s} ref={rs} cand={cs}"); }
                n += 2;
            }
            Console.WriteLine($"checked {n} cases, mismatches {bad}");
        } else if (what == "bench") {
            int N = 1 << 20;
            var xs = new long[N]; var ys = new long[N];
            for (int i = 0; i < N; i++) { xs[i] = Sample(rng, 2); do { ys[i] = Sample(rng, 2); } while (ys[i] == 0); }
            long acc = 0;
            foreach (var (name, f) in new (string, Func<long,long,long>)[] { ("ref div", Ref.Div), ("cand div", Cand.Div) }) {
                for (int w = 0; w < 3; w++) { var sw = Stopwatch.StartNew(); for (int k = 0; k < 8; k++) for (int i = 0; i < N; i++) acc += f(xs[i], ys[i]); sw.Stop(); if (w == 2) Console.WriteLine($"{name}: {sw.Elapsed.TotalMilliseconds * 1e6 / (8.0 * N):F1} ns/op"); }
            }
            foreach (var (name, f) in new (string, Func<long,long>)[] { ("ref sqrt", Ref.Sqrt), ("cand sqrt", Cand.Sqrt) }) {
                for (int w = 0; w < 3; w++) { var sw = Stopwatch.StartNew(); for (int k = 0; k < 8; k++) for (int i = 0; i < N; i++) acc += f(xs[i] < 0 ? -xs[i] : xs[i]); sw.Stop(); if (w == 2) Console.WriteLine($"{name}: {sw.Elapsed.TotalMilliseconds * 1e6 / (8.0 * N):F1} ns/op"); }
            }
            Console.WriteLine(acc);
        }
    }
}
