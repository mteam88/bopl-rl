using System;
using BoplFixedMath;

public static class MathOracle
{
    public static void Main()
    {
        string line;
        while ((line = Console.ReadLine()) != null)
        {
            string[] parts = line.Split('\t');
            string op = parts[0];
            Fix a = Fix.FromRaw(long.Parse(parts[1]));
            Fix b = Fix.FromRaw(long.Parse(parts[2]));
            try
            {
                Fix result = op switch {
                    "add" => a + b, "sub" => a - b, "mul" => a * b, "div" => a / b,
                    "slow_add" => Fix.SlowAdd(a,b), "slow_sub" => Fix.SlowSub(a,b),
                    "slow_mul" => Fix.SlowMul(a,b), "rem" => a % b, "slow_rem" => Fix.SlowMod(a,b),
                    "neg" => -a, "abs" => Fix.Abs(a), "slow_abs" => Fix.SlowAbs(a),
                    "floor" => Fix.Floor(a), "ceil" => Fix.Ceiling(a), "round" => Fix.Round(a),
                    "sqrt" => Fix.Sqrt(a), "sin" => Fix.Sin(a), "cos" => Fix.Cos(a),
                    "fast_sin" => Fix.FastSin(a), "fast_cos" => Fix.FastCos(a), "tan" => Fix.Tan(a),
                    "acos" => Fix.Acos(a), "atan" => Fix.Atan(a), "atan2" => Fix.Atan2(a,b),
                    _ => throw new ArgumentException("unknown operation: " + op)
                };
                Console.WriteLine(result.m_rawValue);
            }
            catch (DivideByZeroException) { Console.WriteLine("division_by_zero"); }
            catch (ArgumentOutOfRangeException) { Console.WriteLine("domain"); }
            catch (OverflowException) { Console.WriteLine("overflow"); }
        }
    }
}
