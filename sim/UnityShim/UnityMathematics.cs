using System;
using System.Runtime.CompilerServices;

namespace Unity.Mathematics
{
	// Bit-exact copy of Unity.Mathematics.Random (decompiled from the game's
	// shipped Assets/Plugins/Unity.Mathematics.dll). Gameplay determinism
	// (Updater.RandomFix etc.) depends on this exact xorshift sequence.
	[Serializable]
	public struct Random
	{
		public uint state;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Random(uint seed)
		{
			state = seed;
			NextState();
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Random CreateFromIndex(uint index) => new Random(WangHash(index + 62));

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		internal static uint WangHash(uint n)
		{
			n = n ^ 0x3Du ^ (n >> 16);
			n *= 9;
			n ^= n >> 4;
			n *= 668265261;
			n ^= n >> 15;
			return n;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void InitState(uint seed = 1851936439u)
		{
			state = seed;
			NextState();
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public bool NextBool() => (NextState() & 1) == 1;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public int NextInt() => (int)NextState() ^ int.MinValue;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public int NextInt(int max) => (int)(NextState() * (ulong)max >> 32);

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public int NextInt(int min, int max)
		{
			uint num = (uint)(max - min);
			return (int)((ulong)((long)NextState() * (long)num) >> 32) + min;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public uint NextUInt() => NextState() - 1;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public uint NextUInt(uint max) => (uint)((ulong)((long)NextState() * (long)max) >> 32);

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public uint NextUInt(uint min, uint max)
		{
			uint num = max - min;
			return (uint)(int)((ulong)((long)NextState() * (long)num) >> 32) + min;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public float NextFloat() => math.asfloat(0x3F800000u | (NextState() >> 9)) - 1f;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public float NextFloat(float max) => NextFloat() * max;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public float NextFloat(float min, float max) => NextFloat() * (max - min) + min;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public double NextDouble()
		{
			ulong num = ((ulong)NextState() << 20) ^ NextState();
			return math.asdouble(0x3FF0000000000000uL | num) - 1.0;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private uint NextState()
		{
			uint result = state;
			state ^= state << 13;
			state ^= state >> 17;
			state ^= state << 5;
			return result;
		}
	}

	public static class math
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static unsafe float asfloat(uint x) => *(float*)&x;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static unsafe uint asuint(float x) => *(uint*)&x;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static unsafe double asdouble(ulong x) => *(double*)&x;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static unsafe ulong asulong(double x) => *(ulong*)&x;

		public static float min(float a, float b) => a < b ? a : b;
		public static float max(float a, float b) => a > b ? a : b;
		public static int min(int a, int b) => a < b ? a : b;
		public static int max(int a, int b) => a > b ? a : b;
		public static float clamp(float x, float a, float b) => max(a, min(b, x));
		public static int clamp(int x, int a, int b) => max(a, min(b, x));
		public static float abs(float x) => Math.Abs(x);
		public static float sqrt(float x) => (float)Math.Sqrt(x);
		public static float floor(float x) => (float)Math.Floor(x);
		public static float ceil(float x) => (float)Math.Ceiling(x);
		public static float lerp(float a, float b, float t) => a + t * (b - a);
	}
}
