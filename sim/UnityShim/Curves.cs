using System;

namespace UnityEngine
{
	public enum WrapMode
	{
		Once = 1, Loop = 2, PingPong = 4, Default = 0, ClampForever = 8, Clamp = 1
	}

	[Serializable]
	public struct Keyframe
	{
		// Field names/order match Unity's struct exactly: ReferenceCapture reflects
		// private instance fields and prints them by name (Keyframe.m_Time ...).
		private float m_Time;
		private float m_Value;
		private float m_InTangent;
		private float m_OutTangent;
		private int m_WeightedMode;
		private float m_InWeight;
		private float m_OutWeight;

		public float time { get => m_Time; set => m_Time = value; }
		public float value { get => m_Value; set => m_Value = value; }
		public float inTangent { get => m_InTangent; set => m_InTangent = value; }
		public float outTangent { get => m_OutTangent; set => m_OutTangent = value; }
		public float inWeight { get => m_InWeight; set => m_InWeight = value; }
		public float outWeight { get => m_OutWeight; set => m_OutWeight = value; }
		public WeightedMode weightedMode { get => (WeightedMode)m_WeightedMode; set => m_WeightedMode = (int)value; }

		public Keyframe(float time, float value)
		{
			m_Time = time; m_Value = value;
			m_InTangent = 0f; m_OutTangent = 0f; m_InWeight = 0f; m_OutWeight = 0f; m_WeightedMode = 0;
		}

		public Keyframe(float time, float value, float inTangent, float outTangent)
		{
			m_Time = time; m_Value = value;
			m_InTangent = inTangent; m_OutTangent = outTangent;
			m_InWeight = 0f; m_OutWeight = 0f; m_WeightedMode = 0;
		}

		// Export loader aliases: the exporter flattens the NATIVE serialization
		// names (inSlope/outSlope/tangentMode) as exact f32 leaves next to its
		// Fix-raw summary; these let the property walker overwrite the summary's
		// preimage floats with the player's exact bits, including the ±Infinity
		// constant-mode tangents the Fix summary collapses (REVIEW-loader-2 L1).
		internal float inSlope { get => m_InTangent; set => m_InTangent = value; }
		internal float outSlope { get => m_OutTangent; set => m_OutTangent = value; }
		internal int tangentMode { get => 0; set { } } // editor-only tangent UI flags
	}

	public enum WeightedMode { None = 0, In = 1, Out = 2, Both = 3 }

	// Gameplay's deterministic path is AnimationCurveFixed, which only reads
	// keys/length/preWrapMode/postWrapMode. Float Evaluate below mirrors the
	// same Hermite basis (used by presentation code only).
	[Serializable]
	public class AnimationCurve
	{
		public Keyframe[] keys = Array.Empty<Keyframe>();
		public WrapMode preWrapMode = WrapMode.ClampForever;
		public WrapMode postWrapMode = WrapMode.ClampForever;

		public AnimationCurve() { }
		public AnimationCurve(params Keyframe[] keys) { this.keys = keys; }

		// Export loader aliases for the flattened native serialization (see
		// Keyframe). m_Pre/PostInfinity duplicate the summary's preWrap/postWrap
		// (different enum encoding); m_RotationOrder is euler-curve-only. No-ops
		// so the walker stays silent (REVIEW-loader-2 L2).
		internal Keyframe[] m_Curve { get => keys; set => keys = value; }
		internal int m_PreInfinity { get => 0; set { } }
		internal int m_PostInfinity { get => 0; set { } }
		internal int m_RotationOrder { get => 0; set { } }

		public int length => keys.Length;

		public Keyframe this[int index] => keys[index];

		public int AddKey(float time, float value) => AddKey(new Keyframe(time, value));

		public int AddKey(Keyframe key)
		{
			var list = new System.Collections.Generic.List<Keyframe>(keys) { key };
			list.Sort((a, b) => a.time.CompareTo(b.time));
			keys = list.ToArray();
			return list.FindIndex(k => k.time == key.time && k.value == key.value);
		}

		public void RemoveKey(int index)
		{
			var list = new System.Collections.Generic.List<Keyframe>(keys);
			list.RemoveAt(index);
			keys = list.ToArray();
		}

		public float Evaluate(float t)
		{
			if (keys.Length == 0) return 0f;
			if (keys.Length == 1) return keys[0].value;
			float first = keys[0].time, last = keys[keys.Length - 1].time, span = last - first;
			if (t < first)
			{
				switch (preWrapMode)
				{
					case WrapMode.Loop:
						t -= first;
						t = span - Mathf.Repeat(Mathf.Abs(t), span);
						t += first;
						break;
					case WrapMode.PingPong:
						t -= first;
						int cycles = (int)Mathf.Abs(t / span);
						t = span - Mathf.Repeat(Mathf.Abs(t), span);
						t = Mathf.Repeat(t, span);
						if (cycles % 2 == 0) t = span - t;
						t += first;
						break;
					default:
						return keys[0].value;
				}
			}
			else if (t > last)
			{
				switch (postWrapMode)
				{
					case WrapMode.Loop:
						t -= first;
						t = Mathf.Repeat(t, span);
						t += first;
						break;
					case WrapMode.PingPong:
						t -= first;
						int cycles = (int)Mathf.Abs(t / span);
						t = Mathf.Repeat(t, span);
						if (cycles % 2 != 0) t = span - t;
						t += first;
						break;
					default:
						return keys[keys.Length - 1].value;
				}
			}
			for (int i = 1; i < keys.Length; i++)
			{
				if (t <= keys[i].time)
					return EvaluateSegment(t, keys[i - 1], keys[i]);
			}
			return keys[keys.Length - 1].value;
		}

		private static float EvaluateSegment(float t, Keyframe k1, Keyframe k2)
		{
			float dt = k2.time - k1.time;
			if (dt <= 0f) return k1.value;
			float u = (t - k1.time) / dt;
			if (float.IsInfinity(k1.outTangent) || float.IsInfinity(k2.inTangent)) return k1.value;
			float m0 = k1.outTangent * dt;
			float m1 = k2.inTangent * dt;
			float u2 = u * u, u3 = u2 * u;
			float h00 = 2f * u3 - 3f * u2 + 1f;
			float h10 = u3 - 2f * u2 + u;
			float h01 = -2f * u3 + 3f * u2;
			float h11 = u3 - u2;
			return h00 * k1.value + h10 * m0 + h11 * m1 + h01 * k2.value;
		}

		public static AnimationCurve Linear(float timeStart, float valueStart, float timeEnd, float valueEnd)
		{
			float tangent = (valueEnd - valueStart) / (timeEnd - timeStart);
			return new AnimationCurve(new Keyframe(timeStart, valueStart, 0f, tangent), new Keyframe(timeEnd, valueEnd, tangent, 0f));
		}

		public static AnimationCurve Constant(float timeStart, float timeEnd, float value)
			=> new AnimationCurve(new Keyframe(timeStart, value), new Keyframe(timeEnd, value));

		public static AnimationCurve EaseInOut(float timeStart, float valueStart, float timeEnd, float valueEnd)
			=> new AnimationCurve(new Keyframe(timeStart, valueStart, 0f, 0f), new Keyframe(timeEnd, valueEnd, 0f, 0f));
	}
}
