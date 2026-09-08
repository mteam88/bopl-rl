using System;

namespace UnityEngine
{
	[Serializable]
	public struct Vector2 : IEquatable<Vector2>
	{
		public float x, y;

		public Vector2(float x, float y) { this.x = x; this.y = y; }

		public float this[int index]
		{
			get => index == 0 ? x : y;
			set { if (index == 0) x = value; else y = value; }
		}

		public static Vector2 zero => new Vector2(0f, 0f);
		public static Vector2 one => new Vector2(1f, 1f);
		public static Vector2 up => new Vector2(0f, 1f);
		public static Vector2 down => new Vector2(0f, -1f);
		public static Vector2 left => new Vector2(-1f, 0f);
		public static Vector2 right => new Vector2(1f, 0f);
		public static Vector2 positiveInfinity => new Vector2(float.PositiveInfinity, float.PositiveInfinity);
		public static Vector2 negativeInfinity => new Vector2(float.NegativeInfinity, float.NegativeInfinity);

		public float magnitude => Mathf.Sqrt(x * x + y * y);
		public float sqrMagnitude => x * x + y * y;

		public Vector2 normalized
		{
			get
			{
				float m = magnitude;
				return m > 1e-5f ? this / m : zero;
			}
		}

		public void Normalize()
		{
			float m = magnitude;
			if (m > 1e-5f) { x /= m; y /= m; } else { x = 0f; y = 0f; }
		}

		public void Set(float newX, float newY) { x = newX; y = newY; }

		public static float Dot(Vector2 a, Vector2 b) => a.x * b.x + a.y * b.y;
		public static float Distance(Vector2 a, Vector2 b) => (a - b).magnitude;
		public float SqrMagnitude() => x * x + y * y;
		public static float SqrMagnitude(Vector2 a) => a.x * a.x + a.y * a.y;
		public static Vector2 Lerp(Vector2 a, Vector2 b, float t) { t = Mathf.Clamp01(t); return new Vector2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t); }
		public static Vector2 LerpUnclamped(Vector2 a, Vector2 b, float t) => new Vector2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
		public static Vector2 MoveTowards(Vector2 current, Vector2 target, float maxDistanceDelta)
		{
			Vector2 d = target - current;
			float m = d.magnitude;
			if (m <= maxDistanceDelta || m == 0f) return target;
			return current + d / m * maxDistanceDelta;
		}
		public static Vector2 Scale(Vector2 a, Vector2 b) => new Vector2(a.x * b.x, a.y * b.y);
		public void Scale(Vector2 s) { x *= s.x; y *= s.y; }
		public static Vector2 Reflect(Vector2 inDirection, Vector2 inNormal) => inDirection - 2f * Dot(inDirection, inNormal) * inNormal;
		public static Vector2 Perpendicular(Vector2 inDirection) => new Vector2(-inDirection.y, inDirection.x);
		public static float Angle(Vector2 from, Vector2 to)
		{
			float d = Mathf.Sqrt(from.sqrMagnitude * to.sqrMagnitude);
			if (d < 1e-15f) return 0f;
			return Mathf.Acos(Mathf.Clamp(Dot(from, to) / d, -1f, 1f)) * Mathf.Rad2Deg;
		}
		public static float SignedAngle(Vector2 from, Vector2 to)
			=> Angle(from, to) * Mathf.Sign(from.x * to.y - from.y * to.x);
		public static Vector2 ClampMagnitude(Vector2 vector, float maxLength)
		{
			if (vector.sqrMagnitude > maxLength * maxLength) return vector.normalized * maxLength;
			return vector;
		}
		public static Vector2 Min(Vector2 a, Vector2 b) => new Vector2(Mathf.Min(a.x, b.x), Mathf.Min(a.y, b.y));
		public static Vector2 Max(Vector2 a, Vector2 b) => new Vector2(Mathf.Max(a.x, b.x), Mathf.Max(a.y, b.y));
		public static Vector2 SmoothDamp(Vector2 current, Vector2 target, ref Vector2 currentVelocity, float smoothTime)
			=> SmoothDamp(current, target, ref currentVelocity, smoothTime, Mathf.Infinity, Time.deltaTime);
		public static Vector2 SmoothDamp(Vector2 current, Vector2 target, ref Vector2 currentVelocity, float smoothTime, float maxSpeed, float deltaTime)
		{
			float vx = currentVelocity.x, vy = currentVelocity.y;
			float rx = Mathf.SmoothDamp(current.x, target.x, ref vx, smoothTime, maxSpeed, deltaTime);
			float ry = Mathf.SmoothDamp(current.y, target.y, ref vy, smoothTime, maxSpeed, deltaTime);
			currentVelocity = new Vector2(vx, vy);
			return new Vector2(rx, ry);
		}

		public static Vector2 operator +(Vector2 a, Vector2 b) => new Vector2(a.x + b.x, a.y + b.y);
		public static Vector2 operator -(Vector2 a, Vector2 b) => new Vector2(a.x - b.x, a.y - b.y);
		public static Vector2 operator -(Vector2 a) => new Vector2(-a.x, -a.y);
		public static Vector2 operator *(Vector2 a, float d) => new Vector2(a.x * d, a.y * d);
		public static Vector2 operator *(float d, Vector2 a) => new Vector2(a.x * d, a.y * d);
		public static Vector2 operator *(Vector2 a, Vector2 b) => new Vector2(a.x * b.x, a.y * b.y);
		public static Vector2 operator /(Vector2 a, float d) => new Vector2(a.x / d, a.y / d);
		public static Vector2 operator /(Vector2 a, Vector2 b) => new Vector2(a.x / b.x, a.y / b.y);
		public static bool operator ==(Vector2 a, Vector2 b) => (a - b).sqrMagnitude < 9.99999944E-11f;
		public static bool operator !=(Vector2 a, Vector2 b) => !(a == b);
		public static implicit operator Vector2(Vector3 v) => new Vector2(v.x, v.y);
		public static implicit operator Vector3(Vector2 v) => new Vector3(v.x, v.y, 0f);

		public bool Equals(Vector2 other) => x == other.x && y == other.y;
		public override bool Equals(object other) => other is Vector2 v && Equals(v);
		public override int GetHashCode() => x.GetHashCode() ^ (y.GetHashCode() << 2);
		public override string ToString() => $"({x:F2}, {y:F2})";
	}

	[Serializable]
	public struct Vector3 : IEquatable<Vector3>
	{
		public float x, y, z;

		public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
		public Vector3(float x, float y) { this.x = x; this.y = y; z = 0f; }

		public float this[int index]
		{
			get => index == 0 ? x : index == 1 ? y : z;
			set { if (index == 0) x = value; else if (index == 1) y = value; else z = value; }
		}

		public static Vector3 zero => new Vector3(0f, 0f, 0f);
		public static Vector3 one => new Vector3(1f, 1f, 1f);
		public static Vector3 up => new Vector3(0f, 1f, 0f);
		public static Vector3 down => new Vector3(0f, -1f, 0f);
		public static Vector3 left => new Vector3(-1f, 0f, 0f);
		public static Vector3 right => new Vector3(1f, 0f, 0f);
		public static Vector3 forward => new Vector3(0f, 0f, 1f);
		public static Vector3 back => new Vector3(0f, 0f, -1f);
		public static Vector3 positiveInfinity => new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
		public static Vector3 negativeInfinity => new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);

		public float magnitude => Mathf.Sqrt(x * x + y * y + z * z);
		public float sqrMagnitude => x * x + y * y + z * z;

		public Vector3 normalized
		{
			get
			{
				float m = magnitude;
				return m > 1e-5f ? this / m : zero;
			}
		}

		public void Normalize()
		{
			float m = magnitude;
			if (m > 1e-5f) { x /= m; y /= m; z /= m; } else { x = y = z = 0f; }
		}

		public void Set(float newX, float newY, float newZ) { x = newX; y = newY; z = newZ; }

		public static float Dot(Vector3 a, Vector3 b) => a.x * b.x + a.y * b.y + a.z * b.z;
		public static Vector3 Cross(Vector3 a, Vector3 b)
			=> new Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
		public static float Distance(Vector3 a, Vector3 b) => (a - b).magnitude;
		public static Vector3 Lerp(Vector3 a, Vector3 b, float t) { t = Mathf.Clamp01(t); return new Vector3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t); }
		public static Vector3 LerpUnclamped(Vector3 a, Vector3 b, float t) => new Vector3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
		public static Vector3 Slerp(Vector3 a, Vector3 b, float t)
		{
			t = Mathf.Clamp01(t);
			float dot = Mathf.Clamp(Dot(a.normalized, b.normalized), -1f, 1f);
			float theta = Mathf.Acos(dot) * t;
			Vector3 relative = (b - a * dot).normalized;
			return a * Mathf.Cos(theta) + relative * Mathf.Sin(theta) * a.magnitude;
		}
		public static Vector3 MoveTowards(Vector3 current, Vector3 target, float maxDistanceDelta)
		{
			Vector3 d = target - current;
			float m = d.magnitude;
			if (m <= maxDistanceDelta || m == 0f) return target;
			return current + d / m * maxDistanceDelta;
		}
		public static Vector3 Scale(Vector3 a, Vector3 b) => new Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
		public void Scale(Vector3 s) { x *= s.x; y *= s.y; z *= s.z; }
		public static Vector3 Project(Vector3 vector, Vector3 onNormal)
		{
			float sqrMag = Dot(onNormal, onNormal);
			if (sqrMag < Mathf.Epsilon) return zero;
			return onNormal * Dot(vector, onNormal) / sqrMag;
		}
		public static Vector3 ProjectOnPlane(Vector3 vector, Vector3 planeNormal) => vector - Project(vector, planeNormal);
		public static Vector3 Reflect(Vector3 inDirection, Vector3 inNormal) => inDirection - 2f * Dot(inDirection, inNormal) * inNormal;
		public static float Angle(Vector3 from, Vector3 to)
		{
			float d = Mathf.Sqrt(from.sqrMagnitude * to.sqrMagnitude);
			if (d < 1e-15f) return 0f;
			return Mathf.Acos(Mathf.Clamp(Dot(from, to) / d, -1f, 1f)) * Mathf.Rad2Deg;
		}
		public static float SignedAngle(Vector3 from, Vector3 to, Vector3 axis)
			=> Angle(from, to) * Mathf.Sign(Dot(axis, Cross(from, to)));
		public static Vector3 ClampMagnitude(Vector3 vector, float maxLength)
		{
			if (vector.sqrMagnitude > maxLength * maxLength) return vector.normalized * maxLength;
			return vector;
		}
		public static Vector3 Min(Vector3 a, Vector3 b) => new Vector3(Mathf.Min(a.x, b.x), Mathf.Min(a.y, b.y), Mathf.Min(a.z, b.z));
		public static Vector3 Max(Vector3 a, Vector3 b) => new Vector3(Mathf.Max(a.x, b.x), Mathf.Max(a.y, b.y), Mathf.Max(a.z, b.z));
		public static Vector3 Normalize(Vector3 value) => value.normalized;
		public static Vector3 SmoothDamp(Vector3 current, Vector3 target, ref Vector3 currentVelocity, float smoothTime)
			=> SmoothDamp(current, target, ref currentVelocity, smoothTime, Mathf.Infinity, Time.deltaTime);
		public static Vector3 SmoothDamp(Vector3 current, Vector3 target, ref Vector3 currentVelocity, float smoothTime, float maxSpeed, float deltaTime)
		{
			float vx = currentVelocity.x, vy = currentVelocity.y, vz = currentVelocity.z;
			float rx = Mathf.SmoothDamp(current.x, target.x, ref vx, smoothTime, maxSpeed, deltaTime);
			float ry = Mathf.SmoothDamp(current.y, target.y, ref vy, smoothTime, maxSpeed, deltaTime);
			float rz = Mathf.SmoothDamp(current.z, target.z, ref vz, smoothTime, maxSpeed, deltaTime);
			currentVelocity = new Vector3(vx, vy, vz);
			return new Vector3(rx, ry, rz);
		}

		public static Vector3 operator +(Vector3 a, Vector3 b) => new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
		public static Vector3 operator -(Vector3 a, Vector3 b) => new Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
		public static Vector3 operator -(Vector3 a) => new Vector3(-a.x, -a.y, -a.z);
		public static Vector3 operator *(Vector3 a, float d) => new Vector3(a.x * d, a.y * d, a.z * d);
		public static Vector3 operator *(float d, Vector3 a) => new Vector3(a.x * d, a.y * d, a.z * d);
		public static Vector3 operator /(Vector3 a, float d) => new Vector3(a.x / d, a.y / d, a.z / d);
		public static bool operator ==(Vector3 a, Vector3 b) => (a - b).sqrMagnitude < 9.99999944E-11f;
		public static bool operator !=(Vector3 a, Vector3 b) => !(a == b);

		public bool Equals(Vector3 other) => x == other.x && y == other.y && z == other.z;
		public override bool Equals(object other) => other is Vector3 v && Equals(v);
		public override int GetHashCode() => x.GetHashCode() ^ (y.GetHashCode() << 2) ^ (z.GetHashCode() >> 2);
		public override string ToString() => $"({x:F2}, {y:F2}, {z:F2})";
	}

	[Serializable]
	public struct Vector4 : IEquatable<Vector4>
	{
		public float x, y, z, w;

		public Vector4(float x, float y, float z, float w) { this.x = x; this.y = y; this.z = z; this.w = w; }
		public Vector4(float x, float y, float z) : this(x, y, z, 0f) { }
		public Vector4(float x, float y) : this(x, y, 0f, 0f) { }

		public static Vector4 zero => new Vector4(0f, 0f, 0f, 0f);
		public static Vector4 one => new Vector4(1f, 1f, 1f, 1f);

		public float magnitude => Mathf.Sqrt(x * x + y * y + z * z + w * w);
		public float sqrMagnitude => x * x + y * y + z * z + w * w;
		public Vector4 normalized { get { float m = magnitude; return m > 1e-5f ? this / m : zero; } }

		public static float Dot(Vector4 a, Vector4 b) => a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
		public static Vector4 Lerp(Vector4 a, Vector4 b, float t) { t = Mathf.Clamp01(t); return a + (b - a) * t; }

		public static Vector4 operator +(Vector4 a, Vector4 b) => new Vector4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
		public static Vector4 operator -(Vector4 a, Vector4 b) => new Vector4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
		public static Vector4 operator *(Vector4 a, float d) => new Vector4(a.x * d, a.y * d, a.z * d, a.w * d);
		public static Vector4 operator /(Vector4 a, float d) => new Vector4(a.x / d, a.y / d, a.z / d, a.w / d);
		public static bool operator ==(Vector4 a, Vector4 b) => (a - b).sqrMagnitude < 9.99999944E-11f;
		public static bool operator !=(Vector4 a, Vector4 b) => !(a == b);
		public static implicit operator Vector4(Vector3 v) => new Vector4(v.x, v.y, v.z, 0f);
		public static implicit operator Vector3(Vector4 v) => new Vector3(v.x, v.y, v.z);
		public static implicit operator Vector4(Vector2 v) => new Vector4(v.x, v.y, 0f, 0f);
		public static implicit operator Vector2(Vector4 v) => new Vector2(v.x, v.y);

		public bool Equals(Vector4 other) => x == other.x && y == other.y && z == other.z && w == other.w;
		public override bool Equals(object other) => other is Vector4 v && Equals(v);
		public override int GetHashCode() => x.GetHashCode() ^ (y.GetHashCode() << 2) ^ (z.GetHashCode() >> 2) ^ (w.GetHashCode() >> 1);
		public override string ToString() => $"({x:F2}, {y:F2}, {z:F2}, {w:F2})";
	}

	[Serializable]
	public struct Vector2Int : IEquatable<Vector2Int>
	{
		public int x, y;
		public Vector2Int(int x, int y) { this.x = x; this.y = y; }
		public static Vector2Int zero => new Vector2Int(0, 0);
		public static Vector2Int one => new Vector2Int(1, 1);
		public static Vector2Int operator +(Vector2Int a, Vector2Int b) => new Vector2Int(a.x + b.x, a.y + b.y);
		public static Vector2Int operator -(Vector2Int a, Vector2Int b) => new Vector2Int(a.x - b.x, a.y - b.y);
		public static bool operator ==(Vector2Int a, Vector2Int b) => a.x == b.x && a.y == b.y;
		public static bool operator !=(Vector2Int a, Vector2Int b) => !(a == b);
		public static implicit operator Vector2(Vector2Int v) => new Vector2(v.x, v.y);
		public bool Equals(Vector2Int other) => this == other;
		public override bool Equals(object other) => other is Vector2Int v && Equals(v);
		public override int GetHashCode() => x.GetHashCode() ^ (y.GetHashCode() << 2);
		public override string ToString() => $"({x}, {y})";
	}

	[Serializable]
	public struct Vector3Int : IEquatable<Vector3Int>
	{
		public int x, y, z;
		public Vector3Int(int x, int y, int z) { this.x = x; this.y = y; this.z = z; }
		public static Vector3Int zero => new Vector3Int(0, 0, 0);
		public static bool operator ==(Vector3Int a, Vector3Int b) => a.x == b.x && a.y == b.y && a.z == b.z;
		public static bool operator !=(Vector3Int a, Vector3Int b) => !(a == b);
		public bool Equals(Vector3Int other) => this == other;
		public override bool Equals(object other) => other is Vector3Int v && Equals(v);
		public override int GetHashCode() => x.GetHashCode() ^ (y.GetHashCode() << 2) ^ (z.GetHashCode() >> 2);
	}

	[Serializable]
	public struct Quaternion : IEquatable<Quaternion>
	{
		public float x, y, z, w;

		public Quaternion(float x, float y, float z, float w) { this.x = x; this.y = y; this.z = z; this.w = w; }

		public static Quaternion identity => new Quaternion(0f, 0f, 0f, 1f);

		public static Quaternion operator *(Quaternion a, Quaternion b) => new Quaternion(
			a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
			a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
			a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
			a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);

		public static Vector3 operator *(Quaternion rotation, Vector3 point)
		{
			float x2 = rotation.x * 2f, y2 = rotation.y * 2f, z2 = rotation.z * 2f;
			float xx = rotation.x * x2, yy = rotation.y * y2, zz = rotation.z * z2;
			float xy = rotation.x * y2, xz = rotation.x * z2, yz = rotation.y * z2;
			float wx = rotation.w * x2, wy = rotation.w * y2, wz = rotation.w * z2;
			return new Vector3(
				(1f - (yy + zz)) * point.x + (xy - wz) * point.y + (xz + wy) * point.z,
				(xy + wz) * point.x + (1f - (xx + zz)) * point.y + (yz - wx) * point.z,
				(xz - wy) * point.x + (yz + wx) * point.y + (1f - (xx + yy)) * point.z);
		}

		public static bool operator ==(Quaternion a, Quaternion b) => Dot(a, b) > 0.999999f;
		public static bool operator !=(Quaternion a, Quaternion b) => !(a == b);

		public static float Dot(Quaternion a, Quaternion b) => a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

		public static Quaternion Inverse(Quaternion q)
		{
			float n = Dot(q, q);
			if (n < Mathf.Epsilon) return identity;
			return new Quaternion(-q.x / n, -q.y / n, -q.z / n, q.w / n);
		}

		public static Quaternion Normalize(Quaternion q)
		{
			float m = Mathf.Sqrt(Dot(q, q));
			if (m < Mathf.Epsilon) return identity;
			return new Quaternion(q.x / m, q.y / m, q.z / m, q.w / m);
		}

		public void Normalize() => this = Normalize(this);
		public Quaternion normalized => Normalize(this);

		public static Quaternion AngleAxis(float angle, Vector3 axis)
		{
			axis = axis.normalized;
			float rad = angle * Mathf.Deg2Rad * 0.5f;
			float s = Mathf.Sin(rad);
			return new Quaternion(axis.x * s, axis.y * s, axis.z * s, Mathf.Cos(rad));
		}

		public void ToAngleAxis(out float angle, out Vector3 axis)
		{
			Quaternion q = normalized;
			angle = 2f * Mathf.Acos(Mathf.Clamp(q.w, -1f, 1f)) * Mathf.Rad2Deg;
			float s = Mathf.Sqrt(1f - q.w * q.w);
			axis = s < 1e-4f ? Vector3.right : new Vector3(q.x / s, q.y / s, q.z / s);
		}

		// ZXY order (Unity convention)
		public static Quaternion Euler(float x, float y, float z)
		{
			float cx = Mathf.Cos(x * Mathf.Deg2Rad * 0.5f), sx = Mathf.Sin(x * Mathf.Deg2Rad * 0.5f);
			float cy = Mathf.Cos(y * Mathf.Deg2Rad * 0.5f), sy = Mathf.Sin(y * Mathf.Deg2Rad * 0.5f);
			float cz = Mathf.Cos(z * Mathf.Deg2Rad * 0.5f), sz = Mathf.Sin(z * Mathf.Deg2Rad * 0.5f);
			return new Quaternion(
				sx * cy * cz + cx * sy * sz,
				cx * sy * cz - sx * cy * sz,
				cx * cy * sz - sx * sy * cz,
				cx * cy * cz + sx * sy * sz);
		}

		public static Quaternion Euler(Vector3 euler) => Euler(euler.x, euler.y, euler.z);

		public Vector3 eulerAngles
		{
			get
			{
				Quaternion q = normalized;
				// Unity ZXY: extract via rotation matrix
				float m21 = 2f * (q.y * q.z - q.w * q.x); // -sin(x)
				float xRad = Mathf.Asin(Mathf.Clamp(-m21, -1f, 1f));
				float yRad, zRad;
				if (Mathf.Abs(m21) < 0.9999999f)
				{
					yRad = Mathf.Atan2(2f * (q.x * q.z + q.w * q.y), 1f - 2f * (q.x * q.x + q.y * q.y));
					zRad = Mathf.Atan2(2f * (q.x * q.y + q.w * q.z), 1f - 2f * (q.x * q.x + q.z * q.z));
				}
				else
				{
					yRad = Mathf.Atan2(-2f * (q.x * q.z - q.w * q.y), 1f - 2f * (q.y * q.y + q.x * q.x));
					zRad = 0f;
				}
				Vector3 e = new Vector3(xRad, yRad, zRad) * Mathf.Rad2Deg;
				if (e.x < 0f) e.x += 360f;
				if (e.y < 0f) e.y += 360f;
				if (e.z < 0f) e.z += 360f;
				return e;
			}
			set => this = Euler(value);
		}

		public static Quaternion FromToRotation(Vector3 from, Vector3 to)
		{
			Vector3 f = from.normalized, t = to.normalized;
			float d = Vector3.Dot(f, t);
			if (d >= 1f - 1e-6f) return identity;
			if (d <= -1f + 1e-6f)
			{
				Vector3 orth = Vector3.Cross(Vector3.right, f);
				if (orth.sqrMagnitude < 1e-6f) orth = Vector3.Cross(Vector3.up, f);
				return AngleAxis(180f, orth.normalized);
			}
			Vector3 c = Vector3.Cross(f, t);
			float s = Mathf.Sqrt((1f + d) * 2f);
			return new Quaternion(c.x / s, c.y / s, c.z / s, s * 0.5f);
		}

		public static Quaternion LookRotation(Vector3 forward) => LookRotation(forward, Vector3.up);

		public static Quaternion LookRotation(Vector3 forward, Vector3 upwards)
		{
			Vector3 f = forward.normalized;
			Vector3 r = Vector3.Cross(upwards, f).normalized;
			if (r.sqrMagnitude < 1e-8f) r = Vector3.right;
			Vector3 u = Vector3.Cross(f, r);
			float m00 = r.x, m01 = u.x, m02 = f.x;
			float m10 = r.y, m11 = u.y, m12 = f.y;
			float m20 = r.z, m21 = u.z, m22 = f.z;
			float trace = m00 + m11 + m22;
			Quaternion q = new Quaternion();
			if (trace > 0f)
			{
				float s = Mathf.Sqrt(trace + 1f) * 2f;
				q.w = 0.25f * s; q.x = (m21 - m12) / s; q.y = (m02 - m20) / s; q.z = (m10 - m01) / s;
			}
			else if (m00 > m11 && m00 > m22)
			{
				float s = Mathf.Sqrt(1f + m00 - m11 - m22) * 2f;
				q.w = (m21 - m12) / s; q.x = 0.25f * s; q.y = (m01 + m10) / s; q.z = (m02 + m20) / s;
			}
			else if (m11 > m22)
			{
				float s = Mathf.Sqrt(1f + m11 - m00 - m22) * 2f;
				q.w = (m02 - m20) / s; q.x = (m01 + m10) / s; q.y = 0.25f * s; q.z = (m12 + m21) / s;
			}
			else
			{
				float s = Mathf.Sqrt(1f + m22 - m00 - m11) * 2f;
				q.w = (m10 - m01) / s; q.x = (m02 + m20) / s; q.y = (m12 + m21) / s; q.z = 0.25f * s;
			}
			return q;
		}

		public static float Angle(Quaternion a, Quaternion b)
		{
			float d = Mathf.Min(Mathf.Abs(Dot(a, b)), 1f);
			return d > 0.999999f ? 0f : Mathf.Acos(d) * 2f * Mathf.Rad2Deg;
		}

		public static Quaternion Lerp(Quaternion a, Quaternion b, float t) => Slerp(a, b, Mathf.Clamp01(t));
		public static Quaternion LerpUnclamped(Quaternion a, Quaternion b, float t) => SlerpUnclamped(a, b, t);
		public static Quaternion Slerp(Quaternion a, Quaternion b, float t) => SlerpUnclamped(a, b, Mathf.Clamp01(t));

		public static Quaternion SlerpUnclamped(Quaternion a, Quaternion b, float t)
		{
			float d = Dot(a, b);
			if (d < 0f) { b = new Quaternion(-b.x, -b.y, -b.z, -b.w); d = -d; }
			if (d > 0.9995f)
			{
				return Normalize(new Quaternion(
					a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
					a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t));
			}
			float theta0 = Mathf.Acos(d);
			float theta = theta0 * t;
			float sin0 = Mathf.Sin(theta0);
			float s0 = Mathf.Sin(theta0 - theta) / sin0;
			float s1 = Mathf.Sin(theta) / sin0;
			return new Quaternion(
				a.x * s0 + b.x * s1, a.y * s0 + b.y * s1,
				a.z * s0 + b.z * s1, a.w * s0 + b.w * s1);
		}

		public static Quaternion RotateTowards(Quaternion from, Quaternion to, float maxDegreesDelta)
		{
			float angle = Angle(from, to);
			if (angle == 0f) return to;
			return SlerpUnclamped(from, to, Mathf.Min(1f, maxDegreesDelta / angle));
		}

		public void Set(float newX, float newY, float newZ, float newW) { x = newX; y = newY; z = newZ; w = newW; }

		public bool Equals(Quaternion other) => x == other.x && y == other.y && z == other.z && w == other.w;
		public override bool Equals(object other) => other is Quaternion q && Equals(q);
		public override int GetHashCode() => x.GetHashCode() ^ (y.GetHashCode() << 2) ^ (z.GetHashCode() >> 2) ^ (w.GetHashCode() >> 1);
		public override string ToString() => $"({x:F5}, {y:F5}, {z:F5}, {w:F5})";
	}

	public static class Mathf
	{
		public const float PI = (float)Math.PI;
		public const float Infinity = float.PositiveInfinity;
		public const float NegativeInfinity = float.NegativeInfinity;
		public const float Deg2Rad = PI / 180f;
		public const float Rad2Deg = 180f / PI;
		public static readonly float Epsilon = float.Epsilon;

		public static float Sin(float f) => (float)Math.Sin(f);
		public static float Cos(float f) => (float)Math.Cos(f);
		public static float Tan(float f) => (float)Math.Tan(f);
		public static float Asin(float f) => (float)Math.Asin(f);
		public static float Acos(float f) => (float)Math.Acos(f);
		public static float Atan(float f) => (float)Math.Atan(f);
		public static float Atan2(float y, float x) => (float)Math.Atan2(y, x);
		public static float Sqrt(float f) => (float)Math.Sqrt(f);
		public static float Abs(float f) => Math.Abs(f);
		public static int Abs(int value) => Math.Abs(value);
		public static float Min(float a, float b) => a < b ? a : b;
		public static float Min(params float[] values) { float m = values[0]; foreach (var v in values) if (v < m) m = v; return m; }
		public static int Min(int a, int b) => a < b ? a : b;
		public static int Min(params int[] values) { int m = values[0]; foreach (var v in values) if (v < m) m = v; return m; }
		public static float Max(float a, float b) => a > b ? a : b;
		public static float Max(params float[] values) { float m = values[0]; foreach (var v in values) if (v > m) m = v; return m; }
		public static int Max(int a, int b) => a > b ? a : b;
		public static int Max(params int[] values) { int m = values[0]; foreach (var v in values) if (v > m) m = v; return m; }
		public static float Pow(float f, float p) => (float)Math.Pow(f, p);
		public static float Exp(float power) => (float)Math.Exp(power);
		public static float Log(float f, float p) => (float)Math.Log(f, p);
		public static float Log(float f) => (float)Math.Log(f);
		public static float Log10(float f) => (float)Math.Log10(f);
		public static float Ceil(float f) => (float)Math.Ceiling(f);
		public static float Floor(float f) => (float)Math.Floor(f);
		public static float Round(float f) => (float)Math.Round(f);
		public static int CeilToInt(float f) => (int)Math.Ceiling(f);
		public static int FloorToInt(float f) => (int)Math.Floor(f);
		public static int RoundToInt(float f) => (int)Math.Round(f);
		public static float Sign(float f) => f >= 0f ? 1f : -1f;
		public static float Clamp(float value, float min, float max) => value < min ? min : value > max ? max : value;
		public static int Clamp(int value, int min, int max) => value < min ? min : value > max ? max : value;
		public static float Clamp01(float value) => value < 0f ? 0f : value > 1f ? 1f : value;
		public static float Lerp(float a, float b, float t) => a + (b - a) * Clamp01(t);
		public static float LerpUnclamped(float a, float b, float t) => a + (b - a) * t;
		public static float LerpAngle(float a, float b, float t)
		{
			float delta = Repeat(b - a, 360f);
			if (delta > 180f) delta -= 360f;
			return a + delta * Clamp01(t);
		}
		public static float MoveTowards(float current, float target, float maxDelta)
		{
			if (Abs(target - current) <= maxDelta) return target;
			return current + Sign(target - current) * maxDelta;
		}
		public static float MoveTowardsAngle(float current, float target, float maxDelta)
		{
			float delta = DeltaAngle(current, target);
			if (-maxDelta < delta && delta < maxDelta) return target;
			return MoveTowards(current, current + delta, maxDelta);
		}
		public static float SmoothStep(float from, float to, float t)
		{
			t = Clamp01(t);
			t = -2f * t * t * t + 3f * t * t;
			return to * t + from * (1f - t);
		}
		public static float Repeat(float t, float length) => Clamp(t - Floor(t / length) * length, 0f, length);
		public static float PingPong(float t, float length)
		{
			t = Repeat(t, length * 2f);
			return length - Abs(t - length);
		}
		public static float InverseLerp(float a, float b, float value) => a != b ? Clamp01((value - a) / (b - a)) : 0f;
		public static float DeltaAngle(float current, float target)
		{
			float delta = Repeat(target - current, 360f);
			if (delta > 180f) delta -= 360f;
			return delta;
		}
		public static bool Approximately(float a, float b)
			=> Abs(b - a) < Max(1E-06f * Max(Abs(a), Abs(b)), Epsilon * 8f);
		// Deterministic 2D value-noise stand-in for Unity's Perlin (presentation only).
		public static float PerlinNoise(float x, float y)
		{
			int xi = FloorToInt(x), yi = FloorToInt(y);
			float xf = x - xi, yf = y - yi;
			float u = xf * xf * (3f - 2f * xf), v = yf * yf * (3f - 2f * yf);
			float a = Hash01(xi, yi), b = Hash01(xi + 1, yi), c = Hash01(xi, yi + 1), d = Hash01(xi + 1, yi + 1);
			return Lerp(Lerp(a, b, u), Lerp(c, d, u), v);
		}
		private static float Hash01(int x, int y)
		{
			unchecked
			{
				uint h = (uint)(x * 374761393 + y * 668265263);
				h = (h ^ (h >> 13)) * 1274126177u;
				return (h ^ (h >> 16)) / 4294967295f;
			}
		}
		public static float SmoothDamp(float current, float target, ref float currentVelocity, float smoothTime)
			=> SmoothDamp(current, target, ref currentVelocity, smoothTime, Infinity, Time.deltaTime);
		public static float SmoothDamp(float current, float target, ref float currentVelocity, float smoothTime, float maxSpeed)
			=> SmoothDamp(current, target, ref currentVelocity, smoothTime, maxSpeed, Time.deltaTime);
		public static float SmoothDamp(float current, float target, ref float currentVelocity, float smoothTime, float maxSpeed, float deltaTime)
		{
			smoothTime = Max(0.0001f, smoothTime);
			float omega = 2f / smoothTime;
			float x = omega * deltaTime;
			float exp = 1f / (1f + x + 0.48f * x * x + 0.235f * x * x * x);
			float change = current - target;
			float originalTo = target;
			float maxChange = maxSpeed * smoothTime;
			change = Clamp(change, -maxChange, maxChange);
			target = current - change;
			float temp = (currentVelocity + omega * change) * deltaTime;
			currentVelocity = (currentVelocity - omega * temp) * exp;
			float output = target + (change + temp) * exp;
			if (originalTo - current > 0f == output > originalTo)
			{
				output = originalTo;
				currentVelocity = (output - originalTo) / deltaTime;
			}
			return output;
		}
		public static float SmoothDampAngle(float current, float target, ref float currentVelocity, float smoothTime)
			=> SmoothDampAngle(current, target, ref currentVelocity, smoothTime, Infinity, Time.deltaTime);
		public static float SmoothDampAngle(float current, float target, ref float currentVelocity, float smoothTime, float maxSpeed, float deltaTime)
		{
			target = current + DeltaAngle(current, target);
			return SmoothDamp(current, target, ref currentVelocity, smoothTime, maxSpeed, deltaTime);
		}
	}

	[Serializable]
	public struct Color : IEquatable<Color>
	{
		public float r, g, b, a;

		public Color(float r, float g, float b, float a) { this.r = r; this.g = g; this.b = b; this.a = a; }
		public Color(float r, float g, float b) : this(r, g, b, 1f) { }

		public static Color red => new Color(1f, 0f, 0f, 1f);
		public static Color green => new Color(0f, 1f, 0f, 1f);
		public static Color blue => new Color(0f, 0f, 1f, 1f);
		public static Color white => new Color(1f, 1f, 1f, 1f);
		public static Color black => new Color(0f, 0f, 0f, 1f);
		public static Color yellow => new Color(1f, 235f / 255f, 4f / 255f, 1f);
		public static Color cyan => new Color(0f, 1f, 1f, 1f);
		public static Color magenta => new Color(1f, 0f, 1f, 1f);
		public static Color gray => new Color(0.5f, 0.5f, 0.5f, 1f);
		public static Color grey => gray;
		public static Color clear => new Color(0f, 0f, 0f, 0f);

		public float this[int index]
		{
			get => index switch { 0 => r, 1 => g, 2 => b, _ => a };
			set { switch (index) { case 0: r = value; break; case 1: g = value; break; case 2: b = value; break; default: a = value; break; } }
		}

		public static Color Lerp(Color a, Color b, float t)
		{
			t = Mathf.Clamp01(t);
			return new Color(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t);
		}
		public static Color LerpUnclamped(Color a, Color b, float t)
			=> new Color(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t);

		public static Color operator +(Color a, Color b) => new Color(a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a);
		public static Color operator -(Color a, Color b) => new Color(a.r - b.r, a.g - b.g, a.b - b.b, a.a - b.a);
		public static Color operator *(Color a, Color b) => new Color(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a);
		public static Color operator *(Color a, float f) => new Color(a.r * f, a.g * f, a.b * f, a.a * f);
		public static Color operator *(float f, Color a) => a * f;
		public static Color operator /(Color a, float f) => new Color(a.r / f, a.g / f, a.b / f, a.a / f);
		public static bool operator ==(Color a, Color b) => a.Equals(b);
		public static bool operator !=(Color a, Color b) => !a.Equals(b);
		public static implicit operator Vector4(Color c) => new Vector4(c.r, c.g, c.b, c.a);
		public static implicit operator Color(Vector4 v) => new Color(v.x, v.y, v.z, v.w);

		public Color WithAlpha(float alpha) => new Color(r, g, b, alpha);
		public bool Equals(Color other) => r == other.r && g == other.g && b == other.b && a == other.a;
		public override bool Equals(object other) => other is Color c && Equals(c);
		public override int GetHashCode() => ((Vector4)this).GetHashCode();
		public override string ToString() => $"RGBA({r:F3}, {g:F3}, {b:F3}, {a:F3})";

		// F6: verbatim from ilspycmd decompile of the game's
		// UnityEngine.CoreModule (BeamColors.Mix runs these on the sim path and
		// the results are trace-captured; float op order matters).
		public static Color HSVToRGB(float H, float S, float V) => HSVToRGB(H, S, V, hdr: true);

		public static Color HSVToRGB(float H, float S, float V, bool hdr)
		{
			Color result = white;
			if (S == 0f)
			{
				result.r = V;
				result.g = V;
				result.b = V;
			}
			else if (V == 0f)
			{
				result.r = 0f;
				result.g = 0f;
				result.b = 0f;
			}
			else
			{
				result.r = 0f;
				result.g = 0f;
				result.b = 0f;
				float num = H * 6f;
				int num2 = (int)Mathf.Floor(num);
				float num3 = num - (float)num2;
				float num4 = V * (1f - S);
				float num5 = V * (1f - S * num3);
				float num6 = V * (1f - S * (1f - num3));
				switch (num2)
				{
					case 0: result.r = V; result.g = num6; result.b = num4; break;
					case 1: result.r = num5; result.g = V; result.b = num4; break;
					case 2: result.r = num4; result.g = V; result.b = num6; break;
					case 3: result.r = num4; result.g = num5; result.b = V; break;
					case 4: result.r = num6; result.g = num4; result.b = V; break;
					case 5: result.r = V; result.g = num4; result.b = num5; break;
					case 6: result.r = V; result.g = num6; result.b = num4; break;
					case -1: result.r = V; result.g = num4; result.b = num5; break;
				}
				if (!hdr)
				{
					result.r = Mathf.Clamp(result.r, 0f, 1f);
					result.g = Mathf.Clamp(result.g, 0f, 1f);
					result.b = Mathf.Clamp(result.b, 0f, 1f);
				}
			}
			return result;
		}

		public static void RGBToHSV(Color rgbColor, out float H, out float S, out float V)
		{
			if (rgbColor.b > rgbColor.g && rgbColor.b > rgbColor.r)
				RGBToHSVHelper(4f, rgbColor.b, rgbColor.r, rgbColor.g, out H, out S, out V);
			else if (rgbColor.g > rgbColor.r)
				RGBToHSVHelper(2f, rgbColor.g, rgbColor.b, rgbColor.r, out H, out S, out V);
			else
				RGBToHSVHelper(0f, rgbColor.r, rgbColor.g, rgbColor.b, out H, out S, out V);
		}

		private static void RGBToHSVHelper(float offset, float dominantcolor, float colorone, float colortwo, out float H, out float S, out float V)
		{
			V = dominantcolor;
			if (V != 0f)
			{
				float num = 0f;
				num = ((!(colorone > colortwo)) ? colorone : colortwo);
				float num2 = V - num;
				if (num2 != 0f)
				{
					S = num2 / V;
					H = offset + (colorone - colortwo) / num2;
				}
				else
				{
					S = 0f;
					H = offset + (colorone - colortwo);
				}
				H /= 6f;
				if (H < 0f)
				{
					H += 1f;
				}
			}
			else
			{
				S = 0f;
				H = 0f;
			}
		}
	}

	[Serializable]
	public struct Color32
	{
		public byte r, g, b, a;
		public Color32(byte r, byte g, byte b, byte a) { this.r = r; this.g = g; this.b = b; this.a = a; }
		public static implicit operator Color(Color32 c) => new Color(c.r / 255f, c.g / 255f, c.b / 255f, c.a / 255f);
		public static implicit operator Color32(Color c) => new Color32(
			(byte)Mathf.Round(Mathf.Clamp01(c.r) * 255f), (byte)Mathf.Round(Mathf.Clamp01(c.g) * 255f),
			(byte)Mathf.Round(Mathf.Clamp01(c.b) * 255f), (byte)Mathf.Round(Mathf.Clamp01(c.a) * 255f));
		public override string ToString() => $"RGBA({r}, {g}, {b}, {a})";
	}

	[Serializable]
	public struct Rect
	{
		public float x, y, width, height;

		public Rect(float x, float y, float width, float height) { this.x = x; this.y = y; this.width = width; this.height = height; }
		public Rect(Vector2 position, Vector2 size) : this(position.x, position.y, size.x, size.y) { }

		public static Rect zero => new Rect(0f, 0f, 0f, 0f);

		public Vector2 position { get => new Vector2(x, y); set { x = value.x; y = value.y; } }
		public Vector2 size { get => new Vector2(width, height); set { width = value.x; height = value.y; } }
		public Vector2 center { get => new Vector2(x + width / 2f, y + height / 2f); set { x = value.x - width / 2f; y = value.y - height / 2f; } }
		public Vector2 min { get => new Vector2(xMin, yMin); set { xMin = value.x; yMin = value.y; } }
		public Vector2 max { get => new Vector2(xMax, yMax); set { xMax = value.x; yMax = value.y; } }
		public float xMin { get => x; set { float old = xMax; x = value; width = old - x; } }
		public float yMin { get => y; set { float old = yMax; y = value; height = old - y; } }
		public float xMax { get => x + width; set => width = value - x; }
		public float yMax { get => y + height; set => height = value - y; }

		public bool Contains(Vector2 point) => point.x >= xMin && point.x < xMax && point.y >= yMin && point.y < yMax;
		public bool Contains(Vector3 point) => Contains(new Vector2(point.x, point.y));
		public bool Overlaps(Rect other) => other.xMax > xMin && other.xMin < xMax && other.yMax > yMin && other.yMin < yMax;
		public override string ToString() => $"(x:{x:F2}, y:{y:F2}, width:{width:F2}, height:{height:F2})";
	}

	[Serializable]
	public struct Bounds
	{
		private Vector3 m_Center;
		private Vector3 m_Extents;

		public Bounds(Vector3 center, Vector3 size) { m_Center = center; m_Extents = size * 0.5f; }

		public Vector3 center { get => m_Center; set => m_Center = value; }
		public Vector3 size { get => m_Extents * 2f; set => m_Extents = value * 0.5f; }
		public Vector3 extents { get => m_Extents; set => m_Extents = value; }
		public Vector3 min { get => m_Center - m_Extents; set => SetMinMax(value, max); }
		public Vector3 max { get => m_Center + m_Extents; set => SetMinMax(min, value); }

		public void SetMinMax(Vector3 min, Vector3 max)
		{
			m_Extents = (max - min) * 0.5f;
			m_Center = min + m_Extents;
		}

		public void Encapsulate(Vector3 point) => SetMinMax(Vector3.Min(min, point), Vector3.Max(max, point));
		public void Encapsulate(Bounds bounds) { Encapsulate(bounds.min); Encapsulate(bounds.max); }
		public void Expand(float amount) => m_Extents += new Vector3(amount, amount, amount) * 0.5f;
		public bool Contains(Vector3 point)
		{
			Vector3 mn = min, mx = max;
			return point.x >= mn.x && point.x <= mx.x && point.y >= mn.y && point.y <= mx.y && point.z >= mn.z && point.z <= mx.z;
		}
		public bool Intersects(Bounds bounds)
		{
			Vector3 mn = min, mx = max, bmn = bounds.min, bmx = bounds.max;
			return mn.x <= bmx.x && mx.x >= bmn.x && mn.y <= bmx.y && mx.y >= bmn.y && mn.z <= bmx.z && mx.z >= bmn.z;
		}
		public override string ToString() => $"Center: {m_Center}, Extents: {m_Extents}";
	}

	[Serializable]
	public struct LayerMask
	{
		private int m_Mask;

		public int value { get => m_Mask; set => m_Mask = value; }

		public static implicit operator int(LayerMask mask) => mask.m_Mask;
		public static implicit operator LayerMask(int intVal) => new LayerMask { m_Mask = intVal };

		public static int NameToLayer(string layerName)
		{
			var names = UnityShim.World.Current.layerNames;
			for (int i = 0; i < names.Length; i++)
				if (names[i] == layerName) return i;
			return -1;
		}

		public static string LayerToName(int layer)
		{
			var names = UnityShim.World.Current.layerNames;
			return layer >= 0 && layer < names.Length ? names[layer] : "";
		}

		public static int GetMask(params string[] layerNames)
		{
			int mask = 0;
			foreach (var name in layerNames)
			{
				int layer = NameToLayer(name);
				if (layer >= 0) mask |= 1 << layer;
			}
			return mask;
		}
	}

	// Presentation-only in this game (gameplay randomness goes through
	// Updater's Unity.Mathematics.Random). Backed by System.Random; NOT
	// Unity's Xorshift128 - documented deviation in sim/PORTING.md.
	public static class Random
	{
		private static System.Random rng = new System.Random(42);

		public static void InitState(int seed) => rng = new System.Random(seed);

		public static float value => (float)rng.NextDouble();

		public static float Range(float minInclusive, float maxInclusive)
			=> minInclusive + (float)rng.NextDouble() * (maxInclusive - minInclusive);

		public static int Range(int minInclusive, int maxExclusive)
			=> minInclusive >= maxExclusive ? minInclusive : rng.Next(minInclusive, maxExclusive);

		public static Vector2 insideUnitCircle
		{
			get
			{
				float angle = value * Mathf.PI * 2f;
				float radius = Mathf.Sqrt(value);
				return new Vector2(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius);
			}
		}

		public static Vector3 insideUnitSphere
		{
			get
			{
				Vector3 v = onUnitSphere * Mathf.Pow(value, 1f / 3f);
				return v;
			}
		}

		public static Vector3 onUnitSphere
		{
			get
			{
				float z = Range(-1f, 1f);
				float a = value * Mathf.PI * 2f;
				float r = Mathf.Sqrt(1f - z * z);
				return new Vector3(r * Mathf.Cos(a), r * Mathf.Sin(a), z);
			}
		}

		public static Quaternion rotation => Quaternion.Euler(Range(0f, 360f), Range(0f, 360f), Range(0f, 360f));
		public static Quaternion rotationUniform => rotation;
	}
}
