using System;

namespace UnityEngine
{
	// Unity 2D physics types. The game's own DetPhysics does all gameplay
	// collision; these exist only so presentation/legacy code compiles.

	public enum ForceMode2D { Force, Impulse }

	public class Collider2D : Behaviour
	{
		public bool isTrigger { get; set; }
		public Rigidbody2D attachedRigidbody { get; set; }
		public Bounds bounds { get; set; }
		public Vector2 offset { get; set; }
		public PhysicsMaterial2D sharedMaterial { get; set; }
	}

	public class BoxCollider2D : Collider2D
	{
		public Vector2 size { get; set; } = Vector2.one;
	}

	public class CircleCollider2D : Collider2D
	{
		public float radius { get; set; } = 0.5f;
	}

	public class PolygonCollider2D : Collider2D
	{
		public Vector2[] points { get; set; } = Array.Empty<Vector2>();
		public int pathCount { get; set; }
	}

	public class CapsuleCollider2D : Collider2D
	{
		public Vector2 size { get; set; }
	}

	public class EdgeCollider2D : Collider2D
	{
		public Vector2[] points { get; set; } = Array.Empty<Vector2>();
	}

	public class Rigidbody2D : Component
	{
		public Vector2 velocity { get; set; }
		public float angularVelocity { get; set; }
		public float mass { get; set; } = 1f;
		public float gravityScale { get; set; } = 1f;
		public float drag { get; set; }
		public float angularDrag { get; set; }
		public bool isKinematic { get; set; }
		public bool simulated { get; set; } = true;
		public RigidbodyType2D bodyType { get; set; }
		public Vector2 position { get; set; }
		public float rotation { get; set; }
		public bool freezeRotation { get; set; }
		public RigidbodyConstraints2D constraints { get; set; }

		public void AddForce(Vector2 force) { }
		public void AddForce(Vector2 force, ForceMode2D mode) { }
		public void AddTorque(float torque) { }
		public void AddTorque(float torque, ForceMode2D mode) { }
		public void MovePosition(Vector2 position) => this.position = position;
		public void MoveRotation(float rotation) => this.rotation = rotation;
	}

	public enum RigidbodyType2D { Dynamic, Kinematic, Static }

	[Flags]
	public enum RigidbodyConstraints2D { None = 0, FreezePositionX = 1, FreezePositionY = 2, FreezeRotation = 4, FreezePosition = 3, FreezeAll = 7 }

	public class PhysicsMaterial2D : Object
	{
		public float friction { get; set; } = 0.4f;
		public float bounciness { get; set; }
	}

	public struct RaycastHit2D
	{
		// Field names/layout match Unity's struct exactly (ReferenceCapture
		// reflects private fields by name; PlayerPhysics raycast buffers are
		// trace-captured). m_Collider is Unity's native collider id - always 0
		// here because Physics2D raycasts produce no hits headless.
		private Vector2 m_Centroid;
		private Vector2 m_Point;
		private Vector2 m_Normal;
		private float m_Distance;
		private float m_Fraction;
		private int m_Collider;

		public Vector2 centroid { get => m_Centroid; set => m_Centroid = value; }
		public Vector2 point { get => m_Point; set => m_Point = value; }
		public Vector2 normal { get => m_Normal; set => m_Normal = value; }
		public float distance { get => m_Distance; set => m_Distance = value; }
		public float fraction { get => m_Fraction; set => m_Fraction = value; }
		public Collider2D collider => null;
		public Rigidbody2D rigidbody => null;
		public Transform transform => null;
		public static implicit operator bool(RaycastHit2D hit) => hit.collider != null;
	}

	public struct ContactPoint2D
	{
		public Vector2 point { get; set; }
		public Vector2 normal { get; set; }
		public Collider2D collider { get; set; }
		public Collider2D otherCollider { get; set; }
	}

	public class Collision2D
	{
		public Collider2D collider { get; set; }
		public GameObject gameObject => collider != null ? collider.gameObject : null;
		public Transform transform => collider != null ? collider.transform : null;
		public Rigidbody2D rigidbody { get; set; }
		public Vector2 relativeVelocity { get; set; }
		public ContactPoint2D[] contacts { get; set; } = Array.Empty<ContactPoint2D>();
		public int contactCount => contacts.Length;
		public ContactPoint2D GetContact(int index) => contacts[index];
	}

	// 3D leftovers occasionally referenced by menu/presentation code.
	public class Collider : Component
	{
		public bool isTrigger { get; set; }
		public Bounds bounds { get; set; }
	}

	public class BoxCollider : Collider
	{
		public Vector3 size { get; set; } = Vector3.one;
		public Vector3 center { get; set; }
	}

	public class Rigidbody : Component
	{
		public Vector3 velocity { get; set; }
		public bool isKinematic { get; set; }
		public bool useGravity { get; set; } = true;
		public float mass { get; set; } = 1f;
		public void AddForce(Vector3 force) { }
	}

	public struct RaycastHit
	{
		public Collider collider { get; set; }
		public Vector3 point { get; set; }
		public Vector3 normal { get; set; }
		public float distance { get; set; }
		public Transform transform => collider != null ? collider.transform : null;
	}

	public static class Physics
	{
		public static bool Raycast(Vector3 origin, Vector3 direction, out RaycastHit hitInfo, float maxDistance)
		{
			hitInfo = default;
			return false;
		}
		public static bool Raycast(Ray ray, out RaycastHit hitInfo, float maxDistance = Mathf.Infinity)
		{
			hitInfo = default;
			return false;
		}
	}
}
