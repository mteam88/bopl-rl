using System;
using System.Collections;
using System.Collections.Generic;
using UnityShim;

namespace UnityEngine
{
	public class Transform : Component, IEnumerable
	{
		internal Vector3 localPositionField = Vector3.zero;
		internal Quaternion localRotationField = Quaternion.identity;
		internal Vector3 localScaleField = Vector3.one;

		internal Transform parentField;
		internal readonly List<Transform> children = new List<Transform>();

		// ---- hierarchy -------------------------------------------------------

		public Transform parent
		{
			get => parentField;
			set => SetParent(value, true);
		}

		public void SetParent(Transform newParent) => SetParent(newParent, true);

		public void SetParent(Transform newParent, bool worldPositionStays)
		{
			if (newParent == parentField) return;
			if (worldPositionStays)
			{
				Vector3 pos = position;
				Quaternion rot = rotation;
				Vector3 scale = lossyScale;
				SetParentInternal(newParent, true);
				position = pos;
				rotation = rot;
				// approximate scale preservation (no shear support)
				Vector3 parentScale = parentField != null ? parentField.lossyScale : Vector3.one;
				localScaleField = new Vector3(
					parentScale.x != 0f ? scale.x / parentScale.x : scale.x,
					parentScale.y != 0f ? scale.y / parentScale.y : scale.y,
					parentScale.z != 0f ? scale.z / parentScale.z : scale.z);
			}
			else
			{
				SetParentInternal(newParent, true);
			}
		}

		internal void SetParentInternal(Transform newParent, bool manageRoots = true)
		{
			if (parentField != null) parentField.children.Remove(this);
			else if (manageRoots && !isAsset) World.Current.roots.Remove(gameObject);
			parentField = newParent;
			if (newParent != null)
			{
				newParent.children.Add(this);
				// Reparenting moves scene membership (matters for children attached
				// under DontDestroyOnLoad roots - ReferenceCapture prints scene names).
				if (!isAsset && !newParent.isAsset && gameObject.scene != newParent.gameObject.scene)
					StampScene(this, newParent.gameObject.scene);
			}
			else if (manageRoots && !isAsset && !gameObject.destroyed && !World.Current.roots.Contains(gameObject))
				World.Current.roots.Add(gameObject);
		}

		private static void StampScene(Transform t, UnityEngine.SceneManagement.Scene scene)
		{
			t.gameObject.scene = scene;
			for (int i = 0; i < t.children.Count; i++) StampScene(t.children[i], scene);
		}

		public int childCount => children.Count;

		public Transform GetChild(int index) => children[index];

		public int GetSiblingIndex() => parentField == null
			? World.Current.roots.IndexOf(gameObject)
			: parentField.children.IndexOf(this);

		public void SetSiblingIndex(int index)
		{
			var list = parentField != null ? parentField.children : null;
			if (list == null) return;
			list.Remove(this);
			list.Insert(Math.Clamp(index, 0, list.Count), this);
		}

		public void SetAsFirstSibling() => SetSiblingIndex(0);
		public void SetAsLastSibling() { if (parentField != null) { parentField.children.Remove(this); parentField.children.Add(this); } }

		public Transform root => parentField == null ? this : parentField.root;

		public Transform Find(string path)
		{
			if (string.IsNullOrEmpty(path)) return null;
			int slash = path.IndexOf('/');
			string head = slash < 0 ? path : path.Substring(0, slash);
			foreach (var child in children)
			{
				if (child.gameObject.name != head) continue;
				if (slash < 0) return child;
				return child.Find(path.Substring(slash + 1));
			}
			return null;
		}

		public void DetachChildren()
		{
			foreach (var child in children.ToArray()) child.SetParent(null, true);
		}

		public bool IsChildOf(Transform other)
		{
			for (var t = this; t != null; t = t.parentField)
				if (t == other) return true;
			return false;
		}

		public IEnumerator GetEnumerator() => children.GetEnumerator();

		// ---- local TRS -------------------------------------------------------

		public Vector3 localPosition { get => localPositionField; set => localPositionField = value; }
		public Quaternion localRotation { get => localRotationField; set => localRotationField = value; }
		public Vector3 localScale { get => localScaleField; set => localScaleField = value; }

		public Vector3 localEulerAngles
		{
			get => localRotationField.eulerAngles;
			set => localRotationField = Quaternion.Euler(value);
		}

		// ---- world TRS -------------------------------------------------------

		public Vector3 position
		{
			get => parentField == null ? localPositionField : parentField.TransformPoint(localPositionField);
			set => localPositionField = parentField == null ? value : parentField.InverseTransformPoint(value);
		}

		public Quaternion rotation
		{
			get => parentField == null ? localRotationField : parentField.rotation * localRotationField;
			set => localRotationField = parentField == null ? value : Quaternion.Inverse(parentField.rotation) * value;
		}

		public Vector3 eulerAngles
		{
			get => rotation.eulerAngles;
			set => rotation = Quaternion.Euler(value);
		}

		public Vector3 lossyScale
		{
			get
			{
				if (parentField == null) return localScaleField;
				Vector3 p = parentField.lossyScale;
				return new Vector3(p.x * localScaleField.x, p.y * localScaleField.y, p.z * localScaleField.z);
			}
		}

		public Vector3 up { get => rotation * Vector3.up; set => rotation = Quaternion.FromToRotation(Vector3.up, value); }
		public Vector3 right { get => rotation * Vector3.right; set => rotation = Quaternion.FromToRotation(Vector3.right, value); }
		public Vector3 forward { get => rotation * Vector3.forward; set => rotation = Quaternion.LookRotation(value); }

		public Vector3 TransformPoint(Vector3 point)
		{
			Vector3 scaled = new Vector3(point.x * localScaleField.x, point.y * localScaleField.y, point.z * localScaleField.z);
			Vector3 local = localPositionField + localRotationField * scaled;
			return parentField == null ? local : parentField.TransformPoint(local);
		}

		public Vector3 InverseTransformPoint(Vector3 point)
		{
			Vector3 local = parentField == null ? point : parentField.InverseTransformPoint(point);
			Vector3 p = Quaternion.Inverse(localRotationField) * (local - localPositionField);
			return new Vector3(
				localScaleField.x != 0f ? p.x / localScaleField.x : p.x,
				localScaleField.y != 0f ? p.y / localScaleField.y : p.y,
				localScaleField.z != 0f ? p.z / localScaleField.z : p.z);
		}

		public Vector3 TransformDirection(Vector3 direction) => rotation * direction;
		public Vector3 InverseTransformDirection(Vector3 direction) => Quaternion.Inverse(rotation) * direction;

		public Vector3 TransformVector(Vector3 vector)
		{
			Vector3 s = lossyScale;
			return rotation * new Vector3(vector.x * s.x, vector.y * s.y, vector.z * s.z);
		}

		public Vector3 InverseTransformVector(Vector3 vector)
		{
			Vector3 s = lossyScale;
			Vector3 v = Quaternion.Inverse(rotation) * vector;
			return new Vector3(s.x != 0f ? v.x / s.x : v.x, s.y != 0f ? v.y / s.y : v.y, s.z != 0f ? v.z / s.z : v.z);
		}

		public void SetPositionAndRotation(Vector3 position, Quaternion rotation)
		{
			this.position = position;
			this.rotation = rotation;
		}

		public void SetLocalPositionAndRotation(Vector3 localPosition, Quaternion localRotation)
		{
			localPositionField = localPosition;
			localRotationField = localRotation;
		}

		public void Translate(Vector3 translation) => Translate(translation, Space.Self);

		public void Translate(Vector3 translation, Space relativeTo)
		{
			if (relativeTo == Space.World) position += translation;
			else position += TransformDirection(translation);
		}

		public void Translate(float x, float y, float z) => Translate(new Vector3(x, y, z), Space.Self);

		public void Rotate(Vector3 eulers) => Rotate(eulers, Space.Self);

		public void Rotate(Vector3 eulers, Space relativeTo)
		{
			Quaternion q = Quaternion.Euler(eulers);
			if (relativeTo == Space.Self) localRotationField = localRotationField * q;
			else rotation = q * rotation;
		}

		public void Rotate(float x, float y, float z) => Rotate(new Vector3(x, y, z), Space.Self);

		public void Rotate(Vector3 axis, float angle) => Rotate(axis, angle, Space.Self);

		public void Rotate(Vector3 axis, float angle, Space relativeTo)
		{
			if (relativeTo == Space.Self)
				localRotationField = localRotationField * Quaternion.AngleAxis(angle, axis);
			else
				rotation = Quaternion.AngleAxis(angle, axis) * rotation;
		}

		public void RotateAround(Vector3 point, Vector3 axis, float angle)
		{
			Quaternion q = Quaternion.AngleAxis(angle, axis);
			position = point + q * (position - point);
			rotation = q * rotation;
		}

		public void LookAt(Transform target) { if (target != null) LookAt(target.position); }

		public void LookAt(Vector3 worldPosition)
		{
			Vector3 dir = worldPosition - position;
			if (dir.sqrMagnitude > 1e-12f) rotation = Quaternion.LookRotation(dir);
		}
	}

	public enum Space { World, Self }
}
