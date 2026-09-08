using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UnityShim
{
	// Stepped once per host frame, after Update, before LateUpdate (Unity order).
	public class CoroutineScheduler
	{
		private readonly List<Coroutine> active = new List<Coroutine>();

		public Coroutine Start(MonoBehaviour owner, IEnumerator routine, string methodName = null)
		{
			var co = new Coroutine(owner, routine, methodName);
			// Unity runs a coroutine synchronously up to its first yield.
			if (!Advance(co)) return co;
			active.Add(co);
			return co;
		}

		public void Stop(Coroutine co)
		{
			if (co != null) co.finished = true;
		}

		public void Stop(MonoBehaviour owner, IEnumerator routine)
		{
			foreach (var co in active)
				if (co.owner == owner && co.routine == routine) co.finished = true;
		}

		public void Stop(MonoBehaviour owner, string methodName)
		{
			foreach (var co in active)
				if (co.owner == owner && co.methodName == methodName) co.finished = true;
		}

		public void Step()
		{
			for (int i = 0; i < active.Count; i++)
			{
				var co = active[i];
				if (co.finished || co.owner == null || co.owner.destroyed || co.owner.gameObject == null || co.owner.gameObject.destroyed)
				{
					active.RemoveAt(i); i--; continue;
				}
				// Disabled GameObject stops coroutines; disabled component does not.
				if (!co.owner.gameObject.activeInHierarchy) { active.RemoveAt(i); i--; continue; }
				if (!co.Ready()) continue;
				if (Advance(co)) { active.RemoveAt(i); i--; }
			}
		}

		// Returns true when finished.
		private bool Advance(Coroutine co)
		{
			while (true)
			{
				bool moved;
				try { moved = co.routine.MoveNext(); }
				catch { co.finished = true; throw; }
				if (!moved) { co.finished = true; return true; }
				object yielded = co.routine.Current;
				switch (yielded)
				{
					case null:
						co.waitUntil = 0f; co.waitFor = null; return false;
					case WaitForSeconds wfs:
						co.waitUntil = Time.time + wfs.seconds; co.waitFor = null; return false;
					case WaitForSecondsRealtime wfsr:
						co.waitUntil = Time.unscaledTime + wfsr.waitTime; co.waitForRealtime = true; co.waitFor = null; return false;
					case WaitForFixedUpdate:
					case WaitForEndOfFrame:
						co.waitUntil = 0f; co.waitFor = null; return false; // treated as one frame
					case Coroutine inner:
						co.waitFor = inner; return false;
					case CustomYieldInstruction custom:
						co.customWait = custom; return false;
					case IEnumerator nested:
						co.waitFor = Start(co.owner, nested); return false;
					default:
						co.waitUntil = 0f; co.waitFor = null; return false; // unknown => one frame
				}
			}
		}
	}
}

namespace UnityEngine
{
	public class YieldInstruction { }

	public sealed class Coroutine : YieldInstruction
	{
		internal readonly MonoBehaviour owner;
		internal readonly IEnumerator routine;
		internal readonly string methodName;
		internal bool finished;
		internal float waitUntil;
		internal bool waitForRealtime;
		internal Coroutine waitFor;
		internal CustomYieldInstruction customWait;

		internal Coroutine(MonoBehaviour owner, IEnumerator routine, string methodName)
		{
			this.owner = owner;
			this.routine = routine;
			this.methodName = methodName;
		}

		internal bool Ready()
		{
			if (waitFor != null)
			{
				if (!waitFor.finished) return false;
				waitFor = null;
			}
			if (customWait != null)
			{
				if (customWait.keepWaiting) return false;
				customWait = null;
			}
			if (waitUntil > 0f)
			{
				float now = waitForRealtime ? Time.unscaledTime : Time.time;
				if (now < waitUntil) return false;
				waitUntil = 0f; waitForRealtime = false;
			}
			return true;
		}
	}

	public sealed class WaitForSeconds : YieldInstruction
	{
		internal readonly float seconds;
		public WaitForSeconds(float seconds) { this.seconds = seconds; }
	}

	public class WaitForSecondsRealtime : CustomYieldInstruction
	{
		public float waitTime;
		private float due = -1f;
		public WaitForSecondsRealtime(float time) { waitTime = time; }
		public override bool keepWaiting
		{
			get
			{
				if (due < 0f) due = Time.unscaledTime + waitTime;
				return Time.unscaledTime < due;
			}
		}
	}

	public sealed class WaitForFixedUpdate : YieldInstruction { }
	public sealed class WaitForEndOfFrame : YieldInstruction { }

	public abstract class CustomYieldInstruction : IEnumerator
	{
		public abstract bool keepWaiting { get; }
		public object Current => null;
		public bool MoveNext() => keepWaiting;
		public void Reset() { }
	}

	public class WaitUntil : CustomYieldInstruction
	{
		private readonly Func<bool> predicate;
		public WaitUntil(Func<bool> predicate) { this.predicate = predicate; }
		public override bool keepWaiting => !predicate();
	}

	public class WaitWhile : CustomYieldInstruction
	{
		private readonly Func<bool> predicate;
		public WaitWhile(Func<bool> predicate) { this.predicate = predicate; }
		public override bool keepWaiting => predicate();
	}
}
