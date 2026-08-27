using System;
using UnityEngine;
using UnityEngine.InputSystem;

// Menu navigation with standard console UX: dominant-axis quantization, a lower
// menu-only deadzone than gameplay, and delayed auto-shift (fire on press, then
// repeat while held). Replaces the original one-shot Press interaction on the
// MenuMap Move actions, which forced one item per stick deflection.
public class MenuNavRepeater : MonoBehaviour
{
	private const float Deadzone = 0.25f;

	private const float InitialDelay = 0.35f;

	private const float RepeatInterval = 0.07f;

	private static MenuNavRepeater instance;

	public static MenuNavRepeater Instance
	{
		get
		{
			if (instance == null)
			{
				GameObject gameObject = new GameObject("MenuNavRepeater");
				UnityEngine.Object.DontDestroyOnLoad(gameObject);
				instance = gameObject.AddComponent<MenuNavRepeater>();
			}
			return instance;
		}
	}

	public static MenuNavRepeater InstanceIfExists => instance;

	public event Action<Vector2> Navigated;

	private Vector2 heldDir;

	private float heldTime;

	private float nextRepeatAt;

	private void Update()
	{
		MenuControls.MenuMapActions menuMap = MenuInputHandler.Instance.MenuMap;
		if (!menuMap.Get().enabled)
		{
			heldDir = Vector2.zero;
			return;
		}
		Vector2 vector = BestVector(menuMap);
		Vector2 vector2 = Quantize(vector);
		if (vector2 != heldDir)
		{
			heldDir = vector2;
			heldTime = 0f;
			nextRepeatAt = InitialDelay;
			if (vector2 != Vector2.zero)
			{
				this.Navigated?.Invoke(vector2);
			}
			return;
		}
		if (!(vector2 == Vector2.zero))
		{
			heldTime += Time.unscaledDeltaTime;
			if (heldTime >= nextRepeatAt)
			{
				nextRepeatAt += RepeatInterval;
				this.Navigated?.Invoke(vector2);
			}
		}
	}

	private static Vector2 BestVector(MenuControls.MenuMapActions map)
	{
		Vector2 vector = map.Move0.ReadValue<Vector2>();
		Vector2 vector2 = map.Move1.ReadValue<Vector2>();
		Vector2 vector3 = map.Move2.ReadValue<Vector2>();
		Vector2 vector4 = map.Move3.ReadValue<Vector2>();
		Vector2 vector5 = map.Move4.ReadValue<Vector2>();
		Vector2 result = vector;
		if (vector2.sqrMagnitude > result.sqrMagnitude)
		{
			result = vector2;
		}
		if (vector3.sqrMagnitude > result.sqrMagnitude)
		{
			result = vector3;
		}
		if (vector4.sqrMagnitude > result.sqrMagnitude)
		{
			result = vector4;
		}
		if (vector5.sqrMagnitude > result.sqrMagnitude)
		{
			result = vector5;
		}
		return result;
	}

	private static Vector2 Quantize(Vector2 v)
	{
		if (Mathf.Abs(v.x) > Mathf.Abs(v.y))
		{
			if (!(Mathf.Abs(v.x) >= Deadzone))
			{
				return Vector2.zero;
			}
			return new Vector2(Mathf.Sign(v.x), 0f);
		}
		if (!(Mathf.Abs(v.y) >= Deadzone))
		{
			return Vector2.zero;
		}
		return new Vector2(0f, Mathf.Sign(v.y));
	}
}
