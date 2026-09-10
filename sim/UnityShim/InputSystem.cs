using System;
using System.Collections;
using System.Collections.Generic;

// UnityEngine.InputSystem stubs. Headless sim has no devices:
// Keyboard.current / Gamepad.current are null and Gamepad.all is empty, which
// is exactly what the game sees on the RL headless player. Input reaches
// gameplay via Player.OverrideInputWithNetworkInput.
namespace UnityEngine.InputSystem
{
	using UnityEngine.InputSystem.Controls;
	using UnityEngine.InputSystem.Utilities;

	public static class InputSystem
	{
		public static float pollingFrequency { get; set; } = 60f;
		public static ReadOnlyArray<InputDevice> devices => new ReadOnlyArray<InputDevice>(Array.Empty<InputDevice>());
		public static ReadOnlyArray<InputDevice> disconnectedDevices => new ReadOnlyArray<InputDevice>(Array.Empty<InputDevice>());
		public static event Action<InputDevice, InputDeviceChange> onDeviceChange { add { } remove { } }
		public static void EnableDevice(InputDevice device) { }
		public static void DisableDevice(InputDevice device) { }
		public static InputDevice AddDevice(string layout) => null;
		public static void RemoveDevice(InputDevice device) { }
		public static void Update() { }
	}

	public enum InputDeviceChange { Added, Removed, Disconnected, Reconnected, Enabled, Disabled }

	public class InputDevice
	{
		public string name { get; set; } = "";
		public string displayName => name;
		public int deviceId { get; set; }
		public bool added => false;
		public bool enabled => false;
		public InputDeviceDescription description { get; set; }
		public Utilities.ReadOnlyArray<InputControl> allControls => new Utilities.ReadOnlyArray<InputControl>(Array.Empty<InputControl>());
		public virtual bool IsActuated(float threshold = 0f) => false;
	}

	public struct InputDeviceDescription
	{
		public string manufacturer { get; set; }
		public string product { get; set; }
		public string interfaceName { get; set; }
	}

	public class Keyboard : InputDevice
	{
		public static Keyboard current => null;

		public KeyControl escapeKey => KeyControl.dummy;
		public KeyControl f8Key => KeyControl.dummy;
		public KeyControl f9Key => KeyControl.dummy;
		public KeyControl f10Key => KeyControl.dummy;
		public KeyControl spaceKey => KeyControl.dummy;
		public KeyControl enterKey => KeyControl.dummy;
		public KeyControl backspaceKey => KeyControl.dummy;
		public KeyControl tabKey => KeyControl.dummy;
		public KeyControl deleteKey => KeyControl.dummy;
		public KeyControl leftShiftKey => KeyControl.dummy;
		public KeyControl rightShiftKey => KeyControl.dummy;
		public KeyControl leftCtrlKey => KeyControl.dummy;
		public KeyControl wKey => KeyControl.dummy;
		public KeyControl aKey => KeyControl.dummy;
		public KeyControl sKey => KeyControl.dummy;
		public KeyControl dKey => KeyControl.dummy;
		public KeyControl upArrowKey => KeyControl.dummy;
		public KeyControl downArrowKey => KeyControl.dummy;
		public KeyControl leftArrowKey => KeyControl.dummy;
		public KeyControl rightArrowKey => KeyControl.dummy;
		public KeyControl anyKey => KeyControl.dummy;
		public IEnumerable<KeyControl> allKeys => Array.Empty<KeyControl>();
		public KeyControl this[Key key] => KeyControl.dummy;
	}

	public class Mouse : InputDevice
	{
		public static Mouse current => null;
		public ButtonControl leftButton => ButtonControl.dummy;
		public ButtonControl rightButton => ButtonControl.dummy;
		public ButtonControl middleButton => ButtonControl.dummy;
		public ButtonControl forwardButton => ButtonControl.dummy;
		public ButtonControl backButton => ButtonControl.dummy;
		public Vector2Control position => Vector2Control.dummy;
		public Vector2Control delta => Vector2Control.dummy;
		public DeltaControl scroll => DeltaControl.dummy;
	}

	public class Gamepad : InputDevice
	{
		public static Gamepad current => null;
		public static ReadOnlyArray<Gamepad> all => new ReadOnlyArray<Gamepad>(Array.Empty<Gamepad>());

		public StickControl leftStick => StickControl.dummy;
		public StickControl rightStick => StickControl.dummy;
		public ButtonControl buttonSouth => ButtonControl.dummy;
		public ButtonControl buttonNorth => ButtonControl.dummy;
		public ButtonControl buttonEast => ButtonControl.dummy;
		public ButtonControl buttonWest => ButtonControl.dummy;
		public ButtonControl leftShoulder => ButtonControl.dummy;
		public ButtonControl rightShoulder => ButtonControl.dummy;
		public ButtonControl leftTrigger => ButtonControl.dummy;
		public ButtonControl rightTrigger => ButtonControl.dummy;
		public ButtonControl startButton => ButtonControl.dummy;
		public ButtonControl selectButton => ButtonControl.dummy;
		public ButtonControl aButton => ButtonControl.dummy;
		public ButtonControl bButton => ButtonControl.dummy;
		public ButtonControl xButton => ButtonControl.dummy;
		public ButtonControl yButton => ButtonControl.dummy;
		public ButtonControl leftStickButton => ButtonControl.dummy;
		public ButtonControl rightStickButton => ButtonControl.dummy;
		public DpadControl dpad => DpadControl.dummy;
		public void SetMotorSpeeds(float lowFrequency, float highFrequency) { }
		public void PauseHaptics() { }
		public void ResumeHaptics() { }
		public void ResetHaptics() { }
	}

	public enum Key
	{
		None, Space, Enter, Tab, Backquote, Quote, Semicolon, Comma, Period, Slash,
		Backslash, LeftBracket, RightBracket, Minus, Equals,
		A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
		Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9, Digit0,
		LeftShift, RightShift, LeftAlt, RightAlt, LeftCtrl, RightCtrl, LeftMeta, RightMeta,
		Escape, LeftArrow, RightArrow, UpArrow, DownArrow, Backspace, PageDown, PageUp,
		Home, End, Insert, Delete, CapsLock, NumLock, PrintScreen, ScrollLock, Pause,
		F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12
	}

	public class InputControl
	{
		public string name { get; set; } = "";
		public string path { get; set; } = "";
		public InputDevice device { get; set; }
		public InputControl parent { get; set; }
		public string displayName => name;
		public virtual bool IsActuated(float threshold = 0f) => false;
		public virtual bool IsPressed() => false;
	}

	public class InputControl<TValue> : InputControl where TValue : struct
	{
		public virtual TValue ReadValue() => default;
	}

	public class InputAction : IDisposable
	{
		public string name { get; set; } = "";
		public bool enabled { get; private set; }
		public InputActionMap actionMap { get; set; }
		public InputActionType type { get; set; }

		public event Action<CallbackContext> started { add { } remove { } }
		public event Action<CallbackContext> performed { add { } remove { } }
		public event Action<CallbackContext> canceled { add { } remove { } }

		public InputAction() { }
		public InputAction(string name) { this.name = name; }
		public InputAction(string name, InputActionType type) { this.name = name; this.type = type; }

		public void Enable() => enabled = true;
		public void Disable() => enabled = false;
		public void Dispose() { }

		public TValue ReadValue<TValue>() where TValue : struct => default;
		public bool IsPressed() => false;
		public bool WasPressedThisFrame() => false;
		public bool WasReleasedThisFrame() => false;
		public bool WasPerformedThisFrame() => false;
		public bool triggered => false;
		public InputControl activeControl => null;
		public ReadOnlyArray<InputControl> controls => new ReadOnlyArray<InputControl>(Array.Empty<InputControl>());
		public void AddBinding(string path) { }
		public void ApplyBindingOverride(string overridePath) { }
		public void ApplyBindingOverride(int bindingIndex, string overridePath) { }

		public struct CallbackContext
		{
			public InputAction action => null;
			public InputControl control => null;
			public bool started => false;
			public bool performed => false;
			public bool canceled => false;
			public TValue ReadValue<TValue>() where TValue : struct => default;
			public bool ReadValueAsButton() => false;
			public double time => 0.0;
		}
	}

	public enum InputActionType { Value, Button, PassThrough }

	public interface IInputActionCollection : IEnumerable<InputAction>
	{
		InputBinding? bindingMask { get; set; }
		ReadOnlyArray<InputDevice>? devices { get; set; }
		ReadOnlyArray<InputControlScheme> controlSchemes { get; }
		bool Contains(InputAction action);
		void Enable();
		void Disable();
	}

	public interface IInputActionCollection2 : IInputActionCollection
	{
		IEnumerable<InputBinding> bindings { get; }
		InputAction FindAction(string actionNameOrId, bool throwIfNotFound = false);
		int FindBinding(InputBinding bindingMask, out InputAction action);
	}

	public class InputActionMap : IEnumerable<InputAction>
	{
		public string name { get; set; } = "";
		public bool enabled { get; private set; }
		internal readonly List<InputAction> actions = new List<InputAction>();
		public InputActionAsset asset { get; internal set; }
		public InputActionMap() { }
		public InputActionMap(string name) { this.name = name; }
		public void Enable() { enabled = true; foreach (var a in actions) a.Enable(); }
		public void Disable() { enabled = false; foreach (var a in actions) a.Disable(); }
		public InputAction FindAction(string actionName, bool throwIfNotFound = false)
		{
			var found = actions.Find(a => a.name == actionName);
			if (found == null && throwIfNotFound) throw new ArgumentException($"No action '{actionName}' in map '{name}'");
			return found;
		}
		public InputAction AddAction(string name) { var a = new InputAction(name) { actionMap = this }; actions.Add(a); return a; }
		public bool Contains(InputAction action) => actions.Contains(action);
		public IEnumerator<InputAction> GetEnumerator() => actions.GetEnumerator();
		IEnumerator IEnumerable.GetEnumerator() => actions.GetEnumerator();
	}

	public class InputActionAsset : ScriptableObject, IInputActionCollection2
	{
		internal readonly List<InputActionMap> maps = new List<InputActionMap>();
		private InputControlScheme[] schemes = Array.Empty<InputControlScheme>();

		public ReadOnlyArray<InputActionMap> actionMaps => new ReadOnlyArray<InputActionMap>(maps.ToArray());
		public InputBinding? bindingMask { get; set; }
		public ReadOnlyArray<InputDevice>? devices { get; set; }
		public ReadOnlyArray<InputControlScheme> controlSchemes => new ReadOnlyArray<InputControlScheme>(schemes);
		public IEnumerable<InputBinding> bindings => Array.Empty<InputBinding>();

		public void Enable() { foreach (var m in maps) m.Enable(); }
		public void Disable() { foreach (var m in maps) m.Disable(); }
		public bool Contains(InputAction action) { foreach (var m in maps) if (m.Contains(action)) return true; return false; }

		public InputAction FindAction(string actionNameOrId, bool throwIfNotFound = false)
		{
			foreach (var m in maps)
			{
				var a = m.FindAction(actionNameOrId);
				if (a != null) return a;
			}
			if (throwIfNotFound) throw new ArgumentException($"No action '{actionNameOrId}'");
			return null;
		}

		public int FindBinding(InputBinding mask, out InputAction action) { action = null; return -1; }

		public InputActionMap FindActionMap(string nameOrId, bool throwIfNotFound = false)
		{
			var found = maps.Find(m => m.name == nameOrId);
			if (found == null && throwIfNotFound) throw new ArgumentException($"No action map '{nameOrId}'");
			return found;
		}

		public int FindControlSchemeIndex(string name)
		{
			for (int i = 0; i < schemes.Length; i++)
				if (schemes[i].name == name) return i;
			return -1;
		}

		// Parses the generated wrapper JSON just enough for FindActionMap/FindAction.
		public static InputActionAsset FromJson(string json)
		{
			var asset = new InputActionAsset();
			var root = Newtonsoft.Json.Linq.JObject.Parse(json);
			asset.name = (string)root["name"] ?? "";
			var jmaps = root["maps"] as Newtonsoft.Json.Linq.JArray;
			if (jmaps != null)
			{
				foreach (var jmap in jmaps)
				{
					var map = new InputActionMap((string)jmap["name"] ?? "") { asset = asset };
					var jactions = jmap["actions"] as Newtonsoft.Json.Linq.JArray;
					if (jactions != null)
						foreach (var ja in jactions)
						{
							var action = map.AddAction((string)ja["name"] ?? "");
							action.type = ((string)ja["type"]) switch
							{
								"Button" => InputActionType.Button,
								"PassThrough" => InputActionType.PassThrough,
								_ => InputActionType.Value,
							};
						}
					asset.maps.Add(map);
				}
			}
			var jschemes = root["controlSchemes"] as Newtonsoft.Json.Linq.JArray;
			if (jschemes != null)
			{
				var list = new List<InputControlScheme>();
				foreach (var js in jschemes)
					list.Add(new InputControlScheme { name = (string)js["name"], bindingGroup = (string)js["bindingGroup"] });
				asset.schemes = list.ToArray();
			}
			return asset;
		}

		public IEnumerator<InputAction> GetEnumerator()
		{
			foreach (var m in maps)
				foreach (var a in m.actions)
					yield return a;
		}

		IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
	}

	public struct InputBinding
	{
		public string path { get; set; }
		public string groups { get; set; }
		public string action { get; set; }
		public string effectivePath => path;
	}

	public struct InputControlScheme
	{
		public string name { get; set; }
		public string bindingGroup { get; set; }
	}

	public class PlayerInputManager : MonoBehaviour
	{
		public static PlayerInputManager instance { get => UnityShim.World.Current.playerInputManager; set => UnityShim.World.Current.playerInputManager = value; }
		public bool joiningEnabled { get; private set; }
		public int playerCount => 0;
		public int maxPlayerCount { get; set; } = -1;
		public PlayerNotifications notificationBehavior { get; set; }
		public GameObject playerPrefab { get; set; }
		public event Action<PlayerInput> onPlayerJoined { add { } remove { } }
		public event Action<PlayerInput> onPlayerLeft { add { } remove { } }
		public void EnableJoining() => joiningEnabled = true;
		public void DisableJoining() => joiningEnabled = false;
		public PlayerInput JoinPlayer(int playerIndex = -1, int splitScreenIndex = -1, string controlScheme = null, InputDevice pairWithDevice = null) => null;
	}

	public enum PlayerNotifications { SendMessages, BroadcastMessages, InvokeUnityEvents, InvokeCSharpEvents }

	public class PlayerInput : MonoBehaviour
	{
		public static ReadOnlyArray<PlayerInput> all => new ReadOnlyArray<PlayerInput>(Array.Empty<PlayerInput>());
		public int playerIndex { get; set; }
		public InputActionAsset actions { get; set; }
		public InputActionMap currentActionMap { get; set; }
		public bool neverAutoSwitchControlSchemes { get; set; }
		public bool inputIsActive { get; private set; }
		public void ActivateInput() => inputIsActive = true;
		public void DeactivateInput() => inputIsActive = false;
		public string currentControlScheme => "";
		public ReadOnlyArray<InputDevice> devices => new ReadOnlyArray<InputDevice>(Array.Empty<InputDevice>());
		public event Action<PlayerInput> onDeviceLost { add { } remove { } }
		public event Action<PlayerInput> onDeviceRegained { add { } remove { } }
		public void SwitchCurrentControlScheme(string scheme, params InputDevice[] devices) { }
		public InputAction FindAction(string name) => actions?.FindAction(name);
	}
}

namespace UnityEngine.InputSystem.Controls
{
	public class ButtonControl : InputControl<float>
	{
		internal static readonly ButtonControl dummy = new ButtonControl();
		public bool isPressed => false;
		public bool wasPressedThisFrame => false;
		public bool wasReleasedThisFrame => false;
		public override bool IsPressed() => false;
		public override bool IsActuated(float threshold = 0f) => false;
	}

	public class KeyControl : ButtonControl
	{
		internal static new readonly KeyControl dummy = new KeyControl();
		public Key keyCode { get; set; }
	}

	public class Vector2Control : InputControl<Vector2>
	{
		internal static readonly Vector2Control dummy = new Vector2Control();
		public override Vector2 ReadValue() => Vector2.zero;
	}

	public class StickControl : Vector2Control
	{
		internal static new readonly StickControl dummy = new StickControl();
		public ButtonControl up => ButtonControl.dummy;
		public ButtonControl down => ButtonControl.dummy;
		public ButtonControl left => ButtonControl.dummy;
		public ButtonControl right => ButtonControl.dummy;
	}

	public class DpadControl : Vector2Control
	{
		internal static new readonly DpadControl dummy = new DpadControl();
		public ButtonControl up => ButtonControl.dummy;
		public ButtonControl down => ButtonControl.dummy;
		public ButtonControl left => ButtonControl.dummy;
		public ButtonControl right => ButtonControl.dummy;
	}

	public class DeltaControl : Vector2Control
	{
		internal static new readonly DeltaControl dummy = new DeltaControl();
		public FloatControl y => FloatControl.dummy;
		public FloatControl x => FloatControl.dummy;
	}

	public class FloatControl : InputControl<float>
	{
		internal static readonly FloatControl dummy = new FloatControl();
	}
}

namespace UnityEngine.InputSystem.UI
{
	// Present in exported scenes; UI input is irrelevant to the sim.
	public class InputSystemUIInputModule : MonoBehaviour
	{
	}
}

namespace UnityEngine.InputSystem.Utilities
{
	public struct ReadOnlyArray<TValue> : IEnumerable<TValue>
	{
		private readonly TValue[] values;
		public ReadOnlyArray(TValue[] array) { values = array ?? Array.Empty<TValue>(); }
		public int Count => values?.Length ?? 0;
		public TValue this[int index] => values[index];
		public IEnumerator<TValue> GetEnumerator() => ((IEnumerable<TValue>)(values ?? Array.Empty<TValue>())).GetEnumerator();
		IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
		public TValue[] ToArray() => (TValue[])(values ?? Array.Empty<TValue>()).Clone();
	}
}
