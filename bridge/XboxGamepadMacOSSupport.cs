#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.LowLevel;
using UnityEngine.InputSystem.Utilities;
using UnityEngine.InputSystem.XInput;
using UnityEngine.Scripting;

// The shipped Unity.InputSystem.dll was ripped from the Windows build, so the
// macOS-only Xbox layouts (compiled under UNITY_STANDALONE_OSX in the package's
// XInputSupport) are missing: a Bluetooth Xbox controller shows up as a generic
// HID, never a Gamepad, and the "GamePad" control scheme ignores it. These are
// the package's own layouts (com.unity.inputsystem 1.11.2,
// InputSystem/Plugins/XInput/XboxGamepadMacOS.cs) registered from game code.

namespace BoplMacOS
{
    // Wired Xbox 360 / One pad behind the 360Controller driver: 14-byte report,
    // 2-byte header, 16-bit buttons, byte triggers, signed 16-bit sticks.
    [StructLayout(LayoutKind.Explicit, Size = 14)]
    public struct XboxWiredMacOSState : IInputStateTypeInfo
    {
        public static FourCC kFormat => new FourCC('H', 'I', 'D');

        [FieldOffset(0)] public ushort padding;

        [InputControl(name = "dpad", layout = "Dpad", sizeInBits = 4, bit = 0)]
        [InputControl(name = "dpad/up", bit = 0)]
        [InputControl(name = "dpad/down", bit = 1)]
        [InputControl(name = "dpad/left", bit = 2)]
        [InputControl(name = "dpad/right", bit = 3)]
        [InputControl(name = "start", bit = 4, displayName = "Start")]
        [InputControl(name = "select", bit = 5, displayName = "Select")]
        [InputControl(name = "leftStickPress", bit = 6)]
        [InputControl(name = "rightStickPress", bit = 7)]
        [InputControl(name = "leftShoulder", bit = 8)]
        [InputControl(name = "rightShoulder", bit = 9)]
        [InputControl(name = "buttonSouth", bit = 12, displayName = "A")]
        [InputControl(name = "buttonEast", bit = 13, displayName = "B")]
        [InputControl(name = "buttonWest", bit = 14, displayName = "X")]
        [InputControl(name = "buttonNorth", bit = 15, displayName = "Y")]
        [FieldOffset(2)] public ushort buttons;

        [InputControl(name = "leftTrigger", format = "BYTE")]
        [FieldOffset(4)] public byte leftTrigger;
        [InputControl(name = "rightTrigger", format = "BYTE")]
        [FieldOffset(5)] public byte rightTrigger;

        [InputControl(name = "leftStick", layout = "Stick", format = "VC2S")]
        [InputControl(name = "leftStick/x", offset = 0, format = "SHRT", parameters = "")]
        [InputControl(name = "leftStick/left", offset = 0, format = "SHRT", parameters = "")]
        [InputControl(name = "leftStick/right", offset = 0, format = "SHRT", parameters = "")]
        [InputControl(name = "leftStick/y", offset = 2, format = "SHRT", parameters = "invert")]
        [InputControl(name = "leftStick/up", offset = 2, format = "SHRT", parameters = "clamp=1,clampMin=-1,clampMax=0,invert=true")]
        [InputControl(name = "leftStick/down", offset = 2, format = "SHRT", parameters = "clamp=1,clampMin=0,clampMax=1,invert=false")]
        [FieldOffset(6)] public short leftStickX;
        [FieldOffset(8)] public short leftStickY;

        [InputControl(name = "rightStick", layout = "Stick", format = "VC2S")]
        [InputControl(name = "rightStick/x", offset = 0, format = "SHRT", parameters = "")]
        [InputControl(name = "rightStick/left", offset = 0, format = "SHRT", parameters = "")]
        [InputControl(name = "rightStick/right", offset = 0, format = "SHRT", parameters = "")]
        [InputControl(name = "rightStick/y", offset = 2, format = "SHRT", parameters = "invert")]
        [InputControl(name = "rightStick/up", offset = 2, format = "SHRT", parameters = "clamp=1,clampMin=-1,clampMax=0,invert=true")]
        [InputControl(name = "rightStick/down", offset = 2, format = "SHRT", parameters = "clamp=1,clampMin=0,clampMax=1,invert=false")]
        [FieldOffset(10)] public short rightStickX;
        [FieldOffset(12)] public short rightStickY;

        public FourCC format => kFormat;
    }

    // Bluetooth Xbox One S / Series pad (product "Xbox Wireless Controller"):
    // report id, four unsigned 16-bit stick axes, two 10-bit triggers, a hat
    // switch byte and a 32-bit button mask.
    [StructLayout(LayoutKind.Explicit, Size = 18)]
    public struct XboxWirelessMacOSState : IInputStateTypeInfo
    {
        public static FourCC kFormat => new FourCC('H', 'I', 'D');

        [FieldOffset(0)] public byte padding;

        [InputControl(name = "leftStick", layout = "Stick", format = "VC2S")]
        [InputControl(name = "leftStick/x", offset = 0, format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5")]
        [InputControl(name = "leftStick/left", offset = 0, format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5,clamp=1,clampMin=0,clampMax=0.5,invert")]
        [InputControl(name = "leftStick/right", offset = 0, format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5,clamp=1,clampMin=0.5,clampMax=1")]
        [InputControl(name = "leftStick/y", offset = 2, format = "USHT", parameters = "invert,normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5")]
        [InputControl(name = "leftStick/up", offset = 2, format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5,clamp=1,clampMin=0,clampMax=0.5,invert")]
        [InputControl(name = "leftStick/down", offset = 2, format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5,clamp=1,clampMin=0.5,clampMax=1,invert=false")]
        [FieldOffset(1)] public ushort leftStickX;
        [FieldOffset(3)] public ushort leftStickY;

        [InputControl(name = "rightStick", layout = "Stick", format = "VC2S")]
        [InputControl(name = "rightStick/x", offset = 0, format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5")]
        [InputControl(name = "rightStick/left", offset = 0, format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5,clamp=1,clampMin=0,clampMax=0.5,invert")]
        [InputControl(name = "rightStick/right", offset = 0, format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5,clamp=1,clampMin=0.5,clampMax=1")]
        [InputControl(name = "rightStick/y", offset = 2, format = "USHT", parameters = "invert,normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5")]
        [InputControl(name = "rightStick/up", offset = 2, format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5,clamp=1,clampMin=0,clampMax=0.5,invert")]
        [InputControl(name = "rightStick/down", offset = 2, format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=1,normalizeZero=0.5,clamp=1,clampMin=0.5,clampMax=1,invert=false")]
        [FieldOffset(5)] public ushort rightStickX;
        [FieldOffset(7)] public ushort rightStickY;

        [InputControl(name = "leftTrigger", format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=0.01560998")]
        [FieldOffset(9)] public ushort leftTrigger;
        [InputControl(name = "rightTrigger", format = "USHT", parameters = "normalize,normalizeMin=0,normalizeMax=0.01560998")]
        [FieldOffset(11)] public ushort rightTrigger;

        [InputControl(name = "dpad", format = "BIT", layout = "Dpad", sizeInBits = 4, defaultState = 8)]
        [InputControl(name = "dpad/up", format = "BIT", layout = "DiscreteButton", parameters = "minValue=8,maxValue=2,nullValue=0,wrapAtValue=9", bit = 0, sizeInBits = 4)]
        [InputControl(name = "dpad/right", format = "BIT", layout = "DiscreteButton", parameters = "minValue=2,maxValue=4", bit = 0, sizeInBits = 4)]
        [InputControl(name = "dpad/down", format = "BIT", layout = "DiscreteButton", parameters = "minValue=4,maxValue=6", bit = 0, sizeInBits = 4)]
        [InputControl(name = "dpad/left", format = "BIT", layout = "DiscreteButton", parameters = "minValue=6, maxValue=8", bit = 0, sizeInBits = 4)]
        [FieldOffset(13)] public byte dpad;

        [InputControl(name = "start", bit = 11, displayName = "Start")]
        [InputControl(name = "select", bit = 10, displayName = "Select")]
        [InputControl(name = "leftStickPress", bit = 13)]
        [InputControl(name = "rightStickPress", bit = 14)]
        [InputControl(name = "leftShoulder", bit = 6)]
        [InputControl(name = "rightShoulder", bit = 7)]
        [InputControl(name = "buttonSouth", bit = 0, displayName = "A")]
        [InputControl(name = "buttonEast", bit = 1, displayName = "B")]
        [InputControl(name = "buttonWest", bit = 3, displayName = "X")]
        [InputControl(name = "buttonNorth", bit = 4, displayName = "Y")]
        [FieldOffset(14)] public uint buttons;

        public FourCC format => kFormat;
    }

    [Preserve]
    [InputControlLayout(displayName = "Xbox Controller", stateType = typeof(XboxWiredMacOSState), hideInUI = true)]
    public class XboxGamepadMacOS : XInputController
    {
    }

    [Preserve]
    [InputControlLayout(displayName = "Wireless Xbox Controller", stateType = typeof(XboxWirelessMacOSState), hideInUI = true)]
    public class XboxOneGamepadMacOSWireless : XInputController
    {
    }

    public static class XboxGamepadMacOSSupport
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Register()
        {
            // Registering re-creates already connected devices that now match,
            // so a pad paired before launch is picked up too.
            InputSystem.RegisterLayout<XboxGamepadMacOS>(
                matches: new InputDeviceMatcher().WithInterface("HID").WithProduct("Xbox.*Wired Controller"));
            InputSystem.RegisterLayout<XboxOneGamepadMacOSWireless>(
                matches: new InputDeviceMatcher().WithInterface("HID").WithProduct("Xbox.*Wireless Controller"));
            Debug.Log("XboxGamepadMacOSSupport: registered macOS Xbox controller layouts");
        }
    }
}
#endif
