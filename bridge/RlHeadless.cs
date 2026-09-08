using UnityEngine;
using UnityEngine.Rendering;

// Skip presentation-only work when the player has no graphics device. Desktop
// play and rendered replays retain their graphics without a separate switch.
internal static class RlHeadless
{
    internal static readonly bool Render = SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null;
}
