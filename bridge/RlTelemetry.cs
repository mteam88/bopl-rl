using System;
using System.IO;

// Episode counters for research only. These never enter gameplay, policy
// observations, rewards, or the replay. Warmup ticks are excluded.
internal static class RlTelemetry
{
    internal const int Fields = 9;
    private static readonly uint[,] counters = new uint[6, Fields];
    internal static bool Counting;
    internal static bool Enabled => RlBridge.Enabled && Counting;

    internal static void Reset()
    {
        Array.Clear(counters, 0, counters.Length);
        Counting = false;
    }

    private static void Add(int playerId, int field)
    {
        if (Enabled && playerId >= 1 && playerId <= 6) counters[playerId - 1, field]++;
    }

    internal static void AbilityActivated(int playerId, int slot)
    {
        if (slot >= 0 && slot < 3) Add(playerId, slot);
    }

    internal static void ArrowFired(int playerId) => Add(playerId, 3);
    // Arrow collisions that killed the victim, excluding hits on the owner.
    // This is not all geometric contacts or a count of unique projectiles.
    internal static void ArrowHit(int owner, int victim)
    {
        if (owner != victim) Add(owner, 4);
    }

    internal static void Death(int victim, int killer)
    {
        if (!Enabled) return;
        int field = killer == Constants.ENVIRONMENTAL_PLAYER_ID ? 5
            : killer == victim ? 7
            : PlayerHandler.Get().GetPlayer(killer) != null ? 6 : 8;
        Add(victim, field);
    }

    internal static void Write(BinaryWriter output, int playerId)
    {
        for (int field = 0; field < Fields; field++) output.Write(counters[playerId - 1, field]);
    }
}
