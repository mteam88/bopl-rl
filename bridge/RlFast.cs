using System;

// Opt-in fast paths for the RL harness that must stay bit-exact with the
// default build (checked with check_game_variant.py / check_pipeline.py).
internal static class RlFast
{
    // BOPL_RL_RAYCULL=1: in DetPhysics.RaycastToClosestRoundedRect, skip rounded
    // rects whose axis-aligned bounds (expanded by radius plus a one-unit margin)
    // do not touch the bounds of the ray segment. Every hit the full test can
    // report lies on the segment within rounding, so culled rects cannot hit.
    internal static readonly bool RayCull = Environment.GetEnvironmentVariable("BOPL_RL_RAYCULL") == "1";
}
