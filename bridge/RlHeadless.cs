using System;

// BOPL_RL_HEADLESS=1 skips work that only feeds rendering (debug lines,
// material/shader parameters, sprite tints). Simulation state is untouched, so
// the headless game must stay bit-exact with the default build; the RL harness
// checks that with check_game_variant.py. Off by default so the desktop game
// and replay viewer behave exactly as before.
internal static class RlHeadless
{
    internal static readonly bool Render = Environment.GetEnvironmentVariable("BOPL_RL_HEADLESS") != "1";
}
