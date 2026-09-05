// Supplies the custom cooldown-ignore property while retaining Ability's real
// GetCooldown, IsCastable, ground gate and time-stop configuration.
internal sealed class ReferenceRoutingAbility : Ability
{
    internal bool IgnoreCooldown;
    public override bool IgnoreNextCooldown => IgnoreCooldown;
}
