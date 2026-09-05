using System;
using BoplFixedMath;

// Scripted motor/dispatch adapter for composed component tests. These delegates
// call original PlayerPhysics/PlayerBody methods; this is not SlimeController.
public sealed class ReferenceActorDriver : MonoUpdatable
{
    public bool EnabledFlag = true;
    public Func<bool> Activity;
    public Action Initialize;
    public Action<Fix> Step;
    public Action<Fix> Late;
    public override bool IsEnabled() => EnabledFlag&&(Activity==null||Activity());
    public override void Init() => Initialize();
    public override void UpdateSim(Fix dt) => Step(dt);
    public override void LateUpdateSim(Fix dt) => Late(dt);
}
