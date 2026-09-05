using System;
using BoplFixedMath;

// A callback endpoint; all scheduling is performed by the actual Updater.
public sealed class ReferenceScheduledComponent : MonoUpdatable
{
    public int CanonicalId;
    public bool EnabledFlag;
    public bool IgnoreActivity;
    public Action<string,int> Notify;
    public override bool IsEnabled()=>EnabledFlag&&(IgnoreActivity||gameObject.activeInHierarchy);
    public override void Init()=>Notify("init",CanonicalId);
    public override void UpdateSim(Fix dt)=>Notify("update",CanonicalId);
    public override void LateUpdateSim(Fix dt)=>Notify("late",CanonicalId);
    public override void OnDestroyUpdatable()=>Notify("destroy_updatable",CanonicalId);
}
