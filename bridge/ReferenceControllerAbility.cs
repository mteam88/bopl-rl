using System;
using BoplFixedMath;
using UnityEngine;

// Records the ability callback boundary while the original controller runs.
public sealed class ReferenceControllerAbility : MonoBehaviour, IAbilityComponent
{
    public Action<AbilityExitInfo> OnExit;
    public void OnEnterAbility() { }
    public void OnScaleChanged(Fix multiplier) { }
    public void ExitAbility(AbilityExitInfo info) => OnExit(info);
}
