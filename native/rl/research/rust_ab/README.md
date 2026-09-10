# Scripted Rust vs .NET A/B tools

Small C drivers that open both backends through `bopl_rl.c` and print
per-decision traces side by side. Each takes `<BoplSimNative.so>
<libbopl_sim.so>` and expects the fidelity gate's bootstrap replay at
`/tmp/gate_bootstrap.rep`. Build like the fidelity gate:

```sh
cd native/rl && cc -O2 -I. -o /tmp/move_ab research/rust_ab/move_ab.c bopl_rl.c -ldl -lm
```

- `move_ab.c`: fixed-action angle sweep, positions per decision.
- `state_ab.c`: idle/run/turn/stop, SlimeController state and timers.
- `bow_ab.c`: hold arrow N decisions then release, bow slot flags/timers.
- `aim_map.c`: full-charge arrow velocity per angle byte.
- `hit_ab.c`: charge and fire at the opponent, hit outcome and death cause.
- `bow_test.c`: Rust-only bow scenario used while porting.
- `platform_ab.c`: grounded platform transform overlap, jump-and-land player
  carry, and the map-2 AntiLockPlatform trajectory.
- `platform_phase_ab.c`: tick-level platform button/anticipation/spawn/resize
  phases and transformed-player following.
- `recast_ab.c`: two platform casts, including the evicted platform's live
  shrink/removal interval.
- `stock_respawn_ab.c`: stock consumption, directional point selection, early
  jump confirmation, and the accelerated spawn interval.

See RUST_ROLLOUTS.md "Fidelity gate" for the validated scopes and measured
numeric tolerances. Do not infer fidelity from an all-identical dual-backend
trace unless `bopl_rl.c` includes the library-identity pool fix (`1cd34b2`).
