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

All of them printed byte-identical traces on 2026-09-09 (see
RUST_ROLLOUTS.md "Fidelity gate").
