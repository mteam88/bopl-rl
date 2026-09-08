# Day-1 checkpoint — 2026-09-08 20:49 UTC

Commit: `7fdf96e`

## Build

- **pass** (0 warning lines)

## Oracle replay (Unity replayed.jsonl.gz vs port trace, presentation + known export gaps excluded)

| fixture | ticks | result | detail |
|---|---|---|---|
| l0-p2 | 480 | **match** | match: 480 ticks |
| l0-p2-g0 | 509 | **match** | match: 509 ticks |
| l0-p2-g1 | 720 | **match** | match: 720 ticks |
| l0-p2-g2 | 720 | diverged at 279/720 | `/value/updatables/105/state/SimpleSparkNode.origColor/Color.r/$f32` (updatables) |
| l0-p2-g3 | 720 | diverged at 372/720 | `/value/transforms/8/state/FixTransform.position/Vec2.x/$fix` (transforms) |
| l0-p2-g4 | 720 | **replay crashed** |    at BoplSim.Host.Program.Main(String[] args) in /home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject/dotne |
| l0-p3-g0 | 557 | **match** | match: 557 ticks |
| l1-p2 | 444 | **match** | match: 444 ticks |
| l2-p2 | 480 | **match** | match: 480 ticks |
| l3-p2 | 480 | **match** | match: 480 ticks |
| l4-p2 | 418 | **match** | match: 418 ticks |

## Throughput (untraced)

- Port: **1388 ticks/s** wall (2400 ticks in 1.73s, includes dotnet run + world load)
- Port self-reported: ticks=2400/2400 elapsed=0.206s ticks/s=11634 simulationTicks=2400 gameEnded=False
- Unity baseline (sim/Oracle/README.md): ~5,000 ticks/s marginal per process
  (0.2 ms/tick mono), 40-50 ticks/s with tracing on; boot ~9 s per process.

