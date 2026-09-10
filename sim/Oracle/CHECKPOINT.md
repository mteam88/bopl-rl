# Day-1 checkpoint — 2026-09-09 17:08 UTC

Commit: `fd9d301`

## Build

- **pass** (4 warning lines)

## Oracle replay (Unity replayed.jsonl.gz vs port trace, presentation + known export gaps excluded)

| fixture | ticks | result | detail |
|---|---|---|---|
| l0-p2 | 480 | **match** | match: 480 ticks |
| l0-p2-g0 | 509 | **match** | match: 509 ticks |
| l0-p2-g1 | 720 | **match** | match: 720 ticks |
| l0-p2-g2 | 720 | **match** | match: 720 ticks |
| l0-p2-g3 | 720 | **match** | match: 720 ticks |
| l0-p2-g4 | 720 | **match** | match: 720 ticks |
| l0-p3-g0 | 557 | **match** | match: 557 ticks |
| l1-p2 | 444 | **match** | match: 444 ticks |
| l1-p2-g4 | 720 | **match** | match: 720 ticks |
| l2-p2 | 480 | **match** | match: 480 ticks |
| l2-p2-g4 | 720 | **match** | match: 720 ticks |
| l3-p2 | 480 | **match** | match: 480 ticks |
| l3-p2-g4 | 720 | **match** | match: 720 ticks |
| l4-p2 | 418 | **match** | match: 418 ticks |
| l4-p2-g4 | 720 | **match** | match: 720 ticks |

## Throughput (untraced)

- Port: **981 ticks/s** wall (2400 ticks in 2.45s, includes dotnet run + world load)
- Port self-reported: ticks=2400/2400 elapsed=0.393s ticks/s=6110 simulationTicks=2400 gameEnded=False
- Unity baseline (sim/Oracle/README.md): ~5,000 ticks/s marginal per process
  (0.2 ms/tick mono), 40-50 ticks/s with tracing on; boot ~9 s per process.

