# Day-1 checkpoint — 2026-09-09 22:24 UTC

Commit: `6ccb43b`

## Build

- **pass** (0 warning lines)

## Oracle replay (Unity replayed.jsonl.gz vs port trace, presentation + known export gaps excluded)

| fixture | ticks | result | detail |
|---|---|---|---|
| l0-p2 | 480 | **match** | match: 480 ticks |
| l1-p2 | 444 | **match** | match: 444 ticks |
| l2-p2 | 480 | **match** | match: 480 ticks |
| l3-p2 | 480 | **match** | match: 480 ticks |
| l4-p2 | 418 | **match** | match: 418 ticks |

## Throughput (untraced)

- Port: **896 ticks/s** wall (2400 ticks in 2.68s, includes dotnet run + world load)
- Port self-reported: ticks=2400/2400 elapsed=0.363s ticks/s=6615 simulationTicks=2400 gameEnded=False
- Unity baseline (sim/Oracle/README.md): ~5,000 ticks/s marginal per process
  (0.2 ms/tick mono), 40-50 ticks/s with tracing on; boot ~9 s per process.

