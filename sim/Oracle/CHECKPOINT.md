# Day-1 checkpoint — 2026-09-08 20:47 UTC

Commit: `31a5e67`

## Build

- **pass** (0 warning lines)

## Oracle replay (Unity replayed.jsonl.gz vs port trace, presentation excluded)

| level | ticks | result | detail |
|---|---|---|---|
| 0 | 480 | **match** | match: 480 ticks |
| 1 | 444 | **match** | match: 444 ticks |
| 2 | 480 | **match** | match: 480 ticks |
| 3 | 480 | **match** | match: 480 ticks |
| 4 | 418 | **match** | match: 418 ticks |

## Throughput (untraced)

- Port: **1282 ticks/s** wall (2400 ticks in 1.87s, includes dotnet run + world load)
- Port self-reported: ticks=2400/2400 elapsed=0.200s ticks/s=11975 simulationTicks=2400 gameEnded=False
- Unity baseline (sim/Oracle/README.md): ~5,000 ticks/s marginal per process
  (0.2 ms/tick mono), 40-50 ticks/s with tracing on; boot ~9 s per process.

