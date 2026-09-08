# Day-1 checkpoint — 2026-09-08 20:18 UTC

Commit: `2e2dca7`

## Build

- **pass** (12 warning lines)

## Oracle replay (Unity replayed.jsonl.gz vs port trace, presentation excluded)

| level | ticks | result | detail |
|---|---|---|---|
| 0 | 480 | diverged at 0/480 | `/value/players/0/Player.playerDied` (players) |
| 1 | 444 | diverged at 0/444 | `/value/players/0/Player.playerDied` (players) |
| 2 | 480 | diverged at 0/480 | `/value/players/0/Player.playerDied` (players) |
| 3 | 480 | diverged at 0/480 | `/value/players/0/Player.playerDied` (players) |
| 4 | 418 | diverged at 0/418 | `/value/players/0/Player.playerDied` (players) |

## Throughput (untraced)

- Port: **1342 ticks/s** wall (2400 ticks in 1.79s, includes dotnet run + world load)
- Port self-reported: ticks=2400/2400 elapsed=0.203s ticks/s=11832 simulationTicks=2400 gameEnded=False
- Unity baseline (sim/Oracle/README.md): ~5,000 ticks/s marginal per process
  (0.2 ms/tick mono), 40-50 ticks/s with tracing on; boot ~9 s per process.

## Shim-thread footnote: the single divergence is one export gap

Every level diverges at tick 0 on the same two fields —
`Player.playerDied`/`playerWon` C# event subscriber lists, whose sole
subscriber (`HandleAbilitySelectUI`) lives in the unexported
`WinnerTriangle.unity` additive scene (export in progress on the oracle
thread). With exactly those two fields filtered from both traces, all five
fixtures match in full, including the early-terminating episodes (player
deaths) on l1/l4:

| level | filtered result |
|---|---|
| 0 | match: 480 ticks |
| 1 | match: 444 ticks |
| 2 | match: 480 ticks |
| 3 | match: 480 ticks |
| 4 | match: 418 ticks |

(Filter: drop `Player.playerDied`/`Player.playerWon` everywhere; diff
otherwise identical to the table above, `--ignore-presentation`.)
