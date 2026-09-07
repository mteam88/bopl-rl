# Hourly supervision

Matthew authorized Fable to research Bopl throughput overnight, with this Codex
thread watching hourly. Keep it running until Matthew stops it; no invented
morning deadline. Fable owns optimization and has full access and broad freedom.

Run identity lives in `native/artifacts/rl/autoresearch-monitor/config.json`.
Fable runs in the dedicated `throughput-autoresearch` Rift. Its required flags
are `--model claude-fable-5-1 --effort high --dangerously-skip-permissions`.
The official Ralph Loop plugin repeats at turn completion without a limit.

At each check, inspect `claude agents --json`, the session's recent conversation
and subprocesses, `results.tsv`, and the evidence behind new retained results.
Distinguish a long benchmark/build from an actual stall. If recovery is needed,
inspect first and avoid duplicate workers. `claude --bg --resume <session-id>`
can resume a stopped background conversation; include the model, effort,
permission and Ralph plugin flags and load the private `.env` as run.sh does.
Never restart a session Matthew deliberately stopped. Preserve artifacts and
do not reset or edit Fable's experiments while it works.

The graph reads `native/rl/autoresearch/results.tsv` in Fable's Rift. If Fable
changes its format, adapt the reader or coordinate the format; do not silently
lose measurements. Distinct workloads get distinct plots. The setup smoke is
not the baseline. Invalid and unconfirmed outcomes must not inflate the retained
curve. Record monitoring findings in the monitor artifact directory.

`hourly.sh` runs as a BB script automation, updates the exported plot and queues
a check into the owner thread. The live page updates independently every 15s:
https://powerspec-g728.tailb55536.ts.net:8876/autoresearch/

Monitor service: `bopl-autoresearch-monitor.service` on 127.0.0.1:8877, routed
through the existing tailnet-only HTTPS listener on 8876 at `/autoresearch`.
When asked to stop everything, stop Fable, pause the named BB automation, and
preserve the results and chart. Stopping research should not remove its history.
