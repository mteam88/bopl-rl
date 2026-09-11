# Hourly supervision

Matthew authorized Fable to research Bopl throughput overnight, with this Codex
thread watching hourly. Keep it running until Matthew stops it; no invented
morning deadline. Fable owns optimization and has full access and broad freedom.

Matthew challenged the initial benchmark's representativeness: it trains only
map 0 from scratch, with ~41s startup versus ~74s active training. Its baseline
is 3965 learner/s active versus 2502 including startup. The earlier 4008/s was
aggregate across two concurrent eight-worker jobs, not this single 16-worker
workload. A supervisor message asks Fable to redesign toward sustained,
representative training and log `learner_dps_active` as well as wall throughput.
Do not celebrate startup-only gains as equivalent to faster long training.

BB terminal `term_avddk96bb9` attaches to the live Claude session for supervision.
`bb terminal send --enter` sends LF: a pasted message may need a separate CR
(`--text $'\r'`) to submit. Verify receipt in the actual conversation, not merely
the terminal-send acknowledgment. Never interrupt a useful benchmark casually.

Run identity lives in `native/artifacts/rl/autoresearch-monitor/config.json`.
Fable runs in the dedicated `throughput-autoresearch` Rift. Its required flags
are `--model claude-fable-5-1 --effort high --dangerously-skip-permissions`.
The official Ralph Loop plugin repeats at turn completion without a limit.

At each check, inspect `claude agents --json`, the session's recent conversation
and subprocesses, `results.tsv`, and the evidence behind new retained results.
Distinguish a long benchmark/build from an actual stall. If recovery is needed,
inspect first and avoid duplicate workers. `claude --bg --resume <session-id>`
can resume a stopped background conversation. For an existing background session,
DO NOT supply option overrides: Claude Code starts a copy when flags are supplied.
It restores saved model/effort/permissions/plugin options itself; verify `/proc`
after resuming. Load the private `.env` as run.sh does.
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

On 2026-09-07 at 04:53:35 UTC systemd-oomd killed peace.service and its 62
processes, including Fable and exp01 after two complete runs. Recovered the same
session at 05:07:47 UTC in the separate user service bopl-throughput-fable.service
with MemoryHigh=12G, MemoryMax=16G, MemorySwapMax=2G. Keep research outside
peace.service. The recovery launcher is in the monitor artifacts as
resume-fable.sh. Its RemainAfterExit=yes unit holds the background descendants;
active(exited) is normal, so inspect actual Claude and benchmark PIDs.
Matthew explicitly corrected the supervisor: do not prescribe short initial
tests, experiment cadence, repeat counts, or a screening workflow. Fable must
use its own judgment under the original autoresearch program. Earlier pacing
guidance was retracted. Supervise availability and report facts; leave research
design, duration, validation, and workflow to Fable.

Fable raised the research unit limits at runtime for exp03: current MemoryHigh
15G and MemoryMax 18G (MemorySwapMax remains 2G), verified 2026-09-07 08:46 UTC.
The original 12G/16G settings above describe recovery history. Preserve Fable’s
current settings when recovering; check live properties before acting. The
7331 memory.high events came from the first exp03 attempt, logged invalid;
there were no additional events in its completed six-run rerun.

Latest live limits, verified 2026-09-07 09:46 UTC: Fable raised MemoryHigh to
20G and MemoryMax to 24G after exp04 hit the former 15G high limit. Exp05
records zero memory.high events in all six runs. Preserve the live settings.
