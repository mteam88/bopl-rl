# dash-perf

Branch `dash-perf` off main `31f7d75`. Live dashboard on 8876 stays untouched; local server on **8890**.

## Findings (2026-09-12)
- `/api/ladder` 14s was `run_summary` × ~107 runs × `rated_curve` → `scoring.evaluation_checkpoints` parsing every eval `result.json` (overnight-5 alone: 618 files / 231 MB), then `run_iqm` did it again. `load_pool` verify and `rate_human` were fine (5 ms / 37 ms).
- `/api/run` shipped `metrics.jsonl` via the 2400-row store: overnight-5-013 is 95 MB (0.8 s); long-run-1b-007 is 1.4 GB. UI never charts those rows.
- Compacted 20 evals were 1.23 MB JSON; latest-only is ~70–126 KB.

## Done
- `rated_curve` reads rating files + sidecar `learner_steps` only (no eval JSON, no bootstrap refit).
- `serve.py`: per-run summary cache keyed by file stamps; background thread rebuilds ladder/sweep; GET serves the last good snapshot instantly; ETags; `/api/run` first paint without metrics.jsonl; lazy `/api/run/<name>/metrics` and `/evaluation/<dir>`.
- `app.js` stale-while-revalidate + lazy eval fetch.
- Contract + API test updated.

## Latency (8890, after change)

| endpoint | cold process | warm | uncompressed | gzip |
|---|---|---|---|---|
| GET /api/ladder | 1.26 s (start→200) | 0.7–1.4 ms | 183 KB | 40 KB |
| GET /api/sweep/overnight-5 | (warm after listen) 2.1 ms | 1.1 ms | 71 KB | 24 KB |
| GET /api/run/overnight-5-013 | 19 ms first hit | 1.0 ms | 70 KB | 12 KB |
| GET /api/run/long-run-1b-007 | 2 ms (cached after UI) | 1.8 ms | 26 KB | 6 KB |
| GET /api/run/overnight-5-013/metrics | 989 ms first parse of 95 MB jsonl | 0.7 ms | 265 KB | 85 KB |

Before (base, warm process): ladder 14.5 s then 21 s (178 KB); sweep 3.5 s (68 KB); run 12–13 MB / ~1 s.

## Verify
- `check_dashboard_api.py` PASS
- research + sweep unit tests OK
- `node check_dashboard.mjs http://127.0.0.1:8890/` → DASHBOARD_OK

## Serve invocation
Same flags as `~/.local/share/bopl-null-hypothesis/start-dashboard-8876.sh`, port **8890**, `--root`/`--pool`/`--run-root*` pointed at the **base** checkout artifacts (`/home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl`). `serve.py` from this rift. See `/tmp/dash-perf-prof/start-8890.sh`.

## Left
- Metrics endpoint still full-scans jsonl on first fetch (lazy, not first paint).
- Smoke test opened an unrated standalone run so `#sweepNav` stayed hidden (`sweepView: null`); ladder/run/mobile still DASHBOARD_OK.
