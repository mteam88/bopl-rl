# dash-perf

Branch `dash-perf` off main `31f7d75`. Live dashboard on 8876 stays untouched; local server on **8890**.

## Plan
1. Profile ladder / sweep / run assembly.
2. Stop rebuilding ladder from `evaluations/*/result.json` and `metrics.jsonl`.
3. Cache per (path, mtime, size); background thread keeps ladder/sweep JSON warm.
4. Thin `/api/run` (latest eval only, metrics on a lazy endpoint). Client SWR cache.
5. Verify: check_dashboard_api, unit tests, headless Firefox, latency table.

## Findings (2026-09-12)
- `/api/ladder` 14s was `run_summary` × 107 runs × `rated_curve` → `evaluation_checkpoints` parsing every eval `result.json` (overnight-5 alone: 618 files / 231 MB), then `run_iqm` did it again.
- `load_pool` verify and `rate_human` bootstrap were fine (5 ms / 37 ms).
- `/api/run` shipped `metrics.jsonl` via the 2400-row store: overnight-5-013 is 95 MB (0.8 s parse); long-run-1b-007 is 1.4 GB. Compacted 20 evals were 1.23 MB JSON; latest-only is 126 KB.
- Current UI never charts `metrics`; first paint does not need them.

## Done
- `rated_curve` reads rating files + sidecar `learner_steps` only.
- `serve.py`: per-run summary cache, warm ladder/sweep thread, ETags, `/api/run` without metrics, `/api/run/<name>/metrics` and `/evaluation/<dir>`.
- `app.js` stale-while-revalidate + lazy eval fetch.
- Contract + API test updated.

## Left
- Start 8890, latency table, unit tests, headless Firefox.
- WIP commit.
