# PGO profile corpus

`corpus.tar.xz` holds 40 BOPLREP2 games: every fifth of games 0-199 of
`native/artifacts/rl/env-corpus` (eval/perf/record_corpus.py, 2026-10-06).
They are generalist-111's final checkpoint against itself, on Matthew's 15
kits and the 111 training maps, with 3-minute games and delay 9.
`tools/pgo_build.sh` re-steps them through an instrumented sim (`pgo-train`)
and builds the release library with the profile.

Held-out games 200-399 measure 22% fewer sim µs per decision with the
identical state hash (docs/THROUGHPUT_V5.md). A stale profile costs speed,
never correctness: re-record it after large sim changes.
