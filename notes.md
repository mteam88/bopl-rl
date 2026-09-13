# Elo roster (WIP)

Date: 2026-09-13. Branch `elo-roster` off `acbeb73`. Rift child only.

## Plan

Replace `--scripted-decay` with an Elo-matchmade roster of `(kind, level)` entries.

- `--scripted-roster` names (`family[-vN]-level`); `default` = 4 v2 kinds × easy/medium/hard + search-medium + search-hard (14).
- `--scripted-bots/--scripted-level` stay as a one-entry (or N-kind) roster.
- Per-episode C draw: `w_i = p_i(1-p_i) + 0.1/n`, `p_i = 1/(1+10^((elo_i-elo_learner)/400))`.
- Share: `clamp(share_min, share_max * I)` with `I = max 4p(1-p)`.
- Entry Elo = `elo_init + pool-v5 rating` (unrated levels: hard − 75 per step). Update from decided games only, pool `update_elo` once per log interval per entry (timeouts excluded). Persist `pool/roster.json`.
- Drop `--scripted-decay` everywhere it is live (train.py, sweep_config TRAIN_FLAGS/ALWAYS_INTEGER, overnight-7, smoke, bench_shape).

# Status (2026-09-13)

- [x] Read SCRIPTED_BOTS.md, train.py, bopl_puffer.h, sweep_config, overnight-7, pool-v5, METRICS_CONTRACT
- [x] Fix rift venv editable finder (points at this rift's PufferLib)
- [x] `scripted_roster.py` + C globals/draw/entry logs + train.py flags/persist/metrics
- [x] sweep_config + overnight-7 params + smoke.toml + bench_shape.sh + docs
- [ ] Rebuild `_C --binding-only`
- [ ] Unit tests, research tests, check_tick_step, resume
- [ ] sps 20M default-roster vs evasive-v2-hard; 40M smoke; driver --dry-run 2

## Findings

- Copied venv finder pointed at `/home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/.../PufferLib-.../pufferlib`; now this rift.
- Pool Elo is one `K * (window_rate - expected)` update per bank per log interval, not per game. Roster matches that (decided games only).
- With overlap, weight push lags one rollout (same as the old decay share).
- `policy_banks = all banks` when `share_min < 1` so leftover matches are real frozen opponents.
- overnight-7 now has 7 Protein params (decay replaced by share_max + share_min). Parent finalises boxes; seed defaults 0.15 / 0.02 sit inside the asked log_normal ranges. Smoke uses train.py `--scripted-share-max 1`.
