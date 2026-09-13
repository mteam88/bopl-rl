# Elo roster

Date: 2026-09-13. Branch `elo-roster` (WIP commit 56fc4b2 + this). Rift child only.

## Done

- Rift venv finder now points at this rift's PufferLib; `_C` rebuilt `--binding-only`; `libbopl_policy_eval.so` rebuilt (stale copy was missing `bopl_policy_eval_create_aim`).
- `--scripted-decay` removed. New flags: `--scripted-roster` (`default` = 14), `--scripted-share-min` (0), `--scripted-share-max` (1).
- C: process-wide roster weights, per-episode draw from `bot_rng`, per-entry decided-game logs.
- Persist `pool/roster.json`. Metrics: `scripted_share`, `scripted_learner_elo`, per-entry `{bank, elo, games, winrate, weight, window_games, window_wins}`.
- overnight-7.toml: `scripted_roster = "default"`, params `scripted_share_max` 0.02–0.5 (default 0.15) and `scripted_share_min` 0.005–0.1 (default 0.02). Driver `--dry-run 2` validates.
- Tests: sweep unittest 80 OK; research 22 OK; `test_native_selfplay` 5 OK; `check_tick_step --scripted mixed-v2-hard` and `search-hard` TICK_STEP_OK; idle `check_scripted_bots` OK (search skipped: that process resolved the Unity game).

## sps (w128 b4 p4 aim_fourier 4, share forced to 1 so the bot actually plays)

| label | median learner_sps | elapsed 20M |
| --- | --- | --- |
| evasive-v2-hard | 1,096,508 | 19.7 s |
| default 14-entry roster | 177,738 | 113.7 s |

Roster is **−83.8%** learner_sps (far past the ~5% ask) because search-hard/medium are rollout stragglers (~1.2 ms/decision vs ~0.2 ms). Do not silently drop them: proposed mitigation is to make the default training roster the 12 v2 levels only and keep `search-*` as eval opponents / an explicit `--scripted-roster` opt-in. A `--scripted-search-weight` floor would be another guessed constant.

At the real curriculum (`share_min=0`, `share_max=1`) 40M median learner_sps was ~230k–350k: share started at 0.27 (mixed-v2-easy is only ~450 Elo, I≈0.26) and rose to 0.56 as the learner climbed to 230 Elo, so search still shows up enough to hurt.

## 40M smoke (`sweeps/smoke-roster/40m`, overnight-7 fixed + share_max 1, no-eval)

Share 0.27 → 0.56 (the opposite of time decay: more bot games as the learner approaches the roster). Learner WR vs easy v2 ~12–17% by 30–40M, vs evasive-v2-hard still ~4–12%. Pick mass at 0–10M is easy v2 (11–12%); by 30–40M the hard/search entries have a larger weight share.

## Resume

`--resume` restores `pool/roster.json` and continues counts (10M 2517 decided games → first post-resume window 2517+ε). Uninterrupted 20M vs 10M+10M is **not** bit-identical even with `BOPL_TF32=0` (CUDA training). Report that as GPU nondeterminism, not a roster bug.

## Left for the parent

- Finalise overnight-7: 7 Protein params now; `scripted_share_min` max (0.1) overlaps `scripted_share_max` min (0.02), so Protein can suggest min>max and train.py will reject the trial.
- Decide whether default roster keeps search.
- Historical overnight-3..6 TOML still contain `scripted_decay` and will not load.
- No `bopl-*` units other than `bopl-roster-*` were touched; those scopes have exited.

## Candidate memories

- Rift-copied venv `__editable___pufferlib_*_finder.py` hardcodes the source checkout; sed to the rift and rebuild `_C` or training silently uses the stale binding (scripted WR ~50%).
- Search bots in a shared env pool dominate rollout wall as stragglers even at ~5–12% of scripted games.
- PufferLib pool Elo is one K-update per log interval on the window rate, not per game; roster matches that.
