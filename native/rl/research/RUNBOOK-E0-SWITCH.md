# Runbook: switch E0's evals to the fast path, then launch anneal-ctl and off1 (2026-10-19)

T = /home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject/tp-spec. E0 (unit `research-e0`)
runs T's arm.py, which is 14 commits behind main. Its trainers are fine, but its evals use the old
batch-4 path at about 4.9 CPU-s a game under load (13 in the Fable review). The new path (1f8c9741,
b9f56a5d) takes 0.52 CPU-s a game and gives the same rows: 40/40 games of l2-s301 anneal vs final-5,
new == old == E0's stored rows. At 19:16 only l2-s301 had all its evals, and E0's evaluator
can't catch up before training ends.

## Why the switch is safe

- `--eval-only` never trains. It reads each stage's `status.json`, skips arms whose stages aren't
  complete, and logs them as `skipped`.
- Done evals are skipped: arm.py checks each eval's `.jsonl`, and a killed partial eval is rerun
  from scratch (its command starts with `rm -f`).
- The seeds are unchanged. Endpoint e against opponent slot o plays seed base 211001 +
  50000·(8·o + e), and the old slots 0-7 keep their numbers (checked: obsfix-s301 ema/final-5 is
  261001 and l2-s302 raw/g111 is 1561001 under both codes).
- New in e0.toml are four pinned on-kit opponents in slots 8-11 (x2-drill, x2-missile,
  x2-blinkplat, l2-me32-trap, on their own kits, 400 games, annealed endpoint only). They run as
  missing evals.
- New rows also carry `credited` (v4.2 kill credit). Rows from before the switch don't have it,
  so `kh_pre_cast` is reported only where every row has it. The primary KH_PRE uses game credit.

## 1. Wait for E0 training to finish

Training uses T's binding and must finish under the old code. The interleave order ends with
l2-s305 and l2-s306, so the last unit is `research-e0-l2-s306-anneal` (expected around 10-21).

```bash
T=/home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject/tp-spec
systemctl --user list-units 'research-e0-*-constant.service' 'research-e0-*-anneal.service' --no-legend
for f in $T/native/artifacts/rl/research/e0/*/anneal/status.json; do
  echo "$f $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' $f)"; done
```

Go on when the first command prints nothing and all 10 arms print `complete`.

## 2. Stop E0's evaluator

`research-e0`'s children are separate transient units, and stopping the parent leaves them
running. Stop both:

```bash
systemctl --user stop research-e0
systemctl --user stop 'research-e0-*-eval-*'
systemctl --user list-units 'research-e0*' --no-legend   # must be empty
```

## 3. Update T and rebuild the eval library

```bash
cd $T && git status --short          # only gate-v7d/ is untracked; nothing else may be dirty
git pull --ff-only origin main
PY=$T/native/artifacts/rl/venv/bin/python
$PY eval/native_policy.py --build    # arm.py's preflight refuses a stale library
```

## 4. Relaunch E0 evals only

Same unit and command as before, plus the two flags. `--eval-jobs 2` runs two 6-worker
evals, which uses all 12 threads. Use `--eval-jobs 1` if step 6 trains at the same time.

```bash
systemd-run --user --unit=research-e0 -p WorkingDirectory=$T \
  $PY -u train/research/arm.py train/research/arms/e0.toml --eval-only --eval-jobs 2
journalctl --user -u research-e0 -f
```

Remaining work: about 2.9 CPU-h an arm, or about 4.5 h wall for 9 arms at 12 threads.

## 5. Analyze E0

```bash
$PY train/research/analyze.py train/research/arms/e0.toml
```

The pre-registered E0 rule is unchanged. The new section `human regime at anneal` gives KH_PRE,
the sudden-death rate, the pre-SD score, and the panel score with SD games counted as draws.
`exploit panel` now includes the on-kit rows.

## 6. Rebuild the trainer binding, check it, then launch anneal-ctl and off1

The pulled train.py and PufferLib patch (TRAINER-BUGS-2026-10.md) need a fresh binding. Run the
GPU check while the GPU is idle (about 2 min):

```bash
$PY train/build.py --device cuda
bash train/research/check_trainer_fixes_gpu.sh
$PY train/research/arm.py train/research/arms/anneal-ctl.toml --dry-run | tail -3
$PY train/research/arm.py train/research/arms/off1.toml --dry-run | tail -3
systemd-run --user --unit=research-offense -p WorkingDirectory=$T \
  $PY -u train/research/arm.py train/research/arms/anneal-ctl.toml train/research/arms/off1.toml
```

anneal-ctl goes first because it is short (9 runs, 12B, about 5.2 GPU-h, plus 7.7 CPU-h of eval).
It starts from E0's l2 6B constant checkpoints, which are fresh runs and not resumes, so the
resume bugs don't apply. off1 is 32 runs of 6B constant + 1B anneal, about 96 GPU-h at
0.43 GPU-h/B, with about 67 CPU-h of eval that overlaps training. Analyze each with
`analyze.py <toml>`. The rules are in PLAN-2026-10.md, "2026-10-19 pre-registration: off1 and
anneal-ctl".

If E0's evals are still running, keep them at `--eval-jobs 1` (6 threads), since the trainer
needs the rest.
