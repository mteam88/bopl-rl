#!/usr/bin/env bash
# GPU checks for native/rl/research/TRAINER-BUGS-2026-10.md that the CPU tests
# (train/test_trainer_fixes.py) cannot make. Run on an idle GPU from a checkout
# whose binding is current (train/build.py --device cuda):
#
#   bash train/research/check_trainer_fixes_gpu.sh [out-dir]
#
# 1. check_v4.py: the row-packing cases, `overflow` (the default budget drops
#    rows in batch order) and `full_rows` / `full_rows_caps` (--v4-full-rows:
#    nothing dropped, the other half of a batch bit-identical when one half
#    gets crowded).
# 2. Resume continuity of the LR cosine and the prioritized-replay beta: a
#    10-update run against the same run split 5 + 5 by --resume, once with the
#    native schedules (expected to DIFFER after the resume: bugs 1 and 4) and
#    once with --anneal-lr-fix 1 --prio-beta-fix 1 --schedule-steps <10 updates>
#    (expected to match exactly; without --schedule-steps the first leg's
#    --steps would set a shorter horizon, the rewind schedule_rewinds warns of).
#    About 2 minutes at 16 workers.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$ROOT/native/artifacts/rl/trainer-fixes-check}
PY=$ROOT/native/artifacts/rl/venv/bin/python
export PYTHONPATH=$ROOT/native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9
export BOPL_RL_BACKEND=rust
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$ROOT"

echo '== check_v4 (row packing)'
"$PY" train/check_v4.py --output "$OUT/check_v4" | tail -c 2000
"$PY" - "$OUT/check_v4/result.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r['pass_all'], 'check_v4 failed'
assert r['full_rows']['other_half_identical'] and r['full_rows_caps']['other_half_identical']
print('check_v4: pass_all, full_rows other half identical')
EOF

BATCH=3072   # 16 workers, historical-every 2: 24 learner rows x horizon 128
# The research arms' policy (arm.py): v4, 256 wide, aim K=4. The default linear encoder can't
# checkpoint (its native input is OBS_STRIDE wide, the writer expects layout 3's OBS_SIZE).
COMMON=(--workers 16 --opponents self --no-eval --no-in-run-rating --anneal-lr 1 --min-lr-ratio 0.1
        --checkpoint-every $BATCH --seed 7 --encoder v4 --hidden 256 --aim-fourier 4
        --v4-critic-dim 384 --v4-ffn-dim 0 --v4-set-inner 64)
run() {   # run <dir> <extra args...>: 10 updates straight, and 5 + --resume 5
    local dir=$1; shift
    "$PY" train/train.py --output "$OUT/$dir-full" --steps $((10 * BATCH)) "${COMMON[@]}" "$@" > "$OUT/$dir-full.log" 2>&1
    "$PY" train/train.py --output "$OUT/$dir-split" --steps $((5 * BATCH)) "${COMMON[@]}" "$@" > "$OUT/$dir-split.log" 2>&1
    "$PY" train/train.py --output "$OUT/$dir-split" --resume --steps $((10 * BATCH)) "${COMMON[@]}" "$@" >> "$OUT/$dir-split.log" 2>&1
}
echo '== resume continuity (native schedules)'
run native
echo '== resume continuity (--anneal-lr-fix 1 --prio-beta-fix 1 --schedule-steps)'
run fixed --anneal-lr-fix 1 --prio-beta-fix 1 --schedule-steps $((10 * BATCH))

"$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
def schedule(run):
    rows = [json.loads(line) for line in (out / run / 'metrics.jsonl').open()]
    return {r['learner_steps']: (r['learning_rate'], r.get('prio_beta')) for r in rows}
for arm, expect_equal in (('native', False), ('fixed', True)):
    full, split = schedule(f'{arm}-full'), schedule(f'{arm}-split')
    assert set(full) == set(split), (arm, sorted(full), sorted(split))
    differ = [s for s in sorted(full) if full[s] != split[s]]
    print(f'{arm}: {len(differ)} of {len(full)} updates differ after the split' + (f' (first at {differ[0]})' if differ else ''))
    for s in sorted(full):
        print(f'  {s:>8} full lr {full[s][0]:.6g} beta {full[s][1]:.4f} | split lr {split[s][0]:.6g} beta {split[s][1]:.4f}')
    assert (not differ) == expect_equal, f'{arm}: expected {"identical" if expect_equal else "different"} schedules'
print('resume continuity: native restarts (bugs 1, 4 reproduced), fixed matches')
EOF
