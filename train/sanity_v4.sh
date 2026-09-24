#!/usr/bin/env bash
# Learning gate for the policy v4 throughput campaign, the
# v4 counterpart of the earlier 150M sanity run: the
# v4-lite-aim recipe (train/sweep/configs/v4-lite-aim.toml [fixed], the production
# shape the campaign optimises: encoder v4, hidden 256, set inner 64, no FFN,
# critic 384, writer bits 10, aim Fourier 4, 256 workers x 4 buffers, the b4
# recipe of 2026-09-19 (learner_passes 1, minibatch_div 2: two gradient steps
# of 24576 prio-drawn rows per 49152-row update; LR 4.852e-4 = 3.431e-4 x
# sqrt 2), 4 banks with 3 scripted, seed 73) run verbatim for
# its first 160M learner decisions, then read against the reference run
# sweeps/v4-aim/000 (thread policy-v4, 2026-09-16, the same recipe at 1.73B).
#
# The gate: the league rating (league candidate, the sweep scorer's rater, 240
# games) of the run's 160M checkpoint >= the reference's 160M checkpoint's minus
# SPREAD, and the trainer's explained_variance / entropy / kl at 160M within 10%
# of the reference's metrics.jsonl at the same learner step (0.299 / 3.97 /
# 0.0034 at 160M). Both checkpoints are rated on the league as it is now, so the
# two numbers share one scale. SPREAD is 75 Elo, the same-shape seed spread
# measured on the old frozen pool; re-measure it on the league before reading a
# near miss as a failure. A candidate that trades exactness for speed (a
# "learning gate" lever) must pass this before it is kept; an exact lever
# (bit-identical rows) needs only the CPU-side row check (train/check_dump.py)
# and the throughput A/B.
#
#   bash train/sanity_v4.sh run <label> [-- extra train.py args]
#       trains into native/artifacts/rl/sanity/<label> (log <label>.log next to
#       it), in its own systemd scope under the GPU bench flock (bench_shape.sh
#       shares it), nice 5. About 16 min at the reference's 167k learner/s plus
#       four evaluations; the v4 trainer takes ~6 GB of device memory.
#   bash train/sanity_v4.sh report <label>
#       eval/readout.py of the run against the v4-aim reference (league
#       ratings, wins on training vs held-out maps, value-head explained
#       variance, death taxonomy) followed by the gate verdict as one JSON line
#       (also appended to sanity/results.jsonl).
#   STEPS=160000000 SEED=73 override the budget / seed.
#   REFERENCE=<sweep run dir> REF_NAME=<readout name> read against another
#   reference of the same recipe (sweeps/v4-lite/000 for the v4-lite shape).
#   Shape / recipe flags go after -- and override the defaults (e.g.
#   -- --learner-passes 2 --minibatch-div 1 --workers 128 --learning-rate 0.0003431012662975937 for the pre-b4 p2 recipe).
#   WALL_SECONDS=<s> (report): the equal-wall-clock comparison for recipe
#   changes: the run's evaluation checkpoint whose trainer elapsed is nearest
#   that many seconds (v4-aim/000 reached 160M learner steps in 955 s; the run
#   needs STEPS >= 520M to pass 955 s at the b4 rate) is rated next to the
#   reference's 160M checkpoint. Informational: the pass/fail verdict stays the
#   equal-steps gate.
set -euo pipefail
MODE=${1:?run|report}; LABEL=${2:?label}; shift 2
[ "${1:-}" = "--" ] && shift
R=$(cd "$(dirname "$0")/.." && pwd)
cd "$R"
OUT=${OUT:-$R/native/artifacts/rl/sanity}
DIR="$OUT/$LABEL"
REFERENCE=${REFERENCE:-$R/native/artifacts/rl/sweeps/v4-aim/000}
REF_NAME=${REF_NAME:-v4-aim}
STEPS=${STEPS:-160000000}
EVAL_EVERY=${EVAL_EVERY:-40000000}
SEED=${SEED:-73}
PUF=$(ls -d "$R"/native/artifacts/rl/dependencies/PufferLib-* | sort | tail -1)
PY=native/artifacts/rl/venv/bin/python

case "$MODE" in
run)
  [ -f native/target/release/libbopl_sim.so ] || { echo "libbopl_sim.so missing: cargo build --release in native/" >&2; exit 1; }
  mkdir -p "$OUT"; rm -rf "$DIR" "$DIR.log"
  ulimit -c 0
  echo "$(date -u '+%F %T UTC') $LABEL: waiting for GPU bench lock" >&2
  exec 9>/tmp/bopl-gpu-bench.lock
  flock 9
  echo "$(date -u '+%F %T UTC') $LABEL: v4-aim recipe to $STEPS learner steps, eval every $EVAL_EVERY, seed $SEED, load $(cut -d' ' -f1-3 /proc/loadavg)" >&2
  set +e
  # train/sweep/configs/v4-lite-aim.toml [fixed], verbatim except steps / eval_every / seed
  # (the driver passes the toml keys as flags).
  env BOPL_RL_BACKEND=rust PYTHONPATH="$PUF" \
    BOPL_OMP_THREADS=${BOPL_OMP_THREADS:-12} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive} BOPL_ENV_THREADS=${BOPL_ENV_THREADS:-8} \
  systemd-run --user --scope --collect --quiet -p MemoryMax=10G --unit "bopl-sanity-$(echo "$LABEL" | tr -c 'a-zA-Z0-9\n' '-')-$$" -- \
    nice -n 5 "$PY" train/train.py --output "$DIR" \
    --aim-fourier 4 --banks 4 --buffers 4 --checkpoint-every 20000000 --cudagraphs 1 \
    --encoder v4 --hidden 256 --layers 1 --v4-ablate 10 --v4-set-inner 64 --v4-ffn-dim 0 --v4-critic-dim 384 \
    --entropy 0.01532588085540654 --entropy-floor 0.001 --entropy-free-rows 1 --entropy-half-life 718611646 \
    --entropy-head-scale 0.028821467398601395,1.5,1 --episode-ticks 7200 \
    --eval-every "$EVAL_EVERY" --evaluation-games 8 --evaluation-maps 0,1,2 --evaluation-workers 4 \
    --heldout-games 4 --heldout-maps 3,4 --horizon 128 --learner-passes 1 --minibatch-div 2 \
    --learning-rate 0.0004852184640654399 --reward-kill 0.2883693237829292 --reward-timeout -1 \
    --scripted-banks 3 --scripted-opponents mixed-hard,mixed-v2-hard,archer-v2-hard,pursuer-v2-hard,evasive-v2-hard \
    --scripted-roster default --scripted-share-max 1 --scripted-share-min 0.11048204896277423 \
    --seed "$SEED" --steps "$STEPS" --training-maps 0,1,2 --workers 256 "$@" > "$DIR.log" 2>&1
  rc=$?
  set -e
  echo "exit $rc" >> "$DIR.log"
  echo "$(date -u '+%F %T UTC') $LABEL: exit $rc" >&2
  exit $rc
  ;;
report)
  [ -d "$DIR" ] || { echo "no run at $DIR" >&2; exit 1; }
  BOPL_RL_BACKEND=rust "$PY" eval/readout.py "$DIR" --name "$LABEL" --baseline "$REF_NAME=$REFERENCE" --json "$DIR/readout.json" || true
  # League ratings of the 160M checkpoints (the sweep scorer's rater, CPU only,
  # ~10 min per checkpoint) for the run and the reference, and of the checkpoint
  # nearest WALL_SECONDS. Each lands in its run's league/ directory and is reused.
  nearest() {  # the checkpoint at physical step $2 in run $1, or the nearest within 1%
    ls "$1"/checkpoints/policy-*.bin 2>/dev/null | python3 -c '
import sys, re
step, files = int(sys.argv[1]), [p.strip() for p in sys.stdin if p.strip()]
at = lambda p: int(re.search(r"policy-(\d+)", p).group(1))
best = min(files, key=lambda p: abs(at(p) - step), default="")
print(best if best and abs(at(best) - step) <= step // 100 else "")' "$2"
  }
  rate() {
    [ -n "$1" ] && BOPL_RL_BACKEND=rust "$PY" league candidate "$1" --games "${BAND_GAMES:-240}" \
      --workers "${BAND_WORKERS:-3}" > /dev/null || echo "league rating of $1 failed" >&2
  }
  run_ck=$(nearest "$DIR" 213352448); ref_ck=$(nearest "$REFERENCE" 213352448)
  rate "$run_ck"; rate "$ref_ck"
  wall_ck=
  if [ -n "${WALL_SECONDS:-}" ]; then
    wall_step=$(python3 - "$DIR" "$WALL_SECONDS" <<'PY'
import json, sys, glob, os
d, wall = sys.argv[1], float(sys.argv[2])
def metrics_open(d):  # plain, or packed by eval/compact_history.py
    p = os.path.join(d, 'metrics.jsonl')
    return os.popen(f"zstd -dcq '{p}.zst'") if not os.path.exists(p) and os.path.exists(p + '.zst') else open(p)
evals = {int(os.path.basename(p)) for p in glob.glob(os.path.join(d, 'evaluations', '*')) if os.path.basename(p).isdigit()}
best = None
for line in metrics_open(d):
    r = json.loads(line)
    if r['step'] in evals and (best is None or abs(r['elapsed'] - wall) < abs(best[1] - wall)): best = (r['step'], r['elapsed'])
print(best[0] if best else '')
PY
)
    [ -n "$wall_step" ] && wall_ck="$DIR/checkpoints/policy-$(printf '%09d' "$wall_step").bin" && rate "$wall_ck"
  fi
  python3 - "$DIR" "$LABEL" "$REFERENCE" "$run_ck" "$ref_ck" "$wall_ck" "${WALL_SECONDS:-}" <<'PY' | tee -a "$OUT/results.jsonl"
import json, os, sys
d, label, reference, run_ck, ref_ck, wall_ck, wall_seconds = sys.argv[1:]
sys.path.insert(0, 'league')
import candidate
def metrics_open(d):  # plain, or packed by eval/compact_history.py
    p = os.path.join(d, 'metrics.jsonl')
    return os.popen(f"zstd -dcq '{p}.zst'") if not os.path.exists(p) and os.path.exists(p + '.zst') else open(p)
SPREAD = 75            # same-shape seed spread, measured on the old frozen pool
LEARNER = 160_000_000
ratings = candidate.reference()
def rated(checkpoint):
    r = candidate.load(candidate.result_path(checkpoint), ratings) if checkpoint else None
    return r and dict(checkpoint=os.path.basename(checkpoint), rating=round(r['rating'], 1),
                      interval=[round(x, 1) for x in r['interval']], games=r['games'])
def metrics_at(run, learner_step):
    best = None
    for line in metrics_open(run):
        r = json.loads(line)
        if r.get('learner_steps', 0) <= learner_step: best = r
    return best or {}
out = dict(label=label, reference=os.path.relpath(reference, os.path.dirname(os.path.dirname(d))), spread=SPREAD)
run, ref = rated(run_ck), rated(ref_ck)
out['league_160M'] = dict(run=run, reference=ref)
out['passed'] = bool(run and ref and run['rating'] >= ref['rating'] - SPREAD)
if ref:
    out['league_160M']['floor'] = round(ref['rating'] - SPREAD, 1)
m, ref_m = metrics_at(d, LEARNER), metrics_at(reference, LEARNER)
for k in ('explained_variance', 'entropy', 'kl', 'learner_sps', 'sps'):
    if k in m and k in ref_m:
        out[k] = dict(run=round(m[k], 4), reference=round(ref_m[k], 4), ratio=round(m[k] / ref_m[k], 3) if ref_m[k] else None)
        if k in ('explained_variance', 'entropy', 'kl') and ref_m[k] and abs(m[k] / ref_m[k] - 1) > 0.10:
            out.setdefault('drift', []).append(k)
if wall_seconds:
    out['wall'] = dict(target_s=float(wall_seconds), league=rated(wall_ck), reference_160M=ref)
out['load1'] = round(os.getloadavg()[0], 1)
print(json.dumps(out))
PY
  ;;
*) echo "mode must be run or report" >&2; exit 2 ;;
esac
