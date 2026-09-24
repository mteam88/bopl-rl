#!/usr/bin/env bash
# Learning gate for the policy v4 throughput campaign, the
# v4 counterpart of the earlier 150M sanity run: the
# v4-lite-aim recipe (train/sweep/configs/v4-lite-aim.toml [fixed], the production
# shape the campaign optimises: encoder v4, hidden 256, set inner 64, no FFN,
# critic 384, writer bits 10, aim Fourier 4, 256 workers x 4 buffers, the b4
# recipe of 2026-09-19 (learner_passes 1, minibatch_div 2: two gradient steps
# of 24576 prio-drawn rows per 49152-row update; LR 4.852e-4 = 3.431e-4 x
# sqrt 2), 4 banks with 3 scripted, seed 73) run verbatim for
# its first 160M learner decisions with the in-run evaluation on pool v7, then
# read against the reference run sweeps/v4-aim/000 (thread policy-v4, 2026-09-16,
# the same recipe at 1.73B) at the evaluation steps both share.
#
# The reference's pool-v7 Elo at the two evaluations the gate reads (readout.py's
# refit of the stored in-run games, 48 games each; the driver's 160-game
# in-run records in reference/d48f4b64/ say 399 [352, 447] and 570 [528, 611]):
#   80M learner (physical step 106692608)   379  [245, 516]
#   160M learner (physical step 213352448)  646  [504, 898]
# Same-shape pool-v7 seed spread is ~75 Elo, so the
# gate is: refit Elo at 80M >= 304 and at 160M >= 571 (reference minus the
# spread), the band rating (240 games, the sweep scorer's rater) of the 160M
# checkpoint >= the reference's 160M band rating minus 75, and the trainer's
# explained_variance / entropy / kl at 160M within 10% of the reference's
# metrics.jsonl at the same learner step (0.299 / 3.97 / 0.0034 at 160M).
# The in-run intervals are wide (+-135); the band rating is the number that
# decides, the in-run curve the early warning. A candidate that trades
# exactness for speed (a "learning gate" lever) must pass
# this before it is kept; an exact lever (bit-identical rows) needs only the
# CPU-side row check (train/check_dump.py) and the throughput A/B.
#
#   bash train/sanity_v4.sh run <label> [-- extra train.py args]
#       trains into native/artifacts/rl/sanity/<label> (log <label>.log next to
#       it), in its own systemd scope under the GPU bench flock (bench_shape.sh
#       shares it), nice 5. About 16 min at the reference's 167k learner/s plus
#       four evaluations; the v4 trainer takes ~6 GB of device memory.
#   bash train/sanity_v4.sh report <label>
#       eval/readout.py of the run against the v4-aim reference (in-run
#       Elo curve at the shared evaluation steps, wins on training vs held-out
#       maps, value-head explained variance, death taxonomy) followed by the
#       gate verdict as one JSON line (also appended to sanity/results.jsonl).
#   STEPS=160000000 EVAL_EVERY=40000000 SEED=73 override the budget / cadence / seed;
#   the 80M and 160M evaluations must stay on the cadence for the gate to read.
#   REFERENCE=<sweep run dir> REF_NAME=<readout name> read against another
#   reference of the same recipe (sweeps/v4-lite/000 for the v4-lite shape).
#   Shape / recipe flags go after -- and override the defaults (e.g.
#   -- --learner-passes 2 --minibatch-div 1 --workers 128 --learning-rate 0.0003431012662975937 for the pre-b4 p2 recipe).
#   WALL_SECONDS=<s> (report): the equal-wall-clock comparison for recipe
#   changes (the campaign's recipe gate: band rating at 955 s against the
#   aim-only p2 run's 572.8 [529, 617] and the b4 pair's 612.0 / 579.2; the
#   run needs STEPS >= 520M to pass 955 s at the b4 rate): the run's
#   evaluation whose trainer elapsed is
#   nearest that many seconds (v4-aim/000 reached 160M learner steps in 955 s)
#   is read out (refit Elo, band rating of its checkpoint into band/wall) next
#   to the reference's 160M numbers; a run must have been given enough STEPS
#   (a multiple of EVAL_EVERY) to pass that wall time. Informational: the
#   pass/fail verdict stays the equal-steps gate.
set -euo pipefail
MODE=${1:?run|report}; LABEL=${2:?label}; shift 2
[ "${1:-}" = "--" ] && shift
R=$(cd "$(dirname "$0")/.." && pwd)
cd "$R"
OUT=${OUT:-$R/native/artifacts/rl/sanity}
DIR="$OUT/$LABEL"
REFERENCE=${REFERENCE:-$R/native/artifacts/rl/sweeps/v4-aim/000}
REF_NAME=${REF_NAME:-v4-aim}
POOL=${POOL:-$R/native/artifacts/rl/sweeps/anchors/pool-v7/pool.json}
STEPS=${STEPS:-160000000}
EVAL_EVERY=${EVAL_EVERY:-40000000}
SEED=${SEED:-73}
PUF=$(ls -d "$R"/native/artifacts/rl/dependencies/PufferLib-* | sort | tail -1)
PY=native/artifacts/rl/venv/bin/python

case "$MODE" in
run)
  [ -f native/target/release/libbopl_sim.so ] || { echo "libbopl_sim.so missing: cargo build --release in native/" >&2; exit 1; }
  [ -f "$POOL" ] || { echo "pool v7 missing at $POOL" >&2; exit 1; }
  mkdir -p "$OUT"; rm -rf "$DIR" "$DIR.log"
  ulimit -c 0
  echo "$(date -u '+%F %T UTC') $LABEL: waiting for GPU bench lock" >&2
  exec 9>/tmp/bopl-gpu-bench.lock
  flock 9
  echo "$(date -u '+%F %T UTC') $LABEL: v4-aim recipe to $STEPS learner steps, eval every $EVAL_EVERY, seed $SEED, load $(cut -d' ' -f1-3 /proc/loadavg)" >&2
  set +e
  # train/sweep/configs/v4-lite-aim.toml [fixed], verbatim except steps / eval_every / seed
  # (the driver passes the toml keys as flags; evaluation_pool = [scoring].pool).
  env BOPL_RL_BACKEND=rust PYTHONPATH="$PUF" \
    BOPL_OMP_THREADS=${BOPL_OMP_THREADS:-12} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive} BOPL_ENV_THREADS=${BOPL_ENV_THREADS:-8} \
  systemd-run --user --scope --collect --quiet -p MemoryMax=10G --unit "bopl-sanity-$(echo "$LABEL" | tr -c 'a-zA-Z0-9\n' '-')-$$" -- \
    nice -n 5 "$PY" train/train.py --output "$DIR" \
    --aim-fourier 4 --banks 4 --buffers 4 --checkpoint-every 20000000 --cudagraphs 1 \
    --encoder v4 --hidden 256 --layers 1 --v4-ablate 10 --v4-set-inner 64 --v4-ffn-dim 0 --v4-critic-dim 384 \
    --entropy 0.01532588085540654 --entropy-floor 0.001 --entropy-free-rows 1 --entropy-half-life 718611646 \
    --entropy-head-scale 0.028821467398601395,1.5,1 --episode-ticks 7200 \
    --eval-every "$EVAL_EVERY" --evaluation-games 8 --evaluation-maps 0,1,2 --evaluation-pool "$POOL" --evaluation-workers 4 \
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
  # Band rating of the 160M checkpoints (the sweep scorer's rater: 240 games
  # on 15-85% matchups, CPU only, ~10 min per checkpoint at 3 workers) for the
  # run and, once, for the reference; the in-run 48-game refit's interval is
  # +-135 Elo, too wide to see the 75 spread on its own. BAND_GAMES=0 skips it.
  if [ "${BAND_GAMES:-240}" -gt 0 ]; then
    for side in run reference; do
      case $side in run) src=$DIR;; reference) src=$REFERENCE;; esac
      # the 160M checkpoint, or the run's nearest within 1% (a w256 run's updates are 65536 physical steps and land one update off)
      ck=$(ls "$src"/checkpoints/policy-*.bin 2>/dev/null | python3 -c '
import sys, re
files = [p.strip() for p in sys.stdin if p.strip()]
step = lambda p: int(re.search(r"policy-(\d+)", p).group(1))
best = min(files, key=lambda p: abs(step(p) - 213352448), default="")
print(best if best and abs(step(best) - 213352448) <= 2133524 else "")')
      if [ -n "$ck" ] && [ -f "$ck" ] && ! ls "$DIR/band/$side"/*.json >/dev/null 2>&1; then
        mkdir -p "$DIR/band/$side"
        BOPL_RL_BACKEND=rust "$PY" train/sweep/reference_pool.py rate --pool "$POOL" --checkpoint "$ck" \
          --out "$DIR/band/$side" --games "${BAND_GAMES:-240}" --workers "${BAND_WORKERS:-3}" > "$DIR/band/$side.log" 2>&1 || echo "band rating of $side failed: $(tail -1 "$DIR/band/$side.log")" >&2
      fi
    done
  fi
  # Equal wall-clock: rate the checkpoint of the evaluation nearest WALL_SECONDS.
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
    ck="$DIR/checkpoints/policy-$(printf '%09d' "${wall_step:-0}").bin"
    if [ -n "$wall_step" ] && [ -f "$ck" ] && [ "${BAND_GAMES:-240}" -gt 0 ] && ! ls "$DIR"/band/wall/*.json >/dev/null 2>&1; then
      mkdir -p "$DIR/band/wall"
      BOPL_RL_BACKEND=rust "$PY" train/sweep/reference_pool.py rate --pool "$POOL" --checkpoint "$ck" \
        --out "$DIR/band/wall" --games "${BAND_GAMES:-240}" --workers "${BAND_WORKERS:-3}" > "$DIR/band/wall.log" 2>&1 || echo "band rating of the wall checkpoint failed: $(tail -1 "$DIR/band/wall.log")" >&2
    fi
  fi
  python3 - "$DIR" "$LABEL" "$REFERENCE" "$REF_NAME" "${WALL_SECONDS:-}" <<'PY' | tee -a "$OUT/results.jsonl"
import glob, json, os, re, sys
d, label, reference, ref_name, wall_seconds = sys.argv[1:]
def metrics_open(d):  # plain, or packed by eval/compact_history.py
    p = os.path.join(d, 'metrics.jsonl')
    return os.popen(f"zstd -dcq '{p}.zst'") if not os.path.exists(p) and os.path.exists(p + '.zst') else open(p)
SPREAD = 75            # same-shape pool-v7 seed spread
# The evaluation steps the gate reads, in the trainer's physical step count
# (checkpoint names): 80M and 160M learner decisions of the v4-aim recipe.
GATE_STEPS = {80_000_000: 106_692_608, 160_000_000: 213_352_448}
def band(side, step=213352448):
    # the rating stored for `step`, or for the nearest rated checkpoint within 1% of it
    files = glob.glob(os.path.join(d, 'band', side, '*policy-*.json'))
    def step_of(f):
        m = re.search(r'policy-(\d+)', os.path.basename(f)); return int(m.group(1)) if m else -1
    files = [f for f in files if abs(step_of(f) - step) <= 0.01 * step]
    if not files: return None
    r = json.load(open(min(files, key=lambda f: abs(step_of(f) - step))))
    return dict(rating=round(r['rating'], 1), interval=[round(x, 1) for x in r['interval']] if r.get('interval') else None, games=r.get('games'))
def refit_elo(name):
    """physical step -> (rating, interval): readout.py's pool-v7 refit of the
    stored evaluation games (elo_refit), else the eval hook's logged elo."""
    try:
        r = json.load(open(os.path.join(d, 'readout.json')))[name]
    except (FileNotFoundError, KeyError):
        return {}
    curve = {int(s): (e['rating'], e.get('interval')) for s, e in r.get('elo_refit', {}).items()}
    return curve or {int(s): (e['elo'] if isinstance(e, dict) else e, e.get('interval') if isinstance(e, dict) else None) for s, e in r.get('elo', {}).items()}
def metrics_at(run, learner_step):
    best = None
    for line in metrics_open(run):
        r = json.loads(line)
        if r.get('learner_steps', 0) <= learner_step: best = r
    return best or {}
def nearest(curve, step, tolerance=0.1):
    hits = [s for s in curve if abs(s - step) <= tolerance * step]
    return min(hits, key=lambda s: abs(s - step)) if hits else None
run_elo, ref_elo = refit_elo(label), refit_elo(ref_name)
out = dict(label=label, reference=os.path.relpath(reference, os.path.dirname(os.path.dirname(d))), spread=SPREAD, gate={}, passed=True)
for learner, step in GATE_STEPS.items():
    rs, fs = nearest(run_elo, step), nearest(ref_elo, step)
    if rs is None or fs is None:
        out['gate'][str(learner)] = dict(missing=True, run_steps=sorted(run_elo), reference_steps=sorted(ref_elo)); out['passed'] = False; continue
    elo, interval = run_elo[rs]; ref, ref_interval = ref_elo[fs]
    ok = elo >= ref - SPREAD
    out['gate'][str(learner)] = dict(run_step=rs, elo=round(elo, 1), interval=[round(x, 1) for x in interval] if interval else None,
                                     reference_step=fs, reference_elo=round(ref, 1), floor=round(ref - SPREAD, 1), ok=ok)
    out['passed'] &= ok
run_band, ref_band = band('run'), band('reference')
if run_band and ref_band:
    ok = run_band['rating'] >= ref_band['rating'] - SPREAD
    out['band_160M'] = dict(run=run_band, reference=ref_band, floor=round(ref_band['rating'] - SPREAD, 1), ok=ok)
    out['passed'] &= ok
elif run_band or ref_band:
    out['band_160M'] = dict(run=run_band, reference=ref_band, missing=True)
last = max(GATE_STEPS)
m, ref_m = metrics_at(d, last), metrics_at(reference, last)
for k in ('explained_variance', 'entropy', 'kl', 'learner_sps', 'sps'):
    if k in m and k in ref_m:
        out[k] = dict(run=round(m[k], 4), reference=round(ref_m[k], 4), ratio=round(m[k] / ref_m[k], 3) if ref_m[k] else None)
        if k in ('explained_variance', 'entropy', 'kl') and ref_m[k] and abs(m[k] / ref_m[k] - 1) > 0.10:
            out.setdefault('drift', []).append(k)
if wall_seconds:
    wall = float(wall_seconds)
    evals = {int(os.path.basename(p)) for p in glob.glob(os.path.join(d, 'evaluations', '*')) if os.path.basename(p).isdigit()}
    rows = [json.loads(l) for l in metrics_open(d)]
    at = min((r for r in rows if r['step'] in evals), key=lambda r: abs(r['elapsed'] - wall), default=None)
    if at:
        elo = run_elo.get(at['step'])
        out['wall'] = dict(target_s=wall, run_step=at['step'], learner_steps=at['learner_steps'], elapsed_s=round(at['elapsed']),
                           elo=round(elo[0], 1) if elo else None, interval=[round(x, 1) for x in elo[1]] if elo and elo[1] else None,
                           band=band('wall', at['step']), reference_160M=dict(elo=out['gate'].get('160000000', {}).get('reference_elo'), band=ref_band))
out['load1'] = round(os.getloadavg()[0], 1)
print(json.dumps(out))
PY
  ;;
*) echo "mode must be run or report" >&2; exit 2 ;;
esac
