#!/usr/bin/env bash
# Learning gate for the policy v4 throughput campaign (THROUGHPUT_V4.md), the
# v4 counterpart of the 150M sanity run in THROUGHPUT.md section 4: the
# v4-aim recipe (sweep/configs/v4-aim.toml [fixed], the production shape the
# campaign optimises: encoder v4, hidden 256, writer bits 8, aim Fourier 4,
# 128 workers x 4 buffers, 4 banks with 3 scripted, seed 73) run verbatim for
# its first 160M learner decisions with the in-run evaluation on pool v7, then
# read against the reference run sweeps/v4-aim/000 (thread policy-v4, 2026-09-16,
# the same recipe at 1.73B) at the evaluation steps both share.
#
# The reference's pool-v7 Elo at the two evaluations the gate reads (readout.py's
# refit of the stored in-run games, 48 games each; the driver's 160-game
# in-run records in reference/d48f4b64/ say 399 [352, 447] and 570 [528, 611]):
#   80M learner (physical step 106692608)   379  [245, 516]
#   160M learner (physical step 213352448)  646  [504, 898]
# Same-shape pool-v7 seed spread is ~75 Elo (POLICY_V4_PLAN.md 7c), so the
# gate is: refit Elo at 80M >= 304 and at 160M >= 571 (reference minus the
# spread), the band rating (240 games, the sweep scorer's rater) of the 160M
# checkpoint >= the reference's 160M band rating minus 75, and the trainer's
# explained_variance / entropy / kl at 160M within 10% of the reference's
# metrics.jsonl at the same learner step (0.299 / 3.97 / 0.0034 at 160M).
# The in-run intervals are wide (+-135); the band rating is the number that
# decides, the in-run curve the early warning. A candidate that trades
# exactness for speed (THROUGHPUT_V4.md's "learning gate" levers) must pass
# this before it is kept; an exact lever (bit-identical rows) needs only the
# CPU-side row check (puffer/check_dump.py) and the throughput A/B.
#
#   bash native/rl/puffer/sanity_v4.sh run <label> [-- extra train.py args]
#       trains into native/artifacts/rl/sanity/<label> (log <label>.log next to
#       it), in its own systemd scope under the GPU bench flock (bench_shape.sh
#       shares it), nice 5. About 16 min at the reference's 167k learner/s plus
#       four evaluations; the v4 trainer takes ~6 GB of device memory.
#   bash native/rl/puffer/sanity_v4.sh report <label>
#       research/readout.py of the run against the v4-aim reference (in-run
#       Elo curve at the shared evaluation steps, wins on training vs held-out
#       maps, value-head explained variance, death taxonomy) followed by the
#       gate verdict as one JSON line (also appended to sanity/results.jsonl).
#   STEPS=160000000 EVAL_EVERY=40000000 SEED=73 override the budget / cadence / seed;
#   the 80M and 160M evaluations must stay on the cadence for the gate to read.
set -euo pipefail
MODE=${1:?run|report}; LABEL=${2:?label}; shift 2
[ "${1:-}" = "--" ] && shift
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
OUT=${OUT:-$R/native/artifacts/rl/sanity}
DIR="$OUT/$LABEL"
REFERENCE=${REFERENCE:-$R/native/artifacts/rl/sweeps/v4-aim/000}
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
  # sweep/configs/v4-aim.toml [fixed], verbatim except steps / eval_every / seed
  # (the driver passes the toml keys as flags; evaluation_pool = [scoring].pool).
  env BOPL_RL_BACKEND=rust PYTHONPATH="$PUF" \
    BOPL_OMP_THREADS=${BOPL_OMP_THREADS:-12} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive} BOPL_ENV_THREADS=${BOPL_ENV_THREADS:-8} \
  systemd-run --user --scope --collect --quiet -p MemoryMax=10G --unit "bopl-sanity-$(echo "$LABEL" | tr -c 'a-zA-Z0-9\n' '-')-$$" -- \
    nice -n 5 "$PY" native/rl/puffer/train.py --output "$DIR" \
    --aim-fourier 4 --banks 4 --buffers 4 --checkpoint-every 20000000 --cudagraphs 1 \
    --encoder v4 --hidden 256 --layers 1 --v4-ablate 8 \
    --entropy 0.01532588085540654 --entropy-floor 0.001 --entropy-free-rows 1 --entropy-half-life 718611646 \
    --entropy-head-scale 0.028821467398601395,1.5,1 --episode-ticks 7200 \
    --eval-every "$EVAL_EVERY" --evaluation-games 8 --evaluation-maps 0,1,2 --evaluation-pool "$POOL" --evaluation-workers 4 \
    --heldout-games 4 --heldout-maps 3,4 --horizon 128 --learner-passes 4 \
    --learning-rate 0.0003431012662975937 --reward-kill 0.2883693237829292 --reward-timeout -1 \
    --scripted-banks 3 --scripted-opponents mixed-hard,mixed-v2-hard,archer-v2-hard,pursuer-v2-hard,evasive-v2-hard \
    --scripted-roster default --scripted-share-max 1 --scripted-share-min 0.11048204896277423 \
    --seed "$SEED" --steps "$STEPS" --training-maps 0,1,2 --workers 128 "$@" > "$DIR.log" 2>&1
  rc=$?
  set -e
  echo "exit $rc" >> "$DIR.log"
  echo "$(date -u '+%F %T UTC') $LABEL: exit $rc" >&2
  exit $rc
  ;;
report)
  [ -d "$DIR" ] || { echo "no run at $DIR" >&2; exit 1; }
  BOPL_RL_BACKEND=rust "$PY" native/rl/research/readout.py "$DIR" --name "$LABEL" --baseline "v4-aim=$REFERENCE" --json "$DIR/readout.json" || true
  # Band rating of the 160M checkpoints (the sweep scorer's rater: 240 games
  # on 15-85% matchups, CPU only, ~10 min per checkpoint at 3 workers) for the
  # run and, once, for the reference; the in-run 48-game refit's interval is
  # +-135 Elo, too wide to see the 75 spread on its own. BAND_GAMES=0 skips it.
  if [ "${BAND_GAMES:-240}" -gt 0 ]; then
    for side in run reference; do
      case $side in run) src=$DIR;; reference) src=$REFERENCE;; esac
      ck="$src/checkpoints/policy-213352448.bin"
      if [ -f "$ck" ] && [ ! -f "$DIR/band/$side/policy-213352448.json" ]; then
        mkdir -p "$DIR/band/$side"
        BOPL_RL_BACKEND=rust "$PY" native/rl/sweep/reference_pool.py rate --pool "$POOL" --checkpoint "$ck" \
          --out "$DIR/band/$side" --games "${BAND_GAMES:-240}" --workers "${BAND_WORKERS:-3}" > "$DIR/band/$side.log" 2>&1 || echo "band rating of $side failed: $(tail -1 "$DIR/band/$side.log")" >&2
      fi
    done
  fi
  python3 - "$DIR" "$LABEL" "$REFERENCE" <<'PY' | tee -a "$OUT/results.jsonl"
import glob, json, os, sys
d, label, reference = sys.argv[1:]
SPREAD = 75            # same-shape pool-v7 seed spread, POLICY_V4_PLAN.md 7c
# The evaluation steps the gate reads, in the trainer's physical step count
# (checkpoint names): 80M and 160M learner decisions of the v4-aim recipe.
GATE_STEPS = {80_000_000: 106_692_608, 160_000_000: 213_352_448}
def band(side):
    files = glob.glob(os.path.join(d, 'band', side, '*policy-213352448*.json'))
    if not files: return None
    r = json.load(open(files[0]))
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
    for line in open(os.path.join(run, 'metrics.jsonl')):
        r = json.loads(line)
        if r.get('learner_steps', 0) <= learner_step: best = r
    return best or {}
def nearest(curve, step, tolerance=0.1):
    hits = [s for s in curve if abs(s - step) <= tolerance * step]
    return min(hits, key=lambda s: abs(s - step)) if hits else None
run_elo, ref_elo = refit_elo(label), refit_elo('v4-aim')
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
out['load1'] = round(os.getloadavg()[0], 1)
print(json.dumps(out))
PY
  ;;
*) echo "mode must be run or report" >&2; exit 2 ;;
esac
