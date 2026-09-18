#!/usr/bin/env bash
# Canonical throughput benchmark at the overnight-4 training shape (THROUGHPUT.md),
# with a policy v4 arm (THROUGHPUT_V4.md). Runs train.py for a fixed number of
# updates on the Rust sim, in its own systemd scope (never inside the bb/peace
# cgroup), serialized across rifts with a flock so two benches never share the
# GPU, and prints one JSON line: median sps / learner_sps, the update's
# rollout / train / wall seconds, per-buffer-step ms for rollout, eval_gpu,
# eval_env, and (BOPL_BENCH_PROF=1, the default) the finer split the native
# trainer prints on stderr: the env pool's c_step mean, env CPU per
# buffer-step, straggler and launch time, the c_step phases per env kind
# (begin / sim / sample / finish) and the stream-event split of eval_gpu
# (h2d / forward / d2h). Steady state = the updates after the first 20.
#
# Usage: bench_shape.sh <label> [updates=300] [-- extra train.py args...]
#   BOPL_BENCH_WORKERS=128 BOPL_BENCH_BUFFERS=4 BOPL_BENCH_PASSES=4 override the shape.
#   BOPL_BENCH_ENCODER=set|v4 (default set = the v3 set encoder at hidden 128).
#     v4 is policy v4 at the production shape (sweep/configs/v4-aim.toml:
#     hidden 256, layout.V4_DEFAULTS, writer bits BOPL_BENCH_V4_ABLATE=8,
#     aim Fourier 4); BOPL_BENCH_HIDDEN overrides the width of either arm.
#   BOPL_BENCH_PROF=1 sets BOPL_ENV_PROF=2 BOPL_GPU_PROF=1 for the run (both
#     arms of an A/B share it; 0 for rows comparable to THROUGHPUT.md's tables).
#   OUT=<dir> (default native/artifacts/rl/bench-shape in this checkout).
# Results: $OUT/<label>/ (run dir, log) and $OUT/results.jsonl (appended).
#
# A/B mode: BOPL_BENCH_AB="KEY=VALUE KEY=VALUE" [BOPL_BENCH_AB_ARGS="--flag v"]
#   [BOPL_BENCH_REPEATS=3] bench_shape.sh <label> [updates] [-- args]
#   runs the incumbent (the environment as is) and the candidate (the
#   incumbent plus the BOPL_BENCH_AB assignments and BOPL_BENCH_AB_ARGS
#   train.py flags) interleaved, alternating which goes first per repeat, each
#   as a normal run (<label>-inc-<i>, <label>-cand-<i> in results.jsonl), then
#   prints one summary line: per arm the median and [min, max] of sps,
#   learner_sps, rollout_s, train_s, wall_s over the repeats, and the
#   candidate/incumbent ratio of the medians. Load1 travels with every row.
set -euo pipefail
LABEL=${1:?label}; shift
UPDATES=${1:-300}; [ $# -gt 0 ] && shift
[ "${1:-}" = "--" ] && shift
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
OUT=${OUT:-$R/native/artifacts/rl/bench-shape}
mkdir -p "$OUT"
WORKERS=${BOPL_BENCH_WORKERS:-128}
BUFFERS=${BOPL_BENCH_BUFFERS:-4}
PASSES=${BOPL_BENCH_PASSES:-4}
ENCODER=${BOPL_BENCH_ENCODER:-set}
PROF=${BOPL_BENCH_PROF:-1}
HORIZON=128
STEPS=$((UPDATES * WORKERS * 3 / 2 * HORIZON))  # learner rows x horizon per update
PUF=$(ls -d "$R"/native/artifacts/rl/dependencies/PufferLib-* | sort | tail -1)
[ -f native/target/release/libbopl_sim.so ] || { echo "libbopl_sim.so missing: cargo build --release in native/" >&2; exit 1; }
ulimit -c 0
case "$ENCODER" in
  set) ENCODER_ARGS=(--encoder set --hidden "${BOPL_BENCH_HIDDEN:-128}") ;;
  v4)  ENCODER_ARGS=(--encoder v4 --hidden "${BOPL_BENCH_HIDDEN:-256}" --v4-ablate "${BOPL_BENCH_V4_ABLATE:-8}" --aim-fourier 4) ;;
  *) echo "BOPL_BENCH_ENCODER must be set or v4" >&2; exit 2 ;;
esac
PROF_ENV=()
[ "$PROF" = 1 ] && PROF_ENV=(BOPL_ENV_PROF=2 BOPL_GPU_PROF=1)

# run_one <label> <"KEY=VALUE ..." extra env> [extra train.py args...]: one
# benchmark run; prints its results.jsonl row on stdout (appended to the file too).
run_one() {
  local label=$1 ab_env=$2; shift 2
  local dir="$OUT/$label" unit rc
  rm -rf "$dir"
  unit="bopl-bench-$(echo "$label" | tr -c 'a-zA-Z0-9\n' '-')-$$"
  echo "$(date -u '+%F %T UTC') $label: waiting for GPU bench lock" >&2
  exec 9>/tmp/bopl-gpu-bench.lock
  flock 9
  echo "$(date -u '+%F %T UTC') $label: running $UPDATES updates ($ENCODER w$WORKERS b$BUFFERS p$PASSES${ab_env:+, $ab_env}), load $(cut -d' ' -f1-3 /proc/loadavg)" >&2
  set +e
  # shellcheck disable=SC2086
  env BOPL_RL_BACKEND=rust PYTHONPATH="$PUF" \
    BOPL_OMP_THREADS=${BOPL_OMP_THREADS:-12} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive} "${PROF_ENV[@]}" $ab_env \
  systemd-run --user --scope --collect --quiet -p MemoryMax=20G --unit "$unit" -- \
    nice -n 5 native/artifacts/rl/venv/bin/python native/rl/puffer/train.py \
    --output "$dir" --banks 4 --buffers "$BUFFERS" --checkpoint-every 1000000000 --cudagraphs 1 \
    "${ENCODER_ARGS[@]}" --entropy 0.008 --entropy-floor 0.001 --entropy-free-rows 1 \
    --entropy-half-life 200000000 --entropy-head-scale 0.15,1.5,1 --episode-ticks 7200 \
    --no-eval --horizon "$HORIZON" --layers 1 --learner-passes "$PASSES" \
    --learning-rate 0.00125 --reward-kill 0.27 --reward-timeout -1 \
    --scripted-bots evasive-v2 --scripted-level hard \
    --scripted-opponents mixed-hard,mixed-v2-hard,archer-v2-hard,pursuer-v2-hard,evasive-v2-hard \
    --seed 73 --steps "$STEPS" --training-maps 0,1,2 --workers "$WORKERS" "${EXTRA_ARGS[@]}" "$@" > "$dir.log" 2>&1
  rc=$?
  set -e
  flock -u 9
  if [ $rc -ne 0 ]; then
    echo "{\"label\": \"$label\", \"failed\": true, \"rc\": $rc}" | tee -a "$OUT/results.jsonl"
    tail -n 30 "$dir.log" >&2
    return $rc
  fi
  python3 - "$dir" "$label" "$WORKERS" "$BUFFERS" "$PASSES" "$HORIZON" "$ENCODER" "$ab_env" <<'PY' | tee -a "$OUT/results.jsonl"
import json, re, statistics as st, sys, os
d, label, workers, buffers, passes, horizon, encoder, ab_env = sys.argv[1:]
rows = [json.loads(l) for l in open(os.path.join(d, 'metrics.jsonl'))]
rows = [r for r in rows if 'native_perf' in r][20:]
H = int(horizon)
med = lambda key: round(st.median(r[key] for r in rows), 4)
per = {k: round(st.median(r['native_perf'][k] for r in rows) * 1000 / H, 4)
       for k in rows[0]['native_perf'] if k in ('rollout', 'eval_gpu', 'eval_env') and rows[0]['native_perf'][k] is not None}
# With BOPL_ASYNC (default) rollout_s and train_s overlap; wall_s is the update's
# wall time (sps = physical decisions / wall_s), so rollout_s + train_s != wall_s.
out = dict(label=label, encoder=encoder, workers=int(workers), buffers=int(buffers), passes=int(passes), updates=len(rows),
           sps=round(st.median(r['sps'] for r in rows)), learner_sps=round(st.median(r['learner_sps'] for r in rows)),
           rollout_s=med('rollout_seconds'), train_s=med('train_seconds'),
           wall_s=round(st.median(r.get('wall_seconds', r['rollout_seconds'] + r['train_seconds']) for r in rows), 4),
           overlap=rows[-1].get('overlap'),
           ms_per_step=per, cuda_mb=rows[-1].get('cuda_memory_mb'), load1=round(os.getloadavg()[0], 1))
if ab_env:
    out['ab_env'] = ab_env
if rows[-1].get('set_overflow') is not None:   # policy v4: rows the packed set buffers dropped (players, slots, entities, slimes)
    out['set_overflow_max'] = [max(r['set_overflow'][i] for r in rows if r.get('set_overflow')) for i in range(4)]
for k in ('explained_variance', 'entropy', 'kl', 'gradient_norm'):
    if k in rows[-1]:
        out[k] = round(rows[-1][k], 4)
# The native trainer's finer split (BOPL_ENV_PROF=2 BOPL_GPU_PROF=1, printed
# every 50 rollouts to stderr; vecenv.h read_profile). Medians over the prints
# after the first (warm-up); us per env-step / per buffer-step as printed.
log = open(d + '.log', errors='replace').read()
def medians(pattern, names, skip=1):
    hits = [[float(x) for x in m] for m in re.findall(pattern, log)][skip:]
    return {n: round(st.median(h[i] for h in hits), 3) for i, n in enumerate(names)} if hits else None
pool = medians(r'bopl env pool: ([\d.]+) us c_step \(mean\), ([\d.]+) ms env CPU per buffer-step, envs owner ([\d.]+)% worker ([\d.]+)% stolen ([\d.]+)%, '
               r'([\d.]+) futex sleeps and ([\d.]+) gpu polls per buffer-step, straggler wait ([\d.]+) us, launch ([\d.]+) us',
               ('c_step_us', 'env_cpu_ms_per_buffer_step', 'owner_pct', 'worker_pct', 'stolen_pct', 'sleeps', 'polls', 'straggler_us', 'launch_us'))
if pool: out['env_pool'] = pool
gpu = medians(r'bopl gpu step \(stream events, us\): h2d ([\d.]+) \+ forward ([\d.]+) \+ d2h ([\d.]+) = ([\d.]+) of eval_gpu ([\d.]+)',
              ('h2d_us', 'forward_us', 'd2h_us', 'sum_us', 'eval_gpu_us'))
if gpu: out['gpu_step'] = gpu
kinds = {}
for kind in ('self-play', 'historical', 'scripted'):
    k = medians(r'bopl c_step %-10s\s*([\d.]+)%% of envs:\s*([\d.]+) us = begin\s*([\d.]+) \+ sim\s*([\d.]+) \+ sample\s*([\d.]+) \+ finish\s*([\d.]+) \(([\d.]+) ticks\)' % re.escape(kind),
                ('share_pct', 'us', 'begin_us', 'sim_us', 'sample_us', 'finish_us', 'ticks'))
    if k: kinds[kind] = k
if kinds: out['c_step'] = kinds
print(json.dumps(out))
PY
}

EXTRA_ARGS=("$@")
if [ -z "${BOPL_BENCH_AB:-}" ]; then
  run_one "$LABEL" ""
  exit
fi

# ---- A/B: interleaved incumbent / candidate repeats ----
REPEATS=${BOPL_BENCH_REPEATS:-3}
# shellcheck disable=SC2206
AB_ARGS=(${BOPL_BENCH_AB_ARGS:-})
ROWS=$(mktemp)
for i in $(seq 1 "$REPEATS"); do
  order="inc cand"; [ $((i % 2)) -eq 0 ] && order="cand inc"
  for arm in $order; do
    if [ "$arm" = inc ]; then run_one "$LABEL-inc-$i" "" >> "$ROWS" || true
    else run_one "$LABEL-cand-$i" "$BOPL_BENCH_AB" "${AB_ARGS[@]}" >> "$ROWS" || true; fi
  done
done
python3 - "$ROWS" "$LABEL" "$BOPL_BENCH_AB" "${BOPL_BENCH_AB_ARGS:-}" <<'PY' | tee -a "$OUT/results.jsonl"
import json, statistics as st, sys, os
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
label, ab_env, ab_args = sys.argv[2:5]
arms = {arm: [r for r in rows if not r.get('failed') and r['label'].startswith(f'{label}-{arm}-')] for arm in ('inc', 'cand')}
keys = ('sps', 'learner_sps', 'rollout_s', 'train_s', 'wall_s')
out = dict(label=label, ab=True, ab_env=ab_env, ab_args=ab_args, repeats={arm: len(v) for arm, v in arms.items()},
           failed=[r['label'] for r in rows if r.get('failed')], load1=round(os.getloadavg()[0], 1))
for arm, v in arms.items():
    if v:
        out[arm] = {k: [round(st.median(r[k] for r in v), 4), round(min(r[k] for r in v), 4), round(max(r[k] for r in v), 4)] for k in keys}
        out[arm]['ms_per_step'] = {p: round(st.median(r['ms_per_step'][p] for r in v if p in r['ms_per_step']), 4)
                                   for p in ('rollout', 'eval_gpu', 'eval_env') if any(p in r['ms_per_step'] for r in v)}
if arms['inc'] and arms['cand']:
    out['cand_over_inc'] = {k: round(out['cand'][k][0] / out['inc'][k][0], 4) for k in keys if out['inc'][k][0]}
    # Does every candidate repeat beat every incumbent repeat (the bar THROUGHPUT.md's kept rows met)?
    out['cand_sps_all_above_inc'] = min(r['sps'] for r in arms['cand']) > max(r['sps'] for r in arms['inc'])
print(json.dumps(out))
PY
rm -f "$ROWS"
