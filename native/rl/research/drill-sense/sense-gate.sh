#!/bin/bash
# Gates for layout v4.4, "the line to the opponent" (writer bit 64, BOPL4_ABLATE_LINE), adapted from
# round 1's drill-gate.sh (tp-spec native/artifacts/rl/research/drill-gate.sh). Outputs in
# native/artifacts/rl/research/drill-sense/{pre,post}/ of this checkout.
#
# Every training run here has GEMM autotuning off (BOPL_GEMM_TUNE=0, BOPL_V4_LT_TUNE=0): the trainer picks
# GEMM kernels by timing them, which a shared GPU flips between runs, so with tuning on two identical runs
# differ in the last bits (and then everywhere, at a working learning rate). With it off a run is
# bit-deterministic, and the bit-off training gate can ask for byte-identical checkpoints.
#
#   SENSE_TREE=<checkout of the commit before the sense> sense-gate.sh pre
#                        baselines from the pre-change tree (its own _C and binaries; outputs still go to this
#                        checkout's $G/pre):
#                        - env_bench over the tp-spec corpus (120 games) at writer bits 0, 16 and 48 (48 with
#                          the Drill form export on, as a bits-48 run has it)
#                        - 10 updates of training at the working rate, bits 16 twice (must be byte-identical:
#                          the trainer is deterministic here, or the gate means nothing) and 48; the same
#                          with --dump-rows (the trainer's GPU rows over the first rollout, GPU vs CPU)
#                        - check_style_binding host-step hashes at bits 16 and 48
#   sense-gate.sh post   after the change; rebuilds the binaries and _C, then each check is a hard gate (exit 1):
#                        1. env_bench state / train / telemetry hashes equal to pre at bits 0, 16, 48
#                        2. training with the bit off byte-identical to pre (detcmp.py), bits 16 and 48: every
#                           checkpoint and every metrics.jsonl record (losses, gradient norm, KL, behaviour,
#                           matches; not wall-clock keys) of 10 working-rate updates; the trainer's GPU rows
#                           over the first rollout byte-identical with 0 GPU-vs-CPU transport mismatches; and
#                           at bits 112, 0 transport mismatches
#                        3. check_style_binding hashes equal to pre (bits 16 and 48)
#                        4. CPU == CUDA rows (encode_check, 178 tp-spec Drill-kit replays): export on at bits
#                           0,16,32,48,112, export off at 0,16; the line slot's coverage (underground rows,
#                           opponent behind terrain) must be nonzero at 112; and two negative controls (GPU
#                           line cast without the exit march; GPU samples a half step early) must mismatch,
#                           in the line slot only
#                        5. check_v4.py --build
#                        6. smoke: bits 112 from round 1's SO-s7121 checkpoint (bits 48) against v4.0 league
#                           members (bits 0), a few updates: exits 0, writes a checkpoint
set -uo pipefail
HERE=$(cd "$(dirname "$0")/../../../.." && pwd)
R=${SENSE_TREE:-$HERE}   # the tree whose binaries and trainer run (pre: the pre-change checkout)
T=/home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject/tp-spec   # read only: corpora, the SO init
RL=$R/native/artifacts/rl
PY=$RL/venv/bin/python
G=$HERE/native/artifacts/rl/research/drill-sense   # outputs, always in this checkout
LIB=$R/native/target/release/libbopl_sim.so
EBCORPUS=$T/native/artifacts/rl/tp-spec/corpus
ENCCORPUS=$T/native/artifacts/rl/research/drill-gate/corpus
DET=$HERE/native/rl/research/drill-sense/detcmp.py
SO_INIT=$T/native/artifacts/rl/research/drill/SO-s7121/anneal/checkpoints/policy-666697728.bin
export PYTHONPATH=$RL/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9 BOPL_RL_BACKEND=rust
export BOPL_GEMM_TUNE=0 BOPL_V4_LT_TUNE=0
cd $R || exit 1
say() { echo "[sense-gate $(date '+%F %T')] $*"; }
BATCH=3072
COMMON=(--workers 16 --opponents self --no-eval --no-in-run-rating --anneal-lr 1 --min-lr-ratio 0.1
        --checkpoint-every $BATCH --seed 7 --encoder v4 --hidden 256 --aim-fourier 4
        --v4-critic-dim 384 --v4-ffn-dim 0 --v4-set-inner 64)
trainrun() {   # trainrun <dir> <name> <bits> [train.py flags]: 10 updates; prints the final checkpoint's sha256
  rm -rf $1/$2
  "$PY" train/train.py --output $1/$2 --steps $((10 * BATCH)) "${COMMON[@]}" --v4-ablate $3 "${@:4}" > $1/$2.log 2>&1 || { echo FAIL; return 1; }
  local ck=$(ls $1/$2/checkpoints/ | grep -E '^policy-[0-9]+\.bin$' | sort -t- -k2 -n | tail -1)
  [ -n "$ck" ] || { echo FAIL; return 1; }
  sha256sum $1/$2/checkpoints/$ck | cut -c1-64
}
binding() {   # binding <dir> <name> <bits>: the host-step hashes
  local form=; [ $(( $3 & 32 )) = 0 ] || form=1
  BOPL4_ABLATE=$3 BOPL_RL_FORM_BODY=$form "$PY" train/check_style_binding.py --steps 1500 --json $1/$2.json > $1/$2.log 2>&1 || return 1
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['reward_sha256'], d['obs_sha256'])" $1/$2.json
}
eb() {   # eb <dir> <bits>: env_bench over 120 games; prints the summary's three hashes
  local form=; [ $(( $2 & 32 )) = 0 ] || form=1
  BOPL_ENV_BENCH_ABLATE=$2 BOPL_RL_FORM_BODY=$form $1/env_bench $LIB $EBCORPUS 120 > $1/eb-$2.jsonl 2> $1/eb-$2.err || { echo FAIL; return 1; }
  grep '"summary": true' $1/eb-$2.jsonl | grep -o '"hash": "[0-9a-f]*", "train_hash": "[0-9a-f]*", "telemetry_hash": "[0-9a-f]*"'
}

case "${1:-}" in
pre)
  P=$G/pre; mkdir -p $P
  [ -x $P/env_bench ] || make OUT=$P env-bench > $P/make.log 2>&1 || { say "pre env_bench build failed"; exit 1; }
  for b in 0 16 48; do eb $P $b > $P/eb-$b.txt & done
  a1=$(trainrun $P base16-a 16) || { say "baseline training run failed ($P/base16-a.log)"; exit 1; }
  a2=$(trainrun $P base16-b 16) || { say "baseline training run failed ($P/base16-b.log)"; exit 1; }
  a3=$(trainrun $P base48 48) || { say "baseline training run failed ($P/base48.log)"; exit 1; }
  echo "$a1 $a2 $a3" > $P/base-train.txt
  r=$("$PY" $DET metrics $P/base16-a $P/base16-b) || { say "the trainer is not deterministic here: $r"; exit 1; }
  say "baseline training: bits 16 twice: $r; bits 48 $a3"
  for r in dump16-a:16 dump16-b:16 dump48:48; do
    trainrun $P ${r%:*} ${r#*:} --dump-rows $P/${r%:*}/rows.npz > /dev/null || { say "baseline run ${r%:*} failed"; exit 1; }
  done
  r=$("$PY" $DET rows $P/dump16-a $P/dump16-b) || { say "dump-rows not deterministic: $r"; exit 1; }
  say "baseline dump-rows: bits 16 twice: $r; bits 48: $("$PY" $DET zero $P/dump48)"
  for b in 16 48; do binding $P base-binding-$b $b > $P/base-binding-$b.txt || { say "baseline check_style_binding $b failed"; exit 1; }; done
  say "baseline check_style_binding: 16 $(cat $P/base-binding-16.txt), 48 $(cat $P/base-binding-48.txt)"
  wait
  for b in 0 16 48; do say "baseline env_bench bits $b: $(cat $P/eb-$b.txt)"; done
  ;;
post)
  P=$G/pre; O=$G/post; mkdir -p $O
  make OUT=$O env-bench encode-check encode-check-neg > $O/make.log 2>&1 || { say "build failed ($O/make.log)"; exit 1; }
  "$PY" train/build.py --device cuda > $O/build.log 2>&1 || { say "_C build failed ($O/build.log)"; exit 1; }
  for b in 0 16 48; do eb $O $b > $O/eb-$b.txt & done
  for b in 16 48; do
    pre=base$b; [ $b = 16 ] && pre=base16-a
    trainrun $O new$b $b > /dev/null || { say "2 FAIL: training run failed ($O/new$b.log)"; exit 1; }
    r=$("$PY" $DET metrics $P/$pre $O/new$b) && say "2 PASS: bits $b training byte-identical: $r" || { say "2 FAIL: bits $b training: $r"; exit 1; }
    pre=dump$b; [ $b = 16 ] && pre=dump16-a
    trainrun $O dump$b $b --dump-rows $O/dump$b/rows.npz > /dev/null || { say "2 FAIL: training run failed ($O/dump$b.log)"; exit 1; }
    r=$("$PY" $DET rows $P/$pre $O/dump$b) && say "2 PASS: bits $b trainer rows: $r" || { say "2 FAIL: bits $b trainer rows: $r"; exit 1; }
  done
  trainrun $O dump112 112 --dump-rows $O/dump112/rows.npz > /dev/null || { say "2 FAIL: training run failed ($O/dump112.log)"; exit 1; }
  r=$("$PY" $DET zero $O/dump112) && say "2 PASS: bits 112 trainer rows: $r" || { say "2 FAIL: bits 112 trainer rows: $r"; exit 1; }
  for b in 16 48; do
    h=$(binding $O new-binding-$b $b) || { say "3 FAIL: check_style_binding $b failed"; exit 1; }
    [ "$h" = "$(cat $P/base-binding-$b.txt)" ] && say "3 PASS: host-step hashes at bits $b equal ($h)" \
      || { say "3 FAIL: host-step hashes at bits $b moved: $h vs $(cat $P/base-binding-$b.txt)"; exit 1; }
  done
  wait
  for b in 0 16 48; do
    [ "$(cat $O/eb-$b.txt)" = "$(cat $P/eb-$b.txt)" ] && [ -n "$(cat $O/eb-$b.txt)" ] && say "1 PASS: env_bench bits $b: $(cat $O/eb-$b.txt)" \
      || { say "1 FAIL: env_bench bits $b: $(cat $O/eb-$b.txt) vs pre $(cat $P/eb-$b.txt)"; exit 1; }
  done

  E=$O/encode_check
  BOPL_RL_FORM_BODY=1 $E $LIB $ENCCORPUS 0 0,16,32,48,112 > $O/enc-on.txt 2> $O/enc-on.err \
    && say "4 PASS: CPU == CUDA, export on, bits 0,16,32,48,112: $(head -1 $O/enc-on.txt)" \
    || { say "4 FAIL: export on: $(tail -3 $O/enc-on.txt | tr '\n' ' ') $(head -3 $O/enc-on.err)"; exit 1; }
  cov=$(grep '"line_ablation": 112' $O/enc-on.txt)
  "$PY" -c "import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d['inside']>0 and d['ground']>0 and d['inside_ground']>0 else 1)" "$cov" \
    && say "4 PASS: line-slot coverage at 112: $cov" || { say "4 FAIL: the corpus does not exercise the line slot: $cov"; exit 1; }
  $E $LIB $ENCCORPUS 0 0,16 > $O/enc-off.txt 2> $O/enc-off.err \
    && say "4 PASS: CPU == CUDA, export off, bits 0,16: $(head -1 $O/enc-off.txt)" \
    || { say "4 FAIL: export off: $(tail -3 $O/enc-off.txt | tr '\n' ' ') $(head -3 $O/enc-off.err)"; exit 1; }
  for neg in 1 2; do
    BOPL_RL_FORM_BODY=1 $O/encode_check_neg$neg $LIB $ENCCORPUS 0 112 > $O/enc-neg$neg.txt 2> $O/enc-neg$neg.err \
      && { say "4 FAIL: negative control $neg found no mismatch: the check is vacuous"; exit 1; }
    "$PY" -c "
import json,sys
d=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{\"games\"') or l.startswith('{\"bad_')]
s={k:v for x in d for k,v in x.items()}
ok=s.get('rows_mismatched',0)>0 and s.get('bad_line_slot',0)==s.get('floats_mismatched',-1)
print(('PASS' if ok else 'FAIL'), 'rows_mismatched', s.get('rows_mismatched'), 'floats', s.get('floats_mismatched'), 'in the line slot', s.get('bad_line_slot'))
sys.exit(0 if ok else 1)" $O/enc-neg$neg.txt > $O/enc-neg$neg.verdict
    r=$(cat $O/enc-neg$neg.verdict)
    case "$r" in PASS*) say "4 PASS: negative control $neg mismatches: ${r#PASS }" ;; *) say "4 FAIL: negative control $neg: $r"; exit 1 ;; esac
  done

  "$PY" train/check_v4.py --build > $O/check_v4.log 2>&1 && say "5 PASS: check_v4.py --build (pass_all $(tail -1 $O/check_v4.log | "$PY" -c 'import json,sys; print(json.loads(sys.stdin.read())["pass_all"])' 2>/dev/null))" \
    || { say "5 FAIL: check_v4.py --build (see $O/check_v4.log)"; exit 1; }

  rm -rf $O/smoke112
  "$PY" train/train.py --output $O/smoke112 --init-weights $SO_INIT --steps $((6 * BATCH)) \
      --workers 16 --opponents v4-lite-1920m,v4-lite-2133m --banks 2 --no-eval --no-in-run-rating --checkpoint-every $BATCH \
      --seed 7121 --encoder v4 --hidden 256 --aim-fourier 4 --v4-critic-dim 384 --v4-ffn-dim 0 --v4-set-inner 64 \
      --v4-ablate 112 > $O/smoke112.log 2>&1 || { say "6 FAIL: bits-112 smoke exited nonzero (see $O/smoke112.log)"; exit 1; }
  last=$(ls $O/smoke112/checkpoints/ | grep -E '^policy-0*[1-9][0-9]*\.bin$' | sort | tail -1)
  [ -n "$last" ] || { say "6 FAIL: bits-112 smoke wrote no trained checkpoint"; exit 1; }
  init=$(sha256sum < $SO_INIT | cut -c1-16) first=$(sha256sum < $O/smoke112/checkpoints/policy-000000000.bin | cut -c1-16)
  final=$(sha256sum < $O/smoke112/checkpoints/$last | cut -c1-16)
  shape=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['v4_shape']['ablate'])" $O/smoke112/checkpoints/$last.json)
  [ "$first" = "$init" ] && [ "$final" != "$init" ] && [ "$shape" = 112 ] \
    && say "6 PASS: bits-112 smoke from SO-s7121 (bits 48): step 0 = the init ($init), $last moved ($final), sidecar ablate $shape" \
    || { say "6 FAIL: bits-112 smoke: init $init, step 0 $first, $last $final, sidecar ablate $shape"; exit 1; }
  say "ALL PASS"
  ;;
*) echo "usage: sense-gate.sh pre | post"; exit 2 ;;
esac
