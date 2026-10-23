#!/bin/bash
# Stage 2d "overfit one matchup" (native/rl/research/EXPLOITER-GAPS-2026-10.md): fine-tune the l2 main that
# l2-me31-drill-grenade-2000m was trained against, only against that frozen exploiter, for 1B learner steps.
# The arms differ only in the main's kit:
#   mirror  main on the exploiter's own kit (Dash/Drill/Grenade): a best response scores >= 0.5 by symmetry
#   iid     main on its league draw (--loadout-pool all), exploiter pinned: the league's own matchup
#   counter main on kits holding Gold mushroom, Rock or Meteor (its best abilities vs me31) + 2 random
#   check   iid at learning rate ~0 for 4M steps: the GPU ledger of base vs me31 matches the CPU eval (0.246 vs
#           0.261, done 2026-10-19, doc Section 1d)
# Flags = the me-31 exploiter's command (logs/me-31.log) with the roles swapped; no style, main's timeout reward;
# observation v4.2 (--v4-ablate 16, the program base since 2026-10-21; PLAN-2026-10.md 2026-10-22).
# mirror and iid train through train/research/arm.py train/research/arms/x2d-drill.toml (same flags, same run
# layout $OUT/<arm>/constant); `train` here is the reference command and the only way to run check and counter.
#   train/research/arms/x2d-drill.sh train check|mirror|iid|counter     (GPU, one at a time)
#   train/research/arms/x2d-drill.sh eval  check|mirror|iid|counter     (CPU, after train)
# The eval's base is the arm's own step-0 checkpoint: l2-main-125336m's weights under the v4.2 writer.
set -euo pipefail
T=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$T/native/artifacts/rl/venv/bin/python
L=$T/native/artifacts/rl/league-loops/l2
OUT=$T/native/artifacts/rl/research/x2d-drill
INIT=$L/main/checkpoints/policy-125336092672.bin          # l2-main-125336m, me31's target
EXPL=l2-me31-drill-grenade-2000m
EXPL_BIN=$L/me-31/checkpoints/policy-2000027648.bin
arm=$2
steps=1000000000; lr=0.0004852184640654399
case $arm in
  mirror) kits=(--loadout-learner-first --loadout-set native/data/lineups-2d-drill-mirror.txt); seed=5201
          sched=$T/native/data/lineups-2d-drill-mirror.txt ;;
  iid)    kits=(--loadout-pool all --pinned-opponents $EXPL); seed=5203
          sched=$T/native/data/lineups-2d-drill-iid-eval.txt ;;
  check)  kits=(--loadout-pool all --pinned-opponents $EXPL); seed=5205; steps=4000000; lr=1e-12   # train.py needs lr > 0; done 2026-10-19 (doc Section 1d)
          sched=$T/native/data/lineups-2d-drill-iid-eval.txt ;;
  counter) kits=(--loadout-learner-first --loadout-set native/data/lineups-2d-drill-counter.txt); seed=5204
          sched=$T/native/data/lineups-2d-drill-counter.txt ;;
  *) echo "arm: check|mirror|iid|counter"; exit 2 ;;
esac
RUN=$OUT/$arm/constant
mkdir -p $OUT/$arm
if [ "$1" = eval ] && [ $arm = check ]; then   # the readout is the training ledger, not an eval
  exec python3 -c "import json; r=[json.loads(l) for l in open('$RUN/metrics.jsonl')][-1]; print([e for e in r['pool']['entries'] if e['games']])"
fi
if [ "$1" = train ]; then
  cd $T
  exec systemd-run --user --scope --collect -p MemoryHigh=12G --unit x2d-drill-$arm -- \
    env BOPL_RL_BACKEND=rust PYTHONPATH=$T/native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9 $PY $T/train/train.py --output $RUN --steps $steps --checkpoint-every 250000000 \
    --init-weights $INIT --opponents $EXPL --seed $seed "${kits[@]}" \
    --historical-every 1 --bank-groups league,league,league,league --banks 4 --pool-size 2 --pfsp-p 1 \
    --learning-rate $lr --anneal-lr 0 \
    --reward-kill 0.2883693237829292 --reward-timeout -1 --episode-ticks 10800 --training-maps training-111 \
    --aim-fourier 4 --buffers 4 --cudagraphs 1 --encoder v4 --hidden 256 --layers 1 --horizon 128 \
    --entropy 0.0005 --entropy-floor 0.0005 --entropy-free-rows 1 --entropy-half-life 2874446584 \
    --entropy-head-scale 0.028821467398601395,1.5,1 --learner-passes 1 --minibatch-div 2 --mirror-share 0 \
    --v4-ablate 16 --v4-critic-dim 384 --v4-ffn-dim 0 --v4-set-inner 64 --workers 256 --no-eval
fi
# Eval: tuned vs base on the same 600 seed pairs (1200 games each), plus the curve at 250M/500M/750M (200 pairs).
# Outputs $OUT/<arm>/x2d-eval-{final,on-iid,curve}/ (games.jsonl, summary.json); analyze.py x2d-drill reads them.
# exploit_eval.py appends to games.jsonl, so eval starts by deleting this arm's x2d-eval-* directories.
# The mirror arm needs only the mirror schedule; every arm is also scored on the iid schedule (transfer).
ev() { systemd-run --user --wait --collect --unit x2d-eval-$arm-$1 -p MemoryHigh=8G -p CPUWeight=30 \
  --setenv=BOPL_RL_BACKEND=rust --setenv=OMP_NUM_THREADS=1 --working-directory=$T \
  $PY $T/eval/exploit_eval.py "${@:2}" --opponent me31=$EXPL_BIN --workers 6; }
cks() { ls $RUN/checkpoints | grep -E '^policy-[0-9]+\.bin$' | sort -t- -k2 -n | sed "s|^|$RUN/checkpoints/|"; }
[ "$(python3 -c "import json; print(json.load(open('$RUN/status.json'))['status'])")" = complete ] || { echo "$RUN not complete"; exit 1; }
BASE=$(cks | head -1); last=$(cks | tail -1)
rm -rf $OUT/$arm/x2d-eval-final $OUT/$arm/x2d-eval-on-iid $OUT/$arm/x2d-eval-curve
ev final --subject tuned=$last --subject base=$BASE --schedule $sched --label $arm --pairs 600 \
  --seed-base 251001 --out $OUT/$arm/x2d-eval-final
ev iid --subject tuned=$last --subject base=$BASE --schedule $T/native/data/lineups-2d-drill-iid-eval.txt \
  --label iid --pairs 600 --seed-base 261001 --out $OUT/$arm/x2d-eval-on-iid
for step in 250 500 750; do
  ck=$(cks | awk -v s=$step 'NR==s/250+1')   # line 1 is the step-0 init
  ev c$step --subject t$step=$ck --schedule $sched --label $arm --pairs 200 --seed-base 251001 --out $OUT/$arm/x2d-eval-curve
done
