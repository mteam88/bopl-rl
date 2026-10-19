#!/bin/bash
# usage: run.sh UNIT offense_diag args...   (CPU only, capped unit; OFFENSE-2026-10.md section 4)
W=$(cd "$(dirname "$0")/../../../.." && pwd); unit=$1; shift
exec systemd-run --user --unit=offense-$unit --collect -P -p MemoryHigh=8G -p CPUWeight=20 --working-directory=$W \
  --setenv=PYTHONPATH=$W/native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9 \
  --setenv=BOPL_RL_BACKEND=rust --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  $W/native/artifacts/rl/venv/bin/python $W/native/rl/research/offense/offense_diag.py "$@"
