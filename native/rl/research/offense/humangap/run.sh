#!/bin/bash
# HUMAN-GAP-2026-10.md (2026-10-19), rift humangap; capped units, 2 workers each.
# Human data: rsync -a mac:Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl/human/ native/artifacts/rl/human-mac/
W=$(cd "$(dirname "$0")/../../../../.." && pwd); O=$W/native/artifacts/rl/sweeps/humangap; P=$W/native/artifacts/rl/venv/bin/python
T=/home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject/tp-spec/native/artifacts/rl
A=$T/research/e0/l2-s301/anneal/checkpoints/policy-1333395456.bin; F5=$T/sweeps/final-5/run/checkpoints/policy-18666684416.bin
u() { systemd-run --user --unit=humangap-$1 --collect -p MemoryHigh=8G -p CPUWeight=20 --working-directory=$W \
  --setenv=PYTHONPATH=$W/native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9 \
  --setenv=BOPL_RL_BACKEND=rust --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=1 bash -c "$2"; }
mkdir -p $O
u humans "$P eval/human_gap.py --out $O/humans.jsonl --workers 2 > $O/humans.log 2>&1"
u diag-any "$P native/rl/research/offense/offense_diag.py --a $A --b $F5 --kits any --pairs 150 --workers 2 --seed-base 241001 --out $O/s301-f5-any.jsonl > $O/s301-f5-any.log 2>&1"
u diag-dbp "$P native/rl/research/offense/offense_diag.py --a $A --b $F5 --kits fixed --kit 2,4,14 --pairs 100 --workers 2 --seed-base 241001 --out $O/s301-f5-dbp.jsonl > $O/s301-f5-dbp.log 2>&1"
u agg "$P eval/aggression.py --subject s301=$A --subject f5=$F5 --opponent f5=$F5 --pairs 100 --workers 2 --out $O/aggression > $O/aggression.log 2>&1"
# then: $P native/rl/research/offense/human_gap.py $O/humans.jsonl s301-vs-f5-any=$O/s301-f5-any.jsonl s301-vs-f5-dbp=$O/s301-f5-dbp.jsonl --json summary.json
