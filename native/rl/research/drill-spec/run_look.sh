#!/bin/bash
# One drill_look condition in its own capped unit. usage: run_look.sh NAME A AKIT B BKIT GAMES
RL=$(cd "$(dirname "$0")/../../../.." && pwd)/native/artifacts/rl
D=$RL/research/drill-spec
name=$1; shift
mkdir -p $D/look/$name
exec systemd-run --user --unit=drill-spec-$name --collect -p MemoryHigh=8G -p CPUWeight=30 \
  -E BOPL_RL_BACKEND=rust -E PYTHONPATH=$RL/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9 \
  --working-directory=$D \
  /bin/bash -c "$RL/venv/bin/python $(cd "$(dirname "$0")" && pwd)/drill_look.py --a $1 --a-kit $2 --b $3 --b-kit $4 --games $5 --seed-base 4100001 --out $D/look/$name --visual 16 > $D/look/$name/log.txt 2>&1"
