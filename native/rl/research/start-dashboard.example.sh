#!/bin/bash
# Proposed replacement for ~/.local/share/bopl-null-hypothesis/start-dashboard-8876.sh
# Auto-discovers every run under --root. Joint Elo is the ladder; --pool only
# labels frozen-pool IQM (prefer v6). Do not pass the old --run-root-glob list.
cd /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject
exec /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl/venv/bin/python native/rl/research/serve.py \
  --host 127.0.0.1 --port 8876 \
  --root /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl \
  --pool /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl/sweeps/anchors/pool-v6/pool.json \
  --human /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl/human/mac
