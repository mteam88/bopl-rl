#!/bin/bash
# Reference for ~/.local/share/bopl-null-hypothesis/start-dashboard-8876.sh (unit bopl-research-dashboard.service).
# Auto-discovers every run under --root. Joint Elo is the ladder; --pool only picks
# which frozen pool's IQM3 is the primary label (pool v7 = the sweep objective; every
# discovered pool under sweeps/anchors is still shown, labelled by its own name).
cd /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject
exec /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl/venv/bin/python native/rl/research/serve.py \
  --host 127.0.0.1 --port 8876 \
  --root /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl \
  --pool /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl/sweeps/anchors/pool-v7/pool.json \
  --human /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl/human/mac
