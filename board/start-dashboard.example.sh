#!/bin/bash
# Reference for ~/.local/share/bopl-null-hypothesis/start-dashboard-8876.sh
# (unit bopl-research-dashboard.service).
#
# --root   training artifacts; every run under it is discovered, nested trials included
# --league the committed league: league.json (members) + payoff.jsonl (payoff counts)
#
# The frozen-pool and human-record flags are gone with the sweep-centric board:
# the league is the scale now, and it anchors on `random` the same way.
cd /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject
exec /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl/venv/bin/python board/serve.py \
  --host 127.0.0.1 --port 8876 \
  --root /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl \
  --league /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/league
