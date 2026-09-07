#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
plugin=${BOPL_RALPH_PLUGIN:-$HOME/.claude/plugins/marketplaces/claude-plugins-official/plugins/ralph-loop}
if [[ ! -f "$plugin/.claude-plugin/plugin.json" ]]; then
  echo 'Set BOPL_RALPH_PLUGIN to the official Ralph Loop plugin directory.' >&2
  exit 1
fi
if [[ -f .claude/ralph-loop.local.md ]]; then
  echo 'A Ralph loop already exists here; attach to it or cancel it first.' >&2
  exit 1
fi
prompt='Read native/rl/autoresearch/program.md and pursue the research. Keep making and testing improvements until Matthew stops you.'
exec claude --model claude-fable-5-1 --plugin-dir "$plugin" \
  --name bopl-throughput --dangerously-skip-permissions "$@" \
  "First use Bash to initialize Ralph Loop: bash \"$plugin/scripts/setup-ralph-loop.sh\" \"$prompt\". Then follow that research prompt."
