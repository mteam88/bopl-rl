#!/usr/bin/env bash
# Refuses game-derived files in this (public) repository. The game and
# everything extracted from it live in the private game/ submodule; here the
# only legitimate entries at those paths are symlinks into game/ and the
# gitlink itself.
#
#   scripts/check-game-data.sh            # the index (pre-commit)
#   scripts/check-game-data.sh --tree REV # every path in REV's tree
#   scripts/check-game-data.sh --range A..B  # every path added or modified in the range (CI)
set -euo pipefail

mode=${1:---index}
case "$mode" in
  --index) paths=$(git diff --cached --name-only --diff-filter=AMCR) ;;
  --tree)  paths=$(git ls-tree -r --name-only "${2:?rev}") ;;
  --range) paths=$(git diff --name-only --diff-filter=AMCR "${2:?range}") ;;
  *) echo "usage: $0 [--index | --tree REV | --range A..B]" >&2; exit 2 ;;
esac
[ -n "$paths" ] || exit 0

# Paths that are only ever symlinks into game/ (or the gitlink).
is_link_only_path() {
  case "$1" in
    game|Builds|native/fixtures|native/reference-export|sim/Gameplay|sim/Oracle/fixtures) return 0 ;;
    native/data/rollout-goldens|native/data/sinlut.bin|native/data/tanlut.bin|native/data/rollout.json) return 0 ;;
    native/data/level1-movement.json|native/data/level1-movement-[0-9].json) return 0 ;;
    native/data/full-reference.json|native/data/assets-reference.json|native/data/map-coverage.json) return 0 ;;
    sim/Export/settings.json|sim/Export/asset-names.json|sim/Export/execution-order.json|sim/Export/sound-bank.json) return 0 ;;
  esac
  return 1
}

mode_of() {  # git object mode of a path in the index or a tree
  case "$mode" in
    --index) git ls-files --stage -- "$1" | awk '{print $1; exit}' ;;
    --tree)  git ls-tree "$2" -- "$1" | awk '{print $1; exit}' ;;
    --range) git ls-tree "${2#*..}" -- "$1" | awk '{print $1; exit}' ;;
  esac
}

bad=0
while IFS= read -r p; do
  [ -n "$p" ] || continue
  reason=""
  if is_link_only_path "$p"; then
    m=$(mode_of "$p" "${2:-}")
    [ "$m" = 120000 ] || [ "$m" = 160000 ] || reason="must be a symlink into game/ (mode $m)"
  else
    case "$p" in
      Assets/*|ProjectSettings/*|Packages/*|UserSettings/*|*.alf) reason="Unity project path (belongs in game/)" ;;
      native/fixtures/*|native/reference-export/*|sim/Gameplay/*|sim/Oracle/fixtures/*|native/data/rollout-goldens/*|game/*) reason="game-derived data path (belongs in game/)" ;;
      *.unity|*.prefab|*.asset|*.mat|*.anim|*.controller|*.overrideController|*.physicsMaterial2D|*.cubemap|*.flare|*.shader|*.cginc|*.meta) reason="Unity asset type" ;;
      *.ogg|*.wav|*.mp3|*.ttf|*.otf|*.dll|*.so|*.dylib|*.bundle) reason="binary asset or library" ;;
      *.rep|*.jsonl.gz|*.jsonl.xz|*lut.bin) reason="recorded game trace or extracted table" ;;
    esac
    # bridge/ keeps the Unity .meta sidecars of our own scripts.
    case "$p" in bridge/*.cs.meta) reason="" ;; esac
  fi
  if [ -n "$reason" ]; then echo "refused: $p: $reason" >&2; bad=1; fi
done <<< "$paths"

if [ "$bad" != 0 ]; then
  echo "game-derived files belong in the private game/ submodule, never here (scripts/check-game-data.sh)" >&2
  exit 1
fi
