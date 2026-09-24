"""Compare a C-controlled episode with ordinary replay playback in the game."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native/tools"))
from reference import run_reference, read_trace, first_difference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--players", type=int, default=6)
parser.add_argument("--ability-group", type=int, default=0)
parser.add_argument("--level", type=int, default=0, help="Map index recorded into the episode; replay playback follows the archive")
parser.add_argument("--ticks", type=int, default=180)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--compare-only", action="store_true", help="Recheck existing, hash-verified captures with the current projection")
args = parser.parse_args()
out = args.output.resolve()
episode, trace = out / "episode.rep", out / "controlled.jsonl.gz"
game = Path(os.environ.get("BOPL_RL_GAME") or ROOT / "Builds/Linux/BoplBattle.x86_64").resolve()
replayed = out / "replayed.jsonl.gz"
sources = {"replay": episode, "controlled": trace, "replayed": replayed}
# Identify the player by whichever runtime payload it ships: a Mono build's managed assembly,
# or an IL2CPP build's native library and metadata (no Assembly-CSharp.dll exists there).
mono_assembly = game.parent / "BoplBattle_Data/Managed/Assembly-CSharp.dll"
if mono_assembly.exists():
    sources["assembly"] = mono_assembly
if (game.parent / "GameAssembly.so").exists():
    sources["game_assembly"] = game.parent / "GameAssembly.so"
    sources["metadata"] = game.parent / "BoplBattle_Data/il2cpp_data/Metadata/global-metadata.dat"
if "assembly" not in sources and "game_assembly" not in sources:
    raise FileNotFoundError(f"No Mono assembly or IL2CPP payload next to {game}")
if args.compare_only:
    previous = json.loads((out / "report.json").read_text())
    if previous.get("game", str(ROOT / "Builds/Linux/BoplBattle.x86_64")) != str(game):
        raise ValueError("Existing capture was made with a different player")
    for name, path in sources.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != previous[name + "_sha256"]:
            raise ValueError(f"Existing capture identity changed: {name}")
    metadata = {"ticks": previous["requested_ticks"]}
    args.players, args.ability_group = previous["players"], previous.get("ability_group", 0)
    args.level = previous.get("level", 0)
else:
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, BOPL_TRACE=str(trace), BOPL_TRACE_TICKS="100000", BOPL_TRACE_BATCH="16")
    command = [str(ROOT / "native/artifacts/rl/record"), str(game), str(ROOT / "native/fixtures/movement.rep"),
               str(out / "controlled-game.log"), str(episode), str(args.players), str(args.ability_group), str(args.ticks), str(args.level)]
    result = run_reference(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=600, check=True)
    metadata = json.loads(result.stdout)
    subprocess.run([sys.executable, str(ROOT / "native/tools/reference.py"), "capture", str(episode), str(replayed),
                    "--ticks", str(metadata["ticks"]), "--batch", "16", "--timeout", "600"], check=True)
# Presentation fields (cosmetic) plus fields holding raw GetInstanceID values,
# which are process-global and cannot match between two Unity processes.
ignored = frozenset(json.loads((ROOT / "native/data/presentation-fields.json").read_text())["fields"]) \
    | frozenset(json.loads((ROOT / "native/data/process-identity-fields.json").read_text())["fields"])
left, right = read_trace(trace), read_trace(replayed)
left_header, right_header = next(left), next(right)
if left_header["setup"] != right_header["setup"]:
    raise AssertionError("Recorded replay setup differs from the controlled episode")
ticks = 0
report = dict(players=args.players, ability_group=args.ability_group, level=args.level, projection="gameplay-with-audited-exclusions", excluded_fields=sorted(ignored), match=True)
while True:
    a, b = next(left, None), next(right, None)
    if a and b and a["kind"] == b["kind"] == "end":
        report["ability_entries"] = a.get("abilityEntries", {})
        # End reason is transport-specific; state and ability-entry counts must agree.
        a.pop("reason", None); b.pop("reason", None)
    diff = first_difference(a, b, ignored_fields=ignored)
    if diff:
        report.update(match=False, path=diff[0], left=diff[1], right=diff[2]); break
    if a is None:
        break
    if a["kind"] == "state":
        ticks += 1
report["matched_ticks"] = ticks
report["requested_ticks"] = metadata["ticks"]
report["game"] = str(game)
report["backend"] = "il2cpp" if "game_assembly" in sources else "mono"
for name, path in sources.items():
    report[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["match"] else 1)
