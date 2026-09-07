"""Run Unity reference captures and find the first exact state difference."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]


def run_reference(command, **kwargs):
    """Keep Unity's heap/cache outside an IDE service's memory limit.

    Long captures also run one at a time on a systemd host. Other hosts keep
    the ordinary subprocess path; no machine-wide service settings change.
    """
    runtime = Path(f'/run/user/{os.getuid()}') if sys.platform == 'linux' else None
    systemd = shutil.which('systemd-run') if runtime and (runtime/'bus').exists() else None
    if systemd is None:
        return subprocess.run(command, **kwargs)
    import fcntl
    with (runtime/'bopl-reference-capture.lock').open('a') as gate:
        fcntl.flock(gate, fcntl.LOCK_EX)
        unit = f'bopl-reference-{os.getpid()}-{uuid.uuid4().hex[:8]}'
        wrapper = [systemd, '--user', '--scope', '--collect', '--quiet', '--unit='+unit]
        if kwargs.get('timeout') is not None:
            wrapper += ['--property=RuntimeMaxSec='+str(kwargs['timeout']),
                        '--property=TimeoutStopSec=5s']
        return subprocess.run(wrapper+list(command), **kwargs)


def read_trace(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        header = json.loads(next(stream))
        if header.get("kind") != "header" or header.get("schema") not in ("bopl-managed-state-v1", "bopl-managed-state-v2", "bopl-managed-state-v3"):
            raise ValueError(f"invalid trace header: {path}")
        yield header
        count = 0
        awaiting_state = False
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "input":
                if awaiting_state or row["index"] != count:
                    raise ValueError(f"input sequence invalid at {count}")
                awaiting_state = True
            elif row["kind"] == "state":
                if not awaiting_state or row["index"] != count:
                    raise ValueError(f"state sequence invalid at {count}")
                awaiting_state = False
                count += 1
            elif row["kind"] == "end":
                if awaiting_state or count == 0 or row["ticks"] != count or next(stream, None) is not None:
                    raise ValueError(f"invalid trace ending: {path}")
                yield row
                return
            else:
                raise ValueError(f"unexpected row: {row['kind']}")
            yield row
        raise ValueError(f"trace has no completed end marker: {path}")


def first_difference(left, right, path="", ignored_fields=frozenset()):
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        for key in sorted((set(left) | set(right)) - ignored_fields):
            if key not in left or key not in right:
                return path + "/" + key, left.get(key, "<missing>"), right.get(key, "<missing>")
            difference = first_difference(left[key], right[key], path + "/" + key, ignored_fields)
            if difference:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + "/length", len(left), len(right)
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, path + "/" + str(index), ignored_fields)
            if difference:
                return difference
    elif left != right:
        return path, left, right
    return None


def provenance(path):
    path = Path(path)
    record = json.loads(path.with_suffix(".provenance.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["trace_sha256"]:
        raise ValueError(f"trace hash does not match provenance: {path}")
    return record


def compare(left, right, gameplay=False):
    ignored_fields = frozenset(json.loads((ROOT / "native/data/presentation-fields.json").read_text())["fields"]) if gameplay else frozenset()
    view = {"projection": "gameplay-with-audited-exclusions" if gameplay else "raw-managed-state", "excluded_fields": sorted(ignored_fields)}
    pa, pb = provenance(left), provenance(right)
    for key in ("replay_sha256", "game_assembly_sha256", "unity_player_sha256"):
        if pa[key] != pb[key]:
            raise ValueError(f"incompatible provenance {key}")
    for key in ("player_binary_sha256", "il2cpp_metadata_sha256"):
        if pa.get(key) != pb.get(key):
            raise ValueError(f"incompatible provenance {key}")
    a, b = read_trace(left), read_trace(right)
    header_a, header_b = next(a), next(b)
    for key in ("schema", "reference", "unity", "scope"):
        if header_a[key] != header_b[key]:
            raise ValueError(f"incompatible {key}: {header_a[key]} vs {header_b[key]}")
    ticks = 0
    while True:
        x, y = next(a, None), next(b, None)
        difference = first_difference(x, y, ignored_fields=ignored_fields)
        if difference:
            path, old, new = difference
            print(json.dumps({**view, "match": False, "matched_ticks": ticks, "path": path,
                "left": old, "right": new}, indent=2))
            return 1
        if x is None:
            break
        if x["kind"] == "state":
            ticks += 1
    print(json.dumps({**view, "match": True, "ticks": ticks, "left": str(left), "right": str(right)}))
    return 0


def resolve_game(game=None):
    return Path(game or os.environ.get("BOPL_RL_GAME")
                or ROOT / "Builds/LinuxIL2CPP/BoplBattle.x86_64").resolve()


def player_sources(binary):
    """Identify the runtime actually launched, including IL2CPP metadata."""
    sources = {"player_binary_sha256": binary,
               "unity_player_sha256": binary.parent / "UnityPlayer.so"}
    native = binary.parent / "GameAssembly.so"
    if native.exists():
        sources["game_assembly_sha256"] = native
        sources["il2cpp_metadata_sha256"] = binary.parent / "BoplBattle_Data/il2cpp_data/Metadata/global-metadata.dat"
    else:
        sources["game_assembly_sha256"] = binary.parent / "BoplBattle_Data/Managed/Assembly-CSharp.dll"
    return sources


def capture(args):
    if args.ticks <= 0 or args.batch <= 0:
        raise ValueError("ticks and batch must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or Path(str(output) + ".failure.json").exists():
        raise ValueError(f"refusing to overwrite {output}")
    binary = resolve_game(getattr(args, "game", None))
    env = dict(os.environ, BOPL_TRACE=str(output), BOPL_TRACE_TICKS=str(args.ticks), BOPL_TRACE_BATCH=str(args.batch))
    physics = getattr(args, "physics", False)
    if physics: env["BOPL_TRACE_PHYSICS"] = "1"
    else: env.pop("BOPL_TRACE_PHYSICS", None)
    for name in ("BOPL_KERNEL_INPUT", "BOPL_KERNEL_OUTPUT"):
        env.pop(name, None)
    sources = {"replay_sha256": args.replay, **player_sources(binary)}
    before = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
    log = output.with_suffix(".unity.log")
    command = [str(binary), str(args.replay.resolve()), "-batchmode", "-nographics", "-logFile", str(log)]
    started = time.monotonic()
    with output.with_suffix(".stdout.log").open("w") as stdout:
        run_reference(command, env=env, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
    after = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
    if before != after:
        raise ValueError("reference inputs changed while capture was running")
    provenance = {
        "command": command, "batch": args.batch, "requested_ticks": args.ticks, "physics_stages": physics,
        "elapsed_seconds": time.monotonic() - started,
        **before,
        "trace_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    failure = Path(str(output) + ".failure.json")
    if failure.exists():
        provenance["status"] = "simulation exception; full trace rejected"
        provenance["failure_sha256"] = hashlib.sha256(failure.read_bytes()).hexdigest()
        failure.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        raise ValueError(f"Unity simulation failed; boundary evidence: {failure}")
    ticks = sum(row["kind"] == "state" for row in read_trace(output))
    if ticks != args.ticks:
        raise ValueError(f"only {ticks}/{args.ticks} requested ticks captured")
    provenance["ticks"] = ticks
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("capture")
    run.add_argument("replay", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--ticks", type=int, default=600)
    run.add_argument("--batch", type=int, default=1)
    run.add_argument("--timeout", type=int, default=180)
    run.add_argument("--game", type=Path, help="Player executable (default: BOPL_RL_GAME or Builds/Linux)")
    run.add_argument("--physics", action="store_true", help="capture full rigid-body stages before gameplay callbacks")
    diff = sub.add_parser("compare")
    diff.add_argument("left", type=Path)
    diff.add_argument("right", type=Path)
    diff.add_argument("--gameplay", action="store_true", help="exclude only source-audited presentation fields; retain raw trace")
    args = parser.parse_args()
    return capture(args) if args.command == "capture" else compare(args.left, args.right, args.gameplay)


if __name__ == "__main__":
    sys.exit(main())
