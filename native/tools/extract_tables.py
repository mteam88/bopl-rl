"""Copy the reference's literal lookup tables without regenerating their values."""
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "67484d0"
SOURCE = "Assets/Scripts/Assembly-CSharp/BoplFixedMath/Fix.cs"


def main():
    source = subprocess.check_output(["git", "show", f"{REFERENCE}:{SOURCE}"], cwd=ROOT)
    target = ROOT / "native/data"
    target.mkdir(exist_ok=True)
    tables = {}
    for name in ("SinLut", "TanLut"):
        match = re.search(rf"{name} = new long\[205887\]\s*\{{(.*?)\}};", source.decode(), re.S)
        if match is None:
            raise ValueError(f"reference table {name} missing")
        values = [int(x) for x in re.findall(r"(-?\d+)L", match[1])]
        if len(values) != 205887:
            raise ValueError(f"wrong {name} length: {len(values)}")
        data = struct.pack(f"<{len(values)}q", *values)
        filename = name.lower() + ".bin"
        (target / filename).write_bytes(data)
        tables[filename] = {"entries": len(values), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "reference_commit": subprocess.check_output(["git", "rev-parse", REFERENCE], cwd=ROOT, text=True).strip(),
        "reference_tree": subprocess.check_output(["git", "rev-parse", f"{REFERENCE}^{{tree}}"], cwd=ROOT, text=True).strip(),
        "unity_version": "2022.3.60f1",
        "math_source": SOURCE,
        "math_source_sha256": hashlib.sha256(source).hexdigest(),
        "tables": tables,
    }
    (target / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
