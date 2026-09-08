#!/usr/bin/env python3
"""Extract ProjectSettings needed by the .NET sim into sim/Export/settings.json.

Sources (Unity text-serialized YAML, all relative to the project root):
  ProjectSettings/TagManager.asset        tags, 32 layer names
  ProjectSettings/Physics2DSettings.asset m_LayerCollisionMatrix, 2D gravity
  ProjectSettings/TimeManager.asset       Fixed Timestep, m_TimeScale
  ProjectSettings/DynamicsManager.asset   3D gravity (completeness only)

m_LayerCollisionMatrix encoding
-------------------------------
The YAML value looks like

    c80000/6c80000/2c80000/2ffffff/f... (256 chars total, 32 slashes)

It is the concatenation of 32 groups of 8 characters, one group per layer
row a (mask over layers b). Each group is the LITTLE-ENDIAN BYTE hex of the
uint32 mask -- chars [0:2]=byte0 (bits 0-7), [2:4]=byte1, [4:6]=byte2,
[6:8]=byte3 -- except char index 6 (the high nibble of byte3, i.e. bits
28-31) is written as '/'. Layers 28-31 are unnamed in this project, and
Unity's default for any layer pair is "collide" (bit SET), so bits 28-31
are 1 in every row; we reconstruct that nibble as 0xF.

This was verified against the built player: Builds/Linux/BoplBattle_Data/
globalgamemanagers stores the same matrix as 32 raw little-endian uint32s
(count marker 0x20 at offset 61132, rows at 61136..61263 in the current
build), and every decoded row matches, e.g. row 0 (Default) = 0xf60000c8,
row 8 (Player) = 0xf78d2fd8. This script re-runs that cross-check when the
build file is present and fails loudly on any mismatch.

Semantics: bit b of mask a SET means layers a and b COLLIDE (the checkbox
in Unity's Layer Collision Matrix UI is ticked). Physics2D.
GetIgnoreLayerCollision(a, b) returns true when the bit is CLEAR:
"whether collisions between layer1 and layer2 are being ignored"
(https://docs.unity3d.com/ScriptReference/Physics2D.GetIgnoreLayerCollision.html);
IgnoreLayerCollision(a, b, true) clears the bit. A fresh project has every
row = 0xffffffff (see DynamicsManager.asset here, all 'f') and
GetIgnoreLayerCollision returns false everywhere, confirming SET = collide.
The matrix is symmetric: bit b of row a == bit a of row b (asserted below).
"""

import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "sim" / "Export" / "settings.json"

SOURCES = [
    "ProjectSettings/TagManager.asset",
    "ProjectSettings/Physics2DSettings.asset",
    "ProjectSettings/TimeManager.asset",
    "ProjectSettings/DynamicsManager.asset",
]


def load_unity_yaml(path: Path) -> dict:
    """Unity asset YAML -> the single document's inner mapping.

    Strips the '%YAML'/'%TAG' directives and the '!u!<class> &<id>' tag on the
    document marker so plain yaml.safe_load accepts it.
    """
    text = path.read_text()
    lines = [l for l in text.splitlines() if not l.startswith("%")]
    lines = [re.sub(r"^--- !u!\d+ &\d+.*$", "---", l) for l in lines]
    doc = yaml.safe_load("\n".join(lines))
    # {'TagManager': {...}} -> inner mapping
    (inner,) = doc.values()
    return inner


def decode_collision_matrix(raw: str) -> list[int]:
    assert len(raw) == 256, f"expected 256 chars, got {len(raw)}"
    rows = []
    for i in range(32):
        g = raw[i * 8 : i * 8 + 8]
        assert g[6] == "/", f"group {i}: expected '/' at index 6, got {g!r}"
        b0 = int(g[0:2], 16)
        b1 = int(g[2:4], 16)
        b2 = int(g[4:6], 16)
        b3 = 0xF0 | int(g[7], 16)  # high nibble (bits 28-31) elided as '/', always F
        rows.append(b0 | b1 << 8 | b2 << 16 | b3 << 24)
    # symmetry: layers a,b collide iff b,a collide
    for a in range(32):
        for b in range(32):
            assert (rows[a] >> b) & 1 == (rows[b] >> a) & 1, f"asymmetry at {a},{b}"
    return rows


def verify_against_build(rows: list[int]) -> str:
    """Cross-check the decoded matrix against the built player's binary
    globalgamemanagers (32 consecutive raw LE uint32s). Returns a status
    string; raises on mismatch when the pattern is locatable."""
    ggm = ROOT / "Builds" / "Linux" / "BoplBattle_Data" / "globalgamemanagers"
    if not ggm.exists():
        return "skipped (no build)"
    data = ggm.read_bytes()
    pattern = struct.pack("<32I", *rows)
    if pattern in data:
        return f"ok (found at offset {data.index(pattern)})"
    raise AssertionError(
        "decoded m_LayerCollisionMatrix not found in built globalgamemanagers; "
        "the bits-28-31 reconstruction or the group decoding is wrong"
    )


def main() -> None:
    tag_mgr = load_unity_yaml(ROOT / "ProjectSettings" / "TagManager.asset")
    time_mgr = load_unity_yaml(ROOT / "ProjectSettings" / "TimeManager.asset")
    dyn_mgr = load_unity_yaml(ROOT / "ProjectSettings" / "DynamicsManager.asset")

    layers = [l if l is not None else "" for l in tag_mgr["layers"]]
    assert len(layers) == 32, f"expected 32 layers, got {len(layers)}"
    tags = list(tag_mgr["tags"])

    # PyYAML would mis-split the matrix token on '/': grab it verbatim instead.
    p2d_path = ROOT / "ProjectSettings" / "Physics2DSettings.asset"
    p2d_text = p2d_path.read_text()
    matrix_raw = re.search(r"m_LayerCollisionMatrix: (\S+)", p2d_text).group(1)
    matrix = decode_collision_matrix(matrix_raw)
    build_check = verify_against_build(matrix)

    p2d = load_unity_yaml(p2d_path)
    gravity2d = p2d["m_Gravity"]  # {x: 0, y: -40}

    settings = {
        "layers": layers,
        "tags": tags,
        "collisionMatrix": matrix,
        "fixedDeltaTime": time_mgr["Fixed Timestep"],
        "timeScale": time_mgr["m_TimeScale"],
        "gravity2D": [gravity2d["x"], gravity2d["y"]],
        "gravity3D": [dyn_mgr["m_Gravity"][a] for a in ("x", "y", "z")],
        "source": {
            rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            for rel in SOURCES
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(f"  layers: {sum(1 for l in layers if l)} named / 32")
    print(f"  tags: {len(tags)}")
    print(f"  fixedDeltaTime: {settings['fixedDeltaTime']}")
    print(f"  collision matrix build cross-check: {build_check}")


if __name__ == "__main__":
    sys.exit(main())
