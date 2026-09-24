"""Shared paths and result writing for native training and evaluation tools."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
