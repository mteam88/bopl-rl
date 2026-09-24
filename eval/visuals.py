"""Evaluation visual replays: `game.json` while a run is fresh, `game.json.gz` after
compact_evaluations.py has archived a finished run. Readers resolve either.

Reports keep referring to `name.json`; only the bytes on disk move. gzip -1 shrinks
a physics-state replay ~12x, which is what keeps the sweep tree inside the disk.
"""
import gzip
import json
from pathlib import Path


def source(path):
    """The on-disk file behind a visual path (plain or gzipped), or None."""
    path = Path(path)
    if path.suffix == '.gz':
        return path if path.is_file() else None
    if path.is_file():
        return path
    archived = path.with_name(path.name + '.gz')
    return archived if archived.is_file() else None


def exists(path):
    return source(path) is not None


def read_bytes(path):
    found = source(path)
    if found is None:
        raise FileNotFoundError(path)
    raw = found.read_bytes()
    return gzip.decompress(raw) if found.suffix == '.gz' else raw


def load(path):
    return json.loads(read_bytes(path))
