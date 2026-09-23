"""Read a run artifact whether it is plain or packed by compact_history.py.

`metrics.jsonl` and the run logs of a finished run live as `<name>.zst`; readers
resolve either spelling so paths in reports and run state stay unchanged.

Decompression shells out to `zstd -dc` on purpose: the training venv is Python
3.11 with no zstd bindings at all (no `zstandard`, no `pyzstd`, no
`compression.zstd`), even though the system Python 3.14 does have
`compression.zstd`. An import that works in a shell will fail under the venv.
"""
import contextlib
import io
from pathlib import Path
import subprocess


def resolved(path):
    """The spelling of `path` that exists: itself, else its `.zst` sibling."""
    path = Path(path)
    if path.exists():
        return path
    archived = path.with_name(path.name + '.zst')
    return archived if archived.exists() else path


def exists(path):
    return resolved(path).exists()


@contextlib.contextmanager
def open_text(path):
    """`path` (or its `.zst` sibling) as a text line iterator."""
    found = resolved(path)
    if found.suffix != '.zst':
        with found.open() as handle:
            yield handle
        return
    process = subprocess.Popen(['zstd', '-dcq', str(found)], stdout=subprocess.PIPE)
    try:
        yield io.TextIOWrapper(process.stdout, encoding='utf-8')
    finally:
        process.stdout.close()  # a partial read leaves zstd on SIGPIPE; that is fine
        process.wait()


def read_text(path):
    """Whole contents, or '' when neither spelling exists."""
    found = resolved(path)
    if not found.exists():
        return ''
    if found.suffix == '.zst':
        return subprocess.run(['zstd', '-dcq', str(found)], check=True,
                              stdout=subprocess.PIPE).stdout.decode()
    return found.read_text()
