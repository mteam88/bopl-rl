"""Install the pinned RL toolchain in native/artifacts/rl/venv using uv."""
import hashlib
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "native/artifacts/rl"
REVISION = "42f70d6932c30ac977736f861006809c50168ba9"
SHA256 = "c8d8b81a6812854e17f86ade232b55d8069871ab17b59ceed405158b371a2845"
dependencies = OUT / "dependencies"
dependencies.mkdir(parents=True, exist_ok=True)
archive = dependencies / "pufferlib.tar.gz"
if not archive.exists():
    urllib.request.urlretrieve(f"https://codeload.github.com/PufferAI/PufferLib/tar.gz/{REVISION}", archive)
if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
    raise ValueError("PufferLib archive does not match the pinned hash")
source = dependencies / f"PufferLib-{REVISION}"
if not source.exists():
    with tarfile.open(archive) as stream:
        stream.extractall(dependencies, filter="data")
python = OUT / "venv/bin/python"
if not python.exists():
    subprocess.run(["uv", "venv", "--python", "3.11", str(OUT / "venv")], check=True)
subprocess.run(["uv", "pip", "install", "--python", str(python), "-r", str(HERE / "requirements.lock")], check=True)
subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", "--no-build-isolation", "-e", str(source)], check=True)
print(f"Installed. Build with: {python} {HERE / 'build.py'} --device cuda")
