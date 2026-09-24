"""Build the probe binding (research evaluator + privileged labels + raycasts).

    native/artifacts/rl/venv/bin/python eval/probes/build.py
"""
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ABI = ROOT / "sim/abi"
OUT = ROOT / "native/artifacts/rl/probes"
LIBRARY = OUT / "libbopl_probe.so"


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cc", "-O3", "-std=gnu11", "-fPIC", "-shared", "-Wall", "-Wno-unused-function",
                    "-I" + str(ABI), "-I" + str(ROOT / "train"), str(HERE / "probe_binding.c"),
                    str(ABI / "bopl_rl.c"), "-lm", "-lpthread", "-o", str(LIBRARY)], check=True)
    return LIBRARY


if __name__ == "__main__":
    print(build())
