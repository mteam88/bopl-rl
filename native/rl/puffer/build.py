"""Build Bopl's C environment with the pinned upstream PufferLib binding.

Run with the RL virtualenv Python. CUDA builds target RTX 5070 (sm_120).
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sysconfig

import pybind11

HERE = Path(__file__).resolve().parent
REVISION = "42f70d6932c30ac977736f861006809c50168ba9"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--binding-only", action="store_true", help="Reuse the existing CUDA object when only environment code changed")
parser.add_argument("--cuda-host-compiler", default="g++-15")
parser.add_argument("--puffer", type=Path, default=HERE.parents[1] / "artifacts/rl/dependencies" / f"PufferLib-{REVISION}")
args = parser.parse_args()
source = args.puffer.resolve()
out = HERE.parents[1] / "artifacts/rl"
out.mkdir(parents=True, exist_ok=True)
# Keep the pinned dependency tree unchanged. Upstream dereferences a null
# graph array during teardown when cudagraphs=-1; apply the small fix only
# in this generated compilation directory.
overlay = out / "puffer-overlay"
shutil.copytree(source / "src", overlay, dirs_exist_ok=True)
subprocess.run(["patch", "-p1", "-i", str(HERE / "pufferlib-close.patch")], cwd=overlay, check=True)
# Bopl trainer semantics on top of the pinned trainer: episode-boundary
# resets of the MinGRU state (rollout and training scan), training
# sequences start from the real rollout-start state, learner-only replay
# sampling, no reward clamp, per-head entropy / gradient-norm / advantage
# statistics, the bopl branch of create_custom_encoder (BOPL_NATIVE_ENCODER
# = linear|set selects bopl_set_encoder.cu, found via the -I below), per-bank
# rollout graph branches with per-stream cuBLAS handles, and the rollout env
# pool in vecenv.h (persistent work-stealing workers and futex handshakes in
# place of the per-step OpenMP fork/join and the spinning stream sync;
# binding.c compiles vecenv.h, so it too builds against the overlay). See pufferlib-bopl.patch (generated with
# diff -ruN against the pristine overlay: cd artifacts/rl && diff -ruN
# pristine-src puffer-overlay).
subprocess.run(["patch", "-p1", "-i", str(HERE / "pufferlib-bopl.patch")], cwd=overlay, check=True)
subprocess.run(["cc", "-O3", "-fPIC", "-fopenmp", "-DPRECISION_FLOAT", "-I" + str(overlay),
                "-I" + str(HERE.parent), "-c", str(HERE / "binding.c"), "-o", str(out / "bopl_binding.o")], check=True)
subprocess.run(["cc", "-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC", "-c",
                str(HERE.parent / "bopl_rl.c"), "-o", str(out / "bopl_rl.o")], check=True)
target = source / "pufferlib" / ("_C" + sysconfig.get_config_var("EXT_SUFFIX"))
if args.device == "cuda":
    import numpy
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    cuda = nvidia / "cu13"
    includes = [overlay, source / "src", HERE, Path(sysconfig.get_path("include")), Path(pybind11.get_include()),
                Path(numpy.get_include()), cuda / "include", nvidia / "cudnn/include", nvidia / "nccl/include"]
    if args.binding_only:
        # The CUDA object compiles bindings.cu -> pufferlib.cu -> bopl_aim.h, bopl_rollout.cu,
        # ocean.cu -> bopl_set_encoder.cu, plus the two overlay patches. Refuse to relink a
        # stale one rather than ship a binding whose kernels predate their sources.
        cuda_object = out / "puffer_cuda.o"
        cuda_sources = [HERE / name for name in ("bopl_rollout.cu", "bopl_rollout_v4.cu", "bopl_set_encoder.cu", "bopl_v4.cu", "bopl_layout_v4.h", "bopl_aim.h",
                                                  "bopl_layout.h", "pufferlib-close.patch", "pufferlib-bopl.patch")]
        if not cuda_object.exists():
            raise SystemExit(f"--binding-only: {cuda_object} does not exist; run a full build")
        newer = [path.name for path in cuda_sources if path.stat().st_mtime > cuda_object.stat().st_mtime]
        if newer:
            raise SystemExit(f"--binding-only: {cuda_object} is older than {newer}; run a full build")
    else:
        subprocess.run([str(cuda / "bin/nvcc"), "-O3", "-std=c++17", "-arch=sm_120", "-ccbin", args.cuda_host_compiler,
                        "-Xcompiler=-fPIC,-fopenmp", "-DPRECISION_FLOAT", "-DENV_NAME=bopl",
                        "-DOBS_TENSOR_T=FloatTensor", *["-I" + str(path) for path in includes],
                        "-c", str(overlay / "bindings.cu"), "-o", str(out / "puffer_cuda.o")], check=True)

    libraries = [cuda / "lib", nvidia / "cudnn/lib", nvidia / "nccl/lib"]
    names = ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13", "libcusolver.so.12", "libcurand.so.10",
             "libcudnn.so.9", "libnccl.so.2"]
    links = []
    for name in names:
        links.append(str(next(path / name for path in libraries if (path / name).exists())))
    subprocess.run(["c++", "-shared", "-fPIC", "-fopenmp", str(out / "puffer_cuda.o"),
                    str(out / "bopl_binding.o"), str(out / "bopl_rl.o"), *links,
                    *["-Wl,-rpath," + str(path) for path in libraries], "-l:libnvidia-ml.so.1",
                    # Driver API: the trainer partitions the GPU between rollout and
                    # training streams with green contexts (pufferlib.cu bopl_gpu_split_init).
                    "-l:libcuda.so.1",
                    "-lm", "-lpthread", "-o", str(target)], check=True)
else:
    subprocess.run(["c++", "-O3", "-std=c++17", "-fPIC", "-fopenmp", "-DPRECISION_FLOAT", "-DENV_NAME=bopl",
                "-DOBS_TENSOR_T=FloatTensor", "-I" + str(overlay), "-I" + str(source / "src"),
                "-I" + sysconfig.get_path("include"), "-I" + pybind11.get_include(),
                    "-c", str(source / "src/bindings_cpu.cpp"), "-o", str(out / "puffer_cpu.o")], check=True)
    subprocess.run(["c++", "-shared", "-fPIC", "-fopenmp", str(out / "puffer_cpu.o"),
                str(out / "bopl_binding.o"), str(out / "bopl_rl.o"), "-lm", "-lpthread", "-o", str(target)], check=True)
subprocess.run(["cc", "-O3", "-std=c11", "-fPIC", "-shared", "-I" + str(HERE.parent),
                str(HERE / "eval_binding.c"), str(HERE.parent / "bopl_rl.c"), "-lm",
                "-o", str(out / "libbopl_research.so")], check=True)
# policy_eval.c is the CPU forward pass evaluate.py / the pool rater load through
# native_policy.py; it shares the head layout with the trainer, so rebuild it in
# lockstep (overnight-7 lost 4 trials to a stale copy missing _create_aim).
subprocess.run(["cc", "-O3", "-std=gnu11", "-fPIC", "-shared", "-I" + str(source / "src"),
                str(HERE / "policy_eval.c"), "-lm", "-o", str(out / "libbopl_policy_eval.so")], check=True)
shutil.copyfile(HERE / "bopl.ini", source / "config/bopl.ini")
print(target)
