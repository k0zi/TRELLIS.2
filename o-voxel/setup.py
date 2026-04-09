import shutil
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

# ── Detect available compiler ─────────────────────────────────────────────────
HAS_NVCC  = shutil.which("nvcc")  is not None
HAS_HIPCC = shutil.which("hipcc") is not None

if BUILD_TARGET == "cuda":
    USE_CUDA, IS_HIP = True, False
elif BUILD_TARGET == "rocm":
    USE_CUDA, IS_HIP = True, True
elif BUILD_TARGET == "cpu":
    USE_CUDA, IS_HIP = False, False
else:  # "auto"
    try:
        from torch.utils.cpp_extension import IS_HIP_EXTENSION
        IS_HIP = IS_HIP_EXTENSION
    except ImportError:
        IS_HIP = False
    USE_CUDA = HAS_NVCC or HAS_HIPCC

print(f"[o_voxel] BUILD_TARGET={BUILD_TARGET}  USE_CUDA={USE_CUDA}  IS_HIP={IS_HIP}")

# ── Source files ──────────────────────────────────────────────────────────────
CPU_SOURCES = [
    # Convert (pure C++)
    "src/convert/flexible_dual_grid.cpp",
    "src/convert/volumetic_attr.cpp",
    # IO (pure C++)
    "src/io/svo.cpp",
    "src/io/filter_parent.cpp",
    "src/io/filter_neighbor.cpp",
    # Serialize – CPU-only implementation (no CUDA headers)
    "src/serialize/cpu_only.cpp",
    # Entry point
    "src/ext.cpp",
]

CUDA_SOURCES = [
    # Hash (CUDA only)
    "src/hash/hash.cu",
    # Serialize – GPU kernels + CPU wrappers via CUDA compiler
    "src/serialize/api.cu",
    "src/serialize/hilbert.cu",
    "src/serialize/z_order.cu",
    # Rasterize (CUDA only)
    "src/rasterize/rasterize.cu",
]

INCLUDE_DIRS = [os.path.join(ROOT, "third_party/eigen")]

# ── Build extension ───────────────────────────────────────────────────────────
if USE_CUDA:
    from torch.utils.cpp_extension import CUDAExtension

    cc_flag = []
    if IS_HIP:
        archs = os.getenv("GPU_ARCHS", "native").split(";")
        cc_flag = [f"--offload-arch={arch}" for arch in archs]

    ext = CUDAExtension(
        name="o_voxel._C",
        sources=CUDA_SOURCES + CPU_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args={
            "cxx":  ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17"] + cc_flag,
        },
    )
else:
    from torch.utils.cpp_extension import CppExtension

    ext = CppExtension(
        name="o_voxel._C",
        sources=CPU_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        define_macros=[("CPU_ONLY", "1")],
    )

setup(
    name="o_voxel",
    packages=["o_voxel", "o_voxel.convert", "o_voxel.io"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
