import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension

# REC-8: CUDA Version Monkey-Patch — Development Environment Workaround
#
# The line below bypasses PyTorch's built-in CUDA toolkit version compatibility check.
# This is intentional for this development environment where the CUDA driver version
# and the PyTorch-bundled CUDA toolkit version differ (e.g., CUDA 12.x driver with
# PyTorch built against CUDA 11.x headers).
#
# RISK: Bypassing this check means the build will NOT fail if there is a genuine
# CUDA ABI mismatch that could lead to undefined behaviour or silent numerical errors
# in production. This monkey-patch MUST NOT be used in production Docker images or CI.
#
# LONG-TERM FIX: Align the installed CUDA toolkit version with torch.version.cuda.
# Check required version with: python -c "import torch; print(torch.version.cuda)"
# Then install the matching toolkit: https://developer.nvidia.com/cuda-toolkit-archive
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None


# Prevent architecture mismatch errors during compilation.
# sm_86 covers the RTX 3060 Laptop (Ampere). sm_80 covers A100. sm_89 covers Ada.
# Update this list when targeting a different GPU generation.
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0"

# Align PyTorch CUDA version info if empty (e.g., when built with custom CUDA).
# This is only a metadata fix; the actual kernel ABI is set at compile time above.
if hasattr(torch.version, 'cuda') and torch.version.cuda is None:
    torch.version.cuda = '13.3'

setup(
    name='deepvol_cuda',
    ext_modules=[
        CUDAExtension(
            name='deepvol_cuda',
            sources=[
                'src/grey_bergomi.cu',
                'src/binding.cpp'
            ],
            libraries=['cufft'],
            extra_compile_args={
                'cxx': ['-O3', '-fpermissive', '-Wno-template-body', '-std=c++20'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xptxas=-v'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
