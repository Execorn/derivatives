import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension
# Monkey-patch to bypass CUDA version mismatch check
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None


# Prevent architecture mismatch errors during compilation
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0"

# Align PyTorch CUDA version info if empty
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
