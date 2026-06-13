from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6" # Prevent arch errors
if hasattr(torch.version, 'cuda'):
    torch.version.cuda = '13.3'

setup(
    name='lifted_heston_cuda',
    ext_modules=[
        CUDAExtension('lifted_heston_cuda', [
            'src/cuda_engine.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3', 
                '--use_fast_math', 
                '-Xptxas=-v' # To check register usage and spills
            ]
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
