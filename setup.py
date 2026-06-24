import os
import sys
import logging
from setuptools import setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepvol.setup")

# Check env flag for explicit disable
disable_cuda = os.environ.get("DEEPVOL_DISABLE_CUDA", "0").lower() in ("1", "true", "yes")

ext_modules = []
cmdclass = {}

if not disable_cuda:
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        
        # Verify CUDA compiler or runtime is available
        cuda_available = torch.cuda.is_available()
        nvcc_exists = os.system("nvcc --version") == 0
        
        if cuda_available or nvcc_exists:
            logger.info("CUDA Environment found. Configuring lifted_heston_cuda Extension module.")
            
            # Prevent architecture mismatch errors during compilation
            if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0"
                
            # Align PyTorch CUDA version info if empty
            if hasattr(torch.version, 'cuda') and torch.version.cuda is None:
                torch.version.cuda = '13.3'
                
            cuda_ext = CUDAExtension(
                name='deepvol.models.lifted_heston_cuda',
                sources=['src/deepvol/models/cuda/cuda_engine.cu'],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': [
                        '-O3', 
                        '--use_fast_math', 
                        '-Xptxas=-v'
                    ]
                }
            )
            ext_modules = [cuda_ext]
            cmdclass = {'build_ext': BuildExtension}
        else:
            logger.warning("CUDA runtime or nvcc compiler not found. CUDA Extension module compilation skipped.")
    except ImportError as e:
        logger.warning(f"Could not import PyTorch. CUDA compilation skipped. Error: {e}")
        logger.warning("Please install PyTorch first to build CUDA extension modules.")
else:
    logger.info("CUDA compilation explicitly disabled via DEEPVOL_DISABLE_CUDA env var.")

# setup() retrieves metadata and main options declaratively from pyproject.toml
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
