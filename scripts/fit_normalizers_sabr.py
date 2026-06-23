"""
fit_normalizers_sabr.py — Fits and saves input/output z-score normalizers.

Fits:
1. SABR normalizers → param_normalizer_sabr.npz, iv_normalizer_sabr.npz
2. SSVI normalizers → param_normalizer_ssvi.npz, iv_normalizer_ssvi.npz
"""

import os
import sys
import numpy as np

# Ensure src path is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from normalizers import ParameterNormalizer, IVSurfaceNormalizer

# Paths
SABR_DATA_PATH = "data/SABRDataset_v1.npz"
SSVI_DATA_PATH = "data/SSVIDataset_v1.npz"

SABR_NORM_PARAM = "artifacts/models/param_normalizer_sabr.npz"
SABR_NORM_IV = "artifacts/models/iv_normalizer_sabr.npz"
SSVI_NORM_PARAM = "artifacts/models/param_normalizer_ssvi.npz"
SSVI_NORM_IV = "artifacts/models/iv_normalizer_ssvi.npz"


def fit_sabr_normalizers():
    print("=" * 60)
    print("  Fitting SABR Normalizers")
    print("=" * 60)
    
    if not os.path.exists(SABR_DATA_PATH):
        print(f"  [Error] Dataset not found: {SABR_DATA_PATH}")
        return
        
    data = np.load(SABR_DATA_PATH)
    params = data["params"]
    iv = data["iv"]
    
    # Instantiate and customize names
    param_norm = ParameterNormalizer()
    param_norm.PARAM_NAMES = ["alpha", "rho", "nu"]
    param_norm.fit(params)
    
    iv_norm = IVSurfaceNormalizer()
    iv_norm.fit(iv)
    
    os.makedirs(os.path.dirname(SABR_NORM_PARAM), exist_ok=True)
    param_norm.save(SABR_NORM_PARAM)
    iv_norm.save(SABR_NORM_IV)
    
    print(f"  Saved → {SABR_NORM_PARAM}")
    print(f"  Saved → {SABR_NORM_IV}")
    print(param_norm.summary())
    print(iv_norm.summary())
    print()


def fit_ssvi_normalizers():
    print("=" * 60)
    print("  Fitting SSVI Normalizers")
    print("=" * 60)
    
    if not os.path.exists(SSVI_DATA_PATH):
        print(f"  [Error] Dataset not found: {SSVI_DATA_PATH}")
        return
        
    data = np.load(SSVI_DATA_PATH)
    params = data["params"]
    iv = data["iv"]
    
    # Instantiate and customize names
    param_norm = ParameterNormalizer()
    param_norm.PARAM_NAMES = [f"theta_{i+1}" for i in range(8)] + ["rho", "eta", "gamma"]
    param_norm.fit(params)
    
    iv_norm = IVSurfaceNormalizer()
    iv_norm.fit(iv)
    
    os.makedirs(os.path.dirname(SSVI_NORM_PARAM), exist_ok=True)
    param_norm.save(SSVI_NORM_PARAM)
    iv_norm.save(SSVI_NORM_IV)
    
    print(f"  Saved → {SSVI_NORM_PARAM}")
    print(f"  Saved → {SSVI_NORM_IV}")
    print(param_norm.summary())
    print(iv_norm.summary())
    print()


if __name__ == "__main__":
    fit_sabr_normalizers()
    fit_ssvi_normalizers()
