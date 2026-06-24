"""
fit_normalizers_heston.py — Fit parameter and IV surface normalizers for Classic Heston model.

Loads the generated dataset, splits into 80/20 train/val, fits the normalizers on the train split,
and saves the parameters to artifacts/models/param_normalizer_heston.npz and iv_normalizer_heston.npz.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.normalizers import ParameterNormalizerHeston, IVSurfaceNormalizer

DATASET_PATH = 'data/HestonDataset_v1.npz'
PARAM_NORM_PATH = 'artifacts/models/param_normalizer_heston.npz'
IV_NORM_PATH = 'artifacts/models/iv_normalizer_heston.npz'

def fit_normalizers():
    print("=" * 60)
    print("  Fitting Heston Normalizers")
    print("=" * 60)
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run scripts/generate_dataset_heston.py first.")
        
    data = np.load(DATASET_PATH)
    params = data['params']  # (N, 5)
    iv = data['iv']          # (N, 8, 11)
    
    N = params.shape[0]
    print(f"  Loaded dataset with {N:,} samples.")
    
    # 80/20 train/val split using a fixed seed (same seed as in FNO training)
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    split = int(0.8 * N)
    tr_idx = idx[:split]
    
    X_tr = params[tr_idx]
    Y_tr = iv[tr_idx]
    
    print(f"  Fitting normalizers on train split of {X_tr.shape[0]:,} samples...")
    
    # Initialize and fit normalizers
    param_norm = ParameterNormalizerHeston().fit(X_tr)
    iv_norm = IVSurfaceNormalizer().fit(Y_tr)
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(PARAM_NORM_PATH), exist_ok=True)
    
    # Save
    param_norm.save(PARAM_NORM_PATH)
    iv_norm.save(IV_NORM_PATH)
    
    print(f"  Normalizers successfully saved:")
    print(f"    - {PARAM_NORM_PATH}")
    print(f"    - {IV_NORM_PATH}")
    print()
    print(param_norm.summary())
    print(iv_norm.summary())

if __name__ == '__main__':
    fit_normalizers()
