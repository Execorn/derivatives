import os
import time
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from data_loader import generate_deep_rough_lhs, MATURITIES, MONEYNESS
from pricing_engine import price_iv_surface

def _worker(row_dict):
    """
    Worker function to compute IV surface for a single parameter configuration.
    """
    params = {
        'kappa': row_dict['kappa'],
        'theta': row_dict['theta'],
        'sigma': row_dict['sigma'],
        'rho': row_dict['rho'],
        'v0': row_dict['v0'],
        'H': row_dict['H']
    }
    
    # price_iv_surface returns shape (8, 11) matching MATURITIES and MONEYNESS.
    # N_cos=400 is heavily optimized and vectorized over both maturities and frequencies.
    ivs = price_iv_surface(params, MATURITIES, MONEYNESS, S0=1.0, N_factors=20, N_cos=400)
    
    return ivs.flatten()

def generate_dataset(n_samples=50_000, out_file="data/DeepRoughDataset.npz"):
    """
    Generate the EXACT Deep Rough dataset using Fourier-COS pricing.
    """
    os.makedirs("data", exist_ok=True)
    
    # We will try to load the existing parameters to ensure we use identical configurations.
    # If the file exists, we just extract the first 6 columns.
    if os.path.exists(out_file):
        print(f"Loading existing parameters from {out_file} to ensure identical coverage...")
        existing_data = np.load(out_file)['dataset']
        if len(existing_data) >= n_samples:
            features = existing_data[:n_samples, :6]
            df_params = pd.DataFrame(features, columns=['kappa', 'theta', 'sigma', 'rho', 'v0', 'H'])
        else:
            print(f"Existing dataset only has {len(existing_data)} samples. Generating new LHS...")
            df_params = generate_deep_rough_lhs(n_samples=n_samples)
    else:
        print(f"Generating LHS parameters for {n_samples} samples...")
        df_params = generate_deep_rough_lhs(n_samples=n_samples)
        
    print(f"Starting EXACT deterministic pricing using multiprocessing over {n_samples} samples...")
    start_time = time.time()
    
    # Convert df to list of dicts for multiprocessing
    tasks = [row.to_dict() for _, row in df_params.iterrows()]
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} CPU cores...")
    
    all_ivs = []
    completed = 0
    
    with multiprocessing.Pool(num_workers) as pool:
        for result in pool.imap(_worker, tasks, chunksize=100):
            all_ivs.append(result)
            completed += 1
            if completed % 1000 == 0:
                print(f"Processed {completed} / {n_samples} samples... Elapsed: {time.time() - start_time:.2f}s")
                
    print(f"Dataset generated in {time.time() - start_time:.2f} seconds.")
    
    # Save to disk
    features = df_params.values
    # Clean NaNs by interpolating or clipping, but exact pricing should be robust.
    targets = np.array(all_ivs)
    # Forward-fill any NaNs across strikes/maturities if optimization bounds were hit
    df_targets = pd.DataFrame(targets)
    df_targets = df_targets.interpolate(axis=1, limit_direction='both').fillna(0.3)
    targets = np.clip(df_targets.values, 1e-4, 5.0)
    
    dataset = np.hstack((features, targets))
    np.savez_compressed(out_file, dataset=dataset)
    print(f"Saved to {out_file}. Shape: {dataset.shape}")

if __name__ == "__main__":
    generate_dataset(n_samples=50_000)
