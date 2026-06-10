import os
import time
import torch
import numpy as np
import pandas as pd
from data_loader import generate_deep_rough_lhs, MATURITIES, MONEYNESS
from iv_inverter import jaeckel_iv
from validate_cuda import compute_weights_and_speeds
import lifted_heston_cuda

def generate_dataset(n_samples=10_000, batch_size=1000, out_file="data/DeepRoughDataset.npz"):
    """
    Generate the Deep Rough dataset.
    Note: To generate the full 100k, you'd set n_samples=100_000.
    For this test script, we default to 10k to ensure it finishes quickly.
    """
    os.makedirs("data", exist_ok=True)
    
    print(f"Generating LHS parameters for {n_samples} samples...")
    df_params = generate_deep_rough_lhs(n_samples=n_samples)
    
    # We will simulate 8 maturities x 11 strikes = 88 prices per sample.
    # The grid:
    # Forward F=100
    S0 = 100.0
    F = 100.0
    
    K_grid = F * np.exp(MONEYNESS)
    T_grid = MATURITIES
    
    # We will process in batches
    num_paths = 10_000 # Number of MC paths per price
    
    all_ivs = []
    
    print("Starting CUDA pricing and IV inversion...")
    start_time = time.time()
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_df = df_params.iloc[start_idx:end_idx]
        
        batch_ivs = []
        for local_idx, (_, row) in enumerate(batch_df.iterrows()):
            kappa = row['kappa']
            theta = row['theta']
            sigma = row['sigma']
            rho = row['rho']
            v0 = row['v0']
            H = row['H']
            
            # FIX: unique seed per sample prevents correlated MC paths across rows.
            # Previously seed=42 was hardcoded for all samples, making the entire
            # dataset share the same Brownian realisation — a silent dataset bias.
            global_sample_idx = start_idx + local_idx
            sample_seed = 42 + global_sample_idx  # deterministic but unique per row
            
            c_weights, x_speeds = compute_weights_and_speeds(20, H)
            c_weights_cuda = c_weights.cuda().contiguous()
            x_speeds_cuda = x_speeds.cuda().contiguous()
            
            iv_row = []
            for t_idx, T in enumerate(T_grid):
                num_steps = int(T * 252) # Daily steps
                if num_steps == 0: num_steps = 1
                dt = T / num_steps
                
                prices = lifted_heston_cuda.simulate_lifted_heston(
                    num_paths, num_steps, float(dt), float(S0), float(v0),
                    float(rho), float(kappa), float(theta), float(sigma),
                    c_weights_cuda, x_speeds_cuda,
                    seed=sample_seed, call_index=t_idx
                )

                # Compute payoffs for all strikes on GPU
                payoffs = torch.relu(prices.unsqueeze(1) - torch.tensor(K_grid, dtype=torch.float32, device='cuda').unsqueeze(0))
                option_prices = payoffs.mean(dim=0).cpu()
                
                # Invert to IV
                ivs = jaeckel_iv(option_prices, F, torch.tensor(K_grid, dtype=torch.float32), T)
                iv_row.extend(ivs.numpy().tolist())
                
            batch_ivs.append(iv_row)
            
        all_ivs.extend(batch_ivs)
        
        if (start_idx + batch_size) % 1000 == 0:
            print(f"Processed {start_idx + batch_size} / {n_samples} samples...")
            
    print(f"Dataset generated in {time.time() - start_time:.2f} seconds.")
    
    # Save to disk
    features = df_params.values
    # Clip any negative IV values produced by failed NR inversions.
    # ~1% of deep-OTM options yield IV < 0 from the Newton-Raphson solver;
    # keeping them as training targets teaches the FNO to output negatives.
    targets = np.clip(np.array(all_ivs), 1e-4, None)
    
    dataset = np.hstack((features, targets))
    np.savez_compressed(out_file, dataset=dataset)
    print(f"Saved to {out_file}. Shape: {dataset.shape}")

if __name__ == "__main__":
    # 50k samples: 6-dimensional parameter space needs dense LHS coverage.
    # 1k was a test stub left in accidentally — caused single-batch training.
    # At ~30s per 1000 samples on CUDA, 50k ≈ 25 min.
    generate_dataset(n_samples=50_000)
