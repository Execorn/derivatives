import time
import numpy as np
import torch
from src.pricing.heston import batch_heston_iv_surface
from src.pricing.rbergomi_gpu import batch_rbergomi_iv_surface
from src.pricing.local_vol import check_arbitrage_free_batch, check_arbitrage_free
from src.pricing.sabr import sabr_iv_surface, ssvi_iv_surface

def benchmark_heston():
    print("=== Heston GPU Surface Benchmarking ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    # Batch of 512 parameter sets
    B = 512
    kappa = torch.empty(B, 1).uniform_(0.1, 5.0)
    theta = torch.empty(B, 1).uniform_(0.01, 0.15)
    sigma = torch.empty(B, 1).uniform_(0.1, 1.0)
    rho = torch.empty(B, 1).uniform_(-0.9, -0.1)
    v0 = torch.empty(B, 1).uniform_(0.01, 0.15)
    params = torch.cat([kappa, theta, sigma, rho, v0], dim=1)
    
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], device=device)
    K_grid = torch.linspace(-0.5, 0.5, 11, device=device)
    
    # Warmup
    _ = batch_heston_iv_surface(params, T_grid, K_grid, device=device)
    
    t0 = time.time()
    for _ in range(10):
        ivs = batch_heston_iv_surface(params, T_grid, K_grid, device=device)
    t1 = time.time()
    
    avg_time_ms = ((t1 - t0) / 10) * 1000
    print(f"Average time for {B} surfaces: {avg_time_ms:.2f} ms")
    print(f"Time per surface: {avg_time_ms / B:.3f} ms")
    print()

def benchmark_rbergomi():
    print("=== Rough Bergomi GPU Surface Benchmarking ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    B = 32
    v0 = np.random.uniform(0.01, 0.15, B)
    H = np.random.uniform(0.05, 0.25, B)
    eta = np.random.uniform(0.5, 2.5, B)
    rho = np.random.uniform(-0.9, -0.1, B)
    params = np.column_stack([v0, H, eta, rho])
    
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    # Warmup
    _ = batch_rbergomi_iv_surface(params, T_grid, K_grid, N_paths=10000, device=device)
    
    t0 = time.time()
    for _ in range(5):
        _ = batch_rbergomi_iv_surface(params, T_grid, K_grid, N_paths=10000, device=device)
    t1 = time.time()
    
    avg_time_ms = ((t1 - t0) / 5) * 1000
    print(f"Average time for {B} surfaces (10k paths each): {avg_time_ms:.2f} ms")
    print(f"Time per surface: {avg_time_ms / B:.3f} ms")
    print()

def benchmark_local_vol_arbitrage_check():
    print("=== Local Volatility Arbitrage Check Benchmarking ===")
    B = 2048
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    a0 = np.random.uniform(0.01, 0.15, size=B)
    b0 = np.random.uniform(0.05, 0.4, size=B)
    rho_base = np.random.uniform(-0.85, -0.15, size=B)
    m_base = np.random.uniform(-0.15, 0.15, size=B)
    sigma_base = np.random.uniform(0.05, 0.35, size=B)
    
    svi_params = np.zeros((B, 8, 5))
    for j in range(8):
        T = T_grid[j]
        scale = T * np.random.uniform(0.9, 1.1, size=B)
        svi_params[:, j, 0] = a0 * scale * np.random.uniform(0.95, 1.05, size=B)
        svi_params[:, j, 1] = b0 * scale * np.random.uniform(0.95, 1.05, size=B)
        svi_params[:, j, 2] = np.clip(rho_base + np.random.uniform(-0.02, 0.02, size=B), -0.95, -0.05)
        svi_params[:, j, 3] = m_base + np.random.uniform(-0.01, 0.01, size=B)
        svi_params[:, j, 4] = np.clip(sigma_base + np.random.uniform(-0.01, 0.01, size=B), 0.01, 0.5)

    # 1. Vectorized batch check
    t0 = time.time()
    batch_res = check_arbitrage_free_batch(T_grid, K_grid, svi_params)
    t1 = time.time()
    batch_time = (t1 - t0) * 1000
    
    # 2. Sequential loop check
    t2 = time.time()
    seq_res = []
    for i in range(B):
        seq_res.append(check_arbitrage_free(T_grid, K_grid, svi_params[i]))
    t3 = time.time()
    seq_time = (t3 - t2) * 1000
    
    assert np.all(batch_res == seq_res), "Error: batched result does not match sequential result!"
    
    print(f"Sequential loop time for {B} samples: {seq_time:.2f} ms")
    print(f"Vectorized batch time for {B} samples: {batch_time:.2f} ms")
    print(f"Speedup: {seq_time / batch_time:.2f}x")
    print()

def benchmark_sabr_ssvi_batch():
    print("=== SABR & SSVI Batch Pricing Benchmarking ===")
    B = 2048
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    alpha = np.random.uniform(0.05, 0.8, B)
    rho = np.random.uniform(-0.9, 0.9, B)
    nu = np.random.uniform(0.1, 1.2, B)
    
    # SABR Batched vs Loop
    t0 = time.time()
    sabr_batch_res = sabr_iv_surface(1.0, T_grid, K_grid, alpha, np.ones(B), rho, nu)
    t1 = time.time()
    sabr_batch_time = (t1 - t0) * 1000
    
    t2 = time.time()
    sabr_seq_res = []
    for i in range(B):
        sabr_seq_res.append(sabr_iv_surface(1.0, T_grid, K_grid, alpha[i], 1.0, rho[i], nu[i]))
    sabr_seq_res = np.array(sabr_seq_res)
    t3 = time.time()
    sabr_seq_time = (t3 - t2) * 1000
    
    assert np.allclose(sabr_batch_res, sabr_seq_res, equal_nan=True), "Error: SABR batch results mismatch!"
    print(f"SABR Sequential loop time for {B} samples: {sabr_seq_time:.2f} ms")
    print(f"SABR Vectorized batch time for {B} samples: {sabr_batch_time:.2f} ms")
    print(f"SABR Speedup: {sabr_seq_time / sabr_batch_time:.2f}x")
    
    # SSVI Batched vs Loop
    theta = np.random.uniform(0.01, 0.8, (B, 8))
    rho_ssvi = np.random.uniform(-0.9, 0.9, B)
    eta = np.random.uniform(0.1, 2.0, B)
    gamma = np.random.uniform(0.1, 0.5, B)
    
    t0 = time.time()
    ssvi_batch_res = ssvi_iv_surface(T_grid, K_grid, theta, rho_ssvi, eta, gamma)
    t1 = time.time()
    ssvi_batch_time = (t1 - t0) * 1000
    
    t2 = time.time()
    ssvi_seq_res = []
    for i in range(B):
        ssvi_seq_res.append(ssvi_iv_surface(T_grid, K_grid, theta[i], rho_ssvi[i], eta[i], gamma[i]))
    ssvi_seq_res = np.array(ssvi_seq_res)
    t3 = time.time()
    ssvi_seq_time = (t3 - t2) * 1000
    
    assert np.allclose(ssvi_batch_res, ssvi_seq_res, equal_nan=True), "Error: SSVI batch results mismatch!"
    print(f"SSVI Sequential loop time for {B} samples: {ssvi_seq_time:.2f} ms")
    print(f"SSVI Vectorized batch time for {B} samples: {ssvi_batch_time:.2f} ms")
    print(f"SSVI Speedup: {ssvi_seq_time / ssvi_batch_time:.2f}x")
    print()

if __name__ == '__main__':
    benchmark_heston()
    benchmark_rbergomi()
    benchmark_local_vol_arbitrage_check()
    benchmark_sabr_ssvi_batch()
